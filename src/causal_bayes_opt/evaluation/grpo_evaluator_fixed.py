"""
GRPO Evaluator with Proper Surrogate Integration

This module provides the fixed GRPO evaluator that properly integrates
the bootstrap surrogate for structure learning during evaluation.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pickle
import json
import time

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as onp
import haiku as hk

from .base_evaluator import BaseEvaluator
from .result_types import ExperimentResult, StepResult
from ..data_structures.scm import (
    get_target, get_variables, get_parents
)
from ..mechanisms.linear import sample_from_linear_scm
from ..interventions.handlers import create_perfect_intervention
from ..acquisition import create_acquisition_state
from ..analysis.trajectory_metrics import (
    compute_f1_score_from_marginals,
    compute_shd_from_marginals
)

logger = logging.getLogger(__name__)


class GRPOEvaluatorFixed(BaseEvaluator):
    """
    Fixed evaluator for GRPO that properly integrates the bootstrap surrogate.
    
    Key improvements:
    - Loads and uses bootstrap surrogate for structure learning
    - Properly tracks marginal parent probabilities
    - Respects optimization direction during evaluation
    - Updates state with surrogate predictions after each intervention
    """
    
    def __init__(self, checkpoint_path: Path, name: Optional[str] = None):
        """
        Initialize GRPO evaluator.
        
        Args:
            checkpoint_path: Path to GRPO checkpoint directory
            name: Optional custom name (defaults to checkpoint name)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")
            
        # Extract name from checkpoint if not provided
        if name is None:
            name = f"GRPO_{checkpoint_path.name}"
            
        super().__init__(
            name=name,
            checkpoint_paths={'policy': checkpoint_path}
        )
        
        self.checkpoint_path = checkpoint_path
        self.policy_params = None
        self.policy_fn = None
        self.optimization_direction = None
        self.surrogate_params = None
        self.surrogate_fn = None
        
    def initialize(self) -> None:
        """Load GRPO checkpoint and initialize evaluation components."""
        logger.info(f"Initializing GRPO evaluator from {self.checkpoint_path}")
        
        try:
            # Load checkpoint metadata
            metadata_path = self.checkpoint_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                # Extract optimization direction
                opt_config = metadata.get('optimization_config', {})
                self.optimization_direction = opt_config.get('direction', 'MAXIMIZE')
                self.training_config = metadata.get('training_config', {})
                self.reward_weights = self.training_config.get('reward_weights', {})
                logger.info(f"Loaded optimization direction: {self.optimization_direction}")
            
            # Load policy parameters
            checkpoint_file = self.checkpoint_path / "checkpoint.pkl"
            if not checkpoint_file.exists():
                raise FileNotFoundError(f"No checkpoint.pkl found in {self.checkpoint_path}")
                
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                
            self.policy_params = checkpoint_data.get('policy_params')
            if self.policy_params is None:
                raise ValueError("No policy_params found in checkpoint")
                
            # Store policy config
            self.policy_config = checkpoint_data.get('policy_config', {})
            
            # Import trainer components to reuse existing logic
            from ..training.enriched_trainer import EnrichedGRPOTrainer
            from ..training.modular_trainer import StateConverter, PolicyFactory
            from omegaconf import DictConfig
            
            # Create minimal config for state converter
            dummy_config = DictConfig({
                'training': {
                    'architecture': self.policy_config.get('architecture', {}),
                    'state_config': self.policy_config.get('state_config', {
                        'standardize_values': True,
                        'include_temporal_features': True,
                        'max_history_size': 100,
                        'num_channels': 5
                    }),
                    'n_variables': 10  # Max variables to support
                }
            })
            
            # Initialize state converter
            self.state_converter = StateConverter(dummy_config)
            
            # Create policy function using PolicyFactory
            policy_factory = PolicyFactory(dummy_config)
            self.policy_fn, _ = policy_factory.create_policy()
            
            # Check surrogate integration
            self.use_surrogate = self.training_config.get('surrogate_integration', {}).get('enabled', False)
            
            if self.use_surrogate:
                # Load bootstrap surrogate if available
                self._load_bootstrap_surrogate(checkpoint_data)
            else:
                logger.warning("Surrogate integration disabled - structure learning metrics will be empty")
            
            self._initialized = True
            logger.info("GRPO evaluator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GRPO evaluator: {e}")
            raise
    
    def _load_bootstrap_surrogate(self, checkpoint_data: Dict[str, Any]) -> None:
        """Load bootstrap surrogate model from checkpoint."""
        try:
            # Check if surrogate params are in the checkpoint
            self.surrogate_params = checkpoint_data.get('surrogate_params')
            
            if self.surrogate_params is None:
                # Try to load from separate surrogate checkpoint
                surrogate_path = self.checkpoint_path.parent / "bootstrap_surrogate"
                if surrogate_path.exists():
                    surrogate_file = surrogate_path / "checkpoint.pkl"
                    if surrogate_file.exists():
                        with open(surrogate_file, 'rb') as f:
                            surrogate_data = pickle.load(f)
                        self.surrogate_params = surrogate_data.get('model_params')
                        logger.info("Loaded surrogate from separate checkpoint")
                    
            if self.surrogate_params is not None:
                # Create surrogate function
                from ..surrogates.continuous_parent_set import ContinuousParentSetPredictionModel
                
                def surrogate_fn(X: jnp.ndarray) -> Dict[str, jnp.ndarray]:
                    model = ContinuousParentSetPredictionModel(
                        hidden_dim=256,
                        num_layers=4,
                        num_heads=8,
                        dropout_rate=0.1,
                        support_empty_parents=True
                    )
                    return model(X)
                
                self.surrogate_fn = hk.transform(surrogate_fn)
                logger.info("Bootstrap surrogate loaded successfully")
            else:
                logger.warning("No surrogate parameters found - structure learning will be limited")
                self.use_surrogate = False
                
        except Exception as e:
            logger.warning(f"Failed to load bootstrap surrogate: {e}")
            self.use_surrogate = False
    
    def _compute_marginal_probs_from_surrogate(
        self,
        samples_data: Dict[str, jnp.ndarray],
        target_var: str,
        variables: List[str],
        key: jax.random.PRNGKey
    ) -> Dict[str, float]:
        """Compute marginal parent probabilities using the bootstrap surrogate."""
        if not self.use_surrogate or self.surrogate_fn is None:
            return {}
            
        try:
            # Prepare data for surrogate
            n_samples = len(samples_data[variables[0]])
            n_vars = len(variables)
            
            # Create data matrix
            X = jnp.zeros((n_samples, n_vars))
            for i, var in enumerate(variables):
                X = X.at[:, i].set(samples_data[var])
            
            # Add batch dimension
            X_batch = X[None, ...]  # [1, n_samples, n_vars]
            
            # Run surrogate
            output = self.surrogate_fn.apply(self.surrogate_params, key, X_batch)
            
            # Extract parent probabilities for target variable
            target_idx = variables.index(target_var)
            
            # Get marginal probabilities
            marginal_probs = {}
            
            if 'parent_probs' in output:
                # Shape: [batch, n_vars, n_vars]
                parent_probs = output['parent_probs'][0]  # Remove batch dim
                
                for i, var in enumerate(variables):
                    if var != target_var:
                        # Probability that var is a parent of target
                        prob = float(parent_probs[target_idx, i])
                        marginal_probs[var] = prob
            
            return marginal_probs
            
        except Exception as e:
            logger.warning(f"Failed to compute surrogate predictions: {e}")
            return {}
    
    def evaluate_single_run(
        self,
        scm: Any,
        config: Dict[str, Any],
        seed: int,
        run_idx: int = 0
    ) -> ExperimentResult:
        """
        Run GRPO evaluation on a single SCM with proper surrogate integration.
        
        Args:
            scm: Structural Causal Model
            config: Evaluation configuration
            seed: Random seed
            run_idx: Run index
            
        Returns:
            ExperimentResult with learning history
        """
        if not self._initialized:
            self.initialize()
            
        start_time = time.time()
        
        # Extract configuration
        max_interventions = config.get('experiment', {}).get('target', {}).get(
            'max_interventions', 10
        )
        n_observational_samples = config.get('experiment', {}).get('target', {}).get(
            'n_observational_samples', 100
        )
        
        # Initialize random key
        key = random.PRNGKey(seed)
        
        # Get SCM info
        target_var = get_target(scm)
        variables = list(get_variables(scm))
        true_parents = list(get_parents(scm, target_var))
        
        # Sample initial observational data
        key, obs_key = random.split(key)
        obs_samples = sample_from_linear_scm(scm, n_observational_samples, seed=int(obs_key[0]))
        
        # Convert to dict format
        obs_data = {}
        for var in variables:
            obs_data[var] = jnp.array([s.get('values', {}).get(var, 0.0) for s in obs_samples])
        
        # Compute initial target value
        initial_value = float(jnp.mean(obs_data[target_var]))
        
        # Create initial state with surrogate predictions
        key, surrogate_key = random.split(key)
        initial_marginals = self._compute_marginal_probs_from_surrogate(
            obs_data, target_var, variables, surrogate_key
        )
        
        # Create proper acquisition state
        current_state = self._create_acquisition_state(
            obs_data, target_var, variables, initial_value, initial_marginals
        )
        
        # Initialize tracking
        learning_history = []
        best_value = initial_value
        cumulative_samples = obs_data.copy()
        
        # Store initial state
        learning_history.append(StepResult(
            step=0,
            intervention={},
            outcome_value=initial_value,
            marginals=initial_marginals,
            uncertainty=self._compute_uncertainty(initial_marginals),
            reward=0.0,
            computation_time=0.0
        ))
        
        # Run intervention loop
        for step in range(1, max_interventions + 1):
            step_start = time.time()
            
            # Convert state to enriched input for policy
            enriched_input = self.state_converter.convert_state_to_enriched_input(current_state)
            
            # Get target variable index
            target_idx = variables.index(target_var)
            
            # Apply policy to select intervention
            key, policy_key = random.split(key)
            policy_output = self.policy_fn.apply(
                self.policy_params, policy_key, enriched_input, target_idx, False
            )
            
            # Convert policy output to intervention
            selected_var_idx, intervention_value = self._policy_output_to_action(
                policy_output, variables, target_var, self.optimization_direction
            )
            selected_var = variables[selected_var_idx]
            
            # Create intervention dict
            intervention_dict = {selected_var: float(intervention_value)}
            
            # Apply intervention and sample
            key, int_key = random.split(key)
            intervention_obj = create_perfect_intervention(
                targets=frozenset([selected_var]),
                values={selected_var: intervention_value}
            )
            
            # Sample with intervention
            from ..environments.sampling import sample_with_intervention
            int_samples = sample_with_intervention(scm, intervention_obj, 100, seed=int(int_key[0]))
            
            # Convert to dict format
            int_data = {}
            for var in variables:
                int_data[var] = jnp.array([s['values'][var] for s in int_samples])
            
            # Compute outcome value
            outcome_value = float(jnp.mean(int_data[target_var]))
            
            # Update best value based on optimization direction
            if self.optimization_direction == "MINIMIZE":
                if outcome_value < best_value:
                    best_value = outcome_value
            else:
                if outcome_value > best_value:
                    best_value = outcome_value
            
            # Update cumulative data
            for var in variables:
                cumulative_samples[var] = jnp.concatenate([
                    cumulative_samples[var], int_data[var]
                ])
            
            # Compute updated marginals using surrogate
            key, surrogate_key = random.split(key)
            current_marginals = self._compute_marginal_probs_from_surrogate(
                cumulative_samples, target_var, variables, surrogate_key
            )
            
            # Update state for next iteration
            current_state = self._update_acquisition_state(
                current_state, intervention_dict, outcome_value, 
                best_value, current_marginals, step
            )
            
            # Calculate reward based on optimization direction
            if self.optimization_direction == "MINIMIZE":
                reward = initial_value - outcome_value
            else:
                reward = outcome_value - initial_value
            
            step_result = StepResult(
                step=step,
                intervention=intervention_dict,
                outcome_value=outcome_value,
                marginals=current_marginals,
                uncertainty=self._compute_uncertainty(current_marginals),
                reward=reward,
                computation_time=time.time() - step_start
            )
            
            learning_history.append(step_result)
        
        # Compute final metrics
        final_value = learning_history[-1].outcome_value if learning_history else initial_value
        final_marginals = learning_history[-1].marginals if learning_history else {}
        
        # Compute structure learning metrics
        final_f1 = compute_f1_score_from_marginals(final_marginals, true_parents)
        final_shd = compute_shd_from_marginals(final_marginals, true_parents)
        
        # Calculate improvement based on optimization direction
        if self.optimization_direction == "MINIMIZE":
            improvement = initial_value - final_value
        else:
            improvement = final_value - initial_value
        
        final_metrics = {
            'initial_value': initial_value,
            'final_value': final_value,
            'best_value': best_value,
            'improvement': improvement,
            'n_interventions': len(learning_history) - 1,
            'final_f1': final_f1,
            'final_shd': final_shd,
            'optimization_direction': self.optimization_direction,
            'surrogate_used': self.use_surrogate
        }
        
        # Create result
        return ExperimentResult(
            learning_history=learning_history,
            final_metrics=final_metrics,
            metadata={
                'method': self.name,
                'checkpoint': str(self.checkpoint_path),
                'optimization_direction': self.optimization_direction,
                'surrogate_integration': self.use_surrogate,
                'scm_info': {
                    'n_variables': len(variables),
                    'target': target_var,
                    'true_parents': true_parents
                },
                'run_idx': run_idx,
                'seed': seed,
                'reward_weights': self.reward_weights
            },
            success=True,
            total_time=time.time() - start_time
        )
    
    def _create_acquisition_state(
        self,
        obs_data: Dict[str, jnp.ndarray],
        target_var: str,
        variables: List[str],
        initial_value: float,
        initial_marginals: Dict[str, float]
    ) -> Any:
        """Create initial acquisition state with proper structure."""
        from ..acquisition.state import AcquisitionState
        from ..avici_integration.parent_set.posterior import create_parent_set_posterior
        from ..data_structures.buffer import ExperienceBuffer
        from ..data_structures.sample import create_sample
        import pyrsistent as pyr
        
        # Create buffer with observational data
        buffer = ExperienceBuffer()
        n_samples = len(obs_data[variables[0]])
        
        for i in range(n_samples):
            sample_values = {var: float(obs_data[var][i]) for var in variables}
            sample = create_sample(values=sample_values)
            buffer.add_observation(sample)
        
        # Create posterior with marginal probabilities
        parent_sets = []
        probabilities = []
        
        # Add empty parent set
        parent_sets.append(frozenset())
        probabilities.append(0.1)  # Small probability for empty set
        
        # Add single parent sets based on marginals
        for var in variables:
            if var != target_var:
                parent_sets.append(frozenset([var]))
                prob = initial_marginals.get(var, 0.1)
                probabilities.append(prob)
        
        # Normalize probabilities
        probabilities = jnp.array(probabilities)
        probabilities = probabilities / jnp.sum(probabilities)
        
        posterior = create_parent_set_posterior(
            target_variable=target_var,
            parent_sets=parent_sets,
            probabilities=probabilities
        )
        
        # Create state with proper marginals
        state = AcquisitionState(
            posterior=posterior,
            buffer=buffer,
            best_value=initial_value,
            current_target=target_var,
            step=0,
            metadata=pyr.m(marginal_parent_probs=initial_marginals)
        )
        
        return state
    
    def _update_acquisition_state(
        self,
        current_state: Any,
        intervention: Dict[str, float],
        outcome_value: float,
        best_value: float,
        new_marginals: Dict[str, float],
        step: int
    ) -> Any:
        """Update acquisition state after intervention."""
        from ..acquisition.state import AcquisitionState
        import pyrsistent as pyr
        
        # Update metadata with new marginals
        new_metadata = current_state.metadata.set('marginal_parent_probs', new_marginals)
        
        # Create updated state
        return AcquisitionState(
            posterior=current_state.posterior,
            buffer=current_state.buffer,
            best_value=best_value,
            current_target=current_state.current_target,
            step=step,
            metadata=new_metadata
        )
    
    def _compute_uncertainty(self, marginals: Dict[str, float]) -> float:
        """Compute uncertainty from marginal probabilities."""
        if not marginals:
            return 0.0
            
        # Compute entropy of marginal distribution
        probs = list(marginals.values())
        probs = jnp.array(probs)
        
        # Add small epsilon to avoid log(0)
        probs = jnp.clip(probs, 1e-10, 1.0)
        
        # Binary entropy for each variable
        entropies = -probs * jnp.log2(probs) - (1 - probs) * jnp.log2(1 - probs)
        
        # Average entropy as uncertainty measure
        return float(jnp.mean(entropies))
    
    def _policy_output_to_action(
        self,
        policy_output: Dict[str, jnp.ndarray],
        variables: List[str],
        target: str,
        optimization_direction: str
    ) -> Tuple[int, float]:
        """Convert policy output to action with optimization direction awareness."""
        # Extract outputs
        variable_logits = policy_output.get('variable_logits', jnp.zeros(len(variables)))
        value_params = policy_output.get('value_params', jnp.zeros((len(variables), 2)))
        
        # Get target index
        target_idx = variables.index(target) if target in variables else -1
        
        # Mask target variable
        masked_logits = variable_logits.at[target_idx].set(-jnp.inf)
        
        # Select variable with highest logit (deterministic for evaluation)
        selected_var_idx = int(jnp.argmax(masked_logits))
        
        # Get value parameters for selected variable
        selected_mean = value_params[selected_var_idx, 0]
        selected_log_std = value_params[selected_var_idx, 1]
        
        # For evaluation, use mean value
        intervention_value = float(selected_mean)
        
        # Apply optimization direction hint
        # For minimization, prefer negative values; for maximization, prefer positive
        if optimization_direction == "MINIMIZE" and intervention_value > 0:
            # Consider trying negative value for minimization
            intervention_value = -abs(intervention_value)
        elif optimization_direction == "MAXIMIZE" and intervention_value < 0:
            # Consider trying positive value for maximization
            intervention_value = abs(intervention_value)
        
        return selected_var_idx, intervention_value