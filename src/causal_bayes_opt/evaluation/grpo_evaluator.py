"""
GRPO Evaluator Adapter

Adapts GRPO evaluation to the unified evaluation interface.
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


class GRPOEvaluator(BaseEvaluator):
    """
    Evaluator for GRPO (Gradient-based Reward Policy Optimization) methods.
    
    This adapter loads a trained GRPO checkpoint and evaluates it following
    the unified evaluation interface.
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
                    'state_config': self.policy_config.get('state_config', {}),
                    'n_variables': 10  # Max variables to support
                }
            })
            
            # Initialize state converter
            self.state_converter = StateConverter(dummy_config, max_variables=10)
            
            # Create policy function using PolicyFactory
            policy_factory = PolicyFactory(dummy_config, max_variables=10)
            self.policy_fn, _ = policy_factory.create_policy()
            
            # Store surrogate integration flag
            self.use_surrogate = self.training_config.get('surrogate_integration', {}).get('enabled', False)
            
            self._initialized = True
            logger.info("GRPO evaluator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GRPO evaluator: {e}")
            raise
    
    def evaluate_single_run(
        self,
        scm: Any,
        config: Dict[str, Any],
        seed: int,
        run_idx: int = 0
    ) -> ExperimentResult:
        """
        Run GRPO evaluation on a single SCM.
        
        This uses the actual trained policy to select interventions and
        tracks both optimization and structure learning metrics.
        
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
        
        # Create initial state using proper ACBO state management
        from ..avici_integration.parent_set import ParentSetPosterior
        from ..data_structures.buffer import ExperienceBuffer
        import pyrsistent as pyr
        
        # Create minimal components for state
        buffer = ExperienceBuffer()
        # Add observational data to buffer
        for i in range(n_observational_samples):
            sample_values = {var: float(obs_data[var][i]) for var in variables}
            from ..data_structures.sample import create_sample
            sample = create_sample(values=sample_values)
            buffer.add_observation(sample)
        
        # Create proper acquisition state for GRPO
        from ..acquisition.state import AcquisitionState
        from ..avici_integration.parent_set.posterior import create_parent_set_posterior
        
        # Create initial posterior with uniform distribution
        parent_sets = [frozenset()]  # Empty parent set
        for var in variables:
            if var != target_var:
                parent_sets.append(frozenset([var]))
        
        n_sets = len(parent_sets)
        probabilities = jnp.ones(n_sets) / n_sets
        
        posterior = create_parent_set_posterior(
            target_variable=target_var,
            parent_sets=parent_sets,
            probabilities=probabilities
        )
        
        current_state = AcquisitionState(
            posterior=posterior,
            buffer=buffer,
            best_value=initial_value,
            current_target=target_var,
            step=0,
            metadata=pyr.m()
        )
        
        # Initialize tracking
        learning_history = []
        best_value = initial_value
        cumulative_samples = obs_data.copy()
        
        # Store initial state
        initial_marginals = dict(current_state.marginal_parent_probs)
        learning_history.append(StepResult(
            step=0,
            intervention={},  # No intervention yet
            outcome_value=initial_value,
            marginals=initial_marginals,
            uncertainty=current_state.uncertainty,
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
                self.policy_params, policy_key, enriched_input, target_idx, False  # Not training
            )
            
            # Convert policy output to intervention
            selected_var_idx, intervention_value = self._policy_output_to_action(
                policy_output, variables, target_var
            )
            selected_var = variables[selected_var_idx]
            
            # Create intervention dict
            intervention_dict = {selected_var: float(intervention_value)}
            
            # Apply intervention and sample
            key, int_key = random.split(key)
            # Create intervention
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
            
            # Create updated state for next iteration (AcquisitionState is immutable)
            current_state = AcquisitionState(
                posterior=current_state.posterior,
                buffer=current_state.buffer,
                best_value=best_value,
                current_target=current_state.current_target,
                step=current_state.step + 1,
                metadata=current_state.metadata
            )
            
            # Extract marginals from posterior
            current_marginals = {}
            
            # Create step result
            # Calculate reward based on optimization direction
            if self.optimization_direction == "MINIMIZE":
                reward = initial_value - outcome_value  # Negative values are better
            else:
                reward = outcome_value - initial_value  # Positive values are better
            
            step_result = StepResult(
                step=step,
                intervention=intervention_dict,
                outcome_value=outcome_value,
                marginals=current_marginals,
                uncertainty=current_state.uncertainty,
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
            improvement = initial_value - final_value  # Lower is better
        else:
            improvement = final_value - initial_value  # Higher is better
        
        final_metrics = {
            'initial_value': initial_value,
            'final_value': final_value,
            'best_value': best_value,
            'improvement': improvement,
            'n_interventions': len(learning_history) - 1,  # Exclude initial state
            'final_f1': final_f1,
            'final_shd': final_shd,
            'optimization_direction': self.optimization_direction
        }
        
        # Create result
        return ExperimentResult(
            learning_history=learning_history,
            final_metrics=final_metrics,
            metadata={
                'method': self.name,
                'checkpoint': str(self.checkpoint_path),
                'optimization_direction': self.optimization_direction,
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
    
    def _policy_output_to_action(self,
                                policy_output: Dict[str, jnp.ndarray],
                                variables: List[str],
                                target: str) -> Tuple[int, float]:
        """Convert policy output to action using same logic as training."""
        # Extract outputs
        variable_logits = policy_output.get('variable_logits', jnp.zeros(len(variables)))
        value_params = policy_output.get('value_params', jnp.zeros((len(variables), 2)))
        
        # Get target index
        target_idx = variables.index(target) if target in variables else -1
        
        # Use deterministic selection for evaluation (no temperature)
        # Select variable with highest logit (excluding target)
        variable_probs = jax.nn.softmax(variable_logits)
        
        # For evaluation, use argmax instead of sampling
        selected_var_idx = int(jnp.argmax(variable_logits))
        
        # Get value parameters for selected variable
        selected_mean = value_params[selected_var_idx, 0]
        selected_log_std = value_params[selected_var_idx, 1]
        
        # For evaluation, use mean value (no sampling)
        intervention_value = float(selected_mean)
        
        # Ensure we don't select target
        if selected_var_idx == target_idx:
            # Find next best variable
            masked_logits = variable_logits.at[target_idx].set(-jnp.inf)
            selected_var_idx = int(jnp.argmax(masked_logits))
            selected_mean = value_params[selected_var_idx, 0]
            intervention_value = float(selected_mean)
        
        return selected_var_idx, intervention_value