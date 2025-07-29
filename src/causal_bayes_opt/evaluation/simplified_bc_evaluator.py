"""
Simplified BC Evaluator

Clean evaluator for BC models trained with SimplifiedBCTrainer.
Works directly with the new checkpoint format without adapters.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pickle
import time

import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
import numpy as onp

from .base_evaluator import BaseEvaluator
from .result_types import ExperimentResult, StepResult
from ..data_structures.scm import get_target, get_variables, get_parents
from ..mechanisms.linear import sample_from_linear_scm
from ..environments.sampling import sample_with_intervention
from ..interventions.handlers import create_perfect_intervention
from ..analysis.trajectory_metrics import (
    compute_f1_score_from_marginals,
    compute_shd_from_marginals
)

logger = logging.getLogger(__name__)


class SimplifiedBCEvaluator(BaseEvaluator):
    """
    Clean evaluator for BC models from SimplifiedBCTrainer.
    
    Features:
    - Direct checkpoint loading  
    - Clean model recreation
    - Supports surrogate-only, acquisition-only, or both
    - Simple evaluation logic
    """
    
    def __init__(
        self,
        surrogate_checkpoint: Optional[Path] = None,
        acquisition_checkpoint: Optional[Path] = None,
        name: Optional[str] = None
    ):
        """
        Initialize BC evaluator.
        
        Args:
            surrogate_checkpoint: Path to BC surrogate checkpoint file
            acquisition_checkpoint: Path to BC acquisition checkpoint file
            name: Optional custom name
        """
        # Validate inputs
        if surrogate_checkpoint is None and acquisition_checkpoint is None:
            raise ValueError("At least one checkpoint must be provided")
            
        # Convert to Path objects
        if surrogate_checkpoint:
            surrogate_checkpoint = Path(surrogate_checkpoint)
            if not surrogate_checkpoint.exists():
                raise ValueError(f"Surrogate checkpoint not found: {surrogate_checkpoint}")
                
        if acquisition_checkpoint:
            acquisition_checkpoint = Path(acquisition_checkpoint)
            if not acquisition_checkpoint.exists():
                raise ValueError(f"Acquisition checkpoint not found: {acquisition_checkpoint}")
                
        # Generate name if not provided
        if name is None:
            if surrogate_checkpoint and acquisition_checkpoint:
                name = "BC_Combined"
            elif surrogate_checkpoint:
                name = "BC_Surrogate"
            else:
                name = "BC_Acquisition"
                
        super().__init__(name=name)
        
        self.surrogate_checkpoint = surrogate_checkpoint
        self.acquisition_checkpoint = acquisition_checkpoint
        
        # Model components
        self.surrogate_params = None
        self.surrogate_fn = None
        self.acquisition_params = None
        self.acquisition_fn = None
        
        # Configurations
        self.surrogate_config = None
        self.acquisition_config = None
        
    def initialize(self) -> None:
        """Load checkpoints and initialize models."""
        logger.info(f"Initializing SimplifiedBCEvaluator: {self.name}")
        
        try:
            # Load surrogate if provided
            if self.surrogate_checkpoint:
                self._load_surrogate()
                
            # Load acquisition if provided
            if self.acquisition_checkpoint:
                self._load_acquisition()
                
            self._initialized = True
            logger.info("BC evaluator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize BC evaluator: {e}")
            raise
            
    def _load_surrogate(self) -> None:
        """Load surrogate model from checkpoint."""
        logger.info(f"Loading surrogate from {self.surrogate_checkpoint}")
        
        with open(self.surrogate_checkpoint, 'rb') as f:
            checkpoint = pickle.load(f)
            
        # Extract components
        self.surrogate_params = checkpoint.get("params")
        if self.surrogate_params is None:
            raise ValueError("No params found in surrogate checkpoint")
            
        # Get config
        self.surrogate_config = checkpoint.get("config", {})
        model_type = checkpoint.get("model_type", "surrogate")
        
        if model_type != "surrogate":
            logger.warning(f"Expected surrogate model but got {model_type}")
            
        # Create surrogate function
        self._create_surrogate_fn()
        
        logger.info("Surrogate model loaded successfully")
        
    def _load_acquisition(self) -> None:
        """Load acquisition model from checkpoint."""
        logger.info(f"Loading acquisition from {self.acquisition_checkpoint}")
        
        with open(self.acquisition_checkpoint, 'rb') as f:
            checkpoint = pickle.load(f)
            
        # Extract components
        self.acquisition_params = checkpoint.get("params")
        if self.acquisition_params is None:
            raise ValueError("No params found in acquisition checkpoint")
            
        # Get config
        self.acquisition_config = checkpoint.get("config", {})
        model_type = checkpoint.get("model_type", "acquisition")
        
        if model_type != "acquisition":
            logger.warning(f"Expected acquisition model but got {model_type}")
            
        # Create acquisition function
        self._create_acquisition_fn()
        
        logger.info("Acquisition model loaded successfully")
        
    def _create_surrogate_fn(self) -> None:
        """Create surrogate model function."""
        from ..avici_integration.continuous.model import ContinuousParentSetPredictionModel
        
        # Extract config
        hidden_dim = self.surrogate_config.get("hidden_dim", 256)
        num_layers = self.surrogate_config.get("num_layers", 4)
        num_heads = self.surrogate_config.get("num_heads", 8)
        
        def surrogate_model(X: jnp.ndarray) -> Dict[str, jnp.ndarray]:
            model = ContinuousParentSetPredictionModel(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout_rate=0.1,
                support_empty_parents=True
            )
            return model(X)
            
        self.surrogate_fn = hk.transform(surrogate_model)
        
    def _create_acquisition_fn(self) -> None:
        """Create acquisition model function."""
        from ..acquisition.enhanced_policy_network import EnhancedPolicyNetwork
        
        # Extract config
        hidden_dim = self.acquisition_config.get("hidden_dim", 256)
        num_layers = self.acquisition_config.get("num_layers", 4)
        
        def acquisition_model(state: jnp.ndarray, target_idx: int) -> Dict[str, jnp.ndarray]:
            # Simple policy network
            model = EnhancedPolicyNetwork(
                num_variables=state.shape[0],
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                target_idx=target_idx
            )
            return model(state)
            
        self.acquisition_fn = hk.transform(acquisition_model)
        
    def evaluate_single_run(
        self,
        scm: Any,
        config: Dict[str, Any],
        seed: int,
        run_idx: int = 0
    ) -> ExperimentResult:
        """
        Evaluate BC models on a single SCM.
        
        Args:
            scm: Structural Causal Model
            config: Evaluation configuration
            seed: Random seed
            run_idx: Run index
            
        Returns:
            ExperimentResult with trajectory
        """
        if not self._initialized:
            self.initialize()
            
        start_time = time.time()
        
        # Extract SCM info
        target_var = get_target(scm)
        variables = list(get_variables(scm))
        true_parents = list(get_parents(scm, target_var))
        n_vars = len(variables)
        
        # Initialize random key
        key = random.PRNGKey(seed)
        
        # Extract evaluation config
        max_interventions = config.get('max_interventions', 10)
        n_observational = config.get('n_observational_samples', 100)
        intervention_range = config.get('intervention_value_range', (-2.0, 2.0))
        
        # Sample initial observational data
        key, obs_key = random.split(key)
        obs_samples = sample_from_linear_scm(scm, n_observational, seed=int(obs_key[0]))
        
        # Convert to dict format
        obs_data = {}
        for var in variables:
            obs_data[var] = jnp.array([s['values'][var] for s in obs_samples])
            
        # Compute initial target value
        initial_value = float(jnp.mean(obs_data[target_var]))
        
        # Initialize tracking
        learning_history = []
        best_value = initial_value
        cumulative_samples = obs_data.copy()
        
        # Compute initial marginals if using surrogate
        initial_marginals = {}
        if self.surrogate_fn:
            key, surrogate_key = random.split(key)
            initial_marginals = self._compute_marginals_with_surrogate(
                cumulative_samples, target_var, variables, surrogate_key
            )
            
        # Store initial state
        learning_history.append(StepResult(
            step=0,
            intervention={},
            outcome_value=initial_value,
            marginals=initial_marginals,
            uncertainty=self._compute_uncertainty(initial_marginals) if initial_marginals else 1.0,
            reward=0.0,
            computation_time=0.0
        ))
        
        # Simple state representation
        current_state = jnp.zeros((n_vars, 32))
        
        # Run intervention loop
        for step in range(1, max_interventions + 1):
            step_start = time.time()
            
            # Select intervention
            if self.acquisition_fn:
                # Use trained acquisition model
                key, policy_key = random.split(key)
                target_idx = variables.index(target_var)
                
                policy_output = self.acquisition_fn.apply(
                    self.acquisition_params, policy_key, current_state, target_idx
                )
                
                # Extract action
                var_logits = policy_output.get('variable_logits', jnp.zeros(n_vars))
                var_logits = var_logits.at[target_idx].set(-jnp.inf)  # Mask target
                
                selected_var_idx = int(jnp.argmax(var_logits))
                selected_var = variables[selected_var_idx]
                
                # Sample value (simplified)
                key, val_key = random.split(key)
                intervention_value = float(random.uniform(
                    val_key, (),
                    minval=intervention_range[0],
                    maxval=intervention_range[1]
                ))
            else:
                # Random intervention
                key, var_key, val_key = random.split(key, 3)
                
                # Select non-target variable
                non_target_vars = [v for v in variables if v != target_var]
                var_idx = int(random.randint(var_key, (), 0, len(non_target_vars)))
                selected_var = non_target_vars[var_idx]
                selected_var_idx = variables.index(selected_var)
                
                # Random value
                intervention_value = float(random.uniform(
                    val_key, (),
                    minval=intervention_range[0],
                    maxval=intervention_range[1]
                ))
                
            # Apply intervention
            intervention = create_perfect_intervention(
                targets=frozenset([selected_var]),
                values={selected_var: intervention_value}
            )
            
            key, int_key = random.split(key)
            int_samples = sample_with_intervention(scm, intervention, 100, seed=int(int_key[0]))
            
            # Convert to dict format
            int_data = {}
            for var in variables:
                int_data[var] = jnp.array([s['values'][var] for s in int_samples])
                
            # Compute outcome
            outcome_value = float(jnp.mean(int_data[target_var]))
            
            # Update best value
            if outcome_value < best_value:  # Assuming minimization
                best_value = outcome_value
                
            # Update cumulative data
            for var in variables:
                cumulative_samples[var] = jnp.concatenate([
                    cumulative_samples[var], int_data[var]
                ])
                
            # Compute updated marginals if using surrogate
            current_marginals = {}
            if self.surrogate_fn:
                key, surrogate_key = random.split(key)
                current_marginals = self._compute_marginals_with_surrogate(
                    cumulative_samples, target_var, variables, surrogate_key
                )
                
            # Simple reward
            reward = initial_value - outcome_value
            
            # Update state
            current_state = current_state.at[selected_var_idx, 0].set(intervention_value)
            
            # Create step result
            step_result = StepResult(
                step=step,
                intervention={selected_var: intervention_value},
                outcome_value=outcome_value,
                marginals=current_marginals,
                uncertainty=self._compute_uncertainty(current_marginals) if current_marginals else 1.0,
                reward=reward,
                computation_time=time.time() - step_start
            )
            
            learning_history.append(step_result)
            
        # Compute final metrics
        final_value = learning_history[-1].outcome_value
        final_marginals = learning_history[-1].marginals
        
        # Structure metrics
        if final_marginals:
            final_f1 = compute_f1_score_from_marginals(final_marginals, true_parents)
            final_shd = compute_shd_from_marginals(final_marginals, true_parents)
        else:
            final_f1 = 0.0
            final_shd = len(true_parents)
            
        final_metrics = {
            'initial_value': initial_value,
            'final_value': final_value,
            'best_value': best_value,
            'improvement': initial_value - final_value,
            'n_interventions': len(learning_history) - 1,
            'final_f1': final_f1,
            'final_shd': final_shd,
            'uses_surrogate': self.surrogate_fn is not None,
            'uses_acquisition': self.acquisition_fn is not None
        }
        
        return ExperimentResult(
            learning_history=learning_history,
            final_metrics=final_metrics,
            metadata={
                'method': self.name,
                'scm_info': {
                    'n_variables': n_vars,
                    'target': target_var,
                    'true_parents': true_parents
                },
                'run_idx': run_idx,
                'seed': seed,
                'surrogate_checkpoint': str(self.surrogate_checkpoint) if self.surrogate_checkpoint else None,
                'acquisition_checkpoint': str(self.acquisition_checkpoint) if self.acquisition_checkpoint else None
            },
            success=True,
            total_time=time.time() - start_time
        )
        
    def _compute_marginals_with_surrogate(
        self,
        samples_data: Dict[str, jnp.ndarray],
        target_var: str,
        variables: List[str],
        key: jax.random.PRNGKey
    ) -> Dict[str, float]:
        """Compute marginal parent probabilities using surrogate."""
        try:
            # Prepare data
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
            
            # Extract parent probabilities
            target_idx = variables.index(target_var)
            marginal_probs = {}
            
            if 'parent_probs' in output:
                parent_probs = output['parent_probs'][0]  # Remove batch
                
                for i, var in enumerate(variables):
                    if var != target_var:
                        prob = float(parent_probs[target_idx, i])
                        marginal_probs[var] = prob
                        
            return marginal_probs
            
        except Exception as e:
            logger.warning(f"Failed to compute marginals: {e}")
            return {}
            
    def _compute_uncertainty(self, marginals: Dict[str, float]) -> float:
        """Compute uncertainty from marginal probabilities."""
        if not marginals:
            return 1.0
            
        # Binary entropy for each variable
        probs = jnp.array(list(marginals.values()))
        probs = jnp.clip(probs, 1e-10, 1.0)
        
        entropies = -probs * jnp.log2(probs) - (1 - probs) * jnp.log2(1 - probs)
        
        return float(jnp.mean(entropies))