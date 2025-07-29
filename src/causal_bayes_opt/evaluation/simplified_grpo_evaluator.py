"""
Simplified GRPO Evaluator

Clean evaluator for GRPO policies trained with SimplifiedGRPOTrainer.
Works directly with the new checkpoint format without adapters.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import pickle
import time

import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk

from .base_evaluator import BaseEvaluator
from .result_types import ExperimentResult, StepResult
from ..data_structures.scm import get_target, get_variables, get_parents
from ..mechanisms.linear import sample_from_linear_scm
from ..environments.sampling import sample_with_intervention
from ..interventions.handlers import create_perfect_intervention
from ..acquisition.enhanced_policy_network import create_enhanced_policy_for_grpo
# from ..jax_native.state import unroll_state_trajectories  # Not used

logger = logging.getLogger(__name__)


class SimplifiedGRPOEvaluator(BaseEvaluator):
    """
    Clean evaluator for GRPO policies from SimplifiedGRPOTrainer.
    
    Features:
    - Direct checkpoint loading
    - Clean policy recreation
    - Optional surrogate support
    - Simple state tracking
    """
    
    def __init__(self, checkpoint_path: Path, name: Optional[str] = None):
        """
        Initialize GRPO evaluator.
        
        Args:
            checkpoint_path: Path to checkpoint file (not directory)
            name: Optional custom name
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")
            
        # Generate name from checkpoint
        if name is None:
            name = f"GRPO_{checkpoint_path.stem}"
            
        super().__init__(name=name)
        
        self.checkpoint_path = checkpoint_path
        self.policy_params = None
        self.policy_fn = None
        self.policy_config = None
        self.optimization_direction = "MINIMIZE"
        
    def initialize(self) -> None:
        """Load checkpoint and initialize policy."""
        logger.info(f"Initializing SimplifiedGRPOEvaluator from {self.checkpoint_path}")
        
        try:
            # Load checkpoint
            with open(self.checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
                
            # Extract components
            self.policy_params = checkpoint.get("params")
            if self.policy_params is None:
                raise ValueError("No params found in checkpoint")
                
            config = checkpoint.get("config", {})
            self.optimization_direction = config.get("optimization_direction", "MINIMIZE")
            self.architecture_level = config.get("architecture_level", "simplified")
            self.reward_weights = config.get("reward_weights", {})
            
            # Store metadata
            self.metadata = checkpoint.get("metadata", {})
            
            logger.info(f"Loaded checkpoint: optimization={self.optimization_direction}, "
                       f"architecture={self.architecture_level}")
            
            # Policy will be created per-SCM in evaluate_single_run
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize evaluator: {e}")
            raise
            
    def evaluate_single_run(
        self,
        scm: Any,
        config: Dict[str, Any],
        seed: int,
        run_idx: int = 0
    ) -> ExperimentResult:
        """
        Evaluate GRPO policy on a single SCM.
        
        Args:
            scm: Structural Causal Model
            config: Evaluation configuration
            seed: Random seed
            run_idx: Run index
            
        Returns:
            ExperimentResult with trajectory
        """
        start_time = time.time()
        
        # Extract SCM info
        target_var = get_target(scm)
        variables = list(get_variables(scm))
        true_parents = list(get_parents(scm, target_var))
        n_vars = len(variables)
        
        # Create policy for this SCM
        policy_fn, _ = create_enhanced_policy_for_grpo(
            variables=variables,
            target_variable=target_var,
            architecture_level=self.architecture_level,
            performance_mode="balanced"
        )
        
        # Transform for evaluation
        self.policy_fn = hk.without_apply_rng(hk.transform(policy_fn))
        
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
        
        # Simple state representation
        current_state = jnp.zeros((n_vars, 32))  # Simple feature representation
        
        # Store initial state
        learning_history.append(StepResult(
            step=0,
            intervention={},
            outcome_value=initial_value,
            marginals={var: 0.0 for var in variables if var != target_var},
            uncertainty=1.0,
            reward=0.0,
            computation_time=0.0
        ))
        
        # Run intervention loop
        for step in range(1, max_interventions + 1):
            step_start = time.time()
            
            # Get policy action
            key, policy_key = random.split(key)
            policy_output = self.policy_fn.apply(self.policy_params, current_state)
            
            # Select intervention variable
            var_logits = policy_output.get('variable_logits', policy_output.get('intervention_logits'))
            key, var_key = random.split(key)
            var_probs = jax.nn.softmax(var_logits)
            selected_var_idx = random.categorical(var_key, var_probs)
            selected_var = variables[int(selected_var_idx)]
            
            # Sample intervention value (simplified)
            key, val_key = random.split(key)
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
            if self.optimization_direction == "MINIMIZE":
                if outcome_value < best_value:
                    best_value = outcome_value
            else:
                if outcome_value > best_value:
                    best_value = outcome_value
                    
            # Simple reward
            if self.optimization_direction == "MINIMIZE":
                reward = initial_value - outcome_value
            else:
                reward = outcome_value - initial_value
                
            # Update state (simplified)
            current_state = current_state.at[selected_var_idx, 0].set(intervention_value)
            
            # Create step result
            step_result = StepResult(
                step=step,
                intervention={selected_var: intervention_value},
                outcome_value=outcome_value,
                marginals={var: 0.0 for var in variables if var != target_var},  # No structure learning
                uncertainty=1.0,
                reward=reward,
                computation_time=time.time() - step_start
            )
            
            learning_history.append(step_result)
            
        # Compute final metrics
        final_value = learning_history[-1].outcome_value
        
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
            'optimization_direction': self.optimization_direction
        }
        
        # Add structure metrics (empty for now)
        final_metrics['final_f1'] = 0.0
        final_metrics['final_shd'] = len(true_parents)  # Max SHD
        
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
                'checkpoint': str(self.checkpoint_path)
            },
            success=True,
            total_time=time.time() - start_time
        )