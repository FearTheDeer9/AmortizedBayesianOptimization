"""
Universal ACBO Evaluator - Generic evaluation loop for any acquisition method.

This module implements the correct architecture where:
1. Models are simple functions (surrogate_fn, acquisition_fn)
2. Evaluation logic is generic and method-agnostic
3. No wrappers or adapters needed

The evaluator runs the standard ACBO loop:
1. Sample observational data
2. For each step:
   - Get structure prediction from surrogate
   - Get intervention from acquisition function
   - Execute intervention and observe outcome
   - Update buffer and metrics
3. Return comprehensive results
"""

import logging
import time
from typing import Callable, Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import jax.numpy as jnp

from ..training.three_channel_converter import buffer_to_three_channel_tensor
from ..data_structures.buffer import ExperienceBuffer
from ..data_structures.sample import create_sample, get_values
from ..data_structures.scm import get_variables, get_target, get_parents
from ..mechanisms.linear import sample_from_linear_scm
from ..environments.sampling import sample_with_intervention
from ..interventions.handlers import create_perfect_intervention
from ..acquisition.clean_rewards import compute_clean_reward
from ..analysis.trajectory_metrics import (
    compute_f1_score_from_marginals,
    compute_shd_from_marginals
)
from ..avici_integration.parent_set.posterior import (
    ParentSetPosterior,
    get_marginal_parent_probabilities
)

logger = logging.getLogger(__name__)


# Type aliases for clarity
Tensor = jnp.ndarray  # [T, n_vars, 3] in 3-channel format
Posterior = Optional[Dict[str, Any]]  # Structure prediction (optional)
Intervention = Dict[str, Any]  # {'targets': set, 'values': dict}

# Function signatures for models
SurrogateFn = Callable[[Tensor, str], Posterior]
AcquisitionFn = Callable[[Tensor, Posterior, str, List[str]], Intervention]


@dataclass
class StepMetrics:
    """Metrics for a single intervention step."""
    step: int
    intervention: Intervention
    outcome_value: float
    reward: float
    best_value: float
    time_elapsed: float
    marginals: Optional[Dict[str, float]] = None
    entropy: Optional[float] = None


@dataclass
class EvaluationResult:
    """Complete evaluation results."""
    history: List[StepMetrics]
    final_metrics: Dict[str, Any]
    scm_info: Dict[str, Any]
    total_time: float
    success: bool = True
    error_message: Optional[str] = None


class UniversalACBOEvaluator:
    """
    Generic ACBO evaluation loop that works with any surrogate/acquisition pair.
    
    This evaluator implements the correct architecture where models are just
    functions and the evaluation loop is generic.
    """
    
    def __init__(self, name: str = "UniversalACBO"):
        """
        Initialize universal evaluator.
        
        Args:
            name: Name for logging and identification
        """
        self.name = name
        
    def evaluate(self,
                 acquisition_fn: AcquisitionFn,
                 scm: Any,
                 config: Dict[str, Any],
                 surrogate_fn: Optional[SurrogateFn] = None,
                 surrogate_update_fn: Optional[Callable] = None,
                 surrogate_params: Optional[Any] = None,
                 surrogate_opt_state: Optional[Any] = None,
                 seed: int = 42) -> EvaluationResult:
        """
        Run ACBO evaluation loop with provided functions.
        
        This is the core evaluation logic that:
        1. Samples initial observational data
        2. Runs the intervention loop
        3. Computes comprehensive metrics
        4. Optionally updates surrogate during evaluation (active learning)
        
        Args:
            acquisition_fn: Function that selects interventions
            scm: Structural Causal Model to evaluate on
            config: Evaluation configuration
            surrogate_fn: Optional function for structure prediction
            surrogate_update_fn: Optional function to update surrogate online
            surrogate_params: Initial surrogate parameters (for active learning)
            surrogate_opt_state: Initial optimizer state (for active learning)
            seed: Random seed for reproducibility
            
        Returns:
            EvaluationResult with complete trajectory and metrics
        """
        start_time = time.time()
        
        try:
            # Extract configuration
            n_observational = config.get('n_observational', 100)
            max_interventions = config.get('max_interventions', 10)
            n_intervention_samples = config.get('n_intervention_samples', 100)
            optimization_direction = config.get('optimization_direction', 'MINIMIZE')
            
            # Get SCM information
            variables = list(get_variables(scm))
            target_var = get_target(scm)
            true_parents = set(get_parents(scm, target_var))
            
            logger.info(
                f"Evaluating {self.name} on SCM with {len(variables)} variables, "
                f"target='{target_var}', true_parents={true_parents}"
            )
            
            # Initialize buffer and history
            buffer = ExperienceBuffer()
            history = []
            
            # Track whether active learning is being used
            active_learning_enabled = (
                surrogate_update_fn is not None and 
                surrogate_params is not None
            )
            
            # Sample observational data
            obs_samples = sample_from_linear_scm(scm, n_observational, seed=seed)
            for sample in obs_samples:
                buffer.add_observation(sample)
            
            # Compute initial metrics
            initial_value = self._compute_target_mean(buffer, target_var)
            best_value = initial_value
            
            # Store initial state (no intervention)
            history.append(StepMetrics(
                step=0,
                intervention={},
                outcome_value=initial_value,
                reward=0.0,
                best_value=best_value,
                time_elapsed=0.0
            ))
            
            # Intervention loop
            for step in range(1, max_interventions + 1):
                step_start = time.time()
                
                # Convert buffer to tensor
                tensor, var_order = buffer_to_three_channel_tensor(
                    buffer, target_var, standardize=True
                )
                
                # Get structure prediction (if surrogate provided)
                posterior = None
                if surrogate_fn is not None:
                    try:
                        posterior = surrogate_fn(tensor, target_var)
                    except Exception as e:
                        logger.warning(f"Surrogate prediction failed: {e}")
                
                # Get intervention from acquisition function
                # Pass variable order to help acquisition functions
                intervention = acquisition_fn(tensor, posterior, target_var, var_order)
                
                # Validate intervention
                if not self._validate_intervention(intervention, variables, target_var):
                    logger.warning(f"Invalid intervention at step {step}: {intervention}")
                    continue
                
                # Execute intervention
                # Convert intervention to proper format
                intervention_obj = create_perfect_intervention(
                    targets=intervention['targets'],
                    values=intervention['values']
                )
                intervention_samples = sample_with_intervention(
                    scm, intervention_obj, n_intervention_samples, seed=seed + step
                )
                
                # Add to buffer
                for sample in intervention_samples:
                    buffer.add_intervention(intervention_obj, sample)
                
                # Update surrogate if active learning is enabled
                if surrogate_update_fn is not None and surrogate_params is not None:
                    try:
                        # Get all samples from buffer for update
                        all_samples = buffer.get_all_samples()
                        
                        # Update surrogate with new data
                        surrogate_params, surrogate_opt_state, update_metrics = surrogate_update_fn(
                            surrogate_params, 
                            surrogate_opt_state,
                            posterior,
                            all_samples,
                            var_order,
                            target_var
                        )
                        
                        # Update the surrogate function closure to use new params
                        def updated_surrogate_fn(tensor, target):
                            # This assumes the surrogate uses the updated params
                            # The actual implementation depends on the surrogate type
                            return surrogate_fn(tensor, target)
                        
                        logger.debug(f"Surrogate updated at step {step}, metrics: {update_metrics}")
                    except Exception as e:
                        logger.warning(f"Surrogate update failed at step {step}: {e}")
                
                # Compute outcome and reward
                outcome_value = self._compute_target_mean_from_samples(
                    intervention_samples, target_var
                )
                
                # Compute reward using clean reward system
                reward_info = compute_clean_reward(
                    buffer_before=buffer,
                    intervention=intervention,
                    outcome=intervention_samples[0],
                    target_variable=target_var,
                    config={'optimization_direction': optimization_direction}
                )
                
                # Update best value
                if optimization_direction == 'MINIMIZE':
                    if outcome_value < best_value:
                        best_value = outcome_value
                else:
                    if outcome_value > best_value:
                        best_value = outcome_value
                
                # Extract marginals if posterior available
                marginals = None
                entropy = None
                if posterior:
                    if isinstance(posterior, dict):
                        # Handle dictionary format
                        if 'marginal_parent_probs' in posterior:
                            marginals = dict(posterior['marginal_parent_probs'])
                        if 'entropy' in posterior:
                            entropy = float(posterior['entropy'])
                    elif isinstance(posterior, ParentSetPosterior):
                        # Handle ParentSetPosterior objects
                        marginals = get_marginal_parent_probabilities(posterior, var_order)
                        entropy = float(posterior.uncertainty)
                
                # Record step metrics
                history.append(StepMetrics(
                    step=step,
                    intervention=intervention,
                    outcome_value=outcome_value,
                    reward=reward_info['total'],
                    best_value=best_value,
                    time_elapsed=time.time() - step_start,
                    marginals=marginals,
                    entropy=entropy
                ))
                
                logger.debug(
                    f"Step {step}: intervened on {intervention.get('targets', {})}, "
                    f"outcome={outcome_value:.3f}, reward={reward_info['total']:.3f}"
                )
            
            # Compute final metrics
            final_metrics = self._compute_final_metrics(
                history, initial_value, best_value, true_parents, optimization_direction
            )
            
            # Add active learning info
            final_metrics['active_learning_enabled'] = active_learning_enabled
            final_metrics['has_surrogate'] = surrogate_fn is not None
            
            # Create SCM info
            scm_info = {
                'n_variables': len(variables),
                'target': target_var,
                'true_parents': list(true_parents),
                'variables': variables
            }
            
            return EvaluationResult(
                history=history,
                final_metrics=final_metrics,
                scm_info=scm_info,
                total_time=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return EvaluationResult(
                history=[],
                final_metrics={},
                scm_info={},
                total_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _validate_intervention(self, 
                              intervention: Intervention,
                              variables: List[str],
                              target_var: str) -> bool:
        """Validate that intervention is well-formed."""
        if not isinstance(intervention, dict):
            return False
        
        targets = intervention.get('targets', set())
        values = intervention.get('values', {})
        
        # Check targets exist and don't include target variable
        for var in targets:
            if var not in variables:
                return False
            if var == target_var:
                return False
        
        # Check values are provided for all targets
        for var in targets:
            if var not in values:
                return False
        
        return True
    
    def _compute_target_mean(self, buffer: ExperienceBuffer, target_var: str) -> float:
        """Compute mean target value from buffer."""
        all_samples = buffer.get_all_samples()
        target_values = []
        
        for sample in all_samples[-100:]:  # Use recent samples
            values = get_values(sample)
            if target_var in values:
                target_values.append(float(values[target_var]))
        
        if target_values:
            return float(jnp.mean(jnp.array(target_values)))
        return 0.0
    
    def _compute_target_mean_from_samples(self, samples: List[Any], target_var: str) -> float:
        """Compute mean target value from sample list."""
        target_values = []
        
        for sample in samples:
            values = get_values(sample)
            if target_var in values:
                target_values.append(float(values[target_var]))
        
        if target_values:
            return float(jnp.mean(jnp.array(target_values)))
        return 0.0
    
    def _compute_final_metrics(self,
                              history: List[StepMetrics],
                              initial_value: float,
                              best_value: float,
                              true_parents: set,
                              optimization_direction: str) -> Dict[str, Any]:
        """Compute comprehensive final metrics."""
        final_step = history[-1] if history else None
        
        # Optimization metrics
        if optimization_direction == 'MINIMIZE':
            improvement = initial_value - best_value
        else:
            improvement = best_value - initial_value
        
        metrics = {
            'initial_value': initial_value,
            'final_value': final_step.outcome_value if final_step else initial_value,
            'best_value': best_value,
            'improvement': improvement,
            'n_interventions': len(history) - 1,  # Exclude initial state
            'optimization_direction': optimization_direction
        }
        
        # Structure learning metrics (if marginals available)
        if final_step and final_step.marginals:
            pred_parents = {
                var for var, prob in final_step.marginals.items() 
                if prob > 0.5
            }
            metrics['final_f1'] = compute_f1_score_from_marginals(
                final_step.marginals, true_parents
            )
            metrics['final_shd'] = compute_shd_from_marginals(
                final_step.marginals, true_parents
            )
            metrics['predicted_parents'] = list(pred_parents)
        else:
            metrics['final_f1'] = 0.0
            metrics['final_shd'] = len(true_parents)  # Worst case
            metrics['predicted_parents'] = []
        
        # Trajectory statistics
        rewards = [step.reward for step in history[1:]]  # Skip initial
        if rewards:
            metrics['mean_reward'] = float(jnp.mean(jnp.array(rewards)))
            metrics['total_reward'] = float(jnp.sum(jnp.array(rewards)))
        else:
            metrics['mean_reward'] = 0.0
            metrics['total_reward'] = 0.0
        
        # Compute mean trajectory value
        trajectory_values = [step.outcome_value for step in history]
        if trajectory_values:
            metrics['mean_trajectory_value'] = float(jnp.mean(jnp.array(trajectory_values)))
        else:
            metrics['mean_trajectory_value'] = initial_value
        
        return metrics


def create_universal_evaluator() -> UniversalACBOEvaluator:
    """Factory function to create universal evaluator."""
    return UniversalACBOEvaluator()