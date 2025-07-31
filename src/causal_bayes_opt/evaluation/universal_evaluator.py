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
SurrogateFn = Callable[[Tensor, str, List[str]], Posterior]  # Added variables parameter
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
    predicted_parents: Optional[List[str]] = None  # Parents predicted by surrogate
    prediction_confidence: Optional[float] = None  # Average confidence of predictions


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
                
                # Log tensor information
                logger.info(f"\nStep {step} - Tensor Input:")
                logger.info(f"  Tensor shape: {tensor.shape}")
                logger.info(f"  Variable order: {var_order}")
                logger.info(f"  Last timestep values:")
                last_timestep = tensor[-1]  # [n_vars, 3]
                for i, var in enumerate(var_order):
                    value = last_timestep[i, 0]
                    is_intervened = last_timestep[i, 1]
                    is_target = last_timestep[i, 2]
                    logger.info(f"    {var}: value={value:.3f}, intervened={is_intervened:.0f}, is_target={is_target:.0f}")
                
                # Get structure prediction (if surrogate provided)
                posterior = None
                if surrogate_fn is not None:
                    try:
                        logger.info(f"\n  Calling surrogate with tensor shape {tensor.shape} and variables {var_order}...")
                        posterior = surrogate_fn(tensor, target_var, var_order)
                        logger.info(f"  Surrogate returned: {type(posterior)}")
                    except Exception as e:
                        logger.warning(f"Surrogate prediction failed: {e}")
                
                # Get intervention from acquisition function
                # Pass variable order to help acquisition functions
                logger.info(f"\n  Calling acquisition function...")
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
                predicted_parents = None
                prediction_confidence = None
                
                if posterior:
                    if isinstance(posterior, dict):
                        # Handle dictionary format
                        if 'marginal_parent_probs' in posterior:
                            marginals = dict(posterior['marginal_parent_probs'])
                        if 'entropy' in posterior:
                            entropy = float(posterior['entropy'])
                    elif isinstance(posterior, ParentSetPosterior):
                        # Handle ParentSetPosterior objects
                        # Check metadata first (BC surrogate stores marginals there)
                        if hasattr(posterior, 'metadata') and 'marginal_parent_probs' in posterior.metadata:
                            raw_marginals = dict(posterior.metadata['marginal_parent_probs'])
                            
                            # Map from surrogate variable names to actual SCM variable names
                            marginals = {}
                            for var_name, prob in raw_marginals.items():
                                # First check if the variable name is already in var_order
                                if var_name in var_order:
                                    marginals[var_name] = prob
                                # If surrogate uses zero-indexed names (X0, X1, etc), map to actual names
                                elif var_name.startswith('X') and var_name[1:].isdigit():
                                    idx = int(var_name[1:])
                                    if idx < len(var_order):
                                        actual_name = var_order[idx]
                                        marginals[actual_name] = prob
                            
                            logger.info(f"Variable mapping: {var_order}")
                            logger.info(f"Raw marginals: {raw_marginals}")
                            logger.info(f"Mapped marginals: {marginals}")
                        else:
                            marginals = get_marginal_parent_probabilities(posterior, var_order)
                        entropy = float(posterior.uncertainty)
                
                # Extract predicted parents and confidence
                if marginals:
                    # Log detailed marginals for debugging
                    logger.info(f"\n  Step {step} - Detailed marginal probabilities:")
                    logger.info(f"  Method: {self.name}")
                    for var in sorted(marginals.keys()):
                        prob = marginals[var]
                        is_parent = var in true_parents
                        logger.info(f"    {var}: {prob:.6f} (actual parent: {is_parent})")
                    
                    threshold = 0.5
                    predicted_parents = [var for var, prob in marginals.items() 
                                       if var != target_var and prob > threshold]
                    
                    # Calculate average confidence (distance from 0.5)
                    confidences = [abs(prob - 0.5) * 2 for var, prob in marginals.items() 
                                 if var != target_var]
                    prediction_confidence = float(jnp.mean(jnp.array(confidences))) if confidences else 0.0
                    
                    logger.info(f"  Predicted parents (step {step}): {predicted_parents}")
                    logger.info(f"  Prediction confidence: {prediction_confidence:.3f}")
                
                # Record step metrics
                history.append(StepMetrics(
                    step=step,
                    intervention=intervention,
                    outcome_value=outcome_value,
                    reward=reward_info['total'],
                    best_value=best_value,
                    time_elapsed=time.time() - step_start,
                    marginals=marginals,
                    entropy=entropy,
                    predicted_parents=predicted_parents,
                    prediction_confidence=prediction_confidence
                ))
                
                # Detailed logging for debugging
                logger.info(
                    f"\n{'='*60}\n"
                    f"Step {step} Summary:\n"
                    f"  Intervention: {list(intervention.get('targets', []))} = "
                    f"{[intervention['values'][t] for t in intervention.get('targets', [])]}\n"
                    f"  Outcome value: {outcome_value:.3f}\n"
                    f"  Best value so far: {best_value:.3f}\n"
                    f"  Buffer size: {len(buffer._observations) + len(buffer._interventions)}\n"
                )
                
                if marginals:
                    logger.info("  Surrogate predictions (marginal probabilities):")
                    for var, prob in sorted(marginals.items()):
                        is_true_parent = var in true_parents
                        logger.info(f"    {var}: {prob:.3f} {'✓ (true parent)' if is_true_parent else ''}")
                    
                    # Log F1 calculation details
                    threshold = 0.5
                    predicted_parents = {var for var, prob in marginals.items() if prob > threshold}
                    true_positives = len(predicted_parents & true_parents)
                    false_positives = len(predicted_parents - true_parents)
                    false_negatives = len(true_parents - predicted_parents)
                    
                    logger.info(f"\n  F1 Calculation (threshold={threshold}):")
                    logger.info(f"    True parents: {sorted(true_parents)}")
                    logger.info(f"    Predicted parents (prob > {threshold}): {sorted(predicted_parents)}")
                    logger.info(f"    True positives: {true_positives}")
                    logger.info(f"    False positives: {false_positives}")
                    logger.info(f"    False negatives: {false_negatives}")
                    
                    if len(predicted_parents) > 0 and len(true_parents) > 0:
                        precision = true_positives / len(predicted_parents)
                        recall = true_positives / len(true_parents)
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        logger.info(f"    Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
                    else:
                        logger.info(f"    F1: 0.0 (no predictions or no true parents)")
                else:
                    logger.info("  No surrogate predictions available")
                
                logger.info(f"{'='*60}\n")
            
            # Compute final metrics
            logger.info(f"\n{'='*70}")
            logger.info("Computing final metrics...")
            logger.info(f"  Initial value: {initial_value:.3f}")
            logger.info(f"  Best value: {best_value:.3f}")
            logger.info(f"  Improvement (best - initial): {initial_value - best_value:.3f}")
            
            # Calculate mean trajectory value
            trajectory_values = [step.outcome_value for step in history]
            mean_traj = float(jnp.mean(jnp.array(trajectory_values))) if trajectory_values else initial_value
            logger.info(f"  Mean trajectory value: {mean_traj:.3f}")
            logger.info(f"  Trajectory: {[f'{v:.2f}' for v in trajectory_values[:10]]}...")
            
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
        
        # Add prediction history analysis
        prediction_history = []
        confidence_history = []
        for step in history[1:]:  # Skip initial state
            if step.predicted_parents is not None:
                prediction_history.append({
                    'step': step.step,
                    'predicted': step.predicted_parents,
                    'confidence': step.prediction_confidence
                })
                confidence_history.append(step.prediction_confidence)
        
        if prediction_history:
            metrics['prediction_history'] = prediction_history
            metrics['mean_prediction_confidence'] = float(jnp.mean(jnp.array(confidence_history)))
            
            # Check prediction consistency
            all_predictions = [tuple(sorted(p['predicted'])) for p in prediction_history]
            unique_predictions = len(set(all_predictions))
            metrics['prediction_consistency'] = 1.0 / unique_predictions if unique_predictions > 0 else 1.0
            logger.info(f"\nPrediction diversity: {unique_predictions} unique predictions across {len(prediction_history)} steps")
        
        # Structure learning metrics (if marginals available)
        if final_step and final_step.marginals:
            pred_parents = {
                var for var, prob in final_step.marginals.items() 
                if prob > 0.5
            }
            
            logger.info("\nFinal F1 Score Calculation:")
            logger.info(f"  Final marginal probabilities:")
            for var, prob in sorted(final_step.marginals.items()):
                logger.info(f"    {var}: {prob:.3f}")
            
            metrics['final_f1'] = compute_f1_score_from_marginals(
                final_step.marginals, true_parents
            )
            metrics['final_shd'] = compute_shd_from_marginals(
                final_step.marginals, true_parents
            )
            metrics['predicted_parents'] = list(pred_parents)
            
            logger.info(f"  Final F1 score: {metrics['final_f1']:.3f}")
            logger.info(f"  Final SHD: {metrics['final_shd']}")
            logger.info(f"  Predicted parents: {sorted(pred_parents)}")
            logger.info(f"  True parents: {sorted(true_parents)}")
        else:
            logger.info("\nNo marginal probabilities available - F1 score = 0.0")
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