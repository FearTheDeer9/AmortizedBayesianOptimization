"""
Metrics Collector for ACBO Comparison Framework

This module provides standardized metrics collection and trajectory analysis
for ACBO experiments. It ensures consistent metric computation across all methods.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import jax.numpy as jnp
import numpy as onp
import pyrsistent as pyr

# Import standardized interfaces
from scripts.notebooks.pipeline_interfaces import (
    MetricNames, TrajectoryData, MethodResult
)
from scripts.notebooks.interface_adapters import LegacyMetricAdapter

from causal_bayes_opt.data_structures.scm import get_target, get_parents, get_variables
from causal_bayes_opt.analysis.trajectory_metrics import (
    compute_trajectory_metrics, analyze_convergence_trajectory,
    extract_learning_curves, compute_intervention_efficiency
)
from .structure_metrics_helper import (
    compute_f1_from_marginals, compute_parent_probability,
    compute_shd_from_marginals, add_structure_metrics_to_result
)

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collector for standardized ACBO experiment metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.collected_metrics = {}
        self.trajectory_data = {}
    
    def collect_run_metrics(self, result: Dict[str, Any], scm: pyr.PMap, 
                           method_name: str, run_idx: int, scm_idx: int) -> MethodResult:
        """
        Collect comprehensive metrics from a single experimental run.
        
        Args:
            result: Raw result from method execution
            scm: The SCM used in the experiment
            method_name: Name of the method
            run_idx: Run index for this method
            scm_idx: SCM index
            
        Returns:
            Standardized MethodResult object
        """
        try:
            # Convert legacy result to standardized format
            method_result = MethodResult.from_legacy_result(
                result, method_name, run_idx, scm_idx
            )
            
            # Enhance with additional metrics
            self._enhance_method_result(method_result, result, scm)
            
            # Store trajectory data for aggregation
            if method_result.trajectory:
                trajectory_key = f"{method_name}_{run_idx}_{scm_idx}"
                self.trajectory_data[trajectory_key] = method_result.trajectory.to_dict()
            
            logger.debug(f"Collected metrics for {method_name} run {run_idx}")
            return method_result
            
        except Exception as e:
            logger.error(f"Failed to collect metrics for {method_name}: {e}")
            # Return error result
            error_result = MethodResult(
                method_name=method_name,
                run_idx=run_idx,
                scm_idx=scm_idx,
                final_target_value=0.0,
                intervention_count=0,
                error=str(e)
            )
            return error_result
    
    def _enhance_method_result(self, method_result: MethodResult, 
                              raw_result: Dict[str, Any], scm: pyr.PMap) -> None:
        """Enhance method result with additional computed metrics."""
        # Compute structure metrics if not already present
        if method_result.structure_metrics is None and method_result.final_marginals:
            target = get_target(scm)
            true_parents = list(get_parents(scm, target)) if target else []
            
            if true_parents and method_result.final_marginals:
                structure_metrics = self._compute_structure_metrics_from_marginals(
                    method_result.final_marginals, true_parents, target
                )
                method_result.structure_metrics = structure_metrics
        
        # Extract execution time if available
        if 'execution_time' in raw_result:
            method_result.execution_time = raw_result['execution_time']
        elif 'duration' in raw_result:
            method_result.execution_time = raw_result['duration']
    
    def _compute_structure_metrics_from_marginals(self, marginals: Dict[str, float],
                                                 true_parents: List[str], 
                                                 target: str) -> Dict[str, float]:
        """Compute structure metrics from marginal probabilities."""
        from .structure_metrics_helper import (
            compute_f1_from_marginals, compute_shd_from_marginals, 
            compute_parent_probability
        )
        
        f1, precision, recall = compute_f1_from_marginals(marginals, true_parents, target)
        shd = compute_shd_from_marginals(marginals, true_parents, target)
        parent_prob = compute_parent_probability(marginals, true_parents)
        
        return {
            MetricNames.F1_SCORE: f1,
            'precision': precision,
            'recall': recall,
            MetricNames.SHD: shd,
            MetricNames.TRUE_PARENT_LIKELIHOOD: parent_prob
        }
    
    def aggregate_method_metrics(self, method_name: str, 
                               all_results: List[MethodResult]) -> Dict[str, Any]:
        """
        Aggregate metrics across all runs for a specific method.
        
        Args:
            method_name: Name of the method
            all_results: List of MethodResult objects for this method
            
        Returns:
            Aggregated metrics using standard names
        """
        if not all_results:
            return {}
        
        valid_results = [r for r in all_results if r and not (hasattr(r, 'error') and r.error) and not (isinstance(r, dict) and r.get('error'))]
        
        if not valid_results:
            logger.warning(f"No valid results for {method_name}")
            return {'method_name': method_name, 'valid_runs': 0}
        
        aggregated = {
            'method_name': method_name,
            'total_runs': len(all_results),
            'valid_runs': len(valid_results),
            'success_rate': len(valid_results) / len(all_results) if all_results else 0
        }
        
        # Aggregate core metrics using standard names
        # Handle both dict and object results
        target_values = []
        intervention_counts = []
        execution_times = []
        
        for r in valid_results:
            if isinstance(r, dict):
                target_values.append(r.get('final_target_value', 0))
                intervention_counts.append(r.get('intervention_count', 0))
                exec_time = r.get('execution_time', 0)
                if exec_time > 0:
                    execution_times.append(exec_time)
            else:
                target_values.append(getattr(r, 'final_target_value', 0))
                intervention_counts.append(getattr(r, 'intervention_count', 0))
                exec_time = getattr(r, 'execution_time', 0)
                if exec_time > 0:
                    execution_times.append(exec_time)
        
        if target_values:
            aggregated.update({
                f'{MetricNames.TARGET_VALUE}_mean': float(onp.mean(target_values)),
                f'{MetricNames.TARGET_VALUE}_std': float(onp.std(target_values)),
                f'{MetricNames.TARGET_VALUE}_min': float(onp.min(target_values)),
                f'{MetricNames.TARGET_VALUE}_max': float(onp.max(target_values))
            })
        
        if intervention_counts:
            aggregated.update({
                f'{MetricNames.INTERVENTION_COUNT}_mean': float(onp.mean(intervention_counts)),
                f'{MetricNames.INTERVENTION_COUNT}_std': float(onp.std(intervention_counts))
            })
        
        if execution_times:
            aggregated['execution_time_mean'] = float(onp.mean(execution_times))
            aggregated['execution_time_std'] = float(onp.std(execution_times))
        
        # Aggregate structure metrics if available
        self._aggregate_structure_metrics(aggregated, valid_results)
        
        # Aggregate trajectory metrics
        aggregated.update(self._aggregate_trajectory_metrics_new(method_name, valid_results))
        
        logger.info(f"Aggregated metrics for {method_name}: {aggregated['valid_runs']} valid runs")
        return aggregated
    
    def _aggregate_structure_metrics(self, aggregated: Dict[str, Any], 
                                   valid_results: List[Any]) -> None:
        """Aggregate structure learning metrics."""
        f1_scores = []
        shd_values = []
        parent_probs = []
        
        for result in valid_results:
            if isinstance(result, dict):
                structure_metrics = result.get('structure_metrics', {})
                if MetricNames.F1_SCORE in structure_metrics:
                    f1_scores.append(structure_metrics[MetricNames.F1_SCORE])
                if MetricNames.SHD in structure_metrics:
                    shd_values.append(structure_metrics[MetricNames.SHD])
                if MetricNames.TRUE_PARENT_LIKELIHOOD in structure_metrics:
                    parent_probs.append(structure_metrics[MetricNames.TRUE_PARENT_LIKELIHOOD])
            else:
                if hasattr(result, 'structure_metrics') and result.structure_metrics:
                    if MetricNames.F1_SCORE in result.structure_metrics:
                        f1_scores.append(result.structure_metrics[MetricNames.F1_SCORE])
                    if MetricNames.SHD in result.structure_metrics:
                        shd_values.append(result.structure_metrics[MetricNames.SHD])
                    if MetricNames.TRUE_PARENT_LIKELIHOOD in result.structure_metrics:
                        parent_probs.append(result.structure_metrics[MetricNames.TRUE_PARENT_LIKELIHOOD])
        
        if f1_scores:
            aggregated.update({
                f'{MetricNames.F1_SCORE}_mean': float(onp.mean(f1_scores)),
                f'{MetricNames.F1_SCORE}_std': float(onp.std(f1_scores))
            })
        
        if shd_values:
            aggregated.update({
                f'{MetricNames.SHD}_mean': float(onp.mean(shd_values)),
                f'{MetricNames.SHD}_std': float(onp.std(shd_values))
            })
        
        if parent_probs:
            aggregated.update({
                f'{MetricNames.TRUE_PARENT_LIKELIHOOD}_mean': float(onp.mean(parent_probs)),
                f'{MetricNames.TRUE_PARENT_LIKELIHOOD}_std': float(onp.std(parent_probs))
            })
    
    def _aggregate_trajectory_metrics_new(self, method_name: str, 
                                        valid_results: List[Any]) -> Dict[str, Any]:
        """Aggregate trajectory metrics from results."""
        trajectories = []
        trajectory_dicts = []
        
        for r in valid_results:
            if isinstance(r, dict):
                traj = r.get('trajectory')
                if traj:
                    if isinstance(traj, dict):
                        trajectory_dicts.append(traj)
                    else:
                        trajectories.append(traj)
            else:
                if hasattr(r, 'trajectory') and r.trajectory:
                    trajectories.append(r.trajectory)
        
        # Convert trajectory objects to dicts if needed
        for t in trajectories:
            if hasattr(t, 'to_dict'):
                trajectory_dicts.append(t.to_dict())
            elif isinstance(t, dict):
                trajectory_dicts.append(t)
        
        if not trajectory_dicts:
            return {}
        
        try:
            return self._aggregate_trajectory_dicts(trajectory_dicts)
            
        except Exception as e:
            logger.warning(f"Failed to aggregate trajectories for {method_name}: {e}")
            return {}
    
    def _aggregate_trajectory_dicts(self, trajectory_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate trajectory dictionaries into mean/std statistics."""
        if not trajectory_dicts:
            return {}
        
        # Find max length across all trajectories
        max_length = max(
            len(t.get('target_values', [])) for t in trajectory_dicts
            if t.get('target_values')
        ) if any(t.get('target_values') for t in trajectory_dicts) else 0
        
        if max_length == 0:
            return {}
        
        aggregated = {
            'steps': list(range(max_length)),
            'target_mean': [],
            'target_std': [],
            'f1_mean': [],
            'f1_std': [],
            'shd_mean': [],
            'shd_std': []
        }
        
        # Compute statistics at each step
        for i in range(max_length):
            # Target values
            target_vals = [
                t['target_values'][i] for t in trajectory_dicts
                if 'target_values' in t and i < len(t['target_values'])
            ]
            if target_vals:
                aggregated['target_mean'].append(float(onp.mean(target_vals)))
                aggregated['target_std'].append(float(onp.std(target_vals)))
            else:
                aggregated['target_mean'].append(0.0)
                aggregated['target_std'].append(0.0)
            
            # F1 scores
            f1_vals = [
                t['f1_scores'][i] for t in trajectory_dicts
                if 'f1_scores' in t and i < len(t['f1_scores'])
            ]
            if f1_vals:
                aggregated['f1_mean'].append(float(onp.mean(f1_vals)))
                aggregated['f1_std'].append(float(onp.std(f1_vals)))
            else:
                aggregated['f1_mean'].append(0.0)
                aggregated['f1_std'].append(0.0)
            
            # SHD values
            shd_vals = [
                t['shd_values'][i] for t in trajectory_dicts
                if 'shd_values' in t and i < len(t['shd_values'])
            ]
            if shd_vals:
                aggregated['shd_mean'].append(float(onp.mean(shd_vals)))
                aggregated['shd_std'].append(float(onp.std(shd_vals)))
            else:
                aggregated['shd_mean'].append(1.0)  # Default SHD
                aggregated['shd_std'].append(0.0)
        
        return aggregated
    
    def compute_learning_curves(self, method_trajectories: Dict[str, Any]) -> Dict[str, Any]:
        """Compute learning curves from trajectory data."""
        if not method_trajectories:
            return {}
        
        try:
            learning_curves = extract_learning_curves(method_trajectories)
            logger.info(f"Computed learning curves for {len(method_trajectories)} methods")
            return learning_curves
            
        except Exception as e:
            logger.error(f"Failed to compute learning curves: {e}")
            return {}
    
    def _collect_performance_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Collect basic performance metrics."""
        return {
            'final_target_value': result.get('final_target_value', 
                                           result.get('final_best', 0.0)),
            'target_reduction': result.get('target_reduction', 
                                        result.get('reduction', 
                                        result.get('target_improvement',
                                        result.get('improvement', 0.0)))),
            'sample_efficiency': result.get('sample_efficiency', 0.0),
            'total_samples': result.get('total_samples', 0),
            'intervention_count': result.get('intervention_count', 
                                           len(result.get('learning_history', [])))
        }
    
    def _collect_structure_metrics(self, result: Dict[str, Any], scm: pyr.PMap) -> Dict[str, Any]:
        """Collect structure learning accuracy metrics."""
        structure_accuracy = self._compute_structure_accuracy(result, scm)
        
        # Extract final structure metrics from trajectory if available
        target = get_target(scm)
        true_parents = list(get_parents(scm, target)) if target else []
        
        # Try to get final F1/SHD from trajectory data
        detailed_results = result.get('detailed_results', {})
        f1_final = None
        shd_final = None
        parent_likelihood_final = None
        
        # Check for pre-computed trajectory metrics
        if 'f1_scores' in detailed_results and detailed_results['f1_scores']:
            f1_final = detailed_results['f1_scores'][-1]
        if 'shd_values' in detailed_results and detailed_results['shd_values']:
            shd_final = detailed_results['shd_values'][-1]
        if 'true_parent_likelihood' in detailed_results and detailed_results['true_parent_likelihood']:
            parent_likelihood_final = detailed_results['true_parent_likelihood'][-1]
        
        # If not found, compute from final marginals
        final_marginals = result.get('final_marginal_probs', {})
        if final_marginals and (f1_final is None or shd_final is None):
            if f1_final is None:
                f1_final, _, _ = compute_f1_from_marginals(final_marginals, true_parents, target)
            if shd_final is None:
                shd_final = compute_shd_from_marginals(final_marginals, true_parents, target)
            if parent_likelihood_final is None:
                parent_likelihood_final = compute_parent_probability(final_marginals, true_parents)
        
        # Set defaults for non-learning methods
        if f1_final is None:
            f1_final = 0.0
        if shd_final is None:
            shd_final = len(true_parents) if true_parents else 0
        if parent_likelihood_final is None:
            parent_likelihood_final = 0.0
        
        return {
            'structure_accuracy': structure_accuracy,
            'final_marginal_probs': final_marginals,
            'converged_to_truth': result.get('converged_to_truth', False),
            'f1_score_final': f1_final,
            'shd_final': shd_final,
            'true_parent_likelihood_final': parent_likelihood_final
        }
    
    def _collect_convergence_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Collect convergence-related metrics."""
        learning_history = result.get('learning_history', [])
        target_progress = result.get('target_progress', [])
        
        convergence_steps = len(target_progress)
        
        # Compute convergence rate
        if target_progress and len(target_progress) > 1:
            initial_value = target_progress[0] if target_progress else 0.0
            final_value = target_progress[-1] if target_progress else 0.0
            improvement = final_value - initial_value
            convergence_rate = improvement / max(1, convergence_steps)
        else:
            convergence_rate = 0.0
        
        return {
            'convergence_steps': convergence_steps,
            'convergence_rate': convergence_rate,
            'final_uncertainty': result.get('final_uncertainty', 0.0)
        }
    
    def _collect_efficiency_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Collect intervention efficiency metrics."""
        learning_history = result.get('learning_history', [])
        
        if not learning_history:
            return {'intervention_efficiency': 0.0}
        
        # Compute intervention efficiency as improvement per intervention
        target_values = [step.get('outcome_value', 0.0) for step in learning_history]
        
        if target_values and len(target_values) > 1:
            total_improvement = max(target_values) - target_values[0]
            efficiency = total_improvement / len(target_values)
        else:
            efficiency = 0.0
        
        return {'intervention_efficiency': efficiency}
    
    def _collect_trajectory_metrics(self, result: Dict[str, Any], scm: pyr.PMap) -> Dict[str, Any]:
        """Collect detailed trajectory metrics."""
        target = get_target(scm)
        true_parents = list(get_parents(scm, target)) if target else []
        
        try:
            # Use existing trajectory analysis infrastructure
            trajectory_metrics = self._extract_trajectory_from_result(result, scm)
            
            # Extract final values
            final_metrics = {}
            for metric_name, values in trajectory_metrics.items():
                if values and isinstance(values, list):
                    final_metrics[f'{metric_name}_final'] = values[-1]
                    final_metrics[f'{metric_name}_trajectory'] = values
            
            return final_metrics
            
        except Exception as e:
            logger.warning(f"Failed to extract trajectory metrics: {e}")
            return self._create_fallback_trajectory_metrics(result, true_parents, target)
    
    def _extract_trajectory_from_result(self, result: Dict[str, Any], scm: pyr.PMap) -> Dict[str, List[float]]:
        """Extract trajectory metrics from experiment result."""
        target = get_target(scm)
        true_parents = list(get_parents(scm, target)) if target else []
        
        # First check if trajectory data is in detailed_results (from method_registry)
        detailed_results = result.get('detailed_results', {})
        
        # Check for pre-computed trajectory data in detailed_results
        if any(key in detailed_results for key in ['target_progress', 'f1_scores', 'shd_values']):
            return {
                'steps': detailed_results.get('steps', list(range(len(detailed_results.get('target_progress', []))))),
                'true_parent_likelihood': detailed_results.get('true_parent_likelihood', 
                                                              detailed_results.get('parent_probability', [])),
                'f1_scores': detailed_results.get('f1_scores', []),
                'shd_values': detailed_results.get('shd_values', []),
                'target_values': detailed_results.get('target_progress', 
                                                    detailed_results.get('target_values', [])),
                'uncertainty_bits': detailed_results.get('uncertainty_progress',
                                                       detailed_results.get('uncertainty_bits', []))
            }
        
        # Try to use existing analysis infrastructure
        try:
            from causal_bayes_opt.analysis.trajectory_metrics import extract_metrics_from_experiment_result
            # Create a properly formatted result for the trajectory metrics function
            # It expects either detailed_results with the data, or the data at top level
            if 'marginal_prob_progress' in result:
                # Data is at top level - wrap it for the function
                formatted_result = {
                    'detailed_results': {
                        'marginal_prob_progress': result.get('marginal_prob_progress', []),
                        'target_progress': result.get('target_progress', []),
                        'uncertainty_progress': result.get('uncertainty_progress', [])
                    }
                }
            else:
                # Try with existing format
                formatted_result = {**result, 'detailed_results': detailed_results}
            return extract_metrics_from_experiment_result(formatted_result, true_parents)
        except Exception as e:
            logger.debug(f"extract_metrics_from_experiment_result failed: {e}")
            # Fallback: create basic trajectory from learning history
            # Check both result and detailed_results for learning_history
            learning_history = detailed_results.get('learning_history', result.get('learning_history', []))
            combined_result = {**result, **detailed_results, 'learning_history': learning_history}
            return self._create_fallback_trajectory_metrics(combined_result, true_parents, target)
    
    def _create_fallback_trajectory_metrics(self, result: Dict[str, Any], 
                                          true_parents: List[str], target: str) -> Dict[str, List[float]]:
        """Create fallback trajectory metrics from learning history."""
        learning_history = result.get('learning_history', [])
        
        if not learning_history:
            return {
                'steps': [0],
                'true_parent_likelihood': [0.0],
                'f1_scores': [0.0],
                'shd_values': [len(true_parents)],
                'target_values': [result.get('final_target_value', 0.0)],
                'uncertainty_bits': [result.get('final_uncertainty', 0.0)]
            }
        
        steps = list(range(len(learning_history)))
        target_values = []
        uncertainty_values = []
        
        # Extract target values - handle different key names
        for step in learning_history:
            # Try multiple possible keys for target value
            target_val = step.get('outcome_value', 
                               step.get('target_value',
                                       step.get('reward', 0.0)))
            target_values.append(target_val)
            
            # Extract uncertainty
            uncertainty_values.append(step.get('uncertainty', 0.0))
        
        # Try to compute structure metrics from marginal probabilities
        f1_scores = []
        parent_probs = []
        shd_values = []
        
        for step in learning_history:
            # Check for marginals with both possible key names
            marginals = step.get('marginals', step.get('marginal_probs', {}))
            
            # Check if we have precomputed structure metrics
            if 'f1_score' in step and 'shd' in step:
                # Use precomputed values
                f1_scores.append(step['f1_score'])
                parent_probs.append(step.get('true_parent_likelihood', 
                                           step.get('parent_probability', 0.0)))
                shd_values.append(step['shd'])
            elif marginals and true_parents:
                # Compute from marginals
                f1, _, _ = compute_f1_from_marginals(marginals, true_parents, target)
                parent_prob = compute_parent_probability(marginals, true_parents)
                shd = compute_shd_from_marginals(marginals, true_parents, target)
                
                f1_scores.append(f1)
                parent_probs.append(parent_prob)
                shd_values.append(shd)
            else:
                # Default values - distinguish between non-learning and missing data
                # For non-learning methods (empty marginals), use 0.0
                # For missing data, this should be handled differently
                is_non_learning = marginals == {}  # Explicitly empty dict means no learning
                if is_non_learning:
                    f1_scores.append(0.0)
                    parent_probs.append(0.0)
                    shd_values.append(len(true_parents))  # Worst case
                else:
                    # Missing data - log warning
                    logger.debug(f"Missing marginals at step {len(f1_scores)}, using defaults")
                    f1_scores.append(0.0)
                    parent_probs.append(0.0)
                    shd_values.append(len(true_parents))
        
        return {
            'steps': steps,
            'true_parent_likelihood': parent_probs,
            'f1_scores': f1_scores,
            'shd_values': shd_values,
            'target_values': target_values,
            'uncertainty_bits': uncertainty_values
        }
    
    def _compute_structure_accuracy(self, result: Dict[str, Any], scm: pyr.PMap) -> float:
        """Compute structure learning accuracy."""
        try:
            target = get_target(scm)
            if not target:
                return 0.0
            
            true_parents = set(get_parents(scm, target))
            final_marginals = result.get('final_marginal_probs', {})
            
            if not final_marginals:
                return 0.0
            
            # Compute accuracy based on thresholded predictions
            correct = 0
            total = 0
            
            for var, prob in final_marginals.items():
                if var != target:
                    predicted_parent = prob > 0.5
                    is_true_parent = var in true_parents
                    
                    if predicted_parent == is_true_parent:
                        correct += 1
                    total += 1
            
            return correct / total if total > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Failed to compute structure accuracy: {e}")
            return 0.0
    
    def _aggregate_trajectory_metrics(self, method_name: str, 
                                    valid_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate trajectory metrics across runs."""
        trajectory_metrics = {}
        
        # Find trajectory keys for this method
        method_keys = [key for key in self.trajectory_data.keys() 
                      if key.startswith(method_name)]
        
        if not method_keys:
            # Also try to extract from valid_results directly
            all_trajectories = []
            for result in valid_results:
                # Check both main result and detailed_results
                detailed = result.get('detailed_results', {})
                if any(key in detailed for key in ['target_progress', 'f1_scores', 'shd_values']):
                    trajectory = {
                        'target_values': detailed.get('target_progress', detailed.get('target_values', [])),
                        'f1_scores': detailed.get('f1_scores', []),
                        'shd_values': detailed.get('shd_values', []),
                        'true_parent_likelihood': detailed.get('true_parent_likelihood', []),
                        'uncertainty_bits': detailed.get('uncertainty_progress', detailed.get('uncertainty_bits', []))
                    }
                    all_trajectories.append(trajectory)
            
            if not all_trajectories:
                return {}
        else:
            # Aggregate trajectory data from stored trajectories
            all_trajectories = [self.trajectory_data[key] for key in method_keys]
        
        try:
            # Use extract_learning_curves for proper aggregation
            learning_curves = extract_learning_curves({method_name: all_trajectories})
            trajectory_metrics.update(learning_curves.get(method_name, {}))
            
            # Also preserve raw trajectory data for detailed analysis
            if all_trajectories:
                # Calculate mean trajectories
                trajectory_lengths = [len(t.get('target_values', [])) for t in all_trajectories]
                if trajectory_lengths:
                    max_length = max(trajectory_lengths)
                    
                    # Initialize aggregated trajectories
                    mean_trajectories = {
                        'target_values_mean': [],
                        'f1_scores_mean': [],
                        'shd_values_mean': [],
                        'true_parent_likelihood_mean': [],
                        'uncertainty_mean': []
                    }
                    
                    # Calculate means at each step
                    for i in range(max_length):
                        target_vals = [t['target_values'][i] for t in all_trajectories 
                                     if i < len(t.get('target_values', []))]
                        f1_vals = [t['f1_scores'][i] for t in all_trajectories 
                                 if i < len(t.get('f1_scores', []))]
                        shd_vals = [t['shd_values'][i] for t in all_trajectories 
                                  if i < len(t.get('shd_values', []))]
                        parent_probs = [t['true_parent_likelihood'][i] for t in all_trajectories 
                                      if i < len(t.get('true_parent_likelihood', []))]
                        uncertainties = [t['uncertainty_bits'][i] for t in all_trajectories 
                                       if i < len(t.get('uncertainty_bits', []))]
                        
                        if target_vals:
                            mean_trajectories['target_values_mean'].append(float(onp.mean(target_vals)))
                        if f1_vals:
                            mean_trajectories['f1_scores_mean'].append(float(onp.mean(f1_vals)))
                        if shd_vals:
                            mean_trajectories['shd_values_mean'].append(float(onp.mean(shd_vals)))
                        if parent_probs:
                            mean_trajectories['true_parent_likelihood_mean'].append(float(onp.mean(parent_probs)))
                        if uncertainties:
                            mean_trajectories['uncertainty_mean'].append(float(onp.mean(uncertainties)))
                    
                    trajectory_metrics.update(mean_trajectories)
            
        except Exception as e:
            logger.warning(f"Failed to aggregate trajectories for {method_name}: {e}")
        
        return trajectory_metrics
    
    def aggregate_trajectory_data(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Public method to aggregate trajectory data from multiple runs.
        
        Args:
            runs: List of run results containing trajectory data
            
        Returns:
            Aggregated trajectory data suitable for plotting
        """
        if not runs:
            return {}
        
        # Extract trajectory data from all runs
        all_trajectories = []
        
        for run in runs:
            # Check detailed_results for trajectory data
            detailed = run.get('detailed_results', {})
            
            # Look for trajectory arrays in detailed results
            if any(key in detailed for key in ['target_progress', 'f1_scores', 'shd_values']):
                trajectory = {
                    'target_values': detailed.get('target_progress', detailed.get('target_values', [])),
                    'f1_scores': detailed.get('f1_scores', []),
                    'shd_values': detailed.get('shd_values', []),
                    'true_parent_likelihood': detailed.get('true_parent_likelihood', []),
                    'uncertainty_bits': detailed.get('uncertainty_progress', detailed.get('uncertainty_bits', [])),
                    'steps': detailed.get('steps', [])
                }
                all_trajectories.append(trajectory)
        
        if not all_trajectories:
            logger.warning("No trajectory data found in runs")
            return {}
        
        # Find the maximum length to handle variable-length trajectories
        trajectory_lengths = [len(t.get('target_values', [])) for t in all_trajectories]
        if not trajectory_lengths:
            return {}
        
        max_length = max(trajectory_lengths)
        n_runs = len(all_trajectories)
        
        # Initialize aggregated results
        aggregated = {
            'n_runs': n_runs,
            'max_length': max_length,
            'target_values_mean': [],
            'target_values_std': [],
            'f1_scores_mean': [],
            'f1_scores_std': [],
            'shd_values_mean': [],
            'shd_values_std': [],
            'true_parent_likelihood_mean': [],
            'true_parent_likelihood_std': [],
            'uncertainty_mean': [],
            'uncertainty_std': [],
            'steps': list(range(max_length))
        }
        
        # Calculate statistics at each step
        for i in range(max_length):
            # Target values
            target_vals = [t['target_values'][i] for t in all_trajectories 
                         if i < len(t.get('target_values', []))]
            if target_vals:
                aggregated['target_values_mean'].append(float(onp.mean(target_vals)))
                aggregated['target_values_std'].append(float(onp.std(target_vals)))
            else:
                aggregated['target_values_mean'].append(0.0)
                aggregated['target_values_std'].append(0.0)
            
            # F1 scores
            f1_vals = [t['f1_scores'][i] for t in all_trajectories 
                     if i < len(t.get('f1_scores', []))]
            if f1_vals:
                aggregated['f1_scores_mean'].append(float(onp.mean(f1_vals)))
                aggregated['f1_scores_std'].append(float(onp.std(f1_vals)))
            else:
                aggregated['f1_scores_mean'].append(0.0)
                aggregated['f1_scores_std'].append(0.0)
            
            # SHD values
            shd_vals = [t['shd_values'][i] for t in all_trajectories 
                      if i < len(t.get('shd_values', []))]
            if shd_vals:
                aggregated['shd_values_mean'].append(float(onp.mean(shd_vals)))
                aggregated['shd_values_std'].append(float(onp.std(shd_vals)))
            else:
                aggregated['shd_values_mean'].append(1.0)  # Default SHD
                aggregated['shd_values_std'].append(0.0)
            
            # True parent likelihood
            parent_probs = [t['true_parent_likelihood'][i] for t in all_trajectories 
                          if i < len(t.get('true_parent_likelihood', []))]
            if parent_probs:
                aggregated['true_parent_likelihood_mean'].append(float(onp.mean(parent_probs)))
                aggregated['true_parent_likelihood_std'].append(float(onp.std(parent_probs)))
            else:
                aggregated['true_parent_likelihood_mean'].append(0.0)
                aggregated['true_parent_likelihood_std'].append(0.0)
            
            # Uncertainty
            uncertainties = [t['uncertainty_bits'][i] for t in all_trajectories 
                           if i < len(t.get('uncertainty_bits', []))]
            if uncertainties:
                aggregated['uncertainty_mean'].append(float(onp.mean(uncertainties)))
                aggregated['uncertainty_std'].append(float(onp.std(uncertainties)))
            else:
                aggregated['uncertainty_mean'].append(0.0)
                aggregated['uncertainty_std'].append(0.0)
        
        logger.info(f"Aggregated trajectory data: {n_runs} runs, {max_length} steps")
        return aggregated

    def export_metrics(self, filepath: str) -> None:
        """Export collected metrics to file."""
        import json
        
        try:
            export_data = {
                'collected_metrics': self.collected_metrics,
                'trajectory_data': {k: self._serialize_trajectory(v) 
                                  for k, v in self.trajectory_data.items()}
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported metrics to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def _serialize_trajectory(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize trajectory data for JSON export."""
        serialized = {}
        for key, value in trajectory_data.items():
            if isinstance(value, (list, tuple)):
                serialized[key] = [float(x) if isinstance(x, (int, float, jnp.ndarray)) else str(x) 
                                 for x in value]
            elif isinstance(value, (int, float, jnp.ndarray)):
                serialized[key] = float(value)
            else:
                serialized[key] = str(value)
        return serialized