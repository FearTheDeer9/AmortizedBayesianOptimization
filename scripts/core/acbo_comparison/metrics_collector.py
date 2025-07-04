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

from causal_bayes_opt.data_structures.scm import get_target, get_parents, get_variables
from causal_bayes_opt.analysis.trajectory_metrics import (
    compute_trajectory_metrics, analyze_convergence_trajectory,
    extract_learning_curves, compute_intervention_efficiency
)

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collector for standardized ACBO experiment metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.collected_metrics = {}
        self.trajectory_data = {}
    
    def collect_run_metrics(self, result: Dict[str, Any], scm: pyr.PMap, 
                           method_name: str, run_idx: int, scm_idx: int) -> Dict[str, Any]:
        """
        Collect comprehensive metrics from a single experimental run.
        
        Args:
            result: Raw result from method execution
            scm: The SCM used in the experiment
            method_name: Name of the method
            run_idx: Run index for this method
            scm_idx: SCM index
            
        Returns:
            Dictionary of collected metrics
        """
        metrics = {}
        
        try:
            # Basic performance metrics
            metrics.update(self._collect_performance_metrics(result))
            
            # Structure learning metrics
            metrics.update(self._collect_structure_metrics(result, scm))
            
            # Convergence metrics
            metrics.update(self._collect_convergence_metrics(result))
            
            # Intervention efficiency metrics
            metrics.update(self._collect_efficiency_metrics(result))
            
            # Trajectory metrics
            trajectory_metrics = self._collect_trajectory_metrics(result, scm)
            metrics.update(trajectory_metrics)
            
            # Store trajectory data for aggregation
            trajectory_key = f"{method_name}_{run_idx}_{scm_idx}"
            self.trajectory_data[trajectory_key] = trajectory_metrics
            
            # Add context information
            metrics.update({
                'method_name': method_name,
                'run_idx': run_idx,
                'scm_idx': scm_idx,
                'scm_variables': len(get_variables(scm)),
                'scm_target': get_target(scm)
            })
            
            logger.debug(f"Collected metrics for {method_name} run {run_idx}")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics for {method_name}: {e}")
            return {
                'method_name': method_name,
                'run_idx': run_idx,
                'scm_idx': scm_idx,
                'collection_error': str(e)
            }
    
    def aggregate_method_metrics(self, method_name: str, 
                               all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate metrics across all runs for a specific method.
        
        Args:
            method_name: Name of the method
            all_results: List of all results for this method
            
        Returns:
            Aggregated metrics
        """
        if not all_results:
            return {}
        
        valid_results = [r for r in all_results if r and 'collection_error' not in r]
        
        if not valid_results:
            logger.warning(f"No valid results for {method_name}")
            return {'method_name': method_name, 'valid_runs': 0}
        
        aggregated = {
            'method_name': method_name,
            'total_runs': len(all_results),
            'valid_runs': len(valid_results),
            'success_rate': len(valid_results) / len(all_results)
        }
        
        # Aggregate numeric metrics
        numeric_metrics = [
            'final_target_value', 'target_improvement', 'structure_accuracy',
            'sample_efficiency', 'intervention_count', 'convergence_steps',
            'true_parent_likelihood_final', 'f1_score_final', 'shd_final',
            'uncertainty_final', 'convergence_rate', 'intervention_efficiency'
        ]
        
        for metric in numeric_metrics:
            values = [r.get(metric, 0.0) for r in valid_results if metric in r]
            if values:
                aggregated.update({
                    f'{metric}_mean': float(onp.mean(values)),
                    f'{metric}_std': float(onp.std(values)),
                    f'{metric}_min': float(onp.min(values)),
                    f'{metric}_max': float(onp.max(values))
                })
        
        # Aggregate trajectory metrics
        aggregated.update(self._aggregate_trajectory_metrics(method_name, valid_results))
        
        logger.info(f"Aggregated metrics for {method_name}: {aggregated['valid_runs']} valid runs")
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
            'target_improvement': result.get('target_improvement', 
                                           result.get('improvement', 0.0)),
            'sample_efficiency': result.get('sample_efficiency', 0.0),
            'total_samples': result.get('total_samples', 0),
            'intervention_count': result.get('intervention_count', 
                                           len(result.get('learning_history', [])))
        }
    
    def _collect_structure_metrics(self, result: Dict[str, Any], scm: pyr.PMap) -> Dict[str, Any]:
        """Collect structure learning accuracy metrics."""
        structure_accuracy = self._compute_structure_accuracy(result, scm)
        
        return {
            'structure_accuracy': structure_accuracy,
            'final_marginal_probs': result.get('final_marginal_probs', {}),
            'converged_to_truth': result.get('converged_to_truth', False)
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
            return self._create_fallback_trajectory_metrics(result, true_parents)
    
    def _extract_trajectory_from_result(self, result: Dict[str, Any], scm: pyr.PMap) -> Dict[str, List[float]]:
        """Extract trajectory metrics from experiment result."""
        target = get_target(scm)
        true_parents = list(get_parents(scm, target)) if target else []
        
        # Try to use existing analysis infrastructure
        try:
            from causal_bayes_opt.analysis.trajectory_metrics import extract_metrics_from_experiment_result
            return extract_metrics_from_experiment_result(result, true_parents)
        except:
            # Fallback: create basic trajectory from learning history
            return self._create_fallback_trajectory_metrics(result, true_parents)
    
    def _create_fallback_trajectory_metrics(self, result: Dict[str, Any], 
                                          true_parents: List[str]) -> Dict[str, List[float]]:
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
        target_values = [step.get('outcome_value', 0.0) for step in learning_history]
        uncertainty_values = [step.get('uncertainty', 0.0) for step in learning_history]
        
        return {
            'steps': steps,
            'true_parent_likelihood': [0.0] * len(steps),  # Not tracked by baselines
            'f1_scores': [0.0] * len(steps),  # Not tracked by baselines
            'shd_values': [len(true_parents)] * len(steps),  # Worst case for baselines
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
            return {}
        
        # Aggregate trajectory data
        all_trajectories = [self.trajectory_data[key] for key in method_keys]
        
        try:
            # Use extract_learning_curves for proper aggregation
            learning_curves = extract_learning_curves({method_name: all_trajectories})
            trajectory_metrics.update(learning_curves.get(method_name, {}))
            
        except Exception as e:
            logger.warning(f"Failed to aggregate trajectories for {method_name}: {e}")
        
        return trajectory_metrics
    
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