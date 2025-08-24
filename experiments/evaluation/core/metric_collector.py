"""
Flexible metrics collection for experiments.

This module provides a flexible framework for collecting and analyzing
experiment-specific metrics during evaluation.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class InterventionRecord:
    """Record of a single intervention."""
    step: int
    variable: str
    value: float
    target_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class PosteriorRecord:
    """Record of surrogate posterior predictions."""
    step: int
    parent_probabilities: Dict[str, np.ndarray]
    confidence_scores: Dict[str, float]
    prediction_metadata: Dict[str, Any] = field(default_factory=dict)


class MetricCollector:
    """Collect and compute experiment-specific metrics."""
    
    def __init__(self, metrics_to_track: List[str]):
        """
        Initialize metric collector.
        
        Args:
            metrics_to_track: List of metrics to compute. Options include:
                - 'convergence': Track convergence rate and final values
                - 'sample_efficiency': Track interventions to threshold
                - 'parent_accuracy': Track parent selection accuracy
                - 'regret': Track cumulative regret
                - 'exploration': Track exploration vs exploitation
        """
        self.metrics_to_track = metrics_to_track
        self.trajectories = []
        self.current_trajectory = []
        self.current_posterior_trajectory = []
        self.method_results = defaultdict(list)
        self.current_method = None
        self.current_scm = None
        
        # For tracking optimal values (needed for regret)
        self.optimal_values = {}
        
        # For structure learning metrics
        self.true_parents = {}
        
        logger.info(f"Initialized MetricCollector tracking: {metrics_to_track}")
        
    def start_trajectory(self, method_name: str, scm_name: str):
        """
        Start recording a new trajectory.
        
        Args:
            method_name: Name of the method being evaluated
            scm_name: Name/identifier of the SCM being used
        """
        if self.current_trajectory:
            logger.warning("Starting new trajectory without ending previous one")
            self.end_trajectory()
            
        self.current_trajectory = []
        self.current_posterior_trajectory = []
        self.current_method = method_name
        self.current_scm = scm_name
        logger.debug(f"Started trajectory for {method_name} on {scm_name}")
        
    def record_intervention(self, 
                           variable: str,
                           value: float,
                           target_value: float,
                           **metadata):
        """
        Record a single intervention.
        
        Args:
            variable: Variable that was intervened on
            value: Intervention value
            target_value: Resulting target value
            **metadata: Additional metadata (e.g., is_parent, surrogate_prediction)
        """
        record = InterventionRecord(
            step=len(self.current_trajectory),
            variable=variable,
            value=value,
            target_value=target_value,
            metadata=metadata
        )
        self.current_trajectory.append(record)
    
    def record_posterior(self,
                        parent_probabilities: Dict[str, np.ndarray],
                        confidence_scores: Optional[Dict[str, float]] = None,
                        **metadata):
        """
        Record surrogate posterior predictions.
        
        Args:
            parent_probabilities: Dictionary mapping variables to parent probability arrays
            confidence_scores: Optional confidence scores for predictions
            **metadata: Additional prediction metadata
        """
        if confidence_scores is None:
            confidence_scores = {}
        
        record = PosteriorRecord(
            step=len(self.current_posterior_trajectory),
            parent_probabilities=parent_probabilities,
            confidence_scores=confidence_scores,
            prediction_metadata=metadata
        )
        self.current_posterior_trajectory.append(record)
    
    def set_true_parents(self, true_parents: Dict[str, List[str]]):
        """
        Set true parent relationships for structure learning evaluation.
        
        Args:
            true_parents: Dictionary mapping variables to their true parents
        """
        self.true_parents = true_parents
        
    def set_optimal_value(self, scm_name: str, optimal_value: float):
        """
        Set the optimal achievable value for an SCM (used for regret calculation).
        
        Args:
            scm_name: SCM identifier
            optimal_value: Best achievable target value
        """
        self.optimal_values[scm_name] = optimal_value
        
    def end_trajectory(self) -> Dict[str, Any]:
        """
        Finish current trajectory and compute metrics.
        
        Returns:
            Dictionary of computed metrics for this trajectory
        """
        if not self.current_trajectory:
            logger.warning("Ending empty trajectory")
            return {}
            
        trajectory_data = {
            'method': self.current_method,
            'scm': self.current_scm,
            'interventions': self.current_trajectory,
            'metrics': self._compute_trajectory_metrics()
        }
        
        self.trajectories.append(trajectory_data)
        self.method_results[self.current_method].append(trajectory_data)
        
        logger.debug(f"Ended trajectory for {self.current_method}: "
                    f"{len(self.current_trajectory)} interventions")
        
        # Clear current trajectory
        self.current_trajectory = []
        self.current_posterior_trajectory = []
        self.current_method = None
        self.current_scm = None
        
        return trajectory_data['metrics']
        
    def _compute_trajectory_metrics(self) -> Dict[str, float]:
        """Compute metrics for current trajectory."""
        metrics = {}
        
        if 'convergence' in self.metrics_to_track:
            target_values = [r.target_value for r in self.current_trajectory]
            if target_values:
                metrics['final_value'] = target_values[-1]
                metrics['best_value'] = min(target_values)
                metrics['convergence_rate'] = self._compute_convergence_rate(target_values)
                metrics['improvement'] = target_values[0] - min(target_values)
            
        if 'sample_efficiency' in self.metrics_to_track:
            metrics['interventions_to_threshold'] = self._compute_sample_efficiency()
            
        if 'parent_accuracy' in self.metrics_to_track:
            parent_selections = [r.metadata.get('is_parent', False) 
                               for r in self.current_trajectory]
            if parent_selections:
                metrics['parent_accuracy'] = np.mean(parent_selections)
                metrics['parent_count'] = sum(parent_selections)
            
        if 'regret' in self.metrics_to_track and self.current_scm in self.optimal_values:
            optimal = self.optimal_values[self.current_scm]
            target_values = [r.target_value for r in self.current_trajectory]
            if target_values:
                # Simple regret: difference from optimal at each step
                regrets = [val - optimal for val in target_values]
                metrics['cumulative_regret'] = sum(regrets)
                metrics['final_regret'] = regrets[-1] if regrets else 0
                metrics['mean_regret'] = np.mean(regrets)
        
        if 'exploration' in self.metrics_to_track:
            # Track unique variables intervened on
            unique_vars = len(set(r.variable for r in self.current_trajectory))
            # Get all variables from first record that has them
            all_vars_set = set()
            for r in self.current_trajectory:
                if 'all_variables' in r.metadata:
                    all_vars_set.update(r.metadata['all_variables'])
                    break
            total_vars = len(all_vars_set) if all_vars_set else unique_vars
            metrics['exploration_ratio'] = unique_vars / max(total_vars, 1)
        
        # Structure learning metrics (if we have posterior predictions and true parents)
        if ('structure_learning' in self.metrics_to_track and 
            self.current_posterior_trajectory and 
            self.true_parents):
            
            structure_metrics = self._compute_structure_learning_metrics()
            metrics.update(structure_metrics)
                
        return metrics
    
    def _compute_convergence_rate(self, values: List[float]) -> float:
        """
        Compute convergence rate (improvement per intervention).
        
        Args:
            values: List of target values over time
            
        Returns:
            Average improvement per intervention
        """
        if len(values) < 2:
            return 0.0
            
        # Compute step-wise improvements (negative for minimization)
        improvements = [values[i] - values[i+1] for i in range(len(values)-1)]
        
        # Return mean of positive improvements
        positive_improvements = [imp for imp in improvements if imp > 0]
        if positive_improvements:
            return np.mean(positive_improvements)
        return 0.0
    
    def _compute_sample_efficiency(self, threshold: float = -5.0) -> int:
        """
        Compute interventions needed to reach threshold.
        
        Args:
            threshold: Target value threshold
            
        Returns:
            Number of interventions to reach threshold (or total if not reached)
        """
        for i, record in enumerate(self.current_trajectory):
            if record.target_value <= threshold:
                return i + 1
        return len(self.current_trajectory)
    
    def _compute_structure_learning_metrics(self) -> Dict[str, float]:
        """
        Compute structure learning metrics from posterior predictions.
        
        Returns:
            Dictionary of structure learning metrics
        """
        if not self.current_posterior_trajectory or not self.true_parents:
            return {}
        
        # Use final posterior predictions for evaluation
        final_posterior = self.current_posterior_trajectory[-1]
        parent_probs = final_posterior.parent_probabilities
        
        # Get variable list from true parents
        variables = list(self.true_parents.keys())
        
        # Import graph metrics for evaluation
        from experiments.evaluation.initial_comparison.src.graph_metrics import evaluate_graph_discovery
        
        try:
            graph_metrics = evaluate_graph_discovery(
                self.true_parents, parent_probs, variables, threshold=0.5
            )
            
            return {
                'parent_f1': graph_metrics.get('f1', 0.0),
                'parent_precision': graph_metrics.get('precision', 0.0),
                'parent_recall': graph_metrics.get('recall', 0.0),
                'structural_hamming_distance': graph_metrics.get('shd', 0.0),
                'true_edges': graph_metrics.get('n_true_edges', 0),
                'predicted_edges': graph_metrics.get('n_predicted_edges', 0)
            }
        except Exception as e:
            logger.warning(f"Failed to compute structure learning metrics: {e}")
            return {
                'parent_f1': 0.0,
                'parent_precision': 0.0, 
                'parent_recall': 0.0,
                'structural_hamming_distance': 0.0
            }
    
    def get_comparison_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get aggregated metrics for all methods.
        
        Returns:
            Dictionary mapping method names to aggregated metrics
        """
        comparison = {}
        
        for method, trajectories in self.method_results.items():
            if not trajectories:
                continue
                
            all_metrics = [t['metrics'] for t in trajectories]
            
            # Aggregate across SCMs
            comparison[method] = {}
            
            # For each metric type, compute mean and std
            metric_keys = set()
            for m in all_metrics:
                metric_keys.update(m.keys())
                
            for key in metric_keys:
                values = [m.get(key, np.nan) for m in all_metrics]
                values = [v for v in values if not np.isnan(v)]
                
                if values:
                    comparison[method][f'mean_{key}'] = np.mean(values)
                    comparison[method][f'std_{key}'] = np.std(values)
                    comparison[method][f'min_{key}'] = np.min(values)
                    comparison[method][f'max_{key}'] = np.max(values)
                    
        return comparison
    
    def get_trajectories_by_method(self, method_name: str) -> List[List[float]]:
        """
        Get all target value trajectories for a specific method.
        
        Args:
            method_name: Name of the method
            
        Returns:
            List of trajectories (each trajectory is a list of target values)
        """
        trajectories = []
        for traj_data in self.method_results.get(method_name, []):
            target_values = [r.target_value for r in traj_data['interventions']]
            trajectories.append(target_values)
        return trajectories
    
    def get_all_trajectories(self) -> Dict[str, List[List[float]]]:
        """
        Get all trajectories grouped by method.
        
        Returns:
            Dictionary mapping method names to lists of trajectories
        """
        all_trajectories = {}
        for method in self.method_results.keys():
            all_trajectories[method] = self.get_trajectories_by_method(method)
        return all_trajectories
    
    def export_results(self) -> Dict[str, Any]:
        """
        Export all collected results.
        
        Returns:
            Dictionary containing all trajectories, metrics, and comparisons
        """
        return {
            'trajectories': self.trajectories,
            'method_results': dict(self.method_results),
            'comparison_metrics': self.get_comparison_metrics(),
            'all_trajectories': self.get_all_trajectories(),
            'metrics_tracked': self.metrics_to_track,
            'optimal_values': self.optimal_values
        }
    
    def summary(self) -> str:
        """
        Generate a text summary of collected metrics.
        
        Returns:
            Formatted summary string
        """
        comparison = self.get_comparison_metrics()
        
        summary_lines = ["=" * 60]
        summary_lines.append("METRIC COLLECTION SUMMARY")
        summary_lines.append("=" * 60)
        
        for method, metrics in comparison.items():
            summary_lines.append(f"\n{method}:")
            summary_lines.append("-" * 40)
            
            # Group metrics by type
            for metric_type in self.metrics_to_track:
                relevant_metrics = {k: v for k, v in metrics.items() 
                                  if metric_type in k}
                if relevant_metrics:
                    summary_lines.append(f"  {metric_type.capitalize()}:")
                    for key, value in relevant_metrics.items():
                        if 'accuracy' in key:
                            summary_lines.append(f"    {key}: {value:.1%}")
                        else:
                            summary_lines.append(f"    {key}: {value:.3f}")
        
        summary_lines.append("\n" + "=" * 60)
        summary_lines.append(f"Total trajectories collected: {len(self.trajectories)}")
        summary_lines.append(f"Methods evaluated: {list(self.method_results.keys())}")
        
        return "\n".join(summary_lines)