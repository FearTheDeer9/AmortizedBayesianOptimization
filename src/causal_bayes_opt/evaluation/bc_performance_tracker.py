"""
Performance tracking utilities for BC method evaluation.

Tracks SHD, F1, and target values during causal Bayesian optimization.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics at a single step."""
    step: int
    intervention_variable: Optional[str]
    intervention_value: Optional[float]
    target_value: float
    shd: Optional[float] = None  # Structural Hamming Distance
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    graph_estimate: Optional[Dict[str, List[str]]] = None
    cumulative_best: Optional[float] = None


@dataclass 
class PerformanceTrajectory:
    """Container for full optimization trajectory."""
    metrics: List[PerformanceMetrics] = field(default_factory=list)
    ground_truth_graph: Optional[Dict[str, List[str]]] = None
    target_variable: Optional[str] = None
    
    def add_step(self, metric: PerformanceMetrics):
        """Add a performance metric for a step."""
        # Update cumulative best
        if self.metrics:
            prev_best = self.metrics[-1].cumulative_best or metric.target_value
            metric.cumulative_best = max(prev_best, metric.target_value)
        else:
            metric.cumulative_best = metric.target_value
        
        self.metrics.append(metric)
    
    def get_arrays(self) -> Dict[str, np.ndarray]:
        """Extract arrays for plotting."""
        if not self.metrics:
            return {}
        
        steps = np.array([m.step for m in self.metrics])
        target_values = np.array([m.target_value for m in self.metrics])
        cumulative_best = np.array([m.cumulative_best or m.target_value for m in self.metrics])
        
        result = {
            'steps': steps,
            'target_values': target_values,
            'cumulative_best': cumulative_best
        }
        
        # Add SHD if available
        if any(m.shd is not None for m in self.metrics):
            shd_values = np.array([m.shd if m.shd is not None else np.nan for m in self.metrics])
            result['shd'] = shd_values
        
        # Add F1 if available
        if any(m.f1_score is not None for m in self.metrics):
            f1_values = np.array([m.f1_score if m.f1_score is not None else np.nan for m in self.metrics])
            result['f1_score'] = f1_values
            
        return result


def extract_ground_truth_graph(scm) -> Dict[str, List[str]]:
    """Extract ground truth graph structure from SCM."""
    graph = {}
    
    # Handle pyrsistent SCM format
    if hasattr(scm, 'get'):
        variables = scm.get('variables', set())
        for var in variables:
            parents = scm.get(f'{var}_parents', set())
            graph[var] = list(parents)
    else:
        # Fallback for other SCM formats
        logger.warning("Unknown SCM format, cannot extract ground truth graph")
        
    return graph


def calculate_shd(true_graph: Dict[str, List[str]], 
                  estimated_graph: Dict[str, List[str]]) -> float:
    """
    Calculate Structural Hamming Distance between two graphs.
    
    SHD = number of edge additions + deletions + reversals needed to transform
    estimated graph into true graph.
    """
    if not true_graph or not estimated_graph:
        return float('nan')
    
    shd = 0
    all_vars = set(true_graph.keys()) | set(estimated_graph.keys())
    
    for var in all_vars:
        true_parents = set(true_graph.get(var, []))
        est_parents = set(estimated_graph.get(var, []))
        
        # Missing edges (in true but not estimated)
        missing = true_parents - est_parents
        shd += len(missing)
        
        # Extra edges (in estimated but not true)
        extra = est_parents - true_parents
        shd += len(extra)
        
        # Check for reversed edges
        for parent in est_parents:
            if parent in true_graph and var in true_graph[parent]:
                # This is a reversal (only count once)
                shd += 0.5
    
    return shd


def calculate_f1_score(true_graph: Dict[str, List[str]], 
                       estimated_graph: Dict[str, List[str]]) -> Tuple[float, float, float]:
    """
    Calculate F1 score, precision, and recall for edge prediction.
    
    Returns:
        (f1_score, precision, recall)
    """
    if not true_graph or not estimated_graph:
        return float('nan'), float('nan'), float('nan')
    
    # Convert to edge sets
    true_edges = set()
    for child, parents in true_graph.items():
        for parent in parents:
            true_edges.add((parent, child))
    
    est_edges = set()
    for child, parents in estimated_graph.items():
        for parent in parents:
            est_edges.add((parent, child))
    
    if not true_edges and not est_edges:
        return 1.0, 1.0, 1.0  # Both empty = perfect match
    
    if not est_edges:
        return 0.0, 0.0, 0.0  # No edges predicted
    
    # Calculate metrics
    tp = len(true_edges & est_edges)  # True positives
    fp = len(est_edges - true_edges)  # False positives
    fn = len(true_edges - est_edges)  # False negatives
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1, precision, recall


class BCPerformanceTracker:
    """Tracks performance metrics during BC method evaluation."""
    
    def __init__(self, scm, target_variable: str):
        """
        Initialize tracker with SCM and target variable.
        
        Args:
            scm: The structural causal model
            target_variable: Name of the target variable to optimize
        """
        self.scm = scm
        self.target_variable = target_variable
        self.ground_truth = extract_ground_truth_graph(scm)
        self.trajectory = PerformanceTrajectory(
            ground_truth_graph=self.ground_truth,
            target_variable=target_variable
        )
        self.step_count = 0
        
    def record_step(self,
                    intervention_variable: Optional[str],
                    intervention_value: Optional[float],
                    target_value: float,
                    graph_estimate: Optional[Dict[str, List[str]]] = None):
        """
        Record metrics for a single optimization step.
        
        Args:
            intervention_variable: Variable that was intervened on
            intervention_value: Value set for intervention
            target_value: Observed value of target variable
            graph_estimate: Current estimate of graph structure
        """
        metrics = PerformanceMetrics(
            step=self.step_count,
            intervention_variable=intervention_variable,
            intervention_value=intervention_value,
            target_value=target_value,
            graph_estimate=graph_estimate
        )
        
        # Calculate structure learning metrics if graph estimate provided
        if graph_estimate and self.ground_truth:
            metrics.shd = calculate_shd(self.ground_truth, graph_estimate)
            metrics.f1_score, metrics.precision, metrics.recall = calculate_f1_score(
                self.ground_truth, graph_estimate
            )
        
        self.trajectory.add_step(metrics)
        self.step_count += 1
        
    def get_trajectory(self) -> PerformanceTrajectory:
        """Get the full performance trajectory."""
        return self.trajectory
    
    def get_final_metrics(self) -> Dict[str, float]:
        """Get final performance metrics."""
        if not self.trajectory.metrics:
            return {}
        
        final = self.trajectory.metrics[-1]
        return {
            'final_target_value': final.target_value,
            'final_shd': final.shd if final.shd is not None else float('nan'),
            'final_f1': final.f1_score if final.f1_score is not None else float('nan'),
            'cumulative_best': final.cumulative_best or final.target_value,
            'improvement': (final.cumulative_best or final.target_value) - self.trajectory.metrics[0].target_value,
            'total_steps': len(self.trajectory.metrics)
        }