"""
Evaluation metrics for ACBO performance assessment.

This module provides pure functions for computing various evaluation metrics
following functional programming principles.
"""

import jax
import jax.numpy as jnp
import numpy as onp
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
import pyrsistent as pyr
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CausalDiscoveryMetrics:
    """Immutable causal discovery evaluation metrics."""
    precision: float
    recall: float
    f1_score: float
    structural_hamming_distance: int
    edge_accuracy: float
    orientation_accuracy: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int


@dataclass(frozen=True)
class OptimizationMetrics:
    """Immutable optimization performance metrics."""
    final_objective_value: float
    optimization_improvement: float
    convergence_steps: int
    intervention_efficiency: float
    sample_efficiency: float
    regret: float
    cumulative_regret: float


@dataclass(frozen=True)
class EfficiencyMetrics:
    """Immutable efficiency and resource usage metrics."""
    total_interventions: int
    computational_time_seconds: float
    memory_usage_gb: float
    interventions_per_second: float
    time_to_convergence: float
    inference_time_per_step: float


@dataclass(frozen=True)
class CompositeMetrics:
    """Immutable composite evaluation metrics."""
    causal_discovery: CausalDiscoveryMetrics
    optimization: OptimizationMetrics
    efficiency: EfficiencyMetrics
    overall_score: float
    weighted_score: Dict[str, float]


# Pure metric computation functions
def compute_causal_discovery_metrics(
    true_graph: jnp.ndarray,
    predicted_graph: jnp.ndarray,
    compute_orientations: bool = True
) -> CausalDiscoveryMetrics:
    """
    Compute causal discovery evaluation metrics.
    
    Args:
        true_graph: Ground truth adjacency matrix [N, N] 
        predicted_graph: Predicted adjacency matrix [N, N]
        compute_orientations: Whether to compute orientation accuracy
        
    Returns:
        CausalDiscoveryMetrics with all computed metrics
    """
    # Ensure binary matrices
    true_binary = (jnp.abs(true_graph) > 1e-6).astype(jnp.int32)
    pred_binary = (jnp.abs(predicted_graph) > 1e-6).astype(jnp.int32)
    
    # Compute confusion matrix elements
    tp = jnp.sum(true_binary * pred_binary)
    fp = jnp.sum((1 - true_binary) * pred_binary)
    fn = jnp.sum(true_binary * (1 - pred_binary))
    tn = jnp.sum((1 - true_binary) * (1 - pred_binary))
    
    # Basic metrics
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Edge accuracy
    edge_accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Structural Hamming Distance
    shd = jnp.sum(jnp.abs(true_binary - pred_binary))
    
    # Orientation accuracy (if requested)
    orientation_accuracy = 0.0
    if compute_orientations:
        # Check orientation for edges that exist in both graphs
        both_edges = true_binary * pred_binary
        if jnp.sum(both_edges) > 0:
            # For directed graphs, check if directions match
            correctly_oriented = jnp.sum(both_edges * (true_graph == predicted_graph))
            orientation_accuracy = correctly_oriented / jnp.sum(both_edges)
    
    return CausalDiscoveryMetrics(
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1_score),
        structural_hamming_distance=int(shd),
        edge_accuracy=float(edge_accuracy),
        orientation_accuracy=float(orientation_accuracy),
        true_positives=int(tp),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_negatives=int(tn)
    )


def compute_optimization_metrics(
    objective_values: jnp.ndarray,
    true_optimum: float,
    intervention_costs: Optional[jnp.ndarray] = None
) -> OptimizationMetrics:
    """
    Compute optimization performance metrics.
    
    Args:
        objective_values: Sequence of objective values over time [T]
        true_optimum: True optimal value (if known)
        intervention_costs: Optional intervention costs [T]
        
    Returns:
        OptimizationMetrics with computed metrics
    """
    if len(objective_values) == 0:
        raise ValueError("objective_values cannot be empty")
    
    # Basic optimization metrics
    final_value = float(objective_values[-1])
    initial_value = float(objective_values[0])
    optimization_improvement = final_value - initial_value
    
    # Convergence analysis
    convergence_threshold = 0.01 * jnp.abs(true_optimum)
    converged_mask = jnp.abs(objective_values - true_optimum) <= convergence_threshold
    convergence_steps = int(jnp.argmax(converged_mask)) if jnp.any(converged_mask) else len(objective_values)
    
    # Regret computation
    simple_regret = jnp.abs(true_optimum - final_value)
    cumulative_regret = jnp.sum(jnp.abs(true_optimum - objective_values))
    
    # Efficiency metrics
    if intervention_costs is not None:
        total_cost = jnp.sum(intervention_costs)
        intervention_efficiency = optimization_improvement / (total_cost + 1e-8)
    else:
        intervention_efficiency = optimization_improvement / len(objective_values)
    
    # Sample efficiency (improvement per step)
    sample_efficiency = optimization_improvement / len(objective_values)
    
    return OptimizationMetrics(
        final_objective_value=final_value,
        optimization_improvement=float(optimization_improvement),
        convergence_steps=convergence_steps,
        intervention_efficiency=float(intervention_efficiency),
        sample_efficiency=float(sample_efficiency),
        regret=float(simple_regret),
        cumulative_regret=float(cumulative_regret)
    )


def compute_efficiency_metrics(
    total_interventions: int,
    computational_time: float,
    memory_usage: float,
    inference_times: Optional[jnp.ndarray] = None,
    convergence_time: Optional[float] = None
) -> EfficiencyMetrics:
    """
    Compute efficiency and resource usage metrics.
    
    Args:
        total_interventions: Total number of interventions performed
        computational_time: Total computational time in seconds
        memory_usage: Peak memory usage in GB
        inference_times: Optional per-step inference times [T]
        convergence_time: Optional time to convergence
        
    Returns:
        EfficiencyMetrics with computed metrics
    """
    # Basic efficiency
    interventions_per_second = total_interventions / (computational_time + 1e-8)
    
    # Average inference time
    if inference_times is not None:
        avg_inference_time = float(jnp.mean(inference_times))
    else:
        avg_inference_time = computational_time / total_interventions
    
    # Time to convergence
    time_to_convergence = convergence_time or computational_time
    
    return EfficiencyMetrics(
        total_interventions=total_interventions,
        computational_time_seconds=computational_time,
        memory_usage_gb=memory_usage,
        interventions_per_second=float(interventions_per_second),
        time_to_convergence=time_to_convergence,
        inference_time_per_step=avg_inference_time
    )


def compute_composite_metrics(
    causal_metrics: CausalDiscoveryMetrics,
    optimization_metrics: OptimizationMetrics,
    efficiency_metrics: EfficiencyMetrics,
    weights: Optional[Dict[str, float]] = None
) -> CompositeMetrics:
    """
    Compute composite evaluation metrics with optional weighting.
    
    Args:
        causal_metrics: Causal discovery metrics
        optimization_metrics: Optimization metrics
        efficiency_metrics: Efficiency metrics
        weights: Optional weights for different metric categories
        
    Returns:
        CompositeMetrics with overall scores
    """
    # Default weights
    default_weights = {
        'causal_discovery': 0.4,
        'optimization': 0.4,
        'efficiency': 0.2
    }
    weights = weights or default_weights
    
    # Normalize metrics to [0, 1] scale for combination
    normalized_causal = causal_metrics.f1_score
    normalized_optimization = jnp.clip(optimization_metrics.sample_efficiency / 10.0, 0.0, 1.0)
    normalized_efficiency = jnp.clip(efficiency_metrics.interventions_per_second / 100.0, 0.0, 1.0)
    
    # Compute weighted scores
    weighted_scores = {
        'causal_discovery': float(normalized_causal * weights['causal_discovery']),
        'optimization': float(normalized_optimization * weights['optimization']),
        'efficiency': float(normalized_efficiency * weights['efficiency'])
    }
    
    # Overall score
    overall_score = sum(weighted_scores.values())
    
    return CompositeMetrics(
        causal_discovery=causal_metrics,
        optimization=optimization_metrics,
        efficiency=efficiency_metrics,
        overall_score=overall_score,
        weighted_score=weighted_scores
    )


# Specialized metric functions
def compute_intervention_quality_metrics(
    interventions: jnp.ndarray,
    targets: jnp.ndarray,
    diversity_threshold: float = 0.1
) -> Dict[str, float]:
    """
    Compute intervention quality and diversity metrics.
    
    Args:
        interventions: Intervention matrices [T, N, N]
        targets: Target variables [T, N]
        diversity_threshold: Threshold for considering interventions diverse
        
    Returns:
        Dictionary of intervention quality metrics
    """
    num_interventions = interventions.shape[0]
    
    # Intervention diversity (unique interventions)
    unique_interventions = []
    for i in range(num_interventions):
        is_unique = True
        for j in range(i):
            if jnp.allclose(interventions[i], interventions[j], atol=diversity_threshold):
                is_unique = False
                break
        if is_unique:
            unique_interventions.append(i)
    
    diversity_ratio = len(unique_interventions) / num_interventions
    
    # Target coverage (how many different targets were explored)
    unique_targets = jnp.unique(jnp.argmax(targets, axis=1))
    target_coverage = len(unique_targets) / targets.shape[1]
    
    # Intervention magnitude statistics
    intervention_magnitudes = jnp.linalg.norm(interventions, axis=(1, 2))
    avg_magnitude = float(jnp.mean(intervention_magnitudes))
    std_magnitude = float(jnp.std(intervention_magnitudes))
    
    return {
        'diversity_ratio': diversity_ratio,
        'target_coverage': float(target_coverage),
        'avg_intervention_magnitude': avg_magnitude,
        'std_intervention_magnitude': std_magnitude,
        'num_unique_interventions': len(unique_interventions)
    }


def compute_learning_curve_metrics(
    objective_values: jnp.ndarray,
    window_size: int = 10
) -> Dict[str, jnp.ndarray]:
    """
    Compute learning curve analysis metrics.
    
    Args:
        objective_values: Sequence of objective values [T]
        window_size: Window size for smoothing
        
    Returns:
        Dictionary of learning curve metrics
    """
    # Smoothed learning curve
    smoothed_values = jnp.convolve(
        objective_values, 
        jnp.ones(window_size) / window_size, 
        mode='valid'
    )
    
    # Learning rate (improvement per step)
    learning_rates = jnp.diff(objective_values)
    smoothed_learning_rates = jnp.convolve(
        learning_rates,
        jnp.ones(min(window_size, len(learning_rates))) / min(window_size, len(learning_rates)),
        mode='valid'
    )
    
    # Plateau detection (where learning rate is consistently low)
    plateau_threshold = 0.01 * jnp.std(learning_rates)
    plateau_mask = jnp.abs(smoothed_learning_rates) < plateau_threshold
    
    return {
        'smoothed_objectives': smoothed_values,
        'learning_rates': learning_rates,
        'smoothed_learning_rates': smoothed_learning_rates,
        'plateau_mask': plateau_mask,
        'plateau_fraction': float(jnp.mean(plateau_mask))
    }


__all__ = [
    'CausalDiscoveryMetrics',
    'OptimizationMetrics',
    'EfficiencyMetrics', 
    'CompositeMetrics',
    'compute_causal_discovery_metrics',
    'compute_optimization_metrics',
    'compute_efficiency_metrics',
    'compute_composite_metrics',
    'compute_intervention_quality_metrics',
    'compute_learning_curve_metrics'
]