"""
Metric Utilities for ACBO Framework

Standardized metric calculations for structure learning and optimization.
Provides F1, SHD, target value metrics, and trajectory aggregation.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def compute_f1_score(
    predicted_parents: List[str],
    true_parents: List[str],
    all_variables: List[str]
) -> Tuple[float, float, float]:
    """
    Compute F1 score for parent set prediction.
    
    Args:
        predicted_parents: List of predicted parent variables
        true_parents: List of true parent variables
        all_variables: List of all variables in the graph
        
    Returns:
        Tuple of (f1_score, precision, recall)
    """
    if not true_parents and not predicted_parents:
        return 1.0, 1.0, 1.0
        
    pred_set = set(predicted_parents)
    true_set = set(true_parents)
    
    true_positives = len(pred_set & true_set)
    false_positives = len(pred_set - true_set)
    false_negatives = len(true_set - pred_set)
    
    if true_positives + false_positives == 0:
        precision = 0.0
    else:
        precision = true_positives / (true_positives + false_positives)
        
    if true_positives + false_negatives == 0:
        recall = 0.0
    else:
        recall = true_positives / (true_positives + false_negatives)
        
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    return f1, precision, recall


def compute_f1_from_marginals(
    marginal_probs: Dict[str, float],
    true_parents: List[str],
    target: str,
    threshold: float = 0.5
) -> Tuple[float, float, float]:
    """
    Compute F1 score from marginal probabilities.
    
    Args:
        marginal_probs: Dictionary of variable -> parent probability
        true_parents: List of true parent variables
        target: Target variable name
        threshold: Probability threshold for parent prediction
        
    Returns:
        Tuple of (f1_score, precision, recall)
    """
    # Get predicted parents based on threshold
    predicted_parents = [
        var for var, prob in marginal_probs.items()
        if var != target and prob > threshold
    ]
    
    # Get all variables from marginals
    all_variables = list(marginal_probs.keys())
    
    return compute_f1_score(predicted_parents, true_parents, all_variables)


def compute_shd(
    predicted_parents: List[str],
    true_parents: List[str]
) -> int:
    """
    Compute Structural Hamming Distance (SHD).
    
    For parent set prediction, SHD is the number of edge differences.
    
    Args:
        predicted_parents: List of predicted parent variables
        true_parents: List of true parent variables
        
    Returns:
        SHD value (number of edge differences)
    """
    pred_set = set(predicted_parents)
    true_set = set(true_parents)
    
    # SHD = edges that need to be added + edges that need to be removed
    return len(pred_set ^ true_set)  # Symmetric difference


def compute_shd_from_marginals(
    marginal_probs: Dict[str, float],
    true_parents: List[str],
    target: str,
    threshold: float = 0.5
) -> int:
    """
    Compute SHD from marginal probabilities.
    
    Args:
        marginal_probs: Dictionary of variable -> parent probability
        true_parents: List of true parent variables
        target: Target variable name
        threshold: Probability threshold for parent prediction
        
    Returns:
        SHD value
    """
    predicted_parents = [
        var for var, prob in marginal_probs.items()
        if var != target and prob > threshold
    ]
    
    return compute_shd(predicted_parents, true_parents)


def aggregate_trajectories(
    trajectories: List[Dict[str, List[float]]]
) -> Dict[str, Any]:
    """
    Aggregate multiple trajectory runs into mean and std.
    
    Args:
        trajectories: List of trajectory dictionaries
        
    Returns:
        Dictionary with aggregated statistics
    """
    if not trajectories:
        return {}
        
    # Find maximum trajectory length
    max_length = max(
        len(traj.get('target_values', [])) 
        for traj in trajectories
    )
    
    if max_length == 0:
        return {}
        
    # Initialize aggregated results
    aggregated = {
        'steps': list(range(max_length)),
        'target_values_mean': [],
        'target_values_std': [],
        'f1_scores_mean': [],
        'f1_scores_std': [],
        'shd_values_mean': [],
        'shd_values_std': [],
        'uncertainty_mean': [],
        'uncertainty_std': []
    }
    
    # Compute statistics at each step
    for i in range(max_length):
        # Target values
        target_vals = [
            traj['target_values'][i] 
            for traj in trajectories
            if 'target_values' in traj and i < len(traj['target_values'])
        ]
        if target_vals:
            aggregated['target_values_mean'].append(float(np.mean(target_vals)))
            aggregated['target_values_std'].append(float(np.std(target_vals)))
        else:
            aggregated['target_values_mean'].append(0.0)
            aggregated['target_values_std'].append(0.0)
            
        # F1 scores
        f1_vals = [
            traj['f1_scores'][i]
            for traj in trajectories
            if 'f1_scores' in traj and i < len(traj['f1_scores'])
        ]
        if f1_vals:
            aggregated['f1_scores_mean'].append(float(np.mean(f1_vals)))
            aggregated['f1_scores_std'].append(float(np.std(f1_vals)))
        else:
            aggregated['f1_scores_mean'].append(0.0)
            aggregated['f1_scores_std'].append(0.0)
            
        # SHD values
        shd_vals = [
            traj['shd_values'][i]
            for traj in trajectories
            if 'shd_values' in traj and i < len(traj['shd_values'])
        ]
        if shd_vals:
            aggregated['shd_values_mean'].append(float(np.mean(shd_vals)))
            aggregated['shd_values_std'].append(float(np.std(shd_vals)))
        else:
            aggregated['shd_values_mean'].append(0.0)
            aggregated['shd_values_std'].append(0.0)
            
        # Uncertainty
        uncertainty_vals = [
            traj.get('uncertainty_bits', [])[i] if i < len(traj.get('uncertainty_bits', [])) else 0.0
            for traj in trajectories
        ]
        if uncertainty_vals:
            aggregated['uncertainty_mean'].append(float(np.mean(uncertainty_vals)))
            aggregated['uncertainty_std'].append(float(np.std(uncertainty_vals)))
            
    return aggregated


def compute_improvement(
    initial_value: float,
    final_value: float,
    optimization_direction: str = "MINIMIZE"
) -> float:
    """
    Compute improvement metric based on optimization direction.
    
    Args:
        initial_value: Starting target value
        final_value: Final target value
        optimization_direction: "MINIMIZE" or "MAXIMIZE"
        
    Returns:
        Improvement value (positive is better)
    """
    if optimization_direction == "MINIMIZE":
        # For minimization, improvement is initial - final
        return initial_value - final_value
    else:
        # For maximization, improvement is final - initial
        return final_value - initial_value


def compute_regret(
    achieved_value: float,
    optimal_value: float,
    optimization_direction: str = "MINIMIZE"
) -> float:
    """
    Compute regret compared to optimal value.
    
    Args:
        achieved_value: Value achieved by method
        optimal_value: Best possible value
        optimization_direction: "MINIMIZE" or "MAXIMIZE"
        
    Returns:
        Regret (non-negative, lower is better)
    """
    if optimization_direction == "MINIMIZE":
        return max(0, achieved_value - optimal_value)
    else:
        return max(0, optimal_value - achieved_value)


def extract_trajectory_from_history(
    learning_history: List[Dict[str, Any]],
    true_parents: List[str],
    target: str
) -> Dict[str, List[float]]:
    """
    Extract trajectory data from learning history.
    
    Args:
        learning_history: List of step dictionaries
        true_parents: True parent variables
        target: Target variable
        
    Returns:
        Dictionary with trajectory arrays
    """
    if not learning_history:
        return {
            'steps': [],
            'target_values': [],
            'f1_scores': [],
            'shd_values': [],
            'uncertainty_bits': []
        }
        
    trajectory = {
        'steps': list(range(len(learning_history))),
        'target_values': [],
        'f1_scores': [],
        'shd_values': [],
        'uncertainty_bits': []
    }
    
    for step in learning_history:
        # Extract target value
        target_val = step.get('outcome_value', step.get('target_value', 0.0))
        trajectory['target_values'].append(float(target_val))
        
        # Extract or compute structure metrics
        if 'f1_score' in step:
            trajectory['f1_scores'].append(float(step['f1_score']))
        elif 'marginals' in step:
            f1, _, _ = compute_f1_from_marginals(
                step['marginals'], true_parents, target
            )
            trajectory['f1_scores'].append(float(f1))
        else:
            trajectory['f1_scores'].append(0.0)
            
        if 'shd' in step:
            trajectory['shd_values'].append(float(step['shd']))
        elif 'marginals' in step:
            shd = compute_shd_from_marginals(
                step['marginals'], true_parents, target
            )
            trajectory['shd_values'].append(float(shd))
        else:
            trajectory['shd_values'].append(float(len(true_parents)))
            
        # Extract uncertainty
        trajectory['uncertainty_bits'].append(
            float(step.get('uncertainty', 0.0))
        )
        
    return trajectory


def compute_area_under_curve(
    values: List[float],
    normalize: bool = True
) -> float:
    """
    Compute area under curve for a trajectory.
    
    Args:
        values: List of values over time
        normalize: Whether to normalize by number of steps
        
    Returns:
        AUC value
    """
    if not values:
        return 0.0
        
    # Use trapezoidal rule
    auc = np.trapz(values)
    
    if normalize and len(values) > 1:
        auc = auc / (len(values) - 1)
        
    return float(auc)


def compute_convergence_rate(
    trajectory: List[float],
    threshold: float = 0.95
) -> Optional[int]:
    """
    Compute steps to convergence.
    
    Args:
        trajectory: Value trajectory
        threshold: Fraction of final value to consider converged
        
    Returns:
        Steps to convergence or None if not converged
    """
    if not trajectory or len(trajectory) < 2:
        return None
        
    final_value = trajectory[-1]
    target_value = final_value * threshold
    
    for i, value in enumerate(trajectory):
        if abs(value - final_value) <= abs(final_value - target_value):
            return i
            
    return None


def compute_sample_efficiency(
    achieved_improvement: float,
    num_interventions: int
) -> float:
    """
    Compute sample efficiency (improvement per intervention).
    
    Args:
        achieved_improvement: Total improvement achieved
        num_interventions: Number of interventions used
        
    Returns:
        Sample efficiency metric
    """
    if num_interventions == 0:
        return 0.0
        
    return achieved_improvement / num_interventions


def format_metrics_table(
    method_metrics: Dict[str, Dict[str, float]]
) -> str:
    """
    Format metrics as a readable table.
    
    Args:
        method_metrics: Dictionary of method -> metrics
        
    Returns:
        Formatted table string
    """
    if not method_metrics:
        return "No metrics available"
        
    # Get all metric names
    all_metrics = set()
    for metrics in method_metrics.values():
        all_metrics.update(metrics.keys())
    metric_names = sorted(all_metrics)
    
    # Build table
    lines = []
    
    # Header
    header = ["Method"] + metric_names
    col_widths = [max(len(h), 20) for h in header]
    
    # Update widths based on values
    for method, metrics in method_metrics.items():
        col_widths[0] = max(col_widths[0], len(method))
        for i, metric in enumerate(metric_names):
            value_str = f"{metrics.get(metric, 0.0):.4f}"
            col_widths[i+1] = max(col_widths[i+1], len(value_str))
            
    # Format header
    header_line = " | ".join(
        h.ljust(w) for h, w in zip(header, col_widths)
    )
    lines.append(header_line)
    lines.append("-" * len(header_line))
    
    # Format rows
    for method, metrics in sorted(method_metrics.items()):
        row = [method]
        for metric in metric_names:
            value = metrics.get(metric, 0.0)
            row.append(f"{value:.4f}")
        
        row_line = " | ".join(
            v.ljust(w) for v, w in zip(row, col_widths)
        )
        lines.append(row_line)
        
    return "\n".join(lines)