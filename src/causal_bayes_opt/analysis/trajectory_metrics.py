"""
Trajectory Metrics Computation

Utility functions for computing derived metrics from trajectory data,
following functional programming principles.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union, FrozenSet
import numpy as onp
import jax.numpy as jnp

# Local imports
from ..acquisition.trajectory import TrajectoryBuffer, TrajectoryStep
from ..acquisition.state import AcquisitionState

logger = logging.getLogger(__name__)


def compute_true_parent_likelihood(
    marginal_probs: Dict[str, float], 
    true_parents: Union[List[str], FrozenSet[str]]
) -> float:
    """
    Compute posterior probability of true parent set from marginal probabilities.
    
    Uses independence assumption: P(true_parent_set) = ∏P(is_parent) × ∏P(not_parent)
    
    Args:
        marginal_probs: Dictionary mapping variables to P(is_parent | data)
        true_parents: Set of variables that are true parents
        
    Returns:
        Probability of the true parent set under current posterior
        
    Example:
        >>> marginals = {'X1': 0.8, 'X2': 0.1, 'X3': 0.9}
        >>> true_parents = ['X1', 'X3']
        >>> likelihood = compute_true_parent_likelihood(marginals, true_parents)
        >>> # Returns: 0.8 * 0.9 * (1 - 0.1) = 0.648
    """
    if not marginal_probs:
        return 0.0
    
    true_parents_set = set(true_parents) if isinstance(true_parents, list) else set(true_parents)
    
    likelihood = 1.0
    
    for variable, prob_is_parent in marginal_probs.items():
        if variable in true_parents_set:
            # True parent: use P(is_parent)
            likelihood *= prob_is_parent
        else:
            # Not a true parent: use P(not_parent) = 1 - P(is_parent)
            likelihood *= (1.0 - prob_is_parent)
    
    return likelihood


def compute_f1_score_from_marginals(
    marginal_probs: Dict[str, float],
    true_parents: Union[List[str], FrozenSet[str]],
    threshold: float = 0.5
) -> float:
    """
    Compute F1 score for structure recovery from marginal probabilities.
    
    Args:
        marginal_probs: Dictionary mapping variables to P(is_parent | data)
        true_parents: Set of variables that are true parents
        threshold: Threshold for considering a variable as predicted parent
        
    Returns:
        F1 score for structure recovery
    """
    if not marginal_probs:
        return 0.0
        
    true_parents_set = set(true_parents) if isinstance(true_parents, list) else set(true_parents)
    
    # Predict parents based on threshold
    predicted_parents = {var for var, prob in marginal_probs.items() if prob > threshold}
    
    # Compute precision, recall, F1
    if not predicted_parents and not true_parents_set:
        return 1.0  # Perfect when both are empty
    
    if not predicted_parents:
        return 0.0  # No predictions but true parents exist
    
    if not true_parents_set:
        return 0.0  # Predictions but no true parents
    
    true_positives = len(predicted_parents & true_parents_set)
    precision = true_positives / len(predicted_parents)
    recall = true_positives / len(true_parents_set)
    
    if precision + recall == 0:
        return 0.0
    
    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score


def extract_state_metrics(
    state: AcquisitionState, 
    true_parents: List[str],
    f1_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Extract metrics from a single acquisition state.
    
    Args:
        state: Acquisition state to analyze
        true_parents: Ground truth parent set
        f1_threshold: Threshold for F1 score calculation
        
    Returns:
        Dictionary with extracted metrics
    """
    marginal_probs = dict(state.marginal_parent_probs)
    
    return {
        'step': state.step,
        'uncertainty_bits': state.uncertainty_bits,
        'best_value': state.best_value,
        'buffer_size': state.buffer_statistics.total_samples,
        'true_parent_likelihood': compute_true_parent_likelihood(marginal_probs, true_parents),
        'f1_score': compute_f1_score_from_marginals(marginal_probs, true_parents, f1_threshold),
        'marginal_probs': marginal_probs
    }


def compute_trajectory_metrics(
    trajectory: TrajectoryBuffer, 
    true_parents: List[str],
    target_variable: str,
    f1_threshold: float = 0.5
) -> Dict[str, List[float]]:
    """
    Extract all metrics from trajectory for visualization.
    
    Args:
        trajectory: Trajectory buffer containing experiment data
        true_parents: Ground truth parent set for target variable
        target_variable: Name of target variable
        f1_threshold: Threshold for F1 score calculation
        
    Returns:
        Dictionary with time series of metrics
    """
    if trajectory.is_empty():
        return {
            'steps': [],
            'true_parent_likelihood': [],
            'f1_scores': [],
            'target_values': [],
            'uncertainty_bits': [],
            'intervention_counts': [],
            'rewards': []
        }
    
    # Get state progression and step metrics
    states = trajectory.get_state_progression()
    steps = [state.step for state in states]
    
    # Compute derived metrics for each state
    state_metrics = [extract_state_metrics(state, true_parents, f1_threshold) for state in states]
    
    # Extract time series
    true_parent_likelihoods = [m['true_parent_likelihood'] for m in state_metrics]
    f1_scores = [m['f1_score'] for m in state_metrics]
    target_values = [m['best_value'] for m in state_metrics]
    uncertainty_values = [m['uncertainty_bits'] for m in state_metrics]
    
    # Get reward progression (from trajectory steps)
    rewards = trajectory.get_reward_history()
    
    # Compute intervention counts over time
    intervention_counts = []
    for state in states:
        intervention_counts.append(state.buffer_statistics.num_interventions)
    
    return {
        'steps': steps,
        'true_parent_likelihood': true_parent_likelihoods,
        'f1_scores': f1_scores,
        'target_values': target_values,
        'uncertainty_bits': uncertainty_values,
        'intervention_counts': intervention_counts,
        'rewards': rewards if len(rewards) == len(steps) else rewards[:len(steps)]
    }


def analyze_convergence_trajectory(
    trajectory_metrics: Dict[str, List[float]],
    convergence_threshold: float = 0.9,
    patience: int = 3
) -> Dict[str, Any]:
    """
    Analyze convergence properties of a trajectory.
    
    Args:
        trajectory_metrics: Output from compute_trajectory_metrics
        convergence_threshold: Threshold for considering convergence achieved
        patience: Number of steps to maintain threshold for convergence
        
    Returns:
        Dictionary with convergence analysis
    """
    if not trajectory_metrics['true_parent_likelihood']:
        return {
            'converged': False,
            'convergence_step': None,
            'final_likelihood': 0.0,
            'max_likelihood': 0.0,
            'convergence_rate': 0.0
        }
    
    likelihoods = trajectory_metrics['true_parent_likelihood']
    steps = trajectory_metrics['steps']
    
    # Find convergence point
    convergence_step = None
    for i in range(len(likelihoods) - patience + 1):
        # Check if likelihood stays above threshold for 'patience' steps
        window = likelihoods[i:i + patience]
        if all(l >= convergence_threshold for l in window):
            convergence_step = steps[i]
            break
    
    # Compute convergence rate (improvement per step)
    if len(likelihoods) > 1:
        # Linear regression on likelihood vs step
        x = onp.array(steps)
        y = onp.array(likelihoods)
        if len(x) > 1:
            convergence_rate = float(onp.polyfit(x, y, 1)[0])  # Slope
        else:
            convergence_rate = 0.0
    else:
        convergence_rate = 0.0
    
    return {
        'converged': convergence_step is not None,
        'convergence_step': convergence_step,
        'final_likelihood': likelihoods[-1],
        'max_likelihood': max(likelihoods),
        'convergence_rate': convergence_rate,
        'total_steps': len(steps)
    }


def extract_learning_curves(
    results_by_method: Dict[str, List[Dict[str, List[float]]]]
) -> Dict[str, Dict[str, Any]]:
    """
    Extract learning curves for comparison across methods.
    
    Args:
        results_by_method: Dictionary mapping method names to lists of trajectory metrics
        
    Returns:
        Dictionary with aggregated learning curves per method
    """
    learning_curves = {}
    
    for method_name, results_list in results_by_method.items():
        if not results_list:
            continue
        
        # Find maximum length across all runs
        max_length = max(len(result.get('steps', [])) for result in results_list)
        
        if max_length == 0:
            continue
        
        # Initialize aggregated arrays
        all_likelihoods = []
        all_f1_scores = []
        all_target_values = []
        
        for result in results_list:
            # Pad sequences to max_length
            likelihood = result.get('true_parent_likelihood', [])
            f1_scores = result.get('f1_scores', [])
            target_values = result.get('target_values', [])
            
            # Extend with last value if shorter (forward fill)
            if len(likelihood) < max_length and len(likelihood) > 0:
                likelihood = likelihood + [likelihood[-1]] * (max_length - len(likelihood))
            if len(f1_scores) < max_length and len(f1_scores) > 0:
                f1_scores = f1_scores + [f1_scores[-1]] * (max_length - len(f1_scores))
            if len(target_values) < max_length and len(target_values) > 0:
                target_values = target_values + [target_values[-1]] * (max_length - len(target_values))
            
            all_likelihoods.append(likelihood[:max_length])
            all_f1_scores.append(f1_scores[:max_length])
            all_target_values.append(target_values[:max_length])
        
        # Compute statistics across runs
        if all_likelihoods:
            likelihood_array = onp.array(all_likelihoods)
            f1_array = onp.array(all_f1_scores) if all_f1_scores else likelihood_array * 0
            target_array = onp.array(all_target_values) if all_target_values else likelihood_array * 0
            
            learning_curves[method_name] = {
                'steps': list(range(max_length)),
                'likelihood_mean': onp.mean(likelihood_array, axis=0).tolist(),
                'likelihood_std': onp.std(likelihood_array, axis=0).tolist(),
                'f1_mean': onp.mean(f1_array, axis=0).tolist(),
                'f1_std': onp.std(f1_array, axis=0).tolist(),
                'target_mean': onp.mean(target_array, axis=0).tolist(),
                'target_std': onp.std(target_array, axis=0).tolist(),
                'n_runs': len(all_likelihoods)
            }
    
    return learning_curves


def compute_f1_with_multiple_thresholds(
    marginal_probs: Dict[str, float],
    true_parents: Union[List[str], FrozenSet[str]],
    thresholds: List[float] = None
) -> Dict[str, float]:
    """
    Compute F1 scores for multiple thresholds to find optimal binarization.
    
    Args:
        marginal_probs: Dictionary mapping variables to P(is_parent | data)
        true_parents: Set of variables that are true parents
        thresholds: List of thresholds to test
        
    Returns:
        Dictionary mapping threshold values to F1 scores
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    f1_scores = {}
    for threshold in thresholds:
        f1_score = compute_f1_score_from_marginals(marginal_probs, true_parents, threshold)
        f1_scores[threshold] = f1_score
    
    return f1_scores


def compute_precision_recall_curve(
    marginal_probs: Dict[str, float],
    true_parents: Union[List[str], FrozenSet[str]],
    thresholds: List[float] = None
) -> Dict[str, List[float]]:
    """
    Compute precision-recall curve for different thresholds.
    
    Args:
        marginal_probs: Dictionary mapping variables to P(is_parent | data)
        true_parents: Set of variables that are true parents
        thresholds: List of thresholds to test
        
    Returns:
        Dictionary with 'thresholds', 'precision', 'recall', 'f1_scores'
    """
    if thresholds is None:
        thresholds = [i/100.0 for i in range(0, 101, 5)]  # 0.0 to 1.0 in steps of 0.05
    
    true_parents_set = set(true_parents) if isinstance(true_parents, list) else set(true_parents)
    
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        predicted_parents = {var for var, prob in marginal_probs.items() if prob > threshold}
        
        if not predicted_parents and not true_parents_set:
            precision, recall, f1 = 1.0, 1.0, 1.0
        elif not predicted_parents:
            precision, recall, f1 = 0.0, 0.0, 0.0
        elif not true_parents_set:
            precision, recall, f1 = 0.0, 0.0, 0.0
        else:
            true_positives = len(predicted_parents & true_parents_set)
            precision = true_positives / len(predicted_parents)
            recall = true_positives / len(true_parents_set)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    return {
        'thresholds': thresholds,
        'precision': precisions,
        'recall': recalls,
        'f1_scores': f1_scores
    }


def find_optimal_f1_threshold(
    marginal_probs: Dict[str, float],
    true_parents: Union[List[str], FrozenSet[str]],
    thresholds: List[float] = None
) -> Tuple[float, float]:
    """
    Find the threshold that maximizes F1 score.
    
    Args:
        marginal_probs: Dictionary mapping variables to P(is_parent | data)
        true_parents: Set of variables that are true parents
        thresholds: List of thresholds to test
        
    Returns:
        Tuple of (optimal_threshold, max_f1_score)
    """
    f1_scores = compute_f1_with_multiple_thresholds(marginal_probs, true_parents, thresholds)
    
    if not f1_scores:
        return 0.5, 0.0
    
    optimal_threshold = max(f1_scores, key=f1_scores.get)
    max_f1_score = f1_scores[optimal_threshold]
    
    return optimal_threshold, max_f1_score


def find_youden_j_threshold(
    marginal_probs: Dict[str, float],
    true_parents: Union[List[str], FrozenSet[str]],
    thresholds: List[float] = None
) -> Tuple[float, float]:
    """
    Find threshold using Youden's J statistic (sensitivity + specificity - 1).
    
    Args:
        marginal_probs: Dictionary mapping variables to P(is_parent | data)
        true_parents: Set of variables that are true parents
        thresholds: List of thresholds to test
        
    Returns:
        Tuple of (optimal_threshold, max_j_statistic)
    """
    if thresholds is None:
        thresholds = [i/100.0 for i in range(0, 101, 5)]
    
    true_parents_set = set(true_parents) if isinstance(true_parents, list) else set(true_parents)
    all_variables = set(marginal_probs.keys())
    true_negatives_set = all_variables - true_parents_set
    
    j_statistics = {}
    
    for threshold in thresholds:
        predicted_parents = {var for var, prob in marginal_probs.items() if prob > threshold}
        predicted_negatives = all_variables - predicted_parents
        
        # Compute confusion matrix components
        true_positives = len(predicted_parents & true_parents_set)
        false_positives = len(predicted_parents & true_negatives_set)
        true_negatives = len(predicted_negatives & true_negatives_set)
        false_negatives = len(predicted_negatives & true_parents_set)
        
        # Compute sensitivity (recall) and specificity
        sensitivity = true_positives / len(true_parents_set) if len(true_parents_set) > 0 else 0.0
        specificity = true_negatives / len(true_negatives_set) if len(true_negatives_set) > 0 else 0.0
        
        # Youden's J statistic = sensitivity + specificity - 1
        j_statistic = sensitivity + specificity - 1
        j_statistics[threshold] = j_statistic
    
    if not j_statistics:
        return 0.5, 0.0
    
    optimal_threshold = max(j_statistics, key=j_statistics.get)
    max_j_statistic = j_statistics[optimal_threshold]
    
    return optimal_threshold, max_j_statistic


def compute_expected_calibration_error(
    marginal_probs: Dict[str, float],
    true_parents: Union[List[str], FrozenSet[str]],
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE) for probability predictions.
    
    ECE measures how well the predicted probabilities match the actual outcomes.
    
    Args:
        marginal_probs: Dictionary mapping variables to P(is_parent | data)
        true_parents: Set of variables that are true parents
        n_bins: Number of bins for calibration computation
        
    Returns:
        Expected Calibration Error value
    """
    if not marginal_probs:
        return 0.0
    
    true_parents_set = set(true_parents) if isinstance(true_parents, list) else set(true_parents)
    
    # Create bins
    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    total_samples = len(marginal_probs)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = []
        bin_accuracies = []
        bin_confidences = []
        
        for var, prob in marginal_probs.items():
            if bin_lower <= prob < bin_upper or (bin_upper == 1.0 and prob == 1.0):
                in_bin.append(var)
                bin_confidences.append(prob)
                # True if variable is actually a parent
                actual = 1.0 if var in true_parents_set else 0.0
                bin_accuracies.append(actual)
        
        if len(in_bin) > 0:
            # Compute average confidence and accuracy in this bin
            avg_confidence = sum(bin_confidences) / len(bin_confidences)
            avg_accuracy = sum(bin_accuracies) / len(bin_accuracies)
            
            # Weight by number of samples in bin
            bin_weight = len(in_bin) / total_samples
            ece += bin_weight * abs(avg_confidence - avg_accuracy)
    
    return ece


def compute_brier_score(
    marginal_probs: Dict[str, float],
    true_parents: Union[List[str], FrozenSet[str]]
) -> float:
    """
    Compute Brier Score for probability predictions.
    
    Brier Score = (1/N) * Σ(predicted_prob - actual)²
    Lower is better (0 = perfect, 1 = worst).
    
    Args:
        marginal_probs: Dictionary mapping variables to P(is_parent | data)
        true_parents: Set of variables that are true parents
        
    Returns:
        Brier Score value
    """
    if not marginal_probs:
        return 0.0
    
    true_parents_set = set(true_parents) if isinstance(true_parents, list) else set(true_parents)
    
    squared_errors = []
    
    for var, predicted_prob in marginal_probs.items():
        # Actual outcome: 1 if parent, 0 if not
        actual = 1.0 if var in true_parents_set else 0.0
        
        # Squared error for this prediction
        squared_error = (predicted_prob - actual) ** 2
        squared_errors.append(squared_error)
    
    # Brier score is mean squared error
    brier_score = sum(squared_errors) / len(squared_errors) if squared_errors else 0.0
    
    return brier_score


def compute_comprehensive_metrics(
    marginal_probs: Dict[str, float],
    true_parents: Union[List[str], FrozenSet[str]],
    f1_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute comprehensive set of metrics for marginal probability predictions.
    
    Args:
        marginal_probs: Dictionary mapping variables to P(is_parent | data)
        true_parents: Set of variables that are true parents
        f1_threshold: Threshold for F1 score calculation
        
    Returns:
        Dictionary with all computed metrics
    """
    # Basic metrics
    likelihood = compute_true_parent_likelihood(marginal_probs, true_parents)
    f1_score = compute_f1_score_from_marginals(marginal_probs, true_parents, f1_threshold)
    
    # Distribution-aware metrics
    ece = compute_expected_calibration_error(marginal_probs, true_parents)
    brier = compute_brier_score(marginal_probs, true_parents)
    
    # Optimal thresholds
    optimal_f1_threshold, max_f1 = find_optimal_f1_threshold(marginal_probs, true_parents)
    youden_threshold, max_j = find_youden_j_threshold(marginal_probs, true_parents)
    
    return {
        'true_parent_likelihood': likelihood,
        'f1_score': f1_score,
        'f1_score_optimal': max_f1,
        'optimal_f1_threshold': optimal_f1_threshold,
        'youden_j_threshold': youden_threshold,
        'max_youden_j': max_j,
        'expected_calibration_error': ece,
        'brier_score': brier,
        'marginal_probs': marginal_probs
    }


def compute_intervention_efficiency(
    trajectory_metrics: Dict[str, List[float]],
    efficiency_threshold: float = 0.8
) -> Dict[str, Any]:
    """
    Compute intervention efficiency metrics.
    
    Args:
        trajectory_metrics: Output from compute_trajectory_metrics
        efficiency_threshold: Threshold for considering good performance
        
    Returns:
        Dictionary with efficiency metrics
    """
    if not trajectory_metrics['true_parent_likelihood']:
        return {
            'steps_to_threshold': None,
            'interventions_to_threshold': None,
            'efficiency_ratio': 0.0
        }
    
    likelihoods = trajectory_metrics['true_parent_likelihood']
    intervention_counts = trajectory_metrics['intervention_counts']
    
    # Find first time we reach threshold
    steps_to_threshold = None
    interventions_to_threshold = None
    
    for i, likelihood in enumerate(likelihoods):
        if likelihood >= efficiency_threshold:
            steps_to_threshold = i + 1
            interventions_to_threshold = intervention_counts[i] if i < len(intervention_counts) else None
            break
    
    # Compute efficiency ratio (improvement per intervention)
    if len(likelihoods) > 1 and len(intervention_counts) > 1:
        likelihood_improvement = likelihoods[-1] - likelihoods[0]
        total_interventions = intervention_counts[-1] - intervention_counts[0]
        efficiency_ratio = likelihood_improvement / total_interventions if total_interventions > 0 else 0.0
    else:
        efficiency_ratio = 0.0
    
    return {
        'steps_to_threshold': steps_to_threshold,
        'interventions_to_threshold': interventions_to_threshold,
        'efficiency_ratio': efficiency_ratio,
        'final_likelihood': likelihoods[-1] if likelihoods else 0.0
    }


# Utility functions for working with experiment results
def extract_metrics_from_experiment_result(
    experiment_result: Dict[str, Any],
    true_parents: List[str],
    f1_threshold: float = 0.5
) -> Dict[str, List[float]]:
    """
    Extract trajectory metrics from experiment result dictionary.
    
    Args:
        experiment_result: Result from running an experiment
        true_parents: Ground truth parent set
        f1_threshold: Threshold for F1 score calculation
        
    Returns:
        Trajectory metrics suitable for visualization
    """
    # Handle different result formats
    if 'detailed_results' in experiment_result and experiment_result['detailed_results']:
        detailed = experiment_result['detailed_results']
        
        # NEW: Extract from learning_history (the actual data format)
        learning_history = detailed.get('learning_history', [])
        
        if learning_history:
            # Extract time series data from learning history
            steps = []
            target_values = []
            uncertainty_values = []
            marginals_list = []
            
            for step_data in learning_history:
                steps.append(step_data.get('step', 0))
                target_values.append(step_data.get('outcome_value', 0.0))
                uncertainty_values.append(step_data.get('uncertainty', 0.0))
                
                # Extract marginals for this step
                marginals = step_data.get('marginals', {})
                marginals_list.append(marginals)
            
            # Compute derived metrics from marginals
            true_parent_likelihoods = []
            f1_scores = []
            
            for marginals in marginals_list:
                if isinstance(marginals, dict) and marginals:
                    likelihood = compute_true_parent_likelihood(marginals, true_parents)
                    f1_score = compute_f1_score_from_marginals(marginals, true_parents, f1_threshold)
                    true_parent_likelihoods.append(likelihood)
                    f1_scores.append(f1_score)
                else:
                    # Handle empty/invalid marginals
                    true_parent_likelihoods.append(0.0)
                    f1_scores.append(0.0)
            
            return {
                'steps': steps,
                'true_parent_likelihood': true_parent_likelihoods,
                'f1_scores': f1_scores,
                'target_values': target_values,
                'uncertainty_bits': uncertainty_values,
                'intervention_counts': steps  # Use step number as proxy
            }
        
        # LEGACY: Try old format (marginal_prob_progress) for backward compatibility
        marginal_progress = detailed.get('marginal_prob_progress', [])
        target_progress = detailed.get('target_progress', [])
        uncertainty_progress = detailed.get('uncertainty_progress', [])
        
        if marginal_progress:
            # Compute derived metrics
            true_parent_likelihoods = []
            f1_scores = []
            
            for marginals in marginal_progress:
                if isinstance(marginals, dict):
                    likelihood = compute_true_parent_likelihood(marginals, true_parents)
                    f1_score = compute_f1_score_from_marginals(marginals, true_parents, f1_threshold)
                    true_parent_likelihoods.append(likelihood)
                    f1_scores.append(f1_score)
            
            return {
                'steps': list(range(len(target_progress))),
                'true_parent_likelihood': true_parent_likelihoods,
                'f1_scores': f1_scores,
                'target_values': target_progress,
                'uncertainty_bits': uncertainty_progress,
                'intervention_counts': list(range(len(target_progress)))  # Simplified
            }
    
    # Fallback for minimal results
    return {
        'steps': [0],
        'true_parent_likelihood': [0.0],
        'f1_scores': [0.0],
        'target_values': [experiment_result.get('target_improvement', 0.0)],
        'uncertainty_bits': [0.0],
        'intervention_counts': [0]
    }


# Export utility functions
__all__ = [
    'compute_true_parent_likelihood',
    'compute_f1_score_from_marginals',
    'compute_f1_with_multiple_thresholds',
    'compute_precision_recall_curve',
    'find_optimal_f1_threshold',
    'find_youden_j_threshold',
    'compute_expected_calibration_error',
    'compute_brier_score',
    'compute_comprehensive_metrics',
    'extract_state_metrics',
    'compute_trajectory_metrics',
    'analyze_convergence_trajectory',
    'extract_learning_curves',
    'compute_intervention_efficiency',
    'extract_metrics_from_experiment_result'
]