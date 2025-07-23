"""
Trajectory Helper for ACBO Comparison

This module provides utilities to ensure all methods return proper trajectory data
for visualization, even if they don't perform structure learning.
"""

import logging
from typing import Dict, List, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def add_trajectory_data_to_result(
    result: Dict[str, Any], 
    scm: Any,
    include_structure_metrics: bool = True
) -> Dict[str, Any]:
    """
    Add trajectory data to a result that lacks it.
    
    This is used for methods like random_untrained that don't track structure learning
    but still need trajectory data for visualization.
    
    Args:
        result: Original result dictionary
        scm: The SCM used in the experiment
        include_structure_metrics: Whether to include F1/SHD (set False for non-learning methods)
        
    Returns:
        Enhanced result with trajectory data
    """
    from causal_bayes_opt.data_structures.scm import get_target, get_parents
    
    # Check if trajectory data already exists in the expected location
    detailed_results = result.get('detailed_results', {})
    if any(key in detailed_results for key in ['f1_scores', 'shd_values', 'true_parent_likelihood']):
        return result  # Already has trajectory data
    
    # Get SCM info
    target = get_target(scm)
    true_parents = list(get_parents(scm, target)) if target else []
    n_true_parents = len(true_parents)
    
    # Extract learning history from either location
    learning_history = result.get('learning_history', detailed_results.get('learning_history', []))
    if not learning_history:
        logger.warning("No learning history found, cannot create trajectory data")
        return result
    
    # Build trajectory arrays
    steps = []
    target_values = []
    f1_scores = []
    shd_values = []
    true_parent_likelihood = []
    
    for i, step_data in enumerate(learning_history):
        steps.append(i)
        
        # Target value (this should always be available)
        target_value = step_data.get('outcome_value', 0.0)
        target_values.append(target_value)
        
        if include_structure_metrics:
            # For learning methods, use marginals if available
            # Check both 'marginals' and 'marginal_probs' keys
            marginals = step_data.get('marginals', step_data.get('marginal_probs', {}))
            
            if marginals:
                # Compute metrics from marginals
                from scripts.core.acbo_comparison.structure_metrics_helper import (
                    compute_f1_from_marginals,
                    compute_parent_probability,
                    compute_shd_from_marginals
                )
                
                f1, _, _ = compute_f1_from_marginals(marginals, true_parents, target)
                parent_prob = compute_parent_probability(marginals, true_parents)
                shd = compute_shd_from_marginals(marginals, true_parents, target)
                
                f1_scores.append(f1)
                true_parent_likelihood.append(parent_prob)
                shd_values.append(shd)
            else:
                # No marginals available - use defaults
                f1_scores.append(0.0)
                true_parent_likelihood.append(0.0)
                shd_values.append(n_true_parents)
        else:
            # For non-learning methods (like random_untrained)
            # Use static values to show no learning
            f1_scores.append(0.0)
            true_parent_likelihood.append(0.0)
            shd_values.append(n_true_parents)
    
    # Create trajectory data
    trajectory_data = {
        'steps': steps,
        'target_values': target_values,
        'f1_scores': f1_scores,
        'shd_values': shd_values,
        'true_parent_likelihood': true_parent_likelihood,
        'n_runs': 1  # Single run
    }
    
    # Add to detailed results
    if 'detailed_results' not in result:
        result['detailed_results'] = {}
    
    result['detailed_results']['trajectory_data'] = trajectory_data
    
    # Also add as separate arrays for compatibility
    result['detailed_results']['target_progress'] = target_values
    result['detailed_results']['f1_scores'] = f1_scores
    result['detailed_results']['shd_values'] = shd_values
    result['detailed_results']['true_parent_likelihood'] = true_parent_likelihood
    
    logger.info(f"Added trajectory data with {len(steps)} steps")
    
    return result


def ensure_demo_methods_have_trajectories(
    method_results: Dict[str, List[Dict[str, Any]]],
    scms: List[Tuple[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Ensure all demo methods have trajectory data for visualization.
    
    Args:
        method_results: Dictionary of method_name -> list of results
        scms: List of (name, scm) tuples used in experiments
        
    Returns:
        Enhanced method results with trajectory data
    """
    enhanced_results = {}
    
    for method_name, results in method_results.items():
        enhanced_method_results = []
        
        # Determine if this method performs structure learning
        include_structure = method_name not in ['Random + Untrained', 'random_untrained']
        
        for i, result in enumerate(results):
            # Get corresponding SCM
            scm_idx = i % len(scms)
            scm_name, scm = scms[scm_idx]
            
            # Add trajectory data if missing
            enhanced_result = add_trajectory_data_to_result(
                result.copy(), 
                scm,
                include_structure_metrics=include_structure
            )
            enhanced_method_results.append(enhanced_result)
        
        enhanced_results[method_name] = enhanced_method_results
        logger.info(f"Enhanced {method_name} with trajectory data")
    
    return enhanced_results


def create_mock_trajectory_for_baseline(
    n_steps: int,
    initial_value: float = 0.0,
    improvement_rate: float = 0.1,
    noise_scale: float = 0.05
) -> Dict[str, List[float]]:
    """
    Create mock trajectory data for baseline methods that don't learn.
    
    This is used for visualization when methods don't track real trajectories.
    
    Args:
        n_steps: Number of intervention steps
        initial_value: Starting target value
        improvement_rate: Rate of improvement per step
        noise_scale: Scale of random noise
        
    Returns:
        Dictionary with trajectory arrays
    """
    steps = list(range(n_steps))
    
    # Target values with some improvement and noise
    target_values = []
    current_value = initial_value
    
    for i in range(n_steps):
        # Add some improvement with noise
        improvement = improvement_rate * (1 + np.random.normal(0, noise_scale))
        current_value += improvement
        target_values.append(current_value)
    
    # Structure metrics stay constant (no learning)
    f1_scores = [0.0] * n_steps
    shd_values = [5.0] * n_steps  # Assume 5 edges different
    true_parent_likelihood = [0.0] * n_steps
    
    return {
        'steps': steps,
        'target_values': target_values,
        'f1_scores': f1_scores,
        'shd_values': shd_values,
        'true_parent_likelihood': true_parent_likelihood
    }