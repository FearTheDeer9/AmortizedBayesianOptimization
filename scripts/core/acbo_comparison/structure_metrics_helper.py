"""
Structure Metrics Helper for ACBO Comparison

This module provides utilities to compute structure learning metrics (F1 scores, parent probabilities)
from marginal probabilities, ensuring all methods can report these metrics.
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def compute_f1_from_marginals(marginal_probs: Dict[str, float], 
                            true_parents: List[str], 
                            target: str,
                            threshold: float = 0.5) -> Tuple[float, float, float]:
    """
    Compute F1 score from marginal parent probabilities.
    
    Args:
        marginal_probs: Dictionary of {variable: probability} for parent relationships
        true_parents: List of true parent variables
        target: Target variable name
        threshold: Probability threshold for binary classification
        
    Returns:
        Tuple of (f1_score, precision, recall)
    """
    if not marginal_probs:
        return 0.0, 0.0, 0.0
    
    # Get all variables except target
    all_vars = [var for var in marginal_probs.keys() if var != target]
    
    if not all_vars:
        return 0.0, 0.0, 0.0
    
    # Compute binary predictions
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    true_parent_set = set(true_parents)
    
    for var in all_vars:
        prob = marginal_probs.get(var, 0.0)
        predicted_parent = prob > threshold
        is_true_parent = var in true_parent_set
        
        if predicted_parent and is_true_parent:
            true_positives += 1
        elif predicted_parent and not is_true_parent:
            false_positives += 1
        elif not predicted_parent and is_true_parent:
            false_negatives += 1
    
    # Calculate metrics
    if true_positives + false_positives == 0:
        precision = 0.0
    else:
        precision = true_positives / (true_positives + false_positives)
    
    if true_positives + false_negatives == 0:
        recall = 0.0
    else:
        recall = true_positives / (true_positives + false_negatives)
    
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return f1_score, precision, recall


def compute_parent_probability(marginal_probs: Dict[str, float],
                             true_parents: List[str]) -> float:
    """
    Compute average probability assigned to true parents.
    
    Args:
        marginal_probs: Dictionary of {variable: probability} for parent relationships
        true_parents: List of true parent variables
        
    Returns:
        Average probability assigned to true parents (P(Parents|Data))
    """
    if not true_parents or not marginal_probs:
        return 0.0
    
    parent_probs = []
    for parent in true_parents:
        if parent in marginal_probs:
            parent_probs.append(marginal_probs[parent])
        else:
            parent_probs.append(0.0)
    
    return np.mean(parent_probs) if parent_probs else 0.0


def compute_shd_from_marginals(marginal_probs: Dict[str, float],
                              true_parents: List[str],
                              target: str,
                              threshold: float = 0.5) -> int:
    """
    Compute Structural Hamming Distance from marginal probabilities.
    
    Args:
        marginal_probs: Dictionary of {variable: probability} for parent relationships
        true_parents: List of true parent variables
        target: Target variable name
        threshold: Probability threshold for binary classification
        
    Returns:
        Structural Hamming Distance (number of edge differences)
    """
    if not marginal_probs:
        return len(true_parents)  # All edges missing
    
    # Get all variables except target
    all_vars = [var for var in marginal_probs.keys() if var != target]
    true_parent_set = set(true_parents)
    
    shd = 0
    
    # Count edge differences
    for var in all_vars:
        prob = marginal_probs.get(var, 0.0)
        predicted_parent = prob > threshold
        is_true_parent = var in true_parent_set
        
        if predicted_parent != is_true_parent:
            shd += 1
    
    # Add missing variables (if any true parents not in marginal_probs)
    for parent in true_parents:
        if parent not in marginal_probs:
            shd += 1
    
    return shd


def add_structure_metrics_to_result(result: Dict[str, Any],
                                  scm: Any,
                                  add_trajectory: bool = True) -> Dict[str, Any]:
    """
    Add structure learning metrics to a result dictionary.
    
    Args:
        result: Original result dictionary
        scm: The SCM used in the experiment
        add_trajectory: Whether to add trajectory data
        
    Returns:
        Enhanced result dictionary with structure metrics
    """
    from causal_bayes_opt.data_structures.scm import get_target, get_parents
    
    target = get_target(scm)
    true_parents = list(get_parents(scm, target)) if target else []
    
    # Check if we already have structure metrics
    if 'f1_score_final' in result or 'true_parent_likelihood_final' in result:
        return result  # Already has metrics
    
    # Get final marginal probabilities
    final_marginals = result.get('final_marginal_probs', {})
    
    if final_marginals:
        # Compute structure metrics
        f1_score, precision, recall = compute_f1_from_marginals(
            final_marginals, true_parents, target
        )
        parent_prob = compute_parent_probability(final_marginals, true_parents)
        shd = compute_shd_from_marginals(final_marginals, true_parents, target)
        
        # Add final metrics
        result['f1_score_final'] = f1_score
        result['precision_final'] = precision
        result['recall_final'] = recall
        result['true_parent_likelihood_final'] = parent_prob
        result['shd_final'] = shd
        
        # Add trajectory if requested and we have learning history
        if add_trajectory and 'learning_history' in result:
            learning_history = result['learning_history']
            
            f1_trajectory = []
            parent_prob_trajectory = []
            shd_trajectory = []
            
            for step in learning_history:
                step_marginals = step.get('marginal_probs', {})
                
                if step_marginals:
                    step_f1, _, _ = compute_f1_from_marginals(
                        step_marginals, true_parents, target
                    )
                    step_parent_prob = compute_parent_probability(
                        step_marginals, true_parents
                    )
                    step_shd = compute_shd_from_marginals(
                        step_marginals, true_parents, target
                    )
                else:
                    # Use defaults for baseline methods
                    step_f1 = 0.0
                    step_parent_prob = 0.0
                    step_shd = len(true_parents)
                
                f1_trajectory.append(step_f1)
                parent_prob_trajectory.append(step_parent_prob)
                shd_trajectory.append(step_shd)
            
            # Add trajectories
            result['f1_scores_trajectory'] = f1_trajectory
            result['true_parent_likelihood_trajectory'] = parent_prob_trajectory
            result['shd_values_trajectory'] = shd_trajectory
            result['steps_trajectory'] = list(range(len(learning_history)))
    
    else:
        # No marginal probabilities available - add defaults
        result['f1_score_final'] = 0.0
        result['true_parent_likelihood_final'] = 0.0
        result['shd_final'] = len(true_parents)
        
        if add_trajectory and 'learning_history' in result:
            n_steps = len(result['learning_history'])
            result['f1_scores_trajectory'] = [0.0] * n_steps
            result['true_parent_likelihood_trajectory'] = [0.0] * n_steps
            result['shd_values_trajectory'] = [len(true_parents)] * n_steps
            result['steps_trajectory'] = list(range(n_steps))
    
    return result


def ensure_all_methods_have_structure_metrics(method_results: Dict[str, List[Dict[str, Any]]],
                                            scms: List[Tuple[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Ensure all methods have structure learning metrics.
    
    Args:
        method_results: Dictionary of method_name -> list of results
        scms: List of (name, scm) tuples used in experiments
        
    Returns:
        Enhanced method results with structure metrics for all methods
    """
    enhanced_results = {}
    
    for method_name, results in method_results.items():
        enhanced_method_results = []
        
        for i, result in enumerate(results):
            # Get corresponding SCM
            scm_idx = i % len(scms)
            scm_name, scm = scms[scm_idx]
            
            # Add structure metrics if missing
            enhanced_result = add_structure_metrics_to_result(result.copy(), scm)
            enhanced_method_results.append(enhanced_result)
        
        enhanced_results[method_name] = enhanced_method_results
        
        logger.info(f"Enhanced {method_name} with structure metrics")
    
    return enhanced_results