#!/usr/bin/env python3
"""
BC Loss Debugging Utilities

This module provides comprehensive logging and debugging utilities for
tracking down the cause of astronomical loss values in BC training.

Key Features:
1. Detailed logging of probability distributions
2. Validation of probability constraints
3. Numerical stability checks
4. Loss computation breakdown

Design Principles (Rich Hickey Approved):
- Pure functions for all validations
- No side effects in debug utilities
- Clear separation of validation logic
- Composable debugging tools
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import jax.numpy as jnp
import numpy as onp

logger = logging.getLogger(__name__)


def validate_probability_distribution(
    probs: jnp.ndarray,
    name: str,
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Validate a probability distribution and return diagnostic info.
    
    Args:
        probs: Probability array to validate
        name: Name for logging
        tolerance: Tolerance for sum-to-1 check
        
    Returns:
        Dictionary with validation results and diagnostics
    """
    diagnostics = {
        'name': name,
        'shape': probs.shape,
        'dtype': str(probs.dtype),
        'min': float(jnp.min(probs)),
        'max': float(jnp.max(probs)),
        'sum': float(jnp.sum(probs)),
        'has_nan': bool(jnp.any(jnp.isnan(probs))),
        'has_inf': bool(jnp.any(jnp.isinf(probs))),
        'all_positive': bool(jnp.all(probs >= 0)),
        'all_in_range': bool(jnp.all((probs >= 0) & (probs <= 1))),
        'sums_to_one': bool(jnp.abs(jnp.sum(probs) - 1.0) < tolerance)
    }
    
    # Add warnings for issues
    issues = []
    if diagnostics['has_nan']:
        issues.append("Contains NaN values")
    if diagnostics['has_inf']:
        issues.append("Contains Inf values")
    if not diagnostics['all_positive']:
        issues.append(f"Contains negative values (min: {diagnostics['min']})")
    if not diagnostics['all_in_range']:
        issues.append(f"Values outside [0,1] range (min: {diagnostics['min']}, max: {diagnostics['max']})")
    if not diagnostics['sums_to_one']:
        issues.append(f"Does not sum to 1.0 (sum: {diagnostics['sum']})")
    
    diagnostics['issues'] = issues
    diagnostics['is_valid'] = len(issues) == 0
    
    return diagnostics


def debug_kl_divergence_computation(
    predicted_probs: jnp.ndarray,
    target_probs: jnp.ndarray,
    epsilon: float = 1e-8
) -> Dict[str, Any]:
    """
    Debug KL divergence computation with detailed logging.
    
    Args:
        predicted_probs: Model predictions
        target_probs: Target distribution
        epsilon: Small value for numerical stability
        
    Returns:
        Dictionary with KL computation details
    """
    # Validate inputs
    pred_diag = validate_probability_distribution(predicted_probs, "predicted")
    target_diag = validate_probability_distribution(target_probs, "target")
    
    # Clip for numerical stability
    pred_clipped = jnp.clip(predicted_probs, epsilon, 1.0 - epsilon)
    target_clipped = jnp.clip(target_probs, epsilon, 1.0 - epsilon)
    
    # Compute KL term by term
    log_ratio = jnp.log(target_clipped / pred_clipped)
    kl_terms = target_clipped * log_ratio
    kl_div = jnp.sum(kl_terms)
    
    # Check for numerical issues
    debug_info = {
        'predicted_validation': pred_diag,
        'target_validation': target_diag,
        'epsilon_used': epsilon,
        'log_ratio_min': float(jnp.min(log_ratio)),
        'log_ratio_max': float(jnp.max(log_ratio)),
        'log_ratio_has_nan': bool(jnp.any(jnp.isnan(log_ratio))),
        'log_ratio_has_inf': bool(jnp.any(jnp.isinf(log_ratio))),
        'kl_terms_min': float(jnp.min(kl_terms)),
        'kl_terms_max': float(jnp.max(kl_terms)),
        'kl_divergence': float(kl_div),
        'kl_is_nan': bool(jnp.isnan(kl_div)),
        'kl_is_inf': bool(jnp.isinf(kl_div))
    }
    
    # Add detailed breakdown if KL is extreme
    if jnp.abs(kl_div) > 100:
        debug_info['extreme_kl_warning'] = True
        debug_info['top_5_kl_terms'] = []
        
        # Find indices of largest KL contributions
        sorted_indices = jnp.argsort(jnp.abs(kl_terms))[-5:]
        for idx in sorted_indices:
            debug_info['top_5_kl_terms'].append({
                'index': int(idx),
                'target_prob': float(target_clipped[idx]),
                'pred_prob': float(pred_clipped[idx]),
                'log_ratio': float(log_ratio[idx]),
                'kl_contribution': float(kl_terms[idx])
            })
    
    return debug_info


def debug_parent_set_conversion(
    parent_sets: List[frozenset],
    probs: jnp.ndarray,
    num_variables: int,
    target_idx: int,
    variable_order: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Debug the conversion from discrete parent sets to continuous probabilities.
    
    Args:
        parent_sets: List of parent sets
        probs: Probabilities for each parent set
        num_variables: Total number of variables
        target_idx: Index of target variable
        
    Returns:
        Dictionary with conversion diagnostics
    """
    # Validate input probabilities
    input_validation = validate_probability_distribution(probs, "input_parent_set_probs")
    
    # Track conversion process
    var_probs = jnp.zeros(num_variables)
    conversion_steps = []
    
    for i, parent_set in enumerate(parent_sets):
        set_prob = probs[i]
        parent_indices = []
        
        for parent in parent_set:
            # Handle both integer indices and string variable names
            if isinstance(parent, int):
                parent_idx = parent
            elif isinstance(parent, str) and variable_order is not None:
                try:
                    parent_idx = variable_order.index(parent)
                except ValueError:
                    continue
            else:
                continue
                
            if parent_idx is not None and 0 <= parent_idx < num_variables:
                parent_indices.append(parent_idx)
                var_probs = var_probs.at[parent_idx].add(set_prob)
        
        conversion_steps.append({
            'set_index': i,
            'parent_set_size': len(parent_set),
            'parent_indices': parent_indices,
            'set_probability': float(set_prob),
            'parents_as_list': list(parent_set)
        })
    
    # Set target to 0
    var_probs = var_probs.at[target_idx].set(0.0)
    
    # Normalize
    total = jnp.sum(var_probs)
    if total > 0:
        var_probs_normalized = var_probs / total
    else:
        # Uniform over non-target
        var_probs_normalized = jnp.ones(num_variables) / (num_variables - 1)
        var_probs_normalized = var_probs_normalized.at[target_idx].set(0.0)
    
    # Validate output
    output_validation = validate_probability_distribution(var_probs_normalized, "continuous_probs")
    
    debug_info = {
        'num_parent_sets': len(parent_sets),
        'num_variables': num_variables,
        'target_idx': target_idx,
        'input_validation': input_validation,
        'output_validation': output_validation,
        'pre_normalization_sum': float(total),
        'conversion_steps': conversion_steps[:5],  # First 5 for brevity
        'non_zero_parent_probs': int(jnp.sum(var_probs_normalized > 0)),
        'max_parent_prob': float(jnp.max(var_probs_normalized)),
        'conversion_preserves_structure': total > 0
    }
    
    return debug_info


def log_training_step_debug(
    batch_idx: int,
    loss_value: float,
    grad_norm: float,
    predicted_probs: Optional[jnp.ndarray] = None,
    target_probs: Optional[jnp.ndarray] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log comprehensive debug information for a training step.
    
    Args:
        batch_idx: Batch index
        loss_value: Computed loss value
        grad_norm: Gradient norm
        predicted_probs: Model predictions (optional)
        target_probs: Target probabilities (optional)
        additional_info: Any additional debug information
    """
    debug_msg = f"\n{'='*60}\n"
    debug_msg += f"Training Step Debug - Batch {batch_idx}\n"
    debug_msg += f"{'='*60}\n"
    debug_msg += f"Loss Value: {loss_value:.6f}\n"
    debug_msg += f"Gradient Norm: {grad_norm:.6f}\n"
    
    # Check for extreme values
    if abs(loss_value) > 1000:
        debug_msg += "⚠️  WARNING: Extremely high loss value!\n"
    
    if grad_norm > 100:
        debug_msg += "⚠️  WARNING: Extremely high gradient norm!\n"
    
    if predicted_probs is not None and target_probs is not None:
        kl_debug = debug_kl_divergence_computation(predicted_probs, target_probs)
        debug_msg += f"\nKL Divergence Debug:\n"
        debug_msg += f"  KL Value: {kl_debug['kl_divergence']:.6f}\n"
        debug_msg += f"  Predicted Valid: {kl_debug['predicted_validation']['is_valid']}\n"
        debug_msg += f"  Target Valid: {kl_debug['target_validation']['is_valid']}\n"
        
        if not kl_debug['predicted_validation']['is_valid']:
            debug_msg += f"  Predicted Issues: {kl_debug['predicted_validation']['issues']}\n"
        if not kl_debug['target_validation']['is_valid']:
            debug_msg += f"  Target Issues: {kl_debug['target_validation']['issues']}\n"
    
    if additional_info:
        debug_msg += f"\nAdditional Info:\n"
        for key, value in additional_info.items():
            debug_msg += f"  {key}: {value}\n"
    
    debug_msg += f"{'='*60}\n"
    
    logger.debug(debug_msg)
    
    # Log error level for extreme values
    if abs(loss_value) > 10000:
        logger.error(f"CRITICAL: Loss value {loss_value} indicates training instability!")


def create_debugging_kl_loss(base_kl_fn):
    """
    Wrap a KL loss function with debugging capabilities.
    
    Args:
        base_kl_fn: Original KL loss function
        
    Returns:
        Wrapped function with debugging
    """
    def debugging_kl_loss(predicted_probs, target_probs, parent_sets=None):
        # Debug before computation
        debug_info = debug_kl_divergence_computation(predicted_probs, target_probs)
        
        # Compute loss
        loss = base_kl_fn(predicted_probs, target_probs, parent_sets)
        
        # Log if loss is extreme
        if jnp.abs(loss) > 100:
            logger.warning(f"Extreme KL loss detected: {float(loss)}")
            logger.warning(f"Debug info: {debug_info}")
        
        return loss
    
    return debugging_kl_loss