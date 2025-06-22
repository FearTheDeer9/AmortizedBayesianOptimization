"""
JAX-compatible utilities for acquisition policy.

This module provides JAX-compilable helper functions for the policy network,
ensuring full compatibility with JAX's JIT compilation requirements.
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional


# Mechanism type encoding constants
MECHANISM_TYPE_ENCODING = {
    'linear': 0,
    'polynomial': 1,
    'gaussian': 2,
    'neural': 3,
    'unknown': 4
}

# Mechanism type scaling factors as JAX array
MECHANISM_SCALING_FACTORS = jnp.array([
    1.0,   # linear: Standard scaling
    0.5,   # polynomial: Smaller interventions for nonlinear effects
    1.2,   # gaussian: Slightly larger for Gaussian processes
    0.8,   # neural: Moderate scaling for neural network mechanisms
    1.0    # unknown: Default scaling
])


def create_variable_lookup_table(variable_order: List[str]) -> Dict[str, int]:
    """
    Create a lookup table for variable name to index mapping.
    
    Args:
        variable_order: Ordered list of variable names
        
    Returns:
        Dictionary mapping variable names to indices
    """
    return {var: idx for idx, var in enumerate(variable_order)}


def create_target_mask_jax(
    variable_order: List[str], 
    target_variable: str
) -> jnp.ndarray:
    """
    Create JAX-compatible target mask using tensor operations.
    
    Args:
        variable_order: Ordered list of variable names
        target_variable: Name of target variable to mask
        
    Returns:
        JAX array with 0 for non-target variables, -inf for target
    """
    # Find target index
    try:
        target_idx = variable_order.index(target_variable)
    except ValueError:
        # Target not found, return zeros (no masking)
        return jnp.zeros(len(variable_order))
    
    # Use tensor-based mask creation
    from .tensor_features import create_target_mask_tensor_jit
    return create_target_mask_tensor_jit(len(variable_order), target_idx)


def extract_mechanism_features_jax(
    variable_order: List[str],
    mechanism_insights: Dict,
    mechanism_confidence: Dict[str, float]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Extract mechanism features in JAX-compatible format.
    
    DEPRECATED: This function uses Python loops and dictionary operations.
    Use create_mechanism_features_tensor() from tensor_features.py instead
    for true JAX performance.
    
    Args:
        variable_order: Ordered list of variable names
        mechanism_insights: Dictionary with mechanism predictions
        mechanism_confidence: Dictionary with confidence scores
        
    Returns:
        Tuple of (predicted_coefficients, uncertainties, type_indices)
    """
    import warnings
    warnings.warn(
        "extract_mechanism_features_jax() is deprecated due to Python loops. "
        "Use create_mechanism_features_tensor() from tensor_features.py instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    n_vars = len(variable_order)
    
    # Initialize arrays
    predicted_coefficients = jnp.ones(n_vars)
    uncertainties = jnp.ones(n_vars) * 0.5
    type_indices = jnp.ones(n_vars, dtype=jnp.int32) * MECHANISM_TYPE_ENCODING['unknown']
    
    # Extract features for each variable
    # WARNING: This loop is a performance bottleneck and defeats JAX compilation
    predicted_effects = mechanism_insights.get('predicted_effects', {})
    mechanism_types = mechanism_insights.get('mechanism_types', {})
    
    for idx, var_name in enumerate(variable_order):
        # Predicted coefficient magnitude
        if var_name in predicted_effects:
            coeff = predicted_effects[var_name]
            predicted_coefficients = predicted_coefficients.at[idx].set(
                jnp.abs(coeff) if coeff != 0 else 1.0
            )
        
        # Uncertainty (1 - confidence)
        if var_name in mechanism_confidence:
            confidence = mechanism_confidence[var_name]
            uncertainties = uncertainties.at[idx].set(1.0 - confidence)
        
        # Mechanism type encoding
        if var_name in mechanism_types:
            mech_type = mechanism_types[var_name]
            type_idx = MECHANISM_TYPE_ENCODING.get(mech_type, MECHANISM_TYPE_ENCODING['unknown'])
            type_indices = type_indices.at[idx].set(type_idx)
    
    return predicted_coefficients, uncertainties, type_indices


def get_mechanism_scaling_factors_jax(type_indices: jnp.ndarray) -> jnp.ndarray:
    """
    Get mechanism scaling factors using JAX-compatible indexing.
    
    Args:
        type_indices: Integer indices for mechanism types
        
    Returns:
        Scaling factors for each variable
    """
    return MECHANISM_SCALING_FACTORS[type_indices]


def create_mechanism_features_vectorized(
    variable_order: List[str],
    mechanism_insights: Dict,
    mechanism_confidence: Dict[str, float]
) -> jnp.ndarray:
    """
    Create mechanism features using the new tensor-based system.
    
    This function now delegates to the proper tensor-based implementation
    that eliminates dictionary lookups and Python loops.
    
    Args:
        variable_order: Ordered list of variable names
        mechanism_insights: Dictionary with mechanism predictions
        mechanism_confidence: Dictionary with confidence scores
        
    Returns:
        JAX array of shape [n_vars, 3] with mechanism features
    """
    # Delegate to the new tensor-based system
    from .tensor_features import create_mechanism_features_tensor, compute_mechanism_features_tensor
    
    tensor_features = create_mechanism_features_tensor(
        variable_order, mechanism_insights, mechanism_confidence
    )
    
    return compute_mechanism_features_tensor(tensor_features)


# JAX-compiled versions of key functions
@jax.jit
def apply_target_mask_jit(logits: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """JAX-compiled function to apply target mask to logits."""
    return logits + mask


@jax.jit
def compute_mechanism_features_jit(
    coefficients: jnp.ndarray,
    uncertainties: jnp.ndarray,
    type_indices: jnp.ndarray
) -> jnp.ndarray:
    """JAX-compiled function to compute mechanism features."""
    scaling_factors = MECHANISM_SCALING_FACTORS[type_indices]
    return jnp.stack([coefficients, uncertainties, scaling_factors], axis=1)


def validate_jax_compatibility(func):
    """
    Decorator to validate JAX compatibility of a function.
    
    Raises:
        RuntimeError: If function is not JAX-compilable
    """
    try:
        # Attempt to JIT compile the function
        jitted_func = jax.jit(func)
        return jitted_func
    except Exception as e:
        raise RuntimeError(f"Function {func.__name__} is not JAX-compatible: {e}")


def prepare_samples_for_history_jax(
    samples: List,
    variable_order: List[str],
    target_variable: str,
    max_history_size: int = 100
) -> jnp.ndarray:
    """
    Convert samples to history format using JAX-compatible operations.
    
    Note: This function still uses Python loops for sample processing,
    but it's called outside the JAX-compiled region. The output is
    a fixed-size tensor suitable for JAX compilation.
    
    Args:
        samples: List of sample objects
        variable_order: Ordered list of variable names
        target_variable: Name of target variable
        max_history_size: Maximum number of samples to include
        
    Returns:
        JAX array of shape [max_history_size, n_vars, 3]
    """
    from ..data_structures.sample import get_values, get_intervention_targets
    
    n_vars = len(variable_order)
    n_samples = len(samples)
    
    # Determine how many samples to use
    if n_samples > max_history_size:
        samples_to_use = samples[-max_history_size:]
        n_used = max_history_size
    else:
        samples_to_use = samples
        n_used = n_samples
    
    # Initialize arrays
    values_array = jnp.zeros((max_history_size, n_vars))
    intervention_array = jnp.zeros((max_history_size, n_vars))
    target_array = jnp.zeros((max_history_size, n_vars))
    
    # Create variable lookup
    var_lookup = create_variable_lookup_table(variable_order)
    
    # Pre-compute target mask
    target_idx = var_lookup.get(target_variable, -1)
    if target_idx >= 0:
        target_array = target_array.at[:, target_idx].set(1.0)
    
    # Process samples (this part still uses loops but is outside JAX compilation)
    for sample_idx, sample in enumerate(samples_to_use):
        sample_values = get_values(sample)
        intervention_targets = get_intervention_targets(sample)
        
        for var_name, value in sample_values.items():
            if var_name in var_lookup:
                var_idx = var_lookup[var_name]
                values_array = values_array.at[sample_idx, var_idx].set(float(value))
                
                if var_name in intervention_targets:
                    intervention_array = intervention_array.at[sample_idx, var_idx].set(1.0)
    
    # Standardize values
    if n_used > 1:
        actual_values = values_array[:n_used, :]
        values_mean = jnp.mean(actual_values, axis=0, keepdims=True)
        values_std = jnp.std(actual_values, axis=0, keepdims=True) + 1e-8
    else:
        values_mean = jnp.zeros((1, n_vars))
        values_std = jnp.ones((1, n_vars))
    
    values_standardized = (values_array - values_mean) / values_std
    
    # Stack into final format
    return jnp.stack([values_standardized, intervention_array, target_array], axis=2)