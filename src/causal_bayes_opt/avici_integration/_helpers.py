"""
Internal helper functions for AVICI data format bridge.

This module contains private implementation details for data conversion.
These functions should not be imported directly by users.
"""

# Standard library imports
from typing import List

# Third-party imports
import jax.numpy as jnp
import pyrsistent as pyr

# Type aliases
SampleList = List[pyr.PMap]
VariableOrder = List[str]

# Constants
EPSILON = 1e-8  # Small epsilon for numerical stability


def _extract_values_matrix(
    samples: SampleList,
    variable_order: VariableOrder
) -> jnp.ndarray:
    """
    Extract variable values from samples into a matrix.
    
    Args:
        samples: List of Sample objects
        variable_order: Ordered list of variable names
        
    Returns:
        JAX array of shape [N, d] with variable values
    """
    n_samples = len(samples)
    n_vars = len(variable_order)
    
    # Initialize values matrix
    values_matrix = jnp.zeros((n_samples, n_vars), dtype=jnp.float32)
    
    # Extract values for each sample
    for i, sample in enumerate(samples):
        sample_values = sample['values']
        
        # Extract values in variable order
        for j, var_name in enumerate(variable_order):
            values_matrix = values_matrix.at[i, j].set(float(sample_values[var_name]))
    
    return values_matrix


def _extract_intervention_indicators(
    samples: SampleList,
    variable_order: VariableOrder
) -> jnp.ndarray:
    """
    Extract intervention indicators from samples.
    
    Args:
        samples: List of Sample objects
        variable_order: Ordered list of variable names
        
    Returns:
        JAX array of shape [N, d] with intervention indicators
        (1 if variable was intervened upon, 0 otherwise)
    """
    n_samples = len(samples)
    n_vars = len(variable_order)
    
    # Initialize intervention indicators matrix
    intervention_matrix = jnp.zeros((n_samples, n_vars), dtype=jnp.float32)
    
    # Extract intervention indicators for each sample
    for i, sample in enumerate(samples):
        if sample['intervention_type'] is not None:
            intervention_targets = sample['intervention_targets']
            
            # Set indicators for intervened variables
            for j, var_name in enumerate(variable_order):
                if var_name in intervention_targets:
                    intervention_matrix = intervention_matrix.at[i, j].set(1.0)
    
    return intervention_matrix


def _create_target_indicators(
    target_variable: str,
    variable_order: VariableOrder,
    n_samples: int
) -> jnp.ndarray:
    """
    Create target indicators matrix.
    
    Args:
        target_variable: Name of target variable
        variable_order: Ordered list of variable names
        n_samples: Number of samples
        
    Returns:
        JAX array of shape [N, d] with target indicators
        (1 for target variable, 0 for all others)
    """
    n_vars = len(variable_order)
    target_indicators = jnp.zeros((n_samples, n_vars), dtype=jnp.float32)
    
    # Find target variable index
    target_idx = variable_order.index(target_variable)
    
    # Set target indicators (1 for target variable, 0 for others)
    target_indicators = target_indicators.at[:, target_idx].set(1.0)
    
    return target_indicators


def _standardize_values(
    values: jnp.ndarray,
    standardization_type: str = "default"
) -> jnp.ndarray:
    """
    Standardize variable values following AVICI's approach.
    
    Args:
        values: JAX array of shape [N, d] with variable values
        standardization_type: Type of standardization ("default" or "count")
        
    Returns:
        Standardized values array of same shape
    """
    if standardization_type == "default":
        # Z-standardization: (x - mean) / std
        mean = jnp.mean(values, axis=0, keepdims=True)
        std = jnp.std(values, axis=0, keepdims=True)
        
        # Avoid division by zero
        std = jnp.where(std == 0.0, 1.0, std)
        
        standardized = (values - mean) / std
        
    elif standardization_type == "count":
        # Count-based standardization (simplified version of AVICI's approach)
        # For now, just apply log transformation and then z-standardization
        
        # Apply log transformation (add small epsilon to avoid log(0))
        log_values = jnp.log(values + EPSILON)
        
        # Then z-standardize
        mean = jnp.mean(log_values, axis=0, keepdims=True)
        std = jnp.std(log_values, axis=0, keepdims=True)
        std = jnp.where(std == 0.0, 1.0, std)
        
        standardized = (log_values - mean) / std
        
    else:
        raise ValueError(f"Unknown standardization type: {standardization_type}")
    
    return standardized