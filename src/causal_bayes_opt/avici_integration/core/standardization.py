"""Pure standardization functions."""

import jax.numpy as jnp
from typing import Dict, Literal, Optional

# Type aliases
StandardizationType = Literal["default", "count", "none"]
StandardizationParams = Dict[str, jnp.ndarray]

# Constants
EPSILON = 1e-8  # Small epsilon for numerical stability


def compute_standardization_params(
    values: jnp.ndarray,
    standardization_type: StandardizationType = "default"
) -> StandardizationParams:
    """
    Pure function: Compute standardization parameters.
    
    Args:
        values: JAX array of shape [N, d] with variable values
        standardization_type: Type of standardization
        
    Returns:
        Dictionary containing standardization parameters
    """
    if standardization_type == "default":
        mean = jnp.mean(values, axis=0, keepdims=True)
        std = jnp.std(values, axis=0, keepdims=True)
        # Avoid division by zero
        std = jnp.where(std == 0.0, 1.0, std)
        
        return {
            "type": "default",
            "mean": mean,
            "std": std
        }
        
    elif standardization_type == "count":
        # Count-based standardization with log transformation
        log_values = jnp.log(values + EPSILON)
        mean = jnp.mean(log_values, axis=0, keepdims=True)
        std = jnp.std(log_values, axis=0, keepdims=True)
        std = jnp.where(std == 0.0, 1.0, std)
        
        return {
            "type": "count",
            "mean": mean,
            "std": std,
            "epsilon": EPSILON
        }
        
    elif standardization_type == "none":
        return {"type": "none"}
        
    else:
        raise ValueError(f"Unknown standardization type: {standardization_type}")


def apply_standardization(
    values: jnp.ndarray,
    params: StandardizationParams
) -> jnp.ndarray:
    """
    Pure function: Apply standardization with parameters.
    
    Args:
        values: JAX array of shape [N, d] with variable values
        params: Standardization parameters from compute_standardization_params
        
    Returns:
        Standardized values array of same shape
    """
    standardization_type = params["type"]
    
    if standardization_type == "default":
        mean = params["mean"]
        std = params["std"]
        return (values - mean) / std
        
    elif standardization_type == "count":
        epsilon = params.get("epsilon", EPSILON)
        mean = params["mean"]
        std = params["std"]
        
        # Apply log transformation then z-standardization
        log_values = jnp.log(values + epsilon)
        return (log_values - mean) / std
        
    elif standardization_type == "none":
        return values
        
    else:
        raise ValueError(f"Unknown standardization type: {standardization_type}")


def reverse_standardization(
    standardized_values: jnp.ndarray,
    params: StandardizationParams
) -> jnp.ndarray:
    """
    Pure function: Reverse standardization to get original scale.
    
    Args:
        standardized_values: Standardized values array
        params: Standardization parameters used for forward transformation
        
    Returns:
        Values in original scale
        
    Note:
        This is primarily for validation and debugging purposes.
    """
    standardization_type = params["type"]
    
    if standardization_type == "default":
        mean = params["mean"]
        std = params["std"]
        return standardized_values * std + mean
        
    elif standardization_type == "count":
        epsilon = params.get("epsilon", EPSILON)
        mean = params["mean"]
        std = params["std"]
        
        # Reverse z-standardization then exp transformation
        log_values = standardized_values * std + mean
        return jnp.exp(log_values) - epsilon
        
    elif standardization_type == "none":
        return standardized_values
        
    else:
        raise ValueError(f"Unknown standardization type: {standardization_type}")
