"""
Data utilities for target-aware AVICI model.

This module provides data processing functions that handle [N, d, 3] input format
instead of AVICI's original [N, d, 2] format.
"""

# Third-party imports
import jax.numpy as jnp
import jax.random as random
import jax.lax as lax

# Local imports
from ._helpers import _standardize_values


def target_aware_standardize_default(x):
    """
    Standardize target-aware data [N, d, 3] format.
    
    Only standardizes the values channel (channel 0), leaves 
    intervention and target channels unchanged.
    
    Args:
        x: Input data of shape [N, d, 3]
        
    Returns:
        Standardized data of same shape
    """
    # Extract values channel
    values = x[..., 0]
    
    # Use helper function for standardization
    standardized_values = _standardize_values(values, standardization_type="default")
    
    # Put back into tensor
    result = x.at[..., 0].set(standardized_values)
    
    return result


def target_aware_get_x(data):
    """
    Extract and standardize input data from training batch.
    
    Args:
        data: Training batch dictionary containing 'x' key
        
    Returns:
        Standardized input tensor [N, d, 3]
    """
    x = data['x']
    
    # Validate input format
    if x.shape[-1] != 3:
        raise ValueError(f"Expected [N, d, 3] format, got shape {x.shape}")
    
    # Standardize (this handles [N, d, 3] format correctly)
    x_standardized = target_aware_standardize_default(x)
    
    return x_standardized


def target_aware_get_train_x(key, data, p_obs_only=0.0):
    """
    Get training data with optional observational-only sampling.
    
    For Phase 1.3 validation, we'll use a simplified version that
    just returns the standardized data.
    
    Args:
        key: Random key (for compatibility with AVICI interface)
        data: Training batch dictionary
        p_obs_only: Probability of using only observational data
        
    Returns:
        Standardized input tensor [N, d, 3]
    """
    # For validation phase, we'll just return standardized data
    # In future versions, this could implement more sophisticated
    # data augmentation similar to AVICI's approach
    
    return target_aware_get_x(data)
