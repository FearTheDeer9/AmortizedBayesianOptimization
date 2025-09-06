"""
Four-channel tensor converter for policies that use 4 channels.

This creates tensors with format [Values, Target, Intervention, Probs] 
without the 5th recency channel that was deemed unnecessary.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional, Callable
import jax.numpy as jnp

from ..data_structures.buffer import ExperienceBuffer
from ..data_structures.sample import get_values, get_intervention_targets
from ..utils.variable_mapping import VariableMapper

logger = logging.getLogger(__name__)


def buffer_to_four_channel_tensor(
    buffer: ExperienceBuffer,
    target_variable: str,
    surrogate_fn: Optional[Callable] = None,
    max_history_size: Optional[int] = 100,
    standardize: bool = True
) -> Tuple[jnp.ndarray, VariableMapper, Dict[str, Any]]:
    """
    Convert experience buffer to 4-channel tensor format.
    
    Creates tensor with channels:
    0. Values: Variable values (standardized per-variable if enabled)
    1. Target: Binary indicator for target variable
    2. Intervention: Binary indicator for intervened variables
    3. Probs: Parent probabilities (uniform 0.5 if no surrogate)
    
    Args:
        buffer: Experience buffer with samples
        target_variable: Name of target variable
        surrogate_fn: Optional function for parent probability predictions
        max_history_size: Maximum number of historical samples (None = use all)
        standardize: Whether to standardize values per-variable
        
    Returns:
        Tuple of (tensor [T, n_vars, 4], mapper, diagnostics)
    """
    # Get all samples from buffer
    all_samples = buffer.get_all_samples()
    if not all_samples:
        raise ValueError("Buffer is empty")
    
    # Get variable order and create mapper
    variable_order = sorted(buffer.get_variable_coverage())
    if target_variable not in variable_order:
        raise ValueError(f"Target '{target_variable}' not in buffer variables: {variable_order}")
    
    mapper = VariableMapper(variable_order, target_variable)
    n_vars = len(variable_order)
    target_idx = mapper.get_index(target_variable)
    
    # Determine actual history size
    if max_history_size is None:
        # Use all available samples
        actual_size = len(all_samples)
        recent_samples = all_samples
        tensor_size = actual_size
    else:
        # Limit to max_history_size
        actual_size = min(len(all_samples), max_history_size)
        recent_samples = all_samples[-actual_size:]
        tensor_size = max_history_size
    
    # Get surrogate predictions for the entire buffer ONCE
    if surrogate_fn is not None:
        try:
            from .three_channel_converter import buffer_to_three_channel_tensor
            
            # Convert buffer to 3-channel format (exactly like end-of-episode)
            tensor_3ch, _ = buffer_to_three_channel_tensor(
                buffer, target_variable, 
                max_history_size=max_history_size,  # Pass through the same limit
                standardize=standardize
            )
            
            # Call surrogate with the 3-channel tensor
            surrogate_result = surrogate_fn(tensor_3ch, target_variable, variable_order)
            
            # Extract parent probabilities
            if isinstance(surrogate_result, dict) and 'parent_probs' in surrogate_result:
                surrogate_probs = jnp.array(surrogate_result['parent_probs'])
                # Ensure correct shape
                if surrogate_probs.shape[0] != n_vars:
                    logger.warning(f"Surrogate returned wrong shape: {surrogate_probs.shape}, expected ({n_vars},)")
                    surrogate_probs = jnp.full(n_vars, 0.5)
                else:
                    pass  # Got valid surrogate predictions
            else:
                logger.warning(f"Surrogate returned unexpected format: {type(surrogate_result)}")
                surrogate_probs = jnp.full(n_vars, 0.5)
        except Exception as e:
            logger.warning(f"Surrogate prediction failed: {e}")
            surrogate_probs = jnp.full(n_vars, 0.5)
    else:
        # Uniform probabilities (no surrogate information)
        surrogate_probs = jnp.full(n_vars, 0.5)
    
    # Initialize 4-channel tensor
    tensor = jnp.zeros((tensor_size, n_vars, 4))
    
    # Fill tensor with recent samples
    for t, sample in enumerate(recent_samples):
        if max_history_size is None:
            tensor_idx = t  # Direct indexing when using all data
        else:
            tensor_idx = max_history_size - actual_size + t  # Place at end when limited
        
        # Channel 0: Values
        values = _extract_values_vector(sample, variable_order)
        
        # Channel 1: Target indicator
        target_mask = jnp.array([1.0 if var == target_variable else 0.0 for var in variable_order])
        
        # Channel 2: Intervention indicator
        intervention_targets = get_intervention_targets(sample)
        intervention_mask = jnp.array([1.0 if var in intervention_targets else 0.0 for var in variable_order])
        
        # Channel 3: Parent probabilities (already computed above for entire buffer)
        parent_probs = surrogate_probs  # Use the same probs for all timesteps
        
        # Set tensor values
        tensor = tensor.at[tensor_idx, :, 0].set(values)
        tensor = tensor.at[tensor_idx, :, 1].set(target_mask)
        tensor = tensor.at[tensor_idx, :, 2].set(intervention_mask)
        tensor = tensor.at[tensor_idx, :, 3].set(parent_probs)
    
    # Standardize values if requested (per-variable)
    if standardize and actual_size > 1:
        tensor = _standardize_values_four_channel(tensor, actual_size)
    
    # Diagnostics
    diagnostics = {
        'n_variables': n_vars,
        'n_samples': len(all_samples),
        'actual_history_size': actual_size,
        'target_variable': target_variable,
        'target_idx': target_idx,
        'has_surrogate': surrogate_fn is not None
    }
    
    return tensor, mapper, diagnostics


def _extract_values_vector(sample: Any, variable_order: List[str]) -> jnp.ndarray:
    """Extract values for all variables in specified order."""
    sample_values = get_values(sample)
    values = []
    for var in variable_order:
        values.append(float(sample_values.get(var, 0.0)))
    return jnp.array(values)


def _standardize_values_four_channel(tensor: jnp.ndarray, actual_size: int) -> jnp.ndarray:
    """Standardize values channel using per-variable statistics (4-channel version)."""
    # Get the actual data region (excluding padding)
    start_idx = tensor.shape[0] - actual_size
    n_vars = tensor.shape[1]
    
    # Create mask for actual data region
    mask = jnp.zeros(tensor.shape[0])
    mask = mask.at[start_idx:].set(1.0)
    
    # Standardize each variable independently
    updated_tensor = tensor
    
    for var_idx in range(n_vars):
        # Get this variable's actual data across time
        var_actual_data = tensor[start_idx:, var_idx, 0]  # Only actual samples
        
        if len(var_actual_data) > 1:  # Need multiple samples for std
            # Per-variable statistics
            var_mean = jnp.mean(var_actual_data)
            var_std = jnp.std(var_actual_data) + 1e-8
            
            # Standardize this variable's values (all timesteps)
            standardized_var = (tensor[:, var_idx, 0] - var_mean) / var_std
            
            # Apply only to actual data region for this variable
            new_var_values = (tensor[:, var_idx, 0] * (1 - mask) + 
                             standardized_var * mask)
            
            updated_tensor = updated_tensor.at[:, var_idx, 0].set(new_var_values)
    
    return updated_tensor