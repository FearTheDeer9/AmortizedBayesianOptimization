"""
Three-channel tensor conversion for ACBO.

This module provides direct conversion from experience buffers to the standard
3-channel tensor format used throughout ACBO, bypassing complex state abstractions.

The 3-channel format:
- Channel 0: past_node_values (standardized values from all samples)
- Channel 1: target_indicator (1.0 for target variable)
- Channel 2: intervened_on_indicator (1.0 if variable was intervened on)
"""

import jax
import jax.numpy as jnp
from typing import List, Tuple, Optional, Dict, Any
import logging

from ..data_structures.buffer import ExperienceBuffer
from ..data_structures.sample import get_values, get_intervention_targets
from ..utils.variable_mapping import VariableMapper

logger = logging.getLogger(__name__)


def buffer_to_three_channel_tensor(
    buffer: ExperienceBuffer,
    target_variable: str,
    max_history_size: Optional[int] = 100,
    standardize: bool = True
) -> Tuple[jnp.ndarray, VariableMapper]:
    """
    Convert experience buffer directly to 3-channel tensor format.
    
    This function creates the standard ACBO tensor representation without
    requiring complex state objects or configuration.
    
    Args:
        buffer: Experience buffer with samples
        target_variable: Name of target variable
        max_history_size: Maximum number of historical samples (None = use all)
        standardize: Whether to standardize values using global statistics
        
    Returns:
        Tuple of:
        - tensor: [T, n_vars, 3] tensor in standard ACBO format
        - mapper: VariableMapper instance for variable name/index conversions
        
    Raises:
        ValueError: If buffer is empty or target not in variables
    """
    logger.debug(f"[3-Channel Converter] Converting buffer with {buffer.size()} samples")
    logger.debug(f"[3-Channel Converter] Target variable: {target_variable}")
    
    # Get all samples from buffer
    all_samples = buffer.get_all_samples()
    if not all_samples:
        raise ValueError("Buffer is empty")
    
    # Get variable order from buffer coverage and create mapper
    variable_order = sorted(buffer.get_variable_coverage())
    if target_variable not in variable_order:
        raise ValueError(f"Target '{target_variable}' not in buffer variables: {variable_order}")
    
    # Create variable mapper
    mapper = VariableMapper(variable_order, target_variable)
    n_vars = len(variable_order)
    logger.debug(f"[3-Channel Converter] Created VariableMapper with variables: {mapper.variables}")
    logger.debug(f"[3-Channel Converter] Target index: {mapper.get_index(target_variable)}")
    
    # Determine samples to use and tensor size
    if max_history_size is None:
        # Use all available samples
        samples = all_samples
        actual_history_size = len(samples)
        tensor_size = actual_history_size
    else:
        # Limit to max history size (most recent samples)
        samples = all_samples[-max_history_size:] if len(all_samples) > max_history_size else all_samples
        actual_history_size = len(samples)
        tensor_size = max_history_size
    
    # Initialize tensor with zeros (padding for short histories when limited)
    tensor = jnp.zeros((tensor_size, n_vars, 3))
    
    # Fill tensor with sample data
    for t, sample in enumerate(samples):
        # Adjust index based on whether we're using all data or limited
        if max_history_size is None:
            tensor_idx = t  # Direct indexing when using all data
        else:
            tensor_idx = max_history_size - actual_history_size + t  # Place at end when limited
        
        # Channel 0: Variable values
        values = _extract_values_vector(sample, variable_order)
        
        # Channel 1: Target indicator
        target_mask = _create_target_mask(variable_order, target_variable)
        
        # Channel 2: Intervention indicator
        intervention_mask = _extract_intervention_mask(sample, variable_order)
        
        # Set tensor values
        tensor = tensor.at[tensor_idx, :, 0].set(values)
        tensor = tensor.at[tensor_idx, :, 1].set(target_mask)
        tensor = tensor.at[tensor_idx, :, 2].set(intervention_mask)
    
    # Standardize values if requested
    if standardize and actual_history_size > 1:
        tensor = _standardize_values_channel(tensor, actual_history_size)
    
    logger.debug(f"[3-Channel Converter] Final tensor shape: {tensor.shape}")
    return tensor, mapper


def _extract_values_vector(sample: Any, variable_order: List[str]) -> jnp.ndarray:
    """Extract ordered values from sample."""
    values = get_values(sample)
    return jnp.array([float(values.get(var, 0.0)) for var in variable_order])


def _create_target_mask(variable_order: List[str], target_variable: str) -> jnp.ndarray:
    """Create binary mask for target variable."""
    return jnp.array([1.0 if var == target_variable else 0.0 for var in variable_order])


def _extract_intervention_mask(sample: Any, variable_order: List[str]) -> jnp.ndarray:
    """Extract intervention indicators from sample."""
    intervention_targets = get_intervention_targets(sample)
    return jnp.array([1.0 if var in intervention_targets else 0.0 for var in variable_order])


def _standardize_values_channel(tensor: jnp.ndarray, actual_history_size: int) -> jnp.ndarray:
    """
    Standardize the values channel using global statistics.
    
    Uses global mean/std across all variables and time to preserve natural
    scale differences between variables from the SCM structure.
    """
    # Get the actual data region (excluding padding)
    start_idx = tensor.shape[0] - actual_history_size
    actual_data = tensor[start_idx:, :, 0]  # Values channel
    
    # Compute global statistics
    global_mean = jnp.mean(actual_data)
    global_std = jnp.std(actual_data) + 1e-8  # Avoid division by zero
    
    # Standardize only the actual data region
    standardized_values = (tensor[:, :, 0] - global_mean) / global_std
    
    # Only update the non-padded region
    mask = jnp.zeros(tensor.shape[0])
    mask = mask.at[start_idx:].set(1.0)
    
    # Apply standardization only to actual data
    new_values = tensor[:, :, 0] * (1 - mask[:, None]) + standardized_values * mask[:, None]
    
    # Update tensor
    return tensor.at[:, :, 0].set(new_values)


def create_training_batch_from_buffer(
    buffer: ExperienceBuffer,
    target_variable: str,
    batch_size: int = 32,
    max_history_size: int = 100,
    standardize: bool = True
) -> Dict[str, Any]:
    """
    Create a training batch directly from buffer for surrogate models.
    
    This creates the format expected by ParentSetPredictionModel training.
    
    Args:
        buffer: Experience buffer
        target_variable: Target variable name
        batch_size: Number of samples in batch
        max_history_size: Max history length per sample
        standardize: Whether to standardize values
        
    Returns:
        Dictionary with:
        - 'x': [batch_size, n_vars, 3] tensor
        - 'variable_order': List of variable names
        - 'target_variable': Target variable name
    """
    # Get base tensor
    full_tensor, variable_order = buffer_to_three_channel_tensor(
        buffer, target_variable, max_history_size, standardize
    )
    
    # For training, we need multiple views of the data
    # Create batch by sampling different time windows
    all_samples = buffer.get_all_samples()
    n_samples = len(all_samples)
    
    if n_samples <= batch_size:
        # If we have fewer samples than batch size, tile the data
        batch = jnp.tile(full_tensor[-1:, :, :], (batch_size, 1, 1))
    else:
        # Sample random windows from the history
        # For now, just take the most recent state repeated
        # TODO: Implement proper temporal sampling
        batch = jnp.tile(full_tensor[-1:, :, :], (batch_size, 1, 1))
    
    return {
        'x': batch,
        'variable_order': variable_order,
        'target_variable': target_variable
    }


def samples_to_three_channel_tensor(
    samples: List[Any],
    variable_order: List[str],
    target_variable: str,
    standardize: bool = True
) -> jnp.ndarray:
    """
    Convert list of samples directly to 3-channel tensor.
    
    Useful for evaluation when you have samples but no buffer.
    
    Args:
        samples: List of samples (observations or interventions)
        variable_order: Order of variables in tensor
        target_variable: Target variable name
        standardize: Whether to standardize values
        
    Returns:
        Tensor of shape [n_samples, n_vars, 3]
    """
    n_samples = len(samples)
    n_vars = len(variable_order)
    
    # Initialize tensor
    tensor = jnp.zeros((n_samples, n_vars, 3))
    
    # Fill with sample data
    for t, sample in enumerate(samples):
        # Channel 0: Values
        values = _extract_values_vector(sample, variable_order)
        
        # Channel 1: Target mask (same for all samples)
        target_mask = _create_target_mask(variable_order, target_variable)
        
        # Channel 2: Intervention mask
        intervention_mask = _extract_intervention_mask(sample, variable_order)
        
        tensor = tensor.at[t, :, 0].set(values)
        tensor = tensor.at[t, :, 1].set(target_mask)
        tensor = tensor.at[t, :, 2].set(intervention_mask)
    
    # Standardize if requested
    if standardize and n_samples > 1:
        values = tensor[:, :, 0]
        global_mean = jnp.mean(values)
        global_std = jnp.std(values) + 1e-8
        standardized = (values - global_mean) / global_std
        tensor = tensor.at[:, :, 0].set(standardized)
    
    return tensor


def validate_three_channel_tensor(tensor: jnp.ndarray, variable_order: List[str]) -> bool:
    """
    Validate that a tensor follows the 3-channel ACBO format.
    
    Args:
        tensor: Tensor to validate
        variable_order: Expected variable order
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if len(tensor.shape) != 3:
        raise ValueError(f"Expected 3D tensor, got shape {tensor.shape}")
    
    if tensor.shape[2] != 3:
        raise ValueError(f"Expected 3 channels, got {tensor.shape[2]}")
    
    if tensor.shape[1] != len(variable_order):
        raise ValueError(f"Expected {len(variable_order)} variables, got {tensor.shape[1]}")
    
    # Check for NaN or infinite values
    if not jnp.all(jnp.isfinite(tensor)):
        raise ValueError("Tensor contains NaN or infinite values")
    
    # Check target channel has exactly one 1.0 per timestep
    target_channel = tensor[:, :, 1]
    target_sums = jnp.sum(target_channel, axis=1)
    if not jnp.allclose(target_sums[target_sums > 0], 1.0):
        raise ValueError("Target channel must have exactly one 1.0 per non-padded timestep")
    
    # Check intervention channel is binary
    intervention_channel = tensor[:, :, 2]
    is_binary = jnp.all((intervention_channel == 0) | (intervention_channel == 1))
    if not is_binary:
        raise ValueError("Intervention channel must be binary (0 or 1)")
    
    return True