#!/usr/bin/env python3
"""
AcquisitionState to JAX Tensor Converter

Converts complex AcquisitionState objects to JAX-compatible tensor representations
for use in JIT-compiled functions. This is critical for BC training with JAX.

Key Features:
1. Extract numerical features from AcquisitionState
2. Handle variable-length histories with padding
3. Ensure consistent tensor shapes for batching
4. Maintain compatibility with policy networks

Design Principles:
- Pure functions for state conversion
- Explicit handling of missing data
- Consistent tensor formats
- No string operations in tensors
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as onp
import pyrsistent as pyr

from ..acquisition.state import AcquisitionState
from ..data_structures.sample import get_values, get_intervention_targets
from ..data_structures.scm import get_variables

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TensorizedAcquisitionState:
    """JAX-compatible tensor representation of AcquisitionState."""
    # Core state tensors
    current_data: jnp.ndarray  # [n_samples, n_vars, n_channels]
    intervention_history: jnp.ndarray  # [history_len, n_vars, n_features]
    uncertainty_estimate: jnp.ndarray  # [n_vars]
    
    # Scalar values
    best_value: jnp.ndarray  # scalar
    step: jnp.ndarray  # scalar
    n_variables: int
    
    # Optional enhanced features
    variable_embeddings: Optional[jnp.ndarray] = None  # [n_vars, embed_dim]
    target_mask: Optional[jnp.ndarray] = None  # [n_vars]


def extract_current_data_tensor(
    state: AcquisitionState,
    max_samples: int = 100,
    n_channels: int = 3
) -> jnp.ndarray:
    """
    Extract current observational data from state as tensor.
    
    Args:
        state: AcquisitionState to extract from
        max_samples: Maximum number of samples to include
        n_channels: Number of channels per variable
        
    Returns:
        JAX array of shape [n_samples, n_vars, n_channels]
    """
    # Get variables for consistent ordering
    variables = []
    if hasattr(state, 'scm'):
        variables = list(get_variables(state.scm))
    elif hasattr(state, 'metadata'):
        if 'scm_info' in state.metadata and 'variables' in state.metadata['scm_info']:
            variables = list(state.metadata['scm_info']['variables'])
        elif 'scm' in state.metadata and 'variables' in state.metadata['scm']:
            variables = list(state.metadata['scm']['variables'])
    elif hasattr(state, 'posterior') and hasattr(state.posterior, 'variable_order'):
        variables = list(state.posterior.variable_order)
    
    # Get samples from buffer
    if hasattr(state, 'buffer') and hasattr(state.buffer, 'get_all_samples'):
        samples = state.buffer.get_all_samples()[-max_samples:]  # Recent samples
    else:
        samples = []
    
    if not samples:
        # Return zeros if no samples
        # Get n_vars from state
        n_vars = 5  # default
        if hasattr(state, 'scm'):
            n_vars = len(get_variables(state.scm))
        elif hasattr(state, 'metadata'):
            if 'scm_info' in state.metadata and 'variables' in state.metadata['scm_info']:
                n_vars = len(state.metadata['scm_info']['variables'])
            elif 'scm' in state.metadata and 'variables' in state.metadata['scm']:
                n_vars = len(state.metadata['scm']['variables'])
        return jnp.zeros((1, n_vars, n_channels))
    
    # Extract values from samples
    data_arrays = []
    for sample in samples:
        values = get_values(sample)
        if isinstance(values, (dict, pyr.PMap)):
            # Convert dict/pmap to array
            # First get consistent variable ordering
            if variables:
                # Use the provided variable ordering
                value_array = jnp.array([float(values.get(var, 0.0)) for var in variables])
            else:
                # Fall back to sorted keys
                var_names = sorted(values.keys())
                value_array = jnp.array([float(values[var]) for var in var_names])
        else:
            value_array = jnp.array(values)
        
        # Reshape to [n_vars, 1] and tile to [n_vars, n_channels]
        if value_array.ndim == 1:
            value_array = value_array[:, None]
        if value_array.shape[1] < n_channels:
            value_array = jnp.tile(value_array, (1, n_channels))[:, :n_channels]
        
        data_arrays.append(value_array)
    
    # Stack samples
    data_tensor = jnp.stack(data_arrays, axis=0)  # [n_samples, n_vars, n_channels]
    
    return data_tensor


def extract_intervention_history_tensor(
    state: AcquisitionState,
    max_history: int = 20,
    n_features: int = 3
) -> jnp.ndarray:
    """
    Extract intervention history as tensor.
    
    Args:
        state: AcquisitionState to extract from
        max_history: Maximum history length
        n_features: Features per intervention
        
    Returns:
        JAX array of shape [history_len, n_vars, n_features]
    """
    # Get variables - check multiple possible locations
    variables = []
    n_vars = 5  # default
    
    # Try to get variables from various possible locations
    if hasattr(state, 'scm'):
        variables = list(get_variables(state.scm))
        n_vars = len(variables)
    elif hasattr(state, 'metadata') and 'scm' in state.metadata:
        scm_data = state.metadata['scm']
        if 'variables' in scm_data:
            variables = list(scm_data['variables'])
            n_vars = len(variables)
    elif hasattr(state, 'metadata') and 'scm_info' in state.metadata:
        scm_info = state.metadata['scm_info']
        if 'variables' in scm_info:
            variables = list(scm_info['variables'])
            n_vars = len(variables)
    elif hasattr(state, 'scm_info') and 'variables' in state.scm_info:
        variables = list(state.scm_info['variables'])
        n_vars = len(variables)
    
    # Get intervention history
    if hasattr(state, 'intervention_history') and state.intervention_history:
        history = state.intervention_history[-max_history:]
    else:
        # Return zeros if no history
        return jnp.zeros((1, n_vars, n_features))
    
    # Convert history to tensor
    history_arrays = []
    for intervention in history:
        # Create intervention tensor [n_vars, n_features]
        intervention_tensor = jnp.zeros((n_vars, n_features))
        
        if isinstance(intervention, dict):
            # Extract intervention targets and values
            targets = intervention.get('variables', set())
            values = intervention.get('values', {})
            
            for i, var in enumerate(variables):
                if var in targets:
                    # Variable was intervened on
                    intervention_tensor = intervention_tensor.at[i, 0].set(1.0)  # Indicator
                    if var in values:
                        intervention_tensor = intervention_tensor.at[i, 1].set(values[var])  # Value
                    intervention_tensor = intervention_tensor.at[i, 2].set(1.0)  # Confidence
        
        history_arrays.append(intervention_tensor)
    
    # Stack history
    history_tensor = jnp.stack(history_arrays, axis=0)  # [history_len, n_vars, n_features]
    
    return history_tensor


def extract_uncertainty_tensor(
    state: AcquisitionState
) -> jnp.ndarray:
    """
    Extract uncertainty estimates as tensor.
    
    Args:
        state: AcquisitionState to extract from
        
    Returns:
        JAX array of shape [n_vars]
    """
    # Get variables - check multiple possible locations
    variables = []
    n_vars = 5  # default
    
    # Try to get variables from various possible locations
    if hasattr(state, 'scm'):
        variables = list(get_variables(state.scm))
        n_vars = len(variables)
    elif hasattr(state, 'metadata') and 'scm' in state.metadata:
        scm_data = state.metadata['scm']
        if 'variables' in scm_data:
            variables = list(scm_data['variables'])
            n_vars = len(variables)
    elif hasattr(state, 'metadata') and 'scm_info' in state.metadata:
        scm_info = state.metadata['scm_info']
        if 'variables' in scm_info:
            variables = list(scm_info['variables'])
            n_vars = len(variables)
    elif hasattr(state, 'scm_info') and 'variables' in state.scm_info:
        variables = list(state.scm_info['variables'])
        n_vars = len(variables)
    
    # Extract uncertainty
    if hasattr(state, 'uncertainty_estimate'):
        if isinstance(state.uncertainty_estimate, dict):
            # Convert dict to array
            uncertainty = jnp.array([
                state.uncertainty_estimate.get(var, 0.5) for var in variables
            ])
        else:
            uncertainty = jnp.array(state.uncertainty_estimate)
    else:
        # Default uncertainty
        uncertainty = jnp.ones(n_vars) * 0.5
    
    return uncertainty


def convert_acquisition_state_to_tensors(
    state: AcquisitionState,
    max_samples: int = 100,
    max_history: int = 20,
    n_channels: int = 3,
    n_features: int = 3
) -> TensorizedAcquisitionState:
    """
    Convert AcquisitionState to JAX-compatible tensor representation.
    
    Args:
        state: AcquisitionState to convert
        max_samples: Maximum number of samples
        max_history: Maximum history length
        n_channels: Channels per variable in data
        n_features: Features per intervention
        
    Returns:
        TensorizedAcquisitionState with all tensors
    """
    # Extract core tensors
    current_data = extract_current_data_tensor(state, max_samples, n_channels)
    intervention_history = extract_intervention_history_tensor(state, max_history, n_features)
    uncertainty_estimate = extract_uncertainty_tensor(state)
    
    # Extract scalar values
    best_value = jnp.array(getattr(state, 'best_value', 0.0))
    step = jnp.array(getattr(state, 'step', 0))
    
    # Get number of variables
    n_variables = current_data.shape[1]
    
    # Create target mask if target variable is specified
    target_mask = None
    if hasattr(state, 'target_variable') or hasattr(state, 'current_target'):
        target_var = getattr(state, 'target_variable', getattr(state, 'current_target', None))
        if target_var:
            # Get variables list
            variables = []
            if hasattr(state, 'scm'):
                variables = list(get_variables(state.scm))
            elif hasattr(state, 'metadata'):
                if 'scm_info' in state.metadata and 'variables' in state.metadata['scm_info']:
                    variables = list(state.metadata['scm_info']['variables'])
                elif 'scm' in state.metadata and 'variables' in state.metadata['scm']:
                    variables = list(state.metadata['scm']['variables'])
            
            if target_var in variables:
                target_idx = variables.index(target_var)
                target_mask = jnp.zeros(n_variables)
                target_mask = target_mask.at[target_idx].set(1.0)
    
    return TensorizedAcquisitionState(
        current_data=current_data,
        intervention_history=intervention_history,
        uncertainty_estimate=uncertainty_estimate,
        best_value=best_value,
        step=step,
        n_variables=n_variables,
        target_mask=target_mask
    )


def create_batch_tensor_state(
    states: List[AcquisitionState],
    **kwargs
) -> Dict[str, jnp.ndarray]:
    """
    Convert list of AcquisitionStates to batched tensor dictionary.
    
    Args:
        states: List of AcquisitionStates
        **kwargs: Arguments for convert_acquisition_state_to_tensors
        
    Returns:
        Dictionary of batched tensors
    """
    # Convert each state
    tensor_states = [
        convert_acquisition_state_to_tensors(state, **kwargs)
        for state in states
    ]
    
    # Find maximum dimensions for padding
    max_samples = max(ts.current_data.shape[0] for ts in tensor_states)
    max_history = max(ts.intervention_history.shape[0] for ts in tensor_states)
    max_vars = max(ts.n_variables for ts in tensor_states)
    
    # Pad and batch tensors
    batched_data = []
    batched_history = []
    batched_uncertainty = []
    
    for ts in tensor_states:
        # Pad current data
        n_samples, n_vars, n_channels = ts.current_data.shape
        if n_samples < max_samples or n_vars < max_vars:
            padded = jnp.zeros((max_samples, max_vars, n_channels))
            padded = padded.at[:n_samples, :n_vars].set(ts.current_data)
            batched_data.append(padded)
        else:
            batched_data.append(ts.current_data[:max_samples, :max_vars])
        
        # Pad history
        history_len, n_vars, n_features = ts.intervention_history.shape
        if history_len < max_history or n_vars < max_vars:
            padded = jnp.zeros((max_history, max_vars, n_features))
            padded = padded.at[:history_len, :n_vars].set(ts.intervention_history)
            batched_history.append(padded)
        else:
            batched_history.append(ts.intervention_history[:max_history, :max_vars])
        
        # Pad uncertainty
        if len(ts.uncertainty_estimate) < max_vars:
            padded = jnp.ones(max_vars) * 0.5  # Default uncertainty
            padded = padded.at[:len(ts.uncertainty_estimate)].set(ts.uncertainty_estimate)
            batched_uncertainty.append(padded)
        else:
            batched_uncertainty.append(ts.uncertainty_estimate[:max_vars])
    
    # Stack batches
    return {
        'current_data': jnp.stack(batched_data),  # [batch, max_samples, max_vars, n_channels]
        'intervention_history': jnp.stack(batched_history),  # [batch, max_history, max_vars, n_features]
        'uncertainty_estimate': jnp.stack(batched_uncertainty),  # [batch, max_vars]
        'best_value': jnp.array([ts.best_value for ts in tensor_states]),  # [batch]
        'step': jnp.array([ts.step for ts in tensor_states]),  # [batch]
    }


def tensor_state_to_dict(tensor_state: TensorizedAcquisitionState) -> Dict[str, jnp.ndarray]:
    """Convert TensorizedAcquisitionState to dictionary for JAX functions."""
    return {
        'current_data': tensor_state.current_data,
        'intervention_history': tensor_state.intervention_history,
        'uncertainty_estimate': tensor_state.uncertainty_estimate,
        'best_value': tensor_state.best_value,
        'step': tensor_state.step
    }