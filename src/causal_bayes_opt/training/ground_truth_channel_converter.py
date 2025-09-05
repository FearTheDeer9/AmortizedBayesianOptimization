"""
Ground truth based 4-channel tensor converter.

Instead of using expensive surrogate predictions, this converter uses
the ground truth SCM structure with progressive certainty based on 
data quantity.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
import jax.numpy as jnp
import numpy as np

from ..data_structures.buffer import ExperienceBuffer
from ..data_structures.sample import get_values, get_intervention_targets, is_observational
from ..data_structures.scm import get_parents
from ..utils.variable_mapping import VariableMapper

logger = logging.getLogger(__name__)


def buffer_to_ground_truth_four_channel_tensor(
    buffer: ExperienceBuffer,
    target_variable: str,
    scm: Dict[str, Any],
    max_history_size: int = 100,
    standardize: bool = True,
    convergence_rate_factor: float = 0.2
) -> Tuple[jnp.ndarray, VariableMapper, Dict[str, Any]]:
    """
    Convert experience buffer to 4-channel tensor using ground truth parent information.
    
    Creates tensor with channels:
    0. Values: Variable values (standardized per-variable if enabled)
    1. Target: Binary indicator for target variable
    2. Intervention: Binary indicator for intervened variables
    3. Probs: Parent probabilities based on ground truth with progressive certainty
    
    Progressive certainty model:
    - Starts at 0.2 (maximum uncertainty) for all variables
    - For observational data: converges to 0.5 for true parents, 0 for non-parents
    - For interventional data: converges to 1.0 for true parents, 0 for non-parents
    - Convergence rate depends on graph size
    
    Args:
        buffer: Experience buffer with samples
        target_variable: Name of target variable
        scm: The ground truth SCM structure
        max_history_size: Maximum number of historical samples
        standardize: Whether to standardize values per-variable
        convergence_rate_factor: Controls convergence speed (smaller = slower)
        
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
    
    # Get ground truth parents
    true_parents = set(get_parents(scm, target_variable))
    
    # Count observational and interventional samples
    n_obs = sum(1 for s in all_samples if is_observational(s))
    n_int = len(all_samples) - n_obs
    
    # Calculate convergence rate based on graph size
    # Larger graphs converge more slowly
    lambda_rate = convergence_rate_factor / n_vars
    
    # Calculate parent probabilities based on data quantity
    parent_probs = np.zeros(n_vars)
    
    for i, var in enumerate(variable_order):
        if var == target_variable:
            # Target variable itself gets 0 probability
            parent_probs[i] = 0.0
        elif var in true_parents:
            # True parents: progressive certainty
            if n_int > 0:
                # We have interventional data: converge from 0.5 to 1.0
                # Use number of interventions as the progression metric
                parent_probs[i] = 0.5 + 0.5 * (1 - np.exp(-lambda_rate * n_int))
            else:
                # Only observational data: converge from 0.2 to 0.5
                parent_probs[i] = 0.2 + 0.3 * (1 - np.exp(-lambda_rate * n_obs))
        else:
            # Non-parents: decay from 0.2 to 0
            if n_int > 0:
                # With interventional data: decay from 0.2 to 0
                parent_probs[i] = 0.2 - 0.2 * (1 - np.exp(-lambda_rate * n_int))
            else:
                # Only observational data: decay from 0.2 to 0
                parent_probs[i] = 0.2 - 0.2 * (1 - np.exp(-lambda_rate * n_obs))
    
    parent_probs = jnp.array(parent_probs)
    
    # Log the progressive certainty state
    logger.debug(f"Ground truth parent probs for {target_variable}:")
    logger.debug(f"  True parents: {true_parents}")
    logger.debug(f"  N_obs: {n_obs}, N_int: {n_int}")
    logger.debug(f"  Lambda rate: {lambda_rate:.4f}")
    logger.debug(f"  Parent probs: {dict(zip(variable_order, parent_probs))}")
    
    # Determine actual history size
    actual_size = min(len(all_samples), max_history_size)
    recent_samples = all_samples[-actual_size:]
    
    # Initialize 4-channel tensor
    tensor = jnp.zeros((max_history_size, n_vars, 4))
    
    # Fill tensor with recent samples
    for t, sample in enumerate(recent_samples):
        tensor_idx = max_history_size - actual_size + t
        
        # Channel 0: Values
        values = _extract_values_vector(sample, variable_order)
        
        # Channel 1: Target indicator
        target_mask = jnp.array([1.0 if var == target_variable else 0.0 for var in variable_order])
        
        # Channel 2: Intervention indicator
        intervention_targets = get_intervention_targets(sample)
        intervention_mask = jnp.array([1.0 if var in intervention_targets else 0.0 for var in variable_order])
        
        # Channel 3: Parent probabilities (progressive per sample)
        # Calculate counts up to and including this sample
        obs_count_at_t = sum(1 for s in recent_samples[:t+1] if not get_intervention_targets(s))
        int_count_at_t = sum(1 for s in recent_samples[:t+1] if get_intervention_targets(s))
        
        # Calculate progressive parent probs for this specific sample
        sample_parent_probs = []
        for i, var in enumerate(variable_order):
            if var == target_variable:
                # Target variable itself gets 0 probability
                sample_parent_probs.append(0.0)
            elif var in true_parents:
                # True parents: progressive certainty based on sample position
                if int_count_at_t > 0:
                    # We have interventional data: converge from 0.5 to 1.0
                    prob = 0.5 + 0.5 * (1 - np.exp(-lambda_rate * int_count_at_t))
                else:
                    # Only observational data: converge from 0.2 to 0.5
                    prob = 0.2 + 0.3 * (1 - np.exp(-lambda_rate * obs_count_at_t))
                sample_parent_probs.append(prob)
            else:
                # Non-parents: decay from 0.2 to 0
                if int_count_at_t > 0:
                    # With interventional data: decay from 0.2 to 0
                    prob = 0.2 - 0.2 * (1 - np.exp(-lambda_rate * int_count_at_t))
                else:
                    # Only observational data: decay from 0.2 to 0
                    prob = 0.2 - 0.2 * (1 - np.exp(-lambda_rate * obs_count_at_t))
                sample_parent_probs.append(prob)
        
        sample_parent_probs = jnp.array(sample_parent_probs)
        
        # Set tensor values
        tensor = tensor.at[tensor_idx, :, 0].set(values)
        tensor = tensor.at[tensor_idx, :, 1].set(target_mask)
        tensor = tensor.at[tensor_idx, :, 2].set(intervention_mask)
        tensor = tensor.at[tensor_idx, :, 3].set(sample_parent_probs)
    
    # Standardize values if requested (per-variable)
    if standardize and actual_size > 1:
        tensor = _standardize_values_four_channel(tensor, actual_size)
    
    # Diagnostics
    diagnostics = {
        'n_variables': n_vars,
        'n_samples': len(all_samples),
        'n_obs': n_obs,
        'n_int': n_int,
        'actual_history_size': actual_size,
        'target_variable': target_variable,
        'target_idx': target_idx,
        'true_parents': list(true_parents),
        'lambda_rate': float(lambda_rate),
        'convergence_state': 'interventional' if n_int > 0 else 'observational'
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