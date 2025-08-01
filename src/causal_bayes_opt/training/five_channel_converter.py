"""
Five-channel tensor conversion for ACBO with surrogate integration.

This module provides conversion from experience buffers to the enhanced
5-channel tensor format that integrates surrogate predictions into policy input.

The 5-channel format:
- Channel 0: past_node_values (standardized values from all samples)
- Channel 1: target_indicator (1.0 for target variable)
- Channel 2: intervened_on_indicator (1.0 if variable was intervened on)
- Channel 3: marginal_parent_probs (from surrogate predictions)
- Channel 4: intervention_recency (steps since last intervention)

Key features:
- Validates surrogate predictions to ensure non-zero signals
- Comprehensive logging for debugging
- Support for missing surrogates (graceful degradation)
- Compatible with both training and evaluation
"""

import jax
import jax.numpy as jnp
from typing import List, Tuple, Optional, Dict, Any, Callable
import logging

from ..data_structures.buffer import ExperienceBuffer
from ..data_structures.sample import get_values, get_intervention_targets
from ..utils.variable_mapping import VariableMapper

logger = logging.getLogger(__name__)


def buffer_to_five_channel_tensor(
    buffer: ExperienceBuffer,
    target_variable: str,
    surrogate_fn: Optional[Callable] = None,
    max_history_size: int = 100,
    standardize: bool = True,
    validate_signals: bool = True
) -> Tuple[jnp.ndarray, VariableMapper, Dict[str, Any]]:
    """
    Convert experience buffer to 5-channel tensor format with surrogate integration.
    
    This function creates the enhanced ACBO tensor representation that includes
    surrogate predictions for structure learning integration.
    
    Args:
        buffer: Experience buffer with samples
        target_variable: Name of target variable
        surrogate_fn: Optional function that predicts parent probabilities
                     Should accept (tensor_3ch, target) and return posterior
        max_history_size: Maximum number of historical samples to include
        standardize: Whether to standardize values using global statistics
        validate_signals: Whether to validate surrogate signals are non-zero
        
    Returns:
        Tuple of:
        - tensor: [T, n_vars, 5] tensor in enhanced ACBO format
        - mapper: VariableMapper instance for variable name/index conversions
        - diagnostics: Dict with validation info and statistics
        
    Raises:
        ValueError: If buffer is empty or target not in variables
    """
    logger.debug(f"[5-Channel Converter] Converting buffer with {buffer.size()} samples")
    logger.debug(f"[5-Channel Converter] Target variable: {target_variable}")
    logger.debug(f"[5-Channel Converter] Has surrogate function: {surrogate_fn is not None}")
    
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
    target_idx = mapper.get_index(target_variable)
    logger.debug(f"[5-Channel Converter] Created VariableMapper with variables: {mapper.variables}")
    logger.debug(f"[5-Channel Converter] Target index: {target_idx}")
    
    # First create 3-channel tensor for surrogate prediction
    tensor_3ch = _create_three_channel_tensor(
        all_samples, variable_order, target_variable, max_history_size, standardize
    )
    
    # Get surrogate predictions if available
    marginal_probs = None
    posterior = None
    if surrogate_fn is not None:
        try:
            # Call surrogate with 3-channel tensor and variable order
            logger.debug(f"[5-Channel Converter] Calling surrogate with tensor shape: {tensor_3ch.shape}")
            posterior = surrogate_fn(tensor_3ch, target_variable, variable_order)
            
            # Extract marginal probabilities
            marginal_probs = _extract_marginal_probs(posterior, variable_order, target_variable)
            
            # Log posterior details
            logger.debug(f"[5-Channel Converter] Posterior type: {type(posterior).__name__}")
            if marginal_probs:
                logger.debug(f"[5-Channel Converter] Extracted marginal probs for {len(marginal_probs)} variables")
                logger.debug(f"[5-Channel Converter] Sample marginals: {list(marginal_probs.items())[:3]}...")
            else:
                logger.debug(f"[5-Channel Converter] No marginal probs extracted")
            
            if validate_signals:
                # Validate non-zero signal
                is_valid, validation_msg = _validate_surrogate_signal(marginal_probs)
                if not is_valid:
                    logger.warning(f"Surrogate validation failed: {validation_msg}")
                else:
                    logger.debug(f"Surrogate signal validated: {validation_msg}")
                    
        except Exception as e:
            logger.warning(f"Surrogate prediction failed: {e}")
            marginal_probs = None
    
    # Create 5-channel tensor
    tensor_5ch = _extend_to_five_channels(
        tensor_3ch, all_samples, variable_order, marginal_probs, max_history_size
    )
    
    # Collect diagnostics
    diagnostics = {
        'has_surrogate': surrogate_fn is not None,
        'surrogate_success': marginal_probs is not None,
        'n_variables': n_vars,
        'n_samples': len(all_samples),
        'actual_history_size': min(len(all_samples), max_history_size),
        'target_variable': target_variable,
        'target_idx': target_idx
    }
    
    if marginal_probs is not None:
        # Add surrogate statistics
        probs_array = jnp.array(list(marginal_probs.values()))
        diagnostics['surrogate_stats'] = {
            'mean_prob': float(jnp.mean(probs_array)),
            'max_prob': float(jnp.max(probs_array)),
            'min_prob': float(jnp.min(probs_array)),
            'std_prob': float(jnp.std(probs_array)),
            'num_nonzero': int(jnp.sum(probs_array > 0.01))
        }
        
        # Log detailed signal info
        logger.info(
            f"Surrogate signal stats: mean={diagnostics['surrogate_stats']['mean_prob']:.3f}, "
            f"max={diagnostics['surrogate_stats']['max_prob']:.3f}, "
            f"nonzero={diagnostics['surrogate_stats']['num_nonzero']}/{n_vars-1}"
        )
    
    return tensor_5ch, mapper, diagnostics


def _create_three_channel_tensor(
    samples: List[Any],
    variable_order: List[str],
    target_variable: str,
    max_history_size: int,
    standardize: bool
) -> jnp.ndarray:
    """Create initial 3-channel tensor from samples."""
    n_vars = len(variable_order)
    actual_size = min(len(samples), max_history_size)
    
    # Initialize tensor
    tensor = jnp.zeros((max_history_size, n_vars, 3))
    
    # Use most recent samples
    recent_samples = samples[-actual_size:]
    
    # Fill tensor
    for t, sample in enumerate(recent_samples):
        tensor_idx = max_history_size - actual_size + t
        
        # Channel 0: Values
        values = _extract_values_vector(sample, variable_order)
        
        # Channel 1: Target indicator
        target_mask = jnp.array([1.0 if var == target_variable else 0.0 for var in variable_order])
        
        # Channel 2: Intervention indicator  
        intervention_targets = get_intervention_targets(sample)
        intervention_mask = jnp.array([1.0 if var in intervention_targets else 0.0 for var in variable_order])
        
        tensor = tensor.at[tensor_idx, :, 0].set(values)
        tensor = tensor.at[tensor_idx, :, 1].set(target_mask)
        tensor = tensor.at[tensor_idx, :, 2].set(intervention_mask)
    
    # Standardize if requested
    if standardize and actual_size > 1:
        tensor = _standardize_values(tensor, actual_size)
    
    return tensor


def _extend_to_five_channels(
    tensor_3ch: jnp.ndarray,
    all_samples: List[Any],
    variable_order: List[str],
    marginal_probs: Optional[Dict[str, float]],
    max_history_size: int
) -> jnp.ndarray:
    """Extend 3-channel tensor to 5 channels with surrogate predictions."""
    T, n_vars, _ = tensor_3ch.shape
    
    # Initialize 5-channel tensor
    tensor_5ch = jnp.zeros((T, n_vars, 5))
    
    # Copy first 3 channels
    tensor_5ch = tensor_5ch.at[:, :, :3].set(tensor_3ch)
    
    # Channel 3: Marginal parent probabilities
    if marginal_probs is not None:
        # Create probability vector
        prob_vector = jnp.array([marginal_probs.get(var, 0.0) for var in variable_order])
        
        # Broadcast to all timesteps (static prediction)
        # In future, could make this time-varying
        tensor_5ch = tensor_5ch.at[:, :, 3].set(prob_vector[None, :])
    else:
        # No surrogate - use zeros
        logger.debug("No surrogate predictions available, using zero probabilities")
    
    # Channel 4: Intervention recency
    recency_matrix = _compute_intervention_recency_matrix(
        all_samples, variable_order, max_history_size
    )
    tensor_5ch = tensor_5ch.at[:, :, 4].set(recency_matrix)
    
    return tensor_5ch


def _extract_marginal_probs(
    posterior: Any,
    variable_order: List[str],
    target_variable: str
) -> Optional[Dict[str, float]]:
    """Extract marginal parent probabilities from posterior using canonical patterns."""
    if posterior is None:
        return None
    
    logger.debug(f"Extracting marginals from posterior type: {type(posterior).__name__}")
    
    # Try canonical extraction first for ParentSetPosterior
    try:
        from ..avici_integration.parent_set.posterior import ParentSetPosterior, get_marginal_parent_probabilities
        
        if isinstance(posterior, ParentSetPosterior):
            logger.debug("Using canonical ParentSetPosterior extraction")
            # Check if it has metadata with marginals first (our BC surrogate format)
            if hasattr(posterior, 'metadata'):
                # Handle both dict and pyrsistent PMap
                try:
                    if 'marginal_parent_probs' in posterior.metadata:
                        logger.debug("Found marginal_parent_probs in ParentSetPosterior metadata")
                        raw_probs = posterior.metadata['marginal_parent_probs']
                        marginal_probs = {}
                        for var in variable_order:
                            if var != target_variable:
                                marginal_probs[var] = float(raw_probs.get(var, 0.0))
                            else:
                                marginal_probs[var] = 0.0
                        logger.debug(f"Extracted marginals from metadata: {marginal_probs}")
                        logger.debug(f"Variable order requested: {variable_order}")
                        logger.debug(f"Raw probs keys: {list(raw_probs.keys())}")
                        return marginal_probs
                except Exception as e:
                    logger.debug(f"Failed to extract from metadata: {e}")
            
            # Fallback to canonical extraction
            marginals = get_marginal_parent_probabilities(posterior, variable_order)
            # Ensure target has zero probability
            marginals[target_variable] = 0.0
            return marginals
    except ImportError:
        logger.debug("ParentSetPosterior not available, using fallback extraction")
    except Exception as e:
        logger.debug(f"Canonical extraction failed: {e}, trying fallbacks")
    
    # Handle different posterior formats
    marginal_probs = {}
    
    if isinstance(posterior, dict):
        # Dictionary format
        if 'marginal_parent_probs' in posterior:
            raw_probs = posterior['marginal_parent_probs']
            # Ensure all variables have entries
            for var in variable_order:
                if var != target_variable:
                    marginal_probs[var] = float(raw_probs.get(var, 0.0))
                else:
                    marginal_probs[var] = 0.0  # Target can't be its own parent
                    
        elif 'metadata' in posterior and 'marginal_parent_probs' in posterior['metadata']:
            # Nested format
            raw_probs = posterior['metadata']['marginal_parent_probs']
            for var in variable_order:
                if var != target_variable:
                    marginal_probs[var] = float(raw_probs.get(var, 0.0))
                else:
                    marginal_probs[var] = 0.0
                    
    elif hasattr(posterior, 'marginal_parent_probs'):
        # Object with attribute
        raw_probs = posterior.marginal_parent_probs
        for var in variable_order:
            if var != target_variable:
                marginal_probs[var] = float(raw_probs.get(var, 0.0))
            else:
                marginal_probs[var] = 0.0
                
    elif hasattr(posterior, 'metadata'):
        # ParentSetPosterior with metadata attribute (fallback if canonical fails)
        logger.debug("Checking metadata attribute on posterior object")
        if hasattr(posterior.metadata, '__getitem__'):
            # It's dict-like
            if 'marginal_parent_probs' in posterior.metadata:
                logger.debug("Found marginal_parent_probs in metadata dict")
                raw_probs = posterior.metadata['marginal_parent_probs']
                for var in variable_order:
                    if var != target_variable:
                        marginal_probs[var] = float(raw_probs.get(var, 0.0))
                    else:
                        marginal_probs[var] = 0.0
            else:
                logger.warning(f"No marginal_parent_probs in posterior metadata: {list(posterior.metadata.keys()) if hasattr(posterior.metadata, 'keys') else 'not a dict'}")
                return None
        else:
            logger.warning(f"Metadata is not dict-like: {type(posterior.metadata)}")
            return None
    else:
        logger.warning(f"Unknown posterior format: {type(posterior).__name__}, attributes: {dir(posterior)[:10]}")
        return None
    
    if marginal_probs:
        logger.debug(f"Successfully extracted {len(marginal_probs)} marginal probabilities")
    
    return marginal_probs


def _validate_surrogate_signal(marginal_probs: Dict[str, float]) -> Tuple[bool, str]:
    """Validate that surrogate predictions contain meaningful signal."""
    if not marginal_probs:
        return False, "No marginal probabilities found"
    
    probs = list(marginal_probs.values())
    
    # Check if all zeros
    if all(p == 0.0 for p in probs):
        return False, "All probabilities are zero"
    
    # Check if all same value (no discrimination)
    if len(set(probs)) == 1:
        return False, f"All probabilities are identical: {probs[0]}"
    
    # Check if any meaningful signal (some prob > threshold)
    meaningful_threshold = 0.01
    num_meaningful = sum(1 for p in probs if p > meaningful_threshold)
    
    if num_meaningful == 0:
        return False, f"No probabilities above threshold {meaningful_threshold}"
    
    # Passed all checks
    max_prob = max(probs)
    min_prob = min(probs)
    return True, f"Valid signal detected: range=[{min_prob:.3f}, {max_prob:.3f}], meaningful={num_meaningful}"


def _compute_intervention_recency_matrix(
    all_samples: List[Any],
    variable_order: List[str],
    max_history_size: int
) -> jnp.ndarray:
    """Compute intervention recency for each variable at each timestep."""
    n_vars = len(variable_order)
    actual_size = min(len(all_samples), max_history_size)
    
    # Initialize with max recency
    recency_matrix = jnp.ones((max_history_size, n_vars))
    
    # Track last intervention for each variable
    last_intervention = {var: -1 for var in variable_order}
    
    # Process samples to compute recency
    recent_samples = all_samples[-actual_size:]
    
    for t, sample in enumerate(recent_samples):
        global_t = len(all_samples) - actual_size + t
        tensor_idx = max_history_size - actual_size + t
        
        # Update last intervention times
        intervention_targets = get_intervention_targets(sample)
        for var in intervention_targets:
            if var in last_intervention:
                last_intervention[var] = global_t
        
        # Compute recency for this timestep
        for i, var in enumerate(variable_order):
            if last_intervention[var] >= 0:
                # Steps since last intervention
                recency = global_t - last_intervention[var]
                # Normalize to [0, 1] range
                normalized_recency = min(recency / float(max_history_size), 1.0)
            else:
                # Never intervened
                normalized_recency = 1.0
            
            recency_matrix = recency_matrix.at[tensor_idx, i].set(normalized_recency)
    
    return recency_matrix


def _extract_values_vector(sample: Any, variable_order: List[str]) -> jnp.ndarray:
    """Extract ordered values from sample."""
    values = get_values(sample)
    return jnp.array([float(values.get(var, 0.0)) for var in variable_order])


def _standardize_values(tensor: jnp.ndarray, actual_size: int) -> jnp.ndarray:
    """Standardize the values channel using global statistics."""
    # Get the actual data region (excluding padding)
    start_idx = tensor.shape[0] - actual_size
    actual_data = tensor[start_idx:, :, 0]  # Values channel
    
    # Compute global statistics
    global_mean = jnp.mean(actual_data)
    global_std = jnp.std(actual_data) + 1e-8
    
    # Standardize only the actual data region
    standardized_values = (tensor[:, :, 0] - global_mean) / global_std
    
    # Only update the non-padded region
    mask = jnp.zeros(tensor.shape[0])
    mask = mask.at[start_idx:].set(1.0)
    
    # Apply standardization only to actual data
    new_values = tensor[:, :, 0] * (1 - mask[:, None]) + standardized_values * mask[:, None]
    
    # Update tensor
    return tensor.at[:, :, 0].set(new_values)


def validate_five_channel_tensor(
    tensor: jnp.ndarray,
    variable_order: List[str],
    check_surrogate: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate that a tensor follows the 5-channel ACBO format.
    
    Args:
        tensor: Tensor to validate
        variable_order: Expected variable order
        check_surrogate: Whether to check surrogate channel validity
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check shape
    if len(tensor.shape) != 3:
        issues.append(f"Expected 3D tensor, got shape {tensor.shape}")
        return False, issues
    
    if tensor.shape[2] != 5:
        issues.append(f"Expected 5 channels, got {tensor.shape[2]}")
        return False, issues
    
    if tensor.shape[1] != len(variable_order):
        issues.append(f"Expected {len(variable_order)} variables, got {tensor.shape[1]}")
        return False, issues
    
    # Check for NaN or infinite values
    if not jnp.all(jnp.isfinite(tensor)):
        issues.append("Tensor contains NaN or infinite values")
    
    # Check target channel (channel 1)
    target_channel = tensor[:, :, 1]
    target_sums = jnp.sum(target_channel, axis=1)
    non_zero_mask = target_sums > 0
    if jnp.any(non_zero_mask) and not jnp.allclose(target_sums[non_zero_mask], 1.0):
        issues.append("Target channel must have exactly one 1.0 per non-padded timestep")
    
    # Check intervention channel (channel 2)
    intervention_channel = tensor[:, :, 2]
    is_binary = jnp.all((intervention_channel == 0) | (intervention_channel == 1))
    if not is_binary:
        issues.append("Intervention channel must be binary (0 or 1)")
    
    # Check surrogate channel (channel 3)
    if check_surrogate:
        surrogate_channel = tensor[:, :, 3]
        # Should be probabilities [0, 1]
        if jnp.any(surrogate_channel < -1e-6) or jnp.any(surrogate_channel > 1.0 + 1e-6):
            issues.append("Surrogate probabilities must be in [0, 1] range")
        
        # Check if all zeros (might indicate missing surrogate)
        if jnp.all(surrogate_channel == 0.0):
            issues.append("Warning: All surrogate probabilities are zero")
    
    # Check recency channel (channel 4)
    recency_channel = tensor[:, :, 4]
    if jnp.any(recency_channel < -1e-6) or jnp.any(recency_channel > 1.0 + 1e-6):
        issues.append("Recency values must be in [0, 1] range")
    
    return len(issues) == 0, issues


def convert_three_to_five_channel(
    tensor_3ch: jnp.ndarray,
    variable_order: List[str],
    target_variable: str,
    surrogate_fn: Optional[Callable] = None,
    all_samples: Optional[List[Any]] = None
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Convert existing 3-channel tensor to 5-channel format.
    
    Useful for backwards compatibility when you have a 3-channel tensor
    and want to add surrogate predictions.
    
    Args:
        tensor_3ch: [T, n_vars, 3] tensor in 3-channel format
        variable_order: List of variable names
        target_variable: Target variable name
        surrogate_fn: Optional surrogate prediction function
        all_samples: Optional sample history for recency calculation
        
    Returns:
        Tuple of (tensor_5ch, diagnostics)
    """
    T, n_vars, _ = tensor_3ch.shape
    
    # Get surrogate predictions
    marginal_probs = None
    if surrogate_fn is not None:
        try:
            posterior = surrogate_fn(tensor_3ch, target_variable, variable_order)
            marginal_probs = _extract_marginal_probs(posterior, variable_order, target_variable)
        except Exception as e:
            logger.warning(f"Surrogate prediction failed during conversion: {e}")
    
    # Create recency matrix (simplified if no samples provided)
    if all_samples is not None:
        recency_matrix = _compute_intervention_recency_matrix(
            all_samples, variable_order, T
        )
    else:
        # Simple decay based on intervention indicators
        recency_matrix = _compute_recency_from_tensor(tensor_3ch, variable_order)
    
    # Build 5-channel tensor
    tensor_5ch = jnp.zeros((T, n_vars, 5))
    tensor_5ch = tensor_5ch.at[:, :, :3].set(tensor_3ch)
    
    if marginal_probs is not None:
        prob_vector = jnp.array([marginal_probs.get(var, 0.0) for var in variable_order])
        tensor_5ch = tensor_5ch.at[:, :, 3].set(prob_vector[None, :])
    
    tensor_5ch = tensor_5ch.at[:, :, 4].set(recency_matrix)
    
    diagnostics = {
        'conversion_successful': True,
        'had_surrogate': marginal_probs is not None,
        'had_samples': all_samples is not None
    }
    
    return tensor_5ch, diagnostics


def _compute_recency_from_tensor(tensor_3ch: jnp.ndarray, variable_order: List[str]) -> jnp.ndarray:
    """Compute recency directly from intervention indicators in tensor."""
    T, n_vars = tensor_3ch.shape[:2]
    intervention_channel = tensor_3ch[:, :, 2]
    
    recency_matrix = jnp.ones((T, n_vars))
    
    for var_idx in range(n_vars):
        last_intervention = -1
        
        for t in range(T):
            if intervention_channel[t, var_idx] == 1.0:
                last_intervention = t
            
            if last_intervention >= 0:
                recency = (t - last_intervention) / float(T)
                recency_matrix = recency_matrix.at[t, var_idx].set(min(recency, 1.0))
    
    return recency_matrix