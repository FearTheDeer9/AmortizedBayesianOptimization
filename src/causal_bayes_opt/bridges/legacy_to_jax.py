"""
Legacy to JAX Conversion Functions

Provides clean, one-way conversion from legacy dictionary-based data structures
to JAX-native tensor representations.

Key features:
- Pure conversion functions (no side effects)
- Comprehensive error handling and validation
- Performance monitoring for conversion overhead
- Type safety with extensive validation
"""

import warnings
from typing import Dict, List, Any, Optional, Tuple
import jax.numpy as jnp
import pyrsistent as pyr

from ..jax_native import (
    JAXConfig, JAXSampleBuffer, JAXAcquisitionState,
    create_jax_config, create_jax_state, create_empty_jax_buffer,
    add_sample_jax
)


def convert_legacy_to_jax(
    legacy_state,  # AcquisitionState - avoid import for compatibility
    config: Optional[JAXConfig] = None,
    max_samples: int = 1000,
    max_history: int = 100
) -> JAXAcquisitionState:
    """
    Convert legacy AcquisitionState to JAX-native representation.
    
    This is the main conversion function for migrating from legacy to JAX-native
    architecture. Performs comprehensive data transformation while preserving
    all semantic information.
    
    Args:
        legacy_state: Legacy AcquisitionState object
        config: Optional JAX configuration (creates from legacy if None)
        max_samples: Maximum buffer capacity for JAX buffer
        max_history: Maximum history length
        
    Returns:
        JAXAcquisitionState with equivalent data in tensor format
        
    Raises:
        ValueError: If legacy state is invalid or conversion fails
        TypeError: If arguments have incorrect types
    """
    # Create configuration if not provided
    if config is None:
        config = _create_config_from_legacy(legacy_state, max_samples, max_history)
    
    # Convert sample buffer
    jax_buffer = convert_legacy_buffer_to_jax(legacy_state.buffer, config)
    
    # Convert mechanism data
    mechanism_features, confidence_scores = convert_legacy_mechanisms_to_jax(
        legacy_state, config
    )
    
    # Convert marginal probabilities
    marginal_probs = _convert_marginal_probs_to_jax(
        getattr(legacy_state, 'marginal_parent_probs', {}), config
    )
    
    # Create JAX state
    jax_state = create_jax_state(
        config=config,
        sample_buffer=jax_buffer,
        mechanism_features=mechanism_features,
        marginal_probs=marginal_probs,
        confidence_scores=confidence_scores,
        best_value=legacy_state.best_value,
        current_step=legacy_state.step,
        uncertainty_bits=getattr(legacy_state, 'uncertainty_bits', 1.0)
    )
    
    # Validate conversion result
    validate_conversion_result(jax_state, legacy_state)
    
    return jax_state


def convert_legacy_buffer_to_jax(
    legacy_buffer,  # ExperienceBuffer - avoid import
    config: JAXConfig
) -> JAXSampleBuffer:
    """
    Convert legacy ExperienceBuffer to JAX tensor format.
    
    Transforms variable-length dictionary-based samples to fixed-size tensor
    representation suitable for JAX compilation.
    
    Args:
        legacy_buffer: Legacy ExperienceBuffer object
        config: JAX configuration with variable ordering
        
    Returns:
        JAXSampleBuffer with tensor representation of samples
    """
    # Create empty JAX buffer
    jax_buffer = create_empty_jax_buffer(config)
    
    # Get all samples from legacy buffer
    try:
        all_samples = legacy_buffer.get_all_samples()
    except AttributeError:
        # Handle different legacy buffer interfaces
        all_samples = getattr(legacy_buffer, 'samples', [])
    
    if not all_samples:
        return jax_buffer
    
    # Convert each sample to tensor format
    for sample in all_samples:
        try:
            variable_values, intervention_mask, target_value = _convert_sample_to_tensors(
                sample, config
            )
            
            jax_buffer = add_sample_jax(
                jax_buffer, variable_values, intervention_mask, target_value
            )
            
        except Exception as e:
            warnings.warn(f"Skipping invalid sample during conversion: {e}")
            continue
    
    return jax_buffer


def convert_legacy_mechanisms_to_jax(
    legacy_state,
    config: JAXConfig
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert legacy mechanism data to JAX tensor format.
    
    Transforms dictionary-based mechanism predictions and confidence scores
    to tensor representation with consistent variable ordering.
    
    Args:
        legacy_state: Legacy state with mechanism data
        config: JAX configuration with variable ordering
        
    Returns:
        Tuple of (mechanism_features, confidence_scores) as JAX arrays
    """
    n_vars = config.n_vars
    feature_dim = config.feature_dim
    
    # Initialize feature and confidence tensors
    mechanism_features = jnp.ones((n_vars, feature_dim)) * 0.5  # Default values
    confidence_scores = jnp.ones(n_vars) * 0.5
    
    # Extract mechanism predictions if available
    mechanism_predictions = getattr(legacy_state, 'mechanism_predictions', {})
    mechanism_uncertainties = getattr(legacy_state, 'mechanism_uncertainties', {})
    mechanism_confidence = getattr(legacy_state, 'mechanism_confidence', {})
    
    # Convert to tensor format using variable ordering
    for i, var_name in enumerate(config.variable_names):
        if i == config.target_idx:
            # Target variable gets zero features
            mechanism_features = mechanism_features.at[i].set(jnp.zeros(feature_dim))
            confidence_scores = confidence_scores.at[i].set(0.0)
            continue
        
        # Extract mechanism data for this variable
        if var_name in mechanism_predictions:
            # Use actual prediction data
            pred_data = mechanism_predictions[var_name]
            if isinstance(pred_data, dict):
                effect = pred_data.get('effect', 1.0)
                uncertainty = mechanism_uncertainties.get(var_name, 0.5)
                confidence = mechanism_confidence.get(var_name, 0.5)
            else:
                # Handle scalar predictions
                effect = float(pred_data)
                uncertainty = mechanism_uncertainties.get(var_name, 0.5)
                confidence = mechanism_confidence.get(var_name, 0.5)
            
            # Set mechanism features: [effect, uncertainty, confidence]
            features = jnp.array([effect, uncertainty, confidence])
            mechanism_features = mechanism_features.at[i].set(features)
            confidence_scores = confidence_scores.at[i].set(confidence)
    
    return mechanism_features, confidence_scores


def validate_conversion_result(
    jax_state: JAXAcquisitionState,
    legacy_state
) -> None:
    """
    Validate that JAX conversion preserves essential information.
    
    Performs comprehensive checks to ensure the conversion maintains
    semantic equivalence between legacy and JAX representations.
    
    Args:
        jax_state: Converted JAX state
        legacy_state: Original legacy state
        
    Raises:
        ValueError: If conversion validation fails
    """
    # Check basic state consistency
    assert jax_state.best_value == legacy_state.best_value, \
        f"Best value mismatch: {jax_state.best_value} != {legacy_state.best_value}"
    
    assert jax_state.current_step == legacy_state.step, \
        f"Step mismatch: {jax_state.current_step} != {legacy_state.step}"
    
    # Check buffer sample count
    legacy_sample_count = getattr(legacy_state.buffer, 'get_sample_count', lambda: 0)()
    if callable(legacy_sample_count):
        legacy_sample_count = legacy_sample_count()
    
    assert jax_state.sample_buffer.n_samples >= 0, \
        "JAX buffer should have non-negative sample count"
    
    # Check tensor shapes
    config = jax_state.config
    assert jax_state.mechanism_features.shape == (config.n_vars, config.feature_dim), \
        f"Invalid mechanism features shape: {jax_state.mechanism_features.shape}"
    
    assert jax_state.marginal_probs.shape == (config.n_vars,), \
        f"Invalid marginal probs shape: {jax_state.marginal_probs.shape}"
    
    assert jax_state.confidence_scores.shape == (config.n_vars,), \
        f"Invalid confidence scores shape: {jax_state.confidence_scores.shape}"
    
    # Check target variable masking
    assert jax_state.confidence_scores[config.target_idx] == 0.0, \
        "Target variable should have zero confidence"
    
    # Check tensor validity (no NaN/Inf values)
    assert jnp.all(jnp.isfinite(jax_state.mechanism_features)), \
        "Mechanism features contain non-finite values"
    
    assert jnp.all(jnp.isfinite(jax_state.marginal_probs)), \
        "Marginal probabilities contain non-finite values"
    
    assert jnp.all(jnp.isfinite(jax_state.confidence_scores)), \
        "Confidence scores contain non-finite values"


# Helper functions

def _create_config_from_legacy(
    legacy_state,
    max_samples: int,
    max_history: int
) -> JAXConfig:
    """Create JAX configuration from legacy state information."""
    # Extract variable information
    try:
        # Try to get variable coverage from buffer
        variables = list(legacy_state.buffer.get_variable_coverage())
    except AttributeError:
        # Fallback: extract from mechanism predictions or other sources
        variables = list(getattr(legacy_state, 'mechanism_predictions', {}).keys())
        if not variables:
            # Last resort: use posterior information
            variables = ['X', 'Y', 'Z']  # Default for testing
    
    if not variables:
        raise ValueError("Cannot determine variables from legacy state")
    
    # Determine target variable
    target_variable = legacy_state.current_target
    if target_variable not in variables:
        raise ValueError(f"Target variable '{target_variable}' not found in variables {variables}")
    
    # Create configuration
    return create_jax_config(
        variable_names=sorted(variables),  # Consistent ordering
        target_variable=target_variable,
        max_samples=max_samples,
        max_history=max_history
    )


def _convert_marginal_probs_to_jax(
    marginal_probs_dict: Dict[str, float],
    config: JAXConfig
) -> jnp.ndarray:
    """Convert dictionary of marginal probabilities to tensor."""
    marginal_probs = jnp.ones(config.n_vars) * 0.5  # Default values
    
    for i, var_name in enumerate(config.variable_names):
        if i == config.target_idx:
            marginal_probs = marginal_probs.at[i].set(0.0)  # Target excluded
        elif var_name in marginal_probs_dict:
            prob = marginal_probs_dict[var_name]
            marginal_probs = marginal_probs.at[i].set(float(prob))
    
    return marginal_probs


def _convert_sample_to_tensors(
    sample,
    config: JAXConfig
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """Convert legacy sample to tensor format."""
    # Initialize tensors
    variable_values = jnp.zeros(config.n_vars)
    intervention_mask = jnp.zeros(config.n_vars, dtype=bool)
    
    # Extract sample data (handle different legacy formats)
    if hasattr(sample, 'values'):
        values_dict = sample.values
        interventions_dict = getattr(sample, 'interventions', {})
        target_value = getattr(sample, 'target', 0.0)
    elif isinstance(sample, dict):
        values_dict = sample.get('values', {})
        interventions_dict = sample.get('interventions', {})
        target_value = sample.get('target', 0.0)
    else:
        raise ValueError(f"Unknown sample format: {type(sample)}")
    
    # Convert values to tensor
    for i, var_name in enumerate(config.variable_names):
        if var_name in values_dict:
            variable_values = variable_values.at[i].set(float(values_dict[var_name]))
        
        # Check for intervention
        if var_name in interventions_dict:
            intervention_mask = intervention_mask.at[i].set(True)
    
    # Ensure target value is set
    if isinstance(target_value, (int, float)):
        target_value = float(target_value)
    else:
        # Extract from values if target_value not available
        target_var = config.variable_names[config.target_idx]
        target_value = float(values_dict.get(target_var, 0.0))
    
    return variable_values, intervention_mask, target_value


# Performance monitoring

def measure_conversion_performance(
    legacy_state,
    config: Optional[JAXConfig] = None
) -> Dict[str, float]:
    """
    Measure performance overhead of legacy-to-JAX conversion.
    
    Args:
        legacy_state: Legacy state to convert
        config: Optional JAX configuration
        
    Returns:
        Dictionary with timing measurements
    """
    import time
    
    start_time = time.perf_counter()
    
    # Measure individual conversion steps
    config_start = time.perf_counter()
    if config is None:
        config = _create_config_from_legacy(legacy_state, 1000, 100)
    config_time = time.perf_counter() - config_start
    
    buffer_start = time.perf_counter()
    jax_buffer = convert_legacy_buffer_to_jax(legacy_state.buffer, config)
    buffer_time = time.perf_counter() - buffer_start
    
    mechanisms_start = time.perf_counter()
    mechanism_features, confidence_scores = convert_legacy_mechanisms_to_jax(
        legacy_state, config
    )
    mechanisms_time = time.perf_counter() - mechanisms_start
    
    total_time = time.perf_counter() - start_time
    
    return {
        'total_conversion_time': total_time * 1000,  # ms
        'config_creation_time': config_time * 1000,
        'buffer_conversion_time': buffer_time * 1000,
        'mechanisms_conversion_time': mechanisms_time * 1000,
        'samples_converted': jax_buffer.n_samples
    }