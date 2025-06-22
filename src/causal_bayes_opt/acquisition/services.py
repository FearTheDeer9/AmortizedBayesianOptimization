"""
Services for creating and updating AcquisitionState objects.

This module contains factory functions and integration logic that coordinate
between the core AcquisitionState data structure and external systems like
surrogate models, AVICI integration, and intervention handling.

These services handle the complex integration logic while keeping the core
state data structure clean and dependency-free.
"""

# Standard library imports
import logging
import time
from typing import Dict, List, Optional, Any, Tuple

# Third-party imports
import jax.numpy as jnp
import pyrsistent as pyr

# Local imports
from .state import AcquisitionState
from ..data_structures.buffer import ExperienceBuffer
from ..data_structures.sample import get_values, get_intervention_targets
from ..avici_integration.parent_set import (
    ParentSetPosterior,
    predict_parent_posterior,  # Now JAX-optimized internally (10-100x faster)
    create_parent_set_posterior,
    create_jax_optimized_model,  # For creating optimized models
    benchmark_model_performance  # For performance validation
)
from ..avici_integration.core import create_training_batch_validated

# Type aliases
InterventionSpec = pyr.PMap  # From interventions framework
Sample = pyr.PMap  # From data_structures framework

logger = logging.getLogger(__name__)


def create_acquisition_state(
    scm: pyr.PMap,
    buffer: ExperienceBuffer,
    surrogate_model: Any,
    surrogate_params: Any,
    target_variable: str,
    step: int = 0,
    metadata: Optional[Dict[str, Any]] = None
) -> AcquisitionState:
    """
    Create acquisition state from current buffer and surrogate model predictions.
    
    This factory function integrates the buffer data with surrogate model predictions
    to create a complete state representation for the acquisition policy.
    
    Args:
        scm: Structural causal model (for context)
        buffer: Current experience buffer with all data
        surrogate_model: Trained surrogate model for posterior prediction
        surrogate_params: Current parameters of the surrogate model
        target_variable: Name of the optimization target variable
        step: Current step number in acquisition process
        metadata: Optional additional context information
        
    Returns:
        Complete AcquisitionState ready for policy input
        
    Raises:
        ValueError: If inputs are inconsistent or insufficient data
        
    Example:
        >>> state = create_acquisition_state(
        ...     scm, buffer, surrogate_model, params, target='Y', step=42
        ... )
        >>> print(f"Uncertainty: {state.uncertainty_bits:.2f} bits")
    """
    # Validate inputs
    if not target_variable:
        raise ValueError("Target variable cannot be empty")
    
    if step < 0:
        raise ValueError("Step must be non-negative")
    
    if buffer.size() == 0:
        raise ValueError("Buffer must contain at least one sample")
    
    # Check target variable is in buffer
    buffer_vars = buffer.get_variable_coverage()
    if target_variable not in buffer_vars:
        raise ValueError(f"Target '{target_variable}' not in buffer variables: {buffer_vars}")
    
    # Get current best value for target variable
    all_samples = buffer.get_all_samples()
    target_values = []
    
    for sample in all_samples:
        values = get_values(sample)
        if target_variable in values:
            target_values.append(float(values[target_variable]))
    
    if not target_values:
        raise ValueError(f"No samples contain target variable '{target_variable}'")
    
    best_value = float(jnp.max(jnp.array(target_values)))
    
    # Get posterior prediction from surrogate model
    try:
        # Create prediction batch from buffer
        batch = create_training_batch_validated(scm, all_samples, target_variable)
        
        # Predict posterior using surrogate model (JAX-optimized)
        posterior = predict_parent_posterior(
            surrogate_model, surrogate_params,
            batch['x'], batch['variable_order'], target_variable
        )
        
    except (ImportError, KeyError, AttributeError) as e:
        logger.error(f"Failed to use surrogate model: {e}")
        # Fallback: create a uniform posterior for testing
        logger.warning("Using uniform posterior fallback for testing")
        
        # Create a simple uniform posterior over empty set and single parents
        parent_candidates = [v for v in buffer_vars if v != target_variable]
        parent_sets = [frozenset()]  # Empty parent set
        parent_sets.extend([frozenset([var]) for var in parent_candidates[:3]])  # Top 3 single parents
        
        n_sets = len(parent_sets)
        uniform_probs = jnp.ones(n_sets) / n_sets
        
        posterior = create_parent_set_posterior(
            target_variable=target_variable,
            parent_sets=parent_sets,
            probabilities=uniform_probs,
            metadata={'fallback_uniform': True, 'step': step}
        )
    
    # Create metadata
    if metadata is None:
        metadata = {}
    
    state_metadata = pyr.pmap({
        **metadata,
        'creation_time': time.time(),
        'scm_variables': set(scm.get('variables', [])) if scm else set(),
        'buffer_creation_time': buffer.get_statistics().creation_time,
        'surrogate_model_type': type(surrogate_model).__name__ if surrogate_model else 'None'
    })
    
    return AcquisitionState(
        posterior=posterior,
        buffer=buffer,
        best_value=best_value,
        current_target=target_variable,
        step=step,
        metadata=state_metadata
    )


def update_state_with_intervention(
    current_state: AcquisitionState,
    intervention: InterventionSpec,
    outcome: Sample,
    new_posterior: ParentSetPosterior
) -> AcquisitionState:
    """
    Create new state after applying an intervention and observing outcome.
    
    This function efficiently updates the acquisition state with new information
    while maintaining immutability principles.
    
    Args:
        current_state: Current acquisition state
        intervention: Applied intervention specification
        outcome: Observed outcome sample from the intervention
        new_posterior: Updated posterior after incorporating new data
        
    Returns:
        New AcquisitionState with updated information
        
    Raises:
        ValueError: If intervention/outcome are inconsistent with current state
        
    Example:
        >>> new_state = update_state_with_intervention(
        ...     state, intervention, outcome, updated_posterior
        ... )
        >>> print(f"Step {new_state.step}: uncertainty reduced by "
        ...       f"{state.uncertainty_bits - new_state.uncertainty_bits:.2f} bits")
    """
    # Validate inputs
    if new_posterior.target_variable != current_state.current_target:
        raise ValueError(
            f"Posterior target '{new_posterior.target_variable}' doesn't match "
            f"state target '{current_state.current_target}'"
        )
    
    # Create updated buffer (this creates a copy for safety)
    new_buffer = ExperienceBuffer()
    
    # Copy existing data
    for obs in current_state.buffer.get_observations():
        new_buffer.add_observation(obs)
    
    for int_spec, int_outcome in current_state.buffer.get_interventions():
        new_buffer.add_intervention(int_spec, int_outcome)
    
    # Add new intervention-outcome pair
    new_buffer.add_intervention(intervention, outcome)
    
    # Update best value
    outcome_values = get_values(outcome)
    current_target_value = outcome_values.get(current_state.current_target)
    
    if current_target_value is not None:
        new_best_value = max(current_state.best_value, float(current_target_value))
    else:
        new_best_value = current_state.best_value
        logger.warning(f"Outcome doesn't contain target variable '{current_state.current_target}'")
    
    # Update metadata
    new_metadata = current_state.metadata.set('last_intervention_step', current_state.step)
    new_metadata = new_metadata.set('last_update_time', time.time())
    
    # Add intervention analysis to metadata
    try:
        intervention_analysis = {
            'intervention_type': intervention.get('type', 'unknown'),
            'intervention_targets': list(intervention.get('targets', set())),
            'outcome_target_value': current_target_value
        }
        new_metadata = new_metadata.set('last_intervention_analysis', intervention_analysis)
    except Exception as e:
        logger.warning(f"Could not analyze intervention: {e}")
    
    # Create new state
    return AcquisitionState(
        posterior=new_posterior,
        buffer=new_buffer,
        best_value=new_best_value,
        current_target=current_state.current_target,
        step=current_state.step + 1,  # Increment step
        metadata=new_metadata
    )


def create_acquisition_state_from_samples(
    samples: list[Sample],
    surrogate_model: Any,
    surrogate_params: Any,
    target_variable: str,
    variable_order: list[str],
    step: int = 0,
    metadata: Optional[Dict[str, Any]] = None
) -> AcquisitionState:
    """
    Create acquisition state directly from a list of samples.
    
    This is a convenience function for cases where you have samples but no
    existing buffer or SCM structure.
    
    Args:
        samples: List of Sample objects (observations + interventions)
        surrogate_model: Trained surrogate model for posterior prediction
        surrogate_params: Current parameters of the surrogate model
        target_variable: Name of the optimization target variable
        variable_order: Ordered list of variable names
        step: Current step number in acquisition process
        metadata: Optional additional context information
        
    Returns:
        Complete AcquisitionState ready for policy input
    """
    # Create buffer from samples
    buffer = ExperienceBuffer()
    
    for sample in samples:
        intervention_targets = get_intervention_targets(sample)
        if intervention_targets:
            # This is an interventional sample - need to reconstruct intervention spec
            # For simplicity, create a basic perfect intervention spec
            intervention_spec = pyr.pmap({
                'type': 'perfect',
                'targets': intervention_targets,
                'values': {var: get_values(sample)[var] for var in intervention_targets}
            })
            buffer.add_intervention(intervention_spec, sample)
        else:
            # This is an observational sample
            buffer.add_observation(sample)
    
    # Create a minimal SCM for context
    scm = pyr.pmap({
        'variables': frozenset(variable_order),
        'target': target_variable
    })
    
    return create_acquisition_state(
        scm=scm,
        buffer=buffer,
        surrogate_model=surrogate_model,
        surrogate_params=surrogate_params,
        target_variable=target_variable,
        step=step,
        metadata=metadata
    )


def validate_state_inputs(
    scm: pyr.PMap,
    buffer: ExperienceBuffer,
    target_variable: str,
    step: int
) -> None:
    """
    Validate inputs for state creation.
    
    Args:
        scm: Structural causal model
        buffer: Experience buffer
        target_variable: Target variable name
        step: Step number
        
    Raises:
        ValueError: If any inputs are invalid
    """
    if not isinstance(scm, pyr.PMap):
        raise ValueError("SCM must be a pyrsistent PMap")
    
    if not isinstance(buffer, ExperienceBuffer):
        raise ValueError("Buffer must be an ExperienceBuffer instance")
    
    if not target_variable or not isinstance(target_variable, str):
        raise ValueError("Target variable must be a non-empty string")
    
    if step < 0:
        raise ValueError("Step must be non-negative")
    
    if buffer.size() == 0:
        raise ValueError("Buffer must contain at least one sample")
    
    # Check target is in SCM variables if specified
    scm_vars = scm.get('variables')
    if scm_vars and target_variable not in scm_vars:
        raise ValueError(f"Target '{target_variable}' not in SCM variables: {scm_vars}")
    
    # Check target is in buffer variables
    buffer_vars = buffer.get_variable_coverage()
    if target_variable not in buffer_vars:
        raise ValueError(f"Target '{target_variable}' not in buffer variables: {buffer_vars}")


def compute_state_delta(
    old_state: AcquisitionState,
    new_state: AcquisitionState
) -> Dict[str, Any]:
    """
    Compute the differences between two acquisition states.
    
    This is useful for analyzing how states change after interventions.
    
    Args:
        old_state: Previous acquisition state
        new_state: Current acquisition state
        
    Returns:
        Dictionary with state change metrics
    """
    return {
        'step_delta': new_state.step - old_state.step,
        'uncertainty_delta_bits': new_state.uncertainty_bits - old_state.uncertainty_bits,
        'best_value_delta': new_state.best_value - old_state.best_value,
        'sample_count_delta': (
            new_state.buffer_statistics.total_samples - 
            old_state.buffer_statistics.total_samples
        ),
        'intervention_count_delta': (
            new_state.buffer_statistics.num_interventions - 
            old_state.buffer_statistics.num_interventions
        ),
        'target_changed': old_state.current_target != new_state.current_target,
        'improvement_achieved': new_state.best_value > old_state.best_value,
        'uncertainty_reduced': new_state.uncertainty_bits < old_state.uncertainty_bits
    }


def create_jax_optimized_surrogate_model(variable_names: list[str],
                                       predict_mechanisms: bool = False,
                                       **config_kwargs):
    """
    Create JAX-optimized surrogate model for acquisition state creation.
    
    ⚠️ MIGRATION UPDATE: This creates high-performance JAX models that provide
    10-100x speedup in parent set prediction for acquisition states.
    
    Args:
        variable_names: List of variable names in the SCM
        predict_mechanisms: Whether to enable mechanism prediction
        **config_kwargs: Additional configuration options
        
    Returns:
        JAX-optimized model ready for use in create_acquisition_state()
        
    Example:
        >>> model = create_jax_optimized_surrogate_model(['X', 'Y', 'Z'])
        >>> model.init(key, sample_data, target_variable='Y')
        >>> state = create_acquisition_state(scm, buffer, model, None, 'Y')
    """
    try:
        model = create_jax_optimized_model(
            variable_names=variable_names,
            predict_mechanisms=predict_mechanisms,
            **config_kwargs
        )
        logger.info(f"Created JAX-optimized surrogate model for acquisition with {len(variable_names)} variables")
        return model
    except ImportError as e:
        logger.error(f"Failed to create JAX-optimized model: {e}")
        logger.info("Falling back to standard model creation")
        # Could fallback to standard model here if needed
        raise


def benchmark_surrogate_performance(surrogate_model, surrogate_params,
                                  test_data, variable_order, target_variable,
                                  n_runs: int = 10) -> Dict[str, Any]:
    """
    Benchmark surrogate model performance for acquisition state creation.
    
    This helps validate that JAX optimizations are working correctly and
    providing expected performance improvements.
    
    Args:
        surrogate_model: Model to benchmark
        surrogate_params: Model parameters
        test_data: Test data for benchmarking
        variable_order: Variable order
        target_variable: Target variable
        n_runs: Number of benchmark runs
        
    Returns:
        Performance metrics and optimization status
    """
    try:
        return benchmark_model_performance(
            surrogate_model, surrogate_params, test_data, 
            variable_order, target_variable, n_runs
        )
    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
        return {
            'error': str(e),
            'jax_optimized': False,
            'mean_time_ms': float('inf')
        }


def create_acquisition_state_with_mechanisms(
    scm: pyr.PMap,
    buffer: ExperienceBuffer,
    surrogate_model: Any,
    surrogate_params: Any,
    target_variable: str,
    step: int = 0,
    extract_mechanisms: bool = True,
    metadata: Optional[Dict[str, Any]] = None
) -> AcquisitionState:
    """
    Create acquisition state with mechanism predictions from JAX unified model.
    
    Architecture Enhancement Pivot - Part C: Integration & Testing
    
    This enhanced version extracts mechanism predictions from JAX unified models
    when available, providing mechanism-aware intervention selection capabilities.
    
    Args:
        scm: Structural causal model (for context)
        buffer: Current experience buffer with all data
        surrogate_model: JAX unified surrogate model with mechanism prediction
        surrogate_params: Current parameters of the surrogate model
        target_variable: Name of the optimization target variable
        step: Current step number in acquisition process
        extract_mechanisms: Whether to extract mechanism predictions (if available)
        metadata: Optional additional context information
        
    Returns:
        AcquisitionState with mechanism predictions if model supports them
        
    Raises:
        ValueError: If inputs are inconsistent or insufficient data
    """
    # Validate inputs same as base function
    if not target_variable:
        raise ValueError("Target variable cannot be empty")
    
    if step < 0:
        raise ValueError("Step must be non-negative")
    
    if buffer.size() == 0:
        raise ValueError("Buffer must contain at least one sample")
    
    # Check target variable is in buffer
    buffer_vars = buffer.get_variable_coverage()
    if target_variable not in buffer_vars:
        raise ValueError(f"Target '{target_variable}' not in buffer variables: {buffer_vars}")
    
    # Get current best value for target variable
    all_samples = buffer.get_all_samples()
    target_values = []
    
    for sample in all_samples:
        values = get_values(sample)
        if target_variable in values:
            target_values.append(float(values[target_variable]))
    
    if not target_values:
        raise ValueError(f"No samples contain target variable '{target_variable}'")
    
    best_value = float(jnp.max(jnp.array(target_values)))
    
    # Initialize mechanism predictions variables
    mechanism_predictions = None
    mechanism_uncertainties = None
    
    # Get posterior and mechanism predictions from surrogate model
    try:
        # Create prediction batch from buffer
        batch = create_training_batch_validated(scm, all_samples, target_variable)
        
        # Check if this is a JAX unified model with mechanism prediction capabilities
        is_jax_unified = hasattr(surrogate_model, 'predict_mechanisms') or (
            hasattr(surrogate_model, 'config') and 
            getattr(surrogate_model.config, 'predict_mechanisms', False)
        )
        
        if extract_mechanisms and is_jax_unified:
            try:
                # Extract mechanism predictions from JAX unified model
                logger.debug("Extracting mechanism predictions from JAX unified model")
                
                # Call model directly to get full outputs including mechanisms
                if hasattr(surrogate_model, 'apply'):
                    # Direct JAX model call
                    full_outputs = surrogate_model.apply(
                        surrogate_params, 
                        batch['x'], 
                        target_variable,  # Will be converted to index internally
                        False  # is_training=False
                    )
                    
                    # Extract mechanism predictions if present
                    if isinstance(full_outputs, dict):
                        if 'mechanism_predictions' in full_outputs:
                            mechanism_predictions = full_outputs['mechanism_predictions']
                            logger.debug("Successfully extracted mechanism predictions")
                        
                        if 'mechanism_uncertainties' in full_outputs:
                            mechanism_uncertainties = full_outputs['mechanism_uncertainties']
                            logger.debug("Successfully extracted mechanism uncertainties")
                        
                        # Convert mechanism uncertainties to proper format if needed
                        if isinstance(mechanism_uncertainties, dict):
                            # Already in correct format
                            pass
                        elif hasattr(mechanism_uncertainties, 'shape'):
                            # Convert tensor to dict using variable order
                            variable_order = batch['variable_order']
                            mechanism_uncertainties = {
                                var: float(mechanism_uncertainties[i]) 
                                for i, var in enumerate(variable_order)
                                if i < mechanism_uncertainties.shape[0] and var != target_variable
                            }
                
            except Exception as e:
                logger.warning(f"Failed to extract mechanism predictions: {e}")
                # Continue without mechanism predictions
                mechanism_predictions = None
                mechanism_uncertainties = None
        
        # Get standard posterior prediction (works for both unified and regular models)
        posterior = predict_parent_posterior(
            surrogate_model, surrogate_params,
            batch['x'], batch['variable_order'], target_variable
        )
        
    except (ImportError, KeyError, AttributeError) as e:
        logger.error(f"Failed to use surrogate model: {e}")
        # Fallback: create a uniform posterior for testing
        logger.warning("Using uniform posterior fallback for testing")
        
        # Create a simple uniform posterior over empty set and single parents
        parent_candidates = [v for v in buffer_vars if v != target_variable]
        parent_sets = [frozenset()]  # Empty parent set
        parent_sets.extend([frozenset([var]) for var in parent_candidates[:3]])  # Top 3 single parents
        
        n_sets = len(parent_sets)
        uniform_probs = jnp.ones(n_sets) / n_sets
        
        posterior = create_parent_set_posterior(
            target_variable=target_variable,
            parent_sets=parent_sets,
            probabilities=uniform_probs,
            metadata={'fallback_uniform': True, 'step': step}
        )
    
    # Create metadata
    if metadata is None:
        metadata = {}
    
    state_metadata = pyr.pmap({
        **metadata,
        'creation_time': time.time(),
        'scm_variables': set(scm.get('variables', [])) if scm else set(),
        'buffer_size': buffer.size(),
        'mechanism_aware': mechanism_predictions is not None,
        'jax_unified_model': is_jax_unified if 'is_jax_unified' in locals() else False
    })
    
    # Create the enhanced acquisition state
    return AcquisitionState(
        posterior=posterior,
        buffer=buffer,
        best_value=best_value,
        current_target=target_variable,
        step=step,
        metadata=state_metadata,
        # Architecture Enhancement Pivot - Part C: mechanism-aware features
        mechanism_predictions=mechanism_predictions,
        mechanism_uncertainties=mechanism_uncertainties
    )


def create_jax_optimized_surrogate_model(
    config: Dict[str, Any],
    variable_names: List[str],
    predict_mechanisms: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    Create JAX-optimized surrogate model with optional mechanism prediction.
    
    Architecture Enhancement Pivot - Part C: Integration & Testing
    
    Args:
        config: Model configuration dictionary
        variable_names: List of variable names for the model
        predict_mechanisms: Whether to enable mechanism prediction capabilities
        
    Returns:
        Tuple of (model, lookup_tables) for JAX-compatible usage
    """
    try:
        # Use the JAX-optimized model creation function
        model, lookup_tables = create_jax_optimized_model(
            variable_names=variable_names,
            predict_mechanisms=predict_mechanisms,
            **config
        )
        
        logger.info(f"Created JAX-optimized surrogate model with mechanisms={predict_mechanisms}")
        return model, lookup_tables
        
    except Exception as e:
        logger.error(f"Failed to create JAX-optimized surrogate model: {e}")
        raise ValueError(f"Could not create JAX-optimized model: {e}")


def benchmark_surrogate_performance(
    surrogate_model: Any,
    surrogate_params: Any,
    test_samples: List[Any],
    target_variable: str,
    variable_order: List[str],
    n_runs: int = 10
) -> Dict[str, Any]:
    """
    Benchmark surrogate model performance for validation.
    
    Architecture Enhancement Pivot - Part C: Performance validation
    
    Args:
        surrogate_model: Model to benchmark
        surrogate_params: Model parameters  
        test_samples: Test samples for benchmarking
        target_variable: Target variable name
        variable_order: Variable order
        n_runs: Number of benchmark runs
        
    Returns:
        Performance metrics including mechanism prediction timing
    """
    try:
        # Convert samples to AVICI format for testing
        from ..avici_integration.core import samples_to_avici_format
        test_data = samples_to_avici_format(test_samples, variable_order, target_variable)
        
        return benchmark_surrogate_performance_helper(
            surrogate_model, surrogate_params, test_data,
            variable_order, target_variable, n_runs
        )
    except Exception as e:
        logger.error(f"Surrogate benchmarking failed: {e}")
        return {
            'error': str(e),
            'jax_optimized': False,
            'mean_time_ms': float('inf'),
            'mechanism_aware': False
        }