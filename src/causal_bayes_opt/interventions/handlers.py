"""
Concrete intervention handlers and factory functions.

Implements specific intervention types (perfect, imperfect, soft)
following the registry pattern.
"""

# Standard library imports
import logging
from typing import Dict, Any, FrozenSet, Callable, Optional

# Third-party imports
import pyrsistent as pyr

# Local imports  
from .registry import register_intervention_handler, InterventionHandler
from ..data_structures.scm import (
    create_scm, get_mechanisms, get_variables, get_edges
)

logger = logging.getLogger(__name__)


# Perfect intervention handler
def perfect_intervention_handler(scm: pyr.PMap, intervention: pyr.PMap) -> pyr.PMap:
    """
    Apply perfect intervention: replace mechanisms with constants.
    
    A perfect intervention completely replaces the mechanism of the target variables
    with deterministic constant values, breaking all causal links from parents.
    
    Args:
        scm: Original SCM
        intervention: Intervention spec with 'targets' and 'values'
        
    Returns:
        Modified SCM with intervened variables
        
    Raises:
        ValueError: If intervention specification is invalid
        
    Example:
        The intervention {'type': 'perfect', 'targets': ['X'], 'values': {'X': 1.0}}
        replaces X's mechanism with a constant function that always returns 1.0
    """
    # Validate intervention specification
    if 'targets' not in intervention:
        raise ValueError("Perfect intervention must specify 'targets'")
    
    if 'values' not in intervention:
        raise ValueError("Perfect intervention must specify 'values'")
    
    targets = frozenset(intervention['targets'])
    values = intervention['values']
    
    # Validate that all targets have specified values
    missing_values = targets - set(values.keys())
    if missing_values:
        raise ValueError(f"Missing values for intervention targets: {sorted(missing_values)}")
    
    # Validate that all targets exist in SCM
    scm_variables = get_variables(scm)
    invalid_targets = targets - scm_variables
    if invalid_targets:
        raise ValueError(f"Intervention targets not in SCM: {sorted(invalid_targets)}")
    
    # Create modified mechanisms
    original_mechanisms = get_mechanisms(scm)
    new_mechanisms = original_mechanisms.copy()
    
    # Replace mechanisms for intervened variables with constant functions
    for target in targets:
        target_value = values[target]
        
        # Create a constant mechanism that ignores parents and noise
        def create_constant_mechanism(value):
            def constant_mechanism(parent_values, noise_key):
                # Ignore all inputs and return the constant value
                return value
            return constant_mechanism
        
        new_mechanisms[target] = create_constant_mechanism(target_value)
        logger.debug(f"Created constant mechanism for '{target}' = {target_value}")
    
    # Create new SCM with modified mechanisms
    modified_scm = create_scm(
        variables=get_variables(scm),
        edges=get_edges(scm),
        mechanisms=new_mechanisms,
        target=scm.get('target'),
        metadata=scm.get('metadata', {})
    )
    
    # Add intervention metadata
    intervention_metadata = {
        'intervention_applied': True,
        'intervention_type': 'perfect',
        'intervention_targets': targets,
        'intervention_values': values
    }
    
    # Merge with existing metadata
    existing_metadata = modified_scm.get('metadata', pyr.m())
    updated_metadata = existing_metadata.update(intervention_metadata)
    modified_scm = modified_scm.set('metadata', updated_metadata)
    
    logger.info(f"Applied perfect intervention on {sorted(targets)}")
    return modified_scm


# Stub implementations for future intervention types
def imperfect_intervention_handler(scm: pyr.PMap, intervention: pyr.PMap) -> pyr.PMap:
    """
    Apply imperfect intervention: add noise to target values.
    
    Note: This is a stub implementation. Not yet implemented.
    
    Args:
        scm: Original SCM
        intervention: Intervention spec with 'targets', 'values', and 'noise_scales'
        
    Returns:
        Modified SCM with imperfect interventions
        
    Raises:
        NotImplementedError: This intervention type is not yet implemented
    """
    raise NotImplementedError("Imperfect interventions not yet implemented")


def soft_intervention_handler(scm: pyr.PMap, intervention: pyr.PMap) -> pyr.PMap:
    """
    Apply soft intervention: modify but don't replace mechanisms.
    
    Note: This is a stub implementation. Not yet implemented.
    
    Args:
        scm: Original SCM
        intervention: Intervention spec with 'targets', 'strength', and 'target_values'
        
    Returns:
        Modified SCM with soft interventions
        
    Raises:
        NotImplementedError: This intervention type is not yet implemented
    """
    raise NotImplementedError("Soft interventions not yet implemented")


# Factory functions for creating intervention specifications
def create_perfect_intervention(
    targets: FrozenSet[str], 
    values: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> pyr.PMap:
    """
    Create a perfect intervention specification.
    
    Args:
        targets: Set of variable names to intervene on
        values: Dictionary mapping target variables to their intervention values
        metadata: Optional additional metadata
        
    Returns:
        Immutable intervention specification
        
    Raises:
        ValueError: If inputs are invalid
        
    Example:
        >>> intervention = create_perfect_intervention(
        ...     targets=frozenset(['X', 'Y']),
        ...     values={'X': 1.0, 'Y': 2.0}
        ... )
    """
    # Validate inputs
    if not isinstance(targets, (frozenset, set)):
        raise ValueError("Targets must be a frozenset or set")
    
    targets = frozenset(targets)  # Ensure it's a frozenset
    
    if not targets:
        raise ValueError("Must specify at least one target variable")
    
    if not isinstance(values, dict):
        raise ValueError("Values must be a dictionary")
    
    # Check that all targets have values
    missing_values = targets - set(values.keys())
    if missing_values:
        raise ValueError(f"Missing values for targets: {sorted(missing_values)}")
    
    # Check that no extra values are provided
    extra_values = set(values.keys()) - targets
    if extra_values:
        raise ValueError(f"Values provided for non-target variables: {sorted(extra_values)}")
    
    # Create intervention specification
    intervention_spec = pyr.m(
        type='perfect',
        targets=targets,
        values=pyr.m(**values),
        metadata=pyr.m(**(metadata or {}))
    )
    
    logger.debug(f"Created perfect intervention spec for targets: {sorted(targets)}")
    return intervention_spec


def create_imperfect_intervention(
    targets: FrozenSet[str],
    values: Dict[str, Any], 
    noise_scales: Dict[str, float],
    metadata: Optional[Dict[str, Any]] = None
) -> pyr.PMap:
    """
    Create an imperfect intervention specification.
    
    Note: This is a stub implementation for future use.
    
    Args:
        targets: Set of variable names to intervene on
        values: Dictionary mapping target variables to their intervention values
        noise_scales: Dictionary mapping target variables to noise scales
        metadata: Optional additional metadata
        
    Returns:
        Immutable intervention specification
        
    Raises:
        NotImplementedError: This intervention type is not yet implemented
    """
    raise NotImplementedError("Imperfect interventions not yet implemented")


def create_soft_intervention(
    targets: FrozenSet[str],
    strength: float,
    target_values: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None  
) -> pyr.PMap:
    """
    Create a soft intervention specification.
    
    Note: This is a stub implementation for future use.
    
    Args:
        targets: Set of variable names to intervene on
        strength: Strength of the soft intervention (0 = no effect, 1 = perfect)
        target_values: Dictionary mapping target variables to their target values
        metadata: Optional additional metadata
        
    Returns:
        Immutable intervention specification
        
    Raises:
        NotImplementedError: This intervention type is not yet implemented
    """
    raise NotImplementedError("Soft interventions not yet implemented")


# Convenience functions
def create_single_variable_perfect_intervention(
    variable: str,
    value: Any,
    metadata: Optional[Dict[str, Any]] = None
) -> pyr.PMap:
    """
    Convenience function to create a perfect intervention on a single variable.
    
    Args:
        variable: Name of the variable to intervene on
        value: Value to set the variable to
        metadata: Optional additional metadata
        
    Returns:
        Immutable intervention specification
        
    Example:
        >>> intervention = create_single_variable_perfect_intervention('X', 1.0)
    """
    return create_perfect_intervention(
        targets=frozenset([variable]),
        values={variable: value},
        metadata=metadata
    )


def validate_perfect_intervention_values(values: Dict[str, Any]) -> bool:
    """
    Validate that perfect intervention values are appropriate.
    
    Args:
        values: Dictionary mapping variables to intervention values
        
    Returns:
        True if values are valid, False otherwise
    """
    try:
        for variable, value in values.items():
            # Variable names must be strings
            if not isinstance(variable, str):
                logger.error(f"Variable name must be string, got: {type(variable)}")
                return False
            
            # Values should be numeric (for most causal models)
            if not isinstance(value, (int, float, complex)):
                logger.warning(f"Non-numeric intervention value for '{variable}': {value}")
                # Don't return False - non-numeric values might be valid in some models
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating intervention values: {e}")
        return False


# Register handlers (called at module import)
def _register_builtin_handlers():
    """Register the built-in intervention handlers."""
    register_intervention_handler("perfect", perfect_intervention_handler)
    # Note: Only registering perfect interventions for now
    # register_intervention_handler("imperfect", imperfect_intervention_handler) 
    # register_intervention_handler("soft", soft_intervention_handler)
    
    logger.debug("Registered built-in intervention handlers")


# Auto-register handlers when module is imported
_register_builtin_handlers()
