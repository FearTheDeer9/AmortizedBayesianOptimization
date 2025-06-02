"""
Intervention registry for polymorphic intervention handling.

Uses function registry pattern to support extensible intervention types
while maintaining pure functional design.
"""

# Standard library imports
import logging
from typing import Dict, Callable, Any, List

# Third-party imports
import pyrsistent as pyr

# Local imports
from ..data_structures.scm import get_variables, get_edges, get_mechanisms

# Type aliases
InterventionHandler = Callable[[pyr.PMap, pyr.PMap], pyr.PMap]  # (scm, intervention) -> modified_scm
InterventionSpec = pyr.PMap  # Immutable intervention specification

logger = logging.getLogger(__name__)

# Global registry - stores intervention type -> handler function mappings
_INTERVENTION_HANDLERS: Dict[str, InterventionHandler] = {}


def register_intervention_handler(intervention_type: str, handler: InterventionHandler) -> None:
    """
    Register a handler function for an intervention type.
    
    Args:
        intervention_type: String identifier for the intervention type (e.g., 'perfect', 'soft')
        handler: Function that takes (scm, intervention_spec) and returns modified SCM
        
    Raises:
        ValueError: If intervention_type is invalid or handler is not callable
        
    Example:
        >>> def my_handler(scm, intervention):
        ...     return apply_my_intervention_logic(scm, intervention)
        >>> register_intervention_handler('my_intervention', my_handler)
    """
    if not isinstance(intervention_type, str) or not intervention_type.strip():
        raise ValueError("Intervention type must be a non-empty string")
    
    if not callable(handler):
        raise ValueError("Handler must be callable")
    
    intervention_type = intervention_type.strip()
    
    if intervention_type in _INTERVENTION_HANDLERS:
        logger.warning(f"Overriding existing handler for intervention type '{intervention_type}'")
    
    _INTERVENTION_HANDLERS[intervention_type] = handler
    logger.debug(f"Registered handler for intervention type '{intervention_type}'")


def get_intervention_handler(intervention_type: str) -> InterventionHandler:
    """
    Get the handler function for an intervention type.
    
    Args:
        intervention_type: String identifier for the intervention type
        
    Returns:
        The registered handler function
        
    Raises:
        ValueError: If no handler is registered for the intervention type
        
    Example:
        >>> handler = get_intervention_handler('perfect')
        >>> modified_scm = handler(scm, intervention_spec)
    """
    if not isinstance(intervention_type, str):
        raise ValueError("Intervention type must be a string")
    
    intervention_type = intervention_type.strip()
    
    if intervention_type not in _INTERVENTION_HANDLERS:
        available_types = sorted(_INTERVENTION_HANDLERS.keys())
        raise ValueError(
            f"No handler registered for intervention type '{intervention_type}'. "
            f"Available types: {available_types}"
        )
    
    return _INTERVENTION_HANDLERS[intervention_type]


def apply_intervention(scm: pyr.PMap, intervention: InterventionSpec) -> pyr.PMap:
    """
    Apply an intervention to an SCM using the registered handler.
    
    This is the main entry point for applying interventions. It looks up the
    appropriate handler based on the intervention type and delegates to that handler.
    
    Args:
        scm: Original structural causal model
        intervention: Intervention specification (must include 'type' field)
        
    Returns:
        Modified SCM with the intervention applied
        
    Raises:
        ValueError: If intervention is invalid or no handler exists for the type
        
    Example:
        >>> intervention = create_perfect_intervention(['X'], {'X': 1.0})
        >>> modified_scm = apply_intervention(scm, intervention)
    """
    # Validate inputs
    if not isinstance(scm, pyr.PMap):
        raise ValueError("SCM must be a pyrsistent PMap")
    
    if not isinstance(intervention, pyr.PMap):
        raise ValueError("Intervention must be a pyrsistent PMap")
    
    # Extract intervention type
    if 'type' not in intervention:
        raise ValueError("Intervention specification must include 'type' field")
    
    intervention_type = intervention['type']
    
    # Get and apply handler
    handler = get_intervention_handler(intervention_type)
    
    try:
        modified_scm = handler(scm, intervention)
        logger.debug(f"Applied intervention of type '{intervention_type}'")
        return modified_scm
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to apply intervention of type '{intervention_type}': {e}"
        ) from e


def list_intervention_types() -> List[str]:
    """
    List all registered intervention types.
    
    Returns:
        Sorted list of intervention type names
        
    Example:
        >>> types = list_intervention_types()
        >>> print(f"Available interventions: {types}")
    """
    return sorted(_INTERVENTION_HANDLERS.keys())


def is_intervention_type_registered(intervention_type: str) -> bool:
    """
    Check if an intervention type has a registered handler.
    
    Args:
        intervention_type: String identifier for the intervention type
        
    Returns:
        True if a handler is registered, False otherwise
    """
    return intervention_type in _INTERVENTION_HANDLERS


def unregister_intervention_handler(intervention_type: str) -> bool:
    """
    Remove a handler from the registry.
    
    Args:
        intervention_type: String identifier for the intervention type
        
    Returns:
        True if handler was removed, False if it wasn't registered
        
    Note:
        This is mainly useful for testing. In production, handlers should
        typically remain registered for the lifetime of the program.
    """
    if intervention_type in _INTERVENTION_HANDLERS:
        del _INTERVENTION_HANDLERS[intervention_type]
        logger.debug(f"Unregistered handler for intervention type '{intervention_type}'")
        return True
    return False


def clear_intervention_registry() -> None:
    """
    Clear all registered intervention handlers.
    
    Note:
        This is mainly useful for testing. Use with caution in production.
    """
    _INTERVENTION_HANDLERS.clear()
    logger.debug("Cleared all intervention handlers from registry")


# Validation functions
def validate_intervention_spec(intervention: InterventionSpec) -> bool:
    """
    Validate that an intervention specification is well-formed.
    
    Args:
        intervention: Intervention specification to validate
        
    Returns:
        True if the intervention is valid, False otherwise
        
    Note:
        This performs basic structural validation. Type-specific validation
        should be performed by the individual handlers.
    """
    try:
        # Must be a PMap
        if not isinstance(intervention, pyr.PMap):
            logger.error("Intervention must be a pyrsistent PMap")
            return False
        
        # Must have a type field
        if 'type' not in intervention:
            logger.error("Intervention must have a 'type' field")
            return False
        
        intervention_type = intervention['type']
        
        # Type must be a string
        if not isinstance(intervention_type, str):
            logger.error("Intervention type must be a string")
            return False
        
        # Type must be registered
        if not is_intervention_type_registered(intervention_type):
            logger.error(f"Intervention type '{intervention_type}' is not registered")
            return False
        
        # Must have targets field (list/set of variable names)
        if 'targets' not in intervention:
            logger.error("Intervention must have a 'targets' field")
            return False
        
        targets = intervention['targets']
        if not isinstance(targets, (frozenset, set, list, tuple)):
            logger.error("Intervention targets must be a collection of variable names")
            return False
        
        # All targets must be strings
        if not all(isinstance(target, str) for target in targets):
            logger.error("All intervention targets must be strings")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating intervention: {e}")
        return False


def validate_intervention_against_scm(scm: pyr.PMap, intervention: InterventionSpec) -> bool:
    """
    Validate that an intervention is compatible with a specific SCM.
    
    Args:
        scm: The structural causal model
        intervention: Intervention specification
        
    Returns:
        True if the intervention is compatible with the SCM, False otherwise
    """
    try:
        # Basic intervention validation
        if not validate_intervention_spec(intervention):
            return False
        
        # Check that all targets exist in the SCM
        scm_variables = get_variables(scm)
        targets = frozenset(intervention['targets'])
        
        missing_targets = targets - scm_variables
        if missing_targets:
            logger.error(f"Intervention targets not in SCM: {sorted(missing_targets)}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating intervention against SCM: {e}")
        return False


# Utility functions for debugging and introspection
def get_registry_info() -> Dict[str, Any]:
    """
    Get information about the current state of the intervention registry.
    
    Returns:
        Dictionary with registry information
    """
    return {
        'registered_types': list_intervention_types(),
        'num_handlers': len(_INTERVENTION_HANDLERS),
        'handlers': {
            intervention_type: {
                'function_name': handler.__name__,
                'module': getattr(handler, '__module__', 'unknown')
            }
            for intervention_type, handler in _INTERVENTION_HANDLERS.items()
        }
    }


def print_registry_status() -> None:
    """Print a human-readable summary of the intervention registry."""
    info = get_registry_info()
    print(f"Intervention Registry Status:")
    print(f"  Registered types: {info['num_handlers']}")
    
    if info['registered_types']:
        print(f"  Available interventions:")
        for intervention_type in info['registered_types']:
            handler_info = info['handlers'][intervention_type]
            print(f"    - {intervention_type}: {handler_info['function_name']}")
    else:
        print(f"  No intervention handlers registered")
