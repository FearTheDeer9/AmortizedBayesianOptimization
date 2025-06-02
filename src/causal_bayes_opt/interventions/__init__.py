"""
Interventions package for causal Bayesian optimization.

This package provides intervention handling using the registry pattern,
supporting extensible intervention types while maintaining functional design.
"""

from .registry import (
    register_intervention_handler,
    get_intervention_handler,
    apply_intervention,
    list_intervention_types,
    is_intervention_type_registered,
    validate_intervention_spec,
    validate_intervention_against_scm,
    get_registry_info,
    InterventionHandler,
    InterventionSpec,
)

from .handlers import (
    perfect_intervention_handler,
    create_perfect_intervention,
    create_single_variable_perfect_intervention,
    validate_perfect_intervention_values,
)

__all__ = [
    # Registry functions
    'register_intervention_handler',
    'get_intervention_handler', 
    'apply_intervention',
    'list_intervention_types',
    'is_intervention_type_registered',
    'validate_intervention_spec',
    'validate_intervention_against_scm',
    'get_registry_info',
    
    # Type aliases
    'InterventionHandler',
    'InterventionSpec',
    
    # Perfect intervention functions
    'perfect_intervention_handler',
    'create_perfect_intervention',
    'create_single_variable_perfect_intervention',
    'validate_perfect_intervention_values',
]
