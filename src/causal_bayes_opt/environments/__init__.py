"""
Environments package for causal Bayesian optimization.

This package provides sampling functionality for SCMs under interventions,
building on the core sampling capabilities.
"""

from .sampling import (
    sample_with_intervention,
    sample_multiple_interventions,
    generate_mixed_dataset,
    generate_intervention_batch,
    generate_random_interventions,
    create_intervention_grid,
    create_factorial_interventions,
    sample_do_intervention,
    compare_intervention_effects,
)

__all__ = [
    # Core intervention sampling
    'sample_with_intervention',
    'sample_multiple_interventions',
    'generate_mixed_dataset',
    
    # Batch utilities
    'generate_intervention_batch',
    'generate_random_interventions',
    
    # Intervention design utilities
    'create_intervention_grid',
    'create_factorial_interventions',
    
    # Convenience functions
    'sample_do_intervention',
    'compare_intervention_effects',
]
