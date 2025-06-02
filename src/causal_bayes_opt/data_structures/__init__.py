"""
Data structures package for causal Bayesian optimization.

This package provides core immutable data structures for representing
structural causal models, samples, and experience buffers.
"""

from .scm import (
    create_scm,
    get_variables,
    get_edges,
    get_mechanisms,
    get_parents,
    get_children,
    get_ancestors,
    get_descendants,
    is_cyclic,
    topological_sort,
    validate_mechanisms,
    validate_edge_consistency,
)

from .sample import (
    create_sample,
    create_observational_sample,
    create_interventional_sample,
    create_perfect_intervention_sample,
    get_values,
    get_interventional_samples,
    get_observational_samples,
    get_metadata,
    is_observational,
    is_interventional,
)

from .buffer import (
    ExperienceBuffer,
    BufferStatistics,
    create_empty_buffer,
    create_buffer_from_samples,
)

__all__ = [
    # SCM functions
    'create_scm',
    'get_variables',
    'get_edges', 
    'get_mechanisms',
    'get_parents',
    'get_children',
    'get_ancestors',
    'get_descendants',
    'is_cyclic',
    'topological_sort',
    'validate_mechanisms',
    'validate_edge_consistency',
    
    # Sample functions
    'create_sample',
    'create_observational_sample',
    'create_interventional_sample',
    'create_perfect_intervention_sample',
    'get_values',
    'get_interventional_samples',
    'get_observational_samples',
    'get_metadata',
    'is_observational',
    'is_interventional',
    
    # Buffer classes and functions
    'ExperienceBuffer',
    'BufferStatistics',
    'create_empty_buffer',
    'create_buffer_from_samples',
]
