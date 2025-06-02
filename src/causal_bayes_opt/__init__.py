"""
Causal Bayesian Optimization Framework.

This package implements the Amortized Causal Bayesian Optimization (ACBO) framework,
combining AVICI's amortized inference with PARENT_SCALE's causal optimization approach.
"""

__version__ = "0.1.0"

# Import key components for easy access
from . import data_structures
from . import mechanisms
from . import interventions
from . import environments
from . import avici_integration
from . import experiments

# Import commonly used functions and classes
from .data_structures import (
    create_scm,
    create_sample,
    ExperienceBuffer,
    create_empty_buffer,
)

from .mechanisms import (
    create_linear_mechanism,
    sample_from_linear_scm,
)

from .interventions import (
    create_perfect_intervention,
    apply_intervention,
)

from .environments import (
    sample_with_intervention,
    generate_mixed_dataset,
)

__all__ = [
    # Submodules
    'data_structures',
    'mechanisms', 
    'interventions',
    'environments',
    'avici_integration',
    'experiments',
    
    # Core functions and classes
    'create_scm',
    'create_sample',
    'ExperienceBuffer',
    'create_empty_buffer',
    'create_linear_mechanism',
    'sample_from_linear_scm',
    'create_perfect_intervention',
    'apply_intervention',
    'sample_with_intervention',
    'generate_mixed_dataset',
]
