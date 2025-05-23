"""
Experiments package for causal Bayesian optimization.

This package provides utilities for creating test SCMs, validation functions,
and experimental configurations for testing and development.
"""

from .test_scms import (
    create_simple_linear_scm,
    create_simple_test_scm,
    create_chain_test_scm
)

__all__ = [
    'create_simple_linear_scm',
    'create_simple_test_scm', 
    'create_chain_test_scm'
]
