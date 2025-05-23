"""
Mechanisms package for causal Bayesian optimization.

This package provides mechanism factories for creating functions that define
causal relationships between variables in structural causal models.

Note: SCM factory functions have been moved to experiments.test_scms for better organization.
"""

from .linear import (
    create_linear_mechanism,
    create_root_mechanism,
    sample_from_linear_scm
)

__all__ = [
    'create_linear_mechanism',
    'create_root_mechanism', 
    'sample_from_linear_scm'
]
