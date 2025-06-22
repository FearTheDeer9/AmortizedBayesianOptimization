"""
JAX-Native Causal Bayesian Optimization

This module provides a pure JAX implementation of causal Bayesian optimization
with no dictionary operations, Python loops, or other JAX compilation blockers
in the hot paths.

Key components:
- JAXConfig: Static configuration with fixed tensor shapes
- JAXSampleBuffer: Circular buffer with pure tensor operations
- JAXAcquisitionState: Immutable state with JAX-compiled computations
- Pure JAX operations throughout the pipeline

Design Principles:
- Immutable data structures following functional programming principles
- Static tensor shapes for optimal JAX compilation
- Pure functions with no side effects
- Type safety with comprehensive hints
- Single responsibility for each component
"""

from .config import JAXConfig, create_jax_config, validate_jax_config
from .sample_buffer import JAXSampleBuffer, create_empty_jax_buffer, add_sample_jax
from .state import JAXAcquisitionState, create_jax_state, update_jax_state
from .operations import (
    compute_mechanism_confidence_jax,
    compute_optimization_progress_jax,
    compute_exploration_coverage_jax,
    compute_policy_features_jax
)

__all__ = [
    # Configuration
    "JAXConfig",
    "create_jax_config", 
    "validate_jax_config",
    
    # Sample Buffer
    "JAXSampleBuffer",
    "create_empty_jax_buffer",
    "add_sample_jax",
    
    # State Management
    "JAXAcquisitionState",
    "create_jax_state",
    "update_jax_state",
    
    # JAX Operations
    "compute_mechanism_confidence_jax",
    "compute_optimization_progress_jax", 
    "compute_exploration_coverage_jax",
    "compute_policy_features_jax"
]