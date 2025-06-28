#!/usr/bin/env python3
"""
PARENT_SCALE Integration Package

This package provides integration between our ACBO framework and the PARENT_SCALE
neural doubly robust causal discovery method for generating expert demonstrations.

Main API:
- run_full_parent_scale_algorithm: Generate expert trajectories 
- run_batch_expert_demonstrations: Collect multiple trajectories
- ensure_parent_scale_imports: Ensure PARENT_SCALE can be imported
"""

# Ensure PARENT_SCALE path is set up before any imports
from .data_processing import setup_parent_scale_path
setup_parent_scale_path()

# Core integration functions
from .algorithm_runner import (
    run_full_parent_scale_algorithm,
    run_parent_discovery,
    run_batch_expert_demonstrations,
    convert_trajectory_to_acbo_format
)

# Extended algorithm with history tracking
from .algorithm_runner_with_history import (
    run_full_parent_scale_algorithm_with_history
)

# Data processing utilities  
from .data_processing import (
    ensure_parent_scale_imports,
    generate_parent_scale_data_with_scm,
    scm_to_graph_structure,
    samples_to_parent_scale_data
)

# Trajectory and validation utilities
from .trajectory_extraction import (
    extract_trajectory_components,
    create_expert_trajectory
)

from .validation import (
    validate_trajectory_completeness,
    validate_algorithm_configuration
)

# Legacy data conversion functions
try:
    from .data_conversion import parent_scale_results_to_posterior
except ImportError:
    def parent_scale_results_to_posterior(*args, **kwargs):
        raise ImportError("Legacy data conversion functions not available")

# Package metadata
__version__ = "2.0.0"
__author__ = "ACBO Team"
__description__ = "PARENT_SCALE integration for expert demonstration collection"

# Main public API
__all__ = [
    # Core functions
    'run_full_parent_scale_algorithm',
    'run_full_parent_scale_algorithm_with_history',
    'run_parent_discovery',
    'run_batch_expert_demonstrations', 
    'convert_trajectory_to_acbo_format',
    
    # Utilities
    'ensure_parent_scale_imports',
    'generate_parent_scale_data_with_scm',
    'scm_to_graph_structure',
    'samples_to_parent_scale_data',
    'parent_scale_results_to_posterior',
    'extract_trajectory_components',
    'create_expert_trajectory',
    'validate_trajectory_completeness',
    'validate_algorithm_configuration'
]