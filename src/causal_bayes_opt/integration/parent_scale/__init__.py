#!/usr/bin/env python3
"""
PARENT_SCALE Integration Package (Refactored)

This package provides integration between our ACBO framework and the PARENT_SCALE
neural doubly robust causal discovery method. It has been refactored into focused
modules following functional programming principles and clean architecture.

The package is organized into focused modules:
- data_generation: Pure functions for PARENT_SCALE data generation
- trajectory_extraction: Extract expert demonstrations from algorithm results
- validation: Validate trajectories and algorithm outputs
- algorithm_runner: Main API functions for algorithm execution
- data_conversion: Legacy format conversion functions
- bridge: High-level coordination and API (legacy compatibility)
"""

# Import main API functions from refactored algorithm_runner
from .algorithm_runner import (
    # Main algorithm execution functions
    run_full_parent_scale_algorithm,
    run_parent_discovery,
    run_batch_expert_demonstrations,
    convert_trajectory_to_acbo_format
)

# Import data generation utilities  
from .data_generation import (
    check_parent_scale_availability,
    generate_parent_scale_data,
    validate_data_completeness,
    get_data_summary,
    create_graph_instance
)

# Import trajectory extraction utilities
from .trajectory_extraction import (
    validate_algorithm_results,
    extract_trajectory_components,
    create_expert_trajectory,
    create_failed_trajectory,
    convert_to_acbo_format
)

# Import validation utilities
from .validation import (
    validate_trajectory_completeness,
    validate_algorithm_configuration,
    compare_trajectories,
    compute_trajectory_statistics,
    validate_demonstration_quality,
    diagnose_trajectory_failure
)

# Import legacy bridge functions for backward compatibility
try:
    from .bridge import (
        # Bridge setup and validation (legacy)
        create_parent_scale_bridge,
        calculate_data_requirements, 
        validate_conversion,
        
        # Data conversion functions (legacy)
        scm_to_graph_structure,
        samples_to_parent_scale_data,
        parent_scale_results_to_posterior,
        samples_to_parent_scale_dict_format,
        create_exploration_set,
        generate_parent_scale_data_original,
        
        # Graph structure (legacy)
        ACBOGraphStructure,
        
        # PARENT_SCALE algorithm class
        PARENT_SCALE,
        PARENT_SCALE_AVAILABLE
    )
except ImportError:
    # Bridge module not available - only core functionality available
    PARENT_SCALE_AVAILABLE = False

# Package metadata
__version__ = "1.0.0"
__author__ = "ACBO Team"
__description__ = "PARENT_SCALE integration for Amortized Causal Bayesian Optimization"

# Export main API (refactored)
__all__ = [
    # Main algorithm execution functions
    'run_full_parent_scale_algorithm',
    'run_parent_discovery',
    'run_batch_expert_demonstrations',
    'convert_trajectory_to_acbo_format',
    
    # Data generation utilities
    'check_parent_scale_availability',
    'generate_parent_scale_data',
    'validate_data_completeness',
    'get_data_summary',
    'create_graph_instance',
    
    # Trajectory extraction utilities
    'validate_algorithm_results',
    'extract_trajectory_components',
    'create_expert_trajectory',
    'create_failed_trajectory',
    'convert_to_acbo_format',
    
    # Validation utilities
    'validate_trajectory_completeness',
    'validate_algorithm_configuration',
    'compare_trajectories',
    'compute_trajectory_statistics',
    'validate_demonstration_quality',
    'diagnose_trajectory_failure',
    
    # Legacy bridge functions (when available)
    'create_parent_scale_bridge',
    'calculate_data_requirements', 
    'validate_conversion',
    'scm_to_graph_structure',
    'samples_to_parent_scale_data',
    'parent_scale_results_to_posterior',
    'samples_to_parent_scale_dict_format',
    'create_exploration_set',
    'generate_parent_scale_data_original',
    'ACBOGraphStructure',
    'PARENT_SCALE',
    'PARENT_SCALE_AVAILABLE'
]