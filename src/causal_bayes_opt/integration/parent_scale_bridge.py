#!/usr/bin/env python3
"""
PARENT_SCALE Data Bridge - Simplified for Refactored Version

Provides backward compatibility by re-exporting the main functions from the
refactored parent_scale module. This maintains API compatibility while
using the new modular structure.
"""

# Import main functions from refactored algorithm_runner
from .parent_scale.algorithm_runner import (
    run_full_parent_scale_algorithm,
    run_parent_discovery,
    run_batch_expert_demonstrations,
    convert_trajectory_to_acbo_format
)

# Import legacy data conversion functions if available
try:
    from .parent_scale.data_conversion import (
        scm_to_graph_structure,
        samples_to_parent_scale_data,
        parent_scale_results_to_posterior,
        create_parent_scale_bridge,
        calculate_data_requirements
    )
    DATA_CONVERSION_AVAILABLE = True
except ImportError:
    DATA_CONVERSION_AVAILABLE = False

# Import availability check  
from .parent_scale.data_processing import ensure_parent_scale_imports

# Check availability
try:
    ensure_parent_scale_imports()
    PARENT_SCALE_AVAILABLE = True
except ImportError:
    PARENT_SCALE_AVAILABLE = False

# Re-export main functions for backward compatibility
__all__ = [
    # Main algorithm execution functions
    'run_full_parent_scale_algorithm',
    'run_parent_discovery', 
    'run_batch_expert_demonstrations',
    'convert_trajectory_to_acbo_format',
    
    # Availability check
    'PARENT_SCALE_AVAILABLE'
]

# Add data conversion functions if available
if DATA_CONVERSION_AVAILABLE:
    __all__.extend([
        'scm_to_graph_structure',
        'samples_to_parent_scale_data',
        'parent_scale_results_to_posterior',
        'create_parent_scale_bridge',
        'calculate_data_requirements'
    ])

# Alias for backward compatibility
generate_parent_scale_data_original = run_full_parent_scale_algorithm