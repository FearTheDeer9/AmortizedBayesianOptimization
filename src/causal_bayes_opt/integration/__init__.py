"""
PARENT_SCALE Integration Module

Provides pure functions for integrating ACBO with PARENT_SCALE neural doubly
robust method. All functions follow functional programming principles.
"""

from .parent_scale_bridge import (
    run_full_parent_scale_algorithm,
    run_parent_discovery,
    run_batch_expert_demonstrations,
    convert_trajectory_to_acbo_format,
    check_parent_scale_availability,
    PARENT_SCALE_AVAILABLE
)

# Try to import legacy functions if available
try:
    from .parent_scale_bridge import (
        scm_to_graph_structure,
        samples_to_parent_scale_data,
        parent_scale_results_to_posterior
    )
    LEGACY_FUNCTIONS_AVAILABLE = True
except ImportError:
    LEGACY_FUNCTIONS_AVAILABLE = False

__all__ = [
    # Main algorithm functions
    "run_full_parent_scale_algorithm",
    "run_parent_discovery",
    "run_batch_expert_demonstrations", 
    "convert_trajectory_to_acbo_format",
    "check_parent_scale_availability",
    "PARENT_SCALE_AVAILABLE"
]

# Add legacy functions if available
if LEGACY_FUNCTIONS_AVAILABLE:
    __all__.extend([
        "scm_to_graph_structure",
        "samples_to_parent_scale_data", 
        "parent_scale_results_to_posterior"
    ])