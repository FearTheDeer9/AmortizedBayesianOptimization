"""
PARENT_SCALE Integration Module

Provides pure functions for integrating ACBO with PARENT_SCALE neural doubly
robust method. All functions follow functional programming principles.
"""

from .parent_scale_bridge import (
    scm_to_graph_structure,
    samples_to_parent_scale_data,
    parent_scale_results_to_posterior,
    run_parent_discovery,
    validate_conversion,
    create_parent_scale_bridge,
    calculate_data_requirements
)

__all__ = [
    "scm_to_graph_structure",
    "samples_to_parent_scale_data", 
    "parent_scale_results_to_posterior",
    "run_parent_discovery",
    "validate_conversion",
    "create_parent_scale_bridge",
    "calculate_data_requirements"
]