#!/usr/bin/env python3
"""
PARENT_SCALE Data Bridge - Refactored Version

Provides conversion functions between our ACBO data structures and PARENT_SCALE's
expected formats. This is the main entry point that imports from the refactored
modular components for better maintainability.

Key Conversions:
1. Our SCM → PARENT_SCALE GraphStructure  
2. Our Sample list → PARENT_SCALE Data(samples, nodes)
3. PARENT_SCALE prob_estimate → Our ParentSetPosterior
4. Round-trip validation for data integrity
"""

# Import functions from the refactored modules for backward compatibility
try:
    from .parent_scale import (
        # Main algorithm execution functions
        run_full_parent_scale_algorithm,
        run_parent_discovery,
        
        # Legacy bridge functions (when available)
        create_parent_scale_bridge,
        calculate_data_requirements, 
        validate_conversion,
        scm_to_graph_structure,
        samples_to_parent_scale_data,
        parent_scale_results_to_posterior,
        samples_to_parent_scale_dict_format,
        create_exploration_set,
        generate_parent_scale_data_original,
    
    # Algorithm execution
    run_parent_discovery,
    run_full_parent_scale_algorithm,
    
    # Graph structure
    ACBOGraphStructure,
    
    # PARENT_SCALE algorithm class
    PARENT_SCALE,
    PARENT_SCALE_AVAILABLE
)

# Keep the old function name for backward compatibility
generate_parent_scale_data = generate_parent_scale_data_original

# Re-export everything for backward compatibility
__all__ = [
    # Bridge setup and validation
    'create_parent_scale_bridge',
    'calculate_data_requirements', 
    'validate_conversion',
    
    # Data conversion functions
    'scm_to_graph_structure',
    'samples_to_parent_scale_data',
    'parent_scale_results_to_posterior',
    'samples_to_parent_scale_dict_format',
    'create_exploration_set',
    'generate_parent_scale_data_original',
    'generate_parent_scale_data',  # Backward compatibility
    
    # Algorithm execution
    'run_parent_discovery',
    'run_full_parent_scale_algorithm',
    
    # Graph structure
    'ACBOGraphStructure',
    
    # PARENT_SCALE algorithm class
    'PARENT_SCALE',
    'PARENT_SCALE_AVAILABLE'
]


if __name__ == "__main__":
    # Simple test of the bridge
    from .parent_scale.bridge import main_test
    main_test()