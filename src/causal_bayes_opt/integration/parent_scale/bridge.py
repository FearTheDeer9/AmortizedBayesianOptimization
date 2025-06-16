#!/usr/bin/env python3
"""
PARENT_SCALE Integration Bridge

High-level coordination and API for PARENT_SCALE integration.
This module provides the main interface for using PARENT_SCALE components
with our ACBO data structures.
"""

from typing import List, Dict, Any, Optional

import pyrsistent as pyr

# Import all components
from .data_conversion import (
    create_parent_scale_bridge, calculate_data_requirements, validate_conversion,
    scm_to_graph_structure, samples_to_parent_scale_data, 
    parent_scale_results_to_posterior, samples_to_parent_scale_dict_format,
    create_exploration_set, generate_parent_scale_data_original
)
from .algorithm_runner import run_parent_discovery, run_full_parent_scale_algorithm, PARENT_SCALE, PARENT_SCALE_AVAILABLE
from .graph_structure import ACBOGraphStructure


# Re-export main API functions
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
    
    # Algorithm execution
    'run_parent_discovery',
    'run_full_parent_scale_algorithm',
    
    # Graph structure
    'ACBOGraphStructure',
    
    # PARENT_SCALE algorithm class
    'PARENT_SCALE',
    'PARENT_SCALE_AVAILABLE'
]


def main_test():
    """Test the bridge functionality."""
    try:
        print("PARENT_SCALE Data Bridge Test")
        print("=" * 40)
        
        create_parent_scale_bridge()  # Validate availability
        print("✅ Bridge functions available")
        
        # Test data requirements calculation
        for n_nodes in [5, 10, 20]:
            req = calculate_data_requirements(n_nodes)
            print(f"  {n_nodes} nodes: {req['total_samples']} samples, {req['bootstrap_samples']} bootstraps")
        
        print("✅ Bridge functionality validated")
        
    except ImportError as e:
        print(f"❌ PARENT_SCALE not available: {e}")
        print("Please ensure external/parent_scale is properly set up.")
    except Exception as e:
        print(f"❌ Bridge test failed: {e}")


if __name__ == "__main__":
    main_test()