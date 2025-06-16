#!/usr/bin/env python3
"""
Expert Demonstration Collection for ACBO - Refactored Version

Collects expert demonstrations from PARENT_SCALE neural doubly robust method
using validated data scaling requirements. This is the main entry point that
imports from the refactored modular components for better maintainability.

Key Features:
1. Uses validated O(d^2.5) data scaling for reliable parent discovery
2. Generates diverse SCM problems for comprehensive training coverage
3. Collects both (data → posterior) pairs and (state → action) sequences
4. Validates demonstration quality before saving
5. Supports batch collection for efficient training data generation
"""

# Import all components from the refactored modules for backward compatibility
from .expert_collection import (
    # Data structures
    ExpertDemonstration,
    ExpertTrajectoryDemonstration, 
    DemonstrationBatch,
    
    # SCM generation
    generate_scm_problems,
    generate_scm,
    
    # Main collector
    ExpertDemonstrationCollector,
    
    # Main function
    collect_expert_demonstrations_main
)

# Re-export everything for backward compatibility
__all__ = [
    # Data structures
    'ExpertDemonstration',
    'ExpertTrajectoryDemonstration', 
    'DemonstrationBatch',
    
    # SCM generation
    'generate_scm_problems',
    'generate_scm',
    
    # Main collector
    'ExpertDemonstrationCollector',
    
    # Main function
    'collect_expert_demonstrations_main'
]


if __name__ == "__main__":
    collect_expert_demonstrations_main()