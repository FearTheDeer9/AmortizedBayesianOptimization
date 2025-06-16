#!/usr/bin/env python3
"""
Expert Collection Package

Refactored expert demonstration collection functionality organized into
focused modules for better maintainability and single responsibility.

Modules:
- data_structures: ExpertDemonstration, ExpertTrajectoryDemonstration, DemonstrationBatch
- scm_generation: Functions for generating diverse SCM problems
- collector: Main ExpertDemonstrationCollector class
- main: Entry point for standard collection workflows
"""

# Import main components
from .data_structures import (
    ExpertDemonstration, 
    ExpertTrajectoryDemonstration, 
    DemonstrationBatch
)
from .scm_generation import generate_scm_problems, generate_scm
from .collector import ExpertDemonstrationCollector
from .main import collect_expert_demonstrations_main

# Package metadata
__version__ = "1.0.0"
__author__ = "ACBO Team"
__description__ = "Expert demonstration collection for ACBO training"

# Export main components
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