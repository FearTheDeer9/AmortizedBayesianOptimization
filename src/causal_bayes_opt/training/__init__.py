"""
Expert Demonstration Collection for ACBO Training

This module implements expert demonstration collection using PARENT_SCALE neural
doubly robust method with validated data scaling requirements.

Key Components:
- ExpertDemonstrationCollector: Main collection class  
- ExpertDemonstration: Data structure for demonstrations
- DemonstrationBatch: Batch collection management

All components follow functional programming principles and use validated
scaling formulas for reliable parent discovery.
"""

from .expert_demonstration_collection import (
    ExpertDemonstration,
    DemonstrationBatch,
    ExpertDemonstrationCollector,
    collect_expert_demonstrations_main
)

__all__ = [
    "ExpertDemonstration",
    "DemonstrationBatch", 
    "ExpertDemonstrationCollector",
    "collect_expert_demonstrations_main"
]