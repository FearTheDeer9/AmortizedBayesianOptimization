#!/usr/bin/env python3
"""
Expert Demonstration Data Structures

Defines the data structures used for storing expert demonstrations
from PARENT_SCALE algorithm runs.
"""

import time
from typing import List, Dict, Any, FrozenSet
from dataclasses import dataclass, field

import numpy as onp
import pyrsistent as pyr


@dataclass
class ExpertDemonstration:
    """Complete expert demonstration from one PARENT_SCALE run."""
    
    # Problem setup
    scm: pyr.PMap
    target_variable: str
    n_nodes: int
    graph_type: str
    
    # Data and results
    observational_samples: List[pyr.PMap]
    interventional_samples: List[pyr.PMap] 
    discovered_parents: FrozenSet[str]
    confidence: float
    accuracy: float  # Against ground truth
    
    # Training data extracted
    parent_posterior: Dict[str, Any]
    data_requirements: Dict[str, int]
    
    # Performance metrics
    inference_time: float
    total_samples_used: int
    
    # Metadata
    collection_timestamp: float = field(default_factory=time.time)
    validation_passed: bool = True


@dataclass
class ExpertTrajectoryDemonstration:
    """Complete expert trajectory from full PARENT_SCALE CBO algorithm."""
    
    # Problem setup
    scm: pyr.PMap
    target_variable: str
    n_nodes: int
    graph_type: str
    
    # Initial data
    initial_observational_samples: List[pyr.PMap]
    initial_interventional_samples: List[pyr.PMap]
    
    # Complete expert trajectory
    expert_trajectory: Dict[str, Any]  # Full CBO trajectory data
    
    # Algorithm performance
    algorithm_time: float
    total_samples_used: int
    final_optimum: float
    total_improvement: float
    convergence_rate: float
    exploration_efficiency: float
    
    # Configuration
    data_requirements: Dict[str, int]
    algorithm_config: Dict[str, Any]
    
    # Metadata
    collection_timestamp: float = field(default_factory=time.time)
    validation_passed: bool = True


@dataclass 
class DemonstrationBatch:
    """Batch of demonstrations for training."""
    
    demonstrations: List[ExpertDemonstration]
    batch_id: str
    collection_config: Dict[str, Any]
    
    # Summary statistics
    total_demonstrations: int = field(init=False)
    avg_accuracy: float = field(init=False)
    graph_types_covered: List[str] = field(init=False)
    node_sizes_covered: List[int] = field(init=False)
    
    def __post_init__(self):
        self.total_demonstrations = len(self.demonstrations)
        self.avg_accuracy = onp.mean([d.accuracy for d in self.demonstrations])
        self.graph_types_covered = list(set(d.graph_type for d in self.demonstrations))
        self.node_sizes_covered = list(set(d.n_nodes for d in self.demonstrations))