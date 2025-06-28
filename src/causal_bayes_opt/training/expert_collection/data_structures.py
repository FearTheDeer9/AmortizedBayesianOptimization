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
    
    def __getstate__(self) -> Dict[str, Any]:
        """Custom serialization to handle SCM with mechanism functions."""
        from ...data_structures.scm import serialize_scm_for_storage
        
        # Serialize the SCM separately to handle mechanism functions
        state = self.__dict__.copy()
        
        try:
            # Replace SCM with serializable version
            state['scm_serialized'] = serialize_scm_for_storage(self.scm)
            del state['scm']  # Remove original SCM with functions
        except Exception as e:
            # Fallback: try to serialize just the structure without mechanisms
            from ...data_structures.scm import get_variables, get_edges, get_target
            state['scm_fallback'] = {
                'variables': list(get_variables(self.scm)),
                'edges': list(get_edges(self.scm)),
                'target': get_target(self.scm),
                'metadata': dict(self.scm.get('metadata', {})),
                'serialization_error': str(e)
            }
            del state['scm']
        
        return state
    
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Custom deserialization to reconstruct SCM with mechanism functions."""
        from ...data_structures.scm import deserialize_scm_from_storage, create_scm
        
        # Restore SCM from serialized data
        if 'scm_serialized' in state:
            try:
                scm = deserialize_scm_from_storage(state['scm_serialized'])
                del state['scm_serialized']
            except Exception as e:
                raise RuntimeError(f"Failed to deserialize SCM: {e}")
        elif 'scm_fallback' in state:
            # Fallback reconstruction (without mechanisms)
            fallback_data = state['scm_fallback']
            scm = create_scm(
                variables=frozenset(fallback_data['variables']),
                edges=frozenset((p, c) for p, c in fallback_data['edges']),
                mechanisms={},  # Empty mechanisms
                target=fallback_data.get('target'),
                metadata=fallback_data.get('metadata', {})
            )
            del state['scm_fallback']
        else:
            raise RuntimeError("No SCM data found in serialized state")
        
        # Restore all attributes
        self.__dict__.update(state)
        self.scm = scm


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