"""
Trajectory Replay Buffer for Joint ACBO Training.

This module provides a replay buffer that stores complete trajectories
(full episodes) to avoid mixing samples from different contexts.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import random
import numpy as np

from ..data_structures.buffer import ExperienceBuffer

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """A complete trajectory from one episode."""
    buffer: ExperienceBuffer  # Complete episode data
    scm_metadata: Dict[str, Any]  # SCM information
    performance_delta: float  # Performance improvement
    timestamp: float  # When collected
    episode: int  # Episode number
    phase: str  # Training phase when collected
    
    @property
    def priority(self) -> float:
        """Compute priority for sampling."""
        # Higher priority for: recent, high performance, diverse
        recency_score = 1.0 / (1.0 + time.time() - self.timestamp)
        performance_score = abs(self.performance_delta)
        return recency_score * 0.5 + performance_score * 0.5


class TrajectoryReplayBuffer:
    """
    Replay buffer that stores complete trajectories.
    
    Key features:
    - Stores full episodes (no sample mixing)
    - Prioritized sampling based on recency and performance
    - SCM-aware grouping for consistent sampling
    """
    
    def __init__(self, 
                 capacity: int = 100,
                 prioritize_recent: bool = True,
                 group_by_scm: bool = True):
        """
        Initialize trajectory replay buffer.
        
        Args:
            capacity: Maximum number of trajectories to store
            prioritize_recent: Whether to prioritize recent trajectories
            group_by_scm: Whether to group trajectories by SCM type
        """
        self.capacity = capacity
        self.prioritize_recent = prioritize_recent
        self.group_by_scm = group_by_scm
        
        # Main storage
        self.trajectories = deque(maxlen=capacity)
        
        # SCM grouping for consistent sampling
        self.scm_groups = {}  # SCM characteristics -> list of trajectory indices
        
        # Statistics
        self.total_added = 0
        self.total_sampled = 0
        
        logger.info(f"Initialized TrajectoryReplayBuffer with capacity {capacity}")
    
    def add_trajectory(self,
                      buffer: ExperienceBuffer,
                      scm_metadata: Dict[str, Any],
                      performance_delta: float,
                      episode: int = 0,
                      phase: str = "unknown"):
        """
        Add a complete trajectory to the buffer.
        
        Args:
            buffer: Complete episode experience buffer
            scm_metadata: Metadata about the SCM
            performance_delta: Performance improvement from this trajectory
            episode: Episode number
            phase: Training phase when collected
        """
        trajectory = Trajectory(
            buffer=buffer,
            scm_metadata=scm_metadata,
            performance_delta=performance_delta,
            timestamp=time.time(),
            episode=episode,
            phase=phase
        )
        
        # Add to main storage
        self.trajectories.append(trajectory)
        self.total_added += 1
        
        # Update SCM grouping
        if self.group_by_scm:
            scm_key = self._get_scm_key(scm_metadata)
            if scm_key not in self.scm_groups:
                self.scm_groups[scm_key] = []
            # Store index in trajectories deque
            self.scm_groups[scm_key].append(len(self.trajectories) - 1)
            
            # Clean up invalid indices due to deque rotation
            self._cleanup_scm_groups()
        
        logger.debug(f"Added trajectory {self.total_added} to replay buffer "
                    f"(current size: {len(self.trajectories)})")
    
    def sample_batch(self, 
                    batch_size: int = 4,
                    same_scm: bool = False) -> List[Trajectory]:
        """
        Sample a batch of trajectories.
        
        Args:
            batch_size: Number of trajectories to sample
            same_scm: Whether to sample from the same SCM type
            
        Returns:
            List of sampled trajectories
        """
        if len(self.trajectories) == 0:
            return []
        
        actual_batch_size = min(batch_size, len(self.trajectories))
        
        if same_scm and self.group_by_scm:
            # Sample from the same SCM family
            batch = self._sample_same_scm(actual_batch_size)
        else:
            # Mixed sampling
            if self.prioritize_recent:
                batch = self._prioritized_sample(actual_batch_size)
            else:
                batch = random.sample(list(self.trajectories), actual_batch_size)
        
        self.total_sampled += len(batch)
        return batch
    
    def _sample_same_scm(self, batch_size: int) -> List[Trajectory]:
        """Sample trajectories from the same SCM family."""
        # Find the largest SCM group
        valid_groups = {}
        for scm_key, indices in self.scm_groups.items():
            # Filter valid indices
            valid_indices = [i for i in indices if 0 <= i < len(self.trajectories)]
            if valid_indices:
                valid_groups[scm_key] = valid_indices
        
        if not valid_groups:
            # Fallback to mixed sampling
            return self._prioritized_sample(batch_size)
        
        # Choose the largest group
        largest_key = max(valid_groups.keys(), key=lambda k: len(valid_groups[k]))
        group_indices = valid_groups[largest_key]
        
        # Sample from this group
        sample_size = min(batch_size, len(group_indices))
        sampled_indices = random.sample(group_indices, sample_size)
        
        return [self.trajectories[i] for i in sampled_indices]
    
    def _prioritized_sample(self, batch_size: int) -> List[Trajectory]:
        """Sample with priority based on recency and performance."""
        # Compute priorities
        priorities = [traj.priority for traj in self.trajectories]
        total_priority = sum(priorities)
        
        if total_priority == 0:
            # Uniform sampling if all priorities are zero
            return random.sample(list(self.trajectories), batch_size)
        
        # Weighted sampling
        probabilities = [p / total_priority for p in priorities]
        indices = np.random.choice(
            len(self.trajectories),
            size=batch_size,
            replace=False,
            p=probabilities
        )
        
        return [self.trajectories[i] for i in indices]
    
    def _get_scm_key(self, metadata: Dict[str, Any]) -> Tuple:
        """Get a hashable key for SCM grouping."""
        # Group by number of variables and graph type
        n_vars = metadata.get('n_variables', 0)
        graph_type = metadata.get('graph_type', 'unknown')
        level = metadata.get('curriculum_level', 0)
        
        return (n_vars, graph_type, level)
    
    def _cleanup_scm_groups(self):
        """Remove invalid indices from SCM groups due to deque rotation."""
        max_valid_index = len(self.trajectories) - 1
        
        for scm_key in list(self.scm_groups.keys()):
            # Filter out invalid indices
            valid_indices = [
                i for i in self.scm_groups[scm_key]
                if 0 <= i <= max_valid_index
            ]
            
            if valid_indices:
                self.scm_groups[scm_key] = valid_indices
            else:
                # Remove empty group
                del self.scm_groups[scm_key]
    
    def get_recent_trajectories(self, n: int = 10) -> List[Trajectory]:
        """Get the n most recent trajectories."""
        n = min(n, len(self.trajectories))
        return list(self.trajectories)[-n:]
    
    def get_best_trajectories(self, n: int = 10) -> List[Trajectory]:
        """Get the n best performing trajectories."""
        sorted_trajs = sorted(
            self.trajectories,
            key=lambda t: t.performance_delta,
            reverse=True
        )
        return sorted_trajs[:n]
    
    def clear(self):
        """Clear the replay buffer."""
        self.trajectories.clear()
        self.scm_groups.clear()
        self.total_added = 0
        self.total_sampled = 0
        logger.info("Cleared replay buffer")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        if len(self.trajectories) == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'total_added': self.total_added,
                'total_sampled': self.total_sampled
            }
        
        performance_deltas = [t.performance_delta for t in self.trajectories]
        phases = [t.phase for t in self.trajectories]
        
        return {
            'size': len(self.trajectories),
            'capacity': self.capacity,
            'total_added': self.total_added,
            'total_sampled': self.total_sampled,
            'mean_performance': np.mean(performance_deltas),
            'std_performance': np.std(performance_deltas),
            'best_performance': max(performance_deltas),
            'worst_performance': min(performance_deltas),
            'phase_distribution': {
                phase: phases.count(phase) / len(phases)
                for phase in set(phases)
            },
            'n_scm_groups': len(self.scm_groups)
        }
    
    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.trajectories)
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"TrajectoryReplayBuffer(size={len(self.trajectories)}/{self.capacity}, "
                f"scm_groups={len(self.scm_groups)})")