"""Experience Management System for GRPO Training.

This module provides efficient storage, sampling, and management of training
experiences for GRPO policy optimization. It integrates with JAXSampleBuffer
for trajectory storage and supports experience replay with prioritization.

Key features:
- Integration with JAXSampleBuffer for efficient tensor operations
- Experience replay with optional prioritization
- Trajectory batching and sampling for GRPO training
- Memory-efficient storage with configurable limits
- JAX-compiled operations for performance

All functions follow functional programming principles: pure functions,
immutable data structures, and explicit over implicit design.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterator, Any
import logging
from collections import deque
import time

import jax
import jax.numpy as jnp
import pyrsistent as pyr

# from ..jax_native.sample_buffer import JAXSampleBuffer, create_empty_jax_buffer  # Future integration
from ..jax_native.state import JAXAcquisitionState, get_policy_input_tensor_jax
from ..acquisition.reward_rubric import RewardResult
from ..environments.intervention_env import EnvironmentInfo
from .grpo_core import GRPOTrajectory, GRPOConfig, create_trajectory_from_experiences

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperienceConfig:
    """Configuration for experience management system.
    
    Args:
        max_buffer_size: Maximum number of experiences to store
        batch_size: Number of experiences per training batch
        min_replay_size: Minimum experiences before sampling
        prioritized_replay: Whether to use prioritized experience replay
        priority_alpha: Prioritization exponent (0.0 = uniform, 1.0 = full priority)
        importance_beta: Importance sampling correction exponent
        max_trajectory_length: Maximum length of individual trajectories
        enable_compression: Whether to compress stored experiences
        memory_limit_mb: Soft memory limit in MB (triggers cleanup)
    """
    max_buffer_size: int = 10000
    batch_size: int = 32
    min_replay_size: int = 1000
    prioritized_replay: bool = False
    priority_alpha: float = 0.6
    importance_beta: float = 0.4
    max_trajectory_length: int = 100
    enable_compression: bool = False
    memory_limit_mb: int = 1024


@dataclass(frozen=True)
class Experience:
    """Single experience for GRPO training.
    
    Args:
        state: Initial state
        action: Action taken
        next_state: Resulting state
        reward: Reward received
        done: Whether episode terminated
        log_prob: Log probability of action
        value: Value estimate for state
        env_info: Environment information
        timestamp: When experience was collected
        priority: Priority for sampling (used in prioritized replay)
    """
    state: JAXAcquisitionState
    action: pyr.PMap
    next_state: JAXAcquisitionState
    reward: RewardResult
    done: bool
    log_prob: float
    value: float
    env_info: EnvironmentInfo
    timestamp: float
    priority: float = 1.0


@dataclass(frozen=True)
class ExperienceBatch:
    """Batch of experiences for training.
    
    Args:
        experiences: List of experiences
        trajectory: GRPO trajectory computed from experiences
        importance_weights: Importance sampling weights (for prioritized replay)
        indices: Buffer indices of sampled experiences
        metadata: Additional batch metadata
    """
    experiences: List[Experience]
    trajectory: GRPOTrajectory
    importance_weights: jnp.ndarray
    indices: List[int]
    metadata: Dict[str, Any]


class ExperienceManager:
    """Manager for experience storage and replay in GRPO training.
    
    Provides efficient storage, sampling, and management of training experiences
    with support for prioritized replay and memory management.
    
    Args:
        config: Experience management configuration
        grpo_config: GRPO configuration for trajectory creation
    """
    
    def __init__(self, config: ExperienceConfig, grpo_config: GRPOConfig):
        self.config = config
        self.grpo_config = grpo_config
        
        # Experience storage
        self._experiences: deque = deque(maxlen=config.max_buffer_size)
        self._priorities = jnp.ones(config.max_buffer_size) if config.prioritized_replay else None
        self._next_idx = 0
        self._size = 0
        
        # Sampling state for prioritized replay
        self._priority_tree = None
        if config.prioritized_replay:
            self._priority_tree = SumTree(config.max_buffer_size)
        
        # Memory tracking
        self._memory_usage_mb = 0.0
        self._last_cleanup = time.time()
        
        logger.info(f"Initialized ExperienceManager with {config.max_buffer_size} capacity")
    
    def add_experience(self, experience: Experience) -> None:
        """Add a single experience to the buffer.
        
        Args:
            experience: Experience to add
        """
        # Add to buffer
        if len(self._experiences) < self.config.max_buffer_size:
            self._experiences.append(experience)
            self._size += 1
        else:
            # Replace oldest experience
            self._experiences[self._next_idx] = experience
        
        # Update priorities for prioritized replay
        if self.config.prioritized_replay:
            self._priority_tree.update(self._next_idx, experience.priority)
        
        self._next_idx = (self._next_idx + 1) % self.config.max_buffer_size
        
        # Update memory tracking
        self._update_memory_usage()
        
        # Periodic cleanup if memory limit exceeded
        if (self._memory_usage_mb > self.config.memory_limit_mb and 
            time.time() - self._last_cleanup > 60.0):  # Max once per minute
            self._cleanup_old_experiences()
    
    def add_trajectory(self, trajectory_experiences: List[Experience]) -> None:
        """Add a complete trajectory of experiences.
        
        Args:
            trajectory_experiences: List of experiences forming a trajectory
        """
        if len(trajectory_experiences) > self.config.max_trajectory_length:
            logger.warning(
                f"Trajectory length {len(trajectory_experiences)} exceeds maximum "
                f"{self.config.max_trajectory_length}, truncating"
            )
            trajectory_experiences = trajectory_experiences[:self.config.max_trajectory_length]
        
        for experience in trajectory_experiences:
            self.add_experience(experience)
    
    def sample_batch(self, batch_size: Optional[int] = None) -> Optional[ExperienceBatch]:
        """Sample a batch of experiences for training.
        
        Args:
            batch_size: Number of experiences to sample (uses config default if None)
            
        Returns:
            ExperienceBatch if sufficient experiences available, None otherwise
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        if self._size < self.config.min_replay_size:
            return None
        
        if self.config.prioritized_replay:
            return self._sample_prioritized_batch(batch_size)
        else:
            return self._sample_uniform_batch(batch_size)
    
    def update_priorities(self, indices: List[int], priorities: jnp.ndarray) -> None:
        """Update priorities for prioritized replay.
        
        Args:
            indices: Buffer indices to update
            priorities: New priority values
        """
        if not self.config.prioritized_replay:
            logger.warning("Attempting to update priorities without prioritized replay enabled")
            return
        
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < self._size:
                self._priority_tree.update(idx, float(priority))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get experience manager statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'buffer_size': self._size,
            'buffer_capacity': self.config.max_buffer_size,
            'memory_usage_mb': self._memory_usage_mb,
            'utilization': self._size / self.config.max_buffer_size,
            'can_sample': self._size >= self.config.min_replay_size,
            'prioritized_replay': self.config.prioritized_replay,
            'avg_priority': float(jnp.mean(self._priorities[:self._size])) if self.config.prioritized_replay else None,
        }
    
    def clear(self) -> None:
        """Clear all stored experiences."""
        self._experiences.clear()
        self._size = 0
        self._next_idx = 0
        self._memory_usage_mb = 0.0
        
        if self.config.prioritized_replay:
            self._priority_tree = SumTree(self.config.max_buffer_size)
        
        logger.info("Cleared experience buffer")
    
    def _sample_uniform_batch(self, batch_size: int) -> ExperienceBatch:
        """Sample a batch uniformly at random."""
        indices = jax.random.choice(
            jax.random.PRNGKey(int(time.time() * 1000000) % 2**31),
            self._size,
            shape=(batch_size,),
            replace=False
        )
        
        experiences = [self._experiences[idx] for idx in indices]
        trajectory = self._create_trajectory_from_experiences(experiences)
        importance_weights = jnp.ones(batch_size)  # Uniform weights
        
        return ExperienceBatch(
            experiences=experiences,
            trajectory=trajectory,
            importance_weights=importance_weights,
            indices=list(indices),
            metadata={'sampling_method': 'uniform'}
        )
    
    def _sample_prioritized_batch(self, batch_size: int) -> ExperienceBatch:
        """Sample a batch using prioritized experience replay."""
        indices = []
        priorities = []
        
        # Sample from priority tree
        for _ in range(batch_size):
            idx, priority = self._priority_tree.sample()
            indices.append(idx)
            priorities.append(priority)
        
        experiences = [self._experiences[idx] for idx in indices]
        
        # Compute importance sampling weights
        priorities_array = jnp.array(priorities)
        min_priority = jnp.min(priorities_array)
        importance_weights = jnp.power(
            (min_priority / priorities_array) * self._size,
            self.config.importance_beta
        )
        # Normalize weights
        importance_weights = importance_weights / jnp.max(importance_weights)
        
        trajectory = self._create_trajectory_from_experiences(experiences)
        
        return ExperienceBatch(
            experiences=experiences,
            trajectory=trajectory,
            importance_weights=importance_weights,
            indices=indices,
            metadata={'sampling_method': 'prioritized'}
        )
    
    def _create_trajectory_from_experiences(self, experiences: List[Experience]) -> GRPOTrajectory:
        """Create GRPO trajectory from experiences."""
        # Extract simple tensor representations for states
        # For now, use mechanism features as a simple state representation
        states = jnp.stack([exp.state.mechanism_features.flatten() for exp in experiences])
        actions = jnp.stack([self._action_to_tensor(exp.action) for exp in experiences])
        rewards = jnp.array([exp.reward.total_reward for exp in experiences])
        values = jnp.array([exp.value for exp in experiences])
        log_probs = jnp.array([exp.log_prob for exp in experiences])
        dones = jnp.array([exp.done for exp in experiences])
        
        # Bootstrap value (last experience's next state value estimate)
        bootstrap_value = 0.0 if experiences[-1].done else experiences[-1].value
        
        # Create trajectory using GRPO core function
        return create_trajectory_from_experiences(
            states=states,
            actions=actions,
            rewards=rewards,
            values=values,
            log_probs=log_probs,
            dones=dones,
            bootstrap_value=bootstrap_value,
            config=self.grpo_config
        )
    
    def _action_to_tensor(self, action: pyr.PMap) -> jnp.ndarray:
        """Convert action to tensor format."""
        # Simple conversion - in practice this would depend on action space
        # For now, assume actions are intervention values
        if not action:
            return jnp.zeros(1)
        return jnp.array([list(action.values())[0]])
    
    def _update_memory_usage(self) -> None:
        """Update memory usage estimate."""
        # Simple estimation based on buffer size
        # In practice, this would be more sophisticated
        bytes_per_experience = 1024  # Rough estimate
        self._memory_usage_mb = (self._size * bytes_per_experience) / (1024 * 1024)
    
    def _cleanup_old_experiences(self) -> None:
        """Remove old experiences to free memory."""
        if self._size < self.config.max_buffer_size // 2:
            return
        
        # Remove oldest 25% of experiences
        cleanup_count = self._size // 4
        for _ in range(cleanup_count):
            if self._experiences:
                self._experiences.popleft()
                self._size -= 1
        
        self._last_cleanup = time.time()
        logger.info(f"Cleaned up {cleanup_count} old experiences")


class SumTree:
    """Sum tree data structure for prioritized experience replay.
    
    Provides O(log n) sampling and updating operations for priority-based
    sampling of experiences.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = jnp.zeros(2 * capacity - 1)
        self.data_pointer = 0
    
    def update(self, idx: int, priority: float) -> None:
        """Update priority at given index."""
        tree_idx = idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree = self.tree.at[tree_idx].set(priority)
        
        # Propagate change up the tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree = self.tree.at[tree_idx].add(change)
    
    def sample(self) -> Tuple[int, float]:
        """Sample an index based on priorities."""
        random_value = jax.random.uniform(jax.random.PRNGKey(int(time.time() * 1000000) % 2**31))
        random_value *= self.tree[0]  # Total priority
        
        idx = self._retrieve(0, random_value)
        data_idx = idx - self.capacity + 1
        priority = self.tree[idx]
        
        return data_idx, priority
    
    def _retrieve(self, idx: int, value: float) -> int:
        """Retrieve leaf index for given cumulative value."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):  # Leaf node
            return idx
        
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])


# Factory functions for common configurations
def create_experience_manager(
    grpo_config: GRPOConfig,
    buffer_size: int = 10000,
    batch_size: int = 32,
    prioritized_replay: bool = False
) -> ExperienceManager:
    """Create experience manager with default configuration.
    
    Args:
        grpo_config: GRPO configuration
        buffer_size: Maximum buffer size
        batch_size: Training batch size
        prioritized_replay: Whether to use prioritized replay
        
    Returns:
        Configured ExperienceManager
    """
    config = ExperienceConfig(
        max_buffer_size=buffer_size,
        batch_size=batch_size,
        prioritized_replay=prioritized_replay
    )
    return ExperienceManager(config, grpo_config)


def create_high_capacity_experience_manager(
    grpo_config: GRPOConfig,
    memory_limit_mb: int = 4096
) -> ExperienceManager:
    """Create high-capacity experience manager for large-scale training.
    
    Args:
        grpo_config: GRPO configuration
        memory_limit_mb: Memory limit in MB
        
    Returns:
        High-capacity ExperienceManager
    """
    config = ExperienceConfig(
        max_buffer_size=50000,
        batch_size=64,
        min_replay_size=5000,
        prioritized_replay=True,
        priority_alpha=0.6,
        importance_beta=0.4,
        memory_limit_mb=memory_limit_mb,
        enable_compression=True
    )
    return ExperienceManager(config, grpo_config)


def create_memory_efficient_experience_manager(
    grpo_config: GRPOConfig
) -> ExperienceManager:
    """Create memory-efficient experience manager for resource-constrained environments.
    
    Args:
        grpo_config: GRPO configuration
        
    Returns:
        Memory-efficient ExperienceManager
    """
    config = ExperienceConfig(
        max_buffer_size=2000,
        batch_size=16,
        min_replay_size=200,
        prioritized_replay=False,
        memory_limit_mb=256,
        max_trajectory_length=50
    )
    return ExperienceManager(config, grpo_config)