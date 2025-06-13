"""
Trajectory buffer for RL training in ACBO.

This module provides trajectory storage that composes with ExperienceBuffer,
enabling both causal discovery (Phase 2) and RL training (Phase 3) from the same data.

The TrajectoryBuffer maintains complete (state, action, reward, next_state) tuples
for GRPO training while automatically synchronizing with ExperienceBuffer for
AVICI-based causal discovery.
"""

# Standard library imports
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Iterator

# Third-party imports
import jax.numpy as jnp
import pyrsistent as pyr

# Local imports
from .state import AcquisitionState
from ..data_structures.buffer import ExperienceBuffer, BufferStatistics

# Type aliases  
InterventionSpec = pyr.PMap
Sample = pyr.PMap 
TrajectoryBatch = List['TrajectoryStep']

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrajectoryStep:
    """
    Immutable representation of a complete RL trajectory step.
    
    Contains all information needed for GRPO training:
    - Decision context (state when intervention was chosen)
    - Action taken (intervention specification)
    - Environment response (outcome sample)
    - Reward signal (computed reward)
    - Resulting context (next state after intervention)
    
    This enables proper RL training with full decision context,
    unlike simple (intervention, outcome) pairs.
    """
    # Core RL components
    state: AcquisitionState           # s_t: Decision context
    intervention: InterventionSpec    # a_t: Action taken
    outcome: Sample                   # Environment response to action
    reward: float                     # r_t: Computed reward signal
    next_state: AcquisitionState      # s_{t+1}: Resulting context
    
    # Training context
    step_index: int                   # Global step number
    training_metadata: pyr.PMap = pyr.m()  # Additional training context
    
    def __post_init__(self):
        """Validate trajectory step consistency."""
        self._validate_trajectory_consistency()
    
    def _validate_trajectory_consistency(self) -> None:
        """Validate that trajectory step is internally consistent."""
        # Check step progression
        if self.next_state.step != self.state.step + 1:
            raise ValueError(
                f"Inconsistent step progression: {self.state.step} -> {self.next_state.step}"
            )
        
        # Check target consistency
        if self.state.current_target != self.next_state.current_target:
            raise ValueError(
                f"Target changed during step: {self.state.current_target} -> {self.next_state.current_target}"
            )
        
        # Check reward is finite
        if not jnp.isfinite(self.reward):
            raise ValueError(f"Reward must be finite, got {self.reward}")
        
        # Check step index consistency
        if self.step_index != self.state.step:
            logger.warning(
                f"Step index {self.step_index} doesn't match state step {self.state.step}"
            )
    
    def get_state_transition(self) -> Tuple[AcquisitionState, AcquisitionState]:
        """Get (state, next_state) pair for analysis."""
        return self.state, self.next_state
    
    def get_uncertainty_reduction(self) -> float:
        """Get uncertainty reduction from this step (information gain)."""
        return self.state.uncertainty_bits - self.next_state.uncertainty_bits
    
    def get_target_improvement(self) -> float:
        """Get target variable improvement from this step."""
        return self.next_state.best_value - self.state.best_value
    
    def summary(self) -> Dict[str, Any]:
        """Get human-readable summary of trajectory step."""
        return {
            'step_index': self.step_index,
            'intervention_type': self.intervention.get('type', 'unknown'),
            'intervention_targets': list(self.intervention.get('targets', set())),
            'reward': self.reward,
            'uncertainty_reduction_bits': self.get_uncertainty_reduction(),
            'target_improvement': self.get_target_improvement(),
            'state_buffer_size': self.state.buffer_statistics.total_samples,
            'next_state_buffer_size': self.next_state.buffer_statistics.total_samples,
        }


class TrajectoryBuffer:
    """
    RL trajectory storage that composes with ExperienceBuffer.
    
    Provides two complementary views of the same data:
    1. ExperienceBuffer view: (intervention, outcome) for causal discovery
    2. Trajectory view: (state, action, reward, next_state) for RL training
    
    Key benefits:
    - Automatic synchronization between views
    - Backward compatibility with Phase 2 AVICI integration
    - Rich context for GRPO training
    - Comprehensive analysis capabilities
    """
    
    def __init__(self, experience_buffer: Optional[ExperienceBuffer] = None):
        """
        Initialize trajectory buffer.
        
        Args:
            experience_buffer: Existing buffer to compose with, or None for new buffer
        """
        # Composition: reuse existing buffer or create new one
        self._experience_buffer = experience_buffer or ExperienceBuffer()
        
        # Trajectory-specific storage
        self._trajectory_steps: List[TrajectoryStep] = []
        
        # Indexing for efficient queries
        self._steps_by_reward: Dict[float, List[int]] = {}
        self._steps_by_target: Dict[str, List[int]] = {}
        
        # Metadata
        self._creation_time = time.time()
        self._last_trajectory_time = self._creation_time
        
        logger.debug(f"Created TrajectoryBuffer with {self._experience_buffer.size()} existing samples")
    
    # === RL Training Interface ===
    
    def add_trajectory_step(self, step: TrajectoryStep) -> None:
        """
        Add complete trajectory step for RL training.
        
        Automatically synchronizes with underlying ExperienceBuffer
        to maintain consistency for causal discovery.
        
        Args:
            step: Complete trajectory step to add
        """
        # Validate step
        if not isinstance(step, TrajectoryStep):
            raise ValueError("Must provide TrajectoryStep instance")
        
        # Add to trajectory storage
        step_index = len(self._trajectory_steps)
        self._trajectory_steps.append(step)
        
        # Automatically sync to experience buffer for causal discovery
        # Note: This assumes the intervention-outcome pair isn't already in the buffer
        # In practice, the next_state.buffer should already contain this data
        try:
            self._experience_buffer.add_intervention(step.intervention, step.outcome)
        except Exception as e:
            # If already added or other issue, log but continue
            logger.debug(f"Could not sync to experience buffer: {e}")
        
        # Update indices
        self._update_indices(step, step_index)
        
        # Update metadata
        self._last_trajectory_time = time.time()
        
        logger.debug(f"Added trajectory step {step_index}: reward={step.reward:.3f}")
    
    def get_trajectory_batch(
        self, 
        batch_size: int, 
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None
    ) -> TrajectoryBatch:
        """
        Get batch of trajectory steps for GRPO training.
        
        Args:
            batch_size: Maximum number of steps to return
            start_idx: Starting index (default: random sampling)
            end_idx: Ending index (default: from start_idx + batch_size)
            
        Returns:
            List of trajectory steps for training
        """
        if not self._trajectory_steps:
            return []
        
        total_steps = len(self._trajectory_steps)
        
        if start_idx is None:
            # Random sampling for better training dynamics
            if batch_size >= total_steps:
                return self._trajectory_steps.copy()
            else:
                # Sample without replacement
                import random
                indices = random.sample(range(total_steps), batch_size)
                return [self._trajectory_steps[i] for i in sorted(indices)]
        else:
            # Sequential sampling
            start_idx = max(0, start_idx)
            if end_idx is None:
                end_idx = min(start_idx + batch_size, total_steps)
            else:
                end_idx = min(end_idx, total_steps)
            
            return self._trajectory_steps[start_idx:end_idx]
    
    def get_recent_trajectories(self, n: int) -> TrajectoryBatch:
        """Get the n most recent trajectory steps."""
        if n <= 0:
            return []
        return self._trajectory_steps[-n:]
    
    def get_high_reward_trajectories(self, top_k: int = 10) -> TrajectoryBatch:
        """Get the top-k highest reward trajectory steps."""
        if not self._trajectory_steps:
            return []
        
        # Sort by reward descending
        sorted_steps = sorted(self._trajectory_steps, key=lambda s: s.reward, reverse=True)
        return sorted_steps[:top_k]
    
    def get_trajectories_by_target(self, target_variable: str) -> TrajectoryBatch:
        """Get all trajectory steps for a specific target variable."""
        if target_variable in self._steps_by_target:
            indices = self._steps_by_target[target_variable]
            return [self._trajectory_steps[i] for i in indices]
        return []
    
    # === Causal Discovery Interface (Delegated to ExperienceBuffer) ===
    
    def get_interventions(self) -> List[Tuple[InterventionSpec, Sample]]:
        """Delegate to underlying experience buffer for causal discovery."""
        return self._experience_buffer.get_interventions()
    
    def get_observations(self) -> List[Sample]:
        """Delegate to underlying experience buffer for causal discovery."""
        return self._experience_buffer.get_observations()
    
    def get_all_samples(self) -> List[Sample]:
        """Delegate to underlying experience buffer for causal discovery."""
        return self._experience_buffer.get_all_samples()
    
    def get_variable_coverage(self) -> frozenset[str]:
        """Delegate to underlying experience buffer."""
        return self._experience_buffer.get_variable_coverage()
    
    def get_statistics(self) -> BufferStatistics:
        """Get statistics from underlying experience buffer."""
        return self._experience_buffer.get_statistics()
    
    # === Analysis Interface ===
    
    def get_reward_history(self) -> List[float]:
        """Get reward progression for analysis."""
        return [step.reward for step in self._trajectory_steps]
    
    def get_state_progression(self) -> List[AcquisitionState]:
        """Get state evolution throughout training."""
        return [step.state for step in self._trajectory_steps]
    
    def get_uncertainty_progression(self) -> List[float]:
        """Get uncertainty progression (bits) throughout training."""
        return [step.state.uncertainty_bits for step in self._trajectory_steps]
    
    def get_target_value_progression(self) -> List[float]:
        """Get best target value progression throughout training."""
        return [step.state.best_value for step in self._trajectory_steps]
    
    def analyze_intervention_effectiveness(self) -> Dict[str, Any]:
        """
        Analyze which interventions led to highest rewards.
        
        Returns:
            Dictionary with intervention effectiveness analysis
        """
        if not self._trajectory_steps:
            return {'error': 'No trajectory steps available'}
        
        # Group by intervention type
        by_type = {}
        for step in self._trajectory_steps:
            int_type = step.intervention.get('type', 'unknown')
            if int_type not in by_type:
                by_type[int_type] = []
            by_type[int_type].append(step.reward)
        
        # Compute statistics by type
        type_stats = {}
        for int_type, rewards in by_type.items():
            rewards_array = jnp.array(rewards)
            type_stats[int_type] = {
                'count': len(rewards),
                'mean_reward': float(jnp.mean(rewards_array)),
                'std_reward': float(jnp.std(rewards_array)),
                'max_reward': float(jnp.max(rewards_array)),
                'min_reward': float(jnp.min(rewards_array))
            }
        
        # Group by intervention targets
        by_targets = {}
        for step in self._trajectory_steps:
            targets = frozenset(step.intervention.get('targets', set()))
            target_key = ', '.join(sorted(targets)) if targets else 'none'
            if target_key not in by_targets:
                by_targets[target_key] = []
            by_targets[target_key].append(step.reward)
        
        # Compute statistics by targets
        target_stats = {}
        for target_key, rewards in by_targets.items():
            rewards_array = jnp.array(rewards)
            target_stats[target_key] = {
                'count': len(rewards),
                'mean_reward': float(jnp.mean(rewards_array)),
                'std_reward': float(jnp.std(rewards_array))
            }
        
        # Overall statistics
        all_rewards = jnp.array(self.get_reward_history())
        
        return {
            'total_steps': len(self._trajectory_steps),
            'overall_stats': {
                'mean_reward': float(jnp.mean(all_rewards)),
                'std_reward': float(jnp.std(all_rewards)),
                'max_reward': float(jnp.max(all_rewards)),
                'min_reward': float(jnp.min(all_rewards)),
                'reward_trend': float(jnp.mean(all_rewards[-10:]) - jnp.mean(all_rewards[:10])) if len(all_rewards) >= 20 else 0.0
            },
            'by_intervention_type': type_stats,
            'by_intervention_targets': target_stats,
        }
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        experience_stats = self.get_statistics()
        
        return {
            # Trajectory-specific info
            'trajectory_steps': len(self._trajectory_steps),
            'creation_time': time.ctime(self._creation_time),
            'last_trajectory_time': time.ctime(self._last_trajectory_time),
            
            # Experience buffer info (delegated)
            'total_samples': experience_stats.total_samples,
            'observational_samples': experience_stats.num_observations,
            'interventional_samples': experience_stats.num_interventions,
            'unique_variables': experience_stats.unique_variables,
            
            # Training effectiveness
            'intervention_effectiveness': self.analyze_intervention_effectiveness(),
            
            # Data coverage
            'variable_coverage': sorted(self.get_variable_coverage()),
            'target_variables_used': sorted(self._steps_by_target.keys()),
        }
    
    # === Batch Processing for RL Training ===
    
    def batch_iterator(
        self, 
        batch_size: int, 
        shuffle: bool = True,
        include_partial: bool = True
    ) -> Iterator[TrajectoryBatch]:
        """
        Iterate over trajectory steps in batches for training.
        
        Args:
            batch_size: Size of each batch
            shuffle: Whether to shuffle the data
            include_partial: Whether to include final partial batch
            
        Yields:
            Batches of trajectory steps
        """
        steps = self._trajectory_steps.copy()
        
        if shuffle:
            import random
            random.shuffle(steps)
        
        for i in range(0, len(steps), batch_size):
            batch = steps[i:i + batch_size]
            if len(batch) == batch_size or include_partial:
                yield batch
    
    # === Internal Helper Methods ===
    
    def _update_indices(self, step: TrajectoryStep, step_index: int) -> None:
        """Update internal indices for efficient querying."""
        # Index by reward (for high-reward queries)
        reward_bucket = round(step.reward, 2)  # Bucket by reward to nearest 0.01
        if reward_bucket not in self._steps_by_reward:
            self._steps_by_reward[reward_bucket] = []
        self._steps_by_reward[reward_bucket].append(step_index)
        
        # Index by target variable
        target = step.state.current_target
        if target not in self._steps_by_target:
            self._steps_by_target[target] = []
        self._steps_by_target[target].append(step_index)
    
    # === Size and Status ===
    
    def size(self) -> int:
        """Get total number of trajectory steps."""
        return len(self._trajectory_steps)
    
    def is_empty(self) -> bool:
        """Check if trajectory buffer is empty."""
        return len(self._trajectory_steps) == 0
    
    def __len__(self) -> int:
        """Support len() operation."""
        return self.size()
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"TrajectoryBuffer(trajectory_steps={len(self._trajectory_steps)}, "
            f"experience_samples={self._experience_buffer.size()})"
        )


# === Factory Functions ===

def create_trajectory_buffer(
    initial_experience_buffer: Optional[ExperienceBuffer] = None
) -> TrajectoryBuffer:
    """
    Create trajectory buffer, optionally composing with existing experience buffer.
    
    Args:
        initial_experience_buffer: Existing buffer to compose with
        
    Returns:
        New TrajectoryBuffer ready for RL training
    """
    return TrajectoryBuffer(initial_experience_buffer)


def create_trajectory_step(
    state: AcquisitionState,
    intervention: InterventionSpec,
    outcome: Sample,
    reward: float,
    next_state: AcquisitionState,
    training_metadata: Optional[Dict[str, Any]] = None
) -> TrajectoryStep:
    """
    Factory function for creating trajectory steps.
    
    Args:
        state: State when intervention was chosen
        intervention: Intervention that was applied
        outcome: Observed outcome from intervention
        reward: Computed reward for this step
        next_state: State after applying intervention
        training_metadata: Optional training context
        
    Returns:
        Validated TrajectoryStep ready for storage
    """
    if training_metadata is None:
        training_metadata = {}
    
    return TrajectoryStep(
        state=state,
        intervention=intervention,
        outcome=outcome,
        reward=reward,
        next_state=next_state,
        step_index=state.step,
        training_metadata=pyr.pmap(training_metadata)
    )


def migrate_experience_to_trajectory_buffer(
    experience_buffer: ExperienceBuffer,
    default_reward: float = 0.0,
    create_mock_states: bool = True
) -> TrajectoryBuffer:
    """
    Convert existing ExperienceBuffer to TrajectoryBuffer for testing.
    
    Creates mock trajectory steps from intervention-outcome pairs.
    Useful for transitioning existing data to new trajectory format.
    
    Args:
        experience_buffer: Existing experience buffer to migrate
        default_reward: Default reward to assign to existing steps
        create_mock_states: Whether to create mock states or skip
        
    Returns:
        TrajectoryBuffer with migrated data
        
    Note:
        This is primarily for testing/migration. Real trajectory steps
        should be created during actual RL training with proper states.
    """
    trajectory_buffer = TrajectoryBuffer(experience_buffer)
    
    if not create_mock_states:
        logger.info("Created trajectory buffer without mock states")
        return trajectory_buffer
    
    # Create mock trajectory steps from existing interventions
    interventions = experience_buffer.get_interventions()
    
    for i, (intervention, outcome) in enumerate(interventions):
        # Create minimal mock states for testing
        # In real usage, these would come from actual training
        try:
            from .services import create_acquisition_state
            
            # Mock SCM for state creation
            mock_scm = pyr.pmap({
                'variables': experience_buffer.get_variable_coverage(),
                'target': list(experience_buffer.get_variable_coverage())[0]  # Use first variable
            })
            
            # Create mock states
            mock_state = create_acquisition_state(
                scm=mock_scm,
                buffer=experience_buffer,
                surrogate_model=None,  # Will use fallback
                surrogate_params=None,
                target_variable=mock_scm['target'],
                step=i
            )
            
            mock_next_state = create_acquisition_state(
                scm=mock_scm,
                buffer=experience_buffer,
                surrogate_model=None,
                surrogate_params=None,
                target_variable=mock_scm['target'],
                step=i + 1
            )
            
            # Create mock trajectory step
            mock_step = create_trajectory_step(
                state=mock_state,
                intervention=intervention,
                outcome=outcome,
                reward=default_reward,
                next_state=mock_next_state,
                training_metadata={'migrated': True, 'original_index': i}
            )
            
            trajectory_buffer.add_trajectory_step(mock_step)
            
        except Exception as e:
            logger.warning(f"Could not create mock trajectory step {i}: {e}")
            continue
    
    logger.info(f"Migrated {len(interventions)} interventions to {trajectory_buffer.size()} trajectory steps")
    return trajectory_buffer
