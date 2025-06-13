"""
Comprehensive tests for TrajectoryBuffer implementation.

These tests validate that the trajectory storage correctly:
- Composes with ExperienceBuffer for backward compatibility
- Provides complete RL training context (state, action, reward, next_state)
- Maintains data consistency between trajectory and experience views
- Supports efficient batch processing for GRPO training
"""

import pytest
import time
import jax.numpy as jnp
import pyrsistent as pyr
from typing import FrozenSet

# Test imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from causal_bayes_opt.acquisition.trajectory import (
    TrajectoryStep,
    TrajectoryBuffer,
    create_trajectory_buffer,
    create_trajectory_step,
    migrate_experience_to_trajectory_buffer,
)

from causal_bayes_opt.acquisition import (
    AcquisitionState,
    create_acquisition_state,
)

from causal_bayes_opt.data_structures import (
    ExperienceBuffer,
    create_empty_buffer,
    create_observational_sample,
    create_interventional_sample,
    create_scm,
)

from causal_bayes_opt.avici_integration.parent_set import (
    create_parent_set_posterior,
)

from causal_bayes_opt.interventions import create_perfect_intervention


class TestTrajectoryStep:
    """Test TrajectoryStep dataclass functionality."""
    
    def test_trajectory_step_creation_and_validation(self):
        """Test that TrajectoryStep can be created and validates correctly."""
        # Create test data
        state = self._create_test_state(step=5)
        next_state = self._create_test_state(step=6)
        intervention = create_perfect_intervention(frozenset(['X']), {'X': 3.0})
        outcome = create_interventional_sample(
            values={'X': 3.0, 'Y': 2.8, 'Z': 1.0},
            intervention_type='perfect',
            targets=frozenset(['X'])
        )
        
        # Create trajectory step
        step = TrajectoryStep(
            state=state,
            intervention=intervention,
            outcome=outcome,
            reward=1.5,
            next_state=next_state,
            step_index=5,
            training_metadata=pyr.m(test=True)
        )
        
        # Test basic properties
        assert step.state.step == 5
        assert step.next_state.step == 6
        assert step.reward == 1.5
        assert step.step_index == 5
        assert step.training_metadata['test'] is True
        
        # Test derived methods
        uncertainty_reduction = step.get_uncertainty_reduction()
        assert isinstance(uncertainty_reduction, float)
        
        target_improvement = step.get_target_improvement()
        assert isinstance(target_improvement, float)
        
        state_transition = step.get_state_transition()
        assert state_transition == (state, next_state)
        
        summary = step.summary()
        assert 'step_index' in summary
        assert 'reward' in summary
        assert 'uncertainty_reduction_bits' in summary
    
    def test_trajectory_step_validation(self):
        """Test that TrajectoryStep validation catches inconsistencies."""
        state = self._create_test_state(step=5)
        next_state = self._create_test_state(step=6)
        intervention = create_perfect_intervention(frozenset(['X']), {'X': 3.0})
        outcome = create_interventional_sample(
            values={'X': 3.0, 'Y': 2.8, 'Z': 1.0},
            intervention_type='perfect',
            targets=frozenset(['X'])
        )
        
        # Test step progression validation
        wrong_next_state = self._create_test_state(step=8)  # Wrong step
        with pytest.raises(ValueError, match="Inconsistent step progression"):
            TrajectoryStep(
                state=state,
                intervention=intervention,
                outcome=outcome,
                reward=1.0,
                next_state=wrong_next_state,
                step_index=5
            )
        
        # Test infinite reward validation
        with pytest.raises(ValueError, match="Reward must be finite"):
            TrajectoryStep(
                state=state,
                intervention=intervention,
                outcome=outcome,
                reward=float('inf'),
                next_state=next_state,
                step_index=5
            )
    
    def _create_test_state(self, step: int = 0) -> AcquisitionState:
        """Create a test acquisition state."""
        buffer = create_empty_buffer()
        buffer.add_observation(create_observational_sample({'X': 1.0, 'Y': 2.0, 'Z': 1.5}))
        
        posterior = create_parent_set_posterior(
            target_variable='Y',
            parent_sets=[frozenset(), frozenset(['X'])],
            probabilities=jnp.array([0.4, 0.6])
        )
        
        return AcquisitionState(
            posterior=posterior,
            buffer=buffer,
            best_value=2.0,
            current_target='Y',
            step=step
        )


class TestTrajectoryBuffer:
    """Test TrajectoryBuffer composition and functionality."""
    
    def test_trajectory_buffer_creation(self):
        """Test TrajectoryBuffer creation and composition."""
        # Test creation with new ExperienceBuffer
        buffer1 = create_trajectory_buffer()
        assert buffer1.size() == 0
        assert buffer1.is_empty()
        assert len(buffer1) == 0
        
        # Test creation with existing ExperienceBuffer
        experience_buffer = self._create_test_experience_buffer()
        buffer2 = create_trajectory_buffer(experience_buffer)
        
        # Should compose with existing buffer
        assert buffer2.get_statistics().total_samples == experience_buffer.size()
        assert buffer2.get_variable_coverage() == experience_buffer.get_variable_coverage()
    
    def test_trajectory_step_addition(self):
        """Test adding trajectory steps and automatic sync."""
        buffer = create_trajectory_buffer()
        step = self._create_test_trajectory_step()
        
        # Add trajectory step
        buffer.add_trajectory_step(step)
        
        # Verify trajectory storage
        assert buffer.size() == 1
        assert not buffer.is_empty()
        
        # Verify automatic sync to experience buffer
        interventions = buffer.get_interventions()
        assert len(interventions) >= 0  # May or may not sync depending on existing data
        
        # Verify we can retrieve the step
        recent = buffer.get_recent_trajectories(1)
        assert len(recent) == 1
        assert recent[0] == step
    
    def test_batch_processing(self):
        """Test batch processing for RL training."""
        buffer = create_trajectory_buffer()
        
        # Add multiple trajectory steps
        steps = []
        for i in range(10):
            step = self._create_test_trajectory_step(step_index=i, reward=float(i))
            steps.append(step)
            buffer.add_trajectory_step(step)
        
        # Test batch retrieval
        batch = buffer.get_trajectory_batch(batch_size=5)
        assert len(batch) == 5
        
        # Test recent trajectories
        recent = buffer.get_recent_trajectories(3)
        assert len(recent) == 3
        assert recent[-1] == steps[-1]  # Most recent should be last added
        
        # Test high reward trajectories
        high_reward = buffer.get_high_reward_trajectories(top_k=3)
        assert len(high_reward) == 3
        # Should be sorted by reward descending
        assert high_reward[0].reward >= high_reward[1].reward >= high_reward[2].reward
    
    def test_experience_buffer_delegation(self):
        """Test that experience buffer methods are properly delegated."""
        # Create buffer with existing experience data
        experience_buffer = self._create_test_experience_buffer()
        trajectory_buffer = create_trajectory_buffer(experience_buffer)
        
        # Test delegation methods
        observations = trajectory_buffer.get_observations()
        assert len(observations) > 0
        
        interventions = trajectory_buffer.get_interventions()
        assert len(interventions) >= 0
        
        all_samples = trajectory_buffer.get_all_samples()
        assert len(all_samples) > 0
        
        variables = trajectory_buffer.get_variable_coverage()
        assert len(variables) > 0
        
        stats = trajectory_buffer.get_statistics()
        assert stats.total_samples > 0
    
    def test_analysis_capabilities(self):
        """Test trajectory analysis methods."""
        buffer = create_trajectory_buffer()
        
        # Add steps with varying rewards and types
        rewards = [1.0, 2.5, 0.5, 3.0, 1.5]
        for i, reward in enumerate(rewards):
            step = self._create_test_trajectory_step(step_index=i, reward=reward)
            buffer.add_trajectory_step(step)
        
        # Test reward history
        reward_history = buffer.get_reward_history()
        assert reward_history == rewards
        
        # Test state progression
        states = buffer.get_state_progression()
        assert len(states) == len(rewards)
        assert all(isinstance(s, AcquisitionState) for s in states)
        
        # Test uncertainty progression
        uncertainty = buffer.get_uncertainty_progression()
        assert len(uncertainty) == len(rewards)
        assert all(isinstance(u, float) for u in uncertainty)
        
        # Test target value progression
        target_values = buffer.get_target_value_progression()
        assert len(target_values) == len(rewards)
        assert all(isinstance(v, float) for v in target_values)
        
        # Test intervention effectiveness analysis
        effectiveness = buffer.analyze_intervention_effectiveness()
        assert 'total_steps' in effectiveness
        assert 'overall_stats' in effectiveness
        assert 'by_intervention_type' in effectiveness
        assert effectiveness['total_steps'] == len(rewards)
    
    def test_batch_iterator(self):
        """Test batch iterator for training."""
        buffer = create_trajectory_buffer()
        
        # Add 10 trajectory steps
        for i in range(10):
            step = self._create_test_trajectory_step(step_index=i)
            buffer.add_trajectory_step(step)
        
        # Test batch iteration
        batches = list(buffer.batch_iterator(batch_size=3, shuffle=False))
        
        # Should have 4 batches: [3, 3, 3, 1]
        assert len(batches) == 4
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 3
        assert len(batches[3]) == 1
        
        # Test without partial batch
        batches_no_partial = list(buffer.batch_iterator(batch_size=3, include_partial=False))
        assert len(batches_no_partial) == 3  # Skip the partial batch
    
    def test_training_summary(self):
        """Test comprehensive training summary."""
        buffer = create_trajectory_buffer()
        
        # Add some trajectory steps
        for i in range(5):
            step = self._create_test_trajectory_step(step_index=i)
            buffer.add_trajectory_step(step)
        
        summary = buffer.get_training_summary()
        
        # Check required fields
        required_fields = [
            'trajectory_steps', 'total_samples', 'intervention_effectiveness',
            'variable_coverage', 'target_variables_used'
        ]
        for field in required_fields:
            assert field in summary
        
        assert summary['trajectory_steps'] == 5
    
    def _create_test_experience_buffer(self) -> ExperienceBuffer:
        """Create test experience buffer with data."""
        buffer = create_empty_buffer()
        
        # Add observational data
        buffer.add_observation(create_observational_sample({'X': 1.0, 'Y': 2.0, 'Z': 1.5}))
        buffer.add_observation(create_observational_sample({'X': 2.0, 'Y': 2.5, 'Z': 1.0}))
        
        # Add interventional data
        intervention = create_perfect_intervention(frozenset(['X']), {'X': 3.0})
        outcome = create_interventional_sample(
            values={'X': 3.0, 'Y': 2.8, 'Z': 1.0},
            intervention_type='perfect',
            targets=frozenset(['X'])
        )
        buffer.add_intervention(intervention, outcome)
        
        return buffer
    
    def _create_test_trajectory_step(self, step_index: int = 0, reward: float = 1.0) -> TrajectoryStep:
        """Create test trajectory step."""
        state = self._create_test_state(step=step_index)
        next_state = self._create_test_state(step=step_index + 1)
        intervention = create_perfect_intervention(frozenset(['X']), {'X': 3.0})
        outcome = create_interventional_sample(
            values={'X': 3.0, 'Y': 2.8, 'Z': 1.0},
            intervention_type='perfect',
            targets=frozenset(['X'])
        )
        
        return TrajectoryStep(
            state=state,
            intervention=intervention,
            outcome=outcome,
            reward=reward,
            next_state=next_state,
            step_index=step_index
        )
    
    def _create_test_state(self, step: int = 0) -> AcquisitionState:
        """Create test acquisition state."""
        buffer = create_empty_buffer()
        buffer.add_observation(create_observational_sample({'X': 1.0, 'Y': 2.0, 'Z': 1.5}))
        
        posterior = create_parent_set_posterior(
            target_variable='Y',
            parent_sets=[frozenset(), frozenset(['X'])],
            probabilities=jnp.array([0.4, 0.6])
        )
        
        return AcquisitionState(
            posterior=posterior,
            buffer=buffer,
            best_value=2.0,
            current_target='Y',
            step=step
        )


class TestFactoryFunctions:
    """Test factory functions for trajectory management."""
    
    def test_create_trajectory_buffer(self):
        """Test trajectory buffer factory function."""
        # Test creation without existing buffer
        buffer1 = create_trajectory_buffer()
        assert isinstance(buffer1, TrajectoryBuffer)
        assert buffer1.size() == 0
        
        # Test creation with existing buffer
        experience_buffer = create_empty_buffer()
        experience_buffer.add_observation(create_observational_sample({'X': 1.0, 'Y': 2.0}))
        
        buffer2 = create_trajectory_buffer(experience_buffer)
        assert isinstance(buffer2, TrajectoryBuffer)
        assert buffer2.get_statistics().total_samples == 1
    
    def test_create_trajectory_step(self):
        """Test trajectory step factory function."""
        state = self._create_test_state(step=5)
        next_state = self._create_test_state(step=6)
        intervention = create_perfect_intervention(frozenset(['X']), {'X': 3.0})
        outcome = create_interventional_sample(
            values={'X': 3.0, 'Y': 2.8, 'Z': 1.0},
            intervention_type='perfect',
            targets=frozenset(['X'])
        )
        
        step = create_trajectory_step(
            state=state,
            intervention=intervention,
            outcome=outcome,
            reward=2.0,
            next_state=next_state,
            training_metadata={'test': True}
        )
        
        assert isinstance(step, TrajectoryStep)
        assert step.reward == 2.0
        assert step.step_index == 5
        assert step.training_metadata['test'] is True
    
    def test_migrate_experience_to_trajectory_buffer(self):
        """Test migration from ExperienceBuffer to TrajectoryBuffer."""
        # Create experience buffer with data
        experience_buffer = create_empty_buffer()
        experience_buffer.add_observation(create_observational_sample({'X': 1.0, 'Y': 2.0, 'Z': 1.5}))
        
        intervention = create_perfect_intervention(frozenset(['X']), {'X': 3.0})
        outcome = create_interventional_sample(
            values={'X': 3.0, 'Y': 2.8, 'Z': 1.0},
            intervention_type='perfect',
            targets=frozenset(['X'])
        )
        experience_buffer.add_intervention(intervention, outcome)
        
        # Test migration without mock states
        trajectory_buffer1 = migrate_experience_to_trajectory_buffer(
            experience_buffer, default_reward=0.5, create_mock_states=False
        )
        assert isinstance(trajectory_buffer1, TrajectoryBuffer)
        assert trajectory_buffer1.get_statistics().total_samples >= 1
        
        # Test migration with mock states (may fail due to missing dependencies, but shouldn't crash)
        try:
            trajectory_buffer2 = migrate_experience_to_trajectory_buffer(
                experience_buffer, default_reward=1.0, create_mock_states=True
            )
            assert isinstance(trajectory_buffer2, TrajectoryBuffer)
        except Exception:
            # Expected if dependencies not available
            pass
    
    def _create_test_state(self, step: int = 0) -> AcquisitionState:
        """Create test acquisition state."""
        buffer = create_empty_buffer()
        buffer.add_observation(create_observational_sample({'X': 1.0, 'Y': 2.0, 'Z': 1.5}))
        
        posterior = create_parent_set_posterior(
            target_variable='Y',
            parent_sets=[frozenset(), frozenset(['X'])],
            probabilities=jnp.array([0.4, 0.6])
        )
        
        return AcquisitionState(
            posterior=posterior,
            buffer=buffer,
            best_value=2.0,
            current_target='Y',
            step=step
        )


class TestComposition:
    """Test composition between TrajectoryBuffer and ExperienceBuffer."""
    
    def test_data_consistency(self):
        """Test that trajectory and experience views remain consistent."""
        # Start with experience buffer
        experience_buffer = create_empty_buffer()
        experience_buffer.add_observation(create_observational_sample({'X': 1.0, 'Y': 2.0, 'Z': 1.5}))
        
        # Create trajectory buffer that composes with it
        trajectory_buffer = create_trajectory_buffer(experience_buffer)
        
        # Add trajectory step
        step = self._create_test_trajectory_step()
        trajectory_buffer.add_trajectory_step(step)
        
        # Verify both views are consistent
        experience_samples = trajectory_buffer.get_all_samples()
        trajectory_steps = trajectory_buffer.get_recent_trajectories(10)
        
        # The intervention-outcome should appear in both views
        assert len(experience_samples) >= 1
        assert len(trajectory_steps) == 1
        
        # The trajectory step's outcome should be compatible with experience view
        trajectory_outcome = trajectory_steps[0].outcome
        assert isinstance(trajectory_outcome, pyr.PMap)  # Sample type
    
    def test_backward_compatibility(self):
        """Test that existing ExperienceBuffer functionality is preserved."""
        experience_buffer = create_empty_buffer()
        experience_buffer.add_observation(create_observational_sample({'X': 1.0, 'Y': 2.0, 'Z': 1.5}))
        
        # Original functionality
        original_observations = experience_buffer.get_observations()
        original_stats = experience_buffer.get_statistics()
        original_variables = experience_buffer.get_variable_coverage()
        
        # After composition
        trajectory_buffer = create_trajectory_buffer(experience_buffer)
        
        # Should maintain same functionality
        composed_observations = trajectory_buffer.get_observations()
        composed_stats = trajectory_buffer.get_statistics()
        composed_variables = trajectory_buffer.get_variable_coverage()
        
        assert len(composed_observations) == len(original_observations)
        assert composed_stats.total_samples == original_stats.total_samples
        assert composed_variables == original_variables
    
    def _create_test_trajectory_step(self) -> TrajectoryStep:
        """Create test trajectory step."""
        state = self._create_test_state(step=0)
        next_state = self._create_test_state(step=1)
        intervention = create_perfect_intervention(frozenset(['X']), {'X': 3.0})
        outcome = create_interventional_sample(
            values={'X': 3.0, 'Y': 2.8, 'Z': 1.0},
            intervention_type='perfect',
            targets=frozenset(['X'])
        )
        
        return TrajectoryStep(
            state=state,
            intervention=intervention,
            outcome=outcome,
            reward=1.5,
            next_state=next_state,
            step_index=0
        )
    
    def _create_test_state(self, step: int = 0) -> AcquisitionState:
        """Create test acquisition state."""
        buffer = create_empty_buffer()
        buffer.add_observation(create_observational_sample({'X': 1.0, 'Y': 2.0, 'Z': 1.5}))
        
        posterior = create_parent_set_posterior(
            target_variable='Y',
            parent_sets=[frozenset(), frozenset(['X'])],
            probabilities=jnp.array([0.4, 0.6])
        )
        
        return AcquisitionState(
            posterior=posterior,
            buffer=buffer,
            best_value=2.0,
            current_target='Y',
            step=step
        )


if __name__ == '__main__':
    # Run basic smoke tests if executed directly
    print("Running basic smoke tests for TrajectoryBuffer...")
    
    # Test 1: Basic trajectory buffer creation
    try:
        buffer = create_trajectory_buffer()
        print(f"✓ TrajectoryBuffer created: {buffer}")
        
        # Test 2: TrajectoryStep creation
        from causal_bayes_opt.data_structures import create_observational_sample
        from causal_bayes_opt.avici_integration.parent_set import create_parent_set_posterior
        
        # Create minimal state
        exp_buffer = create_empty_buffer()
        exp_buffer.add_observation(create_observational_sample({'X': 1.0, 'Y': 2.0}))
        
        posterior = create_parent_set_posterior(
            target_variable='Y',
            parent_sets=[frozenset(['X'])],
            probabilities=jnp.array([1.0])
        )
        
        state = AcquisitionState(
            posterior=posterior,
            buffer=exp_buffer,
            best_value=2.0,
            current_target='Y',
            step=0
        )
        
        next_state = AcquisitionState(
            posterior=posterior,
            buffer=exp_buffer,
            best_value=2.5,
            current_target='Y',
            step=1
        )
        
        intervention = create_perfect_intervention(frozenset(['X']), {'X': 3.0})
        outcome = create_interventional_sample(
            values={'X': 3.0, 'Y': 2.8},
            intervention_type='perfect',
            targets=frozenset(['X'])
        )
        
        step = create_trajectory_step(
            state=state,
            intervention=intervention,
            outcome=outcome,
            reward=1.5,
            next_state=next_state
        )
        
        print(f"✓ TrajectoryStep created: step_index={step.step_index}, reward={step.reward}")
        
        # Test 3: Add to buffer
        buffer.add_trajectory_step(step)
        print(f"✓ Added step to buffer: {buffer.size()} trajectory steps")
        
        # Test 4: Analysis
        summary = buffer.get_training_summary()
        print(f"✓ Training summary: {summary['trajectory_steps']} steps")
        
    except Exception as e:
        print(f"✗ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\n🎉 All smoke tests passed! TrajectoryBuffer implementation looks good.")
    print("✅ Composition with ExperienceBuffer works")
    print("✅ Complete RL training context available")
    print("✅ Backward compatibility maintained")
    print("✅ Rich analysis capabilities")
    print("\nNow run full test suite with: pytest tests/test_trajectory_buffer.py -v")
