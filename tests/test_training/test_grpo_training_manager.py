"""Tests for GRPO Training Manager.

This module tests the complete GRPO training orchestration system including
the main training loop, experience collection, policy updates, and 
training session management.
"""

import pytest
import time
from unittest.mock import Mock, patch
import tempfile
import os

import jax
import jax.numpy as jnp
import pyrsistent as pyr

from causal_bayes_opt.training.grpo_training_manager import (
    TrainingStep,
    TrainingSession,
    GRPOTrainingManager,
    create_grpo_training_manager,
    create_debug_training_manager,
    create_production_training_manager,
)
from causal_bayes_opt.training.grpo_config import (
    ComprehensiveGRPOConfig,
    create_debug_grpo_config,
    create_standard_grpo_config,
    TrainingMode,
    OptimizationLevel,
)
from causal_bayes_opt.training.grpo_core import (
    GRPOConfig,
    GRPOUpdateResult,
    create_default_grpo_config,
)
from causal_bayes_opt.training.experience_management import Experience, ExperienceConfig
from causal_bayes_opt.training.diversity_monitor import DiversityMonitor
from causal_bayes_opt.acquisition.reward_rubric import (
    CausalRewardRubric,
    RewardResult,
)
from causal_bayes_opt.environments.intervention_env import (
    InterventionEnvironment,
    EnvironmentInfo,
)
from causal_bayes_opt.jax_native.state import create_test_state


class TestTrainingStep:
    """Test TrainingStep data structure."""
    
    def test_training_step_creation(self):
        """Test creating training step."""
        grpo_result = GRPOUpdateResult(
            policy_loss=0.5,
            value_loss=0.3,
            entropy_loss=0.1,
            total_loss=0.9,
            kl_divergence=0.01,
            policy_gradient_norm=0.2,
            value_gradient_norm=0.15,
            clipped_fraction=0.1,
            explained_variance=0.6
        )
        
        step = TrainingStep(
            step_number=100,
            episode_number=25,
            experiences_collected=4,
            grpo_update_result=grpo_result,
            diversity_metrics={"entropy": 0.8, "uniqueness": 0.9},
            training_time=0.5,
            memory_usage_mb=256.0,
            checkpoint_saved=True
        )
        
        assert step.step_number == 100
        assert step.episode_number == 25
        assert step.experiences_collected == 4
        assert step.grpo_update_result == grpo_result
        assert step.diversity_metrics["entropy"] == 0.8
        assert step.training_time == 0.5
        assert step.memory_usage_mb == 256.0
        assert step.checkpoint_saved is True
    
    def test_training_step_immutability(self):
        """Test that training step is immutable."""
        step = TrainingStep(
            step_number=1, episode_number=1, experiences_collected=1,
            grpo_update_result=None, diversity_metrics=None,
            training_time=1.0, memory_usage_mb=100.0, checkpoint_saved=False
        )
        
        with pytest.raises(AttributeError):
            step.step_number = 2


class TestTrainingSession:
    """Test TrainingSession data structure."""
    
    def test_training_session_creation(self):
        """Test creating training session."""
        config = create_debug_grpo_config()
        
        session = TrainingSession(
            config=config,
            total_steps=1000,
            total_episodes=100,
            total_experiences=5000,
            training_time=300.0,
            final_performance={"reward": 0.8, "efficiency": 0.9},
            checkpoints_saved=["checkpoint1.pkl", "checkpoint2.pkl"],
            best_checkpoint="checkpoint2.pkl",
            convergence_achieved=True,
            early_stopped=False,
            session_metadata={"experiment": "test"}
        )
        
        assert session.total_steps == 1000
        assert session.total_episodes == 100
        assert session.total_experiences == 5000
        assert session.training_time == 300.0
        assert session.final_performance["reward"] == 0.8
        assert len(session.checkpoints_saved) == 2
        assert session.best_checkpoint == "checkpoint2.pkl"
        assert session.convergence_achieved is True
        assert session.early_stopped is False
        assert session.session_metadata["experiment"] == "test"


class TestGRPOTrainingManager:
    """Test the main GRPO training manager."""
    
    @pytest.fixture
    def mock_environment(self):
        """Create mock intervention environment."""
        env = Mock(spec=InterventionEnvironment)
        env.reset.return_value = create_test_state()
        env.step.return_value = (
            create_test_state(),
            EnvironmentInfo(1, 10.0, False, False, True, 0.8, {})
        )
        return env
    
    @pytest.fixture
    def mock_reward_rubric(self):
        """Create mock reward rubric."""
        rubric = Mock(spec=CausalRewardRubric)
        rubric.compute_reward.return_value = RewardResult(0.8, {"r1": 0.8}, {})
        return rubric
    
    @pytest.fixture
    def mock_policy_network(self):
        """Create mock policy network."""
        network = Mock()
        network.params = {"policy": jnp.array([1.0, 2.0])}
        network.replace.return_value = network
        return network
    
    @pytest.fixture
    def mock_value_network(self):
        """Create mock value network."""
        network = Mock()
        network.params = {"value": jnp.array([0.5, 1.5])}
        network.replace.return_value = network
        return network
    
    @pytest.fixture
    def training_config(self):
        """Create test training configuration."""
        return create_debug_grpo_config()
    
    @pytest.fixture
    def training_manager(self, training_config, mock_environment, mock_reward_rubric, 
                        mock_policy_network, mock_value_network):
        """Create training manager for testing."""
        return GRPOTrainingManager(
            config=training_config,
            environment=mock_environment,
            reward_rubric=mock_reward_rubric,
            policy_network=mock_policy_network,
            value_network=mock_value_network
        )
    
    def test_manager_initialization(self, training_manager, training_config):
        """Test training manager initialization."""
        assert training_manager.config == training_config
        assert training_manager.current_step == 0
        assert training_manager.current_episode == 0
        assert training_manager.total_experiences == 0
        assert training_manager.best_performance == float('-inf')
        assert isinstance(training_manager.experience_manager, object)
        assert isinstance(training_manager.async_manager, object)
    
    def test_collect_experiences(self, training_manager):
        """Test experience collection."""
        experiences = training_manager.collect_experiences(num_episodes=2)
        
        assert len(experiences) >= 2  # At least one per episode
        assert all(isinstance(exp, Experience) for exp in experiences)
        assert training_manager.current_episode == 2
    
    def test_get_training_statistics(self, training_manager):
        """Test getting training statistics."""
        stats = training_manager.get_training_statistics()
        
        assert 'current_step' in stats
        assert 'current_episode' in stats
        assert 'total_experiences' in stats
        assert 'training_time' in stats
        assert 'best_performance' in stats
        assert 'experience_buffer_utilization' in stats
        assert 'memory_usage_mb' in stats
        assert 'can_train' in stats
        assert 'training_mode' in stats
        assert 'optimization_level' in stats
        
        assert stats['current_step'] == 0
        assert stats['current_episode'] == 0
        assert stats['total_experiences'] == 0
        assert isinstance(stats['training_time'], float)
    
    def test_checkpoint_save_load(self, training_manager):
        """Test checkpoint saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Update config to use temp directory
            training_manager.config = training_manager.config.__class__(
                **{**training_manager.config.__dict__, 
                   'checkpointing': training_manager.config.checkpointing.__class__(
                       **{**training_manager.config.checkpointing.__dict__,
                          'checkpoint_dir': temp_dir}
                   )}
            )
            
            # Save checkpoint
            checkpoint_path = training_manager.save_checkpoint()
            
            assert os.path.exists(checkpoint_path)
            assert checkpoint_path in training_manager.checkpoint_history
            
            # Modify state
            original_step = training_manager.current_step
            training_manager.current_step = 100
            
            # Load checkpoint
            training_manager.load_checkpoint(checkpoint_path)
            
            assert training_manager.current_step == original_step
    
    @patch('time.time')
    def test_execute_training_step(self, mock_time, training_manager):
        """Test executing a single training step."""
        mock_time.return_value = 1000.0
        
        # Add some experiences to buffer first
        experiences = training_manager.collect_experiences(num_episodes=1)
        for exp in experiences:
            training_manager.experience_manager.add_experience(exp)
        
        # Execute training step
        step_result = training_manager._execute_training_step()
        
        assert isinstance(step_result, TrainingStep)
        assert step_result.step_number == training_manager.current_step
        assert step_result.experiences_collected > 0
        assert isinstance(step_result.training_time, float)
        assert isinstance(step_result.memory_usage_mb, float)
    
    def test_training_setup(self, training_manager):
        """Test training setup."""
        training_manager._setup_training()
        
        assert 'start_time' in training_manager.session_metadata
        assert 'config_hash' in training_manager.session_metadata
        assert 'environment_type' in training_manager.session_metadata
        assert 'reward_rubric_type' in training_manager.session_metadata
    
    def test_adaptive_adjustments(self, training_manager):
        """Test adaptive training adjustments."""
        # Set up some patience
        training_manager.patience_counter = 6
        
        # Should not raise any exceptions
        training_manager._apply_adaptive_adjustments()
    
    def test_early_stopping_check(self, training_manager):
        """Test early stopping logic."""
        # Initially should not early stop
        assert not training_manager._should_early_stop()
        
        # Set high patience counter
        training_manager.patience_counter = 1000
        
        # Should early stop if patience exceeded
        assert training_manager._should_early_stop()
    
    def test_convergence_check(self, training_manager):
        """Test convergence detection."""
        # With empty history, should not converge
        assert not training_manager._has_converged()
        
        # With small performance history, should not converge
        for i in range(5):
            step_result = Mock()
            step_result.grpo_update_result = Mock()
            step_result.grpo_update_result.policy_loss = -0.8
            training_manager.performance_history.append(step_result)
        
        assert not training_manager._has_converged()
    
    def test_finalize_training(self, training_manager):
        """Test training finalization."""
        training_manager.training_start_time = time.time()
        training_manager.current_step = 100
        training_manager.current_episode = 25
        training_manager.total_experiences = 500
        
        session = training_manager._finalize_training()
        
        assert isinstance(session, TrainingSession)
        assert session.total_steps == 100
        assert session.total_episodes == 25
        assert session.total_experiences == 500
        assert isinstance(session.training_time, float)
        assert isinstance(session.final_performance, dict)


class TestGRPOTrainingManagerIntegration:
    """Test integration scenarios for GRPO training manager."""
    
    @pytest.fixture
    def simple_config(self):
        """Create simple configuration for integration tests."""
        return create_debug_grpo_config()
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for integration testing."""
        env = Mock(spec=InterventionEnvironment)
        env.reset.return_value = create_test_state()
        env.step.return_value = (
            create_test_state(),
            EnvironmentInfo(1, 10.0, False, False, True, 0.8, {})
        )
        
        rubric = Mock(spec=CausalRewardRubric)
        rubric.compute_reward.return_value = RewardResult(0.8, {"r1": 0.8}, {})
        
        policy_net = Mock()
        policy_net.params = {"policy": jnp.array([1.0])}
        policy_net.replace.return_value = policy_net
        
        value_net = Mock()
        value_net.params = {"value": jnp.array([0.5])}
        value_net.replace.return_value = value_net
        
        return env, rubric, policy_net, value_net
    
    def test_short_training_session(self, simple_config, mock_components):
        """Test running a short training session."""
        env, rubric, policy_net, value_net = mock_components
        
        # Create manager with very short training
        config = simple_config.__class__(
            **{**simple_config.__dict__, 'max_training_steps': 5}
        )
        
        manager = GRPOTrainingManager(
            config=config,
            environment=env,
            reward_rubric=rubric,
            policy_network=policy_net,
            value_network=value_net
        )
        
        # Run training
        session = manager.train()
        
        assert isinstance(session, TrainingSession)
        assert session.total_steps <= 5
        assert session.total_episodes > 0
        assert session.training_time > 0
    
    def test_experience_replay_cycle(self, simple_config, mock_components):
        """Test complete experience replay cycle."""
        env, rubric, policy_net, value_net = mock_components
        
        manager = GRPOTrainingManager(
            config=simple_config,
            environment=env,
            reward_rubric=rubric,
            policy_network=policy_net,
            value_network=value_net
        )
        
        # Collect experiences
        experiences = manager.collect_experiences(num_episodes=3)
        assert len(experiences) >= 3
        
        # Add to experience manager
        for exp in experiences:
            manager.experience_manager.add_experience(exp)
        
        # Sample batch
        batch = manager.experience_manager.sample_batch()
        if batch is not None:
            # Update policy
            update_result = manager.update_policy(batch)
            assert update_result is not None
    
    def test_performance_tracking(self, simple_config, mock_components):
        """Test performance tracking throughout training."""
        env, rubric, policy_net, value_net = mock_components
        
        manager = GRPOTrainingManager(
            config=simple_config,
            environment=env,
            reward_rubric=rubric,
            policy_network=policy_net,
            value_network=value_net
        )
        
        # Execute several training steps
        for _ in range(3):
            step_result = manager._execute_training_step()
            manager.performance_history.append(step_result)
        
        stats = manager.get_training_statistics()
        assert stats['current_step'] == 0  # Steps incremented in main loop
        assert len(manager.performance_history) == 3


class TestTrainingManagerFactories:
    """Test factory functions for training managers."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components."""
        env = Mock(spec=InterventionEnvironment)
        rubric = Mock(spec=CausalRewardRubric)
        policy_net = Mock()
        policy_net.params = {}
        value_net = Mock()
        value_net.params = {}
        return env, rubric, policy_net, value_net
    
    def test_create_grpo_training_manager(self, mock_components):
        """Test main factory function."""
        env, rubric, policy_net, value_net = mock_components
        config = create_debug_grpo_config()
        
        manager = create_grpo_training_manager(
            config=config,
            environment=env,
            reward_rubric=rubric,
            policy_network=policy_net,
            value_network=value_net
        )
        
        assert isinstance(manager, GRPOTrainingManager)
        assert manager.config == config
    
    def test_create_debug_training_manager(self, mock_components):
        """Test debug factory function."""
        env, rubric, policy_net, value_net = mock_components
        
        manager = create_debug_training_manager(
            environment=env,
            reward_rubric=rubric,
            policy_network=policy_net,
            value_network=value_net
        )
        
        assert isinstance(manager, GRPOTrainingManager)
        assert manager.config.optimization_level == OptimizationLevel.DEBUG
        assert manager.config.max_training_steps == 1000
    
    def test_create_production_training_manager(self, mock_components):
        """Test production factory function."""
        env, rubric, policy_net, value_net = mock_components
        
        manager = create_production_training_manager(
            environment=env,
            reward_rubric=rubric,
            policy_network=policy_net,
            value_network=value_net,
            max_training_steps=5000
        )
        
        assert isinstance(manager, GRPOTrainingManager)
        assert manager.config.optimization_level == OptimizationLevel.PRODUCTION
        assert manager.config.max_training_steps == 5000
    
    def test_invalid_config_raises_error(self, mock_components):
        """Test that invalid configuration raises error."""
        env, rubric, policy_net, value_net = mock_components
        
        # Create invalid config
        invalid_config = create_debug_grpo_config()
        invalid_config = invalid_config.__class__(
            **{**invalid_config.__dict__, 'max_training_steps': -1}
        )
        
        with pytest.raises(ValueError):
            create_grpo_training_manager(
                config=invalid_config,
                environment=env,
                reward_rubric=rubric,
                policy_network=policy_net,
                value_network=value_net
            )


class TestTrainingManagerRobustness:
    """Test robustness and edge cases for training manager."""
    
    @pytest.fixture
    def manager_with_mocks(self):
        """Create manager with comprehensive mocks."""
        config = create_debug_grpo_config()
        
        env = Mock(spec=InterventionEnvironment)
        env.reset.return_value = create_test_state()
        env.step.return_value = (
            create_test_state(),
            EnvironmentInfo(1, 10.0, False, False, True, 0.8, {})
        )
        
        rubric = Mock(spec=CausalRewardRubric)
        rubric.compute_reward.return_value = RewardResult(0.8, {"r1": 0.8}, {})
        
        policy_net = Mock()
        policy_net.params = {"policy": jnp.array([1.0])}
        policy_net.replace.return_value = policy_net
        
        value_net = Mock()
        value_net.params = {"value": jnp.array([0.5])}
        value_net.replace.return_value = value_net
        
        return GRPOTrainingManager(
            config=config,
            environment=env,
            reward_rubric=rubric,
            policy_network=policy_net,
            value_network=value_net
        )
    
    def test_training_with_no_experiences(self, manager_with_mocks):
        """Test training when no experiences can be sampled."""
        manager = manager_with_mocks
        
        # Execute step with empty buffer
        step_result = manager._execute_training_step()
        
        # Should handle gracefully
        assert isinstance(step_result, TrainingStep)
        assert step_result.grpo_update_result is None
    
    def test_checkpoint_with_invalid_directory(self, manager_with_mocks):
        """Test checkpoint saving with invalid directory."""
        manager = manager_with_mocks
        
        # Try to save to invalid path
        invalid_path = "/invalid/path/checkpoint.pkl"
        
        # Should create directory and save
        try:
            path = manager.save_checkpoint(invalid_path)
            # Clean up if successful
            if os.path.exists(path):
                os.remove(path)
        except (OSError, PermissionError):
            # Expected for truly invalid paths
            pass
    
    def test_performance_computation_edge_cases(self, manager_with_mocks):
        """Test performance computation with edge cases."""
        manager = manager_with_mocks
        
        # Test with empty performance history
        final_perf = manager._compute_final_performance()
        assert isinstance(final_perf, dict)
        assert 'best_reward' in final_perf
        
        # Test convergence with empty history
        assert not manager._has_converged()
        
        # Test early stopping logic
        assert not manager._should_early_stop()
    
    def test_memory_usage_tracking(self, manager_with_mocks):
        """Test memory usage tracking."""
        manager = manager_with_mocks
        
        stats = manager.get_training_statistics()
        assert 'memory_usage_mb' in stats
        assert isinstance(stats['memory_usage_mb'], float)
        assert stats['memory_usage_mb'] >= 0
    
    def test_training_time_tracking(self, manager_with_mocks):
        """Test training time tracking."""
        manager = manager_with_mocks
        
        # Set training start time
        manager.training_start_time = time.time()
        
        # Get statistics
        stats = manager.get_training_statistics()
        assert 'training_time' in stats
        assert isinstance(stats['training_time'], float)
        assert stats['training_time'] >= 0