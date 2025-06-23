"""Integration Tests for Complete GRPO Training System.

This module tests the integration of all GRPO components including configuration,
experience management, training orchestration, and the complete training pipeline.

Key test areas:
- Component integration and data flow
- Configuration compatibility across subsystems
- Training pipeline end-to-end functionality
- Performance and memory efficiency
- Error handling and robustness
"""

import pytest
import time
import tempfile
import os
from unittest.mock import Mock, patch
from typing import List, Dict, Any

import jax
import jax.numpy as jnp
import pyrsistent as pyr

from causal_bayes_opt.training import (
    # GRPO Core
    GRPOConfig,
    GRPOTrajectory,
    GRPOUpdateResult,
    create_default_grpo_config,
    
    # Experience Management
    ExperienceConfig,
    Experience,
    ExperienceBatch,
    ExperienceManager,
    create_experience_manager,
    
    # Configuration System
    ComprehensiveGRPOConfig,
    TrainingMode,
    OptimizationLevel,
    create_debug_grpo_config,
    create_standard_grpo_config,
    create_production_grpo_config,
    validate_comprehensive_grpo_config,
    
    # Training Manager
    GRPOTrainingManager,
    TrainingStep,
    TrainingSession,
    create_grpo_training_manager,
    create_debug_training_manager,
    
    # Async Training and Diversity
    AsyncTrainingConfig,
    DiversityMonitor,
    create_diversity_monitor,
)

from causal_bayes_opt.acquisition.reward_rubric import (
    CausalRewardRubric,
    RewardResult,
    RewardComponent,
    create_training_rubric,
)

from causal_bayes_opt.environments.intervention_env import (
    InterventionEnvironment,
    EnvironmentInfo,
    EnvironmentConfig,
)

from causal_bayes_opt.jax_native.state import (
    JAXAcquisitionState,
    create_test_state,
)


class TestComponentIntegration:
    """Test integration between different GRPO components."""
    
    def test_config_subsystem_compatibility(self):
        """Test that all configuration subsystems are compatible."""
        # Create comprehensive config
        config = create_standard_grpo_config()
        
        # Validate configuration
        validate_comprehensive_grpo_config(config)
        
        # Test that all subsystem configs can be extracted
        assert isinstance(config.grpo_algorithm, GRPOConfig)
        assert isinstance(config.experience_management, ExperienceConfig)
        assert isinstance(config.async_training, AsyncTrainingConfig)
        
        # Test batch size compatibility
        async_batch = config.async_training.batch_size
        exp_batch = config.experience_management.batch_size
        assert async_batch == exp_batch, "Batch sizes must be compatible"
        
        # Test buffer size compatibility
        buffer_size = config.experience_management.max_buffer_size
        min_replay = config.experience_management.min_replay_size
        assert buffer_size > min_replay, "Buffer must be larger than min replay"
    
    def test_experience_manager_grpo_integration(self):
        """Test integration between experience manager and GRPO algorithm."""
        grpo_config = create_default_grpo_config()
        exp_manager = create_experience_manager(grpo_config)
        
        # Create sample experiences
        experiences = []
        for i in range(15):
            state = create_test_state()
            action = pyr.pmap({'X': float(i % 3)})
            reward = RewardResult(0.5 + i * 0.1, {'r1': 0.5}, {})
            env_info = EnvironmentInfo(i, 10.0, False, False, False, 0.8, {})
            
            exp = Experience(
                state=state,
                action=action,
                next_state=state,
                reward=reward,
                done=(i % 5 == 4),
                log_prob=-0.1 * i,
                value=0.6 + i * 0.05,
                env_info=env_info,
                timestamp=time.time()
            )
            experiences.append(exp)
            exp_manager.add_experience(exp)
        
        # Sample batch and verify GRPO trajectory creation
        batch = exp_manager.sample_batch()
        assert batch is not None
        assert isinstance(batch.trajectory, GRPOTrajectory)
        
        # Verify trajectory has correct structure
        trajectory = batch.trajectory
        assert trajectory.states.shape[0] == len(batch.experiences)
        assert trajectory.actions.shape[0] == len(batch.experiences)
        assert trajectory.rewards.shape == (len(batch.experiences),)
        assert trajectory.values.shape == (len(batch.experiences),)
        assert trajectory.advantages.shape == (len(batch.experiences),)
        assert trajectory.returns.shape == (len(batch.experiences),)
    
    def test_reward_rubric_environment_integration(self):
        """Test integration between reward rubric and environment."""
        # Create reward rubric
        rubric = create_training_rubric(
            components=[
                RewardComponent("structure_score", weight=0.6),
                RewardComponent("intervention_efficiency", weight=0.4)
            ]
        )
        
        # Mock environment
        env = Mock(spec=InterventionEnvironment)
        env.reset.return_value = create_test_state()
        env.step.return_value = (
            create_test_state(),
            EnvironmentInfo(1, 10.0, False, False, True, 0.8, {})
        )
        
        # Test reward computation in environment context
        state = create_test_state()
        action = pyr.pmap({'X': 1.0})
        next_state = create_test_state()
        
        reward_result = rubric.compute_reward(state, action, next_state)
        assert isinstance(reward_result, RewardResult)
        assert reward_result.total_reward >= 0
        assert len(reward_result.component_rewards) == 2
    
    def test_diversity_monitor_training_integration(self):
        """Test integration of diversity monitor with training."""
        # Create diversity monitor
        monitor = create_diversity_monitor()
        
        # Simulate training actions
        actions = [
            pyr.pmap({'X': 1.0}),
            pyr.pmap({'X': 2.0}),
            pyr.pmap({'Y': 1.5}),
            pyr.pmap({'X': 1.0}),  # Duplicate
        ]
        
        alerts = []
        for action in actions:
            alert = monitor.add_action(action)
            if alert:
                alerts.append(alert)
        
        # Get metrics
        metrics = monitor.get_metrics()
        assert metrics.total_actions == 4
        assert metrics.unique_actions <= 4
        assert 0 <= metrics.action_entropy <= 2.0  # Log2 of max unique actions
    
    def test_complete_training_manager_integration(self):
        """Test complete integration through training manager."""
        # Create mock components
        env = Mock(spec=InterventionEnvironment)
        env.reset.return_value = create_test_state()
        env.step.return_value = (
            create_test_state(),
            EnvironmentInfo(1, 10.0, False, False, True, 0.8, {})
        )
        
        rubric = Mock(spec=CausalRewardRubric)
        rubric.compute_reward.return_value = RewardResult(0.8, {'r1': 0.8}, {})
        
        policy_net = Mock()
        policy_net.params = {'policy': jnp.array([1.0])}
        policy_net.replace.return_value = policy_net
        
        value_net = Mock()
        value_net.params = {'value': jnp.array([0.5])}
        value_net.replace.return_value = value_net
        
        # Create training manager
        config = create_debug_grpo_config()
        manager = GRPOTrainingManager(
            config=config,
            environment=env,
            reward_rubric=rubric,
            policy_network=policy_net,
            value_network=value_net
        )
        
        # Test complete integration flow
        # 1. Collect experiences
        experiences = manager.collect_experiences(num_episodes=2)
        assert len(experiences) >= 2
        assert all(isinstance(exp, Experience) for exp in experiences)
        
        # 2. Add to experience manager
        for exp in experiences:
            manager.experience_manager.add_experience(exp)
        
        # 3. Sample batch
        batch = manager.experience_manager.sample_batch()
        if batch is not None:
            # 4. Update policy
            update_result = manager.update_policy(batch)
            assert isinstance(update_result, GRPOUpdateResult)
        
        # 5. Get statistics
        stats = manager.get_training_statistics()
        assert 'current_step' in stats
        assert 'total_experiences' in stats
        assert stats['total_experiences'] >= 2


class TestConfigurationIntegration:
    """Test configuration system integration."""
    
    def test_all_factory_configs_valid(self):
        """Test that all factory-created configs are valid."""
        configs = [
            create_debug_grpo_config(),
            create_standard_grpo_config(),
            create_production_grpo_config(),
        ]
        
        for config in configs:
            # Should not raise validation errors
            validate_comprehensive_grpo_config(config)
            
            # Test subsystem compatibility
            assert config.async_training.batch_size == config.experience_management.batch_size
            
            # Test reasonable defaults
            assert config.max_training_steps > 0
            assert config.evaluation_frequency > 0
            assert config.seed >= 0
    
    def test_config_mode_consistency(self):
        """Test configuration mode consistency."""
        # Debug config should have debug settings
        debug_config = create_debug_grpo_config()
        assert debug_config.optimization_level == OptimizationLevel.DEBUG
        assert debug_config.compile_mode == "eager"
        assert debug_config.logging.log_level == "DEBUG"
        
        # Production config should have production settings
        prod_config = create_production_grpo_config()
        assert prod_config.optimization_level == OptimizationLevel.PRODUCTION
        assert prod_config.compile_mode == "jit"
        assert prod_config.precision == "float32"
    
    def test_config_adaptation_compatibility(self):
        """Test configuration adaptation features compatibility."""
        config = create_standard_grpo_config()
        
        # Test adaptive training compatibility
        if config.training_mode == TrainingMode.ADAPTIVE:
            assert any([
                config.adaptive.enable_adaptive_lr,
                config.adaptive.enable_adaptive_exploration,
                config.adaptive.enable_adaptive_curriculum
            ])
        
        # Test curriculum compatibility
        if config.training_mode == TrainingMode.CURRICULUM:
            assert config.curriculum.enable_curriculum
    
    def test_config_serialization_compatibility(self):
        """Test that configs can be serialized and deserialized."""
        import pickle
        
        original_config = create_standard_grpo_config()
        
        # Serialize
        serialized = pickle.dumps(original_config)
        
        # Deserialize
        restored_config = pickle.loads(serialized)
        
        # Should be equal
        assert restored_config.max_training_steps == original_config.max_training_steps
        assert restored_config.training_mode == original_config.training_mode
        assert restored_config.grpo_algorithm.learning_rate == original_config.grpo_algorithm.learning_rate


class TestDataFlowIntegration:
    """Test data flow through the complete system."""
    
    def test_state_tensor_flow(self):
        """Test state tensor flow through system."""
        # Create test state
        state = create_test_state()
        
        # Verify state structure
        assert hasattr(state, 'mechanism_features')
        assert isinstance(state.mechanism_features, jnp.ndarray)
        
        # Test state can be used in experience
        action = pyr.pmap({'X': 1.0})
        reward = RewardResult(0.8, {'r1': 0.8}, {})
        env_info = EnvironmentInfo(1, 10.0, False, False, True, 0.8, {})
        
        experience = Experience(
            state=state,
            action=action,
            next_state=state,
            reward=reward,
            done=False,
            log_prob=-0.5,
            value=0.7,
            env_info=env_info,
            timestamp=time.time()
        )
        
        # Verify experience can be processed
        assert isinstance(experience.state, JAXAcquisitionState)
        assert experience.reward.total_reward == 0.8
    
    def test_trajectory_creation_flow(self):
        """Test trajectory creation from experiences."""
        grpo_config = create_default_grpo_config()
        exp_manager = create_experience_manager(grpo_config)
        
        # Add experiences with varied rewards
        for i in range(10):
            state = create_test_state()
            action = pyr.pmap({'X': float(i)})
            reward = RewardResult(0.5 + i * 0.05, {'r1': 0.5}, {})
            env_info = EnvironmentInfo(i, 10.0, False, False, i == 9, 0.8, {})
            
            exp = Experience(
                state=state,
                action=action,
                next_state=state,
                reward=reward,
                done=(i == 9),
                log_prob=-0.1 * i,
                value=0.6 + i * 0.03,
                env_info=env_info,
                timestamp=time.time()
            )
            exp_manager.add_experience(exp)
        
        # Sample and verify trajectory
        batch = exp_manager.sample_batch()
        assert batch is not None
        
        trajectory = batch.trajectory
        
        # Verify trajectory properties
        assert trajectory.states.shape[0] > 0
        assert trajectory.rewards.shape == trajectory.values.shape
        assert trajectory.advantages.shape == trajectory.returns.shape
        
        # Verify advantages and returns are computed
        assert not jnp.allclose(trajectory.advantages, 0.0)
        assert not jnp.allclose(trajectory.returns, 0.0)
    
    def test_priority_update_flow(self):
        """Test priority update flow in prioritized replay."""
        grpo_config = create_default_grpo_config()
        exp_config = ExperienceConfig(prioritized_replay=True)
        exp_manager = ExperienceManager(exp_config, grpo_config)
        
        # Add experiences
        for i in range(15):
            state = create_test_state()
            exp = Experience(
                state=state,
                action=pyr.pmap({'X': 1.0}),
                next_state=state,
                reward=RewardResult(0.5, {}, {}),
                done=False,
                log_prob=-0.1,
                value=0.6,
                env_info=EnvironmentInfo(0, 10.0, False, False, False, 0.5, {}),
                timestamp=time.time()
            )
            exp_manager.add_experience(exp)
        
        # Sample batch
        batch = exp_manager.sample_batch()
        assert batch is not None
        assert batch.metadata['sampling_method'] == 'prioritized'
        
        # Update priorities
        new_priorities = jnp.array([1.0, 2.0, 0.5] + [1.0] * (len(batch.indices) - 3))
        exp_manager.update_priorities(batch.indices, new_priorities)
        
        # Sample again - should reflect priority changes
        batch2 = exp_manager.sample_batch()
        assert batch2 is not None


class TestErrorHandlingIntegration:
    """Test error handling across integrated components."""
    
    def test_invalid_config_handling(self):
        """Test handling of invalid configurations."""
        # Create config with incompatible settings
        base_config = create_debug_grpo_config()
        
        # Test invalid training steps
        invalid_config = base_config.__class__(
            **{**base_config.__dict__, 'max_training_steps': -1}
        )
        
        with pytest.raises(ValueError, match="max_training_steps must be positive"):
            validate_comprehensive_grpo_config(invalid_config)
    
    def test_insufficient_experience_handling(self):
        """Test handling when insufficient experiences available."""
        grpo_config = create_default_grpo_config()
        exp_manager = create_experience_manager(grpo_config, batch_size=32, buffer_size=100)
        
        # Add only a few experiences (less than min_replay_size)
        for i in range(5):
            state = create_test_state()
            exp = Experience(
                state=state,
                action=pyr.pmap({'X': 1.0}),
                next_state=state,
                reward=RewardResult(0.5, {}, {}),
                done=False,
                log_prob=-0.1,
                value=0.6,
                env_info=EnvironmentInfo(0, 10.0, False, False, False, 0.5, {}),
                timestamp=time.time()
            )
            exp_manager.add_experience(exp)
        
        # Should return None when insufficient experiences
        batch = exp_manager.sample_batch()
        assert batch is None
    
    def test_memory_limit_handling(self):
        """Test memory limit handling in experience manager."""
        grpo_config = create_default_grpo_config()
        exp_config = ExperienceConfig(
            max_buffer_size=20,
            memory_limit_mb=1,  # Very low limit
            batch_size=4
        )
        exp_manager = ExperienceManager(exp_config, grpo_config)
        
        # Add many experiences
        for i in range(25):  # More than buffer capacity
            state = create_test_state()
            exp = Experience(
                state=state,
                action=pyr.pmap({'X': 1.0}),
                next_state=state,
                reward=RewardResult(0.5, {}, {}),
                done=False,
                log_prob=-0.1,
                value=0.6,
                env_info=EnvironmentInfo(0, 10.0, False, False, False, 0.5, {}),
                timestamp=time.time()
            )
            exp_manager.add_experience(exp)
        
        # Should respect buffer limits
        stats = exp_manager.get_statistics()
        assert stats['buffer_size'] <= exp_config.max_buffer_size
    
    def test_training_manager_error_recovery(self):
        """Test training manager error recovery."""
        # Create training manager with mock components
        env = Mock(spec=InterventionEnvironment)
        env.reset.return_value = create_test_state()
        env.step.return_value = (
            create_test_state(),
            EnvironmentInfo(1, 10.0, False, False, True, 0.8, {})
        )
        
        rubric = Mock(spec=CausalRewardRubric)
        rubric.compute_reward.return_value = RewardResult(0.8, {'r1': 0.8}, {})
        
        policy_net = Mock()
        policy_net.params = {'policy': jnp.array([1.0])}
        policy_net.replace.return_value = policy_net
        
        value_net = Mock()
        value_net.params = {'value': jnp.array([0.5])}
        value_net.replace.return_value = value_net
        
        config = create_debug_grpo_config()
        manager = GRPOTrainingManager(
            config=config,
            environment=env,
            reward_rubric=rubric,
            policy_network=policy_net,
            value_network=value_net
        )
        
        # Test graceful handling of empty experience buffer
        step_result = manager._execute_training_step()
        assert isinstance(step_result, TrainingStep)
        assert step_result.grpo_update_result is None  # No update due to insufficient experiences


class TestPerformanceIntegration:
    """Test performance aspects of integrated system."""
    
    def test_batch_processing_efficiency(self):
        """Test efficiency of batch processing."""
        grpo_config = create_default_grpo_config()
        exp_manager = create_experience_manager(grpo_config, batch_size=64)
        
        # Add many experiences
        start_time = time.time()
        for i in range(1000):
            state = create_test_state()
            exp = Experience(
                state=state,
                action=pyr.pmap({'X': float(i % 10)}),
                next_state=state,
                reward=RewardResult(0.5 + (i % 10) * 0.05, {}, {}),
                done=(i % 20 == 19),
                log_prob=-0.1,
                value=0.6,
                env_info=EnvironmentInfo(0, 10.0, False, False, False, 0.5, {}),
                timestamp=time.time()
            )
            exp_manager.add_experience(exp)
        
        add_time = time.time() - start_time
        
        # Sample multiple batches
        start_time = time.time()
        for _ in range(10):
            batch = exp_manager.sample_batch()
            assert batch is not None
        
        sample_time = time.time() - start_time
        
        # Should be reasonably fast
        assert add_time < 5.0  # Adding 1000 experiences should take < 5 seconds
        assert sample_time < 1.0  # Sampling 10 batches should take < 1 second
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking accuracy."""
        grpo_config = create_default_grpo_config()
        exp_manager = create_experience_manager(grpo_config)
        
        # Get initial memory usage
        initial_stats = exp_manager.get_statistics()
        initial_memory = initial_stats['memory_usage_mb']
        
        # Add experiences
        for i in range(100):
            state = create_test_state()
            exp = Experience(
                state=state,
                action=pyr.pmap({'X': 1.0}),
                next_state=state,
                reward=RewardResult(0.5, {}, {}),
                done=False,
                log_prob=-0.1,
                value=0.6,
                env_info=EnvironmentInfo(0, 10.0, False, False, False, 0.5, {}),
                timestamp=time.time()
            )
            exp_manager.add_experience(exp)
        
        # Get final memory usage
        final_stats = exp_manager.get_statistics()
        final_memory = final_stats['memory_usage_mb']
        
        # Memory usage should increase
        assert final_memory >= initial_memory
        assert isinstance(final_memory, float)
    
    def test_configuration_factory_performance(self):
        """Test performance of configuration factories."""
        start_time = time.time()
        
        configs = []
        for _ in range(100):
            configs.extend([
                create_debug_grpo_config(),
                create_standard_grpo_config(),
                create_production_grpo_config(),
            ])
        
        creation_time = time.time() - start_time
        
        # Should create configs quickly
        assert creation_time < 1.0  # 300 configs in < 1 second
        assert len(configs) == 300
        
        # All configs should be valid
        for config in configs[:10]:  # Test a subset for validation performance
            validate_comprehensive_grpo_config(config)


class TestRobustnessIntegration:
    """Test robustness of integrated system."""
    
    def test_checkpoint_recovery_integration(self):
        """Test checkpoint save/load integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create training manager
            env = Mock(spec=InterventionEnvironment)
            env.reset.return_value = create_test_state()
            env.step.return_value = (
                create_test_state(),
                EnvironmentInfo(1, 10.0, False, False, True, 0.8, {})
            )
            
            rubric = Mock(spec=CausalRewardRubric)
            rubric.compute_reward.return_value = RewardResult(0.8, {'r1': 0.8}, {})
            
            policy_net = Mock()
            policy_net.params = {'policy': jnp.array([1.0, 2.0])}
            policy_net.replace.return_value = policy_net
            
            value_net = Mock()
            value_net.params = {'value': jnp.array([0.5, 1.5])}
            value_net.replace.return_value = value_net
            
            # Update config to use temp directory
            config = create_debug_grpo_config()
            config = config.__class__(
                **{**config.__dict__,
                   'checkpointing': config.checkpointing.__class__(
                       **{**config.checkpointing.__dict__,
                          'checkpoint_dir': temp_dir}
                   )}
            )
            
            manager = GRPOTrainingManager(
                config=config,
                environment=env,
                reward_rubric=rubric,
                policy_network=policy_net,
                value_network=value_net
            )
            
            # Set some state
            manager.current_step = 100
            manager.current_episode = 25
            manager.best_performance = 0.95
            
            # Save checkpoint
            checkpoint_path = manager.save_checkpoint()
            assert os.path.exists(checkpoint_path)
            
            # Modify state
            manager.current_step = 200
            manager.best_performance = 0.5
            
            # Load checkpoint
            manager.load_checkpoint(checkpoint_path)
            
            # State should be restored
            assert manager.current_step == 100
            assert manager.current_episode == 25
            assert manager.best_performance == 0.95
    
    def test_concurrent_access_safety(self):
        """Test safety under concurrent access patterns."""
        grpo_config = create_default_grpo_config()
        exp_manager = create_experience_manager(grpo_config)
        
        # Simulate concurrent experience addition
        def add_experiences(start_idx: int, count: int):
            for i in range(start_idx, start_idx + count):
                state = create_test_state()
                exp = Experience(
                    state=state,
                    action=pyr.pmap({'X': float(i)}),
                    next_state=state,
                    reward=RewardResult(0.5, {}, {}),
                    done=False,
                    log_prob=-0.1,
                    value=0.6,
                    env_info=EnvironmentInfo(0, 10.0, False, False, False, 0.5, {}),
                    timestamp=time.time()
                )
                exp_manager.add_experience(exp)
        
        # Add experiences sequentially (simulating concurrent access)
        add_experiences(0, 50)
        add_experiences(50, 50)
        
        # Should handle gracefully
        stats = exp_manager.get_statistics()
        assert stats['buffer_size'] == 100
        
        # Should be able to sample
        batch = exp_manager.sample_batch()
        assert batch is not None
    
    def test_edge_case_handling(self):
        """Test handling of various edge cases."""
        # Empty action spaces
        empty_action = pyr.pmap({})
        state = create_test_state()
        reward = RewardResult(0.0, {}, {})
        env_info = EnvironmentInfo(0, 0.0, True, True, True, 0.0, {})
        
        # Should handle empty action gracefully
        exp = Experience(
            state=state,
            action=empty_action,
            next_state=state,
            reward=reward,
            done=True,
            log_prob=0.0,
            value=0.0,
            env_info=env_info,
            timestamp=time.time()
        )
        
        assert isinstance(exp, Experience)
        assert exp.action == empty_action
        
        # Zero rewards
        zero_reward = RewardResult(0.0, {'r1': 0.0}, {})
        assert zero_reward.total_reward == 0.0
        
        # Extreme values
        extreme_reward = RewardResult(1e6, {'r1': 1e6}, {})
        assert extreme_reward.total_reward == 1e6