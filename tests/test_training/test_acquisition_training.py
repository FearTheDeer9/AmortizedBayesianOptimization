"""
Tests for acquisition model training pipeline.

Tests the complete training pipeline including behavioral cloning, GRPO fine-tuning,
and integration with existing ACBO infrastructure while following TDD principles.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List, Dict, Any
from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import pyrsistent as pyr

from src.causal_bayes_opt.training.acquisition_training import (
    AcquisitionGRPOConfig,
    BehavioralCloningConfig,
    AcquisitionTrainingConfig,
    TrainingResults,
    train_acquisition_model,
    behavioral_cloning_phase,
    grpo_fine_tuning_phase,
    _validate_training_inputs,
    _initialize_policy_network,
    _prepare_bc_training_data,
    _bc_training_step,
    _create_bc_batches,
    _evaluate_bc_model,
    _create_enhanced_grpo_trainer,
    _collect_enhanced_grpo_batch,
    _detect_reward_hacking,
    _evaluate_grpo_performance,
    create_jax_bc_training_step,
    _train_acquisition_model_pure
)
from src.causal_bayes_opt.training.acquisition_config import (
    TrainingConfig as CompleteTrainingConfig,
    create_standard_config
)
from src.causal_bayes_opt.acquisition.state import AcquisitionState
from src.causal_bayes_opt.acquisition.policy import PolicyConfig
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.data_structures.sample import create_sample
# Import ParentScaleTrajectory from acquisition_training (which handles the import gracefully)
from src.causal_bayes_opt.training.acquisition_training import ParentScaleTrajectory


class TestAcquisitionTrainingConfig:
    """Test acquisition training configuration classes."""
    
    def test_acquisition_grpo_config_defaults(self):
        """Test default GRPO configuration values."""
        config = AcquisitionGRPOConfig()
        
        # Verify 2024 research findings are applied
        assert config.kl_penalty_coeff == 0.0  # Zero KL penalty confirmed best practice
        assert config.adaptive_advantage_scaling is True  # Handle multi-objective variance
        assert config.single_update_per_batch is True  # Training stability
        assert config.reward_hacking_detection is True  # Monitor exploitation
        
        # Verify reasonable defaults
        assert config.group_size == 64  # Appropriate for intervention selection
        assert config.clip_ratio == 0.2
        assert config.entropy_coeff == 0.01
        assert config.learning_rate == 3e-4
        assert config.max_grad_norm == 1.0
    
    def test_behavioral_cloning_config_defaults(self):
        """Test BC configuration defaults."""
        config = BehavioralCloningConfig()
        
        assert config.learning_rate == 1e-3
        assert config.batch_size == 32
        assert config.epochs == 50
        assert config.validation_split == 0.2
        assert config.early_stopping_patience == 10
        assert config.use_jax_compilation is True  # Performance optimization
    
    def test_acquisition_training_config_composition(self):
        """Test that complete training config composes properly."""
        config = AcquisitionTrainingConfig()
        
        assert isinstance(config.bc_config, BehavioralCloningConfig)
        assert isinstance(config.grpo_config, AcquisitionGRPOConfig)
        assert isinstance(config.policy_config, PolicyConfig)
        
        # Verify reward weights
        expected_weights = {'optimization': 1.0, 'structure': 0.5, 'parent': 0.3, 'exploration': 0.1}
        assert config.reward_weights == expected_weights


class TestTrainingValidation:
    """Test training input validation functions."""
    
    def test_validate_training_inputs_success(self):
        """Test successful validation of training inputs."""
        # Create valid expert trajectories
        expert_trajectories = self._create_mock_expert_trajectories(100)
        config = AcquisitionTrainingConfig()
        
        # Should not raise exception
        _validate_training_inputs(expert_trajectories, config)
    
    def test_validate_training_inputs_insufficient_trajectories(self):
        """Test validation fails with insufficient expert trajectories."""
        expert_trajectories = self._create_mock_expert_trajectories(10)  # Too few
        # Create config with higher requirement using replace (since config is frozen)
        config = AcquisitionTrainingConfig(min_expert_trajectory_count=50)
        
        with pytest.raises(ValueError, match="Insufficient expert trajectories"):
            _validate_training_inputs(expert_trajectories, config)
    
    def test_validate_training_inputs_malformed_trajectories(self):
        """Test validation fails with malformed trajectories."""
        # Create trajectories without required attributes by using spec
        class IncompleteTrajectory:
            pass  # No states or actions attributes
        
        expert_trajectories = [IncompleteTrajectory() for _ in range(100)]
        config = AcquisitionTrainingConfig()
        
        with pytest.raises(ValueError, match="must have 'states' and 'actions' attributes"):
            _validate_training_inputs(expert_trajectories, config)
    
    def test_validate_training_inputs_negative_reward_weights(self):
        """Test validation fails with negative reward weights."""
        expert_trajectories = self._create_mock_expert_trajectories(100)
        # Create config with negative reward weight (since config is frozen)
        invalid_weights = {'optimization': -1.0, 'structure': 0.5, 'parent': 0.3, 'exploration': 0.1}
        config = AcquisitionTrainingConfig(reward_weights=invalid_weights)
        
        with pytest.raises(ValueError, match="reward weights must be non-negative"):
            _validate_training_inputs(expert_trajectories, config)
    
    def _create_mock_expert_trajectories(self, count: int) -> List[ParentScaleTrajectory]:
        """Create mock expert trajectories for testing."""
        trajectories = []
        for i in range(count):
            # Create mock trajectory with required attributes
            trajectory = Mock(spec=ParentScaleTrajectory)
            trajectory.states = [Mock() for _ in range(5)]  # 5 states per trajectory
            trajectory.actions = [Mock() for _ in range(5)]  # 5 actions per trajectory
            trajectories.append(trajectory)
        return trajectories


class TestBehavioralCloningPhase:
    """Test behavioral cloning training phase."""
    
    def test_prepare_bc_training_data(self):
        """Test preparation of BC training data from expert trajectories."""
        # Create mock trajectories
        trajectories = []
        for i in range(3):
            trajectory = Mock(spec=ParentScaleTrajectory)
            trajectory.states = [Mock() for _ in range(2)]
            trajectory.actions = [Mock() for _ in range(2)]
            trajectories.append(trajectory)
        
        key = jax.random.PRNGKey(42)
        validation_split = 0.2
        
        train_data, val_data = _prepare_bc_training_data(trajectories, validation_split, key)
        
        # Should have 6 total (state, action) pairs
        total_pairs = len(train_data) + len(val_data)
        assert total_pairs == 6
        
        # Should split roughly according to validation_split
        expected_train_size = int(6 * 0.8)
        assert len(train_data) == expected_train_size
        assert len(val_data) == 6 - expected_train_size
        
        # Each item should be a (state, action) tuple
        for state, action in train_data:
            assert state is not None
            assert action is not None
    
    def test_create_bc_batches(self):
        """Test creation of BC training batches."""
        # Create mock training data
        data = [(Mock(), Mock()) for _ in range(10)]
        batch_size = 3
        key = jax.random.PRNGKey(42)
        
        batches = _create_bc_batches(data, batch_size, key)
        
        # Should create batches of exact size (incomplete batches dropped)
        expected_num_batches = 10 // 3  # 3 complete batches
        assert len(batches) == expected_num_batches
        
        # Each batch should have exact batch_size
        for batch in batches:
            assert len(batch) == batch_size
    
    def test_bc_training_step_pure_function(self):
        """Test that BC training step is a pure function."""
        # Create minimal mock inputs
        params = {'weights': jnp.array([1.0, 2.0])}
        opt_state = Mock()
        batch = [(Mock(), Mock()) for _ in range(2)]
        policy_network = Mock()
        optimizer = Mock()
        
        # Mock the optimizer update
        mock_updates = {'weights': jnp.array([0.1, 0.2])}
        mock_new_opt_state = Mock()
        optimizer.update.return_value = (mock_updates, mock_new_opt_state)
        
        # Mock policy network outputs
        policy_network.apply.return_value = {
            'variable_logits': jnp.array([1.0, 2.0]),
            'value_params': jnp.array([[0.0, 1.0], [1.0, 0.0]])
        }
        
        # Mock compute_action_log_probability to return finite values
        with patch('src.causal_bayes_opt.training.acquisition_training.compute_action_log_probability') as mock_log_prob:
            mock_log_prob.return_value = -1.0  # Reasonable log probability
            
            # Call the function
            new_params, new_opt_state, loss, accuracy = _bc_training_step(
                params, opt_state, batch, policy_network, optimizer
            )
            
            # Verify outputs are reasonable
            assert new_params is not None
            assert new_opt_state == mock_new_opt_state
            assert jnp.isfinite(loss)
            assert 0.0 <= accuracy <= 1.0
            
            # Verify original inputs weren't modified (immutability)
            assert jnp.allclose(params['weights'], jnp.array([1.0, 2.0]))  # Original unchanged
    
    def test_evaluate_bc_model_empty_data(self):
        """Test BC model evaluation with empty validation data."""
        params = Mock()
        policy_network = Mock()
        val_data = []
        
        val_loss, val_accuracy = _evaluate_bc_model(params, policy_network, val_data)
        
        assert val_loss == 0.0
        assert val_accuracy == 0.0
    
    def test_evaluate_bc_model_with_data(self):
        """Test BC model evaluation with real data."""
        params = Mock()
        policy_network = Mock()
        val_data = [(Mock(), Mock()) for _ in range(3)]
        
        # Mock policy outputs
        policy_network.apply.return_value = {
            'variable_logits': jnp.array([1.0, 2.0]),
            'value_params': jnp.array([[0.0, 1.0], [1.0, 0.0]])
        }
        
        with patch('src.causal_bayes_opt.training.acquisition_training.compute_action_log_probability') as mock_log_prob:
            mock_log_prob.return_value = -2.0  # Reasonable log probability
            
            val_loss, val_accuracy = _evaluate_bc_model(params, policy_network, val_data)
            
            assert jnp.isfinite(val_loss)
            assert val_loss > 0  # Should be positive (negative log likelihood)
            assert 0.0 <= val_accuracy <= 1.0


class TestGRPOEnhancements:
    """Test enhanced GRPO functionality with 2024 improvements."""
    
    def test_create_enhanced_grpo_trainer(self):
        """Test creation of enhanced GRPO trainer."""
        policy_network = Mock()
        config = AcquisitionGRPOConfig()
        
        trainer, optimizer_init = _create_enhanced_grpo_trainer(policy_network, config)
        
        assert callable(trainer)
        assert callable(optimizer_init)
    
    def test_enhanced_grpo_adaptive_advantage_scaling(self):
        """Test adaptive advantage scaling for high variance rewards."""
        policy_network = Mock()
        config = AcquisitionGRPOConfig(adaptive_advantage_scaling=True)
        
        trainer, optimizer_init = _create_enhanced_grpo_trainer(policy_network, config)
        
        # Create mock batch with high variance rewards
        high_variance_rewards = jnp.array([10.0, -5.0, 15.0, -10.0])  # High std
        batch_data = {
            'rewards': high_variance_rewards,
            'states': [Mock() for _ in range(4)],
            'actions': [Mock() for _ in range(4)],
            'old_log_probs': jnp.array([-1.0, -1.5, -2.0, -1.2])
        }
        
        # Mock the base trainer to capture the modified batch
        with patch('src.causal_bayes_opt.training.acquisition_training.create_grpo_trainer') as mock_create:
            mock_base_trainer = Mock()
            mock_create.return_value = (mock_base_trainer, optimizer_init)
            
            trainer, _ = _create_enhanced_grpo_trainer(policy_network, config)
            
            # Call enhanced trainer
            params = Mock()
            opt_state = Mock()
            trainer(params, opt_state, batch_data)
            
            # Should have called base trainer (adaptive scaling should trigger)
            mock_base_trainer.assert_called_once()
    
    def test_detect_reward_hacking(self):
        """Test reward hacking detection functionality."""
        # Test normal rewards (no hacking)
        normal_batch = {'rewards': jnp.array([0.1, 0.3, -0.2, 0.4])}
        normal_metrics = {'mean_reward': [0.2, 0.3, 0.25]}
        
        assert not _detect_reward_hacking(normal_batch, normal_metrics)
        
        # Test suspiciously high rewards
        high_reward_batch = {'rewards': jnp.array([5.0, 4.8, 5.2, 4.9])}
        
        assert _detect_reward_hacking(high_reward_batch, normal_metrics)
        
        # Test suspiciously uniform rewards
        uniform_batch = {'rewards': jnp.array([1.0001, 1.0002, 1.0003, 1.0001])}
        
        assert _detect_reward_hacking(uniform_batch, normal_metrics)
        
        # Test sudden reward spikes (need >5 elements)
        spike_metrics = {'mean_reward': [0.2, 0.3, 0.25, 0.2, 0.3, 4.0]}  # Sudden spike
        
        assert _detect_reward_hacking(normal_batch, spike_metrics)


class TestTrainingIntegration:
    """Test integration of training components."""
    
    def test_training_config_integration(self):
        """Test integration between our config and existing config system."""
        # Test that our config works with existing infrastructure
        standard_config = create_standard_config()
        
        assert standard_config is not None
        assert hasattr(standard_config, 'bc_config')
        assert hasattr(standard_config, 'grpo_config')
        assert hasattr(standard_config, 'reward_config')
    
    def test_immutability_of_config_objects(self):
        """Test that configuration objects are properly immutable."""
        config = AcquisitionGRPOConfig()
        
        # Should not be able to modify frozen dataclass
        with pytest.raises((AttributeError, FrozenInstanceError)):
            config.group_size = 128  # Attempt to modify frozen field
    
    def test_config_validation_comprehensive(self):
        """Test comprehensive configuration validation."""
        from src.causal_bayes_opt.training.acquisition_config import validate_config_compatibility
        
        # Create valid config
        config = create_standard_config()
        warnings = validate_config_compatibility(config)
        
        # Should return list (empty if no warnings)
        assert isinstance(warnings, list)
        
        # Create problematic config
        problematic_config = create_standard_config()
        problematic_config.bc_config.accuracy_threshold = 0.99  # Very high
        problematic_config.grpo_config.learning_rate = 5e-3  # Very high
        
        warnings = validate_config_compatibility(problematic_config)
        assert len(warnings) > 0  # Should have warnings


class TestPropertyBasedValidation:
    """Property-based tests for configuration validation."""
    
    def test_reward_weights_always_non_negative(self):
        """Property: All reward weights should always be non-negative."""
        # Test multiple random configurations
        for seed in range(10):
            key = jax.random.PRNGKey(seed)
            
            # Generate random positive weights
            weights = jax.random.uniform(key, (4,), minval=0.0, maxval=2.0)
            
            # Create config with random positive weights (since config is frozen)
            random_weights = {
                'optimization': float(weights[0]),
                'structure': float(weights[1]),
                'parent': float(weights[2]),
                'exploration': float(weights[3])
            }
            config = AcquisitionTrainingConfig(reward_weights=random_weights)
            
            # Should validate successfully
            expert_trajectories = self._create_mock_expert_trajectories(100)
            _validate_training_inputs(expert_trajectories, config)  # Should not raise
    
    def test_grpo_group_size_scaling_property(self):
        """Property: GRPO group size should scale reasonably with problem size."""
        from src.causal_bayes_opt.training.acquisition_config import get_recommended_config_for_problem_size
        
        # Larger problems should generally get larger group sizes
        small_config = get_recommended_config_for_problem_size(n_variables=3, n_expert_trajectories=100)
        large_config = get_recommended_config_for_problem_size(n_variables=15, n_expert_trajectories=100)
        
        assert large_config.grpo_config.group_size >= small_config.grpo_config.group_size
    
    def _create_mock_expert_trajectories(self, count: int) -> List[ParentScaleTrajectory]:
        """Create mock expert trajectories for testing."""
        trajectories = []
        for i in range(count):
            trajectory = Mock(spec=ParentScaleTrajectory)
            trajectory.states = [Mock() for _ in range(5)]
            trajectory.actions = [Mock() for _ in range(5)]
            trajectories.append(trajectory)
        return trajectories


class TestFunctionalPurity:
    """Test that functions follow functional programming principles."""
    
    def test_bc_training_step_is_pure(self):
        """Test that BC training step doesn't modify inputs."""
        import copy
        
        # Create inputs
        params = {'weights': jnp.array([1.0, 2.0])}
        opt_state = {'momentum': jnp.array([0.1, 0.2])}
        batch = [(Mock(), Mock())]
        policy_network = Mock()
        optimizer = Mock()
        
        # Deep copy inputs to verify they're not modified
        params_copy = copy.deepcopy(params)
        opt_state_copy = copy.deepcopy(opt_state)
        batch_copy = copy.deepcopy(batch)
        
        # Mock necessary functions
        optimizer.update.return_value = ({'weights': jnp.array([0.9, 1.8])}, {'momentum': jnp.array([0.11, 0.21])})
        policy_network.apply.return_value = {'variable_logits': jnp.array([1.0]), 'value_params': jnp.array([[0.0, 1.0]])}
        
        with patch('src.causal_bayes_opt.training.acquisition_training.compute_action_log_probability') as mock_log_prob:
            mock_log_prob.return_value = -1.0
            
            # Call function
            _bc_training_step(params, opt_state, batch, policy_network, optimizer)
            
            # Verify inputs weren't modified (use JAX array comparison)
            assert jnp.allclose(params['weights'], params_copy['weights'])
            assert jnp.allclose(opt_state['momentum'], opt_state_copy['momentum'])
            # Note: batch contains mocks so deep equality check may not work perfectly
    
    def test_prepare_bc_training_data_deterministic(self):
        """Test that BC data preparation is deterministic given same inputs."""
        trajectories = []
        for i in range(3):
            trajectory = Mock(spec=ParentScaleTrajectory)
            trajectory.states = [f"state_{i}_{j}" for j in range(2)]
            trajectory.actions = [f"action_{i}_{j}" for j in range(2)]
            trajectories.append(trajectory)
        
        key = jax.random.PRNGKey(42)
        validation_split = 0.2
        
        # Call twice with same inputs
        result1 = _prepare_bc_training_data(trajectories, validation_split, key)
        result2 = _prepare_bc_training_data(trajectories, validation_split, key)
        
        # Should get identical results
        train1, val1 = result1
        train2, val2 = result2
        
        assert len(train1) == len(train2)
        assert len(val1) == len(val2)
        
        # Results should be deterministic (same random key produces same shuffle)
        for (s1, a1), (s2, a2) in zip(train1, train2):
            assert s1 == s2
            assert a1 == a2
    
    def test_pure_training_function_no_side_effects(self):
        """Test that the pure training function has no side effects."""
        # Create mock trajectories inline since this class doesn't have the helper
        expert_trajectories = []
        for i in range(100):
            trajectory = Mock(spec=ParentScaleTrajectory)
            trajectory.states = [Mock() for _ in range(5)]
            trajectory.actions = [Mock() for _ in range(5)]
            expert_trajectories.append(trajectory)
        config = AcquisitionTrainingConfig()
        surrogate_model = Mock()
        surrogate_params = Mock()
        key = jax.random.PRNGKey(42)
        
        # Mock the required functions to prevent actual training
        with patch('src.causal_bayes_opt.training.acquisition_training.behavioral_cloning_phase') as mock_bc, \
             patch('src.causal_bayes_opt.training.acquisition_training.grpo_fine_tuning_phase') as mock_grpo, \
             patch('src.causal_bayes_opt.training.acquisition_training._initialize_policy_network') as mock_init, \
             patch('src.causal_bayes_opt.training.acquisition_training._evaluate_final_model') as mock_eval:
            
            mock_init.return_value = (Mock(), Mock())
            mock_bc.return_value = (Mock(), {'train_loss': [0.5]})
            mock_grpo.return_value = (Mock(), {'policy_loss': [0.3]})
            mock_eval.return_value = {'performance': 0.8}
            
            # Call the pure function multiple times - should be deterministic
            result1 = _train_acquisition_model_pure(
                expert_trajectories, surrogate_model, surrogate_params, config, key
            )
            result2 = _train_acquisition_model_pure(
                expert_trajectories, surrogate_model, surrogate_params, config, key
            )
            
            # Results should be consistent (deterministic)
            assert len(result1) == 4  # final_params, bc_metrics, grpo_metrics, final_evaluation
            assert len(result2) == 4


class TestJAXCompilation:
    """Test JAX compilation functionality."""
    
    def test_create_jax_bc_training_step(self):
        """Test creation of JAX-compiled BC training step."""
        policy_network = Mock()
        optimizer = Mock()
        
        # Should be able to create the JAX training step
        jax_step = create_jax_bc_training_step(policy_network, optimizer)
        
        # Should return a callable
        assert callable(jax_step)
    
    def test_jax_compilation_flag_respected(self):
        """Test that JAX compilation flag is respected in BC config."""
        # Test with JAX compilation enabled
        config_with_jax = BehavioralCloningConfig(use_jax_compilation=True)
        assert config_with_jax.use_jax_compilation is True
        
        # Test with JAX compilation disabled
        config_without_jax = BehavioralCloningConfig(use_jax_compilation=False)
        assert config_without_jax.use_jax_compilation is False


# Integration test with real components (when available)
class TestRealComponentIntegration:
    """Integration tests with real ACBO components when available."""
    
    @pytest.mark.integration
    def test_integration_with_existing_grpo(self):
        """Test integration with existing GRPO implementation."""
        # This test requires the actual GRPO implementation to be available
        try:
            from src.causal_bayes_opt.acquisition.grpo import create_grpo_trainer, GRPOConfig
            
            # Create a simple policy network mock
            policy_network = Mock()
            config = GRPOConfig()
            
            # Should be able to create trainer without errors
            trainer, optimizer_init = create_grpo_trainer(policy_network, config)
            
            assert callable(trainer)
            assert callable(optimizer_init)
            
        except ImportError:
            pytest.skip("GRPO implementation not available for integration test")
    
    @pytest.mark.integration
    def test_integration_with_existing_rewards(self):
        """Test integration with existing rewards system."""
        try:
            from src.causal_bayes_opt.acquisition.rewards import create_default_reward_config
            
            # Should be able to create reward config
            reward_config = create_default_reward_config()
            
            assert reward_config is not None
            assert 'reward_weights' in reward_config
            
        except ImportError:
            pytest.skip("Rewards implementation not available for integration test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])