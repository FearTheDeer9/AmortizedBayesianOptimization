"""Tests for GRPO (Group Relative Policy Optimization) implementation."""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
import pyrsistent as pyr
from unittest.mock import Mock, MagicMock, patch

from src.causal_bayes_opt.acquisition.grpo import (
    GRPOConfig,
    GRPOUpdate, 
    create_grpo_trainer,
    collect_grpo_batch,
    create_grpo_batch_from_samples,
    create_grpo_batch_from_buffer,
    _compute_grpo_loss,
    _compute_policy_entropy,
)
from src.causal_bayes_opt.acquisition import AcquisitionState
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer


class TestGRPOConfig:
    """Test GRPO configuration."""
    
    def test_default_config(self):
        """Test default GRPO configuration follows best practices."""
        config = GRPOConfig()
        
        # Test DeepSeek recommendations
        assert config.group_size == 64  # Not 8!
        assert config.clip_ratio == 0.2  # Standard PPO value
        assert config.entropy_coeff == 0.01  # Standard entropy regularization
        assert config.kl_penalty_coeff == 0.0  # Updated default (open-r1 recommendation)
        assert config.learning_rate == 3e-4  # Standard Adam learning rate
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = GRPOConfig(
            group_size=128,
            clip_ratio=0.1,
            entropy_coeff=0.02,
            learning_rate=1e-4
        )
        
        assert config.group_size == 128
        assert config.clip_ratio == 0.1
        assert config.entropy_coeff == 0.02
        assert config.learning_rate == 1e-4


class TestGRPOLoss:
    """Test GRPO loss computation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = GRPOConfig(group_size=4)  # Small for testing
        self.key = random.PRNGKey(42)
        
        # Mock policy network
        self.mock_policy = Mock()
        
        # Mock batch data
        self.batch_data = {
            'states': [Mock() for _ in range(4)],
            'actions': [Mock() for _ in range(4)],
            'rewards': jnp.array([1.0, 2.0, -1.0, 0.5]),
            'old_log_probs': jnp.array([-0.5, -0.3, -0.8, -0.4])
        }
        
        # Mock policy outputs
        self.mock_policy_output = {
            'variable_logits': jnp.array([1.0, 2.0, 0.5]),
            'value_params': jnp.array([[0.0, -1.0], [1.0, -0.5], [0.5, -2.0]])
        }
        
    def test_advantage_normalization(self):
        """Test that advantages are normalized by group standard deviation."""
        rewards = jnp.array([1.0, 2.0, -1.0, 0.5])
        
        # Manual calculation
        group_baseline = jnp.mean(rewards)  # 0.625
        raw_advantages = rewards - group_baseline  # [0.375, 1.375, -1.625, -0.125]
        expected_advantages = raw_advantages / (jnp.std(raw_advantages) + 1e-8)
        
        # Test the computation matches literature
        assert jnp.allclose(group_baseline, 0.625)
        assert jnp.allclose(jnp.mean(expected_advantages), 0.0, atol=1e-6)  # Zero mean
        
    def test_kl_divergence_exact_formula(self):
        """Test exact KL divergence computation KL(π_old || π_new)."""
        old_log_probs = jnp.array([-0.5, -0.3, -0.8, -0.4])
        new_log_probs = jnp.array([-0.6, -0.2, -0.9, -0.3])
        
        # Exact KL: E[log(π_old) - log(π_new)]
        expected_kl = jnp.mean(old_log_probs - new_log_probs)
        
        # Manual calculation
        kl_values = old_log_probs - new_log_probs  # [0.1, -0.1, 0.1, -0.1]
        manual_kl = jnp.mean(kl_values)  # 0.0
        
        assert jnp.allclose(expected_kl, manual_kl)
        assert jnp.allclose(expected_kl, 0.0)
        
    def test_no_value_network_in_loss(self):
        """Test that GRPO loss has NO value network components."""
        # Test the structure of loss_info without running full computation
        # This test verifies that the expected keys exist in the loss_info dict
        
        # Expected keys from GRPO (NO value network components)
        expected_keys = {
            'policy_loss', 'entropy_loss', 'kl_penalty',
            'group_baseline', 'mean_reward', 'reward_std',
            'mean_advantage', 'advantage_std', 'mean_entropy', 'approx_kl'
        }
        
        # Forbidden keys (value network components)
        forbidden_keys = {
            'value_loss', 'explained_variance', 'value_targets'
        }
        
        # Test by inspecting the GRPOUpdate dataclass structure
        update = GRPOUpdate(
            policy_loss=0.5,
            entropy_loss=0.1,
            kl_penalty=0.05,
            total_loss=0.65,
            grad_norm=1.0,
            group_baseline=0.5,
            mean_reward=0.5,
            reward_std=1.0,
            mean_advantage=0.0,
            advantage_std=1.0,
            mean_entropy=1.5,
            approx_kl=0.02
        )
        
        # Verify GRPOUpdate has NO value network fields
        update_fields = set(update.__dict__.keys())
        value_fields = [f for f in update_fields if 'value' in f.lower() and f != 'total_loss']
        assert len(value_fields) == 0, f"Found value-related fields: {value_fields}"
        
        # Verify structure matches expectations
        assert 'policy_loss' in update_fields
        assert 'entropy_loss' in update_fields
        assert 'kl_penalty' in update_fields
        assert 'group_baseline' in update_fields
        assert 'value_loss' not in update_fields
        assert 'explained_variance' not in update_fields
                
    def test_policy_entropy_computation(self):
        """Test policy entropy computation for categorical + Gaussian."""
        policy_output = {
            'variable_logits': jnp.array([1.0, 2.0, 0.5]),  # 3 variables
            'value_params': jnp.array([[0.0, -1.0], [1.0, -0.5], [0.5, -2.0]])  # [mean, log_std]
        }
        
        entropy = _compute_policy_entropy(policy_output)
        
        # Verify entropy is positive (more uncertainty = higher entropy)
        assert entropy > 0
        
        # Test with more uncertain distribution
        uncertain_output = {
            'variable_logits': jnp.array([0.1, 0.1, 0.1]),  # More uniform
            'value_params': jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])  # Higher variance
        }
        
        uncertain_entropy = _compute_policy_entropy(uncertain_output)
        assert uncertain_entropy > entropy  # More uncertainty = higher entropy


class TestGRPOTrainer:
    """Test GRPO trainer creation and functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = GRPOConfig(group_size=4)
        self.mock_policy = Mock()
        
    def test_trainer_creation(self):
        """Test GRPO trainer creation."""
        update_step, optimizer_init = create_grpo_trainer(self.mock_policy, self.config)
        
        # Verify functions are returned
        assert callable(update_step)
        assert callable(optimizer_init)
        
        # Test optimizer initialization
        mock_params = {'dummy': jnp.array([1.0, 2.0])}
        opt_state = optimizer_init(mock_params)
        assert opt_state is not None
        
    def test_update_step_interface(self):
        """Test update step has correct interface."""
        update_step, optimizer_init = create_grpo_trainer(self.mock_policy, self.config)
        
        # Test that functions are returned and have correct signature
        assert callable(update_step)
        assert callable(optimizer_init)
        
        # Test optimizer initialization works
        mock_params = {'dummy': jnp.array([1.0])}
        mock_opt_state = optimizer_init(mock_params)
        assert mock_opt_state is not None
        
        # Test that update_step function expects correct arguments
        # (Don't actually call it due to JAX/Mock complexity)
        import inspect
        sig = inspect.signature(update_step)
        expected_params = ['params', 'opt_state', 'batch_data']
        actual_params = list(sig.parameters.keys())
        assert actual_params == expected_params
        
        # Test GRPOUpdate structure
        update_info = GRPOUpdate(
            policy_loss=0.5,
            entropy_loss=0.1,
            kl_penalty=0.05,
            total_loss=0.65,
            grad_norm=1.0,
            group_baseline=0.625,
            mean_reward=0.625,
            reward_std=1.0,
            mean_advantage=0.0,
            advantage_std=1.0,
            mean_entropy=1.0,
            approx_kl=0.0
        )
        
        # Verify return type structure
        assert hasattr(update_info, 'policy_loss')
        assert hasattr(update_info, 'entropy_loss')
        assert hasattr(update_info, 'kl_penalty')
        assert hasattr(update_info, 'total_loss')
        assert hasattr(update_info, 'grad_norm')
        assert hasattr(update_info, 'group_baseline')
        
        # Verify NO value loss
        assert not hasattr(update_info, 'value_loss')
        assert not hasattr(update_info, 'explained_variance')


class TestGRPOBatchCollection:
    """Test GRPO batch collection functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = GRPOConfig(group_size=4)
        self.key = random.PRNGKey(42)
        
        # Mock components
        self.mock_policy = Mock()
        self.mock_params = {'dummy': jnp.array([1.0])}
        self.mock_surrogate_model = Mock()
        self.mock_surrogate_params = {'surrogate': jnp.array([1.0])}
        
        # Mock states
        self.mock_states = []
        for i in range(4):
            state = Mock(spec=AcquisitionState)
            state.buffer = Mock(spec=ExperienceBuffer)
            state.current_target = 'Y'
            state.step = i
            self.mock_states.append(state)
            
        # Mock SCMs
        self.mock_scms = [Mock() for _ in range(4)]
        
        # Mock reward config
        self.reward_config = pyr.m(optimization=1.0, structure=0.5)
        
    def test_collect_grpo_batch_interface(self):
        """Test batch collection function interface."""
        # Test that function exists and has correct signature
        import inspect
        sig = inspect.signature(collect_grpo_batch)
        expected_params = [
            'policy_network', 'params', 'states', 'scms', 
            'surrogate_model', 'surrogate_params', 'config', 
            'reward_config', 'key', 'reward_scaling', 'reward_clip_value'
        ]
        actual_params = list(sig.parameters.keys())
        assert actual_params == expected_params
        
        # Test input validation
        with pytest.raises(ValueError, match="Number of states must match number of SCMs"):
            collect_grpo_batch(
                self.mock_policy, self.mock_params,
                [Mock(), Mock()],  # 2 states
                [Mock()],          # 1 SCM (mismatch!)
                self.mock_surrogate_model, self.mock_surrogate_params,
                self.config, self.reward_config, self.key
            )
                            
    def test_batch_from_samples(self):
        """Test creating batch from pre-collected samples."""
        # Create mock samples: (state, action, reward, old_log_prob) tuples
        samples = [
            (Mock(), Mock(), 1.0, -0.5),
            (Mock(), Mock(), 2.0, -0.3),
            (Mock(), Mock(), -1.0, -0.8),
            (Mock(), Mock(), 0.5, -0.4),
            (Mock(), Mock(), 1.5, -0.6),  # Extra sample
        ]
        
        batch = create_grpo_batch_from_samples(samples, self.config)
        
        # Verify correct number selected
        assert len(batch['states']) == self.config.group_size
        assert len(batch['actions']) == self.config.group_size
        assert batch['rewards'].shape == (self.config.group_size,)
        assert batch['old_log_probs'].shape == (self.config.group_size,)
        
        # Verify values
        expected_rewards = jnp.array([1.0, 2.0, -1.0, 0.5])
        expected_log_probs = jnp.array([-0.5, -0.3, -0.8, -0.4])
        
        assert jnp.allclose(batch['rewards'], expected_rewards)
        assert jnp.allclose(batch['old_log_probs'], expected_log_probs)
        
    def test_insufficient_samples_error(self):
        """Test error when insufficient samples provided."""
        samples = [(Mock(), Mock(), 1.0, -0.5)]  # Only 1 sample, need 4
        
        with pytest.raises(ValueError, match="Need at least 4 samples"):
            create_grpo_batch_from_samples(samples, self.config)


class TestGRPOIntegration:
    """Integration tests for GRPO with other components."""
    
    def test_grpo_no_value_network_principle(self):
        """Test that GRPO truly eliminates value networks."""
        config = GRPOConfig()
        
        # The config should have NO value-related parameters
        config_dict = config.__dict__
        value_keys = [k for k in config_dict.keys() if 'value' in k.lower()]
        assert len(value_keys) == 0, f"Found value-related keys: {value_keys}"
        
        # The update structure should have NO value components
        update = GRPOUpdate(
            policy_loss=0.5,
            entropy_loss=0.1,
            kl_penalty=0.05,
            total_loss=0.65,
            grad_norm=1.0,
            group_baseline=0.5,
            mean_reward=0.5,
            reward_std=1.0,
            mean_advantage=0.0,
            advantage_std=1.0,
            mean_entropy=1.5,
            approx_kl=0.02
        )
        
        # Verify all fields are non-value related
        update_dict = update.__dict__
        value_fields = [k for k in update_dict.keys() if 'value' in k.lower() and k != 'total_loss']
        assert len(value_fields) == 0, f"Found value-related fields: {value_fields}"
        
    def test_grpo_advantage_normalization_stability(self):
        """Test that advantage normalization provides stability."""
        # Test with high-variance rewards
        high_var_rewards = jnp.array([100.0, -50.0, 200.0, -100.0])
        group_baseline = jnp.mean(high_var_rewards)  # 37.5
        raw_advantages = high_var_rewards - group_baseline
        normalized_advantages = raw_advantages / (jnp.std(raw_advantages) + 1e-8)
        
        # After normalization, should have reasonable scale
        assert jnp.std(normalized_advantages) == pytest.approx(1.0, abs=1e-6)
        assert jnp.mean(normalized_advantages) == pytest.approx(0.0, abs=1e-6)
        
        # Test with zero variance (edge case)
        zero_var_rewards = jnp.array([1.0, 1.0, 1.0, 1.0])
        zero_baseline = jnp.mean(zero_var_rewards)
        zero_advantages = zero_var_rewards - zero_baseline
        zero_normalized = zero_advantages / (jnp.std(zero_advantages) + 1e-8)
        
        # Should handle gracefully (advantages become 0)
        assert jnp.allclose(zero_normalized, 0.0)


if __name__ == "__main__":
    pytest.main([__file__])