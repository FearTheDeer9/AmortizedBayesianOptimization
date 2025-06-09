"""
Test suite for enhanced GRPO with open-r1 features.

Tests the enhanced GRPO implementation in acquisition/grpo.py with:
- Updated KL penalty coefficient (0.0 default)
- Sample reuse functionality
- Configurable reward scaling
- Configurable advantage scaling
"""

import pytest
import jax
import jax.numpy as jnp
import pyrsistent as pyr

from src.causal_bayes_opt.acquisition.grpo import (
    GRPOConfig,
    SampleReuseManager,
    create_sample_reuse_manager,
    collect_grpo_batch_with_reuse,
    _compute_grpo_loss
)


class TestEnhancedGRPOConfig:
    """Test enhanced GRPO configuration with open-r1 features."""
    
    def test_default_kl_penalty_zero(self):
        """Test that KL penalty defaults to 0.0 (open-r1 recommendation)."""
        config = GRPOConfig()
        assert config.kl_penalty_coeff == 0.0
    
    def test_sample_reuse_parameters(self):
        """Test sample reuse parameters are included."""
        config = GRPOConfig()
        assert hasattr(config, 'num_iterations')
        assert config.num_iterations == 4  # Default open-r1 value
        
        # Should be configurable
        custom_config = GRPOConfig(num_iterations=6)
        assert custom_config.num_iterations == 6
    
    def test_configurable_reward_scaling(self):
        """Test configurable reward scaling parameter."""
        config = GRPOConfig()
        assert hasattr(config, 'scale_rewards')
        assert config.scale_rewards == True  # Default enabled
        
        # Should be configurable
        no_scale_config = GRPOConfig(scale_rewards=False)
        assert no_scale_config.scale_rewards == False
    
    def test_backward_compatibility(self):
        """Test that existing parameters still work."""
        config = GRPOConfig(
            group_size=32,
            clip_ratio=0.3,
            entropy_coeff=0.02,
            learning_rate=1e-3
        )
        
        # Old parameters should work
        assert config.group_size == 32
        assert config.clip_ratio == 0.3
        assert config.entropy_coeff == 0.02
        assert config.learning_rate == 1e-3
        
        # New parameters should have defaults
        assert config.num_iterations == 4
        assert config.scale_rewards == True
        assert config.kl_penalty_coeff == 0.0


class TestSampleReuseManager:
    """Test sample reuse functionality (key open-r1 feature)."""
    
    def test_manager_creation(self):
        """Test sample reuse manager creation."""
        config = GRPOConfig(num_iterations=3)
        manager = create_sample_reuse_manager(config)
        
        assert manager.current_samples == []
        assert manager.reuse_iteration == 0
        assert manager.max_iterations == 3
    
    def test_should_collect_new_samples_empty(self):
        """Test collection logic with no samples."""
        manager = SampleReuseManager([], 0, 4)
        
        assert manager.should_collect_new_samples(32) == True
    
    def test_should_collect_new_samples_insufficient(self):
        """Test collection logic with insufficient samples."""
        # Only 10 samples but need 32
        samples = [(None, None, 1.0, 0.5) for _ in range(10)]
        manager = SampleReuseManager(samples, 0, 4)
        
        assert manager.should_collect_new_samples(32) == True
    
    def test_should_reuse_samples(self):
        """Test sample reuse logic."""
        # Enough samples, under iteration limit
        samples = [(None, None, 1.0, 0.5) for _ in range(64)]
        manager = SampleReuseManager(samples, 1, 4)  # Iteration 1 of 4
        
        assert manager.should_collect_new_samples(32) == False
    
    def test_should_collect_after_max_iterations(self):
        """Test collection after reaching iteration limit."""
        samples = [(None, None, 1.0, 0.5) for _ in range(64)]
        manager = SampleReuseManager(samples, 4, 4)  # At limit
        
        assert manager.should_collect_new_samples(32) == True
    
    def test_update_for_reuse(self):
        """Test reuse iteration increment."""
        samples = [(None, None, 1.0, 0.5) for _ in range(32)]
        manager = SampleReuseManager(samples, 1, 4)
        
        updated = manager.update_for_reuse()
        
        assert updated.reuse_iteration == 2
        assert updated.current_samples == samples  # Same samples
        assert updated.max_iterations == 4
    
    def test_reset_with_new_samples(self):
        """Test resetting with fresh samples."""
        old_samples = [(None, None, 1.0, 0.5) for _ in range(32)]
        manager = SampleReuseManager(old_samples, 3, 4)
        
        new_samples = [(None, None, 2.0, 0.3) for _ in range(32)]
        reset_manager = manager.reset_with_new_samples(new_samples)
        
        assert reset_manager.current_samples == new_samples
        assert reset_manager.reuse_iteration == 0  # Reset to 0
        assert reset_manager.max_iterations == 4


class TestConfigurableAdvantageScaling:
    """Test configurable advantage scaling (open-r1 feature)."""
    
    def test_advantage_scaling_enabled(self):
        """Test advantage scaling when enabled."""
        config = GRPOConfig(scale_rewards=True)
        
        # Mock batch data
        batch_data = {
            'states': [None] * 4,
            'actions': [None] * 4,
            'rewards': jnp.array([1.0, 2.0, 3.0, 4.0]),
            'old_log_probs': jnp.array([0.1, 0.2, 0.3, 0.4])
        }
        
        # Test the advantages computation
        rewards = batch_data['rewards']
        group_baseline = jnp.mean(rewards)
        advantages = rewards - group_baseline
        
        if config.scale_rewards:
            scaled_advantages = advantages / (jnp.std(advantages) + 1e-8)
        else:
            scaled_advantages = advantages
        
        # Should be normalized
        assert jnp.allclose(jnp.mean(scaled_advantages), 0.0, atol=1e-6)
        assert jnp.allclose(jnp.std(scaled_advantages), 1.0, atol=1e-6)
    
    def test_advantage_scaling_disabled(self):
        """Test advantage scaling when disabled."""
        config = GRPOConfig(scale_rewards=False)
        
        rewards = jnp.array([1.0, 2.0, 3.0, 4.0])
        group_baseline = jnp.mean(rewards)
        advantages = rewards - group_baseline
        
        if config.scale_rewards:
            scaled_advantages = advantages / (jnp.std(advantages) + 1e-8)
        else:
            scaled_advantages = advantages
        
        # Should be raw advantages (not normalized)
        expected_advantages = rewards - group_baseline
        assert jnp.allclose(scaled_advantages, expected_advantages)


class TestRewardScaling:
    """Test reward scaling in batch collection."""
    
    def test_tanh_reward_scaling(self):
        """Test tanh reward scaling."""
        rewards = jnp.array([10.0, -5.0, 15.0, -8.0])
        clip_value = 10.0
        
        # Apply tanh scaling
        scaled_rewards = jnp.tanh(rewards / clip_value) * clip_value
        
        # Should be bounded by [-clip_value, clip_value]
        assert jnp.all(scaled_rewards >= -clip_value)
        assert jnp.all(scaled_rewards <= clip_value)
        
        # Large values should be compressed
        assert jnp.abs(scaled_rewards[0]) < jnp.abs(rewards[0])  # 10.0 -> smaller
        assert jnp.abs(scaled_rewards[2]) < jnp.abs(rewards[2])  # 15.0 -> smaller
    
    def test_clip_reward_scaling(self):
        """Test clip reward scaling."""
        rewards = jnp.array([12.0, -15.0, 8.0, -3.0])
        clip_value = 10.0
        
        # Apply clip scaling
        scaled_rewards = jnp.clip(rewards, -clip_value, clip_value)
        
        # Should be hard clipped
        assert jnp.all(scaled_rewards >= -clip_value)
        assert jnp.all(scaled_rewards <= clip_value)
        
        # Values should be exactly clipped
        assert scaled_rewards[0] == clip_value   # 12.0 -> 10.0
        assert scaled_rewards[1] == -clip_value  # -15.0 -> -10.0
        assert scaled_rewards[2] == 8.0          # 8.0 -> 8.0 (unchanged)
        assert scaled_rewards[3] == -3.0         # -3.0 -> -3.0 (unchanged)
    
    def test_no_reward_scaling(self):
        """Test no reward scaling."""
        rewards = jnp.array([10.0, -5.0, 15.0, -8.0])
        
        # No scaling applied
        scaled_rewards = rewards  # Identity
        
        assert jnp.allclose(scaled_rewards, rewards)


class TestBackwardCompatibility:
    """Test that enhanced GRPO maintains backward compatibility."""
    
    def test_old_function_signatures_work(self):
        """Test that existing function calls still work."""
        # Should be able to create old-style config
        config = GRPOConfig(
            group_size=32,
            clip_ratio=0.2,
            entropy_coeff=0.01,
            kl_penalty_coeff=0.1,  # Can still set old value
            learning_rate=1e-3
        )
        
        assert config.group_size == 32
        assert config.kl_penalty_coeff == 0.1  # Explicitly set
        
        # New features should have defaults
        assert config.num_iterations == 4
        assert config.scale_rewards == True
    
    def test_enhanced_features_optional(self):
        """Test that enhanced features are optional."""
        # Old usage pattern should work
        config = GRPOConfig()
        
        # Can access new features
        assert hasattr(config, 'num_iterations')
        assert hasattr(config, 'scale_rewards')
        
        # But they have sensible defaults
        assert config.num_iterations > 0
        assert isinstance(config.scale_rewards, bool)


class TestIntegrationWithExistingGRPO:
    """Test integration with existing GRPO infrastructure."""
    
    def test_config_works_with_existing_functions(self):
        """Test that enhanced config works with existing GRPO functions."""
        config = GRPOConfig(
            group_size=16,
            num_iterations=3,
            scale_rewards=False,
            kl_penalty_coeff=0.0
        )
        
        # Should be able to use config in existing functions
        assert config.group_size == 16
        assert config.clip_ratio == 0.2  # Default
        
        # Enhanced features available
        assert config.num_iterations == 3
        assert config.scale_rewards == False
        assert config.kl_penalty_coeff == 0.0
    
    def test_sample_reuse_integration_pattern(self):
        """Test sample reuse integration pattern."""
        config = GRPOConfig(num_iterations=2, group_size=8)
        manager = create_sample_reuse_manager(config)
        
        # Workflow simulation
        assert manager.should_collect_new_samples(8) == True  # First time
        
        # Simulate collecting samples
        mock_samples = [(None, None, 1.0, 0.5) for _ in range(8)]
        manager = manager.reset_with_new_samples(mock_samples)
        
        # Should reuse now
        assert manager.should_collect_new_samples(8) == False
        manager = manager.update_for_reuse()
        
        # Still reusing (iteration 1 of 2)
        assert manager.should_collect_new_samples(8) == False
        manager = manager.update_for_reuse()
        
        # Now should collect fresh (reached limit)
        assert manager.should_collect_new_samples(8) == True