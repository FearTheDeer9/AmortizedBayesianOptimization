"""Tests for enhanced acquisition training configuration.

This module tests the AcquisitionTrainingConfig enhancements that support
the reward rubric system and async training features.
"""

import pytest
import pyrsistent as pyr

from causal_bayes_opt.training.acquisition_training import (
    AcquisitionTrainingConfig,
    BehavioralCloningConfig,
    AcquisitionGRPOConfig,
    create_training_config,
    create_deployment_config,
    create_ablation_config,
)
from causal_bayes_opt.acquisition.policy import PolicyConfig
from causal_bayes_opt.acquisition.reward_rubric import (
    CausalRewardRubric,
    create_training_rubric,
    create_deployment_rubric,
)


class TestAcquisitionTrainingConfigEnhanced:
    """Test enhanced acquisition training configuration."""
    
    def test_default_config_creation(self):
        """Test creating config with defaults."""
        config = AcquisitionTrainingConfig()
        
        # Check default values
        assert config.use_hybrid_rewards is True
        assert config.reward_rubric is None
        assert config.enable_async_training is True
        assert config.diversity_monitoring is True
        assert config.diversity_threshold == 0.3
        assert not config.is_legacy_mode()
    
    def test_config_with_custom_rubric(self):
        """Test config with custom reward rubric."""
        custom_rubric = create_training_rubric(
            improvement_weight=3.0,
            mechanism_impact_weight=2.0
        )
        
        config = AcquisitionTrainingConfig(
            reward_rubric=custom_rubric,
            diversity_threshold=0.5
        )
        
        assert config.reward_rubric == custom_rubric
        assert config.diversity_threshold == 0.5
        assert config.get_reward_rubric() == custom_rubric
    
    def test_config_immutability(self):
        """Test that config is immutable."""
        config = AcquisitionTrainingConfig()
        
        with pytest.raises(AttributeError):
            config.use_hybrid_rewards = False
    
    def test_get_reward_rubric_with_explicit_rubric(self):
        """Test get_reward_rubric when rubric is specified."""
        rubric = create_deployment_rubric()
        config = AcquisitionTrainingConfig(reward_rubric=rubric)
        
        assert config.get_reward_rubric() == rubric
    
    def test_get_reward_rubric_with_hybrid_enabled(self):
        """Test get_reward_rubric with hybrid rewards but no explicit rubric."""
        config = AcquisitionTrainingConfig(
            use_hybrid_rewards=True,
            reward_rubric=None
        )
        
        rubric = config.get_reward_rubric()
        assert isinstance(rubric, CausalRewardRubric)
        # Should be training rubric by default
        component_names = {c.name for c in rubric.components}
        assert "mechanism_impact" in component_names  # Training mode indicator
    
    def test_get_reward_rubric_legacy_fallback(self):
        """Test get_reward_rubric in legacy mode."""
        config = AcquisitionTrainingConfig(
            use_hybrid_rewards=False,
            reward_rubric=None
        )
        
        assert config.is_legacy_mode()
        rubric = config.get_reward_rubric()
        assert isinstance(rubric, CausalRewardRubric)
        
        # Should be observable-only in legacy mode
        gt_required = [c for c in rubric.components if c.requires_ground_truth]
        assert len(gt_required) == 0
    
    def test_legacy_mode_detection(self):
        """Test legacy mode detection logic."""
        # Hybrid mode
        config1 = AcquisitionTrainingConfig(use_hybrid_rewards=True)
        assert not config1.is_legacy_mode()
        
        # Explicit rubric
        config2 = AcquisitionTrainingConfig(
            use_hybrid_rewards=False,
            reward_rubric=create_training_rubric()
        )
        assert not config2.is_legacy_mode()
        
        # Legacy mode
        config3 = AcquisitionTrainingConfig(
            use_hybrid_rewards=False,
            reward_rubric=None
        )
        assert config3.is_legacy_mode()
    
    def test_async_training_configuration(self):
        """Test async training configuration options."""
        # Async enabled
        config1 = AcquisitionTrainingConfig(enable_async_training=True)
        assert config1.enable_async_training
        
        # Async disabled
        config2 = AcquisitionTrainingConfig(enable_async_training=False)
        assert not config2.enable_async_training
    
    def test_diversity_monitoring_configuration(self):
        """Test diversity monitoring configuration."""
        config = AcquisitionTrainingConfig(
            diversity_monitoring=True,
            diversity_threshold=0.4
        )
        
        assert config.diversity_monitoring
        assert config.diversity_threshold == 0.4


class TestConfigurationFactoryFunctions:
    """Test factory functions for creating standard configurations."""
    
    def test_create_training_config(self):
        """Test creating training configuration."""
        config = create_training_config(
            bc_epochs=60,
            grpo_epochs=120,
            improvement_weight=2.5,
            mechanism_impact_weight=2.0,
            exploration_weight=0.8
        )
        
        assert config.use_hybrid_rewards
        assert config.bc_config.epochs == 60
        assert config.grpo_config.max_episodes == 120
        assert config.enable_async_training
        assert config.diversity_monitoring
        
        # Check rubric has correct components
        rubric = config.get_reward_rubric()
        component_names = {c.name for c in rubric.components}
        assert "target_improvement" in component_names
        assert "mechanism_impact" in component_names
        assert "exploration_diversity" in component_names
        
        # Should have supervised components for training
        gt_required = [c for c in rubric.components if c.requires_ground_truth]
        assert len(gt_required) > 0
    
    def test_create_deployment_config(self):
        """Test creating deployment configuration."""
        config = create_deployment_config(
            bc_epochs=25,
            grpo_epochs=40,
            improvement_weight=3.5,
            exploration_weight=1.2
        )
        
        assert config.use_hybrid_rewards
        assert config.bc_config.epochs == 25
        assert config.grpo_config.max_episodes == 40
        assert config.enable_async_training
        
        # Check rubric has only observable components
        rubric = config.get_reward_rubric()
        gt_required = [c for c in rubric.components if c.requires_ground_truth]
        assert len(gt_required) == 0
        
        # Should still have core components
        component_names = {c.name for c in rubric.components}
        assert "target_improvement" in component_names
        assert "exploration_diversity" in component_names
    
    def test_create_ablation_config_supervised_only(self):
        """Test creating ablation config with only supervised signals."""
        config = create_ablation_config(
            use_supervised=True,
            use_observable=False,
            bc_epochs=35,
            diversity_threshold=0.4
        )
        
        assert config.use_hybrid_rewards
        assert config.bc_config.epochs == 35
        assert config.diversity_threshold == 0.4
        
        # Check rubric has only supervised components
        rubric = config.get_reward_rubric()
        gt_required = [c for c in rubric.components if c.requires_ground_truth]
        gt_not_required = [c for c in rubric.components if not c.requires_ground_truth]
        
        assert len(gt_required) > 0
        assert len(gt_not_required) == 0
    
    def test_create_ablation_config_observable_only(self):
        """Test creating ablation config with only observable signals."""
        config = create_ablation_config(
            use_supervised=False,
            use_observable=True,
            grpo_epochs=75
        )
        
        assert config.grpo_config.max_episodes == 75
        
        # Check rubric has only observable components
        rubric = config.get_reward_rubric()
        gt_required = [c for c in rubric.components if c.requires_ground_truth]
        gt_not_required = [c for c in rubric.components if not c.requires_ground_truth]
        
        assert len(gt_required) == 0
        assert len(gt_not_required) > 0
    
    def test_create_ablation_config_invalid(self):
        """Test creating ablation config with invalid combination."""
        with pytest.raises(ValueError, match="At least one reward type must be enabled"):
            create_ablation_config(
                use_supervised=False,
                use_observable=False
            )
    
    def test_factory_kwargs_passthrough(self):
        """Test that factory functions pass through additional kwargs."""
        config = create_training_config(
            expert_trajectory_count=1000,
            checkpoint_frequency=2000,
            logging_frequency=50
        )
        
        assert config.expert_trajectory_count == 1000
        assert config.checkpoint_frequency == 2000
        assert config.logging_frequency == 50


class TestConfigurationBackwardCompatibility:
    """Test backward compatibility with legacy reward system."""
    
    def test_legacy_reward_weights_preserved(self):
        """Test that legacy reward weights are preserved."""
        legacy_weights = {
            'optimization': 2.0,
            'structure': 1.0,
            'parent': 0.8,
            'exploration': 0.2
        }
        
        config = AcquisitionTrainingConfig(
            use_hybrid_rewards=False,
            reward_weights=legacy_weights
        )
        
        assert config.reward_weights == legacy_weights
        assert config.is_legacy_mode()
    
    def test_default_reward_weights(self):
        """Test default reward weights match expected values."""
        config = AcquisitionTrainingConfig()
        
        expected_weights = {
            'optimization': 1.0,
            'structure': 0.5,
            'parent': 0.3,
            'exploration': 0.1
        }
        
        assert config.reward_weights == expected_weights
    
    def test_mixed_configuration_validation(self):
        """Test behavior with mixed legacy/new configuration."""
        # Hybrid enabled with legacy weights should prefer rubric
        config = AcquisitionTrainingConfig(
            use_hybrid_rewards=True,
            reward_weights={'optimization': 5.0}  # Should be ignored
        )
        
        assert not config.is_legacy_mode()
        rubric = config.get_reward_rubric()
        assert isinstance(rubric, CausalRewardRubric)


class TestConfigurationIntegration:
    """Test integration with other system components."""
    
    def test_config_with_policy_config(self):
        """Test config integration with policy configuration."""
        policy_config = PolicyConfig(
            hidden_dim=256,
            num_heads=8
        )
        
        config = AcquisitionTrainingConfig(policy_config=policy_config)
        
        assert config.policy_config == policy_config
        assert config.policy_config.hidden_dim == 256
        assert config.policy_config.num_heads == 8
    
    def test_config_serialization(self):
        """Test config can be converted to dict (excluding functions)."""
        config = create_training_config()
        
        # Should be able to extract basic config info
        assert config.use_hybrid_rewards
        assert config.enable_async_training
        assert config.diversity_monitoring
        
        # Rubric should have serializable config
        rubric = config.get_reward_rubric()
        rubric_config = rubric.to_config()
        assert "components" in rubric_config
        assert "diversity_threshold" in rubric_config
    
    def test_config_with_custom_thresholds(self):
        """Test config with various threshold settings."""
        config = AcquisitionTrainingConfig(
            diversity_threshold=0.2,
            expert_trajectory_count=1000,
            min_expert_trajectory_count=200
        )
        
        assert config.diversity_threshold == 0.2
        assert config.expert_trajectory_count == 1000
        assert config.min_expert_trajectory_count == 200
        
        # Rubric should respect threshold
        rubric = config.get_reward_rubric()
        assert rubric.diversity_threshold == 0.3  # Default rubric threshold