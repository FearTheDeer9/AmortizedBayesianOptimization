"""
Test suite for training configuration system.

Tests Pydantic validation, immutable config creation, and integration
with existing acquisition module configurations.
"""

import pytest
from pydantic import ValidationError

from src.causal_bayes_opt.training.config import (
    TrainingConfig,
    GRPOTrainingConfig,
    RewardTrainingConfig,
    ExplorationTrainingConfig,
    EvaluationConfig,
    create_training_config,
    validate_training_config,
    create_default_training_config,
    TrainingSchema,
    GRPOTrainingSchema,
    RewardTrainingSchema,
    ExplorationTrainingSchema,
)


class TestPydanticValidation:
    """Test Pydantic schema validation."""
    
    def test_grpo_schema_valid_config(self):
        """Test valid GRPO configuration."""
        config = GRPOTrainingSchema(
            group_size=64,
            num_iterations=4,
            learning_rate=3e-4,
            kl_penalty_coeff=0.0
        )
        assert config.group_size == 64
        assert config.num_iterations == 4
        assert config.kl_penalty_coeff == 0.0  # open-r1 recommendation
    
    def test_grpo_schema_group_size_bounds(self):
        """Test group size validation bounds."""
        # Too small
        with pytest.raises(ValidationError, match="greater than or equal to 8"):
            GRPOTrainingSchema(group_size=4)
        
        # Too large
        with pytest.raises(ValidationError, match="less than or equal to 256"):
            GRPOTrainingSchema(group_size=300)
        
        # Valid boundaries
        assert GRPOTrainingSchema(group_size=8).group_size == 8
        assert GRPOTrainingSchema(group_size=256).group_size == 256
    
    def test_grpo_schema_num_iterations_bounds(self):
        """Test num_iterations validation."""
        # Too small
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            GRPOTrainingSchema(num_iterations=0)
        
        # Too large
        with pytest.raises(ValidationError, match="less than or equal to 10"):
            GRPOTrainingSchema(num_iterations=15)
        
        # Valid range
        assert GRPOTrainingSchema(num_iterations=1).num_iterations == 1
        assert GRPOTrainingSchema(num_iterations=10).num_iterations == 10
    
    def test_grpo_schema_learning_rate_bounds(self):
        """Test learning rate validation."""
        with pytest.raises(ValidationError):
            GRPOTrainingSchema(learning_rate=0.0)  # Must be > 0
        
        with pytest.raises(ValidationError):
            GRPOTrainingSchema(learning_rate=0.1)  # Too large
        
        assert GRPOTrainingSchema(learning_rate=1e-5).learning_rate == 1e-5
        assert GRPOTrainingSchema(learning_rate=0.01).learning_rate == 0.01
    
    def test_reward_schema_weight_validation(self):
        """Test reward weight validation."""
        # Valid weights
        config = RewardTrainingSchema(
            optimization_weight=1.0,
            structure_weight=0.5,
            parent_weight=0.3,
            exploration_weight=0.1
        )
        assert config.optimization_weight == 1.0
        
        # Negative weights should fail
        with pytest.raises(ValidationError):
            RewardTrainingSchema(optimization_weight=-1.0)
    
    def test_exploration_schema_temperature_order(self):
        """Test temperature ordering validation."""
        # Valid ordering
        config = ExplorationTrainingSchema(
            initial_temperature=2.0,
            final_temperature=0.1
        )
        assert config.initial_temperature > config.final_temperature
        
        # Invalid ordering
        with pytest.raises(ValidationError, match="less than initial"):
            ExplorationTrainingSchema(
                initial_temperature=0.1,
                final_temperature=2.0
            )
    
    def test_training_schema_nested_validation(self):
        """Test nested configuration validation."""
        config = TrainingSchema(
            total_steps=1000,
            warmup_steps=50,
            grpo=GRPOTrainingSchema(group_size=32),
            rewards=RewardTrainingSchema(optimization_weight=2.0)
        )
        assert config.total_steps == 1000
        assert config.grpo.group_size == 32
        assert config.rewards.optimization_weight == 2.0


class TestImmutableConfigs:
    """Test immutable configuration dataclasses."""
    
    def test_grpo_config_immutability(self):
        """Test GRPO config is immutable."""
        config = GRPOTrainingConfig(group_size=64)
        
        # Should not be able to modify
        with pytest.raises(AttributeError):
            config.group_size = 128
    
    def test_grpo_config_to_acquisition_conversion(self):
        """Test conversion to acquisition module config."""
        training_config = GRPOTrainingConfig(
            group_size=64,
            learning_rate=1e-3,
            clip_ratio=0.3
        )
        
        acquisition_config = training_config.to_grpo_config()
        
        assert acquisition_config.group_size == 64
        assert acquisition_config.learning_rate == 1e-3
        assert acquisition_config.clip_ratio == 0.3
    
    def test_reward_config_get_weights(self):
        """Test reward config weight extraction."""
        config = RewardTrainingConfig(
            optimization_weight=1.5,
            structure_weight=0.8,
            parent_weight=0.2,
            exploration_weight=0.05
        )
        
        weights = config.get_reward_weights()
        
        assert weights['optimization'] == 1.5
        assert weights['structure'] == 0.8
        assert weights['parent'] == 0.2
        assert weights['exploration'] == 0.05
        
        # Weights should be immutable
        import pyrsistent as pyr
        assert isinstance(weights, pyr.PMap)
    
    def test_exploration_config_conversion(self):
        """Test exploration config conversion."""
        training_config = ExplorationTrainingConfig(
            strategy_type="adaptive",
            uncertainty_weight=2.0,
            initial_temperature=3.0
        )
        
        exploration_config = training_config.to_exploration_config()
        
        assert exploration_config.uncertainty_weight == 2.0
        assert exploration_config.initial_temperature == 3.0
    
    def test_training_config_immutability(self):
        """Test top-level training config immutability."""
        config = TrainingConfig(total_steps=1000)
        
        with pytest.raises(AttributeError):
            config.total_steps = 2000
        
        # Nested configs should also be immutable
        with pytest.raises(AttributeError):
            config.grpo.group_size = 128


class TestConfigFactory:
    """Test configuration factory functions."""
    
    def test_create_training_config_with_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "total_steps": 2000,
            "grpo": {
                "group_size": 32,
                "num_iterations": 6
            },
            "rewards": {
                "optimization_weight": 2.0
            }
        }
        
        config = create_training_config(config_dict)
        
        assert config.total_steps == 2000
        assert config.grpo.group_size == 32
        assert config.grpo.num_iterations == 6
        assert config.rewards.optimization_weight == 2.0
    
    def test_create_training_config_with_kwargs(self):
        """Test creating config with kwargs."""
        config = create_training_config(
            total_steps=3000,
            log_frequency=20
        )
        
        assert config.total_steps == 3000
        assert config.log_frequency == 20
    
    def test_create_training_config_merge_dict_kwargs(self):
        """Test merging dictionary and kwargs (kwargs override)."""
        config_dict = {"total_steps": 1000}
        
        config = create_training_config(
            config_dict,
            total_steps=2000,  # Should override
            log_frequency=15
        )
        
        assert config.total_steps == 2000  # kwargs override
        assert config.log_frequency == 15
    
    def test_create_training_config_validation_error(self):
        """Test validation error handling."""
        with pytest.raises(ValueError, match="Configuration validation failed"):
            create_training_config({
                "grpo": {
                    "group_size": 2  # Too small
                }
            })
    
    def test_create_default_config(self):
        """Test default configuration creation."""
        config = create_default_training_config()
        
        # Check open-r1 recommendations are applied
        assert config.grpo.num_iterations == 4  # Sample reuse
        assert config.grpo.kl_penalty_coeff == 0.0  # No KL penalty
        assert config.grpo.scale_rewards == True  # Configurable scaling
        assert config.grpo.group_size == 64  # Recommended size
        
        # Check dual-objective weights
        assert config.rewards.optimization_weight == 1.0
        assert config.rewards.structure_weight == 0.5
        
        # Check adaptive exploration
        assert config.exploration.strategy_type == "adaptive"


class TestConfigValidation:
    """Test configuration validation functions."""
    
    def test_validate_training_config_valid(self):
        """Test validation of valid configuration."""
        config = create_default_training_config()
        assert validate_training_config(config) == True
    
    def test_validate_warmup_steps_error(self):
        """Test warmup steps validation."""
        config = TrainingConfig(
            total_steps=100,
            warmup_steps=150  # Invalid: > total_steps
        )
        
        with pytest.raises(ValueError, match="Warmup steps must be less"):
            validate_training_config(config)
    
    def test_validate_eval_frequency_error(self):
        """Test evaluation frequency validation."""
        config = TrainingConfig(
            total_steps=100,
            evaluation=EvaluationConfig(eval_frequency=200)  # > total_steps
        )
        
        with pytest.raises(ValueError, match="Evaluation frequency"):
            validate_training_config(config)
    
    def test_validate_reward_weights_zero(self):
        """Test validation fails when all reward weights are zero."""
        config = TrainingConfig(
            rewards=RewardTrainingConfig(
                optimization_weight=0.0,
                structure_weight=0.0,
                parent_weight=0.0,
                exploration_weight=0.0
            )
        )
        
        with pytest.raises(ValueError, match="At least one reward weight"):
            validate_training_config(config)
    
    def test_validate_temperature_order(self):
        """Test exploration temperature validation."""
        config = TrainingConfig(
            exploration=ExplorationTrainingConfig(
                strategy_type="adaptive",
                initial_temperature=0.1,
                final_temperature=2.0  # Invalid: > initial
            )
        )
        
        with pytest.raises(ValueError, match="Final temperature must be less"):
            validate_training_config(config)


class TestOpenR1Integration:
    """Test integration of open-r1 recommendations."""
    
    def test_sample_reuse_parameter(self):
        """Test num_iterations parameter for sample reuse."""
        config = create_training_config({
            "grpo": {"num_iterations": 6}
        })
        
        assert config.grpo.num_iterations == 6
    
    def test_kl_penalty_zero_default(self):
        """Test KL penalty defaults to 0.0 per open-r1."""
        config = create_default_training_config()
        assert config.grpo.kl_penalty_coeff == 0.0
    
    def test_configurable_reward_scaling(self):
        """Test configurable reward scaling option."""
        # Default should be True
        config = create_default_training_config()
        assert config.grpo.scale_rewards == True
        
        # Should be configurable
        config = create_training_config({
            "grpo": {"scale_rewards": False}
        })
        assert config.grpo.scale_rewards == False
    
    def test_group_size_bounds(self):
        """Test GRPO group size bounds."""
        # Should accept recommended range
        config = create_training_config({
            "grpo": {"group_size": 128}
        })
        assert config.grpo.group_size == 128
        
        # Should reject too small
        with pytest.raises(ValueError):
            create_training_config({
                "grpo": {"group_size": 4}
            })


class TestConfigMetadata:
    """Test configuration metadata handling."""
    
    def test_metadata_preservation(self):
        """Test that original config dict is preserved in metadata."""
        original_dict = {
            "total_steps": 1500,
            "custom_field": "test_value",
            "grpo": {"group_size": 32}
        }
        
        config = create_training_config(original_dict)
        
        # Metadata should contain original config
        assert config.metadata['total_steps'] == 1500
        assert config.metadata['custom_field'] == "test_value"
        assert config.metadata['grpo']['group_size'] == 32
    
    def test_metadata_immutability(self):
        """Test metadata is immutable."""
        config = create_default_training_config()
        
        # Metadata should be immutable PMap
        import pyrsistent as pyr
        assert isinstance(config.metadata, pyr.PMap)


class TestBackwardCompatibility:
    """Test backward compatibility with existing configs."""
    
    def test_grpo_config_compatibility(self):
        """Test compatibility with existing GRPOConfig."""
        training_config = GRPOTrainingConfig(
            group_size=64,
            learning_rate=1e-3,
            clip_ratio=0.15,
            entropy_coeff=0.02
        )
        
        grpo_config = training_config.to_grpo_config()
        
        # Should have all required fields
        assert hasattr(grpo_config, 'group_size')
        assert hasattr(grpo_config, 'learning_rate')
        assert hasattr(grpo_config, 'clip_ratio')
        assert hasattr(grpo_config, 'entropy_coeff')
        
        # Values should match
        assert grpo_config.group_size == 64
        assert grpo_config.learning_rate == 1e-3
        assert grpo_config.clip_ratio == 0.15
        assert grpo_config.entropy_coeff == 0.02