"""Tests for GRPO configuration system.

This module tests the comprehensive GRPO configuration system including
validation, factory functions, and compatibility checking between subsystems.
"""

import pytest
from unittest.mock import Mock

from causal_bayes_opt.training.grpo_config import (
    TrainingMode,
    OptimizationLevel,
    PolicyNetworkConfig,
    ValueNetworkConfig,
    CurriculumConfig,
    AdaptiveConfig,
    CheckpointingConfig,
    LoggingConfig,
    ComprehensiveGRPOConfig,
    validate_comprehensive_grpo_config,
    create_standard_grpo_config,
    create_research_grpo_config,
    create_production_grpo_config,
    create_debug_grpo_config,
    get_recommended_config_for_problem_size,
)
from causal_bayes_opt.training.grpo_core import GRPOConfig
from causal_bayes_opt.training.experience_management import ExperienceConfig
from causal_bayes_opt.training.async_training import AsyncTrainingConfig


class TestEnumsAndDataclasses:
    """Test enum definitions and basic dataclasses."""
    
    def test_training_mode_enum(self):
        """Test TrainingMode enum values."""
        assert TrainingMode.BEHAVIORAL_CLONING.value == "behavioral_cloning"
        assert TrainingMode.GRPO_FINE_TUNING.value == "grpo_fine_tuning"
        assert TrainingMode.FULL_GRPO.value == "full_grpo"
        assert TrainingMode.CURRICULUM.value == "curriculum"
        assert TrainingMode.ADAPTIVE.value == "adaptive"
    
    def test_optimization_level_enum(self):
        """Test OptimizationLevel enum values."""
        assert OptimizationLevel.DEBUG.value == "debug"
        assert OptimizationLevel.DEVELOPMENT.value == "development"
        assert OptimizationLevel.PRODUCTION.value == "production"
        assert OptimizationLevel.RESEARCH.value == "research"


class TestPolicyNetworkConfig:
    """Test policy network configuration."""
    
    def test_config_creation(self):
        """Test creating policy network config."""
        config = PolicyNetworkConfig(
            hidden_dims=(512, 256, 128),
            activation="gelu",
            dropout_rate=0.2,
            use_batch_norm=False,
            use_residual=True,
            attention_heads=8,
            max_sequence_length=200,
            embedding_dim=64
        )
        
        assert config.hidden_dims == (512, 256, 128)
        assert config.activation == "gelu"
        assert config.dropout_rate == 0.2
        assert config.use_batch_norm is False
        assert config.use_residual is True
        assert config.attention_heads == 8
        assert config.max_sequence_length == 200
        assert config.embedding_dim == 64
    
    def test_config_defaults(self):
        """Test default policy network config values."""
        config = PolicyNetworkConfig()
        
        assert config.hidden_dims == (256, 128, 64)
        assert config.activation == "relu"
        assert config.dropout_rate == 0.1
        assert config.use_batch_norm is True
        assert config.use_residual is False
        assert config.attention_heads is None
        assert config.max_sequence_length == 100
        assert config.embedding_dim == 32
    
    def test_config_immutability(self):
        """Test that policy network config is immutable."""
        config = PolicyNetworkConfig()
        
        with pytest.raises(AttributeError):
            config.hidden_dims = (128, 64)


class TestValueNetworkConfig:
    """Test value network configuration."""
    
    def test_config_creation(self):
        """Test creating value network config."""
        config = ValueNetworkConfig(
            hidden_dims=(256, 128),
            activation="tanh",
            dropout_rate=0.15,
            use_batch_norm=False,
            use_residual=True,
            shared_layers=2,
            ensemble_size=5
        )
        
        assert config.hidden_dims == (256, 128)
        assert config.activation == "tanh"
        assert config.dropout_rate == 0.15
        assert config.use_batch_norm is False
        assert config.use_residual is True
        assert config.shared_layers == 2
        assert config.ensemble_size == 5
    
    def test_config_defaults(self):
        """Test default value network config values."""
        config = ValueNetworkConfig()
        
        assert config.hidden_dims == (256, 128)
        assert config.activation == "relu"
        assert config.dropout_rate == 0.1
        assert config.use_batch_norm is True
        assert config.use_residual is False
        assert config.shared_layers == 0
        assert config.ensemble_size == 1


class TestCurriculumConfig:
    """Test curriculum learning configuration."""
    
    def test_config_creation(self):
        """Test creating curriculum config."""
        config = CurriculumConfig(
            enable_curriculum=True,
            difficulty_schedule="exponential",
            adaptation_rate=0.05,
            success_threshold=0.9,
            failure_threshold=0.2,
            min_episodes_per_level=200,
            max_episodes_per_level=2000,
            difficulty_factors={"noise": 0.2, "complexity": 0.8}
        )
        
        assert config.enable_curriculum is True
        assert config.difficulty_schedule == "exponential"
        assert config.adaptation_rate == 0.05
        assert config.success_threshold == 0.9
        assert config.failure_threshold == 0.2
        assert config.min_episodes_per_level == 200
        assert config.max_episodes_per_level == 2000
        assert config.difficulty_factors["noise"] == 0.2
    
    def test_config_defaults(self):
        """Test default curriculum config values."""
        config = CurriculumConfig()
        
        assert config.enable_curriculum is False
        assert config.difficulty_schedule == "linear"
        assert config.adaptation_rate == 0.1
        assert config.success_threshold == 0.8
        assert config.failure_threshold == 0.3
        assert config.min_episodes_per_level == 100
        assert config.max_episodes_per_level == 1000


class TestAdaptiveConfig:
    """Test adaptive training configuration."""
    
    def test_config_creation(self):
        """Test creating adaptive config."""
        config = AdaptiveConfig(
            enable_adaptive_lr=False,
            enable_adaptive_exploration=True,
            enable_adaptive_curriculum=True,
            performance_window=2000,
            adaptation_frequency=200,
            lr_adaptation_factor=0.95,
            exploration_adaptation_factor=0.9,
            early_stopping_patience=5000,
            target_performance=0.95
        )
        
        assert config.enable_adaptive_lr is False
        assert config.enable_adaptive_exploration is True
        assert config.enable_adaptive_curriculum is True
        assert config.performance_window == 2000
        assert config.adaptation_frequency == 200
        assert config.lr_adaptation_factor == 0.95
        assert config.exploration_adaptation_factor == 0.9
        assert config.early_stopping_patience == 5000
        assert config.target_performance == 0.95
    
    def test_config_defaults(self):
        """Test default adaptive config values."""
        config = AdaptiveConfig()
        
        assert config.enable_adaptive_lr is True
        assert config.enable_adaptive_exploration is True
        assert config.enable_adaptive_curriculum is False
        assert config.performance_window == 1000
        assert config.adaptation_frequency == 100
        assert config.lr_adaptation_factor == 0.9
        assert config.exploration_adaptation_factor == 0.95
        assert config.early_stopping_patience == 10000
        assert config.target_performance == 0.9


class TestCheckpointingConfig:
    """Test checkpointing configuration."""
    
    def test_config_creation(self):
        """Test creating checkpointing config."""
        config = CheckpointingConfig(
            enable_checkpointing=False,
            checkpoint_frequency=2000,
            keep_best_only=True,
            save_optimizer_state=False,
            checkpoint_dir="./custom_checkpoints",
            max_checkpoints=10,
            save_metrics=False,
            compression_level=9
        )
        
        assert config.enable_checkpointing is False
        assert config.checkpoint_frequency == 2000
        assert config.keep_best_only is True
        assert config.save_optimizer_state is False
        assert config.checkpoint_dir == "./custom_checkpoints"
        assert config.max_checkpoints == 10
        assert config.save_metrics is False
        assert config.compression_level == 9
    
    def test_config_defaults(self):
        """Test default checkpointing config values."""
        config = CheckpointingConfig()
        
        assert config.enable_checkpointing is True
        assert config.checkpoint_frequency == 1000
        assert config.keep_best_only is False
        assert config.save_optimizer_state is True
        assert config.checkpoint_dir == "./checkpoints"
        assert config.max_checkpoints == 5
        assert config.save_metrics is True
        assert config.compression_level == 6


class TestLoggingConfig:
    """Test logging configuration."""
    
    def test_config_creation(self):
        """Test creating logging config."""
        config = LoggingConfig(
            log_level="DEBUG",
            log_frequency=50,
            log_gradients=True,
            log_weights=True,
            log_activations=True,
            enable_tensorboard=True,
            enable_wandb=True,
            project_name="test_project",
            tags=["experiment", "test"]
        )
        
        assert config.log_level == "DEBUG"
        assert config.log_frequency == 50
        assert config.log_gradients is True
        assert config.log_weights is True
        assert config.log_activations is True
        assert config.enable_tensorboard is True
        assert config.enable_wandb is True
        assert config.project_name == "test_project"
        assert config.tags == ["experiment", "test"]
    
    def test_config_defaults(self):
        """Test default logging config values."""
        config = LoggingConfig()
        
        assert config.log_level == "INFO"
        assert config.log_frequency == 100
        assert config.log_gradients is False
        assert config.log_weights is False
        assert config.log_activations is False
        assert config.enable_tensorboard is False
        assert config.enable_wandb is False
        assert config.project_name == "causal_bayes_opt"
        assert config.tags == []


class TestComprehensiveGRPOConfig:
    """Test the main comprehensive GRPO configuration."""
    
    @pytest.fixture
    def minimal_config(self):
        """Create minimal valid config for testing."""
        return ComprehensiveGRPOConfig(
            grpo_algorithm=GRPOConfig(),
            experience_management=ExperienceConfig(),
            async_training=AsyncTrainingConfig(),
            policy_network=PolicyNetworkConfig(),
            value_network=ValueNetworkConfig()
        )
    
    def test_config_creation(self, minimal_config):
        """Test creating comprehensive config."""
        config = minimal_config
        
        assert isinstance(config.grpo_algorithm, GRPOConfig)
        assert isinstance(config.experience_management, ExperienceConfig)
        assert isinstance(config.async_training, AsyncTrainingConfig)
        assert isinstance(config.policy_network, PolicyNetworkConfig)
        assert isinstance(config.value_network, ValueNetworkConfig)
        assert config.training_mode == TrainingMode.FULL_GRPO
        assert config.optimization_level == OptimizationLevel.PRODUCTION
        assert config.max_training_steps == 100000
        assert config.seed == 42
    
    def test_config_with_all_components(self):
        """Test creating config with all optional components."""
        curriculum = CurriculumConfig(enable_curriculum=True)
        adaptive = AdaptiveConfig()
        checkpointing = CheckpointingConfig()
        logging = LoggingConfig()
        
        config = ComprehensiveGRPOConfig(
            grpo_algorithm=GRPOConfig(),
            experience_management=ExperienceConfig(),
            async_training=AsyncTrainingConfig(),
            policy_network=PolicyNetworkConfig(),
            value_network=ValueNetworkConfig(),
            curriculum=curriculum,
            adaptive=adaptive,
            checkpointing=checkpointing,
            logging=logging,
            training_mode=TrainingMode.CURRICULUM,
            optimization_level=OptimizationLevel.RESEARCH,
            max_training_steps=50000,
            seed=123
        )
        
        assert config.curriculum.enable_curriculum is True
        assert config.training_mode == TrainingMode.CURRICULUM
        assert config.optimization_level == OptimizationLevel.RESEARCH
        assert config.max_training_steps == 50000
        assert config.seed == 123
    
    def test_config_immutability(self, minimal_config):
        """Test that comprehensive config is immutable."""
        config = minimal_config
        
        with pytest.raises(AttributeError):
            config.max_training_steps = 50000


class TestConfigValidation:
    """Test configuration validation functions."""
    
    def test_valid_config_passes(self):
        """Test that valid config passes validation."""
        config = ComprehensiveGRPOConfig(
            grpo_algorithm=GRPOConfig(),
            experience_management=ExperienceConfig(),
            async_training=AsyncTrainingConfig(),
            policy_network=PolicyNetworkConfig(),
            value_network=ValueNetworkConfig()
        )
        
        # Should not raise any exceptions
        validate_comprehensive_grpo_config(config)
    
    def test_invalid_training_steps(self):
        """Test validation of invalid training steps."""
        config = ComprehensiveGRPOConfig(
            grpo_algorithm=GRPOConfig(),
            experience_management=ExperienceConfig(),
            async_training=AsyncTrainingConfig(),
            policy_network=PolicyNetworkConfig(),
            value_network=ValueNetworkConfig(),
            max_training_steps=0  # Invalid
        )
        
        with pytest.raises(ValueError, match="max_training_steps must be positive"):
            validate_comprehensive_grpo_config(config)
    
    def test_invalid_device(self):
        """Test validation of invalid device."""
        config = ComprehensiveGRPOConfig(
            grpo_algorithm=GRPOConfig(),
            experience_management=ExperienceConfig(),
            async_training=AsyncTrainingConfig(),
            policy_network=PolicyNetworkConfig(),
            value_network=ValueNetworkConfig(),
            device="invalid_device"
        )
        
        with pytest.raises(ValueError, match="device must be one of"):
            validate_comprehensive_grpo_config(config)
    
    def test_invalid_precision(self):
        """Test validation of invalid precision."""
        config = ComprehensiveGRPOConfig(
            grpo_algorithm=GRPOConfig(),
            experience_management=ExperienceConfig(),
            async_training=AsyncTrainingConfig(),
            policy_network=PolicyNetworkConfig(),
            value_network=ValueNetworkConfig(),
            precision="invalid_precision"
        )
        
        with pytest.raises(ValueError, match="precision must be one of"):
            validate_comprehensive_grpo_config(config)
    
    def test_curriculum_mode_without_curriculum(self):
        """Test curriculum mode without curriculum enabled."""
        config = ComprehensiveGRPOConfig(
            grpo_algorithm=GRPOConfig(),
            experience_management=ExperienceConfig(),
            async_training=AsyncTrainingConfig(),
            policy_network=PolicyNetworkConfig(),
            value_network=ValueNetworkConfig(),
            training_mode=TrainingMode.CURRICULUM,
            curriculum=CurriculumConfig(enable_curriculum=False)
        )
        
        with pytest.raises(ValueError, match="Training mode is CURRICULUM but curriculum is not enabled"):
            validate_comprehensive_grpo_config(config)
    
    def test_adaptive_mode_without_adaptive_features(self):
        """Test adaptive mode without adaptive features enabled."""
        adaptive = AdaptiveConfig(
            enable_adaptive_lr=False,
            enable_adaptive_exploration=False,
            enable_adaptive_curriculum=False
        )
        
        config = ComprehensiveGRPOConfig(
            grpo_algorithm=GRPOConfig(),
            experience_management=ExperienceConfig(),
            async_training=AsyncTrainingConfig(),
            policy_network=PolicyNetworkConfig(),
            value_network=ValueNetworkConfig(),
            training_mode=TrainingMode.ADAPTIVE,
            adaptive=adaptive
        )
        
        with pytest.raises(ValueError, match="Training mode is ADAPTIVE but no adaptive features are enabled"):
            validate_comprehensive_grpo_config(config)
    
    def test_invalid_network_config(self):
        """Test validation of invalid network configuration."""
        policy_network = PolicyNetworkConfig(
            hidden_dims=(),  # Empty - invalid
            dropout_rate=1.5  # > 1 - invalid
        )
        
        config = ComprehensiveGRPOConfig(
            grpo_algorithm=GRPOConfig(),
            experience_management=ExperienceConfig(),
            async_training=AsyncTrainingConfig(),
            policy_network=policy_network,
            value_network=ValueNetworkConfig()
        )
        
        with pytest.raises(ValueError, match="policy_network.hidden_dims cannot be empty"):
            validate_comprehensive_grpo_config(config)
    
    def test_invalid_curriculum_config(self):
        """Test validation of invalid curriculum configuration."""
        curriculum = CurriculumConfig(
            enable_curriculum=True,
            success_threshold=0.3,
            failure_threshold=0.8  # Greater than success - invalid
        )
        
        config = ComprehensiveGRPOConfig(
            grpo_algorithm=GRPOConfig(),
            experience_management=ExperienceConfig(),
            async_training=AsyncTrainingConfig(),
            policy_network=PolicyNetworkConfig(),
            value_network=ValueNetworkConfig(),
            curriculum=curriculum
        )
        
        with pytest.raises(ValueError, match="success_threshold must be greater than failure_threshold"):
            validate_comprehensive_grpo_config(config)


class TestConfigFactories:
    """Test configuration factory functions."""
    
    def test_standard_config_factory(self):
        """Test standard config factory."""
        config = create_standard_grpo_config(
            max_training_steps=25000,
            batch_size=16,
            buffer_size=5000
        )
        
        assert config.max_training_steps == 25000
        assert config.async_training.batch_size == 16
        assert config.experience_management.max_buffer_size == 5000
        assert config.training_mode == TrainingMode.FULL_GRPO
        assert config.optimization_level == OptimizationLevel.PRODUCTION
        
        # Should pass validation
        validate_comprehensive_grpo_config(config)
    
    def test_research_config_factory(self):
        """Test research config factory."""
        config = create_research_grpo_config(
            max_training_steps=75000,
            enable_curriculum=True,
            enable_adaptive=True
        )
        
        assert config.max_training_steps == 75000
        assert config.curriculum.enable_curriculum is True
        assert config.adaptive.enable_adaptive_lr is True
        assert config.training_mode == TrainingMode.CURRICULUM
        assert config.optimization_level == OptimizationLevel.RESEARCH
        assert config.experience_management.prioritized_replay is True
        assert config.value_network.ensemble_size == 3
        assert config.logging.enable_tensorboard is True
        
        # Should pass validation
        validate_comprehensive_grpo_config(config)
    
    def test_production_config_factory(self):
        """Test production config factory."""
        config = create_production_grpo_config(
            max_training_steps=30000,
            batch_size=32,
            enable_checkpointing=True
        )
        
        assert config.max_training_steps == 30000
        assert config.async_training.batch_size == 32
        assert config.checkpointing.enable_checkpointing is True
        assert config.optimization_level == OptimizationLevel.PRODUCTION
        assert config.compile_mode == "jit"
        assert config.precision == "float32"
        
        # Should pass validation
        validate_comprehensive_grpo_config(config)
    
    def test_debug_config_factory(self):
        """Test debug config factory."""
        config = create_debug_grpo_config(
            max_training_steps=500,
            batch_size=4
        )
        
        assert config.max_training_steps == 500
        assert config.async_training.batch_size == 4
        assert config.optimization_level == OptimizationLevel.DEBUG
        assert config.async_training.enable_compilation is False
        assert config.compile_mode == "eager"
        assert config.logging.log_level == "DEBUG"
        assert config.policy_network.hidden_dims == (64, 32)
        
        # Should pass validation
        validate_comprehensive_grpo_config(config)
    
    def test_recommended_config_small_problem(self):
        """Test recommended config for small problems."""
        config = get_recommended_config_for_problem_size(
            n_variables=5,
            max_samples=1000,
            computational_budget="low"
        )
        
        assert config.optimization_level == OptimizationLevel.DEBUG
        assert config.max_training_steps == 5000
        
        # Should pass validation
        validate_comprehensive_grpo_config(config)
    
    def test_recommended_config_large_problem(self):
        """Test recommended config for large problems."""
        config = get_recommended_config_for_problem_size(
            n_variables=25,
            max_samples=10000,
            computational_budget="high"
        )
        
        assert config.optimization_level == OptimizationLevel.RESEARCH
        assert config.max_training_steps == 100000
        # Should have larger networks for large problems
        assert config.policy_network.hidden_dims[0] > 512
        assert config.value_network.hidden_dims[0] > 512
        
        # Should pass validation
        validate_comprehensive_grpo_config(config)
    
    def test_recommended_config_medium_problem(self):
        """Test recommended config for medium problems."""
        config = get_recommended_config_for_problem_size(
            n_variables=10,
            max_samples=5000,
            computational_budget="medium"
        )
        
        assert config.optimization_level == OptimizationLevel.PRODUCTION
        assert config.max_training_steps == 25000
        
        # Should pass validation
        validate_comprehensive_grpo_config(config)


class TestConfigIntegration:
    """Test integration scenarios for GRPO configuration."""
    
    def test_all_factory_configs_are_valid(self):
        """Test that all factory configs pass validation."""
        configs = [
            create_standard_grpo_config(),
            create_research_grpo_config(),
            create_production_grpo_config(),
            create_debug_grpo_config(),
            get_recommended_config_for_problem_size(5, 1000, "low"),
            get_recommended_config_for_problem_size(15, 5000, "medium"),
            get_recommended_config_for_problem_size(30, 10000, "high"),
        ]
        
        for config in configs:
            validate_comprehensive_grpo_config(config)
    
    def test_subsystem_compatibility(self):
        """Test that subsystem configurations are compatible."""
        config = create_standard_grpo_config(batch_size=64)
        
        # Batch sizes should be consistent
        assert config.async_training.batch_size == config.experience_management.batch_size
        
        # Buffer size should be reasonable relative to batch size
        assert config.experience_management.max_buffer_size > config.experience_management.batch_size * 10
        
        # Min replay size should be achievable
        assert config.experience_management.min_replay_size < config.experience_management.max_buffer_size
    
    def test_config_modification_immutability(self):
        """Test that configurations remain immutable after creation."""
        config = create_standard_grpo_config()
        
        # All major components should be immutable
        with pytest.raises(AttributeError):
            config.max_training_steps = 50000
        
        with pytest.raises(AttributeError):
            config.grpo_algorithm.learning_rate = 1e-3
        
        with pytest.raises(AttributeError):
            config.policy_network.hidden_dims = (128, 64)
    
    def test_config_serialization_readiness(self):
        """Test that configs can be converted to dictionaries (serialization ready)."""
        config = create_research_grpo_config()
        
        # Should be able to access all attributes as dict
        config_dict = config.__dict__
        
        assert 'grpo_algorithm' in config_dict
        assert 'experience_management' in config_dict
        assert 'async_training' in config_dict
        assert 'policy_network' in config_dict
        assert 'value_network' in config_dict
        assert 'max_training_steps' in config_dict
        
        # Nested configs should also be accessible
        grpo_dict = config_dict['grpo_algorithm'].__dict__
        assert 'learning_rate' in grpo_dict
        assert 'clip_ratio' in grpo_dict