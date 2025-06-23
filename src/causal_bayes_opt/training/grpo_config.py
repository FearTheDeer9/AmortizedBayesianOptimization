"""GRPO Configuration System for Enhanced Training.

This module provides comprehensive configuration management for GRPO training,
integrating the core algorithm, experience management, reward systems, and
training infrastructure into a unified configuration framework.

Key features:
- Hierarchical configuration system with validation
- Integration with reward rubric and diversity monitoring
- Support for curriculum learning and adaptive training
- Configuration factories for different training scenarios
- Compatibility validation between different subsystems
- Performance optimization settings

All configurations follow functional programming principles with immutable
data structures and comprehensive validation.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from ..acquisition.reward_rubric import CausalRewardRubric
from ..environments.intervention_env import EnvironmentConfig
from .async_training import AsyncTrainingConfig
from .diversity_monitor import DiversityMonitor
from .experience_management import ExperienceConfig
from .grpo_core import GRPOConfig, validate_grpo_config

logger = logging.getLogger(__name__)


class TrainingMode(Enum):
    """Training mode configuration."""
    BEHAVIORAL_CLONING = "behavioral_cloning"
    GRPO_FINE_TUNING = "grpo_fine_tuning"
    FULL_GRPO = "full_grpo"
    CURRICULUM = "curriculum"
    ADAPTIVE = "adaptive"


class OptimizationLevel(Enum):
    """Performance optimization level."""
    DEBUG = "debug"           # No optimization, full debugging
    DEVELOPMENT = "development"  # Basic optimization
    PRODUCTION = "production"    # Full optimization
    RESEARCH = "research"        # Research-focused settings


@dataclass(frozen=True)
class PolicyNetworkConfig:
    """Configuration for policy network architecture.
    
    Args:
        hidden_dims: Hidden layer dimensions
        activation: Activation function name
        dropout_rate: Dropout rate for regularization
        use_batch_norm: Whether to use batch normalization
        use_residual: Whether to use residual connections
        attention_heads: Number of attention heads (if using attention)
        max_sequence_length: Maximum sequence length for attention
        embedding_dim: Embedding dimension for categorical variables
    """
    hidden_dims: Tuple[int, ...] = (256, 128, 64)
    activation: str = "relu"
    dropout_rate: float = 0.1
    use_batch_norm: bool = True
    use_residual: bool = False
    attention_heads: Optional[int] = None
    max_sequence_length: int = 100
    embedding_dim: int = 32


@dataclass(frozen=True)
class ValueNetworkConfig:
    """Configuration for value network architecture.
    
    Args:
        hidden_dims: Hidden layer dimensions
        activation: Activation function name
        dropout_rate: Dropout rate for regularization
        use_batch_norm: Whether to use batch normalization
        use_residual: Whether to use residual connections
        shared_layers: Number of layers to share with policy network
        ensemble_size: Number of value networks in ensemble (if using)
    """
    hidden_dims: Tuple[int, ...] = (256, 128)
    activation: str = "relu"
    dropout_rate: float = 0.1
    use_batch_norm: bool = True
    use_residual: bool = False
    shared_layers: int = 0
    ensemble_size: int = 1


@dataclass(frozen=True)
class CurriculumConfig:
    """Configuration for curriculum learning.
    
    Args:
        enable_curriculum: Whether to use curriculum learning
        difficulty_schedule: Difficulty progression schedule
        adaptation_rate: Rate of difficulty adaptation
        success_threshold: Success rate threshold for advancement
        failure_threshold: Failure rate threshold for regression
        min_episodes_per_level: Minimum episodes before advancement
        max_episodes_per_level: Maximum episodes per difficulty level
        difficulty_factors: Factors that control difficulty
    """
    enable_curriculum: bool = False
    difficulty_schedule: str = "linear"  # "linear", "exponential", "adaptive"
    adaptation_rate: float = 0.1
    success_threshold: float = 0.8
    failure_threshold: float = 0.3
    min_episodes_per_level: int = 100
    max_episodes_per_level: int = 1000
    difficulty_factors: Dict[str, float] = field(default_factory=lambda: {
        "noise_level": 0.1,
        "intervention_cost": 1.0,
        "target_complexity": 0.5
    })


@dataclass(frozen=True)
class AdaptiveConfig:
    """Configuration for adaptive training strategies.
    
    Args:
        enable_adaptive_lr: Whether to use adaptive learning rates
        enable_adaptive_exploration: Whether to use adaptive exploration
        enable_adaptive_curriculum: Whether to use adaptive curriculum
        performance_window: Window size for performance tracking
        adaptation_frequency: Frequency of adaptation (in steps)
        lr_adaptation_factor: Learning rate adaptation factor
        exploration_adaptation_factor: Exploration adaptation factor
        early_stopping_patience: Patience for early stopping
        target_performance: Target performance level
    """
    enable_adaptive_lr: bool = True
    enable_adaptive_exploration: bool = True
    enable_adaptive_curriculum: bool = False
    performance_window: int = 1000
    adaptation_frequency: int = 100
    lr_adaptation_factor: float = 0.9
    exploration_adaptation_factor: float = 0.95
    early_stopping_patience: int = 10000
    target_performance: float = 0.9


@dataclass(frozen=True)
class CheckpointingConfig:
    """Configuration for model checkpointing and saving.
    
    Args:
        enable_checkpointing: Whether to save checkpoints
        checkpoint_frequency: Frequency of checkpointing (in steps)
        keep_best_only: Whether to keep only the best checkpoint
        save_optimizer_state: Whether to save optimizer state
        checkpoint_dir: Directory for saving checkpoints
        max_checkpoints: Maximum number of checkpoints to keep
        save_metrics: Whether to save training metrics
        compression_level: Compression level for checkpoints
    """
    enable_checkpointing: bool = True
    checkpoint_frequency: int = 1000
    keep_best_only: bool = False
    save_optimizer_state: bool = True
    checkpoint_dir: str = "./checkpoints"
    max_checkpoints: int = 5
    save_metrics: bool = True
    compression_level: int = 6


@dataclass(frozen=True)
class LoggingConfig:
    """Configuration for training logging and monitoring.
    
    Args:
        log_level: Logging level
        log_frequency: Frequency of logging (in steps)
        log_gradients: Whether to log gradient statistics
        log_weights: Whether to log weight statistics
        log_activations: Whether to log activation statistics
        enable_tensorboard: Whether to use TensorBoard logging
        enable_wandb: Whether to use Weights & Biases logging
        project_name: Project name for experiment tracking
        tags: Tags for experiment organization
    """
    log_level: str = "INFO"
    log_frequency: int = 100
    log_gradients: bool = False
    log_weights: bool = False
    log_activations: bool = False
    enable_tensorboard: bool = False
    enable_wandb: bool = False
    project_name: str = "causal_bayes_opt"
    tags: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ComprehensiveGRPOConfig:
    """Comprehensive GRPO configuration integrating all subsystems.
    
    This is the main configuration class that coordinates all aspects of
    GRPO training including algorithm parameters, network architectures,
    experience management, and training infrastructure.
    
    Args:
        grpo_algorithm: Core GRPO algorithm configuration
        experience_management: Experience replay configuration
        async_training: Async training infrastructure configuration
        policy_network: Policy network architecture configuration
        value_network: Value network architecture configuration
        reward_rubric: Reward system configuration
        environment_config: Environment configuration
        diversity_monitor: Diversity monitoring configuration
        curriculum: Curriculum learning configuration
        adaptive: Adaptive training configuration
        checkpointing: Checkpointing configuration
        logging: Logging configuration
        training_mode: Training mode selection
        optimization_level: Performance optimization level
        max_training_steps: Maximum number of training steps
        evaluation_frequency: Frequency of evaluation (in steps)
        seed: Random seed for reproducibility
        device: Device for training ('cpu', 'gpu', 'auto')
        precision: Numerical precision ('float32', 'float16', 'bfloat16')
        compile_mode: JAX compilation mode ('eager', 'jit', 'aot')
    """
    # Core configurations
    grpo_algorithm: GRPOConfig
    experience_management: ExperienceConfig
    async_training: AsyncTrainingConfig

    # Network architectures
    policy_network: PolicyNetworkConfig
    value_network: ValueNetworkConfig

    # Training systems
    reward_rubric: Optional[CausalRewardRubric] = None
    environment_config: Optional[EnvironmentConfig] = None
    diversity_monitor: Optional[DiversityMonitor] = None

    # Training strategies
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)

    # Infrastructure
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Training parameters
    training_mode: TrainingMode = TrainingMode.FULL_GRPO
    optimization_level: OptimizationLevel = OptimizationLevel.PRODUCTION
    max_training_steps: int = 100000
    evaluation_frequency: int = 1000
    seed: int = 42
    device: str = "auto"
    precision: str = "float32"
    compile_mode: str = "jit"

    def __post_init__(self):
        """Initialize configuration (validation is manual)."""
        # Validation is performed manually via validate_comprehensive_grpo_config()
        # This allows for creating invalid configs for testing purposes
        pass


def validate_comprehensive_grpo_config(config: ComprehensiveGRPOConfig) -> None:
    """Validate comprehensive GRPO configuration.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate core GRPO config
    validate_grpo_config(config.grpo_algorithm)

    # Validate training parameters
    if config.max_training_steps <= 0:
        raise ValueError(f"max_training_steps must be positive, got {config.max_training_steps}")

    if config.evaluation_frequency <= 0:
        raise ValueError(f"evaluation_frequency must be positive, got {config.evaluation_frequency}")

    if config.seed < 0:
        raise ValueError(f"seed must be non-negative, got {config.seed}")

    # Validate device
    valid_devices = {"cpu", "gpu", "auto"}
    if config.device not in valid_devices:
        raise ValueError(f"device must be one of {valid_devices}, got {config.device}")

    # Validate precision
    valid_precisions = {"float32", "float16", "bfloat16"}
    if config.precision not in valid_precisions:
        raise ValueError(f"precision must be one of {valid_precisions}, got {config.precision}")

    # Validate compile mode
    valid_compile_modes = {"eager", "jit", "aot"}
    if config.compile_mode not in valid_compile_modes:
        raise ValueError(f"compile_mode must be one of {valid_compile_modes}, got {config.compile_mode}")

    # Validate network configurations
    _validate_network_config(config.policy_network, "policy_network")
    _validate_network_config(config.value_network, "value_network")

    # Validate curriculum configuration
    if config.curriculum.enable_curriculum:
        _validate_curriculum_config(config.curriculum)

    # Validate adaptive configuration
    _validate_adaptive_config(config.adaptive)

    # Validate compatibility between subsystems
    _validate_subsystem_compatibility(config)

    logger.info("GRPO configuration validation successful")


def _validate_network_config(network_config: Union[PolicyNetworkConfig, ValueNetworkConfig], name: str) -> None:
    """Validate network configuration."""
    if not network_config.hidden_dims:
        raise ValueError(f"{name}.hidden_dims cannot be empty")

    if any(dim <= 0 for dim in network_config.hidden_dims):
        raise ValueError(f"{name}.hidden_dims must contain only positive integers")

    if not 0 <= network_config.dropout_rate <= 1:
        raise ValueError(f"{name}.dropout_rate must be in [0,1], got {network_config.dropout_rate}")

    if hasattr(network_config, 'embedding_dim') and network_config.embedding_dim <= 0:
        raise ValueError(f"{name}.embedding_dim must be positive, got {network_config.embedding_dim}")


def _validate_curriculum_config(curriculum: CurriculumConfig) -> None:
    """Validate curriculum learning configuration."""
    valid_schedules = {"linear", "exponential", "adaptive"}
    if curriculum.difficulty_schedule not in valid_schedules:
        raise ValueError(f"difficulty_schedule must be one of {valid_schedules}")

    if not 0 < curriculum.adaptation_rate <= 1:
        raise ValueError(f"adaptation_rate must be in (0,1], got {curriculum.adaptation_rate}")

    if not 0 <= curriculum.success_threshold <= 1:
        raise ValueError(f"success_threshold must be in [0,1], got {curriculum.success_threshold}")

    if not 0 <= curriculum.failure_threshold <= 1:
        raise ValueError(f"failure_threshold must be in [0,1], got {curriculum.failure_threshold}")

    if curriculum.success_threshold <= curriculum.failure_threshold:
        raise ValueError("success_threshold must be greater than failure_threshold")

    if curriculum.min_episodes_per_level <= 0:
        raise ValueError("min_episodes_per_level must be positive")

    if curriculum.max_episodes_per_level < curriculum.min_episodes_per_level:
        raise ValueError("max_episodes_per_level must be >= min_episodes_per_level")


def _validate_adaptive_config(adaptive: AdaptiveConfig) -> None:
    """Validate adaptive training configuration."""
    if adaptive.performance_window <= 0:
        raise ValueError(f"performance_window must be positive, got {adaptive.performance_window}")

    if adaptive.adaptation_frequency <= 0:
        raise ValueError(f"adaptation_frequency must be positive, got {adaptive.adaptation_frequency}")

    if not 0 < adaptive.lr_adaptation_factor <= 1:
        raise ValueError(f"lr_adaptation_factor must be in (0,1], got {adaptive.lr_adaptation_factor}")

    if not 0 < adaptive.exploration_adaptation_factor <= 1:
        raise ValueError(f"exploration_adaptation_factor must be in (0,1], got {adaptive.exploration_adaptation_factor}")

    if adaptive.early_stopping_patience <= 0:
        raise ValueError(f"early_stopping_patience must be positive, got {adaptive.early_stopping_patience}")


def _validate_subsystem_compatibility(config: ComprehensiveGRPOConfig) -> None:
    """Validate compatibility between different subsystems."""
    # Check batch size compatibility
    grpo_batch = config.async_training.batch_size
    exp_batch = config.experience_management.batch_size

    if grpo_batch != exp_batch:
        logger.warning(
            f"Batch size mismatch: async_training={grpo_batch}, experience_management={exp_batch}. "
            "Using async_training batch size."
        )

    # Check buffer size compatibility
    min_replay = config.experience_management.min_replay_size
    max_buffer = config.experience_management.max_buffer_size

    if min_replay >= max_buffer:
        raise ValueError(
            f"min_replay_size ({min_replay}) must be less than max_buffer_size ({max_buffer})"
        )

    # Check training mode compatibility
    if config.training_mode == TrainingMode.CURRICULUM and not config.curriculum.enable_curriculum:
        raise ValueError("Training mode is CURRICULUM but curriculum is not enabled")

    if config.training_mode == TrainingMode.ADAPTIVE and not any([
        config.adaptive.enable_adaptive_lr,
        config.adaptive.enable_adaptive_exploration,
        config.adaptive.enable_adaptive_curriculum
    ]):
        raise ValueError("Training mode is ADAPTIVE but no adaptive features are enabled")


# Factory functions for common configurations
def create_standard_grpo_config(
    max_training_steps: int = 50000,
    batch_size: int = 32,
    buffer_size: int = 10000
) -> ComprehensiveGRPOConfig:
    """Create standard GRPO configuration for general use.
    
    Args:
        max_training_steps: Maximum training steps
        batch_size: Training batch size
        buffer_size: Experience buffer size
        
    Returns:
        Standard GRPO configuration
    """
    grpo_config = GRPOConfig()

    experience_config = ExperienceConfig(
        max_buffer_size=buffer_size,
        batch_size=batch_size,
        min_replay_size=batch_size * 10
    )

    async_config = AsyncTrainingConfig(
        batch_size=batch_size,
        max_parallel_envs=min(16, batch_size),
        enable_compilation=True
    )

    return ComprehensiveGRPOConfig(
        grpo_algorithm=grpo_config,
        experience_management=experience_config,
        async_training=async_config,
        policy_network=PolicyNetworkConfig(),
        value_network=ValueNetworkConfig(),
        max_training_steps=max_training_steps,
        training_mode=TrainingMode.FULL_GRPO,
        optimization_level=OptimizationLevel.PRODUCTION
    )


def create_research_grpo_config(
    max_training_steps: int = 100000,
    enable_curriculum: bool = True,
    enable_adaptive: bool = True
) -> ComprehensiveGRPOConfig:
    """Create research-focused GRPO configuration.
    
    Args:
        max_training_steps: Maximum training steps
        enable_curriculum: Whether to enable curriculum learning
        enable_adaptive: Whether to enable adaptive training
        
    Returns:
        Research GRPO configuration
    """
    # More conservative GRPO settings for research
    grpo_config = GRPOConfig(
        learning_rate=1e-4,
        value_learning_rate=5e-4,
        clip_ratio=0.1,
        entropy_coefficient=0.02
    )

    # Larger buffer for better sample diversity
    experience_config = ExperienceConfig(
        max_buffer_size=50000,
        batch_size=64,
        min_replay_size=5000,
        prioritized_replay=True
    )

    async_config = AsyncTrainingConfig(
        batch_size=64,
        max_parallel_envs=32,
        enable_compilation=True
    )

    # Larger networks for research
    policy_network = PolicyNetworkConfig(
        hidden_dims=(512, 256, 128),
        dropout_rate=0.2,
        use_batch_norm=True
    )

    value_network = ValueNetworkConfig(
        hidden_dims=(512, 256),
        dropout_rate=0.2,
        ensemble_size=3  # Value ensemble for better estimation
    )

    curriculum = CurriculumConfig(
        enable_curriculum=enable_curriculum,
        difficulty_schedule="adaptive",
        success_threshold=0.85,
        failure_threshold=0.3
    )

    adaptive = AdaptiveConfig(
        enable_adaptive_lr=enable_adaptive,
        enable_adaptive_exploration=enable_adaptive,
        enable_adaptive_curriculum=enable_adaptive and enable_curriculum
    )

    logging_config = LoggingConfig(
        log_frequency=50,
        log_gradients=True,
        log_weights=True,
        enable_tensorboard=True
    )

    return ComprehensiveGRPOConfig(
        grpo_algorithm=grpo_config,
        experience_management=experience_config,
        async_training=async_config,
        policy_network=policy_network,
        value_network=value_network,
        curriculum=curriculum,
        adaptive=adaptive,
        logging=logging_config,
        max_training_steps=max_training_steps,
        training_mode=TrainingMode.CURRICULUM if enable_curriculum else TrainingMode.FULL_GRPO,
        optimization_level=OptimizationLevel.RESEARCH
    )


def create_production_grpo_config(
    max_training_steps: int = 25000,
    batch_size: int = 64,
    enable_checkpointing: bool = True
) -> ComprehensiveGRPOConfig:
    """Create production-ready GRPO configuration.
    
    Args:
        max_training_steps: Maximum training steps
        batch_size: Training batch size
        enable_checkpointing: Whether to enable checkpointing
        
    Returns:
        Production GRPO configuration
    """
    # High-performance GRPO settings
    grpo_config = GRPOConfig(
        learning_rate=5e-4,
        value_learning_rate=1e-3,
        clip_ratio=0.2,
        entropy_coefficient=0.01,
        max_grad_norm=1.0
    )

    # Memory-efficient experience management
    experience_config = ExperienceConfig(
        max_buffer_size=20000,
        batch_size=batch_size,
        min_replay_size=batch_size * 5,
        memory_limit_mb=2048
    )

    async_config = AsyncTrainingConfig(
        batch_size=batch_size,
        max_parallel_envs=batch_size,
        enable_compilation=True,
        max_memory_mb=8192
    )

    # Optimized network architectures
    policy_network = PolicyNetworkConfig(
        hidden_dims=(256, 128),
        dropout_rate=0.1,
        use_batch_norm=False  # Faster without batch norm
    )

    value_network = ValueNetworkConfig(
        hidden_dims=(256, 128),
        dropout_rate=0.1,
        use_batch_norm=False
    )

    checkpointing = CheckpointingConfig(
        enable_checkpointing=enable_checkpointing,
        checkpoint_frequency=2500,
        keep_best_only=True,
        compression_level=9
    )

    logging_config = LoggingConfig(
        log_frequency=250,
        log_gradients=False,
        log_weights=False
    )

    return ComprehensiveGRPOConfig(
        grpo_algorithm=grpo_config,
        experience_management=experience_config,
        async_training=async_config,
        policy_network=policy_network,
        value_network=value_network,
        checkpointing=checkpointing,
        logging=logging_config,
        max_training_steps=max_training_steps,
        training_mode=TrainingMode.FULL_GRPO,
        optimization_level=OptimizationLevel.PRODUCTION,
        precision="float32",
        compile_mode="jit"
    )


def create_debug_grpo_config(
    max_training_steps: int = 1000,
    batch_size: int = 8
) -> ComprehensiveGRPOConfig:
    """Create debug GRPO configuration for development and testing.
    
    Args:
        max_training_steps: Maximum training steps
        batch_size: Training batch size
        
    Returns:
        Debug GRPO configuration
    """
    grpo_config = GRPOConfig(
        learning_rate=1e-3,
        value_learning_rate=1e-3
    )

    experience_config = ExperienceConfig(
        max_buffer_size=1000,
        batch_size=batch_size,
        min_replay_size=batch_size * 2
    )

    async_config = AsyncTrainingConfig(
        batch_size=batch_size,
        max_parallel_envs=4,
        enable_compilation=False,  # Disable for debugging
        progress_logging_interval=10
    )

    policy_network = PolicyNetworkConfig(
        hidden_dims=(64, 32),
        dropout_rate=0.0  # No dropout for debugging
    )

    value_network = ValueNetworkConfig(
        hidden_dims=(64, 32),
        dropout_rate=0.0
    )

    logging_config = LoggingConfig(
        log_level="DEBUG",
        log_frequency=10,
        log_gradients=True,
        log_weights=True
    )

    return ComprehensiveGRPOConfig(
        grpo_algorithm=grpo_config,
        experience_management=experience_config,
        async_training=async_config,
        policy_network=policy_network,
        value_network=value_network,
        logging=logging_config,
        max_training_steps=max_training_steps,
        training_mode=TrainingMode.FULL_GRPO,
        optimization_level=OptimizationLevel.DEBUG,
        compile_mode="eager"
    )


def get_recommended_config_for_problem_size(
    n_variables: int,
    max_samples: int,
    computational_budget: str = "medium"
) -> ComprehensiveGRPOConfig:
    """Get recommended configuration based on problem characteristics.
    
    Args:
        n_variables: Number of variables in the causal model
        max_samples: Maximum number of samples expected
        computational_budget: Computational budget ("low", "medium", "high")
        
    Returns:
        Recommended GRPO configuration
    """
    if computational_budget == "low":
        base_config = create_debug_grpo_config(max_training_steps=5000)
    elif computational_budget == "medium":
        base_config = create_standard_grpo_config(max_training_steps=25000)
    else:  # high
        base_config = create_research_grpo_config(max_training_steps=100000)

    # Adjust based on problem size
    if n_variables > 20:
        # Large problems need bigger networks
        policy_hidden = tuple(dim * 2 for dim in base_config.policy_network.hidden_dims)
        value_hidden = tuple(dim * 2 for dim in base_config.value_network.hidden_dims)

        policy_network = PolicyNetworkConfig(
            **{**base_config.policy_network.__dict__, 'hidden_dims': policy_hidden}
        )
        value_network = ValueNetworkConfig(
            **{**base_config.value_network.__dict__, 'hidden_dims': value_hidden}
        )

        # Larger buffer for complex problems
        buffer_size = min(base_config.experience_management.max_buffer_size * 2, 100000)
        experience_config = ExperienceConfig(
            **{**base_config.experience_management.__dict__, 'max_buffer_size': buffer_size}
        )

        return ComprehensiveGRPOConfig(
            **{**base_config.__dict__,
               'policy_network': policy_network,
               'value_network': value_network,
               'experience_management': experience_config}
        )

    return base_config
