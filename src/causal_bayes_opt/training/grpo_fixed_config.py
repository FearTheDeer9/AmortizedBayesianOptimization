"""GRPO Configuration with Collapse Prevention Fixes.

This module provides GRPO configurations that incorporate the fixes identified
during the posterior collapse investigation:

1. Global standardization instead of per-variable standardization
2. Increased entropy coefficient (0.1 instead of 0.01)
3. Bootstrap surrogate with structural priors
4. Adaptive reward system that shifts from discovery to optimization

These fixes prevent the collapse where all variable embeddings become identical
(similarity > 0.96) after ~100 episodes of training.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .grpo_config import (
    ComprehensiveGRPOConfig,
    PolicyNetworkConfig,
    TrainingMode,
    OptimizationLevel,
    CheckpointingConfig,
    LoggingConfig,
    AdaptiveConfig
)
from ..acquisition.grpo import GRPOConfig
from .experience_management import ExperienceConfig
from .async_training import AsyncTrainingConfig
from ..surrogate.phase_manager import PhaseConfig
from ..surrogate.bootstrap import BootstrapConfig


@dataclass(frozen=True)
class EnrichedEncoderConfig:
    """Configuration for EnrichedAttentionEncoder to prevent collapse."""
    num_layers: int = 2
    num_heads: int = 4  # Matches notebook configuration
    hidden_dim: int = 128
    dropout: float = 0.1
    use_residual: bool = True
    use_layer_norm: bool = True


@dataclass(frozen=True)
class StateEnrichmentConfig:
    """Configuration for state enrichment with global standardization."""
    standardize_values: bool = True
    use_global_standardization: bool = True  # Key fix for collapse
    channels: Tuple[str, ...] = ("values", "interventions", "target", "parent_probs", "recency")


@dataclass(frozen=True)
class AdaptiveRewardConfig:
    """Configuration for adaptive reward system."""
    enabled: bool = True
    structure_threshold: float = 0.95
    adaptation_rate: float = 0.1
    initial_weights: Dict[str, float] = field(default_factory=lambda: {
        "discovery": 0.7,
        "optimization": 0.3
    })
    final_weights: Dict[str, float] = field(default_factory=lambda: {
        "discovery": 0.05,
        "optimization": 0.95
    })
    update_frequency: int = 10


@dataclass(frozen=True)
class CollapseMonitoringConfig:
    """Configuration for monitoring embedding collapse."""
    monitor_embeddings: bool = True
    embedding_similarity_threshold: float = 0.95
    parent_prob_variance_threshold: float = 0.01
    log_frequency: int = 100


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Configuration for early stopping to prevent over-training."""
    enabled: bool = True
    convergence_accuracy_threshold: float = 0.95
    convergence_patience: int = 10
    min_episodes_per_scm: int = 20
    max_episodes_per_scm: int = 100
    reward_variance_threshold: float = 0.1


def create_grpo_config_with_fixes(
    max_training_steps: int = 50000,
    batch_size: int = 64,
    group_size: int = 64,
    use_bootstrap: bool = True,
    use_adaptive_rewards: bool = True,
    entropy_coefficient: float = 0.1,  # Increased from default 0.01
    use_early_stopping: bool = True
) -> ComprehensiveGRPOConfig:
    """Create GRPO configuration with collapse prevention fixes.
    
    This configuration incorporates all fixes identified during the investigation:
    - Global standardization in state enrichment
    - Increased entropy coefficient for exploration
    - Bootstrap surrogate with structural priors
    - Adaptive reward system
    
    Args:
        max_training_steps: Maximum training steps
        batch_size: Training batch size
        group_size: GRPO group size for advantage computation
        use_bootstrap: Whether to use bootstrap surrogate
        use_adaptive_rewards: Whether to use adaptive reward system
        entropy_coefficient: Entropy coefficient for exploration
        
    Returns:
        GRPO configuration with collapse prevention measures
    """
    # GRPO algorithm config with increased entropy
    grpo_config = GRPOConfig(
        group_size=group_size,
        interventions_per_state=min(16, group_size),  # Same-state batching
        clip_ratio=0.2,
        entropy_coeff=entropy_coefficient,  # Key fix: increased from 0.01
        kl_penalty_coeff=0.0,
        max_grad_norm=1.0,
        learning_rate=3e-4,
        num_iterations=4,
        scale_rewards=True
    )
    
    # Experience management with appropriate buffer size
    experience_config = ExperienceConfig(
        max_buffer_size=20000,
        batch_size=batch_size,
        min_replay_size=batch_size * 5,
        prioritized_replay=False,  # Not needed with good exploration
        memory_limit_mb=2048
    )
    
    # Async training configuration
    async_config = AsyncTrainingConfig(
        batch_size=batch_size,
        max_parallel_envs=min(32, batch_size),
        enable_compilation=True
    )
    
    # Policy network for enriched encoder
    policy_network = PolicyNetworkConfig(
        hidden_dims=(256, 128),
        activation="relu",
        dropout_rate=0.1,
        use_batch_norm=False,  # Faster without batch norm
        attention_heads=4,  # For enriched encoder
        embedding_dim=128
    )
    
    # Adaptive configuration if enabled
    adaptive_config = AdaptiveConfig(
        enable_adaptive_lr=False,  # Focus on reward adaptation
        enable_adaptive_exploration=use_adaptive_rewards,
        enable_adaptive_curriculum=False,
        performance_window=1000,
        adaptation_frequency=100
    )
    
    # Checkpointing with collapse monitoring
    checkpointing = CheckpointingConfig(
        enable_checkpointing=True,
        checkpoint_frequency=1000,
        keep_best_only=False,
        save_optimizer_state=True,
        checkpoint_dir="./checkpoints/grpo_with_fixes",
        max_checkpoints=5,
        save_metrics=True
    )
    
    # Logging with embedding monitoring
    logging_config = LoggingConfig(
        log_level="INFO",
        log_frequency=100,
        log_gradients=False,
        log_weights=False,
        enable_tensorboard=True,
        project_name="causal_bayes_opt_fixed"
    )
    
    # Early stopping configuration
    early_stopping_config = None
    if use_early_stopping:
        early_stopping_config = {
            'early_stopping_enabled': True,
            'convergence_accuracy_threshold': 0.95,
            'convergence_patience': 10,
            'min_episodes_per_scm': 20,
            'max_episodes_per_scm': 100,
            'reward_variance_threshold': 0.1
        }
    
    return ComprehensiveGRPOConfig(
        grpo_algorithm=grpo_config,
        experience_management=experience_config,
        async_training=async_config,
        policy_network=policy_network,
        adaptive=adaptive_config,
        checkpointing=checkpointing,
        logging=logging_config,
        early_stopping=early_stopping_config,
        max_training_steps=max_training_steps,
        training_mode=TrainingMode.ADAPTIVE if use_adaptive_rewards else TrainingMode.FULL_GRPO,
        optimization_level=OptimizationLevel.PRODUCTION
    )


def create_bootstrap_phase_config(
    bootstrap_steps: int = 100,
    transition_steps: int = 50
) -> PhaseConfig:
    """Create phase configuration for bootstrap surrogate.
    
    Args:
        bootstrap_steps: Steps to use bootstrap surrogate
        transition_steps: Steps for gradual transition
        
    Returns:
        Phase configuration for bootstrap surrogate
    """
    return PhaseConfig(
        bootstrap_steps=bootstrap_steps,
        transition_steps=transition_steps,
        exploration_noise_start=0.5,
        exploration_noise_end=0.1,
        transition_schedule="linear"
    )


def create_bootstrap_config() -> BootstrapConfig:
    """Create bootstrap configuration with structural priors.
    
    Returns:
        Bootstrap configuration for initial surrogate
    """
    return BootstrapConfig(
        structure_encoding_dim=128,
        use_graph_distance=True,
        use_structural_priors=True,
        min_noise_factor=0.05,
        noise_schedule="linear_decay"
    )


def create_research_config_with_fixes(
    max_training_steps: int = 100000,
    enable_all_monitoring: bool = True
) -> ComprehensiveGRPOConfig:
    """Create research configuration with all fixes and extensive monitoring.
    
    Args:
        max_training_steps: Maximum training steps
        enable_all_monitoring: Enable all diagnostic monitoring
        
    Returns:
        Research GRPO configuration with fixes
    """
    base_config = create_grpo_config_with_fixes(
        max_training_steps=max_training_steps,
        batch_size=64,
        group_size=64,
        entropy_coefficient=0.1
    )
    
    # Enhanced logging for research
    research_logging = LoggingConfig(
        log_level="DEBUG" if enable_all_monitoring else "INFO",
        log_frequency=50,
        log_gradients=enable_all_monitoring,
        log_weights=enable_all_monitoring,
        log_activations=enable_all_monitoring,
        enable_tensorboard=True,
        enable_wandb=True,
        project_name="causal_bayes_opt_research",
        tags=["grpo", "fixed", "research", "monitoring"]
    )
    
    # Larger policy network for research
    research_policy = PolicyNetworkConfig(
        hidden_dims=(512, 256, 128),
        activation="relu", 
        dropout_rate=0.2,
        use_batch_norm=True,
        use_residual=True,
        attention_heads=8,
        embedding_dim=256
    )
    
    # Update configuration
    return ComprehensiveGRPOConfig(
        grpo_algorithm=base_config.grpo_algorithm,
        experience_management=base_config.experience_management,
        async_training=base_config.async_training,
        policy_network=research_policy,
        adaptive=base_config.adaptive,
        checkpointing=base_config.checkpointing,
        logging=research_logging,
        max_training_steps=max_training_steps,
        training_mode=TrainingMode.ADAPTIVE,
        optimization_level=OptimizationLevel.RESEARCH
    )


def create_quick_training_config_with_early_stopping(
    n_episodes: int = 200,
    n_scms: int = 10
) -> Dict[str, any]:
    """Create quick training configuration with early stopping enabled.
    
    This configuration is optimized for fast training runs that demonstrate
    the early stopping functionality.
    
    Args:
        n_episodes: Total number of episodes
        n_scms: Number of SCMs to rotate through
        
    Returns:
        Training configuration dictionary
    """
    return {
        'training': {
            'n_episodes': n_episodes,
            'episode_length': 10,
            'learning_rate': 3e-4,
            'batch_size': 64,
            'reward_weights': {
                'optimization': 0.3,
                'discovery': 0.7,
                'efficiency': 0.0
            },
            'use_real_observations': True,
            'early_stopping_enabled': True,
            'convergence_accuracy_threshold': 0.95,
            'convergence_patience': 10,
            'min_episodes_per_scm': 10,
            'max_episodes_per_scm': 50
        },
        'experiment': {
            'scm_generation': {
                'use_variable_factory': True,
                'num_scms': n_scms,
                'variable_range': [3, 5],
                'structure_types': ['fork', 'chain', 'collider'],
                'rotation_frequency': 20  # Fallback if early stopping disabled
            }
        },
        'optimization': {
            'direction': 'MINIMIZE'
        },
        'seed': 42
    }


def validate_fixed_config(config: ComprehensiveGRPOConfig) -> None:
    """Validate that configuration includes collapse prevention fixes.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValueError: If configuration is missing critical fixes
    """
    # Check entropy coefficient
    if config.grpo_algorithm.entropy_coeff < 0.05:
        raise ValueError(
            f"Entropy coefficient {config.grpo_algorithm.entropy_coeff} is too low. "
            "Recommended minimum is 0.05 to prevent collapse."
        )
    
    # Check for attention heads (needed for enriched encoder)
    if config.policy_network.attention_heads is None:
        raise ValueError(
            "Policy network must have attention_heads configured for enriched encoder."
        )
    
    # Warn about exploration settings
    if not config.adaptive.enable_adaptive_exploration:
        import logging
        logging.warning(
            "Adaptive exploration is disabled. This may lead to collapse "
            "after structure discovery phase."
        )


# Export configurations
__all__ = [
    "create_grpo_config_with_fixes",
    "create_bootstrap_phase_config", 
    "create_bootstrap_config",
    "create_research_config_with_fixes",
    "validate_fixed_config",
    "EnrichedEncoderConfig",
    "StateEnrichmentConfig",
    "AdaptiveRewardConfig",
    "CollapseMonitoringConfig"
]