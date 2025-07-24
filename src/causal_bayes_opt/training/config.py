#!/usr/bin/env python3
"""
Training Configuration System for ACBO

Provides immutable, validated configuration objects for training the surrogate
and acquisition models. Uses Pydantic for validation and functional programming
principles for configuration management.

Following open-r1 recommendations and best practices from the literature.

Related Files:
- surrogate_training.py: Uses SurrogateTrainingConfig for model training
- acquisition_training.py: Uses GRPOTrainingConfig for policy training  
- acquisition_config.py: Additional acquisition-specific configurations
- docs/architecture/adr/005_jax_performance_optimization.md: JAX compilation settings
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict
import pyrsistent as pyr

from ..acquisition.grpo import GRPOConfig


# ============================================================================
# Pydantic Validation Schemas
# ============================================================================

class GRPOTrainingSchema(BaseModel):
    """Pydantic schema for GRPO training configuration validation."""
    
    group_size: int = Field(default=64, ge=8, le=256, description="GRPO group size")
    num_iterations: int = Field(default=4, ge=1, le=10, description="Sample reuse iterations")
    learning_rate: float = Field(default=3e-4, gt=0, le=0.01, description="Learning rate")
    clip_ratio: float = Field(default=0.2, gt=0, le=1.0, description="Policy clipping ratio")
    entropy_coeff: float = Field(default=0.01, ge=0, le=1.0, description="Entropy coefficient")
    kl_penalty_coeff: float = Field(default=0.0, ge=0, le=1.0, description="KL penalty coefficient")
    max_grad_norm: float = Field(default=1.0, gt=0, description="Maximum gradient norm")
    scale_rewards: bool = Field(default=True, description="Whether to scale advantages")
    
    model_config = ConfigDict(extra="forbid")


class RewardTrainingSchema(BaseModel):
    """Pydantic schema for reward configuration validation."""
    
    optimization_weight: float = Field(default=1.0, ge=0, description="Optimization reward weight")
    structure_weight: float = Field(default=0.5, ge=0, description="Structure learning weight")
    parent_weight: float = Field(default=0.3, ge=0, description="Parent intervention weight")
    exploration_weight: float = Field(default=0.1, ge=0, description="Exploration bonus weight")
    
    model_config = ConfigDict(extra="forbid")


class ExplorationTrainingSchema(BaseModel):
    """Pydantic schema for exploration configuration validation."""
    
    strategy_type: str = Field(default="adaptive", description="Exploration strategy type")
    uncertainty_weight: float = Field(default=1.0, ge=0, description="Uncertainty weighting")
    initial_temperature: float = Field(default=2.0, gt=0, description="Initial temperature")
    final_temperature: float = Field(default=0.1, gt=0, description="Final temperature")
    
    @field_validator('final_temperature')
    def final_less_than_initial(cls, v, info):
        if info.data and 'initial_temperature' in info.data and v >= info.data['initial_temperature']:
            raise ValueError('Final temperature must be less than initial temperature')
        return v
    
    model_config = ConfigDict(extra="forbid")


class SurrogateTrainingSchema(BaseModel):
    """Pydantic schema for surrogate model training configuration."""
    
    model_hidden_dim: int = Field(default=128, ge=32, le=512, description="Hidden dimension")
    model_n_layers: int = Field(default=6, ge=1, le=16, description="Number of layers")
    learning_rate: float = Field(default=1e-3, gt=0, le=0.1, description="Learning rate")
    batch_size: int = Field(default=32, ge=1, le=256, description="Batch size")
    max_epochs: int = Field(default=100, ge=1, description="Maximum epochs")
    early_stopping_patience: int = Field(default=10, ge=1, description="Early stopping patience")
    validation_frequency: int = Field(default=5, ge=1, description="Validation frequency")
    weight_decay: float = Field(default=1e-4, ge=0, description="Weight decay")
    max_parent_size: int = Field(default=5, ge=1, le=10, description="Maximum parent set size")
    dropout: float = Field(default=0.1, ge=0, le=0.5, description="Dropout rate")
    
    model_config = ConfigDict(extra="forbid")


class EvaluationSchema(BaseModel):
    """Pydantic schema for evaluation configuration."""
    
    eval_frequency: int = Field(default=50, ge=1, description="Evaluation frequency")
    n_eval_trajectories: int = Field(default=20, ge=1, description="Number of evaluation trajectories")
    eval_scm_sizes: list = Field(default=[5, 8, 10], description="SCM sizes for evaluation")
    
    model_config = ConfigDict(extra="forbid")


class TrainingSchema(BaseModel):
    """Top-level training configuration schema."""
    
    total_steps: int = Field(default=1000, ge=1, description="Total training steps")
    warmup_steps: int = Field(default=100, ge=0, description="Warmup steps")
    log_frequency: int = Field(default=10, ge=1, description="Logging frequency")
    save_frequency: int = Field(default=100, ge=1, description="Model saving frequency")
    random_seed: int = Field(default=42, ge=0, description="Random seed")
    
    # Nested configurations
    grpo: GRPOTrainingSchema = Field(default_factory=GRPOTrainingSchema)
    rewards: RewardTrainingSchema = Field(default_factory=RewardTrainingSchema)
    exploration: ExplorationTrainingSchema = Field(default_factory=ExplorationTrainingSchema)
    surrogate: SurrogateTrainingSchema = Field(default_factory=SurrogateTrainingSchema)
    evaluation: EvaluationSchema = Field(default_factory=EvaluationSchema)
    
    @field_validator('warmup_steps')
    def warmup_less_than_total(cls, v, info):
        if info.data and 'total_steps' in info.data and v >= info.data['total_steps']:
            raise ValueError('Warmup steps must be less than total steps')
        return v
    
    model_config = ConfigDict(extra="ignore")


# ============================================================================
# Immutable Configuration Dataclasses
# ============================================================================

@dataclass(frozen=True)
class GRPOTrainingConfig:
    """Immutable GRPO training configuration."""
    
    group_size: int = 64
    num_iterations: int = 4  # Sample reuse (open-r1)
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.01
    kl_penalty_coeff: float = 0.0  # Open-r1 recommendation
    max_grad_norm: float = 1.0
    scale_rewards: bool = True  # Configurable scaling
    
    def to_grpo_config(self) -> GRPOConfig:
        """Convert to acquisition module GRPOConfig."""
        return GRPOConfig(
            group_size=self.group_size,
            clip_ratio=self.clip_ratio,
            entropy_coeff=self.entropy_coeff,
            kl_penalty_coeff=self.kl_penalty_coeff,
            max_grad_norm=self.max_grad_norm,
            learning_rate=self.learning_rate
        )


@dataclass(frozen=True)
class RewardTrainingConfig:
    """Immutable reward training configuration."""
    
    optimization_weight: float = 1.0
    structure_weight: float = 0.5
    parent_weight: float = 0.3
    exploration_weight: float = 0.1
    
    def get_reward_weights(self) -> pyr.PMap:
        """Get reward weights as immutable map."""
        return pyr.m(
            optimization=self.optimization_weight,
            structure=self.structure_weight,
            parent=self.parent_weight,
            exploration=self.exploration_weight
        )


@dataclass(frozen=True)
class ExplorationTrainingConfig:
    """Immutable exploration training configuration."""
    
    strategy_type: str = "adaptive"
    uncertainty_weight: float = 1.0
    initial_temperature: float = 2.0
    final_temperature: float = 0.1
    
    def to_exploration_config(self) -> pyr.PMap:
        """Convert to exploration configuration map."""
        return pyr.m(
            strategy_type=self.strategy_type,
            uncertainty_weight=self.uncertainty_weight,
            initial_temperature=self.initial_temperature,
            final_temperature=self.final_temperature
        )


@dataclass(frozen=True)
class SurrogateTrainingConfig:
    """Immutable surrogate model training configuration."""
    
    model_hidden_dim: int = 128
    model_n_layers: int = 6
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    validation_frequency: int = 5
    weight_decay: float = 1e-4
    max_parent_size: int = 5
    dropout: float = 0.1
    
    # BC-specific configuration for dynamic dimensions
    use_continuous_model: bool = True
    use_scm_aware_batching: bool = True
    use_jax_unified: bool = False
    
    def get_model_kwargs(self) -> Dict[str, Any]:
        """Get model configuration parameters."""
        return {
            'layers': self.model_n_layers,
            'dim': self.model_hidden_dim,
            'dropout': self.dropout,
            'max_parent_size': self.max_parent_size
        }


@dataclass(frozen=True)
class EvaluationConfig:
    """Immutable evaluation configuration."""
    
    eval_frequency: int = 50
    n_eval_trajectories: int = 20
    eval_scm_sizes: tuple = (5, 8, 10)


@dataclass(frozen=True)
class TrainingConfig:
    """Top-level immutable training configuration."""
    
    total_steps: int = 1000
    warmup_steps: int = 50  # Default to less than any test total_steps
    log_frequency: int = 10
    save_frequency: int = 100
    random_seed: int = 42
    
    # Nested configurations
    grpo: GRPOTrainingConfig = field(default_factory=GRPOTrainingConfig)
    rewards: RewardTrainingConfig = field(default_factory=RewardTrainingConfig)
    exploration: ExplorationTrainingConfig = field(default_factory=ExplorationTrainingConfig)
    surrogate: SurrogateTrainingConfig = field(default_factory=SurrogateTrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Metadata preservation
    metadata: pyr.PMap = field(default_factory=lambda: pyr.m())


# ============================================================================
# Configuration Factory Functions
# ============================================================================

def create_training_config(
    config_dict: Optional[Dict[str, Any]] = None,
    **kwargs
) -> TrainingConfig:
    """
    Create validated training configuration from dictionary and kwargs.
    
    Args:
        config_dict: Configuration dictionary to validate
        **kwargs: Additional configuration overrides
        
    Returns:
        Validated TrainingConfig object
        
    Raises:
        ValueError: If configuration validation fails
    """
    # Merge config_dict and kwargs (kwargs override)
    if config_dict is None:
        config_dict = {}
    
    merged_config = {**config_dict, **kwargs}
    
    try:
        # Validate with Pydantic schema
        validated = TrainingSchema(**merged_config)
        
        # Convert to immutable dataclass
        config = TrainingConfig(
            total_steps=validated.total_steps,
            warmup_steps=validated.warmup_steps,
            log_frequency=validated.log_frequency,
            save_frequency=validated.save_frequency,
            random_seed=validated.random_seed,
            
            grpo=GRPOTrainingConfig(
                group_size=validated.grpo.group_size,
                num_iterations=validated.grpo.num_iterations,
                learning_rate=validated.grpo.learning_rate,
                clip_ratio=validated.grpo.clip_ratio,
                entropy_coeff=validated.grpo.entropy_coeff,
                kl_penalty_coeff=validated.grpo.kl_penalty_coeff,
                max_grad_norm=validated.grpo.max_grad_norm,
                scale_rewards=validated.grpo.scale_rewards
            ),
            
            rewards=RewardTrainingConfig(
                optimization_weight=validated.rewards.optimization_weight,
                structure_weight=validated.rewards.structure_weight,
                parent_weight=validated.rewards.parent_weight,
                exploration_weight=validated.rewards.exploration_weight
            ),
            
            exploration=ExplorationTrainingConfig(
                strategy_type=validated.exploration.strategy_type,
                uncertainty_weight=validated.exploration.uncertainty_weight,
                initial_temperature=validated.exploration.initial_temperature,
                final_temperature=validated.exploration.final_temperature
            ),
            
            surrogate=SurrogateTrainingConfig(
                model_hidden_dim=validated.surrogate.model_hidden_dim,
                model_n_layers=validated.surrogate.model_n_layers,
                learning_rate=validated.surrogate.learning_rate,
                batch_size=validated.surrogate.batch_size,
                max_epochs=validated.surrogate.max_epochs,
                early_stopping_patience=validated.surrogate.early_stopping_patience,
                validation_frequency=validated.surrogate.validation_frequency,
                weight_decay=validated.surrogate.weight_decay,
                max_parent_size=validated.surrogate.max_parent_size,
                dropout=validated.surrogate.dropout
            ),
            
            evaluation=EvaluationConfig(
                eval_frequency=validated.evaluation.eval_frequency,
                n_eval_trajectories=validated.evaluation.n_eval_trajectories,
                eval_scm_sizes=tuple(validated.evaluation.eval_scm_sizes)
            ),
            
            # Preserve original config as metadata
            metadata=pyr.m(**merged_config)
        )
        
        return config
        
    except ValidationError as e:
        raise ValueError(f"Configuration validation failed: {e}")


def create_default_training_config() -> TrainingConfig:
    """Create default training configuration with open-r1 recommendations."""
    return create_training_config({
        "grpo": {
            "group_size": 64,           # DeepSeek recommendation
            "num_iterations": 4,        # Sample reuse
            "kl_penalty_coeff": 0.0,    # Open-r1 default
            "scale_rewards": True       # Configurable scaling
        },
        "rewards": {
            "optimization_weight": 1.0,
            "structure_weight": 0.5
        },
        "exploration": {
            "strategy_type": "adaptive"
        }
    })


def validate_training_config(config: TrainingConfig) -> bool:
    """
    Validate training configuration for consistency.
    
    Args:
        config: TrainingConfig to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    # Check warmup steps
    if config.warmup_steps >= config.total_steps:
        raise ValueError(
            f"Warmup steps ({config.warmup_steps}) must be less than total steps ({config.total_steps}). "
            f"Suggested: set warmup_steps to {config.total_steps // 10} (10% of total steps)"
        )
    
    # Check evaluation frequency
    if config.evaluation.eval_frequency > config.total_steps:
        raise ValueError(
            f"Evaluation frequency ({config.evaluation.eval_frequency}) cannot exceed total steps ({config.total_steps}). "
            f"Suggested: set eval_frequency to {min(50, config.total_steps // 10)} for regular evaluation"
        )
    
    # Check that at least one reward weight is non-zero
    weights = config.rewards.get_reward_weights()
    if all(w == 0.0 for w in weights.values()):
        raise ValueError(
            f"At least one reward weight must be non-zero. Current weights: {dict(weights)}. "
            f"Suggested: set optimization_weight=1.0 and structure_weight=0.5 for balanced training"
        )
    
    # Check temperature ordering
    if config.exploration.final_temperature >= config.exploration.initial_temperature:
        raise ValueError(
            f"Final temperature ({config.exploration.final_temperature}) must be less than "
            f"initial temperature ({config.exploration.initial_temperature}) for proper annealing. "
            f"Suggested: initial=2.0, final=0.1"
        )
    
    return True


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Pydantic schemas
    'TrainingSchema',
    'GRPOTrainingSchema', 
    'RewardTrainingSchema',
    'ExplorationTrainingSchema',
    'SurrogateTrainingSchema',
    'EvaluationSchema',
    
    # Immutable configs
    'TrainingConfig',
    'GRPOTrainingConfig',
    'RewardTrainingConfig', 
    'ExplorationTrainingConfig',
    'SurrogateTrainingConfig',
    'EvaluationConfig',
    
    # Factory functions
    'create_training_config',
    'create_default_training_config',
    'validate_training_config'
]