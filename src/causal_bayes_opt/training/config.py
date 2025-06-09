# Training Configuration System with Pydantic Validation
# Implements recommendations from open-r1 for robust GRPO training

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, field_validator
import pyrsistent as pyr

from ..acquisition.grpo import GRPOConfig
from ..acquisition.exploration import ExplorationConfig


# ============================================================================
# Pydantic Validation Schemas (for input validation)
# ============================================================================

class GRPOTrainingSchema(BaseModel):
    """Pydantic schema for GRPO training configuration validation."""
    
    # Core GRPO parameters (following open-r1 recommendations)
    group_size: int = Field(ge=8, le=256, default=64, description="GRPO group size")
    num_iterations: int = Field(ge=1, le=10, default=4, description="Sample reuse iterations")
    
    # Learning parameters
    learning_rate: float = Field(gt=0.0, le=0.01, default=3e-4, description="Adam learning rate")
    max_grad_norm: float = Field(gt=0.0, le=10.0, default=1.0, description="Gradient clipping norm")
    
    # GRPO-specific parameters
    clip_ratio: float = Field(gt=0.0, le=1.0, default=0.2, description="PPO-style clipping ratio")
    entropy_coeff: float = Field(ge=0.0, le=1.0, default=0.01, description="Entropy regularization")
    kl_penalty_coeff: float = Field(ge=0.0, le=1.0, default=0.0, description="KL penalty (default 0.0 per open-r1)")
    
    # Advantage computation (based on recommendations)
    scale_rewards: bool = Field(default=True, description="Scale advantages by std deviation")
    
    @field_validator('group_size')
    @classmethod
    def validate_group_size(cls, v):
        """Ensure group size is reasonable for GRPO."""
        if v < 8:
            raise ValueError("Group size should be at least 8 for stable GRPO training")
        return v
    
    @field_validator('num_iterations')
    @classmethod
    def validate_num_iterations(cls, v):
        """Validate sample reuse iterations."""
        if v > 10:
            raise ValueError("Too many iterations may lead to overfitting")
        return v


class RewardTrainingSchema(BaseModel):
    """Pydantic schema for reward configuration validation."""
    
    # Dual-objective weights
    optimization_weight: float = Field(ge=0.0, le=10.0, default=1.0, description="Target optimization weight")
    structure_weight: float = Field(ge=0.0, le=10.0, default=0.5, description="Structure discovery weight") 
    parent_weight: float = Field(ge=0.0, le=10.0, default=0.3, description="Parent intervention bonus weight")
    exploration_weight: float = Field(ge=0.0, le=10.0, default=0.1, description="Exploration bonus weight")
    
    # Reward scaling
    reward_scaling: str = Field(default="tanh", pattern="^(tanh|clip|none)$", description="Reward scaling method")
    reward_clip_value: float = Field(gt=0.0, default=10.0, description="Reward clipping value")
    
    @field_validator('optimization_weight', 'structure_weight', 'parent_weight', 'exploration_weight')
    @classmethod
    def validate_weights_positive(cls, v):
        """Ensure all weights are non-negative."""
        if v < 0:
            raise ValueError("Weight must be non-negative")
        return v
    
    def validate_total_weights(self) -> bool:
        """Check that total weights make sense."""
        total = self.optimization_weight + self.structure_weight + self.parent_weight + self.exploration_weight
        if total <= 0:
            raise ValueError("At least one reward weight must be positive")
        return True


class ExplorationTrainingSchema(BaseModel):
    """Pydantic schema for exploration strategy configuration."""
    
    strategy_type: str = Field(default="adaptive", pattern="^(uncertainty_guided|adaptive)$")
    
    # Uncertainty-guided parameters
    uncertainty_weight: float = Field(ge=0.0, le=10.0, default=1.0)
    count_weight: float = Field(ge=0.0, le=1.0, default=0.1)
    variable_uncertainty_weight: float = Field(ge=0.0, le=10.0, default=0.5)
    temperature: float = Field(gt=0.0, le=10.0, default=1.0)
    
    # Adaptive exploration parameters
    initial_temperature: float = Field(gt=0.0, le=10.0, default=2.0)
    final_temperature: float = Field(gt=0.0, le=10.0, default=0.1)
    adaptation_steps: int = Field(ge=100, le=10000, default=1000)
    stagnation_threshold: int = Field(ge=10, le=1000, default=100)
    
    @field_validator('final_temperature')
    @classmethod 
    def validate_temperature_order(cls, v, info):
        """Ensure final temperature is less than initial."""
        if info.data and 'initial_temperature' in info.data and v >= info.data['initial_temperature']:
            raise ValueError("Final temperature must be less than initial temperature")
        return v


class EvaluationSchema(BaseModel):
    """Pydantic schema for evaluation configuration."""
    
    # Evaluation frequency
    eval_frequency: int = Field(ge=1, le=1000, default=100, description="Steps between evaluations")
    checkpoint_frequency: int = Field(ge=1, le=10000, default=500, description="Steps between checkpoints")
    
    # Evaluation metrics
    track_optimization: bool = Field(default=True, description="Track optimization metrics")
    track_structure_learning: bool = Field(default=True, description="Track structure discovery metrics")
    track_intervention_diversity: bool = Field(default=True, description="Track intervention diversity")
    
    # Early stopping
    early_stopping_enabled: bool = Field(default=False, description="Enable early stopping")
    early_stopping_patience: int = Field(ge=1, le=1000, default=200, description="Early stopping patience")
    early_stopping_threshold: float = Field(gt=0.0, default=1e-4, description="Early stopping threshold")


class TrainingSchema(BaseModel):
    """Top-level training configuration schema."""
    
    # Training schedule
    total_steps: int = Field(ge=1, le=100000, default=5000, description="Total training steps")
    warmup_steps: int = Field(ge=0, le=10000, default=100, description="Warmup steps")
    
    # Update frequencies
    surrogate_update_frequency: int = Field(ge=1, le=1000, default=50, description="Surrogate model update frequency")
    
    # Logging
    log_frequency: int = Field(ge=1, le=1000, default=10, description="Logging frequency")
    save_trajectory_data: bool = Field(default=True, description="Save trajectory data for analysis")
    
    # Nested configurations
    grpo: GRPOTrainingSchema = Field(default_factory=GRPOTrainingSchema)
    rewards: RewardTrainingSchema = Field(default_factory=RewardTrainingSchema)
    exploration: ExplorationTrainingSchema = Field(default_factory=ExplorationTrainingSchema)
    evaluation: EvaluationSchema = Field(default_factory=EvaluationSchema)
    
    @field_validator('warmup_steps')
    @classmethod
    def validate_warmup_steps(cls, v, info):
        """Ensure warmup steps don't exceed total steps."""
        if info.data and 'total_steps' in info.data and v >= info.data['total_steps']:
            raise ValueError("Warmup steps must be less than total steps")
        return v


# ============================================================================
# Immutable Configuration Dataclasses (for internal use)
# ============================================================================

@dataclass(frozen=True)
class GRPOTrainingConfig:
    """Immutable GRPO training configuration following functional principles."""
    
    # Core parameters
    group_size: int = 64
    num_iterations: int = 4  # Sample reuse (open-r1 recommendation)
    
    # Learning parameters
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    
    # GRPO parameters
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.01
    kl_penalty_coeff: float = 0.0  # Default 0.0 per open-r1 recommendation
    
    # Advantage computation
    scale_rewards: bool = True  # Configurable reward scaling
    
    def to_grpo_config(self) -> GRPOConfig:
        """Convert to acquisition module's GRPOConfig."""
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
    
    # Dual-objective weights
    optimization_weight: float = 1.0
    structure_weight: float = 0.5
    parent_weight: float = 0.3
    exploration_weight: float = 0.1
    
    # Reward processing
    reward_scaling: str = "tanh"
    reward_clip_value: float = 10.0
    
    def get_reward_weights(self) -> pyr.PMap:
        """Get reward weights as immutable map."""
        return pyr.pmap({
            'optimization': self.optimization_weight,
            'structure': self.structure_weight,
            'parent': self.parent_weight,
            'exploration': self.exploration_weight
        })


@dataclass(frozen=True)
class ExplorationTrainingConfig:
    """Immutable exploration training configuration."""
    
    strategy_type: str = "adaptive"
    
    # Uncertainty-guided parameters
    uncertainty_weight: float = 1.0
    count_weight: float = 0.1
    variable_uncertainty_weight: float = 0.5
    temperature: float = 1.0
    
    # Adaptive parameters
    initial_temperature: float = 2.0
    final_temperature: float = 0.1
    adaptation_steps: int = 1000
    stagnation_threshold: int = 100
    
    def to_exploration_config(self) -> ExplorationConfig:
        """Convert to acquisition module's ExplorationConfig."""
        return ExplorationConfig(
            uncertainty_weight=self.uncertainty_weight,
            count_weight=self.count_weight,
            variable_uncertainty_weight=self.variable_uncertainty_weight,
            temperature=self.temperature,
            initial_temperature=self.initial_temperature,
            final_temperature=self.final_temperature,
            adaptation_steps=self.adaptation_steps,
            stagnation_threshold=self.stagnation_threshold
        )


@dataclass(frozen=True)
class EvaluationConfig:
    """Immutable evaluation configuration."""
    
    # Frequencies
    eval_frequency: int = 100
    checkpoint_frequency: int = 500
    
    # Metrics to track
    track_optimization: bool = True
    track_structure_learning: bool = True
    track_intervention_diversity: bool = True
    
    # Early stopping
    early_stopping_enabled: bool = False
    early_stopping_patience: int = 200
    early_stopping_threshold: float = 1e-4


@dataclass(frozen=True)
class TrainingConfig:
    """Top-level immutable training configuration."""
    
    # Training schedule
    total_steps: int = 5000
    warmup_steps: int = 100
    surrogate_update_frequency: int = 50
    
    # Logging
    log_frequency: int = 10
    save_trajectory_data: bool = True
    
    # Component configurations
    grpo: GRPOTrainingConfig = field(default_factory=GRPOTrainingConfig)
    rewards: RewardTrainingConfig = field(default_factory=RewardTrainingConfig)
    exploration: ExplorationTrainingConfig = field(default_factory=ExplorationTrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Additional metadata
    metadata: pyr.PMap = field(default_factory=pyr.pmap)


# ============================================================================
# Factory Functions and Validation
# ============================================================================

def create_training_config(
    config_dict: Optional[Dict[str, Any]] = None,
    **kwargs
) -> TrainingConfig:
    """
    Create validated training configuration from dictionary or kwargs.
    
    Args:
        config_dict: Optional configuration dictionary
        **kwargs: Additional configuration parameters
        
    Returns:
        TrainingConfig: Validated immutable configuration
        
    Raises:
        ValueError: If configuration validation fails
    """
    # Merge config_dict and kwargs
    if config_dict is None:
        config_dict = {}
    config_dict.update(kwargs)
    
    # Validate with Pydantic schema
    try:
        validated_schema = TrainingSchema(**config_dict)
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")
    
    # Convert to immutable dataclasses
    grpo_config = GRPOTrainingConfig(**validated_schema.grpo.model_dump())
    reward_config = RewardTrainingConfig(**validated_schema.rewards.model_dump())
    exploration_config = ExplorationTrainingConfig(**validated_schema.exploration.model_dump())
    evaluation_config = EvaluationConfig(**validated_schema.evaluation.model_dump())
    
    # Create top-level config
    return TrainingConfig(
        total_steps=validated_schema.total_steps,
        warmup_steps=validated_schema.warmup_steps,
        surrogate_update_frequency=validated_schema.surrogate_update_frequency,
        log_frequency=validated_schema.log_frequency,
        save_trajectory_data=validated_schema.save_trajectory_data,
        grpo=grpo_config,
        rewards=reward_config,
        exploration=exploration_config,
        evaluation=evaluation_config,
        metadata=pyr.pmap(config_dict)  # Store original config for reference
    )


def validate_training_config(config: TrainingConfig) -> bool:
    """
    Validate training configuration consistency.
    
    Args:
        config: Training configuration to validate
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Cross-component validation
    if config.warmup_steps >= config.total_steps:
        raise ValueError("Warmup steps must be less than total steps")
    
    if config.evaluation.eval_frequency > config.total_steps:
        raise ValueError("Evaluation frequency should be less than total steps")
    
    if config.surrogate_update_frequency > config.total_steps:
        raise ValueError("Surrogate update frequency should be less than total steps")
    
    # Validate reward weights sum to something reasonable
    total_weight = (
        config.rewards.optimization_weight +
        config.rewards.structure_weight +
        config.rewards.parent_weight +
        config.rewards.exploration_weight
    )
    if total_weight <= 0:
        raise ValueError("At least one reward weight must be positive")
    
    # Validate exploration temperature ordering
    if (config.exploration.strategy_type == "adaptive" and 
        config.exploration.final_temperature >= config.exploration.initial_temperature):
        raise ValueError("Final temperature must be less than initial temperature")
    
    return True


def create_default_training_config() -> TrainingConfig:
    """Create default training configuration with recommended settings."""
    return create_training_config({
        "total_steps": 5000,
        "grpo": {
            "group_size": 64,
            "num_iterations": 4,  # Sample reuse
            "kl_penalty_coeff": 0.0,  # open-r1 recommendation
            "scale_rewards": True
        },
        "rewards": {
            "optimization_weight": 1.0,
            "structure_weight": 0.5,
            "parent_weight": 0.3,
            "exploration_weight": 0.1
        },
        "exploration": {
            "strategy_type": "adaptive",
            "initial_temperature": 2.0,
            "final_temperature": 0.1
        }
    })


# Export main functions and classes
__all__ = [
    'TrainingConfig',
    'GRPOTrainingConfig',
    'RewardTrainingConfig', 
    'ExplorationTrainingConfig',
    'EvaluationConfig',
    'create_training_config',
    'validate_training_config',
    'create_default_training_config',
]