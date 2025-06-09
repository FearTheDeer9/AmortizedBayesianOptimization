# Training Configuration API Reference

## Overview
The training configuration module provides a robust, validated configuration system for the ACBO training pipeline. It implements recommendations from open-r1 for stable GRPO training and uses Pydantic for validation with immutable dataclasses for internal representation.

## Design Philosophy
- **Validation First**: Pydantic schemas validate all inputs
- **Immutable Configs**: Internal configs are frozen dataclasses
- **Hierarchical Structure**: Nested configurations for different components
- **Type Safety**: Full type annotations and runtime validation
- **Default Best Practices**: Defaults follow open-r1 recommendations

## Configuration Structure

### TrainingConfig (Top Level)
```python
@dataclass(frozen=True)
class TrainingConfig:
    # Training schedule
    total_steps: int = 5000
    warmup_steps: int = 100
    surrogate_update_frequency: int = 50
    
    # Logging
    log_frequency: int = 10
    save_trajectory_data: bool = True
    
    # Component configurations
    grpo: GRPOTrainingConfig
    rewards: RewardTrainingConfig
    exploration: ExplorationTrainingConfig
    evaluation: EvaluationConfig
    
    # Additional metadata
    metadata: pyr.PMap
```

### GRPOTrainingConfig
```python
@dataclass(frozen=True)
class GRPOTrainingConfig:
    # Core parameters (following open-r1)
    group_size: int = 64              # DeepSeek recommendation
    num_iterations: int = 4           # Sample reuse iterations
    
    # Learning parameters
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    
    # GRPO parameters
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.01
    kl_penalty_coeff: float = 0.0     # Default 0.0 per open-r1
    
    # Advantage computation
    scale_rewards: bool = True        # Configurable scaling
```

### RewardTrainingConfig
```python
@dataclass(frozen=True)
class RewardTrainingConfig:
    # Dual-objective weights
    optimization_weight: float = 1.0
    structure_weight: float = 0.5
    parent_weight: float = 0.3
    exploration_weight: float = 0.1
    
    # Reward processing
    reward_scaling: str = "tanh"      # "tanh", "clip", or "none"
    reward_clip_value: float = 10.0
```

### ExplorationTrainingConfig
```python
@dataclass(frozen=True)
class ExplorationTrainingConfig:
    strategy_type: str = "adaptive"   # "uncertainty_guided" or "adaptive"
    
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
```

### EvaluationConfig
```python
@dataclass(frozen=True)
class EvaluationConfig:
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
```

## Factory Functions

### create_training_config(config_dict=None, **kwargs)
Create validated training configuration from dictionary or kwargs.

**Parameters:**
- config_dict: Optional[Dict[str, Any]] - Configuration dictionary
- **kwargs: Additional configuration parameters

**Returns:**
TrainingConfig - Validated immutable configuration

**Raises:**
- ValueError: If configuration validation fails

**Example:**
```python
# From dictionary
config_dict = {
    "total_steps": 10000,
    "grpo": {
        "group_size": 64,
        "num_iterations": 4,
        "kl_penalty_coeff": 0.0
    },
    "rewards": {
        "optimization_weight": 2.0,
        "structure_weight": 1.0
    }
}
config = create_training_config(config_dict)

# From kwargs
config = create_training_config(
    total_steps=10000,
    warmup_steps=200,
    grpo={"group_size": 128}
)
```

### create_default_training_config()
Create default training configuration with recommended settings.

**Returns:**
TrainingConfig - Default configuration following best practices

**Example:**
```python
config = create_default_training_config()
print(f"Total steps: {config.total_steps}")  # 5000
print(f"GRPO group size: {config.grpo.group_size}")  # 64
```

### validate_training_config(config)
Validate training configuration consistency.

**Parameters:**
- config: TrainingConfig - Configuration to validate

**Returns:**
bool - True if valid

**Raises:**
- ValueError: If configuration is invalid

**Example:**
```python
config = create_training_config({"warmup_steps": 6000, "total_steps": 5000})
# Raises: ValueError: Warmup steps must be less than total steps
```

## Validation Rules

### Pydantic Schema Validation

#### Value Ranges
- `group_size`: 8 ≤ n ≤ 256
- `num_iterations`: 1 ≤ n ≤ 10
- `learning_rate`: 0 < lr ≤ 0.01
- `max_grad_norm`: 0 < norm ≤ 10.0
- `clip_ratio`: 0 < ratio ≤ 1.0
- All weights: n ≥ 0

#### String Enums
- `reward_scaling`: "tanh", "clip", or "none"
- `strategy_type`: "uncertainty_guided" or "adaptive"

#### Cross-Field Validation
- `warmup_steps < total_steps`
- `final_temperature < initial_temperature`
- At least one reward weight > 0

### Custom Validation
```python
# Group size validation
if config.grpo.group_size < 8:
    raise ValueError("Group size should be at least 8 for stable GRPO")

# Temperature ordering
if config.exploration.final_temperature >= config.exploration.initial_temperature:
    raise ValueError("Final temperature must be less than initial")
```

## Usage Patterns

### Basic Configuration
```python
# Simple configuration with defaults
config = create_training_config(total_steps=10000)
```

### GRPO-Focused Configuration
```python
config = create_training_config({
    "grpo": {
        "group_size": 128,          # Larger groups
        "num_iterations": 6,        # More sample reuse
        "learning_rate": 1e-4,      # Lower learning rate
        "scale_rewards": True,      # Enable scaling
        "kl_penalty_coeff": 0.0     # No KL penalty (open-r1)
    }
})
```

### Multi-Objective Tuning
```python
config = create_training_config({
    "rewards": {
        "optimization_weight": 2.0,  # Emphasize optimization
        "structure_weight": 1.0,     # Standard structure learning
        "parent_weight": 0.5,        # Moderate parent bonus
        "exploration_weight": 0.2,   # Light exploration
        "reward_scaling": "tanh",    # Smooth scaling
        "reward_clip_value": 20.0    # Higher clip threshold
    }
})
```

### Adaptive Exploration
```python
config = create_training_config({
    "exploration": {
        "strategy_type": "adaptive",
        "initial_temperature": 5.0,  # Start very exploratory
        "final_temperature": 0.01,   # End very focused
        "adaptation_steps": 2000,    # Slow adaptation
        "stagnation_threshold": 50   # Quick response to stagnation
    }
})
```

### Evaluation and Checkpointing
```python
config = create_training_config({
    "evaluation": {
        "eval_frequency": 50,        # Frequent evaluation
        "checkpoint_frequency": 250,  # Regular checkpoints
        "early_stopping_enabled": True,
        "early_stopping_patience": 500,
        "early_stopping_threshold": 1e-5
    }
})
```

## Integration Examples

### With GRPO Training
```python
# Create configuration
config = create_training_config({
    "total_steps": 10000,
    "grpo": {"group_size": 64, "num_iterations": 4}
})

# Convert to GRPO module config
grpo_config = config.grpo.to_grpo_config()

# Create trainer
update_step, optimizer_init = create_grpo_trainer(policy_network, grpo_config)
```

### With Reward Computation
```python
# Get reward weights
reward_weights = config.rewards.get_reward_weights()

# Compute rewards
reward = compute_reward(
    state, intervention, outcome, next_state,
    weights=reward_weights,
    scaling=config.rewards.reward_scaling,
    clip_value=config.rewards.reward_clip_value
)
```

### With Training Loop
```python
config = create_default_training_config()

for step in range(config.total_steps):
    # Warmup phase
    if step < config.warmup_steps:
        learning_rate = warmup_schedule(step, config.warmup_steps)
    
    # Evaluation
    if step % config.evaluation.eval_frequency == 0:
        evaluate_model(...)
    
    # Checkpointing
    if step % config.evaluation.checkpoint_frequency == 0:
        save_checkpoint(...)
    
    # Logging
    if step % config.log_frequency == 0:
        log_metrics(...)
```

## Best Practices

### 1. Use Validation
```python
# Always validate custom configs
try:
    config = create_training_config(my_config_dict)
    validate_training_config(config)
except ValueError as e:
    print(f"Invalid configuration: {e}")
```

### 2. Start from Defaults
```python
# Get defaults and modify
default_config = create_default_training_config()

# Create modified version
custom_config = create_training_config({
    "total_steps": 20000,  # Only change what's needed
    "grpo": {"group_size": 128}
})
```

### 3. Store Configuration
```python
# Configuration is immutable and serializable
config_dict = {
    "total_steps": config.total_steps,
    "grpo": {
        "group_size": config.grpo.group_size,
        # ... other fields
    }
}
save_json(config_dict, "training_config.json")
```

### 4. Document Changes
```python
config = create_training_config({
    "grpo": {
        "group_size": 128,  # Increased for stability with large SCMs
        "learning_rate": 1e-4  # Reduced for complex optimization landscape
    }
})
```

## Advanced Features

### Metadata Storage
```python
# Original config is stored in metadata
config = create_training_config({"custom_field": "value"})
print(config.metadata["custom_field"])  # "value"
```

### Component Conversion
```python
# Convert to component-specific configs
grpo_config = config.grpo.to_grpo_config()
exploration_config = config.exploration.to_exploration_config()
```

### Validation Composition
```python
# Custom validation on top of Pydantic
def validate_custom_rules(config: TrainingConfig):
    if config.grpo.group_size > 100 and config.total_steps < 10000:
        raise ValueError("Large group sizes need more training steps")
    
    if config.rewards.exploration_weight > 1.0 and config.exploration.strategy_type != "adaptive":
        raise ValueError("High exploration weight requires adaptive strategy")
```

## Common Configurations

### Fast Experimentation
```python
fast_config = create_training_config({
    "total_steps": 1000,
    "eval_frequency": 50,
    "grpo": {"group_size": 32}
})
```

### Production Training
```python
production_config = create_training_config({
    "total_steps": 50000,
    "grpo": {
        "group_size": 64,
        "num_iterations": 4,
        "learning_rate": 3e-4
    },
    "evaluation": {
        "checkpoint_frequency": 1000,
        "early_stopping_enabled": True
    }
})
```

### Exploration-Heavy
```python
exploration_config = create_training_config({
    "rewards": {
        "exploration_weight": 2.0,
        "structure_weight": 2.0,
        "optimization_weight": 0.5
    },
    "exploration": {
        "initial_temperature": 10.0,
        "strategy_type": "adaptive"
    }
})
```

## Troubleshooting

### Common Errors

**Invalid Group Size:**
```python
config = create_training_config({"grpo": {"group_size": 4}})
# ValueError: Group size should be at least 8 for stable GRPO training
```

**Temperature Ordering:**
```python
config = create_training_config({
    "exploration": {
        "initial_temperature": 1.0,
        "final_temperature": 2.0
    }
})
# ValueError: Final temperature must be less than initial temperature
```

**Inconsistent Steps:**
```python
config = create_training_config({
    "total_steps": 100,
    "warmup_steps": 200
})
# ValueError: Warmup steps must be less than total steps
```