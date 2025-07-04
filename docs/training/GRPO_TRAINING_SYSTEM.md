# GRPO Training System Documentation

## Overview

This document describes the complete GRPO (Guided Reward Policy Optimization) training system implemented for causal Bayesian optimization. The system provides production-ready tools for training acquisition policies, comparing models, and conducting systematic experiments.

## System Architecture

### Core Components

1. **Full-Scale Training** (`scripts/train_full_scale_grpo.py`)
   - Production-ready GRPO policy training
   - Continuous reward system with SCM-objective optimization
   - Comprehensive WandB integration
   - Automatic checkpoint management

2. **Model Comparison** (`scripts/compare_grpo_models.py`)
   - Statistical comparison of trained models
   - Performance evaluation on multiple test environments
   - Automated ranking and visualization
   - Significance testing

3. **Workflow Runner** (`scripts/run_grpo_workflow.py`)
   - End-to-end training and comparison workflow
   - Multi-run experiment management
   - Automated configuration testing

### Key Features

- **Continuous Reward System**: Replaces binary rewards with continuous SCM-objective optimization
- **Component Validation**: Comprehensive testing framework for reward components
- **Statistical Analysis**: Significance testing, effect sizes, ANOVA comparisons
- **Experiment Tracking**: Full WandB integration with automatic artifact management
- **Checkpoint Management**: Versioned model storage with metadata
- **Configuration Management**: Hydra-based configuration with command-line overrides

## Reward System Design

### Continuous SCM-Objective Rewards

The training system uses a continuous reward formulation that encourages optimal interventions:

```python
def _compute_scm_objective_reward(
    state_before: AcquisitionState,
    target_value: float,
    target_variable: str
) -> Optional[float]:
    """Compute reward based on distance to theoretical SCM optimum."""
```

**Key Properties:**
- Continuous gradients for better learning signals
- Theoretical optimum alignment with true SCM structure
- Adaptive normalization across variable ranges
- Component-wise validation and isolation testing

### Reward Components

1. **Optimization Reward** (weight: 1.0)
   - Measures progress toward target optimization
   - Based on continuous improvement metrics

2. **Structure Discovery Reward** (weight: 0.5)
   - Encourages causal structure learning
   - Parent relationship discovery

3. **Parent Discovery Reward** (weight: 0.3)
   - Specific parent variable identification
   - Marginal probability updates

4. **Exploration Reward** (weight: 0.1)
   - Balanced exploration vs exploitation
   - Intervention diversity encouragement

## Training Configuration

### Network Architecture

```yaml
training:
  hidden_size: 64          # Hidden layer size
  num_layers: 2            # Number of hidden layers
  state_dim: 10            # Input state dimension
  action_dim: 1            # Output action dimension
  max_intervention_value: 3.0  # Intervention value bounds
```

### Learning Parameters

```yaml
training:
  n_episodes: 300          # Training episodes
  episode_length: 20       # Steps per episode
  learning_rate: 0.0003    # Adam learning rate
  gamma: 0.99              # Discount factor
```

### Reward Weights

```yaml
training:
  reward_weights:
    optimization: 1.0      # Target optimization
    structure: 0.5         # Structure discovery
    parent: 0.3            # Parent discovery
    exploration: 0.1       # Exploration bonus
```

## Usage Guide

### Basic Training

```bash
# Train with default configuration
poetry run python scripts/train_full_scale_grpo.py

# Custom hyperparameters
poetry run python scripts/train_full_scale_grpo.py \
    training.learning_rate=0.001 \
    training.hidden_size=128

# Enable WandB logging
poetry run python scripts/train_full_scale_grpo.py \
    logging.wandb.enabled=true
```

### Multi-Run Experiments

```bash
# Hyperparameter sweep
poetry run python scripts/train_full_scale_grpo.py --multirun \
    training.learning_rate=0.0003,0.001,0.003 \
    training.hidden_size=32,64,128
```

### Model Comparison

```bash
# Compare all models in checkpoints/
poetry run python scripts/compare_grpo_models.py

# With WandB logging
poetry run python scripts/compare_grpo_models.py \
    logging.wandb.enabled=true

# Compare specific models
poetry run python scripts/compare_grpo_models.py \
    --model_paths checkpoints/model1 checkpoints/model2
```

### Complete Workflow

```bash
# Run full training and comparison workflow
poetry run python scripts/run_grpo_workflow.py --n-runs 3 --enable-wandb
```

## Validation Framework

### Component Testing

The system includes comprehensive validation for individual reward components:

1. **Zero-Out Component Tests**: Isolate and validate individual reward components
2. **Adversarial Testing**: Test for potential reward gaming exploits
3. **Normalization Testing**: Validate across extreme value ranges
4. **Integration Testing**: End-to-end policy improvement validation

### Test Files

- `tests/test_training/test_continuous_reward_validation.py`
- `tests/test_training/test_adversarial_reward_exploits.py`
- `tests/test_training/test_reward_normalization.py`

### Validation Script

```bash
# Run short training validation
poetry run python scripts/validate_policy_training.py
```

## Model Comparison Methodology

### Statistical Analysis

The comparison system provides comprehensive statistical analysis:

1. **Pairwise t-tests**: Compare performance between model pairs
2. **Effect sizes**: Cohen's d for practical significance
3. **ANOVA**: Overall significance across multiple models
4. **Performance ranking**: Systematic model ordering

### Evaluation Metrics

- **Mean Performance**: Average reward across test episodes
- **Success Rate**: Percentage of episodes with positive rewards
- **Convergence Time**: Episodes until performance stabilization
- **Standard Deviation**: Performance consistency measure

### Test Environments

Models are evaluated on multiple SCM variants:
- Chain structures (X₀ → X₁ → ... → Xₙ)
- Star structures (X₀, X₁, ... → Xₙ)
- Fork structures (X₀ → X₁, X₂, ...)
- Random sparse structures

## Checkpoint Management

### Model Storage

Trained models are automatically saved with comprehensive metadata:

```
checkpoints/grpo_training_<timestamp>/
├── model.pkl          # Model parameters and state
├── metrics.json       # Training performance metrics
└── config.yaml        # Training configuration
```

### Checkpoint Contents

- **Policy Parameters**: Neural network weights for policy
- **Value Parameters**: Neural network weights for value function
- **Optimizer State**: Adam optimizer internal state
- **Training Metrics**: Performance analysis and trends
- **Configuration**: Complete Hydra configuration used

## WandB Integration

### Experiment Tracking

- **Real-time Metrics**: Episode rewards, values, losses
- **Model Artifacts**: Automatic checkpoint uploading
- **Configuration Logging**: Complete hyperparameter tracking
- **Comparison Tables**: Side-by-side model performance

### Custom Metrics

```python
wandb.define_metric("episode")
wandb.define_metric("mean_reward", step_metric="episode")
wandb.define_metric("final_best_value", step_metric="episode")
```

## Performance Characteristics

### Training Performance

- **Typical Training Time**: 5-15 minutes for 300 episodes
- **Memory Usage**: ~2GB RAM for standard configuration
- **GPU Acceleration**: Automatic JAX GPU utilization
- **Convergence**: Usually within 100-200 episodes

### Scalability

- **Variable Count**: Tested up to 10 variables
- **Episode Length**: Scales linearly with episode steps
- **Batch Training**: Supports multi-run parallel execution

## Migration from Legacy System

### Key Changes

1. **Reward System**: Binary → Continuous SCM-objective rewards
2. **Validation**: Added comprehensive component testing
3. **Comparison**: Statistical significance testing
4. **Tracking**: Full WandB integration
5. **Configuration**: Hydra-based management

### Removed Components

- `verifiable_rewards.py`: Old binary reward system (deleted)
- Silent failure modes in training managers
- Mock result returns without actual computation

## Best Practices

### Training Configuration

1. **Start with defaults**: Use provided configuration as baseline
2. **Incremental changes**: Modify one hyperparameter at a time
3. **Statistical validation**: Use comparison framework for evaluation
4. **Checkpoint everything**: Save all training runs for comparison

### Experiment Design

1. **Multiple seeds**: Run with different random seeds
2. **Proper baselines**: Include standard configurations
3. **Statistical testing**: Use significance tests for claims
4. **Documentation**: Log all configuration changes

### Model Selection

1. **Multiple metrics**: Don't rely only on final reward
2. **Convergence analysis**: Check training stability
3. **Test generalization**: Evaluate on multiple SCMs
4. **Significance testing**: Validate performance differences

## Troubleshooting

### Common Issues

1. **JAX Memory Errors**: Reduce batch size or network size
2. **NaN Values**: Check learning rate and gradient clipping
3. **No Improvement**: Verify reward configuration and SCM setup
4. **WandB Failures**: Check internet connection and API key

### Debug Mode

```bash
# Run with debug logging
poetry run python scripts/train_full_scale_grpo.py \
    logging.level=DEBUG
```

### Validation

```bash
# Quick validation of training setup
poetry run python scripts/validate_policy_training.py
```

## Future Extensions

### Planned Features

1. **Curriculum Learning**: Progressive difficulty increase
2. **Meta-Learning**: Adaptation across SCM families
3. **Distributed Training**: Multi-GPU and multi-node support
4. **Online Learning**: Real-time policy updates

### Research Directions

1. **Reward Shaping**: Advanced reward design techniques
2. **Architecture Search**: Automated network design
3. **Transfer Learning**: Cross-domain policy transfer
4. **Theoretical Analysis**: Convergence guarantees

## References

- GRPO Training Manager: `src/causal_bayes_opt/training/grpo_training_manager.py`
- Reward System: `src/causal_bayes_opt/acquisition/rewards.py`
- Validation Tests: `tests/test_training/`
- Configuration: `config/full_scale_grpo_config.yaml`