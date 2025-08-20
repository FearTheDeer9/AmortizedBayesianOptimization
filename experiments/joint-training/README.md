# Joint ACBO Training

This directory contains the joint training implementation that alternates between policy and surrogate model updates in a GAN-like fashion.

## Features

- **Flexible Initialization**: Train from scratch or load pretrained models
- **Configurable Reward Weights**: Full control over GRPO reward components
- **Alternating Training**: Automatic switching between policy and surrogate phases
- **F1-based Rotation**: Rotate SCMs when surrogate reaches high accuracy
- **Comprehensive Logging**: Track metrics for both models

## Quick Start

### Train from Scratch
```bash
python train_joint.py --episodes 200
```

### Quick Test (2 episodes)
```bash
python train_joint.py --quick-test
```

### Load Pretrained Policy Only
```bash
python train_joint.py \
    --policy-checkpoint experiments/policy-only-training/checkpoints/diverse_fixed_3to10/joint_ep2/policy.pkl \
    --episodes 200
```

### Load Both Pretrained Models
```bash
python train_joint.py \
    --policy-checkpoint experiments/policy-only-training/checkpoints/diverse_fixed_3to10/joint_ep2/policy.pkl \
    --surrogate-checkpoint experiments/surrogate-only-training/checkpoints/avici_runs/avici_style_20250818_161941/best_model.pkl \
    --episodes 200
```

### Custom Reward Weights
```bash
python train_joint.py \
    --target-weight 0.8 \
    --parent-weight 0.15 \
    --info-weight 0.05 \
    --exploration-bonus 0.0
```

## Command Line Arguments

### Model Loading
- `--policy-checkpoint PATH`: Path to pretrained policy checkpoint
- `--surrogate-checkpoint PATH`: Path to pretrained surrogate checkpoint

### Training Settings
- `--episodes N`: Total number of training episodes (default: 200)
- `--policy-episodes N`: Policy episodes per phase (default: 5)
- `--surrogate-steps N`: Surrogate training steps per phase (default: 1000)
- `--f1-threshold F`: F1 threshold for SCM rotation (default: 0.9)

### GRPO Reward Weights
- `--target-weight W`: Weight for target value improvement (default: 0.7)
- `--parent-weight W`: Weight for parent selection accuracy (default: 0.2)
- `--info-weight W`: Weight for information gain (default: 0.1)
- `--exploration-bonus W`: Exploration bonus weight (default: 0.0)

### Model Architecture
- `--hidden-dim N`: Hidden dimension for models (default: 128)
- `--num-layers N`: Number of layers for surrogate (default: 8)

### SCM Generation
- `--min-vars N`: Minimum number of variables (default: 3)
- `--max-vars N`: Maximum number of variables (default: 30)

### Learning Rates
- `--learning-rate LR`: Policy learning rate (default: 5e-4)
- `--surrogate-lr LR`: Surrogate learning rate (default: 1e-4)

### Other
- `--seed N`: Random seed (default: 42)
- `--verbose`: Enable verbose logging
- `--no-surrogate`: Disable surrogate (pure GRPO training)
- `--quick-test`: Quick test with 2 episodes

## Training Process

1. **Policy Phase**: Train policy for N episodes using GRPO
   - Generate diverse SCMs
   - Use surrogate predictions for rewards (if enabled)
   - Apply configurable reward weights
   - Update policy parameters

2. **Surrogate Phase**: Train surrogate for M steps
   - Generate diverse graph batches
   - Train on structure learning task
   - Monitor F1 score
   - Rotate SCM if F1 > threshold

3. **Alternating**: Automatically switch between phases

## Output

- **Checkpoints**: Saved in `experiments/joint-training/checkpoints/`
- **Results**: Saved in `experiments/joint-training/results/`
- **Logs**: Printed to console with phase-specific metrics

## Configuration Files

- `configs/base_config.py`: Base configuration
- `configs/quick_test.yaml`: Quick test configuration

## Implementation Notes

- Uses `JointACBOTrainer` as base class
- Borrows heavily from:
  - `experiments/policy-only-training/train_grpo_diverse_fixed.py`
  - `experiments/surrogate-only-training/scripts/train_avici_style.py`
- Handles both old (pickle) and new (checkpoint_utils) formats
- Includes monkey-patch for missing `model_io` module