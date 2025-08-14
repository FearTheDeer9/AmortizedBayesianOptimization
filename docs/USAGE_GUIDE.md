# ACBO Usage Guide

This guide provides step-by-step instructions for training and evaluating Amortized Causal Bayesian Optimization (ACBO) models.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Training Models](#training-models)
3. [Checkpoint Management](#checkpoint-management)
4. [Evaluating Models](#evaluating-models)
5. [Common Workflows](#common-workflows)
6. [Parameter Reference](#parameter-reference)
7. [Troubleshooting](#troubleshooting)

## Quick Start

The ACBO system consists of two main scripts:
- `scripts/main/train.py` - Train surrogate and policy models
- `scripts/main/evaluate.py` - Evaluate trained models

### Basic Training Example
```bash
# Train a BC surrogate model for structure learning
python scripts/main/train.py --method surrogate --episodes 100

# Train a BC policy using expert demonstrations
python scripts/main/train.py --method bc --episodes 100

# Train a GRPO policy with reinforcement learning
python scripts/main/train.py --method grpo --episodes 1000
```

### Basic Evaluation Example
```bash
# Evaluate with baselines
python scripts/main/evaluate.py \
  --include_baselines \
  --baseline_surrogate trained \
  --register_surrogate trained checkpoints/bc_surrogate_final \
  --n_scms 10 \
  --n_interventions 20
```

## Training Models

### 1. Training a Surrogate Model (Structure Learning)

The surrogate model learns to predict causal structures from observational and interventional data.

```bash
python scripts/main/train.py \
  --method surrogate \
  --episodes 100 \
  --demo_path expert_demonstrations/raw/raw_demonstrations \
  --max_demos 50 \
  --encoder_type node_feature \
  --surrogate_hidden_dim 128 \
  --surrogate_layers 4 \
  --surrogate_heads 8
```

**Key Parameters:**
- `--encoder_type`: Type of encoder (`node_feature` recommended)
- `--max_demos`: Limit demonstrations for faster training/testing
- `--surrogate_hidden_dim`: Hidden dimension size (128-256)
- `--surrogate_layers`: Number of transformer layers (4-6)
- `--surrogate_heads`: Number of attention heads (8)

### 2. Training a BC Policy

Behavioral Cloning (BC) learns an intervention policy from expert demonstrations.

```bash
python scripts/main/train.py \
  --method bc \
  --episodes 100 \
  --demo_path expert_demonstrations/raw/raw_demonstrations \
  --max_demos 50 \
  --architecture alternating_attention \
  --use_permutation \
  --label_smoothing 0.1
```

**Key Parameters:**
- `--architecture`: Model architecture (`alternating_attention` recommended)
- `--use_permutation`: Prevent position shortcuts in learning
- `--label_smoothing`: Regularization (0.1 recommended)

### 3. Training a GRPO Policy

Group Relative Policy Optimization (GRPO) learns through reinforcement learning.

```bash
python scripts/main/train.py \
  --method grpo \
  --episodes 1000 \
  --scm_type mixed \
  --min_vars 3 \
  --max_vars 8 \
  --use_surrogate \
  --surrogate_checkpoint checkpoints/bc_surrogate_final
```

**Key Parameters:**
- `--scm_type`: Type of SCMs to train on (`mixed`, `random`, `chain`, etc.)
- `--use_surrogate`: Enable structure learning during training
- `--surrogate_checkpoint`: Path to pre-trained surrogate

### 4. Combined Training Workflow

Train surrogate first, then use it for GRPO training:

```bash
python scripts/main/train.py \
  --method grpo_with_surrogate \
  --episodes 1000 \
  --max_demos 50
```

This automatically:
1. Trains a BC surrogate model
2. Uses it to train a GRPO policy with structure learning

## Checkpoint Management

### Overview

The checkpoint system stores complete model state including architecture configuration, ensuring compatibility when loading models.

### Checkpoint Structure

Each checkpoint contains:
```python
{
    'model_type': 'policy' or 'surrogate',
    'architecture': {
        'hidden_dim': 256,
        'num_layers': 4,
        'num_heads': 8,
        'encoder_type': 'node_feature',  # For surrogates
        # ... other architecture params
    },
    'params': jax_params,  # Model weights
    'optimizer_state': opt_state,  # Optimizer state
    'training_metrics': {...},  # Training history
    'metadata': {...}  # Additional info
}
```

### Default Checkpoint Locations

Training automatically saves checkpoints to:
- **BC Policy**: `checkpoints/bc_final`
- **BC Surrogate**: `checkpoints/bc_surrogate_final`
- **GRPO Policy**: `checkpoints/unified_grpo_final`

### Loading Checkpoints for Evaluation

#### Correct Usage
```bash
# Register a trained surrogate
python scripts/main/evaluate.py \
  --register_surrogate trained checkpoints/bc_surrogate_final \
  --register_policy bc checkpoints/bc_final \
  --evaluate_pairs bc trained
```

#### Loading Best Practices

1. **Verify Architecture Compatibility**
   - Checkpoints store architecture configuration automatically
   - Models with different architectures cannot share weights

2. **Use Correct Paths**
   - Paths are relative to project root
   - Use absolute paths if running from different directories

3. **Test After Training**
   - Always verify checkpoint loads correctly after training
   - Run a quick evaluation to confirm expected behavior

### Using Pre-trained Models

#### Load Surrogate for GRPO Training
```bash
python scripts/main/train.py \
  --method grpo \
  --use_surrogate \
  --surrogate_checkpoint checkpoints/bc_surrogate_final
```

#### Load Multiple Models for Evaluation
```bash
python scripts/main/evaluate.py \
  --register_surrogate trained checkpoints/bc_surrogate_final \
  --register_policy bc_policy checkpoints/bc_final \
  --register_policy grpo_policy checkpoints/unified_grpo_final \
  --evaluate_pairs bc_policy trained \
  --evaluate_pairs grpo_policy trained
```

### Checkpoint Compatibility

⚠️ **Critical**: Models trained with different architectures are NOT compatible:
- `encoder_type` must match for surrogates (e.g., `node_feature` vs `simple`)
- `architecture` must match for policies (e.g., `alternating_attention` vs `simple`)

### Verifying Checkpoint Contents

To inspect a checkpoint:
```python
import pickle
from pathlib import Path

checkpoint_path = Path('checkpoints/bc_surrogate_final')
with open(checkpoint_path, 'rb') as f:
    checkpoint = pickle.load(f)
    
print(f"Model type: {checkpoint['model_type']}")
print(f"Architecture: {checkpoint['architecture']}")
print(f"Training epochs: {checkpoint['training_metrics']['epochs_trained']}")
```

### Best Practices

1. **Consistent Architecture**: Use `node_feature` encoder for surrogates, `alternating_attention` for policies
2. **Organized Storage**: Use descriptive checkpoint names for different model variants
3. **Documentation**: Record architecture and training parameters for each checkpoint
4. **Validation**: Test checkpoint loading and inference immediately after training

## Evaluating Models

### Basic Evaluation

```bash
python scripts/main/evaluate.py \
  --include_baselines \
  --register_surrogate trained checkpoints/bc_surrogate_final \
  --register_policy bc checkpoints/bc_final \
  --evaluate_pairs bc trained \
  --n_scms 10 \
  --n_interventions 20 \
  --plot \
  --plot_trajectories
```

### Comprehensive Evaluation

Evaluate all policy-surrogate combinations:

```bash
python scripts/utils/comprehensive_evaluation.py
```

This evaluates:
- Random policy + Dummy surrogate (baseline)
- Random policy + Trained surrogate
- Oracle policy + Dummy surrogate
- Oracle policy + Trained surrogate

### Evaluation Parameters

- `--n_scms`: Number of test SCMs (default: 10)
- `--n_obs`: Initial observations (default: 100)
- `--n_interventions`: Number of interventions (default: 20)
- `--n_samples`: Samples per intervention (default: 10)
- `--plot`: Generate comparison plots
- `--plot_trajectories`: Generate trajectory plots

## Common Workflows

### Workflow 1: Complete Pipeline

```bash
# Step 1: Train surrogate model
python scripts/main/train.py --method surrogate --episodes 100 --max_demos 50

# Step 2: Train BC policy
python scripts/main/train.py --method bc --episodes 100 --max_demos 50

# Step 3: Train GRPO with surrogate
python scripts/main/train.py --method grpo --episodes 1000 --use_surrogate \
  --surrogate_checkpoint checkpoints/bc_surrogate_final

# Step 4: Evaluate all methods
python scripts/main/evaluate.py \
  --include_baselines \
  --register_surrogate trained checkpoints/bc_surrogate_final \
  --register_policy bc checkpoints/bc_final \
  --register_policy grpo checkpoints/unified_grpo_final \
  --evaluate_pairs bc trained \
  --evaluate_pairs grpo trained \
  --n_scms 10 \
  --plot
```

### Workflow 2: Quick Testing

```bash
# Train with limited data for testing
python scripts/main/train.py --method surrogate --episodes 10 --max_demos 5

# Quick evaluation
python scripts/main/evaluate.py \
  --include_baselines \
  --register_surrogate test checkpoints/bc_surrogate_final \
  --n_scms 3 \
  --n_interventions 5
```

### Workflow 3: Active Learning

Enable surrogate updates during evaluation:

```bash
python scripts/main/evaluate.py \
  --register_surrogate trained checkpoints/bc_surrogate_final \
  --register_policy bc checkpoints/bc_final \
  --evaluate_pairs bc trained \
  --surrogate_update_strategy bic \
  --n_scms 5
```

## Parameter Reference

### Training Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--episodes` | Training episodes/epochs | 1000 | 100 for BC, 1000 for GRPO |
| `--seed` | Random seed | 42 | Any integer |
| `--batch_size` | Batch size | 32 | 32-64 |
| `--learning_rate` | Learning rate | 3e-4 | 1e-4 to 3e-4 |
| `--hidden_dim` | Hidden dimension | 256 | 128-256 |
| `--max_demos` | Max demonstrations | None | 50-100 for testing |

### SCM Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--scm_type` | Type of SCMs | mixed | random, chain, fork, collider, mixed |
| `--min_vars` | Min variables | 3 | 3-5 |
| `--max_vars` | Max variables | 8 | 6-10 |

### Architecture Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--architecture` | Policy architecture | alternating_attention | simple, attention, alternating_attention |
| `--encoder_type` | Surrogate encoder | node_feature | node_feature, node, simple |

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `--batch_size`
   - Reduce `--hidden_dim`
   - Use fewer `--max_demos`

2. **Training Not Converging**
   - Increase `--episodes`
   - Adjust `--learning_rate`
   - Check data with smaller `--max_demos`

3. **F1 Score = 0 or Very Low**
   - Ensure surrogate is properly trained (100+ episodes recommended)
   - Verify correct checkpoint is loaded
   - Check demonstrations contain structure information
   - Confirm encoder_type consistency

4. **Model Outputs Uniform Predictions**
   - Indicates architecture mismatch or initialization issue
   - Checkpoints include architecture configuration for automatic compatibility
   - Verify encoder_type matches expected configuration

5. **Gradient Issues During Training**
   - Model uses Orthogonal initialization for stability
   - If gradients explode: reduce learning rate or enable gradient clipping
   - If gradients vanish: check batch normalization and activation functions

6. **Inconsistent Results Across Environments**
   - Verify model architecture files are consistent
   - Check initialization schemes match
   - Ensure checkpoint architecture configuration is preserved

7. **Slow Training**
   - Use `--max_demos` to limit data
   - Reduce model size parameters
   - Use smaller `--n_scms` for evaluation

### Debugging Tips

1. **Test with Small Data First**
   ```bash
   python scripts/main/train.py --method surrogate --episodes 10 --max_demos 5
   ```

2. **Check Model Loading**
   ```bash
   python scripts/main/evaluate.py \
     --register_surrogate test path/to/checkpoint \
     --n_scms 1 \
     --n_interventions 3
   ```

3. **Verify Demonstrations**
   ```bash
   ls -la expert_demonstrations/raw/raw_demonstrations/*.pkl | head -5
   ```

### Getting Help

- Check logs in terminal output
- Review checkpoint files in `checkpoints/`
- Examine evaluation results in `evaluation_results/`

## Next Steps

- See [CANONICAL_PATTERNS.md](../CANONICAL_PATTERNS.md) for development patterns
- See [TRAINING_COMMANDS.md](../TRAINING_COMMANDS.md) for more examples
- See example scripts in `scripts/examples/` for automation