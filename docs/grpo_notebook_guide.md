# GRPO Modular Notebooks Guide

This guide explains how to use the new modular GRPO training and evaluation notebooks that support both minimization and maximization objectives.

## Overview

The modular notebooks provide:
- **No silent failures** - explicit errors when things go wrong
- **Independent cells** - run any cell without dependencies
- **Optimization direction support** - handle both MIN and MAX objectives
- **Consistent with PARENT_SCALE** - correctly handles minimization

## Key Concepts

### Optimization Direction

GRPO training now supports two optimization directions:

1. **MINIMIZE**: Optimize to reduce the target value (like PARENT_SCALE)
   - Lower target values are better
   - Used when the target represents cost, error, or loss
   - Internally converts to maximization for GRPO

2. **MAXIMIZE**: Optimize to increase the target value
   - Higher target values are better
   - Used when the target represents reward, profit, or score
   - Direct mapping to GRPO rewards

### Target Signal Mismatch Fix

Previously, there was a critical mismatch:
- PARENT_SCALE minimizes target values
- GRPO training maximized rewards
- This created conflicting optimization objectives

The new implementation fixes this by:
- Storing optimization direction in checkpoint metadata
- Converting rewards appropriately during training
- Displaying metrics correctly in evaluation

## Training Notebook Usage

### Quick Start

```python
# Cell 2: Configure training
TRAINING_MODE = "QUICK"
OPTIMIZATION_OBJECTIVE = "TARGET_MINIMIZE"  # or "TARGET_MAXIMIZE"
RANDOM_SEED = 42
```

### Training Modes

- **QUICK**: 5 minutes, 32 SCMs, for testing
- **STANDARD**: 10 minutes, 48 SCMs, for development
- **FULL**: 15 minutes, 64 SCMs, for production
- **PRECISION**: 30 minutes, 128 SCMs, for best quality

### Optimization Objectives

- **TARGET_MINIMIZE**: Like PARENT_SCALE, minimize target
- **TARGET_MAXIMIZE**: Traditional RL, maximize target
- **STRUCTURE_FOCUSED**: Emphasize causal discovery
- **BALANCED**: Balance all objectives

### Resuming Training

To resume from a checkpoint:

```python
# Cell 2: Set checkpoint path
RESUME_FROM_CHECKPOINT = "checkpoints/grpo_training/grpo_quick_minimize_20250722_120000"
```

The notebook will:
1. Load the checkpoint
2. Verify optimization compatibility
3. Resume from the last completed episode

## Evaluation Notebook Usage

### Evaluation Modes

1. **SINGLE_CHECKPOINT**: Evaluate one model
   ```python
   EVALUATION_MODE = "SINGLE_CHECKPOINT"
   ```

2. **COMPARE_CHECKPOINTS**: Compare multiple training runs
   ```python
   EVALUATION_MODE = "COMPARE_CHECKPOINTS"
   ```

3. **COMPARE_OBJECTIVES**: Compare MIN vs MAX policies
   ```python
   EVALUATION_MODE = "COMPARE_OBJECTIVES"
   ```

### Checkpoint Selection

The notebook automatically discovers checkpoints and shows:
- Optimization direction
- Training mode
- Timestamp

You can select by:
- Index: `selected_checkpoints = [available_checkpoints[0]]`
- Name: Filter by specific checkpoint name
- Direction: Filter by MINIMIZE or MAXIMIZE

### Understanding Results

For **MINIMIZE** checkpoints:
- Lower values are better
- Plots show "(↓ better)" labels
- Y-axis may be inverted for intuitive visualization

For **MAXIMIZE** checkpoints:
- Higher values are better
- Plots show "(↑ better)" labels
- Standard Y-axis orientation

## Common Workflows

### 1. Train and Evaluate a Minimization Policy

```bash
# Step 1: Train
# In grpo_training_modular.ipynb:
TRAINING_MODE = "QUICK"
OPTIMIZATION_OBJECTIVE = "TARGET_MINIMIZE"
# Run all cells

# Step 2: Evaluate
# In grpo_evaluation_modular.ipynb:
EVALUATION_MODE = "SINGLE_CHECKPOINT"
# The latest checkpoint is selected automatically
# Run all cells
```

### 2. Compare Minimization vs Maximization

```bash
# Step 1: Train minimization policy
OPTIMIZATION_OBJECTIVE = "TARGET_MINIMIZE"
# Run training

# Step 2: Train maximization policy
OPTIMIZATION_OBJECTIVE = "TARGET_MAXIMIZE"
# Run training

# Step 3: Compare in evaluation
EVALUATION_MODE = "COMPARE_OBJECTIVES"
# Both checkpoints selected automatically
# Run evaluation
```

### 3. Convert Checkpoint Direction

```python
from scripts.notebooks.checkpoint_utils import (
    find_checkpoint_by_name, convert_checkpoint_optimization
)

# Find checkpoint
ckpt = find_checkpoint_by_name(checkpoint_dir, "grpo_quick_minimize_20250722_120000")

# Convert from MINIMIZE to MAXIMIZE
new_ckpt = convert_checkpoint_optimization(
    ckpt, 
    new_direction="MAXIMIZE",
    new_checkpoint_dir=checkpoint_dir
)
```

## Checkpoint Management

### List Available Checkpoints

```python
from scripts.notebooks.checkpoint_utils import list_checkpoints, create_checkpoint_summary_table

# List all checkpoints
checkpoints = list_checkpoints(checkpoint_dir)

# Filter by optimization direction
minimize_checkpoints = list_checkpoints(
    checkpoint_dir, 
    filter_by={'optimization_direction': 'MINIMIZE'}
)

# Display as table
print(create_checkpoint_summary_table(checkpoints))
```

### Load Checkpoint Metadata

```python
from scripts.notebooks.checkpoint_utils import find_checkpoint_by_name

ckpt = find_checkpoint_by_name(checkpoint_dir, "grpo_quick_minimize_20250722_120000")
print(f"Direction: {ckpt.optimization_config.direction}")
print(f"Training mode: {ckpt.training_config['mode']}")
```

## Troubleshooting

### "Optimization direction mismatch" Error

This occurs when trying to resume training with a different optimization direction than the checkpoint. Solutions:
1. Use a checkpoint with matching direction
2. Train from scratch with desired direction
3. Convert the checkpoint (see above)

### Silent Failures

The old notebooks had many places where errors were silently ignored. The new notebooks will raise explicit `NotebookError` exceptions. This is intentional - it's better to know immediately when something fails.

### Checkpoint Not Found

Make sure:
1. The checkpoint directory exists
2. The checkpoint has metadata.json
3. The path is correct

Use `list_checkpoints()` to see all available checkpoints.

### Metrics Look Wrong

Check the optimization direction:
- For MINIMIZE: Lower is better, may show negative improvements
- For MAXIMIZE: Higher is better, shows positive improvements

The plots automatically adjust labels and formatting.

## Migration from Old Notebooks

If you have checkpoints from the old notebooks:
1. They default to MAXIMIZE direction
2. Add metadata manually if needed
3. Use conversion utilities to change direction

Example metadata fix:
```python
# Add optimization config to old checkpoint
metadata = {
    "optimization_config": {
        "direction": "MAXIMIZE",
        "target_baseline": 0.0
    },
    # ... other metadata
}
```

## Best Practices

1. **Always specify optimization direction** explicitly in training
2. **Check checkpoint compatibility** before comparing
3. **Use descriptive names** when saving checkpoints
4. **Document your objectives** in experiment notes
5. **Verify metric interpretation** matches your use case

## Example Experiment

Here's a complete experiment comparing minimization vs maximization:

```python
# Experiment: Does optimization direction affect exploration?

# 1. Train minimization policy
# grpo_training_modular.ipynb
TRAINING_MODE = "STANDARD"
OPTIMIZATION_OBJECTIVE = "TARGET_MINIMIZE"
RANDOM_SEED = 42
# Run training -> checkpoint: grpo_standard_minimize_[timestamp]

# 2. Train maximization policy  
# grpo_training_modular.ipynb
OPTIMIZATION_OBJECTIVE = "TARGET_MAXIMIZE"  
RANDOM_SEED = 42  # Same seed for fair comparison
# Run training -> checkpoint: grpo_standard_maximize_[timestamp]

# 3. Evaluate both
# grpo_evaluation_modular.ipynb
EVALUATION_MODE = "COMPARE_OBJECTIVES"
NUM_TEST_SCMS = 20
RUNS_PER_METHOD = 5
# Run evaluation

# Results will show:
# - Which direction achieves better final values
# - How exploration strategies differ
# - Whether one generalizes better
```

## Summary

The modular notebooks provide a robust, failure-explicit approach to GRPO training and evaluation with proper support for both minimization and maximization objectives. This ensures compatibility with algorithms like PARENT_SCALE while maintaining the flexibility to optimize in either direction.