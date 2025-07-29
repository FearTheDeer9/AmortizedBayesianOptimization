# Two-Phase GRPO + Active Learning Training

## Overview

The two-phase training approach combines the strengths of GRPO (Gradient-based Reward Policy Optimization) with active learning for causal structure discovery:

1. **Phase 1**: Train GRPO policy with bootstrap features (structure-aware but static)
2. **Phase 2**: Use trained GRPO policy with active learning surrogate (discovers structure from data)

This approach addresses the key limitation of the current GRPO training: while bootstrap features provide good intervention guidance, they don't actually learn causal structure from data.

## Key Benefits

- **Better Structure Learning**: Active surrogate discovers true causal structure vs static bootstrap
- **Guided Exploration**: GRPO policy provides intelligent intervention selection
- **Modular Design**: Reuses existing components with minimal new code
- **Measurable Progress**: Track F1/SHD improvements during Phase 2

## Implementation

### Phase 1: GRPO Training (Standard)

Train GRPO as usual with bootstrap features:

```bash
# Run standard GRPO training (notebooks or scripts)
# This creates checkpoints with policy params saved separately
```

The training now saves policy parameters in a separate file (`policy_params.pkl`) within each checkpoint directory.

### Phase 2: Active Learning with GRPO Policy

Use the trained GRPO policy to guide active learning:

```bash
# Run Phase 2 with trained GRPO policy
poetry run python scripts/core/two_phase_training.py \
    --grpo-checkpoint checkpoints/enriched_grpo_final \
    --scm-type easy \
    --n-interventions 50 \
    --learning-rate 1e-3
```

Options:
- `--grpo-checkpoint`: Path to trained GRPO checkpoint directory
- `--scm-type`: SCM difficulty (easy, medium, hard)
- `--n-interventions`: Number of intervention steps
- `--n-observational`: Initial observational samples (default: 30)
- `--learning-rate`: Learning rate for active surrogate (default: 1e-3)
- `--compare`: Compare multiple approaches
- `--output-dir`: Directory for results (default: phase2_results)

### Comparison Mode

Compare different approaches:

```bash
poetry run python scripts/core/two_phase_training.py \
    --grpo-checkpoint checkpoints/enriched_grpo_final \
    --scm-type medium \
    --compare
```

This compares:
1. Random + Active Learning (baseline)
2. GRPO + Active Learning (Phase 2)
3. GRPO + Bootstrap (reference)

## Architecture

### Key Components

1. **GRPO Policy Loader** (`grpo_policy_loader.py`):
   - Loads trained GRPO checkpoint
   - Creates intervention function compatible with active learning
   - Maintains internal state for history tracking

2. **Phase 2 Script** (`two_phase_training.py`):
   - Orchestrates active learning with GRPO policy
   - Tracks structure learning metrics (F1, SHD)
   - Provides comparison framework

3. **Checkpoint Manager Updates**:
   - Saves policy params separately for easy loading
   - Maintains backward compatibility

### How It Works

1. **Policy Loading**:
   ```python
   loaded_policy = load_grpo_policy(checkpoint_path)
   ```

2. **Intervention Function Creation**:
   ```python
   intervention_fn = create_grpo_intervention_fn(
       loaded_policy=loaded_policy,
       scm=scm,
       intervention_range=(-2.0, 2.0)
   )
   ```

3. **Active Learning Integration**:
   ```python
   result = run_progressive_learning_demo_with_scm(
       scm=scm,
       config=config,
       pretrained_surrogate=None,  # Create learning surrogate
       pretrained_acquisition=intervention_fn  # Use GRPO policy
   )
   ```

## Results Tracking

Phase 2 tracks comprehensive metrics:

- **Target Optimization**: Final value and improvement
- **Structure Learning**: F1 scores and SHD over time
- **Sample Efficiency**: Convergence speed
- **Intervention Diversity**: Variable coverage

Results are saved as pickle files for analysis.

## Example Workflow

1. Train GRPO policy (Phase 1):
   ```bash
   # Run training notebook or script
   # Checkpoint saved to: checkpoints/enriched_grpo_final/
   ```

2. Run Phase 2 active learning:
   ```bash
   poetry run python scripts/core/two_phase_training.py \
       --grpo-checkpoint checkpoints/enriched_grpo_final \
       --scm-type easy \
       --n-interventions 100
   ```

3. Analyze results:
   ```python
   import pickle
   with open('phase2_results/phase2_result_easy.pkl', 'rb') as f:
       result = pickle.load(f)
   
   # Access metrics
   print(f"Final F1: {result['structure_learning_metrics']['final_f1']}")
   print(f"Final SHD: {result['structure_learning_metrics']['final_shd']}")
   ```

## Testing

Run tests to verify implementation:

```bash
poetry run pytest tests/test_training/test_two_phase_training.py -v
```

## Future Enhancements

1. **Online Learning**: Update surrogate during GRPO training
2. **Joint Optimization**: Train policy and surrogate together
3. **Transfer Learning**: Use learned structures across SCMs
4. **Adaptive Switching**: Dynamically switch between bootstrap and learned features