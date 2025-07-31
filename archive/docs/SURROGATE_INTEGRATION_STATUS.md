# Surrogate Integration Status

## Current State (As of Jan 2025)

### What Works:
- BC surrogate training from demonstrations
- Surrogate model architecture (ContinuousParentSetPredictionModel)
- Checkpoint saving/loading infrastructure
- Active learning surrogates for evaluation

### What's Missing:

#### 1. GRPO Reward Integration
The GRPO trainer does NOT use structure-aware rewards even when surrogate is present:
- Uses `compute_clean_reward` instead of `compute_structure_aware_reward`
- Structure rewards (info_gain, parent_bonus) are defined but return 0.0
- No reward shaping based on structure predictions

#### 2. Pre-trained Surrogate Loading
- Infrastructure exists (`load_bc_surrogate_model`) but isn't used
- GRPO always initializes fresh surrogate with random weights
- No command-line option to load pre-trained surrogate checkpoint

#### 3. Joint Training
- Surrogate update function exists but is never called
- No actual parameter updates during GRPO training
- "Joint training" is currently just co-existence, not co-learning

## How to Use Pre-trained Surrogates (Current Workaround)

### Option 1: Use Active Learning in Evaluation
```bash
# This works! Surrogates adapt during evaluation
poetry run python scripts/evaluate_acbo_methods.py \
    --use_active_learning \
    --surrogate_checkpoint checkpoints/bc_surrogate/checkpoint_final.pkl
```

### Option 2: Manual Integration (Not Implemented)
Would require modifying GRPO trainer to:
1. Accept --surrogate_checkpoint argument
2. Load pre-trained parameters
3. Implement proper structure-aware rewards

## Recommended Training Pipeline

### 1. Train BC Surrogate First
```bash
poetry run python scripts/train_acbo_methods.py \
    --method bc \
    --model_type surrogate \
    --episodes 1500 \
    --checkpoint_dir checkpoints/bc_surrogate
```

### 2. Train GRPO Policy (without surrogate integration)
```bash
poetry run python scripts/train_acbo_methods.py \
    --method grpo \
    --episodes 2000 \
    --checkpoint_dir checkpoints/grpo_solo
```

### 3. Evaluate with Active Learning
```bash
poetry run python scripts/evaluate_acbo_methods.py \
    --grpo checkpoints/grpo_solo/checkpoint_final.pkl \
    --use_active_learning \
    --surrogate_checkpoint checkpoints/bc_surrogate/checkpoint_final.pkl
```

## Future Work

To properly integrate surrogates into GRPO training:

1. **Implement structure-aware reward computation**
   - Complete the TODO in `compute_structure_aware_reward`
   - Add entropy reduction rewards
   - Add parent intervention bonuses

2. **Add surrogate checkpoint loading to GRPO**
   - Add --surrogate_checkpoint argument
   - Load pre-trained parameters instead of random init

3. **Enable true joint training**
   - Call surrogate_update_fn during episodes
   - Balance policy and surrogate learning rates
   - Add surrogate loss to metrics

4. **Validate impact**
   - Compare rewards with/without structure bonuses
   - Measure if policy learns to trust surrogate
   - Track structure learning convergence