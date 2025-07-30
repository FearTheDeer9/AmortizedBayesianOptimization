# ACBO Pipeline Status - January 29, 2025

## Executive Summary

The ACBO (Adaptive Causal Bayesian Optimization) pipeline has undergone significant consolidation and cleanup. The GRPO training is now using the true GRPO algorithm with batch advantages (not REINFORCE), while the BC (Behavioral Cloning) training has been fixed by separating policy and surrogate trainers. This document captures the current state, architecture, and next steps.

## Session Accomplishments (January 29)

### Fixed BC Training
- **Separated trainers**: Created PolicyBCTrainer and SurrogateBCTrainer to avoid model_type branching
- **Shape issues resolved**: Each trainer now handles its specific data format correctly
- **Clean implementation**: BC training now completes successfully from start to end

### Enhanced Evaluation Metrics
- **Mean trajectory value tracking**: Added `mean_trajectory_value` to track average target value throughout optimization (not just best achieved)
- **Fixed F1 score calculation**: Now properly handles ParentSetPosterior objects from AVICI integration
- **Oracle structure metrics**: Added F1/SHD reporting for Oracle when paired with surrogates
- **Improved insights**: Mean trajectory values reveal that Random stays near initial values while Oracle consistently optimizes

### Discovered Surrogate Integration Gaps
- **No reward integration**: GRPO uses `compute_clean_reward` which ignores structure predictions
- **No joint training**: Surrogate parameters never update during GRPO episodes
- **Unused infrastructure**: Pre-trained surrogate loading exists but isn't exposed in CLI
- **Key finding**: Current "joint training" provides no benefit - surrogate stays at random initialization

### Created Training Resources
- **Training scripts**: `train_focused.sh` and `train_long_comprehensive.sh` for easy multi-model training
- **Documentation**: Created SURROGATE_INTEGRATION_STATUS.md documenting the gaps
- **Commands reference**: Updated TRAINING_COMMANDS.md with practical examples

## Current Architecture

### Core Components

1. **GRPO Training** (`src/causal_bayes_opt/training/unified_grpo_trainer.py`)
   - ✅ Implements true GRPO with batch advantages
   - ✅ Uses 3-channel tensor format [T, n_vars, 3]
   - ✅ Supports flexible SCM input (list, dict, callable)
   - ✅ Includes convergence detection
   - ✅ Integrates surrogate learning
   - Status: **WORKING**

2. **BC Training** (Separated trainers)
   - **PolicyBCTrainer** (`src/causal_bayes_opt/training/policy_bc_trainer.py`)
     - ✅ Trains intervention selection policies
     - ✅ Handles variable selection and value prediction
     - ✅ Works with 3-channel tensor format
   - **SurrogateBCTrainer** (`src/causal_bayes_opt/training/surrogate_bc_trainer.py`)
     - ✅ Trains structure learning models
     - ✅ Learns parent probability predictions
     - ✅ Compatible with AVICI integration
   - Status: **WORKING**

3. **Data Preprocessing** (`src/causal_bayes_opt/training/data_preprocessing.py`)
   - ✅ Cleanly separates data preparation from training
   - ✅ Handles intervention variable tuple format
   - ✅ Converts demonstrations to 3-channel tensors
   - Status: **WORKING**

4. **Evaluation** (`src/causal_bayes_opt/evaluation/universal_evaluator.py`)
   - ✅ Universal evaluator for all methods (GRPO, BC, Random, Oracle)
   - ✅ Handles checkpoint loading and model inference
   - ✅ Enhanced metrics: mean trajectory values, F1/SHD scores
   - ✅ Active learning surrogate support
   - Status: **WORKING**

5. **Surrogate Integration** (Incomplete)
   - ⚠️ Structure-aware rewards defined but not used
   - ⚠️ No parameter updates during GRPO training
   - ⚠️ Pre-trained loading not exposed in CLI
   - Status: **NEEDS INTEGRATION**

## Key Design Decisions

### 1. True GRPO Implementation
```python
# Compute advantages using group baseline (key GRPO innovation)
group_baseline = jnp.mean(rewards)
advantages = rewards - group_baseline
```
This is the critical difference from REINFORCE - using batch statistics for variance reduction.

### 2. 3-Channel Tensor Format
All models use consistent tensor format: `[T, n_vars, 3]` where:
- T: Time steps (observation history)
- n_vars: Number of variables in the SCM
- 3: Channels (values, masks, timestamps)

### 3. Variable-Agnostic Models
The surrogate models (like `ContinuousParentSetPredictionModel`) are designed to handle variable numbers of nodes, similar to set transformers. This is why JAX shape requirements conflict with the flexibility needed.

### 4. Data Flow
```
Expert Demonstrations → Data Preprocessing → 3-Channel Tensors → BC Training
                                           ↓
                                    Policy/Surrogate Models
```

## Remaining Implementation Gaps

### Gap 1: Continue Training from Checkpoint
- **Status**: Infrastructure exists but not exposed
- **What's needed**:
  ```python
  # Add to train_acbo_methods.py
  parser.add_argument('--resume_from', type=str, help='Resume from checkpoint')
  
  # In trainer initialization:
  if args.resume_from:
      checkpoint = load_checkpoint(args.resume_from)
      trainer.load_state(checkpoint)
  ```

### Gap 2: Dynamic Policy-Surrogate Pairing
- **Status**: Partially working via `--use_active_learning`
- **What's needed**:
  - Better CLI for pairing arbitrary policy + surrogate checkpoints
  - Currently can only use active learning or loaded surrogates, not mix-and-match freely

### Gap 3: Trajectory Visualization
- **Status**: Basic plots exist (bar charts, heatmaps)
- **What's needed**:
  - Line plots showing per-step metrics (SHD, F1, target value)
  - Store StepMetrics history in evaluation results
  - Plot functions for:
    ```python
    # Per-SCM trajectories
    plot_scm_trajectory(history, metrics=['f1', 'shd', 'target_value'])
    
    # Mean trajectories across SCMs
    plot_mean_trajectories(all_histories, methods=['GRPO', 'BC', 'Oracle'])
    ```

### Gap 4: Final Code Cleanup
- **Already deleted** (this session):
  - All clean/simplified trainers and evaluators
  - Redundant BC training files
- **Still to delete**:
  ```
  # Root directory
  - test_bc_*.py
  - find_gradient_issue.py
  - fix_bc_notebook_cells.py
  - check_bc_checkpoint.py
  - debug_structure_learning.json
  - Old .md files (CLEAN_ACBO_SUMMARY.md, etc.)
  
  # Keep only:
  - ACBO_PIPELINE_STATUS_20250129.md
  - TRAINING_COMMANDS.md
  - SURROGATE_INTEGRATION_STATUS.md
  - README.md
  ```

## Testing Results

### Updated Pipeline Status
1. **GRPO Training**: ✅ SUCCESS
   - Trains successfully with unified trainer
   - Saves checkpoints correctly
   - Uses true GRPO algorithm with batch advantages

2. **BC Training**: ✅ SUCCESS (after separation)
   - PolicyBCTrainer: Successfully trains intervention policies
   - SurrogateBCTrainer: Successfully trains structure models
   - Both complete training without errors

3. **Evaluation**: ✅ SUCCESS (enhanced)
   - Loads checkpoints correctly
   - Computes all metrics including mean trajectory values
   - Handles ParentSetPosterior objects properly
   - Active learning surrogates work during evaluation

## Code to Delete

Based on our investigation, the following can be safely deleted:

### Trainers (Redundant)
- `src/causal_bayes_opt/training/clean_grpo_trainer.py` - Uses REINFORCE, not true GRPO
- `src/causal_bayes_opt/training/simplified_grpo_trainer.py` - Also REINFORCE
- `src/causal_bayes_opt/training/clean_bc_trainer.py` - Incomplete implementation
- `src/causal_bayes_opt/training/simplified_bc_trainer.py` - Duplicate functionality

### Evaluators (Redundant)
- `src/causal_bayes_opt/evaluation/clean_grpo_evaluator.py`
- `src/causal_bayes_opt/evaluation/simplified_grpo_evaluator.py`
- `src/causal_bayes_opt/evaluation/simplified_bc_evaluator.py`
- `src/causal_bayes_opt/evaluation/grpo_evaluator_debug.py`
- `src/causal_bayes_opt/evaluation/grpo_evaluator_diagnostic.py`
- `src/causal_bayes_opt/evaluation/grpo_evaluator_fixed.py`

### Other Files
- `src/causal_bayes_opt/training/modular_trainer.py` - Already deleted, but referenced
- Various test/debug scripts in root directory

## Migration Guide

### For GRPO Training
```python
# Old (REINFORCE)
from training.clean_grpo_trainer import CleanGRPOTrainer
trainer = CleanGRPOTrainer(config)

# New (True GRPO)
from training.unified_grpo_trainer import create_unified_grpo_trainer
trainer = create_unified_grpo_trainer(config)
```

### For BC Training
```python
# Policy training
from training.policy_bc_trainer import PolicyBCTrainer
trainer = PolicyBCTrainer(hidden_dim=256, ...)
results = trainer.train(demo_path)

# Surrogate training
from training.surrogate_bc_trainer import SurrogateBCTrainer
trainer = SurrogateBCTrainer(hidden_dim=128, num_layers=4, ...)
results = trainer.train(demo_path)
```

### For Evaluation
```python
# Use universal evaluator
from evaluation.universal_evaluator import UniversalEvaluator
evaluator = UniversalEvaluator()
results = evaluator.evaluate_checkpoints(grpo_path, bc_path)
```

## Lessons Learned

1. **Avoid Debugging Loops**: Multiple times we entered loops of small patches rather than addressing root causes
2. **Test Early and Often**: Many issues could have been caught earlier with end-to-end testing
3. **Clean Architecture First**: Having duplicate "clean" and "simplified" versions caused confusion
4. **Document Design Decisions**: Understanding why models are variable-agnostic would have saved time
5. **Separate Concerns**: Data preprocessing should be separate from training logic

## Next Steps

### Immediate (Priority 1)
1. ✅ Fix BC training shape issues (DONE - separated trainers)
2. ✅ Delete redundant trainers/evaluators (DONE)
3. Clean up root directory files (listed in Gap 4 above)

### Short Term (Priority 2)
1. Implement checkpoint resuming (add `--resume_from`)
2. Add flexible policy-surrogate pairing interface
   - Allow `--grpo_policy checkpoint1 --bc_surrogate checkpoint2`
   - Mix and match any trained components
3. Add trajectory plotting functions
4. Fix surrogate reward integration in GRPO

### Long Term (Priority 3)
1. Implement true joint training (surrogate updates during GRPO)
2. Create comprehensive evaluation dashboard
3. Add automated hyperparameter tuning

## Configuration Reference

### GRPO Training
```yaml
method: grpo
episodes: 1000
batch_size: 64  # GRPO group size
learning_rate: 3e-4
use_surrogate: true
checkpoint_dir: checkpoints/grpo
```

### BC Training
```yaml
method: bc
episodes: 500
batch_size: 32
learning_rate: 1e-3
demo_path: expert_demonstrations/raw/raw_demonstrations
checkpoint_dir: checkpoints/bc
```

## File Structure
```
causal_bayes_opt/
├── src/causal_bayes_opt/
│   ├── training/
│   │   ├── unified_grpo_trainer.py    # ✅ Keep - True GRPO
│   │   ├── policy_bc_trainer.py       # ✅ Keep - Policy BC
│   │   ├── surrogate_bc_trainer.py    # ✅ Keep - Surrogate BC
│   │   ├── data_preprocessing.py      # ✅ Keep - Working
│   │   └── active_learning.py         # ✅ Keep - AL surrogates
│   └── evaluation/
│       └── universal_evaluator.py      # ✅ Keep - Enhanced evaluator
├── scripts/
│   ├── train_acbo_methods.py          # ✅ Keep - Main training
│   ├── evaluate_acbo_methods.py       # ✅ Keep - Main evaluation
│   ├── train_focused.sh               # ✅ Keep - Quick training
│   └── train_long_comprehensive.sh    # ✅ Keep - Full training
└── docs/
    ├── ACBO_PIPELINE_STATUS_20250129.md
    ├── TRAINING_COMMANDS.md
    ├── SURROGATE_INTEGRATION_STATUS.md
    └── README.md
```

## Key Insights

- **Minimization works correctly**: Rewards are properly computed for minimization direction
- **Random baseline issue**: Only 4 SCMs tested by default, causing high variance in comparisons
- **Mean trajectory value is more informative**: Shows consistent optimization behavior vs lucky best values
- **Surrogate integration currently provides no benefit**: GRPO training ignores structure predictions
- **Active learning during evaluation**: Best current approach for structure-aware optimization

## Summary

The ACBO pipeline has been successfully consolidated and enhanced:

✅ **Working Components**:
- GRPO trainer with true batch advantages algorithm
- Separated BC trainers for policies and surrogates
- Universal evaluator with enhanced metrics
- Active learning surrogates for evaluation

🚧 **Remaining Gaps**:
- Checkpoint resuming not exposed in CLI
- Policy-surrogate pairing needs better interface
- Trajectory visualization missing line plots
- Surrogate rewards not integrated in GRPO

This represents a significant improvement with a clean, maintainable codebase ready for the remaining enhancements.