# Task Plan: Fix Phase 2 Evaluation Issues
## Date: 2025-01-28

## Objective
Fix Phase 2 evaluation to show meaningful structure learning and optimization results by addressing training duration, SCM rotation timing, and evaluation metrics.

## Current State Analysis
- Phase 2 evaluation shows poor results: large variance, no structure learning
- Training uses QUICK mode (3.2 minutes, 96 episodes) - insufficient for learning
- SCM rotation happens too quickly (every 32 episodes)
- Reward signal is correct (uses target value, not intervention values)
- Collapse prevention is implemented (global standardization)
- Missing oracle baseline in Phase 2 makes it hard to assess performance

## Implementation Plan

### Step 1: Add Training Mode Selection to Notebook
- Add dropdown for QUICK/FULL/PRECISION modes in training notebook
- Show expected duration and episodes for each mode
- Default to FULL for meaningful results

### Step 2: Add Dynamic SCM Rotation Option
- Add checkbox for "Dynamic SCM Progression" 
- When enabled, rotate based on convergence rather than fixed episodes
- Show convergence metrics during training

### Step 3: Fix Oracle Baseline in Phase 2
- Re-add oracle policy evaluation that was removed
- Shows best possible minimization for each SCM
- Provides performance benchmark

### Step 4: Add Structure Learning Metrics
- Track F1 score progression during training
- Show marginal parent probabilities evolution
- Display structural Hamming distance (SHD)

### Step 5: Create Comprehensive Evaluation
- Compare random vs structure-aware vs GRPO vs oracle
- Show both optimization and structure learning metrics
- Visualize learning curves over episodes

## Design Decisions Log
[APPEND ONLY - Record all design decisions as they're made]

2025-01-28: Decision to prioritize training duration fix first as it's the root cause
2025-01-28: Keep existing reward signal as it's correctly implemented
2025-01-28: Add oracle baseline back to provide clear performance benchmark

## Problems & Solutions
[APPEND ONLY - Document issues encountered and resolutions]

2025-01-28: Problem - QUICK mode insufficient for learning. Solution - Add mode selection
2025-01-28: Problem - Can't see convergence. Solution - Add convergence monitoring

## Progress Updates
[APPEND ONLY - Regular updates on implementation progress]

2025-01-28: Created plan document
2025-01-28: Starting with training mode selection implementation
2025-01-28: Fixed Haiku parameter mismatch in grpo_policy_loader.py
2025-01-28: Updated parameter names to match training exactly: enriched_history, target_variable_idx, is_training
2025-01-28: Fixed export error in evaluation notebook Cell 15 to handle different result types
2025-01-28: Added comprehensive result conversion logic with NumpyEncoder for JSON serialization
2025-01-28: Fixed UnboundLocalError in complete_workflow_demo.py where true_parents was used before definition
2025-01-28: Moved true_parents definition to beginning of run_progressive_learning_demo_with_scm function
2025-01-28: Fixed misleading print statement that said "BC acquisition policy" when using GRPO
2025-01-28: Fixed negative uncertainty validation error by improving numerical precision handling
2025-01-28: Added tolerance for near-zero negative values and improved clamping in uncertainty calculation
2025-01-28: Discovered root cause of RoleBasedProjection Haiku error - use_role_based_projection not saved in config
2025-01-28: Fixed PolicyFactory to save use_role_based_projection in architecture config
2025-01-28: Ensured both training and loading use consistent network parameters

## CORRECTIONS BASED ON USER FEEDBACK

### Reward Signal Analysis
After user's question, I investigated and found:

1. **Optimization Reward**: The actual optimization reward is computed in `_compute_optimization_reward()` which:
   - First tries SCM-objective reward (distance to theoretical optimum)
   - Falls back to improved relative reward (comparing to expected baseline)
   - Does NOT simply use `-target_value`

2. **Magnitude Bonus**: Found in `optimization_adapter.py`:
   ```python
   magnitude = sum(abs(v) for v in intervention_values.values())
   magnitude_bonus = min(0.5, magnitude * 0.1)
   ```
   This is indeed 1/10 of intervention magnitude, capped at 0.5.

3. **Reward Weights**: From training config:
   - Optimization: 80% (0.8)
   - Structure discovery: 10% (0.1)
   - Efficiency/exploration: 10% (0.1)

### SCM Rotation Analysis
The SCM rotation is already sophisticated:

1. **Dynamic Progression**: Already implemented in `SCMRotationManager`
   - Checks `dynamic_progression_enabled` (set by `early_stopping_enabled`)
   - If enabled, rotates based on convergence
   - If disabled, uses fixed rotation frequency

2. **Early Stopping Config**: Already configured in Cell 2.6:
   - Convergence threshold: 95% accuracy
   - Patience: 5 episodes
   - Min episodes per SCM: 5
   - Max episodes per SCM: 30

### Revised Implementation Plan

Given these findings, the main issues are:

1. **Training Duration**: QUICK mode (96 episodes) is still too short
   - Need to enable FULL mode (960 episodes) or PRECISION (1920 episodes)
   - Document expected training times clearly

2. **Export Error in Phase 2**: Cell 15 fails because Phase 2 results don't have `to_dict()` method
   - Need to handle Phase 2 results differently in export

3. **Missing Visibility**: Users can't see:
   - Which training mode was used
   - How convergence detection worked
   - Structure learning progression

### Updated Steps

1. **Fix Export Error** (Quick fix for Cell 15)
2. **Add Training Mode Visibility** (Make it clear which mode to use)
3. **Add Convergence Monitoring** (Show when SCMs converge)
4. **Add Structure Learning Metrics** (Track F1 progression)
5. **Create Comprehensive Comparison** (Including oracle baseline)