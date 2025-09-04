# Single-SCM Learning Analysis

## Purpose

This analysis framework demonstrates that our GRPO system CAN learn optimal policies when trained on a single, fixed SCM. This helps us:

1. **Prove the learning algorithm works** - Show convergence to optimal interventions
2. **Isolate the generalization problem** - Separate learning from transfer issues
3. **Test reward structures** - Verify that our reward design drives correct behavior
4. **Analyze trajectories** - Understand how the policy improves over time

## Usage

### Basic Single-SCM Training

```bash
# Train on a fork structure with 4 variables for 100 interventions
python experiments/policy-only-training/train_grpo_single_scm_with_surrogate.py \
    --interventions 100 \
    --scm-type fork \
    --num-vars 4 \
    --verbose
```

### Automated Analysis

```bash
# Run training and analyze the learning trajectory
python experiments/policy-only-training/analyze_single_scm_learning.py \
    --interventions 50 \
    --scm-type fork \
    --num-vars 4
```

## Expected Behavior

### For Fork Structure
- **Target**: Usually the sink node (e.g., X3)
- **Parents**: All other nodes feed into target
- **Optimal Policy**: Intervene on the parent with largest coefficient
- **Expected Convergence**: Should select optimal parent >80% after ~30 interventions

### For Chain Structure  
- **Target**: End of the chain
- **Parents**: Single parent (previous node in chain)
- **Optimal Policy**: Intervene on the single parent
- **Expected Convergence**: Rapid convergence (<20 interventions)

### For Collider Structure
- **Target**: Central node with multiple parents
- **Parents**: Multiple independent parents
- **Optimal Policy**: Parent with strongest effect
- **Expected Convergence**: Similar to fork (~30 interventions)

## Key Metrics

1. **Convergence Rate**: Percentage of optimal selections over time
2. **Final Performance**: Optimal selection rate in last 10 interventions
3. **Reward Progression**: Improvement from first to last quarter
4. **Selection Probability**: Confidence in optimal action selection

## Interpretation

### Success Indicators
- ✅ >80% optimal selections in final interventions
- ✅ Monotonically increasing optimal selection rate
- ✅ High selection probability (>0.9) for optimal action
- ✅ Positive reward improvement trend

### Failure Modes
- ❌ Random selection pattern (25% optimal for 4 variables)
- ❌ Oscillating between suboptimal actions
- ❌ Declining performance over time
- ❌ Low confidence in all actions

## Example Analysis Output

```
LEARNING TRAJECTORY ANALYSIS
============================================================

SCM STRUCTURE:
  Target: X3
  Parents: ['X0', 'X1', 'X2']
  Variables: ['X0', 'X1', 'X2', 'X3']

CONVERGENCE METRICS:
  Total interventions: 50
  Interventions 1-10: 3/10 optimal (30.0%)
  Interventions 11-20: 5/10 optimal (50.0%)
  Interventions 21-30: 7/10 optimal (70.0%)
  Interventions 31-40: 8/10 optimal (80.0%)
  Interventions 41-50: 9/10 optimal (90.0%)

  FINAL PERFORMANCE (last 10 interventions):
    Optimal selections: 9/10 (90.0%)
    ✅ CONVERGED to optimal policy!

REWARD PROGRESSION:
  Mean reward: 0.623
  Std reward: 0.287
  First quarter mean: 0.412
  Last quarter mean: 0.834
  Improvement: +0.422
```

## Debugging Tips

1. **If not converging**: 
   - Check reward weights are sensible
   - Verify SCM has clear optimal parent
   - Increase learning rate slightly
   - Check if entropy coefficient is too high

2. **If oscillating**:
   - Learning rate may be too high
   - GRPO clip ratio might be too restrictive
   - Check advantage normalization

3. **If random behavior persists**:
   - Entropy coefficient might be too high
   - Policy might be stuck in local minimum
   - Try different initialization seed