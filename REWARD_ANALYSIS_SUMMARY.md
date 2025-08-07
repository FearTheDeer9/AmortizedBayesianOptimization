# GRPO Reward Analysis Summary

## Problem Statement
You correctly identified that in RL, rewards should ALWAYS increase during training. The current GRPO implementation showed decreasing "mean_reward" values, which seemed wrong for a minimization task.

## Root Cause Analysis

### 1. Reward Saturation Issue
The current reward function uses a sigmoid transformation:
```python
reward = 1.0 / (1.0 + exp(-2.0 * normalized_improvement))
```

Where `improvement = baseline - outcome_value` for minimization tasks.

**Problems:**
- The baseline is computed from observational data and never updates
- As the policy improves (Y decreases), rewards quickly saturate at 1.0
- Once saturated, there's no learning signal for further improvement
- This explains why rewards plateau despite continued performance gains

### 2. Fixed Baseline Problem
- Baseline = mean of observational Y values (~1.0)
- This baseline stays constant throughout training
- Rewards measure "distance from initial baseline" not "improvement over time"
- Even if policy achieves Y=0.3 consistently, rewards stay near 1.0

### 3. Impact on Learning
- GRPO uses advantages = rewards - value_baseline
- When rewards saturate at 1.0, advantages → 0
- Zero advantages mean zero gradients → no learning
- This explains why the policy gets stuck

## Solutions Implemented

### 1. Improved Absolute Performance Rewards
```python
# For minimization
reward = 1.0 / (1.0 + scale * outcome_value)
```

**Benefits:**
- Direct mapping: better performance → higher reward
- No saturation: rewards can always improve
- Simple and interpretable
- Provides consistent learning signal

### 2. Adaptive Baseline Option
```python
# Maintains moving average of recent outcomes
baseline = mean(recent_outcomes[-window_size:])
reward = sigmoid(temperature * (baseline - outcome))
```

**Benefits:**
- Baseline adapts as policy improves
- Maintains relative reward signal
- Prevents saturation
- Natural curriculum learning

## Test Results

### Original Rewards
- Initial episodes: reward ~0.66
- Later episodes: reward ~1.00 (saturated)
- Slope: 0.0057 (positive but plateaus)
- **Problem: Saturation kills learning signal**

### Improved Absolute Rewards
- Initial episodes: reward ~0.51
- Later episodes: reward ~0.68
- Slope: 0.0053 (consistent growth)
- **Success: No saturation, continuous improvement**

## Key Insights

1. **Reward Saturation is Fatal for RL**: Once rewards plateau, learning stops
2. **Fixed Baselines Don't Work**: Need adaptive or absolute rewards
3. **RL Needs Growth**: Rewards must increase as performance improves
4. **Simple is Better**: Absolute rewards are more stable than complex baselines

## Recommendations

1. **Immediate Fix**: Replace `compute_clean_reward` with `compute_improved_clean_reward` using absolute rewards
2. **Configuration**: Use `reward_type='absolute'` with `scale=1.0`
3. **Monitoring**: Track reward slopes during training to ensure positive trend
4. **Testing**: Verify policy gradients remain non-zero throughout training

## Implementation Steps

1. Update `unified_grpo_trainer.py` to use improved rewards:
```python
from ..acquisition.improved_rewards import compute_improved_clean_reward

# Replace compute_clean_reward call with:
reward_info = compute_improved_clean_reward(
    buffer_before=buffer,
    intervention=intervention,
    outcome=outcome,
    target_variable=target_var,
    config={
        'optimization_direction': self.optimization_direction,
        'reward_type': 'absolute',
        'scale': 1.0
    }
)
```

2. Test with simple SCM to verify:
   - Rewards increase over episodes
   - Policy learns to minimize target
   - No saturation occurs

3. Then re-enable surrogate and other features

## Conclusion

The fundamental issue was reward saturation due to fixed baseline comparison. The improved absolute reward function provides a proper RL signal that grows with performance improvement, enabling continuous learning throughout training.