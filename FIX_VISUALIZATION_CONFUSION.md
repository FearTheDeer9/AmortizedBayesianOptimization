# Fix for Visualization Confusion

## The Problem

The current visualization is confusing because:

1. **Data**: We calculate `reduction = initial_mean - intervention_mean`
   - Positive reduction = good (target value decreased)
   - Negative reduction = bad (target value increased)

2. **Plot**: Shows `actual_value = -reduction` with label "Target Value (Lower is Better)"
   - This creates a double negative
   - Higher bars show better performance but label says "lower is better"
   - Users think they're seeing target values but they're seeing negated reductions

## Solution Options

### Option 1: Plot Actual Target Values (Recommended)
```python
# Instead of plotting reductions, plot actual final target values
values = [v['final_target_value'] for v in checkpoint_methods.values()]

# Label appropriately
if opt_config.is_minimizing:
    ax1.set_ylabel('Final Target Value (Lower is Better)')
    # Don't invert y-axis - let lower values naturally appear lower
else:
    ax1.set_ylabel('Final Target Value (Higher is Better)')
```

### Option 2: Plot Reductions with Clear Labels
```python
# Keep plotting reductions but fix the label
values = [v['reduction'] for v in checkpoint_methods.values()]

# Label clearly
ax1.set_ylabel('Target Reduction (Higher is Better)')
# Don't invert y-axis - higher reductions are better
```

### Option 3: Plot Percentage Improvement
```python
# Calculate percentage improvement from baseline
baseline = initial_mean
improvements = [(baseline - v['final_value']) / abs(baseline) * 100 
                for v in checkpoint_methods.values()]

ax1.set_ylabel('Improvement from Baseline (%)')
# Always positive scale, higher is always better
```

## Implementation Fix

The fix should be applied in:
1. `experiments/grpo_evaluation_modular.ipynb` - Cell 6 visualization
2. `scripts/core/acbo_comparison/visualization.py` - For unified pipeline plots
3. `scripts/unified_pipeline.py` - Summary plot generation

## Key Principle

**Be explicit about what you're plotting:**
- If plotting target values → "Target Value"
- If plotting reductions → "Target Reduction" or "Improvement"
- If plotting percentages → "% Improvement"

Never mix concepts like plotting reductions but labeling as target values!