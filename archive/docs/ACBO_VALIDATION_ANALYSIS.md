# ACBO Validation Results Analysis

## Executive Summary

The validation results reveal several important patterns that require investigation:

1. **Random baseline performs surprisingly well** (mean improvement: 3.578)
2. **All methods show identical F1=0.000 and SHD=1.500** (suggesting no structure learning)
3. **BC surrogate and active learning produce identical results** (suggesting surrogate not being used)
4. **GRPO performs worse than random** (2.612 vs 3.578 improvement)

## Detailed Analysis

### 1. Random Baseline Performance

The random baseline achieving mean improvement of 3.578 is actually reasonable given:
- Initial target values are positive (0.057 to 0.944)
- Random interventions with N(0,1) values can easily produce negative outcomes
- Chain_5 SCM shows extreme improvement (10.010) which skews the mean
- This suggests the SCMs have high sensitivity to interventions

**Conclusion**: The -3.492 improvement mentioned by the user is consistent with these results.

### 2. Zero F1 Scores Across All Methods

All methods show F1=0.000, which indicates:
- No structure learning is happening
- The surrogate model is either:
  - Not being trained on graph supervision
  - Not updating its parameters
  - Returning zero probabilities for all edges

**Key Evidence**:
- grpo_unified+bc_surrogate has identical results to grpo_unified+active_learning
- This confirms the surrogate predictions are not being used effectively

### 3. GRPO Underperformance

GRPO (2.612) performs worse than random (3.578), which suggests:
- The policy might be early in training (only 100 episodes)
- The 5-channel integration might not be working correctly
- The policy might be avoiding interventions on high-impact variables

### 4. Oracle Performance

Oracle (4.496) only slightly outperforms random (3.578), which is suspicious because:
- Oracle knows the true causal structure
- It should significantly outperform random selection
- This suggests the oracle implementation might not be optimal

## Root Causes Identified

1. **Surrogate Not Learning Structure**:
   - F1=0.000 means no edges are being predicted
   - The surrogate likely needs graph supervision during training
   - 100 episodes is too few for meaningful structure learning

2. **5-Channel Integration Issues**:
   - The surrogate predictions (channel 3) might be all zeros
   - This would explain why bc_surrogate and active_learning are identical

3. **Training Duration**:
   - 100 episodes is minimal for both policy and surrogate learning
   - Need 1000+ episodes for meaningful results

## Recommendations

1. **Immediate Actions**:
   - Check if surrogate training includes graph supervision
   - Log the actual values in the 5-channel tensors
   - Verify surrogate parameters are updating during training

2. **Training Improvements**:
   - Increase training to 1000+ episodes
   - Add logging for surrogate predictions during evaluation
   - Monitor F1 scores during surrogate training

3. **Debugging Steps**:
   - Run a simple 2-variable SCM test
   - Log raw posterior probabilities from surrogate
   - Check if surrogate loss is decreasing during training

## Code to Run Next

```bash
# 1. Check detailed results with more logging
poetry run python scripts/diagnose_acbo_results.py

# 2. Train with more episodes
poetry run python scripts/train_acbo_methods.py \
    --method surrogate \
    --episodes 1000 \
    --checkpoint_dir checkpoints/debug

# 3. Evaluate with verbose logging
LOGGING_LEVEL=DEBUG poetry run python scripts/evaluate_acbo_methods.py \
    --grpo checkpoints/validation/unified_grpo_final \
    --bc checkpoints/validation/bc_final \
    --surrogate checkpoints/validation/bc_surrogate_final \
    --n_scms 2 \
    --n_interventions 5
```

## Conclusion

The results show that while the 5-channel tensor integration is working mechanically (no crashes), the surrogate is not providing meaningful structure predictions. This is likely due to:
1. Insufficient training (100 episodes)
2. Missing graph supervision in surrogate training
3. Possible frozen or stub surrogate implementation

The random baseline performing well is actually expected given the SCM characteristics, not a bug.