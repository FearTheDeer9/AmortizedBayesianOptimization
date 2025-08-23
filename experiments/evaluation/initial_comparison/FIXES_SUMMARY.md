# Evaluation Fixes Summary

## Issues Fixed

### 1. Value Explosion in Large SCMs
**Problem**: Target values in 100-variable SCMs reached enormous values (millions/billions) due to unbounded linear mechanism outputs cascading through deep graphs.

**Solution**: Implemented two-layer protection:
- Activated the unused `clip_intervention_values` function in `interventions/handlers.py`
- Added `output_bounds` parameter to linear mechanisms with sqrt-scaled bounds based on graph size
- Modified `variable_scm_factory.py` to use bounds: `bound_scale = 10.0 * sqrt(num_variables / 5.0)`

### 2. Architecture Deprecation Issue  
**Problem**: "permutation_invariant" was deprecated in favor of "simple_permutation_invariant" but this wasn't propagated through the codebase.

**Solution**: Made principled refactoring:
- Updated `clean_policy_factory.py` to map "permutation_invariant" → "simple_permutation_invariant"
- Fixed checkpoint detection in `ModelLoader.detect_checkpoint_architecture()` to apply the same mapping
- Updated untrained policy creation to match checkpoint settings

### 3. use_fixed_std Detection Bug
**Problem**: Detection logic defaulted to False instead of True, which is the standard in training.

**Solution**: 
- Fixed detection in `ModelLoader.detect_checkpoint_architecture()` to default to `use_fixed_std=True`
- Only set to False if we find evidence of learned std (shape [*, 2] in val_mlp_output)

### 4. Scalar Conversion Error
**Problem**: With `use_fixed_std=True`, the policy output shape differed, causing "Only scalar arrays can be converted to Python scalars" error.

**Solution**:
- Added shape handling in `ModelLoader._create_acquisition_fn()` to properly extract scalars
- Added `.item()` conversion for array values before returning

## Files Modified

1. **src/causal_bayes_opt/interventions/handlers.py**
   - Activated intervention value clipping

2. **src/causal_bayes_opt/mechanisms/linear.py**
   - Added output_bounds parameter support

3. **src/causal_bayes_opt/experiments/variable_scm_factory.py**
   - Added sqrt-scaled bounds calculation

4. **src/causal_bayes_opt/policies/clean_policy_factory.py**
   - Made permutation_invariant an alias for simple version

5. **experiments/evaluation/core/model_loader.py**
   - Fixed checkpoint architecture detection
   - Fixed use_fixed_std detection logic
   - Added scalar conversion handling

## Verification

The evaluation now runs successfully with:
- No value explosions in large SCMs
- Proper loading of trained policies with 4-channel architecture
- Correct detection of use_fixed_std=True
- Successful completion of all experiments

Test command:
```bash
python scripts/run_experiment.py \
  --config configs/test_with_trained.yaml \
  --policy-checkpoint /path/to/checkpoint/policy.pkl \
  --output-dir results/fixed_comparison_final
```