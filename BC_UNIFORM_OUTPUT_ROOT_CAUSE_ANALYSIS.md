# Root Cause Analysis: BC Model Uniform Output Issue

## Summary

The BC surrogate model was producing identical outputs `[0.333, 0.333, 0.333, 0.0]` for all inputs due to a fundamental flaw in the attention mechanism design.

## Root Cause

The issue was in the `ParentAttentionLayer` class in `model.py`:

```python
def __call__(self, query, key_value):
    # PROBLEM: Query is replicated identically for all variables
    query_expanded = jnp.tile(query[None, :], (n_vars, 1))  # [n_vars, hidden_dim]
    
    # This results in identical attention computation for each row
    attended = attention(
        query=query_expanded,    # All rows identical!
        key=key_value,          
        value=key_value         
    )
```

When all queries are identical, the attention mechanism produces identical outputs for each potential parent variable, resulting in uniform probabilities after softmax.

## Why This Happened

The original design attempted to use multi-head attention in a way that doesn't make sense for this task:
- It expanded the target embedding to create multiple identical queries
- Each identical query attended to the same set of keys/values
- This produced identical attention outputs for all variables

## The Fix

The fixed implementation (`fixed_model.py`) uses proper query-key attention:

```python
def __call__(self, query, key_value):
    # Transform query and keys to attention space
    q = query_projection(query)  # [key_size] - single query for target
    k = key_projection(key_value)  # [n_vars, key_size] - keys for each parent
    
    # Compute attention scores properly
    scores = jnp.dot(k, q) / jnp.sqrt(self.key_size)  # [n_vars]
```

This computes different scores for each potential parent based on their relationship to the target.

## Additional Improvements

The fixed model also includes:

1. **Better node encoding**: Preserves variable-specific statistics instead of averaging everything
2. **Proper feature extraction**: Computes meaningful features like mean, variance, intervention rates
3. **No destructive aggregation**: Maintains distinctions between variables throughout

## Verification

Testing confirmed:
- Original model: Always outputs `[0.333, 0.333, 0.333, 0.0]`
- Fixed model: Produces varied outputs based on input patterns

## Impact on BC Training

This explains why BC training appeared to work but produced poor results:
- The model could minimize training loss by always predicting uniform distributions
- But it couldn't actually learn parent relationships
- Any checkpoint from this training would be useless

## Next Steps

1. Replace the flawed architecture with the fixed version
2. Retrain BC models with proper architecture
3. Add validation checks during training to catch uniform outputs early
4. Consider adding tests that verify models produce varied outputs for different inputs