#!/usr/bin/env python3
"""
Debug variable selection head masking logic.

The isolation test revealed that variable logits are producing extreme negative values 
around -1 billion. This focused test examines the variable selection head masking.
"""

import jax
import jax.numpy as jnp
import haiku as hk
import sys
import os
sys.path.append('/Users/harellidar/Documents/Imperial/Individual_Project/causal_bayes_opt')

from src.causal_bayes_opt.acquisition.enriched.policy_heads import SimplifiedPolicyHeads


def test_variable_masking():
    """Test the variable selection masking logic."""
    print("🔍 DEBUGGING VARIABLE SELECTION MASKING")
    print("=" * 50)
    
    def policy_heads_fn(variable_embeddings, target_idx, dropout_rate):
        heads = SimplifiedPolicyHeads(hidden_dim=128, dropout=0.0)
        return heads._variable_selection_head(variable_embeddings, target_idx, dropout_rate)
    
    init_fn = hk.transform(policy_heads_fn)
    apply_fn = init_fn.apply
    
    # Test with different target indices
    n_vars = 4
    dummy_embeddings = jnp.ones((n_vars, 128)) * 0.1  # Small positive values
    
    key = jax.random.PRNGKey(42)
    params = init_fn.init(key, dummy_embeddings, 1, 0.0)
    
    print(f"Testing with {n_vars} variables, embeddings mean: {jnp.mean(dummy_embeddings):.6f}")
    
    for target_idx in range(n_vars):
        print(f"\n--- Target index: {target_idx} ---")
        
        # Get masked outputs
        masked_logits = apply_fn(params, key, dummy_embeddings, target_idx, 0.0)
        
        print(f"  Logits: {masked_logits}")
        print(f"  Range: [{jnp.min(masked_logits):.6f}, {jnp.max(masked_logits):.6f}]")
        
        # Check which variables have extreme values (likely the masked ones)
        extreme_mask = jnp.abs(masked_logits) > 1e6
        print(f"  Variables with extreme values: {extreme_mask}")
        print(f"  Expected target {target_idx} to be extreme: {extreme_mask[target_idx] if target_idx < len(extreme_mask) else 'INDEX OUT OF RANGE'}")
        
        # Check if ALL variables are getting extreme values (this would be the bug)
        extreme_count = jnp.sum(extreme_mask)
        print(f"  Variables with extreme values (>1e6): {extreme_count}/{n_vars}")
        
        if extreme_count == n_vars:
            print("  ❌ BUG FOUND: ALL variables have extreme values!")
        elif extreme_count == 1 and extreme_mask[target_idx]:
            print("  ✅ Masking looks correct: only target has extreme value")
        else:
            print("  ⚠️  Unexpected masking pattern")
        
        # Softmax test
        try:
            probs = jax.nn.softmax(masked_logits)
            print(f"  Softmax probabilities: {probs}")
            print(f"  Softmax sum: {jnp.sum(probs):.6f}")
            print(f"  Max probability: {jnp.max(probs):.6f}")
        except Exception as e:
            print(f"  Softmax failed: {e}")


def test_better_masking():
    """Test a better masking approach."""
    print("\n🔧 TESTING IMPROVED MASKING APPROACH")
    print("=" * 50)
    
    def improved_variable_selection(variable_embeddings, target_idx, dropout_rate):
        """Improved variable selection with safer masking."""
        n_vars = variable_embeddings.shape[0]
        
        # Same MLP as before
        x = variable_embeddings
        x = hk.Linear(64, name="var_select_linear1")(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="var_select_norm1")(x)
        x = jax.nn.relu(x)
        
        x = hk.Linear(64, name="var_select_linear2")(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="var_select_norm2")(x)
        x = jax.nn.relu(x)
        
        # Output layer with better initialization
        variable_logits = hk.Linear(1, name="var_select_output")(x).squeeze(-1)
        
        # BETTER MASKING: Use much smaller negative value and ensure only target is masked
        target_mask = jnp.arange(n_vars) == target_idx
        
        # Use -10 instead of -1e9 for more stable numerical computation
        masked_logits = jnp.where(target_mask, -10.0, variable_logits)
        
        return masked_logits
    
    init_fn = hk.transform(improved_variable_selection)
    apply_fn = init_fn.apply
    
    n_vars = 4
    dummy_embeddings = jnp.ones((n_vars, 128)) * 0.1
    key = jax.random.PRNGKey(123)
    params = init_fn.init(key, dummy_embeddings, 1, 0.0)
    
    print(f"Testing improved masking with {n_vars} variables")
    
    for target_idx in range(n_vars):
        print(f"\n--- Target index: {target_idx} ---")
        
        logits = apply_fn(params, key, dummy_embeddings, target_idx, 0.0)
        probs = jax.nn.softmax(logits)
        
        print(f"  Logits: {logits}")
        print(f"  Probabilities: {probs}")
        print(f"  Target probability: {probs[target_idx]:.6f} (should be ~0)")
        print(f"  Max non-target probability: {jnp.max(probs[jnp.arange(n_vars) != target_idx]):.6f}")
        print(f"  Probability sum: {jnp.sum(probs):.6f}")


if __name__ == "__main__":
    test_variable_masking()
    test_better_masking()