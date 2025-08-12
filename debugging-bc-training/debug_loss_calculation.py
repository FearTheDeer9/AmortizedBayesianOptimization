#!/usr/bin/env python3
"""
Debug the loss calculation to understand why it's so high.
"""

import numpy as np
import jax
import jax.numpy as jnp

def debug_loss_calculation():
    """Simulate the loss calculation to understand the high values."""
    
    print("="*80)
    print("LOSS CALCULATION DEBUGGING")
    print("="*80)
    
    # Simulate typical values
    n_vars = 5
    
    # Variable selection loss
    print("\n1. VARIABLE SELECTION LOSS:")
    print("-" * 40)
    
    # Case 1: Random uniform predictions
    var_logits_uniform = jnp.zeros(n_vars)  # Uniform logits
    var_probs_uniform = jax.nn.softmax(var_logits_uniform)
    print(f"Uniform logits: {var_logits_uniform}")
    print(f"Softmax probs: {var_probs_uniform}")
    
    # Loss for each possible target
    for target_idx in range(n_vars):
        var_loss = -jnp.log(var_probs_uniform[target_idx])
        print(f"  Target {target_idx}: loss = {var_loss:.3f}")
    
    print(f"\nExpected loss (uniform): {-jnp.log(1/n_vars):.3f}")
    
    # Case 2: Very confident wrong prediction
    var_logits_confident = jnp.array([-10, 10, -10, -10, -10])  # Very confident in index 1
    var_probs_confident = jax.nn.softmax(var_logits_confident)
    print(f"\nConfident wrong logits: {var_logits_confident}")
    print(f"Softmax probs: {var_probs_confident}")
    
    for target_idx in range(n_vars):
        var_loss = -jnp.log(jnp.maximum(var_probs_confident[target_idx], 1e-10))
        print(f"  Target {target_idx}: loss = {var_loss:.3f}")
    
    print("\n2. VALUE PREDICTION LOSS:")
    print("-" * 40)
    
    # Typical value prediction scenario
    target_value = 0.5
    value_mean = 0.0  # Model prediction
    value_log_std = 0.0  # log(1) = 0
    value_std = jnp.exp(value_log_std)
    
    value_loss = 0.5 * jnp.log(2 * jnp.pi) + value_log_std + \
                0.5 * ((target_value - value_mean) / value_std) ** 2
    
    print(f"Target value: {target_value}")
    print(f"Predicted mean: {value_mean}")
    print(f"Predicted std: {value_std}")
    print(f"Value loss: {value_loss:.3f}")
    print(f"  - Constant term: {0.5 * jnp.log(2 * jnp.pi):.3f}")
    print(f"  - Log std term: {value_log_std:.3f}")
    print(f"  - MSE term: {0.5 * ((target_value - value_mean) / value_std) ** 2:.3f}")
    
    print("\n3. COMBINED LOSS:")
    print("-" * 40)
    
    # Total loss as computed in the trainer
    var_loss_typical = -jnp.log(1/n_vars)  # Random guess
    total_loss = var_loss_typical + 0.5 * value_loss
    
    print(f"Variable loss: {var_loss_typical:.3f}")
    print(f"Value loss (weighted): {0.5 * value_loss:.3f}")
    print(f"Total loss: {total_loss:.3f}")
    
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    print("""
For a 5-variable problem:
- Variable selection loss (random): 1.61
- Value prediction loss (typical): 0.92 + MSE
- Total expected loss: ~2.5-3.0

But we're seeing loss of 4.8, which means either:
1. The model is VERY confident in wrong variable predictions (loss ~10-20)
2. The value prediction is very bad (high MSE)
3. There's numerical instability (log of very small probabilities)

Most likely: The model outputs very confident WRONG predictions,
possibly due to the variable index mismatch we found.
""")
    
    # Check what loss we'd get with the sorting issue
    print("\n4. SORTING ISSUE IMPACT:")
    print("-" * 40)
    
    print("""
If the model learned:
- "X3 is at index 3" from 5-var SCMs
- "X3 is at index 5" from 12-var SCMs

Then when it sees a 12-var SCM and outputs index 3 (thinking it's X3),
but the label says index 5 is correct:
- The model is confident in index 3
- The loss is -log(softmax[5]) where softmax is peaked at 3
- This could easily give loss > 10!
""")

if __name__ == "__main__":
    debug_loss_calculation()