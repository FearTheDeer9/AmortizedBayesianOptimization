#!/usr/bin/env python3
"""
Minimal test to verify BC fix.
"""

import os
import sys
import pickle
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.numpy as jnp
from src.causal_bayes_opt.training.behavioral_cloning_adapter import (
    load_demonstration_batch,
    create_surrogate_training_example
)
from src.causal_bayes_opt.training.bc_surrogate_trainer import (
    convert_parent_sets_to_continuous_probs,
    convert_to_jax_batch,
    kl_divergence_loss_jax
)


def test_minimal():
    """Minimal test of the fix."""
    print("Testing BC fix with minimal example...")
    
    # Load one demo
    demo_file = Path("expert_demonstrations/raw/raw_demonstrations/batch_1751266609.pkl")
    batch = load_demonstration_batch(str(demo_file))
    demo = batch.demonstrations[0]
    
    # Create training example
    avici_data = jnp.ones((100, demo.n_nodes, 3))
    example = create_surrogate_training_example(demo, 0, avici_data)
    
    print(f"\nTraining example:")
    print(f"  Parent sets: {example.parent_sets}")
    print(f"  Variable order: {example.variable_order}")
    print(f"  Target: {example.target_variable}")
    
    # Test conversion
    target_idx = example.variable_order.index(example.target_variable)
    continuous_probs = convert_parent_sets_to_continuous_probs(
        parent_sets=example.parent_sets,
        probs=example.expert_probs,
        num_variables=len(example.variable_order),
        target_idx=target_idx,
        variable_order=example.variable_order
    )
    
    print(f"\nContinuous probabilities:")
    print(f"  Values: {continuous_probs}")
    print(f"  Non-zero indices: {jnp.where(continuous_probs > 0.01)[0]}")
    
    # Expected parents are X0 (index 0), X1 (index 1), X2 (index 4)
    expected_indices = [0, 1, 4]
    actual_indices = [int(i) for i in jnp.where(continuous_probs > 0.01)[0]]
    
    if set(actual_indices) == set(expected_indices):
        print(f"  ✓ Correct parent indices identified!")
    else:
        print(f"  ✗ Wrong parent indices. Expected {expected_indices}, got {actual_indices}")
    
    # Test batch conversion
    jax_batch = convert_to_jax_batch([example])
    batch_probs = jax_batch.expert_probs[0][:len(example.variable_order)]
    
    print(f"\nBatch probabilities:")
    print(f"  Values: {batch_probs}")
    print(f"  Matches continuous probs: {jnp.allclose(batch_probs, continuous_probs)}")
    
    # Test KL loss
    uniform_pred = jnp.ones(len(continuous_probs)) / (len(continuous_probs) - 1)
    uniform_pred = uniform_pred.at[target_idx].set(0.0)
    
    kl_loss = kl_divergence_loss_jax(uniform_pred, continuous_probs)
    print(f"\nKL loss (uniform vs expert): {float(kl_loss)}")
    
    if abs(kl_loss) < 2.0:
        print("✓ Reasonable KL loss - fix is working!")
    else:
        print("✗ High KL loss - something may still be wrong")


if __name__ == "__main__":
    test_minimal()