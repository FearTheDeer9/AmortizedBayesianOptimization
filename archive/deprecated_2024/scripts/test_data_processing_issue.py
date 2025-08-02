#!/usr/bin/env python3
"""
Test data processing to find where astronomical losses come from.
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
    create_surrogate_training_example,
    extract_posterior_history
)
from src.causal_bayes_opt.training.bc_surrogate_trainer import (
    convert_parent_sets_to_continuous_probs,
    convert_to_jax_batch
)


def test_data_processing():
    """Test the full data processing pipeline."""
    print("Testing data processing pipeline for BC training")
    
    # Load one demonstration batch
    demo_file = Path("expert_demonstrations/raw/raw_demonstrations/batch_1751266609.pkl")
    batch = load_demonstration_batch(str(demo_file))
    demo = batch.demonstrations[0]
    
    print(f"\nDemo info:")
    print(f"  Target variable: {demo.target_variable}")
    print(f"  Number of nodes: {demo.n_nodes}")
    print(f"  Graph type: {demo.graph_type}")
    
    # Check parent posterior
    print(f"\nParent posterior:")
    if isinstance(demo.parent_posterior, dict) and 'posterior_distribution' in demo.parent_posterior:
        post_dist = demo.parent_posterior['posterior_distribution']
        print(f"  Distribution: {post_dist}")
        
        # Extract parent sets and probabilities
        parent_sets = list(post_dist.keys())
        probs = list(post_dist.values())
        print(f"  Number of parent sets: {len(parent_sets)}")
        print(f"  Probabilities: {probs}")
        print(f"  Sum of probabilities: {sum(probs)}")
    
    # Create training example
    print(f"\nCreating training example...")
    
    # Create dummy AVICI data
    avici_data = jnp.ones((100, demo.n_nodes, 3))
    
    # Get posterior history to create training example
    posterior_history = extract_posterior_history(demo)
    print(f"  Posterior history length: {len(posterior_history)}")
    
    if posterior_history:
        print(f"  First posterior: {posterior_history[0]}")
    
    # Create training example
    training_example = create_surrogate_training_example(demo, 0, avici_data)
    
    print(f"\nTraining example created:")
    print(f"  Parent sets: {training_example.parent_sets}")
    print(f"  Expert probs: {training_example.expert_probs}")
    print(f"  Expert probs sum: {jnp.sum(training_example.expert_probs)}")
    print(f"  Variable order: {training_example.variable_order}")
    print(f"  Target variable: {training_example.target_variable}")
    
    # Test parent set to continuous conversion
    print(f"\nTesting parent set to continuous conversion:")
    target_idx = training_example.variable_order.index(training_example.target_variable)
    
    continuous_probs = convert_parent_sets_to_continuous_probs(
        parent_sets=training_example.parent_sets,
        probs=training_example.expert_probs,
        num_variables=len(training_example.variable_order),
        target_idx=target_idx
    )
    
    print(f"  Continuous probs: {continuous_probs}")
    print(f"  Sum: {jnp.sum(continuous_probs)}")
    print(f"  Target prob (should be 0): {continuous_probs[target_idx]}")
    
    # Create a batch
    print(f"\nCreating JAX batch...")
    try:
        jax_batch = convert_to_jax_batch([training_example])
        print(f"  Batch created successfully!")
        print(f"  Expert probs shape in batch: {jax_batch.expert_probs.shape}")
        print(f"  Expert probs in batch: {jax_batch.expert_probs[0]}")
        
        # Check if the conversion is correct
        if not jnp.array_equal(jax_batch.expert_probs[0][:len(continuous_probs)], continuous_probs):
            print(f"  WARNING: Batch expert probs don't match continuous probs!")
            print(f"  Expected: {continuous_probs}")
            print(f"  Got: {jax_batch.expert_probs[0][:len(continuous_probs)]}")
            
    except Exception as e:
        print(f"  Batch creation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Check for potential issues
    print(f"\n\nPotential issues:")
    
    # Issue 1: Single parent set with prob 1.0
    if len(training_example.parent_sets) == 1 and training_example.expert_probs[0] == 1.0:
        print("  - Expert has 100% confidence in single parent set")
        print("  - This creates a very specific continuous distribution")
        
    # Issue 2: Parent set has many parents
    max_parents = max(len(ps) for ps in training_example.parent_sets)
    if max_parents > 3:
        print(f"  - Large parent sets (max {max_parents} parents)")
        print("  - Probability gets spread thin in continuous conversion")
    
    # Issue 3: Check if continuous probs are very skewed
    if jnp.max(continuous_probs) > 0.9:
        print(f"  - Highly skewed continuous distribution (max prob = {jnp.max(continuous_probs)})")
    
    # Issue 4: Check if model would predict uniform
    uniform_pred = jnp.ones(len(continuous_probs)) / (len(continuous_probs) - 1)
    uniform_pred = uniform_pred.at[target_idx].set(0.0)
    
    from src.causal_bayes_opt.training.bc_surrogate_trainer import kl_divergence_loss_jax
    kl_uniform = kl_divergence_loss_jax(uniform_pred, continuous_probs)
    print(f"\n  - KL divergence (uniform prediction): {float(kl_uniform)}")
    
    if abs(kl_uniform) > 10:
        print("  - High KL divergence expected even for reasonable predictions!")


if __name__ == "__main__":
    test_data_processing()