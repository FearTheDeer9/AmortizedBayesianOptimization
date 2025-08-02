#!/usr/bin/env python3
"""
Verify that the BC training fix resolves astronomical losses.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import jax.random as random

from src.causal_bayes_opt.training.bc_data_pipeline import process_all_demonstrations
from src.causal_bayes_opt.training.bc_surrogate_trainer import (
    BCSurrogateTrainer,
    BCTrainingConfig,
    convert_to_jax_batch,
    kl_divergence_loss_jax
)
from src.causal_bayes_opt.training.config import SurrogateTrainingConfig
from src.causal_bayes_opt.training.trajectory_processor import DifficultyLevel


def verify_fix():
    """Verify the BC fix works correctly."""
    print("Verifying BC training fix...")
    
    # Load and process demonstrations
    demo_dir = Path("expert_demonstrations/raw/raw_demonstrations")
    processed = process_all_demonstrations(str(demo_dir), max_examples_per_demo=5)
    
    # Get easy dataset
    if DifficultyLevel.EASY in processed.surrogate_datasets:
        dataset = processed.surrogate_datasets[DifficultyLevel.EASY]
    else:
        dataset = next(iter(processed.surrogate_datasets.values()))
    
    print(f"\nDataset has {len(dataset.training_examples)} examples")
    
    # Take first batch of examples
    batch_size = 4
    batch_examples = dataset.training_examples[:batch_size]
    
    # Convert to JAX batch
    print("\nConverting to JAX batch...")
    jax_batch = convert_to_jax_batch(batch_examples)
    
    print(f"JAX batch created:")
    print(f"  Obs data shape: {jax_batch.observational_data.shape}")
    print(f"  Expert probs shape: {jax_batch.expert_probs.shape}")
    
    # Check expert probabilities
    print("\nExpert probabilities in batch:")
    for i in range(min(3, batch_size)):
        example = batch_examples[i]
        batch_probs = jax_batch.expert_probs[i]
        num_vars = len(example.variable_order)
        
        print(f"\nExample {i}:")
        print(f"  Original parent sets: {example.parent_sets[:2]}...")
        print(f"  Variable order: {example.variable_order}")
        print(f"  Target: {example.target_variable}")
        print(f"  Batch probs: {batch_probs[:num_vars]}")
        print(f"  Sum: {jnp.sum(batch_probs[:num_vars])}")
        
        # Check if probabilities are reasonable (not uniform)
        non_zero_probs = batch_probs[batch_probs > 0.01]
        if len(non_zero_probs) > 0 and len(non_zero_probs) < num_vars - 1:
            print(f"  ✓ Probabilities are concentrated (good!)")
        else:
            print(f"  ✗ Probabilities look uniform (bad!)")
    
    # Test KL loss computation
    print("\n\nTesting KL loss computation:")
    
    # Simulate model predictions (uniform)
    for i in range(min(3, batch_size)):
        num_vars = len(batch_examples[i].variable_order)
        target_idx = batch_examples[i].variable_order.index(batch_examples[i].target_variable)
        
        # Create uniform prediction
        uniform_pred = jnp.ones(num_vars) / (num_vars - 1)
        uniform_pred = uniform_pred.at[target_idx].set(0.0)
        
        # Get expert probs for this example
        expert_probs = jax_batch.expert_probs[i][:num_vars]
        
        # Compute KL loss
        kl_loss = kl_divergence_loss_jax(uniform_pred, expert_probs)
        
        print(f"\nExample {i}:")
        print(f"  Expert probs: {expert_probs}")
        print(f"  Uniform pred: {uniform_pred}")
        print(f"  KL loss: {float(kl_loss)}")
        
        if abs(kl_loss) < 10:
            print(f"  ✓ Reasonable KL loss")
        else:
            print(f"  ✗ High KL loss!")
    
    print("\n\nSummary:")
    print("If you see concentrated probabilities and reasonable KL losses, the fix is working!")


if __name__ == "__main__":
    verify_fix()