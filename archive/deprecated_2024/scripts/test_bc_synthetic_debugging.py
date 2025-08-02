#!/usr/bin/env python3
"""
Test BC loss with synthetic data to isolate the issue.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import jax.random as random

from src.causal_bayes_opt.training.data_structures import TrainingExample
from src.causal_bayes_opt.training.bc_surrogate_trainer import (
    convert_parent_sets_to_continuous_probs,
    kl_divergence_loss_jax,
    convert_to_jax_batch
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_synthetic_example(num_vars=5, num_parent_sets=10, target_idx=0):
    """Create a synthetic training example."""
    # Create observational data [N, d, 3]
    n_samples = 100
    obs_data = jnp.ones((n_samples, num_vars, 3))
    
    # Create parent sets - simple example where each set has 1-2 parents
    parent_sets = []
    for i in range(num_parent_sets):
        # Don't include target in parent sets
        available_parents = [j for j in range(num_vars) if j != target_idx]
        if i < num_vars - 1:
            # Single parent sets
            parent_sets.append(frozenset([available_parents[i % len(available_parents)]]))
        else:
            # Some two-parent sets
            idx1 = i % len(available_parents)
            idx2 = (i + 1) % len(available_parents)
            if idx1 != idx2:
                parent_sets.append(frozenset([available_parents[idx1], available_parents[idx2]]))
            else:
                parent_sets.append(frozenset([available_parents[idx1]]))
    
    # Create probabilities that sum to 1
    raw_probs = jnp.array([1.0 / (i + 1) for i in range(num_parent_sets)])
    expert_probs = raw_probs / jnp.sum(raw_probs)
    
    # Create variable names
    variable_order = [f"X{i}" for i in range(num_vars)]
    target_variable = variable_order[target_idx]
    
    return TrainingExample(
        observational_data=obs_data,
        expert_posterior=None,  # Not used in BC training
        target_variable=target_variable,
        variable_order=variable_order,
        expert_accuracy=0.9,
        problem_difficulty="easy",
        parent_sets=parent_sets,
        expert_probs=expert_probs
    )


def test_synthetic_loss():
    """Test loss calculation with synthetic data."""
    logger.info("Testing BC loss with synthetic data")
    
    # Create synthetic examples
    examples = []
    for i in range(4):
        num_vars = 5 + i  # Varying number of variables
        target_idx = i % num_vars
        example = create_synthetic_example(num_vars, 15, target_idx)
        examples.append(example)
        
        logger.info(f"\nExample {i}:")
        logger.info(f"  Num vars: {num_vars}")
        logger.info(f"  Target idx: {target_idx}")
        logger.info(f"  Num parent sets: {len(example.parent_sets)}")
        logger.info(f"  Expert probs sum: {jnp.sum(example.expert_probs)}")
    
    # Test parent set conversion
    logger.info("\nTesting parent set to continuous conversion:")
    for i, example in enumerate(examples):
        num_vars = len(example.variable_order)
        target_idx = example.variable_order.index(example.target_variable)
        
        continuous_probs = convert_parent_sets_to_continuous_probs(
            parent_sets=example.parent_sets,
            probs=example.expert_probs,
            num_variables=num_vars,
            target_idx=target_idx
        )
        
        logger.info(f"\nExample {i} conversion:")
        logger.info(f"  Continuous probs: {continuous_probs}")
        logger.info(f"  Sum: {jnp.sum(continuous_probs)}")
        logger.info(f"  Target prob: {continuous_probs[target_idx]}")
        
        # Test KL loss
        # Case 1: Uniform prediction
        uniform = jnp.ones(num_vars) / (num_vars - 1)
        uniform = uniform.at[target_idx].set(0.0)
        kl_uniform = kl_divergence_loss_jax(uniform, continuous_probs)
        logger.info(f"  KL (uniform): {float(kl_uniform)}")
        
        # Case 2: Perfect match
        kl_perfect = kl_divergence_loss_jax(continuous_probs, continuous_probs)
        logger.info(f"  KL (perfect match): {float(kl_perfect)}")
        
        # Case 3: Very wrong prediction
        wrong = jnp.zeros(num_vars)
        wrong = wrong.at[(target_idx + 1) % num_vars].set(1.0)
        kl_wrong = kl_divergence_loss_jax(wrong, continuous_probs)
        logger.info(f"  KL (wrong): {float(kl_wrong)}")
    
    # Test batch conversion
    logger.info("\nTesting batch conversion:")
    try:
        jax_batch = convert_to_jax_batch(examples)
        logger.info("Batch conversion successful!")
        logger.info(f"  Obs shape: {jax_batch.observational_data.shape}")
        logger.info(f"  Expert probs shape: {jax_batch.expert_probs.shape}")
        logger.info(f"  Target variables: {jax_batch.target_variables}")
        
        # Check if expert probs are correctly converted
        for i in range(len(examples)):
            logger.info(f"\nBatch example {i}:")
            logger.info(f"  Original expert probs sum: {jnp.sum(examples[i].expert_probs)}")
            logger.info(f"  Batch expert probs: {jax_batch.expert_probs[i]}")
            logger.info(f"  Batch expert probs sum: {jnp.sum(jax_batch.expert_probs[i])}")
            
    except Exception as e:
        logger.error(f"Batch conversion failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_synthetic_loss()