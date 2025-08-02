#!/usr/bin/env python3
"""
Direct test of BC loss debugging using notebook's data loading approach.
"""

import os
import sys
import logging
import pickle
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import jax.random as random

# Import minimal required modules
from src.causal_bayes_opt.training.bc_data_pipeline import process_all_demonstrations
from src.causal_bayes_opt.training.trajectory_processor import DifficultyLevel
from src.causal_bayes_opt.training.data_structures import TrainingExample

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_loss_calculation():
    """Test loss calculation with actual data from notebook."""
    logger.info("Testing BC loss calculation with debugging")
    
    # Load demonstrations using the same approach as notebook
    demo_dir = Path("expert_demonstrations/raw/raw_demonstrations")
    
    # Process demonstrations
    logger.info("Processing demonstrations...")
    processed = process_all_demonstrations(str(demo_dir), max_examples_per_demo=10)
    
    # Get easy dataset
    if DifficultyLevel.EASY in processed.surrogate_datasets:
        dataset = processed.surrogate_datasets[DifficultyLevel.EASY]
    else:
        dataset = next(iter(processed.surrogate_datasets.values()))
    
    logger.info(f"Dataset has {len(dataset.training_examples)} examples")
    
    # Examine first few training examples
    for i, example in enumerate(dataset.training_examples[:3]):
        logger.info(f"\nExample {i}:")
        logger.info(f"  Obs data shape: {example.observational_data.shape}")
        logger.info(f"  Parent sets: {len(example.parent_sets)}")
        logger.info(f"  Expert probs shape: {example.expert_probs.shape}")
        logger.info(f"  Expert probs sum: {jnp.sum(example.expert_probs)}")
        logger.info(f"  Expert probs min/max: {jnp.min(example.expert_probs)}/{jnp.max(example.expert_probs)}")
        logger.info(f"  Target variable: {example.target_variable}")
        logger.info(f"  Variable order: {example.variable_order}")
        
        # Check if expert probs are valid
        if not jnp.allclose(jnp.sum(example.expert_probs), 1.0, atol=1e-6):
            logger.warning(f"  WARNING: Expert probs don't sum to 1.0!")
        
        # Check parent sets
        logger.info(f"  First 3 parent sets: {example.parent_sets[:3]}")
        logger.info(f"  First 3 probs: {example.expert_probs[:3]}")
    
    # Test conversion to continuous probabilities
    from src.causal_bayes_opt.training.bc_surrogate_trainer import (
        convert_parent_sets_to_continuous_probs,
        kl_divergence_loss_jax
    )
    
    logger.info("\nTesting parent set to continuous conversion:")
    example = dataset.training_examples[0]
    num_vars = len(example.variable_order)
    target_idx = example.variable_order.index(example.target_variable)
    
    continuous_probs = convert_parent_sets_to_continuous_probs(
        parent_sets=example.parent_sets,
        probs=example.expert_probs,
        num_variables=num_vars,
        target_idx=target_idx
    )
    
    logger.info(f"Continuous probs shape: {continuous_probs.shape}")
    logger.info(f"Continuous probs: {continuous_probs}")
    logger.info(f"Sum: {jnp.sum(continuous_probs)}")
    logger.info(f"Target prob (should be 0): {continuous_probs[target_idx]}")
    
    # Test KL loss with uniform predictions
    uniform_probs = jnp.ones(num_vars) / (num_vars - 1)
    uniform_probs = uniform_probs.at[target_idx].set(0.0)
    
    kl_loss = kl_divergence_loss_jax(uniform_probs, continuous_probs)
    logger.info(f"\nKL loss (uniform vs expert): {float(kl_loss)}")
    
    # Test with very wrong predictions
    wrong_probs = jnp.zeros(num_vars)
    wrong_probs = wrong_probs.at[0].set(1.0)  # All prob on first variable
    if target_idx == 0:
        wrong_probs = wrong_probs.at[1].set(1.0)
        wrong_probs = wrong_probs.at[0].set(0.0)
    
    kl_loss_wrong = kl_divergence_loss_jax(wrong_probs, continuous_probs)
    logger.info(f"KL loss (wrong vs expert): {float(kl_loss_wrong)}")
    
    # Test batch conversion
    from src.causal_bayes_opt.training.bc_surrogate_trainer import convert_to_jax_batch
    
    logger.info("\nTesting batch conversion:")
    batch_examples = dataset.training_examples[:4]
    
    try:
        jax_batch = convert_to_jax_batch(batch_examples)
        logger.info(f"JAX batch created successfully")
        logger.info(f"  Obs data shape: {jax_batch.observational_data.shape}")
        logger.info(f"  Expert probs shape: {jax_batch.expert_probs.shape}")
        logger.info(f"  Target variables: {jax_batch.target_variables}")
    except Exception as e:
        logger.error(f"Batch conversion failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_loss_calculation()