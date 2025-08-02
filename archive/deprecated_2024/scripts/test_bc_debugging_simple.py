#!/usr/bin/env python3
"""
Simple test for BC debugging - load one demonstration file and check data formats.
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

from src.causal_bayes_opt.training.behavioral_cloning_adapter import (
    create_surrogate_training_example,
    convert_trajectory_to_acquisition_steps
)
from src.causal_bayes_opt.training.trajectory_processor import (
    prepare_surrogate_dataset,
    DifficultyLevel
)
from src.causal_bayes_opt.training.bc_surrogate_trainer import (
    BCSurrogateTrainer,
    BCTrainingConfig,
    convert_parent_sets_to_continuous_probs,
    kl_divergence_loss_jax
)
from src.causal_bayes_opt.training.config import SurrogateTrainingConfig

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_data_conversion():
    """Test data conversion and loss calculation with debugging."""
    logger.info("Starting data conversion test")
    
    # Load one demonstration file
    demo_file = Path("expert_demonstrations/raw/raw_demonstrations/batch_1751266609.pkl")
    
    if not demo_file.exists():
        logger.error(f"Demo file not found: {demo_file}")
        return
    
    logger.info(f"Loading demonstrations from {demo_file}")
    
    with open(demo_file, 'rb') as f:
        demos = pickle.load(f)
    
    logger.info(f"Loaded {len(demos)} demonstrations")
    
    # Take first demonstration
    demo = demos[0]
    logger.info(f"First demo type: {type(demo)}")
    logger.info(f"Demo attributes: {dir(demo)}")
    
    # Extract trajectory steps
    trajectory_steps = []
    for step_idx in range(len(demo.trajectory.states) - 1):
        step_data = convert_trajectory_to_acquisition_steps(demo, step_idx)
        if step_data:
            trajectory_steps.append(step_data)
    
    logger.info(f"Created {len(trajectory_steps)} trajectory steps")
    
    # Create surrogate training examples
    training_examples = []
    for step in trajectory_steps:
        # Create AVICI data - [N, d, 3] format
        state = step.state
        n_samples = 100
        n_vars = len(state.node_names)
        
        # Simple synthetic data for testing
        avici_data = jnp.ones((n_samples, n_vars, 3))
        
        example = create_surrogate_training_example(
            demo=demo,
            step=step,
            avici_data=avici_data
        )
        
        if example:
            training_examples.append(example)
    
    logger.info(f"Created {len(training_examples)} training examples")
    
    if not training_examples:
        logger.error("No training examples created")
        return
    
    # Test data conversion on first example
    example = training_examples[0]
    logger.info(f"\nFirst training example:")
    logger.info(f"  Observational data shape: {example.observational_data.shape}")
    logger.info(f"  Number of parent sets: {len(example.parent_sets)}")
    logger.info(f"  Expert probs shape: {example.expert_probs.shape}")
    logger.info(f"  Target variable: {example.target_variable}")
    logger.info(f"  Number of variables: {len(example.variable_order)}")
    
    # Test parent set to continuous conversion
    num_vars = len(example.variable_order)
    target_idx = example.variable_order.index(example.target_variable)
    
    logger.info(f"\nTesting parent set conversion:")
    logger.info(f"  Input parent sets: {example.parent_sets[:3]}...")  # Show first 3
    logger.info(f"  Input probs: {example.expert_probs[:5]}...")  # Show first 5
    
    continuous_probs = convert_parent_sets_to_continuous_probs(
        parent_sets=example.parent_sets,
        probs=example.expert_probs,
        num_variables=num_vars,
        target_idx=target_idx
    )
    
    logger.info(f"  Output continuous probs shape: {continuous_probs.shape}")
    logger.info(f"  Output continuous probs: {continuous_probs}")
    logger.info(f"  Sum of continuous probs: {jnp.sum(continuous_probs)}")
    
    # Test KL loss computation with dummy predictions
    predicted_probs = jnp.ones(num_vars) / (num_vars - 1)  # Uniform over non-target
    predicted_probs = predicted_probs.at[target_idx].set(0.0)
    
    logger.info(f"\nTesting KL loss computation:")
    logger.info(f"  Predicted probs: {predicted_probs}")
    logger.info(f"  Target probs: {continuous_probs}")
    
    kl_loss = kl_divergence_loss_jax(predicted_probs, continuous_probs)
    logger.info(f"  KL divergence: {float(kl_loss)}")
    
    # Test with actual model predictions (if astronomical loss comes from here)
    # Create dataset for training
    dataset = prepare_surrogate_dataset(
        [demo],
        difficulty_level=DifficultyLevel.EASY,
        max_examples_per_demo=10
    )
    
    logger.info(f"\nPrepared dataset with {len(dataset.training_examples)} examples")
    
    # Create small BC trainer config
    surrogate_config = SurrogateTrainingConfig(
        learning_rate=1e-3,
        batch_size=4,
        max_epochs=1,
        use_continuous_model=True
    )
    
    bc_config = BCTrainingConfig(
        surrogate_config=surrogate_config,
        curriculum_learning=False,
        use_jax_compilation=False,  # Disable JAX for simpler debugging
        enable_wandb_logging=False
    )
    
    trainer = BCSurrogateTrainer(bc_config)
    
    # Run one training step manually
    logger.info("\nRunning one training step...")
    
    # Create training batch from first few examples
    batch_examples = dataset.training_examples[:4]
    
    # Compute loss for this batch
    batch_loss = trainer._compute_batch_loss(batch_examples, trainer._initialize_training_state())
    logger.info(f"Batch loss: {float(batch_loss)}")


if __name__ == "__main__":
    test_data_conversion()