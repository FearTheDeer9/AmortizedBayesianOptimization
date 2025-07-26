#!/usr/bin/env python3
"""
Test script to diagnose BC acquisition trainer loss issues.

This script runs a few training steps with comprehensive logging to understand
why we're getting zero gradients.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
import jax
import jax.random as random
from src.causal_bayes_opt.training.bc_acquisition_trainer import create_bc_acquisition_trainer
from src.causal_bayes_opt.training.bc_data_pipeline import process_all_demonstrations

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting BC loss diagnostics test...")
    
    # Process demonstration data
    demo_dir = Path(__file__).parent.parent / "expert_demonstrations" / "raw" / "raw_demonstrations"
    
    logger.info(f"Loading demonstrations from {demo_dir}")
    processed_data = process_all_demonstrations(
        demo_dir=str(demo_dir),
        split_ratios=(0.7, 0.15, 0.15),
        random_seed=42,
        max_examples_per_demo=10  # Use small subset for testing
    )
    
    # Create BC acquisition trainer with diagnostics enabled
    logger.info("Creating BC acquisition trainer...")
    trainer = create_bc_acquisition_trainer(
        learning_rate=3e-4,
        batch_size=32,
        use_curriculum=True,
        use_jax=True,
        checkpoint_dir="checkpoints/bc_diagnostics",
        enable_wandb_logging=False,
        experiment_name="bc_loss_diagnostics"
    )
    
    # Run a few training steps
    logger.info("Running training with diagnostics...")
    
    # Only train for 1 epoch to see diagnostics
    trainer.config = trainer.config._replace(max_epochs_per_level=1)
    
    # Train on curriculum
    results = trainer.train_on_curriculum(
        curriculum_datasets=processed_data.acquisition_datasets,
        validation_datasets=processed_data.acquisition_datasets,
        random_key=random.PRNGKey(42)
    )
    
    logger.info("Training complete. Check logs above for loss diagnostics.")
    logger.info(f"Final accuracy: {results.final_state.best_validation_accuracy:.4f}")

if __name__ == "__main__":
    main()