#!/usr/bin/env python3
"""
Test BC training with debugging to identify cause of astronomical loss values.
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
import pickle

from src.causal_bayes_opt.training.bc_data_pipeline import process_all_demonstrations
from src.causal_bayes_opt.training.bc_surrogate_trainer import BCSurrogateTrainer, BCTrainingConfig
from src.causal_bayes_opt.training.config import SurrogateTrainingConfig
from src.causal_bayes_opt.training.trajectory_processor import DifficultyLevel

# Configure logging to show debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bc_debug_training.log')
    ]
)

logger = logging.getLogger(__name__)


def test_bc_training_with_debug():
    """Test BC training with debugging enabled."""
    logger.info("Starting BC training test with debugging")
    
    # Configuration
    batch_size = 8  # Small batch for debugging
    max_epochs_per_level = 5  # Few epochs for quick debugging
    
    # Create BC configuration
    surrogate_config = SurrogateTrainingConfig(
        learning_rate=1e-3,
        batch_size=batch_size,
        max_epochs=max_epochs_per_level,
        early_stopping_patience=5,
        validation_frequency=1,
        use_continuous_model=True  # Use continuous model for dynamic dimensions
    )
    
    bc_config = BCTrainingConfig(
        surrogate_config=surrogate_config,
        curriculum_learning=False,  # Single level for simplicity
        batch_size=batch_size,
        max_epochs_per_level=max_epochs_per_level,
        min_epochs_per_level=1,
        advancement_threshold=0.5,
        validation_patience=5,
        use_jax_compilation=True,
        checkpoint_dir="checkpoints/bc_debug",
        save_frequency=1,
        enable_wandb_logging=False,  # Disable wandb for debugging
        experiment_name="bc_debug"
    )
    
    # Load small expert dataset
    logger.info("Loading expert demonstrations...")
    expert_demos_dir = Path("expert_demonstrations/raw/raw_demonstrations")
    
    if not expert_demos_dir.exists():
        logger.error(f"Expert demonstrations directory not found: {expert_demos_dir}")
        return
    
    # Process all demonstrations from directory
    processed_dataset = process_all_demonstrations(
        str(expert_demos_dir),
        curriculum_learning=False,  # Single difficulty for debugging
        max_files=1  # Process only one file for quick debugging
    )
    
    # Get the surrogate datasets
    surrogate_datasets = processed_dataset.surrogate_datasets
    
    if not surrogate_datasets:
        logger.error("No surrogate datasets created")
        return
    
    # Get first available dataset
    if DifficultyLevel.EASY in surrogate_datasets:
        train_dataset = surrogate_datasets[DifficultyLevel.EASY]
    else:
        train_dataset = next(iter(surrogate_datasets.values()))
    
    logger.info(f"Training dataset has {len(train_dataset.training_examples)} examples")
    
    # Create small validation set
    val_dataset = train_dataset  # Use same data for quick test
    
    # Create trainer
    trainer = BCSurrogateTrainer(bc_config)
    
    # Create training datasets dict
    curriculum_datasets = {DifficultyLevel.EASY: train_dataset}
    validation_datasets = {DifficultyLevel.EASY: val_dataset}
    
    # Run training with debugging
    logger.info("Starting training with debugging enabled...")
    random_key = random.PRNGKey(42)
    
    try:
        results = trainer.train_on_curriculum(
            curriculum_datasets=curriculum_datasets,
            validation_datasets=validation_datasets,
            random_key=random_key
        )
        
        logger.info(f"Training completed!")
        logger.info(f"Final loss: {results.final_state.best_validation_loss}")
        logger.info(f"Total epochs: {results.final_state.epoch}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_bc_training_with_debug()