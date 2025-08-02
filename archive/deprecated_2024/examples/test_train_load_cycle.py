#!/usr/bin/env python3
"""
Test the complete train-save-load cycle with the fixed Haiku compatibility.

This demo:
1. Trains a small GRPO model
2. Saves checkpoint
3. Loads in a different context
4. Verifies inference works correctly
"""

import logging
import tempfile
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from omegaconf import DictConfig
from src.causal_bayes_opt.training.clean_grpo_trainer import create_clean_grpo_trainer
from src.causal_bayes_opt.evaluation.model_interfaces import create_grpo_acquisition
from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_train_save_load():
    """Test the complete train-save-load cycle."""
    logger.info("=== Testing Train-Save-Load Cycle ===\n")
    
    # Create temporary directory for checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "test_checkpoint"
        
        # 1. Training Phase
        logger.info("1. Training Phase")
        logger.info("-" * 40)
        
        # Create minimal training config
        config = DictConfig({
            'seed': 42,
            'max_episodes': 10,  # Very short for testing
            'n_variables_range': [3, 3],
            'obs_per_episode': 50,
            'max_interventions': 5,
            'batch_size': 16,
            'learning_rate': 1e-3,
            'checkpoint_dir': str(checkpoint_dir),
            'architecture': {
                'num_layers': 2,
                'num_heads': 4,
                'hidden_dim': 64,  # Small for testing
                'key_size': 16,
                'dropout': 0.0
            }
        })
        
        # Create trainer
        trainer = create_clean_grpo_trainer(config)
        
        # Simple SCM generator
        def scm_generator():
            return create_fork_scm(noise_scale=1.0, target="Y")
        
        # Train
        logger.info("Training GRPO model...")
        results = trainer.train(scm_generator)
        logger.info(f"Training completed in {results['training_time']:.2f}s")
        logger.info(f"Final reward: {results['final_metrics']['mean_reward']:.3f}\n")
        
        # 2. Loading Phase
        logger.info("2. Loading Phase")
        logger.info("-" * 40)
        
        # Path to saved checkpoint
        final_checkpoint = checkpoint_dir / "clean_grpo_final"
        assert final_checkpoint.exists(), "Checkpoint not found!"
        
        logger.info(f"Loading checkpoint from {final_checkpoint}")
        
        # Create acquisition function (simulates loading in different context)
        try:
            grpo_acquisition = create_grpo_acquisition(final_checkpoint, seed=123)
            logger.info("✓ Successfully loaded GRPO checkpoint!\n")
        except Exception as e:
            logger.error(f"✗ Failed to load checkpoint: {e}")
            return False
        
        # 3. Inference Phase
        logger.info("3. Inference Phase")
        logger.info("-" * 40)
        
        # Create test tensor
        import jax.numpy as jnp
        test_tensor = jnp.ones((20, 3, 3))  # [T=20, n_vars=3, channels=3]
        
        logger.info("Running inference with loaded model...")
        try:
            # Call acquisition function
            intervention = grpo_acquisition(
                tensor=test_tensor,
                posterior=None,
                target='Y',
                variables=['X', 'Y', 'Z']
            )
            
            logger.info(f"✓ Inference successful!")
            logger.info(f"  Selected intervention: {intervention}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Inference failed: {e}")
            return False


def main():
    """Run the train-save-load test."""
    logger.info("Train-Save-Load Cycle Test")
    logger.info("=" * 60)
    logger.info("This test verifies that the Haiku parameter fix works correctly.")
    logger.info("Models trained in one context can be loaded in another.\n")
    
    success = test_train_save_load()
    
    logger.info("\n" + "=" * 60)
    if success:
        logger.info("✓ SUCCESS: Train-save-load cycle works correctly!")
        logger.info("  The shared policy factory prevents Haiku parameter mismatches.")
        logger.info("  No more wasted training time due to loading errors!")
    else:
        logger.error("✗ FAILED: There are still issues with the train-save-load cycle.")
    
    return success


if __name__ == "__main__":
    main()