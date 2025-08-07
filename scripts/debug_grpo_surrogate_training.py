#!/usr/bin/env python3
"""
Debug script to test GRPO with surrogate training.
Runs minimal training to identify checkpoint saving issues.
"""

import logging
import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from omegaconf import DictConfig
from scripts.train_acbo_methods import train_grpo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_grpo_with_surrogate():
    """Test GRPO training with pre-trained surrogate."""
    logger.info("=" * 80)
    logger.info("DEBUG: Testing GRPO with Surrogate Training")
    logger.info("=" * 80)
    
    # Minimal config for debugging
    config = DictConfig({
        'method': 'grpo_with_surrogate',
        'episodes': 50,  # Minimal episodes for quick test
        'batch_size': 32,
        'learning_rate': 3e-4,
        'hidden_dim': 256,  # Smaller for faster training
        'use_surrogate': True,
        'surrogate_checkpoint': 'checkpoints/comprehensive_20250804_190724/bc_surrogate_final',
        'surrogate_lr': 5e-4,
        'scm_type': 'fork',  # Simple SCM for testing
        'min_vars': 3,
        'max_vars': 3,
        'checkpoint_dir': 'checkpoints/debug_grpo_surrogate',
        'seed': 123,
        'reward_weights': {
            'optimization': 0.5,
            'discovery': 0.1,
            'efficiency': 0.1,
            'info_gain': 0.3  # Higher weight to test info gain
        },
        'log_every': 5,  # More frequent logging
        'use_early_stopping': False
    })
    
    # Create checkpoint directory
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Check if surrogate checkpoint exists
    surrogate_path = Path(config.surrogate_checkpoint)
    if not surrogate_path.exists():
        logger.error(f"Surrogate checkpoint not found: {surrogate_path}")
        logger.info("Please train a BC surrogate first or update the path")
        return
    
    logger.info(f"\n✓ Found surrogate checkpoint: {surrogate_path}")
    
    try:
        # Train GRPO with detailed logging
        logger.info("\nStarting GRPO training with surrogate...")
        start_time = time.time()
        
        results = train_grpo(config)
        
        training_time = time.time() - start_time
        logger.info(f"\n✓ Training completed in {training_time:.1f}s")
        
        # Check if checkpoint was saved
        checkpoint_path = Path(config.checkpoint_dir) / 'unified_grpo_final'
        if checkpoint_path.exists():
            logger.info(f"✓ Checkpoint saved successfully: {checkpoint_path}")
            
            # Check checkpoint contents
            checkpoint_file = checkpoint_path / 'checkpoint.pkl'
            if checkpoint_file.exists():
                import pickle
                with open(checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                logger.info(f"  - Model type: {checkpoint.get('model_type', 'unknown')}")
                logger.info(f"  - Episodes trained: {checkpoint.get('episodes', 'unknown')}")
                logger.info(f"  - Final reward: {results.get('final_reward', 'unknown')}")
        else:
            logger.error(f"✗ Checkpoint NOT saved at: {checkpoint_path}")
            
        # Log training results
        logger.info("\nTraining Results:")
        for key, value in results.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
                
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.exception("Full traceback:")
        

def test_grpo_without_surrogate():
    """Test GRPO training without surrogate as baseline."""
    logger.info("\n" + "=" * 80)
    logger.info("DEBUG: Testing GRPO WITHOUT Surrogate (Baseline)")
    logger.info("=" * 80)
    
    config = DictConfig({
        'method': 'grpo',
        'episodes': 50,
        'batch_size': 32,
        'learning_rate': 3e-4,
        'hidden_dim': 256,
        'use_surrogate': False,
        'scm_type': 'fork',
        'min_vars': 3,
        'max_vars': 3,
        'checkpoint_dir': 'checkpoints/debug_grpo_no_surrogate',
        'seed': 124,
        'reward_weights': {
            'optimization': 0.8,
            'discovery': 0.1,
            'efficiency': 0.1
        },
        'log_every': 5,
        'use_early_stopping': False
    })
    
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info("\nStarting GRPO training without surrogate...")
        start_time = time.time()
        
        results = train_grpo(config)
        
        training_time = time.time() - start_time
        logger.info(f"\n✓ Training completed in {training_time:.1f}s")
        
        # This should be renamed to grpo_no_surrogate_final
        checkpoint_path = Path(config.checkpoint_dir) / 'grpo_no_surrogate_final'
        if checkpoint_path.exists():
            logger.info(f"✓ Checkpoint saved and renamed: {checkpoint_path}")
        else:
            # Check if it's still at the original location
            orig_path = Path(config.checkpoint_dir) / 'unified_grpo_final'
            if orig_path.exists():
                logger.warning(f"Checkpoint at original location: {orig_path}")
                logger.warning("Rename logic may have failed")
            else:
                logger.error("✗ No checkpoint found!")
                
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.exception("Full traceback:")


if __name__ == "__main__":
    # Test both scenarios
    test_grpo_with_surrogate()
    test_grpo_without_surrogate()
    
    logger.info("\n" + "=" * 80)
    logger.info("DEBUG COMPLETE")
    logger.info("=" * 80)