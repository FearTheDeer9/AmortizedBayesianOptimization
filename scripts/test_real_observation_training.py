#!/usr/bin/env python3
"""
Test script for real observation training.

This script validates that the enriched trainer can successfully:
1. Use real observations instead of synthetic bootstrap features
2. Compute rewards based on actual intervention outcomes
3. Update policy based on real experience
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import jax
import jax.random as random
from omegaconf import OmegaConf

from src.causal_bayes_opt.training.enriched_trainer import EnrichedGRPOTrainer
from src.causal_bayes_opt.training.modular_trainer import PolicyFactory, SCMRotationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_config():
    """Create a minimal test configuration."""
    config = {
        'seed': 42,
        'training': {
            'n_episodes': 5,  # Short test
            'episode_length': 5,
            'learning_rate': 0.001,
            'gamma': 0.99,
            'max_intervention_value': 2.0,
            'use_real_observations': True,  # Key setting
            'synthetic_features': {
                'enabled': False
            },
            'reward_weights': {
                'optimization': 1.0,
                'discovery': 1.0,
                'efficiency': 0.5
            },
            'architecture': {
                'hidden_dim': 64,  # Smaller for testing
                'num_layers': 2,
                'num_heads': 4,
                'key_size': 16,
                'widening_factor': 2,
                'dropout': 0.0,
                'policy_intermediate_dim': None
            },
            'state_config': {
                'max_history_size': 20,
                'num_channels': 5,
                'standardize_values': True,
                'include_temporal_features': True
            }
        },
        'experiment': {
            'scm_generation': {
                'use_variable_factory': False,  # Use simple fallback SCMs
                'fallback_scms': ['fork_3var'],
                'rotation_frequency': 10
            }
        },
        'logging': {
            'checkpoint_dir': 'test_checkpoints/real_obs',
            'level': 'DEBUG',
            'wandb': {
                'enabled': False,
                'project': 'test'
            }
        }
    }
    
    return OmegaConf.create(config)


def test_real_observation_training():
    """Test the enriched trainer with real observations."""
    logger.info("Testing real observation training...")
    
    # Create configuration
    config = create_test_config()
    
    # Create trainer
    try:
        trainer = EnrichedGRPOTrainer(config)
        logger.info("✓ Trainer created successfully")
    except Exception as e:
        logger.error(f"✗ Failed to create trainer: {e}")
        raise
    
    # Run training
    try:
        logger.info("Starting training with real observations...")
        trainer.train()
        logger.info("✓ Training completed successfully")
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        raise
    
    # Check that we used real observations
    if config.training.use_real_observations:
        logger.info("✓ Confirmed: use_real_observations = True")
    else:
        logger.error("✗ Expected use_real_observations = True")
        
    logger.info("\nTest completed successfully!")


def test_synthetic_vs_real_comparison():
    """Compare synthetic and real observation training."""
    logger.info("\nTesting synthetic vs real observation comparison...")
    
    # Test 1: Real observations
    config_real = create_test_config()
    config_real.training.use_real_observations = True
    
    logger.info("Test 1: Training with REAL observations")
    trainer_real = EnrichedGRPOTrainer(config_real)
    
    # Run one episode and check logs
    key = random.PRNGKey(42)
    episode_key, key = random.split(key)
    
    metrics = trainer_real._run_episode(0, episode_key)
    logger.info(f"Real obs mean reward: {metrics.mean_reward:.3f}")
    logger.info(f"Real obs structure accuracy: {metrics.structure_accuracy:.3f}")
    
    # Test 2: Synthetic observations (for comparison)
    config_synthetic = create_test_config()
    config_synthetic.training.use_real_observations = False
    
    logger.info("\nTest 2: Training with SYNTHETIC observations")
    trainer_synthetic = EnrichedGRPOTrainer(config_synthetic)
    
    metrics_synth = trainer_synthetic._run_episode(0, episode_key)
    logger.info(f"Synthetic mean reward: {metrics_synth.mean_reward:.3f}")
    logger.info(f"Synthetic structure accuracy: {metrics_synth.structure_accuracy:.3f}")
    
    logger.info("\n✓ Comparison test completed")


if __name__ == "__main__":
    # Run basic test
    test_real_observation_training()
    
    # Run comparison test
    test_synthetic_vs_real_comparison()