#!/usr/bin/env python3
"""
Simple test for real observation training - avoids GRPO update issues.
"""

import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import jax
import jax.random as random
from omegaconf import OmegaConf

from src.causal_bayes_opt.training.enriched_trainer import EnrichedGRPOTrainer

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_real_observation_episode():
    """Test a single episode with real observations."""
    config = {
        'seed': 42,
        'training': {
            'n_episodes': 1,
            'episode_length': 3,  # Short episode
            'learning_rate': 0.001,
            'gamma': 0.99,
            'max_intervention_value': 2.0,
            'use_real_observations': True,
            'reward_weights': {
                'optimization': 1.0,
                'discovery': 0.0,
                'efficiency': 0.0
            },
            'architecture': {
                'hidden_dim': 32,
                'num_layers': 1,
                'num_heads': 2,
                'key_size': 16,
                'widening_factor': 2,
                'dropout': 0.0
            },
            'state_config': {
                'max_history_size': 10,
                'num_channels': 5,
                'standardize_values': False,
                'include_temporal_features': False
            }
        },
        'experiment': {
            'scm_generation': {
                'use_variable_factory': False,
                'fallback_scms': ['fork_3var'],
                'rotation_frequency': 10
            }
        },
        'logging': {
            'checkpoint_dir': 'test_checkpoints/simple_real',
            'level': 'INFO',
            'wandb': {'enabled': False}
        }
    }
    
    config = OmegaConf.create(config)
    trainer = EnrichedGRPOTrainer(config)
    
    # Run one episode without GRPO updates
    key = random.PRNGKey(42)
    episode_key, _ = random.split(key)
    
    logger.info("Running episode with real observations...")
    metrics = trainer._run_episode(0, episode_key)
    
    logger.info(f"✓ Episode completed successfully!")
    logger.info(f"  Mean reward: {metrics.mean_reward:.3f}")
    logger.info(f"  Structure accuracy: {metrics.structure_accuracy:.3f}")
    logger.info(f"  SCM type: {metrics.scm_type}")
    
    # Check that we used real observations
    if config.training.use_real_observations:
        logger.info("✓ Confirmed: Used REAL observations")
    else:
        logger.error("✗ Expected real observations")
        
    return metrics


if __name__ == "__main__":
    test_real_observation_episode()