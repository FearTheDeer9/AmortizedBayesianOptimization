#!/usr/bin/env python3
"""
Unified training script for ACBO methods (GRPO and BC).

This script provides a clean interface to train both GRPO and BC models
with consistent configurations and checkpoint formats.

Usage:
    python scripts/train_acbo_methods.py --method grpo --episodes 1000
    python scripts/train_acbo_methods.py --method bc --episodes 500 --expert oracle
"""

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from omegaconf import DictConfig
import pyrsistent as pyr

from src.causal_bayes_opt.training.clean_grpo_trainer import create_clean_grpo_trainer
from src.causal_bayes_opt.training.clean_bc_trainer import create_clean_bc_trainer
from src.causal_bayes_opt.experiments.benchmark_scms import (
    create_dense_scm,
    create_sparse_scm,
    create_chain_scm,
    create_fork_scm,
    create_collider_scm
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_scm_generator(scm_type: str = 'random', n_variables_range: list = [3, 8]):
    """
    Create SCM generator function.
    
    Args:
        scm_type: Type of SCM ('random', 'chain', 'fork', 'collider', 'mixed')
        n_variables_range: Range of variables for random SCMs
        
    Returns:
        Function that generates SCMs
    """
    import random as py_random
    
    if scm_type == 'random':
        def generator():
            n_vars = py_random.randint(n_variables_range[0], n_variables_range[1])
            # Use sparse SCM for random graphs
            return create_sparse_scm(
                num_vars=n_vars,
                edge_prob=0.3,
                noise_scale=1.0
            )
    
    elif scm_type == 'chain':
        def generator():
            n_vars = py_random.randint(max(3, n_variables_range[0]), n_variables_range[1])
            return create_chain_scm(chain_length=n_vars)
    
    elif scm_type == 'fork':
        def generator():
            return create_fork_scm(noise_scale=1.0)
    
    elif scm_type == 'collider':
        def generator():
            return create_collider_scm(noise_scale=1.0)
    
    elif scm_type == 'mixed':
        # Randomly select from different types
        generators = [
            lambda: create_sparse_scm(
                num_vars=py_random.randint(n_variables_range[0], n_variables_range[1]),
                edge_prob=0.3,
                noise_scale=1.0
            ),
            lambda: create_chain_scm(
                chain_length=py_random.randint(max(3, n_variables_range[0]), n_variables_range[1])
            ),
            lambda: create_fork_scm(noise_scale=1.0),
            lambda: create_collider_scm(noise_scale=1.0)
        ]
        
        def generator():
            gen = py_random.choice(generators)
            return gen()
    
    else:
        raise ValueError(f"Unknown SCM type: {scm_type}")
    
    return generator


def train_grpo(config: DictConfig) -> Dict[str, Any]:
    """
    Train GRPO model.
    
    Args:
        config: Training configuration
        
    Returns:
        Training results
    """
    logger.info("=" * 60)
    logger.info("Training GRPO Model")
    logger.info("=" * 60)
    
    # Create trainer
    trainer = create_clean_grpo_trainer(config)
    
    # Create SCM generator
    scm_generator = create_scm_generator(
        config.get('scm_type', 'mixed'),
        config.get('n_variables_range', [3, 8])
    )
    
    # Train
    results = trainer.train(scm_generator)
    
    logger.info(f"\nTraining completed in {results['training_time']:.2f}s")
    logger.info(f"Final reward: {results['final_metrics']['mean_reward']:.3f}")
    
    if config.get('use_surrogate', True) and 'structure_metrics' in results['final_metrics']:
        sm = results['final_metrics']['structure_metrics']
        if sm.get('f1_score', 0) > 0:
            logger.info(f"Final structure F1: {sm['f1_score']:.3f}")
    
    return results


def train_bc(config: DictConfig) -> Dict[str, Any]:
    """
    Train BC model.
    
    Args:
        config: Training configuration
        
    Returns:
        Training results
    """
    logger.info("=" * 60)
    logger.info("Training BC Model")
    logger.info("=" * 60)
    logger.info(f"Expert strategy: {config.get('expert_strategy', 'oracle')}")
    
    # Create trainer
    trainer = create_clean_bc_trainer(config)
    
    # Create SCM generator
    scm_generator = create_scm_generator(
        config.get('scm_type', 'mixed'),
        config.get('n_variables_range', [3, 8])
    )
    
    # Train
    results = trainer.train(scm_generator)
    
    logger.info(f"\nTraining completed in {results['training_time']:.2f}s")
    logger.info(f"Trained on {results['n_demonstrations']} demonstrations")
    logger.info(f"Final accuracy: {results['final_metrics']['var_accuracy']:.3f}")
    logger.info(f"Final value RMSE: {results['final_metrics']['value_rmse']:.3f}")
    
    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ACBO methods')
    
    # Method selection
    parser.add_argument(
        '--method', 
        type=str, 
        required=True,
        choices=['grpo', 'bc', 'both'],
        help='Method to train'
    )
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    # SCM parameters
    parser.add_argument('--scm_type', type=str, default='mixed', 
                       choices=['random', 'chain', 'fork', 'collider', 'mixed'],
                       help='Type of SCMs to train on')
    parser.add_argument('--min_vars', type=int, default=3, help='Minimum variables')
    parser.add_argument('--max_vars', type=int, default=8, help='Maximum variables')
    
    # BC-specific parameters
    parser.add_argument('--expert', type=str, default='oracle',
                       choices=['oracle', 'random'],
                       help='Expert strategy for BC')
    parser.add_argument('--demo_episodes', type=int, default=100, 
                       help='Number of demonstration episodes for BC')
    
    # Surrogate parameters
    parser.add_argument('--use_surrogate', action='store_true', help='Enable surrogate learning')
    parser.add_argument('--surrogate_lr', type=float, default=1e-3, help='Surrogate learning rate')
    
    args = parser.parse_args()
    
    # Create configuration
    config = DictConfig({
        'seed': args.seed,
        'max_episodes': args.episodes,
        'n_variables_range': [args.min_vars, args.max_vars],
        'obs_per_episode': 100,
        'max_interventions': 10,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'checkpoint_dir': args.checkpoint_dir,
        'scm_type': args.scm_type,
        'use_surrogate': args.use_surrogate,
        'surrogate_lr': args.surrogate_lr,
        'hidden_dim': args.hidden_dim,
        
        # BC-specific
        'expert_strategy': args.expert,
        'demonstration_episodes': args.demo_episodes,
        
        # Architecture (GRPO)
        'architecture': {
            'num_layers': 4,
            'num_heads': 8,
            'hidden_dim': args.hidden_dim,
            'key_size': 32,
            'dropout': 0.1
        }
    })
    
    logger.info(f"Training configuration:")
    logger.info(f"  Method: {args.method}")
    logger.info(f"  Episodes: {args.episodes}")
    logger.info(f"  SCM type: {args.scm_type}")
    logger.info(f"  Variable range: [{args.min_vars}, {args.max_vars}]")
    logger.info(f"  Use surrogate: {args.use_surrogate}")
    
    # Train selected method(s)
    if args.method == 'grpo' or args.method == 'both':
        grpo_results = train_grpo(config)
        logger.info("\n✓ GRPO training completed")
    
    if args.method == 'bc' or args.method == 'both':
        bc_results = train_bc(config)
        logger.info("\n✓ BC training completed")
    
    logger.info("\n" + "=" * 60)
    logger.info("Training completed successfully!")
    logger.info(f"Checkpoints saved to: {args.checkpoint_dir}/")
    
    if args.method == 'both':
        logger.info("\nNext step: Run evaluation script to compare methods")
        logger.info("  python scripts/evaluate_acbo_methods.py")


if __name__ == "__main__":
    main()