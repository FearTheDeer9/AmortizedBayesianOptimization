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

from src.causal_bayes_opt.training.unified_grpo_trainer import create_unified_grpo_trainer
from src.causal_bayes_opt.training.policy_bc_trainer import PolicyBCTrainer
from src.causal_bayes_opt.training.surrogate_bc_trainer import SurrogateBCTrainer
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
    logger.info("Training GRPO Model (Unified Trainer with True GRPO)")
    logger.info("=" * 60)
    
    # Disable convergence detection for now (incompatible TrainingMetrics structure)
    config['use_early_stopping'] = False
    
    # Map reward weights if needed
    if 'reward_weights' not in config:
        config['reward_weights'] = {
            'optimization': 0.8,
            'discovery': 0.1,
            'efficiency': 0.1
        }
    
    # Create trainer
    trainer = create_unified_grpo_trainer(config)
    
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
    
    logger.info(f"Converged: {results.get('converged', False)}")
    
    return results


def train_bc(config: DictConfig) -> Dict[str, Any]:
    """
    Train BC policy model on expert demonstrations.
    
    Args:
        config: Training configuration
        
    Returns:
        Training results
    """
    logger.info("=" * 60)
    logger.info("Training BC Policy Model")
    logger.info("=" * 60)
    
    demo_path = config.get('demo_path', 'expert_demonstrations/raw/raw_demonstrations')
    logger.info(f"Loading expert demonstrations from: {demo_path}")
    
    # Log max_demos parameter
    max_demos = config.get('max_demos')
    if max_demos is not None:
        logger.info(f"*** Limiting to {max_demos} demonstration files ***")
    
    # Create policy BC trainer
    trainer = PolicyBCTrainer(
        hidden_dim=config.get('hidden_dim', 256),
        learning_rate=config.get('learning_rate', 3e-4),
        batch_size=config.get('batch_size', 32),
        max_epochs=config.get('max_episodes', 1000),  # Use episodes as epochs
        early_stopping_patience=10,
        validation_split=0.2,
        gradient_clip=1.0,
        weight_decay=1e-4,
        seed=config.get('seed', 42)
    )
    
    # Train on demonstrations
    results = trainer.train(demonstrations_path=demo_path, max_demos=config.get('max_demos'))
    
    logger.info(f"\nTraining completed in {results['metrics']['training_time']:.2f}s")
    logger.info(f"Trained for {results['metrics']['epochs_trained']} epochs")
    logger.info(f"Best validation loss: {results['metrics']['best_val_loss']:.4f}")
    logger.info(f"Final train loss: {results['metrics']['final_train_loss']:.4f}")
    
    # Save checkpoint
    checkpoint_path = Path(config.get('checkpoint_dir', 'checkpoints')) / 'bc_final'
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(checkpoint_path, results)
    logger.info(f"Saved BC policy checkpoint to {checkpoint_path}")
    
    # Convert results to match expected format
    results = {
        'training_time': results['metrics']['training_time'],
        'final_metrics': {
            'loss': results['metrics']['final_train_loss'],
            'val_loss': results['metrics']['best_val_loss']
        },
        'all_metrics': results['metrics'],
        'policy_params': results['params'],
        'n_demonstrations': results['metadata']['n_train_samples']
    }
    
    return results


def train_surrogate(config: DictConfig) -> Dict[str, Any]:
    """
    Train BC surrogate model for structure learning.
    
    Args:
        config: Training configuration
        
    Returns:
        Training results
    """
    logger.info("=" * 60)
    logger.info("Training BC Surrogate Model (Structure Learning)")
    logger.info("=" * 60)
    
    demo_path = config.get('demo_path', 'expert_demonstrations/raw/raw_demonstrations')
    logger.info(f"Loading expert demonstrations from: {demo_path}")
    
    # Create surrogate BC trainer
    trainer = SurrogateBCTrainer(
        hidden_dim=config.get('surrogate_hidden_dim', 128),
        num_layers=config.get('surrogate_layers', 4),
        num_heads=config.get('surrogate_heads', 8),
        key_size=config.get('architecture', {}).get('key_size', 32),
        learning_rate=config.get('surrogate_lr', 1e-3),
        batch_size=config.get('batch_size', 32),
        max_epochs=config.get('max_episodes', 1000),
        early_stopping_patience=10,
        validation_split=0.2,
        gradient_clip=1.0,
        dropout=config.get('architecture', {}).get('dropout', 0.1),
        weight_decay=1e-4,
        seed=config.get('seed', 42)
    )
    
    # Train on demonstrations
    results = trainer.train(demonstrations_path=demo_path, max_demos=config.get('max_demos'))
    
    logger.info(f"\nTraining completed in {results['metrics']['training_time']:.2f}s")
    logger.info(f"Trained for {results['metrics']['epochs_trained']} epochs")
    logger.info(f"Best validation loss: {results['metrics']['best_val_loss']:.4f}")
    logger.info(f"Final train loss: {results['metrics']['final_train_loss']:.4f}")
    
    # Save checkpoint
    checkpoint_path = Path(config.get('checkpoint_dir', 'checkpoints')) / 'bc_surrogate_final'
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(checkpoint_path, results)
    logger.info(f"Saved BC surrogate checkpoint to {checkpoint_path}")
    
    # Convert results to match expected format
    results = {
        'training_time': results['metrics']['training_time'],
        'final_metrics': {
            'loss': results['metrics']['final_train_loss'],
            'val_loss': results['metrics']['best_val_loss']
        },
        'all_metrics': results['metrics'],
        'params': results['params'],
        'n_demonstrations': results['metadata']['n_train_samples'],
        'model_type': 'surrogate'
    }
    
    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ACBO methods')
    
    # Method selection
    parser.add_argument(
        '--method', 
        type=str, 
        required=True,
        choices=['grpo', 'bc', 'surrogate', 'both'],
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
    parser.add_argument('--demo_path', type=str, 
                       default='expert_demonstrations/raw/raw_demonstrations',
                       help='Path to expert demonstrations for BC training')
    parser.add_argument('--max_demos', type=int, default=None,
                       help='Maximum number of demo files to load (for testing)')
    
    # Surrogate parameters
    parser.add_argument('--use_surrogate', action='store_true', help='Enable surrogate learning in GRPO')
    parser.add_argument('--surrogate_lr', type=float, default=1e-3, help='Surrogate learning rate')
    parser.add_argument('--surrogate_hidden_dim', type=int, default=128, help='Surrogate hidden dimension')
    parser.add_argument('--surrogate_layers', type=int, default=4, help='Surrogate number of layers')
    parser.add_argument('--surrogate_heads', type=int, default=8, help='Surrogate number of attention heads')
    
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
        'surrogate_hidden_dim': args.surrogate_hidden_dim,
        'surrogate_layers': args.surrogate_layers,
        'surrogate_heads': args.surrogate_heads,
        'hidden_dim': args.hidden_dim,
        
        # BC-specific
        'demo_path': args.demo_path,
        'max_demos': args.max_demos,
        
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
        logger.info("\n✓ BC acquisition training completed")
        
    if args.method == 'surrogate':
        surrogate_results = train_surrogate(config)
        logger.info("\n✓ BC surrogate training completed")
    
    logger.info("\n" + "=" * 60)
    logger.info("Training completed successfully!")
    logger.info(f"Checkpoints saved to: {args.checkpoint_dir}/")
    
    if args.method == 'both':
        logger.info("\nNext step: Run evaluation script to compare methods")
        logger.info("  python scripts/evaluate_acbo_methods.py")


if __name__ == "__main__":
    main()