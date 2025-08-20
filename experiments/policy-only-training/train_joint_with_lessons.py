#!/usr/bin/env python3
"""
Training script for JointACBOTrainer incorporating all debugging lessons.

This script ensures all the lessons learned from debugging are properly applied:
1. Using simple_permutation_invariant architecture (no feature extraction)
2. Fixed std = 0.5 for exploration
3. Proper GRPO group size = 10
4. No double normalization (already fixed in trainer)
5. Variable range limits with soft tanh mapping
6. Optimal learning rate = 5e-4
7. Disabled phase rotation for policy-only training
8. Choice between SCMCurriculumFactory and VariableSCMFactory
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import jax
import jax.numpy as jnp

from src.causal_bayes_opt.training.joint_acbo_trainer import JointACBOTrainer
from src.causal_bayes_opt.training.curriculum_factory import SCMCurriculumFactory
from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
from src.causal_bayes_opt.training.utils.wandb_setup import WandBManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def make_serializable(obj: Any) -> Any:
    """
    Recursively convert JAX arrays and other non-serializable objects to Python types.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if hasattr(obj, 'tolist'):
        # JAX array
        return obj.tolist()
    elif isinstance(obj, frozenset):
        # Convert frozenset to sorted list string (for intervention targets)
        return str(sorted(list(obj)))
    elif isinstance(obj, set):
        # Convert set to sorted list
        return sorted(list(obj))
    elif isinstance(obj, dict):
        # Recursively process dictionary, converting keys if needed
        result = {}
        for k, v in obj.items():
            # Convert key if it's not JSON-serializable
            if isinstance(k, (frozenset, set)):
                key = str(sorted(list(k)))
            elif isinstance(k, (str, int, float, bool, type(None))):
                key = k
            else:
                key = str(k)
            result[key] = make_serializable(v)
        return result
    elif isinstance(obj, list):
        # Recursively process list
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        # Convert tuple to list
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        # Already serializable
        return obj
    elif hasattr(obj, '__dict__'):
        # Try to extract attributes from objects
        return make_serializable(obj.__dict__)
    else:
        # Fall back to string representation
        return str(obj)


def create_optimized_config(
    max_episodes: int = 100,
    scm_type: str = "curriculum",
    enable_surrogate: bool = True,
    verbose: bool = False,
    enable_wandb: bool = False
) -> Dict[str, Any]:
    """
    Create configuration incorporating all debugging lessons.
    
    Args:
        max_episodes: Number of training episodes
        scm_type: Type of SCM factory ("curriculum" or "variable")
        enable_surrogate: Whether to use surrogate model
        verbose: Enable verbose logging
        
    Returns:
        Configuration dictionary
    """
    
    config = {
        # Core episode settings
        'max_episodes': max_episodes,
        'obs_per_episode': 10,
        'max_interventions': 30,
        
        # CRITICAL LESSON 1: Use simple_permutation_invariant (no feature extraction)
        'policy_architecture': 'simple_permutation_invariant',
        'architecture_level': 'simple',  # Backup in case
        
        # CRITICAL LESSON 2: Fixed std = 0.5 for consistent exploration
        'use_fixed_std': True,
        'fixed_std': 0.5,
        
        # CRITICAL LESSON 3: Optimal learning rate from debugging
        'learning_rate': 5e-4,
        
        # CRITICAL LESSON 4: GRPO configuration that worked
        'grpo_config': {
            'group_size': 10,  # Tested and working
            'entropy_coefficient': 0.001,  # Low for exploitation
            'clip_ratio': 0.2,
            'gradient_clip': 1.0,
            'ppo_epochs': 4,
            'normalize_advantages': True  # GRPO handles normalization
        },
        
        # CRITICAL LESSON 5: Disable phase rotation for policy-only training
        'episodes_per_phase': 1000000,  # Effectively never switch
        
        # Surrogate settings
        'use_surrogate': enable_surrogate,
        'surrogate_hidden_dim': 128,
        'surrogate_layers': 4,
        'surrogate_heads': 8,
        'surrogate_lr': 1e-3,
        
        # Joint training configuration
        'joint_training': {
            'initial_phase': 'policy',  # Start with policy
            
            # CRITICAL LESSON 6: Reward weights that work
            'loss_weights': {
                'policy': {
                    'target_delta': 0.9,  # Heavy focus on target
                    'direct_parent': 0.1,  # Small bonus for correct structure
                    'information_gain': 0.0  # Disabled based on debugging
                }
            },
            
            # Replay buffer settings
            'replay': {
                'enabled': True,
                'capacity': 100,
                'batch_size': 4,
                'min_size': 10,
                'prioritize_recent': True
            },
            
            # SCM management
            'scm_management': {
                'use_curriculum': (scm_type == "curriculum"),
                'rotation_episodes': 20,
                'convergence_f1_threshold': 0.9
            },
            
            # Disable adaptive features for consistency
            'adaptive': {
                'use_adaptive_weights': False,
                'use_performance_rotation': False
            },
            
            # Logging
            'log_every': 10,
            'validate_every': 50,
            'save_checkpoints': True
        },
        
        # GRPO rewards enabled
        'use_grpo_rewards': True,
        
        # Reward configuration
        'grpo_reward_config': {
            'use_trajectory_info_gain': False,  # Disabled based on debugging
            'info_gain_weight': 0.0,
            'parent_accuracy_weight': 0.1,
            'target_improvement_weight': 0.9
        },
        
        # General settings
        'batch_size': 32,
        'seed': 42,
        'verbose': verbose,
        'checkpoint_dir': 'checkpoints/joint_with_lessons',
        
        # Architecture details (backup)
        'architecture': {
            'hidden_dim': 256,
            'encoder_dim': 256,
            'use_batch_norm': False,
            'dropout_rate': 0.0
        },
        
        # WandB logging configuration
        'logging': {
            'wandb': {
                'enabled': enable_wandb,
                'project': 'causal-bayes-opt-grpo',
                'name': f'joint_{scm_type}_{max_episodes}ep',
                'tags': ['joint_training', 'grpo', 'fixed_std', 'simple_permutation'],
                'notes': 'Training with all debugging lessons incorporated',
                'group': f'joint_{scm_type}',
                'config_exclude_keys': [],  # Log all config
                'log_frequency': 1  # Log every episode
            }
        }
    }
    
    return config


def create_scm_factory(scm_type: str, seed: int = 42):
    """
    Create SCM factory based on type.
    
    Args:
        scm_type: "curriculum" or "variable"
        seed: Random seed
        
    Returns:
        SCM factory instance
    """
    
    if scm_type == "curriculum":
        logger.info("Creating SCMCurriculumFactory for progressive difficulty")
        return SCMCurriculumFactory(
            start_level=1,
            max_level=3,
            mode="progressive",
            seed=seed
        )
    elif scm_type == "variable":
        logger.info("Creating VariableSCMFactory for diverse structures")
        factory = VariableSCMFactory(
            noise_scale=1.0,
            coefficient_range=(-2.0, 2.0),
            seed=seed
        )
        # Create a callable that generates SCMs with varying structures
        import random
        random.seed(seed)  # Set seed for reproducibility
        structure_types = ["collider", "fork", "chain", "mixed"]
        
        def generate_scm():
            # Randomly select structure type and number of variables
            structure = random.choice(structure_types)
            num_vars = random.randint(3, 5)
            return factory.create_variable_scm(
                num_variables=num_vars,
                structure_type=structure,
                target=None  # Let factory choose target
            )
        
        return generate_scm
    else:
        raise ValueError(f"Unknown SCM type: {scm_type}")


def verify_configuration(trainer: JointACBOTrainer):
    """
    Verify that all debugging lessons are properly incorporated.
    
    Args:
        trainer: The initialized trainer
    """
    
    logger.info("\n" + "="*70)
    logger.info("VERIFYING CONFIGURATION")
    logger.info("="*70)
    
    checks = []
    
    # Check 1: Policy architecture
    if hasattr(trainer, 'policy_architecture'):
        arch = trainer.policy_architecture
        if arch == 'simple_permutation_invariant':
            checks.append("✅ Using simple_permutation_invariant policy (no feature extraction)")
        else:
            checks.append(f"❌ Wrong architecture: {arch} (should be simple_permutation_invariant)")
    else:
        checks.append("⚠️  Could not verify policy architecture")
    
    # Check 2: Fixed std
    if hasattr(trainer, 'use_fixed_std') and trainer.use_fixed_std:
        if hasattr(trainer, 'fixed_std') and trainer.fixed_std == 0.5:
            checks.append("✅ Using fixed std = 0.5")
        else:
            checks.append(f"❌ Wrong fixed_std: {getattr(trainer, 'fixed_std', 'unknown')}")
    else:
        checks.append("❌ Not using fixed std")
    
    # Check 3: Learning rate
    if hasattr(trainer, 'learning_rate'):
        lr = trainer.learning_rate
        if abs(lr - 5e-4) < 1e-6:
            checks.append("✅ Using optimal learning rate = 5e-4")
        else:
            checks.append(f"⚠️  Different learning rate: {lr}")
    
    # Check 4: GRPO group size
    if hasattr(trainer, 'grpo_config'):
        group_size = trainer.grpo_config.group_size
        if group_size == 10:
            checks.append("✅ GRPO group size = 10")
        else:
            checks.append(f"❌ Wrong GRPO group size: {group_size}")
    
    # Check 5: Phase rotation
    if hasattr(trainer, 'episodes_per_phase'):
        if trainer.episodes_per_phase >= 100000:
            checks.append("✅ Phase rotation effectively disabled")
        else:
            checks.append(f"⚠️  Phase rotation at {trainer.episodes_per_phase} episodes")
    
    # Check 6: Reward weights
    if hasattr(trainer, 'policy_loss_weights'):
        weights = trainer.policy_loss_weights
        if weights.get('target_delta', 0) >= 0.7:
            checks.append("✅ Heavy focus on target minimization")
        else:
            checks.append(f"⚠️  Target weight only {weights.get('target_delta', 0)}")
    
    # Print all checks
    for check in checks:
        logger.info(check)
    
    # Count successes
    successes = sum(1 for c in checks if c.startswith("✅"))
    total = len(checks)
    
    logger.info(f"\nConfiguration score: {successes}/{total}")
    if successes == total:
        logger.info("🎉 All debugging lessons properly incorporated!")
    elif successes >= total - 1:
        logger.info("⚠️  Most lessons incorporated, training should work")
    else:
        logger.warning("❌ Missing critical configuration, training may not work well")
    
    logger.info("="*70 + "\n")
    
    return successes == total


def main():
    """Main training function."""
    
    parser = argparse.ArgumentParser(description="Train JointACBO with debugging lessons")
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of training episodes')
    parser.add_argument('--scm-type', choices=['curriculum', 'variable'], default='curriculum',
                        help='Type of SCM factory to use')
    parser.add_argument('--no-surrogate', action='store_true',
                        help='Disable surrogate model')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test with 10 episodes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable WandB logging for stability monitoring')
    
    args = parser.parse_args()
    
    # Override episodes for quick test
    if args.quick_test:
        args.episodes = 10
        logger.info("Running quick test with 10 episodes")
    
    # Create configuration
    logger.info("\n" + "="*70)
    logger.info("INITIALIZING TRAINING WITH DEBUGGING LESSONS")
    logger.info("="*70)
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"SCM Type: {args.scm_type}")
    logger.info(f"Surrogate: {'Enabled' if not args.no_surrogate else 'Disabled'}")
    logger.info(f"Seed: {args.seed}")
    logger.info("="*70 + "\n")
    
    config = create_optimized_config(
        max_episodes=args.episodes,
        scm_type=args.scm_type,
        enable_surrogate=not args.no_surrogate,
        verbose=args.verbose,
        enable_wandb=args.wandb
    )
    
    # Update seed
    config['seed'] = args.seed
    
    # Log critical settings
    logger.info("Critical settings from debugging:")
    logger.info(f"  - Policy: {config['policy_architecture']}")
    logger.info(f"  - Fixed std: {config['fixed_std']}")
    logger.info(f"  - Learning rate: {config['learning_rate']}")
    logger.info(f"  - GRPO group size: {config['grpo_config']['group_size']}")
    logger.info(f"  - Target weight: {config['joint_training']['loss_weights']['policy']['target_delta']}")
    logger.info("")
    
    # Initialize WandB if enabled
    wandb_manager = None
    if args.wandb:
        logger.info("Initializing WandB for stability monitoring...")
        wandb_manager = WandBManager()
        wandb_run = wandb_manager.setup(config, experiment_name=f"joint_{args.scm_type}")
        if wandb_run:
            logger.info(f"✅ WandB initialized: {wandb_run.url if hasattr(wandb_run, 'url') else 'running'}")
        else:
            logger.warning("⚠️  WandB initialization failed, continuing without logging")
            wandb_manager = None
    
    # Initialize trainer
    logger.info("Initializing JointACBOTrainer...")
    trainer = JointACBOTrainer(config=config)
    logger.info("✅ Trainer initialized\n")
    
    # Verify configuration
    config_ok = verify_configuration(trainer)
    if not config_ok and not args.quick_test:
        response = input("\n⚠️  Configuration may not be optimal. Continue? (y/n): ")
        if response.lower() != 'y':
            logger.info("Training cancelled")
            if wandb_manager:
                wandb_manager.finish()
            return
    
    # Create SCM factory
    scm_factory = create_scm_factory(args.scm_type, args.seed)
    
    # Start training
    logger.info("\n" + "="*70)
    logger.info("STARTING TRAINING")
    logger.info("="*70 + "\n")
    
    try:
        # Add callback for WandB logging if enabled
        if wandb_manager and wandb_manager.enabled:
            original_train = trainer.train
            
            def train_with_logging(*args, **kwargs):
                # Call original train
                results = original_train(*args, **kwargs)
                
                # Log episode metrics to WandB
                if 'episode_metrics' in results and wandb_manager:
                    for metrics in results['episode_metrics']:
                        wandb_manager.log({
                            'episode': metrics.get('episode', 0),
                            'mean_reward': metrics.get('mean_reward', 0),
                            'target_value': metrics.get('target_value', 0),
                            'policy_loss': metrics.get('policy_loss', 0),
                            'grad_norm': metrics.get('grad_norm', 0),
                            'param_change': metrics.get('param_change', 0),
                            'f1_score': metrics.get('f1_score', 0),
                            'target_improvement': metrics.get('target_improvement', 0),
                            'info_gain': metrics.get('information_gain', 0),
                        }, step=metrics.get('episode', 0))
                
                return results
            
            trainer.train = train_with_logging
        
        results = trainer.train(scms=scm_factory)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/joint_with_lessons_{args.scm_type}_{timestamp}.json"
        os.makedirs("results", exist_ok=True)
        
        with open(results_file, 'w') as f:
            # Use recursive serialization helper
            serializable_results = make_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"\n✅ Training complete! Results saved to {results_file}")
        
        # Print summary
        if 'episode_metrics' in results:
            metrics = results['episode_metrics']
            if metrics:
                final_metrics = metrics[-1]
                logger.info("\nFinal metrics:")
                logger.info(f"  - Episode: {final_metrics.get('episode', 'N/A')}")
                logger.info(f"  - Training phase: {final_metrics.get('training_phase', 'unknown')}")
                
                # Handle mean reward
                mean_reward = final_metrics.get('mean_reward', 'N/A')
                if mean_reward != 'N/A':
                    logger.info(f"  - Mean reward: {mean_reward:.4f}")
                else:
                    logger.info(f"  - Mean reward: N/A")
                
                # Handle F1 score - it's N/A in policy-only mode without surrogate
                f1_value = final_metrics.get('f1_score', None)
                if f1_value is not None and f1_value != 'N/A':
                    logger.info(f"  - F1 score: {f1_value:.4f}")
                else:
                    # Check if we're in policy-only mode
                    phase = final_metrics.get('training_phase', 'unknown')
                    if phase == 'policy' and not args.no_surrogate:
                        logger.info(f"  - F1 score: N/A (surrogate not active in policy phase)")
                    elif args.no_surrogate:
                        logger.info(f"  - F1 score: N/A (surrogate disabled)")
                    else:
                        logger.info(f"  - F1 score: N/A")
                
                # Handle target improvement
                target_improvement = final_metrics.get('target_improvement', 'N/A')
                if target_improvement != 'N/A':
                    logger.info(f"  - Target improvement: {target_improvement:.4f}")
                else:
                    logger.info(f"  - Target improvement: N/A")
        
        # Log final summary to WandB
        if wandb_manager and wandb_manager.enabled and 'episode_metrics' in results:
            # Handle F1 score for WandB - use 0 if N/A or None
            f1_for_wandb = final_metrics.get('f1_score', 0)
            if f1_for_wandb is None or f1_for_wandb == 'N/A':
                f1_for_wandb = 0
            
            final_summary = {
                'final_mean_reward': final_metrics.get('mean_reward', 0),
                'final_target_value': final_metrics.get('target_value', 0),
                'final_f1_score': f1_for_wandb,
                'final_training_phase': final_metrics.get('training_phase', 'unknown'),
                'total_episodes': len(results['episode_metrics']),
            }
            wandb_manager.log(final_summary)
            logger.info("\n📊 Final metrics logged to WandB")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        if wandb_manager:
            wandb_manager.finish()
        return 1
    
    finally:
        # Ensure WandB is properly closed
        if wandb_manager:
            wandb_manager.finish()
            logger.info("WandB run finished")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())