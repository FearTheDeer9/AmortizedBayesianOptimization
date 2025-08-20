#!/usr/bin/env python3
"""
Train GRPO with diverse SCMs (3-99 variables) using AVICI-style generation.
Incorporates all debugging lessons from previous experiments.

Key features:
- Generates new SCM each episode for maximum diversity
- Covers 3-99 variables uniformly
- Uses proven configurations from debugging
- Tracks performance by SCM size category
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.causal_bayes_opt.training.unified_grpo_trainer import UnifiedGRPOTrainer
from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
from src.causal_bayes_opt.training.utils.wandb_setup import WandBManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class DiverseSCMGenerator:
    """
    Generates diverse SCMs on-the-fly for GRPO training.
    Matches the approach from surrogate training but adapted for GRPO.
    """
    
    def __init__(self, min_vars: int = 3, max_vars: int = 99, seed: int = 42):
        """
        Initialize the generator.
        
        Args:
            min_vars: Minimum number of variables
            max_vars: Maximum number of variables
            seed: Random seed for reproducibility
        """
        self.factory = VariableSCMFactory(
            noise_scale=1.0,
            coefficient_range=(-2.0, 2.0),
            seed=seed
        )
        self.min_vars = min_vars
        self.max_vars = max_vars
        self.rng = np.random.RandomState(seed)
        self.generated_count = 0
        
        # Track generation statistics
        self.structure_counts = {
            'collider': 0,
            'fork': 0,
            'chain': 0,
            'mixed': 0,
            'random': 0
        }
        self.size_counts = {
            'small': 0,    # 3-10 vars
            'medium': 0,   # 11-30 vars
            'large': 0,    # 31-50 vars
            'very_large': 0  # 51-99 vars
        }
        
    def __call__(self):
        """Generate a single SCM with random properties."""
        # Uniformly sample number of variables
        num_vars = self.rng.randint(self.min_vars, self.max_vars + 1)
        
        # Sample structure type with equal probability
        structures = ['collider', 'fork', 'chain', 'mixed', 'random']
        structure = self.rng.choice(structures)
        
        # Vary edge density based on structure
        if structure == 'chain':
            edge_density = 1.0 / (num_vars - 1) if num_vars > 1 else 0.0
        elif structure in ['fork', 'collider']:
            edge_density = self.rng.uniform(0.2, 0.4)
        elif structure == 'mixed':
            edge_density = self.rng.uniform(0.25, 0.35)
        else:  # random
            edges_per_var = self.rng.uniform(1.0, 3.0)
            edge_density = min(edges_per_var / (num_vars - 1), 0.5)
        
        self.generated_count += 1
        
        # Generate the SCM
        scm = self.factory.create_variable_scm(
            num_variables=num_vars,
            structure_type=structure,
            edge_density=edge_density
        )
        
        # Track statistics
        self.structure_counts[structure] += 1
        size_category = self._get_size_category(num_vars)
        self.size_counts[size_category] += 1
        
        # Add generation metadata
        metadata = scm.get('metadata', {})
        metadata.update({
            'generation_id': self.generated_count,
            'size_category': size_category,
            'generation_method': 'diverse_generator'
        })
        scm = scm.update({'metadata': metadata})
        
        # Log generation info every 10 SCMs
        if self.generated_count % 10 == 0:
            logger.info(f"Generated {self.generated_count} SCMs - "
                       f"Latest: {num_vars} vars, {structure} structure")
        
        return scm
    
    def _get_size_category(self, num_vars: int) -> str:
        """Categorize SCM by size."""
        if num_vars <= 10:
            return 'small'
        elif num_vars <= 30:
            return 'medium'
        elif num_vars <= 50:
            return 'large'
        else:
            return 'very_large'
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return {
            'total_generated': self.generated_count,
            'structure_distribution': self.structure_counts.copy(),
            'size_distribution': self.size_counts.copy()
        }


def create_diverse_training_config(
    max_episodes: int = 200,
    min_vars: int = 3,
    max_vars: int = 99,
    verbose: bool = False,
    enable_wandb: bool = False,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Create configuration with all debugging lessons for diverse SCM training.
    
    Args:
        max_episodes: Number of training episodes
        min_vars: Minimum number of variables in SCMs
        max_vars: Maximum number of variables in SCMs
        verbose: Enable verbose logging
        enable_wandb: Enable WandB logging
        seed: Random seed
        
    Returns:
        Configuration dictionary with all critical settings
    """
    
    config = {
        # Core episode settings
        'max_episodes': max_episodes,
        'obs_per_episode': 10,
        'max_interventions': 30,
        
        # CRITICAL LESSON 1: Use simple_permutation_invariant (no feature extraction)
        'policy_architecture': 'simple_permutation_invariant',
        'architecture_level': 'simple',
        
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
        
        # CRITICAL LESSON 5: Reward weights from debugging
        'grpo_reward_config': {
            'target_improvement_weight': 0.9,  # Heavy focus on target
            'parent_accuracy_weight': 0.1,     # Small bonus for structure
            'info_gain_weight': 0.0,           # Disabled based on debugging
            'use_trajectory_info_gain': False
        },
        
        # No surrogate for pure GRPO training
        'use_surrogate': False,
        
        # Enable GRPO rewards
        'use_grpo_rewards': True,
        
        # SCM generation settings
        'scm_generation': {
            'min_vars': min_vars,
            'max_vars': max_vars,
            'new_scm_each_episode': True,
            'generator_type': 'diverse'
        },
        
        # General settings
        'batch_size': 32,
        'seed': seed,
        'verbose': verbose,
        'checkpoint_dir': f'checkpoints/diverse_{min_vars}to{max_vars}',
        
        # Architecture details
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
                'name': f'diverse_{min_vars}to{max_vars}_{max_episodes}ep',
                'tags': ['diverse_scms', 'grpo', 'fixed_std', 'simple_permutation'],
                'notes': 'GRPO training with diverse SCMs (3-99 variables)',
                'group': f'diverse_{min_vars}to{max_vars}',
                'config_exclude_keys': [],
                'log_frequency': 1
            }
        }
    }
    
    return config


def verify_trainer_configuration(trainer: UnifiedGRPOTrainer, config: Dict[str, Any]) -> bool:
    """
    Verify that all debugging lessons are properly incorporated in the trainer.
    
    Args:
        trainer: The initialized trainer
        config: The configuration used
        
    Returns:
        True if all checks pass
    """
    
    logger.info("\n" + "="*70)
    logger.info("VERIFYING CONFIGURATION")
    logger.info("="*70)
    
    checks = []
    
    # Check 1: Policy architecture
    if hasattr(trainer, 'policy_architecture'):
        arch = trainer.policy_architecture
        if arch == 'simple_permutation_invariant':
            checks.append("✅ Using simple_permutation_invariant policy")
        else:
            checks.append(f"❌ Wrong architecture: {arch}")
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
    
    # Check 5: Reward weights
    if config.get('grpo_reward_config', {}).get('target_improvement_weight', 0) >= 0.8:
        checks.append("✅ Heavy focus on target minimization")
    else:
        weight = config.get('grpo_reward_config', {}).get('target_improvement_weight', 0)
        checks.append(f"⚠️  Target weight only {weight}")
    
    # Check 6: No surrogate
    if not config.get('use_surrogate', True):
        checks.append("✅ Surrogate disabled for pure GRPO")
    else:
        checks.append("⚠️  Surrogate enabled (not needed for diverse training)")
    
    # Print all checks
    for check in checks:
        logger.info(check)
    
    # Count successes
    successes = sum(1 for c in checks if c.startswith("✅"))
    total = len(checks)
    
    logger.info(f"\nConfiguration score: {successes}/{total}")
    if successes == total:
        logger.info("🎉 All critical settings properly configured!")
    elif successes >= total - 1:
        logger.info("⚠️  Most settings correct, training should work")
    else:
        logger.warning("❌ Missing critical settings, training may not work well")
    
    logger.info("="*70 + "\n")
    
    return successes >= total - 1


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
        # Convert frozenset to sorted list string
        return str(sorted(list(obj)))
    elif isinstance(obj, set):
        # Convert set to sorted list
        return sorted(list(obj))
    elif isinstance(obj, dict):
        # Recursively process dictionary
        result = {}
        for k, v in obj.items():
            # Convert key if needed
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


def analyze_results_by_size(episode_metrics: List[Dict], generator: DiverseSCMGenerator) -> Dict[str, Any]:
    """
    Analyze training results by SCM size category.
    
    Args:
        episode_metrics: List of episode metrics
        generator: The SCM generator with statistics
        
    Returns:
        Analysis results by size category
    """
    # Group metrics by size category
    size_metrics = {
        'small': [],    # 3-10 vars
        'medium': [],   # 11-30 vars
        'large': [],    # 31-50 vars
        'very_large': []  # 51-99 vars
    }
    
    for metric in episode_metrics:
        size_cat = metric.get('scm_metadata', {}).get('size_category', 'unknown')
        if size_cat in size_metrics:
            size_metrics[size_cat].append(metric.get('mean_reward', 0))
    
    # Compute statistics for each category
    analysis = {}
    for category, rewards in size_metrics.items():
        if rewards:
            analysis[category] = {
                'count': len(rewards),
                'mean_reward': float(np.mean(rewards)),
                'std_reward': float(np.std(rewards)),
                'best_reward': float(np.max(rewards)),
                'worst_reward': float(np.min(rewards)),
                'improvement': float(np.mean(rewards[-10:]) - np.mean(rewards[:10])) if len(rewards) >= 20 else 0
            }
        else:
            analysis[category] = {
                'count': 0,
                'mean_reward': 0,
                'std_reward': 0,
                'best_reward': 0,
                'worst_reward': 0,
                'improvement': 0
            }
    
    # Add overall statistics
    all_rewards = [r for rewards in size_metrics.values() for r in rewards]
    if all_rewards:
        analysis['overall'] = {
            'total_episodes': len(all_rewards),
            'mean_reward': float(np.mean(all_rewards)),
            'std_reward': float(np.std(all_rewards)),
            'best_reward': float(np.max(all_rewards)),
            'improvement': float(np.mean(all_rewards[-20:]) - np.mean(all_rewards[:20])) if len(all_rewards) >= 40 else 0
        }
    
    # Add generation statistics
    analysis['generation_stats'] = generator.get_statistics()
    
    return analysis


def main():
    """Main training function."""
    
    parser = argparse.ArgumentParser(description="Train GRPO with diverse SCMs")
    parser.add_argument('--episodes', type=int, default=200,
                        help='Number of training episodes')
    parser.add_argument('--min-vars', type=int, default=3,
                        help='Minimum number of variables in SCMs')
    parser.add_argument('--max-vars', type=int, default=99,
                        help='Maximum number of variables in SCMs')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test with 10 episodes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable WandB logging')
    parser.add_argument('--checkpoint-freq', type=int, default=50,
                        help='Save checkpoint every N episodes')
    
    args = parser.parse_args()
    
    # Override episodes for quick test
    if args.quick_test:
        args.episodes = 10
        args.checkpoint_freq = 5
        logger.info("Running quick test with 10 episodes")
    
    # Create configuration
    logger.info("\n" + "="*70)
    logger.info("GRPO TRAINING WITH DIVERSE SCMs")
    logger.info("="*70)
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"SCM Variables: {args.min_vars}-{args.max_vars}")
    logger.info(f"Seed: {args.seed}")
    logger.info("="*70 + "\n")
    
    config = create_diverse_training_config(
        max_episodes=args.episodes,
        min_vars=args.min_vars,
        max_vars=args.max_vars,
        verbose=args.verbose,
        enable_wandb=args.wandb,
        seed=args.seed
    )
    
    # Log critical settings
    logger.info("Critical settings from debugging:")
    logger.info(f"  - Policy: {config['policy_architecture']}")
    logger.info(f"  - Fixed std: {config['fixed_std']}")
    logger.info(f"  - Learning rate: {config['learning_rate']}")
    logger.info(f"  - GRPO group size: {config['grpo_config']['group_size']}")
    logger.info(f"  - Target weight: {config['grpo_reward_config']['target_improvement_weight']}")
    logger.info("")
    
    # Initialize WandB if enabled
    wandb_manager = None
    if args.wandb:
        logger.info("Initializing WandB...")
        try:
            wandb_manager = WandBManager()
            wandb_run = wandb_manager.setup(config, experiment_name=f"diverse_{args.min_vars}to{args.max_vars}")
            if wandb_run:
                logger.info(f"✅ WandB initialized: {wandb_run.url if hasattr(wandb_run, 'url') else 'running'}")
            else:
                logger.warning("⚠️  WandB initialization failed, continuing without logging")
                wandb_manager = None
        except Exception as e:
            logger.warning(f"⚠️  WandB initialization failed: {e}")
            wandb_manager = None
    
    # Initialize trainer
    logger.info("Initializing UnifiedGRPOTrainer...")
    trainer = UnifiedGRPOTrainer(config=config)
    logger.info("✅ Trainer initialized\n")
    
    # Verify configuration
    config_ok = verify_trainer_configuration(trainer, config)
    if not config_ok and not args.quick_test:
        response = input("\n⚠️  Configuration may not be optimal. Continue? (y/n): ")
        if response.lower() != 'y':
            logger.info("Training cancelled")
            if wandb_manager:
                wandb_manager.finish()
            return 1
    
    # Create SCM generator
    logger.info("Creating diverse SCM generator...")
    scm_generator = DiverseSCMGenerator(
        min_vars=args.min_vars,
        max_vars=args.max_vars,
        seed=args.seed
    )
    logger.info(f"✅ Generator ready for {args.min_vars}-{args.max_vars} variable SCMs\n")
    
    # Start training
    logger.info("\n" + "="*70)
    logger.info("STARTING TRAINING")
    logger.info("="*70 + "\n")
    
    try:
        # Train with the generator as a callable
        results = trainer.train(scms=scm_generator)
        
        # Analyze results by size category
        if 'episode_metrics' in results:
            logger.info("\n" + "="*70)
            logger.info("ANALYZING RESULTS BY SCM SIZE")
            logger.info("="*70)
            
            analysis = analyze_results_by_size(results['episode_metrics'], scm_generator)
            
            # Print analysis
            for category in ['small', 'medium', 'large', 'very_large']:
                if category in analysis and analysis[category]['count'] > 0:
                    stats = analysis[category]
                    logger.info(f"\n{category.upper()} SCMs:")
                    logger.info(f"  Episodes: {stats['count']}")
                    logger.info(f"  Mean reward: {stats['mean_reward']:.4f} ± {stats['std_reward']:.4f}")
                    logger.info(f"  Best reward: {stats['best_reward']:.4f}")
                    logger.info(f"  Improvement: {stats['improvement']:.4f}")
            
            if 'overall' in analysis:
                overall = analysis['overall']
                logger.info(f"\nOVERALL:")
                logger.info(f"  Total episodes: {overall['total_episodes']}")
                logger.info(f"  Mean reward: {overall['mean_reward']:.4f} ± {overall['std_reward']:.4f}")
                logger.info(f"  Best reward: {overall['best_reward']:.4f}")
                logger.info(f"  Improvement: {overall['improvement']:.4f}")
            
            # Add analysis to results
            results['size_analysis'] = analysis
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results/diverse_training")
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"diverse_{args.min_vars}to{args.max_vars}_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            serializable_results = make_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"\n✅ Training complete! Results saved to {results_file}")
        
        # Log final summary to WandB
        if wandb_manager and wandb_manager.enabled:
            if 'size_analysis' in results and 'overall' in results['size_analysis']:
                final_summary = {
                    'final_mean_reward': results['size_analysis']['overall']['mean_reward'],
                    'final_improvement': results['size_analysis']['overall']['improvement'],
                    'total_episodes': results['size_analysis']['overall']['total_episodes'],
                    'scm_diversity': len(scm_generator.structure_counts)
                }
                wandb_manager.log(final_summary)
                logger.info("📊 Final metrics logged to WandB")
        
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
    
    # Print generation statistics
    gen_stats = scm_generator.get_statistics()
    logger.info("\n" + "="*70)
    logger.info("SCM GENERATION STATISTICS")
    logger.info("="*70)
    logger.info(f"Total SCMs generated: {gen_stats['total_generated']}")
    logger.info("\nStructure distribution:")
    for structure, count in gen_stats['structure_distribution'].items():
        logger.info(f"  {structure}: {count}")
    logger.info("\nSize distribution:")
    for size, count in gen_stats['size_distribution'].items():
        logger.info(f"  {size}: {count}")
    logger.info("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())