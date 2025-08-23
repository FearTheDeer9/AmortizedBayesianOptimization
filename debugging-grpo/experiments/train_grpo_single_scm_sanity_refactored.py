#!/usr/bin/env python3
"""
Single SCM sanity check using the refactored modular architecture.

This version uses:
- PureGRPOTrainer (focused only on policy training)
- JointOrchestrator (handles coordination)
- Proper separation of concerns

Should produce the same results as the original monolithic trainer.
"""

import os
import sys
import json
import argparse
import logging
import csv
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import jax.random as random

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Use new modular components
from src.training.pure_grpo_trainer import PureGRPOTrainer, PureGRPOConfig
from src.training.joint_orchestrator import JointOrchestrator, JointTrainingConfig

# Keep original SCM and data structure imports
from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
from src.causal_bayes_opt.data_structures.scm import get_parents, get_variables, get_target
from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
from src.causal_bayes_opt.environments.sampling import sample_with_intervention
from src.causal_bayes_opt.training.utils.wandb_setup import WandBManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_deterministic_test_scm():
    """Create deterministic test SCM with structure: X -> Y, Z (isolated)"""
    from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm
    from src.causal_bayes_opt.data_structures.scm import create_scm
    from src.causal_bayes_opt.mechanisms.linear import create_linear_mechanism
    import pyrsistent as pyr
    
    # Define the specific structure we want
    variables = frozenset(['X', 'Y', 'Z'])
    edges = frozenset([('X', 'Y')])  # Only X causes Y, Z isolated
    target = 'Y'
    
    # Create mechanisms
    mechanisms = {}
    
    # X mechanism: root variable with noise
    mechanisms['X'] = create_linear_mechanism(
        parents=[],
        coefficients={},
        intercept=0.0,
        noise_scale=1.0
    )
    
    # Y mechanism: Y = 10*X (deterministic, no noise)
    mechanisms['Y'] = create_linear_mechanism(
        parents=['X'],
        coefficients={'X': 10.0},
        intercept=0.0,
        noise_scale=0.0
    )
    
    # Z mechanism: isolated root variable with noise
    mechanisms['Z'] = create_linear_mechanism(
        parents=[],
        coefficients={},
        intercept=0.0,
        noise_scale=1.0
    )
    
    # Create SCM with metadata
    metadata = {
        'structure_type': 'deterministic_test',
        'variable_ranges': {
            'X': (-100.0, 100.0),
            'Y': (-1000.0, 1000.0),
            'Z': (-10.0, 10.0)
        },
        'coefficients': {('X', 'Y'): 10.0},
        'noise_scales': {'X': 1.0, 'Y': 0.0, 'Z': 1.0},
        'description': 'Deterministic test: Y = 10*X, Z isolated',
        'expected_behavior': 'X interventions strongly affect Y, Z interventions do nothing',
        'optimal_strategy': 'Minimize Y by setting X = -100 → Y = -1000'
    }
    
    scm = create_scm(
        variables=variables,
        edges=edges,
        mechanisms=mechanisms,
        target=target,
        metadata=metadata
    )
    
    logger.info("Created deterministic test SCM: X -> Y (coeff=10.0), Z isolated")
    logger.info(f"  Structure: Y = 10*X (no noise), Z isolated")
    logger.info(f"  Expected: X interventions have huge impact, Z interventions have no impact")
    logger.info(f"  Optimal: X = -100 → Y = -1000 (theoretical minimum)")
    
    return scm


class RefactoredTrainer:
    """
    Adapter class to provide the same interface as DiverseGRPOTrainer
    but using the new modular architecture.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Create pure GRPO trainer
        self.policy_trainer = PureGRPOTrainer(config)
        
        # Create joint orchestrator
        joint_config = JointTrainingConfig(
            episodes_per_phase=config.get('episodes_per_phase', 999999),
            initial_phase='policy',
            max_episodes=config.get('max_episodes', 2)
        )
        
        self.orchestrator = JointOrchestrator(
            policy_trainer=self.policy_trainer,
            surrogate_trainer=None,  # No surrogate for pure GRPO
            config=joint_config
        )
        
        # Episode tracking for compatibility
        self.episode_performances = []
        self.convergence_metrics = {
            'rolling_averages': [],
            'best_per_category': {'single': float('inf')}
        }
        
        logger.info("Initialized RefactoredTrainer with modular architecture")
    
    def train(self, scms):
        """Train using modular architecture."""
        results = self.orchestrator.train(scms)
        
        # Convert to compatible format
        for episode_data in results['all_metrics']:
            self.episode_performances.append({
                'episode': episode_data['episode'],
                'size_category': 'single',
                'num_vars': episode_data['n_variables'],
                'mean_target': episode_data['mean_reward'],
                'best_target': episode_data['mean_reward'],
                'parent_selection_rate': 0.5  # Placeholder
            })
        
        return results
    
    def _track_episode_performance(self, episode_idx):
        """Compatibility method - no-op since tracking happens in train()."""
        pass


def create_single_scm_config(
    max_episodes: int = 100,
    verbose: bool = False,
    enable_wandb: bool = False,
    seed: int = 42
) -> Dict[str, Any]:
    """Create configuration for single SCM sanity check using modular architecture."""
    
    config = {
        # Core episode settings
        'max_episodes': max_episodes,
        'obs_per_episode': 10,
        'max_interventions': 5,  # Reduced for faster testing
        
        # CRITICAL: Disable phase switching for pure GRPO
        'episodes_per_phase': 999999,
        
        # CRITICAL: Use simplified_permutation_invariant
        'policy_architecture': 'simplified_permutation_invariant',
        
        # CRITICAL: Pure GRPO training - no surrogate!
        'use_surrogate': False,
        'use_grpo_rewards': True,
        
        # CRITICAL: Fixed std for exploration
        'use_fixed_std': True,
        'fixed_std': 1.0,
        
        # CRITICAL: Higher learning rate
        'learning_rate': 1e-1,
        
        # GRPO configuration
        'grpo_config': {
            'group_size': 8,  # Smaller for faster testing
            'entropy_coefficient': 0.001,
            'clip_ratio': 1.0,
            'gradient_clip': 10.0,
            'ppo_epochs': 4,
            'normalize_advantages': False
        },
        
        # Reward weights - using binary reward system
        'reward_weights': {
            'target': 1.0,
            'parent': 0.0,
            'info_gain': 0.0
        },
        
        # Use binary reward: +1 if above mean, -1 if below mean
        'reward_type': 'binary',
        
        # General settings
        'batch_size': 8,
        'seed': seed,
        'verbose': verbose,
        'checkpoint_dir': 'debugging-grpo/checkpoints/single_scm_sanity',
        
        # WandB logging configuration
        'logging': {
            'wandb': {
                'enabled': enable_wandb,
                'project': 'causal-bayes-opt-grpo-refactored',
                'name': f'refactored_single_scm_{max_episodes}ep',
                'tags': ['single_scm', 'grpo', 'refactored', 'binary_reward'],
                'log_frequency': 1
            }
        }
    }
    
    return config


def main():
    """Main training function using refactored architecture."""
    
    parser = argparse.ArgumentParser(description="Single SCM GRPO sanity check (refactored)")
    parser.add_argument('--episodes', type=int, default=2,
                        help='Number of training episodes')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable WandB logging')
    
    args = parser.parse_args()
    
    # Create configuration
    logger.info("\n" + "="*70)
    logger.info("REFACTORED SINGLE SCM GRPO SANITY CHECK")
    logger.info("="*70)
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"Architecture: Modular (PureGRPOTrainer + JointOrchestrator)")
    logger.info(f"Reward Type: Binary (+1 if above mean, -1 if below mean)")
    logger.info(f"Seed: {args.seed}")
    logger.info("="*70 + "\n")
    
    config = create_single_scm_config(
        max_episodes=args.episodes,
        verbose=args.verbose,
        enable_wandb=args.wandb,
        seed=args.seed
    )
    
    # Log critical settings
    logger.info("Critical settings:")
    logger.info(f"  - Architecture: Modular (PureGRPOTrainer + JointOrchestrator)")
    logger.info(f"  - Policy: {config['policy_architecture']}")
    logger.info(f"  - Reward type: {config['reward_type']}")
    logger.info(f"  - Fixed std: {config['fixed_std']}")
    logger.info(f"  - Learning rate: {config['learning_rate']}")
    logger.info(f"  - GRPO group size: {config['grpo_config']['group_size']}")
    logger.info("")
    
    # Create deterministic test SCM
    logger.info("Creating deterministic test SCM...")
    fixed_scm = create_deterministic_test_scm()
    
    # Log SCM details
    target_var = get_target(fixed_scm)
    parents = get_parents(fixed_scm, target_var)
    variables = get_variables(fixed_scm)
    logger.info(f"✅ Created SCM: variables={variables}, target={target_var}, parents={parents}")
    
    # Initialize refactored trainer
    logger.info("\nInitializing refactored GRPO trainer...")
    trainer = RefactoredTrainer(config=config)
    logger.info("✅ Refactored trainer initialized")
    
    # Start training
    logger.info("\n" + "="*70)
    logger.info("STARTING REFACTORED TRAINING")
    logger.info("="*70 + "\n")
    
    try:
        # Train with the fixed SCM
        results = trainer.train(scms=lambda: fixed_scm)
        
        # Analyze results
        logger.info("\n" + "="*70)
        logger.info("ANALYZING REFACTORED RESULTS")
        logger.info("="*70)
        
        all_metrics = trainer.episode_performances
        if all_metrics:
            final_episodes = all_metrics[-2:] if len(all_metrics) >= 2 else all_metrics
            
            analysis = {
                'total_episodes': len(all_metrics),
                'final_mean_target': np.mean([m['mean_target'] for m in final_episodes]),
                'best_target_overall': np.min([m['best_target'] for m in all_metrics]),
                'architecture': 'Modular (PureGRPOTrainer + JointOrchestrator)',
                'reward_system': 'Binary (+1/-1 based on mean)'
            }
            
            logger.info(f"\nREFACTORED PERFORMANCE:")
            logger.info(f"  Architecture: {analysis['architecture']}")
            logger.info(f"  Reward system: {analysis['reward_system']}")
            logger.info(f"  Total episodes: {analysis['total_episodes']}")
            logger.info(f"  Final mean target: {analysis['final_mean_target']:.3f}")
            logger.info(f"  Best target overall: {analysis['best_target_overall']:.3f}")
            
            # Success criteria check
            logger.info(f"\nSUCCESS CRITERIA CHECK:")
            target_success = analysis['best_target_overall'] < 0.0  # Any negative value
            
            logger.info(f"  Target optimization (< 0.0): {'✅' if target_success else '❌'} {analysis['best_target_overall']:.3f}")
            
            if target_success:
                logger.info(f"  🎉 REFACTORED SANITY CHECK PASSED!")
            else:
                logger.info(f"  ⚠️ REFACTORED SANITY CHECK NEEDS INVESTIGATION")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("debugging-grpo/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"refactored_sanity_{timestamp}.json"
        
        # Prepare serializable results
        save_data = {
            'config': config,
            'architecture': 'Modular (PureGRPOTrainer + JointOrchestrator)',
            'scm_details': {
                'variables': list(get_variables(fixed_scm)),
                'target': get_target(fixed_scm),
                'parents': list(get_parents(fixed_scm, get_target(fixed_scm))),
                'structure_type': 'deterministic_test',
                'num_variables': 3
            },
            'analysis': analysis if 'analysis' in locals() else {},
            'episode_performances': trainer.episode_performances,
            'training_time': results.get('training_time', 0),
            'policy_episodes': results.get('policy_episodes', 0),
            'surrogate_episodes': results.get('surrogate_episodes', 0)
        }
        
        with open(results_file, 'w') as f:
            # Simple serialization
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert(v) for v in obj]
                return obj
            
            json.dump(convert(save_data), f, indent=2)
        
        logger.info(f"\n✅ Refactored sanity check complete! Results saved to {results_file}")
        
        # Compare with original
        logger.info(f"\n" + "="*70)
        logger.info("COMPARISON GUIDE")
        logger.info("="*70)
        logger.info("To compare with original implementation:")
        logger.info("1. Run: python experiments/policy-only-training/train_grpo_single_scm_sanity.py --episodes 2")
        logger.info("2. Look for '[BINARY TARGET REWARD]' logs in both")
        logger.info("3. Compare final target values and learning behavior")
        logger.info("4. Both should show similar binary reward patterns and target optimization")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())