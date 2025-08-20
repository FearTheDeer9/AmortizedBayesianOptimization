#!/usr/bin/env python3
"""
Production GRPO Training with Curriculum Learning

Based on working test_scaled_training.py with curriculum factory integration.
"""

import sys
import os
from pathlib import Path
import numpy as np
import time
import pickle
import json
from typing import Dict, Any, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.training.joint_acbo_trainer import JointACBOTrainer
from src.causal_bayes_opt.training.curriculum_factory import SCMCurriculumFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionTrainer(JointACBOTrainer):
    """Production trainer with curriculum tracking and checkpointing."""
    
    def __init__(self, config):
        super().__init__(config=config)
        self.episode_performances = []
        self.intervention_ranges = []
        self.curriculum_levels = []
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def _run_collaborative_episode(self, episode_idx):
        """Override to track per-episode performance."""
        # Track intervention values this episode
        self.current_intervention_values = []
        
        # Run normal episode
        result = super()._run_collaborative_episode(episode_idx)
        
        # Record episode performance
        if hasattr(self, 'episode_rewards'):
            avg_reward = np.mean(self.episode_rewards[-20:]) if len(self.episode_rewards) >= 20 else 0
            self.episode_performances.append(avg_reward)
            
        # Record intervention range usage
        if self.current_intervention_values:
            value_range = max(self.current_intervention_values) - min(self.current_intervention_values)
            self.intervention_ranges.append(value_range)
            
        # Track curriculum level
        if hasattr(self, 'current_scm_metadata'):
            level = self.current_scm_metadata.get('curriculum_level', 
                    self.current_scm_metadata.get('level', 0))
            self.curriculum_levels.append(level)
            
        return result
    
    def _select_best_grpo_intervention(self, candidates):
        """Track intervention values."""
        best = super()._select_best_grpo_intervention(candidates)
        
        if 'value' in best:
            self.current_intervention_values.append(best['value'])
            
        return best
    
    def save_checkpoint(self, episode: int, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'episode': episode,
            'policy_params': self.policy_params,
            'surrogate_params': self.surrogate_params,
            'episode_performances': self.episode_performances,
            'intervention_ranges': self.intervention_ranges,
            'curriculum_levels': self.curriculum_levels,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_ep{episode}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Save as best if specified
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pkl"
            with open(best_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            logger.info(f"Saved best model at episode {episode}")
        
        # Always save as latest
        latest_path = self.checkpoint_dir / "latest.pkl"
        with open(latest_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Saved checkpoint at episode {episode}")


def create_config(config_name: str = "production") -> Dict[str, Any]:
    """Create configuration based on working settings."""
    
    base_config = {
        # Episode settings (from working scripts)
        'obs_per_episode': 10,
        'max_interventions': 30,
        
        # CRITICAL: Use exact architecture that works
        'policy_architecture': 'simple_permutation_invariant',
        'use_fixed_std': True,
        'fixed_std': 0.5,
        
        # Training settings - disable phase rotation for policy-only
        'episodes_per_phase': 100000,  # Very high to effectively disable phase switching
        'use_surrogate': True,
        'use_grpo_rewards': True,
        
        'learning_rate': 5e-4,
        
        # GRPO config (from working scripts)
        'grpo_config': {
            'group_size': 10,
            'entropy_coefficient': 0.001,
            'clip_ratio': 0.2,
            'gradient_clip': 1.0,
            'ppo_epochs': 4
        },
        
        # Joint training (exact weights that work)
        'joint_training': {
            'initial_phase': 'policy',  # Start with policy training
            'loss_weights': {
                'policy': {
                    'target_delta': 0.9,  # Heavy focus on target minimization
                    'direct_parent': 0.1,
                    'information_gain': 0.0  # Disabled based on debugging
                }
            },
            # Disable adaptive features for consistent training
            'adaptive': {
                'use_adaptive_weights': False,
                'use_performance_rotation': False
            }
        },
        
        'verbose': False
    }
    
    # Config variants
    if config_name == "production":
        base_config.update({
            'max_episodes': 1000,
            'checkpoint_every': 50,
            'checkpoint_dir': 'checkpoints/production',
            'curriculum_start_level': 1,
            'curriculum_max_level': 10
        })
    elif config_name == "quick_test":
        base_config.update({
            'max_episodes': 50,
            'checkpoint_every': 10,
            'checkpoint_dir': 'checkpoints/quick_test',
            'curriculum_start_level': 1,
            'curriculum_max_level': 3
        })
    elif config_name == "debug":
        base_config.update({
            'max_episodes': 10,
            'checkpoint_every': 5,
            'checkpoint_dir': 'checkpoints/debug',
            'curriculum_start_level': 1,
            'curriculum_max_level': 2,
            'verbose': True
        })
    
    return base_config


def run_training(config_name: str = "production"):
    """Run production training with curriculum."""
    
    print("\n" + "="*80)
    print(f"PRODUCTION GRPO TRAINING - {config_name.upper()}")
    print("="*80)
    
    # Get configuration
    config = create_config(config_name)
    
    print("\n📊 Configuration:")
    print(f"  Episodes: {config['max_episodes']}")
    print(f"  Interventions per episode: {config['max_interventions']}")
    print(f"  Curriculum levels: {config['curriculum_start_level']}-{config['curriculum_max_level']}")
    print(f"  Architecture: {config['policy_architecture']}")
    print(f"  Fixed std: {config['fixed_std']}")
    
    # Create curriculum factory
    print("\n🎓 Initializing curriculum...")
    curriculum = SCMCurriculumFactory(
        start_level=config['curriculum_start_level'],
        max_level=config['curriculum_max_level'],
        mode="progressive",
        seed=42
    )
    
    # Create trainer
    print("🚀 Initializing trainer...")
    trainer = ProductionTrainer(config)
    
    # Training loop with checkpointing
    print("\n" + "-"*80)
    print("Starting training...")
    print("-"*80)
    
    start_time = time.time()
    best_performance = float('-inf')
    
    try:
        # Run training - pass curriculum factory directly
        # (JointACBOTrainer will wrap it in AdaptiveSCMGenerator)
        results = trainer.train(scms=curriculum)
        
        # Track best model
        for episode in range(0, config['max_episodes'], config['checkpoint_every']):
            if episode < len(trainer.episode_performances):
                recent_perf = np.mean(trainer.episode_performances[max(0, episode-10):episode+1])
                
                # Save checkpoint
                is_best = recent_perf > best_performance
                if is_best:
                    best_performance = recent_perf
                
                trainer.save_checkpoint(episode, is_best=is_best)
                
                # Log progress
                if episode % 100 == 0:
                    current_level = trainer.curriculum_levels[-1] if trainer.curriculum_levels else 1
                    print(f"\nEpisode {episode}: Avg reward = {recent_perf:.3f}, Level = {current_level}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    
    training_time = time.time() - start_time
    
    # Final checkpoint
    trainer.save_checkpoint(config['max_episodes'], is_best=False)
    
    # Analyze results
    print("\n" + "="*80)
    print("TRAINING RESULTS")
    print("="*80)
    
    if trainer.episode_performances:
        print("\n📈 Performance Trajectory:")
        perfs = trainer.episode_performances
        
        # Show progression
        for i in range(0, len(perfs), max(1, len(perfs)//10)):
            end_idx = min(i+10, len(perfs))
            avg = np.mean(perfs[i:end_idx])
            level = trainer.curriculum_levels[i] if i < len(trainer.curriculum_levels) else "?"
            print(f"  Episodes {i+1}-{end_idx}: avg = {avg:.3f}, level = {level}")
        
        # Overall improvement
        early_avg = np.mean(perfs[:10]) if len(perfs) >= 10 else perfs[0] if perfs else 0
        late_avg = np.mean(perfs[-10:]) if len(perfs) >= 10 else perfs[-1] if perfs else 0
        improvement = ((late_avg - early_avg) / abs(early_avg)) * 100 if early_avg != 0 else 0
        
        print(f"\n✨ Overall Performance:")
        print(f"  Early (first 10): {early_avg:.3f}")
        print(f"  Late (last 10): {late_avg:.3f}")
        print(f"  Improvement: {improvement:+.1f}%")
    
    # Curriculum progression
    if trainer.curriculum_levels:
        print(f"\n🎓 Curriculum Progression:")
        print(f"  Starting level: {trainer.curriculum_levels[0]}")
        print(f"  Final level: {trainer.curriculum_levels[-1]}")
        print(f"  Max level reached: {max(trainer.curriculum_levels)}")
    
    print(f"\n⏱️ Training time: {training_time:.1f} seconds")
    print(f"   ({training_time/config['max_episodes']:.1f} sec/episode)")
    
    # Save summary
    summary = {
        'config_name': config_name,
        'total_episodes': config['max_episodes'],
        'training_time': training_time,
        'final_performance': late_avg if trainer.episode_performances else 0,
        'improvement': improvement if trainer.episode_performances else 0,
        'max_curriculum_level': max(trainer.curriculum_levels) if trainer.curriculum_levels else 1,
        'episode_performances': trainer.episode_performances[-100:] if trainer.episode_performances else []
    }
    
    summary_path = Path(config['checkpoint_dir']) / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved to {config['checkpoint_dir']}")
    
    return trainer, results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Production GRPO Training')
    parser.add_argument('--config', type=str, default='quick_test',
                       choices=['production', 'quick_test', 'debug'],
                       help='Configuration to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Run training
    trainer, results = run_training(args.config)
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()