#!/usr/bin/env python3
"""
Simple training with SCM rotation - based on our working test_scaled_training.py
Just adds SCM rotation every N episodes.
"""

import sys
import os
from pathlib import Path
import numpy as np
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.experiments.benchmark_scms import (
    create_chain_scm, create_fork_scm, create_collider_scm,
    create_sparse_scm, create_diamond_scm
)
from src.causal_bayes_opt.training.joint_acbo_trainer import JointACBOTrainer


class RotatingTrainer(JointACBOTrainer):
    """JointACBOTrainer with SCM rotation."""
    
    def __init__(self, config, scm_pool):
        super().__init__(config=config)
        self.scm_pool = scm_pool
        self.current_scm_idx = 0
        self.rotation_interval = config.get('rotation_interval', 10)
        self.episode_performances = []
        self.x1_ranges_by_episode = []
        
        # Set initial SCM
        self.current_scm = self.scm_pool[0]
        print(f"Starting with SCM: {self.current_scm.get('metadata', {}).get('description', 'Unknown')}")
        
    def _run_collaborative_episode(self, episode_idx):
        """Override to handle SCM rotation."""
        # Check if we should rotate
        if episode_idx > 0 and episode_idx % self.rotation_interval == 0:
            self.current_scm_idx = (self.current_scm_idx + 1) % len(self.scm_pool)
            self.current_scm = self.scm_pool[self.current_scm_idx]
            print(f"\n🔄 Rotating to SCM {self.current_scm_idx}: {self.current_scm.get('metadata', {}).get('description', 'Unknown')}")
        
        # Track X1 values this episode
        self.current_x1_values = []
        
        # Run normal episode
        result = super()._run_collaborative_episode(episode_idx)
        
        # Record episode performance
        if hasattr(self, 'episode_rewards'):
            avg_reward = np.mean(self.episode_rewards[-20:]) if len(self.episode_rewards) >= 20 else 0
            self.episode_performances.append(avg_reward)
            
        # Record X1 range usage
        if self.current_x1_values:
            x1_range = max(self.current_x1_values) - min(self.current_x1_values)
            self.x1_ranges_by_episode.append(x1_range)
            
        return result
    
    def _select_best_grpo_intervention(self, candidates):
        """Track X1 values."""
        best = super()._select_best_grpo_intervention(candidates)
        
        if 'X1' in str(best.get('variable', '')):
            self.current_x1_values.append(best['value'])
            
        return best


def run_training_with_rotation():
    """Run training with SCM rotation."""
    
    print("\n" + "="*80)
    print("GRPO TRAINING WITH SCM ROTATION")
    print("="*80)
    
    # Create SCM pool
    scm_pool = [
        create_chain_scm(3),
        create_chain_scm(4),
        create_fork_scm(),
        create_collider_scm(),
        create_sparse_scm(num_vars=5, edge_prob=0.3),
    ]
    
    print(f"\nCreated pool of {len(scm_pool)} SCMs")
    
    # Config based on what was working
    config = {
        'max_episodes': 50,  # Can scale up for production
        'obs_per_episode': 10,
        'max_interventions': 20,
        
        # Use the simplified architecture that was working
        'policy_architecture': 'simple_permutation_invariant',
        'use_fixed_std': True,
        'fixed_std': 0.5,
        
        'episodes_per_phase': 1000,
        'use_surrogate': True,
        'use_grpo_rewards': True,
        
        'learning_rate': 5e-4,
        
        'grpo_config': {
            'group_size': 10,
            'entropy_coefficient': 0.001,
            'clip_ratio': 0.2,
            'gradient_clip': 1.0,
            'ppo_epochs': 4
        },
        
        'joint_training': {
            'policy_loss_weight': 1.0,
            'surrogate_loss_weight': 0.1,
            'info_gain_weight': 0.0,
            'target_weight': 0.9,
            'parent_weight': 0.1,
        },
        
        'rotation_interval': 10,  # Rotate every 10 episodes
        'enable_checkpointing': True,
        'checkpoint_every': 20,
    }
    
    # Create trainer
    trainer = RotatingTrainer(config, scm_pool)
    
    # Run training
    print("\nStarting training...")
    trainer.train()
    
    # Print final results
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    if trainer.episode_performances:
        print(f"\nPerformance progression:")
        print(f"  First 10 episodes: {np.mean(trainer.episode_performances[:10]):.3f}")
        print(f"  Last 10 episodes:  {np.mean(trainer.episode_performances[-10:]):.3f}")
        
    if trainer.x1_ranges_by_episode:
        print(f"\nRange utilization progression:")
        print(f"  First 10 episodes: {np.mean(trainer.x1_ranges_by_episode[:10]):.2f}")
        print(f"  Last 10 episodes:  {np.mean(trainer.x1_ranges_by_episode[-10:]):.2f}")
    
    # Save checkpoint
    if config.get('enable_checkpointing'):
        import pickle
        checkpoint_path = Path("checkpoints/final_model.pkl")
        checkpoint_path.parent.mkdir(exist_ok=True)
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'policy_params': trainer.policy_params,
                'surrogate_params': trainer.surrogate_params,
                'config': config,
                'performances': trainer.episode_performances
            }, f)
        print(f"\nSaved final model to {checkpoint_path}")


if __name__ == "__main__":
    run_training_with_rotation()