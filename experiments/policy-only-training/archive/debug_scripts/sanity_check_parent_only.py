#!/usr/bin/env python3
"""
Sanity check: Test if GRPO can learn the trivial case of always selecting X1.

With parent-only rewards:
- direct_parent: 1.0 (all reward)
- target_delta: 0.0 (no reward for outcomes)
- info_gain: 0.0 (no reward for info)

This should result in 100% X1 selection since X1 is the only parent of X2.
If this doesn't work, GRPO implementation is fundamentally broken.
"""

import sys
import os
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.experiments.benchmark_scms import create_chain_scm
from src.causal_bayes_opt.training.joint_acbo_trainer import JointACBOTrainer


class SanityCheckTrainer(JointACBOTrainer):
    """Trainer with detailed tracking for sanity check."""
    
    def __init__(self, config):
        super().__init__(config=config)
        
        # Tracking
        self.all_selections = []
        self.all_rewards = []
        self.episode_x1_rates = []
        
    def _select_best_intervention_grpo(self, buffer, posterior, target_var, variables, scm, policy_params, rng_key):
        """Override to track selections."""
        
        # Call parent method
        best = super()._select_best_intervention_grpo(
            buffer, posterior, target_var, variables, scm, policy_params, rng_key
        )
        
        # Track selection
        self.all_selections.append(best['variable'])
        
        # Print every 10 interventions only if not reaching target
        if len(self.all_selections) % 10 == 0:
            recent = self.all_selections[-10:]
            x1_count = sum(1 for v in recent if v == 'X1')
            if x1_count < 9:  # Only print if not performing well
                print(f"\n📊 Last 10 selections: X1={x1_count}/10 ({100*x1_count/10:.0f}%)")
                print(f"   Selections: {recent}")
        
        return best
    
    def _end_episode(self, episode_idx, scm):
        """Track episode-level metrics."""
        
        # Calculate X1 rate for this episode
        if self.all_selections:
            episode_selections = self.all_selections[-self.config['max_interventions']:]
            x1_rate = sum(1 for v in episode_selections if v == 'X1') / len(episode_selections)
            self.episode_x1_rates.append(x1_rate)
            
            print(f"\n{'='*70}")
            print(f"EPISODE {episode_idx + 1} COMPLETE")
            print(f"X1 selection rate: {100*x1_rate:.1f}%")
            
            # Show trend if we have multiple episodes
            if len(self.episode_x1_rates) > 1:
                print(f"Episode progression: {[f'{100*r:.0f}%' for r in self.episode_x1_rates]}")
                
                # Check if improving
                if len(self.episode_x1_rates) >= 3:
                    recent_avg = np.mean(self.episode_x1_rates[-3:])
                    early_avg = np.mean(self.episode_x1_rates[:3])
                    if recent_avg > early_avg + 0.1:
                        print("✅ Learning to prefer X1!")
                    elif recent_avg < early_avg - 0.1:
                        print("❌ Moving away from X1")
                    else:
                        print("⚠️ No clear learning trend")
        
        super()._end_episode(episode_idx, scm)


def run_sanity_check():
    """Run the parent-only reward sanity check."""
    
    print("\n" + "="*80)
    print("GRPO SANITY CHECK: PARENT-ONLY REWARDS")
    print("="*80)
    print("\n🎯 Expected Behavior:")
    print("   - Should ALWAYS select X1 (100% rate)")
    print("   - X1 is the only parent of X2, gets reward=1.0")
    print("   - X0 is not a parent, gets reward=0.0")
    print("   - Values don't matter, only variable selection")
    
    # Create simple 3-node chain SCM
    scm = create_chain_scm(chain_length=3)
    
    config = {
        # Episode configuration
        'max_episodes': 2,  # Just 1 episode for ultra-quick test
        'obs_per_episode': 10,
        'max_interventions': 20,  # Very few interventions for quick test
        
        # Architecture
        'policy_architecture': 'permutation_invariant',
        
        # Keep in policy phase entire time
        'episodes_per_phase': 1000,
        'use_surrogate': True,
        'use_grpo_rewards': True,
        
        # Learning parameters
        'learning_rate': 2e-4,  # 5x higher since PPO epochs aren't actually running
        
        # GRPO configuration
        'grpo_config': {
            'group_size': 20,
            'entropy_coefficient': 0.01,  # Small exploration
            'clip_ratio': 0.2,
            'gradient_clip': 1.0,
            'ppo_epochs': 4
        },
        
        # CRITICAL: Parent-only rewards - use the correct config structure
        'joint_training': {
            'loss_weights': {
                'policy': {
                    'target_delta': 1.0,        # NO reward for outcomes
                    'direct_parent': 0.0,       # ALL reward for parent selection 
                    'information_gain': 0.0,    # NO info gain reward
                    'absolute_target': 1.0      # NO absolute target reward
                }
            }
        },
        
        'checkpoint_dir': 'sanity_check',
        'verbose': False
    }
    
    print(f"\n🔧 Configuration:")
    print(f"  Episodes: {config['max_episodes']}")
    print(f"  Interventions per episode: {config['max_interventions']}")
    print(f"  Group size: {config['grpo_config']['group_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    
    print(f"\n💰 Reward Weights (PARENT ONLY):")
    policy_weights = config['joint_training']['loss_weights']['policy']
    print(f"  target_delta: {policy_weights['target_delta']} (outcomes don't matter)")
    print(f"  direct_parent: {policy_weights['direct_parent']} (X1 gets this, X0 gets 0)")
    print(f"  information_gain: {policy_weights['information_gain']} (no info reward)")
    
    print(f"\n🎯 Success Criteria:")
    print(f"  1. X1 selection rate should quickly reach ~100%")
    print(f"  2. Should see consistent improvement across episodes")
    print(f"  3. Final episodes should have >95% X1 selection")
    
    print(f"\n" + "-"*80)
    print("STARTING SANITY CHECK")
    print("-"*80)
    
    # Create trainer
    trainer = SanityCheckTrainer(config)
    
    # Run training
    results = trainer.train([scm])
    
    # Final analysis
    print("\n" + "="*80)
    print("SANITY CHECK RESULTS")
    print("="*80)
    
    # Overall statistics
    total_selections = len(trainer.all_selections)
    x1_selections = sum(1 for v in trainer.all_selections if v == 'X1')
    overall_x1_rate = x1_selections / total_selections if total_selections > 0 else 0
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total interventions: {total_selections}")
    print(f"  X1 selections: {x1_selections}")
    print(f"  X0 selections: {total_selections - x1_selections}")
    print(f"  Overall X1 rate: {100*overall_x1_rate:.1f}%")
    
    # Episode progression
    if trainer.episode_x1_rates:
        print(f"\n📈 Episode Progression:")
        for i, rate in enumerate(trainer.episode_x1_rates):
            status = "✅" if rate > 0.9 else ("⚠️" if rate > 0.7 else "❌")
            print(f"  Episode {i+1}: {100*rate:.1f}% {status}")
        
        # First vs last comparison
        if len(trainer.episode_x1_rates) >= 2:
            first_half = trainer.episode_x1_rates[:len(trainer.episode_x1_rates)//2]
            second_half = trainer.episode_x1_rates[len(trainer.episode_x1_rates)//2:]
            
            first_avg = np.mean(first_half)
            second_avg = np.mean(second_half)
            
            print(f"\n📊 Learning Analysis:")
            print(f"  First half average: {100*first_avg:.1f}%")
            print(f"  Second half average: {100*second_avg:.1f}%")
            print(f"  Improvement: {100*(second_avg - first_avg):+.1f}%")
    
    # Final verdict
    print(f"\n{'='*80}")
    print("VERDICT")
    print("="*80)
    
    final_rate = trainer.episode_x1_rates[-1] if trainer.episode_x1_rates else 0
    
    if final_rate > 0.95:
        print("✅ SUCCESS! GRPO can learn trivial parent selection.")
        print("   The basic mechanism works. Issues must be in reward calculation.")
    elif final_rate > 0.75:
        print("⚠️ PARTIAL SUCCESS. GRPO shows learning but not converging.")
        print("   May need more episodes or learning rate tuning.")
    else:
        print("❌ FAILURE! GRPO cannot learn even trivial parent selection.")
        print("   There is a fundamental bug in the GRPO implementation.")
        print(f"   Final X1 rate: {100*final_rate:.1f}% (expected >95%)")
    
    return trainer, results


if __name__ == "__main__":
    trainer, results = run_sanity_check()