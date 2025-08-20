#!/usr/bin/env python3
"""
Achieve 100% success rate with proper GRPO configuration.
"""

import sys
import os
from pathlib import Path
import numpy as np
import jax

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.experiments.benchmark_scms import create_chain_scm
from src.causal_bayes_opt.training.joint_acbo_trainer import JointACBOTrainer


class SuccessTracker(JointACBOTrainer):
    """Track success metrics."""
    
    def __init__(self, config):
        super().__init__(config=config)
        self.all_interventions = []
        self.episode_performances = []
        
    def _select_best_intervention_grpo(self, buffer, posterior, target_var, variables, scm, policy_params, rng_key):
        """Track interventions."""
        result = super()._select_best_intervention_grpo(
            buffer, posterior, target_var, variables, scm, policy_params, rng_key
        )
        
        # Track intervention
        self.all_interventions.append({
            'variable': result.get('variable'),
            'value': float(result.get('value', 0)),
            'target_value': float(result.get('target_value', 0))
        })
        
        return result
    
    def _end_episode(self, episode_idx, scm):
        """Track episode performance."""
        if self.all_interventions:
            # Get interventions from this episode
            episode_start = max(0, len(self.all_interventions) - self.config['max_interventions'])
            episode_interventions = self.all_interventions[episode_start:]
            
            # Calculate metrics
            x1_selections = sum(1 for i in episode_interventions if i['variable'] == 'X1')
            x1_values = [i['value'] for i in episode_interventions if i['variable'] == 'X1']
            x2_values = [i['target_value'] for i in episode_interventions]
            
            metrics = {
                'episode': episode_idx + 1,
                'x1_selection_rate': x1_selections / len(episode_interventions) if episode_interventions else 0,
                'mean_x1': np.mean(x1_values) if x1_values else 0,
                'mean_x2': np.mean(x2_values) if x2_values else 0,
                'best_x2': np.min(x2_values) if x2_values else 0,
                'negative_x1_rate': sum(1 for v in x1_values if v < 0) / len(x1_values) if x1_values else 0
            }
            
            self.episode_performances.append(metrics)
            
            # Print progress
            print(f"\n📊 Episode {episode_idx + 1} Performance:")
            print(f"   X1 selection: {100*metrics['x1_selection_rate']:.1f}%")
            print(f"   Mean X1: {metrics['mean_x1']:.3f}")
            print(f"   Mean X2: {metrics['mean_x2']:.3f}")
            print(f"   Best X2: {metrics['best_x2']:.3f}")
            print(f"   Negative X1: {100*metrics['negative_x1_rate']:.1f}%")
        
        super()._end_episode(episode_idx, scm)


def test_for_100_percent():
    """Test with proper configuration for 100% success."""
    
    print("\n" + "="*80)
    print("ACHIEVING 100% SUCCESS RATE")
    print("="*80)
    
    scm = create_chain_scm(chain_length=3)
    
    config = {
        'max_episodes': 10,  # More episodes for convergence
        'obs_per_episode': 10,
        'max_interventions': 30,  # More interventions per episode
        'policy_architecture': 'permutation_invariant',
        'episodes_per_phase': 1000,
        'use_surrogate': True,
        'use_grpo_rewards': True,
        'learning_rate': 2e-5,  # Stable learning rate
        
        # PROPER GRPO CONFIG
        'grpo_config': {
            'group_size': 20,  # Large group for better statistics
            'entropy_coefficient': 0.0,  # No exploration - pure exploitation
            'clip_ratio': 0.2,
            'gradient_clip': 1.0,
            'ppo_epochs': 4
        },
        
        # Focus entirely on target minimization
        'grpo_reward_weights': {
            'target_delta': 1.0,
            'info_gain': 0.0,
            'direct_parent': 0.0
        },
        
        'checkpoint_dir': 'achieve_100',
        'verbose': False
    }
    
    print(f"\n🔧 Configuration:")
    print(f"  Episodes: {config['max_episodes']}")
    print(f"  Interventions per episode: {config['max_interventions']}")
    print(f"  Group size: {config['grpo_config']['group_size']} (FIXED)")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Entropy: {config['grpo_config']['entropy_coefficient']} (pure exploitation)")
    print(f"  Total GRPO updates: {config['max_episodes'] * config['max_interventions']}")
    
    print("\n🎯 Success Criteria:")
    print("  1. X1 selection rate > 95%")
    print("  2. Mean X1 < -1.0")
    print("  3. Mean X2 < -2.0")
    print("  4. >90% negative X1 values")
    print("  5. Consistent improvement across episodes")
    
    trainer = SuccessTracker(config)
    
    print("\n" + "-"*80)
    print("TRAINING PROGRESS")
    print("-"*80)
    
    results = trainer.train([scm])
    
    # Final analysis
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    if trainer.episode_performances:
        # Compare first vs last episodes
        first_3 = trainer.episode_performances[:3]
        last_3 = trainer.episode_performances[-3:]
        
        print("\n📈 Learning Trajectory:")
        print("\nFirst 3 episodes:")
        for m in first_3:
            print(f"  Episode {m['episode']}: X1={m['mean_x1']:.3f}, X2={m['mean_x2']:.3f}, "
                  f"X1 selection={100*m['x1_selection_rate']:.0f}%, "
                  f"Negative X1={100*m['negative_x1_rate']:.0f}%")
        
        print("\nLast 3 episodes:")
        for m in last_3:
            print(f"  Episode {m['episode']}: X1={m['mean_x1']:.3f}, X2={m['mean_x2']:.3f}, "
                  f"X1 selection={100*m['x1_selection_rate']:.0f}%, "
                  f"Negative X1={100*m['negative_x1_rate']:.0f}%")
        
        # Success evaluation
        final_metrics = last_3[-1] if last_3 else trainer.episode_performances[-1]
        
        print("\n" + "="*80)
        print("SUCCESS CRITERIA EVALUATION")
        print("="*80)
        
        criteria = [
            ("X1 selection > 95%", final_metrics['x1_selection_rate'] > 0.95),
            ("Mean X1 < -1.0", final_metrics['mean_x1'] < -1.0),
            ("Mean X2 < -2.0", final_metrics['mean_x2'] < -2.0),
            (">90% negative X1", final_metrics['negative_x1_rate'] > 0.90),
            ("Improvement shown", last_3[-1]['mean_x2'] < first_3[0]['mean_x2'] if first_3 else False)
        ]
        
        for criterion, passed in criteria:
            print(f"  {criterion}: {'✅' if passed else '❌'}")
        
        success_rate = sum(1 for _, p in criteria if p) / len(criteria)
        print(f"\n🎯 Success Rate: {100*success_rate:.0f}%")
        
        if success_rate >= 0.8:
            print("\n🎉 SUCCESS! Near-optimal performance achieved!")
            print("The GRPO mechanism works correctly with proper configuration.")
        elif success_rate >= 0.6:
            print("\n✅ Good progress! More training may achieve 100%.")
        else:
            print("\n⚠️ More debugging needed.")
        
        # Show best single episode
        best_episode = min(trainer.episode_performances, key=lambda x: x['mean_x2'])
        print(f"\n🌟 Best Episode Performance:")
        print(f"  Episode {best_episode['episode']}: Mean X2 = {best_episode['mean_x2']:.3f}")
        
    return trainer


if __name__ == "__main__":
    trainer = test_for_100_percent()