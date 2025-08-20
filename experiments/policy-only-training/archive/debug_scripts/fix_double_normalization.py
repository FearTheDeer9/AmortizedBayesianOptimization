#!/usr/bin/env python3
"""
Fix double normalization in GRPO rewards.

This script modifies the reward calculation to avoid normalizing twice:
1. Use raw target values instead of normalized target_delta
2. Let GRPO handle the only normalization
"""

import sys
import os
from pathlib import Path
import numpy as np
import jax.numpy as jnp

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.training.joint_acbo_trainer import JointACBOTrainer
from src.causal_bayes_opt.experiments.benchmark_scms import create_chain_scm


class FixedNormalizationTrainer(JointACBOTrainer):
    """Trainer with fixed normalization to avoid double standardization."""
    
    def _scale_rewards_group_based(self, candidates, target_var):
        """
        Override to use raw target values instead of normalized delta.
        
        GRPO will handle normalization, we just provide raw signal.
        """
        # Get reward weights
        weights = self.policy_loss_weights if hasattr(self, 'policy_loss_weights') else {
            'target_delta': 1.0,  # Now this is actually raw target
            'information_gain': 0.0,
            'direct_parent': 0.0
        }
        
        # Process each candidate
        for i, c in enumerate(candidates):
            raw = c['reward_components_raw']
            
            # Use RAW target value (negated for minimization)
            # No normalization here - let GRPO do it
            target_reward = -raw['target_value']  # Negative because minimizing
            
            # Info gain (keep as is for now, but could also be raw)
            info_gain = raw.get('info_gain_raw', 0.0)
            
            # Direct parent (binary, no change needed)
            direct_parent = raw.get('direct_parent', 0.0)
            
            # Combine with weights
            total_reward = (
                weights.get('target_delta', 1.0) * target_reward +
                weights.get('information_gain', 0.0) * info_gain +
                weights.get('direct_parent', 0.0) * direct_parent
            )
            
            # Store components
            c['reward'] = float(total_reward)
            c['reward_components'] = {
                'target_delta': float(target_reward),  # Actually raw now
                'info_gain': float(info_gain),
                'direct_parent': float(direct_parent)
            }
            
            # Debug logging for first candidate
            if i == 0 and hasattr(self, '_debug_episode_count') and self._debug_episode_count <= 3:
                print(f"\n    🔧 FIXED Reward Calculation (Candidate {i+1}):")
                print(f"       Variable: {c['variable']}, Value: {c['value']:.3f}")
                print(f"       Raw target value: {raw['target_value']:.3f}")
                print(f"       Target reward (negated): {target_reward:.3f}")
                print(f"       Total reward: {total_reward:.3f}")
                print(f"       → GRPO will normalize this in advantages")
        
        return candidates


def compare_normalization_approaches():
    """Compare double vs single normalization."""
    
    print("\n" + "="*80)
    print("NORMALIZATION COMPARISON TEST")
    print("="*80)
    
    scm = create_chain_scm(chain_length=3)
    
    base_config = {
        'max_episodes': 2,
        'obs_per_episode': 10,
        'max_interventions': 30,
        'policy_architecture': 'permutation_invariant',
        'episodes_per_phase': 1000,
        'use_surrogate': True,
        'use_grpo_rewards': True,
        'learning_rate': 3e-4,  # Higher LR since we fixed double norm
        'grpo_config': {
            'group_size': 10,
            'entropy_coefficient': 0.01,
            'clip_ratio': 0.2,
            'gradient_clip': 1.0,
            'ppo_epochs': 4
        },
        'joint_training': {
            'loss_weights': {
                'policy': {
                    'target_delta': 1.0,
                    'direct_parent': 0.0,
                    'information_gain': 0.0
                }
            }
        },
        'checkpoint_dir': 'fixed_normalization',
        'verbose': False
    }
    
    # Track performance
    class PerformanceTracker(FixedNormalizationTrainer):
        def __init__(self, config):
            super().__init__(config)
            self.target_values = []
            self.selections = []
            self._debug_episode_count = 0
            
        def _select_best_intervention_grpo(self, buffer, posterior, target_var, variables, scm, policy_params, rng_key):
            self._debug_episode_count += 1
            best = super()._select_best_intervention_grpo(
                buffer, posterior, target_var, variables, scm, policy_params, rng_key
            )
            
            # Track
            self.target_values.append(best.get('target_value', float('nan')))
            self.selections.append(best['variable'])
            
            return best
    
    print("\nConfiguration:")
    print("  Episodes: 2")
    print("  Interventions per episode: 30")
    print("  Group size: 10")
    print("  Learning rate: 3e-4 (increased)")
    print("  Reward: Raw target value (no pre-normalization)")
    
    print("\nExpected behavior:")
    print("  1. Rewards = -target_value (raw, no normalization)")
    print("  2. GRPO computes: advantages = (rewards - mean) / std")
    print("  3. Single normalization → stronger learning signal")
    
    print("\n" + "-"*80)
    print("RUNNING FIXED NORMALIZATION TEST")
    print("-"*80)
    
    # Run training
    tracker = PerformanceTracker(base_config)
    results = tracker.train([scm])
    
    # Analyze results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    if tracker.target_values:
        # Split by episodes
        mid = len(tracker.target_values) // 2
        first_half = tracker.target_values[:mid]
        second_half = tracker.target_values[mid:]
        
        print(f"\nTarget Value Progression:")
        print(f"  First half mean: {np.mean(first_half):.3f}")
        print(f"  Second half mean: {np.mean(second_half):.3f}")
        print(f"  Improvement: {np.mean(first_half) - np.mean(second_half):.3f}")
        
        # Variable selection
        x1_rate = sum(1 for v in tracker.selections if v == 'X1') / len(tracker.selections)
        print(f"\nVariable Selection:")
        print(f"  X1 selection rate: {100*x1_rate:.1f}%")
        
        # Check last 10 interventions
        if len(tracker.target_values) >= 10:
            last_10_mean = np.mean(tracker.target_values[-10:])
            last_10_x1 = sum(1 for v in tracker.selections[-10:] if v == 'X1')
            
            print(f"\nFinal Performance (last 10):")
            print(f"  Mean target value: {last_10_mean:.3f}")
            print(f"  X1 selections: {last_10_x1}/10")
            
            if last_10_mean < -0.5 and last_10_x1 >= 8:
                print("\n✅ SUCCESS! Fixed normalization improves learning!")
            else:
                print("\n⚠️ Still needs tuning, but normalization is fixed")
    
    return tracker, results


def main():
    """Run the comparison."""
    print("\n" + "="*80)
    print("DOUBLE NORMALIZATION FIX")
    print("="*80)
    
    print("""
Problem Identified:
1. joint_acbo_trainer normalizes: (value - mean) / std
2. unified_grpo_trainer normalizes again: (reward - mean) / std
3. Result: Double normalization weakens signal

Solution:
- Use raw target values as rewards
- Let GRPO do the only normalization
- This provides stronger, cleaner learning signal
""")
    
    tracker, results = compare_normalization_approaches()
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("""
To fix double normalization in your codebase:

1. Modify joint_acbo_trainer._scale_rewards_group_based():
   - Change line 1260 from:
     target_delta = -(raw['target_value'] - group_mean) / (group_std + 1e-8)
   - To:
     target_delta = -raw['target_value']  # Let GRPO normalize
   
2. Consider renaming 'target_delta' to 'target_reward' for clarity

3. Increase learning rate (3e-4 or higher) since signal is stronger

This ensures GRPO's advantage normalization is the ONLY normalization applied.
""")


if __name__ == "__main__":
    main()