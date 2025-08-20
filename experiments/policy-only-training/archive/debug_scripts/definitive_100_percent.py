#!/usr/bin/env python3
"""
Definitive test to achieve 100% success rate on chain SCM with comprehensive diagnostics.
This script addresses all identified issues:
1. Verifies group_size is actually being used
2. Forces continuous policy training (no phase switching)
3. Uses optimal reward weights (0.7 target, 0.3 parent)
4. Tracks detailed convergence metrics
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


class DefinitiveDiagnosticTrainer(JointACBOTrainer):
    """Enhanced trainer with comprehensive diagnostics."""
    
    def __init__(self, config):
        super().__init__(config=config)
        
        # Tracking metrics
        self.all_interventions = []
        self.episode_performances = []
        self.group_sizes_seen = []
        self.phase_history = []
        self.reward_component_history = []
        
        # Per-episode detailed tracking
        self.current_episode_data = {
            'x1_selections': 0,
            'x0_selections': 0,
            'x2_selections': 0,
            'x1_values': [],
            'x2_outcomes': [],
            'rewards': []
        }
        
    def _select_best_intervention_grpo(self, buffer, posterior, target_var, variables, scm, policy_params, rng_key):
        """Override to add comprehensive diagnostics."""
        
        # Track current phase
        self.phase_history.append(self.current_phase)
        
        # Call parent method first to get candidates
        # But we need to intercept the candidate generation
        
        # Prepare state for policy
        buffer_dict = self._prepare_buffer_dict(buffer)
        state = self._prepare_state(buffer_dict, posterior, target_var)
        
        # Generate candidates
        self.rng_key, sample_key = jax.random.split(self.rng_key)
        candidates = []
        
        # DIAGNOSTIC: Track actual group size
        actual_group_size = self.group_size  # Should be 20 from config
        print(f"\n🔍 DIAGNOSTIC: Generating {actual_group_size} candidates (configured: {self.config.get('grpo_config', {}).get('group_size', 'NOT SET')})")
        
        for i in range(actual_group_size):
            self.rng_key, candidate_key = jax.random.split(self.rng_key)
            
            # Get policy output
            policy_output = self.policy.forward(
                params=policy_params,
                state=state,
                rng_key=candidate_key
            )
            
            # Sample intervention
            var_probs = policy_output['variable_probs']
            selected_idx = jax.random.categorical(candidate_key, jnp.log(var_probs + 1e-8))
            selected_var = variables[int(selected_idx)]
            
            # Sample value
            value_params = policy_output['value_params']
            mean = value_params[selected_idx, 0]
            log_std = value_params[selected_idx, 1]
            std = jnp.exp(log_std)
            
            self.rng_key, value_key = jax.random.split(self.rng_key)
            value = mean + std * jax.random.normal(value_key)
            
            # Compute log probability
            var_log_prob = jnp.log(var_probs[selected_idx] + 1e-8)
            value_log_prob = -0.5 * ((value - mean) / std) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi)
            total_log_prob = var_log_prob + value_log_prob
            
            # Apply intervention and get outcome
            intervention = {selected_var: float(value)}
            outcome = scm.intervene(intervention)
            target_value = outcome.get(target_var, 0)
            
            candidate = {
                'variable': selected_var,
                'value': float(value),
                'target_value': float(target_value),
                'log_prob': float(total_log_prob),
                'var_log_prob': float(var_log_prob),
                'value_log_prob': float(value_log_prob)
            }
            candidates.append(candidate)
        
        # Track actual group size
        self.group_sizes_seen.append(len(candidates))
        
        # Compute rewards with component tracking
        rewards = self._compute_grpo_rewards_with_tracking(
            candidates, target_var, scm, buffer, variables
        )
        
        # Add rewards to candidates
        for candidate, reward in zip(candidates, rewards):
            candidate['reward'] = float(reward)
        
        # Select best based on rewards
        best_idx = np.argmax(rewards)
        best = candidates[best_idx]
        
        # Track intervention details
        self.all_interventions.append({
            'variable': best['variable'],
            'value': best['value'],
            'target_value': best['target_value'],
            'reward': best['reward'],
            'phase': self.current_phase,
            'group_size': len(candidates)
        })
        
        # Update episode tracking
        if best['variable'] == 'X1':
            self.current_episode_data['x1_selections'] += 1
            self.current_episode_data['x1_values'].append(best['value'])
        elif best['variable'] == 'X0':
            self.current_episode_data['x0_selections'] += 1
        elif best['variable'] == 'X2':
            self.current_episode_data['x2_selections'] += 1
        
        self.current_episode_data['x2_outcomes'].append(best['target_value'])
        self.current_episode_data['rewards'].append(best['reward'])
        
        # Print diagnostic info every 10 interventions
        total_interventions = len(self.all_interventions)
        if total_interventions % 10 == 0:
            recent = self.all_interventions[-10:]
            x1_count = sum(1 for i in recent if i['variable'] == 'X1')
            x1_vals = [i['value'] for i in recent if i['variable'] == 'X1']
            x2_vals = [i['target_value'] for i in recent]
            
            print(f"\n📊 Progress (interventions {total_interventions-9}-{total_interventions}):")
            print(f"   X1 selection: {x1_count}/10 ({100*x1_count/10:.0f}%)")
            if x1_vals:
                print(f"   Mean X1 value: {np.mean(x1_vals):.3f}")
            print(f"   Mean X2 outcome: {np.mean(x2_vals):.3f}")
            print(f"   Training phase: {self.current_phase}")
            print(f"   Group sizes: {set(self.group_sizes_seen[-10:])}")
        
        # Update policy if in policy phase
        if self.current_phase == 'policy':
            self._update_policy_with_grpo(candidates)
        else:
            print(f"   ⚠️ Skipping GRPO update - in {self.current_phase} phase")
        
        # Add to buffer
        buffer.add_intervention(
            targets=frozenset([best['variable']]),
            values={best['variable']: best['value']},
            outcome={target_var: best['target_value']}
        )
        
        return best
    
    def _compute_grpo_rewards_with_tracking(self, candidates, target_var, scm, buffer, variables):
        """Compute rewards with detailed component tracking."""
        
        rewards = []
        component_details = []
        
        # Get reward weights
        weights = self.config.get('grpo_reward_weights', {
            'target_delta': 0.7,
            'direct_parent': 0.3,
            'info_gain': 0.0
        })
        
        # Get true parents for structural reward
        from src.causal_bayes_opt.data_structures.scm import get_parents
        true_parents = list(get_parents(scm, target_var))
        
        for i, candidate in enumerate(candidates):
            # Target minimization component
            target_component = -candidate['target_value'] * weights.get('target_delta', 0.7)
            
            # Parent selection component
            is_parent = 1.0 if candidate['variable'] in true_parents else 0.0
            parent_component = is_parent * weights.get('direct_parent', 0.3)
            
            # Total reward
            total_reward = target_component + parent_component
            rewards.append(total_reward)
            
            # Track components
            component_details.append({
                'variable': candidate['variable'],
                'target': target_component,
                'parent': parent_component,
                'total': total_reward
            })
        
        # Store component history
        self.reward_component_history.append(component_details)
        
        # Print component breakdown for first candidate in some updates
        if len(self.all_interventions) % 20 == 0 and component_details:
            print(f"\n🎯 Reward Components (first 3 candidates):")
            for j in range(min(3, len(component_details))):
                comp = component_details[j]
                cand = candidates[j]
                print(f"   {cand['variable']}={cand['value']:.2f} → X2={cand['target_value']:.2f}:")
                print(f"     Target: {comp['target']:.3f}, Parent: {comp['parent']:.3f}, Total: {comp['total']:.3f}")
        
        return np.array(rewards)
    
    def _end_episode(self, episode_idx, scm):
        """Track episode performance."""
        
        # Calculate episode metrics
        if self.current_episode_data['x2_outcomes']:
            total_selections = (self.current_episode_data['x1_selections'] + 
                              self.current_episode_data['x0_selections'] + 
                              self.current_episode_data['x2_selections'])
            
            metrics = {
                'episode': episode_idx + 1,
                'x1_selection_rate': self.current_episode_data['x1_selections'] / total_selections if total_selections > 0 else 0,
                'x0_selection_rate': self.current_episode_data['x0_selections'] / total_selections if total_selections > 0 else 0,
                'mean_x1': np.mean(self.current_episode_data['x1_values']) if self.current_episode_data['x1_values'] else 0,
                'mean_x2': np.mean(self.current_episode_data['x2_outcomes']),
                'best_x2': np.min(self.current_episode_data['x2_outcomes']),
                'negative_x1_rate': sum(1 for v in self.current_episode_data['x1_values'] if v < 0) / len(self.current_episode_data['x1_values']) if self.current_episode_data['x1_values'] else 0,
                'mean_reward': np.mean(self.current_episode_data['rewards'])
            }
            
            self.episode_performances.append(metrics)
            
            # Print episode summary
            print(f"\n{'='*70}")
            print(f"EPISODE {episode_idx + 1} COMPLETE")
            print(f"{'='*70}")
            print(f"📊 Performance:")
            print(f"   X1 selection: {100*metrics['x1_selection_rate']:.1f}%")
            print(f"   X0 selection: {100*metrics['x0_selection_rate']:.1f}%")
            print(f"   Mean X1 value: {metrics['mean_x1']:.3f}")
            print(f"   Mean X2 outcome: {metrics['mean_x2']:.3f}")
            print(f"   Best X2 achieved: {metrics['best_x2']:.3f}")
            print(f"   Negative X1 rate: {100*metrics['negative_x1_rate']:.1f}%")
            print(f"   Mean reward: {metrics['mean_reward']:.3f}")
            
            # Check phase consistency
            phase_counts = {}
            for phase in self.phase_history[-self.config['max_interventions']:]:
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
            print(f"   Phase distribution: {phase_counts}")
            
            # Check group size consistency
            if self.group_sizes_seen:
                recent_sizes = self.group_sizes_seen[-self.config['max_interventions']:]
                print(f"   Group sizes used: {set(recent_sizes)} (should be {self.config.get('grpo_config', {}).get('group_size', 'NOT SET')})")
        
        # Reset episode data
        self.current_episode_data = {
            'x1_selections': 0,
            'x0_selections': 0,
            'x2_selections': 0,
            'x1_values': [],
            'x2_outcomes': [],
            'rewards': []
        }
        
        super()._end_episode(episode_idx, scm)


def run_definitive_test():
    """Run the definitive test with all fixes."""
    
    print("\n" + "="*80)
    print("DEFINITIVE 100% SUCCESS RATE TEST")
    print("="*80)
    
    # Create simple 3-node chain SCM
    scm = create_chain_scm(chain_length=3)
    
    config = {
        # Episode configuration
        'max_episodes': 20,
        'obs_per_episode': 10,
        'max_interventions': 50,
        
        # Architecture
        'policy_architecture': 'permutation_invariant',
        
        # CRITICAL: Keep in policy phase entire time
        'episodes_per_phase': 1000,  # Never switch phases
        'use_surrogate': True,
        'use_grpo_rewards': True,
        
        # Stable learning parameters
        'learning_rate': 2e-5,
        
        # GRPO configuration with proper group size
        'grpo_config': {
            'group_size': 20,  # Large group for better statistics
            'entropy_coefficient': 0.0,  # Pure exploitation
            'clip_ratio': 0.2,
            'gradient_clip': 1.0,
            'ppo_epochs': 4
        },
        
        # Optimal reward weights (per user request)
        'grpo_reward_weights': {
            'target_delta': 0.7,    # Primary focus on minimizing X2
            'direct_parent': 0.3,   # Strong signal for selecting X1
            'info_gain': 0.0        # No information gain component
        },
        
        'checkpoint_dir': 'definitive_100',
        'verbose': False
    }
    
    print(f"\n🔧 Configuration:")
    print(f"  Episodes: {config['max_episodes']}")
    print(f"  Interventions per episode: {config['max_interventions']}")
    print(f"  Total GRPO updates: {config['max_episodes'] * config['max_interventions']}")
    print(f"  Group size: {config['grpo_config']['group_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Episodes per phase: {config['episodes_per_phase']} (continuous policy training)")
    print(f"  Reward weights: target={config['grpo_reward_weights']['target_delta']}, "
          f"parent={config['grpo_reward_weights']['direct_parent']}, "
          f"info={config['grpo_reward_weights']['info_gain']}")
    
    print(f"\n🎯 Success Criteria:")
    print(f"  1. >95% X1 selection rate in final 5 episodes")
    print(f"  2. Mean X1 < -1.5 in final 5 episodes")
    print(f"  3. Mean X2 < -3.0 in final 5 episodes")
    print(f"  4. Consistent improvement from first to last episodes")
    print(f"  5. Group size = 20 consistently (not 4)")
    
    print(f"\n" + "-"*80)
    print("STARTING TRAINING")
    print("-"*80)
    
    # Create trainer with diagnostics
    trainer = DefinitiveDiagnosticTrainer(config)
    
    # Run training
    results = trainer.train([scm])
    
    # Final analysis
    print("\n" + "="*80)
    print("FINAL ANALYSIS")
    print("="*80)
    
    if trainer.episode_performances:
        # Analyze first vs last 5 episodes
        first_5 = trainer.episode_performances[:5]
        last_5 = trainer.episode_performances[-5:]
        
        print("\n📈 Learning Trajectory:")
        print("\nFirst 5 episodes:")
        for m in first_5:
            print(f"  Episode {m['episode']:2d}: X1 rate={100*m['x1_selection_rate']:5.1f}%, "
                  f"Mean X1={m['mean_x1']:6.3f}, Mean X2={m['mean_x2']:6.3f}")
        
        print("\nLast 5 episodes:")
        for m in last_5:
            print(f"  Episode {m['episode']:2d}: X1 rate={100*m['x1_selection_rate']:5.1f}%, "
                  f"Mean X1={m['mean_x1']:6.3f}, Mean X2={m['mean_x2']:6.3f}")
        
        # Calculate aggregate metrics
        first_5_x1_rate = np.mean([m['x1_selection_rate'] for m in first_5])
        last_5_x1_rate = np.mean([m['x1_selection_rate'] for m in last_5])
        first_5_mean_x1 = np.mean([m['mean_x1'] for m in first_5])
        last_5_mean_x1 = np.mean([m['mean_x1'] for m in last_5])
        first_5_mean_x2 = np.mean([m['mean_x2'] for m in first_5])
        last_5_mean_x2 = np.mean([m['mean_x2'] for m in last_5])
        
        print(f"\n📊 Aggregate Comparison:")
        print(f"  X1 selection rate: {100*first_5_x1_rate:.1f}% → {100*last_5_x1_rate:.1f}% "
              f"(Δ={100*(last_5_x1_rate-first_5_x1_rate):+.1f}%)")
        print(f"  Mean X1 value: {first_5_mean_x1:.3f} → {last_5_mean_x1:.3f} "
              f"(Δ={last_5_mean_x1-first_5_mean_x1:+.3f})")
        print(f"  Mean X2 outcome: {first_5_mean_x2:.3f} → {last_5_mean_x2:.3f} "
              f"(Δ={last_5_mean_x2-first_5_mean_x2:+.3f})")
        
        # Group size verification
        print(f"\n🔍 Group Size Verification:")
        unique_sizes = set(trainer.group_sizes_seen)
        print(f"  Unique group sizes seen: {unique_sizes}")
        print(f"  Expected: {{20}}")
        group_size_correct = unique_sizes == {20}
        
        # Phase distribution
        print(f"\n🔄 Training Phase Distribution:")
        phase_counts = {}
        for phase in trainer.phase_history:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        total_phases = len(trainer.phase_history)
        for phase, count in phase_counts.items():
            print(f"  {phase}: {count}/{total_phases} ({100*count/total_phases:.1f}%)")
        mostly_policy = phase_counts.get('policy', 0) / total_phases > 0.9
        
        # Success evaluation
        print("\n" + "="*80)
        print("SUCCESS CRITERIA EVALUATION")
        print("="*80)
        
        criteria = [
            ("X1 selection >95% (last 5)", last_5_x1_rate > 0.95),
            ("Mean X1 < -1.5 (last 5)", last_5_mean_x1 < -1.5),
            ("Mean X2 < -3.0 (last 5)", last_5_mean_x2 < -3.0),
            ("Improvement shown", last_5_mean_x2 < first_5_mean_x2 - 0.5),
            ("Group size = 20", group_size_correct),
            ("Continuous policy training", mostly_policy)
        ]
        
        for criterion, passed in criteria:
            status = "✅" if passed else "❌"
            print(f"  {status} {criterion}")
        
        passed_count = sum(1 for _, p in criteria if p)
        total_count = len(criteria)
        success_rate = passed_count / total_count
        
        print(f"\n🎯 Overall Success: {passed_count}/{total_count} ({100*success_rate:.0f}%)")
        
        if success_rate >= 0.9:
            print("\n🎉 SUCCESS! Near-optimal performance achieved!")
            print("The system correctly learns to:")
            print("  1. Select X1 (the direct parent) consistently")
            print("  2. Use negative values to minimize X2")
            print("  3. Achieve strong performance with proper configuration")
        elif success_rate >= 0.7:
            print("\n✅ Good progress! System is learning effectively.")
            print("Consider more episodes for full convergence.")
        else:
            print("\n⚠️ Performance below expectations.")
            print("Check diagnostic output for issues.")
        
        # Best single episode
        best_episode = min(trainer.episode_performances, key=lambda x: x['mean_x2'])
        print(f"\n🌟 Best Single Episode:")
        print(f"  Episode {best_episode['episode']}: Mean X2 = {best_episode['mean_x2']:.3f}, "
              f"X1 rate = {100*best_episode['x1_selection_rate']:.1f}%")
    
    return trainer, results


if __name__ == "__main__":
    trainer, results = run_definitive_test()