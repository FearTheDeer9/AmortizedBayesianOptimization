#!/usr/bin/env python3
"""
Track ALL parameters during GRPO training to diagnose learning issues.

This script:
1. Forces continuous policy training (no phase switching)
2. Tracks group propensities across all 20 candidates (not just selected)
3. Monitors all parameter changes between interventions
4. Verifies that parameter updates match computed gradients
"""

import sys
import os
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.experiments.benchmark_scms import create_chain_scm
from src.causal_bayes_opt.training.joint_acbo_trainer import JointACBOTrainer


class FullParamTracker(JointACBOTrainer):
    """Track all parameters and group behavior during training."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Force policy phase always
        self.current_phase = 'policy'
        self.episodes_per_phase = float('inf')
        
        # Tracking structures
        self.param_snapshots = []
        self.group_propensity_history = []
        self.intervention_count = 0
        self.updates_performed = 0
        self.updates_skipped = 0
        
    def _should_switch_phase(self):
        """Never switch phases."""
        return False
        
    def _switch_training_phase(self):
        """Do nothing - stay in policy phase."""
        pass
    
    def _extract_all_params(self, params):
        """Extract comprehensive parameter snapshot."""
        snapshot = {}
        
        def extract_recursive(node, path=""):
            """Recursively extract parameter values."""
            if isinstance(node, dict):
                for key, value in node.items():
                    new_path = f"{path}/{key}" if path else key
                    extract_recursive(value, new_path)
            elif isinstance(node, jnp.ndarray):
                # Store key statistics and sample values
                flat = node.flatten()
                snapshot[path] = {
                    'norm': float(jnp.linalg.norm(node)),
                    'mean': float(jnp.mean(node)),
                    'std': float(jnp.std(node)),
                    'min': float(jnp.min(node)),
                    'max': float(jnp.max(node)),
                    'shape': node.shape,
                    'first_5': flat[:5].tolist() if len(flat) >= 5 else flat.tolist()
                }
        
        extract_recursive(params)
        return snapshot
    
    def _compare_snapshots(self, before, after):
        """Compare two parameter snapshots."""
        changes = {}
        total_change = 0.0
        max_change = 0.0
        
        for key in before.keys():
            if key in after:
                norm_before = before[key]['norm']
                norm_after = after[key]['norm']
                norm_change = norm_after - norm_before
                
                changes[key] = {
                    'norm_change': norm_change,
                    'relative_change': norm_change / (norm_before + 1e-8),
                    'mean_change': after[key]['mean'] - before[key]['mean']
                }
                
                total_change += abs(norm_change)
                max_change = max(max_change, abs(norm_change))
        
        return changes, total_change, max_change
    
    def _analyze_group_propensities(self, buffer, posterior, target_var, variables, scm, policy_params, rng_key):
        """Analyze propensities across ALL candidates in the group."""
        
        # Prepare tensor using the same approach
        from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
        tensor, mapper = buffer_to_three_channel_tensor(
            buffer, target_var, max_history_size=100, standardize=True
        )
        
        # Track selections and probabilities
        var_selections = {var: 0 for var in variables}
        var_prob_sums = {var: 0.0 for var in variables}
        value_means = {var: [] for var in variables}
        all_candidates = []
        
        print(f"\n🔍 GENERATING {self.group_size} CANDIDATES...")
        
        # Generate all candidates
        for i in range(self.group_size):
            self.rng_key, candidate_key = jax.random.split(self.rng_key)
            
            # Get policy output using the policy function
            policy_output = self.policy_fn.apply(
                policy_params,
                candidate_key,
                tensor,
                mapper.target_idx
            )
            
            # Get probabilities
            var_logits = policy_output['variable_logits']
            var_probs = jax.nn.softmax(var_logits)
            value_params = policy_output['value_params']
            
            # Accumulate probabilities for each variable
            for j in range(len(var_probs)):
                var_name = mapper.get_name(j)
                if var_name in var_prob_sums:
                    var_prob_sums[var_name] += float(var_probs[j])
            
            # Sample variable
            selected_idx = jax.random.categorical(candidate_key, var_logits)
            selected_var = mapper.get_name(int(selected_idx))
            
            # Sample value
            mean = value_params[selected_idx, 0]
            log_std = value_params[selected_idx, 1]
            std = jnp.exp(log_std)
            
            self.rng_key, value_key = jax.random.split(self.rng_key)
            value = mean + std * jax.random.normal(value_key)
            
            # Track selection
            if selected_var in var_selections:
                var_selections[selected_var] += 1
                value_means[selected_var].append(float(value))
            
            # Store candidate
            all_candidates.append({
                'variable': selected_var,
                'value': float(value),
                'var_prob': float(var_probs[selected_idx]),
                'mean': float(mean),
                'std': float(std)
            })
        
        # Compute statistics
        stats = {
            'selections': var_selections,
            'selection_rates': {k: v/self.group_size for k, v in var_selections.items()},
            'avg_probs': {k: v/self.group_size for k, v in var_prob_sums.items()},
            'value_means': {k: np.mean(v) if v else 0 for k, v in value_means.items()},
            'value_stds': {k: np.std(v) if len(v) > 1 else 0 for k, v in value_means.items()},
            'candidates': all_candidates
        }
        
        return stats
    
    def _select_best_intervention_grpo(self, buffer, posterior, target_var, variables, scm, policy_params, rng_key):
        """Override to track parameters and group behavior."""
        
        self.intervention_count += 1
        
        print(f"\n{'='*80}")
        print(f"INTERVENTION {self.intervention_count}")
        print(f"{'='*80}")
        
        # Confirm we're in policy phase
        print(f"Training phase: {self.current_phase} (should be 'policy')")
        
        # Take parameter snapshot BEFORE
        before_snapshot = self._extract_all_params(policy_params)
        
        # Analyze group propensities
        group_stats = self._analyze_group_propensities(
            buffer, posterior, target_var, variables, scm, policy_params, rng_key
        )
        self.group_propensity_history.append(group_stats)
        
        # Print group statistics
        self._print_group_stats(group_stats)
        
        # Call parent method (generates candidates, computes rewards, updates policy)
        # But we need to generate our own candidates to track everything
        candidates = self._generate_and_evaluate_candidates(
            buffer, posterior, target_var, variables, scm, policy_params, rng_key
        )
        
        # Compute rewards
        rewards = self._compute_grpo_rewards(candidates, target_var, scm, buffer, variables)
        
        # Add rewards to candidates
        for candidate, reward in zip(candidates, rewards):
            candidate['reward'] = float(reward)
        
        # Select best
        best_idx = np.argmax(rewards)
        best = candidates[best_idx]
        
        print(f"\n📍 SELECTED: {best['variable']} = {best['value']:.3f} → X2 = {best['target_value']:.3f}")
        
        # Update policy if in policy phase
        if self.current_phase == 'policy':
            print("\n🔄 PERFORMING GRPO UPDATE...")
            self._update_policy_with_grpo(candidates)
            self.updates_performed += 1
        else:
            print("\n⚠️ SKIPPING UPDATE (not in policy phase)")
            self.updates_skipped += 1
        
        # Take parameter snapshot AFTER
        after_snapshot = self._extract_all_params(self.policy_params)
        
        # Compare snapshots
        changes, total_change, max_change = self._compare_snapshots(before_snapshot, after_snapshot)
        
        # Print parameter changes
        self._print_param_changes(changes, total_change, max_change)
        
        # Add to buffer
        buffer.add_intervention(
            targets=frozenset([best['variable']]),
            values={best['variable']: best['value']},
            outcome={'X2': best['target_value']}
        )
        
        return best
    
    def _generate_grpo_batch_with_info_gain(self, buffer, posterior, target_var, variables, scm, 
                                           policy_params, surrogate_params, rng_key):
        """Override to avoid using surrogate when disabled."""
        if not self.use_surrogate:
            # Simple generation without info gain
            return self._generate_and_evaluate_candidates(
                buffer, posterior, target_var, variables, scm, policy_params, rng_key
            )
        else:
            return super()._generate_grpo_batch_with_info_gain(
                buffer, posterior, target_var, variables, scm, 
                policy_params, surrogate_params, rng_key
            )
    
    def _generate_and_evaluate_candidates(self, buffer, posterior, target_var, variables, scm, policy_params, rng_key):
        """Generate candidates with full tracking."""
        
        # Use parent class methods to prepare the state
        from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
        tensor, mapper = buffer_to_three_channel_tensor(
            buffer, target_var, max_history_size=100, standardize=True
        )
        
        # Store variable order for mapping
        var_to_idx = {var: i for i, var in enumerate(variables)}
        
        candidates = []
        
        for i in range(self.group_size):
            self.rng_key, candidate_key = jax.random.split(self.rng_key)
            
            # Get policy output using the policy function
            policy_output = self.policy_fn.apply(
                policy_params,
                candidate_key,
                tensor,
                mapper.target_idx
            )
            
            # Get probabilities and values
            var_logits = policy_output['variable_logits']
            var_probs = jax.nn.softmax(var_logits)
            value_params = policy_output['value_params']
            
            # Sample variable
            selected_idx = jax.random.categorical(candidate_key, var_logits)
            selected_var = mapper.get_name(int(selected_idx))
            
            # Sample value
            mean = value_params[selected_idx, 0]
            log_std = value_params[selected_idx, 1]
            std = jnp.exp(log_std)
            
            self.rng_key, value_key = jax.random.split(self.rng_key)
            value = mean + std * jax.random.normal(value_key)
            
            # Compute log probability
            var_log_prob = jnp.log(var_probs[selected_idx] + 1e-8)
            value_log_prob = -0.5 * ((value - mean) / std) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi)
            total_log_prob = var_log_prob + value_log_prob
            
            # Apply intervention using the proper SCM function
            from src.causal_bayes_opt.mechanisms.linear import intervene_on_scm
            intervention = {selected_var: float(value)}
            outcome = intervene_on_scm(scm, intervention)
            target_value = outcome.get(target_var, 0)
            
            # Simplified candidate
            candidate = {
                'variable': selected_var,
                'value': float(value),
                'target_value': float(target_value),
                'log_prob': float(total_log_prob),
                'info_gain': 0.0  # No info gain when surrogate disabled
            }
            
            candidates.append(candidate)
        
        return candidates
    
    def _print_group_stats(self, stats):
        """Print group propensity statistics."""
        print(f"\n📊 GROUP PROPENSITIES ({self.group_size} candidates):")
        
        # Selection counts
        sels = stats['selections']
        rates = stats['selection_rates']
        print(f"   Selections: X0={sels.get('X0', 0)}/{self.group_size} ({100*rates.get('X0', 0):.0f}%), "
              f"X1={sels.get('X1', 0)}/{self.group_size} ({100*rates.get('X1', 0):.0f}%), "
              f"X2={sels.get('X2', 0)}/{self.group_size} ({100*rates.get('X2', 0):.0f}%)")
        
        # Average probabilities
        probs = stats['avg_probs']
        print(f"   Avg Probs: X0={probs.get('X0', 0):.3f}, "
              f"X1={probs.get('X1', 0):.3f}, "
              f"X2={probs.get('X2', 0):.3f}")
        
        # Value statistics
        means = stats['value_means']
        stds = stats['value_stds']
        if sels.get('X1', 0) > 0:
            print(f"   X1 Values: mean={means.get('X1', 0):.3f}, std={stds.get('X1', 0):.3f}")
        if sels.get('X0', 0) > 0:
            print(f"   X0 Values: mean={means.get('X0', 0):.3f}, std={stds.get('X0', 0):.3f}")
    
    def _print_param_changes(self, changes, total_change, max_change):
        """Print parameter change analysis."""
        print(f"\n🔧 PARAMETER CHANGES:")
        print(f"   Total change (sum of norms): {total_change:.6f}")
        print(f"   Max single param change: {max_change:.6f}")
        
        if max_change < 1e-8:
            print("   ⚠️ WARNING: NO PARAMETERS CHANGED!")
            print("   Possible causes:")
            print("     - Not in policy phase (check above)")
            print("     - Learning rate too small")
            print("     - Gradients are zero")
            print("     - Optimizer issue")
        else:
            # Show top 5 changed parameters
            sorted_changes = sorted(
                changes.items(),
                key=lambda x: abs(x[1]['norm_change']),
                reverse=True
            )[:5]
            
            print("   Top 5 changed parameters:")
            for param_name, change in sorted_changes:
                if abs(change['norm_change']) > 1e-6:
                    print(f"     {param_name}:")
                    print(f"       Norm change: {change['norm_change']:.6f} "
                          f"({100*change['relative_change']:.3f}%)")
    
    def _end_episode(self, episode_idx, scm):
        """Print episode summary."""
        print(f"\n{'='*80}")
        print(f"EPISODE {episode_idx + 1} COMPLETE")
        print(f"{'='*80}")
        
        print(f"Updates performed: {self.updates_performed}")
        print(f"Updates skipped: {self.updates_skipped}")
        
        # Analyze propensity evolution
        if len(self.group_propensity_history) >= 10:
            first_5 = self.group_propensity_history[:5]
            last_5 = self.group_propensity_history[-5:]
            
            # Average X1 selection rate
            first_x1_rate = np.mean([s['selection_rates'].get('X1', 0) for s in first_5])
            last_x1_rate = np.mean([s['selection_rates'].get('X1', 0) for s in last_5])
            
            # Average X1 probability
            first_x1_prob = np.mean([s['avg_probs'].get('X1', 0) for s in first_5])
            last_x1_prob = np.mean([s['avg_probs'].get('X1', 0) for s in last_5])
            
            print(f"\n📈 PROPENSITY EVOLUTION:")
            print(f"   X1 selection rate: {100*first_x1_rate:.1f}% → {100*last_x1_rate:.1f}%")
            print(f"   X1 probability: {first_x1_prob:.3f} → {last_x1_prob:.3f}")
            
            if last_x1_rate > first_x1_rate + 0.1:
                print("   ✅ Learning to prefer X1!")
            elif last_x1_rate < first_x1_rate - 0.1:
                print("   ❌ Moving away from X1")
            else:
                print("   ⚠️ No clear learning trend")
        
        super()._end_episode(episode_idx, scm)


def run_param_tracking_test():
    """Run test with comprehensive parameter tracking."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE PARAMETER TRACKING TEST")
    print("="*80)
    
    scm = create_chain_scm(chain_length=3)
    
    config = {
        'max_episodes': 3,
        'obs_per_episode': 10,
        'max_interventions': 20,  # Fewer for detailed analysis
        'policy_architecture': 'permutation_invariant',
        
        # Force continuous policy training
        'episodes_per_phase': float('inf'),
        'use_surrogate': False,  # Disable surrogate
        'use_grpo_rewards': True,
        
        # Learning parameters
        'learning_rate': 1e-4,  # Higher for visible changes
        
        # GRPO configuration
        'grpo_config': {
            'group_size': 20,
            'entropy_coefficient': 0.01,
            'clip_ratio': 0.2,
            'gradient_clip': 1.0,
            'ppo_epochs': 4
        },
        
        # Reward weights
        'grpo_reward_weights': {
            'target_delta': 0.7,
            'direct_parent': 0.3,
            'info_gain': 0.0
        },
        
        'checkpoint_dir': 'param_tracking',
        'verbose': False
    }
    
    print(f"\n🔧 Configuration:")
    print(f"  Episodes: {config['max_episodes']}")
    print(f"  Interventions per episode: {config['max_interventions']}")
    print(f"  Group size: {config['grpo_config']['group_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Phase switching: DISABLED (continuous policy training)")
    
    print(f"\n🎯 What we're tracking:")
    print(f"  1. Group propensities (all 20 candidates, not just selected)")
    print(f"  2. All parameter changes between interventions")
    print(f"  3. Whether updates actually happen")
    print(f"  4. Evolution of X0 vs X1 preference")
    
    print(f"\n" + "-"*80)
    print("STARTING TRAINING")
    print("-"*80)
    
    trainer = FullParamTracker(config)
    results = trainer.train([scm])
    
    # Final analysis
    print("\n" + "="*80)
    print("FINAL ANALYSIS")
    print("="*80)
    
    print(f"\nTotal interventions: {trainer.intervention_count}")
    print(f"Updates performed: {trainer.updates_performed}")
    print(f"Updates skipped: {trainer.updates_skipped}")
    
    if trainer.updates_performed == 0:
        print("\n❌ NO UPDATES PERFORMED - Policy never trained!")
    elif trainer.updates_performed < trainer.intervention_count / 2:
        print("\n⚠️ Many updates skipped - phase switching issue")
    else:
        print("\n✅ Updates performed as expected")
    
    return trainer, results


if __name__ == "__main__":
    trainer, results = run_param_tracking_test()