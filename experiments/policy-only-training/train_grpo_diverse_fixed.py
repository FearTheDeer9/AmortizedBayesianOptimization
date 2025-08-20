#!/usr/bin/env python3
"""
Train GRPO with diverse SCMs using proven patterns from definitive_100_percent.py.
This version uses JointACBOTrainer as base and implements direct intervention generation.
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

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.causal_bayes_opt.training.joint_acbo_trainer import JointACBOTrainer
from src.causal_bayes_opt.training.diverse_scm_generator import DiverseSCMGenerator
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


# DiverseSCMGenerator moved to src.causal_bayes_opt.training.diverse_scm_generator
# Using import from there instead of duplicate definition


class DiverseGRPOTrainer(JointACBOTrainer):
    """
    Trainer for diverse SCMs using proven patterns from definitive_100_percent.py.
    Overrides intervention selection to use direct generation instead of batch system.
    """
    
    def __init__(self, config):
        # Store config for override method
        self.config = config
        
        # CRITICAL: Override episodes_per_phase BEFORE calling parent init
        if not config.get('use_surrogate', True):
            config['episodes_per_phase'] = 999999  # Never switch phases
            config['initial_phase'] = 'policy'  # Ensure we start in policy phase
            logger.info("Configuring for pure GRPO: phase switching disabled")
        
        super().__init__(config=config)
        
        # FORCE override after parent init (regardless of config)
        # This ensures we never switch phases in pure GRPO mode
        if not config.get('use_surrogate', True):
            self.episodes_per_phase = 999999  # Force this value
            self.current_phase = 'policy'  # Force policy phase
            self.phase_episode_count = 0  # Reset counter
            logger.info(f"FORCED OVERRIDE: episodes_per_phase={self.episodes_per_phase}, current_phase={self.current_phase}, phase_count={self.phase_episode_count}")
        
        # Tracking metrics by SCM size
        self.size_category_metrics = {
            'small': [],
            'medium': [],
            'large': []
        }
        
        # Track all interventions for analysis
        self.all_interventions = []
        self.episode_performances = []
        
        # Comprehensive metrics tracking
        self.convergence_metrics = {
            'rolling_average_window': 20,
            'rolling_averages': [],
            'best_per_category': {'small': float('inf'), 'medium': float('inf'), 'large': float('inf')},
            'episodes_to_threshold': {},  # Track when we hit performance thresholds
            'value_patterns': {'parent_values': [], 'non_parent_values': []},
            'exploration_ratio': []  # Track exploration vs exploitation
        }
        
        # CSV logging setup
        self.csv_data = []
        self.checkpoint_episodes = [25, 50, 75, 100, 125, 150, 175, 200]
        
        # Current episode tracking
        self.current_episode_data = {
            'selections': [],
            'target_values': [],
            'rewards': [],
            'size_category': None,
            'num_vars': 0,
            'intervention_values': [],  # Track actual intervention values
            'parent_interventions': 0,   # Count parent vs non-parent
            'non_parent_interventions': 0
        }
        
        # Override group size if needed
        if hasattr(self, 'grpo_config'):
            self.group_size = self.grpo_config.group_size
        else:
            self.group_size = config.get('grpo_config', {}).get('group_size', 10)
        
        logger.info(f"Initialized DiverseGRPOTrainer with group_size={self.group_size}")
    
    def _rotate_scm(self, episode_metrics):
        """Override to properly call the SCM generator each episode."""
        # Store previous SCM name for logging
        prev_scm_name = self.current_scm.name if hasattr(self.current_scm, 'name') else 'unknown'
        
        # Generate new SCM from our diverse generator
        if hasattr(self, 'scm_generator_callable'):
            logger.info(f"Generating new diverse SCM...")
            new_scm = self.scm_generator_callable()
            
            # Extract metadata
            metadata = new_scm.get('metadata', {})
            size_cat = metadata.get('size_category', 'unknown')
            num_vars = metadata.get('num_variables', len(new_scm.get('variables', [])))
            structure = metadata.get('structure_type', 'unknown')
            
            logger.info(f"  Generated {structure} SCM: {num_vars} vars ({size_cat})")
            
            # Update current SCM
            self.current_scm = new_scm
            # Convert PMap to mutable dict before modifying
            self.current_scm_metadata = dict(metadata)  # Convert to mutable dict
            self.current_scm_metadata['num_variables'] = num_vars
            self.episodes_on_current_scm = 0
        else:
            # Fallback to parent implementation
            super()._rotate_scm(episode_metrics)
    
    def _should_switch_phase(self):
        """Override to never switch phases in pure GRPO mode."""
        if not self.config.get('use_surrogate', True):
            return False  # Never switch phases
        return super()._should_switch_phase()
    
    def _run_collaborative_episode(self, episode_idx):
        """Override to track metrics by SCM size and ensure SCM rotation."""
        # Rotate SCM at the start of each episode for diversity
        if episode_idx > 0:  # Don't rotate on first episode (already have one)
            self._rotate_scm([])
        
        # Get current SCM metadata
        if hasattr(self, 'current_scm_metadata'):
            size_cat = self.current_scm_metadata.get('size_category', 'unknown')
            num_vars = self.current_scm_metadata.get('num_variables', 0)
            self.current_episode_data['size_category'] = size_cat
            self.current_episode_data['num_vars'] = num_vars
            
            logger.info(f"\nEpisode {episode_idx}: {size_cat} SCM with {num_vars} variables")
        
        # Run the episode
        result = super()._run_collaborative_episode(episode_idx)
        
        # Track episode performance
        self._track_episode_performance(episode_idx)
        
        # Reset episode data
        self.current_episode_data = {
            'selections': [],
            'target_values': [],
            'rewards': [],
            'size_category': None,
            'num_vars': 0,
            'intervention_values': [],
            'parent_interventions': 0,
            'non_parent_interventions': 0
        }
        
        return result
    
    def _save_checkpoint(self, episode_idx):
        """Save checkpoint at specified episodes."""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        episode_dir = checkpoint_dir / f"episode_{episode_idx}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        
        # Save policy using checkpoint_utils
        from src.causal_bayes_opt.utils.checkpoint_utils import save_checkpoint
        
        # Get architecture from parent class (UnifiedGRPOTrainer sets these)
        # The parent class already has self.hidden_dim and policy_architecture
        architecture = {
            'hidden_dim': self.hidden_dim,  # Set by UnifiedGRPOTrainer.__init__
            'num_layers': self.config.get('architecture', {}).get('num_layers', 4),
            'num_heads': self.config.get('architecture', {}).get('num_heads', 8),
            'key_size': self.config.get('architecture', {}).get('key_size', 32),
            'dropout': self.config.get('architecture', {}).get('dropout', 0.1),
            'architecture_type': self.config.get('policy_architecture', 'simple_permutation_invariant')
        }
        
        save_checkpoint(
            path=episode_dir / "policy.pkl",
            params=self.policy_params,
            architecture=architecture,
            model_type='policy',
            model_subtype='grpo',
            training_config={
                'learning_rate': self.config.get('learning_rate', 5e-4),
                'episode': episode_idx,
                'total_episodes': self.config.get('max_episodes', 200)
            },
            metadata={
                'episode': episode_idx,
                'convergence_metrics': self.convergence_metrics,
                'best_per_category': self.convergence_metrics['best_per_category'],
                'trainer': 'DiverseGRPOTrainer'
            }
        )
        
        logger.info(f"  💾 Saved checkpoint at episode {episode_idx}")
    
    def _generate_grpo_batch_with_info_gain(self, buffer, posterior, target_var, variables, 
                                           scm, policy_params, surrogate_params, rng_key):
        """
        Override to use direct intervention generation following definitive_100_percent.py pattern.
        This replaces the complex batch generation with simple, direct generation.
        """
        
        # Track that we're in policy phase (should always be true now)
        if hasattr(self, 'current_phase') and self.current_phase != 'policy':
            logger.warning(f"GRPO called in {self.current_phase} phase - should be 'policy'")
        
        # DIAGNOSTIC: Log group size verification
        logger.info(f"\n🔍 GRPO Batch Generation:")
        logger.info(f"  Configured group size: {self.group_size}")
        logger.info(f"  Target: {target_var}, True parents: {get_parents(scm, target_var)}")
        
        # Prepare state for policy using the 5-channel tensor
        from src.causal_bayes_opt.training.five_channel_converter import buffer_to_five_channel_tensor_with_posteriors
        tensor_5ch, mapper, debug_info = buffer_to_five_channel_tensor_with_posteriors(
            buffer=buffer,
            target_variable=target_var,
            max_history_size=100,
            standardize=True
        )
        
        # Generate candidates directly (following definitive_100_percent.py pattern)
        candidates = []
        variable_selections = {}  # Track selection distribution
        
        for i in range(self.group_size):
            rng_key, candidate_key = random.split(rng_key)
            
            # Get policy output directly
            # Note: Using the transformed policy apply function
            # Signature: apply(params, rng, input_tensor, target_idx)
            policy_output = self.policy_fn.apply(
                policy_params,
                candidate_key,  # RNG must be second argument
                tensor_5ch,
                mapper.target_idx
            )
            
            # Extract variable probabilities and value parameters
            if isinstance(policy_output, dict):
                var_probs = policy_output.get('variable_probs', None)
                value_params = policy_output.get('value_params', None)
            else:
                # Handle tuple output (var_probs, value_params)
                var_probs, value_params = policy_output
            
            # Ensure var_probs is normalized
            if var_probs is not None:
                var_probs = jnp.abs(var_probs)
                var_probs = var_probs / (jnp.sum(var_probs) + 1e-8)
            else:
                # Uniform if not available
                var_probs = jnp.ones(len(variables)) / len(variables)
            
            # Mask out target variable
            var_probs = var_probs.at[mapper.target_idx].set(0.0)
            var_probs = var_probs / (jnp.sum(var_probs) + 1e-8)
            
            # Sample variable
            rng_key, var_key = random.split(rng_key)
            selected_idx = random.categorical(var_key, jnp.log(var_probs + 1e-8))
            selected_var = mapper.get_name(int(selected_idx))
            
            # Track selection
            variable_selections[selected_var] = variable_selections.get(selected_var, 0) + 1
            
            # Sample value
            if self.use_fixed_std:
                std = self.fixed_std
            else:
                std = 0.5  # Default
            
            rng_key, value_key = random.split(rng_key)
            value = float(random.normal(value_key) * std)
            
            # CRITICAL: Clip intervention value to SCM-specific ranges
            from src.causal_bayes_opt.interventions.handlers import clip_intervention_values
            clipped_values = clip_intervention_values(
                {selected_var: value},
                scm,
                default_range=(-3.0, 3.0)  # Fallback if SCM doesn't have ranges
            )
            value = clipped_values[selected_var]
            
            # Apply intervention and get outcome
            intervention = create_perfect_intervention(
                targets=frozenset([selected_var]),
                values={selected_var: value}
            )
            
            # Sample outcome
            outcome_samples = sample_with_intervention(
                scm, intervention, n_samples=1, seed=int(rng_key[0]) % 1000000
            )
            
            if outcome_samples:
                outcome = outcome_samples[0]
                target_value = float(get_values(outcome).get(target_var, 0.0))
            else:
                target_value = 0.0
            
            # Compute log probability for GRPO
            var_log_prob = float(jnp.log(var_probs[selected_idx] + 1e-8))
            # Simple Gaussian log prob for value
            value_log_prob = -0.5 * (value / std) ** 2 - np.log(std * np.sqrt(2 * np.pi))
            total_log_prob = var_log_prob + value_log_prob
            
            candidate = {
                'intervention': intervention,
                'outcome': outcome if outcome_samples else None,
                'variable': selected_var,
                'variable_idx': int(selected_idx),
                'value': value,
                'target_value': target_value,
                'log_prob': total_log_prob,
                'old_log_prob': total_log_prob,  # For GRPO
                'reward': 0.0,  # Will be computed next
                'posterior_after': posterior  # Keep same posterior
            }
            candidates.append(candidate)
        
        # Compute rewards manually with tracking
        rewards = self._compute_grpo_rewards_with_tracking(
            candidates, target_var, scm, buffer, variables
        )
        
        # Add rewards to candidates
        for candidate, reward in zip(candidates, rewards):
            candidate['reward'] = float(reward)
            
            # Add reward components for transparency
            candidate['reward_components'] = {
                'target_delta': -candidate['target_value'] * 0.9,
                'direct_parent': 0.1 if candidate['variable'] in get_parents(scm, target_var) else 0.0,
                'info_gain': 0.0
            }
            candidate['reward_components_raw'] = {
                'target_value': candidate['target_value'],
                'direct_parent': 1.0 if candidate['variable'] in get_parents(scm, target_var) else 0.0,
                'info_gain_raw': 0.0
            }
        
        # DIAGNOSTIC: Log selection distribution and reward components
        logger.info(f"  Variable selection distribution: {variable_selections}")
        logger.info(f"  Actual candidates generated: {len(candidates)}")
        
        # Show first 3 candidates for debugging
        logger.info(f"\n  First 3 candidates (for debugging):")
        for i in range(min(3, len(candidates))):
            c = candidates[i]
            is_parent = "✓" if c['variable'] in get_parents(scm, target_var) else "✗"
            logger.info(f"    {i+1}. {c['variable']} (parent:{is_parent}) = {c['value']:.2f} → {target_var}={c['target_value']:.2f}, reward={c['reward']:.3f}")
        
        return candidates
    
    def _compute_grpo_rewards_with_tracking(self, candidates, target_var, scm, buffer, variables):
        """
        Compute rewards manually with component tracking (from definitive_100_percent.py).
        """
        rewards = []
        
        # Get reward weights from config
        weights = self.config.get('grpo_reward_config', {
            'target_improvement_weight': 0.9,
            'parent_accuracy_weight': 0.1,
            'info_gain_weight': 0.0
        })
        
        # Get true parents for structural reward
        true_parents = set(get_parents(scm, target_var))
        
        for i, candidate in enumerate(candidates):
            # Target minimization component (negative because we want to minimize)
            target_component = -candidate['target_value'] * weights.get('target_improvement_weight', 0.9)
            
            # Parent selection component
            is_parent = 1.0 if candidate['variable'] in true_parents else 0.0
            parent_component = is_parent * weights.get('parent_accuracy_weight', 0.1)
            
            # Total reward
            total_reward = target_component + parent_component
            rewards.append(total_reward)
            
            # Log first few for debugging
            if i < 3 and len(self.all_interventions) % 50 == 0:
                logger.debug(f"  Candidate {i}: {candidate['variable']}={candidate['value']:.2f} "
                           f"→ {target_var}={candidate['target_value']:.2f}, "
                           f"reward={total_reward:.3f} (target:{target_component:.3f}, parent:{parent_component:.3f})")
        
        return np.array(rewards)
    
    def _select_best_grpo_intervention(self, candidates):
        """
        Override to track interventions and handle diverse SCMs.
        """
        # Select best based on rewards
        rewards = [c['reward'] for c in candidates]
        best_idx = np.argmax(rewards)
        best = candidates[best_idx]
        
        # Get true parents for tracking
        target_var = get_target(self.current_scm)
        true_parents = set(get_parents(self.current_scm, target_var))
        is_parent = best['variable'] in true_parents
        
        # Track intervention with enhanced metrics
        self.all_interventions.append({
            'variable': best['variable'],
            'value': best['value'],
            'target_value': best['target_value'],
            'reward': best['reward'],
            'phase': self.current_phase,
            'group_size': len(candidates),
            'size_category': self.current_episode_data.get('size_category', 'unknown'),
            'num_vars': self.current_episode_data.get('num_vars', 0),
            'is_parent': is_parent,
            'episode': len(self.episode_performances)
        })
        
        # Update episode tracking
        self.current_episode_data['selections'].append(best['variable'])
        self.current_episode_data['target_values'].append(best['target_value'])
        self.current_episode_data['rewards'].append(best['reward'])
        self.current_episode_data['intervention_values'].append(best['value'])
        
        # Track parent vs non-parent interventions
        if is_parent:
            self.current_episode_data['parent_interventions'] += 1
            self.convergence_metrics['value_patterns']['parent_values'].append(abs(best['value']))
        else:
            self.current_episode_data['non_parent_interventions'] += 1
            self.convergence_metrics['value_patterns']['non_parent_values'].append(abs(best['value']))
        
        # Log progress every 10 interventions
        if len(self.all_interventions) % 10 == 0:
            recent = self.all_interventions[-10:]
            mean_target = np.mean([i['target_value'] for i in recent])
            mean_reward = np.mean([i['reward'] for i in recent])
            
            logger.info(f"Progress (last 10): mean_target={mean_target:.3f}, mean_reward={mean_reward:.3f}")
        
        return best
    
    def _track_episode_performance(self, episode_idx):
        """Track performance metrics for the episode."""
        if not self.current_episode_data['target_values']:
            return
        
        # Calculate episode metrics
        target_values = self.current_episode_data['target_values']
        rewards = self.current_episode_data['rewards']
        selections = self.current_episode_data['selections']
        intervention_values = self.current_episode_data['intervention_values']
        
        # Get true parents for this SCM
        target_var = get_target(self.current_scm)
        true_parents = set(get_parents(self.current_scm, target_var))
        
        # Calculate parent selection rate
        parent_selections = sum(1 for v in selections if v in true_parents)
        parent_rate = parent_selections / len(selections) if selections else 0
        
        # Calculate value statistics
        mean_abs_value = np.mean([abs(v) for v in intervention_values]) if intervention_values else 0
        value_std = np.std(intervention_values) if len(intervention_values) > 1 else 0
        
        metrics = {
            'episode': episode_idx,
            'size_category': self.current_episode_data.get('size_category', 'unknown'),
            'num_vars': self.current_episode_data.get('num_vars', 0),
            'mean_target': np.mean(target_values),
            'best_target': np.min(target_values),
            'worst_target': np.max(target_values),
            'target_std': np.std(target_values),
            'mean_reward': np.mean(rewards),
            'parent_selection_rate': parent_rate,
            'num_interventions': len(target_values),
            'mean_abs_intervention_value': mean_abs_value,
            'intervention_value_std': value_std,
            'parent_interventions': self.current_episode_data['parent_interventions'],
            'non_parent_interventions': self.current_episode_data['non_parent_interventions']
        }
        
        self.episode_performances.append(metrics)
        
        # Update best per category
        size_cat = self.current_episode_data.get('size_category', 'unknown')
        if size_cat in self.convergence_metrics['best_per_category']:
            current_best = self.convergence_metrics['best_per_category'][size_cat]
            if metrics['best_target'] < current_best:
                self.convergence_metrics['best_per_category'][size_cat] = metrics['best_target']
                logger.info(f"  🎯 New best for {size_cat} SCMs: {metrics['best_target']:.3f}")
        
        # Calculate rolling average
        if len(self.episode_performances) >= self.convergence_metrics['rolling_average_window']:
            recent_episodes = self.episode_performances[-self.convergence_metrics['rolling_average_window']:]
            rolling_avg = np.mean([ep['mean_target'] for ep in recent_episodes])
            self.convergence_metrics['rolling_averages'].append(rolling_avg)
            logger.info(f"  Rolling avg (last {self.convergence_metrics['rolling_average_window']}): {rolling_avg:.3f}")
        
        # Add to size category metrics
        if size_cat in self.size_category_metrics:
            self.size_category_metrics[size_cat].append(metrics)
        
        # Save checkpoint if needed
        if episode_idx in self.checkpoint_episodes:
            self._save_checkpoint(episode_idx)
        
        # Log to CSV
        self.csv_data.append(metrics)
        
        # Log episode summary
        logger.info(f"\nEpisode {episode_idx} Complete:")
        logger.info(f"  Size: {size_cat} ({metrics['num_vars']} vars)")
        logger.info(f"  Mean target: {metrics['mean_target']:.3f}")
        logger.info(f"  Best target: {metrics['best_target']:.3f}")
        logger.info(f"  Parent selection: {100*metrics['parent_selection_rate']:.1f}%")
        logger.info(f"  Mean |intervention|: {metrics['mean_abs_intervention_value']:.3f}")


def create_diverse_training_config(
    max_episodes: int = 200,
    min_vars: int = 3,
    max_vars: int = 30,
    verbose: bool = False,
    enable_wandb: bool = False,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Create configuration using proven values from definitive_100_percent.py
    and our debugging lessons.
    """
    
    config = {
        # Core episode settings
        'max_episodes': max_episodes,
        'obs_per_episode': 10,
        'max_interventions': 30,
        
        # CRITICAL: Disable phase switching for pure GRPO
        'episodes_per_phase': 999999,  # Never switch phases
        
        # CRITICAL: Use simple_permutation_invariant (our lesson)
        'policy_architecture': 'simple_permutation_invariant',
        
        # CRITICAL: Pure GRPO training - no surrogate!
        'use_surrogate': False,  # Disable surrogate completely
        'use_grpo_rewards': True,
        
        # CRITICAL: Fixed std for exploration (our lesson)
        'use_fixed_std': True,
        'fixed_std': 0.5,
        
        # CRITICAL: Optimal learning rate (our lesson)
        'learning_rate': 5e-4,
        
        # GRPO configuration (combination of our lessons and definitive)
        'grpo_config': {
            'group_size': 10,  # Our tested size
            'entropy_coefficient': 0.001,  # Low for exploitation
            'clip_ratio': 0.2,
            'gradient_clip': 1.0,
            'ppo_epochs': 4,
            'normalize_advantages': True
        },
        
        # Reward weights (our optimized values)
        'grpo_reward_config': {
            'target_improvement_weight': 0.9,
            'parent_accuracy_weight': 0.1,
            'info_gain_weight': 0.0
        },
        
        # SCM generation settings
        'scm_generation': {
            'min_vars': min_vars,
            'max_vars': max_vars,
            'new_scm_each_episode': True,
            'generator_type': 'diverse'
        },
        
        # Joint training config (just in case it's checked)
        'joint_training': {
            'episodes_per_phase': 999999,  # Never switch
            'initial_phase': 'policy',
            'adaptive': {
                'use_performance_rotation': False,  # Disable performance-based rotation
                'plateau_patience': 999999  # Never detect plateau
            }
        },
        
        # General settings
        'batch_size': 32,
        'seed': seed,
        'verbose': verbose,
        'checkpoint_dir': f'checkpoints/diverse_fixed_{min_vars}to{max_vars}',
        
        # WandB logging configuration
        'logging': {
            'wandb': {
                'enabled': enable_wandb,
                'project': 'causal-bayes-opt-grpo',
                'name': f'diverse_fixed_{min_vars}to{max_vars}_{max_episodes}ep',
                'tags': ['diverse_scms', 'grpo', 'fixed_implementation'],
                'log_frequency': 1
            }
        }
    }
    
    return config


def analyze_results_by_size(trainer: DiverseGRPOTrainer) -> Dict[str, Any]:
    """
    Analyze training results by SCM size category.
    """
    analysis = {}
    
    # Analyze each size category
    for category in ['small', 'medium', 'large']:
        metrics = trainer.size_category_metrics.get(category, [])
        if metrics:
            analysis[category] = {
                'count': len(metrics),
                'mean_target': np.mean([m['mean_target'] for m in metrics]),
                'best_target': np.min([m['best_target'] for m in metrics]),
                'mean_parent_rate': np.mean([m['parent_selection_rate'] for m in metrics]),
                'improvement': 0  # Calculate if we have enough data
            }
            
            # Calculate improvement (first vs last episodes)
            if len(metrics) >= 10:
                first_5 = metrics[:5]
                last_5 = metrics[-5:]
                first_mean = np.mean([m['mean_target'] for m in first_5])
                last_mean = np.mean([m['mean_target'] for m in last_5])
                analysis[category]['improvement'] = first_mean - last_mean
    
    # Overall statistics
    all_metrics = trainer.episode_performances
    if all_metrics:
        analysis['overall'] = {
            'total_episodes': len(all_metrics),
            'mean_target': np.mean([m['mean_target'] for m in all_metrics]),
            'best_target': np.min([m['best_target'] for m in all_metrics]),
            'mean_parent_rate': np.mean([m['parent_selection_rate'] for m in all_metrics])
        }
        
        # Overall improvement
        if len(all_metrics) >= 20:
            first_10 = all_metrics[:10]
            last_10 = all_metrics[-10:]
            first_mean = np.mean([m['mean_target'] for m in first_10])
            last_mean = np.mean([m['mean_target'] for m in last_10])
            analysis['overall']['improvement'] = first_mean - last_mean
    
    return analysis


def main():
    """Main training function."""
    
    parser = argparse.ArgumentParser(description="Train GRPO with diverse SCMs (fixed implementation)")
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of training episodes')
    parser.add_argument('--min-vars', type=int, default=3,
                        help='Minimum number of variables in SCMs')
    parser.add_argument('--max-vars', type=int, default=30,
                        help='Maximum number of variables in SCMs')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test with 5 episodes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable WandB logging')
    
    args = parser.parse_args()
    
    # Override episodes for quick test
    if args.quick_test:
        args.episodes = 5  # Need at least 5 to see if it switches after episode 3
        logger.info("Running quick test with 5 episodes")
    
    # Create configuration
    logger.info("\n" + "="*70)
    logger.info("GRPO TRAINING WITH DIVERSE SCMs (FIXED IMPLEMENTATION)")
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
    logger.info("Critical settings:")
    logger.info(f"  - Policy: {config['policy_architecture']}")
    logger.info(f"  - Fixed std: {config['fixed_std']}")
    logger.info(f"  - Learning rate: {config['learning_rate']}")
    logger.info(f"  - GRPO group size: {config['grpo_config']['group_size']}")
    logger.info(f"  - Pure GRPO training: surrogate disabled")
    logger.info("")
    
    # Initialize WandB if enabled
    wandb_manager = None
    if args.wandb:
        logger.info("Initializing WandB...")
        try:
            wandb_manager = WandBManager()
            wandb_run = wandb_manager.setup(config, experiment_name=f"diverse_fixed_{args.min_vars}to{args.max_vars}")
            if wandb_run:
                logger.info(f"✅ WandB initialized")
            else:
                logger.warning("⚠️  WandB initialization failed, continuing without logging")
                wandb_manager = None
        except Exception as e:
            logger.warning(f"⚠️  WandB initialization failed: {e}")
            wandb_manager = None
    
    # Initialize trainer
    logger.info("Initializing DiverseGRPOTrainer...")
    trainer = DiverseGRPOTrainer(config=config)
    logger.info("✅ Trainer initialized\n")
    
    # Create SCM generator
    logger.info("Creating diverse SCM generator...")
    scm_generator = DiverseSCMGenerator(
        min_vars=args.min_vars,
        max_vars=args.max_vars,
        seed=args.seed
    )
    logger.info(f"✅ Generator ready for {args.min_vars}-{args.max_vars} variable SCMs\n")
    
    # Store the generator in the trainer for use in _rotate_scm
    trainer.scm_generator_callable = scm_generator
    
    # Start training
    logger.info("\n" + "="*70)
    logger.info("STARTING TRAINING")
    logger.info("="*70 + "\n")
    
    try:
        # Train with the generator as a callable
        results = trainer.train(scms=scm_generator)
        
        # Analyze results by size category
        logger.info("\n" + "="*70)
        logger.info("ANALYZING RESULTS BY SCM SIZE")
        logger.info("="*70)
        
        analysis = analyze_results_by_size(trainer)
        
        # Print analysis
        for category in ['small', 'medium', 'large']:
            if category in analysis and analysis[category]['count'] > 0:
                stats = analysis[category]
                logger.info(f"\n{category.upper()} SCMs ({category}):")
                logger.info(f"  Episodes: {stats['count']}")
                logger.info(f"  Mean target: {stats['mean_target']:.3f}")
                logger.info(f"  Best target: {stats['best_target']:.3f}")
                logger.info(f"  Parent selection: {100*stats['mean_parent_rate']:.1f}%")
                if 'improvement' in stats and stats['improvement'] != 0:
                    logger.info(f"  Improvement: {stats['improvement']:.3f}")
        
        if 'overall' in analysis:
            overall = analysis['overall']
            logger.info(f"\nOVERALL:")
            logger.info(f"  Total episodes: {overall['total_episodes']}")
            logger.info(f"  Mean target: {overall['mean_target']:.3f}")
            logger.info(f"  Best target: {overall['best_target']:.3f}")
            logger.info(f"  Parent selection: {100*overall['mean_parent_rate']:.1f}%")
            if 'improvement' in overall:
                logger.info(f"  Improvement: {overall['improvement']:.3f}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results/diverse_training_fixed")
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"diverse_fixed_{args.min_vars}to{args.max_vars}_{timestamp}.json"
        csv_file = results_dir / f"diverse_fixed_{args.min_vars}to{args.max_vars}_{timestamp}.csv"
        
        # Save CSV data for detailed analysis
        if trainer.csv_data:
            df = pd.DataFrame(trainer.csv_data)
            df.to_csv(csv_file, index=False)
            logger.info(f"  📊 Saved episode metrics to {csv_file}")
        
        # Prepare serializable results with comprehensive metrics
        save_data = {
            'config': config,
            'analysis': analysis,
            'generator_stats': scm_generator.get_statistics(),
            'episode_performances': trainer.episode_performances,
            'convergence_metrics': {
                'best_per_category': trainer.convergence_metrics['best_per_category'],
                'rolling_averages': trainer.convergence_metrics['rolling_averages'][-50:] if trainer.convergence_metrics['rolling_averages'] else [],
                'final_parent_selection_rate': np.mean([ep['parent_selection_rate'] for ep in trainer.episode_performances[-20:]]) if len(trainer.episode_performances) >= 20 else 0
            }
        }
        
        with open(results_file, 'w') as f:
            # Simple serialization - convert numpy types
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
        
        logger.info(f"\n✅ Training complete! Results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
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
    # Import needed for intervention handling
    from src.causal_bayes_opt.data_structures.sample import get_values
    
    sys.exit(main())