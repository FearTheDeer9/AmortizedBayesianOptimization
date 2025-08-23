#!/usr/bin/env python3
"""
Single SCM maximization test for GRPO training.
Uses exactly one 3-variable fork SCM to validate GRPO can learn to MAXIMIZE target (opposite of minimization).
This serves as a control test to ensure we observe opposite behavior with flipped reward signal.
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


class MaximizingGRPOTrainer(JointACBOTrainer):
    """
    Trainer that maximizes target instead of minimizing (flipped reward signal).
    Used as control test to ensure GRPO can learn opposite objective.
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
        
        # Track all interventions for analysis
        self.all_interventions = []
        self.episode_performances = []
        
        # Comprehensive metrics tracking
        self.convergence_metrics = {
            'rolling_average_window': 20,
            'rolling_averages': [],
            'best_per_category': {'single': float('-inf')},  # Changed: best is now highest (maximization)
            'episodes_to_threshold': {},
            'value_patterns': {'parent_values': [], 'non_parent_values': []},
            'exploration_ratio': []
        }
        
        # CSV logging setup
        self.csv_data = []
        self.checkpoint_episodes = [25, 50, 75, 100]
        
        # Current episode tracking
        self.current_episode_data = {
            'selections': [],
            'target_values': [],
            'rewards': [],
            'size_category': 'single',
            'num_vars': 3,
            'intervention_values': [],
            'parent_interventions': 0,
            'non_parent_interventions': 0
        }
        
        # Override group size if needed
        if hasattr(self, 'grpo_config'):
            self.group_size = self.grpo_config.group_size
        else:
            self.group_size = config.get('grpo_config', {}).get('group_size', 8)
        
        logger.info(f"Initialized MaximizingGRPOTrainer with group_size={self.group_size}")
    
    def _rotate_scm(self, episode_metrics):
        """Override to use fixed single SCM - never actually rotate."""
        # For single SCM sanity check, don't rotate - keep same SCM
        # Just increment the episode counter
        self.episodes_on_current_scm += 1
        
        # Log that we're keeping the same SCM
        target_var = get_target(self.current_scm) if hasattr(self, 'current_scm') else 'unknown'
        num_vars = len(get_variables(self.current_scm)) if hasattr(self, 'current_scm') else 0
        logger.info(f"Keeping same single SCM: {num_vars} vars, target={target_var}")
    
    def _should_switch_phase(self):
        """Override to never switch phases in pure GRPO mode."""
        if not self.config.get('use_surrogate', True):
            return False  # Never switch phases
        return super()._should_switch_phase()
    
    def _run_collaborative_episode(self, episode_idx):
        """Override to track metrics by SCM size and ensure SCM rotation."""
        # For single SCM, don't rotate except on first episode setup
        if episode_idx > 0:
            self._rotate_scm([])
        
        # Get current SCM metadata
        if hasattr(self, 'current_scm_metadata'):
            size_cat = self.current_scm_metadata.get('size_category', 'single')
            num_vars = self.current_scm_metadata.get('num_variables', 3)
            self.current_episode_data['size_category'] = size_cat
            self.current_episode_data['num_vars'] = num_vars
            
            logger.info(f"\nEpisode {episode_idx}: Single SCM with {num_vars} variables (MAXIMIZATION mode)")
        
        # Run the episode
        result = super()._run_collaborative_episode(episode_idx)
        
        # Track episode performance
        self._track_episode_performance(episode_idx)
        
        # Reset episode data
        self.current_episode_data = {
            'selections': [],
            'target_values': [],
            'rewards': [],
            'size_category': 'single',
            'num_vars': 3,
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
                'total_episodes': self.config.get('max_episodes', 100)
            },
            metadata={
                'episode': episode_idx,
                'convergence_metrics': self.convergence_metrics,
                'best_per_category': self.convergence_metrics['best_per_category'],
                'trainer': 'MaximizingGRPOTrainer'
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
        logger.info(f"\n🔍 GRPO Batch Generation (MAXIMIZATION):")
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
            
            # Add reward components for transparency (FLIPPED for maximization)
            candidate['reward_components'] = {
                'target_delta': candidate['target_value'] * 0.9,  # POSITIVE for maximization
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
        Compute rewards manually with component tracking - FLIPPED for maximization.
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
            # Target MAXIMIZATION component (POSITIVE because we want to maximize)
            target_component = candidate['target_value'] * weights.get('target_improvement_weight', 0.9)
            
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
        Select RANDOM candidate instead of best to eliminate selection bias.
        GRPO still learns from all candidates, but execution is unbiased.
        """
        # Get rewards for analysis
        rewards = [c['reward'] for c in candidates]
        best_idx = np.argmax(rewards)
        best_reward = rewards[best_idx]
        
        # Select RANDOM candidate for execution (eliminates selection bias)
        import random
        selected_idx = random.randint(0, len(candidates) - 1)
        selected = candidates[selected_idx]
        selected_reward = rewards[selected_idx]
        
        # Log selection vs best every 10 interventions for transparency
        if len(self.all_interventions) % 10 == 0:
            logger.info(f"🎲 Random selection: reward={selected_reward:.3f}, Best available: {best_reward:.3f}")
        
        # Get true parents for tracking
        target_var = get_target(self.current_scm)
        true_parents = set(get_parents(self.current_scm, target_var))
        is_parent = selected['variable'] in true_parents
        
        # Track intervention with enhanced metrics (using SELECTED candidate)
        self.all_interventions.append({
            'variable': selected['variable'],
            'value': selected['value'],
            'target_value': selected['target_value'],
            'reward': selected['reward'],
            'phase': self.current_phase,
            'group_size': len(candidates),
            'size_category': self.current_episode_data.get('size_category', 'single'),
            'num_vars': self.current_episode_data.get('num_vars', 3),
            'is_parent': is_parent,
            'episode': len(self.episode_performances),
            'best_available_reward': best_reward,  # Track what we could have gotten
            'selection_advantage': selected_reward - np.mean(rewards)  # How much better/worse than average
        })
        
        # Update episode tracking (using SELECTED candidate)
        self.current_episode_data['selections'].append(selected['variable'])
        self.current_episode_data['target_values'].append(selected['target_value'])
        self.current_episode_data['rewards'].append(selected['reward'])
        self.current_episode_data['intervention_values'].append(selected['value'])
        
        # Track parent vs non-parent interventions
        if is_parent:
            self.current_episode_data['parent_interventions'] += 1
            self.convergence_metrics['value_patterns']['parent_values'].append(abs(selected['value']))
        else:
            self.current_episode_data['non_parent_interventions'] += 1
            self.convergence_metrics['value_patterns']['non_parent_values'].append(abs(selected['value']))
        
        # Log progress every 10 interventions
        if len(self.all_interventions) % 10 == 0:
            recent = self.all_interventions[-10:]
            mean_target = np.mean([i['target_value'] for i in recent])
            mean_reward = np.mean([i['reward'] for i in recent])
            mean_best_available = np.mean([i['best_available_reward'] for i in recent])
            
            logger.info(f"Progress (last 10): mean_target={mean_target:.3f}, mean_reward={mean_reward:.3f}")
            logger.info(f"  Mean best available: {mean_best_available:.3f} (policy learning potential)")
        
        return selected
    
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
            'size_category': self.current_episode_data.get('size_category', 'single'),
            'num_vars': self.current_episode_data.get('num_vars', 3),
            'mean_target': np.mean(target_values),
            'best_target': np.max(target_values),  # Changed: best is now highest (maximization)
            'worst_target': np.min(target_values),  # Changed: worst is now lowest
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
        
        # Update best per category (using 'single' as category) - Changed for maximization
        size_cat = 'single'
        if size_cat not in self.convergence_metrics['best_per_category']:
            self.convergence_metrics['best_per_category'][size_cat] = float('-inf')
        
        current_best = self.convergence_metrics['best_per_category'][size_cat]
        if metrics['best_target'] > current_best:  # Changed: > for maximization
            self.convergence_metrics['best_per_category'][size_cat] = metrics['best_target']
            logger.info(f"  🎯 New best for single SCM (MAXIMIZATION): {metrics['best_target']:.3f}")
        
        # Calculate rolling average
        if len(self.episode_performances) >= self.convergence_metrics['rolling_average_window']:
            recent_episodes = self.episode_performances[-self.convergence_metrics['rolling_average_window']:]
            rolling_avg = np.mean([ep['mean_target'] for ep in recent_episodes])
            self.convergence_metrics['rolling_averages'].append(rolling_avg)
            logger.info(f"  Rolling avg (last {self.convergence_metrics['rolling_average_window']}): {rolling_avg:.3f}")
        
        # Save checkpoint if needed
        if episode_idx in self.checkpoint_episodes:
            self._save_checkpoint(episode_idx)
        
        # Log to CSV
        self.csv_data.append(metrics)
        
        # Log episode summary
        logger.info(f"\nEpisode {episode_idx} Complete (MAXIMIZATION):")
        logger.info(f"  Size: {size_cat} ({metrics['num_vars']} vars)")
        logger.info(f"  Mean target: {metrics['mean_target']:.3f}")
        logger.info(f"  Best target: {metrics['best_target']:.3f}")
        logger.info(f"  Parent selection: {100*metrics['parent_selection_rate']:.1f}%")
        logger.info(f"  Mean |intervention|: {metrics['mean_abs_intervention_value']:.3f}")


def create_maximization_config(
    max_episodes: int = 100,
    verbose: bool = False,
    enable_wandb: bool = False,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Create configuration for single SCM maximization test.
    """
    
    config = {
        # Core episode settings (reduced for sanity check)
        'max_episodes': 5,
        'obs_per_episode': 10,
        'max_interventions': 30,
        
        # CRITICAL: Disable phase switching for pure GRPO
        'episodes_per_phase': 999999,  # Never switch phases
        
        # CRITICAL: Use simple_permutation_invariant (our lesson)
        'policy_architecture': 'simplified_permutation_invariant',
        
        # CRITICAL: Pure GRPO training - no surrogate!
        'use_surrogate': False,  # Disable surrogate completely
        'use_grpo_rewards': True,
        
        # CRITICAL: Fixed std for exploration (our lesson)
        'use_fixed_std': True,
        'fixed_std': 0.5,
        
        # CRITICAL: Optimal learning rate (our lesson)
        'learning_rate': 1e-3,
        
        # GRPO configuration (combination of our lessons and definitive)
        'grpo_config': {
            'group_size': 16,  # Our tested size
            'entropy_coefficient': 0.001,  # Low for exploitation
            'clip_ratio': 0.2,
            'gradient_clip': 1.0,
            'ppo_epochs': 4,
            'normalize_advantages': True
        },
        
        # Reward weights (FLIPPED for maximization test)
        'grpo_reward_config': {
            'target_improvement_weight': 0.0,  # Still 0.9 but POSITIVE target component
            'parent_accuracy_weight': 1.0,
            'info_gain_weight': 0.0
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
        'checkpoint_dir': 'checkpoints/single_scm_maximize',
        
        # WandB logging configuration
        'logging': {
            'wandb': {
                'enabled': enable_wandb,
                'project': 'causal-bayes-opt-grpo',
                'name': f'single_scm_maximize_{max_episodes}ep',
                'tags': ['single_scm', 'grpo', 'maximize_test'],
                'log_frequency': 1
            }
        }
    }
    
    return config


def main():
    """Main training function for single SCM maximization test."""
    
    parser = argparse.ArgumentParser(description="Single SCM GRPO maximization test")
    parser.add_argument('--episodes', type=int, default=100,
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
    logger.info("SINGLE SCM GRPO MAXIMIZATION TEST")
    logger.info("="*70)
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"SCM: 3-variable fork structure")
    logger.info(f"Objective: MAXIMIZE target (opposite of minimization)")
    logger.info(f"Seed: {args.seed}")
    logger.info("="*70 + "\n")
    
    config = create_maximization_config(
        max_episodes=args.episodes,
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
    logger.info(f"  - REWARD FLIPPED: Target maximization instead of minimization")
    logger.info("")
    
    # Initialize WandB if enabled
    wandb_manager = None
    if args.wandb:
        logger.info("Initializing WandB...")
        try:
            wandb_manager = WandBManager()
            wandb_run = wandb_manager.setup(config, experiment_name=f"single_scm_maximize")
            if wandb_run:
                logger.info(f"✅ WandB initialized")
            else:
                logger.warning("⚠️  WandB initialization failed, continuing without logging")
                wandb_manager = None
        except Exception as e:
            logger.warning(f"⚠️  WandB initialization failed: {e}")
            wandb_manager = None
    
    # Create single 3-variable fork SCM (same as minimization test)
    logger.info("Creating single 3-variable fork SCM...")
    scm_factory = VariableSCMFactory(seed=args.seed, noise_scale= 0.01)
    fixed_scm = scm_factory.create_variable_scm(
        num_variables=3,        # Simple 3-variable setup
        structure_type='chain',  # Fork structure: X0, X2 → X1
        target_variable='X2'    # Middle variable as target
    )
    
    # Log SCM details
    target_var = get_target(fixed_scm)
    parents = get_parents(fixed_scm, target_var)
    variables = get_variables(fixed_scm)
    logger.info(f"✅ Created SCM: variables={variables}, target={target_var}, parents={parents}")
    
    # Initialize trainer
    logger.info("\nInitializing Maximizing GRPO trainer...")
    trainer = MaximizingGRPOTrainer(config=config)
    logger.info("✅ Trainer initialized")
    
    # Set fixed SCM in trainer (this is the key change for single SCM)
    trainer.current_scm = fixed_scm
    trainer.current_scm_metadata = dict(fixed_scm.get('metadata', {}))
    trainer.current_scm_metadata['size_category'] = 'single'
    trainer.current_scm_metadata['num_variables'] = 3
    
    # Create lambda that always returns the same SCM
    trainer.scm_generator_callable = lambda: fixed_scm
    
    logger.info(f"✅ Fixed SCM set: {len(get_variables(fixed_scm))} variables, target={get_target(fixed_scm)}")
    
    # Start training
    logger.info("\n" + "="*70)
    logger.info("STARTING SINGLE SCM MAXIMIZATION TRAINING")
    logger.info("="*70 + "\n")
    
    try:
        # Train with the fixed SCM
        results = trainer.train(scms=lambda: fixed_scm)
        
        # Analyze results
        logger.info("\n" + "="*70)
        logger.info("ANALYZING SINGLE SCM MAXIMIZATION RESULTS")
        logger.info("="*70)
        
        all_metrics = trainer.episode_performances
        if all_metrics:
            # Overall statistics
            final_episodes = all_metrics[-10:] if len(all_metrics) >= 10 else all_metrics
            
            analysis = {
                'total_episodes': len(all_metrics),
                'final_mean_target': np.mean([m['mean_target'] for m in final_episodes]),
                'best_target_overall': np.max([m['best_target'] for m in all_metrics]),  # Changed: max for maximization
                'final_parent_rate': np.mean([m['parent_selection_rate'] for m in final_episodes]),
                'improvement': 0
            }
            
            # Calculate improvement (first vs last episodes) - Changed for maximization
            if len(all_metrics) >= 20:
                first_10 = all_metrics[:10]
                last_10 = all_metrics[-10:]
                first_mean = np.mean([m['mean_target'] for m in first_10])
                last_mean = np.mean([m['mean_target'] for m in last_10])
                analysis['improvement'] = last_mean - first_mean  # Changed: last - first for maximization
            
            logger.info(f"\nSINGLE SCM MAXIMIZATION PERFORMANCE:")
            logger.info(f"  Total episodes: {analysis['total_episodes']}")
            logger.info(f"  Final mean target: {analysis['final_mean_target']:.3f}")
            logger.info(f"  Best target overall: {analysis['best_target_overall']:.3f}")
            logger.info(f"  Final parent selection: {100*analysis['final_parent_rate']:.1f}%")
            if analysis['improvement'] != 0:
                logger.info(f"  Improvement: {analysis['improvement']:.3f}")
            
            # Success criteria check (FLIPPED for maximization)
            logger.info(f"\nSUCCESS CRITERIA CHECK (MAXIMIZATION):")
            target_success = analysis['best_target_overall'] > 1.0  # Changed: > 1.0 for maximization
            parent_success = analysis['final_parent_rate'] > 0.6  # Same: want parent selection
            
            logger.info(f"  Target maximization (> 1.0): {'✅' if target_success else '❌'} {analysis['best_target_overall']:.3f}")
            logger.info(f"  Parent selection (> 60%): {'✅' if parent_success else '❌'} {100*analysis['final_parent_rate']:.1f}%")
            
            if target_success and parent_success:
                logger.info(f"  🎉 MAXIMIZATION TEST PASSED!")
            else:
                logger.info(f"  ⚠️  MAXIMIZATION TEST NEEDS INVESTIGATION")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results/single_scm_maximize")
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"single_scm_maximize_{timestamp}.json"
        csv_file = results_dir / f"single_scm_maximize_{timestamp}.csv"
        
        # Save CSV data for detailed analysis
        if trainer.csv_data:
            df = pd.DataFrame(trainer.csv_data)
            df.to_csv(csv_file, index=False)
            logger.info(f"\n📊 Saved episode metrics to {csv_file}")
        
        # Prepare serializable results
        save_data = {
            'config': config,
            'scm_details': {
                'variables': list(get_variables(fixed_scm)),  # Convert frozenset to list
                'target': get_target(fixed_scm),
                'parents': list(get_parents(fixed_scm, get_target(fixed_scm))),  # Convert frozenset to list
                'structure_type': 'fork',
                'num_variables': 3
            },
            'analysis': analysis if 'analysis' in locals() else {},
            'episode_performances': trainer.episode_performances,
            'convergence_metrics': {
                'best_overall': trainer.convergence_metrics['best_per_category'].get('single', float('-inf')),
                'rolling_averages': trainer.convergence_metrics['rolling_averages'][-20:] if trainer.convergence_metrics['rolling_averages'] else []
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
                elif isinstance(obj, frozenset):
                    return list(obj)  # Convert frozenset to list
                return obj
            
            json.dump(convert(save_data), f, indent=2)
        
        logger.info(f"\n✅ Single SCM maximization test complete! Results saved to {results_file}")
        
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
    
    return 0


if __name__ == "__main__":
    # Import needed for intervention handling
    from src.causal_bayes_opt.data_structures.sample import get_values
    
    sys.exit(main())