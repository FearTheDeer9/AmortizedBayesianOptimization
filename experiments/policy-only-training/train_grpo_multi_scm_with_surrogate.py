#!/usr/bin/env python3
"""
Multi-SCM GRPO Training with Enhanced Surrogate Integration and Dynamic Rotation.

This script combines:
1. Multi-SCM training with diverse graph structures
2. AVICI surrogate integration for structure learning
3. Dynamic SCM rotation after each episode
4. Early rotation based on convergence detection

Key features:
- Rotates SCMs after each episode (when max_interventions reached)
- Early rotation if same (variable, quantile) selected 3+ times with >90% probability
- Surrogate-guided learning with composite rewards
- Configurable convergence patience and threshold
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
from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory, get_scm_info
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


class EnhancedGRPOTrainer(JointACBOTrainer):
    """
    Enhanced trainer with dynamic SCM rotation and convergence detection.
    
    Features:
    - Rotates SCMs after each episode
    - Early rotation on convergence detection
    - Tracks intervention selection probabilities
    - Integrates surrogate for structure learning
    """
    
    def __init__(self, config):
        """Initialize enhanced trainer with rotation and tracking capabilities."""
        # Store config for override methods
        self.config = config
        
        # Initialize parent trainer
        super().__init__(config=config)
        
        # Enhanced rotation configuration
        rotation_config = config.get('scm_rotation', {})
        self.rotate_every_episode = rotation_config.get('rotate_every_episode', True)
        self.convergence_patience = rotation_config.get('convergence_patience', 3)
        self.convergence_threshold = rotation_config.get('convergence_threshold', 0.9)
        self.enable_early_rotation = rotation_config.get('enable_early_rotation', True)
        
        # Intervention tracking for convergence detection
        self.intervention_history = []  # Store (var_idx, quantile_idx, probability) tuples
        self.consecutive_same_count = 0
        self.last_intervention = None  # (var_idx, quantile_idx) tuple
        
        # SCM management
        self.scm_factory = None
        self.scm_history = []  # Track all SCMs used
        self.episodes_on_current_scm = 0
        self.total_rotations = 0
        self.early_rotations = 0
        self.episode_end_rotations = 0
        
        # Performance tracking
        self.episode_performances = []
        self.convergence_metrics = {
            'rolling_average_window': 20,
            'rolling_averages': [],
            'best_per_size': {'small': float('inf'), 'medium': float('inf'), 'large': float('inf')},
            'convergence_events': []  # Track when convergence triggers rotation
        }
        
        # CSV logging setup
        self.csv_data = []
        self.checkpoint_episodes = [10, 25, 50, 75, 100, 150, 200]
        
        # Episode-level tracking
        self.current_episode_interventions = []
        self.current_episode_data = {
            'selections': [],
            'probabilities': [],
            'target_values': [],
            'rewards': [],
            'num_vars': 0,
            'convergence_triggered': False
        }
        
        logger.info(f"Initialized EnhancedGRPOTrainer:")
        logger.info(f"  - Rotate every episode: {self.rotate_every_episode}")
        logger.info(f"  - Convergence patience: {self.convergence_patience}")
        logger.info(f"  - Convergence threshold: {self.convergence_threshold:.1%}")
        logger.info(f"  - Early rotation enabled: {self.enable_early_rotation}")
        logger.info(f"  - Surrogate enabled: {config.get('use_surrogate', True)}")
    
    def _check_convergence(self, var_idx: int, quantile_idx: int, probability: float) -> bool:
        """
        Check if convergence criteria met for early rotation.
        
        Returns True if same (variable, quantile) selected consecutively
        with probability > threshold for patience iterations.
        """
        if not self.enable_early_rotation:
            return False
        
        current_intervention = (var_idx, quantile_idx)
        
        # Check if probability exceeds threshold
        if probability > self.convergence_threshold:
            if current_intervention == self.last_intervention:
                self.consecutive_same_count += 1
                
                if self.consecutive_same_count >= self.convergence_patience:
                    logger.info(f"🎯 CONVERGENCE DETECTED: {current_intervention} selected "
                              f"{self.consecutive_same_count} times with >{self.convergence_threshold:.1%} probability")
                    self.convergence_metrics['convergence_events'].append({
                        'episode': self.episode_count if hasattr(self, 'episode_count') else 0,
                        'intervention': current_intervention,
                        'count': self.consecutive_same_count,
                        'probability': probability
                    })
                    return True
            else:
                # Different intervention selected, reset counter
                self.consecutive_same_count = 1
                self.last_intervention = current_intervention
        else:
            # Probability too low, don't count as consecutive
            self.consecutive_same_count = 0
            self.last_intervention = None
        
        return False
    
    def _reset_convergence_tracking(self):
        """Reset convergence tracking for new episode/SCM."""
        self.consecutive_same_count = 0
        self.last_intervention = None
        self.current_episode_interventions = []
        self.current_episode_data['convergence_triggered'] = False
    
    def train(self, scms):
        """
        Override train method to force usage of our enhanced episode runner.
        This ensures our tracking and rotation logic is actually used.
        """
        logger.info("🎯 EnhancedGRPOTrainer.train() called - our enhanced version is active!")
        
        # Store the original method and replace with our enhanced version
        original_run_episode = self._run_grpo_episode
        self._run_grpo_episode = self._run_grpo_episode_with_tracking
        
        try:
            # Call parent's train method with our method in place
            result = super().train(scms)
        finally:
            # Restore original method (good practice, though train usually runs once)
            self._run_grpo_episode = original_run_episode
        
        return result
    
    def _rotate_scm(self, reason: str = "episode_end"):
        """
        Rotate to a new SCM and track rotation statistics.
        
        Args:
            reason: Why rotation occurred ("episode_end" or "convergence")
        """
        # Track rotation statistics
        self.total_rotations += 1
        if reason == "convergence":
            self.early_rotations += 1
        else:
            self.episode_end_rotations += 1
        
        # Log rotation event
        if hasattr(self, 'current_scm') and self.current_scm:
            old_target = get_target(self.current_scm)
            old_vars = len(get_variables(self.current_scm))
            
            # Track SCM history
            scm_info = get_scm_info(self.current_scm)
            scm_info['episodes_used'] = self.episodes_on_current_scm
            scm_info['rotation_reason'] = reason
            self.scm_history.append(scm_info)
            
            logger.info(f"🔄 ROTATING SCM: {old_vars} vars, target={old_target}")
            logger.info(f"   Reason: {reason}")
            logger.info(f"   Episodes on SCM: {self.episodes_on_current_scm}")
            logger.info(f"   Total rotations: {self.total_rotations} (early: {self.early_rotations}, "
                       f"episode_end: {self.episode_end_rotations})")
        
        # Generate new SCM
        if self.scm_factory and hasattr(self, 'sampling_config'):
            new_scm = self.scm_factory.get_random_scm(**self.sampling_config)
            self.current_scm = new_scm
            self.current_scm_metadata = dict(new_scm.get('metadata', {}))
            
            # Extract detailed SCM information
            new_target = get_target(new_scm)
            new_vars = list(get_variables(new_scm))
            new_parents = list(get_parents(new_scm, new_target))
            
            # Extract coefficients for parents
            coefficients = {}
            if hasattr(new_scm, 'mechanisms') and new_target in new_scm['mechanisms']:
                target_mechanism = new_scm['mechanisms'][new_target]
                if hasattr(target_mechanism, 'coefficients'):
                    coefficients = dict(target_mechanism['coefficients'])
            
            # Print detailed SCM information
            logger.info(f"\n" + "="*60)
            logger.info(f"📊 NEW SCM DETAILS:")
            logger.info(f"   Variables: {new_vars}")
            logger.info(f"   Target: {new_target}")
            logger.info(f"   True Parents: {new_parents}")
            if coefficients:
                logger.info(f"   Parent Coefficients:")
                for parent, coeff in coefficients.items():
                    logger.info(f"      {parent} → {new_target}: {coeff:.3f}")
            else:
                logger.info(f"   Parent Coefficients: Not available")
            logger.info(f"   Structure Type: {new_scm.get('metadata', {}).get('structure_type', 'unknown')}")
            logger.info(f"="*60 + "\n")
        
        # Reset episode counter and convergence tracking
        self.episodes_on_current_scm = 0
        self._reset_convergence_tracking()
    
    def _run_grpo_episode_with_tracking(self, episode_idx, scm, scm_name, key):
        """
        Enhanced episode runner with intervention tracking and early rotation.
        
        This wraps the parent's _run_grpo_episode to add:
        1. Intervention probability tracking
        2. Convergence detection
        3. Early rotation triggering
        """
        # Reset episode tracking
        self._reset_convergence_tracking()
        self.current_episode_data['num_vars'] = len(get_variables(scm))
        
        # Store original max_interventions
        original_max_interventions = self.max_interventions
        
        # Track if we trigger early rotation
        early_rotation_triggered = False
        
        # Custom intervention loop with tracking
        target_var = get_target(scm)
        variables = list(get_variables(scm))
        true_parents = list(get_parents(scm, target_var))
        
        # Print SCM info at start of episode
        logger.info(f"\n" + "="*60)
        logger.info(f"📊 EPISODE {episode_idx} SCM DETAILS:")
        logger.info(f"   Variables: {variables}")
        logger.info(f"   Target: {target_var}")
        logger.info(f"   True Parents: {true_parents}")
        
        # Extract coefficients
        coefficients = {}
        if hasattr(scm, 'mechanisms') and target_var in scm['mechanisms']:
            target_mechanism = scm['mechanisms'][target_var]
            if hasattr(target_mechanism, 'coefficients'):
                coefficients = dict(target_mechanism['coefficients'])
        
        if coefficients:
            logger.info(f"   Parent Coefficients:")
            for parent, coeff in coefficients.items():
                logger.info(f"      {parent} → {target_var}: {coeff:.3f}")
        else:
            logger.info(f"   Parent Coefficients: Not available")
        logger.info(f"="*60)
        
        logger.info(f"\nEpisode {episode_idx}: SCM with {len(variables)} vars, "
                   f"target={target_var}, parents={true_parents}")
        
        # Initialize buffer for episode
        from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
        from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
        
        buffer = ExperienceBuffer()
        
        # Add observational data
        obs_samples = sample_from_linear_scm(scm, self.obs_per_episode, seed=int(key[0]) % 1000000)
        for sample in obs_samples:
            buffer.add_observation(sample)
        
        # Intervention loop with tracking
        for int_idx in range(self.max_interventions):
            # Generate intervention using policy
            key, int_key = random.split(key)
            
            # Get policy output (need to access internals)
            from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
            from src.causal_bayes_opt.utils.variable_mapping import VariableMapper
            
            mapper = VariableMapper(variables, target_variable=target_var)
            tensor, _ = buffer_to_three_channel_tensor(buffer, target_var, max_history_size=100, standardize=False)
            
            # Apply policy
            key, policy_key = random.split(key)
            policy_output = self.policy_fn.apply(
                self.policy_params, policy_key, tensor, mapper.target_idx
            )
            
            # Extract probabilities based on architecture
            if 'quantile_scores' in policy_output:
                # Quantile architecture
                quantile_scores = policy_output['quantile_scores']
                flat_scores = quantile_scores.flatten()
                probabilities = jax.nn.softmax(flat_scores)
                
                # Sample intervention
                key, sample_key = random.split(key)
                flat_idx = random.categorical(sample_key, jnp.log(probabilities + 1e-8))
                
                # Convert flat index to (var, quantile)
                num_vars = len(variables)
                num_quantiles = flat_scores.shape[0] // num_vars
                var_idx = int(flat_idx // num_quantiles)
                quantile_idx = int(flat_idx % num_quantiles)
                
                # Get probability of selected intervention
                selection_prob = float(probabilities[flat_idx])
                
                # Track intervention
                self.current_episode_interventions.append({
                    'var_idx': var_idx,
                    'quantile_idx': quantile_idx,
                    'probability': selection_prob,
                    'var_name': variables[var_idx] if var_idx < len(variables) else 'unknown'
                })
                
                # Check for convergence
                if self._check_convergence(var_idx, quantile_idx, selection_prob):
                    logger.info(f"   Early rotation triggered at intervention {int_idx + 1}")
                    self.current_episode_data['convergence_triggered'] = True
                    early_rotation_triggered = True
                    break
                
                # Generate actual intervention value (simplified for now)
                key, value_key = random.split(key)
                if var_idx < len(variables) and variables[var_idx] != target_var:
                    # Map quantile to value (simplified - use quantiles of standard normal)
                    quantiles = jnp.array([0.25, 0.5, 0.75])  # 25%, 50%, 75% quantiles
                    if quantile_idx < len(quantiles):
                        intervention_value = float(random.normal(value_key) * 2.0)
                    else:
                        intervention_value = float(random.normal(value_key))
                    
                    # Create and apply intervention
                    intervention = create_perfect_intervention(
                        targets=frozenset([variables[var_idx]]),
                        values={variables[var_idx]: intervention_value}
                    )
                    
                    # Sample post-intervention
                    post_data = sample_with_intervention(scm, intervention, 1, seed=int(value_key[0]) % 1000000)
                    if post_data:
                        buffer.add_intervention(intervention, post_data[0])
                        
                        # Track target value
                        target_value = post_data[0].values[target_var]
                        self.current_episode_data['target_values'].append(target_value)
            else:
                # Traditional architecture (simplified handling)
                logger.debug("Traditional architecture intervention tracking not fully implemented")
        
        # Get surrogate parent probabilities for the last intervention
        logger.info(f"\n🔍 CHECKING SURROGATE AVAILABILITY...")
        if not hasattr(self, 'surrogate_predict_fn'):
            logger.info("   ❌ No surrogate_predict_fn attribute")
        elif self.surrogate_predict_fn is None:
            logger.info("   ❌ surrogate_predict_fn is None")
        else:
            logger.info("   ✅ Surrogate is available")
            
        if buffer and hasattr(self, 'surrogate_predict_fn') and self.surrogate_predict_fn:
            try:
                # Get final buffer state and create tensor
                final_tensor, final_mapper = buffer_to_three_channel_tensor(
                    buffer, target_var, max_history_size=100, standardize=False
                )
                
                # Get surrogate predictions
                key, surrogate_key = random.split(key)
                surrogate_output = self.surrogate_predict_fn(
                    final_tensor, target_var, variables
                )
                
                if 'parent_probs' in surrogate_output:
                    parent_probs = surrogate_output['parent_probs']
                    
                    # Print surrogate parent probabilities
                    logger.info(f"\n🔮 SURROGATE PARENT PROBABILITIES (Last Intervention):")
                    logger.info(f"   Target: {target_var}, True Parents: {true_parents}")
                    
                    for i, var in enumerate(variables):
                        if var != target_var and i < len(parent_probs):
                            prob = float(parent_probs[i])
                            is_parent = var in true_parents
                            marker = "✅" if is_parent else ""
                            logger.info(f"     {var}: {prob:.3f} {marker}")
                    
                    # Calculate discrimination score
                    true_parent_probs = []
                    false_parent_probs = []
                    for i, var in enumerate(variables):
                        if var != target_var and i < len(parent_probs):
                            prob = float(parent_probs[i])
                            if var in true_parents:
                                true_parent_probs.append(prob)
                            else:
                                false_parent_probs.append(prob)
                    
                    if true_parent_probs and false_parent_probs:
                        avg_true = np.mean(true_parent_probs)
                        avg_false = np.mean(false_parent_probs)
                        discrimination = avg_true - avg_false
                        logger.info(f"   Discrimination Score: {discrimination:+.3f} "
                                  f"(True avg: {avg_true:.3f}, False avg: {avg_false:.3f})")
                else:
                    logger.info("   ⚠️ Surrogate output format unexpected")
            except Exception as e:
                logger.debug(f"Could not get surrogate predictions: {e}")
        
        # Log episode summary
        if self.current_episode_interventions:
            avg_prob = np.mean([i['probability'] for i in self.current_episode_interventions])
            logger.info(f"   Interventions: {len(self.current_episode_interventions)}, "
                       f"Avg probability: {avg_prob:.3f}")
            
            # Check which variables were selected most
            var_counts = {}
            for intervention in self.current_episode_interventions:
                var_name = intervention['var_name']
                var_counts[var_name] = var_counts.get(var_name, 0) + 1
            
            top_vars = sorted(var_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            logger.info(f"   Top selected vars: {top_vars}")
        
        # Increment episode counter on current SCM
        self.episodes_on_current_scm += 1
        
        # Determine if rotation needed
        should_rotate = False
        rotation_reason = None
        
        if early_rotation_triggered:
            should_rotate = True
            rotation_reason = "convergence"
        elif self.rotate_every_episode:
            should_rotate = True
            rotation_reason = "episode_end"
        
        # Perform rotation if needed
        if should_rotate:
            logger.info(f"\n🔄 ROTATION TRIGGERED: {rotation_reason}")
            self._rotate_scm(rotation_reason)
        else:
            logger.info(f"\n📍 NO ROTATION: Staying on same SCM (episode {self.episodes_on_current_scm})")
        
        # Call parent's episode runner for GRPO updates
        # Note: We've already done the episode, so this is mainly for the GRPO update
        # We might need to refactor this to avoid double execution
        result = super()._run_grpo_episode(episode_idx, scm, scm_name, key)
        
        # Track episode performance
        self._track_episode_performance(episode_idx)
        
        return result
    
    def _track_episode_performance(self, episode_idx):
        """Track performance metrics for analysis."""
        # Calculate metrics from episode data
        metrics = {
            'episode': episode_idx,
            'num_vars': self.current_episode_data['num_vars'],
            'num_interventions': len(self.current_episode_interventions),
            'convergence_triggered': self.current_episode_data['convergence_triggered'],
            'mean_target': np.mean(self.current_episode_data['target_values']) if self.current_episode_data['target_values'] else 0,
            'best_target': np.min(self.current_episode_data['target_values']) if self.current_episode_data['target_values'] else 0,
            'total_rotations': self.total_rotations,
            'early_rotations': self.early_rotations
        }
        
        self.episode_performances.append(metrics)
        
        # Save checkpoint if needed
        if episode_idx in self.checkpoint_episodes:
            self._save_checkpoint(episode_idx)
        
        # Log summary every 10 episodes
        if episode_idx % 10 == 0:
            recent_metrics = self.episode_performances[-10:]
            avg_interventions = np.mean([m['num_interventions'] for m in recent_metrics])
            convergence_rate = np.mean([m['convergence_triggered'] for m in recent_metrics])
            
            logger.info(f"\n📊 Last 10 episodes summary:")
            logger.info(f"   Avg interventions: {avg_interventions:.1f}")
            logger.info(f"   Convergence rate: {convergence_rate:.1%}")
            logger.info(f"   Total rotations: {self.total_rotations}")
            logger.info(f"   Early/Total: {self.early_rotations}/{self.total_rotations}")
    
    def _save_checkpoint(self, episode_idx):
        """Save checkpoint with enhanced metrics."""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        episode_dir = checkpoint_dir / f"episode_{episode_idx}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        
        # Save using checkpoint_utils
        from src.causal_bayes_opt.utils.checkpoint_utils import save_checkpoint
        
        architecture = {
            'hidden_dim': self.hidden_dim,
            'num_layers': self.config.get('architecture', {}).get('num_layers', 4),
            'num_heads': self.config.get('architecture', {}).get('num_heads', 8),
            'key_size': self.config.get('architecture', {}).get('key_size', 32),
            'dropout': self.config.get('architecture', {}).get('dropout', 0.1),
            'architecture_type': self.config.get('policy_architecture', 'quantile')
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
                'total_episodes': self.config.get('max_episodes', 200),
                'rotation_config': self.config.get('scm_rotation', {})
            },
            metadata={
                'episode': episode_idx,
                'total_rotations': self.total_rotations,
                'early_rotations': self.early_rotations,
                'convergence_events': len(self.convergence_metrics['convergence_events']),
                'scm_diversity': len(self.scm_history),
                'trainer': 'EnhancedGRPOTrainer'
            }
        )
        
        logger.info(f"  💾 Saved checkpoint at episode {episode_idx}")
    
    def _run_grpo_episode(self, episode_idx, scm, scm_name, key):
        """Override to use our enhanced tracking version."""
        return self._run_grpo_episode_with_tracking(episode_idx, scm, scm_name, key)


def create_enhanced_config(
    max_episodes: int = 200,
    verbose: bool = False,
    enable_wandb: bool = False,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Create configuration for enhanced multi-SCM training with surrogate and dynamic rotation.
    """
    
    config = {
        # Core episode settings
        'max_episodes': max_episodes,
        'obs_per_episode': 20,
        'max_interventions': 10,  # Per episode before rotation
        
        # Phase management (for JointACBOTrainer compatibility)
        'episodes_per_phase': 999999,  # Effectively disable phase switching
        'initial_phase': 'policy',
        
        # Architecture
        'policy_architecture': 'quantile',
        
        # Surrogate integration
        'use_surrogate': True,
        'use_grpo_rewards': True,
        
        # Exploration settings
        'use_fixed_std': True,
        'fixed_std': 1.0,
        
        # Learning settings
        'learning_rate': 1e-3,
        
        # GRPO configuration
        'grpo_config': {
            'group_size': 32,
            'entropy_coefficient': 0.001,
            'clip_ratio': 1.0,
            'gradient_clip': 10.0,
            'ppo_epochs': 4,
            'normalize_advantages': True
        },
        
        # Reward weights (with surrogate info gain)
        'reward_weights': {
            'target': 0.0,
            'parent': 0.0,
            'info_gain': 0  # From surrogate
        },
        
        # Composite reward for surrogate integration
        'reward_type': 'composite',
        
        # NEW: Enhanced SCM rotation configuration  
        'rotate_after_episode': True,          # Rotate after each episode (for JointACBOTrainer)
        'convergence_patience': 3,             # Consecutive same interventions for early rotation
        'convergence_threshold': 0.9,          # Probability threshold for convergence  
        'enable_early_rotation': True,         # Enable convergence-based rotation
        
        # Keep nested structure for backward compatibility
        'scm_rotation': {
            'rotate_every_episode': True,      # Always rotate after episode
            'convergence_patience': 3,         # Consecutive same interventions for early rotation
            'convergence_threshold': 0.9,      # Probability threshold for convergence
            'enable_early_rotation': True,     # Enable convergence-based rotation
        },
        
        # Surrogate model configuration
        'surrogate_checkpoint_path': 'experiments/surrogate-only-training/scripts/checkpoints/avici_runs/avici_style_20250822_213115/checkpoint_step_200.pkl',
        'surrogate_lr': 1e-3,
        'surrogate_hidden_dim': 128,
        'surrogate_layers': 4,
        'surrogate_heads': 8,
        
        # Wall clock timeout
        'wall_clock_timeout_minutes': 30,
        
        # General settings
        'batch_size': 32,
        'seed': seed,
        'verbose': verbose,
        'checkpoint_dir': 'checkpoints/grpo_enhanced',
        
        # WandB logging
        'logging': {
            'wandb': {
                'enabled': enable_wandb,
                'project': 'causal-bayes-opt-grpo-enhanced',
                'name': f'multi_scm_surrogate_{max_episodes}ep',
                'tags': ['multi_scm', 'grpo', 'surrogate', 'dynamic_rotation'],
                'log_frequency': 1
            }
        }
    }
    
    return config


def main():
    """Main training function for enhanced multi-SCM GRPO with surrogate and dynamic rotation."""
    
    # Version confirmation
    logger.info("🚀 SCRIPT VERSION: Enhanced with SCM info, surrogate probs, and convergence detection v2")
    logger.info("📍 Script location: train_grpo_multi_scm_with_surrogate.py")
    
    parser = argparse.ArgumentParser(description="Enhanced Multi-SCM GRPO training")
    parser.add_argument('--episodes', type=int, default=200,
                        help='Number of training episodes')
    parser.add_argument('--patience', type=int, default=3,
                        help='Convergence patience (consecutive same interventions)')
    parser.add_argument('--threshold', type=float, default=0.9,
                        help='Convergence threshold (probability)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable WandB logging')
    parser.add_argument('--max-time-minutes', type=int, default=None,
                        help='Maximum training time in minutes')
    parser.add_argument('--surrogate-checkpoint', type=str, default=None,
                        help='Path to surrogate checkpoint (overrides config)')
    parser.add_argument('--policy-checkpoint', type=str, default=None,
                        help='Path to policy checkpoint to resume from')
    parser.add_argument('--checkpoint-output', type=str, default=None,
                        help='Explicit path for final checkpoint output')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--target-weight', type=float, default=None,
                        help='Target improvement reward weight')
    parser.add_argument('--parent-weight', type=float, default=None,
                        help='Parent accuracy reward weight')
    parser.add_argument('--info-weight', type=float, default=None,
                        help='Information gain reward weight')
    parser.add_argument('--max-interventions', type=int, default=None,
                        help='Max interventions per episode')
    parser.add_argument('--obs-per-episode', type=int, default=None,
                        help='Observations per episode')
    
    args = parser.parse_args()
    
    # Create timestamped run directory
    run_name = f"grpo_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = Path("checkpoints/grpo_enhanced") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create configuration first
    config = create_enhanced_config(
        max_episodes=args.episodes,
        verbose=args.verbose,
        enable_wandb=args.wandb,
        seed=args.seed
    )
    
    # Override convergence settings from args (set both locations)
    config['convergence_patience'] = args.patience
    config['convergence_threshold'] = args.threshold
    config['scm_rotation']['convergence_patience'] = args.patience
    config['scm_rotation']['convergence_threshold'] = args.threshold
    
    # Override surrogate checkpoint if provided
    if args.surrogate_checkpoint:
        config['surrogate_checkpoint_path'] = args.surrogate_checkpoint
        logger.info(f"Using surrogate checkpoint: {args.surrogate_checkpoint}")
    
    # Override learning rate if provided
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    
    # Override reward weights if provided
    if args.target_weight is not None:
        config['reward_weights']['target'] = args.target_weight
    if args.parent_weight is not None:
        config['reward_weights']['parent'] = args.parent_weight
    if args.info_weight is not None:
        config['reward_weights']['info_gain'] = args.info_weight
    
    # Override episode parameters if provided
    if args.max_interventions is not None:
        config['max_interventions'] = args.max_interventions
    if args.obs_per_episode is not None:
        config['obs_per_episode'] = args.obs_per_episode
    
    # Set time limit if provided
    if args.max_time_minutes:
        config['wall_clock_timeout_minutes'] = args.max_time_minutes
    
    # Update checkpoint directory
    config['checkpoint_dir'] = str(checkpoint_dir)
    
    # Print header with configuration details
    logger.info("\n" + "="*70)
    logger.info("ENHANCED MULTI-SCM GRPO TRAINING WITH SURROGATE")
    logger.info("="*70)
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"Convergence patience: {args.patience}")
    logger.info(f"Convergence threshold: {args.threshold:.1%}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    logger.info("="*70 + "\n")
    
    # Log critical settings
    logger.info("Critical settings:")
    logger.info(f"  - Policy: {config['policy_architecture']}")
    logger.info(f"  - Surrogate: {'Enabled' if config['use_surrogate'] else 'Disabled'}")
    logger.info(f"  - Rotation: Every episode + convergence detection")
    logger.info(f"  - Convergence: {args.patience} consecutive @ >{args.threshold:.1%}")
    logger.info(f"  - Learning rate: {config['learning_rate']}")
    logger.info(f"  - GRPO group size: {config['grpo_config']['group_size']}")
    
    # Document reward components and entropy
    logger.info("\n📊 REWARD COMPONENTS & ENTROPY:")
    logger.info(f"  Reward Type: {config['reward_type']}")
    logger.info(f"  Reward Weights:")
    for component, weight in config['reward_weights'].items():
        logger.info(f"    - {component}: {weight}")
    logger.info(f"  Entropy Coefficient: {config['grpo_config']['entropy_coefficient']}")
    logger.info(f"  Note: Higher entropy encourages exploration, lower encourages exploitation")
    logger.info("")
    
    # Initialize WandB if enabled
    wandb_manager = None
    if args.wandb:
        logger.info("Initializing WandB...")
        try:
            wandb_manager = WandBManager()
            wandb_run = wandb_manager.setup(config, experiment_name=run_name)
            if wandb_run:
                logger.info(f"✅ WandB initialized")
            else:
                logger.warning("⚠️  WandB initialization failed, continuing without logging")
                wandb_manager = None
        except Exception as e:
            logger.warning(f"⚠️  WandB initialization failed: {e}")
            wandb_manager = None
    
    # Create SCM factory for random generation
    logger.info("Setting up SCM factory for diverse graph generation...")
    
    scm_factory = VariableSCMFactory(
        seed=args.seed,
        noise_scale=0.5,
        coefficient_range=(-3.0, 3.0),
        vary_intervention_ranges=True,
        use_output_bounds=True
    )
    
    # Define sampling configuration
    sampling_config = {
        'variable_counts': [3, 4, 5, 6, 8, 10],  # Diverse sizes
        'structure_types': ["fork", "chain", "collider", "mixed", "random"],
        'edge_density_range': (0.2, 0.6),
        'name_prefix': 'enhanced'
    }
    
    logger.info(f"✅ SCM factory configured:")
    logger.info(f"  Variable counts: {sampling_config['variable_counts']}")
    logger.info(f"  Structure types: {sampling_config['structure_types']}")
    logger.info(f"  Edge density: {sampling_config['edge_density_range']}")
    
    # Create SCM generator function
    def random_scm_generator():
        return scm_factory.get_random_scm(**sampling_config)
    
    # Initialize trainer
    logger.info("\nInitializing enhanced GRPO trainer...")
    trainer = EnhancedGRPOTrainer(config=config)
    
    # Load policy checkpoint if provided
    if args.policy_checkpoint and Path(args.policy_checkpoint).exists():
        logger.info(f"Loading policy checkpoint from: {args.policy_checkpoint}")
        from src.causal_bayes_opt.utils.checkpoint_utils import load_checkpoint
        checkpoint = load_checkpoint(Path(args.policy_checkpoint))
        trainer.policy_params = checkpoint['params']
        
        # Set starting episode for continuation
        # Check multiple possible fields for episode count
        last_episode = (
            checkpoint.get('metadata', {}).get('episode', 0) or 
            checkpoint.get('metadata', {}).get('total_episodes', 0) or
            0
        )
        trainer.start_episode = last_episode + 1 if last_episode > 0 else 0
        logger.info(f"  Loaded policy checkpoint with {last_episode} completed episodes")
        logger.info(f"  Will continue training from episode {trainer.start_episode}")
    else:
        trainer.start_episode = 0  # Start from scratch
    
    # Set SCM factory and config for rotation
    trainer.scm_factory = scm_factory
    trainer.sampling_config = sampling_config
    
    logger.info("✅ Trainer initialized")
    
    # Save experiment configuration
    experiment_config = {
        'args': vars(args),
        'config': config,
        'run_name': run_name,
        'sampling_config': sampling_config
    }
    
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(experiment_config, f, indent=2, default=str)
    
    logger.info(f"💾 Saved experiment config: {checkpoint_dir / 'config.json'}")
    
    # Start training
    logger.info("\n" + "="*70)
    logger.info("STARTING ENHANCED TRAINING")
    logger.info("="*70 + "\n")
    
    try:
        # Train with diverse SCMs and dynamic rotation
        results = trainer.train(scms=random_scm_generator)
        
        # Save final checkpoint
        logger.info(f"\n💾 Saving final checkpoint...")
        
        from src.causal_bayes_opt.utils.checkpoint_utils import save_checkpoint
        
        # Use explicit output path if provided
        if args.checkpoint_output:
            final_checkpoint_path = Path(args.checkpoint_output)
        else:
            final_checkpoint_path = checkpoint_dir / 'final_policy.pkl'
        
        save_checkpoint(
            path=final_checkpoint_path,
            params=trainer.policy_params,
            architecture={
                'hidden_dim': trainer.hidden_dim,
                'num_layers': trainer.num_layers,
                'num_heads': trainer.num_heads,
                'key_size': trainer.key_size,
                'dropout': trainer.dropout,
                'policy_architecture': config['policy_architecture'],
                'architecture_type': 'quantile'
            },
            model_type='policy',
            model_subtype='grpo',
            training_config=config,
            metadata={
                'experiment_type': 'enhanced_multi_scm',
                'run_name': run_name,
                'total_episodes': len(trainer.episode_performances),
                'total_rotations': trainer.total_rotations,
                'early_rotations': trainer.early_rotations,
                'convergence_events': len(trainer.convergence_metrics['convergence_events']),
                'scm_diversity': len(trainer.scm_history),
                'trainer_class': 'EnhancedGRPOTrainer'
            }
        )
        
        # Save detailed results
        results_data = {
            'config': config,
            'sampling_config': sampling_config,
            'scm_history': trainer.scm_history,
            'episode_performances': trainer.episode_performances,
            'convergence_metrics': trainer.convergence_metrics,
            'rotation_stats': {
                'total': trainer.total_rotations,
                'early': trainer.early_rotations,
                'episode_end': trainer.episode_end_rotations,
                'early_ratio': trainer.early_rotations / max(trainer.total_rotations, 1)
            }
        }
        
        results_file = checkpoint_dir / 'training_results.json'
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"✅ Training complete!")
        logger.info(f"📊 Results saved to: {results_file}")
        logger.info(f"🎯 Checkpoint directory: {checkpoint_dir}")
        
        # Print summary statistics
        logger.info("\n" + "="*70)
        logger.info("TRAINING SUMMARY")
        logger.info("="*70)
        logger.info(f"Total episodes: {len(trainer.episode_performances)}")
        logger.info(f"Total SCM rotations: {trainer.total_rotations}")
        logger.info(f"Early rotations (convergence): {trainer.early_rotations}")
        logger.info(f"Episode-end rotations: {trainer.episode_end_rotations}")
        logger.info(f"Early rotation rate: {trainer.early_rotations / max(trainer.total_rotations, 1):.1%}")
        logger.info(f"Unique SCMs used: {len(trainer.scm_history)}")
        logger.info(f"Convergence events: {len(trainer.convergence_metrics['convergence_events'])}")
        
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