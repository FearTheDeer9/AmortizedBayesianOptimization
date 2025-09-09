#!/usr/bin/env python3
"""
Ground Truth COEFFICIENT Channel GRPO Training with Per-Intervention SCM Rotation.

This script modifies train_grpo_ground_truth_rotation.py to use actual coefficient
values in the fourth channel instead of progressive parent probabilities.

Key features:
1. Uses ground truth parent information with ACTUAL COEFFICIENTS
2. Each SCM maintains persistent buffer that grows over time
3. Option to pre-populate with initial interventions
4. Per-intervention SCM rotation to prevent hyperspecialization

Coefficient channel model:
- Non-parents get value 0
- Target itself gets value 0
- Direct parents get their actual coefficient value connecting them to the target
- This provides direct causal strength information to the policy
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
from src.causal_bayes_opt.data_structures.scm import get_parents, get_variables, get_target, get_mechanisms
from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
from src.causal_bayes_opt.environments.sampling import sample_with_intervention
from src.causal_bayes_opt.training.utils.wandb_setup import WandBManager

# Additional imports needed for the working method
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm  
from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
from src.causal_bayes_opt.training.four_channel_converter import buffer_to_four_channel_tensor
from src.causal_bayes_opt.data_structures.sample import get_intervention_targets, is_interventional
from src.causal_bayes_opt.training.five_channel_converter import create_uniform_posterior
from src.causal_bayes_opt.training.ground_truth_channel_converter import buffer_to_ground_truth_four_channel_tensor
from src.causal_bayes_opt.data_structures.sample import get_values
import time

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
        self.episodes_per_scm = config.get('episodes_per_scm', 5)  # Rotate every N episodes
        self.total_rotations = 0
        self.early_rotations = 0
        self.episode_end_rotations = 0
        self.scheduled_rotations = 0  # Track scheduled rotations
        self.convergence_detected = False  # Flag for early rotation due to convergence
        
        # NEW: Per-batch rotation management with persistent buffers
        self.scm_pool = []  # Pre-generated pool of (SCM, Buffer) pairs
        self.scm_pool_metadata = []  # Metadata for each SCM in pool
        self.current_pool_index = 0
        self.batch_rotation_count = 0
        self.rotate_after_batch = config.get('rotate_after_batch', True)
        
        # Ground truth channel configuration
        self.use_ground_truth_channel = config.get('use_ground_truth_channel', False)
        self.convergence_rate_factor = config.get('convergence_rate_factor', 0.2)
        self.initial_interventions_per_scm = config.get('initial_interventions_per_scm', 0)
        # Pre-population ranges
        self.min_obs = config.get('min_obs', 10)
        self.max_obs = config.get('max_obs', 150)
        self.min_int = config.get('min_int', 5)
        self.max_int = config.get('max_int', 20)
        
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
        self.completed_episodes_count = 0  # Simple episode counter
        self.current_episode_data = {
            'selections': [],
            'probabilities': [],
            'target_values': [],
            'rewards': [],
            'num_vars': 0,
            'convergence_triggered': False
        }
        
        # Wall-clock based checkpointing
        self.training_start_time = time.time()
        self.last_checkpoint_time = time.time()
        self.checkpoint_interval_minutes = 30  # Save every 30 minutes
        self.total_interventions_count = 0
        self.interventions_per_checkpoint = 100  # Save every 100 interventions
        
        logger.info(f"Initialized EnhancedGRPOTrainer:")
        logger.info(f"  - Rotate every episode: {self.rotate_every_episode}")
        logger.info(f"  - Convergence patience: {self.convergence_patience}")
        logger.info(f"  - Convergence threshold: {self.convergence_threshold:.1%}")
        logger.info(f"  - Early rotation enabled: {self.enable_early_rotation}")
        logger.info(f"  - Surrogate enabled: {config.get('use_surrogate', True)}")
        logger.info(f"  - Per-batch rotation: {self.rotate_after_batch}")
        logger.info(f"  - Ground truth channel: {self.use_ground_truth_channel}")
        logger.info(f"  📁 Checkpoint strategy:")
        logger.info(f"    - Episode checkpoints: {self.checkpoint_episodes}")
        logger.info(f"    - Wall-clock checkpoint: Every {self.checkpoint_interval_minutes} minutes")
        logger.info(f"    - Intervention checkpoint: Every 100 interventions")
        if self.use_ground_truth_channel:
            logger.info(f"  - Convergence rate factor: {self.convergence_rate_factor}")
            logger.info(f"  - Initial interventions per SCM: {self.initial_interventions_per_scm}")
            
            # Monkey-patch buffer_to_four_channel_tensor when using ground truth
            self._original_buffer_to_four_channel = buffer_to_four_channel_tensor
            
            # Store the coefficient channel function as an instance method
            self.buffer_to_coefficient_channel_tensor = self._create_coefficient_channel_function()
            
            def ground_truth_wrapper(buffer, target_var, **kwargs):
                """Wrapper that uses coefficient channel when available."""
                if hasattr(self, '_current_intervention_scm'):
                    # Use coefficient channel
                    tensor, mapper, diagnostics = self.buffer_to_coefficient_channel_tensor(
                        buffer,
                        target_var,
                        self._current_intervention_scm,
                        **kwargs
                    )
                    
                    # Log state
                    logger.info(f"    📊 Coefficient channel tensor created:")
                    logger.info(f"       Channel type: COEFFICIENTS (not probabilities)")
                    logger.info(f"       True parents: {diagnostics['true_parents']}")
                    logger.info(f"       Coefficients: {diagnostics['coefficients']}")
                    
                    return tensor, mapper, diagnostics
                else:
                    # Fall back to original
                    return self._original_buffer_to_four_channel(buffer, target_var, **kwargs)
            
            # Replace the import in the module
            import src.causal_bayes_opt.training.unified_grpo_trainer as trainer_module
            trainer_module.buffer_to_four_channel_tensor = ground_truth_wrapper
    
    def _create_coefficient_channel_function(self):
        """Create the coefficient channel tensor function."""
        def buffer_to_coefficient_channel_tensor(buffer, target_var, scm, **kwargs):
                """Create tensor with coefficient values in fourth channel."""
                from src.causal_bayes_opt.data_structures.scm import get_parents, get_mechanisms
                from src.causal_bayes_opt.data_structures.sample import get_values, get_intervention_targets, is_observational
                from src.causal_bayes_opt.utils.variable_mapping import VariableMapper
                
                # Get all samples from buffer
                all_samples = buffer.get_all_samples()
                if not all_samples:
                    raise ValueError("Buffer is empty")
                
                # Get variable order and create mapper
                variable_order = sorted(buffer.get_variable_coverage())
                if target_var not in variable_order:
                    raise ValueError(f"Target '{target_var}' not in buffer variables: {variable_order}")
                
                mapper = VariableMapper(variable_order, target_var)
                n_vars = len(variable_order)
                target_idx = mapper.get_index(target_var)
                
                # Get ground truth parents and coefficients
                true_parents = set(get_parents(scm, target_var))
                
                # Extract coefficients from target mechanism
                mechanisms = get_mechanisms(scm)
                target_mechanism = mechanisms.get(target_var)
                coefficients = {}
                if target_mechanism and hasattr(target_mechanism, 'coefficients'):
                    for parent, coeff in target_mechanism.coefficients.items():
                        coefficients[parent] = coeff
                
                # Create coefficient array for all variables
                coefficient_values = np.zeros(n_vars)
                for i, var in enumerate(variable_order):
                    if var == target_var:
                        # Target itself gets 0
                        coefficient_values[i] = 0.0
                    elif var in coefficients:
                        # Parents get their actual coefficient
                        coefficient_values[i] = coefficients[var]
                    else:
                        # Non-parents get 0
                        coefficient_values[i] = 0.0
                
                coefficient_values = jnp.array(coefficient_values)
                
                # Log the coefficient values
                logger.info(f"    🔢 COEFFICIENT CHANNEL for {target_var}:")
                logger.info(f"       True parents: {true_parents}")
                logger.info(f"       Coefficients: {coefficients}")
                for var, coeff_val in zip(variable_order, coefficient_values):
                    if coeff_val != 0:
                        logger.info(f"         {var} → {target_var}: coefficient = {coeff_val:.3f}")
                
                # Determine actual history size
                max_history_size = kwargs.get('max_history_size', 100)
                actual_size = min(len(all_samples), max_history_size)
                recent_samples = all_samples[-actual_size:]
                
                # Initialize 4-channel tensor
                tensor = jnp.zeros((max_history_size, n_vars, 4))
                
                # Fill tensor with recent samples
                for t, sample in enumerate(recent_samples):
                    tensor_idx = max_history_size - actual_size + t
                    
                    # Channel 0: Values
                    sample_values = get_values(sample)
                    values = jnp.array([float(sample_values.get(var, 0.0)) for var in variable_order])
                    
                    # Channel 1: Target indicator
                    target_mask = jnp.array([1.0 if var == target_var else 0.0 for var in variable_order])
                    
                    # Channel 2: Intervention indicator
                    intervention_targets = get_intervention_targets(sample)
                    intervention_mask = jnp.array([1.0 if var in intervention_targets else 0.0 for var in variable_order])
                    
                    # Channel 3: Coefficient values (constant for all samples)
                    # This is the KEY CHANGE - using actual coefficients instead of probabilities
                    
                    # Set tensor values
                    tensor = tensor.at[tensor_idx, :, 0].set(values)
                    tensor = tensor.at[tensor_idx, :, 1].set(target_mask)
                    tensor = tensor.at[tensor_idx, :, 2].set(intervention_mask)
                    tensor = tensor.at[tensor_idx, :, 3].set(coefficient_values)  # COEFFICIENTS!
                
                # Standardize values if requested (per-variable)
                if kwargs.get('standardize', True) and actual_size > 1:
                    # Standardize channel 0 (values) per-variable
                    start_idx = max_history_size - actual_size
                    for var_idx in range(n_vars):
                        var_actual_data = tensor[start_idx:, var_idx, 0]
                        if len(var_actual_data) > 1:
                            var_mean = jnp.mean(var_actual_data)
                            var_std = jnp.std(var_actual_data) + 1e-8
                            standardized_var = (tensor[:, var_idx, 0] - var_mean) / var_std
                            mask = jnp.zeros(max_history_size)
                            mask = mask.at[start_idx:].set(1.0)
                            new_var_values = tensor[:, var_idx, 0] * (1 - mask) + standardized_var * mask
                            tensor = tensor.at[:, var_idx, 0].set(new_var_values)
                
                # Count obs and int samples for diagnostics
                n_obs = sum(1 for s in all_samples if is_observational(s))
                n_int = len(all_samples) - n_obs
                
                # Diagnostics
                diagnostics = {
                    'n_variables': n_vars,
                    'n_samples': len(all_samples),
                    'n_obs': n_obs,
                    'n_int': n_int,
                    'actual_history_size': actual_size,
                    'target_variable': target_var,
                    'target_idx': target_idx,
                    'true_parents': list(true_parents),
                    'coefficients': coefficients,
                    'coefficient_channel': 'ACTIVE'  # Different from probability channel
                }
                
                return tensor, mapper, diagnostics
        
        return buffer_to_coefficient_channel_tensor
    
    def setup_scm_pool(self, pool_size: int = 50):
        """Pre-generate diverse SCM pool with persistent buffers for per-batch rotation."""
        if not self.scm_factory or not hasattr(self, 'sampling_config'):
            logger.warning("SCM factory or sampling config not available, skipping pool setup")
            return
        
        logger.info(f"🏗️  Pre-generating SCM pool of size {pool_size} with persistent buffers...")
        
        for i in range(pool_size):
            try:
                scm_name, scm = self.scm_factory.get_random_scm(**self.sampling_config)
                target_var = get_target(scm)
                variables = list(get_variables(scm))
                
                # Create persistent buffer for this SCM
                buffer = ExperienceBuffer()
                
                # Pre-populate with random number of observational samples
                min_obs = getattr(self, 'min_obs', 10) if hasattr(self, 'min_obs') else 10
                max_obs = getattr(self, 'max_obs', 150) if hasattr(self, 'max_obs') else 150
                n_obs_samples = np.random.randint(min_obs, max_obs + 1)
                obs_samples = sample_from_linear_scm(scm, n_obs_samples, seed=42 + i * 1000)
                for sample in obs_samples:
                    buffer.add_observation(sample)
                logger.info(f"  Pre-populated with {n_obs_samples} observations")
                
                # Pre-populate with random number of interventions
                min_int = getattr(self, 'min_int', 5) if hasattr(self, 'min_int') else 5
                max_int = getattr(self, 'max_int', 20) if hasattr(self, 'max_int') else 20
                n_interventions = np.random.randint(min_int, max_int + 1)
                logger.info(f"  Pre-populating SCM {i} with {n_interventions} interventions")
                
                # Get valid intervention variables (exclude target)
                valid_vars = [v for v in variables if v != target_var]
                
                for int_idx in range(n_interventions):
                        # Random intervention variable
                        int_var = np.random.choice(valid_vars)
                        int_value = np.random.normal(0, 2.0)
                        
                        # Create and apply intervention
                        intervention = create_perfect_intervention(
                            targets=frozenset([int_var]),
                            values={int_var: int_value}
                        )
                        
                        # Sample post-intervention
                        post_data = sample_with_intervention(scm, intervention, 1, seed=42000 + i * 100 + int_idx)
                        if post_data:
                            buffer.add_intervention(intervention, post_data[0])
                
                # Check buffer statistics after pre-population
                stats = buffer.get_statistics()
                logger.info(f"  Buffer stats after pre-pop: {stats.num_observations} obs, {stats.num_interventions} int")
                
                # Print verification that SCM is unique
                try:
                    scm_info = get_scm_info(scm)
                    coeffs = scm_info.get('coefficients', {})
                    logger.debug(f"  SCM #{i}: Target: {target_var}, Parents: {list(get_parents(scm, target_var))}, Coefficients: {coeffs}, Buffer: {stats.num_observations} obs, {stats.num_interventions} int")
                except:
                    pass
                
                # Store SCM with its persistent buffer
                self.scm_pool.append((scm_name, scm, buffer))
                
                # Store metadata
                metadata = {
                    'scm_name': scm_name,
                    'target': target_var,
                    'num_vars': len(variables),
                    'true_parents': list(get_parents(scm, target_var)),
                    'usage_count': 0,
                    'buffer_size': buffer.size(),
                    'n_obs': n_obs_samples,
                    'n_int': self.initial_interventions_per_scm
                }
                self.scm_pool_metadata.append(metadata)
                
            except Exception as e:
                logger.warning(f"Failed to generate SCM {i}: {e}")
        
        logger.info(f"✅ Generated {len(self.scm_pool)} diverse SCMs with persistent buffers")
        logger.info(f"   Each buffer pre-populated with {self.min_obs}-{self.max_obs} random observations")
        logger.info(f"   Each buffer pre-populated with {self.min_int}-{self.max_int} random interventions")
    
    def _get_next_scm_from_pool(self):
        """Get next SCM with its persistent buffer from pool with round-robin rotation."""
        if not self.scm_pool:
            return None, None, None
        
        old_index = self.current_pool_index
        scm_name, scm, buffer = self.scm_pool[self.current_pool_index]
        metadata = self.scm_pool_metadata[self.current_pool_index]
        
        # Track usage
        metadata['usage_count'] += 1
        metadata['buffer_size'] = buffer.size()
        self.batch_rotation_count += 1
        
        # Move to next SCM
        self.current_pool_index = (self.current_pool_index + 1) % len(self.scm_pool)
        
        # Log rotation message
        logger.info(f"\nROTATING SCM: #{old_index} → #{self.current_pool_index}")
        try:
            scm_info = get_scm_info(scm)
            logger.debug(f"  New SCM coefficients: {scm_info.get('coefficients', {})}")
        except:
            pass
        logger.debug(f"  Buffer from pool: {buffer.num_observations()} obs, {buffer.num_interventions()} int")
        logger.debug(f"  Total rotations so far: {self.batch_rotation_count}")
        
        return scm_name, scm, buffer
    
    def _get_random_scm_from_pool(self):
        """Randomly select an SCM and its buffer from the pool."""
        if not self.scm_pool:
            return None, None, None, None
        
        # Random selection instead of round-robin
        idx = np.random.randint(0, len(self.scm_pool))
        scm_name, scm, buffer = self.scm_pool[idx]
        metadata = self.scm_pool_metadata[idx]
        
        # Track usage
        metadata['usage_count'] += 1
        metadata['buffer_size'] = buffer.size()
        
        return idx, scm_name, scm, buffer
    
    # REMOVED: Convergence detection methods (_check_convergence and _reset_convergence_tracking)
    # These are incompatible with per-batch rotation since each intervention uses a different SCM
    # with potentially different variables and structure. Convergence can't be meaningfully detected
    # when the underlying causal graph changes after every intervention.
    
    def _prepare_scms(self, scms):
        """Override to return dummy SCMs when using pool to prevent parent consuming generator."""
        if self.scm_pool and self.rotate_after_batch:
            # Return dummy list - we'll use pool directly in episode runner
            logger.info("Using SCM pool - returning dummy list for parent class")
            # Return first SCM from pool repeatedly (won't actually be used)
            dummy_list = []
            if self.scm_pool:
                scm_name, scm, _ = self.scm_pool[0]
                for i in range(min(self.config.get('max_episodes', 10), 10)):
                    dummy_list.append((scm_name, scm))
            return dummy_list
        else:
            # Use parent's normal preparation
            return super()._prepare_scms(scms)
    
    def train(self, scms):
        """
        Override train method for per-intervention random SCM sampling.
        Uses SCM pool if available, otherwise falls back to original behavior.
        """
        logger.info("🎯 EnhancedGRPOTrainer.train() called - per-intervention random sampling active!")
        
        # If we have a pool, use it; otherwise use original generator approach
        if self.scm_pool:
            logger.info(f"Using pre-generated SCM pool with {len(self.scm_pool)} SCMs")
            
            # Create generator that uses our pool but only returns SCM name and SCM
            def scm_pool_generator():
                if self.scm_pool:
                    # Just return first SCM - actual selection happens per-intervention
                    scm_name, scm, _ = self.scm_pool[0]
                    return scm_name, scm
                else:
                    # Fallback to original generator
                    return scms()
            
            # Train with per-intervention random sampling
            result = super().train(scm_pool_generator)
            return result
        else:
            # Fallback to original behavior
            logger.info("Using original behavior without SCM pool")
            result = super().train(scms)
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
        
        # Reset episode counter
        self.episodes_on_current_scm = 0
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
        self.completed_episodes_count += 1  # Simple episode counter
        
        # Episode-based checkpoint
        if episode_idx in self.checkpoint_episodes:
            logger.info(f"📌 Episode checkpoint: Saving at episode {episode_idx}")
            self._save_checkpoint(episode_idx)
        
        # Wall-clock based checkpointing
        current_time = time.time()
        elapsed_minutes = (current_time - self.last_checkpoint_time) / 60
        if elapsed_minutes >= self.checkpoint_interval_minutes:
            logger.info(f"⏰ Wall-clock checkpoint: {elapsed_minutes:.1f} minutes elapsed")
            self._save_checkpoint(f"time_{int(elapsed_minutes)}min")
            self.last_checkpoint_time = current_time
        
        # Intervention-based checkpointing
        if self.total_interventions_count >= self.interventions_per_checkpoint:
            logger.info(f"🔄 Intervention checkpoint: {self.total_interventions_count} interventions completed")
            self._save_checkpoint(f"int_{self.total_interventions_count}")
            self.interventions_per_checkpoint += 100  # Next checkpoint at +100 interventions
        
        # Log summary every 10 episodes
        if episode_idx % 10 == 0:
            recent_metrics = self.episode_performances[-10:]
            avg_interventions = np.mean([m['num_interventions'] for m in recent_metrics]) if recent_metrics else 0
            convergence_rate = np.mean([m['convergence_triggered'] for m in recent_metrics]) if recent_metrics else 0
            
            total_elapsed = (current_time - self.training_start_time) / 60
            logger.info(f"\n📊 Episode {episode_idx} summary:")
            logger.info(f"   Episodes completed: {self.completed_episodes_count}")
            logger.info(f"   Total interventions: {self.total_interventions_count}")
            logger.info(f"   Time elapsed: {total_elapsed:.1f} minutes")
            if recent_metrics:
                logger.info(f"   Last 10 episodes - Avg interventions: {avg_interventions:.1f}")
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
            },
            optimizer_state=self.optimizer_state  # Save optimizer state for continual learning
        )
        
        logger.info(f"  💾 Saved checkpoint at episode {episode_idx}")
         
    def _run_policy_episode_with_interventions(self, episode_idx: int, scm: Any, scm_name: str, key) -> Dict[str, Any]:
        """
        Override to use random SCM sampling for each intervention.
        
        Key changes from parent:
        - Each intervention randomly samples an SCM from the pool
        - Uses that SCM's persistent buffer (which grows over time)
        - No episode-level buffer initialization
        """
        from src.causal_bayes_opt.data_structures.scm import get_variables, get_target, get_parents
        from src.causal_bayes_opt.data_structures.sample import get_values
        import time
        
        # Print episode header
        logger.debug(f"EPISODE {episode_idx} - RANDOM SCM SAMPLING")
        
        if self.scm_pool:
            logger.debug(f"  Will randomly sample from pool of {len(self.scm_pool)} SCMs with persistent buffers")
        else:
            # Fallback if no pool - shouldn't happen in our use case
            logger.error(f"  ERROR: No SCM pool available!")
            return {}
        
        # Track metrics across all interventions
        intervention_metrics = []
        all_rewards = []
        all_target_values = []
        scm_usage_counts = {}  # Track which SCMs are used
        
        # Track initial buffer sizes to verify persistence
        initial_buffer_sizes = {i: self.scm_pool[i][2].size() for i in range(len(self.scm_pool))}
        
        # INTERVENTION LOOP with random SCM sampling
        for intervention_idx in range(self.max_interventions):
            intervention_start = time.time()
            
            # CRITICAL: Randomly sample SCM and its buffer for THIS intervention
            scm_idx, scm_name, scm, buffer = self._get_random_scm_from_pool()
            
            # Get SCM info for the randomly selected SCM
            variables = list(get_variables(scm))
            target_var = get_target(scm)
            true_parents = list(get_parents(scm, target_var))
            
            # Print intervention header with SCM info
            logger.debug(f"INTERVENTION {intervention_idx+1}/{self.max_interventions} - SCM #{scm_idx}/{len(self.scm_pool)}, Target: {target_var}, Parents: {true_parents}, Buffer: {buffer.num_observations()} obs/{buffer.num_interventions()} int")
            
            # Track usage
            scm_usage_counts[scm_idx] = scm_usage_counts.get(scm_idx, 0) + 1
            
            # Show coefficients to verify different SCMs
            try:
                from src.causal_bayes_opt.experiments.variable_scm_factory import get_scm_info
                scm_info = get_scm_info(scm)
                if 'coefficients' in scm_info:
                    logger.debug(f"   Coefficients: {scm_info['coefficients']}")
            except:
                pass
            
            # Generate intervention key
            key, intervention_key = random.split(key)
            
            # Run single GRPO intervention with THIS SCM and ITS buffer
            single_result = self._run_single_grpo_intervention(
                buffer, scm, target_var, variables, intervention_key
            )
            
            # CRITICAL: Add the best intervention to this SCM's buffer
            if 'best_intervention' in single_result and single_result['best_intervention']['outcome'] is not None:
                buffer.add_intervention(
                    single_result['best_intervention']['intervention'],
                    single_result['best_intervention']['outcome'],
                    posterior=single_result['best_intervention'].get('posterior')
                )
                
            # Track metrics
            if 'best_intervention' in single_result and single_result['best_intervention']['outcome'] is not None:
                outcome = single_result['best_intervention']['outcome']
                target_value = get_values(outcome).get(target_var, 0.0)
                all_target_values.append(target_value)
                logger.debug(f"   Target value: {target_value:.3f}, Buffer after: {buffer.num_observations()} obs/{buffer.num_interventions()} int")
            
            intervention_metrics.append(single_result)
            all_rewards.extend(single_result.get('candidate_rewards', []))
            
            # Log timing
            intervention_duration = time.time() - intervention_start
            logger.debug(f"   Duration: {intervention_duration:.1f}s")
        
        # Episode summary
        logger.info(f"EPISODE {episode_idx} COMPLETE - Total interventions: {len(intervention_metrics)}, Mean reward: {np.mean(all_rewards) if all_rewards else 0:.3f}")
        
        # Show SCM usage distribution and buffer growth
        logger.debug(f"SCM USAGE & BUFFER GROWTH:")
        final_buffer_sizes = {i: self.scm_pool[i][2].size() for i in range(len(self.scm_pool))}
        
        for scm_idx in sorted(scm_usage_counts.keys()):
            count = scm_usage_counts[scm_idx]
            growth = final_buffer_sizes[scm_idx] - initial_buffer_sizes[scm_idx]
            logger.debug(f"   SCM #{scm_idx}: used {count} times, buffer grew by {growth} samples")
            if growth != count:
                logger.warning(f"      WARNING: Expected growth {count}, actual growth {growth}")
        
        unused = set(range(len(self.scm_pool))) - set(scm_usage_counts.keys())
        if unused:
            logger.debug(f"   Unused SCMs: {sorted(unused)}")
        
        # Target progression
        if all_target_values:
            logger.debug(f"TARGET PROGRESSION - Best: {min(all_target_values):.3f}, Worst: {max(all_target_values):.3f}, Mean: {np.mean(all_target_values):.3f}")
        
        # Track episode performance and save checkpoints
        self.current_episode_data['target_values'] = all_target_values
        self.current_episode_data['num_vars'] = len(variables) if 'variables' in locals() else 0
        self._track_episode_performance(episode_idx)
        
        # Update total interventions count
        self.total_interventions_count += len(intervention_metrics)
        
        # Return metrics for compatibility with parent class
        return {
            'episode': episode_idx,
            'mean_reward': float(np.mean(all_rewards)) if all_rewards else 0.0,
            'n_interventions': len(intervention_metrics),
            'intervention_metrics': intervention_metrics,
            'n_variables': len(variables) if 'variables' in locals() else 0,
            'scm_usage': scm_usage_counts,
            'target_values': all_target_values,
            'best_target': min(all_target_values) if all_target_values else 0.0,
            'target_improvement': (all_target_values[0] - all_target_values[-1]) if len(all_target_values) >= 2 else 0.0,
        }
    
    def _run_single_grpo_intervention(self, buffer, scm, target_var, variables, key):
        """
        Override to use ground truth channel instead of surrogate.
        This is copied from UnifiedGRPOTrainer with modifications for ground truth.
        """
        # Get true parents for the target variable
        true_parents = list(get_parents(scm, target_var))
        
        # Find parent with best coefficient × range product (optimal for minimization)
        optimal_var = None
        optimal_coefficient = 0.0
        optimal_score = 0.0
        optimal_range = (-10, 10)
        
        # Get coefficients from target's mechanism
        mechanisms = get_mechanisms(scm)
        target_mechanism = mechanisms.get(target_var)
        coefficients = {}
        if target_mechanism and hasattr(target_mechanism, 'coefficients'):
            for parent, coeff in target_mechanism.coefficients.items():
                coefficients[(parent, target_var)] = coeff
        
        # Get variable ranges
        variable_ranges = scm.get('variable_ranges', {})
        if not variable_ranges:
            metadata = scm.get('metadata', {})
            variable_ranges = metadata.get('variable_ranges', {})
        
        for parent in true_parents:
            coeff = coefficients.get((parent, target_var), 0.0)
            parent_range = variable_ranges.get(parent, (-10, 10))
            range_size = parent_range[1] - parent_range[0]
            score = abs(coeff) * range_size
            
            if score > optimal_score:
                optimal_score = score
                optimal_coefficient = coeff
                optimal_var = parent
                optimal_range = parent_range
        
        # With per-intervention rotation, we don't need detailed convergence tracking
        
        # Store old params for change tracking
        old_params = self.policy_params
        
        # Collect GRPO batch of candidates
        grpo_batch_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'old_log_probs': [],
            'target_idx': None,
            'intervention_details': [],
            'raw_target_values': [],
            'raw_info_gains': [],
            'parent_flags': [],
            'selected_vars': []
        }
        
        # Generate batch of candidates
        for step in range(self.grpo_config.group_size):
            key, step_key = random.split(key)
            
            # CRITICAL CHANGE: Use ground truth channel if configured
            if self.use_ground_truth_channel:
                # Store the current SCM so the wrapper can access it
                self._current_intervention_scm = scm
                
                # Use ground truth four-channel tensor through the wrapper
                actual_buffer_size = len(buffer.get_all_samples())
                history_size = min(self.max_history_size, max(actual_buffer_size, self.min_history_size)) if self.adaptive_history else self.max_history_size
                
                # This will call our coefficient channel function directly
                tensor, mapper, diagnostics = self.buffer_to_coefficient_channel_tensor(
                    buffer, target_var, scm,
                    max_history_size=history_size,
                    standardize=True
                )
                
                # Log ground truth state once per batch
                if step == 0:
                    logger.info(f"  🎯 Coefficient channel active:")
                    logger.info(f"     Channel type: COEFFICIENTS")
                    logger.info(f"     N_obs: {diagnostics['n_obs']}, N_int: {diagnostics['n_int']}")
                    logger.info(f"     True parents: {diagnostics['true_parents']}")
                    logger.info(f"     Coefficients: {diagnostics.get('coefficients', {})}")
                    
                    # Debug: Print tensor channel information
                    logger.info("  📊 Tensor channels (most recent 3 samples):")
                    n_samples = min(3, actual_buffer_size)
                    for i in range(-n_samples, 0):
                        # i=-3 is 3rd newest, i=-2 is 2nd newest, i=-1 is newest
                        sample_age = "newest" if i == -1 else f"{abs(i)}th newest"
                        logger.info(f"     Sample ({sample_age}):")
                        # Format each channel's values nicely
                        ch0_vals = [f"{v:.2f}" for v in tensor[i, :, 0]]
                        ch1_vals = [f"{v:.2f}" for v in tensor[i, :, 1]]
                        ch2_vals = [f"{int(v)}" for v in tensor[i, :, 2]]
                        ch3_vals = [f"{v:.3f}" for v in tensor[i, :, 3]]
                        logger.info(f"       Ch0 (Values): [{', '.join(ch0_vals)}]")
                        logger.info(f"       Ch1 (Target): [{', '.join(ch1_vals)}]")
                        logger.info(f"       Ch2 (IntFlag): [{', '.join(ch2_vals)}]")
                        logger.info(f"       Ch3 (COEFFICIENTS): [{', '.join(ch3_vals)}]")
                    
                    # Additional debug: Show which variables have non-zero coefficients
                    logger.info("  🔍 Non-zero coefficients in channel 3:")
                    for var_idx, var_name in enumerate(mapper.variables):
                        coeff_val = tensor[-1, var_idx, 3]  # Look at newest sample
                        if coeff_val != 0:
                            logger.info(f"       {var_name}: {coeff_val:.3f}")
            else:
                # Original surrogate-based tensor conversion
                surrogate_fn = self.surrogate_predict_fn if (self.use_surrogate and hasattr(self, 'surrogate_predict_fn')) else None
                actual_buffer_size = len(buffer.get_all_samples())
                history_size = min(self.max_history_size, max(actual_buffer_size, self.min_history_size)) if self.adaptive_history else self.max_history_size
                
                tensor, mapper, diagnostics = buffer_to_four_channel_tensor(
                    buffer, target_var, surrogate_fn=surrogate_fn, max_history_size=history_size, standardize=True
                )
            
            # Get policy output
            key, policy_key = random.split(key)
            policy_output = self.policy_fn.apply(
                self.policy_params, policy_key, tensor, mapper.target_idx
            )
            
            # Sample intervention - support both architectures
            key, selection_key = random.split(key)
            
            if 'quantile_scores' in policy_output:
                # QUANTILE ARCHITECTURE: Unified variable+value selection
                from src.causal_bayes_opt.training.quantile_selection import select_quantile_intervention
                
                selected_var, intervention_value, log_prob, debug_info = select_quantile_intervention(
                    policy_output, buffer, scm, mapper.variables, target_var, selection_key, fixed_std=1.0
                )
                selected_var_idx = mapper.variables.index(selected_var)
                
                # Log quantile selection details
                if step == 0:
                    from src.causal_bayes_opt.training.quantile_selection import log_quantile_details
                    log_quantile_details(policy_output['quantile_scores'], debug_info, mapper.variables, target_var, scm)
                
            else:
                # TRADITIONAL ARCHITECTURE: Separate variable and value heads
                var_logits = policy_output['variable_logits']
                value_params = policy_output['value_params']
                
                key, var_key = random.split(key)
                selected_var_idx = random.categorical(var_key, var_logits)
                var_probs = jax.nn.softmax(var_logits)
                
                # Sample intervention value
                key, val_key = random.split(key)
                mean = value_params[selected_var_idx, 0]
                log_std = value_params[selected_var_idx, 1]
                std = jnp.exp(log_std)
                intervention_value = mean + std * random.normal(val_key)
                
                # Compute log probability
                log_prob = float(jnp.log(var_probs[selected_var_idx] + 1e-8))
                debug_info = {}
            
            # Create and apply intervention
            selected_var = mapper.variables[selected_var_idx] if 'quantile_scores' in policy_output else mapper.get_name(int(selected_var_idx))
            intervention = create_perfect_intervention(
                targets=frozenset([selected_var]),
                values={selected_var: float(intervention_value)}
            )
            
            key, sample_key = random.split(key)
            intervention_samples = sample_with_intervention(
                scm, intervention, n_samples=1, seed=int(sample_key[0])
            )
            
            # Collect raw values for group-based rewards
            outcome_sample = intervention_samples[0] if intervention_samples else None
            if outcome_sample:
                from src.causal_bayes_opt.data_structures.sample import get_values
                target_value = float(get_values(outcome_sample).get(target_var, 0.0))
                grpo_batch_data['raw_target_values'].append(target_value)
                
                # For ground truth channel, we don't compute info gain
                info_gain = 0.0
                grpo_batch_data['raw_info_gains'].append(info_gain)
                
                is_parent = selected_var in true_parents
                grpo_batch_data['parent_flags'].append(is_parent)
                grpo_batch_data['selected_vars'].append(selected_var)
            else:
                grpo_batch_data['raw_target_values'].append(0.0)
                grpo_batch_data['raw_info_gains'].append(0.0)
                grpo_batch_data['parent_flags'].append(False)
                grpo_batch_data['selected_vars'].append(selected_var)
            
            # Store for GRPO batch
            grpo_batch_data['states'].append(tensor)
            
            action_data = {
                'variable': selected_var_idx,
                'value': float(intervention_value)
            }
            
            if 'quantile_scores' in policy_output:
                action_data['quantile_idx'] = debug_info.get('selected_quantile_idx', 0)
                action_data['flat_quantile_idx'] = debug_info.get('winner_idx', 0)
            
            grpo_batch_data['actions'].append(action_data)
            grpo_batch_data['old_log_probs'].append(float(log_prob))
            
            # Store intervention details
            intervention_detail = {
                'intervention': intervention,
                'samples': intervention_samples,
                'posterior': None
            }
            
            # Track convergence
            is_optimal = (selected_var == optimal_var) if optimal_var else False
            selection_prob = float(jnp.exp(log_prob))
            
            if step == 0:
                # With per-intervention rotation, we don't track convergence metrics
                if optimal_var:
                    logger.info(f"  Selection: {selected_var}, "
                              f"Optimal={optimal_var}, "
                              f"IsOptimal={is_optimal}")
                else:
                    logger.info(f"  Selection: {selected_var}, No optimal parent found")
            
            if 'quantile_scores' in policy_output and debug_info:
                intervention_detail['debug_info'] = {
                    'selected_var_idx': selected_var_idx,
                    'selected_quantile': debug_info.get('selected_quantile_idx'),
                    'selection_probability': selection_prob,
                    'is_optimal': is_optimal,
                    'optimal_var': optimal_var
                }
            
            grpo_batch_data['intervention_details'].append(intervention_detail)
            
            if grpo_batch_data['target_idx'] is None:
                grpo_batch_data['target_idx'] = mapper.target_idx
        
        # SECOND PASS: Compute group-based binary rewards
        group_target_mean = np.mean(grpo_batch_data['raw_target_values'])
        group_info_mean = np.mean(grpo_batch_data['raw_info_gains'])
        
        # Get reward weights
        target_weight = self.config.get('reward_weights', {}).get('target', 0.7)
        info_weight = self.config.get('reward_weights', {}).get('info_gain', 0.2) if not self.use_ground_truth_channel else 0.0
        parent_weight = self.config.get('reward_weights', {}).get('parent', 0.1)
        
        # When using ground truth, redistribute info weight to target and parent
        if self.use_ground_truth_channel:
            target_weight = self.config.get('reward_weights', {}).get('target', 0.8)
            parent_weight = self.config.get('reward_weights', {}).get('parent', 0.2)
        
        # Compute binary rewards
        rewards = []
        for i in range(len(grpo_batch_data['raw_target_values'])):
            if self.optimization_direction == "MINIMIZE":
                target_binary = 1.0 if grpo_batch_data['raw_target_values'][i] < group_target_mean else 0.0
            else:
                target_binary = 1.0 if grpo_batch_data['raw_target_values'][i] > group_target_mean else 0.0
            
            info_binary = 1.0 if grpo_batch_data['raw_info_gains'][i] > group_info_mean else 0.0
            parent_binary = 1.0 if grpo_batch_data['parent_flags'][i] else 0.0
            
            total_reward = (target_weight * target_binary + 
                          info_weight * info_binary + 
                          parent_weight * parent_binary)
            
            rewards.append(total_reward)
            
            if i < 5:
                logger.info(f"[REWARD #{i}] Var={grpo_batch_data['selected_vars'][i]}: "
                           f"Target={target_binary:.0f}({grpo_batch_data['raw_target_values'][i]:.3f}) "
                           f"Parent={parent_binary:.0f} "
                           f"→ Total={total_reward:.3f}")
        
        logger.info(f"[GROUP STATS] Size={len(rewards)}, "
                   f"Target mean={group_target_mean:.3f}, "
                   f"Reward range=[{min(rewards):.3f}, {max(rewards):.3f}]")
        
        # Convert to arrays
        grpo_batch_data['rewards'] = jnp.array(rewards)
        grpo_batch_data['old_log_probs'] = jnp.array(grpo_batch_data['old_log_probs'])
        
        # Create GRPO batch
        grpo_batch = self._create_grpo_batch(grpo_batch_data)
        
        # Perform GRPO update
        self.policy_params, self.optimizer_state, grpo_metrics = self._grpo_update(
            self.policy_params, self.optimizer_state, grpo_batch
        )
        
        # Compute parameter change
        param_changes = jax.tree.map(lambda old, new: jnp.linalg.norm(new - old), old_params, self.policy_params)
        total_change = sum(jax.tree.leaves(param_changes))
        
        # Select best intervention
        best_idx = jnp.argmax(grpo_batch_data['rewards'])
        selected_idx = int(best_idx)
        
        best_intervention_info = grpo_batch_data['intervention_details'][selected_idx]
        best_reward = float(grpo_batch_data['rewards'][best_idx])
        selected_reward = float(grpo_batch_data['rewards'][selected_idx])
        
        return {
            'best_intervention': {
                'intervention': best_intervention_info['intervention'],
                'outcome': best_intervention_info['samples'][0] if best_intervention_info['samples'] else None,
                'posterior': best_intervention_info.get('posterior'),
                'debug_info': best_intervention_info.get('debug_info')
            },
            'candidate_rewards': [float(r) for r in grpo_batch_data['rewards']],
            'grpo_metrics': grpo_metrics,
            'param_change': float(total_change),
            'selection_info': {
                'selected_idx': selected_idx,
                'selected_reward': selected_reward,
                'best_idx': int(best_idx),
                'best_reward': best_reward,
                'selection_advantage': selected_reward - best_reward
            }
        }
    
    def _run_grpo_episode(self, episode_idx, scm, scm_name, key):
        """Override to use per-intervention random SCM sampling."""
        # Use our custom implementation with per-intervention random sampling
        return self._run_policy_episode_with_interventions(episode_idx, scm, scm_name, key)


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
        # Core episode settings - match working config
        'max_episodes': max_episodes,
        'obs_per_episode': 10,      # Match working complex script
        'max_interventions': 15,    # Match working complex script
        
        # Phase management (for JointACBOTrainer compatibility)
        'episodes_per_phase': 999999,  # Effectively disable phase switching
        'initial_phase': 'policy',
        
        # Architecture
        'policy_architecture': 'quantile',
        
        # Model architecture configuration - for deeper/larger models
        'architecture': {
            'hidden_dim': 512,     # Increased from default 256 for more capacity
            'num_layers': 8,       # Increased from default 4 for deeper model
            'num_heads': 16,       # Increased from default 8 for more attention heads
            'dropout': 0.15,       # Increased from default 0.1 for better regularization
        },
        
        # Surrogate integration
        'use_surrogate': True,
        'use_grpo_rewards': True,
        
        # Exploration settings
        'use_fixed_std': True,
        'fixed_std': 1.0,
        
        # Learning settings - use proven stable rate
        'learning_rate': 5e-4,
        
        # GRPO configuration
        # PERFORMANCE NOTE: Each intervention takes ~1.5 seconds per candidate due to:
        # - Surrogate forward pass on entire buffer (grows with interventions)
        # - SCM sampling and reward computation
        # Total time per intervention ≈ group_size * 1.5 seconds
        'grpo_config': {
            'group_size': 16,  # Reduced from 32 for 2x faster iterations (~24s vs ~48s per intervention)
            'entropy_coefficient': 0.001,
            'clip_ratio': 1.0,
            'gradient_clip': 10.0,
            'ppo_epochs': 4,
            'normalize_advantages': True
        },
        
        # Reward weights (with surrogate info gain) - use working balanced structure
        'reward_weights': {
            'target': 0.8,     # Target optimization (primary signal)
            'parent': 0.2,     # Parent selection bonus (causal guidance)
            'info_gain': 0.0   # No info gain by default
        },
        
        # Binary reward for clear 0/+1 ranking signals  
        'reward_type': 'binary',
        
        # Use probability change info gain by default
        'info_gain_type': 'probability_change',
        
        # Buffer configuration for adaptive history sizing
        'buffer_config': {
            'max_history_size': 30,  # Reduced from 100 to minimize padding
            'adaptive_history': True,
            'min_history_size': 10
        },
        
        # NEW: Per-batch SCM rotation configuration  
        'rotate_after_episode': False,        # Changed: Don't rotate after episode
        'rotate_after_batch': True,           # NEW: Rotate after each batch
        'convergence_patience': 3,            # Consecutive same interventions for early rotation
        'convergence_threshold': 0.9,         # Probability threshold for convergence  
        'enable_early_rotation': False,       # Disabled for per-batch approach
        
        # Keep nested structure for backward compatibility
        'scm_rotation': {
            'rotate_every_episode': False,     # Changed: per-batch instead
            'convergence_patience': 3,         # Consecutive same interventions for early rotation
            'convergence_threshold': 0.9,      # Probability threshold for convergence
            'enable_early_rotation': False,    # Disabled for per-batch approach
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
    logger.info("🚀 SCRIPT VERSION: Coefficient Channel - Direct Causal Strength")
    logger.info("📍 Script location: train_grpo_ground_truth_coefficients.py")
    
    parser = argparse.ArgumentParser(description="Enhanced Multi-SCM GRPO training")
    parser.add_argument('--episodes', type=int, default=200,
                        help='Number of training episodes')
    parser.add_argument('--pool-size', type=int, default=50,
                        help='Size of pre-generated SCM pool for per-batch rotation')
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
    parser.add_argument('--target-weight', type=float, default=0.8,
                        help='Target improvement reward weight')
    parser.add_argument('--parent-weight', type=float, default=0.2,
                        help='Parent accuracy reward weight')
    parser.add_argument('--info-weight', type=float, default=0.0,
                        help='Information gain reward weight')
    parser.add_argument('--max-interventions', type=int, default=None,
                        help='Max interventions per episode')
    parser.add_argument('--obs-per-episode', type=int, default=None,
                        help='Observations per episode')
    parser.add_argument('--structure-types', type=str, nargs='+',
                        default=['chain'],
                        choices=['random', 'chain', 'fork', 'true_fork', 'collider', 'mixed', 'scale_free', 'two_layer'],
                        help='SCM structure types to train on')
    parser.add_argument('--min-vars', type=int, default=3,
                        help='Minimum number of variables in SCM')
    parser.add_argument('--max-vars', type=int, default=99,
                        help='Maximum number of variables in SCM')
    parser.add_argument('--reward-type', type=str, default='binary',
                        choices=['composite', 'binary', 'clean', 'better_clean'],
                        help='Type of reward computation (binary uses 0/+1 group ranking)')
    parser.add_argument('--info-gain-type', type=str, default='probability_change',
                        choices=['entropy_reduction', 'probability_change'],
                        help='Info gain computation: probability_change (sum absolute changes) or entropy_reduction')
    parser.add_argument('--use-ground-truth', action='store_true',
                        help='Use ground truth channel instead of surrogate')
    parser.add_argument('--convergence-rate', type=float, default=0.2,
                        help='Convergence rate factor for progressive certainty')
    parser.add_argument('--initial-interventions', type=int, default=0,
                        help='Number of initial interventions to pre-populate per SCM (deprecated, use min/max-int)')
    parser.add_argument('--min-obs', type=int, default=10,
                        help='Minimum observations for pre-population')
    parser.add_argument('--max-obs', type=int, default=150,
                        help='Maximum observations for pre-population')
    parser.add_argument('--min-int', type=int, default=5,
                        help='Minimum interventions for pre-population')
    parser.add_argument('--max-int', type=int, default=20,
                        help='Maximum interventions for pre-population')
    parser.add_argument('--save-freq', type=int, default=50,
                        help='Save checkpoint every N episodes')
    parser.add_argument('--log-freq', type=int, default=10,
                        help='Log metrics every N episodes')
    
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
    
    # Add pool size to config
    config['scm_pool_size'] = args.pool_size
    
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
    
    # Override reward type if provided
    if args.reward_type:
        config['reward_type'] = args.reward_type
    
    # Override info gain type if provided
    if args.info_gain_type:
        config['info_gain_type'] = args.info_gain_type
    
    # Configure ground truth channel if requested
    if args.use_ground_truth:
        config['use_ground_truth_channel'] = True
        config['convergence_rate_factor'] = args.convergence_rate
        config['initial_interventions_per_scm'] = args.initial_interventions
        config['use_surrogate'] = False  # Disable surrogate when using ground truth
        # Store pre-population ranges
        config['min_obs'] = args.min_obs
        config['max_obs'] = args.max_obs
        config['min_int'] = args.min_int
        config['max_int'] = args.max_int
        config['checkpoint_frequency'] = args.save_freq
        logger.info(f"🎯 Using COEFFICIENT channel with:")
        logger.info(f"   Channel 3: Actual coefficient values (not probabilities)")
        logger.info(f"   Pre-population: {args.min_obs}-{args.max_obs} obs, {args.min_int}-{args.max_int} int")
        logger.info(f"   Checkpoint every {args.save_freq} episodes")
    
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
    
    # Define sampling configuration with respect to min/max vars
    sampling_config = {
        'variable_counts': list(range(args.min_vars, args.max_vars + 1)),  # Use specified range
        'structure_types': args.structure_types,  # Use command-line argument
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
        
        
        # Load optimizer state if available for continual learning
        if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
            trainer.optimizer_state = checkpoint['optimizer_state']
            logger.info(f"  Loaded optimizer state for continual learning")
        else:
            logger.info(f"  No optimizer state found in checkpoint, will reinitialize")
        
        # Set starting episode for continuation (simple checkpoint-based approach)
        trainer.start_episode = 1  # Resume from episode 1 (any non-zero value)
        logger.info(f"  Policy checkpoint exists - will continue training from episode 1")
        logger.info(f"  (Using checkpoint-based resume, not episode counting)")
    else:
        trainer.start_episode = 0  # Start from scratch
    
    # Set SCM factory and config for rotation
    trainer.scm_factory = scm_factory
    trainer.sampling_config = sampling_config
    
    # NEW: Setup SCM pool for per-batch rotation
    if trainer.rotate_after_batch:
        pool_size = config.get('scm_pool_size', 50)
        trainer.setup_scm_pool(pool_size)
    
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
                'completed_episodes_count': trainer.completed_episodes_count,
                'total_rotations': trainer.total_rotations,
                'early_rotations': trainer.early_rotations,
                'convergence_events': len(trainer.convergence_metrics['convergence_events']),
                'scm_diversity': len(trainer.scm_history),
                'trainer_class': 'EnhancedGRPOTrainer'
            },
            optimizer_state=trainer.optimizer_state  # Save optimizer state for continual learning
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
    sys.exit(main())