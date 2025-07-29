#!/usr/bin/env python3
"""
Clean BC (Behavioral Cloning) trainer implementation.

This module provides a BC trainer that:
1. Uses the same 3-channel tensor format as GRPO
2. Trains on expert demonstrations
3. Compatible with the universal evaluator
4. Includes optional surrogate learning
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
import pickle

import jax
import jax.numpy as jnp
import jax.random as random
import optax
import haiku as hk
from omegaconf import DictConfig
import pyrsistent as pyr

from .three_channel_converter import buffer_to_three_channel_tensor
from ..data_structures.buffer import ExperienceBuffer
from ..data_structures.sample import create_sample, get_values
from ..policies.clean_bc_policy_factory import create_clean_bc_policy, create_bc_loss_fn
from ..data_structures.scm import get_variables, get_target, get_parents
from ..mechanisms.linear import sample_from_linear_scm
from ..interventions.handlers import create_perfect_intervention
from ..environments.sampling import sample_with_intervention
from .continuous_surrogate_integration import (
    create_continuous_learnable_surrogate,
    compute_posterior_from_buffer_continuous,
    compute_structure_metrics_continuous
)

logger = logging.getLogger(__name__)


class CleanBCTrainer:
    """
    Clean BC trainer with 3-channel tensor format.
    
    This trainer:
    - Learns from expert demonstrations
    - Uses behavioral cloning loss
    - Maintains compatibility with universal evaluator
    - Optionally updates surrogate model
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize clean BC trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.rng_key = random.PRNGKey(config.get('seed', 42))
        
        # Extract key configurations
        self.max_episodes = config.get('max_episodes', 1000)
        self.n_variables_range = config.get('n_variables_range', [3, 8])
        self.obs_per_episode = config.get('obs_per_episode', 100)
        self.max_interventions = config.get('max_interventions', 10)
        self.batch_size = config.get('batch_size', 32)
        
        # BC-specific config
        self.expert_strategy = config.get('expert_strategy', 'oracle')  # 'oracle' or 'random'
        self.demonstration_episodes = config.get('demonstration_episodes', 100)
        
        # Architecture config
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # Whether to use surrogate learning
        self.use_surrogate = config.get('use_surrogate', True)
        
        # Initialize components
        self._initialize_policy()
        self._initialize_surrogate()
        self._initialize_optimizer()
        
        # Training state
        self.training_metrics = []
        self.expert_buffer = []  # Store expert demonstrations
        self.episode_count = 0
        
    def _initialize_policy(self):
        """Initialize BC policy network using shared factory."""
        # Use shared policy factory
        policy_fn = create_clean_bc_policy(hidden_dim=self.hidden_dim)
        self.policy_fn = hk.transform(policy_fn)
        
        # Create loss function
        self.loss_fn = create_bc_loss_fn(self.policy_fn)
        
        # Initialize with dummy data
        dummy_tensor = jnp.zeros((10, 5, 3))  # [T=10, n_vars=5, channels=3]
        self.rng_key, init_key = random.split(self.rng_key)
        self.policy_params = self.policy_fn.init(init_key, dummy_tensor, 0)
        
    def _initialize_surrogate(self):
        """Initialize surrogate model for structure learning."""
        if not self.config.get('use_surrogate', True):
            self.surrogate_net = None
            self.surrogate_params = None
            self.surrogate_opt_state = None
            self.surrogate_predict_fn = None
            self.surrogate_update_fn = None
            logger.info("Surrogate learning disabled")
            return
            
        # Initialize learnable surrogate
        self.rng_key, surrogate_key = random.split(self.rng_key)
        
        # Determine max variables from config
        max_vars = max(self.n_variables_range) if isinstance(self.n_variables_range, list) else 10
        
        (self.surrogate_net, 
         self.surrogate_params, 
         self.surrogate_opt_state,
         self.surrogate_predict_fn,
         self.surrogate_update_fn) = create_continuous_learnable_surrogate(
            n_variables=max_vars,
            key=surrogate_key,
            learning_rate=self.config.get('surrogate_lr', 1e-3),
            hidden_dim=self.config.get('surrogate_hidden_dim', 128),
            num_layers=self.config.get('surrogate_layers', 4),
            num_heads=self.config.get('surrogate_heads', 8)
        )
        
        logger.info("Initialized learnable surrogate model for BC")
        
    def _initialize_optimizer(self):
        """Initialize optimizer for BC training."""
        learning_rate = self.config.get('learning_rate', 1e-3)
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.policy_params)
        
    def _generate_expert_demonstration(self, scm: pyr.PMap) -> List[Tuple[Any, Dict, float]]:
        """
        Generate expert demonstration for an episode.
        
        Args:
            scm: Structural causal model
            
        Returns:
            List of (buffer_state, intervention, outcome) tuples
        """
        demonstrations = []
        
        # Get SCM info
        variables = list(get_variables(scm))
        target_var = get_target(scm)
        target_idx = variables.index(target_var)
        true_parents = list(get_parents(scm, target_var)) if hasattr(scm, 'edges') else []
        
        # Initialize buffer
        buffer = ExperienceBuffer()
        self.rng_key, obs_key = random.split(self.rng_key)
        
        # Sample observational data
        obs_samples = sample_from_linear_scm(scm, self.obs_per_episode, seed=int(obs_key[0]))
        for sample in obs_samples:
            buffer.add_observation(sample)
        
        # Generate expert interventions
        for step in range(self.max_interventions):
            # Convert buffer to tensor for consistency
            tensor, var_order = buffer_to_three_channel_tensor(
                buffer, target_var, max_history_size=100, standardize=True
            )
            
            # Expert intervention selection
            if self.expert_strategy == 'oracle' and true_parents:
                # Oracle: rotate through true parents for diversity
                parent_idx = step % len(true_parents)
                selected_var = true_parents[parent_idx]
            else:
                # Random expert (baseline)
                candidates = [v for v in variables if v != target_var]
                self.rng_key, choice_key = random.split(self.rng_key)
                var_idx = random.randint(choice_key, (), 0, len(candidates))
                selected_var = candidates[var_idx]
            
            selected_var_idx = variables.index(selected_var)
            
            # Expert value selection (greedy optimization)
            self.rng_key, val_key = random.split(self.rng_key)
            if self.expert_strategy == 'oracle':
                # Oracle uses greedy optimization
                # For MINIMIZE direction, use negative values
                intervention_value = -2.0 + float(random.uniform(val_key, (), minval=-0.2, maxval=0.2))
            else:
                # Random uses wider range
                intervention_value = float(random.uniform(val_key, (), minval=-2.0, maxval=2.0))
            
            # Create intervention
            intervention = create_perfect_intervention(
                targets=frozenset([selected_var]),
                values={selected_var: intervention_value}
            )
            
            # Apply intervention
            self.rng_key, sample_key = random.split(self.rng_key)
            intervention_samples = sample_with_intervention(
                scm, intervention, n_samples=1, seed=int(sample_key[0])
            )
            
            # Store demonstration (buffer state before intervention)
            demonstrations.append({
                'tensor': tensor.copy(),  # Current state
                'target_idx': target_idx,
                'expert_var_idx': selected_var_idx,
                'expert_value': intervention_value,
                'variables': var_order.copy()
            })
            
            # Update buffer
            for sample in intervention_samples:
                buffer.add_intervention(intervention, sample)
        
        return demonstrations
    
    def train(self, scm_generator: Callable[[], pyr.PMap]) -> Dict[str, Any]:
        """
        Train BC policy on expert demonstrations.
        
        Args:
            scm_generator: Function that generates SCMs for training
            
        Returns:
            Training results and metrics
        """
        logger.info("Starting clean BC training")
        start_time = time.time()
        
        # Phase 1: Collect expert demonstrations
        logger.info(f"Phase 1: Collecting {self.demonstration_episodes} expert demonstrations")
        for demo_ep in range(self.demonstration_episodes):
            scm = scm_generator()
            demonstrations = self._generate_expert_demonstration(scm)
            self.expert_buffer.extend(demonstrations)
            
            if demo_ep % 20 == 0:
                logger.info(f"Collected {len(self.expert_buffer)} demonstrations")
        
        # Phase 2: Behavioral cloning training
        logger.info(f"Phase 2: Training on {len(self.expert_buffer)} demonstrations")
        
        # Convert to training batches
        n_demos = len(self.expert_buffer)
        
        for episode in range(self.max_episodes):
            # Sample batch of demonstrations
            self.rng_key, batch_key = random.split(self.rng_key)
            batch_indices = random.choice(
                batch_key, n_demos, shape=(self.batch_size,), replace=True
            )
            
            # Compute batch loss
            batch_loss = 0.0
            batch_metrics = {
                'var_accuracy': 0.0,
                'value_mse': 0.0,
                'structure_f1': 0.0
            }
            
            for idx in batch_indices:
                demo = self.expert_buffer[idx]
                
                # Compute BC loss
                self.rng_key, loss_key = random.split(self.rng_key)
                loss, metrics = self.loss_fn(
                    self.policy_params,
                    loss_key,
                    demo['tensor'],
                    demo['target_idx'],
                    demo['expert_var_idx'],
                    demo['expert_value']
                )
                
                batch_loss += loss
                
                # Track accuracy
                pred_var = jnp.argmax(metrics['predicted_var_probs'])
                if pred_var == demo['expert_var_idx']:
                    batch_metrics['var_accuracy'] += 1.0
                
                # Track value error
                value_error = jnp.abs(metrics['predicted_mean'] - demo['expert_value'])
                batch_metrics['value_mse'] += value_error ** 2
            
            # Average over batch
            batch_loss /= self.batch_size
            batch_metrics['var_accuracy'] /= self.batch_size
            batch_metrics['value_mse'] /= self.batch_size
            
            # Compute gradients and update
            grads = jax.grad(lambda p: batch_loss)(self.policy_params)
            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.policy_params = optax.apply_updates(self.policy_params, updates)
            
            # Store metrics
            episode_metrics = {
                'episode': episode,
                'loss': float(batch_loss),
                'var_accuracy': float(batch_metrics['var_accuracy']),
                'value_rmse': float(jnp.sqrt(batch_metrics['value_mse'])),
                'expert_strategy': self.expert_strategy
            }
            
            self.training_metrics.append(episode_metrics)
            
            # Log progress
            if episode % 100 == 0:
                logger.info(
                    f"Episode {episode}: "
                    f"loss={episode_metrics['loss']:.3f}, "
                    f"var_acc={episode_metrics['var_accuracy']:.3f}, "
                    f"val_rmse={episode_metrics['value_rmse']:.3f}"
                )
        
        # Save final checkpoint
        self._save_checkpoint(is_final=True)
        
        return {
            'training_time': time.time() - start_time,
            'final_metrics': self.training_metrics[-1],
            'all_metrics': self.training_metrics,
            'policy_params': self.policy_params,
            'n_demonstrations': len(self.expert_buffer)
        }
    
    def _save_checkpoint(self, is_final: bool = False):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        name = "clean_bc_final" if is_final else f"clean_bc_ep{self.episode_count}"
        checkpoint_path = checkpoint_dir / name
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save checkpoint data
        checkpoint_data = {
            'policy_params': self.policy_params,
            'config': dict(self.config),
            'episode': self.episode_count,
            'is_final': is_final,
            'three_channel_format': True,
            'training_metrics': self.training_metrics[-10:] if self.training_metrics else [],
            'expert_strategy': self.expert_strategy,
            'n_demonstrations': len(self.expert_buffer)
        }
        
        # Include surrogate parameters if enabled
        if self.use_surrogate and self.surrogate_params is not None:
            checkpoint_data['surrogate_params'] = self.surrogate_params
            checkpoint_data['has_surrogate'] = True
        
        with open(checkpoint_path / 'checkpoint.pkl', 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"Saved BC checkpoint to {checkpoint_path}")


def create_clean_bc_trainer(config: DictConfig) -> CleanBCTrainer:
    """
    Factory function to create clean BC trainer.
    
    Args:
        config: Training configuration
        
    Returns:
        Initialized CleanBCTrainer
    """
    return CleanBCTrainer(config)