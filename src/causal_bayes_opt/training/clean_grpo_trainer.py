"""
Clean GRPO trainer implementation without AcquisitionState dependency.

This module provides a simplified GRPO training loop that:
1. Uses direct buffer-to-tensor conversion (3-channel format)
2. Updates posteriors properly during training
3. Maintains variable-agnostic processing
4. Removes complex state abstractions
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
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
from ..acquisition.clean_rewards import compute_clean_reward
from ..policies.clean_policy_factory import create_clean_grpo_policy
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


class CleanGRPOTrainer:
    """
    Clean GRPO trainer with direct tensor conversion and proper posterior updates.
    
    This trainer:
    - Converts buffers directly to 3-channel tensors
    - Updates structural posteriors during training
    - Maintains true variable-agnostic processing
    - Uses simple, clear interfaces
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize clean GRPO trainer.
        
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
        
        # Architecture config
        arch_config = config.get('architecture', {})
        self.num_layers = arch_config.get('num_layers', 4)
        self.num_heads = arch_config.get('num_heads', 8)
        self.hidden_dim = arch_config.get('hidden_dim', 256)
        self.key_size = arch_config.get('key_size', 32)
        self.dropout = arch_config.get('dropout', 0.1)
        
        # Whether to use surrogate learning
        self.use_surrogate = config.get('use_surrogate', True)
        
        # Initialize components
        self._initialize_policy()
        self._initialize_surrogate()
        self._initialize_optimizer()
        
        # Training state
        self.training_metrics = []
        self.episode_count = 0
        
    def _initialize_policy(self):
        """Initialize policy network using shared factory."""
        # Use shared policy factory to ensure consistent module paths
        policy_fn = create_clean_grpo_policy(hidden_dim=self.hidden_dim)
        self.policy_fn = hk.transform(policy_fn)
        
        # Initialize with dummy data
        dummy_tensor = jnp.zeros((10, 5, 3))  # [T=10, n_vars=5, channels=3]
        self.rng_key, init_key = random.split(self.rng_key)
        self.policy_params = self.policy_fn.init(init_key, dummy_tensor, 0)
        
    def _initialize_surrogate(self):
        """Initialize surrogate model for structure learning."""
        if not self.use_surrogate:
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
        
        logger.info("Initialized learnable surrogate model")
        
    def _initialize_optimizer(self):
        """Initialize optimizer."""
        learning_rate = self.config.get('learning_rate', 3e-4)
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.policy_params)
        
    def train(self, scm_generator: Callable[[], pyr.PMap]) -> Dict[str, Any]:
        """
        Train GRPO with clean tensor conversion.
        
        Args:
            scm_generator: Function that generates SCMs for training
            
        Returns:
            Training results and metrics
        """
        logger.info("Starting clean GRPO training")
        start_time = time.time()
        
        for episode in range(self.max_episodes):
            # Generate new SCM
            self.rng_key, scm_key = random.split(self.rng_key)
            scm = scm_generator()
            
            # Run episode
            episode_metrics = self._run_episode(scm, episode)
            self.training_metrics.append(episode_metrics)
            
            # Log progress
            if episode % 100 == 0:
                log_msg = (
                    f"Episode {episode}: "
                    f"reward={episode_metrics['mean_reward']:.3f}, "
                    f"loss={episode_metrics['loss']:.3f}"
                )
                
                # Add structure metrics if available
                if self.use_surrogate and episode_metrics.get('structure_metrics'):
                    sm = episode_metrics['structure_metrics']
                    if sm.get('f1_score', 0) > 0:
                        log_msg += f", F1={sm['f1_score']:.3f}, SHD={sm['shd']:.1f}"
                
                logger.info(log_msg)
        
        # Save final checkpoint
        self._save_checkpoint(is_final=True)
        
        return {
            'training_time': time.time() - start_time,
            'final_metrics': self.training_metrics[-1],
            'all_metrics': self.training_metrics,
            'policy_params': self.policy_params
        }
    
    def _run_episode(self, scm: pyr.PMap, episode_idx: int) -> Dict[str, float]:
        """Run single training episode with proper tensor conversion."""
        # Get SCM info
        variables = list(get_variables(scm))
        target_var = get_target(scm)
        target_idx = variables.index(target_var)
        
        # Initialize buffer with observational data
        buffer = ExperienceBuffer()
        self.rng_key, obs_key = random.split(self.rng_key)
        
        # Sample observational data
        obs_samples = sample_from_linear_scm(scm, self.obs_per_episode, seed=int(obs_key[0]))
        for sample in obs_samples:
            buffer.add_observation(sample)
        
        # Training loop for this episode
        total_loss = 0.0
        total_reward = 0.0
        structure_metrics = []
        
        # Get true parents for metrics (if available)
        true_parents = list(get_parents(scm, target_var)) if hasattr(scm, 'edges') else []
        
        for step in range(self.max_interventions):
            # Convert buffer to tensor
            tensor, var_order = buffer_to_three_channel_tensor(
                buffer, target_var, max_history_size=100, standardize=True
            )
            
            # Get posterior prediction if surrogate is enabled
            posterior = None
            if self.use_surrogate and self.surrogate_net is not None:
                # Use continuous model - it needs target index and variable list
                posterior = self.surrogate_predict_fn(tensor, target_idx, variables)
                
                # Compute structure metrics if we have true parents
                if true_parents:
                    metrics = compute_structure_metrics_continuous(posterior, true_parents)
                    structure_metrics.append(metrics)
            
            # Select intervention using policy
            self.rng_key, policy_key = random.split(self.rng_key)
            policy_output = self.policy_fn.apply(
                self.policy_params, policy_key, tensor, target_idx
            )
            
            # Sample intervention
            var_logits = policy_output['variable_logits']
            value_params = policy_output['value_params']
            
            # Sample variable (excluding target)
            self.rng_key, var_key = random.split(self.rng_key)
            var_probs = jax.nn.softmax(var_logits)
            selected_var_idx = random.categorical(var_key, var_logits)
            
            # Sample value
            self.rng_key, val_key = random.split(self.rng_key)
            mean = value_params[selected_var_idx, 0]
            log_std = value_params[selected_var_idx, 1]
            std = jnp.exp(log_std)
            intervention_value = mean + std * random.normal(val_key)
            
            # Create and apply intervention
            selected_var = variables[selected_var_idx]
            intervention = create_perfect_intervention(
                targets=frozenset([selected_var]),
                values={selected_var: float(intervention_value)}
            )
            
            # Sample with intervention
            self.rng_key, sample_key = random.split(self.rng_key)
            intervention_samples = sample_with_intervention(
                scm, intervention, n_samples=10, seed=int(sample_key[0])
            )
            
            # Add to buffer
            for sample in intervention_samples:
                buffer.add_intervention(intervention, sample)
            
            # Compute clean reward using the new reward system
            # Use the first outcome sample for reward computation
            outcome_sample = intervention_samples[0] if intervention_samples else None
            
            if outcome_sample:
                reward_info = compute_clean_reward(
                    buffer_before=buffer,
                    intervention={
                        'targets': frozenset([selected_var]),
                        'values': {selected_var: float(intervention_value)}
                    },
                    outcome=outcome_sample,
                    target_variable=target_var,
                    config={
                        'optimization_direction': 'MINIMIZE',
                        'weights': {
                            'target': 1.0,
                            'diversity': 0.2,
                            'exploration': 0.1
                        }
                    }
                )
                reward = reward_info['total']
                total_reward += reward
                
                # Log reward components for debugging
                if step % 5 == 0:  # Log every 5 steps
                    logger.debug(
                        f"Step {step}: target_reward={reward_info['target']:.3f}, "
                        f"diversity={reward_info['diversity']:.3f}, "
                        f"exploration={reward_info['exploration']:.3f}, "
                        f"total={reward:.3f}"
                    )
            else:
                reward = 0.0
            
            # Compute loss (simple policy gradient)
            log_prob = jnp.log(var_probs[selected_var_idx] + 1e-8)
            loss = -log_prob * reward
            total_loss += loss
            
            # Update policy
            grads = jax.grad(lambda p: loss)(self.policy_params)
            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.policy_params = optax.apply_updates(self.policy_params, updates)
            
            # Update surrogate if enabled
            if self.use_surrogate and self.surrogate_update_fn is not None:
                surrogate_params, surrogate_opt_state, surrogate_metrics = self.surrogate_update_fn(
                    self.surrogate_params, self.surrogate_opt_state, buffer, target_var
                )
                self.surrogate_params = surrogate_params
                self.surrogate_opt_state = surrogate_opt_state
        
        # Compute average structure metrics
        avg_structure_metrics = {}
        if structure_metrics:
            for key in ['f1_score', 'precision', 'recall', 'shd']:
                values = [m[key] for m in structure_metrics if key in m]
                avg_structure_metrics[key] = sum(values) / len(values) if values else 0.0
        
        return {
            'episode': episode_idx,
            'mean_reward': total_reward / self.max_interventions,
            'loss': float(total_loss / self.max_interventions),
            'n_variables': len(variables),
            'scm_type': 'generated',
            'structure_metrics': avg_structure_metrics,
            'has_surrogate': self.use_surrogate
        }
    
    def _save_checkpoint(self, is_final: bool = False):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        name = "clean_grpo_final" if is_final else f"clean_grpo_ep{self.episode_count}"
        checkpoint_path = checkpoint_dir / name
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save checkpoint data
        checkpoint_data = {
            'policy_params': self.policy_params,
            'config': dict(self.config),
            'episode': self.episode_count,
            'is_final': is_final,
            'three_channel_format': True,  # Flag for new format
            'training_metrics': self.training_metrics[-10:] if self.training_metrics else [],
            'has_surrogate': self.use_surrogate
        }
        
        # Include surrogate parameters if enabled
        if self.use_surrogate and self.surrogate_params is not None:
            checkpoint_data['surrogate_params'] = self.surrogate_params
        
        with open(checkpoint_path / 'checkpoint.pkl', 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")


def create_clean_grpo_trainer(config: DictConfig) -> CleanGRPOTrainer:
    """
    Factory function to create clean GRPO trainer.
    
    Args:
        config: Training configuration
        
    Returns:
        Initialized CleanGRPOTrainer
    """
    return CleanGRPOTrainer(config)