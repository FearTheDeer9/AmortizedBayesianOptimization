"""
Streamlined GRPO trainer core - focused on essential training logic.

This module provides the core GRPO training functionality without the complexity
of extensive logging, multiple initialization paths, or overlapping features.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as random
import optax
import haiku as hk
import pyrsistent as pyr

from ..data_structures.buffer import ExperienceBuffer
from ..data_structures.sample import get_values
from ..data_structures.scm import get_variables, get_target, get_parents
from ..mechanisms.linear import sample_from_linear_scm
from ..interventions.handlers import create_perfect_intervention
from ..environments.sampling import sample_with_intervention
from ..policies.clean_policy_factory import create_clean_grpo_policy
from ..acquisition.grpo import GRPOConfig, GRPOUpdate
from .five_channel_converter import buffer_to_five_channel_tensor
from .four_channel_converter import buffer_to_four_channel_tensor
from .grpo_reward_computer import GRPORewardComputer, create_reward_computer_from_config
from .grpo_logger import GRPOLogger

logger = logging.getLogger(__name__)


@dataclass
class GRPOTrainerConfig:
    """Clean configuration for GRPO trainer."""
    # Core training
    max_episodes: int = 100
    obs_per_episode: int = 100
    max_interventions: int = 10
    batch_size: int = 32
    learning_rate: float = 3e-4
    
    # Architecture
    policy_architecture: str = "permutation_invariant"
    hidden_dim: int = 256
    
    # GRPO specific
    entropy_coeff: float = 0.01
    clip_ratio: float = 0.2
    gradient_clip: float = 1.0
    ppo_epochs: int = 4
    
    # Reward
    reward_type: str = "composite"
    optimization_direction: str = "MINIMIZE"
    reward_weights: Dict[str, float] = None
    
    # Other
    use_surrogate: bool = False
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    
    def __post_init__(self):
        if self.reward_weights is None:
            self.reward_weights = {'target': 0.7, 'parent': 0.1, 'info_gain': 0.2}


class GRPOTrainerCore:
    """Streamlined GRPO trainer focused on core training logic."""
    
    def __init__(self, config: Union[Dict[str, Any], GRPOTrainerConfig]):
        """Initialize with clean configuration."""
        # Handle both dict and dataclass config
        if isinstance(config, dict):
            self.config = GRPOTrainerConfig(**{k: v for k, v in config.items() 
                                             if k in GRPOTrainerConfig.__annotations__})
        else:
            self.config = config
        
        # Initialize components
        self.rng_key = random.PRNGKey(self.config.seed)
        
        # Initialize policy
        self._init_policy()
        
        # Initialize GRPO
        self._init_grpo()
        
        # Initialize reward computer
        self.reward_computer = create_reward_computer_from_config(config if isinstance(config, dict) else config.__dict__)
        
        # Initialize logger
        self.logger = GRPOLogger(self.config.optimization_direction)
        
        # Training state
        self.training_step = 0
        
        logger.info(f"Initialized GRPOTrainerCore with {self.config.policy_architecture} architecture")
    
    def _init_policy(self):
        """Initialize policy network."""
        policy_fn = create_clean_grpo_policy(
            hidden_dim=self.config.hidden_dim,
            architecture=self.config.policy_architecture
        )
        
        self.policy_fn = hk.transform(policy_fn)
        
        # Initialize parameters - use correct number of channels
        n_channels = 4 if self.config.policy_architecture in ['quantile', 'permutation_invariant'] else 5
        dummy_tensor = jnp.zeros((10, 5, n_channels))
        self.rng_key, init_key = random.split(self.rng_key)
        self.policy_params = self.policy_fn.init(init_key, dummy_tensor, 0)
    
    def _init_grpo(self):
        """Initialize GRPO optimizer and configuration."""
        self.grpo_config = GRPOConfig(
            group_size=self.config.batch_size,
            interventions_per_state=1,
            learning_rate=self.config.learning_rate,
            clip_ratio=self.config.clip_ratio,
            entropy_coeff=self.config.entropy_coeff,
            max_grad_norm=self.config.gradient_clip
        )
        
        # Create optimizer
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.gradient_clip),
            optax.adam(learning_rate=self.config.learning_rate)
        )
        
        self.optimizer_state = self.optimizer.init(self.policy_params)
    
    def train(self, scms: Union[List[Any], Callable]) -> Dict[str, Any]:
        """
        Train GRPO policy on provided SCMs.
        
        Args:
            scms: Training SCMs (list or callable)
            
        Returns:
            Training results and metrics
        """
        start_time = time.time()
        
        # Convert SCMs to list format
        if callable(scms):
            scm_list = [(f"scm_{i}", scms()) for i in range(10)]
        elif isinstance(scms, list):
            scm_list = [(f"scm_{i}", scm) for i, scm in enumerate(scms)]
        else:
            scm_list = [("single_scm", scms)]
        
        episode_metrics = []
        
        # Training loop
        for episode in range(self.config.max_episodes):
            # Get current SCM
            scm_name, scm = scm_list[episode % len(scm_list)]
            
            # Run episode
            self.rng_key, episode_key = random.split(self.rng_key)
            metrics = self._run_episode(episode, scm, scm_name, episode_key)
            episode_metrics.append(metrics)
            
            # Log progress
            self.logger.log_training_progress(episode, episode_metrics, scm_name)
        
        # Final summary
        self.logger.log_final_summary()
        
        return {
            'training_time': time.time() - start_time,
            'final_metrics': episode_metrics[-1] if episode_metrics else {},
            'all_metrics': episode_metrics,
            'policy_params': self.policy_params
        }
    
    def _run_episode(self, episode_idx: int, scm: Any, scm_name: str, key: jax.random.PRNGKey) -> Dict[str, Any]:
        """Run single GRPO training episode."""
        # Get SCM info
        variables = list(get_variables(scm))
        target_var = get_target(scm)
        true_parents = list(get_parents(scm, target_var))
        
        # Log episode start
        self.logger.log_episode_start(episode_idx, scm_name, target_var, true_parents)
        
        # Initialize buffer with observational data
        buffer = ExperienceBuffer()
        key, obs_key = random.split(key)
        
        obs_samples = sample_from_linear_scm(scm, self.config.obs_per_episode, seed=int(obs_key[0]))
        for sample in obs_samples:
            buffer.add_observation(sample)
        
        # Run multiple interventions per episode
        all_rewards = []
        
        for intervention_idx in range(self.config.max_interventions):
            # Collect GRPO batch for this intervention
            grpo_batch_data = {
                'states': [],
                'actions': [],
                'rewards': [],
                'old_log_probs': [],
                'target_idx': None,
                'intervention_details': []
            }
            
            # Generate batch of candidates
            for step in range(self.grpo_config.group_size):
                key, step_key = random.split(key)
                
                # Convert buffer to tensor
                if self.config.policy_architecture == "permutation_invariant":
                    tensor, mapper, _ = buffer_to_four_channel_tensor(
                        buffer, target_var, max_history_size=100, standardize=True
                    )
                else:
                    tensor, mapper, _ = buffer_to_five_channel_tensor(
                        buffer, target_var, max_history_size=100, standardize=False
                    )
                
                # Get policy output
                key, policy_key = random.split(key)
                policy_output = self.policy_fn.apply(
                    self.policy_params, policy_key, tensor, mapper.target_idx
                )
                
                # Sample intervention
                var_logits = policy_output['variable_logits']
                value_params = policy_output['value_params']
                
                key, var_key = random.split(key)
                selected_var_idx = random.categorical(var_key, var_logits)
                var_probs = jax.nn.softmax(var_logits)
                
                # Log probabilities (first step only)
                if step == 0:
                    self.logger.log_variable_probabilities(var_probs, mapper, target_var, episode_idx, scm_name)
                    if intervention_idx == 0:  # First intervention
                        self.logger.log_buffer_state(buffer)
                        self.logger.log_tensor_analysis(tensor, mapper, target_var)
                
                # Sample intervention value
                key, val_key = random.split(key)
                mean = value_params[selected_var_idx, 0]
                log_std = value_params[selected_var_idx, 1]
                std = jnp.exp(log_std)
                intervention_value = mean + std * random.normal(val_key)
                
                # Create intervention
                selected_var = mapper.get_name(int(selected_var_idx))
                intervention = create_perfect_intervention(
                    targets=frozenset([selected_var]),
                    values={selected_var: float(intervention_value)}
                )
                
                # Apply intervention
                key, sample_key = random.split(key)
                intervention_samples = sample_with_intervention(
                    scm, intervention, n_samples=1, seed=int(sample_key[0])
                )
                
                # Compute reward
                outcome_sample = intervention_samples[0] if intervention_samples else None
                if outcome_sample:
                    reward_info = self.reward_computer.compute_reward(
                        intervention=intervention,
                        outcome_sample=outcome_sample,
                        buffer=buffer,
                        scm=scm,
                        target_variable=target_var,
                        variables=variables,
                        tensor_5ch=tensor,
                        mapper=mapper
                    )
                    reward = reward_info['total']
                    
                    # Track target value
                    target_val = get_values(outcome_sample).get(target_var, 0.0)
                    self.logger.track_target_value(episode_idx, target_var, target_val, scm_name, scm)
                else:
                    reward = 0.0
                
                # Store for GRPO batch
                log_prob = jnp.log(var_probs[selected_var_idx] + 1e-8)
                grpo_batch_data['states'].append(tensor)
                grpo_batch_data['actions'].append({
                    'variable': selected_var_idx,
                    'value': float(intervention_value)
                })
                grpo_batch_data['rewards'].append(reward)
                grpo_batch_data['old_log_probs'].append(float(log_prob))
                grpo_batch_data['intervention_details'].append({
                    'intervention': intervention,
                    'samples': intervention_samples
                })
                
                if grpo_batch_data['target_idx'] is None:
                    grpo_batch_data['target_idx'] = mapper.target_idx
            
            # Convert to arrays
            grpo_batch_data['rewards'] = jnp.array(grpo_batch_data['rewards'])
            grpo_batch_data['old_log_probs'] = jnp.array(grpo_batch_data['old_log_probs'])
            
            # Log candidates and advantages
            if intervention_idx == 0:  # First intervention gets full logging
                self.logger.log_candidates_with_rewards(
                    grpo_batch_data, mapper, target_var, scm, self.reward_computer.legacy_reward_config
                )
                self.logger.log_grpo_advantages(grpo_batch_data['rewards'])
            
            # Create GRPO batch
            grpo_batch = self._create_grpo_batch(grpo_batch_data)
            
            # GRPO update
            old_policy_output = self.policy_fn.apply(
                self.policy_params, random.PRNGKey(42), tensor, mapper.target_idx
            )
            
            self.policy_params, self.optimizer_state, grpo_metrics = self._grpo_update(
                self.policy_params, self.optimizer_state, grpo_batch
            )
            
            # Log policy evolution (first intervention only)
            if intervention_idx == 0:
                new_policy_output = self.policy_fn.apply(
                    self.policy_params, random.PRNGKey(42), tensor, mapper.target_idx
                )
                self.logger.log_policy_evolution(
                    old_policy_output['variable_logits'],
                    new_policy_output['variable_logits']
                )
            
            # Add best intervention to buffer
            best_idx = jnp.argmax(grpo_batch_data['rewards'])
            best_intervention_info = grpo_batch_data['intervention_details'][int(best_idx)]
            
            if best_intervention_info['samples']:
                buffer.add_intervention(
                    best_intervention_info['intervention'],
                    best_intervention_info['samples'][0]
                )
            
            all_rewards.extend(grpo_batch_data['rewards'])
            self.training_step += 1
        
        return {
            'episode': episode_idx,
            'mean_reward': float(jnp.mean(jnp.array(all_rewards))) if all_rewards else 0.0,
            'loss': float(grpo_metrics.total_loss),
            'n_variables': len(variables),
            'scm_type': scm_name,
            'grpo_metrics': {
                'policy_loss': float(grpo_metrics.policy_loss),
                'entropy_loss': float(grpo_metrics.entropy_loss),
                'mean_advantage': float(grpo_metrics.mean_advantage),
                'group_baseline': float(grpo_metrics.group_baseline),
                'grad_norm': float(grpo_metrics.grad_norm)
            }
        }
    
    def _create_grpo_batch(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create GRPO batch format."""
        states_batch = jnp.stack(batch_data['states'])
        action_var_indices = jnp.array([a['variable'] for a in batch_data['actions']])
        action_values = jnp.array([a['value'] for a in batch_data['actions']])
        
        return {
            'states': states_batch,
            'actions': {
                'variables': action_var_indices,
                'values': action_values
            },
            'rewards': batch_data['rewards'],
            'old_log_probs': batch_data['old_log_probs'],
            'target_idx': batch_data['target_idx']
        }
    
    def _grpo_update(self, params: Any, opt_state: Any, batch: Dict[str, Any]) -> Tuple[Any, Any, Any]:
        """Perform GRPO policy update."""
        def loss_fn(p):
            return self._compute_grpo_loss(p, batch)
        
        (loss_value, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        
        # Log gradient analysis
        self.logger.log_gradient_analysis(grads, batch['rewards'])
        
        # Apply updates
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        # Create metrics
        grpo_metrics = GRPOUpdate(
            policy_loss=loss_info['policy_loss'],
            entropy_loss=loss_info['entropy_loss'],
            kl_penalty=0.0,
            total_loss=loss_value,
            grad_norm=optax.global_norm(grads),
            group_baseline=loss_info['group_baseline'],
            mean_reward=loss_info['mean_reward'],
            reward_std=loss_info['reward_std'],
            mean_advantage=loss_info['mean_advantage'],
            advantage_std=loss_info['advantage_std'],
            mean_entropy=loss_info['mean_entropy'],
            approx_kl=loss_info['approx_kl']
        )
        
        # Log loss analysis
        self.logger.log_loss_analysis(loss_info)
        
        return new_params, new_opt_state, grpo_metrics
    
    def _compute_grpo_loss(self, params: Any, batch: Dict[str, Any]) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Compute GRPO loss with group advantages."""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        old_log_probs = batch['old_log_probs']
        
        # GRPO group baseline
        group_baseline = jnp.mean(rewards)
        advantages = rewards - group_baseline
        
        # Normalize advantages
        advantages = advantages / (jnp.std(advantages) + 1e-8)
        
        # Forward pass
        batch_size = states.shape[0]
        new_log_probs = []
        entropy_values = []
        
        for i in range(batch_size):
            self.rng_key, policy_key = random.split(self.rng_key)
            
            policy_output = self.policy_fn.apply(
                params, policy_key, states[i], batch['target_idx']
            )
            
            var_probs = jax.nn.softmax(policy_output['variable_logits'])
            selected_var = actions['variables'][i]
            log_prob = jnp.log(var_probs[selected_var] + 1e-8)
            new_log_probs.append(log_prob)
            
            entropy = -jnp.sum(var_probs * jnp.log(var_probs + 1e-8))
            entropy_values.append(entropy)
        
        new_log_probs = jnp.array(new_log_probs)
        entropy_values = jnp.array(entropy_values)
        
        # PPO clipped loss
        ratio = jnp.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = jnp.clip(ratio, 1.0 - self.grpo_config.clip_ratio, 1.0 + self.grpo_config.clip_ratio) * advantages
        policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
        
        # Entropy loss
        entropy_loss = -self.grpo_config.entropy_coeff * jnp.mean(entropy_values)
        
        # Total loss
        total_loss = policy_loss + entropy_loss
        
        loss_info = {
            'policy_loss': policy_loss,
            'entropy_loss': entropy_loss,
            'group_baseline': group_baseline,
            'mean_reward': jnp.mean(rewards),
            'reward_std': jnp.std(rewards),
            'mean_advantage': jnp.mean(advantages),
            'advantage_std': jnp.std(advantages),
            'mean_entropy': jnp.mean(entropy_values),
            'approx_kl': jnp.mean((new_log_probs - old_log_probs) ** 2),
            'total_loss': total_loss
        }
        
        return total_loss, loss_info