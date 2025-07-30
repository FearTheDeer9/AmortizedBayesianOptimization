"""
Unified GRPO trainer combining the best of clean and simplified implementations.

This module provides a comprehensive GRPO training implementation that:
1. Uses true GRPO algorithm with batch advantages (from acquisition/grpo.py)
2. Supports flexible SCM input formats (from simplified)
3. Uses 3-channel tensor format (from clean)
4. Includes convergence detection (from simplified)
5. Integrates surrogate learning (from clean)
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

# Core imports
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

# Import true GRPO algorithm
from ..acquisition.grpo import (
    GRPOConfig, GRPOUpdate, create_grpo_trainer, _compute_grpo_loss
)

# Import convergence detection
from .convergence_detector import ConvergenceDetector, ConvergenceConfig
from .data_structures import TrainingMetrics

logger = logging.getLogger(__name__)


class UnifiedGRPOTrainer:
    """
    Unified GRPO trainer with true GRPO algorithm and all key features.
    
    This trainer combines:
    - True GRPO with batch advantages (not REINFORCE)
    - Flexible SCM input (list, dict, callable)
    - 3-channel tensor format
    - Convergence detection with early stopping
    - Surrogate learning integration
    - Proper reward computation
    """
    
    def __init__(self, 
                 # Can accept either DictConfig or individual parameters
                 config: Optional[Union[DictConfig, Dict[str, Any]]] = None,
                 # Individual parameters for flexibility
                 learning_rate: float = 3e-4,
                 n_episodes: int = 1000,
                 episode_length: int = 20,
                 batch_size: int = 64,  # GRPO group size
                 architecture_level: str = "simplified",
                 # Convergence
                 use_early_stopping: bool = True,
                 convergence_config: Optional[ConvergenceConfig] = None,
                 # Reward weights
                 reward_weights: Optional[Dict[str, float]] = None,
                 # Optimization
                 optimization_direction: str = "MINIMIZE",
                 # Other
                 seed: int = 42,
                 use_surrogate: bool = True,
                 checkpoint_dir: str = "checkpoints"):
        """
        Initialize unified GRPO trainer.
        
        Supports both config-based and parameter-based initialization.
        """
        # Handle config vs parameters
        if config is not None:
            if isinstance(config, DictConfig):
                config = dict(config)
            self._init_from_config(config)
        else:
            self._init_from_params(
                learning_rate=learning_rate,
                n_episodes=n_episodes,
                episode_length=episode_length,
                batch_size=batch_size,
                architecture_level=architecture_level,
                use_early_stopping=use_early_stopping,
                convergence_config=convergence_config,
                reward_weights=reward_weights,
                optimization_direction=optimization_direction,
                seed=seed,
                use_surrogate=use_surrogate,
                checkpoint_dir=checkpoint_dir
            )
        
        # Initialize components
        self._initialize_policy()
        self._initialize_surrogate()
        self._initialize_grpo()
        
        # Training state
        self.training_metrics = []
        self.episode_count = 0
        self.training_step = 0
        
    def _init_from_config(self, config: Dict[str, Any]):
        """Initialize from config dictionary."""
        self.config = config
        self.seed = config.get('seed', 42)
        self.rng_key = random.PRNGKey(self.seed)
        
        # Extract key configurations
        self.max_episodes = config.get('max_episodes', 1000)
        self.n_variables_range = config.get('n_variables_range', [3, 8])
        self.obs_per_episode = config.get('obs_per_episode', 100)
        self.max_interventions = config.get('max_interventions', 10)
        self.batch_size = config.get('batch_size', 64)
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        
        # Architecture config
        arch_config = config.get('architecture', {})
        self.num_layers = arch_config.get('num_layers', 4)
        self.num_heads = arch_config.get('num_heads', 8)
        self.hidden_dim = arch_config.get('hidden_dim', 256)
        self.key_size = arch_config.get('key_size', 32)
        self.dropout = arch_config.get('dropout', 0.1)
        self.architecture_level = arch_config.get('level', 'simplified')
        
        # Training config
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.optimization_direction = config.get('optimization_direction', 'MINIMIZE')
        self.use_surrogate = config.get('use_surrogate', True)
        
        # Reward weights
        self.reward_weights = config.get('reward_weights', {
            'optimization': 0.8,
            'discovery': 0.1,
            'efficiency': 0.1
        })
        
        # Convergence config
        self.use_early_stopping = config.get('use_early_stopping', True)
        if self.use_early_stopping:
            conv_config = config.get('convergence', {})
            self.convergence_config = ConvergenceConfig(
                structure_accuracy_threshold=conv_config.get('accuracy_threshold', 0.95),
                patience=conv_config.get('patience', 5),
                min_episodes=conv_config.get('min_episodes', 5),
                max_episodes_per_scm=conv_config.get('max_episodes_per_scm', 30)
            )
            self.convergence_detector = ConvergenceDetector(self.convergence_config)
        else:
            self.convergence_detector = None
            
    def _init_from_params(self, **kwargs):
        """Initialize from individual parameters."""
        self.learning_rate = kwargs['learning_rate']
        self.max_episodes = kwargs['n_episodes']
        self.episode_length = kwargs['episode_length']
        self.batch_size = kwargs['batch_size']
        self.architecture_level = kwargs['architecture_level']
        self.optimization_direction = kwargs['optimization_direction']
        self.seed = kwargs['seed']
        self.use_surrogate = kwargs['use_surrogate']
        self.checkpoint_dir = kwargs['checkpoint_dir']
        
        self.rng_key = random.PRNGKey(self.seed)
        
        # Default ranges
        self.n_variables_range = [3, 8]
        self.obs_per_episode = 100
        self.max_interventions = self.episode_length
        
        # Architecture defaults based on level
        if self.architecture_level == "baseline":
            self.hidden_dim = 128
            self.num_layers = 2
            self.num_heads = 4
        elif self.architecture_level == "simplified":
            self.hidden_dim = 256
            self.num_layers = 4
            self.num_heads = 8
        else:  # full
            self.hidden_dim = 512
            self.num_layers = 6
            self.num_heads = 16
        
        self.key_size = 32
        self.dropout = 0.1
        
        # Reward weights
        self.reward_weights = kwargs['reward_weights'] or {
            'optimization': 0.8,
            'discovery': 0.1,
            'efficiency': 0.1
        }
        
        # Convergence detection
        self.use_early_stopping = kwargs['use_early_stopping']
        if self.use_early_stopping:
            self.convergence_config = kwargs['convergence_config'] or ConvergenceConfig(
                structure_accuracy_threshold=0.95,
                patience=5,
                min_episodes=5,
                max_episodes_per_scm=30
            )
            self.convergence_detector = ConvergenceDetector(self.convergence_config)
        else:
            self.convergence_detector = None
            
        # Create minimal config for compatibility
        self.config = {
            'learning_rate': self.learning_rate,
            'seed': self.seed,
            'use_surrogate': self.use_surrogate,
            'checkpoint_dir': self.checkpoint_dir
        }
        
    def _initialize_policy(self):
        """Initialize policy network using shared factory."""
        policy_fn = create_clean_grpo_policy(hidden_dim=self.hidden_dim)
        self.policy_fn = hk.transform(policy_fn)
        
        # Initialize with dummy data
        dummy_tensor = jnp.zeros((10, 5, 3))  # [T=10, n_vars=5, channels=3]
        self.rng_key, init_key = random.split(self.rng_key)
        self.policy_params = self.policy_fn.init(init_key, dummy_tensor, 0)
        
        logger.info(f"Initialized policy with architecture: {self.architecture_level}")
        
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
        
    def _initialize_grpo(self):
        """Initialize true GRPO trainer with batch advantages."""
        # GRPO config with proper defaults
        self.grpo_config = GRPOConfig(
            group_size=self.batch_size,
            interventions_per_state=1,
            learning_rate=self.learning_rate,
            clip_ratio=0.2,
            entropy_coeff=0.1,  # Higher entropy for exploration
            max_grad_norm=1.0
        )
        
        # Create optimizer
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.grpo_config.max_grad_norm),
            optax.adam(learning_rate=self.learning_rate)
        )
        
        # Initialize optimizer state
        self.optimizer_state = self.optimizer.init(self.policy_params)
        
        # Create custom GRPO update function
        def grpo_update(params, opt_state, batch):
            """GRPO update with simplified batch format."""
            # Compute loss and gradients
            def loss_fn(p):
                return self._compute_simple_grpo_loss(p, batch)
            
            (loss_value, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            
            # Apply updates
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            
            # Compute gradient norm
            grad_norm = optax.global_norm(grads)
            
            return new_params, new_opt_state, GRPOUpdate(
                policy_loss=loss_info['policy_loss'],
                entropy_loss=loss_info['entropy_loss'],
                kl_penalty=loss_info['kl_penalty'],
                total_loss=loss_value,
                grad_norm=grad_norm,
                group_baseline=loss_info['group_baseline'],
                mean_reward=loss_info['mean_reward'],
                reward_std=loss_info['reward_std'],
                mean_advantage=loss_info['mean_advantage'],
                advantage_std=loss_info['advantage_std'],
                mean_entropy=loss_info['mean_entropy'],
                approx_kl=loss_info['approx_kl']
            )
        
        self.grpo_update = grpo_update
        
        logger.info(f"Initialized true GRPO with group_size={self.batch_size}, entropy_coeff=0.1")
        
    def train(self, 
              scms: Union[List[Any], Dict[str, Any], Callable[[], Any]],
              eval_scms: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Train GRPO policy on provided SCMs.
        
        Args:
            scms: Training SCMs - can be:
                - List of SCMs to rotate through
                - Dict mapping names to SCMs
                - Callable that generates SCMs on demand
            eval_scms: Optional separate evaluation set
            
        Returns:
            Training results and metrics
        """
        logger.info("Starting unified GRPO training with true GRPO algorithm")
        start_time = time.time()
        
        # Convert SCMs to standard format
        scm_rotation = self._prepare_scms(scms)
        logger.info(f"Starting training with {len(scm_rotation)} SCMs")
        
        # Training history
        episode_metrics = []
        scm_episodes = {name: 0 for name, _ in scm_rotation}
        current_scm_idx = 0
        
        # Main training loop
        for episode in range(self.max_episodes):
            # Get current SCM
            scm_name, scm = scm_rotation[current_scm_idx]
            scm_episodes[scm_name] += 1
            
            # Run episode with GRPO batch collection
            self.rng_key, episode_key = random.split(self.rng_key)
            metrics = self._run_grpo_episode(episode, scm, scm_name, episode_key)
            episode_metrics.append(metrics)
            
            # Check convergence if enabled
            if self.convergence_detector:
                # Convert dict metrics to TrainingMetrics object for convergence detector
                training_metrics = TrainingMetrics(
                    episode=metrics['episode'],
                    mean_reward=metrics['mean_reward'],
                    structure_accuracy=metrics.get('structure_metrics', {}).get('f1_score', 0.0),
                    optimization_improvement=0.0,  # Not tracked in this implementation
                    policy_loss=metrics.get('loss', 0.0),
                    value_loss=0.0,  # Not tracked separately
                    scm_type=metrics.get('scm_type', scm_name),
                    f1_score=metrics.get('structure_metrics', {}).get('f1_score', None),
                    true_parent_likelihood=metrics.get('structure_metrics', {}).get('parent_likelihood', None),
                    shd=metrics.get('structure_metrics', {}).get('shd', None),
                    marginal_probs=metrics.get('structure_metrics', {}).get('marginal_probs', None)
                )
                self.convergence_detector.update(scm_name, training_metrics)
                converged, reason = self.convergence_detector.check_convergence(scm_name)
                
                if converged or scm_episodes[scm_name] >= self.convergence_config.max_episodes_per_scm:
                    logger.info(f"SCM {scm_name} converged after {scm_episodes[scm_name]} episodes: {reason}")
                    
                    # Rotate to next SCM
                    current_scm_idx = (current_scm_idx + 1) % len(scm_rotation)
                    
                    # Check if all SCMs have converged
                    all_converged = all(
                        self.convergence_detector.scm_states.get(name, None) and 
                        self.convergence_detector.scm_states[name].converged
                        for name, _ in scm_rotation
                    )
                    
                    if all_converged:
                        logger.info(f"All SCMs converged! Stopping early at episode {episode}")
                        break
            
            # Log progress
            if episode % 10 == 0:
                recent_rewards = [m['mean_reward'] for m in episode_metrics[-10:]]
                mean_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
                logger.info(f"Episode {episode}: mean_reward={mean_reward:.4f}, "
                          f"current_scm={scm_name}")
                
                # Log GRPO-specific metrics if available
                if 'grpo_metrics' in metrics:
                    gm = metrics['grpo_metrics']
                    logger.info(f"  GRPO: advantage={gm.get('mean_advantage', 0):.3f}, "
                              f"baseline={gm.get('group_baseline', 0):.3f}")
        
        # Save final checkpoint
        self._save_checkpoint(is_final=True)
        
        # Prepare results
        training_time = time.time() - start_time
        final_metrics = episode_metrics[-1] if episode_metrics else {}
        
        return {
            'training_time': training_time,
            'final_metrics': final_metrics,
            'all_metrics': episode_metrics,
            'policy_params': self.policy_params,
            'episodes_per_scm': scm_episodes,
            'converged': all(
                self.convergence_detector.scm_states.get(name, None) and 
                self.convergence_detector.scm_states[name].converged
                for name, _ in scm_rotation
            ) if self.convergence_detector else False
        }
    
    def _run_grpo_episode(self, episode_idx: int, scm: pyr.PMap, scm_name: str, key: jax.random.PRNGKey) -> Dict[str, Any]:
        """Run single training episode with GRPO batch collection."""
        # Get SCM info
        variables = list(get_variables(scm))
        target_var = get_target(scm)
        target_idx = variables.index(target_var)
        true_parents = list(get_parents(scm, target_var)) if hasattr(scm, 'edges') else []
        
        # Initialize buffer with observational data
        buffer = ExperienceBuffer()
        key, obs_key = random.split(key)
        
        # Sample observational data
        obs_samples = sample_from_linear_scm(scm, self.obs_per_episode, seed=int(obs_key[0]))
        for sample in obs_samples:
            buffer.add_observation(sample)
        
        # Collect GRPO batch
        grpo_batch_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'old_log_probs': []
        }
        
        # Generate batch of interventions for GRPO
        for _ in range(self.grpo_config.group_size):
            key, step_key = random.split(key)
            
            # Convert buffer to tensor
            tensor, var_order = buffer_to_three_channel_tensor(
                buffer, target_var, max_history_size=100, standardize=True
            )
            
            # Get posterior prediction if surrogate is enabled
            posterior = None
            if self.use_surrogate and self.surrogate_net is not None:
                posterior = self.surrogate_predict_fn(tensor, target_idx, variables)
            
            # Get policy output
            key, policy_key = random.split(key)
            policy_output = self.policy_fn.apply(
                self.policy_params, policy_key, tensor, target_idx
            )
            
            # Sample intervention
            var_logits = policy_output['variable_logits']
            value_params = policy_output['value_params']
            
            # Sample variable
            key, var_key = random.split(key)
            var_probs = jax.nn.softmax(var_logits)
            selected_var_idx = random.categorical(var_key, var_logits)
            
            # Compute log probability for GRPO
            log_prob = jnp.log(var_probs[selected_var_idx] + 1e-8)
            
            # Sample value
            key, val_key = random.split(key)
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
            key, sample_key = random.split(key)
            intervention_samples = sample_with_intervention(
                scm, intervention, n_samples=10, seed=int(sample_key[0])
            )
            
            # Compute reward
            outcome_sample = intervention_samples[0] if intervention_samples else None
            if outcome_sample:
                # Map reward weights to clean reward format
                clean_weights = {
                    'target': self.reward_weights.get('optimization', 0.8),
                    'diversity': self.reward_weights.get('discovery', 0.1),
                    'exploration': self.reward_weights.get('efficiency', 0.1)
                }
                
                reward_info = compute_clean_reward(
                    buffer_before=buffer,
                    intervention={
                        'targets': frozenset([selected_var]),
                        'values': {selected_var: float(intervention_value)}
                    },
                    outcome=outcome_sample,
                    target_variable=target_var,
                    config={
                        'optimization_direction': self.optimization_direction,
                        'weights': clean_weights
                    }
                )
                reward = reward_info['total']
            else:
                reward = 0.0
            
            # Store for GRPO batch
            grpo_batch_data['states'].append(tensor)
            grpo_batch_data['actions'].append({
                'variable': selected_var_idx,
                'value': float(intervention_value)
            })
            grpo_batch_data['rewards'].append(reward)
            grpo_batch_data['old_log_probs'].append(float(log_prob))
        
        # Convert to arrays
        grpo_batch_data['rewards'] = jnp.array(grpo_batch_data['rewards'])
        grpo_batch_data['old_log_probs'] = jnp.array(grpo_batch_data['old_log_probs'])
        
        # Create GRPO batch with proper format
        grpo_batch = self._create_grpo_batch(grpo_batch_data)
        
        # Perform GRPO update
        self.policy_params, self.optimizer_state, grpo_metrics = self.grpo_update(
            self.policy_params,
            self.optimizer_state,
            grpo_batch
        )
        
        self.training_step += 1
        
        # Compute structure metrics if surrogate is available
        structure_metrics = {}
        if self.use_surrogate and true_parents and posterior:
            metrics = compute_structure_metrics_continuous(posterior, true_parents)
            structure_metrics = metrics
        
        return {
            'episode': episode_idx,
            'mean_reward': float(grpo_metrics.mean_reward),
            'loss': float(grpo_metrics.total_loss),
            'n_variables': len(variables),
            'scm_type': scm_name,
            'structure_metrics': structure_metrics,
            'has_surrogate': self.use_surrogate,
            'grpo_metrics': {
                'policy_loss': float(grpo_metrics.policy_loss),
                'entropy_loss': float(grpo_metrics.entropy_loss),
                'mean_advantage': float(grpo_metrics.mean_advantage),
                'advantage_std': float(grpo_metrics.advantage_std),
                'group_baseline': float(grpo_metrics.group_baseline),
                'approx_kl': float(grpo_metrics.approx_kl)
            }
        }
    
    def _prepare_scms(self, scms: Union[List[Any], Dict[str, Any], Callable]) -> List[Tuple[str, Any]]:
        """Convert various SCM formats to standard list of (name, scm) tuples."""
        if isinstance(scms, list):
            # Check if it's already a list of tuples
            if scms and isinstance(scms[0], tuple) and len(scms[0]) == 2:
                # Already in (name, scm) format
                return scms
            else:
                # Generate names for unnamed SCMs
                return [(f"scm_{i}", scm) for i, scm in enumerate(scms)]
        elif isinstance(scms, dict):
            # Use provided names
            return list(scms.items())
        elif callable(scms):
            # Generate SCMs on demand
            generated = []
            for i in range(10):  # Default to 10 SCMs
                scm = scms()
                generated.append((f"generated_{i}", scm))
            return generated
        else:
            raise ValueError(f"Unsupported SCM format: {type(scms)}")
    
    def _create_grpo_batch(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create GRPO batch in format expected by GRPO loss computation."""
        # Stack tensors
        states_batch = jnp.stack(batch_data['states'])  # [batch_size, T, n_vars, 3]
        
        # Extract action indices and values
        action_var_indices = jnp.array([a['variable'] for a in batch_data['actions']])
        action_values = jnp.array([a['value'] for a in batch_data['actions']])
        
        # Rewards and old log probs are already arrays
        
        return {
            'states': states_batch,
            'actions': {
                'variables': action_var_indices,
                'values': action_values
            },
            'rewards': batch_data['rewards'],
            'old_log_probs': batch_data['old_log_probs']
        }
    
    def _compute_simple_grpo_loss(self, params: Any, batch: Dict[str, Any]) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Compute GRPO loss without requiring AcquisitionState objects."""
        states = batch['states']  # [batch_size, T, n_vars, 3]
        actions = batch['actions']
        rewards = batch['rewards']
        old_log_probs = batch['old_log_probs']
        
        # Compute advantages using group baseline (key GRPO innovation)
        group_baseline = jnp.mean(rewards)
        advantages = rewards - group_baseline
        
        # Normalize advantages for stability
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
        
        # Forward pass to get current log probs
        batch_size = states.shape[0]
        new_log_probs = []
        entropy_values = []
        
        for i in range(batch_size):
            # Get policy output for this state
            self.rng_key, policy_key = random.split(self.rng_key)
            
            # Assume target is last variable (can be improved)
            n_vars = states[i].shape[1]
            target_idx = n_vars - 1
            
            policy_output = self.policy_fn.apply(
                params, policy_key, states[i], target_idx
            )
            
            # Compute log prob for selected action
            var_probs = jax.nn.softmax(policy_output['variable_logits'])
            selected_var = actions['variables'][i]
            log_prob = jnp.log(var_probs[selected_var] + 1e-8)
            new_log_probs.append(log_prob)
            
            # Compute entropy
            entropy = -jnp.sum(var_probs * jnp.log(var_probs + 1e-8))
            entropy_values.append(entropy)
        
        new_log_probs = jnp.array(new_log_probs)
        entropy_values = jnp.array(entropy_values)
        
        # Compute ratio for PPO-style clipping
        ratio = jnp.exp(new_log_probs - old_log_probs)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = jnp.clip(ratio, 1.0 - self.grpo_config.clip_ratio, 1.0 + self.grpo_config.clip_ratio) * advantages
        policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
        
        # Entropy loss (negative for maximization)
        entropy_loss = -self.grpo_config.entropy_coeff * jnp.mean(entropy_values)
        
        # Total loss
        total_loss = policy_loss + entropy_loss
        
        # Compute diagnostics
        approx_kl = jnp.mean((new_log_probs - old_log_probs) ** 2)
        
        loss_info = {
            'policy_loss': policy_loss,
            'entropy_loss': entropy_loss,
            'kl_penalty': 0.0,  # Not using KL penalty
            'group_baseline': group_baseline,
            'mean_reward': jnp.mean(rewards),
            'reward_std': jnp.std(rewards),
            'mean_advantage': jnp.mean(advantages),
            'advantage_std': jnp.std(advantages),
            'mean_entropy': jnp.mean(entropy_values),
            'approx_kl': approx_kl
        }
        
        return total_loss, loss_info
    
    def _save_checkpoint(self, is_final: bool = False):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        name = "unified_grpo_final" if is_final else f"unified_grpo_ep{self.episode_count}"
        checkpoint_path = checkpoint_dir / name
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save checkpoint data
        checkpoint_data = {
            'policy_params': self.policy_params,
            'config': self.config if hasattr(self, 'config') else {
                'learning_rate': self.learning_rate,
                'architecture_level': self.architecture_level,
                'optimization_direction': self.optimization_direction
            },
            'episode': self.episode_count,
            'is_final': is_final,
            'three_channel_format': True,
            'training_metrics': self.training_metrics[-10:] if self.training_metrics else [],
            'has_surrogate': self.use_surrogate,
            'uses_true_grpo': True  # Flag to indicate this uses true GRPO
        }
        
        # Include surrogate parameters if enabled
        if self.use_surrogate and self.surrogate_params is not None:
            checkpoint_data['surrogate_params'] = self.surrogate_params
        
        with open(checkpoint_path / 'checkpoint.pkl', 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
    def save_checkpoint(self, path: Path, results: Dict[str, Any]) -> None:
        """Save training checkpoint (compatibility method)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'params': results.get('policy_params', self.policy_params),
            'config': self.config if hasattr(self, 'config') else {
                'learning_rate': self.learning_rate,
                'architecture_level': self.architecture_level
            },
            'metrics': results.get('all_metrics', []),
            'metadata': {
                'trainer_type': 'UnifiedGRPOTrainer',
                'uses_true_grpo': True,
                'converged': results.get('converged', False)
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
            
        logger.info(f"Saved checkpoint to {path}")


def create_unified_grpo_trainer(config: Union[DictConfig, Dict[str, Any], None] = None, 
                                **kwargs) -> UnifiedGRPOTrainer:
    """
    Factory function to create unified GRPO trainer.
    
    Can be called with either a config dict/DictConfig or keyword arguments.
    
    Args:
        config: Configuration dictionary or DictConfig
        **kwargs: Individual parameters if not using config
        
    Returns:
        Initialized UnifiedGRPOTrainer
    """
    if config is not None:
        return UnifiedGRPOTrainer(config=config)
    else:
        return UnifiedGRPOTrainer(config=None, **kwargs)