"""
Pure GRPO trainer - focused only on policy training.

This trainer handles ONLY:
- GRPO policy episodes
- Reward computation 
- Policy parameter updates
- Policy checkpointing

It does NOT handle:
- Phase switching (orchestrator's job)
- SCM rotation (orchestrator's job)  
- Surrogate training (AVICI's job)
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import optax
import haiku as hk
import pyrsistent as pyr

from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.data_structures.sample import get_values
from src.causal_bayes_opt.data_structures.scm import get_variables, get_target, get_parents
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
from src.causal_bayes_opt.environments.sampling import sample_with_intervention
from src.causal_bayes_opt.policies.clean_policy_factory import create_clean_grpo_policy
from src.causal_bayes_opt.acquisition.grpo import GRPOConfig, GRPOUpdate
from src.causal_bayes_opt.acquisition.composite_reward import compute_composite_reward, RewardConfig
from src.causal_bayes_opt.acquisition.better_rewards import RunningStats
from src.causal_bayes_opt.training.four_channel_converter import buffer_to_four_channel_tensor
from src.causal_bayes_opt.training.five_channel_converter import buffer_to_five_channel_tensor

logger = logging.getLogger(__name__)


@dataclass
class PureGRPOConfig:
    """Configuration for pure GRPO training."""
    # Core settings
    obs_per_episode: int = 100
    max_interventions: int = 10
    batch_size: int = 32
    learning_rate: float = 3e-4
    
    # Architecture
    policy_architecture: str = "permutation_invariant"
    hidden_dim: int = 256
    use_fixed_std: bool = False
    fixed_std: float = 0.5
    
    # GRPO settings
    entropy_coeff: float = 0.01
    clip_ratio: float = 0.2
    gradient_clip: float = 1.0
    ppo_epochs: int = 4
    
    # Reward settings
    reward_type: str = "composite"
    optimization_direction: str = "MINIMIZE"
    reward_weights: Dict[str, float] = None
    
    # Other
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    
    def __post_init__(self):
        if self.reward_weights is None:
            self.reward_weights = {'target': 1.0, 'parent': 0.0, 'info_gain': 0.0}


class PureGRPOTrainer:
    """Pure GRPO trainer focused only on policy training."""
    
    def __init__(self, config: Union[Dict[str, Any], PureGRPOConfig]):
        """Initialize pure GRPO trainer."""
        # Handle config conversion
        if isinstance(config, dict):
            # Extract relevant fields for PureGRPOConfig
            grpo_config_dict = config.get('grpo_config', {})
            
            self.config = PureGRPOConfig(
                obs_per_episode=config.get('obs_per_episode', 100),
                max_interventions=config.get('max_interventions', 10),
                batch_size=grpo_config_dict.get('group_size', config.get('batch_size', 32)),
                learning_rate=config.get('learning_rate', 3e-4),
                policy_architecture=config.get('policy_architecture', 'permutation_invariant'),
                hidden_dim=config.get('architecture', {}).get('hidden_dim', 256),
                use_fixed_std=config.get('use_fixed_std', False),
                fixed_std=config.get('fixed_std', 0.5),
                entropy_coeff=grpo_config_dict.get('entropy_coefficient', 0.01),
                clip_ratio=grpo_config_dict.get('clip_ratio', 0.2),
                gradient_clip=grpo_config_dict.get('gradient_clip', 1.0),
                ppo_epochs=grpo_config_dict.get('ppo_epochs', 4),
                reward_type=config.get('reward_type', 'composite'),
                optimization_direction=config.get('optimization_direction', 'MINIMIZE'),
                reward_weights=config.get('reward_weights', {'target': 1.0, 'parent': 0.0, 'info_gain': 0.0}),
                seed=config.get('seed', 42),
                checkpoint_dir=config.get('checkpoint_dir', 'checkpoints')
            )
            # Store original config for compatibility
            self.original_config = config
        else:
            self.config = config
            self.original_config = {}
        
        # Initialize RNG
        self.rng_key = random.PRNGKey(self.config.seed)
        
        # Initialize components
        self._init_policy()
        self._init_grpo() 
        self._init_rewards()
        
        # Training state
        self.training_step = 0
        
        logger.info(f"Initialized PureGRPOTrainer with {self.config.policy_architecture} architecture")
    
    def _init_policy(self):
        """Initialize policy network."""
        policy_fn = create_clean_grpo_policy(
            hidden_dim=self.config.hidden_dim,
            architecture=self.config.policy_architecture,
            use_fixed_std=self.config.use_fixed_std,
            fixed_std=self.config.fixed_std
        )
        
        self.policy_fn = hk.transform(policy_fn)
        
        # Initialize parameters
        dummy_tensor = jnp.zeros((10, 5, 5))
        self.rng_key, init_key = random.split(self.rng_key)
        self.policy_params = self.policy_fn.init(init_key, dummy_tensor, 0)
        
        logger.info(f"Initialized policy with {self.config.policy_architecture}")
    
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
        
        logger.info(f"Initialized GRPO with group_size={self.config.batch_size}")
    
    def _init_rewards(self):
        """Initialize reward computation."""
        self.reward_stats = RunningStats(window_size=1000)
        
        self.reward_config = RewardConfig(
            target_weight=self.config.reward_weights.get('target', 1.0),
            parent_weight=self.config.reward_weights.get('parent', 0.0),
            info_gain_weight=self.config.reward_weights.get('info_gain', 0.0),
            optimization_direction=self.config.optimization_direction,
            reward_type=self.config.reward_type,
            stats=self.reward_stats
        )
        
        logger.info(f"Initialized rewards with type={self.config.reward_type}")
    
    def _run_grpo_episode(self, episode_idx: int, scm: Any, scm_name: str, key: jax.random.PRNGKey) -> Dict[str, Any]:
        """
        Run single GRPO training episode.
        
        This is the core method that trains the policy using GRPO.
        """
        # Get SCM info
        variables = list(get_variables(scm))
        target_var = get_target(scm)
        true_parents = list(get_parents(scm, target_var))
        
        logger.info(f"\nEpisode {episode_idx}: {scm_name}, Target: {target_var}")
        
        # Initialize buffer with observational data
        buffer = ExperienceBuffer()
        key, obs_key = random.split(key)
        
        obs_samples = sample_from_linear_scm(scm, self.config.obs_per_episode, seed=int(obs_key[0]))
        for sample in obs_samples:
            buffer.add_observation(sample)
        
        # Run multiple interventions per episode
        all_rewards = []
        
        for intervention_idx in range(self.config.max_interventions):
            # Collect GRPO batch
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
                
                # Convert buffer to tensor (use 4-channel for permutation invariant)
                if 'permutation_invariant' in self.config.policy_architecture:
                    tensor, mapper, _ = buffer_to_four_channel_tensor(
                        buffer, target_var, max_history_size=100, standardize=False
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
                if step == 0 and intervention_idx == 0:
                    prob_info = []
                    for i, var_name in enumerate(mapper.variables):
                        if i != mapper.target_idx:
                            prob_info.append(f"{var_name}:{var_probs[i]:.3f}")
                    logger.info(f"  Variable probabilities: {' '.join(prob_info)}")
                
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
                    reward_info = compute_composite_reward(
                        intervention=intervention,
                        outcome_sample=outcome_sample,
                        buffer=buffer,
                        scm=scm,
                        target_variable=target_var,
                        variables=variables,
                        config=self.reward_config,
                        tensor_5ch=tensor,
                        mapper=mapper,
                        reward_type=self.config.reward_type,
                        stats=self.reward_stats
                    )
                    reward = reward_info['total']
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
            
            # Log candidates (first intervention only)
            if intervention_idx == 0:
                self._log_candidates(grpo_batch_data, mapper, target_var, scm)
            
            # Create GRPO batch
            grpo_batch = self._create_grpo_batch(grpo_batch_data)
            
            # GRPO update
            self.policy_params, self.optimizer_state, grpo_metrics = self._grpo_update(
                self.policy_params, self.optimizer_state, grpo_batch
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
    
    def _grpo_update(self, params: Any, opt_state: Any, batch: Dict[str, Any]):
        """Perform GRPO policy update."""
        def loss_fn(p):
            return self._compute_grpo_loss(p, batch)
        
        (loss_value, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        
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
        
        return new_params, new_opt_state, grpo_metrics
    
    def _compute_grpo_loss(self, params: Any, batch: Dict[str, Any]):
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
            'approx_kl': jnp.mean((new_log_probs - old_log_probs) ** 2)
        }
        
        return total_loss, loss_info
    
    def _log_candidates(self, grpo_batch_data: Dict, mapper: Any, target_var: str, scm: Any):
        """Log candidate interventions and rewards."""
        print(f"\n[GRPO CANDIDATES]:")
        
        for i in range(len(grpo_batch_data['actions'])):
            action = grpo_batch_data['actions'][i]
            reward = grpo_batch_data['rewards'][i]
            var_name = mapper.get_name(int(action['variable']))
            
            if i < len(grpo_batch_data['intervention_details']):
                intervention_info = grpo_batch_data['intervention_details'][i]
                if 'samples' in intervention_info and intervention_info['samples']:
                    target_value = get_values(intervention_info['samples'][0]).get(target_var, 0.0)
                    print(f"  Candidate {i+1}: {var_name} = {action['value']:.3f} → TARGET = {target_value:.3f}, REWARD = {reward:.3f}")
        
        # Log advantages
        baseline = jnp.mean(grpo_batch_data['rewards'])
        advantages = grpo_batch_data['rewards'] - baseline
        
        print(f"\n📊 GRPO Advantages:")
        print(f"  Baseline: {baseline:.3f}")
        print(f"  Best advantage: {jnp.max(advantages):+.3f}")
        print(f"  Advantage std: {jnp.std(advantages):.3f}")
        
        if jnp.std(advantages) < 0.01:
            print(f"  ❌ WARNING: Advantages too uniform!")
    
    def save_checkpoint(self, episode: int, checkpoint_dir: Optional[str] = None):
        """Save policy checkpoint."""
        checkpoint_dir = Path(checkpoint_dir or self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"pure_grpo_ep{episode}.pkl"
        
        # Save policy parameters and config
        import pickle
        checkpoint_data = {
            'policy_params': self.policy_params,
            'config': self.config,
            'episode': episode,
            'training_step': self.training_step
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")