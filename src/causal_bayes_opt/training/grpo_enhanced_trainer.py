"""
Enhanced GRPO trainer with group-based advantages and ground truth rewards.

This module extends UnifiedGRPOTrainer to implement:
1. Group-based advantage calculation (true GRPO)
2. Multi-component rewards with ground truth parent information
3. Better credit assignment between variable and value selection
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

from .unified_grpo_trainer import UnifiedGRPOTrainer
from ..acquisition.grpo_rewards import (
    compute_grpo_reward, compute_group_advantages, 
    GRPORewardComponents, analyze_reward_distribution
)
from ..data_structures.sample import get_values, create_sample
from ..environments.sampling import sample_with_intervention
from ..interventions.handlers import create_perfect_intervention

logger = logging.getLogger(__name__)

# Global tracking for test scripts to access
ENHANCED_TARGET_VALUES = []


def compute_param_change(old_params, new_params):
    """Compute magnitude of parameter changes."""
    import jax.tree_util as tree
    
    # Flatten parameters
    old_flat, _ = tree.tree_flatten(old_params)
    new_flat, _ = tree.tree_flatten(new_params)
    
    # Compute total change
    total_change = 0.0
    total_magnitude = 0.0
    
    for old_p, new_p in zip(old_flat, new_flat):
        change = jnp.sum((new_p - old_p) ** 2)
        magnitude = jnp.sum(old_p ** 2)
        total_change += change
        total_magnitude += magnitude
    
    total_change = jnp.sqrt(total_change)
    total_magnitude = jnp.sqrt(total_magnitude)
    
    # Relative change
    relative_change = total_change / (total_magnitude + 1e-8)
    
    return {
        'total': float(total_change),
        'relative': float(relative_change),
        'magnitude': float(total_magnitude)
    }


class GRPOEnhancedTrainer(UnifiedGRPOTrainer):
    """
    Enhanced GRPO trainer with proper group-based advantages and rewards.
    
    Key improvements:
    1. Samples multiple interventions per state (group)
    2. Uses group mean as baseline (true GRPO)
    3. Incorporates ground truth parent information in rewards
    4. Better credit assignment for learning
    """
    
    def __init__(self, **kwargs):
        # Extract enhanced config options
        self.group_size = kwargs.pop('group_size', 4)  # Number of samples per state
        self.use_grpo_rewards = kwargs.pop('use_grpo_rewards', True)
        self.reward_config = kwargs.pop('grpo_reward_config', {
            'reward_weights': {
                'variable_selection': 0.5,
                'value_selection': 0.5,
                'parent_bonus': 0.3,
                'improvement_bonus': 0.2
            },
            'improvement_threshold': 0.1
        })
        
        # Initialize parent class
        super().__init__(**kwargs)
        
        # Override the grpo_update function to include more debugging info
        self._create_enhanced_grpo_update()
        
        # Track reward components for analysis
        self.reward_history = []
        self.gradient_history = []
        self.param_change_history = []
        self.component_reward_history = {
            'target_improvement': [],
            'parent_intervention': [],
            'value_optimization': [],
            'structure_discovery': [],
            'total': []
        }
    
    def _create_enhanced_grpo_update(self):
        """Create enhanced GRPO update function with more debugging info."""
        from ..acquisition.grpo import GRPOUpdate
        from types import SimpleNamespace
        import jax
        import optax
        
        def enhanced_grpo_update(params, opt_state, batch):
            """Enhanced GRPO update with extra debugging."""
            # Compute loss and gradients with our enhanced loss function
            def loss_fn(p):
                return self._compute_simple_grpo_loss(p, batch)
            
            (loss_value, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            
            # Apply updates
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            
            # Compute gradient norm
            grad_norm = optax.global_norm(grads)
            
            # Create enhanced metrics with all debugging info from loss_info
            enhanced_metrics = SimpleNamespace(
                # Standard fields
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
                approx_kl=loss_info['approx_kl'],
                # Enhanced debugging fields
                mean_ratio=loss_info.get('mean_ratio', 1.0),
                ratio_std=loss_info.get('ratio_std', 0.0),
                mean_log_prob_change=loss_info.get('mean_log_prob_change', 0.0),
                surr1_mean=loss_info.get('surr1_mean', 0.0),
                surr2_mean=loss_info.get('surr2_mean', 0.0),
                surr_min_mean=loss_info.get('surr_min_mean', 0.0),
                clip_fraction=loss_info.get('clip_fraction', 0.0),
                positive_advantages=loss_info.get('positive_advantages', 0),
                negative_advantages=loss_info.get('negative_advantages', 0),
                # Diagnostic fields
                loss_terms_sum=loss_info.get('loss_terms_sum', 0.0),
                loss_terms_mean=loss_info.get('loss_terms_mean', 0.0),
                log_prob_variance=loss_info.get('log_prob_variance', 0.0),
                unique_log_probs=loss_info.get('unique_log_probs', 0)
            )
            
            return new_params, new_opt_state, enhanced_metrics
        
        self.grpo_update = enhanced_grpo_update
        
    def _run_grpo_episode(self, episode_idx: int, scm: Any, scm_name: str, key: jax.random.PRNGKey) -> Dict[str, Any]:
        """
        Run episode with group-based sampling for true GRPO.
        
        Key difference: samples multiple interventions per state and uses
        group statistics for advantage calculation.
        """
        # Get SCM info
        from ..data_structures.scm import get_variables, get_target, get_parents
        variables = list(get_variables(scm))
        target_var = get_target(scm)
        target_idx = variables.index(target_var)
        true_parents = list(get_parents(scm, target_var)) if hasattr(scm, 'edges') else []
        
        # Log initial state for first episode or every 10th
        if episode_idx == 0 or episode_idx % 10 == 0:
            logger.info(
                f"\n[EPISODE STATE] Episode {episode_idx}:\n"
                f"  SCM: {scm_name} (target: {target_var})\n"
                f"  True parents of {target_var}: {true_parents}\n"
                f"  Group size: {self.group_size}\n"
                f"  Episode length: {self.episode_length}"
            )
        
        # Initialize buffer with observational data
        from ..data_structures.buffer import ExperienceBuffer
        buffer = ExperienceBuffer()
        key, obs_key = jax.random.split(key)
        
        # Sample observational data
        from ..mechanisms.linear import sample_from_linear_scm
        obs_samples = sample_from_linear_scm(scm, self.obs_per_episode, seed=int(obs_key[0]))
        for sample in obs_samples:
            buffer.add_observation(sample)
        
        # Log initial state for first episode
        if episode_idx == 0:
            logger.info(f"\n[INITIAL STATE] Starting enhanced GRPO training:")
            logger.info(f"  SCM: {scm_name} (target: {target_var})")
            logger.info(f"  True parents of {target_var}: {true_parents}")
            logger.info(f"  Group size: {self.group_size}")
            logger.info(f"  Episode length: {self.episode_length}")
        
        # Collect groups of interventions for GRPO
        grpo_groups = []  # List of groups, each group has multiple interventions
        
        # Generate multiple episodes worth of data
        for episode_step in range(self.episode_length):
            key, step_key = jax.random.split(key)
            
            # Get current state (tensor representation)
            from .five_channel_converter import buffer_to_five_channel_tensor
            if self.use_surrogate and self.surrogate_predict_fn is not None:
                def surrogate_wrapper(t, tgt, vars=None):
                    actual_vars = vars if vars is not None else variables
                    return self.surrogate_predict_fn(t, target_var, actual_vars)
                
                tensor, mapper, diagnostics = buffer_to_five_channel_tensor(
                    buffer, target_var, 
                    surrogate_fn=surrogate_wrapper,
                    max_history_size=100, 
                    standardize=True,
                    validate_signals=True
                )
            else:
                # No surrogate - convert 3-channel to 5-channel
                from .three_channel_converter import buffer_to_three_channel_tensor
                tensor_3ch, mapper = buffer_to_three_channel_tensor(
                    buffer, target_var, max_history_size=100, standardize=True
                )
                T, n_vars, _ = tensor_3ch.shape
                tensor = jnp.zeros((T, n_vars, 5))
                tensor = tensor.at[:, :, :3].set(tensor_3ch)
            
            # Sample a group of interventions for this state
            group_data = {
                'state': tensor,
                'mapper': mapper,
                'interventions': [],
                'outcomes': [],
                'rewards': [],
                'old_log_probs': [],
                'actions': []
            }
            
            # Generate group_size interventions for the same state
            for group_idx in range(self.group_size):
                key, sample_key = jax.random.split(key)
                
                # Get policy output
                policy_output = self.policy_fn.apply(
                    self.policy_params, sample_key, tensor, mapper.target_idx
                )
                
                # Sample intervention
                var_logits = policy_output['variable_logits']
                value_params = policy_output['value_params']
                
                # Sample variable
                key, var_key = jax.random.split(key)
                var_probs = jax.nn.softmax(var_logits)
                selected_var_idx = jax.random.categorical(var_key, var_logits)
                
                # Sample value
                key, val_key = jax.random.split(key)
                mean = value_params[selected_var_idx, 0]
                log_std = value_params[selected_var_idx, 1]
                std = jnp.exp(log_std)
                z = jax.random.normal(val_key)
                intervention_value = mean + std * z
                
                # Compute COMPLETE log probability: log P(var) + log P(value | var)
                log_prob_var = jnp.log(var_probs[selected_var_idx] + 1e-8)
                # Log probability of the sampled value under the Gaussian distribution
                log_prob_value = -0.5 * ((intervention_value - mean) / std) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi)
                log_prob = log_prob_var + log_prob_value
                
                # Create and apply intervention
                selected_var = mapper.get_name(int(selected_var_idx))
                intervention = create_perfect_intervention(
                    targets=frozenset([selected_var]),
                    values={selected_var: float(intervention_value)}
                )
                
                # Log intervention decision every 10 episodes
                if episode_idx % 10 == 0 and episode_step == 0 and group_idx == 0:
                    # Calculate entropy of variable selection
                    var_entropy = -jnp.sum(var_probs * jnp.log(var_probs + 1e-8))
                    max_entropy = jnp.log(len(var_probs))  # Maximum entropy (uniform)
                    
                    logger.info(f"\n[INTERVENTION DECISION] Episode {episode_idx}:")
                    logger.info(f"  Variable logits: {var_logits}")
                    logger.info(f"  Variable probs: {var_probs}")
                    logger.info(f"  Variable entropy: {var_entropy:.3f} (max/uniform: {max_entropy:.3f})")
                    logger.info(f"  Selected: {selected_var} (idx={selected_var_idx})")
                    logger.info(f"  Value: {intervention_value:.3f}")
                    logger.info(f"  Value params - mean: {mean:.3f}, std: {jnp.exp(log_std):.3f}")
                    logger.info(f"  True parents of {target_var}: {true_parents}")
                    
                # Store var probs for first sample in group for diagnostic
                if group_idx == 0:
                    group_data['first_var_probs'] = var_probs
                    group_data['first_var_entropy'] = -jnp.sum(var_probs * jnp.log(var_probs + 1e-8))
                
                # Sample outcome
                key, outcome_key = jax.random.split(key)
                intervention_samples = sample_with_intervention(
                    scm, intervention, n_samples=1, seed=int(outcome_key[0])
                )
                outcome_sample = intervention_samples[0] if intervention_samples else None
                
                # Store group data
                group_data['interventions'].append(intervention)
                group_data['outcomes'].append(outcome_sample)
                group_data['old_log_probs'].append(float(log_prob))
                group_data['actions'].append({
                    'variable': int(selected_var_idx),
                    'value': float(intervention_value)
                })
            
            # Compute rewards for the entire group
            if self.use_grpo_rewards:
                # Use new reward system with ground truth
                reward_components = []
                for intervention, outcome in zip(group_data['interventions'], 
                                                group_data['outcomes']):
                    # Log target values for tracking
                    outcome_values = get_values(outcome)
                    if target_var in outcome_values:
                        target_value = float(outcome_values[target_var])
                        # Store in module-level tracking
                        ENHANCED_TARGET_VALUES.append({
                            'value': target_value,
                            'target_var': target_var,
                            'intervention': intervention
                        })
                        if episode_idx % 20 == 0 and group_idx == 0:
                            logger.info(f"[ENHANCED] Target {target_var} = {target_value:.3f} after intervention on {intervention.get('targets', 'unknown')}")
                    
                    # Get surrogate posteriors if available (for structure discovery)
                    surrogate_posterior_before = None
                    surrogate_posterior_after = None
                    
                    # TODO: If surrogate model provides posteriors, extract them here
                    # For now, we pass None but the infrastructure is ready
                    
                    reward_comp = compute_grpo_reward(
                        scm=scm,
                        intervention=intervention,
                        outcome=outcome,
                        target_variable=target_var,
                        buffer_before=buffer,
                        config={
                            **self.reward_config,
                            'optimization_direction': self.optimization_direction
                        },
                        group_outcomes=list(zip(group_data['interventions'], 
                                               group_data['outcomes'])),
                        reward_type=self.reward_config.get('reward_type', 'squared'),  # Default to squared
                        surrogate_posterior_before=surrogate_posterior_before,
                        surrogate_posterior_after=surrogate_posterior_after
                    )
                    reward_components.append(reward_comp)
                    self.reward_history.append(reward_comp)
                
                # Extract total rewards
                group_rewards = [rc.total_reward for rc in reward_components]
                
                # Track component-wise rewards
                for rc in reward_components:
                    self.component_reward_history['target_improvement'].append(rc.target_improvement)
                    self.component_reward_history['parent_intervention'].append(rc.parent_intervention)
                    self.component_reward_history['value_optimization'].append(rc.value_optimization)
                    self.component_reward_history['structure_discovery'].append(rc.structure_discovery)
                    self.component_reward_history['total'].append(rc.total_reward)
                
                # Log reward decomposition with more detail
                if episode_idx % 10 == 0 and episode_step == 0:
                    # Analyze all rewards in the group
                    parent_rewards = [rc.parent_intervention for rc in reward_components]
                    improvement_rewards = [rc.target_improvement for rc in reward_components]
                    value_rewards = [rc.value_optimization for rc in reward_components]
                    total_rewards = [rc.total_reward for rc in reward_components]
                    
                    logger.info(
                        f"\n[REWARD DECOMPOSITION] Episode {episode_idx}, Group analysis:\n"
                        f"  Parent intervention: min={min(parent_rewards):.3f}, max={max(parent_rewards):.3f}, mean={np.mean(parent_rewards):.3f}\n"
                        f"  Target improvement: min={min(improvement_rewards):.3f}, max={max(improvement_rewards):.3f}, mean={np.mean(improvement_rewards):.3f}\n"
                        f"  Value optimization: min={min(value_rewards):.3f}, max={max(value_rewards):.3f}, mean={np.mean(value_rewards):.3f}\n"
                        f"  Total rewards: min={min(total_rewards):.3f}, max={max(total_rewards):.3f}, mean={np.mean(total_rewards):.3f}\n"
                        f"  Parent selection: {sum(rc.correct_parent for rc in reward_components)}/{len(reward_components)}\n"
                        f"  Improvements: {sum(rc.improved_beyond_threshold for rc in reward_components)}/{len(reward_components)}"
                    )
            else:
                # Use original reward system
                group_rewards = []
                for intervention, outcome in zip(group_data['interventions'], 
                                                group_data['outcomes']):
                    from ..acquisition.better_rewards import compute_better_clean_reward
                    reward_info = compute_better_clean_reward(
                        buffer_before=buffer,
                        intervention=intervention,
                        outcome=outcome,
                        target_variable=target_var,
                        config={
                            'optimization_direction': self.optimization_direction,
                            'reward_type': 'adaptive_sigmoid',
                            'temperature_factor': 2.0
                        },
                        stats=self.reward_stats
                    )
                    group_rewards.append(reward_info['total'])
            
            group_data['rewards'] = group_rewards
            
            # Compute group advantages (true GRPO)
            advantages = compute_group_advantages(group_rewards, method='zscore')
            group_data['advantages'] = advantages
            
            # Log group statistics with more detail
            if episode_step % 5 == 0 or (episode_idx < 5 and episode_step == 0):
                # Get reward range info
                rewards_array = jnp.array(group_rewards)
                reward_min = float(jnp.min(rewards_array))
                reward_max = float(jnp.max(rewards_array))
                reward_range = reward_max - reward_min
                
                # Diagnostic analysis
                unique_rewards = set(group_rewards)
                reward_variance = float(jnp.var(rewards_array))
                
                # Check for symmetric pattern in advantages
                sorted_advantages = sorted(advantages)
                is_symmetric = False
                if len(advantages) == 4:
                    # Check if advantages come in pairs that sum to ~0
                    is_symmetric = (abs(sorted_advantages[0] + sorted_advantages[3]) < 0.01 and 
                                  abs(sorted_advantages[1] + sorted_advantages[2]) < 0.01)
                
                logger.info(
                    f"\n[GROUP STATS] Episode {episode_idx}, Step {episode_step}:\n"
                    f"  Group rewards: {[f'{r:.3f}' for r in group_rewards]}\n"
                    f"  Reward range: [{reward_min:.3f}, {reward_max:.3f}] (range={reward_range:.3f})\n"
                    f"  Group mean: {jnp.mean(rewards_array):.3f}\n"
                    f"  Group std: {jnp.std(rewards_array):.3f}\n"
                    f"  Advantages: {[f'{a:.3f}' for a in advantages]}\n"
                    f"  Positive advantages: {sum(1 for a in advantages if a > 0)}/{len(advantages)}"
                )
                
                # Add policy entropy info if available
                entropy_info = ""
                if 'first_var_entropy' in group_data:
                    max_entropy = jnp.log(len(group_data['first_var_probs']))
                    entropy_ratio = group_data['first_var_entropy'] / max_entropy
                    entropy_info = f"\n  Variable selection entropy: {group_data['first_var_entropy']:.3f} (max: {max_entropy:.3f}, ratio: {entropy_ratio:.3f})"
                
                logger.info(
                    f"\n[DIAGNOSTIC] Reward/Advantage Analysis:\n"
                    f"  Unique rewards: {len(unique_rewards)} values: {sorted(list(unique_rewards))}\n"
                    f"  Reward variance: {reward_variance:.6f}\n"
                    f"  Advantages symmetric: {is_symmetric}\n"
                    f"  Old log probs: {[f'{lp:.3f}' for lp in group_data['old_log_probs']]}\n"
                    f"  Log prob std: {jnp.std(jnp.array(group_data['old_log_probs'])):.6f}"
                    f"{entropy_info}"
                )
                
                # Log which interventions got positive advantages
                if self.use_grpo_rewards and reward_components:
                    for i, (adv, rc) in enumerate(zip(advantages, reward_components)):
                        if adv > 0:
                            logger.debug(
                                f"  Positive advantage {i}: parent={rc.correct_parent}, "
                                f"improved={rc.improved_beyond_threshold}, "
                                f"reward={rc.total_reward:.3f}"
                            )
            
            # Add best outcome from group to buffer for next iteration
            best_idx = jnp.argmax(jnp.array(group_rewards))
            best_outcome = group_data['outcomes'][best_idx]
            best_intervention = group_data['interventions'][best_idx]
            buffer.add_intervention(best_intervention, best_outcome)
            
            # Store group for training
            grpo_groups.append(group_data)
        
        # Create training batch from all groups
        grpo_batch = self._create_grpo_batch_from_groups(grpo_groups)
        
        # Log batch statistics before update
        # Log detailed info for first 20 episodes, then every 5
        if episode_idx < 20 or episode_idx % 5 == 0:
            batch_advantages = grpo_batch['advantages']
            batch_rewards = grpo_batch['rewards']
            logger.info(
                f"\n[BATCH STATISTICS] Episode {episode_idx}:\n"
                f"  Batch size: {len(batch_advantages)}\n"
                f"  Advantages - mean: {jnp.mean(batch_advantages):.6f}, std: {jnp.std(batch_advantages):.6f}\n"
                f"  Rewards - mean: {jnp.mean(batch_rewards):.3f}, std: {jnp.std(batch_rewards):.3f}\n"
                f"  Non-zero advantages: {jnp.sum(jnp.abs(batch_advantages) > 1e-6)}/{len(batch_advantages)}"
            )
        
        # Store old params for tracking changes
        old_params = self.policy_params
        
        # Add detailed logging info to batch for debugging
        grpo_batch['debug_info'] = {
            'episode': episode_idx,
            'rewards_range': [float(jnp.min(grpo_batch['rewards'])), float(jnp.max(grpo_batch['rewards']))],
            'advantages_range': [float(jnp.min(grpo_batch['advantages'])), float(jnp.max(grpo_batch['advantages']))],
            'old_log_probs_range': [float(jnp.min(grpo_batch['old_log_probs'])), float(jnp.max(grpo_batch['old_log_probs']))]
        }
        
        # Run multiple epochs on the same batch (GRPO/PPO style)
        ppo_epochs = self.grpo_config.ppo_epochs if hasattr(self.grpo_config, 'ppo_epochs') else 4
        
        for epoch in range(ppo_epochs):
            # Update policy with GRPO using parent's update function
            self.policy_params, self.optimizer_state, grpo_metrics = self.grpo_update(
                self.policy_params,
                self.optimizer_state,
                grpo_batch
            )
            
            # Log first epoch and last epoch
            if epoch == 0 or epoch == ppo_epochs - 1:
                ratio_str = f"{grpo_metrics.mean_ratio:.3f}" if hasattr(grpo_metrics, 'mean_ratio') else "N/A"
                logger.info(f"\n[EPOCH {epoch}] Episode {episode_idx}: "
                          f"loss={grpo_metrics.policy_loss:.6f}, "
                          f"ratio={ratio_str}")
        
        # Log gradient and loss information after every update for debugging
        # Log detailed info for first 20 episodes, then every 5
        if episode_idx < 20 or episode_idx % 5 == 0:
            logger.info(
                f"\n[GRPO UPDATE] Episode {episode_idx}:\n"
                f"  Policy loss: {grpo_metrics.policy_loss:.6f}\n"
                f"  Entropy loss: {grpo_metrics.entropy_loss:.6f}\n"
                f"  Total loss: {grpo_metrics.total_loss:.6f}\n"
                f"  Gradient norm: {grpo_metrics.grad_norm:.6f}\n"
                f"  Learning rate: {self.learning_rate:.6f}\n"
                f"  Effective update: {grpo_metrics.grad_norm * self.learning_rate:.6f}\n"
                f"  Mean reward: {grpo_metrics.mean_reward:.3f}\n"
                f"  Reward std: {grpo_metrics.reward_std:.3f}\n"
                f"  Mean advantage: {grpo_metrics.mean_advantage:.3f}\n"
                f"  Advantage std: {grpo_metrics.advantage_std:.3f}\n"
                f"  Mean entropy: {grpo_metrics.mean_entropy:.3f}\n"
                f"  Approx KL: {grpo_metrics.approx_kl:.6f}"
            )
            
            # Additional debugging info if available
            if hasattr(grpo_metrics, 'mean_ratio'):
                logger.info(
                    f"\n[LOSS COMPONENTS] Episode {episode_idx}:\n"
                    f"  Mean ratio: {grpo_metrics.mean_ratio:.6f} (std: {grpo_metrics.ratio_std:.6f})\n"
                    f"  Mean log prob change: {grpo_metrics.mean_log_prob_change:.6f}\n"
                    f"  Surr1 mean: {grpo_metrics.surr1_mean:.6f}\n"
                    f"  Surr2 mean: {grpo_metrics.surr2_mean:.6f}\n"
                    f"  Surr min mean: {grpo_metrics.surr_min_mean:.6f}\n"
                    f"  Clip fraction: {grpo_metrics.clip_fraction:.3f}\n"
                    f"  Positive advantages: {grpo_metrics.positive_advantages}/{len(grpo_batch['advantages'])}\n"
                    f"  Negative advantages: {grpo_metrics.negative_advantages}/{len(grpo_batch['advantages'])}"
                )
                
                # Additional diagnostic info
                if hasattr(grpo_metrics, 'loss_terms_sum'):
                    logger.info(
                        f"\n[LOSS DIAGNOSTIC] Episode {episode_idx}:\n"
                        f"  Loss terms sum: {grpo_metrics.loss_terms_sum:.6f}\n"
                        f"  Loss terms mean: {grpo_metrics.loss_terms_mean:.6f}\n"
                        f"  Log prob variance: {grpo_metrics.log_prob_variance:.6f}\n"
                        f"  Unique log probs: {grpo_metrics.unique_log_probs}"
                    )
        
        # Track gradient norm and parameter changes
        if hasattr(grpo_metrics, 'grad_norm'):
            self.gradient_history.append({
                'episode': episode_idx,
                'grad_norm': float(grpo_metrics.grad_norm),
                'learning_rate': self.learning_rate,
                'effective_update': float(grpo_metrics.grad_norm * self.learning_rate)
            })
            
            # Compute parameter change magnitude
            param_change = compute_param_change(old_params, self.policy_params)
            self.param_change_history.append({
                'episode': episode_idx,
                'total_change': param_change['total'],
                'relative_change': param_change['relative'],
                'per_layer': param_change.get('per_layer', {})
            })
            
            # Log every 10 episodes
            if episode_idx % 10 == 0:
                logger.info(
                    f"\n[GRADIENT ANALYSIS] Episode {episode_idx}:\n"
                    f"  Gradient norm: {grpo_metrics.grad_norm:.6f}\n"
                    f"  Learning rate: {self.learning_rate:.6f}\n"
                    f"  Effective update: {grpo_metrics.grad_norm * self.learning_rate:.6f}\n"
                    f"  Parameter change: {param_change['total']:.6f} ({param_change['relative']:.2%})"
                )
        
        self.training_step += 1
        
        # Compute episode metrics
        all_rewards = []
        all_intervention_values = []
        parent_intervention_values = []
        
        for group in grpo_groups:
            all_rewards.extend(group['rewards'])
            # Collect intervention values for analysis
            for action, intervention in zip(group['actions'], group['interventions']):
                value = action['value']
                all_intervention_values.append(value)
                # Check if intervention was on parent
                targets = intervention.get('targets', set())
                if targets & set(true_parents):
                    parent_intervention_values.append(value)
        
        # Log value distribution analysis every 20 episodes
        if episode_idx % 20 == 0 and parent_intervention_values:
            logger.info(
                f"\n[VALUE ANALYSIS] Episode {episode_idx}:\n"
                f"  All intervention values - mean: {np.mean(all_intervention_values):.3f}, "
                f"std: {np.std(all_intervention_values):.3f}\n"
                f"  Parent intervention values ({len(parent_intervention_values)} samples):\n"
                f"    Mean: {np.mean(parent_intervention_values):.3f}\n"
                f"    Std: {np.std(parent_intervention_values):.3f}\n"
                f"    Min: {np.min(parent_intervention_values):.3f}\n"
                f"    Max: {np.max(parent_intervention_values):.3f}\n"
                f"    Negative values: {sum(1 for v in parent_intervention_values if v < 0)}/{len(parent_intervention_values)} "
                f"({100 * sum(1 for v in parent_intervention_values if v < 0) / len(parent_intervention_values):.1f}%)"
            )
        
        # Analyze reward distribution if using new rewards
        if self.use_grpo_rewards and len(self.reward_history) > 100:
            recent_history = self.reward_history[-100:]
            distribution = analyze_reward_distribution(recent_history)
            logger.info(
                f"\n[REWARD ANALYSIS] Recent 100 samples:\n"
                f"  Parent selection rate: {distribution['binary_signals']['parent_selection_rate']:.3f}\n"
                f"  Improvement rate: {distribution['binary_signals']['improvement_rate']:.3f}\n"
                f"  Target improvement mean: {distribution['target_improvement']['mean']:.3f}\n"
                f"  Value optimization mean: {distribution['value_optimization']['mean']:.3f}"
            )
            
        
        metrics = {
            'episode': episode_idx,
            'scm_type': scm_name,
            'mean_reward': float(jnp.mean(jnp.array(all_rewards))),
            'reward_std': float(jnp.std(jnp.array(all_rewards))),
            'max_reward': float(jnp.max(jnp.array(all_rewards))),
            'min_reward': float(jnp.min(jnp.array(all_rewards))),
            'n_groups': len(grpo_groups),
            'group_size': self.group_size
        }
        
        return metrics
    
    def _create_grpo_batch_from_groups(self, groups: List[Dict]) -> Dict[str, Any]:
        """Convert groups of interventions into GRPO training batch."""
        batch_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'advantages': [],
            'old_log_probs': []
        }
        
        for group in groups:
            # Each intervention in the group shares the same state
            state = group['state']
            for i in range(len(group['interventions'])):
                batch_data['states'].append(state)
                batch_data['actions'].append(group['actions'][i])
                batch_data['rewards'].append(group['rewards'][i])
                batch_data['advantages'].append(group['advantages'][i])
                batch_data['old_log_probs'].append(group['old_log_probs'][i])
        
        # Convert to arrays
        return {
            'states': jnp.stack(batch_data['states']),
            'actions': batch_data['actions'],
            'rewards': jnp.array(batch_data['rewards']),
            'advantages': jnp.array(batch_data['advantages']),
            'old_log_probs': jnp.array(batch_data['old_log_probs'])
        }
    
    def _compute_simple_grpo_loss(self, params: Any, batch: Dict[str, Any]) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Compute GRPO loss with proper advantages.
        
        Key difference: uses pre-computed group advantages instead of
        simple baseline subtraction.
        """
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        advantages = batch['advantages']  # Pre-computed group advantages
        old_log_probs = batch['old_log_probs']
        
        # Forward pass to get current log probs
        batch_size = states.shape[0]
        new_log_probs = []
        entropy_values = []
        
        # Track individual log probs for debugging
        log_prob_changes = []
        
        for i in range(batch_size):
            # Get policy output for this state
            self.rng_key, policy_key = jax.random.split(self.rng_key)
            
            # Use the target index from the batch
            target_idx = batch.get('target_idx', states[i].shape[1] - 1)
            
            policy_output = self.policy_fn.apply(
                params, policy_key, states[i], target_idx
            )
            
            # Compute COMPLETE log prob for selected action
            var_probs = jax.nn.softmax(policy_output['variable_logits'])
            selected_var = actions[i]['variable']
            log_prob_var = jnp.log(var_probs[selected_var] + 1e-8)
            
            # Get value distribution parameters
            value_params = policy_output['value_params']
            mean = value_params[selected_var, 0]
            log_std = value_params[selected_var, 1]
            std = jnp.exp(log_std)
            
            # Compute log prob of the actual value under the Gaussian
            actual_value = actions[i]['value']
            log_prob_value = -0.5 * ((actual_value - mean) / std) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi)
            
            # Complete log probability
            log_prob = log_prob_var + log_prob_value
            new_log_probs.append(log_prob)
            
            # Track change from old log prob
            log_prob_changes.append(log_prob - old_log_probs[i])
            
            # Compute entropy
            entropy = -jnp.sum(var_probs * jnp.log(var_probs + 1e-8))
            entropy_values.append(entropy)
        
        new_log_probs = jnp.array(new_log_probs)
        old_log_probs = jnp.array(old_log_probs)
        entropy_values = jnp.array(entropy_values)
        
        # Compute ratio for PPO-style clipping
        log_ratio = new_log_probs - old_log_probs
        ratio = jnp.exp(log_ratio)
        
        # Debug log ratios - removed since we can't log inside JIT
        
        # Clipped surrogate objective with pre-computed advantages
        surr1 = ratio * advantages
        surr2 = jnp.clip(ratio, 1.0 - self.grpo_config.clip_ratio, 
                        1.0 + self.grpo_config.clip_ratio) * advantages
        
        # Track which surrogate is used
        surr_min = jnp.minimum(surr1, surr2)
        
        # Track individual loss terms for diagnostic
        loss_terms = -surr_min  # Individual contributions to loss
        
        policy_loss = -jnp.mean(surr_min)
        
        # Track how many samples are clipped
        clipped_mask = jnp.abs(ratio - 1.0) > self.grpo_config.clip_ratio
        clip_fraction = jnp.mean(clipped_mask)
        
        # Debug computations removed - can't log inside JIT
        
        # Entropy loss
        entropy_loss = -self.grpo_config.entropy_coeff * jnp.mean(entropy_values)
        
        # Total loss
        total_loss = policy_loss + entropy_loss
        
        # Compute diagnostics
        approx_kl = jnp.mean((new_log_probs - old_log_probs) ** 2)
        
        loss_info = {
            'policy_loss': policy_loss,
            'entropy_loss': entropy_loss,
            'kl_penalty': 0.0,  # Not using KL penalty in enhanced version
            'group_baseline': jnp.mean(rewards),  # Using group mean as baseline
            'mean_reward': jnp.mean(rewards),
            'reward_std': jnp.std(rewards),
            'mean_advantage': jnp.mean(advantages),
            'advantage_std': jnp.std(advantages),
            'mean_entropy': jnp.mean(entropy_values),
            'approx_kl': approx_kl,
            'clip_fraction': clip_fraction,
            # Additional debugging info
            'mean_ratio': jnp.mean(ratio),
            'ratio_std': jnp.std(ratio),
            'mean_log_prob_change': jnp.mean(jnp.array(log_prob_changes)),
            'surr1_mean': jnp.mean(surr1),
            'surr2_mean': jnp.mean(surr2),
            'surr_min_mean': jnp.mean(surr_min),
            'positive_advantages': jnp.sum(advantages > 0),
            'negative_advantages': jnp.sum(advantages < 0),
            # Diagnostic info
            'loss_terms_sum': jnp.sum(loss_terms),
            'loss_terms_mean': jnp.mean(loss_terms),
            'log_prob_variance': jnp.var(new_log_probs),
            'unique_log_probs': jnp.unique(new_log_probs).shape[0]
        }
        
        return total_loss, loss_info