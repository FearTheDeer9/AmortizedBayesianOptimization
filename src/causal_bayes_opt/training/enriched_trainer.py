"""
Enriched GRPO Trainer - Main Training Orchestrator

This module provides the main training coordinator that uses modular components
following CLAUDE.md single-responsibility principles.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import replace

import jax
import jax.numpy as jnp
import jax.random as random
import optax
import pyrsistent as pyr
from omegaconf import DictConfig

from .modular_trainer import (
    TrainingMetrics, PolicyFactory, SCMRotationManager, 
    StateConverter, CheckpointManager, MetricsCollector
)
from .grpo_core import (
    GRPOConfig, create_grpo_update_fn, create_default_grpo_config,
    create_trajectory_from_experiences
)
from ..acquisition.rewards import create_default_reward_config
from ..data_structures.scm import get_variables, get_target

logger = logging.getLogger(__name__)


class EnrichedGRPOTrainer:
    """Main trainer that orchestrates modular components."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        
        # Initialize modular components
        self.scm_manager = SCMRotationManager(config)
        self.state_converter = StateConverter(config, self.scm_manager.max_variables)
        self.checkpoint_manager = CheckpointManager(config)
        self.metrics_collector = MetricsCollector()
        
        # Create policy
        policy_factory = PolicyFactory(config, self.scm_manager.max_variables)
        self.policy_fn, self.policy_config = policy_factory.create_policy()
        
        # Initialize policy parameters with proper key threading
        self._initial_key = random.PRNGKey(config.seed)
        init_key, trainer_key = random.split(self._initial_key)
        
        dummy_input = self._create_dummy_input()
        self.policy_params = self.policy_fn.init(init_key, dummy_input)
        self._current_key = trainer_key
        
        # Create GRPO configuration and optimizers - FIXED learning rates
        self.grpo_config = create_default_grpo_config()
        
        # CRITICAL FIX: Use higher learning rates for policy learning
        policy_lr = max(config.training.learning_rate, 1e-3)  # Ensure minimum 1e-3
        value_lr = max(config.training.learning_rate, 1e-3)   # Ensure minimum 1e-3
        
        self.grpo_config = replace(
            self.grpo_config,
            learning_rate=policy_lr,
            value_learning_rate=value_lr,
            # CRITICAL: Ensure advantages are normalized for better learning
            normalize_advantages=True,
            # CRITICAL: Use higher entropy for better exploration
            entropy_coefficient=0.02,
            # CRITICAL: Use more conservative clipping for stability
            clip_ratio=0.2
        )
        
        logger.info(f"GRPO Config: policy_lr={policy_lr:.6f}, value_lr={value_lr:.6f}, entropy={self.grpo_config.entropy_coefficient:.3f}")
        logger.info(f"GRPO Config: normalize_advantages={self.grpo_config.normalize_advantages}, clip_ratio={self.grpo_config.clip_ratio:.2f}")
        
        # Separate optimizers for policy and value networks
        self.policy_optimizer = optax.adam(learning_rate=self.grpo_config.learning_rate)
        self.value_optimizer = optax.adam(learning_rate=self.grpo_config.value_learning_rate)
        
        self.policy_optimizer_state = self.policy_optimizer.init(self.policy_params)
        self.value_optimizer_state = self.value_optimizer.init(self.policy_params)
        
        # Create GRPO policy and value function wrappers
        self.policy_wrapper, self.value_wrapper = self._create_grpo_function_wrappers()
        
        # Create GRPO update function
        self.grpo_update_fn = create_grpo_update_fn(
            self.policy_wrapper, self.value_wrapper,
            self.policy_optimizer, self.value_optimizer,
            self.grpo_config
        )
        
        # Create reward configuration
        self.reward_config = create_default_reward_config(
            optimization_weight=config.training.reward_weights.optimization,
            structure_weight=config.training.reward_weights.discovery,
            exploration_weight=config.training.reward_weights.efficiency
        )
        
        logger.info(f"Initialized trainer with {self.scm_manager.max_variables} max variables")
    
    def _create_dummy_input(self) -> jnp.ndarray:
        """Create dummy input for policy initialization."""
        max_history_size = self.config.training.state_config.get('max_history_size', 100)
        num_channels = self.config.training.state_config.get('num_channels', 10)
        
        return jnp.zeros((max_history_size, self.scm_manager.max_variables, num_channels))
    
    def _create_grpo_function_wrappers(self) -> Tuple[Any, Any]:
        """Create policy and value function wrappers for GRPO compatibility.
        
        Returns:
            Tuple of (policy_wrapper, value_wrapper) functions that match GRPO expected signatures
        """
        
        def policy_wrapper(params, states, actions):
            """Wrapper that extracts log probabilities from enriched policy.
            
            Expected GRPO signature: policy_fn(params, states, actions) -> log_probs
            """
            batch_size = states.shape[0]
            log_probs = []
            
            for i in range(batch_size):
                state = states[i]  # [T, vars, channels]
                action = actions[i]  # [vars]
                
                # Get policy output (requires random key and target_idx)
                key = random.PRNGKey(i)  # Deterministic key for consistency
                target_idx = 0  # Default target for log prob computation
                
                policy_output = self.policy_fn.apply(
                    params, key, state, target_idx, True
                )
                
                # Extract value parameters for log probability computation
                value_params = policy_output.get('value_params', jnp.zeros((len(action), 2)))
                means = value_params[:, 0]
                log_stds = value_params[:, 1]
                stds = jnp.exp(log_stds) + 1e-8  # Add small epsilon for numerical stability
                
                # Debug logging for policy wrapper issues
                if i == 0:  # Log first sample for debugging
                    logger.debug(f"Policy wrapper debug - means: {means[:3]}, log_stds: {log_stds[:3]}, action: {action[:3]}")
                
                # Compute log probability of taken action under current policy
                # Assuming Gaussian policy: log P(a) = -0.5 * ((a - μ) / σ)^2 - log(σ√(2π))
                normalized_action = (action - means) / stds
                log_prob = -0.5 * jnp.sum(normalized_action ** 2) - jnp.sum(log_stds) - 0.5 * len(action) * jnp.log(2 * jnp.pi)
                log_probs.append(log_prob)
            
            return jnp.array(log_probs)
        
        def value_wrapper(params, states):
            """Wrapper that extracts state values from enriched policy.
            
            Expected GRPO signature: value_fn(params, states) -> values
            """
            batch_size = states.shape[0]
            values = []
            
            for i in range(batch_size):
                state = states[i]  # [T, vars, channels]
                
                # Get policy output (requires random key and target_idx)
                key = random.PRNGKey(i)  # Deterministic key for consistency
                target_idx = 0  # Default target for value computation
                
                policy_output = self.policy_fn.apply(
                    params, key, state, target_idx, True
                )
                
                # Extract state value
                value = policy_output.get('state_value', 0.0)
                values.append(value)
            
            return jnp.array(values)
        
        return policy_wrapper, value_wrapper
    
    def train(self) -> Dict[str, Any]:
        """Run complete training process."""
        logger.info("Starting enriched GRPO training")
        start_time = time.time()
        
        # Setup WandB if enabled
        if self.config.logging.wandb.enabled:
            self._setup_wandb()
        
        # Training loop with proper key threading
        training_key = self._current_key
        
        for episode in range(self.config.training.n_episodes):
            # Thread keys properly through episodes
            episode_key, training_key = random.split(training_key)
            
            try:
                # Run episode
                metrics = self._run_episode(episode, episode_key)
                
                # Update metrics collector (immutable)
                self.metrics_collector = self.metrics_collector.add_metrics(metrics)
                
                # Log progress
                if episode % 10 == 0:
                    logger.info(
                        f"Episode {episode}: reward={metrics.mean_reward:.3f}, "
                        f"intervention_rate={metrics.structure_accuracy:.3f}, "
                        f"scm={metrics.scm_type}"
                    )
                
                # Save periodic checkpoints
                if episode % 50 == 0 and episode > 0:
                    self.checkpoint_manager.save_checkpoint(
                        self.policy_params, self.policy_config, episode, metrics
                    )
                
            except Exception as e:
                logger.error(f"Episode {episode} failed: {e}")
                raise
        
        # Save final checkpoint
        final_metrics = self.metrics_collector.get_latest_metrics()
        final_checkpoint = self.checkpoint_manager.save_checkpoint(
            self.policy_params, self.policy_config, 
            self.config.training.n_episodes, final_metrics, is_final=True
        )
        
        # Analyze performance
        total_time = time.time() - start_time
        performance_analysis = self.metrics_collector.analyze_performance(total_time)
        
        logger.info(f"Training completed in {total_time:.1f}s")
        logger.info(f"Final checkpoint: {final_checkpoint}")
        
        return {
            'checkpoint_path': str(final_checkpoint),
            'performance': performance_analysis,
            'policy_config': self.policy_config
        }
    
    def _run_episode(self, episode_idx: int, episode_key: jax.random.PRNGKey) -> TrainingMetrics:
        """Run a single training episode."""
        # Get current SCM
        scm_name, scm = self.scm_manager.get_current_scm(episode_idx)
        variables = list(get_variables(scm))
        target = get_target(scm)
        
        # Collect trajectory
        trajectory = []
        episode_rewards = []
        
        # Thread keys through episode steps
        step_key = episode_key
        
        for step in range(self.config.training.episode_length):
            # Split key for this step
            step_key, action_key = random.split(step_key)
            
            # Create mock state for this step
            state = self._create_mock_state(scm, step, 0.0)
            
            # Convert to enriched input
            enriched_input = self.state_converter.convert_state_to_enriched_input(state)
            
            # Get target variable index
            target_idx = variables.index(target) if target in variables else 0
            
            # Policy forward pass
            policy_output = self.policy_fn.apply(
                self.policy_params, action_key, enriched_input, target_idx, True
            )
            
            # Convert policy output to action (with training episode for exploration)
            action = self._policy_output_to_action(policy_output, variables, target, episode_idx)
            
            # Simulate intervention
            intervention, reward = self._simulate_intervention(scm, action)
            
            # Store step data
            trajectory.append({
                'state': enriched_input,
                'action': action,
                'reward': reward,
                'policy_output': policy_output,
                'target_idx': target_idx,
                'intervention': intervention  # Store intervention for metrics
            })
            episode_rewards.append(reward)
        
        # Update policy
        policy_losses = self._update_policy(trajectory)
        
        # Compute meaningful intervention metrics instead of structure accuracy
        total_interventions = sum(1 for step in trajectory if len(step.get('intervention', {}).get('targets', set())) > 0)
        intervention_rate = total_interventions / len(trajectory) if trajectory else 0.0
        
        # Track action magnitudes for policy learning assessment
        mean_action_magnitude = policy_losses.get('mean_action_magnitude', 0.0)
        
        # Create metrics with meaningful intervention-focused metrics
        metrics = TrainingMetrics(
            episode=episode_idx,
            mean_reward=float(jnp.mean(jnp.array(episode_rewards))),
            structure_accuracy=intervention_rate,  # Repurposed: now represents intervention success rate
            optimization_improvement=float(jnp.mean(jnp.array(episode_rewards)) - episode_rewards[0]),
            policy_loss=policy_losses['policy_loss'],
            value_loss=policy_losses['value_loss'],
            scm_type=scm_name
        )
        
        return metrics
    
    def _create_mock_state(self, scm: pyr.PMap, step: int, best_value: float) -> Any:
        """Create mock acquisition state for training."""
        # Create mock buffer and state
        # This is a simplified version for training
        from unittest.mock import Mock
        
        mock_buffer = Mock()
        mock_buffer.get_all_samples.return_value = []
        mock_buffer.get_variable_coverage.return_value = list(get_variables(scm))
        
        mock_state = Mock()
        mock_state.buffer = mock_buffer
        mock_state.current_target = get_target(scm)
        mock_state.step = step
        mock_state.best_value = best_value
        mock_state.uncertainty_bits = 1.0 + step * 0.1
        mock_state.marginal_parent_probs = {var: 0.5 for var in get_variables(scm)}
        mock_state.mechanism_confidence = {var: 0.7 for var in get_variables(scm)}
        
        def mock_mechanism_insights():
            return {
                'predicted_effects': {var: 0.5 for var in get_variables(scm)},
                'mechanism_types': {var: 'linear' for var in get_variables(scm)}
            }
        
        def mock_optimization_progress():
            return {'best_value': best_value, 'steps_since_improvement': step}
        
        mock_state.get_mechanism_insights = mock_mechanism_insights
        mock_state.get_optimization_progress = mock_optimization_progress
        
        return mock_state
    
    def _policy_output_to_action(self, 
                                policy_output: Dict[str, jnp.ndarray],
                                variables: List[str], 
                                target: str,
                                training_episode: Optional[int] = None) -> jnp.ndarray:
        """Convert policy output to action vector."""
        variable_logits = policy_output.get('variable_logits', jnp.zeros(len(variables)))
        value_params = policy_output.get('value_params', jnp.zeros((len(variables), 2)))
        
        # Simple action: use mean values from policy, zero out target
        action = value_params[:, 0]  # Use means
        
        # Add exploration noise ONLY during training, with decay
        if training_episode is not None:
            # Reduced exploration noise to avoid overwhelming tiny policy outputs
            max_episodes = getattr(self.config.training, 'n_episodes', 100)
            exploration_decay = max(0.05, 1.0 - (training_episode / max_episodes))
            exploration_noise_scale = 0.1 * exploration_decay  # Reduced: starts at 0.1, decays to 0.005
            
            exploration_noise = random.normal(
                random.PRNGKey(int(jnp.sum(action * 1000) % 2**31)),  # Deterministic based on current action
                shape=action.shape
            ) * exploration_noise_scale
            
            action = action + exploration_noise
        
        # Mask target variable (set to zero after adding noise)
        target_idx = variables.index(target) if target in variables else -1
        if target_idx >= 0:
            action = action.at[target_idx].set(0.0)
        
        return action
    
    def _simulate_intervention(self, scm: pyr.PMap, action: jnp.ndarray) -> Tuple[pyr.PMap, float]:
        """Simulate intervention and compute reward."""
        variables = list(get_variables(scm))
        
        # Create intervention from action
        intervention_targets = set()
        intervention_values = {}
        
        for i, var in enumerate(variables):
            if i < len(action) and abs(action[i]) > 0.005:  # Further lowered threshold to help untrained policy
                intervention_targets.add(var)
                intervention_values[var] = float(action[i])
        
        intervention = pyr.m(
            type="perfect",
            targets=intervention_targets,
            values=intervention_values
        )
        
        # For training, use simplified reward calculation
        # In real ACBO, this would use the full verifiable reward system
        target_var = get_target(scm)
        
        # Simple reward: positive for intervening on non-target variables
        reward = 0.0
        if intervention_targets:
            # Bonus for intervening (exploration)
            reward += 0.2
            
            # Penalty if intervening on target (invalid)
            if target_var in intervention_targets:
                reward -= 0.5
            else:
                reward += 0.3  # Bonus for valid intervention
            
            # Scale by intervention magnitude (realistic outcome simulation)
            intervention_magnitude = sum(abs(v) for v in intervention_values.values())
            reward += min(0.5, intervention_magnitude * 0.1)
        
        # Ensure reward is in reasonable range
        reward = max(-1.0, min(1.0, reward))
        
        return intervention, reward
    
    def _update_policy(self, trajectory: List[Dict[str, Any]]) -> Dict[str, float]:
        """Update policy parameters using proper GRPO implementation."""
        if not trajectory:
            return {'policy_loss': 0.0, 'value_loss': 0.0}
        
        try:
            # Convert trajectory to proper GRPO format
            states = jnp.stack([step['state'] for step in trajectory])
            actions = jnp.stack([step['action'] for step in trajectory])
            rewards = jnp.array([step['reward'] for step in trajectory])
            
            # Policy learning diagnostics - track action magnitudes and rewards
            action_magnitudes = jnp.abs(actions)
            mean_action_magnitude = float(jnp.mean(action_magnitudes))
            max_action_magnitude = float(jnp.max(action_magnitudes))
            mean_reward = float(jnp.mean(rewards))
            
            # ENHANCED parameter debugging - trace the full parameter structure
            old_param_sample = None
            param_debug_info = {}
            
            try:
                # Debug: Log the full parameter structure
                logger.debug(f"Parameter structure: {type(self.policy_params)}")
                logger.debug(f"Parameter keys: {list(self.policy_params.keys()) if isinstance(self.policy_params, dict) else 'Not a dict'}")
                
                # Navigate to get a meaningful parameter sample
                param_tree = self.policy_params
                navigation_path = []
                
                while isinstance(param_tree, dict) and param_tree:
                    first_key = next(iter(param_tree.keys()))
                    navigation_path.append(first_key)
                    param_tree = param_tree[first_key]
                    logger.debug(f"Navigated to: {'/'.join(navigation_path)}, type: {type(param_tree)}")
                
                # Extract parameter value
                if hasattr(param_tree, 'flatten') and param_tree.size > 0:
                    old_param_sample = float(param_tree.flatten()[0])
                    param_debug_info['path'] = '/'.join(navigation_path)
                    param_debug_info['shape'] = param_tree.shape
                    param_debug_info['mean'] = float(jnp.mean(param_tree))
                    param_debug_info['std'] = float(jnp.std(param_tree))
                    logger.debug(f"Parameter sample extracted: value={old_param_sample:.8f}, shape={param_tree.shape}")
                else:
                    logger.warning(f"Could not extract parameter: type={type(param_tree)}, hasattr_flatten={hasattr(param_tree, 'flatten')}")
                    old_param_sample = 0.0
                    
            except Exception as e:
                logger.error(f"Parameter extraction failed: {e}")
                import traceback
                traceback.print_exc()
                old_param_sample = None
            
            # Get current value estimates and log probabilities using wrappers
            values = self.value_wrapper(self.policy_params, states)
            log_probs = self.policy_wrapper(self.policy_params, states, actions)
            
            dones = jnp.zeros_like(rewards)  # No episode termination in our case
            
            # Create GRPO trajectory with advantage computation
            grpo_trajectory = create_trajectory_from_experiences(
                states=states,
                actions=actions, 
                rewards=rewards,
                values=values,
                log_probs=log_probs,
                dones=dones,
                bootstrap_value=0.0,  # No bootstrap needed for complete episodes
                config=self.grpo_config
            )
            
            # CRITICAL DEBUG: Log parameters before GRPO update
            logger.debug(f"PRE-UPDATE: policy_params type: {type(self.policy_params)}")
            if old_param_sample is not None:
                logger.debug(f"PRE-UPDATE: sample parameter value: {old_param_sample:.8f}")
                if param_debug_info:
                    logger.debug(f"PRE-UPDATE: param stats - mean: {param_debug_info['mean']:.8f}, std: {param_debug_info['std']:.8f}")
            
            # Apply GRPO update using the proper update function
            logger.debug("Calling GRPO update function...")
            (new_policy_params, _, 
             new_policy_opt_state, new_value_opt_state, 
             update_result) = self.grpo_update_fn(
                self.policy_params,      # policy_params
                self.policy_params,      # value_params (using same network) 
                self.policy_optimizer_state,  # policy_opt_state
                self.value_optimizer_state,   # value_opt_state
                grpo_trajectory
            )
            
            # CRITICAL DEBUG: Log parameters after GRPO update
            logger.debug(f"POST-UPDATE: new_policy_params type: {type(new_policy_params)}")
            logger.debug(f"POST-UPDATE: Are params the same object? {self.policy_params is new_policy_params}")
            
            # Check if parameters actually changed
            try:
                param_tree_new = new_policy_params
                navigation_path = param_debug_info.get('path', '').split('/') if param_debug_info.get('path') else []
                
                for key in navigation_path:
                    if key and isinstance(param_tree_new, dict) and key in param_tree_new:
                        param_tree_new = param_tree_new[key]
                
                if hasattr(param_tree_new, 'flatten') and param_tree_new.size > 0:
                    new_param_value = float(param_tree_new.flatten()[0])
                    new_mean = float(jnp.mean(param_tree_new))
                    new_std = float(jnp.std(param_tree_new))
                    
                    logger.debug(f"POST-UPDATE: sample parameter value: {new_param_value:.8f}")
                    logger.debug(f"POST-UPDATE: param stats - mean: {new_mean:.8f}, std: {new_std:.8f}")
                    
                    if old_param_sample is not None:
                        actual_change = abs(new_param_value - old_param_sample)
                        logger.debug(f"POST-UPDATE: actual parameter change: {actual_change:.12f}")
                        
                        if actual_change < 1e-12:
                            logger.warning("🚨 CRITICAL: Parameters did not change after GRPO update!")
                        else:
                            logger.info(f"✅ Parameters changed by: {actual_change:.12f}")
                            
            except Exception as e:
                logger.error(f"POST-UPDATE parameter check failed: {e}")
            
            # Policy learning diagnostics - ENHANCED parameter change tracking
            param_change = 0.0
            new_param_sample = None
            
            if old_param_sample is not None and param_debug_info:
                try:
                    # Use the same navigation path as in debugging
                    param_tree = new_policy_params
                    navigation_path = param_debug_info.get('path', '').split('/') if param_debug_info.get('path') else []
                    
                    for key in navigation_path:
                        if key and isinstance(param_tree, dict) and key in param_tree:
                            param_tree = param_tree[key]
                    
                    if hasattr(param_tree, 'flatten') and param_tree.size > 0:
                        new_param_sample = float(param_tree.flatten()[0])
                        param_change = float(abs(new_param_sample - old_param_sample))
                        
                        logger.debug(f"Parameter tracking: old={old_param_sample:.8f}, new={new_param_sample:.8f}, change={param_change:.12f}")
                    else:
                        logger.warning(f"Parameter tracking failed: could not access parameter at path {param_debug_info.get('path')}")
                        new_param_sample = 0.0
                        param_change = 0.0
                    
                except Exception as e:
                    logger.error(f"Parameter tracking error: {e}")
                    param_change = 0.0
                    new_param_sample = 0.0
            
            # Log policy learning diagnostics every few updates
            if not hasattr(self, '_update_count'):
                self._update_count = 0
            self._update_count += 1
            
            if self._update_count % 5 == 0:  # Log every 5 updates
                logger.info(f"Policy Learning Diagnostics (update {self._update_count}):")
                logger.info(f"  Action magnitudes: mean={mean_action_magnitude:.6f}, max={max_action_magnitude:.6f}")
                logger.info(f"  Mean reward: {mean_reward:.3f}")
                logger.info(f"  Policy param change: {param_change:.8f}")
                
                # ENHANCED diagnostics - show actual parameter values for debugging
                if old_param_sample is not None and new_param_sample is not None:
                    logger.info(f"  Parameter values: old={old_param_sample:.8f}, new={new_param_sample:.8f}")
                
                logger.info(f"  Advantages: min={float(jnp.min(grpo_trajectory.advantages)):.3f}, max={float(jnp.max(grpo_trajectory.advantages)):.3f}")
                logger.info(f"  GRPO losses: policy={update_result.policy_loss:.6f}, value={update_result.value_loss:.6f}")
                
                # ENHANCED diagnostics - gradient norms and learning rates
                logger.info(f"  Gradient norms: policy={update_result.policy_gradient_norm:.8f}, value={update_result.value_gradient_norm:.8f}")
                logger.info(f"  Learning rates: policy={self.grpo_config.learning_rate:.6f}, value={self.grpo_config.value_learning_rate:.6f}")
                logger.info(f"  KL divergence: {update_result.kl_divergence:.6f}, entropy: {update_result.entropy_loss:.6f}")
                
                # Critical diagnostic: check if policy is learning to output larger actions
                if mean_action_magnitude < 0.005:
                    logger.warning(f"⚠️ Policy outputs very small actions ({mean_action_magnitude:.6f}) - may not trigger interventions!")
                if param_change < 1e-6:
                    logger.warning(f"⚠️ Very small parameter changes ({param_change:.8f}) - learning might be stuck!")
                if update_result.policy_gradient_norm < 1e-6:
                    logger.warning(f"⚠️ Very small policy gradients ({update_result.policy_gradient_norm:.8f}) - no learning signal!")
                if update_result.kl_divergence < 1e-6:
                    logger.warning(f"⚠️ Very small KL divergence ({update_result.kl_divergence:.8f}) - policy not changing!")
            
            # CRITICAL DEBUG: Update stored parameters and optimizer states with verification
            logger.debug("Updating stored parameters...")
            
            # Store old reference for comparison
            old_params_ref = self.policy_params
            
            # Update parameters
            self.policy_params = new_policy_params
            self.policy_optimizer_state = new_policy_opt_state
            self.value_optimizer_state = new_value_opt_state
            
            # CRITICAL VERIFICATION: Ensure parameters actually updated
            logger.debug(f"Parameter update verification:")
            logger.debug(f"  Old params reference ID: {id(old_params_ref)}")
            logger.debug(f"  New params reference ID: {id(self.policy_params)}")
            logger.debug(f"  Are references different? {old_params_ref is not self.policy_params}")
            
            # Double-check by accessing the parameter value again
            if param_debug_info and new_param_sample is not None:
                try:
                    # Verify stored parameters match what we updated to
                    param_tree = self.policy_params
                    navigation_path = param_debug_info.get('path', '').split('/') if param_debug_info.get('path') else []
                    
                    for key in navigation_path:
                        if key and isinstance(param_tree, dict) and key in param_tree:
                            param_tree = param_tree[key]
                    
                    if hasattr(param_tree, 'flatten') and param_tree.size > 0:
                        stored_param_value = float(param_tree.flatten()[0])
                        
                        if abs(stored_param_value - new_param_sample) < 1e-12:
                            logger.debug(f"✅ Parameter storage verified: {stored_param_value:.8f}")
                        else:
                            logger.error(f"🚨 CRITICAL: Parameter storage mismatch! Expected: {new_param_sample:.8f}, Got: {stored_param_value:.8f}")
                            
                except Exception as e:
                    logger.error(f"Parameter storage verification failed: {e}")
            
            return {
                'policy_loss': update_result.policy_loss,
                'value_loss': update_result.value_loss,
                'entropy_loss': update_result.entropy_loss,
                'total_loss': update_result.total_loss,
                'kl_divergence': update_result.kl_divergence,
                'explained_variance': update_result.explained_variance,
                # Add diagnostic metrics
                'mean_action_magnitude': mean_action_magnitude,
                'max_action_magnitude': max_action_magnitude,
                'param_change': param_change
            }
            
        except Exception as e:
            logger.error(f"GRPO update failed: {e}")
            import traceback
            traceback.print_exc()
            # Return zeros to indicate failed update
            return {'policy_loss': 0.0, 'value_loss': 0.0}
    
    def _setup_wandb(self):
        """Setup WandB logging if enabled."""
        try:
            import wandb
            wandb.init(
                project=self.config.logging.wandb.get('project', 'enriched-grpo'),
                config=dict(self.config),
                name=self.config.logging.wandb.get('run_name', 'enriched-training')
            )
            logger.info("WandB logging enabled")
        except ImportError:
            logger.warning("WandB not available, skipping logging setup")