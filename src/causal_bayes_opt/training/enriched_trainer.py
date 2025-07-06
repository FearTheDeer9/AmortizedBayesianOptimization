"""
Enriched GRPO Trainer - Main Training Orchestrator

This module provides the main training coordinator that uses modular components
following CLAUDE.md single-responsibility principles.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any

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
# Use correct GRPO implementation (group-relative, policy-only)
from ..acquisition.grpo import (
    GRPOConfig, GRPOUpdate, create_grpo_trainer, collect_grpo_batch,
    create_grpo_batch_from_samples
)
from ..acquisition.state import AcquisitionState
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
        
        # Create correct GRPO configuration (group-relative) 
        # Use smaller group size that's achievable with episode length
        episode_length = config.training.episode_length
        achievable_group_size = max(16, min(64, episode_length * 8))  # Allow 8 episodes to accumulate
        
        # Increase entropy coefficient for better exploration if default is too small
        entropy_coeff = getattr(config.training, 'entropy_coefficient', 0.01)
        if entropy_coeff < 0.02:
            logger.info(f"⚡ Increasing entropy coefficient from {entropy_coeff:.3f} to 0.02 for better exploration")
            entropy_coeff = 0.02
        
        self.grpo_config = GRPOConfig(
            group_size=achievable_group_size,
            learning_rate=config.training.learning_rate,
            clip_ratio=getattr(config.training, 'clip_ratio', 0.2),
            entropy_coeff=entropy_coeff,
            max_grad_norm=getattr(config.training, 'max_grad_norm', 1.0)
        )
        
        # Create optimizer with learning rate schedule for better training dynamics
        base_lr = self.grpo_config.learning_rate
        
        # Increase learning rate if it's very small (common issue causing poor learning)
        if base_lr < 1e-4:
            logger.info(f"⚡ Increasing learning rate from {base_lr:.2e} to 3e-4 for better training dynamics")
            base_lr = 3e-4
        
        # Create learning rate schedule with warmup
        warmup_steps = 100
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=base_lr * 0.1,  # Start at 10% of base LR
            peak_value=base_lr,        # Reach full LR after warmup
            warmup_steps=warmup_steps,
            decay_steps=10000          # Long decay
        )
        
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.grpo_config.max_grad_norm),
            optax.adam(learning_rate=lr_schedule)
        )
        
        # Initialize optimizer state and step counter for LR schedule
        self.optimizer_state = self.optimizer.init(self.policy_params)
        self.training_step = 0
        
        logger.info(f"Correct GRPO Config: group_size={self.grpo_config.group_size}, lr={self.grpo_config.learning_rate:.6f}")
        logger.info(f"Correct GRPO Config: entropy_coeff={self.grpo_config.entropy_coeff:.3f}, clip_ratio={self.grpo_config.clip_ratio:.2f}")
        
        # Create policy wrapper (no value wrapper needed)
        self.policy_wrapper = self._create_policy_wrapper()
        
        # GRPO update function already created above during initialization
        
        # Create reward configuration
        self.reward_config = create_default_reward_config(
            optimization_weight=config.training.reward_weights.optimization,
            structure_weight=config.training.reward_weights.discovery,
            exploration_weight=config.training.reward_weights.efficiency
        )
        
        # Initialize sample accumulation buffer for GRPO
        self.sample_buffer = []
        self.update_frequency = max(1, self.grpo_config.group_size // episode_length)  # Update every N episodes
        
        logger.info(f"Initialized trainer with {self.scm_manager.max_variables} max variables")
        logger.info(f"GRPO group size: {self.grpo_config.group_size}, update frequency: {self.update_frequency} episodes")
    
    def _create_dummy_input(self) -> jnp.ndarray:
        """Create dummy input for policy initialization."""
        max_history_size = self.config.training.state_config.get('max_history_size', 100)
        num_channels = self.config.training.state_config.get('num_channels', 10)
        
        return jnp.zeros((max_history_size, self.scm_manager.max_variables, num_channels))
    
    def _create_policy_wrapper(self) -> Any:
        """Create policy wrapper for policy-only GRPO compatibility.
        
        Returns:
            Policy wrapper function that extracts log probabilities from enriched policy
        """
        
        def policy_wrapper(params, states, actions):
            """Wrapper that extracts log probabilities from enriched policy.
            
            Expected policy-only GRPO signature: policy_fn(params, states, actions) -> log_probs
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
        
        return policy_wrapper
    
    
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
            
            # Create tensor-backed state for this step
            state = self._create_tensor_backed_state(scm, step, 0.0)
            
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
        
        # Update policy with sample accumulation
        policy_losses = self._update_policy(trajectory, episode_idx)
        
        # Compute meaningful intervention metrics instead of structure accuracy
        total_interventions = sum(1 for step in trajectory if len(step.get('intervention', {}).get('targets', set())) > 0)
        intervention_rate = total_interventions / len(trajectory) if trajectory else 0.0
        
        # Track action magnitudes for policy learning assessment
        max_action_magnitude = policy_losses.get('max_action_magnitude', 0.0)
        
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
    
    def _create_tensor_backed_state(self, scm: pyr.PMap, step: int, best_value: float) -> Any:
        """Create tensor-backed acquisition state for training."""
        from ..jax_native.state import create_tensor_backed_state_from_scm
        
        # Create proper TensorBackedAcquisitionState instead of Mock objects
        return create_tensor_backed_state_from_scm(
            scm=scm,
            step=step,
            best_value=best_value,
            uncertainty_bits=1.0 + step * 0.1,  # Gradual uncertainty increase
            max_samples=self.config.training.state_config.get('max_history_size', 100),
            max_history=50,
            feature_dim=3
        )
    
    
    def _policy_output_to_action(self, 
                                policy_output: Dict[str, jnp.ndarray],
                                variables: List[str], 
                                target: str,
                                training_episode: Optional[int] = None) -> jnp.ndarray:
        """Convert policy output to action vector."""
        variable_logits = policy_output.get('variable_logits', jnp.zeros(len(variables)))
        value_params = policy_output.get('value_params', jnp.zeros((len(variables), 2)))
        
        # DEBUG PHASE 1: Log raw policy outputs
        if not hasattr(self, '_action_debug_count'):
            self._action_debug_count = 0
        self._action_debug_count += 1
        
        # PHASE 1 ENHANCED: Network output validation logging
        if self._action_debug_count % 5 == 0:  # More frequent logging for debugging
            logger.info(f"🔍 PHASE 1 ENHANCED - Network Output Validation (call {self._action_debug_count}):")
            logger.info(f"  Raw variable_logits: {variable_logits}")
            logger.info(f"  Raw value_params shape: {value_params.shape}")
            logger.info(f"  Raw value_params means: {value_params[:, 0]}")
            logger.info(f"  Raw value_params log_stds: {value_params[:, 1]}")
            logger.info(f"  Raw value_params stds: {jnp.exp(value_params[:, 1])}")
            logger.info(f"  Variables: {variables}, Target: {target}")
            
            # CRITICAL VALIDATION: Check for all-zero outputs (indicates dead network)
            means_magnitude = float(jnp.max(jnp.abs(value_params[:, 0])))
            logits_magnitude = float(jnp.max(jnp.abs(variable_logits)))
            
            if means_magnitude < 1e-8:
                logger.warning(f"⚠️ PHASE 1 CRITICAL: Policy means are nearly zero! Magnitude: {means_magnitude:.12f}")
                logger.warning("  This suggests the policy network is not learning or is initialized poorly")
            
            if logits_magnitude < 1e-8:
                logger.warning(f"⚠️ PHASE 1 CRITICAL: Variable logits are nearly zero! Magnitude: {logits_magnitude:.12f}")
                logger.warning("  This suggests the policy network is outputting uniform distributions")
            
            # Check for NaN or infinite values
            if jnp.any(jnp.isnan(value_params)) or jnp.any(jnp.isinf(value_params)):
                logger.error("🚨 PHASE 1 CRITICAL: NaN or Inf detected in value_params!")
            
            if jnp.any(jnp.isnan(variable_logits)) or jnp.any(jnp.isinf(variable_logits)):
                logger.error("🚨 PHASE 1 CRITICAL: NaN or Inf detected in variable_logits!")
            
            # Log standard deviation magnitudes (important for exploration)
            std_magnitudes = jnp.exp(value_params[:, 1])
            min_std = float(jnp.min(std_magnitudes))
            max_std = float(jnp.max(std_magnitudes))
            logger.info(f"  Standard deviation range: [{min_std:.6f}, {max_std:.6f}]")
            
            if max_std > 10.0:
                logger.warning(f"⚠️ PHASE 1: Very large std detected ({max_std:.3f}) - may cause unstable sampling")
            if min_std < 1e-6:
                logger.warning(f"⚠️ PHASE 1: Very small std detected ({min_std:.6f}) - may prevent exploration")
        
        # PHASE 1 CONTINUED: Check raw policy outputs before any processing
        action = value_params[:, 0]  # Use means - this is our raw action before scaling/clipping
        raw_action_magnitude = float(jnp.max(jnp.abs(action)))
        
        if self._action_debug_count % 5 == 0:
            logger.info(f"  PHASE 1: Raw action before any scaling/clipping: {action}")
            logger.info(f"  PHASE 1: Raw action magnitude: {raw_action_magnitude:.8f}")
            
            # CRITICAL: Check if policy is learning to produce non-zero actions
            if raw_action_magnitude < 1e-6:
                logger.warning(f"🚨 PHASE 1 CRITICAL: Policy producing extremely small actions ({raw_action_magnitude:.8f})")
                logger.warning("  This indicates the policy may not be learning to take meaningful actions")
                
            # Check action distribution - are all actions similar or diverse?
            action_std = float(jnp.std(action))
            logger.info(f"  PHASE 1: Action diversity (std): {action_std:.8f}")
            
            if action_std < 1e-6:
                logger.warning(f"⚠️ PHASE 1: Very low action diversity ({action_std:.8f}) - policy may be collapsed")
            
            # Track action evolution over time
            if not hasattr(self, '_action_magnitude_history'):
                self._action_magnitude_history = []
            self._action_magnitude_history.append(raw_action_magnitude)
            
            # Show trend if we have enough history
            if len(self._action_magnitude_history) >= 5:
                recent_magnitudes = self._action_magnitude_history[-5:]
                magnitude_trend = recent_magnitudes[-1] - recent_magnitudes[0]
                logger.info(f"  PHASE 1: Action magnitude trend (last 5): {magnitude_trend:+.8f}")
                
                if abs(magnitude_trend) < 1e-8:
                    logger.warning("⚠️ PHASE 1: Action magnitudes not changing - policy may be stuck")
        
        # Add exploration noise ONLY during training, with decay
        exploration_applied = False
        if training_episode is not None:
            # Reduced exploration noise to avoid overwhelming tiny policy outputs
            max_episodes = getattr(self.config.training, 'n_episodes', 100)
            exploration_decay = max(0.05, 1.0 - (training_episode / max_episodes))
            exploration_noise_scale = 0.1 * exploration_decay  # Reduced: starts at 0.1, decays to 0.005
            
            # Use safer deterministic key generation
            action_sum = float(jnp.sum(jnp.abs(action))) 
            seed_val = int((action_sum * 1000) % 1000000)  # Safer modulo
            exploration_noise = random.normal(
                random.PRNGKey(seed_val),
                shape=action.shape
            ) * exploration_noise_scale
            
            if self._action_debug_count % 10 == 0:
                logger.debug(f"  Training episode: {training_episode}, Exploration decay: {exploration_decay:.3f}")
                logger.debug(f"  Exploration noise scale: {exploration_noise_scale:.6f}")
                logger.debug(f"  Exploration noise: {exploration_noise}")
            
            action = action + exploration_noise
            exploration_applied = True
        
        # Mask target variable (set to zero after adding noise)
        target_idx = variables.index(target) if target in variables else -1
        if target_idx >= 0:
            action = action.at[target_idx].set(0.0)
        
        final_action_magnitude = float(jnp.max(jnp.abs(action)))
        
        if self._action_debug_count % 10 == 0:
            logger.debug(f"  Final action: {action}")
            logger.debug(f"  Final action magnitude: {final_action_magnitude:.8f}")
            logger.debug(f"  Target masked at index: {target_idx}")
            logger.debug(f"  Exploration applied: {exploration_applied}")
            
            # Analysis of action degradation
            magnitude_change = final_action_magnitude - raw_action_magnitude
            logger.debug(f"  Action magnitude change: {magnitude_change:+.8f}")
            
            if final_action_magnitude < 0.01:
                logger.warning(f"⚠️ PHASE 1: Very small final action magnitude ({final_action_magnitude:.8f})")
            if abs(magnitude_change) > raw_action_magnitude * 0.5:
                logger.warning(f"⚠️ PHASE 1: Large magnitude change ({magnitude_change:+.8f}), exploration may be dominating")
        
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
        
        # PHASE 4: Validate reward function incentive structure
        if hasattr(self, '_reward_debug_count'):
            self._reward_debug_count += 1
        else:
            self._reward_debug_count = 1
        
        debug_rewards = (self._reward_debug_count % 10 == 0)  # Debug every 10th reward computation
        
        # For training, use simplified reward calculation
        # In real ACBO, this would use the full verifiable reward system
        target_var = get_target(scm)
        
        # PHASE 4: Detailed reward component analysis
        reward_components = {}
        
        # Simple reward: positive for intervening on non-target variables
        reward = 0.0
        if intervention_targets:
            # Bonus for intervening (exploration)
            exploration_bonus = 0.2
            reward += exploration_bonus
            reward_components['exploration_bonus'] = exploration_bonus
            
            # Penalty if intervening on target (invalid)
            if target_var in intervention_targets:
                target_penalty = -0.5
                reward += target_penalty
                reward_components['target_penalty'] = target_penalty
            else:
                valid_intervention_bonus = 0.3  # Bonus for valid intervention
                reward += valid_intervention_bonus
                reward_components['valid_intervention_bonus'] = valid_intervention_bonus
            
            # Scale by intervention magnitude (realistic outcome simulation)
            intervention_magnitude = sum(abs(v) for v in intervention_values.values())
            magnitude_bonus = min(0.5, intervention_magnitude * 0.1)
            reward += magnitude_bonus
            reward_components['magnitude_bonus'] = magnitude_bonus
        else:
            # No intervention penalty
            no_action_penalty = -0.1
            reward += no_action_penalty
            reward_components['no_action_penalty'] = no_action_penalty
        
        # Ensure reward is in reasonable range
        reward_before_clipping = reward
        reward = max(-1.0, min(1.0, reward))
        
        # PHASE 4: Log reward analysis for debugging incentive structure
        if debug_rewards:
            logger.info(f"🔍 PHASE 4 REWARD ANALYSIS (computation {self._reward_debug_count}):")
            logger.info(f"  Action: {action}")
            logger.info(f"  Intervention targets: {intervention_targets}")
            logger.info(f"  Intervention values: {intervention_values}")
            logger.info(f"  Target variable: {target_var}")
            logger.info(f"  Reward components: {reward_components}")
            logger.info(f"  Total reward before clipping: {reward_before_clipping:.6f}")
            logger.info(f"  Final reward: {reward:.6f}")
            
            # CRITICAL: Check reward incentive alignment
            if intervention_targets:
                if target_var in intervention_targets:
                    logger.info("  ✅ INCENTIVE CHECK: Correctly penalizing intervention on target variable")
                else:
                    logger.info("  ✅ INCENTIVE CHECK: Correctly rewarding intervention on non-target variables")
            else:
                logger.info("  ⚠️ INCENTIVE CHECK: No intervention detected - applying no-action penalty")
            
            # Check for potential reward hacking
            if reward > 0.8:
                logger.info(f"  💰 HIGH REWARD: Policy achieved high reward ({reward:.3f}) - good performance!")
            elif reward < -0.8:
                logger.warning(f"  💸 LOW REWARD: Policy received low reward ({reward:.3f}) - poor performance")
            
            # Track reward distribution over time
            if not hasattr(self, '_reward_history'):
                self._reward_history = []
            self._reward_history.append(reward)
            
            # Analyze reward trends
            if len(self._reward_history) >= 10:
                recent_rewards = self._reward_history[-10:]
                reward_trend = recent_rewards[-1] - recent_rewards[0]
                reward_mean = sum(recent_rewards) / len(recent_rewards)
                logger.info(f"  REWARD TREND: mean={reward_mean:.3f}, trend={reward_trend:+.3f}")
                
                if reward_mean < 0:
                    logger.warning("⚠️ PHASE 4: Negative average reward - policy may not be learning effective actions!")
        
        return intervention, reward
    
    def _update_policy(self, trajectory: List[Dict[str, Any]], episode_idx: int) -> Dict[str, float]:
        """Update policy parameters using proper GRPO implementation with sample accumulation."""
        if not trajectory:
            return {'policy_loss': 0.0, 'value_loss': 0.0}
        
        try:
            # Add trajectory samples to accumulation buffer
            for step in trajectory:
                # Format: (state, action, reward, old_log_prob)
                # Use the enriched input tensor instead of mock state object
                self.sample_buffer.append((step['state'], step['action'], float(step['reward']), 0.0))
            
            # Only update when we have enough samples
            if len(self.sample_buffer) < self.grpo_config.group_size:
                logger.debug(f"Accumulating samples: {len(self.sample_buffer)}/{self.grpo_config.group_size}")
                return {'policy_loss': 0.0, 'value_loss': 0.0, 'samples_accumulated': len(self.sample_buffer)}
            
            logger.debug(f"Running GRPO update with {len(self.sample_buffer)} samples")
            
            # Convert trajectory to diagnostics
            actions = jnp.stack([step['action'] for step in trajectory])
            rewards = jnp.array([step['reward'] for step in trajectory])
            
            # Policy learning diagnostics - track action magnitudes and rewards
            action_magnitudes = jnp.abs(actions)
            max_action_magnitude = float(jnp.max(action_magnitudes))
            mean_reward = float(jnp.mean(rewards))
            
            # SIMPLE parameter monitoring - just check if ANY parameters changed
            old_param_norm = optax.global_norm(self.policy_params)
            logger.debug(f"PRE-UPDATE: Total parameter norm: {float(old_param_norm):.8f}")
            
            # Extract data from sample buffer for GRPO batch creation
            rewards_tensor = jnp.array([sample[2] for sample in self.sample_buffer])  # [batch_size]
            old_log_probs_tensor = jnp.array([sample[3] for sample in self.sample_buffer])  # [batch_size]
            actions_tensor = jnp.stack([sample[1] for sample in self.sample_buffer])  # [batch_size, vars]
            
            # CRITICAL DEBUG: Log parameters before GRPO update
            logger.debug(f"PRE-UPDATE: policy_params type: {type(self.policy_params)}")
            logger.debug(f"PRE-UPDATE: batch shapes - actions: {actions_tensor.shape}, rewards: {rewards_tensor.shape}")
            logger.debug(f"PRE-UPDATE: parameter norm: {float(old_param_norm):.8f}")
            
            # PHASE 3 FIX: Use TensorBackedAcquisitionState directly with updated GRPO
            logger.debug("Creating TensorBackedAcquisitionState objects for GRPO...")
            
            # Get current SCM for state creation
            current_scm_name, current_scm = self.scm_manager.scm_rotation[0]  # Get first SCM from rotation
            
            # Create TensorBackedAcquisitionState objects for each sample in buffer
            # Each sample in buffer corresponds to a step, so create state for each step
            acquisition_states = []
            for i, (enriched_input, action, reward, old_log_prob) in enumerate(self.sample_buffer):
                # Create TensorBackedAcquisitionState for this sample
                step_state = self._create_tensor_backed_state(current_scm, step=i, best_value=reward)
                acquisition_states.append(step_state)
            
            # Convert tensor actions to intervention objects (simplified)
            interventions = []
            variables = list(get_variables(current_scm))
            for action_tensor in actions_tensor:
                # Find the variable with max activation (simplified variable selection)
                selected_var_idx = int(jnp.argmax(jnp.abs(action_tensor)))
                selected_var = variables[selected_var_idx]
                intervention_value = float(action_tensor[selected_var_idx])
                
                # Create intervention using existing framework
                from ..interventions.handlers import create_perfect_intervention
                intervention = create_perfect_intervention(
                    targets=frozenset([selected_var]),
                    values={selected_var: intervention_value}
                )
                interventions.append(intervention)
            
            # Create proper GRPO batch format
            grpo_batch_correct = {
                'states': acquisition_states,
                'actions': interventions,
                'rewards': rewards_tensor,
                'old_log_probs': old_log_probs_tensor
            }
            
            logger.debug("Calling CORRECT GRPO implementation...")
            # Use the correct GRPO loss computation with existing optimizer
            def loss_fn(params):
                from ..acquisition.grpo import _compute_grpo_loss
                return _compute_grpo_loss(params, grpo_batch_correct, self.policy_fn, self.grpo_config)
            
            # Compute loss and gradients
            (loss_value, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(self.policy_params)
            
            # Apply updates using the trainer's existing optimizer
            updates, new_opt_state = self.optimizer.update(grads, self.optimizer_state, self.policy_params)
            new_policy_params = optax.apply_updates(self.policy_params, updates)
            
            # Create update result
            grad_norm = optax.global_norm(grads)
            update_result = GRPOUpdate(
                policy_loss=float(loss_info['policy_loss']),
                entropy_loss=float(loss_info['entropy_loss']),
                kl_penalty=float(loss_info['kl_penalty']),
                total_loss=float(loss_value),
                grad_norm=float(grad_norm),
                group_baseline=float(loss_info['group_baseline']),
                mean_reward=float(loss_info['mean_reward']),
                reward_std=float(loss_info['reward_std']),
                mean_advantage=float(loss_info['mean_advantage']),
                advantage_std=float(loss_info['advantage_std']),
                mean_entropy=float(loss_info['mean_entropy']),
                approx_kl=float(loss_info['approx_kl'])
            )
            
            # CRITICAL DEBUG: Log parameters after GRPO update
            logger.debug(f"POST-UPDATE: new_policy_params type: {type(new_policy_params)}")
            logger.debug(f"POST-UPDATE: Are params the same object? {self.policy_params is new_policy_params}")
            
            # Check if parameters actually changed using simple norm comparison
            try:
                new_param_norm = optax.global_norm(new_policy_params)
                param_norm_change = abs(float(new_param_norm) - float(old_param_norm))
                
                logger.debug(f"POST-UPDATE: Total parameter norm: {float(new_param_norm):.8f}")
                logger.debug(f"POST-UPDATE: Parameter norm change: {param_norm_change:.12f}")
                
                if param_norm_change < 1e-12:
                    logger.warning("🚨 CRITICAL: Parameters did not change after GRPO update!")
                else:
                    logger.info(f"✅ Parameters changed - norm delta: {param_norm_change:.12f}")
                            
            except Exception as e:
                logger.error(f"POST-UPDATE parameter check failed: {e}")
            
            # Policy learning diagnostics - SIMPLE parameter change tracking
            param_change = param_norm_change if 'param_norm_change' in locals() else 0.0
            
            # Log policy learning diagnostics every few updates
            if not hasattr(self, '_update_count'):
                self._update_count = 0
            self._update_count += 1
            
            if self._update_count % 5 == 0:  # Log every 5 updates
                logger.info(f"Policy Learning Diagnostics (update {self._update_count}):")
                logger.info(f"  Action magnitudes: max={max_action_magnitude:.6f}")
                logger.info(f"  Mean reward: {mean_reward:.3f}")
                logger.info(f"  Policy param change: {param_change:.8f}")
                
                # Show parameter norm changes for debugging
                logger.info(f"  Parameter norm change: {param_change:.8f}")
                
                # Get buffer rewards for group statistics (accumulation shows group size)
                buffer_rewards = [sample[2] for sample in self.sample_buffer] if len(self.sample_buffer) > 0 else [0.0]
                logger.info(f"  Rewards: min={min(buffer_rewards):.3f}, max={max(buffer_rewards):.3f}, group_baseline={update_result.group_baseline:.3f}")
                logger.info(f"  GRPO losses: policy={update_result.policy_loss:.6f}, entropy={update_result.entropy_loss:.6f}")
                
                # ENHANCED diagnostics - gradient norms and learning rates
                logger.info(f"  Gradient norm: {update_result.grad_norm:.8f}")
                logger.info(f"  Learning rate: {self.grpo_config.learning_rate:.6f}")
                logger.info(f"  KL penalty: {update_result.kl_penalty:.6f}, approx_kl: {update_result.approx_kl:.6f}")
                
                # Critical diagnostic: check if policy is learning to output larger actions
                if max_action_magnitude < 0.005:
                    logger.warning(f"⚠️ Policy outputs very small actions ({max_action_magnitude:.6f}) - may not trigger interventions!")
                if param_change < 1e-8:
                    logger.warning(f"⚠️ Very small parameter changes ({param_change:.8f}) - learning might be stuck!")
                if update_result.grad_norm < 1e-6:
                    logger.warning(f"⚠️ Very small gradients ({update_result.grad_norm:.8f}) - no learning signal!")
                if update_result.approx_kl < 1e-6:
                    logger.warning(f"⚠️ Very small KL divergence ({update_result.approx_kl:.8f}) - policy not changing!")
            
            # Update stored parameters and optimizer states
            logger.debug("Updating stored parameters...")
            
            # Update parameters (policy-only)
            self.policy_params = new_policy_params
            self.optimizer_state = new_opt_state
            
            logger.debug(f"✅ Parameters and optimizer state updated successfully")
            
            # Verify parameter storage by checking if references changed
            try:
                stored_param_norm = optax.global_norm(self.policy_params)
                storage_norm_change = abs(float(stored_param_norm) - float(new_param_norm))
                
                if storage_norm_change < 1e-12:
                    logger.debug(f"✅ Parameter storage verified: norm={float(stored_param_norm):.8f}")
                else:
                    logger.error(f"🚨 CRITICAL: Parameter storage mismatch! Expected norm: {float(new_param_norm):.8f}, Got: {float(stored_param_norm):.8f}")
                        
            except Exception as e:
                logger.error(f"Parameter storage verification failed: {e}")
            
            # Clear sample buffer after successful update
            self.sample_buffer = []
            logger.debug(f"Cleared sample buffer after GRPO update")
            
            return {
                'policy_loss': update_result.policy_loss,
                'value_loss': 0.0,  # No value loss in policy-only GRPO
                'entropy_loss': update_result.entropy_loss,
                'total_loss': update_result.total_loss,
                'kl_divergence': update_result.approx_kl,
                'explained_variance': 0.0,  # No value function in policy-only GRPO
                # Add diagnostic metrics
                'max_action_magnitude': max_action_magnitude,
                'param_norm_change': param_change,
                'samples_processed': len(rewards_tensor)
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