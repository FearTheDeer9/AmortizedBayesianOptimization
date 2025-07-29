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
from .convergence_detector import ConvergenceDetector, ConvergenceConfig
# Use correct GRPO implementation (group-relative, policy-only)
from ..acquisition.grpo import (
    GRPOConfig, GRPOUpdate, create_grpo_trainer, collect_grpo_batch,
    create_grpo_batch_from_samples
)
from ..acquisition.state import AcquisitionState
from ..acquisition.rewards import create_default_reward_config
from ..data_structures.scm import get_variables, get_target
from ..data_structures.buffer import ExperienceBuffer
from ..data_structures.sample import get_values
from ..analysis.trajectory_metrics import (
    compute_f1_score_from_marginals, 
    compute_true_parent_likelihood,
    compute_shd_from_marginals
)
from ..visualization.metric_dashboard import TrainingMetricsLogger

logger = logging.getLogger(__name__)


class EpisodeState:
    """
    Maintains real observation history for a single episode.
    
    This class tracks actual interventions and outcomes during an episode,
    allowing the policy to learn from real data instead of synthetic features.
    """
    def __init__(self, scm: pyr.PMap):
        """Initialize episode state for the given SCM."""
        self.scm = scm
        self.buffer = ExperienceBuffer()  # Real observation storage
        self.observations = []  # List of (intervention, outcome) pairs
        self.best_value = float('-inf')  # Track best target value seen
        self.target_variable = get_target(scm)
        self.variables = list(get_variables(scm))
        
        # Optional: Maintain posterior (can be expensive to update)
        self.posterior = None
        
        # Add initial observational sample to establish baseline
        self._add_initial_observation()
    
    def _add_initial_observation(self):
        """Add an initial observational sample to establish baseline."""
        from ..environments.sampling import sample_from_linear_scm
        
        # Sample one observational data point
        obs_samples = sample_from_linear_scm(self.scm, n_samples=1, seed=42)
        if obs_samples:
            self.buffer.add_observation(obs_samples[0])
            # Update best value from initial observation
            values = get_values(obs_samples[0])
            if self.target_variable in values:
                self.best_value = float(values[self.target_variable])
    
    def add_intervention_outcome(self, intervention: pyr.PMap, outcome: pyr.PMap):
        """Add a real intervention and its outcome to the history."""
        self.observations.append((intervention, outcome))
        self.buffer.add_intervention(intervention, outcome)
        
        # Update best value
        values = get_values(outcome)
        if self.target_variable in values:
            target_value = float(values[self.target_variable])
            self.best_value = max(self.best_value, target_value)
    
    def get_intervention_count(self) -> int:
        """Get number of interventions performed."""
        return len(self.observations)
    
    def get_unique_intervention_targets(self) -> set:
        """Get set of variables that have been intervened upon."""
        targets = set()
        for intervention, _ in self.observations:
            targets.update(intervention.get('targets', set()))
        return targets


class EnrichedGRPOTrainer:
    """Main trainer that orchestrates modular components."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        
        # Initialize modular components
        self.scm_manager = SCMRotationManager(config)
        self.state_converter = StateConverter(config)
        self.checkpoint_manager = CheckpointManager(config)
        self.metrics_collector = MetricsCollector()
        
        # Initialize convergence detection if enabled
        if config.training.get('early_stopping_enabled', False):
            convergence_config = ConvergenceConfig(
                structure_accuracy_threshold=config.training.get(
                    'convergence_accuracy_threshold', 0.95
                ),
                patience=config.training.get('convergence_patience', 10),
                min_episodes=config.training.get('min_episodes_per_scm', 20),
                max_episodes_per_scm=config.training.get('max_episodes_per_scm', 100)
            )
            self.convergence_detector = ConvergenceDetector(convergence_config)
            logger.info("Early stopping enabled with convergence detection")
        else:
            self.convergence_detector = None
        
        # Create policy
        policy_factory = PolicyFactory(config)
        self.policy_fn, self.policy_config = policy_factory.create_policy()
        
        # Initialize policy parameters with proper key threading
        self._initial_key = random.PRNGKey(config.seed)
        init_key, trainer_key = random.split(self._initial_key)
        
        dummy_input = self._create_dummy_input()
        self.policy_params = self.policy_fn.init(init_key, dummy_input)
        self._current_key = trainer_key
        
        # Create correct GRPO configuration from provided config
        # Use optimized same-state batching parameters
        if hasattr(config.training, 'grpo_config'):
            # Read from config if available
            grpo_cfg = config.training.grpo_config
            self.grpo_config = GRPOConfig(
                group_size=getattr(grpo_cfg, 'group_size', 16),
                interventions_per_state=getattr(grpo_cfg, 'interventions_per_state', 16),
                learning_rate=getattr(grpo_cfg, 'learning_rate', config.training.learning_rate),
                clip_ratio=getattr(grpo_cfg, 'clip_ratio', 0.2),
                entropy_coeff=getattr(grpo_cfg, 'entropy_coeff', 0.01),
                max_grad_norm=getattr(grpo_cfg, 'max_grad_norm', 1.0)
            )
            logger.info(f"✅ Using optimized GRPO config: group_size={self.grpo_config.group_size}, "
                       f"interventions_per_state={self.grpo_config.interventions_per_state}")
        else:
            # Fallback with optimized defaults for same-state batching
            logger.warning("⚠️ No GRPO config found, using optimized defaults")
            self.grpo_config = GRPOConfig(
                group_size=16,  # Optimized for same-state batching
                interventions_per_state=16,  # Same-state batching
                learning_rate=config.training.learning_rate,
                clip_ratio=0.2,
                entropy_coeff=0.01,
                max_grad_norm=1.0
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
        # Get optimization direction from config, default to MAXIMIZE
        optimization_direction = 'MAXIMIZE'  # Default
        if hasattr(config, 'optimization') and hasattr(config.optimization, 'direction'):
            optimization_direction = config.optimization.direction
        elif hasattr(config, 'experiment') and hasattr(config.experiment, 'optimization_direction'):
            optimization_direction = config.experiment.optimization_direction
        elif config.get('optimization_direction'):
            optimization_direction = config.get('optimization_direction')
        
        self.reward_config = create_default_reward_config(
            optimization_weight=config.training.reward_weights.optimization,
            structure_weight=config.training.reward_weights.discovery,
            exploration_weight=config.training.reward_weights.efficiency,
            optimization_direction=optimization_direction
        )
        
        # Initialize sample accumulation buffer for GRPO
        self.sample_buffer = []
        # For same-state batching, we update every episode (no accumulation across episodes)
        self.update_frequency = 1  # Update every episode with same-state batching
        
        # Get max variables from config or SCMs
        max_vars = 10  # default
        if hasattr(self.config.experiment.scm_generation, 'variable_range'):
            max_vars = self.config.experiment.scm_generation.variable_range[1]
        logger.info(f"Initialized trainer with up to {max_vars} variables")
        logger.info(f"GRPO group size: {self.grpo_config.group_size}, update frequency: {self.update_frequency} episodes")
        
        # Enable structure learning metrics tracking
        self.track_structure_metrics = config.training.get('track_structure_metrics', True)
        
        # Initialize metrics dashboard if enabled
        self.enable_dashboard = config.training.get('enable_dashboard', False)
        if self.enable_dashboard:
            self.metrics_logger = TrainingMetricsLogger()
        else:
            self.metrics_logger = None
    
    def _create_dummy_input(self) -> jnp.ndarray:
        """Create dummy input for policy initialization."""
        max_history_size = self.config.training.state_config.get('max_history_size', 100)
        num_channels = self.config.training.state_config.get('num_channels', 5)
        
        # Use a reasonable default for max variables during initialization
        # The actual size will be determined dynamically based on the SCM
        max_variables = 10  # Default max for initialization
        if hasattr(self.config.experiment.scm_generation, 'variable_range'):
            max_variables = self.config.experiment.scm_generation.variable_range[1]
        
        return jnp.zeros((max_history_size, max_variables, num_channels))
    
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
        
        # Start metrics dashboard if enabled
        if self.metrics_logger:
            self.metrics_logger.start_logging()
            logger.info("Started real-time metrics dashboard")
        
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
                
                # Increment episode count for current SCM (only once per episode)
                self.scm_manager.increment_episode_count()
                
                # Update convergence detector if enabled
                if self.convergence_detector:
                    scm_name, _ = self.scm_manager.get_current_scm(episode)
                    self.convergence_detector.update(scm_name, metrics)
                    
                    # Check convergence
                    converged, reason = self.convergence_detector.check_convergence(scm_name)
                    
                    # Check if we should rotate to next SCM
                    if self.scm_manager.should_rotate(converged):
                        scm_info = self.scm_manager.get_current_scm_info()
                        logger.info(
                            f"Episode {episode}: Rotating from {scm_info['scm_name']} "
                            f"(trained for {scm_info['episodes_on_current']} episodes). "
                            f"Reason: {reason}"
                        )
                        
                        # Log convergence summary if available
                        if self.convergence_detector:
                            summary = self.convergence_detector.get_scm_summary(scm_name)
                            logger.info(
                                f"SCM summary - Best accuracy: {summary['best_structure_accuracy']:.3f}, "
                                f"Episodes: {summary['episodes_trained']}"
                            )
                        
                        # Advance to next SCM
                        if not self.scm_manager.advance_to_next_scm():
                            # We've completed all SCMs in rotation
                            logger.info("Completed full SCM rotation")
                
                # Log to dashboard if enabled
                if self.metrics_logger:
                    self.metrics_logger.log_episode_metrics(episode, metrics)
                
                # Log progress
                if episode % 10 == 0:
                    log_msg = (
                        f"Episode {episode}: reward={metrics.mean_reward:.3f}, "
                        f"intervention_rate={metrics.structure_accuracy:.3f}, "
                        f"scm={metrics.scm_type}"
                    )
                    
                    # Add structure learning metrics if available
                    if metrics.f1_score is not None:
                        log_msg += f", F1={metrics.f1_score:.3f}"
                    if metrics.true_parent_likelihood is not None:
                        log_msg += f", P(Parents)={metrics.true_parent_likelihood:.3f}"
                    if metrics.shd is not None:
                        log_msg += f", SHD={metrics.shd}"
                    
                    logger.info(log_msg)
                
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
        
        # Stop metrics dashboard and export final metrics
        if self.metrics_logger:
            self.metrics_logger.stop_logging()
            metrics_file = self.metrics_logger.export_final_metrics()
            training_summary = self.metrics_logger.get_training_summary()
            
            logger.info(f"Exported training metrics to {metrics_file}")
            logger.info(f"Training summary: {training_summary}")
        
        logger.info(f"Training completed in {total_time:.1f}s")
        logger.info(f"Final checkpoint: {final_checkpoint}")
        
        result = {
            'checkpoint_path': str(final_checkpoint),
            'performance': performance_analysis,
            'policy_config': self.policy_config
        }
        
        # Add metrics summary if available
        if self.metrics_logger:
            result['metrics_summary'] = self.metrics_logger.get_training_summary()
            if metrics_file:
                result['metrics_export'] = str(metrics_file)
        
        # Add convergence summary if available
        if self.convergence_detector:
            convergence_summary = self.convergence_detector.get_training_summary()
            result['convergence_summary'] = convergence_summary
            
            # Log key insights
            logger.info(
                f"Convergence summary: {convergence_summary['converged_scms']}/{convergence_summary['total_scms']} SCMs converged"
            )
            logger.info(
                f"Training distribution: {convergence_summary['discovery_ratio']:.1%} discovery, "
                f"{1 - convergence_summary['discovery_ratio']:.1%} exploitation"
            )
        
        return result
    
    def _run_episode(self, episode_idx: int, episode_key: jax.random.PRNGKey) -> TrainingMetrics:
        """Run a single training episode with real observations."""
        # Get current SCM
        scm_name, scm = self.scm_manager.get_current_scm(episode_idx)
        variables = list(get_variables(scm))
        target = get_target(scm)
        
        # Check if we should use real observations
        use_real_observations = self.config.training.get('use_real_observations', True)
        
        # Log configuration
        if hasattr(self.config.training, 'synthetic_features'):
            synthetic_enabled = self.config.training.synthetic_features.get('enabled', False)
            if synthetic_enabled:
                logger.warning("synthetic_features.enabled is DEPRECATED - use use_real_observations instead")
        
        if use_real_observations:
            # Create episode state for tracking real observations
            episode_state = EpisodeState(scm)
            logger.info(f"Episode {episode_idx}: Using REAL observations for {scm_name}")
        else:
            # Fall back to synthetic features (for comparison/debugging)
            logger.warning(f"Episode {episode_idx}: Using SYNTHETIC features (not recommended)")
        
        # Collect trajectory
        trajectory = []
        episode_rewards = []
        
        # Thread keys through episode steps
        step_key = episode_key
        
        # For compatibility with old approach
        best_value = 0.0
        previous_state = None
        
        for step in range(self.config.training.episode_length):
            # Split key for this step
            step_key, action_key = random.split(step_key)
            
            if use_real_observations:
                # Create state from REAL observations
                state = self._create_real_state(episode_state, step)
                
                # Convert to tensor format for policy (this is where we bridge to tensors)
                # For now, we'll use the existing converter but with real state
                tensor_state = self._convert_real_state_to_tensor(state, step)
                enriched_input = self.state_converter.convert_state_to_enriched_input(tensor_state)
            else:
                # Old approach: synthetic tensor-backed state
                state = self._create_tensor_backed_state(scm, step, best_value)
                enriched_input = self.state_converter.convert_state_to_enriched_input(state)
            
            # Get target variable index
            target_idx = variables.index(target) if target in variables else 0
            
            # Log channel statistics for debugging (every 100 episodes on first step)
            if step == 0 and episode_idx % 100 == 0:
                self._log_channel_statistics(enriched_input, variables, target_idx)
            
            # Policy forward pass
            policy_output = self.policy_fn.apply(
                self.policy_params, action_key, enriched_input, target_idx, True
            )
            
            # Convert policy output to action (with training episode for exploration)
            action = self._policy_output_to_action(policy_output, variables, target, episode_idx)
            
            # Simulate intervention and get outcome
            intervention, outcome = self._simulate_intervention_outcome(scm, action)
            
            if use_real_observations:
                # Add real outcome to episode state
                episode_state.add_intervention_outcome(intervention, outcome)
            
            # Calculate reward
            if use_real_observations and step > 0:
                # Use the full verifiable reward system with real observations
                # Create the state before intervention for reward computation
                state_before = self._create_real_state(episode_state, step - 1)
                
                # Try to use verifiable reward system
                try:
                    # Create state after intervention (includes the new outcome)
                    state_after = self._create_real_state(episode_state, step)
                    
                    reward = self._compute_reward(
                        state_before=state_before,
                        state_after=state_after,
                        intervention=intervention,
                        outcome=outcome
                    )
                except Exception as e:
                    logger.warning(f"Verifiable reward failed: {e}, falling back to training reward")
                    reward = self._compute_training_reward(
                        episode_state=episode_state,
                        intervention=intervention,
                        outcome=outcome,
                        step=step
                    )
            else:
                # Fallback to simple reward for first step or synthetic mode
                reward = self._compute_simple_reward(outcome, target)
            
            # Store step data
            trajectory.append({
                'state': enriched_input,
                'action': action,
                'reward': reward,
                'policy_output': policy_output,
                'target_idx': target_idx,
                'intervention': intervention,  # Store intervention for metrics
                'outcome': outcome  # Store outcome for analysis
            })
            episode_rewards.append(reward)
            
            # Update best value for compatibility
            outcome_values = get_values(outcome)
            target_value = float(outcome_values.get(target, 0.0))
            if self.reward_config.get('optimization_direction') == 'MAXIMIZE':
                best_value = max(best_value, target_value)
            else:
                best_value = min(best_value, target_value)
            
            # Update previous state for next iteration
            previous_state = state
        
        # Update policy with sample accumulation
        policy_losses = self._update_policy(trajectory, episode_idx)
        
        # Compute meaningful intervention metrics instead of structure accuracy
        total_interventions = sum(1 for step in trajectory if len(step.get('intervention', {}).get('targets', set())) > 0)
        intervention_rate = total_interventions / len(trajectory) if trajectory else 0.0
        
        # Track action magnitudes for policy learning assessment
        max_action_magnitude = policy_losses.get('max_action_magnitude', 0.0)
        
        # Compute structure learning metrics if enabled
        f1_score, true_parent_likelihood, shd = None, None, None
        marginal_probs_dict = None
        
        if self.track_structure_metrics and trajectory:
            # Extract marginal probabilities from the final state
            final_enriched_input = trajectory[-1]['state']
            marginal_probs_dict = self._extract_marginal_probabilities(final_enriched_input)
            
            # Update marginal probabilities to use actual variable names
            if marginal_probs_dict:
                # Map generic variable names to actual variable names
                actual_marginal_probs = {}
                for i, var_name in enumerate(variables):
                    if i < len(marginal_probs_dict):
                        generic_name = f"X{i}"
                        if generic_name in marginal_probs_dict:
                            actual_marginal_probs[var_name] = marginal_probs_dict[generic_name]
                
                marginal_probs_dict = actual_marginal_probs
                
                # Compute structure learning metrics
                f1_score, true_parent_likelihood, shd = self._compute_structure_learning_metrics(scm, marginal_probs_dict)
        
        # Create metrics with structure learning metrics included
        metrics = TrainingMetrics(
            episode=episode_idx,
            mean_reward=float(jnp.mean(jnp.array(episode_rewards))),
            structure_accuracy=intervention_rate,  # Repurposed: now represents intervention success rate
            optimization_improvement=float(jnp.mean(jnp.array(episode_rewards)) - episode_rewards[0]),
            policy_loss=policy_losses['policy_loss'],
            value_loss=policy_losses['value_loss'],
            scm_type=scm_name,
            f1_score=f1_score,
            true_parent_likelihood=true_parent_likelihood,
            shd=shd,
            marginal_probs=marginal_probs_dict
        )
        
        return metrics
    
    def _create_tensor_backed_state(self, scm: pyr.PMap, step: int, best_value: float) -> Any:
        """Create tensor-backed acquisition state for training with bootstrap surrogate features."""
        from ..jax_native.state import create_tensor_backed_state_from_scm
        
        # Create proper TensorBackedAcquisitionState with bootstrap surrogate features
        # This replaces the problematic constant default values with meaningful variable differentiation
        return create_tensor_backed_state_from_scm(
            scm=scm,
            step=step,
            best_value=best_value,
            uncertainty_bits=1.0 + step * 0.1,  # Gradual uncertainty increase
            max_samples=self.config.training.state_config.get('max_history_size', 100),
            max_history=self.config.training.state_config.get('max_history_size', 100),
            feature_dim=3,
            use_bootstrap_surrogate=True  # NEW: Enable bootstrap surrogate features
        )
    
    def _create_real_state(self, episode_state: EpisodeState, step: int) -> AcquisitionState:
        """
        Create acquisition state from real observations.
        
        This method creates a state that reflects actual intervention history
        rather than synthetic bootstrap features.
        
        Args:
            episode_state: Current episode state with real observations
            step: Current step number
            
        Returns:
            AcquisitionState with real data
        """
        from ..avici_integration.parent_set import ParentSetPosterior
        
        # Use real buffer with actual observations
        buffer = episode_state.buffer
        
        # Create posterior (start uniform if not available)
        if episode_state.posterior is None:
            # Create uniform posterior over parent sets
            n_vars = len(episode_state.variables)
            
            # Create a more realistic initial posterior:
            # Include empty set and all single-parent sets
            parent_sets = [frozenset()]  # Empty parent set
            
            # Add each single variable as a potential parent
            for var in episode_state.variables:
                if var != episode_state.target_variable:
                    parent_sets.append(frozenset([var]))
            
            # Uniform probabilities over all parent sets
            n_sets = len(parent_sets)
            probabilities = jnp.ones(n_sets) / n_sets
            
            # Import the create function
            from ..avici_integration.parent_set.posterior import create_parent_set_posterior
            
            # Create uniform posterior over empty and single-parent sets
            posterior = create_parent_set_posterior(
                target_variable=episode_state.target_variable,
                parent_sets=parent_sets,
                probabilities=probabilities,
                metadata=pyr.m(initial_posterior=True, n_parent_sets_total=2**(n_vars-1))
            )
        else:
            posterior = episode_state.posterior
        
        # Create real acquisition state
        state = AcquisitionState(
            posterior=posterior,
            buffer=buffer,
            best_value=episode_state.best_value,
            current_target=episode_state.target_variable,
            step=step,
            metadata=pyr.m(
                scm=episode_state.scm,
                intervention_count=episode_state.get_intervention_count(),
                unique_targets=episode_state.get_unique_intervention_targets()
            )
        )
        
        return state
    
    def _compute_structure_learning_metrics(self, scm: pyr.PMap, marginal_probs: Dict[str, float]) -> Tuple[Optional[float], Optional[float], Optional[int]]:
        """
        Compute structure learning metrics: F1 score, true parent likelihood, and SHD.
        
        Args:
            scm: The structural causal model
            marginal_probs: Dictionary mapping variables to their marginal parent probabilities
            
        Returns:
            Tuple of (f1_score, true_parent_likelihood, shd)
        """
        try:
            target = get_target(scm)
            if not target:
                return None, None, None
            
            # Get true parents for the target variable
            # In a real implementation, this would extract from the SCM structure
            # For now, we'll use the SCM's adjacency structure
            variables = list(get_variables(scm))
            
            # Extract true parents from SCM (simplified - real implementation depends on SCM format)
            true_parents = []
            adjacency_matrix = scm.get('adjacency_matrix', {})
            
            if adjacency_matrix and target in adjacency_matrix:
                # Get incoming edges to target
                for var in variables:
                    if var != target and adjacency_matrix.get(var, {}).get(target, 0) != 0:
                        true_parents.append(var)
            else:
                # Fallback: try to infer from edge structure if available
                edges = scm.get('edges', frozenset())
                for edge in edges:
                    if len(edge) == 2 and edge[1] == target:
                        true_parents.append(edge[0])
            
            if not marginal_probs:
                logger.debug("No marginal probabilities available for structure metrics")
                return None, None, None
            
            # Compute metrics
            f1_score = compute_f1_score_from_marginals(marginal_probs, true_parents, threshold=0.5)
            parent_likelihood = compute_true_parent_likelihood(marginal_probs, true_parents)
            shd = compute_shd_from_marginals(marginal_probs, true_parents, threshold=0.5)
            
            return f1_score, parent_likelihood, shd
            
        except Exception as e:
            logger.debug(f"Failed to compute structure learning metrics: {e}")
            return None, None, None
    
    def _extract_marginal_probabilities(self, enriched_input: jnp.ndarray) -> Dict[str, float]:
        """
        Extract marginal parent probabilities from enriched input tensor.
        
        Args:
            enriched_input: Enriched history tensor [history_size, n_vars, n_channels]
            
        Returns:
            Dictionary mapping variable names to marginal parent probabilities
        """
        try:
            # Extract marginal probabilities from channel 1 (parent probabilities)
            # Use the most recent step (last in history)
            if enriched_input.shape[0] == 0:
                return {}
            
            recent_probs = enriched_input[-1, :, 1]  # Channel 1 = parent probabilities
            
            # Get variable names (this would need to be passed or stored)
            # For now, create generic variable names
            n_vars = enriched_input.shape[1]
            variable_names = [f"X{i}" for i in range(n_vars)]
            
            marginal_probs = {}
            for i, var_name in enumerate(variable_names):
                marginal_probs[var_name] = float(recent_probs[i])
            
            return marginal_probs
            
        except Exception as e:
            logger.debug(f"Failed to extract marginal probabilities: {e}")
            return {}
    
    
    def _policy_output_to_action(self, 
                                policy_output: Dict[str, jnp.ndarray],
                                variables: List[str], 
                                target: str,
                                training_episode: Optional[int] = None) -> Tuple[int, float]:
        """
        Convert policy output to per-variable action (variable index, intervention value).
        
        This implements explicit variable selection followed by value sampling for the
        selected variable, enabling more efficient GRPO training.
        
        Returns:
            Tuple of (selected_variable_index, intervention_value)
        """
        variable_logits = policy_output.get('variable_logits', jnp.zeros(len(variables)))
        value_params = policy_output.get('value_params', jnp.zeros((len(variables), 2)))
        
        # Debug logging setup
        if not hasattr(self, '_action_debug_count'):
            self._action_debug_count = 0
        self._action_debug_count += 1
        
        # Get target index
        target_idx = variables.index(target) if target in variables else -1
        
        # Log raw outputs periodically
        if self._action_debug_count % 10 == 0:
            logger.info(f"🔍 Per-Variable Encoding - Policy Output (call {self._action_debug_count}):")
            logger.info(f"  Variable logits: {variable_logits}")
            logger.info(f"  Variables: {variables}, Target: {target}")
            
            # Check if target is properly masked
            if target_idx >= 0:
                target_logit = variable_logits[target_idx]
                logger.info(f"  Target variable '{target}' at index {target_idx}, logit: {target_logit}")
                if target_logit > -1e8:
                    logger.warning("⚠️ Target variable not properly masked!")
            
            # Check for identical logits (posterior collapse)
            non_target_indices = [i for i in range(len(variable_logits)) if i != target_idx]
            if non_target_indices:
                non_target_logits = variable_logits[jnp.array(non_target_indices)]
                unique_logits = jnp.unique(non_target_logits)
                logit_variance = jnp.var(non_target_logits)
                
                # Log variance to track collapse
                logger.info(f"  Non-target logit variance: {logit_variance:.6f}")
                
                if len(unique_logits) == 1 or logit_variance < 1e-6:
                    logger.warning(f"⚠️ IDENTICAL LOGITS for all non-target variables: {unique_logits[0]:.6f}")
                    logger.warning("  This indicates posterior collapse - policy cannot differentiate between variables!")
                    logger.warning(f"  Logit variance: {logit_variance:.8f}")
                elif logit_variance < 0.01:
                    logger.warning(f"⚠️ Low logit variance ({logit_variance:.6f}) - approaching posterior collapse")
        
        # Add diversity bonus during training to prevent posterior collapse
        if training_episode is not None and training_episode < 50:
            # Add small random noise to logits to maintain diversity early in training
            noise_scale = 0.1 * (1.0 - training_episode / 50.0)  # Decay noise over time
            key = jax.random.PRNGKey(self._action_debug_count + training_episode)
            noise = jax.random.normal(key, shape=variable_logits.shape) * noise_scale
            # Only add noise to non-target variables
            noise_mask = jnp.arange(len(variables)) != target_idx
            variable_logits = variable_logits + (noise * noise_mask)
        
        # STEP 1: Variable Selection using softmax over logits
        # Apply temperature for exploration control during training
        variable_selection_temp = 1.0  # Default temperature
        if training_episode is not None:
            # Temperature decay: start at 2.0, decay to 0.5
            max_episodes = getattr(self.config.training, 'n_episodes', 100)
            temp_decay = training_episode / max_episodes
            variable_selection_temp = 2.0 - 1.5 * temp_decay  # 2.0 -> 0.5
        
        # Apply temperature scaling
        scaled_logits = variable_logits / variable_selection_temp
        
        # Compute probabilities (softmax handles -inf properly)
        variable_probs = jax.nn.softmax(scaled_logits)
        
        # Sample variable using a deterministic key for reproducibility
        logit_sum = float(jnp.sum(jnp.abs(variable_logits)))
        var_seed = int((logit_sum * 1000 + self._action_debug_count) % 1000000)
        var_key = random.PRNGKey(var_seed)
        
        # Use categorical sampling
        selected_var_idx = int(random.categorical(var_key, scaled_logits))
        
        # STEP 2: Value Selection for the selected variable
        # Get parameters for selected variable
        selected_mean = value_params[selected_var_idx, 0]
        selected_log_std = value_params[selected_var_idx, 1]
        selected_std = jnp.exp(selected_log_std)
        
        # Apply value temperature for exploration
        value_temp = 1.0
        if training_episode is not None:
            # Value temperature: allow more exploration early
            value_temp = 1.5 - 0.5 * temp_decay  # 1.5 -> 1.0
        
        # Sample intervention value from Gaussian
        val_seed = int((float(selected_mean + selected_log_std) * 1000 + self._action_debug_count) % 1000000)
        val_key = random.PRNGKey(val_seed)
        
        # Sample from scaled Gaussian
        noise = random.normal(val_key, shape=())
        intervention_value = float(selected_mean + selected_std * value_temp * noise)
        
        # Log selection details periodically
        if self._action_debug_count % 10 == 0:
            logger.info(f"  Variable selection:")
            logger.info(f"    Temperature: {variable_selection_temp:.2f}")
            logger.info(f"    Probabilities: {variable_probs}")
            logger.info(f"    Selected: {variables[selected_var_idx]} (index {selected_var_idx})")
            logger.info(f"  Value selection:")
            logger.info(f"    Mean: {selected_mean:.4f}, Std: {selected_std:.4f}")
            logger.info(f"    Temperature: {value_temp:.2f}")
            logger.info(f"    Sampled value: {intervention_value:.4f}")
        
        # Validate selection
        if selected_var_idx == target_idx:
            logger.warning(f"⚠️ Selected target variable for intervention! This should be rare.")
        
        return selected_var_idx, intervention_value
    
    def _create_acquisition_state(
        self, 
        scm: pyr.PMap, 
        buffer: Optional[ExperienceBuffer] = None,
        posterior: Optional[Any] = None,
        best_value: Optional[float] = None,
        step: int = 0
    ) -> Any:
        """
        Create an AcquisitionState for reward computation.
        
        Args:
            scm: Structural causal model
            buffer: Experience buffer (will create empty if None)
            posterior: Parent set posterior (will create default if None)
            best_value: Best observed value (will use 0.0 if None)
            step: Current step number
            
        Returns:
            AcquisitionState object
        """
        from ..acquisition.state import AcquisitionState
        from ..data_structures.buffer import ExperienceBuffer
        from ..avici_integration.parent_set import ParentSetPosterior
        
        target_var = get_target(scm)
        variables = list(get_variables(scm))
        
        # Create default buffer if needed
        if buffer is None:
            from ..data_structures.sample import create_sample
            buffer = ExperienceBuffer()
            # Add a dummy observational sample so buffer knows about all variables
            dummy_values = {var: 0.0 for var in variables}
            dummy_sample = create_sample(values=dummy_values)
            buffer.add_observation(dummy_sample)
        
        # Create default posterior if needed
        if posterior is None:
            # Simple uniform posterior for training
            variables = list(get_variables(scm))
            n_parent_sets = 2 ** (len(variables) - 1)  # All possible parent sets
            posterior_probs = jnp.ones(n_parent_sets) / n_parent_sets
            
            # Create minimal posterior object
            # Import the create function if not already imported
            from ..avici_integration.parent_set.posterior import create_parent_set_posterior
            
            # Create initial posterior with empty and single-parent sets
            parent_sets = [frozenset()]  # Empty parent set
            
            # Add single-parent sets
            for var in variables:
                if var != target_var:
                    parent_sets.append(frozenset([var]))
            
            # Uniform probabilities
            n_sets = len(parent_sets)
            probabilities = jnp.ones(n_sets) / n_sets
            
            posterior = create_parent_set_posterior(
                target_variable=target_var,
                parent_sets=parent_sets,
                probabilities=probabilities,
                metadata=pyr.m(initial_posterior=True)
            )
        
        # Use provided or default best value
        if best_value is None:
            best_value = 0.0
        
        # Create acquisition state
        state = AcquisitionState(
            posterior=posterior,
            buffer=buffer,
            best_value=best_value,
            current_target=target_var,
            step=step,
            metadata=pyr.m(scm=scm)
        )
        
        return state
    
    def _simulate_intervention_outcome(self, scm: pyr.PMap, action: Tuple[int, float]) -> Tuple[pyr.PMap, pyr.PMap]:
        """
        Simulate intervention and return the outcome.
        
        Args:
            scm: Structural causal model
            action: Tuple of (selected_variable_index, intervention_value)
            
        Returns:
            Tuple of (intervention object, outcome sample)
        """
        variables = list(get_variables(scm))
        target_var = get_target(scm)
        
        # Extract per-variable action
        selected_var_idx, intervention_value = action
        
        # Create intervention from per-variable action
        intervention_targets = set()
        intervention_values = {}
        
        # Only intervene if value is non-negligible
        if abs(intervention_value) > 0.005:
            selected_var = variables[selected_var_idx]
            intervention_targets.add(selected_var)
            intervention_values[selected_var] = float(intervention_value)
        
        intervention = pyr.m(
            type="perfect",
            targets=intervention_targets,
            values=intervention_values
        )
        
        # Sample outcome from intervention
        from ..environments.sampling import sample_with_intervention
        seed = int(time.time() * 1000000) % 1000000
        outcomes = sample_with_intervention(scm, intervention, n_samples=1, seed=seed)
        outcome = outcomes[0]
        
        return intervention, outcome
    
    def _compute_simple_reward(self, outcome: pyr.PMap, target: str) -> float:
        """
        Compute simple reward for first step (no previous state).
        
        Args:
            outcome: Outcome sample from intervention
            target: Target variable name
            
        Returns:
            Simple reward based on target value
        """
        outcome_values = get_values(outcome)
        target_value = float(outcome_values.get(target, 0.0))
        
        # Simple reward based on optimization direction
        optimization_direction = self.reward_config.get('optimization_direction', 'MAXIMIZE')
        if optimization_direction == 'MINIMIZE':
            # For minimization, negative values are good
            return -target_value * 0.1
        else:
            # For maximization, positive values are good
            return target_value * 0.1
    
    def _compute_reward(
        self, 
        state_before: Any, 
        state_after: Any, 
        intervention: pyr.PMap, 
        outcome: pyr.PMap
    ) -> float:
        """
        Compute reward using the sophisticated reward system.
        
        Args:
            state_before: State before intervention
            state_after: State after intervention
            intervention: Intervention that was applied
            outcome: Outcome from the intervention
            
        Returns:
            Reward value
        """
        from ..acquisition.rewards import compute_verifiable_reward, create_default_reward_config
        
        # Create reward config with user's requirements
        # Use weights from training config
        weights = self.config.training.reward_weights
        reward_config = create_default_reward_config(
            optimization_weight=weights.get('optimization', 1.0),
            structure_weight=weights.get('discovery', 0.5),
            parent_weight=0.3,  # Reward intervening on parents
            exploration_weight=weights.get('efficiency', 0.1),
            optimization_direction=self.reward_config.get('optimization_direction', 'MAXIMIZE')
        )
        
        # Add flag for expected value baseline
        reward_config = reward_config.set('use_expected_value_baseline', True)
        
        try:
            # Compute verifiable reward with all components
            reward_components = compute_verifiable_reward(
                state_before=state_before,
                intervention=intervention,
                outcome=outcome,
                state_after=state_after,
                config=reward_config
            )
            
            reward = float(reward_components.total_reward)
            
            # Log reward decomposition periodically
            if hasattr(self, '_reward_log_count'):
                self._reward_log_count += 1
            else:
                self._reward_log_count = 1
                
            # Log more frequently during early training
            log_freq = 10 if self._reward_log_count < 100 else 100
            if self._reward_log_count % log_freq == 0:
                logger.info(f"Reward decomposition: {reward_components.summary()}")
                logger.info(f"Total reward: {reward:.4f}")
            
            return reward
            
        except Exception as e:
            logger.warning(f"Failed to compute verifiable reward: {e}, using simple fallback")
            # Fallback to simple reward
            return self._compute_simple_reward(outcome, state_before.current_target)
    
    def _compute_training_reward(
        self,
        episode_state: EpisodeState,
        intervention: pyr.PMap,
        outcome: pyr.PMap,
        step: int
    ) -> float:
        """
        Compute training reward without requiring a surrogate model.
        
        This method computes rewards based on real observations without
        requiring pre-trained models or synthetic features.
        
        Args:
            episode_state: Current episode state with real observations
            intervention: The intervention that was applied
            outcome: The observed outcome
            step: Current step number
            
        Returns:
            Reward value
        """
        target_var = episode_state.target_variable
        outcome_values = get_values(outcome)
        target_value = float(outcome_values.get(target_var, 0.0))
        
        # Get optimization direction
        optimization_direction = self.reward_config.get('optimization_direction', 'MAXIMIZE')
        
        # 1. Optimization reward - based on actual improvement
        optimization_reward = 0.0
        
        # Compare to mean of recent observations (expected value baseline)
        recent_values = []
        for obs_intervention, obs_outcome in episode_state.observations[-10:]:  # Last 10 observations
            obs_values = get_values(obs_outcome)
            if target_var in obs_values:
                recent_values.append(float(obs_values[target_var]))
        
        if recent_values:
            baseline_value = sum(recent_values) / len(recent_values)
        else:
            baseline_value = episode_state.best_value
        
        # Compute improvement from baseline
        if optimization_direction == 'MINIMIZE':
            improvement = baseline_value - target_value  # Lower is better
        else:
            improvement = target_value - baseline_value  # Higher is better
        
        # Normalize by typical value range (estimate from observations)
        value_range = 1.0  # Default
        if len(recent_values) > 1:
            value_range = max(recent_values) - min(recent_values)
            if value_range < 0.1:
                value_range = 1.0  # Avoid division by small numbers
        
        optimization_reward = improvement / value_range
        
        # 2. Simple exploration reward - encourage trying different variables
        exploration_reward = 0.0
        intervention_targets = intervention.get('targets', set())
        
        if intervention_targets:
            # Count how many times we've intervened on these variables
            target_counts = {}
            for past_intervention, _ in episode_state.observations:
                past_targets = past_intervention.get('targets', set())
                for var in past_targets:
                    target_counts[var] = target_counts.get(var, 0) + 1
            
            # Reward less-explored variables
            avg_count = sum(target_counts.values()) / max(len(target_counts), 1)
            for var in intervention_targets:
                var_count = target_counts.get(var, 0)
                if var_count <= avg_count:
                    exploration_reward += 0.1  # Bonus for exploring less-visited variables
        
        # 3. Skip structure discovery reward (requires surrogate)
        # 4. Skip parent intervention reward (requires posterior)
        
        # Combine rewards
        total_reward = (
            self.reward_config.get('optimization_weight', 1.0) * optimization_reward +
            self.reward_config.get('exploration_weight', 0.1) * exploration_reward
        )
        
        # Log periodically
        if step % 10 == 0:
            logger.debug(
                f"Training reward: opt={optimization_reward:.3f}, "
                f"explore={exploration_reward:.3f}, total={total_reward:.3f}"
            )
        
        return float(total_reward)
    
    def _convert_real_state_to_tensor(self, state: AcquisitionState, step: int) -> AcquisitionState:
        """
        Convert real AcquisitionState to format expected by StateConverter.
        
        In practice, our real AcquisitionState already has the required interface:
        - buffer with get_all_samples() and get_variable_coverage()
        - current_target attribute
        
        So we can return it directly. This method exists for clarity and
        potential future transformations.
        
        Args:
            state: Real AcquisitionState from _create_real_state
            step: Current step (for potential future use)
            
        Returns:
            The state itself (no conversion needed currently)
        """
        # Verify state has required attributes
        if not hasattr(state, 'buffer'):
            raise ValueError("State missing 'buffer' attribute")
        if not hasattr(state, 'current_target'):
            raise ValueError("State missing 'current_target' attribute")
            
        # Could add step-specific transformations here if needed
        # For now, just return the state as-is
        return state
    
    def _update_policy(self, trajectory: List[Dict[str, Any]], episode_idx: int) -> Dict[str, float]:
        """Update policy parameters using proper GRPO implementation with same-state batching."""
        if not trajectory:
            return {'policy_loss': 0.0, 'value_loss': 0.0}
        
        try:
            # SAME-STATE BATCHING: Use only samples from current episode/trajectory
            # Do not accumulate across different SCMs
            if len(trajectory) < self.grpo_config.interventions_per_state:
                logger.debug(f"Trajectory too short: {len(trajectory)} < {self.grpo_config.interventions_per_state}")
                # For short trajectories, repeat samples to reach minimum
                samples_needed = self.grpo_config.interventions_per_state
                repeated_trajectory = []
                while len(repeated_trajectory) < samples_needed:
                    repeated_trajectory.extend(trajectory)
                trajectory = repeated_trajectory[:samples_needed]
            
            # Clear sample buffer for same-state batching
            self.sample_buffer = []
            
            # Add trajectory samples to buffer (all from same episode/SCM)
            for step in trajectory[:self.grpo_config.group_size]:  # Limit to group size
                # Format: (state, action, reward, old_log_prob)
                self.sample_buffer.append((step['state'], step['action'], float(step['reward']), 0.0))
            
            # Ensure we have exactly group_size samples for same-state batching
            if len(self.sample_buffer) != self.grpo_config.group_size:
                logger.debug(f"Adjusting sample buffer: {len(self.sample_buffer)} -> {self.grpo_config.group_size}")
                # Pad or trim to exact group size
                while len(self.sample_buffer) < self.grpo_config.group_size:
                    # Repeat samples if needed
                    idx = len(self.sample_buffer) % len(trajectory)
                    step = trajectory[idx]
                    self.sample_buffer.append((step['state'], step['action'], float(step['reward']), 0.0))
                self.sample_buffer = self.sample_buffer[:self.grpo_config.group_size]
            
            logger.debug(f"Running GRPO update with {len(self.sample_buffer)} samples")
            
            # Convert trajectory to diagnostics - handle per-variable actions
            # Actions are now tuples (var_idx, value)
            action_values = jnp.array([step['action'][1] for step in trajectory])  # Get intervention values
            action_indices = jnp.array([step['action'][0] for step in trajectory])  # Get selected variables
            rewards = jnp.array([step['reward'] for step in trajectory])
            
            # Policy learning diagnostics - track action magnitudes and rewards
            action_magnitudes = jnp.abs(action_values)
            max_action_magnitude = float(jnp.max(action_magnitudes))
            mean_reward = float(jnp.mean(rewards))
            
            # SIMPLE parameter monitoring - just check if ANY parameters changed
            old_param_norm = optax.global_norm(self.policy_params)
            logger.debug(f"PRE-UPDATE: Total parameter norm: {float(old_param_norm):.8f}")
            
            # Extract data from sample buffer for GRPO batch creation
            rewards_tensor = jnp.array([sample[2] for sample in self.sample_buffer])  # [batch_size]
            old_log_probs_tensor = jnp.array([sample[3] for sample in self.sample_buffer])  # [batch_size]
            # Actions are now tuples (var_idx, value) - need to handle differently
            actions_list = [sample[1] for sample in self.sample_buffer]  # List of (var_idx, value) tuples
            
            # CRITICAL DEBUG: Log parameters before GRPO update
            logger.debug(f"PRE-UPDATE: policy_params type: {type(self.policy_params)}")
            logger.debug(f"PRE-UPDATE: batch shapes - actions: {len(actions_list)}, rewards: {rewards_tensor.shape}")
            logger.debug(f"PRE-UPDATE: parameter norm: {float(old_param_norm):.8f}")
            
            # PHASE 3 FIX: Use TensorBackedAcquisitionState directly with updated GRPO
            logger.debug("Creating TensorBackedAcquisitionState objects for GRPO...")
            
            # SAME-STATE BATCHING: Get current SCM from episode
            current_scm_name, current_scm = self.scm_manager.get_current_scm(episode_idx)
            logger.debug(f"Using SCM '{current_scm_name}' for same-state batch")
            
            # Create TensorBackedAcquisitionState objects for each sample in buffer
            # All samples are from the same episode/SCM for same-state batching
            acquisition_states = []
            for i, (enriched_input, action, reward, old_log_prob) in enumerate(self.sample_buffer):
                # Create TensorBackedAcquisitionState for this sample
                step_state = self._create_tensor_backed_state(current_scm, step=i, best_value=reward)
                acquisition_states.append(step_state)
            
            # Convert per-variable actions to intervention objects
            interventions = []
            variables = list(get_variables(current_scm))
            for action_tuple in actions_list:
                # Extract per-variable action components
                selected_var_idx, intervention_value = action_tuple
                
                # Ensure variable index is valid
                if selected_var_idx < len(variables):
                    selected_var = variables[selected_var_idx]
                    
                    # Create intervention using existing framework
                    from ..interventions.handlers import create_perfect_intervention
                    intervention = create_perfect_intervention(
                        targets=frozenset([selected_var]),
                        values={selected_var: float(intervention_value)}
                    )
                    interventions.append(intervention)
                else:
                    logger.warning(f"Invalid variable index {selected_var_idx} for SCM with {len(variables)} variables")
                    # Skip invalid indices to prevent dimension mismatches
                    continue
            
            # Ensure all batch components have same size for same-state batching
            if len(interventions) != len(acquisition_states):
                logger.warning(f"Intervention count mismatch: {len(interventions)} != {len(acquisition_states)}")
                # Pad interventions with dummy values if needed
                while len(interventions) < len(acquisition_states):
                    # Create a dummy no-op intervention
                    from ..interventions.handlers import create_perfect_intervention
                    dummy_intervention = create_perfect_intervention(targets=frozenset(), values={})
                    interventions.append(dummy_intervention)
            
            # Create proper GRPO batch format
            grpo_batch_correct = {
                'states': acquisition_states,
                'actions': interventions,
                'rewards': rewards_tensor,
                'old_log_probs': old_log_probs_tensor
            }
            
            # Log batch dimensions for debugging
            logger.debug(f"GRPO batch dimensions - states: {len(acquisition_states)}, "
                        f"actions: {len(interventions)}, rewards: {rewards_tensor.shape}")
            
            # Verify same-state batching
            scm_variables = list(get_variables(current_scm))
            logger.debug(f"Same-state batch using SCM with {len(scm_variables)} variables: {scm_variables}")
            
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
    
    def _log_channel_statistics(self, enriched_input: jnp.ndarray, variables: List[str], target_idx: int):
        """
        Log channel statistics to debug posterior collapse.
        
        Args:
            enriched_input: Enriched history tensor [T, n_vars, n_channels]
            variables: List of variable names
            target_idx: Index of target variable
        """
        try:
            # Get the most recent timestep for analysis
            recent_step = enriched_input[-1]  # [n_vars, n_channels]
            n_vars, n_channels = recent_step.shape
            
            logger.info(f"\n=== Channel Statistics (n_vars={n_vars}, n_channels={n_channels}) ===")
            
            # Channel names from state enrichment
            channel_names = [
                "variable_values",
                "intervention_indicators", 
                "target_indicators",
                "marginal_parent_probs",
                "intervention_recency"
            ]
            
            # Analyze each channel
            for ch_idx in range(n_channels):
                if ch_idx < len(channel_names):
                    ch_name = channel_names[ch_idx]
                else:
                    ch_name = f"channel_{ch_idx}"
                
                channel_values = recent_step[:, ch_idx]
                
                # Check if all values are identical (posterior collapse indicator)
                unique_values = jnp.unique(channel_values)
                is_collapsed = len(unique_values) == 1
                
                # Compute statistics
                ch_mean = jnp.mean(channel_values)
                ch_std = jnp.std(channel_values)
                ch_min = jnp.min(channel_values)
                ch_max = jnp.max(channel_values)
                
                # Log channel info
                logger.info(f"\nChannel {ch_idx} ({ch_name}):")
                logger.info(f"  Values: {channel_values}")
                logger.info(f"  Stats: mean={ch_mean:.4f}, std={ch_std:.4f}, min={ch_min:.4f}, max={ch_max:.4f}")
                
                if is_collapsed:
                    logger.warning(f"  ⚠️ COLLAPSED: All values identical ({unique_values[0]:.4f})")
                else:
                    logger.info(f"  ✅ Variable-specific: {len(unique_values)} unique values")
                
                # Special checks for specific channels
                if ch_idx == 2:  # Target indicators
                    target_value = channel_values[target_idx] if target_idx < n_vars else -1
                    logger.info(f"  Target variable {variables[target_idx]} has value: {target_value}")
                    if target_value != 1.0:
                        logger.warning("  ⚠️ Target indicator not set correctly!")
            
            # Check overall differentiation
            all_identical = True
            for i in range(n_vars):
                for j in range(i+1, n_vars):
                    var_i_features = recent_step[i, :]
                    var_j_features = recent_step[j, :]
                    if not jnp.allclose(var_i_features, var_j_features):
                        all_identical = False
                        break
                if not all_identical:
                    break
            
            if all_identical and n_vars > 1:
                logger.error("\n🚨 CRITICAL: All variables have IDENTICAL features - complete posterior collapse!")
            else:
                # Find which channels provide differentiation
                differentiating_channels = []
                for ch_idx in range(n_channels):
                    channel_values = recent_step[:, ch_idx]
                    if len(jnp.unique(channel_values)) > 1:
                        differentiating_channels.append(ch_idx)
                
                logger.info(f"\n✅ Variables are differentiable via channels: {differentiating_channels}")
            
            logger.info("=" * 60 + "\n")
            
        except Exception as e:
            logger.error(f"Failed to log channel statistics: {e}")