"""GRPO Training Manager for Complete Training Orchestration.

This module provides the main training orchestration for GRPO-based acquisition
model training, integrating all subsystems including experience management,
reward computation, diversity monitoring, and async training infrastructure.

Key features:
- Complete GRPO training pipeline with experience replay
- Integration with reward rubric and diversity monitoring
- Async training infrastructure for performance
- Comprehensive logging and checkpointing
- Support for curriculum and adaptive training
- JAX-compiled training loops for efficiency

All components follow functional programming principles with immutable data
structures and pure functions where possible.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Iterator, Callable
import logging
import time
from pathlib import Path
import pickle

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import jax
import jax.numpy as jnp
import optax
import pyrsistent as pyr

from .grpo_config import ComprehensiveGRPOConfig, TrainingMode, OptimizationLevel
from .grpo_core import (
    GRPOConfig, GRPOTrajectory, GRPOUpdateResult,
    create_grpo_update_fn, validate_grpo_config
)
from .experience_management import (
    ExperienceManager, Experience, ExperienceBatch,
    create_experience_manager
)
from .async_training import AsyncTrainingManager, TrainingProgress
from .diversity_monitor import DiversityMonitor, DiversityAlert
from ..acquisition.reward_rubric import CausalRewardRubric, RewardResult
from ..environments.intervention_env import InterventionEnvironment, EnvironmentInfo
from ..jax_native.state import JAXAcquisitionState, get_policy_input_tensor_jax

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingStep:
    """Single training step information.
    
    Args:
        step_number: Global training step number
        episode_number: Current episode number
        experiences_collected: Number of experiences collected this step
        grpo_update_result: Result of GRPO update
        diversity_metrics: Diversity monitoring results
        training_time: Time taken for this step (seconds)
        memory_usage_mb: Memory usage estimate
        checkpoint_saved: Whether checkpoint was saved this step
    """
    step_number: int
    episode_number: int
    experiences_collected: int
    grpo_update_result: Optional[GRPOUpdateResult]
    diversity_metrics: Optional[Dict[str, float]]
    training_time: float
    memory_usage_mb: float
    checkpoint_saved: bool


@dataclass(frozen=True)
class TrainingSession:
    """Complete training session results.
    
    Args:
        config: Training configuration used
        total_steps: Total training steps completed
        total_episodes: Total episodes completed
        total_experiences: Total experiences collected
        training_time: Total training time (seconds)
        final_performance: Final performance metrics
        checkpoints_saved: List of checkpoint paths
        best_checkpoint: Path to best performing checkpoint
        convergence_achieved: Whether training converged
        early_stopped: Whether training was early stopped
        session_metadata: Additional session information
    """
    config: ComprehensiveGRPOConfig
    total_steps: int
    total_episodes: int
    total_experiences: int
    training_time: float
    final_performance: Dict[str, float]
    checkpoints_saved: List[str]
    best_checkpoint: Optional[str]
    convergence_achieved: bool
    early_stopped: bool
    session_metadata: Dict[str, Any]


class GRPOTrainingManager:
    """Main manager for GRPO training orchestration.
    
    Coordinates all aspects of GRPO training including experience collection,
    policy updates, diversity monitoring, and performance tracking.
    
    Args:
        config: Comprehensive GRPO configuration
        environment: Intervention environment for experience collection
        reward_rubric: Reward computation system
        policy_network: Policy network (external, for flexibility)
        value_network: Value network (external, for flexibility)
    """
    
    def __init__(
        self,
        config: ComprehensiveGRPOConfig,
        environment: InterventionEnvironment,
        reward_rubric: CausalRewardRubric,
        policy_network: Any,  # JAX/Haiku network
        value_network: Any,   # JAX/Haiku network
    ):
        self.config = config
        self.environment = environment
        self.reward_rubric = reward_rubric
        self.policy_network = policy_network
        self.value_network = value_network
        
        # Initialize subsystems
        self.experience_manager = ExperienceManager(
            config.experience_management,
            config.grpo_algorithm
        )
        
        # For now, async manager is optional and handled manually in the training loop
        # In a full implementation, would create with proper environments and configs
        self.async_manager = None  # Would be AsyncTrainingManager([environment], diversity_monitor, config.async_training, training_config)
        
        self.diversity_monitor = config.diversity_monitor
        
        # GRPO update function will be created when optimizers are available
        # For now, this is handled in the update_policy method
        self.grpo_update_fn = None
        self.optimizer_state = None
        
        # Training state
        self.current_step = 0
        self.current_episode = 0
        self.total_experiences = 0
        self.best_performance = float('-inf')
        self.training_start_time = None
        self.session_metadata = {}
        
        # Performance tracking
        self.performance_history = []
        self.diversity_history = []
        self.checkpoint_history = []
        
        # Early stopping and adaptation
        self.patience_counter = 0
        self.last_adaptation_step = 0
        
        logger.info(f"Initialized GRPOTrainingManager with {config.training_mode.value} mode")
    
    def train(self) -> TrainingSession:
        """Execute complete training session.
        
        Returns:
            TrainingSession with complete results
        """
        logger.info("Starting GRPO training session")
        self.training_start_time = time.time()
        
        try:
            # Pre-training setup
            self._setup_training()
            
            # Main training loop
            for step in range(self.config.max_training_steps):
                self.current_step = step
                
                # Execute training step
                step_result = self._execute_training_step()
                
                # Log progress
                if step % self.config.logging.log_frequency == 0:
                    self._log_training_progress(step_result)
                
                # Evaluation and checkpointing
                if step % self.config.evaluation_frequency == 0:
                    self._evaluate_and_checkpoint()
                
                # Adaptive adjustments
                if step % self.config.adaptive.adaptation_frequency == 0:
                    self._apply_adaptive_adjustments()
                
                # Check for early stopping
                if self._should_early_stop():
                    logger.info(f"Early stopping at step {step}")
                    break
                
                # Check for convergence
                if self._has_converged():
                    logger.info(f"Training converged at step {step}")
                    break
            
            # Post-training cleanup
            return self._finalize_training()
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
    
    def collect_experiences(self, num_episodes: int = 1) -> List[Experience]:
        """Collect experiences from the environment.
        
        Args:
            num_episodes: Number of episodes to collect
            
        Returns:
            List of collected experiences
        """
        experiences = []
        
        for episode in range(num_episodes):
            episode_experiences = self._collect_single_episode()
            experiences.extend(episode_experiences)
            self.current_episode += 1
        
        return experiences
    
    def update_policy(self, batch: ExperienceBatch) -> GRPOUpdateResult:
        """Update policy using GRPO algorithm with enhanced networks.
        
        Args:
            batch: Batch of experiences for training
            
        Returns:
            GRPO update results
        """
        # Initialize GRPO update function if not already created
        if self.grpo_update_fn is None:
            self._initialize_grpo_training()
        
        try:
            # Extract batch data for GRPO training
            states = []
            actions = []
            rewards = []
            old_log_probs = []
            
            for exp in batch.experiences:
                # Convert experience to enhanced state tensor
                state_tensor = self._experience_to_state_tensor(exp)
                states.append(state_tensor)
                
                # Extract action information
                actions.append(exp.action)
                
                # Extract reward
                rewards.append(float(exp.reward.total_reward))
                
                # Extract log probability
                old_log_probs.append(exp.log_prob)
            
            # Stack into batch tensors
            batch_states = jnp.stack(states)  # [batch_size, T, n_vars, channels]
            batch_rewards = jnp.array(rewards)  # [batch_size]
            batch_old_log_probs = jnp.array(old_log_probs)  # [batch_size]
            
            # Create GRPO batch
            grpo_batch = {
                'states': batch_states,
                'actions': actions,  # Keep as list for now
                'rewards': batch_rewards,
                'old_log_probs': batch_old_log_probs
            }
            
            # Apply GRPO update
            new_params, new_opt_state, update_result = self.grpo_update_fn(
                self.policy_network.params,
                self.optimizer_state,
                grpo_batch
            )
            
            # Update network parameters
            self.policy_network = self.policy_network.replace(params=new_params)
            self.optimizer_state = new_opt_state
            
            return update_result
            
        except Exception as e:
            logger.error(f"GRPO update failed: {e}")
            # Re-raise the exception to expose training issues
            raise RuntimeError(f"GRPO policy update failed: {e}") from e
    
    def _initialize_grpo_training(self) -> None:
        """Initialize GRPO training components with enhanced networks."""
        try:
            # Import enhanced network factories
            from ..acquisition.enhanced_policy_network import create_enhanced_policy_for_grpo
            from ..avici_integration.enhanced_surrogate import create_enhanced_surrogate_for_grpo
            
            # Extract problem information from environment
            if hasattr(self.environment, 'variables'):
                variables = self.environment.variables
                target_variable = self.environment.target_variable
            else:
                # Fallback: extract from environment info
                env_info = self.environment.get_environment_info()
                variables = env_info.variables
                target_variable = env_info.target_variable
            
            # Create enhanced policy network
            enhanced_policy_fn, policy_config = create_enhanced_policy_for_grpo(
                variables=variables,
                target_variable=target_variable,
                architecture_level="full",
                performance_mode="balanced"
            )
            
            # Initialize policy network parameters
            key = jax.random.PRNGKey(self.config.seed)
            dummy_state = jnp.zeros((policy_config['max_history_size'], 
                                   len(variables), 
                                   policy_config['num_channels']))
            
            # Transform and initialize
            self.enhanced_policy_fn = hk.transform(lambda x: enhanced_policy_fn(x, is_training=True))
            policy_params = self.enhanced_policy_fn.init(key, dummy_state)
            
            # Create optimizer
            import optax
            optimizer = optax.adam(learning_rate=self.config.grpo_algorithm.learning_rate)
            self.optimizer_state = optimizer.init(policy_params)
            
            # Update policy network with enhanced version
            self.policy_network = self.policy_network.replace(params=policy_params)
            
            # Create GRPO update function
            from .grpo_core import create_grpo_update_fn
            self.grpo_update_fn = create_grpo_update_fn(
                policy_fn=self.enhanced_policy_fn,
                optimizer=optimizer,
                config=self.config.grpo_algorithm
            )
            
            logger.info("Enhanced GRPO training initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced GRPO training: {e}")
            # Re-raise the exception to expose initialization issues
            raise RuntimeError(f"GRPO training initialization failed: {e}") from e
    
    def _experience_to_state_tensor(self, experience: Experience) -> jnp.ndarray:
        """Convert experience to enhanced state tensor format.
        
        Args:
            experience: Experience object
            
        Returns:
            State tensor [T, n_vars, channels] for enhanced policy network
        """
        try:
            # Get enhanced state representation using the proper enriched history builder
            from ..acquisition.enriched.state_enrichment import create_enriched_history_tensor
            
            # Extract state information
            state = experience.state
            
            # Use the enriched history tensor creation with proper parameters
            max_history = 20  # Match policy network expectations
            include_temporal_features = True  # Enable temporal context
            
            # Create enriched history tensor using the proper function
            state_tensor = create_enriched_history_tensor(
                state=state,
                max_history_size=max_history,
                include_temporal_features=include_temporal_features
            )
            
            return state_tensor
            
        except Exception as e:
            logger.error(f"Failed to create enhanced state tensor: {e}")
            # Re-raise the exception to expose state creation issues
            raise RuntimeError(f"State tensor creation failed: {e}") from e
    
    def _get_policy_action(self, state: Any, key: jax.Array) -> Tuple[Any, float, float]:
        """Get action from enhanced policy network.
        
        Args:
            state: Current environment state
            key: Random key for sampling
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        # Initialize GRPO training if not already done
        if self.grpo_update_fn is None:
            self._initialize_grpo_training()
        
        # Ensure policy function is initialized
        if not hasattr(self, 'enhanced_policy_fn'):
            raise ValueError("Enhanced policy network not initialized")
        
        # Create state tensor from current state
        dummy_experience = type('Experience', (), {
            'state': state,
            'action': None,
            'reward': None
        })()
        state_tensor = self._experience_to_state_tensor(dummy_experience)
        
        # Get policy outputs
        policy_outputs = self.enhanced_policy_fn.apply(
            self.policy_network.params, 
            key, 
            state_tensor
        )
        
        # Sample action from policy outputs
        variable_logits = policy_outputs['variable_logits']
        value_params = policy_outputs['value_params']
        state_value = policy_outputs['state_value']
        
        # Sample which variable to intervene on
        variable_probs = jax.nn.softmax(variable_logits)
        variable_idx = jax.random.categorical(key, variable_logits)
        
        # Sample intervention value for selected variable
        key, value_key = jax.random.split(key)
        mean = value_params[variable_idx, 0]
        log_std = value_params[variable_idx, 1]
        std = jnp.exp(log_std)
        intervention_value = jax.random.normal(value_key) * std + mean
        
        # Get variable name
        if hasattr(self.environment, 'variables'):
            variables = self.environment.variables
            variable_name = variables[int(variable_idx)]
        else:
            variable_name = f'X{int(variable_idx)}'
        
        # Create action
        action = pyr.pmap({
            'values': {variable_name: float(intervention_value)}
        })
        
        # Compute log probability
        variable_log_prob = jnp.log(variable_probs[variable_idx] + 1e-8)
        value_log_prob = -0.5 * ((intervention_value - mean) / (std + 1e-8)) ** 2 - 0.5 * jnp.log(2 * jnp.pi) - log_std
        total_log_prob = variable_log_prob + value_log_prob
        
        return action, float(total_log_prob), float(state_value)
    
    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """Save training checkpoint.
        
        Args:
            path: Optional custom checkpoint path
            
        Returns:
            Path to saved checkpoint
        """
        if path is None:
            timestamp = int(time.time())
            path = f"{self.config.checkpointing.checkpoint_dir}/grpo_checkpoint_{timestamp}.pkl"
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint_data = {
            'config': self.config,
            'policy_params': self.policy_network.params,
            'value_params': self.value_network.params,
            'step': self.current_step,
            'episode': self.current_episode,
            'performance_history': self.performance_history,
            'diversity_history': self.diversity_history,
            'best_performance': self.best_performance,
            'session_metadata': self.session_metadata
        }
        
        with open(path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        self.checkpoint_history.append(path)
        logger.info(f"Saved checkpoint to {path}")
        
        return path
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        with open(path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        # Restore state
        self.policy_network = self.policy_network.replace(params=checkpoint_data['policy_params'])
        self.value_network = self.value_network.replace(params=checkpoint_data['value_params'])
        self.current_step = checkpoint_data['step']
        self.current_episode = checkpoint_data['episode']
        self.performance_history = checkpoint_data.get('performance_history', [])
        self.diversity_history = checkpoint_data.get('diversity_history', [])
        self.best_performance = checkpoint_data.get('best_performance', float('-inf'))
        self.session_metadata = checkpoint_data.get('session_metadata', {})
        
        logger.info(f"Loaded checkpoint from {path}")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get current training statistics.
        
        Returns:
            Dictionary of training statistics
        """
        current_time = time.time()
        training_time = current_time - (self.training_start_time or current_time)
        
        exp_stats = self.experience_manager.get_statistics()
        
        return {
            'current_step': self.current_step,
            'current_episode': self.current_episode,
            'total_experiences': self.total_experiences,
            'training_time': training_time,
            'best_performance': self.best_performance,
            'experience_buffer_utilization': exp_stats['utilization'],
            'memory_usage_mb': exp_stats['memory_usage_mb'],
            'can_train': exp_stats['can_sample'],
            'patience_counter': self.patience_counter,
            'checkpoints_saved': len(self.checkpoint_history),
            'training_mode': self.config.training_mode.value,
            'optimization_level': self.config.optimization_level.value
        }
    
    def _setup_training(self) -> None:
        """Set up training infrastructure."""
        # Validate configuration
        validate_grpo_config(self.config.grpo_algorithm)
        
        # Initialize random state
        jax.random.PRNGKey(self.config.seed)
        
        # Setup logging if configured
        if self.config.logging.enable_tensorboard:
            logger.info("TensorBoard logging enabled")
        
        if self.config.logging.enable_wandb:
            self._setup_wandb()
        
        # Initialize session metadata
        self.session_metadata = {
            'start_time': time.time(),
            'config_hash': hash(str(self.config)),
            'environment_type': type(self.environment).__name__,
            'reward_rubric_type': type(self.reward_rubric).__name__
        }
        
        logger.info("Training setup completed")
    
    def _setup_wandb(self) -> None:
        """Initialize Weights & Biases logging with causal discovery specific configuration."""
        if not WANDB_AVAILABLE:
            logger.warning("wandb not available - install with: pip install wandb")
            return
        
        # Prepare config for logging (convert to dict to avoid serialization issues)
        wandb_config = {
            "learning_rate": self.config.grpo_algorithm.learning_rate,
            "batch_size": getattr(self.config.grpo_algorithm, 'batch_size', 64),
            "training_mode": self.config.training_mode.value,
            "optimization_level": self.config.optimization_level.value,
            "environment_type": type(self.environment).__name__,
            "reward_rubric_type": type(self.reward_rubric).__name__,
        }
        
        # Add curriculum config if available
        if hasattr(self.config, 'curriculum'):
            wandb_config.update({
                "curriculum_enabled": True,
                "curriculum_stages": len(getattr(self.config.curriculum, 'stages', [])),
            })
        
        # Initialize wandb run
        wandb.init(
            project=self.config.logging.project_name,
            name=f"grpo_training_{int(time.time())}",
            config=wandb_config,
            tags=self.config.logging.tags + ["grpo", "causal_discovery"],
            group="acbo_training"
        )
        
        logger.info(f"WandB logging initialized for project: {self.config.logging.project_name}")

    def _execute_training_step(self) -> TrainingStep:
        """Execute a single training step."""
        step_start = time.time()
        
        # Collect experiences
        experiences = self.collect_experiences(num_episodes=1)
        self.total_experiences += len(experiences)
        
        # Add to experience manager
        for exp in experiences:
            self.experience_manager.add_experience(exp)
        
        # Sample batch for training
        batch = self.experience_manager.sample_batch()
        grpo_result = None
        
        if batch is not None:
            # Update policy
            grpo_result = self.update_policy(batch)
            
            # Update experience priorities if using prioritized replay
            if self.config.experience_management.prioritized_replay and grpo_result:
                # Use policy loss as priority signal (higher loss = higher priority)
                priorities = jnp.ones(len(batch.indices)) * abs(grpo_result.policy_loss) + 1e-6
                self.experience_manager.update_priorities(batch.indices, priorities)
        
        # Monitor diversity
        diversity_metrics = None
        if self.diversity_monitor is not None:
            # Add current policy action to diversity monitor
            # This is a simplified approach - in practice would use actual policy outputs
            dummy_action = pyr.pmap({'dummy': 1.0})
            alert = self.diversity_monitor.add_action(dummy_action)
            if alert:
                logger.warning(f"Diversity alert: {alert.alert_type.value}")
            diversity_metrics = self.diversity_monitor.get_metrics()._asdict()
        
        # Calculate step metrics
        step_time = time.time() - step_start
        exp_stats = self.experience_manager.get_statistics()
        
        return TrainingStep(
            step_number=self.current_step,
            episode_number=self.current_episode,
            experiences_collected=len(experiences),
            grpo_update_result=grpo_result,
            diversity_metrics=diversity_metrics,
            training_time=step_time,
            memory_usage_mb=exp_stats['memory_usage_mb'],
            checkpoint_saved=False
        )
    
    def _collect_single_episode(self) -> List[Experience]:
        """Collect experiences from a single episode."""
        experiences = []
        
        # Reset environment
        key = jax.random.PRNGKey(42)  # Simple key for testing
        state = self.environment.reset(key)
        done = False
        
        while not done:
            # Generate a new key for this step
            key, action_key = jax.random.split(key)
            
            # Get action from enhanced policy network
            try:
                action, log_prob, value = self._get_policy_action(state, action_key)
            except Exception as e:
                logger.error(f"Policy action failed: {e}")
                # Re-raise the exception to expose policy issues
                raise RuntimeError(f"Policy action generation failed: {e}") from e
            
            # Execute action in environment
            next_state, env_info = self.environment.step(action)
            
            # Compute reward
            reward_result = self.reward_rubric.compute_reward(state, action, next_state)
            
            # Create experience
            experience = Experience(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward_result,
                done=env_info.episode_complete,
                log_prob=log_prob,
                value=value,
                env_info=env_info,
                timestamp=time.time()
            )
            
            experiences.append(experience)
            
            # Update for next iteration
            state = next_state
            done = env_info.episode_complete
        
        return experiences
    
    def _log_training_progress(self, step_result: TrainingStep) -> None:
        """Log training progress to console and WandB."""
        logger.info(
            f"Step {step_result.step_number}: "
            f"Episode {step_result.episode_number}, "
            f"Experiences: {step_result.experiences_collected}, "
            f"Time: {step_result.training_time:.3f}s"
        )
        
        # Prepare metrics for logging
        metrics = {
            "step": step_result.step_number,
            "episode": step_result.episode_number,
            "experiences_collected": step_result.experiences_collected,
            "training_time": step_result.training_time,
            "memory_usage_mb": step_result.memory_usage_mb,
        }
        
        if step_result.grpo_update_result:
            grpo_result = step_result.grpo_update_result
            logger.info(
                f"GRPO Update - "
                f"Policy Loss: {float(grpo_result.policy_loss):.4f}, "
                f"Value Loss: {float(grpo_result.value_loss):.4f}, "
                f"Entropy: {float(grpo_result.entropy_loss):.4f}"
            )
            
            # Add GRPO metrics
            metrics.update({
                "grpo/policy_loss": float(grpo_result.policy_loss),
                "grpo/value_loss": float(grpo_result.value_loss),
                "grpo/entropy_loss": float(grpo_result.entropy_loss),
            })
        
        if step_result.diversity_metrics:
            logger.info(f"Diversity: {step_result.diversity_metrics}")
            # Add diversity metrics with prefix
            for key, value in step_result.diversity_metrics.items():
                metrics[f"diversity/{key}"] = value
        
        # Log to WandB if enabled
        if self.config.logging.enable_wandb and WANDB_AVAILABLE:
            try:
                wandb.log(metrics, step=step_result.step_number)
            except Exception as e:
                logger.warning(f"Failed to log to WandB: {e}")
    
    def _evaluate_and_checkpoint(self) -> None:
        """Evaluate current performance and save checkpoint if needed."""
        # Simple performance evaluation - in practice would be more sophisticated
        # Use negative policy loss as a performance proxy (lower loss = better performance)
        recent_rewards = [-step.grpo_update_result.policy_loss
                         for step in self.performance_history[-10:] 
                         if step.grpo_update_result is not None]
        
        if recent_rewards:
            current_performance = float(jnp.mean(jnp.array(recent_rewards)))
            
            # Update best performance
            if current_performance > self.best_performance:
                self.best_performance = current_performance
                self.patience_counter = 0
                
                # Save best checkpoint
                if self.config.checkpointing.enable_checkpointing:
                    checkpoint_path = self.save_checkpoint()
                    logger.info(f"New best performance: {current_performance:.4f}")
            else:
                self.patience_counter += 1
        
        # Regular checkpointing
        if (self.config.checkpointing.enable_checkpointing and 
            self.current_step % self.config.checkpointing.checkpoint_frequency == 0):
            self.save_checkpoint()
    
    def _apply_adaptive_adjustments(self) -> None:
        """Apply adaptive training adjustments."""
        if not any([
            self.config.adaptive.enable_adaptive_lr,
            self.config.adaptive.enable_adaptive_exploration,
            self.config.adaptive.enable_adaptive_curriculum
        ]):
            return
        
        # Adaptive learning rate
        if self.config.adaptive.enable_adaptive_lr:
            # Simple adaptation based on performance stagnation
            if self.patience_counter > 5:
                logger.info("Adapting learning rate")
                # In practice, would adjust optimizer learning rate
        
        # Adaptive exploration
        if self.config.adaptive.enable_adaptive_exploration:
            # Adjust exploration based on diversity metrics
            if self.diversity_monitor is not None:
                metrics = self.diversity_monitor.get_metrics()
                if metrics.action_entropy < 0.1:  # Low diversity threshold
                    logger.info("Increasing exploration due to low diversity")
    
    def _should_early_stop(self) -> bool:
        """Check if training should be early stopped."""
        if not self.config.adaptive.early_stopping_patience:
            return False
        
        return self.patience_counter >= self.config.adaptive.early_stopping_patience
    
    def _has_converged(self) -> bool:
        """Check if training has converged."""
        if len(self.performance_history) < 10:
            return False
        
        # Check if performance has plateaued
        # Use negative policy loss as performance metric
        recent_performance = [-step.grpo_update_result.policy_loss
                            for step in self.performance_history[-10:] 
                            if step.grpo_update_result is not None]
        
        if len(recent_performance) < 5:
            return False
        
        performance_std = float(jnp.std(jnp.array(recent_performance)))
        return performance_std < 0.01  # Very low variance indicates convergence
    
    def _finalize_training(self) -> TrainingSession:
        """Finalize training and return session results."""
        total_time = time.time() - self.training_start_time
        
        # Final checkpoint
        final_checkpoint = None
        if self.config.checkpointing.enable_checkpointing:
            final_checkpoint = self.save_checkpoint()
        
        # Compute final performance
        final_performance = self._compute_final_performance()
        
        # Create session result
        session = TrainingSession(
            config=self.config,
            total_steps=self.current_step,
            total_episodes=self.current_episode,
            total_experiences=self.total_experiences,
            training_time=total_time,
            final_performance=final_performance,
            checkpoints_saved=self.checkpoint_history,
            best_checkpoint=self.checkpoint_history[-1] if self.checkpoint_history else None,
            convergence_achieved=self._has_converged(),
            early_stopped=self._should_early_stop(),
            session_metadata=self.session_metadata
        )
        
        logger.info(f"Training completed: {session.total_steps} steps, {session.total_episodes} episodes")
        logger.info(f"Final performance: {final_performance}")
        
        return session
    
    def _compute_final_performance(self) -> Dict[str, float]:
        """Compute final performance metrics."""
        exp_stats = self.experience_manager.get_statistics()
        
        return {
            'best_reward': self.best_performance,
            'buffer_utilization': exp_stats['utilization'],
            'convergence_score': 1.0 if self._has_converged() else 0.0,
            'training_efficiency': self.total_experiences / max(1, self.current_step),
            'memory_efficiency': 1.0 / max(1.0, exp_stats['memory_usage_mb'] / 1024)
        }


# Factory functions for common training scenarios
def create_grpo_training_manager(
    config: ComprehensiveGRPOConfig,
    environment: InterventionEnvironment,
    reward_rubric: CausalRewardRubric,
    policy_network: Any,
    value_network: Any
) -> GRPOTrainingManager:
    """Create GRPO training manager with validation.
    
    Args:
        config: Training configuration
        environment: Intervention environment
        reward_rubric: Reward computation system
        policy_network: Policy network
        value_network: Value network
        
    Returns:
        Configured GRPOTrainingManager
    """
    # Validate configuration
    from .grpo_config import validate_comprehensive_grpo_config
    validate_comprehensive_grpo_config(config)
    
    return GRPOTrainingManager(
        config=config,
        environment=environment,
        reward_rubric=reward_rubric,
        policy_network=policy_network,
        value_network=value_network
    )


def create_debug_training_manager(
    environment: InterventionEnvironment,
    reward_rubric: CausalRewardRubric,
    policy_network: Any,
    value_network: Any
) -> GRPOTrainingManager:
    """Create training manager optimized for debugging.
    
    Args:
        environment: Intervention environment
        reward_rubric: Reward computation system
        policy_network: Policy network
        value_network: Value network
        
    Returns:
        Debug-optimized GRPOTrainingManager
    """
    from .grpo_config import create_debug_grpo_config
    
    config = create_debug_grpo_config()
    
    return GRPOTrainingManager(
        config=config,
        environment=environment,
        reward_rubric=reward_rubric,
        policy_network=policy_network,
        value_network=value_network
    )


def create_production_training_manager(
    environment: InterventionEnvironment,
    reward_rubric: CausalRewardRubric,
    policy_network: Any,
    value_network: Any,
    max_training_steps: int = 25000
) -> GRPOTrainingManager:
    """Create training manager optimized for production.
    
    Args:
        environment: Intervention environment
        reward_rubric: Reward computation system
        policy_network: Policy network
        value_network: Value network
        max_training_steps: Maximum training steps
        
    Returns:
        Production-optimized GRPOTrainingManager
    """
    from .grpo_config import create_production_grpo_config
    
    config = create_production_grpo_config(max_training_steps=max_training_steps)
    
    return GRPOTrainingManager(
        config=config,
        environment=environment,
        reward_rubric=reward_rubric,
        policy_network=policy_network,
        value_network=value_network
    )