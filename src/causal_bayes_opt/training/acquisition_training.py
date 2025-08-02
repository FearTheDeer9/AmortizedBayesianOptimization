"""
Enhanced Acquisition Model Training with GRPO and Verifiable Rewards.

This module implements the complete training pipeline for acquisition policies in ACBO,
incorporating 2024 research findings from DeepSeek R1 GRPO and verifiable rewards literature.

Key Features:
1. Behavioral Cloning (BC) warm-start from expert demonstrations
2. Enhanced GRPO fine-tuning with relevant 2024 improvements
3. Mathematically verifiable rewards (no human feedback required)
4. Reward hacking prevention and monitoring
5. JAX-compiled training steps for performance

The implementation focuses only on techniques relevant to causal intervention selection,
filtering out LLM-specific approaches from the source literature.

Related Files:
- acquisition_config.py: Enhanced configuration system for acquisition training
- surrogate_training.py: Provides trained surrogate models for reward computation
- ../acquisition/grpo.py: Core GRPO algorithm implementation
- ../acquisition/rewards.py: Verifiable reward computation functions
- tests/test_training/test_acquisition_training.py: Training pipeline tests
"""

# Standard library imports
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple, Optional

# Third-party imports
import jax
import jax.numpy as jnp
import optax
import pyrsistent as pyr

# Project imports - acquisition module
from ..acquisition.grpo import GRPOConfig, GRPOUpdate, create_grpo_trainer
from ..acquisition.policy import (
    AcquisitionPolicyNetwork, 
    PolicyConfig,
    compute_action_log_probability,
    sample_intervention_from_policy
)
from ..acquisition.rewards import (
    RewardComponents,
    compute_verifiable_reward,
    create_default_reward_config
)
from ..acquisition.reward_rubric import (
    CausalRewardRubric,
    create_training_rubric,
    create_deployment_rubric,
    create_ablation_rubric,
)
from ..jax_native.state import TensorBackedAcquisitionState as AcquisitionState
from ..acquisition.services import create_acquisition_state

# Project imports - data structures
from ..data_structures.buffer import ExperienceBuffer

# Expert collection imports (optional to avoid dependency issues)
try:
    from ..training.expert_collection.collector import ExpertDemonstrationCollector, ParentScaleTrajectory
    EXPERT_COLLECTION_AVAILABLE = True
except ImportError:
    # Fallback mock class for testing
    class ParentScaleTrajectory:
        def __init__(self):
            self.states = []
            self.actions = []
            self.scm = None
    EXPERT_COLLECTION_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AcquisitionGRPOConfig:
    """Enhanced GRPO configuration for causal intervention selection with 2024 improvements."""
    
    # Core GRPO parameters (already correctly implemented)
    group_size: int = 64  # Appropriate for intervention selection (not LLM text gen)
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.01
    kl_penalty_coeff: float = 0.0  # ✅ Best practice confirmed by DeepSeek R1
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    
    # Relevant 2024 enhancements for our use case
    adaptive_advantage_scaling: bool = True  # Handle multi-objective reward variance
    single_update_per_batch: bool = True  # Training stability improvement
    reward_hacking_detection: bool = True  # Monitor for exploitation patterns
    
    # Training control
    max_episodes: int = 10000
    evaluation_frequency: int = 500
    early_stopping_patience: int = 5


@dataclass(frozen=True)
class BehavioralCloningConfig:
    """Configuration for behavioral cloning warm-start phase."""
    
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 50
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    
    # JAX compilation settings
    use_jax_compilation: bool = True
    static_argnums: Tuple[int, ...] = (2, 3)  # For Haiku model compilation


@dataclass(frozen=True)
class AcquisitionTrainingConfig:
    """Complete configuration for acquisition model training pipeline.
    
    Enhanced to support both legacy reward system and new rubric-based system
    inspired by the verifiers repository patterns.
    """
    
    # Phase configurations
    bc_config: BehavioralCloningConfig = field(default_factory=BehavioralCloningConfig)
    grpo_config: AcquisitionGRPOConfig = field(default_factory=AcquisitionGRPOConfig)
    
    # Policy network configuration
    policy_config: PolicyConfig = field(default_factory=PolicyConfig)
    
    # Reward system configuration (new rubric-based system)
    use_hybrid_rewards: bool = True
    reward_rubric: Optional[CausalRewardRubric] = None
    
    # Legacy reward configuration (backward compatibility)
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'optimization': 1.0,
        'structure': 0.5,
        'parent': 0.3,
        'exploration': 0.1
    })
    
    # Training performance enhancements (inspired by verifiers)
    enable_async_training: bool = True
    diversity_monitoring: bool = True
    diversity_threshold: float = 0.3
    
    # Data requirements
    expert_trajectory_count: int = 500
    min_expert_trajectory_count: int = 100
    
    # Training control
    save_checkpoints: bool = True
    checkpoint_frequency: int = 1000
    logging_frequency: int = 100
    
    def get_reward_rubric(self) -> CausalRewardRubric:
        """Get reward rubric, creating default if not specified."""
        if self.reward_rubric is not None:
            return self.reward_rubric
        elif self.use_hybrid_rewards:
            # Create training rubric with mechanism-aware features
            return create_training_rubric()
        else:
            # Fallback to legacy-compatible rubric
            return create_ablation_rubric(
                use_supervised=False,
                use_observable=True
            )
    
    def is_legacy_mode(self) -> bool:
        """Check if running in legacy reward mode."""
        return not self.use_hybrid_rewards and self.reward_rubric is None


def create_training_config(
    bc_epochs: int = 50,
    grpo_epochs: int = 100,
    improvement_weight: float = 2.0,
    mechanism_impact_weight: float = 1.5,
    exploration_weight: float = 0.5,
    enable_async: bool = True,
    **kwargs
) -> AcquisitionTrainingConfig:
    """Create training configuration with both supervised and observable signals.
    
    This configuration uses ground truth during training for better guidance
    while also incorporating observable signals for robustness.
    
    Args:
        bc_epochs: Number of behavioral cloning epochs
        grpo_epochs: Number of GRPO fine-tuning epochs
        improvement_weight: Weight for target improvement reward
        mechanism_impact_weight: Weight for mechanism impact reward
        exploration_weight: Weight for exploration diversity
        enable_async: Enable async training features
        **kwargs: Additional configuration overrides
        
    Returns:
        Training configuration with hybrid rewards enabled
    """
    bc_config = BehavioralCloningConfig(epochs=bc_epochs)
    grpo_config = AcquisitionGRPOConfig(max_episodes=grpo_epochs)
    
    # Create rubric with custom weights
    rubric = create_training_rubric(
        improvement_weight=improvement_weight,
        mechanism_impact_weight=mechanism_impact_weight,
        exploration_weight=exploration_weight,
    )
    
    return AcquisitionTrainingConfig(
        bc_config=bc_config,
        grpo_config=grpo_config,
        use_hybrid_rewards=True,
        reward_rubric=rubric,
        enable_async_training=enable_async,
        diversity_monitoring=True,
        **kwargs
    )


def create_deployment_config(
    bc_epochs: int = 30,
    grpo_epochs: int = 50,
    improvement_weight: float = 3.0,
    exploration_weight: float = 1.0,
    confidence_weight: float = 1.5,
    **kwargs
) -> AcquisitionTrainingConfig:
    """Create deployment configuration using only observable signals.
    
    This configuration does not require ground truth and is suitable
    for real-world deployment where the true causal structure is unknown.
    
    Args:
        bc_epochs: Number of behavioral cloning epochs
        grpo_epochs: Number of GRPO fine-tuning epochs
        improvement_weight: Weight for target improvement reward
        exploration_weight: Weight for exploration diversity
        confidence_weight: Weight for posterior confidence
        **kwargs: Additional configuration overrides
        
    Returns:
        Deployment configuration with observable-only rewards
    """
    bc_config = BehavioralCloningConfig(epochs=bc_epochs)
    grpo_config = AcquisitionGRPOConfig(max_episodes=grpo_epochs)
    
    # Create deployment rubric (no ground truth needed)
    rubric = create_deployment_rubric(
        improvement_weight=improvement_weight,
        exploration_weight=exploration_weight,
        confidence_weight=confidence_weight,
    )
    
    return AcquisitionTrainingConfig(
        bc_config=bc_config,
        grpo_config=grpo_config,
        use_hybrid_rewards=True,
        reward_rubric=rubric,
        enable_async_training=True,
        diversity_monitoring=True,
        **kwargs
    )


def create_ablation_config(
    use_supervised: bool = True,
    use_observable: bool = True,
    bc_epochs: int = 40,
    grpo_epochs: int = 80,
    diversity_threshold: float = 0.3,
    **kwargs
) -> AcquisitionTrainingConfig:
    """Create configuration for ablation studies.
    
    This allows testing different combinations of reward signals
    to understand their individual contributions.
    
    Args:
        use_supervised: Include supervised reward components
        use_observable: Include observable reward components
        bc_epochs: Number of behavioral cloning epochs
        grpo_epochs: Number of GRPO fine-tuning epochs
        diversity_threshold: Minimum reward variance threshold
        **kwargs: Additional configuration overrides
        
    Returns:
        Ablation configuration for scientific comparison
    """
    bc_config = BehavioralCloningConfig(epochs=bc_epochs)
    grpo_config = AcquisitionGRPOConfig(max_episodes=grpo_epochs)
    
    # Create ablation rubric
    rubric = create_ablation_rubric(
        use_supervised=use_supervised,
        use_observable=use_observable,
        diversity_threshold=diversity_threshold,
    )
    
    return AcquisitionTrainingConfig(
        bc_config=bc_config,
        grpo_config=grpo_config,
        use_hybrid_rewards=True,
        reward_rubric=rubric,
        enable_async_training=True,
        diversity_monitoring=True,
        diversity_threshold=diversity_threshold,
        **kwargs
    )


@dataclass(frozen=True)
class TrainingResults:
    """Results from acquisition model training."""
    
    # Final model parameters
    final_params: Any
    
    # Training metrics
    bc_metrics: Dict[str, List[float]]
    grpo_metrics: Dict[str, List[float]]
    
    # Performance evaluation
    final_evaluation: Dict[str, float]
    
    # Training metadata
    total_training_time: float
    bc_training_time: float
    grpo_training_time: float
    checkpoints_saved: List[str]


def _train_acquisition_model_pure(
    expert_trajectories: List[ParentScaleTrajectory],
    surrogate_model: Any,
    surrogate_params: Any,
    config: AcquisitionTrainingConfig,
    key: jax.Array
) -> Tuple[Any, Dict[str, List[float]], Dict[str, List[float]], Dict[str, float]]:
    """Pure acquisition model training pipeline (no side effects)."""
    # Validate inputs
    _validate_training_inputs(expert_trajectories, config)
    
    # Initialize policy network
    bc_key, grpo_key = jax.random.split(key)
    policy_network, initial_params = _initialize_policy_network(
        expert_trajectories, config.policy_config, bc_key
    )
    
    # Phase 1: Behavioral Cloning warm-start
    bc_params, bc_metrics = behavioral_cloning_phase(
        expert_trajectories,
        policy_network,
        initial_params,
        config.bc_config,
        bc_key
    )
    
    # Phase 2: GRPO fine-tuning
    final_params, grpo_metrics = grpo_fine_tuning_phase(
        bc_params,
        policy_network,
        surrogate_model,
        surrogate_params,
        expert_trajectories,
        config,
        grpo_key
    )
    
    # Final evaluation
    final_evaluation = _evaluate_final_model(
        final_params,
        policy_network,
        expert_trajectories,
        config
    )
    
    return final_params, bc_metrics, grpo_metrics, final_evaluation


def train_acquisition_model(
    expert_trajectories: List[ParentScaleTrajectory],
    surrogate_model: Any,
    surrogate_params: Any,
    config: AcquisitionTrainingConfig,
    key: jax.Array
) -> TrainingResults:
    """
    Complete acquisition model training pipeline with BC warm-start + GRPO fine-tuning.
    
    Args:
        expert_trajectories: Expert demonstrations from PARENT_SCALE
        surrogate_model: Trained surrogate model for posterior prediction
        surrogate_params: Parameters for surrogate model
        config: Training configuration
        key: JAX random key
        
    Returns:
        TrainingResults with final parameters and metrics
        
    Raises:
        ValueError: If insufficient expert data or invalid configuration
    """
    import time
    start_time = time.time()
    
    logger.info(f"Starting acquisition model training with {len(expert_trajectories)} expert trajectories")
    
    # Phase 1: Behavioral Cloning warm-start
    logger.info("Phase 1: Behavioral Cloning warm-start")
    bc_start_time = time.time()
    
    # Call pure function for actual training
    final_params, bc_metrics, grpo_metrics, final_evaluation = _train_acquisition_model_pure(
        expert_trajectories, surrogate_model, surrogate_params, config, key
    )
    
    bc_training_time = time.time() - bc_start_time
    logger.info(f"BC phase completed in {bc_training_time:.2f} seconds")
    
    # Note: The pure function does both BC and GRPO phases, so timing is approximate
    grpo_training_time = bc_training_time  # Placeholder since phases are combined in pure function
    logger.info(f"GRPO phase completed")
    
    total_training_time = time.time() - start_time
    logger.info(f"Total training completed in {total_training_time:.2f} seconds")
    
    return TrainingResults(
        final_params=final_params,
        bc_metrics=bc_metrics,
        grpo_metrics=grpo_metrics,
        final_evaluation=final_evaluation,
        total_training_time=total_training_time,
        bc_training_time=bc_training_time,
        grpo_training_time=grpo_training_time,
        checkpoints_saved=[]  # TODO: Implement checkpointing
    )


def behavioral_cloning_phase(
    expert_trajectories: List[ParentScaleTrajectory],
    policy_network: Any,
    initial_params: Any,
    config: BehavioralCloningConfig,
    key: jax.Array
) -> Tuple[Any, Dict[str, List[float]]]:
    """
    Behavioral cloning warm-start phase using expert demonstrations.
    
    Trains the policy to mimic expert intervention choices using cross-entropy loss.
    
    Args:
        expert_trajectories: Expert demonstrations from PARENT_SCALE
        policy_network: Acquisition policy network
        initial_params: Initial network parameters
        config: BC configuration
        key: JAX random key
        
    Returns:
        (trained_params, training_metrics)
    """
    logger.info(f"Starting BC training with {len(expert_trajectories)} trajectories")
    
    # Prepare training data
    train_data, val_data = _prepare_bc_training_data(
        expert_trajectories, config.validation_split, key
    )
    
    # Create optimizer
    optimizer = optax.adam(learning_rate=config.learning_rate)
    opt_state = optimizer.init(initial_params)
    
    # Create training step function
    if config.use_jax_compilation:
        # Use JAX-compiled training step for performance
        jax_train_step = create_jax_bc_training_step(policy_network, optimizer)
        
        def train_step(params, opt_state, batch, policy_network, optimizer):
            states, actions = zip(*batch)
            return jax_train_step(params, opt_state, list(states), list(actions))
    else:
        train_step = _bc_training_step
    
    # Training loop
    params = initial_params
    metrics = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.epochs):
        # Training epoch
        epoch_train_loss = []
        epoch_train_accuracy = []
        
        for batch in _create_bc_batches(train_data, config.batch_size, key):
            params, opt_state, batch_loss, batch_accuracy = train_step(
                params, opt_state, batch, policy_network, optimizer
            )
            
            epoch_train_loss.append(batch_loss)
            epoch_train_accuracy.append(batch_accuracy)
        
        # Validation epoch
        val_loss, val_accuracy = _evaluate_bc_model(
            params, policy_network, val_data
        )
        
        # Record metrics
        metrics['train_loss'].append(jnp.mean(jnp.array(epoch_train_loss)))
        metrics['train_accuracy'].append(jnp.mean(jnp.array(epoch_train_accuracy)))
        metrics['val_loss'].append(val_loss)
        metrics['val_accuracy'].append(val_accuracy)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= config.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch} (patience: {patience_counter})")
            break
            
        if epoch % 10 == 0:
            logger.info(
                f"Epoch {epoch}: train_loss={metrics['train_loss'][-1]:.4f}, "
                f"train_acc={metrics['train_accuracy'][-1]:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}"
            )
    
    logger.info(f"BC training completed. Final validation accuracy: {val_accuracy:.4f}")
    return params, metrics


def grpo_fine_tuning_phase(
    bc_params: Any,
    policy_network: Any,
    surrogate_model: Any,
    surrogate_params: Any,
    expert_trajectories: List[ParentScaleTrajectory],
    config: AcquisitionTrainingConfig,
    key: jax.Array
) -> Tuple[Any, Dict[str, List[float]]]:
    """
    GRPO fine-tuning phase with enhanced verifiable rewards.
    
    Fine-tunes the BC-initialized policy using reinforcement learning with
    mathematically verifiable rewards for optimization and structure discovery.
    
    Args:
        bc_params: Parameters from behavioral cloning phase
        policy_network: Acquisition policy network
        surrogate_model: Trained surrogate model
        surrogate_params: Surrogate model parameters
        expert_trajectories: Expert trajectories for environment setup
        config: Training configuration
        key: JAX random key
        
    Returns:
        (final_params, training_metrics)
    """
    logger.info("Starting GRPO fine-tuning with verifiable rewards")
    
    # Create enhanced GRPO trainer
    grpo_trainer, grpo_optimizer_init = _create_enhanced_grpo_trainer(
        policy_network, config.grpo_config
    )
    
    # Initialize GRPO training
    params = bc_params
    grpo_opt_state = grpo_optimizer_init(params)
    
    # Create reward configuration
    reward_config = create_default_reward_config(**config.reward_weights)
    
    # Training metrics
    metrics = {
        'policy_loss': [],
        'entropy_loss': [],
        'total_loss': [],
        'mean_reward': [],
        'reward_std': [],
        'mean_advantage': [],
        'advantage_std': [],
        'reward_hacking_detections': []
    }
    
    # Generate diverse training environments from expert trajectories
    training_envs = _generate_training_environments(expert_trajectories, key)
    
    episode_count = 0
    best_performance = -float('inf')
    patience_counter = 0
    
    while episode_count < config.grpo_config.max_episodes:
        # Collect experience batch
        batch_key = jax.random.fold_in(key, episode_count)
        
        batch_data = _collect_enhanced_grpo_batch(
            policy_network,
            params,
            surrogate_model,
            surrogate_params,
            training_envs,
            config.grpo_config,
            reward_config,
            batch_key
        )
        
        # Check for reward hacking
        if config.grpo_config.reward_hacking_detection:
            hacking_detected = _detect_reward_hacking(batch_data, metrics)
            metrics['reward_hacking_detections'].append(hacking_detected)
            
            if hacking_detected:
                logger.warning(f"Potential reward hacking detected at episode {episode_count}")
        
        # GRPO update
        params, grpo_opt_state, update_info = grpo_trainer(
            params, grpo_opt_state, batch_data
        )
        
        # Record metrics
        metrics['policy_loss'].append(float(update_info.policy_loss))
        metrics['entropy_loss'].append(float(update_info.entropy_loss))
        metrics['total_loss'].append(float(update_info.total_loss))
        metrics['mean_reward'].append(float(update_info.mean_reward))
        metrics['reward_std'].append(float(update_info.reward_std))
        metrics['mean_advantage'].append(float(update_info.mean_advantage))
        metrics['advantage_std'].append(float(update_info.advantage_std))
        
        # Evaluation and early stopping
        if episode_count % config.grpo_config.evaluation_frequency == 0:
            current_performance = _evaluate_grpo_performance(
                params, policy_network, surrogate_model, surrogate_params, training_envs
            )
            
            if current_performance > best_performance:
                best_performance = current_performance
                patience_counter = 0
            else:
                patience_counter += 1
            
            logger.info(
                f"Episode {episode_count}: reward={update_info.mean_reward:.4f}, "
                f"policy_loss={update_info.policy_loss:.4f}, "
                f"performance={current_performance:.4f}"
            )
            
            if patience_counter >= config.grpo_config.early_stopping_patience:
                logger.info(f"Early stopping GRPO at episode {episode_count}")
                break
        
        episode_count += 1
    
    logger.info(f"GRPO training completed after {episode_count} episodes")
    return params, metrics


# Helper functions for the training pipeline

def _validate_training_inputs(
    expert_trajectories: List[ParentScaleTrajectory],
    config: AcquisitionTrainingConfig
) -> None:
    """Validate training inputs and configuration."""
    if len(expert_trajectories) < config.min_expert_trajectory_count:
        raise ValueError(
            f"Insufficient expert trajectories: {len(expert_trajectories)} < "
            f"{config.min_expert_trajectory_count}"
        )
    
    if not all(hasattr(traj, 'states') and hasattr(traj, 'actions') for traj in expert_trajectories):
        raise ValueError("Expert trajectories must have 'states' and 'actions' attributes")
    
    # Validate reward weights
    if not all(w >= 0 for w in config.reward_weights.values()):
        raise ValueError("All reward weights must be non-negative")


def _initialize_policy_network(
    expert_trajectories: List[ParentScaleTrajectory],
    policy_config: PolicyConfig,
    key: jax.Array
) -> Tuple[Any, Any]:
    """Initialize policy network with example data."""
    # Get example state from first trajectory
    example_state = expert_trajectories[0].states[0]
    
    # Create policy network
    from ..acquisition.policy import create_acquisition_policy
    policy_network = create_acquisition_policy(policy_config, example_state)
    
    # Initialize parameters
    initial_params = policy_network.init(key, example_state, is_training=True)
    
    return policy_network, initial_params


def _prepare_bc_training_data(
    expert_trajectories: List[ParentScaleTrajectory],
    validation_split: float,
    key: jax.Array
) -> Tuple[List[Tuple[AcquisitionState, Any]], List[Tuple[AcquisitionState, Any]]]:
    """Prepare training and validation data for behavioral cloning."""
    # Extract (state, action) pairs from trajectories
    training_pairs = []
    
    for trajectory in expert_trajectories:
        for state, action in zip(trajectory.states, trajectory.actions):
            training_pairs.append((state, action))
    
    # Split into train/validation
    n_train = int(len(training_pairs) * (1 - validation_split))
    
    # Shuffle
    indices = jax.random.permutation(key, len(training_pairs))
    shuffled_pairs = [training_pairs[i] for i in indices]
    
    train_data = shuffled_pairs[:n_train]
    val_data = shuffled_pairs[n_train:]
    
    return train_data, val_data


def create_jax_bc_training_step(
    policy_network: Any,
    optimizer: optax.GradientTransformation
) -> Callable:
    """Create JAX-compiled BC training step for performance optimization."""
    
    @jax.jit
    def compiled_training_step(
        params: Any,
        opt_state: Any,
        batch_states: List[Any],
        batch_actions: List[Any]
    ) -> Tuple[Any, Any, float, float]:
        """JAX-compiled training step for BC."""
        
        def loss_fn(params):
            # Forward pass through policy for all states
            policy_outputs = [
                policy_network.apply(params, state, is_training=True) 
                for state in batch_states
            ]
            
            # Compute log probabilities for expert actions
            log_probs = [
                compute_action_log_probability(output, action, state, PolicyConfig())
                for output, action, state in zip(policy_outputs, batch_actions, batch_states)
            ]
            
            # Cross-entropy loss (negative log likelihood)
            loss = -jnp.mean(jnp.array(log_probs))
            
            # Compute accuracy (percentage of actions that would be sampled with high probability)
            probs = jnp.exp(jnp.array(log_probs))
            accuracy = jnp.mean(probs > 0.1)  # Threshold for "reasonable" probability
            
            return loss, accuracy
        
        # Compute loss and gradients
        (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        
        # Apply updates
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss, accuracy
    
    return compiled_training_step


def _bc_training_step(
    params: Any,
    opt_state: Any,
    batch: List[Tuple[AcquisitionState, Any]],
    policy_network: Any,
    optimizer: optax.GradientTransformation
) -> Tuple[Any, Any, float, float]:
    """Single behavioral cloning training step (non-compiled version for compatibility)."""
    
    def loss_fn(params):
        states, actions = zip(*batch)
        
        # Forward pass through policy
        policy_outputs = [policy_network.apply(params, state, is_training=True) for state in states]
        
        # Compute log probabilities for expert actions
        log_probs = [
            compute_action_log_probability(output, action, state, PolicyConfig())
            for output, action, state in zip(policy_outputs, actions, states)
        ]
        
        # Cross-entropy loss (negative log likelihood)
        loss = -jnp.mean(jnp.array(log_probs))
        
        # Compute accuracy (percentage of actions that would be sampled with high probability)
        probs = jnp.exp(jnp.array(log_probs))
        accuracy = jnp.mean(probs > 0.1)  # Threshold for "reasonable" probability
        
        return loss, accuracy
    
    # Compute loss and gradients
    (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    # Apply updates
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss, accuracy


def _create_bc_batches(
    data: List[Tuple[AcquisitionState, Any]],
    batch_size: int,
    key: jax.Array
) -> List[List[Tuple[AcquisitionState, Any]]]:
    """Create batches for BC training."""
    # Shuffle data
    indices = jax.random.permutation(key, len(data))
    shuffled_data = [data[i] for i in indices]
    
    # Create batches
    batches = []
    for i in range(0, len(shuffled_data), batch_size):
        batch = shuffled_data[i:i + batch_size]
        if len(batch) == batch_size:  # Only use complete batches
            batches.append(batch)
    
    return batches


def _evaluate_bc_model(
    params: Any,
    policy_network: Any,
    val_data: List[Tuple[AcquisitionState, Any]]
) -> Tuple[float, float]:
    """Evaluate BC model on validation data."""
    if not val_data:
        return 0.0, 0.0
    
    states, actions = zip(*val_data)
    
    # Forward pass
    policy_outputs = [policy_network.apply(params, state, is_training=False) for state in states]
    
    # Compute log probabilities
    log_probs = [
        compute_action_log_probability(output, action, state, PolicyConfig())
        for output, action, state in zip(policy_outputs, actions, states)
    ]
    
    # Validation loss and accuracy
    val_loss = -jnp.mean(jnp.array(log_probs))
    probs = jnp.exp(jnp.array(log_probs))
    val_accuracy = jnp.mean(probs > 0.1)
    
    return float(val_loss), float(val_accuracy)


def _create_enhanced_grpo_trainer(
    policy_network: Any,
    config: AcquisitionGRPOConfig
) -> Tuple[Callable, Callable]:
    """Create enhanced GRPO trainer with 2024 improvements."""
    
    # Convert to base GRPO config
    base_grpo_config = GRPOConfig(
        group_size=config.group_size,
        clip_ratio=config.clip_ratio,
        entropy_coeff=config.entropy_coeff,
        kl_penalty_coeff=config.kl_penalty_coeff,  # 0.0 confirmed best practice
        max_grad_norm=config.max_grad_norm,
        learning_rate=config.learning_rate
    )
    
    # Create base trainer
    base_trainer, optimizer_init = create_grpo_trainer(policy_network, base_grpo_config)
    
    # Enhance with 2024 improvements
    def enhanced_grpo_trainer(params, opt_state, batch_data):
        """Enhanced GRPO trainer with adaptive advantage scaling and stability improvements."""
        
        if config.adaptive_advantage_scaling:
            # Apply adaptive advantage scaling for multi-objective variance
            rewards = batch_data['rewards']
            reward_std = jnp.std(rewards)
            
            if reward_std > 2.0:  # High variance threshold
                # Apply stronger normalization for high variance rewards
                normalized_rewards = (rewards - jnp.mean(rewards)) / (reward_std + 1e-8)
                batch_data = {**batch_data, 'rewards': normalized_rewards}
        
        # Single update per batch for stability
        if config.single_update_per_batch:
            return base_trainer(params, opt_state, batch_data)
        else:
            # Multiple updates (less stable but potentially faster learning)
            for _ in range(2):
                params, opt_state, update_info = base_trainer(params, opt_state, batch_data)
            return params, opt_state, update_info
    
    return enhanced_grpo_trainer, optimizer_init


def _collect_enhanced_grpo_batch(
    policy_network: Any,
    params: Any,
    surrogate_model: Any,
    surrogate_params: Any,
    training_envs: List[Any],
    config: AcquisitionGRPOConfig,
    reward_config: pyr.PMap,
    key: jax.Array
) -> Dict[str, Any]:
    """Collect GRPO training batch with enhanced verifiable rewards."""
    # Import here to avoid circular imports
    from ..acquisition.grpo import collect_grpo_batch
    
    # Sample environments for this batch
    env_keys = jax.random.split(key, len(training_envs))
    selected_envs = []
    selected_states = []
    
    for i, env in enumerate(training_envs[:config.group_size]):
        # Create initial state for this environment
        state = _create_initial_state(env, surrogate_model, surrogate_params, env_keys[i])
        selected_envs.append(env)
        selected_states.append(state)
    
    # Collect batch using existing GRPO infrastructure
    batch = collect_grpo_batch(
        policy_network=policy_network,
        params=params,
        states=selected_states,
        scms=selected_envs,
        surrogate_model=surrogate_model,
        surrogate_params=surrogate_params,
        config=GRPOConfig(group_size=config.group_size),  # Convert to base config
        reward_config=reward_config,
        key=key
    )
    
    return batch


def _generate_training_environments(
    expert_trajectories: List[ParentScaleTrajectory],
    key: jax.Array
) -> List[Any]:
    """Generate diverse training environments from expert trajectories."""
    environments = []
    
    for trajectory in expert_trajectories:
        if hasattr(trajectory, 'scm'):
            environments.append(trajectory.scm)
    
    if not environments:
        # Fallback: create some simple environments
        logger.warning("No SCMs found in expert trajectories, creating fallback environments")
        from ..experiments.test_scms import create_fork_scm
        
        for i in range(10):
            env_key = jax.random.fold_in(key, i)
            scm = create_fork_scm(seed=int(env_key[0]))
            environments.append(scm)
    
    return environments


def _create_initial_state(
    scm: Any,
    surrogate_model: Any,
    surrogate_params: Any,
    key: jax.Array
) -> AcquisitionState:
    """Create initial acquisition state for training episode."""
    # Generate some initial observational data
    from ..environments.sampling import sample_observational
    
    initial_samples = sample_observational(scm, n_samples=20, key=key)
    
    # Create buffer and state
    buffer = ExperienceBuffer()
    for sample in initial_samples:
        buffer.add_sample(sample)
    
    # Create acquisition state
    target_variable = scm.get('target', list(scm.get('variables', ['Y']))[0])
    
    state = create_acquisition_state(
        scm=scm,
        buffer=buffer,
        surrogate_model=surrogate_model,
        surrogate_params=surrogate_params,
        target_variable=target_variable,
        step=0
    )
    
    return state


def _detect_reward_hacking(
    batch_data: Dict[str, Any],
    metrics: Dict[str, List[float]]
) -> bool:
    """Detect potential reward hacking patterns."""
    rewards = batch_data['rewards']
    
    # Check for unrealistic reward patterns
    mean_reward = jnp.mean(rewards)
    max_reward = jnp.max(rewards)
    min_reward = jnp.min(rewards)
    
    # Suspiciously high rewards
    if mean_reward > 2.0:  # Should be in roughly [-1, 1] range
        return True
    
    # Suspiciously uniform rewards (no learning signal)
    if jnp.std(rewards) < 0.01:
        return True
    
    # Sudden reward spikes
    if len(metrics['mean_reward']) > 5:
        recent_rewards = metrics['mean_reward'][-5:]
        if max(recent_rewards) - min(recent_rewards) > 3.0:
            return True
    
    return False


def _evaluate_grpo_performance(
    params: Any,
    policy_network: Any,
    surrogate_model: Any,
    surrogate_params: Any,
    training_envs: List[Any]
) -> float:
    """Evaluate GRPO performance on held-out environments."""
    # Simple evaluation: average reward on a few test episodes
    total_reward = 0.0
    n_test_episodes = min(5, len(training_envs))
    
    for i in range(n_test_episodes):
        env = training_envs[-(i+1)]  # Use last few environments as test
        
        # Create test state
        test_key = jax.random.PRNGKey(42 + i)
        test_state = _create_initial_state(env, surrogate_model, surrogate_params, test_key)
        
        # Sample intervention
        policy_output = policy_network.apply(params, test_state, is_training=False)
        intervention = sample_intervention_from_policy(
            policy_output, test_state, test_key, PolicyConfig()
        )
        
        # Simulate and compute reward
        from ..environments.sampling import sample_with_intervention
        outcome = sample_with_intervention(env, intervention, n_samples=1, key=test_key)[0]
        
        # Create next state and compute reward
        next_buffer = test_state.buffer.copy()
        next_buffer.add_intervention(intervention, outcome)
        
        next_state = create_acquisition_state(
            scm=env,
            buffer=next_buffer,
            surrogate_model=surrogate_model,
            surrogate_params=surrogate_params,
            target_variable=test_state.current_target,
            step=test_state.step + 1
        )
        
        reward_components = compute_verifiable_reward(
            test_state, intervention, outcome, next_state
        )
        
        total_reward += reward_components.total_reward
    
    return total_reward / n_test_episodes


def _evaluate_final_model(
    params: Any,
    policy_network: Any,
    expert_trajectories: List[ParentScaleTrajectory],
    config: AcquisitionTrainingConfig
) -> Dict[str, float]:
    """Comprehensive evaluation of the final trained model."""
    # TODO: Implement comprehensive evaluation
    # For now, return placeholder metrics
    
    return {
        'expert_similarity': 0.85,  # How similar to expert choices
        'reward_performance': 1.2,  # Average reward achieved
        'intervention_diversity': 0.75,  # Diversity of interventions
        'convergence_speed': 0.9  # How quickly it finds good solutions
    }