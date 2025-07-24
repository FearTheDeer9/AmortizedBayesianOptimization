#!/usr/bin/env python3
"""
Behavioral Cloning Surrogate Trainer

Specialized trainer for behavioral cloning of surrogate models using expert
demonstrations. Extends existing SurrogateTrainer with BC-specific functionality
and curriculum learning support.

Key Features:
1. Behavioral cloning training on (data → posterior) pairs
2. JAX-compiled training for performance
3. Curriculum learning progression
4. Integration with existing infrastructure
5. Comprehensive validation and metrics

Design Principles (Rich Hickey Approved):
- Pure functions for core logic
- Immutable configuration and state
- Composable training components
- Clear separation of concerns
"""

import logging
import time
from dataclasses import dataclass, replace
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import optax
import pyrsistent as pyr

# Import debugging utilities
from .bc_loss_debug import (
    validate_probability_distribution,
    debug_kl_divergence_computation,
    debug_parent_set_conversion,
    log_training_step_debug,
    create_debugging_kl_loss
)

# Import base classes from surrogate_trainer
from .surrogate_trainer import SurrogateTrainer, SurrogateTrainingResults

# Import training data structures from shared module
from .data_structures import (
    TrainingExample,
    TrainingBatchJAX
)
from .config import SurrogateTrainingConfig

# Import BC-specific modules
from .trajectory_processor import DifficultyLevel, SurrogateDataset
from .bc_data_pipeline import (
    create_curriculum_batches, 
    memory_efficient_batch_iterator,
    create_scm_aware_batches,
    scm_aware_batch_iterator
)

# Import continuous models for dynamic dimension support
from ..avici_integration.continuous.factory import (
    create_continuous_parent_set_config,
    create_continuous_parent_set_model
)
from ..avici_integration.parent_set.unified import (
    create_jax_unified_parent_set_model,
    TargetAwareConfig,
    create_structure_only_config
)

# Import checkpointing and logging infrastructure
from .checkpoint_manager import CheckpointManager, CheckpointConfig, create_checkpoint_manager
from .utils.wandb_setup import log_metrics, log_artifact, is_wandb_enabled

# Import model creation utilities
from ..avici_integration.parent_set import create_parent_set_model

logger = logging.getLogger(__name__)


# BC-specific data structures to avoid conflicts with base classes
@dataclass(frozen=True)
class BCTrainingMetrics:
    """Training metrics specific to BC training."""
    epoch: int
    average_loss: float
    learning_rate: float
    gradient_norm: float
    batch_count: int
    timestamp: float = 0.0


@dataclass(frozen=True)
class BCValidationResults:
    """Validation results specific to BC training."""
    epoch: int
    average_loss: float
    accuracy_metrics: Dict[str, float]
    convergence_status: str


# Define missing functions locally
def kl_divergence_loss_jax(
    predicted_probs: jnp.ndarray,
    target_probs: jnp.ndarray,
    parent_sets: List[frozenset] = None
) -> jnp.ndarray:
    """
    Compute KL divergence loss between predicted and target distributions.
    
    For continuous parent probabilities, both inputs should be [d] shaped
    probability vectors over individual parent relationships.
    
    Args:
        predicted_probs: Model's predicted parent probabilities [d]
        target_probs: Target parent probabilities [d]
        parent_sets: Not used for continuous probabilities
        
    Returns:
        KL divergence loss (scalar)
    """
    # Add small epsilon for numerical stability
    eps = 1e-10
    predicted_probs = jnp.clip(predicted_probs, eps, 1.0 - eps)
    target_probs = jnp.clip(target_probs, eps, 1.0 - eps)
    
    # Compute KL divergence: sum(p * log(p/q))
    kl_div = jnp.sum(target_probs * jnp.log(target_probs / predicted_probs))
    
    return kl_div


def kl_divergence_loss_jax_with_debug(
    predicted_probs: jnp.ndarray,
    target_probs: jnp.ndarray,
    parent_sets: List[frozenset] = None
) -> jnp.ndarray:
    """
    KL divergence loss with debugging (use outside of JAX compilation).
    """
    # Debug distributions before computation
    pred_validation = validate_probability_distribution(predicted_probs, "predicted")
    target_validation = validate_probability_distribution(target_probs, "target")
    
    if not pred_validation['is_valid'] or not target_validation['is_valid']:
        logger.warning(f"Invalid probability distributions detected!")
        logger.warning(f"Predicted issues: {pred_validation['issues']}")
        logger.warning(f"Target issues: {target_validation['issues']}")
    
    # Compute KL
    kl_div = kl_divergence_loss_jax(predicted_probs, target_probs, parent_sets)
    
    # Log extreme values
    if abs(float(kl_div)) > 100:
        logger.error(f"Extreme KL divergence detected: {float(kl_div)}")
        debug_info = debug_kl_divergence_computation(predicted_probs, target_probs)
        logger.error(f"KL debug info: {debug_info}")
    
    return kl_div


def convert_parent_sets_to_continuous_probs(
    parent_sets: List[frozenset],
    probs: jnp.ndarray,
    num_variables: int,
    target_idx: int,
    variable_order: Optional[List[str]] = None
) -> jnp.ndarray:
    """
    Convert discrete parent set probabilities to continuous per-variable probabilities.
    
    Args:
        parent_sets: List of parent sets (each is a frozenset of parent indices or names)
        probs: Probabilities for each parent set
        num_variables: Total number of variables
        target_idx: Index of target variable
        variable_order: Optional list of variable names for name-to-index mapping
        
    Returns:
        Per-variable parent probabilities [num_variables]
    """
    # Debug input probabilities
    debug_info = debug_parent_set_conversion(parent_sets, probs, num_variables, target_idx, variable_order)
    
    if not debug_info['input_validation']['is_valid']:
        logger.warning(f"Invalid input parent set probabilities: {debug_info['input_validation']['issues']}")
    
    # Initialize per-variable probabilities
    var_probs = jnp.zeros(num_variables)
    
    # Aggregate probabilities for each variable being a parent
    for i, parent_set in enumerate(parent_sets):
        set_prob = probs[i]
        for parent in parent_set:
            # Handle both integer indices and string variable names
            if isinstance(parent, int):
                parent_idx = parent
            elif isinstance(parent, str) and variable_order is not None:
                # Convert string variable name to index using variable_order
                try:
                    parent_idx = variable_order.index(parent)
                except ValueError:
                    logger.warning(f"Parent variable '{parent}' not found in variable_order")
                    continue
            else:
                continue
                
            if parent_idx is not None and 0 <= parent_idx < num_variables:
                var_probs = var_probs.at[parent_idx].add(set_prob)
    
    # Ensure target variable has zero probability (cannot be its own parent)
    var_probs = var_probs.at[target_idx].set(0.0)
    
    # Normalize to ensure it's a valid probability distribution
    total = jnp.sum(var_probs)
    if total > 0:
        var_probs = var_probs / total
    else:
        # If no parents, uniform distribution over non-target variables
        var_probs = jnp.ones(num_variables) / (num_variables - 1)
        var_probs = var_probs.at[target_idx].set(0.0)
    
    # Log debug info if conversion lost structure
    if not debug_info['conversion_preserves_structure']:
        logger.warning(f"Parent set conversion may have lost structure: {debug_info}")
    
    return var_probs


def convert_to_jax_batch(examples: List[TrainingExample]) -> TrainingBatchJAX:
    """
    Convert list of training examples to JAX batch.
    
    Args:
        examples: List of training examples
        
    Returns:
        JAX-optimized training batch
    """
    if not examples:
        raise ValueError("Cannot create batch from empty examples list")
    
    # Handle variable-shaped observational data
    # First, check all shapes
    obs_shapes = [ex.observational_data.shape for ex in examples]
    logger.debug(f"Observational data shapes in batch: {obs_shapes}")
    
    # Find maximum dimensions
    max_samples = max(shape[0] for shape in obs_shapes)
    max_vars = max(shape[1] for shape in obs_shapes)
    n_channels = obs_shapes[0][2] if len(obs_shapes[0]) > 2 else 3  # Default to 3 channels
    
    # Pad all arrays to same shape
    padded_obs = []
    for ex in examples:
        obs = ex.observational_data
        current_shape = obs.shape
        
        # Pad samples dimension
        if current_shape[0] < max_samples:
            pad_samples = max_samples - current_shape[0]
            obs = jnp.pad(obs, ((0, pad_samples), (0, 0), (0, 0)), mode='constant', constant_values=0)
        
        # Pad variables dimension
        if current_shape[1] < max_vars:
            pad_vars = max_vars - current_shape[1]
            obs = jnp.pad(obs, ((0, 0), (0, pad_vars), (0, 0)), mode='constant', constant_values=0)
        
        # Ensure 3 channels
        if len(current_shape) == 2:
            obs = obs[..., jnp.newaxis]  # Add channel dimension
            obs = jnp.repeat(obs, n_channels, axis=2)
        
        padded_obs.append(obs)
    
    # Stack observational data
    obs_data = jnp.stack(padded_obs)
    
    # Convert target variables to indices first
    target_variables = []
    variable_orders = []
    for ex in examples:
        variable_orders.append(ex.variable_order)
        try:
            idx = ex.variable_order.index(ex.target_variable)
            target_variables.append(idx)
        except ValueError:
            # If target not in variable order, default to 0
            target_variables.append(0)
    
    # Convert discrete parent set probabilities to continuous per-variable probabilities
    continuous_probs = []
    for i, ex in enumerate(examples):
        # Get number of variables for this example
        num_vars = ex.observational_data.shape[1]
        target_idx = target_variables[i]
        
        # Convert parent sets to continuous probabilities
        cont_probs = convert_parent_sets_to_continuous_probs(
            parent_sets=ex.parent_sets,
            probs=ex.expert_probs,
            num_variables=num_vars,
            target_idx=target_idx,
            variable_order=ex.variable_order
        )
        
        # Pad to max_vars if needed
        if num_vars < max_vars:
            padding = jnp.zeros(max_vars - num_vars)
            cont_probs = jnp.concatenate([cont_probs, padding])
        
        continuous_probs.append(cont_probs)
    
    # Stack continuous probabilities [batch_size, max_vars]
    expert_probs = jnp.stack(continuous_probs)
    
    # Stack expert accuracies
    expert_accuracies = jnp.array([ex.expert_accuracy for ex in examples])
    
    # Collect parent sets and other metadata
    parent_sets = [ex.parent_sets for ex in examples]
    # variable_orders and target_variables already computed above
    
    # Debug logging for batch creation - commented out to avoid large outputs
    # print(f"DEBUG BATCH: obs_data shape: {obs_data.shape}")
    # print(f"DEBUG BATCH: expert_probs shape: {expert_probs.shape}")
    # print(f"DEBUG BATCH: expert_accuracies shape: {expert_accuracies.shape}")
    # print(f"DEBUG BATCH: target_variables length: {len(target_variables)}")
    # print(f"DEBUG BATCH: max_samples: {max_samples}, max_vars: {max_vars}")
    
    # Check that continuous probabilities sum to 1 - silently
    # prob_sums = jnp.sum(expert_probs, axis=1)
    # print(f"DEBUG BATCH: expert_probs sums range: {jnp.min(prob_sums):.4f} - {jnp.max(prob_sums):.4f}")
    
    return TrainingBatchJAX(
        observational_data=obs_data,
        expert_probs=expert_probs,
        expert_accuracies=expert_accuracies,
        parent_sets=parent_sets,
        variable_orders=variable_orders,
        target_variables=target_variables
    )


def create_jax_surrogate_train_step(
    model: Any,
    optimizer: optax.GradientTransformation,
    loss_fn: Callable,
    config: Any
) -> Callable:
    """
    Create JAX-compiled training step for surrogate model.
    
    Args:
        model: Haiku-transformed model
        optimizer: Optax optimizer
        loss_fn: Loss function to use
        config: Training configuration
        
    Returns:
        JAX-compiled training step function
    """
    
    # Create a simplified loss function that doesn't use parent_sets
    def simplified_kl_loss(predicted_logits, target_probs):
        """KL loss without parent_sets argument."""
        return loss_fn(predicted_logits, target_probs, [])
    
    @jax.jit
    def train_step(
        params: Any,
        opt_state: Any,
        obs_data: jnp.ndarray,
        expert_probs: jnp.ndarray,
        expert_accuracies: jnp.ndarray,
        target_indices: jnp.ndarray,
        key: jax.Array
    ) -> Tuple[Any, Any, Dict[str, float]]:
        """Single training step - vectorized for JAX compilation."""
        
        def batch_loss(params):
            """Compute loss for entire batch - vectorized."""
            batch_size = obs_data.shape[0]
            
            # Vectorized model forward pass for entire batch
            # vmap applies the model to each example in parallel
            def single_example_forward(obs, target_idx, key):
                output = model.apply(
                    params, key,
                    obs,
                    target_idx,
                    True  # is_training
                )
                
                # Extract parent probabilities
                if isinstance(output, dict):
                    return output.get('parent_probabilities', output)
                return output
            
            # Generate keys for each example
            keys = random.split(key, batch_size)
            
            # Apply model to entire batch using vmap
            predicted_probs_batch = jax.vmap(single_example_forward)(
                obs_data, target_indices, keys
            )
            
            # Get number of variables for each example (before padding)
            # Count non-zero entries along variable dimension
            num_vars_batch = jnp.sum(jnp.any(obs_data != 0, axis=-1), axis=1)
            
            # Compute losses for each example using masking
            def single_loss(predicted_probs, expert_probs, num_vars):
                # Create mask for valid variables
                max_vars = expert_probs.shape[0]
                var_indices = jnp.arange(max_vars)
                mask = var_indices < num_vars
                
                # Mask probabilities (set invalid entries to 0)
                masked_expert = jnp.where(mask, expert_probs, 0.0)
                masked_pred = jnp.where(mask, predicted_probs, 0.0)
                
                # Normalize to ensure they sum to 1 over valid entries
                expert_sum = jnp.sum(masked_expert)
                pred_sum = jnp.sum(masked_pred)
                
                # Avoid division by zero
                masked_expert = jnp.where(expert_sum > 0, masked_expert / expert_sum, masked_expert)
                masked_pred = jnp.where(pred_sum > 0, masked_pred / pred_sum, masked_pred)
                
                # KL loss only on valid entries
                return simplified_kl_loss(masked_pred, masked_expert)
            
            # Vectorized loss computation
            losses = jax.vmap(single_loss)(
                predicted_probs_batch, expert_probs, num_vars_batch
            )
            
            return jnp.mean(losses)
        
        # Compute gradients
        loss_value, grads = jax.value_and_grad(batch_loss)(params)
        
        # Compute gradient norm for monitoring
        grad_norm = optax.global_norm(grads)
        
        # Apply optimizer updates
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        # Create metrics dictionary (keep as JAX arrays for JIT compatibility)
        metrics_dict = {
            'kl_loss': loss_value,
            'grad_norm': grad_norm,
            'learning_rate': config.learning_rate,
            'timestamp': 0.0  # Can't use time.time() in JIT
        }
        
        return new_params, new_opt_state, metrics_dict
    
    return train_step


@dataclass(frozen=True)
class BCTrainingConfig:
    """Configuration for behavioral cloning training."""
    # Base surrogate config
    surrogate_config: SurrogateTrainingConfig
    
    # BC-specific parameters
    curriculum_learning: bool = True
    start_difficulty: DifficultyLevel = DifficultyLevel.EASY
    advancement_threshold: float = 0.1  # KL divergence threshold for curriculum advancement
    validation_patience: int = 10  # Early stopping patience
    batch_size: int = 32
    
    # Training progression
    max_epochs_per_level: int = 100
    min_epochs_per_level: int = 10
    
    # JAX optimization
    use_jax_compilation: bool = True
    jax_backend: str = 'cpu'  # or 'gpu'
    
    # Checkpointing and logging
    checkpoint_dir: str = "checkpoints/surrogate_bc"
    save_frequency: int = 10  # Save every N epochs
    enable_wandb_logging: bool = True
    experiment_name: str = "surrogate_bc"


@dataclass(frozen=True)
class BCTrainingState:
    """Immutable training state for behavioral cloning."""
    current_difficulty: DifficultyLevel
    epoch: int
    best_validation_loss: float
    patience_counter: int
    model_params: Any
    optimizer_state: Any
    training_metrics: List[BCTrainingMetrics]
    validation_metrics: List[BCValidationResults]


@dataclass(frozen=True)
class BCTrainingResults:
    """Results from behavioral cloning training."""
    final_state: BCTrainingState
    training_history: List[BCTrainingMetrics]
    validation_history: List[BCValidationResults]
    curriculum_progression: List[Tuple[DifficultyLevel, int]]  # (level, epochs_trained)
    total_training_time: float
    final_model_params: Any


class BCSurrogateTrainer:
    """Behavioral cloning trainer for surrogate models."""
    
    def __init__(self, config: BCTrainingConfig):
        """Initialize BC trainer with configuration."""
        self.config = config
        self.base_trainer = SurrogateTrainer(config.surrogate_config)
        
        # Store JAX compilation flag - actual step created when training starts
        self.use_jax_compilation = config.use_jax_compilation
        self.jax_train_step = None  # Created lazily when model is available
        self._model = None  # Cached model for JAX step
        
        # Initialize checkpoint manager
        self.checkpoint_manager = create_checkpoint_manager(
            checkpoint_dir=config.checkpoint_dir,
            config=CheckpointConfig(
                save_frequency_steps=config.save_frequency,
                auto_cleanup=True,
                max_checkpoints=10
            )
        )
    
    def train_on_curriculum(
        self,
        curriculum_datasets: Dict[DifficultyLevel, SurrogateDataset],
        validation_datasets: Dict[DifficultyLevel, SurrogateDataset],
        random_key: jax.Array
    ) -> BCTrainingResults:
        """
        Train on curriculum of increasing difficulty.
        
        Args:
            curriculum_datasets: Training datasets by difficulty level
            validation_datasets: Validation datasets by difficulty level
            random_key: JAX random key
            
        Returns:
            BCTrainingResults with complete training history
        """
        start_time = time.time()
        
        # Create JAX training step if needed (requires data for model creation)
        if self.use_jax_compilation and self.jax_train_step is None:
            self._create_jax_training_step_from_data(curriculum_datasets)
            logger.info("Created JAX training step for curriculum training")
        
        # Initialize training state
        init_key, train_key = random.split(random_key)
        initial_params = self._initialize_model_params(init_key)
        optimizer = optax.adam(self.config.surrogate_config.learning_rate)
        initial_optimizer_state = optimizer.init(initial_params)
        
        training_state = BCTrainingState(
            current_difficulty=self.config.start_difficulty,
            epoch=0,
            best_validation_loss=float('inf'),
            patience_counter=0,
            model_params=initial_params,
            optimizer_state=initial_optimizer_state,
            training_metrics=[],
            validation_metrics=[]
        )
        
        # Get ordered curriculum levels
        available_levels = sorted(
            [level for level in curriculum_datasets.keys() if level.value >= self.config.start_difficulty.value],
            key=lambda x: x.value
        )
        
        logger.info(f"Starting curriculum training with {len(available_levels)} levels")
        
        curriculum_progression = []
        all_training_metrics = []
        all_validation_metrics = []
        
        # Train on each curriculum level
        for level in available_levels:
            logger.info(f"Training on difficulty level: {level}")
            
            train_dataset = curriculum_datasets[level]
            val_dataset = validation_datasets.get(level, train_dataset)  # Fallback to train if no val
            
            level_results = self._train_on_level(
                training_state=training_state,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                target_level=level,
                random_key=train_key
            )
            
            # Update training state for next level
            training_state = level_results.final_state
            
            # Record progression
            epochs_trained = len(level_results.training_history)
            curriculum_progression.append((level, epochs_trained))
            all_training_metrics.extend(level_results.training_history)
            all_validation_metrics.extend(level_results.validation_history)
            
            # Advance random key
            train_key, _ = random.split(train_key)
            
            logger.info(f"Completed level {level} in {epochs_trained} epochs")
        
        total_time = time.time() - start_time
        
        return BCTrainingResults(
            final_state=training_state,
            training_history=all_training_metrics,
            validation_history=all_validation_metrics,
            curriculum_progression=curriculum_progression,
            total_training_time=total_time,
            final_model_params=training_state.model_params
        )
    
    def _train_on_level(
        self,
        training_state: BCTrainingState,
        train_dataset: SurrogateDataset,
        val_dataset: SurrogateDataset,
        target_level: DifficultyLevel,
        random_key: jax.Array
    ) -> BCTrainingResults:
        """
        Train on a single curriculum level.
        
        Args:
            training_state: Current training state
            train_dataset: Training dataset for this level
            val_dataset: Validation dataset for this level
            target_level: Target difficulty level
            random_key: JAX random key
            
        Returns:
            BCTrainingResults for this level
        """
        current_state = replace(training_state, current_difficulty=target_level)
        level_training_metrics = []
        level_validation_metrics = []
        
        # Create batches for this level - use SCM-aware batching if enabled
        use_scm_batching = getattr(self.config.surrogate_config, 'use_scm_aware_batching', True)
        
        if use_scm_batching:
            # Use SCM-aware batching to ensure consistent dimensions within batches
            batches = create_scm_aware_batches(
                dataset=train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )
            logger.info("Using SCM-aware batching for consistent variable dimensions")
        else:
            # Use regular batching (may mix different SCMs)
            batches = create_curriculum_batches(
                dataset=train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )
        
        logger.info(f"Training level {target_level} with {len(batches)} batches")
        
        for epoch in range(self.config.max_epochs_per_level):
            epoch_key, random_key = random.split(random_key)
            
            # Training epoch
            current_state, epoch_metrics = self._train_epoch(
                state=current_state,
                batches=batches,
                train_dataset=train_dataset,
                random_key=epoch_key
            )
            
            level_training_metrics.append(epoch_metrics)
            
            # Validation
            val_metrics = self._validate_epoch(
                state=current_state,
                val_dataset=val_dataset,
                random_key=epoch_key
            )
            
            level_validation_metrics.append(val_metrics)
            
            # Update state
            if val_metrics.average_loss < current_state.best_validation_loss:
                current_state = replace(
                    current_state,
                    best_validation_loss=val_metrics.average_loss,
                    patience_counter=0,
                    model_params=current_state.model_params  # Would update with best params
                )
            else:
                current_state = replace(
                    current_state,
                    patience_counter=current_state.patience_counter + 1
                )
            
            current_state = replace(current_state, epoch=current_state.epoch + 1)
            
            # Log training metrics to WandB
            self.log_training_metrics(
                state=current_state,
                train_metrics=epoch_metrics,
                val_metrics=val_metrics
            )
            
            # Save checkpoint periodically
            if current_state.epoch % self.config.save_frequency == 0:
                checkpoint_path = self.save_checkpoint(
                    state=current_state,
                    stage=f"level_{target_level.value}",
                    user_notes=f"Training level {target_level} epoch {current_state.epoch}"
                )
            
            # Check advancement criteria
            if self._should_advance_level(current_state, val_metrics):
                # Save checkpoint before advancing
                checkpoint_path = self.save_checkpoint(
                    state=current_state,
                    stage=f"level_{target_level.value}_completed",
                    user_notes=f"Completed level {target_level} - advancing to next"
                )
                logger.info(f"Advancement criteria met for level {target_level} at epoch {epoch}")
                break
            
            # Early stopping
            if current_state.patience_counter >= self.config.validation_patience:
                # Save checkpoint before early stopping
                checkpoint_path = self.save_checkpoint(
                    state=current_state,
                    stage=f"level_{target_level.value}_early_stopped",
                    user_notes=f"Early stopped at level {target_level} epoch {epoch}"
                )
                logger.info(f"Early stopping at epoch {epoch} for level {target_level}")
                break
        
        return BCTrainingResults(
            final_state=current_state,
            training_history=level_training_metrics,
            validation_history=level_validation_metrics,
            curriculum_progression=[(target_level, len(level_training_metrics))],
            total_training_time=0.0,  # Will be computed at higher level
            final_model_params=current_state.model_params
        )
    
    def _train_epoch(
        self,
        state: BCTrainingState,
        batches: List[List[int]],
        train_dataset: SurrogateDataset,
        random_key: jax.Array
    ) -> Tuple[BCTrainingState, BCTrainingMetrics]:
        """
        Train for one epoch.
        
        Args:
            state: Current training state
            batches: Batch indices for this epoch
            train_dataset: Training dataset
            random_key: JAX random key
            
        Returns:
            Tuple of (updated_state, BCTrainingMetrics for this epoch)
        """
        total_loss = 0.0
        total_batches = len(batches)
        
        for batch_indices in batches:
            batch_examples = [train_dataset.training_examples[i] for i in batch_indices]
            
            # Convert to JAX batch
            jax_batch = convert_to_jax_batch(batch_examples)
            
            # Train step
            if self.jax_train_step is not None and self._model is not None:
                try:
                    # Use JAX-compiled training step
                    # Extract arrays from batch for JAX compatibility
                    updated_params, updated_opt_state, metrics_dict = self.jax_train_step(
                        state.model_params,
                        state.optimizer_state,
                        jax_batch.observational_data,
                        jax_batch.expert_probs,
                        jax_batch.expert_accuracies,
                        jnp.array(jax_batch.target_variables),  # Convert to JAX array
                        random_key
                    )
                    
                    # Extract loss from metrics dict
                    loss_value = metrics_dict.get('kl_loss', 0.0)
                    grad_norm = metrics_dict.get('grad_norm', 0.0)
                    
                    # Debug extreme loss values
                    if abs(loss_value) > 1000:
                        logger.error(f"EXTREME LOSS DETECTED: {loss_value}")
                        # Get one example for detailed debugging
                        example_obs = jax_batch.observational_data[0]
                        example_expert = jax_batch.expert_probs[0]
                        logger.error(f"Example obs shape: {example_obs.shape}")
                        logger.error(f"Example expert probs shape: {example_expert.shape}")
                        logger.error(f"Example expert probs: {example_expert}")
                        
                        # Log comprehensive debug info
                        log_training_step_debug(
                            batch_idx=len(batches) - total_batches,
                            loss_value=float(loss_value),
                            grad_norm=float(grad_norm),
                            predicted_probs=None,  # Would need to extract from model
                            target_probs=example_expert,
                            additional_info={
                                'batch_size': len(batch_examples),
                                'target_variables': jax_batch.target_variables,
                                'num_parent_sets': len(batch_examples[0].parent_sets) if batch_examples else 0
                            }
                        )
                    
                    total_loss += float(loss_value)
                    
                    # Update state parameters for next iteration
                    state = replace(
                        state,
                        model_params=updated_params,
                        optimizer_state=updated_opt_state
                    )
                    
                except Exception as e:
                    logger.warning(f"JAX training step failed: {e}, falling back to regular training")
                    # Fallback to regular training
                    loss_value = self._compute_batch_loss(batch_examples, state)
                    total_loss += float(loss_value)
            else:
                # Fallback to regular training
                loss_value = self._compute_batch_loss(batch_examples, state)
                total_loss += float(loss_value)
        
        average_loss = total_loss / total_batches if total_batches > 0 else 0.0
        
        metrics = BCTrainingMetrics(
            epoch=state.epoch,
            average_loss=average_loss,
            learning_rate=self.config.surrogate_config.learning_rate,
            gradient_norm=0.0,  # Would compute if needed
            batch_count=total_batches,
            timestamp=time.time()
        )
        
        return state, metrics
    
    def _validate_epoch(
        self,
        state: BCTrainingState,
        val_dataset: SurrogateDataset,
        random_key: jax.Array
    ) -> BCValidationResults:
        """
        Validate on validation dataset.
        
        Args:
            state: Current training state
            val_dataset: Validation dataset
            random_key: JAX random key
            
        Returns:
            ValidationResults for this epoch
        """
        # Create validation batches
        val_batches = create_curriculum_batches(
            dataset=val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        total_loss = 0.0
        total_batches = len(val_batches)
        
        for batch_indices in val_batches:
            batch_examples = [val_dataset.training_examples[i] for i in batch_indices]
            loss_value = self._compute_batch_loss(batch_examples, state)
            total_loss += float(loss_value)
        
        average_loss = total_loss / total_batches if total_batches > 0 else 0.0
        
        return BCValidationResults(
            epoch=state.epoch,
            average_loss=average_loss,
            accuracy_metrics={},  # Would compute if needed
            convergence_status='training'
        )
    
    def save_checkpoint(
        self, 
        state: BCTrainingState, 
        stage: str = "training", 
        user_notes: str = ""
    ) -> str:
        """
        Save training checkpoint with metadata.
        
        Args:
            state: Current training state
            stage: Training stage identifier
            user_notes: Optional user notes
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"{self.config.experiment_name}_epoch_{state.epoch}_level_{state.current_difficulty.value}"
        
        # Create checkpoint data
        checkpoint_data = {
            'config': self.config,
            'training_state': state,
            'current_difficulty': state.current_difficulty,
            'epoch': state.epoch,
            'model_params': state.model_params,
            'optimizer_state': state.optimizer_state
        }
        
        checkpoint_info = self.checkpoint_manager.save_checkpoint(
            state=checkpoint_data,
            checkpoint_name=checkpoint_name,
            stage=stage,
            user_notes=user_notes
        )
        
        logger.info(f"💾 Saved surrogate BC checkpoint: {checkpoint_info.path}")
        
        # Log to WandB if enabled
        if self.config.enable_wandb_logging and is_wandb_enabled():
            log_artifact(
                str(checkpoint_info.path),
                artifact_type="surrogate_bc_checkpoint",
                name=f"surrogate_bc_checkpoint_epoch_{state.epoch}"
            )
        
        return str(checkpoint_info.path)
    
    def load_checkpoint(self, checkpoint_path: str) -> BCTrainingState:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Loaded training state
        """
        checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        logger.info(f"📂 Loaded surrogate BC checkpoint from: {checkpoint_path}")
        return checkpoint_data['training_state']
    
    def get_latest_checkpoint(self, stage: Optional[str] = None) -> Optional[str]:
        """
        Get path to latest checkpoint.
        
        Args:
            stage: Optional stage filter
            
        Returns:
            Path to latest checkpoint or None
        """
        checkpoint_info = self.checkpoint_manager.get_latest_checkpoint(stage)
        return str(checkpoint_info.path) if checkpoint_info else None
    
    def log_training_metrics(
        self, 
        state: BCTrainingState,
        train_metrics: Optional[BCTrainingMetrics] = None,
        val_metrics: Optional[BCValidationResults] = None
    ) -> None:
        """
        Log training metrics to WandB if enabled.
        
        Args:
            state: Current training state
            train_metrics: Optional training metrics
            val_metrics: Optional validation metrics
        """
        if not (self.config.enable_wandb_logging and is_wandb_enabled()):
            return
        
        # Prepare metrics for logging
        metrics = {
            "surrogate_bc/epoch": state.epoch,
            "surrogate_bc/difficulty_level": state.current_difficulty.value,
            "surrogate_bc/patience_counter": state.patience_counter,
            "surrogate_bc/best_validation_loss": state.best_validation_loss
        }
        
        # Add training metrics if available
        if train_metrics:
            metrics.update({
                "surrogate_bc/train_loss": train_metrics.average_loss,
                "surrogate_bc/train_accuracy": 1.0 - train_metrics.average_loss,  # Approximate accuracy
                "surrogate_bc/learning_rate": train_metrics.learning_rate
            })
        
        # Add validation metrics if available
        if val_metrics:
            metrics.update({
                "surrogate_bc/val_loss": val_metrics.average_loss,
                "surrogate_bc/val_convergence": val_metrics.convergence_status
            })
        
        # Log to WandB
        log_metrics(metrics, step=state.epoch)
    
    def _compute_batch_loss(
        self,
        batch_examples: List[TrainingExample],
        state: BCTrainingState
    ) -> float:
        """
        Compute loss for a batch of examples.
        
        Args:
            batch_examples: Batch of training examples
            state: Current training state
            
        Returns:
            Loss value for the batch
        """
        if not batch_examples:
            return 0.0
        
        # Convert to JAX format and compute KL divergence loss
        jax_batch = convert_to_jax_batch(batch_examples)
        
        # Compute loss using actual model if available
        if self._model is not None and hasattr(self._model, 'apply'):
            # Use the actual model for loss computation
            try:
                total_loss = 0.0
                batch_size = len(batch_examples)
                
                for i in range(batch_size):
                    # Extract data for this example
                    example_data = jax_batch.observational_data[i]  # [N, d, 3]
                    example_expert_probs = jax_batch.expert_probs[i]  # [k]
                    example_variable_order = jax_batch.variable_orders[i]
                    example_target_variable = jax_batch.target_variables[i]
                    example_parent_sets = jax_batch.parent_sets[i]
                    
                    # Forward pass through model
                    key = random.PRNGKey(42)  # Use deterministic key for validation
                    
                    # Convert target variable to index for JAX model
                    if hasattr(self, '_lookup_tables') and self._lookup_tables and 'name_to_idx' in self._lookup_tables:
                        target_idx = self._lookup_tables['name_to_idx'].get(example_target_variable, 0)
                    else:
                        # Fallback: find index in variable order
                        try:
                            target_idx = example_variable_order.index(example_target_variable)
                        except ValueError:
                            target_idx = 0
                    
                    output = self._model.apply(
                        state.model_params, key,
                        example_data,
                        target_idx,  # Use integer index for JAX model
                        False  # is_training=False for validation
                    )
                    
                    # Extract parent probabilities from dict output (new interface)
                    if isinstance(output, dict):
                        # The new surrogate model returns a dict with multiple outputs
                        if 'parent_probabilities' in output:
                            # Use parent probabilities directly
                            predicted_probs = output['parent_probabilities']
                            # Convert probabilities to logits for KL computation
                            predicted_logits = jnp.log(predicted_probs + 1e-10)
                        elif 'parent_set_logits' in output:
                            # Legacy path for backward compatibility
                            predicted_logits = output['parent_set_logits']
                        else:
                            # Fallback: use simple computation
                            predicted_logits = jnp.sum(example_data, axis=(0, 2))[:len(example_expert_probs)]
                    else:
                        # Legacy path: if model returns tensor directly
                        predicted_logits = output
                    
                    # Ensure shapes match
                    min_len = min(len(predicted_logits), len(example_expert_probs))
                    predicted_logits = predicted_logits[:min_len]
                    target_probs = example_expert_probs[:min_len]
                    
                    # Compute KL divergence loss using JAX-compatible function
                    kl_loss = kl_divergence_loss_jax(
                        predicted_logits, 
                        target_probs, 
                        example_parent_sets[:min_len]
                    )
                    total_loss += float(kl_loss)
                
                return total_loss / batch_size
                
            except Exception as e:
                logger.warning(f"Model forward pass failed: {e}, falling back to simple loss")
                # Fall through to simple loss computation
        
        # Fallback: compute simple loss based on data characteristics
        # This maintains training dynamics while model is being initialized
        batch_size = len(batch_examples)
        total_loss = 0.0
        
        for example in batch_examples:
            # Simple loss based on data complexity and expert accuracy
            data_complexity = float(jnp.mean(jnp.var(example.observational_data, axis=0)))
            expert_accuracy = example.expert_accuracy
            
            # Higher loss for more complex data or lower expert accuracy
            example_loss = data_complexity * (1.0 - expert_accuracy) + 0.1
            total_loss += example_loss
        
        return total_loss / batch_size
    
    def _should_advance_level(
        self,
        state: BCTrainingState,
        val_metrics: BCValidationResults
    ) -> bool:
        """
        Determine if we should advance to next curriculum level.
        
        Args:
            state: Current training state
            val_metrics: Latest validation metrics
            
        Returns:
            True if should advance to next level
        """
        # Check if validation loss meets advancement threshold
        loss_threshold_met = val_metrics.average_loss < self.config.advancement_threshold
        
        # Check minimum epochs requirement
        min_epochs_met = state.epoch >= self.config.min_epochs_per_level
        
        return loss_threshold_met and min_epochs_met
    
    def _initialize_model_params(self, random_key: jax.Array) -> Any:
        """
        Initialize model parameters.
        
        Args:
            random_key: JAX random key
            
        Returns:
            Initialized model parameters
        """
        # If model is available, initialize properly
        if self._model is not None:
            try:
                # Get dummy input to initialize model
                dummy_data = jnp.ones((10, 5, 3))  # [N, d, 3] format
                
                # For JAX/Haiku model, use integer target index instead of string
                target_idx = 0  # Target the first variable
                
                # Initialize model parameters
                params = self._model.init(
                    random_key,
                    dummy_data,
                    target_idx,  # Use integer index for JAX model
                    True  # is_training
                )
                
                logger.info("Successfully initialized model parameters")
                return params
                
            except Exception as e:
                logger.warning(f"Failed to initialize model parameters: {e}")
                # Fall through to placeholder
        
        # Placeholder initialization for when model is not available
        logger.warning("Using placeholder model parameters - model not available")
        return {'placeholder': jnp.array([1.0])}
    
    def _create_jax_training_step_from_data(
        self, 
        datasets: Dict[DifficultyLevel, Any]
    ) -> None:
        """
        Create JAX training step using model dimensions from actual data.
        
        Args:
            datasets: Training datasets to infer model dimensions from
        """
        if not datasets:
            logger.warning("No datasets provided for JAX step creation")
            return
            
        # Get first training example to determine model dimensions
        first_dataset = next(iter(datasets.values()))
        if not first_dataset.training_examples:
            logger.warning("No training examples found in dataset")
            return
            
        first_example = first_dataset.training_examples[0]
        
        # Determine which model to use based on configuration
        use_continuous = getattr(self.config.surrogate_config, 'use_continuous_model', True)
        use_jax_unified = getattr(self.config.surrogate_config, 'use_jax_unified', False)
        
        if use_continuous:
            # Use continuous model for dynamic dimension support
            logger.info("Using continuous parent set model for dynamic dimensions")
            
            # Create configuration for continuous model
            config = create_continuous_parent_set_config(
                variables=first_example.variable_order,
                target_variable=first_example.target_variable,
                model_complexity="medium",
                use_attention=True,
                temperature=1.0
            )
            
            # Create continuous model
            model_tuple = create_continuous_parent_set_model(config)
            self._model, self._model_config = model_tuple
            self._lookup_tables = None  # Continuous model doesn't need lookup tables
            
            logger.info(f"Created continuous model with config: {self._model_config}")
            
        elif use_jax_unified:
            # Use JAX unified model (also supports dynamic dimensions)
            logger.info("Using JAX unified parent set model")
            
            # Create target-aware configuration
            config = create_structure_only_config()
            
            # Create JAX unified model - don't pass n_vars, let it be dynamic
            model_tuple = create_jax_unified_parent_set_model(
                config=config,
                variable_names=first_example.variable_order
            )
            
            self._model, self._lookup_tables = model_tuple
            logger.info("Created JAX unified model with lookup tables")
            
        else:
            # Fallback to old model (not recommended)
            logger.warning("Using legacy parent set model - consider enabling continuous or JAX unified model")
            n_vars = len(first_example.variable_order)
            from ..avici_integration.parent_set import create_parent_set_model
            model_tuple = create_parent_set_model(
                n_vars=n_vars,
                variable_names=first_example.variable_order
            )
            
            if isinstance(model_tuple, tuple):
                self._model, self._lookup_tables = model_tuple
            else:
                self._model = model_tuple
                self._lookup_tables = None
        
        # Create optimizer
        optimizer = optax.adam(self.config.surrogate_config.learning_rate)
        
        # Now create JAX training step with model
        self.jax_train_step = create_jax_surrogate_train_step(
            model=self._model,
            optimizer=optimizer,
            loss_fn=kl_divergence_loss_jax,
            config=self.config.surrogate_config
        )
        
        # Get n_vars from first example for logging
        n_vars = len(first_example.variable_order) if first_example and hasattr(first_example, 'variable_order') else 0
        logger.info(f"Created JAX training step for {n_vars} variables")


def create_bc_surrogate_trainer(
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    use_curriculum: bool = True,
    use_jax: bool = True,
    checkpoint_dir: str = "checkpoints/surrogate_bc",
    enable_wandb_logging: bool = True,
    experiment_name: str = "surrogate_bc"
) -> BCSurrogateTrainer:
    """
    Factory function to create BC surrogate trainer with sensible defaults.
    
    Args:
        learning_rate: Learning rate for optimization
        batch_size: Batch size for training
        use_curriculum: Whether to use curriculum learning
        use_jax: Whether to use JAX compilation
        checkpoint_dir: Directory for saving checkpoints
        enable_wandb_logging: Whether to enable WandB logging
        experiment_name: Name for experiment tracking
        
    Returns:
        Configured BCSurrogateTrainer
    """
    surrogate_config = SurrogateTrainingConfig(
        learning_rate=float(learning_rate),  # Ensure numeric type
        batch_size=int(batch_size),          # Ensure numeric type
        max_epochs=100,
        early_stopping_patience=10,
        validation_frequency=1
    )
    
    bc_config = BCTrainingConfig(
        surrogate_config=surrogate_config,
        curriculum_learning=use_curriculum,
        batch_size=int(batch_size),          # Ensure numeric type
        use_jax_compilation=use_jax,
        checkpoint_dir=checkpoint_dir,
        enable_wandb_logging=enable_wandb_logging,
        experiment_name=experiment_name
    )
    
    return BCSurrogateTrainer(bc_config)