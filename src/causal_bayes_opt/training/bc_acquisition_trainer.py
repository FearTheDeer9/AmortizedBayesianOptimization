#!/usr/bin/env python3
"""
Behavioral Cloning Acquisition Trainer

Specialized trainer for behavioral cloning of acquisition policies using expert
intervention demonstrations. Trains policies to mimic expert intervention choices.

Key Features:
1. Behavioral cloning training on (state → action) pairs
2. Cross-entropy loss for intervention selection
3. JAX-compiled training for performance
4. Curriculum learning support
5. Variable-length intervention history handling

Design Principles (Rich Hickey Approved):
- Pure functions for action prediction
- Immutable state and configuration
- Composable training components
- Clear separation of policy and training logic
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
import haiku as hk
import pyrsistent as pyr

# Import existing acquisition infrastructure
from ..acquisition.policy import AcquisitionPolicyNetwork, PolicyConfig
from ..acquisition.state import AcquisitionState
from .acquisition_training import (
    AcquisitionTrainingConfig,
    behavioral_cloning_phase,
    TrainingResults
)

# Import EnhancedPolicyNetwork for dynamic dimension support
from ..acquisition.enhanced_policy_network import EnhancedPolicyNetwork

# Import BC-specific modules
from .trajectory_processor import (
    DifficultyLevel,
    AcquisitionDataset,
    TrajectoryStep
)
from .bc_data_pipeline import (
    create_curriculum_batches,
    create_scm_aware_batches,
    scm_aware_batch_iterator,
    create_acquisition_scm_aware_batches
)

# Import checkpointing and logging infrastructure
from .checkpoint_manager import CheckpointManager, CheckpointConfig, create_checkpoint_manager
from .utils.wandb_setup import log_metrics, log_artifact, is_wandb_enabled

# Import new validation metrics
from .acquisition_validation_metrics import (
    compute_comprehensive_validation_metrics,
    compute_diversity_bonus,
    log_validation_metrics
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BCPolicyConfig:
    """Configuration for behavioral cloning policy training."""
    # Policy architecture
    policy_config: PolicyConfig
    
    # BC-specific parameters
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs_per_level: int = 50
    min_epochs_per_level: int = 5
    
    # Curriculum learning
    curriculum_learning: bool = True
    start_difficulty: DifficultyLevel = DifficultyLevel.EASY
    advancement_threshold: float = 0.8  # Accuracy threshold for advancement
    
    # Training settings
    validation_patience: int = 10
    use_jax_compilation: bool = True
    
    # Loss weighting
    variable_selection_weight: float = 1.0
    intervention_value_weight: float = 0.5
    
    # Checkpointing and logging
    checkpoint_dir: str = "checkpoints/acquisition_bc"
    save_frequency: int = 10  # Save every N epochs
    enable_wandb_logging: bool = True
    experiment_name: str = "acquisition_bc"


@dataclass(frozen=True)
class BCPolicyState:
    """Immutable training state for BC policy."""
    current_difficulty: DifficultyLevel
    epoch: int
    best_validation_accuracy: float
    patience_counter: int
    policy_params: Any
    optimizer_state: Any
    training_metrics: List[Dict[str, float]]
    validation_metrics: List[Dict[str, float]]
    intervention_history: List[int] = None  # Track intervention choices for diversity bonus


@dataclass(frozen=True)
class BCPolicyResults:
    """Results from BC policy training."""
    final_state: BCPolicyState
    training_history: List[Dict[str, float]]
    validation_history: List[Dict[str, float]]
    curriculum_progression: List[Tuple[DifficultyLevel, int]]
    total_training_time: float
    final_policy_params: Any


class BCAcquisitionTrainer:
    """Behavioral cloning trainer for acquisition policies."""
    
    def __init__(self, config: BCPolicyConfig):
        """Initialize BC policy trainer."""
        self.config = config
        
        # Create optimizer
        self.optimizer = optax.adam(config.learning_rate)
        
        # Initialize policy network
        self._policy_network = None  # Will be created when needed
        self._num_variables = None  # Will be set from training data
        
        # Create JAX-compiled functions if enabled
        if config.use_jax_compilation:
            self._create_compiled_functions()
        else:
            self.jax_train_step = None
            self.jax_predict_step = None
        
        # Initialize checkpoint manager
        self.checkpoint_manager = create_checkpoint_manager(
            checkpoint_dir=config.checkpoint_dir,
            config=CheckpointConfig(
                save_frequency_steps=config.save_frequency,
                auto_cleanup=True,
                max_checkpoints=10
            )
        )
    
    def _create_compiled_functions(self):
        """Create JAX-compiled training and prediction functions."""
        
        if not self.config.use_jax_compilation:
            self.jax_train_step = None
            self.jax_predict_step = None
            return
        
        # Create the actual policy network for use in JAX functions
        if self._policy_network is None:
            self._create_policy_network()
            if self._policy_network is None:
                logger.warning("Failed to create policy network for JAX compilation")
                self.jax_train_step = None
                self.jax_predict_step = None
                return
        
        # Store reference to policy network for use in JAX functions
        policy_network = self._policy_network
        
        @jax.jit
        def train_step(params, opt_state, batch_states, batch_actions, key):
            """Compiled training step with actual policy network."""
            
            def loss_fn(params):
                batch_size = batch_actions.shape[0]
                
                # Vectorized computation for all examples
                def compute_single_loss(i, key):
                    # Get state dict for this example
                    state_dict = {
                        'state_tensor': batch_states['state_tensor'][i],
                        'target_variable_idx': batch_states['target_variable_idx'][i],
                        'history_tensor': batch_states['history_tensor'][i],
                        'is_training': True
                    }
                    
                    # Forward pass through policy network
                    # Haiku transformed functions: apply(params, rng, *args)
                    policy_output = policy_network.apply(params, key, state_dict, True)  # is_training=True
                    
                    # Extract predictions - use dynamic dimensions from state tensor
                    n_vars = batch_states['state_tensor'].shape[1]  # Get actual number of variables
                    
                    # Always assume policy_output is a dict (ensured by policy function)
                    variable_logits = policy_output.get('variable_logits', jnp.zeros(n_vars))
                    value_params = policy_output.get('value_params', jnp.zeros((n_vars, 2)))
                    
                    # Note: Debug prints removed for cleaner output
                    
                    # Get expert action
                    expert_var_idx = batch_actions[i, 0].astype(jnp.int32)
                    expert_value = batch_actions[i, 1]
                    
                    # Validate expert_var_idx is within bounds
                    def safe_cross_entropy(logits, label):
                        # Clip label to valid range [0, n_vars-1]
                        clipped_label = jnp.clip(label, 0, n_vars - 1)
                        return optax.softmax_cross_entropy_with_integer_labels(
                            logits[None, :], clipped_label[None]
                        )[0]
                    
                    def safe_value_prediction(params, idx):
                        # Clip idx to valid range [0, n_vars-1]
                        clipped_idx = jnp.clip(idx, 0, n_vars - 1)
                        return params[clipped_idx, 0]
                    
                    # Variable selection loss with label smoothing
                    # Increased from 0.1 to 0.2 to prevent overconfident predictions
                    label_smoothing = 0.2
                    n_classes = variable_logits.shape[0]
                    
                    # Create smoothed labels
                    smooth_labels = jnp.ones(n_classes) * (label_smoothing / (n_classes - 1))
                    smooth_labels = smooth_labels.at[expert_var_idx].set(1.0 - label_smoothing)
                    
                    # Compute cross-entropy with smoothed labels
                    log_probs = jax.nn.log_softmax(variable_logits)
                    base_var_loss = -jnp.sum(smooth_labels * log_probs)
                    
                    # Apply diversity weighting (simplified for JAX)
                    diversity_weight = 1.0  # Could be computed from history outside JAX
                    var_loss = base_var_loss * diversity_weight
                    
                    # Value prediction loss (MSE on mean) with bounds checking
                    pred_mean = safe_value_prediction(value_params, expert_var_idx)
                    value_loss = (pred_mean - expert_value) ** 2
                    
                    # Combined loss with proper weighting
                    raw_combined_loss = (self.config.variable_selection_weight * var_loss + 
                                        self.config.intervention_value_weight * value_loss)
                    
                    # Store raw values for diagnostic logging (will be aggregated outside JAX)
                    raw_var_loss = var_loss
                    raw_value_loss = value_loss
                    
                    # Check for invalid loss values
                    def validate_loss(loss, loss_name):
                        # Use a small epsilon instead of 0 to maintain gradient flow
                        # This prevents gradient collapse when losses become invalid
                        return jnp.where(
                            jnp.isnan(loss) | jnp.isinf(loss),
                            jnp.array(1e-6),  # Small epsilon to maintain gradients
                            loss
                        )
                    
                    var_loss = validate_loss(var_loss, "var_loss")
                    value_loss = validate_loss(value_loss, "value_loss")
                    combined_loss = validate_loss(raw_combined_loss, "combined_loss")
                    
                    # Diagnostic: compute softmax probabilities for entropy
                    probs = jax.nn.softmax(variable_logits)
                    max_prob = jnp.max(probs)
                    entropy = -jnp.sum(probs * jnp.log(probs + 1e-8))
                    
                    # Use softer clipping to avoid gradient issues
                    # Updated: Use larger scale factor to maintain gradients at typical loss values
                    # Analysis showed that at loss=22, gradient with scale=10 is only 0.048
                    # With scale=50, gradient is 0.829, allowing learning to continue
                    clipped_loss = 50.0 * jnp.tanh(combined_loss / 50.0)
                    
                    # Return a tuple with loss and diagnostics
                    # The diagnostics will be aggregated and logged outside JAX
                    diagnostics = {
                        'raw_var_loss': raw_var_loss,
                        'raw_value_loss': raw_value_loss, 
                        'raw_combined_loss': raw_combined_loss,
                        'clipped_loss': clipped_loss,
                        'max_prob': max_prob,
                        'entropy': entropy,
                        'expert_var_idx': expert_var_idx,
                        'n_vars': n_vars
                    }
                    
                    return combined_loss, diagnostics
                
                # Use vmap to vectorize over batch
                keys = random.split(key, batch_size)
                # Now returns (losses, diagnostics)
                results = jax.vmap(compute_single_loss)(jnp.arange(batch_size), keys)
                losses, batch_diagnostics = results
                
                # Aggregate diagnostics for logging
                mean_diagnostics = jax.tree.map(lambda x: jnp.mean(x), batch_diagnostics)
                
                return jnp.mean(losses), mean_diagnostics
            
            # Use has_aux=True to handle the diagnostics tuple
            (loss_value, diagnostics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            
            # Monitor gradients for debugging
            grad_norms = jax.tree.map(lambda x: jnp.linalg.norm(x), grads)
            total_grad_norm = jnp.sqrt(sum(jax.tree.leaves(jax.tree.map(lambda x: jnp.sum(x**2), grads))))
            
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            
            return new_params, new_opt_state, loss_value, total_grad_norm, diagnostics
        
        @jax.jit
        def predict_step(params, state_dict, key):
            """Compiled prediction step."""
            # Use actual policy network for predictions
            # Haiku transformed functions: apply(params, rng, *args)
            policy_output = policy_network.apply(params, key, state_dict, False)
            
            # Extract logits and sample action - use dynamic dimensions
            state_tensor = state_dict.get('state_tensor', jnp.zeros((5, 10)))
            n_vars = state_tensor.shape[0]  # Get actual number of variables from state tensor
            variable_logits = policy_output.get('variable_logits', jnp.zeros(n_vars))
            value_params = policy_output.get('value_params', jnp.zeros((n_vars, 2)))
            
            # Sample variable using Gumbel-max trick for differentiability
            var_key, val_key = random.split(key)
            gumbel_noise = -jnp.log(-jnp.log(random.uniform(var_key, variable_logits.shape)))
            var_idx = jnp.argmax(variable_logits + gumbel_noise)
            
            # Sample value from predicted distribution
            mean, log_std = value_params[var_idx]
            std = jnp.exp(log_std)
            value = mean + std * random.normal(val_key, ())
            
            return {
                'variable_idx': var_idx,
                'value': value,
                'intervention_variables': jnp.array([var_idx]),
                'intervention_values': jnp.array([value])
            }
        
        self.jax_train_step = train_step
        self.jax_predict_step = predict_step
    
    def train_on_curriculum(
        self,
        curriculum_datasets: Dict[DifficultyLevel, AcquisitionDataset],
        validation_datasets: Dict[DifficultyLevel, AcquisitionDataset],
        random_key: jax.Array
    ) -> BCPolicyResults:
        """
        Train policy on curriculum of increasing difficulty.
        
        Args:
            curriculum_datasets: Training datasets by difficulty level
            validation_datasets: Validation datasets by difficulty level
            random_key: JAX random key
            
        Returns:
            BCPolicyResults with complete training history
        """
        start_time = time.time()
        
        # Initialize training state
        init_key, train_key = random.split(random_key)
        initial_params = self._initialize_policy_params(init_key)
        initial_optimizer_state = self.optimizer.init(initial_params)
        
        training_state = BCPolicyState(
            current_difficulty=self.config.start_difficulty,
            epoch=0,
            best_validation_accuracy=0.0,
            patience_counter=0,
            policy_params=initial_params,
            optimizer_state=initial_optimizer_state,
            training_metrics=[],
            validation_metrics=[],
            intervention_history=[]
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
            val_dataset = validation_datasets.get(level, train_dataset)
            
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
        
        return BCPolicyResults(
            final_state=training_state,
            training_history=all_training_metrics,
            validation_history=all_validation_metrics,
            curriculum_progression=curriculum_progression,
            total_training_time=total_time,
            final_policy_params=training_state.policy_params
        )
    
    def _train_on_level(
        self,
        training_state: BCPolicyState,
        train_dataset: AcquisitionDataset,
        val_dataset: AcquisitionDataset,
        target_level: DifficultyLevel,
        random_key: jax.Array
    ) -> BCPolicyResults:
        """Train on a single curriculum level."""
        current_state = replace(training_state, current_difficulty=target_level)
        level_training_metrics = []
        level_validation_metrics = []
        
        # Create batches for this level
        trajectory_steps = train_dataset.trajectory_steps
        n_steps = len(trajectory_steps)
        
        logger.info(f"Training level {target_level} with {n_steps} trajectory steps")
        
        # Extract number of variables from the first state if not already set
        if self._num_variables is None and trajectory_steps:
            old_num_vars = 5  # The default we used before
            self._extract_num_variables_from_data(trajectory_steps)
            
            # If the number of variables changed, we need to recreate everything
            if self._num_variables != old_num_vars:
                logger.info(f"Number of variables changed from {old_num_vars} to {self._num_variables}")
                logger.info("Recreating policy network and JAX functions with correct variable dimensions")
                
                # Clear the old network
                self._policy_network = None
                
                # Recreate both the network and JAX-compiled functions
                if self.config.use_jax_compilation:
                    self._create_compiled_functions()
                else:
                    self._create_policy_network()
                
                # Reinitialize policy parameters with correct dimensions
                logger.info("Reinitializing policy parameters with correct dimensions")
                new_key = random.PRNGKey(42)  # Use a consistent seed for reproducible results
                new_params = self._initialize_policy_params(new_key)
                new_optimizer_state = self.optimizer.init(new_params)
                
                # Update the training state with new parameters
                current_state = replace(
                    current_state,
                    policy_params=new_params,
                    optimizer_state=new_optimizer_state
                )
                logger.info("✅ Successfully updated training state with new parameters")
        
        for epoch in range(self.config.max_epochs_per_level):
            epoch_key, random_key = random.split(random_key)
            
            # Training epoch
            current_state, epoch_metrics = self._train_epoch(
                state=current_state,
                trajectory_steps=trajectory_steps,
                random_key=epoch_key
            )
            
            level_training_metrics.append(epoch_metrics)
            
            # Validation
            val_metrics = self._validate_epoch(
                state=current_state,
                val_trajectory_steps=val_dataset.trajectory_steps,
                random_key=epoch_key
            )
            
            level_validation_metrics.append(val_metrics)
            
            # Update state - use top_3_accuracy as primary metric (more meaningful than exact match)
            current_accuracy = val_metrics.get('top_3_accuracy', val_metrics.get('accuracy', 0.0))
            if current_accuracy > current_state.best_validation_accuracy:
                current_state = replace(
                    current_state,
                    best_validation_accuracy=current_accuracy,
                    patience_counter=0
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
            
            # Log progress with comprehensive metrics
            if epoch % 5 == 0 or epoch == 0:
                logger.info(f"  Epoch {epoch}: loss={epoch_metrics.get('loss', 0.0):.4f}, "
                          f"top_3_acc={val_metrics.get('top_3_accuracy', 0.0):.4f}, "
                          f"mrr={val_metrics.get('mean_reciprocal_rank', 0.0):.4f}, "
                          f"lr={epoch_metrics.get('learning_rate', self.config.learning_rate):.2e}")
                
                # Log comprehensive metrics every 10 epochs for detailed monitoring
                if epoch % 10 == 0 and val_metrics:
                    log_validation_metrics(val_metrics, epoch, str(target_level))
            
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
        
        return BCPolicyResults(
            final_state=current_state,
            training_history=level_training_metrics,
            validation_history=level_validation_metrics,
            curriculum_progression=[(target_level, len(level_training_metrics))],
            total_training_time=0.0,
            final_policy_params=current_state.policy_params
        )
    
    def _update_params_without_jax(
        self, 
        params: Any, 
        batch_states: List[AcquisitionState],
        batch_actions: List[Dict[str, Any]],
        learning_rate: float,
        random_key: jax.Array
    ) -> Any:
        """Update parameters without JAX (gradient-free optimization)."""
        # This is a simplified parameter update for demonstration
        # In practice, you'd want a proper gradient-free optimizer
        
        # Add small random perturbations to parameters
        def add_noise(params_tree):
            return jax.tree.map(
                lambda p: p + learning_rate * 0.1 * random.normal(random_key, p.shape),
                params_tree
            )
        
        # Return slightly perturbed parameters
        return add_noise(params)

    def _train_epoch(
        self,
        state: BCPolicyState,
        trajectory_steps: List[TrajectoryStep],
        random_key: jax.Array
    ) -> Tuple[BCPolicyState, Dict[str, float]]:
        """Train for one epoch.
        
        Args:
            state: BCPolicyState containing policy parameters and optimizer state
            trajectory_steps: List of TrajectoryStep objects (each contains an AcquisitionState)
            random_key: JAX random key
        
        Returns:
            Tuple of (updated_state, metrics_dict)
        
        Note: Be careful not to confuse BCPolicyState (training state) with 
        AcquisitionState (RL observation state contained in trajectory steps).
        """
        # Create SCM-aware batches to ensure consistent variable dimensions
        batch_indices = create_acquisition_scm_aware_batches(
            trajectory_steps=trajectory_steps,
            batch_size=self.config.batch_size,
            shuffle=True,
            random_seed=int(random_key[0])
        )
        
        total_loss = 0.0
        total_batches = len(batch_indices)
        
        for batch_num, batch_idx in enumerate(batch_indices):
            batch_steps = [trajectory_steps[i] for i in batch_idx]
            
            # Extract states and actions
            batch_states = [step.state for step in batch_steps]
            batch_actions = [step.action for step in batch_steps]
            
            # Train step
            if self.jax_train_step is not None:
                # Convert states to arrays for JAX
                from .acquisition_state_converter import create_batch_tensor_state
                
                # Create batched tensor state
                batch_tensor_dict = create_batch_tensor_state(batch_states)
                
                # Convert to expected format for JAX train_step
                # The JAX train_step expects: state_tensor, target_variable_idx, history_tensor
                n_batch = len(batch_states)
                
                # Extract state tensor from current_data
                # current_data shape: [batch, samples, vars, channels]
                # Dummy state expects feature_dim=10, but we have channels=3
                # Solution: Take last few samples to create feature vector of size 10
                n_vars = batch_tensor_dict['current_data'].shape[2]
                n_channels = batch_tensor_dict['current_data'].shape[3]
                
                # Take last 3-4 samples to get ~10 features (3 samples * 3 channels = 9, close to 10)
                n_samples_needed = min(4, batch_tensor_dict['current_data'].shape[1])
                recent_data = batch_tensor_dict['current_data'][:, -n_samples_needed:, :, :]
                # Reshape to [batch, vars, samples*channels]
                state_tensor = recent_data.transpose(0, 2, 1, 3).reshape(n_batch, n_vars, -1)
                
                # Pad or truncate to exactly 10 features to match dummy state
                if state_tensor.shape[2] < 10:
                    # Pad with zeros
                    padding = jnp.zeros((n_batch, n_vars, 10 - state_tensor.shape[2]))
                    state_tensor = jnp.concatenate([state_tensor, padding], axis=2)
                elif state_tensor.shape[2] > 10:
                    # Truncate
                    state_tensor = state_tensor[:, :, :10]
                
                # Extract target variable indices (if available)
                target_indices = []
                for acq_state in batch_states:  # Renamed to avoid shadowing the BCPolicyState parameter
                    target_idx = 0  # default
                    if hasattr(acq_state, 'target_variable') or hasattr(acq_state, 'current_target'):
                        target_var = getattr(acq_state, 'target_variable', getattr(acq_state, 'current_target', None))
                        if target_var and hasattr(acq_state, 'scm_info') and 'variables' in acq_state.scm_info:
                            variables = list(acq_state.scm_info['variables'])
                            if target_var in variables:
                                target_idx = variables.index(target_var)
                    target_indices.append(target_idx)
                
                target_variable_idx = jnp.array(target_indices)
                
                # Use intervention history as history tensor
                # intervention_history shape: [batch, history_len, vars, features]
                # But features=3, while dummy state expects feature_dim=10
                history_tensor = batch_tensor_dict['intervention_history']
                
                # Ensure consistent history length of 3 to match dummy state initialization
                # This prevents Haiku parameter shape mismatches during training
                expected_history_len = 3
                current_history_len = history_tensor.shape[1]
                
                if current_history_len < expected_history_len:
                    # Pad with zeros at the beginning (older history)
                    padding_shape = (n_batch, expected_history_len - current_history_len, n_vars, history_tensor.shape[3])
                    padding = jnp.zeros(padding_shape)
                    history_tensor = jnp.concatenate([padding, history_tensor], axis=1)
                elif current_history_len > expected_history_len:
                    # Keep only the most recent history
                    history_tensor = history_tensor[:, -expected_history_len:, :, :]
                
                # Pad history features to match dummy state feature_dim=10
                history_features = history_tensor.shape[3]
                if history_features < 10:
                    # Pad with zeros to reach 10 features
                    history_padding = jnp.zeros((n_batch, expected_history_len, n_vars, 10 - history_features))
                    history_tensor = jnp.concatenate([history_tensor, history_padding], axis=3)
                elif history_features > 10:
                    # Truncate to 10 features
                    history_tensor = history_tensor[:, :, :, :10]
                
                # Create the expected dictionary format
                batch_tensor_dict = {
                    'state_tensor': state_tensor,
                    'target_variable_idx': target_variable_idx,
                    'history_tensor': history_tensor
                }
                
                # Convert actions to arrays with proper variable indexing
                batch_action_arrays = []
                
                # Get variable list from first state for consistent indexing
                first_state = batch_states[0]
                variables = []
                if hasattr(first_state, 'posterior') and hasattr(first_state.posterior, 'variable_order'):
                    variables = list(first_state.posterior.variable_order)
                    # Reduce logging frequency - only log on first batch
                    if batch_num == 0:
                        logger.info(f"Using posterior variable_order: {variables[:5]}...")  # Show only first 5
                elif hasattr(first_state, 'metadata') and 'scm_info' in first_state.metadata:
                    # Get variables from metadata.scm_info
                    scm_info = first_state.metadata['scm_info']
                    variables = list(scm_info.get('variables', [])) if isinstance(scm_info, dict) else []
                    # Reduce logging frequency - only log on first batch
                    if batch_num == 0:
                        logger.info(f"Using metadata scm_info variables: {variables[:5]}...")  # Show only first 5
                else:
                    logger.warning("No variable information found in state - this will cause mapping failures!")
                
                if not variables:
                    logger.error("Empty variables list - cannot create proper variable mapping!")
                    raise ValueError("No variables found for acquisition training")
                
                # Create variable to index mapping
                var_to_idx = {var: idx for idx, var in enumerate(variables)}
                # Only log variable mapping on first batch to reduce spam
                if batch_idx == batch_indices[0]:
                    logger.debug(f"Created variable mapping for {len(var_to_idx)} variables")
                
                for action in batch_actions:
                    if isinstance(action, dict):
                        intervention_vars = action.get('intervention_variables', frozenset())
                        intervention_vals = action.get('intervention_values', ())
                        
                        if intervention_vars and variables:
                            var_name = next(iter(intervention_vars))
                            if var_name not in var_to_idx:
                                logger.error(f"Variable '{var_name}' not found in mapping {var_to_idx}")
                                raise ValueError(f"Expert variable '{var_name}' not in variable list {variables}")
                            var_idx = var_to_idx[var_name]
                            value = intervention_vals[0] if intervention_vals else 0.0
                            # Remove per-action logging to reduce memory usage
                            pass  # logger.debug(f"Mapped expert action: {var_name} -> {var_idx}, value={value}")
                        else:
                            # Reduce warning spam - only log once per epoch
                            pass  # logger.warning removed to reduce memory usage
                            var_idx = 0
                            value = 0.0
                        
                        action_array = jnp.array([var_idx, value])
                    else:
                        action_array = jnp.array([0, 0.0])
                    batch_action_arrays.append(action_array)
                
                # Stack actions
                batch_actions_tensor = jnp.stack(batch_action_arrays)
                
                # Validate batch actions for debugging
                max_var_idx = jnp.max(batch_actions_tensor[:, 0])
                min_var_idx = jnp.min(batch_actions_tensor[:, 0])
                n_vars_expected = len(variables)
                
                # logger.info(f"Batch validation: var_idx range [{int(min_var_idx)}, {int(max_var_idx)}], expected n_vars: {n_vars_expected}")
                
                # Check for out-of-bounds indices
                if max_var_idx >= n_vars_expected:
                    logger.error(f"Out-of-bounds variable index {int(max_var_idx)} >= {n_vars_expected}")
                    logger.error(f"Variable mapping was: {var_to_idx}")
                    logger.error(f"Expert actions mapped to indices: {batch_actions_tensor[:, 0]}")
                    
                if min_var_idx < 0:
                    logger.error(f"Negative variable index {int(min_var_idx)}")
                    
                # Debug logging commented out for cleaner output
                # unique_indices = jnp.unique(batch_actions_tensor[:, 0])
                # logger.info(f"Unique expert variable indices in batch: {unique_indices}")
                
                # action_counts = {int(idx): int(jnp.sum(batch_actions_tensor[:, 0] == idx)) for idx in unique_indices}
                # logger.info(f"Action distribution: {action_counts}")
                
                # Use JAX train step to update parameters
                updated_params, updated_opt_state, loss_value, grad_norm, diagnostics = self.jax_train_step(
                    state.policy_params,
                    state.optimizer_state,
                    batch_tensor_dict,  # or appropriate state representation
                    batch_actions_tensor,
                    random_key
                )
                
                # Monitor loss and gradients for debugging
                loss_float = float(loss_value)
                grad_norm_float = float(grad_norm)
                
                # Extract and log diagnostics (only on first batch and every 10th batch to avoid spam)
                if batch_num == 0 or batch_num % 10 == 0:
                    diag = {k: float(v) for k, v in diagnostics.items()}
                    logger.info(f"=== Loss Diagnostics (Batch {batch_num}) ===")
                    logger.info(f"Raw var_loss: {diag['raw_var_loss']:.4f}")
                    logger.info(f"Raw value_loss: {diag['raw_value_loss']:.4f}") 
                    logger.info(f"Raw combined_loss: {diag['raw_combined_loss']:.4f}")
                    logger.info(f"Clipped loss: {diag['clipped_loss']:.4f}")
                    logger.info(f"Max probability: {diag['max_prob']:.4f}")
                    logger.info(f"Entropy: {diag['entropy']:.4f}")
                    logger.info(f"Gradient norm: {grad_norm_float:.6f}")
                    
                    # Check for loss saturation with updated threshold
                    if diag['raw_combined_loss'] > 100.0:
                        logger.warning(f"Very high raw loss detected: {diag['raw_combined_loss']:.2f}")
                        logger.warning("This may reduce gradient magnitude (scale=50 threshold)")
                    elif diag['raw_combined_loss'] > 200.0:
                        logger.error(f"Extremely high raw loss: {diag['raw_combined_loss']:.2f}")
                        logger.error("This will cause gradient vanishing even with scale=50")
                
                if jnp.isnan(loss_value) or jnp.isinf(loss_value):
                    logger.error(f"NaN/Inf loss detected: {loss_float}")
                    logger.error("This indicates a fundamental training problem")
                elif jnp.isnan(grad_norm) or jnp.isinf(grad_norm):
                    logger.error(f"NaN/Inf gradient norm detected: {grad_norm_float}")
                    logger.error("Gradients are broken - no learning can occur")
                elif grad_norm_float < 1e-8:
                    logger.warning(f"Zero gradients detected: {grad_norm_float}")
                    logger.warning("No parameter updates will occur - training stuck")
                elif loss_float > 1000000:  # Astronomical loss threshold
                    logger.warning(f"Astronomical loss detected: {loss_float}, grad_norm: {grad_norm_float}")
                    logger.warning(f"Variable mapping: {var_to_idx}")
                    logger.warning(f"Action indices: {batch_actions_tensor[:5, 0]}")  # First 5 for debugging
                    logger.warning(f"Expected n_vars: {n_vars_expected}")
                elif loss_float > 100:  # High but potentially recoverable
                    # Only log every 10th batch to reduce spam
                    if batch_num % 10 == 0:
                        logger.info(f"High loss: {loss_float:.2f}, grad_norm: {grad_norm_float:.4f}")
                else:
                    # Log gradient health for first batch only
                    if batch_num == 0:  # First batch of epoch
                        logger.debug(f"Gradient health - norm: {grad_norm_float:.4f}, loss: {loss_float:.4f}")
                        
                        # Simple gradient vanishing check based on norm
                        if grad_norm_float < 1e-8 and loss_float > 0.01:
                            logger.warning("Potential gradient vanishing detected - check loss computation")
                
                # Update state with new parameters
                state = replace(
                    state,
                    policy_params=updated_params,
                    optimizer_state=updated_opt_state
                )
                
                total_loss += float(loss_value)
            else:
                # Fallback training without JAX
                loss_value = self._compute_batch_loss(batch_states, batch_actions, state)
                total_loss += float(loss_value)
                
                # Update parameters using gradient-free method
                batch_key, random_key = random.split(random_key)
                state = replace(
                    state,
                    policy_params=self._update_params_without_jax(
                        state.policy_params,
                        batch_states,
                        batch_actions,
                        self.config.learning_rate,
                        batch_key
                    )
                )
        
        average_loss = total_loss / total_batches if total_batches > 0 else 0.0
        
        metrics = {
            'epoch': state.epoch,
            'loss': average_loss,
            'learning_rate': self.config.learning_rate
        }
        
        return state, metrics
    
    def _validate_epoch(
        self,
        state: BCPolicyState,
        val_trajectory_steps: List[TrajectoryStep],
        random_key: jax.Array
    ) -> Dict[str, float]:
        """Validate on validation dataset using comprehensive metrics."""
        if not val_trajectory_steps:
            return {'top_3_accuracy': 0.0, 'loss': float('inf')}
        
        # Collect policy logits and expert choices for batch validation
        policy_logits_list = []
        expert_choices_list = []
        total_loss = 0.0
        
        for step in val_trajectory_steps:
            # Get policy output for this state
            try:
                if self.jax_predict_step is not None:
                    # Convert AcquisitionState to tensor dictionary for JAX
                    state_arrays = self._state_to_arrays(step.state)
                    
                    # Get policy logits directly from the policy network
                    # Need to provide RNG key for Haiku transformed functions
                    eval_key = random.PRNGKey(0)  # Deterministic key for evaluation
                    policy_output = self._policy_network.apply(
                        state.policy_params, eval_key, state_arrays, False  # is_training=False
                    )
                    policy_logits = policy_output.get('variable_logits', jnp.zeros(self._num_variables or 5))
                    
                else:
                    # Fallback for non-JAX prediction
                    state_dict = self._state_to_arrays(step.state)
                    # Need to provide RNG key for Haiku transformed functions
                    eval_key = random.PRNGKey(0)  # Deterministic key for evaluation
                    policy_output = self._policy_network.apply(
                        state.policy_params, eval_key, state_dict, False
                    )
                    policy_logits = policy_output.get('variable_logits', jnp.zeros(self._num_variables or 5))
                
                policy_logits_list.append(policy_logits)
                
                # Extract expert choice
                expert_vars = step.action.get('intervention_variables', frozenset())
                if expert_vars:
                    # Convert variable name to index
                    expert_var = list(expert_vars)[0]
                    expert_idx = self._variable_name_to_index(expert_var, step.state)
                    expert_choices_list.append(expert_idx)
                else:
                    expert_choices_list.append(0)  # Default to first variable
                
                # Compute loss for this step (simplified)
                loss = self._compute_single_loss_simple(step)
                total_loss += float(loss)
                
            except Exception as e:
                logger.warning(f"Validation step failed: {e}")
                # Use fallback values
                policy_logits_list.append(jnp.zeros(self._num_variables or 5))
                expert_choices_list.append(0)
                total_loss += 1000.0  # High penalty for failed computation
        
        if not policy_logits_list:
            return {'top_3_accuracy': 0.0, 'loss': float('inf')}
        
        # Convert to JAX arrays for efficient computation
        policy_logits = jnp.stack(policy_logits_list)  # [batch_size, n_variables]
        expert_choices = jnp.array(expert_choices_list)  # [batch_size]
        
        # Compute comprehensive validation metrics
        validation_metrics = compute_comprehensive_validation_metrics(
            policy_logits=policy_logits,
            expert_choices=expert_choices,
            intervention_history=state.intervention_history,
            total_variables=self._num_variables
        )
        
        # Add loss and epoch info
        average_loss = total_loss / len(val_trajectory_steps) if val_trajectory_steps else 0.0
        validation_metrics.update({
            'epoch': state.epoch,
            'loss': average_loss,
            'accuracy': validation_metrics.get('top_1_accuracy', 0.0)  # For backward compatibility
        })
        
        return validation_metrics
    
    def _predict_action(
        self,
        params: Any,
        state: AcquisitionState,
        random_key: jax.Array
    ) -> Dict[str, Any]:
        """
        Predict action for given state.
        
        Args:
            params: Policy parameters
            state: Current acquisition state
            random_key: JAX random key
            
        Returns:
            Predicted action dictionary
        """
        # Get available variables from various possible locations
        variables = []
        
        # Try multiple locations for variables
        if hasattr(state, 'metadata') and 'scm_info' in state.metadata:
            # Get from metadata.scm_info (our fix)
            scm_info = state.metadata['scm_info']
            if isinstance(scm_info, dict) and 'variables' in scm_info:
                variables = list(scm_info['variables'])
            elif 'scm' in state.metadata and 'variables' in state.metadata['scm']:
                variables = list(state.metadata['scm']['variables'])
        elif hasattr(state, 'posterior') and hasattr(state.posterior, 'variable_order'):
            variables = list(state.posterior.variable_order)
        
        if not variables:
            return {
                'intervention_variables': frozenset(),
                'intervention_values': tuple()
            }
        
        # Use actual policy network if available and parameters are valid
        if hasattr(self, '_policy_network') and self._policy_network is not None:
            try:
                # Convert state to proper format for policy network
                state_dict = self._state_to_arrays(state)
                
                # Use the actual policy network
                # Haiku needs RNG key as second argument
                policy_output = self._policy_network.apply(
                    params, random_key, state_dict, False  # is_training=False
                )
                
                # Extract variable logits and value parameters
                variable_logits = policy_output.get('variable_logits', jnp.zeros(len(variables)))
                value_params = policy_output.get('value_params', jnp.zeros((len(variables), 2)))
                
                # Sample variable from logits
                var_probs = jax.nn.softmax(variable_logits)
                var_idx = random.choice(random_key, len(variables), p=var_probs)
                selected_var = variables[var_idx]
                
                # Sample intervention value from predicted distribution
                value_key, _ = random.split(random_key)
                mean, log_std = value_params[var_idx]
                std = jnp.exp(log_std)
                intervention_value = mean + std * random.normal(value_key, ())
                
                return {
                    'intervention_variables': frozenset([selected_var]),
                    'intervention_values': (float(intervention_value),)
                }
                
            except Exception as e:
                logger.warning(f"Policy network forward pass failed: {e}, falling back to heuristic")
                # Fall through to heuristic
        
        # Fallback: Use heuristic based on state information
        # This is better than random as it considers the state
        
        # Prefer variables with higher uncertainty if available
        if hasattr(state, 'uncertainty_info') and state.uncertainty_info:
            uncertainty_info = state.uncertainty_info
            # Find variable with highest uncertainty
            max_uncertainty = 0.0
            best_var = variables[0]
            
            for var in variables:
                var_uncertainty = uncertainty_info.get(var, 0.0)
                if var_uncertainty > max_uncertainty:
                    max_uncertainty = var_uncertainty
                    best_var = var
                    
            selected_var = best_var
        else:
            # If no uncertainty info, sample uniformly from variables
            var_idx = random.randint(random_key, (), 0, len(variables))
            selected_var = variables[var_idx]
        
        # Sample intervention value from reasonable range
        value_key, _ = random.split(random_key)
        intervention_value = random.normal(value_key, ()) * 2.0  # Scale by 2 for reasonable range
        
        return {
            'intervention_variables': frozenset([selected_var]),
            'intervention_values': (float(intervention_value),)
        }
    
    def _actions_match(
        self,
        predicted_action: Dict[str, Any],
        expert_action: Dict[str, Any]
    ) -> bool:
        """Check if predicted action matches expert action."""
        # Check if intervention variables match
        pred_vars = predicted_action.get('intervention_variables', frozenset())
        expert_vars = expert_action.get('intervention_variables', frozenset())
        
        return pred_vars == expert_vars
    
    def _compute_single_loss(
        self,
        predicted_action: Dict[str, Any],
        expert_action: Dict[str, Any]
    ) -> float:
        """Compute loss for a single prediction."""
        # Variable selection loss (cross-entropy)
        var_selection_loss = self._compute_variable_selection_loss(
            [predicted_action], [expert_action]
        )
        
        # Intervention value loss (MSE)
        value_loss = self._compute_intervention_value_loss(
            [predicted_action], [expert_action]
        )
        
        # Weighted combination
        total_loss = (
            self.config.variable_selection_weight * var_selection_loss +
            self.config.intervention_value_weight * value_loss
        )
        
        return total_loss
    
    def _compute_batch_loss(
        self,
        batch_states: List[AcquisitionState],
        batch_actions: List[Dict[str, Any]],
        state: BCPolicyState
    ) -> float:
        """Compute loss for a batch."""
        total_loss = 0.0
        
        for single_state, expert_action in zip(batch_states, batch_actions):
            predicted_action = self._predict_action(
                state.policy_params,
                single_state,
                random.PRNGKey(0)  # Simplified
            )
            loss = self._compute_single_loss(predicted_action, expert_action)
            total_loss += loss
        
        return total_loss / len(batch_states) if batch_states else 0.0
    
    def _compute_variable_selection_loss(
        self,
        predicted_actions: List[Dict[str, Any]],
        expert_actions: List[Dict[str, Any]]
    ) -> float:
        """Compute cross-entropy loss for variable selection."""
        if not predicted_actions or not expert_actions:
            return 0.0
        
        total_loss = 0.0
        valid_comparisons = 0
        
        for pred, exp in zip(predicted_actions, expert_actions):
            # Get predicted and expert intervention variables
            pred_vars = pred.get('intervention_variables', frozenset())
            exp_vars = exp.get('intervention_variables', frozenset())
            
            # For multi-variable case, we need to compute cross-entropy properly
            # For now, use binary classification loss
            if pred_vars == exp_vars:
                # Perfect match
                loss = 0.0
            else:
                # Mismatch - use cross-entropy approximation
                # This could be improved with actual logit predictions
                loss = -jnp.log(0.1)  # Negative log probability for incorrect prediction
            
            total_loss += loss
            valid_comparisons += 1
        
        return float(total_loss / valid_comparisons) if valid_comparisons > 0 else 0.0
    
    def _compute_intervention_value_loss(
        self,
        predicted_actions: List[Dict[str, Any]],
        expert_actions: List[Dict[str, Any]]
    ) -> float:
        """Compute MSE loss for intervention values."""
        if not predicted_actions or not expert_actions:
            return 0.0
        
        total_mse = 0.0
        count = 0
        
        for pred, exp in zip(predicted_actions, expert_actions):
            pred_vals = pred.get('intervention_values', ())
            exp_vals = exp.get('intervention_values', ())
            
            # Only compare values if variables match
            pred_vars = pred.get('intervention_variables', frozenset())
            exp_vars = exp.get('intervention_variables', frozenset())
            
            if pred_vars == exp_vars and len(pred_vals) == len(exp_vals):
                # Variables match, compute MSE on values
                for p_val, e_val in zip(pred_vals, exp_vals):
                    mse = (float(p_val) - float(e_val)) ** 2
                    total_mse += mse
                    count += 1
            elif pred_vars != exp_vars:
                # Variables don't match, add penalty
                # This encourages selecting the right variable first
                penalty = 1.0  # Fixed penalty for wrong variable selection
                total_mse += penalty
                count += 1
            else:
                # Length mismatch, add penalty
                penalty = 1.0
                total_mse += penalty
                count += 1
        
        return float(total_mse / count) if count > 0 else 0.0
    
    def _should_advance_level(
        self,
        state: BCPolicyState,
        val_metrics: Dict[str, float]
    ) -> bool:
        """Determine if should advance to next curriculum level."""
        # Use top_3_accuracy as it's more meaningful than exact match
        top_3_acc = val_metrics.get('top_3_accuracy', 0.0)
        # Also consider mean reciprocal rank as secondary metric
        mrr = val_metrics.get('mean_reciprocal_rank', 0.0)
        
        # Advanced criteria: either high top-3 accuracy OR good ranking performance
        accuracy_threshold_met = top_3_acc >= self.config.advancement_threshold or mrr >= 0.5
        min_epochs_met = state.epoch >= self.config.min_epochs_per_level
        
        return accuracy_threshold_met and min_epochs_met
    
    def _initialize_policy_params(self, random_key: jax.Array) -> Any:
        """Initialize policy parameters."""
        # Create policy network if not already created
        if self._policy_network is None:
            self._create_policy_network()
            # Log status after creation attempt
            if self._policy_network is None:
                logger.warning("Policy network creation failed - _policy_network is still None")
            else:
                logger.info("Policy network created successfully")
        
        # Initialize parameters if we have a policy network
        if self._policy_network is not None:
            try:
                # Create dummy state for initialization
                dummy_state = self._create_dummy_acquisition_state()
                logger.info(f"Dummy state format: {list(dummy_state.keys())}")
                logger.info(f"State tensor shape: {dummy_state['state_tensor'].shape}")
                logger.info(f"History tensor shape: {dummy_state['history_tensor'].shape}")
                logger.info(f"Target variable idx: {dummy_state['target_variable_idx']}")
                
                # Initialize policy parameters with detailed error handling
                logger.info("Calling policy network init...")
                params = self._policy_network.init(random_key, dummy_state, False)
                logger.info("Successfully initialized policy network parameters")
                return params
                
            except Exception as e:
                import traceback
                # Handle case where exception is None or has no string representation
                if e is None:
                    error_msg = "Haiku transformation returned None as exception - this often indicates a shape mismatch or missing required argument"
                else:
                    error_msg = str(e) if str(e) else f"Exception of type {type(e).__name__} with no message"
                
                logger.error(f"Failed to initialize policy network: {error_msg}")
                logger.error(f"Exception type: {type(e).__name__ if e is not None else 'NoneType'}")
                logger.error(f"Dummy state used: {dummy_state}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                
                # Try to provide more specific guidance
                if "shape" in error_msg.lower() or e is None:
                    logger.error("Hint: Check that dummy_state matches the expected input format for EnhancedPolicyNetwork.__call__")
                # Fall through to placeholder
        
        # Placeholder initialization
        logger.warning("Using placeholder policy parameters - network not available")
        return {'placeholder': jnp.array([1.0])}
    
    def _create_policy_network(self):
        """Create the policy network."""
        # Temporarily set debug logging for detailed diagnostics
        current_level = logger.level
        logger.setLevel(logging.DEBUG)
        
        try:
            # Determine which policy network to use
            use_enhanced = getattr(self.config.policy_config, 'use_enhanced_policy', True)
            
            if use_enhanced:
                # Use EnhancedPolicyNetwork for dynamic dimension support
                logger.info("Using EnhancedPolicyNetwork for dynamic variable dimensions")
                
                def policy_fn(state_dict, is_training):
                    # Extract the actual state tensor and parameters from dict
                    # Removed debug logging to avoid excessive output during training
                    
                    # Get state tensor without hardcoded fallback - use actual data dimensions
                    state_tensor = state_dict.get('state_tensor')
                    if state_tensor is None:
                        # If no state tensor provided, create one with actual number of variables
                        n_vars = self._num_variables if self._num_variables is not None else 5
                        state_tensor = jnp.zeros((n_vars, 10))
                        logger.debug(f"Created fallback state tensor for {n_vars} variables")
                    target_idx = state_dict.get('target_variable_idx', 0)
                    history = state_dict.get('history_tensor', None)
                    
                    try:
                        network = EnhancedPolicyNetwork(
                            hidden_dim=self.config.policy_config.hidden_dim,
                            num_layers=self.config.policy_config.num_layers,
                            num_heads=self.config.policy_config.num_heads,
                            key_size=self.config.policy_config.key_size,
                            dropout=self.config.policy_config.dropout,
                            # num_variables will be inferred from data shape
                            num_variables=None
                        )
                        
                        result = network(state_tensor, target_idx, history, is_training)
                        
                        # Always return the result from the network
                        # The network is guaranteed to return a dict with all keys
                        return result
                        
                    except Exception as inner_e:
                        logger.error(f"Error in policy function: {inner_e}")
                        logger.error(f"Inner exception type: {type(inner_e)}")
                        import traceback
                        logger.error(f"Inner traceback:\n{traceback.format_exc()}")
                        # Return valid default instead of raising
                        n_vars = state_tensor.shape[0] if state_tensor is not None else 5
                        return {
                            'variable_logits': jnp.zeros(n_vars),
                            'value_params': jnp.zeros((n_vars, 2)),
                            'state_value': jnp.array(0.0)
                        }
            else:
                # Use standard AcquisitionPolicyNetwork (legacy)
                logger.warning("Using legacy AcquisitionPolicyNetwork - consider enabling enhanced policy")
                
                def policy_fn(state, is_training):
                    network = AcquisitionPolicyNetwork(
                        hidden_dim=self.config.policy_config.hidden_dim,
                        num_layers=self.config.policy_config.num_layers,
                        num_heads=self.config.policy_config.num_heads,
                        dropout=self.config.policy_config.dropout
                    )
                    return network(state, is_training)
            
            # Transform to Haiku function
            logger.info("Transforming policy function with Haiku...")
            self._policy_network = hk.transform(policy_fn)
            logger.info("Successfully created policy network")
            
        except Exception as e:
            import traceback
            logger.error(f"Failed to create policy network: {e}")
            logger.error(f"Exception type: {type(e)}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            self._policy_network = None
        finally:
            # Restore original logging level
            logger.setLevel(current_level)
    
    def _create_dummy_acquisition_state(self) -> Dict[str, jnp.ndarray]:
        """Create a dummy acquisition state for parameter initialization."""
        # Use actual number of variables if available, otherwise fall back to default
        n_vars = self._num_variables if self._num_variables is not None else 5
        feature_dim = 10  # This is a reasonable default
        
        logger.info(f"Creating dummy state for {n_vars} variables (from data: {self._num_variables is not None})")
        
        dummy_state = {
            'state_tensor': jnp.zeros((n_vars, feature_dim)),  # [n_vars, feature_dim] - dynamic
            'target_variable_idx': 0,            # Index of target variable
            'history_tensor': jnp.zeros((3, n_vars, feature_dim)),  # [history_len, n_vars, feature_dim] - dynamic
            'is_training': False
        }
        
        return dummy_state
    
    def _extract_num_variables_from_data(self, trajectory_steps: List) -> None:
        """Extract the number of variables from actual training data."""
        if not trajectory_steps:
            logger.warning("No trajectory steps provided - cannot extract variable count")
            return
        
        # Get the first state to extract variable information
        first_step = trajectory_steps[0]
        first_state = first_step.state
        
        # Try multiple ways to get the variable count
        n_vars = None
        
        # Method 1: From metadata scm_info
        if hasattr(first_state, 'metadata') and 'scm_info' in first_state.metadata:
            scm_info = first_state.metadata['scm_info']
            variables = scm_info.get('variables', []) if isinstance(scm_info, dict) else []
            if variables:
                n_vars = len(variables)
                logger.info(f"Extracted {n_vars} variables from metadata scm_info: {variables[:5]}...")
        
        # Method 2: From posterior variable_order
        if n_vars is None and hasattr(first_state, 'posterior') and hasattr(first_state.posterior, 'variable_order'):
            variables = list(first_state.posterior.variable_order)
            if variables:
                n_vars = len(variables)
                logger.info(f"Extracted {n_vars} variables from posterior variable_order: {variables[:5]}...")
        
        # Method 3: From state tensor shape (if available)
        if n_vars is None:
            try:
                state_arrays = self._state_to_arrays(first_state)
                if 'state_tensor' in state_arrays:
                    state_tensor = state_arrays['state_tensor']
                    n_vars = state_tensor.shape[0]  # First dimension should be n_vars
                    logger.info(f"Extracted {n_vars} variables from state tensor shape: {state_tensor.shape}")
            except Exception as e:
                logger.debug(f"Could not extract variables from state tensor: {e}")
        
        # Set the number of variables
        if n_vars is not None and n_vars > 0:
            self._num_variables = n_vars
            logger.info(f"✅ Set number of variables to {n_vars} from training data")
        else:
            logger.warning("❌ Could not extract number of variables from training data - using default")
    
    def _state_to_arrays(self, state: AcquisitionState) -> Dict[str, jnp.ndarray]:
        """Convert AcquisitionState to array representation for JAX."""
        # Import the new converter
        from .acquisition_state_converter import (
            convert_acquisition_state_to_tensors,
            tensor_state_to_dict
        )
        
        # Convert state to tensors
        tensor_state = convert_acquisition_state_to_tensors(
            state,
            max_samples=10,  # Last 10 samples
            max_history=5,   # Last 5 interventions
            n_channels=3,    # Standard channels
            n_features=3     # Intervention features
        )
        
        # Convert to dictionary for JAX
        return tensor_state_to_dict(tensor_state)
    
    def save_checkpoint(
        self, 
        state: BCPolicyState, 
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
        
        # Determine model type based on policy configuration
        use_enhanced = getattr(self.config.policy_config, 'use_enhanced_policy', True)
        model_type = "enhanced_acquisition" if use_enhanced else "standard_acquisition"
        
        # Extract policy configuration
        model_config = {
            'hidden_dim': self.config.policy_config.hidden_dim,
            'num_layers': self.config.policy_config.num_layers,
            'num_heads': self.config.policy_config.num_heads,
            'key_size': getattr(self.config.policy_config, 'key_size', 32),
            'dropout': self.config.policy_config.dropout,
            'use_enhanced_policy': use_enhanced
        }
        
        # Add number of variables if available
        if hasattr(self, '_num_variables') and self._num_variables is not None:
            model_config['num_variables'] = self._num_variables
            
        # Create checkpoint data
        checkpoint_data = {
            'config': self.config,
            'training_state': state,
            'current_difficulty': state.current_difficulty,
            'epoch': state.epoch,
            'policy_params': state.policy_params,
            'optimizer_state': state.optimizer_state,
            'model_type': model_type,
            'model_config': model_config
        }
        
        checkpoint_info = self.checkpoint_manager.save_checkpoint(
            state=checkpoint_data,
            checkpoint_name=checkpoint_name,
            stage=stage,
            user_notes=user_notes
        )
        
        logger.info(f"💾 Saved acquisition BC checkpoint: {checkpoint_info.path}")
        
        # Log to WandB if enabled
        if self.config.enable_wandb_logging and is_wandb_enabled():
            log_artifact(
                str(checkpoint_info.path),
                artifact_type="acquisition_bc_checkpoint",
                name=f"acquisition_bc_checkpoint_epoch_{state.epoch}"
            )
        
        return str(checkpoint_info.path)
    
    def load_checkpoint(self, checkpoint_path: str) -> BCPolicyState:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Loaded training state
        """
        checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        logger.info(f"📂 Loaded acquisition BC checkpoint from: {checkpoint_path}")
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
        state: BCPolicyState,
        train_metrics: Optional[Dict[str, float]] = None,
        val_metrics: Optional[Dict[str, float]] = None
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
            "acquisition_bc/epoch": state.epoch,
            "acquisition_bc/difficulty_level": state.current_difficulty.value,
            "acquisition_bc/patience_counter": state.patience_counter,
            "acquisition_bc/best_validation_accuracy": state.best_validation_accuracy
        }
        
        # Add training metrics if available
        if train_metrics:
            metrics.update({
                "acquisition_bc/train_loss": train_metrics.get('loss', 0.0),
                "acquisition_bc/learning_rate": train_metrics.get('learning_rate', self.config.learning_rate)
            })
        
        # Add validation metrics if available
        if val_metrics:
            metrics.update({
                "acquisition_bc/val_accuracy": val_metrics.get('accuracy', 0.0),
                "acquisition_bc/val_top_3_accuracy": val_metrics.get('top_3_accuracy', 0.0),
                "acquisition_bc/val_top_5_accuracy": val_metrics.get('top_5_accuracy', 0.0),
                "acquisition_bc/val_mean_reciprocal_rank": val_metrics.get('mean_reciprocal_rank', 0.0),
                "acquisition_bc/val_expert_percentile": val_metrics.get('expert_percentile', 0.0),
                "acquisition_bc/val_diversity_score": val_metrics.get('diversity_score', 0.0),
                "acquisition_bc/val_exploration_coverage": val_metrics.get('exploration_coverage', 0.0),
                "acquisition_bc/val_mean_entropy": val_metrics.get('mean_entropy', 0.0),
                "acquisition_bc/val_loss": val_metrics.get('loss', 0.0)
            })
        
        # Log to WandB
        log_metrics(metrics, step=state.epoch)

    def _variable_name_to_index(self, variable_name: str, state: AcquisitionState) -> int:
        """Convert variable name to index."""
        # Try multiple locations for variable ordering
        variables = []
        
        if hasattr(state, 'metadata') and 'scm_info' in state.metadata:
            scm_info = state.metadata['scm_info']
            if isinstance(scm_info, dict) and 'variables' in scm_info:
                variables = list(scm_info['variables'])
        elif hasattr(state, 'posterior') and hasattr(state.posterior, 'variable_order'):
            variables = list(state.posterior.variable_order)
        
        if variable_name in variables:
            return variables.index(variable_name)
        
        # Fallback: try to extract index from variable name (e.g., 'X0' -> 0)
        try:
            if variable_name.startswith('X'):
                return int(variable_name[1:])
        except (ValueError, IndexError):
            pass
        
        return 0  # Default fallback
    
    def _compute_single_loss_simple(self, step: TrajectoryStep) -> float:
        """Simplified loss computation for validation."""
        try:
            # This is a simplified version for validation - just return a reasonable value
            # In practice, this would compute the actual loss using the policy network
            return 1.0  # Placeholder loss value
        except Exception as e:
            logger.warning(f"Failed to compute simple loss: {e}")
            return 1000.0  # High penalty for failed computation


def create_bc_acquisition_trainer(
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    use_curriculum: bool = True,
    use_jax: bool = True,
    checkpoint_dir: str = "checkpoints/acquisition_bc",
    enable_wandb_logging: bool = True,
    experiment_name: str = "acquisition_bc"
) -> BCAcquisitionTrainer:
    """
    Factory function to create BC acquisition trainer.
    
    Args:
        learning_rate: Learning rate for optimization
        batch_size: Batch size for training
        use_curriculum: Whether to use curriculum learning
        use_jax: Whether to use JAX compilation
        checkpoint_dir: Directory for saving checkpoints
        enable_wandb_logging: Whether to enable WandB logging
        experiment_name: Name for experiment tracking
        
    Returns:
        Configured BCAcquisitionTrainer
    """
    policy_config = PolicyConfig(
        hidden_dim=128,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        exploration_noise=0.1,
        variable_selection_temp=1.0,
        value_selection_temp=1.0
    )
    
    bc_config = BCPolicyConfig(
        policy_config=policy_config,
        learning_rate=float(learning_rate),  # Ensure numeric type
        batch_size=int(batch_size),          # Ensure numeric type
        curriculum_learning=use_curriculum,
        use_jax_compilation=use_jax,
        checkpoint_dir=checkpoint_dir,
        enable_wandb_logging=enable_wandb_logging,
        experiment_name=experiment_name
    )
    
    return BCAcquisitionTrainer(bc_config)