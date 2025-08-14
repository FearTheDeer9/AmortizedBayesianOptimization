"""
Behavioral Cloning Trainer for Policy (Intervention Selection) Models

This module provides a focused BC trainer specifically for learning intervention
policies from expert trajectories. It trains models to select which variables
to intervene on and what values to set.
"""

import time
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
import optax
import numpy as np

from ..policies.clean_bc_policy_factory import create_clean_bc_policy
from .data_preprocessing import load_demonstrations_from_path
from .demonstration_to_tensor import create_bc_training_dataset
from ..utils.variable_mapping import VariableMapper

logger = logging.getLogger(__name__)


def smooth_cross_entropy_loss(logits: jnp.ndarray, target_idx: int, smoothing: float = 0.1) -> jnp.ndarray:
    """
    Cross-entropy loss with label smoothing to prevent overconfidence.
    
    Args:
        logits: Unnormalized log probabilities [n_classes]
        target_idx: Index of the target class
        smoothing: Label smoothing parameter (0 = no smoothing)
        
    Returns:
        Smoothed cross-entropy loss
    """
    n_classes = logits.shape[-1]
    
    # Create smoothed target distribution
    # Instead of one-hot, we use (1-smoothing) for target and smoothing/(n-1) for others
    targets = jnp.ones(n_classes) * smoothing / (n_classes - 1)
    targets = targets.at[target_idx].set(1.0 - smoothing)
    
    # Compute cross-entropy with smoothed targets
    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.sum(targets * log_probs)
    
    return loss


def robust_value_loss(predicted_mean: jnp.ndarray, 
                     predicted_log_std: jnp.ndarray, 
                     target_value: float) -> jnp.ndarray:
    """
    Robust value loss function that prevents explosion.
    
    Uses Huber-style loss for large errors and regularization
    to prevent overconfident predictions.
    
    Args:
        predicted_mean: Predicted mean value
        predicted_log_std: Predicted log standard deviation
        target_value: True target value
        
    Returns:
        Robust value loss
    """
    # Harmonize clipping with model's output range
    log_std = jnp.clip(predicted_log_std, -2.0, 2.0)
    std = jnp.exp(log_std)
    
    # Calculate error
    error = target_value - predicted_mean
    
    # Huber-style loss: quadratic for small errors, linear for large
    huber_delta = 1.0  # Threshold for linear vs quadratic
    normalized_error = jnp.abs(error) / std
    
    # Compute both quadratic and linear terms
    is_small_error = normalized_error <= huber_delta
    quadratic_loss = 0.5 * (error / std) ** 2
    linear_loss = huber_delta * (normalized_error - 0.5 * huber_delta)
    
    # Select appropriate loss based on error magnitude
    mse_term = jnp.where(is_small_error, quadratic_loss, linear_loss)
    
    # Add regularization to prevent overconfident predictions (very small std)
    # This penalizes log_std values approaching -2.0
    std_regularization = 0.01 * jnp.exp(-log_std - 2.0)  # Stronger penalty near boundary
    
    # Combine all terms
    value_loss = 0.5 * jnp.log(2 * jnp.pi) + log_std + mse_term + std_regularization
    
    return value_loss


class PolicyBCTrainer:
    """
    Behavioral cloning trainer for policy models (intervention selection).
    
    This trainer focuses on learning to select interventions from expert
    trajectories, mimicking expert exploration behavior.
    """
    
    def __init__(self,
                 hidden_dim: int = 256,
                 architecture: str = "alternating_attention",
                 learning_rate: float = 3e-4,
                 batch_size: int = 32,
                 max_epochs: int = 1000,
                 early_stopping_patience: int = 10,
                 validation_split: float = 0.2,
                 gradient_clip: float = 1.0,
                 weight_decay: float = 1e-4,
                 seed: int = 42,
                 use_permutation: bool = False,
                 label_smoothing: float = 0.0,
                 permutation_seed: int = 42):
        """
        Initialize policy BC trainer.
        
        Args:
            hidden_dim: Hidden dimension for policy network
            architecture: Architecture type - "simple", "attention", or "alternating_attention"
                         Default is "alternating_attention" which handles permutation symmetries best
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            max_epochs: Maximum number of training epochs
            early_stopping_patience: Patience for early stopping
            validation_split: Fraction of data for validation
            gradient_clip: Gradient clipping value
            weight_decay: Weight decay for AdamW
            seed: Random seed
            use_permutation: Whether to use variable permutation to prevent shortcuts
            label_smoothing: Label smoothing parameter (0 = no smoothing)
            permutation_seed: Seed for permutation generation
        """
        self.hidden_dim = hidden_dim
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.gradient_clip = gradient_clip
        self.weight_decay = weight_decay
        self.seed = seed
        self.use_permutation = use_permutation
        self.label_smoothing = label_smoothing
        self.permutation_seed = permutation_seed
        
        # Initialize random key
        self.key = random.PRNGKey(seed)
        
        if use_permutation:
            logger.info(f"Using variable permutation with seed {permutation_seed}")
        if label_smoothing > 0:
            logger.info(f"Using label smoothing with alpha={label_smoothing}")
        
        # Model components (initialized during training)
        self.net = None
        self.model_params = None
        self.optimizer = None
        self.optimizer_state = None
        
    def train(self, demonstrations_path: str, max_demos: Optional[int] = None, max_trajectory_length: int = 100) -> Dict[str, Any]:
        """
        Train policy model on expert demonstrations.
        
        Args:
            demonstrations_path: Path to expert demonstrations
            max_demos: Maximum number of demonstration files to load (for testing)
            
        Returns:
            Training results including parameters and metrics
        """
        start_time = time.time()
        logger.info("Starting policy BC training")
        
        # Load demonstrations
        logger.info(f"Loading demonstrations from {demonstrations_path}")
        raw_demos = load_demonstrations_from_path(demonstrations_path, max_files=max_demos)
        
        # Flatten demonstrations
        flat_demos = []
        for item in raw_demos:
            if hasattr(item, 'demonstrations'):
                flat_demos.extend(item.demonstrations)
            else:
                flat_demos.append(item)
        
        if max_demos and len(flat_demos) > max_demos:
            flat_demos = flat_demos[:max_demos]
            logger.info(f"Limited to {max_demos} demonstrations")
        
        # Convert to 5-channel training data
        if self.use_permutation:
            logger.info("Converting demonstrations to 5-channel tensors with variable permutation...")
            # Import the permuted version from debugging directory
            import sys
            import os
            from pathlib import Path
            
            # Get the path to debugging-bc-training relative to the repo root
            repo_root = Path(__file__).parent.parent.parent.parent
            debug_path = repo_root / "debugging-bc-training"
            
            if debug_path.exists():
                sys.path.insert(0, str(debug_path))
                from demonstration_to_tensor_permuted import create_bc_training_dataset_permuted
            else:
                # Fallback: try to find it relative to current working directory
                debug_path = Path.cwd() / "debugging-bc-training"
                if debug_path.exists():
                    sys.path.insert(0, str(debug_path))
                    from demonstration_to_tensor_permuted import create_bc_training_dataset_permuted
                else:
                    raise ImportError(f"Cannot find debugging-bc-training directory. Looked in {repo_root / 'debugging-bc-training'} and {debug_path}")
            
            all_inputs, all_labels, dataset_metadata = create_bc_training_dataset_permuted(
                flat_demos, 
                max_trajectory_length,
                use_permutation=True,
                base_seed=self.permutation_seed
            )
        else:
            logger.info("Converting demonstrations to 5-channel tensors with structural knowledge...")
            all_inputs, all_labels, dataset_metadata = create_bc_training_dataset(
                flat_demos, max_trajectory_length
            )
        
        if not all_inputs:
            raise ValueError("No valid training data created from demonstrations")
        
        logger.info(f"Created {len(all_inputs)} training examples from "
                   f"{dataset_metadata['n_demonstrations']} demonstrations")
        logger.info(f"Using 5-channel tensors with shape {dataset_metadata['tensor_shape']}")
        
        # Initialize model and variable mapper
        self._initialize_model(all_inputs[0], dataset_metadata)
        
        # Split data
        n_val = int(len(all_inputs) * self.validation_split)
        self.key, split_key = random.split(self.key)
        indices = random.permutation(split_key, len(all_inputs))
        
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        train_inputs = [all_inputs[i] for i in train_indices]
        train_labels = [all_labels[i] for i in train_indices]
        val_inputs = [all_inputs[i] for i in val_indices] if n_val > 0 else []
        val_labels = [all_labels[i] for i in val_indices] if n_val > 0 else []
        
        logger.info(f"Split data: {len(train_inputs)} train, {len(val_inputs)} validation")
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_params = None
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            epoch_start = time.time()
            
            # Train epoch
            train_loss = self._train_epoch(train_inputs, train_labels)
            train_losses.append(train_loss)
            
            # Validation
            if val_inputs:
                val_loss = self._evaluate(val_inputs, val_labels)
                val_losses.append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = jax.tree.map(lambda x: x, self.model_params)
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.max_epochs}: "
                              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                              f"time={time.time()-epoch_start:.1f}s")
            else:
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.max_epochs}: "
                              f"train_loss={train_loss:.4f}, "
                              f"time={time.time()-epoch_start:.1f}s")
                best_params = self.model_params
        
        # Use best parameters
        if best_params is not None:
            self.model_params = best_params
        
        training_time = time.time() - start_time
        
        # Prepare results
        results = {
            "params": self.model_params,
            "config": {
                "hidden_dim": self.hidden_dim,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size
            },
            "metrics": {
                "training_time": training_time,
                "epochs_trained": len(train_losses),
                "final_train_loss": train_losses[-1] if train_losses else 0.0,
                "best_val_loss": best_val_loss if val_inputs else train_losses[-1],
                "train_history": train_losses,
                "val_history": val_losses
            },
            "metadata": {
                "trainer_type": "PolicyBCTrainer",
                "model_type": "acquisition",
                "n_train_samples": len(train_inputs),
                "n_val_samples": len(val_inputs),
                "n_demonstrations": dataset_metadata['n_demonstrations'],
                "tensor_channels": 5,
                "uses_structural_knowledge": True
            }
        }
        
        logger.info(f"Training completed in {training_time/60:.1f} minutes")
        logger.info(f"Final metrics: train_loss={train_losses[-1]:.4f}, "
                   f"best_val_loss={best_val_loss:.4f}")
        
        return results
    
    def _initialize_model(self, sample_tensor: jnp.ndarray, metadata: Dict[str, Any]):
        """Initialize model architecture and optimizer."""
        logger.info("Initializing policy model for 5-channel input")
        logger.info(f"Input tensor shape: {sample_tensor.shape}")
        
        # Store initial variable info for model shape, but we'll create mappers per example
        self.n_vars = len(metadata['variables'])
        self.original_target_idx = metadata.get('target_idx', 0)
        
        # If using permutation, we need to get the PERMUTED position of the target variable
        if self.use_permutation:
            # Create a mapper to find where the target variable ends up after permutation
            from pathlib import Path
            import sys
            debug_path = Path(__file__).parent.parent.parent.parent / "debugging-bc-training"
            if debug_path.exists():
                sys.path.insert(0, str(debug_path))
                from permuted_variable_mapper import PermutedVariableMapper
                
                mapper = PermutedVariableMapper(
                    metadata['variables'], 
                    seed=self.permutation_seed
                )
                # Get the permuted position of the target variable
                target_var = metadata.get('target_variable')
                if target_var:
                    self.target_idx = mapper.to_permuted_index(target_var)
                    logger.info(f"Using permutation: target variable {target_var} moved from index {self.original_target_idx} to {self.target_idx}")
                else:
                    self.target_idx = self.original_target_idx
                    logger.warning("No target_variable in metadata, using original index")
            else:
                logger.warning(f"Could not find debugging-bc-training for permutation mapper, using original index")
                self.target_idx = self.original_target_idx
        else:
            self.target_idx = self.original_target_idx
        
        # Create policy function with specified architecture
        logger.info(f"Using architecture: {self.architecture}")
        policy_fn = create_clean_bc_policy(
            hidden_dim=self.hidden_dim,
            architecture=self.architecture
        )
        
        # Transform with Haiku
        self.net = hk.transform(policy_fn)
        
        # Initialize parameters with 5-channel tensor
        self.key, init_key = random.split(self.key)
        self.model_params = self.net.init(
            init_key,
            sample_tensor,  # [T, n_vars, 5]
            self.target_idx
        )
        
        # Initialize optimizer
        self.optimizer = optax.adamw(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.optimizer_state = self.optimizer.init(self.model_params)
        
        logger.info("Model initialized successfully for 5-channel tensors with structural knowledge")
    
    def _train_epoch(self, inputs: List[jnp.ndarray], labels: List[Dict[str, Any]]) -> float:
        """Train for one epoch."""
        # Shuffle data
        self.key, shuffle_key = random.split(self.key)
        indices = random.permutation(shuffle_key, len(inputs))
        
        total_loss = 0.0
        n_batches = 0
        
        # Process in batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_inputs = [inputs[idx] for idx in batch_indices]
            batch_labels = [labels[idx] for idx in batch_indices]
            
            self.key, step_key = random.split(self.key)
            loss = self._train_batch(batch_inputs, batch_labels, step_key)
            total_loss += loss
            n_batches += 1
        
        return total_loss / n_batches
    
    def _train_batch(self, batch_inputs: List[jnp.ndarray], 
                     batch_labels: List[Dict[str, Any]], 
                     rng_key: jax.Array) -> float:
        """Train on a batch of examples."""
        def loss_fn(params):
            batch_loss = 0.0
            valid_examples = 0
            
            for input_tensor, label in zip(batch_inputs, batch_labels):
                # Forward pass with 5-channel tensor
                # Use the permuted target index for masking when permutation is enabled
                outputs = self.net.apply(params, rng_key, input_tensor, self.target_idx)
                var_logits = outputs['variable_logits']
                value_params = outputs['value_params']
                
                # Extract intervention target
                target_vars = list(label['targets'])
                if not target_vars:
                    continue
                    
                target_var_name = target_vars[0]
                target_value = label['values'][target_var_name]
                
                # Get target index - either permuted or standard
                if self.use_permutation and 'permuted_target_idx' in label:
                    # Use pre-computed permuted index
                    var_idx = label['permuted_target_idx']
                    if var_idx is None:
                        logger.warning(f"Permuted index is None for target {target_var_name}")
                        continue
                    if not (0 <= var_idx < var_logits.shape[0]):
                        logger.warning(f"Invalid permuted index {var_idx} for n_vars={var_logits.shape[0]}")
                        continue
                else:
                    # Standard mapping
                    example_variables = label.get('variables', [])
                    if not example_variables:
                        logger.warning(f"No variable names in label, skipping")
                        continue
                        
                    example_mapper = VariableMapper(
                        variables=example_variables,
                        target_variable=label.get('target_variable')
                    )
                    
                    # Map variable name to index for this example
                    try:
                        var_idx = example_mapper.get_index(target_var_name)
                    except ValueError:
                        logger.warning(f"Variable {target_var_name} not in example variables: {example_variables}")
                        continue
                
                # Variable selection loss with optional label smoothing
                if self.label_smoothing > 0:
                    var_loss = smooth_cross_entropy_loss(var_logits, var_idx, self.label_smoothing)
                else:
                    var_loss = -jax.nn.log_softmax(var_logits)[var_idx]
                
                # Value prediction loss using robust formulation
                value_mean = value_params[var_idx, 0]
                value_log_std = value_params[var_idx, 1]
                
                # Use robust loss to prevent explosion
                value_loss = robust_value_loss(value_mean, value_log_std, target_value)
                
                total_loss = var_loss + 0.5 * value_loss
                
                # Use jnp.where to handle inf/nan in a JAX-compatible way
                total_loss = jnp.where(
                    jnp.isfinite(total_loss),
                    total_loss,
                    0.0  # Replace inf/nan with 0
                )
                
                batch_loss += total_loss
                valid_examples += 1
            
            if valid_examples == 0:
                logger.warning(f"No valid examples in batch of size {len(batch_inputs)}")
                return 0.0
            
            return batch_loss / valid_examples
        
        # Compute gradients and update
        loss_val, grads = jax.value_and_grad(loss_fn)(self.model_params)
        
        # Clip gradients
        grads, _ = optax.clip_by_global_norm(self.gradient_clip).update(grads, None)
        
        # Apply updates
        updates, self.optimizer_state = self.optimizer.update(
            grads, self.optimizer_state, self.model_params
        )
        self.model_params = optax.apply_updates(self.model_params, updates)
        
        return float(loss_val)
    
    def _evaluate(self, inputs: List[jnp.ndarray], labels: List[Dict[str, Any]]) -> float:
        """Evaluate on dataset."""
        total_loss = 0.0
        n_examples = 0
        
        for input_tensor, label in zip(inputs, labels):
            self.key, eval_key = random.split(self.key)
            
            # Forward pass
            outputs = self.net.apply(self.model_params, eval_key, input_tensor, self.target_idx)
            var_logits = outputs['variable_logits']
            value_params = outputs['value_params']
            
            # Extract target
            target_vars = list(label['targets'])
            if not target_vars:
                continue
                
            target_var_name = target_vars[0]
            target_value = label['values'][target_var_name]
            
            # Get target index - either permuted or standard
            if self.use_permutation and 'permuted_target_idx' in label:
                var_idx = label['permuted_target_idx']
                if var_idx is None:
                    logger.warning(f"Permuted index is None for target {target_var_name} in evaluation")
                    continue
            else:
                # Standard mapping
                example_variables = label.get('variables', [])
                if not example_variables:
                    logger.warning(f"No variable names in label, skipping")
                    continue
                    
                example_mapper = VariableMapper(
                    variables=example_variables,
                    target_variable=label.get('target_variable')
                )
                
                # Map to index
                try:
                    var_idx = example_mapper.get_index(target_var_name)
                except ValueError:
                    logger.warning(f"Variable {target_var_name} not in example variables")
                    continue
            
            # Validate index bounds
            if not (0 <= var_idx < var_logits.shape[0]):
                logger.warning(f"Invalid index {var_idx} for n_vars={var_logits.shape[0]} in evaluation")
                continue
            
            # Compute losses
            var_loss = -jax.nn.log_softmax(var_logits)[var_idx]
            
            value_mean = value_params[var_idx, 0]
            value_log_std = value_params[var_idx, 1]
            
            # Use robust loss to prevent explosion
            value_loss = robust_value_loss(value_mean, value_log_std, target_value)
            
            example_loss = var_loss + 0.5 * value_loss
            
            # Only add finite losses
            if jnp.isfinite(example_loss):
                total_loss += float(example_loss)
                n_examples += 1
            else:
                logger.debug(f"Skipping non-finite loss in evaluation: {example_loss}")
        
        return total_loss / n_examples if n_examples > 0 else 0.0
    
    def save_checkpoint(self, path: Path, results: Dict[str, Any]) -> None:
        """Save training checkpoint using unified format."""
        from ..utils.checkpoint_utils import save_checkpoint
        
        # Extract architecture
        config = results.get('config', {})
        
        architecture = {
            'hidden_dim': config.get('hidden_dim', self.hidden_dim),
            'architecture_type': self.architecture
        }
        
        # Training configuration
        training_config = {
            'learning_rate': config.get('learning_rate', self.learning_rate),
            'batch_size': config.get('batch_size', self.batch_size),
            'max_epochs': self.max_epochs,
            'gradient_clip': self.gradient_clip,
            'weight_decay': self.weight_decay,
            'validation_split': self.validation_split
        }
        
        save_checkpoint(
            path=path,
            params=results['params'],
            architecture=architecture,
            model_type='policy',
            model_subtype='bc',
            training_config=training_config,
            metadata=results.get('metadata', {}),
            metrics=results.get('metrics', {})
        )
    
    @classmethod
    def load_checkpoint(cls, path: Path) -> Tuple['PolicyBCTrainer', Dict[str, Any]]:
        """Load trainer from checkpoint."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Create trainer with saved config
        config = checkpoint['config']
        trainer = cls(
            hidden_dim=config['hidden_dim'],
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size']
        )
        
        # Set loaded parameters
        trainer.model_params = checkpoint['params']
        
        return trainer, checkpoint