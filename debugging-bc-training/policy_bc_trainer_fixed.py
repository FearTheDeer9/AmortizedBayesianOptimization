"""
Behavioral Cloning Trainer for Policy (Intervention Selection) Models - FIXED VERSION

This version uses numerical sorting for variable mapping to match the data preparation.
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

# Import the FIXED variable mapper with numerical sorting
import sys
sys.path.append(str(Path(__file__).parent))
from variable_mapping_fixed import VariableMapper

# Import from parent
sys.path.append(str(Path(__file__).parent.parent))
from src.causal_bayes_opt.policies.clean_bc_policy_factory import create_clean_bc_policy
from src.causal_bayes_opt.training.data_preprocessing import load_demonstrations_from_path
from demonstration_to_tensor_fixed import create_bc_training_dataset

logger = logging.getLogger(__name__)


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
    
    # Add safety check to prevent inf
    value_loss = jnp.where(
        jnp.isfinite(value_loss),
        value_loss,
        100.0  # Cap at large but finite value
    )
    
    return value_loss


class PolicyBCTrainer:
    """
    Behavioral cloning trainer for policy models (intervention selection).
    
    This trainer focuses on learning to select interventions from expert
    trajectories, mimicking expert exploration behavior.
    """
    
    def __init__(self,
                 hidden_dim: int = 256,
                 learning_rate: float = 3e-4,
                 batch_size: int = 32,
                 max_epochs: int = 1000,
                 early_stopping_patience: int = 10,
                 validation_split: float = 0.2,
                 gradient_clip: float = 1.0,
                 weight_decay: float = 1e-4,
                 seed: int = 42):
        """
        Initialize policy BC trainer.
        
        Args:
            hidden_dim: Hidden dimension for policy network
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            max_epochs: Maximum number of training epochs
            early_stopping_patience: Patience for early stopping
            validation_split: Fraction of data for validation
            gradient_clip: Gradient clipping value
            weight_decay: Weight decay for AdamW
            seed: Random seed
        """
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.gradient_clip = gradient_clip
        self.weight_decay = weight_decay
        self.seed = seed
        
        # Initialize random key
        self.key = random.PRNGKey(seed)
        
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
        logger.info("Starting policy BC training with FIXED variable mapping")
        
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
        
        # Convert to 5-channel training data with NUMERICAL SORTING
        logger.info("Converting demonstrations to 5-channel tensors with numerical sorting...")
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
                "uses_structural_knowledge": True,
                "uses_numerical_sorting": True  # Key flag
            }
        }
        
        logger.info(f"Training completed in {training_time/60:.1f} minutes")
        logger.info(f"Final metrics: train_loss={train_losses[-1]:.4f}, "
                   f"best_val_loss={best_val_loss:.4f}")
        
        return results
    
    def _initialize_model(self, sample_tensor: jnp.ndarray, metadata: Dict[str, Any]):
        """Initialize model architecture and optimizer."""
        logger.info("Initializing policy model for 5-channel input with numerical sorting")
        logger.info(f"Input tensor shape: {sample_tensor.shape}")
        
        # Store initial variable info for model shape, but we'll create mappers per example
        self.n_vars = len(metadata['variables'])
        self.target_idx = metadata.get('target_idx', 0)
        
        # Create policy function
        policy_fn = create_clean_bc_policy(hidden_dim=self.hidden_dim)
        
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
        
        logger.info("Model initialized successfully for 5-channel tensors with numerical sorting")
    
    def _train_epoch(self, inputs: List[jnp.ndarray], labels: List[Dict[str, Any]]) -> float:
        """Train for one epoch."""
        # Shuffle data
        self.key, shuffle_key = random.split(self.key)
        indices = random.permutation(shuffle_key, len(inputs))
        
        total_loss = 0.0
        n_batches = 0
        inf_count = 0
        
        # Process in batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_inputs = [inputs[idx] for idx in batch_indices]
            batch_labels = [labels[idx] for idx in batch_indices]
            
            self.key, step_key = random.split(self.key)
            loss = self._train_batch(batch_inputs, batch_labels, step_key)
            
            # Check for inf/nan
            if not jnp.isfinite(loss):
                inf_count += 1
                logger.debug(f"Batch {i//self.batch_size} produced inf/nan loss")
                # Use a large but finite loss for training stability
                loss = 100.0
            
            total_loss += loss
            n_batches += 1
        
        if inf_count > 0:
            logger.warning(f"Epoch had {inf_count}/{n_batches} batches with inf/nan loss")
        
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
                outputs = self.net.apply(params, rng_key, input_tensor, self.target_idx)
                var_logits = outputs['variable_logits']
                value_params = outputs['value_params']
                
                # Extract intervention target
                target_vars = list(label.get('targets', []))
                if not target_vars:
                    continue
                    
                target_var_name = target_vars[0]
                target_value = label['values'].get(target_var_name, None)
                
                if target_value is None:
                    continue
                
                # Create variable mapper with NUMERICAL SORTING for this example
                example_variables = label.get('variables', [])
                if not example_variables:
                    logger.warning(f"No variable names in label, skipping")
                    continue
                    
                # Use the FIXED VariableMapper with numerical sorting
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
                
                # Variable selection loss
                var_loss = -jax.nn.log_softmax(var_logits)[var_idx]
                
                # Value prediction loss using robust formulation
                value_mean = value_params[var_idx, 0]
                value_log_std = value_params[var_idx, 1]
                
                # Use robust loss to prevent explosion
                value_loss = robust_value_loss(value_mean, value_log_std, target_value)
                
                # Combine losses with safety check
                example_loss = var_loss + 0.5 * value_loss
                
                # Only add if finite
                example_loss = jnp.where(
                    jnp.isfinite(example_loss),
                    example_loss,
                    10.0  # Use large but finite penalty for bad examples
                )
                
                batch_loss += example_loss
                valid_examples += 1
            
            # Average over valid examples
            if valid_examples > 0:
                return batch_loss / valid_examples
            else:
                return 0.0
        
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
            target_vars = list(label.get('targets', []))
            if not target_vars:
                continue
                
            target_var_name = target_vars[0]
            target_value = label['values'].get(target_var_name, None)
            
            if target_value is None:
                continue
            
            # Create variable mapper with NUMERICAL SORTING for this example
            example_variables = label.get('variables', [])
            if not example_variables:
                logger.warning(f"No variable names in label, skipping")
                continue
                
            # Use the FIXED VariableMapper with numerical sorting
            example_mapper = VariableMapper(
                variables=example_variables,
                target_variable=label.get('target_variable')
            )
            
            # Map to index
            try:
                var_idx = example_mapper.get_index(target_var_name)
            except ValueError:
                logger.warning(f"Variable {target_var_name} not in example variables: {example_variables}")
                continue
            
            # Compute losses
            var_loss = -jax.nn.log_softmax(var_logits)[var_idx]
            
            value_mean = value_params[var_idx, 0]
            value_log_std = value_params[var_idx, 1]
            
            # Use robust loss to prevent explosion
            value_loss = robust_value_loss(value_mean, value_log_std, target_value)
            
            # Combine with safety check
            example_loss = var_loss + 0.5 * value_loss
            if jnp.isfinite(example_loss):
                total_loss += float(example_loss)
                n_examples += 1
        
        return total_loss / n_examples if n_examples > 0 else 0.0
    
    def save_checkpoint(self, path: Path, results: Dict[str, Any]) -> None:
        """Save training checkpoint using unified format."""
        from src.causal_bayes_opt.utils.checkpoint_utils import save_checkpoint
        
        # Extract architecture
        config = results.get('config', {})
        
        architecture = {
            'hidden_dim': config.get('hidden_dim', self.hidden_dim)
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