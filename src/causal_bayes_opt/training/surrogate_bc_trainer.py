"""
Behavioral Cloning Trainer for Surrogate (Structure Learning) Models

This module provides a focused BC trainer specifically for learning causal structure
from expert demonstrations. It trains models to predict parent probabilities given
observational data.
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

from ..avici_integration.continuous.model import ContinuousParentSetPredictionModel
from .data_preprocessing import (
    SurrogateTrainingData,
    load_demonstrations_from_path,
    preprocess_demonstration_batch
)

logger = logging.getLogger(__name__)


class SurrogateBCTrainer:
    """
    Behavioral cloning trainer for surrogate models (structure learning).
    
    This trainer focuses on learning to predict parent probabilities from
    observational data, mimicking expert structure discovery behavior.
    """
    
    def __init__(self,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 key_size: int = 32,
                 learning_rate: float = 1e-3,
                 batch_size: int = 32,
                 max_epochs: int = 1000,
                 early_stopping_patience: int = 10,
                 validation_split: float = 0.2,
                 gradient_clip: float = 1.0,
                 dropout: float = 0.1,
                 weight_decay: float = 1e-4,
                 seed: int = 42):
        """
        Initialize surrogate BC trainer.
        
        Args:
            hidden_dim: Hidden dimension for transformer
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            key_size: Size of attention keys
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            max_epochs: Maximum number of training epochs
            early_stopping_patience: Patience for early stopping
            validation_split: Fraction of data for validation
            gradient_clip: Gradient clipping value
            dropout: Dropout rate
            weight_decay: Weight decay for AdamW
            seed: Random seed
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.key_size = key_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.gradient_clip = gradient_clip
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.seed = seed
        
        # Initialize random key
        self.key = random.PRNGKey(seed)
        
        # Model components (initialized during training)
        self.net = None
        self.model_params = None
        self.optimizer = None
        self.optimizer_state = None
        
    def train(self, demonstrations_path: str, max_demos: Optional[int] = None) -> Dict[str, Any]:
        """
        Train surrogate model on expert demonstrations.
        
        Args:
            demonstrations_path: Path to expert demonstrations
            max_demos: Maximum number of demonstration files to load (for testing)
            
        Returns:
            Training results including parameters and metrics
        """
        start_time = time.time()
        logger.info("Starting surrogate BC training")
        
        # Load and preprocess demonstrations
        logger.info(f"Loading demonstrations from {demonstrations_path}")
        raw_demos = load_demonstrations_from_path(demonstrations_path, max_files=max_demos)
        
        # Preprocess to get surrogate training data
        logger.info("Preprocessing demonstrations for structure learning")
        surrogate_data = []
        
        # Limit demonstrations if max_demos is small
        demos_to_process = raw_demos
        if max_demos is not None and max_demos <= 2:
            # For small max_demos, process individual demonstrations
            flat_demos = []
            for item in raw_demos:
                if hasattr(item, 'demonstrations'):
                    flat_demos.extend(item.demonstrations[:max_demos])
                else:
                    flat_demos.append(item)
                if len(flat_demos) >= max_demos:
                    break
            demos_to_process = flat_demos[:max_demos]
            logger.info(f"Limited to {len(demos_to_process)} individual demonstrations")
        
        for demo in demos_to_process:
            preprocessed = preprocess_demonstration_batch([demo])
            if preprocessed['surrogate_data']:
                surrogate_data.extend(preprocessed['surrogate_data'])
        
        if not surrogate_data:
            raise ValueError("No valid surrogate training data found in demonstrations")
        
        logger.info(f"Preprocessed {len(surrogate_data)} training examples")
        
        # Initialize model
        self._initialize_model(surrogate_data[0])
        
        # Split data
        n_val = int(len(surrogate_data) * self.validation_split)
        self.key, split_key = random.split(self.key)
        indices = random.permutation(split_key, len(surrogate_data))
        
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        train_data = [surrogate_data[i] for i in train_indices]
        val_data = [surrogate_data[i] for i in val_indices] if n_val > 0 else []
        
        logger.info(f"Split data: {len(train_data)} train, {len(val_data)} validation")
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_params = None
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            epoch_start = time.time()
            
            # Train epoch
            train_loss = self._train_epoch(train_data)
            train_losses.append(train_loss)
            
            # Validation
            if val_data:
                val_loss = self._evaluate(val_data)
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
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "key_size": self.key_size,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size
            },
            "metrics": {
                "training_time": training_time,
                "epochs_trained": len(train_losses),
                "final_train_loss": train_losses[-1] if train_losses else 0.0,
                "best_val_loss": best_val_loss if val_data else train_losses[-1],
                "train_history": train_losses,
                "val_history": val_losses
            },
            "metadata": {
                "trainer_type": "SurrogateBCTrainer",
                "model_type": "surrogate",
                "n_train_samples": len(train_data),
                "n_val_samples": len(val_data)
            }
        }
        
        logger.info(f"Training completed in {training_time/60:.1f} minutes")
        logger.info(f"Final metrics: train_loss={train_losses[-1]:.4f}, "
                   f"best_val_loss={best_val_loss:.4f}")
        
        return results
    
    def _initialize_model(self, sample_data: SurrogateTrainingData):
        """Initialize model architecture and optimizer."""
        logger.info("Initializing surrogate model")
        
        # Define model function
        def surrogate_fn(data: jnp.ndarray, target_variable: int, is_training: bool = False):
            model = ContinuousParentSetPredictionModel(
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                key_size=self.key_size,
                dropout=self.dropout
            )
            return model(data, target_variable, is_training)
        
        # Transform with Haiku
        self.net = hk.transform(surrogate_fn)
        
        # Initialize parameters
        self.key, init_key = random.split(self.key)
        self.model_params = self.net.init(
            init_key, 
            sample_data.state_tensor, 
            sample_data.target_idx, 
            False
        )
        
        # Initialize optimizer
        self.optimizer = optax.adamw(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.optimizer_state = self.optimizer.init(self.model_params)
        
        logger.info("Model initialized successfully")
    
    def _train_epoch(self, data: List[SurrogateTrainingData]) -> float:
        """Train for one epoch."""
        # Shuffle data
        self.key, shuffle_key = random.split(self.key)
        indices = random.permutation(shuffle_key, len(data))
        
        total_loss = 0.0
        n_batches = 0
        
        # Process in batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = [data[idx] for idx in batch_indices]
            
            # Compute batch loss
            self.key, step_key = random.split(self.key)
            loss = self._train_batch(batch, step_key)
            
            total_loss += loss
            n_batches += 1
        
        return total_loss / n_batches
    
    def _train_batch(self, batch: List[SurrogateTrainingData], rng_key: jax.Array) -> float:
        """Train on a single batch."""
        def loss_fn(params):
            batch_loss = 0.0
            
            for example in batch:
                # Forward pass
                output = self.net.apply(
                    params, rng_key, 
                    example.state_tensor, 
                    example.target_idx, 
                    True  # is_training
                )
                pred_probs = output['parent_probabilities']
                
                # Get true probabilities
                true_probs = jnp.zeros(len(example.variables))
                for i, var in enumerate(example.variables):
                    if var in example.marginal_parent_probs:
                        true_probs = true_probs.at[i].set(
                            example.marginal_parent_probs[var]
                        )
                
                # Mask out target variable
                mask = jnp.ones(len(example.variables))
                mask = mask.at[example.target_idx].set(0.0)
                
                # Binary cross-entropy loss
                eps = 1e-7
                bce = -(
                    true_probs * jnp.log(pred_probs + eps) +
                    (1 - true_probs) * jnp.log(1 - pred_probs + eps)
                )
                
                # Masked loss
                masked_bce = bce * mask
                example_loss = jnp.sum(masked_bce) / jnp.sum(mask)
                
                batch_loss += example_loss
            
            return batch_loss / len(batch)
        
        # Compute gradients
        loss_val, grads = jax.value_and_grad(loss_fn)(self.model_params)
        
        # Clip gradients
        grad_norm = optax.global_norm(grads)
        grads = jax.lax.cond(
            grad_norm > self.gradient_clip,
            lambda g: jax.tree.map(lambda x: x * self.gradient_clip / grad_norm, g),
            lambda g: g,
            grads
        )
        
        # Update parameters
        updates, self.optimizer_state = self.optimizer.update(
            grads, self.optimizer_state, self.model_params
        )
        self.model_params = optax.apply_updates(self.model_params, updates)
        
        return float(loss_val)
    
    def _evaluate(self, data: List[SurrogateTrainingData]) -> float:
        """Evaluate on dataset."""
        total_loss = 0.0
        n_examples = len(data)
        
        for example in data:
            self.key, eval_key = random.split(self.key)
            
            # Forward pass (no training mode)
            output = self.net.apply(
                self.model_params, eval_key,
                example.state_tensor,
                example.target_idx,
                False  # is_training
            )
            pred_probs = output['parent_probabilities']
            
            # Get true probabilities
            true_probs = jnp.zeros(len(example.variables))
            for i, var in enumerate(example.variables):
                if var in example.marginal_parent_probs:
                    true_probs = true_probs.at[i].set(
                        example.marginal_parent_probs[var]
                    )
            
            # Compute loss
            mask = jnp.ones(len(example.variables))
            mask = mask.at[example.target_idx].set(0.0)
            
            eps = 1e-7
            bce = -(
                true_probs * jnp.log(pred_probs + eps) +
                (1 - true_probs) * jnp.log(1 - pred_probs + eps)
            )
            
            loss = jnp.sum(bce * mask) / jnp.sum(mask)
            total_loss += float(loss)
        
        return total_loss / n_examples
    
    def save_checkpoint(self, path: Path, results: Dict[str, Any]) -> None:
        """Save training checkpoint using unified format."""
        from ..utils.checkpoint_utils import save_checkpoint
        
        # Extract architecture from config
        config = results.get('config', {})
        
        architecture = {
            'hidden_dim': config.get('hidden_dim', self.hidden_dim),
            'num_layers': config.get('num_layers', self.num_layers),
            'num_heads': config.get('num_heads', self.num_heads),
            'key_size': config.get('key_size', self.key_size),  # Explicit!
            'dropout': config.get('dropout', self.dropout)
        }
        
        # Training configuration
        training_config = {
            'learning_rate': config.get('learning_rate', self.learning_rate),
            'batch_size': config.get('batch_size', self.batch_size),
            'max_epochs': config.get('max_epochs', self.max_epochs),
            'gradient_clip': self.gradient_clip,
            'weight_decay': self.weight_decay
        }
        
        save_checkpoint(
            path=path,
            params=results['params'],
            architecture=architecture,
            model_type='surrogate',
            model_subtype='continuous_parent_set',
            training_config=training_config,
            metadata=results.get('metadata', {}),
            metrics=results.get('metrics', {})
        )
    
    @classmethod
    def load_checkpoint(cls, path: Path) -> Tuple['SurrogateBCTrainer', Dict[str, Any]]:
        """Load trainer from checkpoint."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Create trainer with saved config
        config = checkpoint['config']
        trainer = cls(
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            key_size=config['key_size'],
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size']
        )
        
        # Set loaded parameters
        trainer.model_params = checkpoint['params']
        
        return trainer, checkpoint