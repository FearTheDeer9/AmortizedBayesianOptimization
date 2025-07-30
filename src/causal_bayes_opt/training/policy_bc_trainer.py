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
from .data_preprocessing import (
    PolicyTrainingData,
    load_demonstrations_from_path,
    preprocess_demonstration_batch
)

logger = logging.getLogger(__name__)


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
        
    def train(self, demonstrations_path: str, max_demos: Optional[int] = None) -> Dict[str, Any]:
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
        
        # Load and preprocess demonstrations
        logger.info(f"Loading demonstrations from {demonstrations_path}")
        raw_demos = load_demonstrations_from_path(demonstrations_path, max_files=max_demos)
        
        # Preprocess to get policy training data
        logger.info("Preprocessing demonstrations for policy learning")
        policy_data = []
        
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
            if preprocessed['policy_data']:
                policy_data.extend(preprocessed['policy_data'])
        
        if not policy_data:
            raise ValueError("No valid policy training data found in demonstrations")
        
        logger.info(f"Preprocessed {len(policy_data)} training trajectories")
        
        # Initialize model
        self._initialize_model(policy_data[0])
        
        # Split data
        n_val = int(len(policy_data) * self.validation_split)
        self.key, split_key = random.split(self.key)
        indices = random.permutation(split_key, len(policy_data))
        
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        train_data = [policy_data[i] for i in train_indices]
        val_data = [policy_data[i] for i in val_indices] if n_val > 0 else []
        
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
                "trainer_type": "PolicyBCTrainer",
                "model_type": "acquisition",
                "n_train_samples": len(train_data),
                "n_val_samples": len(val_data)
            }
        }
        
        logger.info(f"Training completed in {training_time/60:.1f} minutes")
        logger.info(f"Final metrics: train_loss={train_losses[-1]:.4f}, "
                   f"best_val_loss={best_val_loss:.4f}")
        
        return results
    
    def _initialize_model(self, sample_data: PolicyTrainingData):
        """Initialize model architecture and optimizer."""
        logger.info("Initializing policy model")
        
        # Create policy function
        policy_fn = create_clean_bc_policy(hidden_dim=self.hidden_dim)
        
        # Transform with Haiku
        self.net = hk.transform(policy_fn)
        
        # Initialize parameters with full trajectory states
        self.key, init_key = random.split(self.key)
        # Policy expects full trajectory [T, n_vars, 3]
        self.model_params = self.net.init(
            init_key,
            sample_data.states,  # Full trajectory [T, n_vars, 3]
            sample_data.target_idx
        )
        
        # Initialize optimizer
        self.optimizer = optax.adamw(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.optimizer_state = self.optimizer.init(self.model_params)
        
        logger.info("Model initialized successfully")
    
    def _train_epoch(self, data: List[PolicyTrainingData]) -> float:
        """Train for one epoch."""
        # Shuffle data
        self.key, shuffle_key = random.split(self.key)
        indices = random.permutation(shuffle_key, len(data))
        
        total_loss = 0.0
        n_examples = 0
        
        # Process each trajectory
        for idx in indices:
            trajectory = data[idx]
            self.key, step_key = random.split(self.key)
            
            # Train on this trajectory
            loss = self._train_trajectory(trajectory, step_key)
            total_loss += loss
            n_examples += 1
        
        return total_loss / n_examples
    
    def _train_trajectory(self, trajectory: PolicyTrainingData, rng_key: jax.Array) -> float:
        """Train on a single trajectory."""
        def loss_fn(params):
            trajectory_loss = 0.0
            n_steps = 0
            
            # Process each step in the trajectory
            for t in range(len(trajectory.intervention_vars)):
                
                # Get state history up to time t
                # Policy expects history [T', n_vars, 3] where T' = t+1
                state_history = trajectory.states[:t+1]  # [t+1, n_vars, 3]
                expert_var = int(trajectory.intervention_vars[t])
                expert_value = trajectory.intervention_values[t]
                
                # Forward pass with state history
                outputs = self.net.apply(params, rng_key, state_history, trajectory.target_idx)
                var_logits = outputs['variable_logits']
                value_params = outputs['value_params']
                
                # Variable selection loss (cross-entropy)
                var_probs = jax.nn.softmax(var_logits)
                var_loss = -jnp.log(var_probs[expert_var] + 1e-8)
                
                # Value prediction loss (negative log likelihood)
                mean = value_params[expert_var, 0]
                log_std = value_params[expert_var, 1]
                std = jnp.exp(jnp.clip(log_std, -5, 2))
                
                # NLL of expert value under predicted Gaussian
                value_loss = 0.5 * jnp.log(2 * jnp.pi) + log_std + \
                            0.5 * ((expert_value - mean) / std) ** 2
                
                # Combined loss (weight value loss less)
                step_loss = var_loss + 0.5 * value_loss
                
                trajectory_loss += step_loss
                n_steps += 1
            
            # Average over steps and ensure scalar
            avg_loss = trajectory_loss / jnp.maximum(n_steps, 1)
            return jnp.squeeze(avg_loss)  # Ensure scalar output
        
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
    
    def _evaluate(self, data: List[PolicyTrainingData]) -> float:
        """Evaluate on dataset."""
        total_loss = 0.0
        n_trajectories = len(data)
        
        for trajectory in data:
            self.key, eval_key = random.split(self.key)
            
            traj_loss = 0.0
            n_steps = 0
            
            # Evaluate each step
            for t in range(len(trajectory.intervention_vars)):
                
                # Get state history up to time t
                state_history = trajectory.states[:t+1]
                expert_var = int(trajectory.intervention_vars[t])
                expert_value = trajectory.intervention_values[t]
                
                # Forward pass with state history
                outputs = self.net.apply(
                    self.model_params, eval_key,
                    state_history, trajectory.target_idx
                )
                var_logits = outputs['variable_logits']
                value_params = outputs['value_params']
                
                # Compute losses
                var_probs = jax.nn.softmax(var_logits)
                var_loss = -jnp.log(var_probs[expert_var] + 1e-8)
                
                mean = value_params[expert_var, 0]
                log_std = value_params[expert_var, 1]
                std = jnp.exp(jnp.clip(log_std, -5, 2))
                
                value_loss = 0.5 * jnp.log(2 * jnp.pi) + log_std + \
                            0.5 * ((expert_value - mean) / std) ** 2
                
                step_loss = var_loss + 0.5 * value_loss
                traj_loss += float(jnp.squeeze(step_loss))
                n_steps += 1
            
            if n_steps > 0:
                total_loss += traj_loss / n_steps
        
        return total_loss / n_trajectories
    
    def save_checkpoint(self, path: Path, results: Dict[str, Any]) -> None:
        """Save training checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'params': results['params'],
            'config': results['config'],
            'metrics': results['metrics'],
            'metadata': results['metadata']
        }
        
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Saved policy checkpoint to {path}")
    
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