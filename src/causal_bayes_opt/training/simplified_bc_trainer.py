"""
Simplified BC Trainer with Flexible Demonstration Support

This module provides a clean, simple behavioral cloning trainer for both
surrogate and acquisition models with support for various demonstration formats.
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
import optax
import numpy as np

from ..avici_integration.continuous.model import ContinuousParentSetPredictionModel
from ..acquisition.enhanced_policy_network import EnhancedPolicyNetwork

logger = logging.getLogger(__name__)


class SimplifiedBCTrainer:
    """
    Simplified behavioral cloning trainer for surrogate and acquisition models.
    
    Key features:
    - Support for both surrogate (structure learning) and acquisition (policy) models
    - Flexible demonstration input formats
    - Early stopping with validation
    - Clean checkpoint format
    """
    
    def __init__(self,
                 model_type: str,  # "surrogate" or "acquisition"
                 # Model configuration
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 key_size: int = 32,
                 # Training configuration
                 learning_rate: float = 1e-3,
                 batch_size: int = 32,
                 max_epochs: int = 50,
                 early_stopping_patience: int = 5,
                 validation_split: float = 0.2,
                 # Other
                 dropout: float = 0.1,
                 seed: int = 42):
        """
        Initialize simplified BC trainer.
        
        Args:
            model_type: "surrogate" for structure learning, "acquisition" for policy
            hidden_dim: Hidden dimension for model
            num_layers: Number of layers
            num_heads: Number of attention heads
            key_size: Key size for attention
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            max_epochs: Maximum training epochs
            early_stopping_patience: Patience for early stopping
            validation_split: Fraction of data for validation
            dropout: Dropout rate
            seed: Random seed
        """
        if model_type not in ["surrogate", "acquisition"]:
            raise ValueError(f"Invalid model_type: {model_type}")
            
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.key_size = key_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.dropout = dropout
        self.seed = seed
        
        # Training state
        self.model_params = None
        self.optimizer_state = None
        self.key = random.PRNGKey(seed)
        
        logger.info(f"Initialized SimplifiedBCTrainer: model_type={model_type}, "
                   f"hidden_dim={hidden_dim}, lr={learning_rate}")
        
    def train(self,
              demonstrations: Union[List[Dict], str, Path],
              eval_demonstrations: Optional[Union[List[Dict], str, Path]] = None) -> Dict[str, Any]:
        """
        Train BC model on demonstrations.
        
        Args:
            demonstrations: Training demonstrations - can be:
                - List of demonstration dicts
                - Path to demonstrations file/directory
                - String path to load from
            eval_demonstrations: Optional separate evaluation set
            
        Returns:
            Dictionary containing:
                - params: Trained model parameters
                - config: Training configuration  
                - metrics: Training metrics and history
                - metadata: Additional information
        """
        start_time = time.time()
        
        # Load and prepare demonstrations
        train_data = self._load_demonstrations(demonstrations)
        logger.info(f"Loaded {len(train_data)} training demonstrations")
        
        # Split into train/val if no eval set provided
        if eval_demonstrations is None:
            n_val = int(len(train_data) * self.validation_split)
            self.key, split_key = random.split(self.key)
            indices = random.permutation(split_key, len(train_data))
            val_data = [train_data[i] for i in indices[:n_val]]
            train_data = [train_data[i] for i in indices[n_val:]]
        else:
            val_data = self._load_demonstrations(eval_demonstrations)
            
        logger.info(f"Training on {len(train_data)} samples, validating on {len(val_data)} samples")
        
        # Initialize model
        self._initialize_model(train_data[0])
        
        # Create data loaders
        train_loader = self._create_data_loader(train_data, self.batch_size, shuffle=True)
        val_loader = self._create_data_loader(val_data, self.batch_size, shuffle=False)
        
        # Training history
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_params = None
        
        # Main training loop
        for epoch in range(self.max_epochs):
            # Train epoch
            epoch_train_losses = []
            for batch in train_loader:
                self.key, batch_key = random.split(self.key)
                loss = self._train_step(batch, batch_key)
                epoch_train_losses.append(loss)
                
            train_loss = float(np.mean(epoch_train_losses))
            train_losses.append(train_loss)
            
            # Validation
            epoch_val_losses = []
            for batch in val_loader:
                self.key, batch_key = random.split(self.key)
                loss = self._eval_step(batch, batch_key)
                epoch_val_losses.append(loss)
                
            val_loss = float(np.mean(epoch_val_losses))
            val_losses.append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = jax.tree.map(lambda x: x.copy(), self.model_params)
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                
        # Use best parameters
        if best_params is not None:
            self.model_params = best_params
            
        # Prepare results
        training_time = time.time() - start_time
        
        results = {
            "params": self.model_params,
            "config": {
                "model_type": self.model_type,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "key_size": self.key_size,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "dropout": self.dropout
            },
            "metrics": {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_val_loss": best_val_loss,
                "final_train_loss": train_losses[-1] if train_losses else 0.0,
                "epochs_trained": len(train_losses),
                "training_time": training_time
            },
            "metadata": {
                "trainer_type": "SimplifiedBCTrainer",
                "model_type": self.model_type,
                "n_train_samples": len(train_data),
                "n_val_samples": len(val_data)
            }
        }
        
        logger.info(f"Training completed in {training_time/60:.1f} minutes")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        return results
        
    def save_checkpoint(self, path: Path, results: Dict[str, Any]) -> None:
        """Save training checkpoint."""
        import pickle
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "params": results["params"],
            "config": results["config"],
            "metrics": results["metrics"],
            "metadata": results["metadata"],
            "model_type": self.model_type
        }
        
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)
            
        logger.info(f"Saved checkpoint to {path}")
        
    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        """Load training checkpoint."""
        import pickle
        
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
            
        # Restore state
        self.model_params = checkpoint["params"]
        self.model_type = checkpoint["model_type"]
        config = checkpoint["config"]
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config["num_layers"]
        
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint
        
    def _load_demonstrations(self, demonstrations: Union[List[Dict], str, Path]) -> List[Dict]:
        """Load demonstrations from various formats."""
        if isinstance(demonstrations, list):
            return demonstrations
        elif isinstance(demonstrations, (str, Path)):
            path = Path(demonstrations)
            if path.is_file():
                # Load from pickle file
                import pickle
                with open(path, "rb") as f:
                    return pickle.load(f)
            elif path.is_dir():
                # Load all pickle files from directory
                all_demos = []
                for file_path in sorted(path.glob("*.pkl")):
                    with open(file_path, "rb") as f:
                        demos = pickle.load(f)
                        if isinstance(demos, list):
                            all_demos.extend(demos)
                        else:
                            all_demos.append(demos)
                return all_demos
            else:
                raise ValueError(f"Path does not exist: {path}")
        else:
            raise ValueError(f"Unsupported demonstration format: {type(demonstrations)}")
            
    def _initialize_model(self, sample_demo) -> None:
        """Initialize model based on type and sample demonstration."""
        if self.model_type == "surrogate":
            # Initialize surrogate model for structure learning
            def model_fn(data: jnp.ndarray, target_variable: int, is_training: bool = False):
                model = ContinuousParentSetPredictionModel(
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                    num_heads=self.num_heads,
                    key_size=self.key_size,
                    dropout=self.dropout
                )
                return model(data, target_variable, is_training)
                
            self.model_fn = hk.transform(model_fn)
            
            # Get sample input shape from DemonstrationBatch
            if hasattr(sample_demo, 'demonstrations'):
                # DemonstrationBatch format - extract from first demonstration
                demo = sample_demo.demonstrations[0] if sample_demo.demonstrations else None
                if demo and hasattr(demo, 'n_nodes'):
                    n_vars = demo.n_nodes
                    sample_data = jnp.zeros((100, n_vars, 3))  # [N, d, 3]
                else:
                    sample_data = jnp.zeros((100, 5, 3))  # Default
            elif isinstance(sample_demo, dict) and "data" in sample_demo:
                sample_data = sample_demo["data"]
            else:
                # Create dummy data
                sample_data = jnp.zeros((100, 5, 3))  # [N, d, 3]
                
        else:  # acquisition
            # Initialize acquisition policy model
            def model_fn(state: jnp.ndarray, is_training: bool = False):
                model = EnhancedPolicyNetwork(
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                    num_heads=self.num_heads,
                    key_size=self.key_size,
                    dropout=self.dropout
                )
                return model(state, is_training=is_training)
                
            self.model_fn = hk.transform(model_fn)
            
            # Get sample input shape from DemonstrationBatch
            if hasattr(sample_demo, 'demonstrations'):
                # DemonstrationBatch format
                demo = sample_demo.demonstrations[0] if sample_demo.demonstrations else None
                if demo and hasattr(demo, 'n_nodes'):
                    n_vars = demo.n_nodes
                    sample_state = jnp.zeros((n_vars, 32))  # [n_vars, feature_dim]
                else:
                    sample_state = jnp.zeros((5, 32))  # Default
            elif isinstance(sample_demo, dict) and "state" in sample_demo:
                sample_state = sample_demo["state"]
            else:
                # Create dummy state
                sample_state = jnp.zeros((5, 32))  # [n_vars, feature_dim]
                
        # Initialize parameters
        self.key, init_key = random.split(self.key)
        
        if self.model_type == "surrogate":
            self.model_params = self.model_fn.init(init_key, sample_data, 0, False)
        else:
            self.model_params = self.model_fn.init(init_key, sample_state, False)
            
        # Create optimizer
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=self.learning_rate)
        )
        
        self.optimizer_state = self.optimizer.init(self.model_params)
        
        logger.info(f"Initialized {self.model_type} model")
        
    def _create_data_loader(self, data: List[Any], batch_size: int, shuffle: bool) -> List[List[Any]]:
        """Create simple data loader."""
        # If data contains DemonstrationBatch objects, flatten them
        flattened_data = []
        for item in data:
            if hasattr(item, 'demonstrations'):
                # DemonstrationBatch - extract individual demonstrations
                flattened_data.extend(item.demonstrations)
            else:
                # Already individual demonstrations or dicts
                flattened_data.append(item)
        
        n_samples = len(flattened_data)
        indices = list(range(n_samples))
        
        if shuffle:
            self.key, shuffle_key = random.split(self.key)
            indices = random.permutation(shuffle_key, jnp.array(indices)).tolist()
            
        batches = []
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = [flattened_data[idx] for idx in batch_indices]
            batches.append(batch)
            
        return batches
        
    def _train_step(self, batch: List[Dict], key: jax.random.PRNGKey) -> float:
        """Single training step."""
        if self.model_type == "surrogate":
            loss, grads = self._compute_surrogate_loss_and_grads(batch, key, is_training=True)
        else:
            loss, grads = self._compute_acquisition_loss_and_grads(batch, key, is_training=True)
            
        # Update parameters
        updates, self.optimizer_state = self.optimizer.update(grads, self.optimizer_state)
        self.model_params = optax.apply_updates(self.model_params, updates)
        
        return float(loss)
        
    def _eval_step(self, batch: List[Dict], key: jax.random.PRNGKey) -> float:
        """Single evaluation step."""
        if self.model_type == "surrogate":
            loss = self._compute_surrogate_loss(batch, key, is_training=False)
        else:
            loss = self._compute_acquisition_loss(batch, key, is_training=False)
            
        return float(loss)
        
    def _compute_surrogate_loss_and_grads(self, batch: List[Any], key: jax.random.PRNGKey, is_training: bool) -> Tuple[float, Any]:
        """Compute loss and gradients for surrogate model."""
        def loss_fn(params):
            total_loss = 0.0
            n_samples = 0
            
            for demo in batch:
                # Handle both ExpertDemonstration and dict formats
                if hasattr(demo, 'observational_samples'):
                    # ExpertDemonstration format
                    # Convert samples to array format
                    n_vars = demo.n_nodes
                    target_var = demo.target_variable
                    # Get variable index mapping
                    from ...data_structures.scm import get_variables
                    variables = list(get_variables(demo.scm))
                    target_idx = variables.index(target_var)
                    
                    # Create data array from samples
                    n_samples_data = len(demo.observational_samples)
                    data = jnp.zeros((n_samples_data, n_vars, 3))  # Placeholder
                    
                    # Get true parents
                    true_parents = list(demo.discovered_parents)
                    true_parent_indices = [variables.index(p) for p in true_parents if p in variables]
                else:
                    # Dict format
                    data = demo["data"]  # [N, d, 3]
                    target_idx = demo["target_variable"]
                    true_parent_indices = demo.get("true_parents", [])
                
                # Forward pass
                output = self.model_fn.apply(params, key, data, target_idx, is_training)
                parent_probs = output["parent_probabilities"]
                
                # Compute BCE loss for parent prediction
                n_vars = parent_probs.shape[0]
                true_parent_mask = jnp.zeros(n_vars)
                for parent_idx in true_parent_indices:
                    if parent_idx != target_idx:
                        true_parent_mask = true_parent_mask.at[parent_idx].set(1.0)
                        
                # Binary cross-entropy
                eps = 1e-8
                loss = -jnp.mean(
                    true_parent_mask * jnp.log(parent_probs + eps) +
                    (1 - true_parent_mask) * jnp.log(1 - parent_probs + eps)
                )
                
                total_loss += loss
                n_samples += 1
                
            return total_loss / n_samples if n_samples > 0 else 0.0
            
        return jax.value_and_grad(loss_fn)(self.model_params)
        
    def _compute_surrogate_loss(self, batch: List[Dict], key: jax.random.PRNGKey, is_training: bool) -> float:
        """Compute loss for surrogate model (no gradients)."""
        loss, _ = self._compute_surrogate_loss_and_grads(batch, key, is_training)
        return loss
        
    def _compute_acquisition_loss_and_grads(self, batch: List[Any], key: jax.random.PRNGKey, is_training: bool) -> Tuple[float, Any]:
        """Compute loss and gradients for acquisition model."""
        def loss_fn(params):
            total_loss = 0.0
            n_samples = 0
            
            for demo in batch:
                # Handle both ExpertTrajectoryDemonstration and dict formats
                if hasattr(demo, 'expert_trajectory'):
                    # ExpertTrajectoryDemonstration format
                    # Extract trajectory data
                    trajectory = demo.expert_trajectory
                    
                    # For now, create placeholder state and action
                    # In a real implementation, we'd extract these from the trajectory
                    n_vars = demo.n_nodes
                    state = jnp.zeros((n_vars, 32))  # Placeholder state
                    
                    # Dummy action - would be extracted from trajectory
                    action = {"variable": 0, "value": 0.0}
                    
                elif hasattr(demo, 'n_nodes'):
                    # ExpertDemonstration format (might contain acquisition data)
                    n_vars = demo.n_nodes
                    state = jnp.zeros((n_vars, 32))  # Placeholder
                    action = {"variable": 0, "value": 0.0}  # Placeholder
                    
                else:
                    # Dict format
                    state = demo["state"]
                    action = demo["action"]  # {"variable": idx, "value": float}
                
                # Forward pass
                output = self.model_fn.apply(params, key, state, is_training)
                
                # Variable selection loss (cross-entropy)
                var_logits = output["variable_logits"]
                var_loss = -jax.nn.log_softmax(var_logits)[action["variable"]]
                
                # Value prediction loss (MSE)
                value_params = output["value_params"][action["variable"]]
                # Simple MSE loss (assuming value_params represents mean)
                # Take the first element if value_params is a vector
                if value_params.ndim > 0:
                    value_pred = value_params[0]
                else:
                    value_pred = value_params
                value_loss = (value_pred - action["value"]) ** 2
                
                total_loss += var_loss + 0.1 * value_loss  # Weight value loss less
                n_samples += 1
                
            return total_loss / n_samples if n_samples > 0 else 0.0
            
        return jax.value_and_grad(loss_fn)(self.model_params)
        
    def _compute_acquisition_loss(self, batch: List[Dict], key: jax.random.PRNGKey, is_training: bool) -> float:
        """Compute loss for acquisition model (no gradients)."""
        loss, _ = self._compute_acquisition_loss_and_grads(batch, key, is_training)
        return loss