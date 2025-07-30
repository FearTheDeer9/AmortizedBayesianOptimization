"""
Production Active Learning for Surrogate Models

This module provides active learning capabilities for surrogate models during
evaluation. Unlike the demo version, this uses actual SCM variable names and
integrates properly with the ACBO workflow.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
import optax
import pyrsistent as pyr
import numpy as np

from ..avici_integration.parent_set.model import ParentSetPredictionModel
from ..avici_integration.parent_set.inference import predict_parent_posterior
from ..avici_integration.core import samples_to_avici_format
from ..data_structures.scm import get_variables
from ..data_structures.sample import get_values, get_intervention_targets

logger = logging.getLogger(__name__)


class ActiveLearningSurrogate:
    """
    Active learning wrapper for surrogate models.
    
    This class provides online learning capabilities for surrogate models,
    allowing them to update their parameters based on new observations
    during evaluation.
    """
    
    def __init__(self,
                 variables: List[str],
                 initial_params: Optional[Any] = None,
                 learning_rate: float = 1e-3,
                 scoring_method: str = "bic",
                 model_config: Optional[Dict[str, Any]] = None,
                 seed: int = 42):
        """
        Initialize active learning surrogate.
        
        Args:
            variables: List of actual SCM variable names
            initial_params: Optional pre-trained parameters to start from
            learning_rate: Learning rate for online updates
            scoring_method: Scoring method for structure learning ("bic", "aic", "mdl", "likelihood")
            model_config: Model architecture configuration
            seed: Random seed
        """
        self.variables = variables
        self.n_vars = len(variables)
        self.learning_rate = learning_rate
        self.scoring_method = scoring_method
        self.key = random.PRNGKey(seed)
        
        # Default model configuration
        default_config = {
            "layers": 4,
            "dim": 128,
            "max_parent_size": min(5, self.n_vars - 1)
        }
        self.model_config = {**default_config, **(model_config or {})}
        
        # Create model
        self._create_model()
        
        # Initialize or use provided parameters
        if initial_params is not None:
            self.params = initial_params
            logger.info("Using provided initial parameters")
        else:
            self._initialize_params()
            logger.info("Initialized fresh parameters")
        
        # Create optimizer
        self._create_optimizer()
        
        # Track update statistics
        self.n_updates = 0
        self.update_history = []
        
    def _create_model(self):
        """Create the surrogate model function."""
        def model_fn(x: jnp.ndarray, variable_order: List[str], target_variable: str, is_training: bool = False):
            model = ParentSetPredictionModel(
                layers=self.model_config["layers"],
                dim=self.model_config["dim"],
                max_parent_size=self.model_config["max_parent_size"]
            )
            return model(x, variable_order, target_variable, is_training)
        
        self.net = hk.transform(model_fn)
        
    def _initialize_params(self):
        """Initialize model parameters."""
        # Create dummy data for initialization
        dummy_data = jnp.zeros((10, self.n_vars, 3))
        self.key, init_key = random.split(self.key)
        self.params = self.net.init(init_key, dummy_data, self.variables, self.variables[0], False)
        
    def _create_optimizer(self):
        """Create optimizer with learning rate schedule."""
        # Use cosine decay for stable convergence
        schedule = optax.cosine_decay_schedule(
            init_value=self.learning_rate,
            decay_steps=1000,
            alpha=0.1  # Final LR = 0.1 * initial
        )
        
        self.optimizer = optax.adam(learning_rate=schedule)
        self.opt_state = self.optimizer.init(self.params)
        
    def predict(self, data: jnp.ndarray, target_variable: str) -> Any:
        """
        Make predictions for a target variable.
        
        Args:
            data: Tensor data in [T, n_vars, 3] format
            target_variable: Name of target variable
            
        Returns:
            ParentSetPosterior object
        """
        return predict_parent_posterior(
            net=self.net,
            params=self.params,
            x=data,
            variable_order=self.variables,
            target_variable=target_variable,
            metadata={
                'model_type': 'ActiveLearningSurrogate',
                'n_updates': self.n_updates,
                'scoring_method': self.scoring_method
            }
        )
    
    def update(self, samples: List[pyr.PMap], target_variable: str) -> Dict[str, float]:
        """
        Update model parameters based on new samples.
        
        Args:
            samples: List of new samples (observational and interventional)
            target_variable: Target variable for structure learning
            
        Returns:
            Dictionary with update metrics
        """
        if len(samples) < 5:
            logger.debug("Skipping update: insufficient samples")
            return {"loss": 0.0, "grad_norm": 0.0, "update_norm": 0.0, "skipped": True}
        
        # Convert samples to tensor format
        data = samples_to_avici_format(samples, self.variables, target_variable)
        
        # Define loss function
        def loss_fn(params):
            # Get model predictions
            self.key, model_key = random.split(self.key)
            output = self.net.apply(params, model_key, data, self.variables, target_variable, False)
            
            # Extract parent set predictions
            logits = output['parent_set_logits']
            parent_sets = output['parent_sets']
            
            # Compute scores for each parent set
            scores = self._compute_parent_set_scores(parent_sets, samples, target_variable)
            
            # Temperature-scaled softmax
            temperature = 2.0
            probs = jax.nn.softmax(logits / temperature)
            
            # Expected score under model distribution
            expected_score = jnp.sum(probs * scores)
            
            # Loss is negative expected score
            loss = -expected_score
            
            # Add L2 regularization
            l2_reg = 1e-4 * sum(jnp.sum(p**2) for p in jax.tree.leaves(params))
            
            return loss + l2_reg
        
        # Compute gradients
        loss_val, grads = jax.value_and_grad(loss_fn)(self.params)
        
        # Clip gradients
        grad_norm = optax.global_norm(grads)
        max_grad_norm = 1.0
        
        if grad_norm > max_grad_norm:
            grads = jax.tree.map(
                lambda g: g * max_grad_norm / (grad_norm + 1e-8),
                grads
            )
            clipped_grad_norm = max_grad_norm
        else:
            clipped_grad_norm = grad_norm
        
        # Update parameters
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, self.params)
        self.params = optax.apply_updates(self.params, updates)
        
        # Track update statistics
        update_norm = optax.global_norm(updates)
        self.n_updates += 1
        
        metrics = {
            "loss": float(loss_val),
            "grad_norm": float(clipped_grad_norm),
            "update_norm": float(update_norm),
            "n_samples": len(samples),
            "n_updates": self.n_updates
        }
        
        self.update_history.append(metrics)
        
        logger.debug(f"Update {self.n_updates}: loss={loss_val:.4f}, "
                    f"grad_norm={clipped_grad_norm:.4f}")
        
        return metrics
    
    def _compute_parent_set_scores(self, parent_sets: List, samples: List[pyr.PMap], 
                                   target_variable: str) -> jnp.ndarray:
        """
        Compute scores for each parent set based on data likelihood.
        
        Args:
            parent_sets: List of parent sets to score
            samples: Data samples
            target_variable: Target variable
            
        Returns:
            Array of scores for each parent set
        """
        # Extract target values
        target_values = jnp.array([get_values(s)[target_variable] for s in samples])
        n_samples = len(target_values)
        
        # Create value matrix for all variables
        all_values = jnp.zeros((n_samples, self.n_vars))
        for i, sample in enumerate(samples):
            sample_values = get_values(sample)
            for j, var_name in enumerate(self.variables):
                if var_name in sample_values:
                    all_values = all_values.at[i, j].set(sample_values[var_name])
        
        scores = []
        
        for parent_set in parent_sets:
            n_params = len(parent_set) + 1  # coefficients + intercept
            
            if len(parent_set) == 0:
                # No parents: simple mean/variance model
                mean_pred = jnp.mean(target_values)
                var_pred = jnp.maximum(jnp.var(target_values), 0.01)
                log_likelihood = -0.5 * jnp.sum(
                    jnp.log(2 * jnp.pi * var_pred) + 
                    (target_values - mean_pred)**2 / var_pred
                )
            else:
                # Linear regression with parents
                parent_indices = [self.variables.index(p) for p in parent_set 
                                if p in self.variables]
                
                if parent_indices and n_samples > len(parent_indices):
                    # Extract parent values
                    parent_values = all_values[:, jnp.array(parent_indices)]
                    
                    # Add intercept
                    X = jnp.column_stack([jnp.ones(n_samples), parent_values])
                    
                    # Solve least squares with regularization
                    XTX = X.T @ X + 1e-6 * jnp.eye(X.shape[1])
                    XTy = X.T @ target_values
                    beta = jnp.linalg.solve(XTX, XTy)
                    
                    # Compute residuals
                    predictions = X @ beta
                    residuals = target_values - predictions
                    residual_var = jnp.maximum(jnp.var(residuals), 0.01)
                    
                    # Log likelihood
                    log_likelihood = -0.5 * jnp.sum(
                        jnp.log(2 * jnp.pi * residual_var) + 
                        residuals**2 / residual_var
                    )
                else:
                    # Fallback to simple model
                    mean_pred = jnp.mean(target_values)
                    var_pred = jnp.maximum(jnp.var(target_values), 0.01)
                    log_likelihood = -0.5 * jnp.sum(
                        jnp.log(2 * jnp.pi * var_pred) + 
                        (target_values - mean_pred)**2 / var_pred
                    )
            
            # Apply scoring method
            if self.scoring_method == "bic":
                score = log_likelihood - 0.5 * n_params * jnp.log(n_samples)
            elif self.scoring_method == "aic":
                score = log_likelihood - n_params
            elif self.scoring_method == "mdl":
                score = log_likelihood - 0.5 * n_params * jnp.log(n_samples)
            else:  # likelihood
                score = log_likelihood
            
            scores.append(score)
        
        return jnp.array(scores)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the active learning process."""
        if not self.update_history:
            return {
                "n_updates": 0,
                "total_samples": 0,
                "avg_loss": 0.0,
                "avg_grad_norm": 0.0
            }
        
        total_samples = sum(h.get("n_samples", 0) for h in self.update_history)
        avg_loss = np.mean([h["loss"] for h in self.update_history])
        avg_grad_norm = np.mean([h["grad_norm"] for h in self.update_history])
        
        return {
            "n_updates": self.n_updates,
            "total_samples": total_samples,
            "avg_loss": float(avg_loss),
            "avg_grad_norm": float(avg_grad_norm),
            "scoring_method": self.scoring_method,
            "model_config": self.model_config
        }


def create_active_learning_surrogate(scm: pyr.PMap,
                                   initial_checkpoint: Optional[Path] = None,
                                   learning_rate: float = 1e-3,
                                   scoring_method: str = "bic",
                                   seed: int = 42) -> Tuple[Callable, Callable]:
    """
    Create an active learning surrogate for the given SCM.
    
    Args:
        scm: The structural causal model
        initial_checkpoint: Optional checkpoint to initialize from
        learning_rate: Learning rate for updates
        scoring_method: Scoring method for structure learning
        seed: Random seed
        
    Returns:
        Tuple of (predict_fn, update_fn)
    """
    # Get actual variable names from SCM
    variables = list(get_variables(scm))
    
    # Load initial parameters if checkpoint provided
    initial_params = None
    if initial_checkpoint and initial_checkpoint.exists():
        import pickle
        with open(initial_checkpoint, "rb") as f:
            checkpoint = pickle.load(f)
            initial_params = checkpoint.get("params", checkpoint.get("surrogate_params"))
            logger.info(f"Loaded initial parameters from {initial_checkpoint}")
    
    # Create active learning surrogate
    surrogate = ActiveLearningSurrogate(
        variables=variables,
        initial_params=initial_params,
        learning_rate=learning_rate,
        scoring_method=scoring_method,
        seed=seed
    )
    
    # Create wrapper functions
    def predict_fn(tensor: jnp.ndarray, target: str) -> Any:
        """Predict parent posterior for target variable."""
        return surrogate.predict(tensor, target)
    
    def update_fn(samples: List[pyr.PMap], target: str) -> Dict[str, float]:
        """Update surrogate with new samples."""
        return surrogate.update(samples, target)
    
    return predict_fn, update_fn