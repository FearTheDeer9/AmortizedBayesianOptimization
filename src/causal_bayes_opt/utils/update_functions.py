"""
Update function interface for active learning in ACBO.

This module provides a standardized interface for model update functions
that can be used during evaluation to adapt model parameters based on
new observations.

Key principles:
- Compatible with JAX transformations
- Works with any optimizer
- Returns standardized metrics
- Supports different update strategies
"""

import logging
from typing import Protocol, Tuple, Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import optax

from ..data_structures.buffer import ExperienceBuffer
from ..data_structures.sample import get_values

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UpdateContext:
    """
    Context for model updates during evaluation.
    
    This immutable dataclass contains all information needed
    for an update function to make decisions.
    """
    buffer: ExperienceBuffer
    target_variable: str
    variables: List[str]
    step: int
    metadata: Dict[str, Any]
    
    def get_recent_samples(self, n: int = 10) -> List[Any]:
        """Get the n most recent samples from buffer."""
        all_samples = self.buffer.get_all_samples()
        return all_samples[-n:] if len(all_samples) > n else all_samples
    
    def get_target_values(self) -> jnp.ndarray:
        """Extract target variable values from all samples."""
        samples = self.buffer.get_all_samples()
        values = []
        for sample in samples:
            sample_values = get_values(sample)
            if self.target_variable in sample_values:
                values.append(float(sample_values[self.target_variable]))
        return jnp.array(values)


class UpdateFunction(Protocol):
    """
    Protocol for model update functions.
    
    All update functions must follow this interface to ensure
    compatibility with the evaluation system.
    """
    
    def __call__(
        self,
        params: Any,
        opt_state: Any,
        context: UpdateContext
    ) -> Tuple[Any, Any, Dict[str, float]]:
        """
        Update model parameters based on context.
        
        Args:
            params: Current model parameters
            opt_state: Current optimizer state
            context: Update context with buffer, target, etc.
            
        Returns:
            Tuple of:
            - new_params: Updated parameters
            - new_opt_state: Updated optimizer state
            - metrics: Dictionary of update metrics
        """
        ...


class NoOpUpdate:
    """No-operation update function for fixed models."""
    
    def __call__(
        self,
        params: Any,
        opt_state: Any,
        context: UpdateContext
    ) -> Tuple[Any, Any, Dict[str, float]]:
        """Return unchanged parameters."""
        return params, opt_state, {
            "loss": 0.0,
            "grad_norm": 0.0,
            "update_norm": 0.0,
            "skipped": True,
            "reason": "no_op"
        }


class GradientUpdate:
    """
    Gradient-based update using data likelihood.
    
    This is the standard update approach that computes gradients
    based on how well the model explains the observed data.
    """
    
    def __init__(
        self,
        loss_fn: Callable,
        optimizer: optax.GradientTransformation,
        min_samples: int = 5,
        max_grad_norm: float = 1.0
    ):
        """
        Initialize gradient update function.
        
        Args:
            loss_fn: Loss function (params, context) -> scalar
            optimizer: Optax optimizer
            min_samples: Minimum samples needed for update
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.min_samples = min_samples
        self.max_grad_norm = max_grad_norm
    
    def __call__(
        self,
        params: Any,
        opt_state: Any,
        context: UpdateContext
    ) -> Tuple[Any, Any, Dict[str, float]]:
        """Perform gradient update."""
        # Check if we have enough samples
        n_samples = context.buffer.size()
        if n_samples < self.min_samples:
            return params, opt_state, {
                "loss": 0.0,
                "grad_norm": 0.0,
                "update_norm": 0.0,
                "skipped": True,
                "reason": f"insufficient_samples ({n_samples} < {self.min_samples})"
            }
        
        # Compute loss and gradients
        loss_val, grads = jax.value_and_grad(self.loss_fn)(params, context)
        
        # Clip gradients
        grad_norm = optax.global_norm(grads)
        if grad_norm > self.max_grad_norm:
            grads = jax.tree.map(
                lambda g: g * self.max_grad_norm / (grad_norm + 1e-8),
                grads
            )
            clipped_grad_norm = self.max_grad_norm
        else:
            clipped_grad_norm = grad_norm
        
        # Update parameters
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        # Compute update norm
        update_norm = optax.global_norm(updates)
        
        return new_params, new_opt_state, {
            "loss": float(loss_val),
            "grad_norm": float(clipped_grad_norm),
            "update_norm": float(update_norm),
            "n_samples": n_samples,
            "skipped": False
        }


class AdaptiveUpdate:
    """
    Adaptive update that adjusts learning rate based on performance.
    
    This update function monitors the loss trajectory and adjusts
    the learning rate to ensure stable convergence.
    """
    
    def __init__(
        self,
        base_update: UpdateFunction,
        patience: int = 5,
        lr_decay: float = 0.5,
        lr_increase: float = 1.1
    ):
        """
        Initialize adaptive update.
        
        Args:
            base_update: Base update function to wrap
            patience: Steps to wait before adjusting LR
            lr_decay: Factor to decrease LR on plateau
            lr_increase: Factor to increase LR on improvement
        """
        self.base_update = base_update
        self.patience = patience
        self.lr_decay = lr_decay
        self.lr_increase = lr_increase
        self.loss_history = []
        self.best_loss = float('inf')
        self.patience_counter = 0
    
    def __call__(
        self,
        params: Any,
        opt_state: Any,
        context: UpdateContext
    ) -> Tuple[Any, Any, Dict[str, float]]:
        """Perform adaptive update with learning rate adjustment."""
        # Perform base update
        new_params, new_opt_state, metrics = self.base_update(params, opt_state, context)
        
        if not metrics.get("skipped", True):
            loss = metrics["loss"]
            self.loss_history.append(loss)
            
            # Check if we should adjust learning rate
            if loss < self.best_loss:
                self.best_loss = loss
                self.patience_counter = 0
                # Could increase LR here if desired
            else:
                self.patience_counter += 1
                
            # Add adaptive metrics
            metrics["best_loss"] = self.best_loss
            metrics["patience_counter"] = self.patience_counter
            
        return new_params, new_opt_state, metrics


def create_update_function(
    strategy: str = "no_op",
    net: Optional[Any] = None,
    learning_rate: float = 1e-3,
    min_samples: int = 5,
    **kwargs
) -> UpdateFunction:
    """
    Factory for creating update functions.
    
    Args:
        strategy: Update strategy ("no_op" or "bic")
        net: Haiku network (required for "bic" strategy)
        learning_rate: Learning rate for optimizer
        min_samples: Minimum samples for updates
        **kwargs: Additional strategy-specific arguments
        
    Returns:
        Update function instance
    """
    if strategy == "no_op":
        return NoOpUpdate()
    
    elif strategy == "bic":
        if net is None:
            raise ValueError("BIC update requires net parameter")
        
        optimizer = optax.adam(learning_rate)
        return BICUpdate(
            net=net,
            optimizer=optimizer,
            min_samples=min_samples,
            max_grad_norm=kwargs.get("max_grad_norm", 1.0),
            temperature=kwargs.get("temperature", 3.0)
        )
    
    else:
        raise ValueError(f"Unknown update strategy: {strategy}. Supported: 'no_op', 'bic'")


class BICUpdate:
    """
    BIC-based update for surrogate models.
    
    This update function uses Bayesian Information Criterion (BIC) scoring
    to update surrogate model parameters based on observed data.
    
    WARNING: This update strategy is incompatible with BC surrogate models that
    only output marginal parent probabilities. The BIC update requires models
    that output parent set predictions (parent_set_logits and parent_sets).
    When used with BC surrogates, the loss will be 0.0 and updates will only
    apply L2 regularization, causing performance degradation over time.
    
    For BC surrogates that output marginal probabilities, consider alternative
    update strategies or use the model without active learning.
    """
    
    def __init__(
        self,
        net: Any,
        optimizer: optax.GradientTransformation,
        min_samples: int = 5,
        max_grad_norm: float = 1.0,
        temperature: float = 3.0
    ):
        """
        Initialize BIC update function.
        
        Args:
            net: Haiku transformed network
            optimizer: Optax optimizer (e.g., adam with lr=1e-3)
            min_samples: Minimum samples needed for update
            max_grad_norm: Maximum gradient norm for clipping
            temperature: Temperature for softmax scaling (higher = smoother gradients)
        """
        self.net = net
        self.optimizer = optimizer
        self.min_samples = min_samples
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature
        
    def __call__(
        self,
        params: Any,
        opt_state: Any,
        context: UpdateContext
    ) -> Tuple[Any, Any, Dict[str, float]]:
        """Perform BIC-based gradient update."""
        # Check if we have enough samples
        n_samples = context.buffer.size()
        if n_samples < self.min_samples:
            return params, opt_state, {
                "loss": 0.0,
                "grad_norm": 0.0,
                "update_norm": 0.0,
                "skipped": True,
                "reason": f"insufficient_samples ({n_samples} < {self.min_samples})"
            }
        
        # Define loss function inline to have access to all context
        def loss_fn(params: Any) -> jnp.ndarray:
            # Convert buffer to tensor
            from ..training.three_channel_converter import buffer_to_three_channel_tensor
            tensor, mapper = buffer_to_three_channel_tensor(
                context.buffer,
                context.target_variable,
                standardize=True
            )
            
            # Get target index
            target_idx = mapper.get_index(context.target_variable)
            variable_order = mapper.variables
            
            # Forward pass
            rng = jax.random.PRNGKey(context.step)
            outputs = self.net.apply(params, rng, tensor, target_idx, False)  # No dropout during inference
            
            # Extract predictions
            logits = outputs.get('parent_set_logits', None)
            parent_sets = outputs.get('parent_sets', None)
            
            if logits is None or parent_sets is None:
                # Model doesn't support parent set predictions
                return jnp.array(0.0)
            
            # Convert logits to probabilities with temperature scaling
            scaled_logits = logits / self.temperature
            probs = jax.nn.softmax(scaled_logits)
            
            # Get samples for BIC computation
            all_samples = context.buffer.get_all_samples()
            recent_samples = all_samples[-100:] if len(all_samples) > 100 else all_samples
            
            # Compute BIC scores for each parent set
            scores_per_set = self._compute_bic_scores(
                parent_sets, recent_samples, context.target_variable, variable_order
            )
            
            # Normalize scores for stable gradients
            expected_ll_per_sample = -1.4  # Expected log-likelihood under simple Gaussian
            scores_normalized = scores_per_set / len(recent_samples)
            scores_centered = scores_normalized - expected_ll_per_sample
            
            # Use log-sum-exp for numerical stability
            log_partition = jax.nn.logsumexp(scores_centered, b=probs)
            
            # Loss is negative log expected score
            loss = -log_partition
            
            # Add moderate L2 regularization
            l2_reg = 1e-4 * sum(
                jnp.sum(p**2) for p in jax.tree.leaves(params)
            )
            
            return loss + l2_reg
        
        # Compute loss and gradients
        logger.debug(f"[BICUpdate] Computing loss and gradients...")
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        logger.debug(f"[BICUpdate] Loss value: {float(loss_val):.4f}")
        
        # Clip gradients
        grad_norm = optax.global_norm(grads)
        logger.debug(f"[BICUpdate] Gradient norm before clipping: {float(grad_norm):.4f}")
        
        if grad_norm > self.max_grad_norm:
            grads = jax.tree.map(
                lambda g: g * self.max_grad_norm / (grad_norm + 1e-8),
                grads
            )
            clipped_grad_norm = self.max_grad_norm
            logger.debug(f"[BICUpdate] Gradients clipped to norm: {clipped_grad_norm}")
        else:
            clipped_grad_norm = grad_norm
        
        # Update parameters
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        # Compute update norm
        update_norm = optax.global_norm(updates)
        logger.debug(f"[BICUpdate] Update norm: {float(update_norm):.4f}")
        logger.debug(f"[BICUpdate] Update completed successfully")
        
        return new_params, new_opt_state, {
            "loss": float(loss_val),
            "grad_norm": float(clipped_grad_norm),
            "update_norm": float(update_norm),
            "n_samples": n_samples,
            "skipped": False
        }
    
    def _compute_bic_scores(
        self,
        parent_sets: List,
        samples: List,
        target_variable: str,
        variable_order: List[str]
    ) -> jnp.ndarray:
        """
        Compute BIC scores for each parent set.
        
        This is a simplified version of compute_likelihood_per_parent_set_jax
        from demo_learning.py, adapted for the update context.
        """
        if not samples:
            return jnp.array([0.0] * len(parent_sets))
        
        # Extract target values
        target_values = jnp.array([get_values(s)[target_variable] for s in samples])
        n_samples = len(target_values)
        
        # Create matrix of all variable values
        all_values = jnp.zeros((n_samples, len(variable_order)))
        for i, sample in enumerate(samples):
            sample_values = get_values(sample)
            for j, var_name in enumerate(variable_order):
                if var_name in sample_values:
                    all_values = all_values.at[i, j].set(sample_values[var_name])
        
        scores = []
        
        for parent_set in parent_sets:
            n_params = len(parent_set) + 1  # coefficients + intercept
            
            if len(parent_set) == 0:
                # No parents: simple mean/variance
                mean_pred = jnp.mean(target_values)
                var_pred = jnp.maximum(jnp.var(target_values), 0.01)
                log_likelihood = -0.5 * jnp.sum(
                    jnp.log(2 * jnp.pi * var_pred) + 
                    (target_values - mean_pred)**2 / var_pred
                )
            else:
                # Has parents: linear regression
                parent_indices = jnp.array([
                    variable_order.index(p) for p in parent_set 
                    if p in variable_order
                ])
                
                if len(parent_indices) > 0 and n_samples > len(parent_indices):
                    parent_values = all_values[:, parent_indices]
                    X = jnp.column_stack([jnp.ones(n_samples), parent_values])
                    
                    # Least squares with regularization
                    XTX = X.T @ X
                    XTy = X.T @ target_values
                    XTX_reg = XTX + 1e-6 * jnp.eye(XTX.shape[0])
                    beta = jnp.linalg.solve(XTX_reg, XTy)
                    
                    predictions = X @ beta
                    residuals = target_values - predictions
                    residual_var = jnp.maximum(jnp.var(residuals), 0.01)
                    
                    log_likelihood = -0.5 * jnp.sum(
                        jnp.log(2 * jnp.pi * residual_var) + 
                        residuals**2 / residual_var
                    )
                else:
                    # Fallback
                    mean_pred = jnp.mean(target_values)
                    var_pred = jnp.maximum(jnp.var(target_values), 0.01)
                    log_likelihood = -0.5 * jnp.sum(
                        jnp.log(2 * jnp.pi * var_pred) + 
                        (target_values - mean_pred)**2 / var_pred
                    )
            
            # BIC score: log_likelihood - 0.5 * n_params * log(n_samples)
            bic_score = log_likelihood - 0.5 * n_params * jnp.log(n_samples)
            scores.append(bic_score)
        
        return jnp.array(scores)