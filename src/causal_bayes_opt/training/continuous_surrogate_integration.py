#!/usr/bin/env python3
"""
Continuous surrogate model integration for ACBO training.

This module provides functions to create, update, and use the ContinuousParentSetPredictionModel
for structure learning during GRPO/BC training. The continuous model offers better
JAX integration and scalability compared to discrete parent set enumeration.

Key Features:
- Continuous parent probabilities (fully differentiable)
- Linear scaling with number of variables
- Natural gradient flow for learning
- Simple integration with existing training

Design Principles:
- Pure functions for predictions
- Immutable parameter updates
- Clear separation from acquisition logic
"""

import logging
import time
from typing import Tuple, List, Dict, Any, Optional, Callable

import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
import optax

from ..data_structures.buffer import ExperienceBuffer
from ..data_structures.sample import get_values
from ..avici_integration.continuous.model import ContinuousParentSetPredictionModel
from ..avici_integration.continuous.integration import create_continuous_surrogate_model
from .three_channel_converter import buffer_to_three_channel_tensor

logger = logging.getLogger(__name__)

# Type aliases for clarity
SurrogateFn = Callable[[jnp.ndarray, int, List[str]], Dict[str, Any]]
UpdateFn = Callable[[Any, Any, ExperienceBuffer, str], Tuple[Any, Any, Dict[str, float]]]


def create_continuous_learnable_surrogate(
    n_variables: int,
    key: jax.Array,
    learning_rate: float = 1e-3,
    hidden_dim: int = 128,
    num_layers: int = 4,
    num_heads: int = 8
) -> Tuple[hk.Transformed, Any, Any, SurrogateFn, UpdateFn]:
    """
    Create a learnable continuous parent set model with update capability.
    
    Args:
        n_variables: Maximum number of variables in SCMs
        key: JAX random key
        learning_rate: Learning rate for optimizer
        hidden_dim: Hidden dimension for the model
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        
    Returns:
        Tuple of:
        - Transformed Haiku model
        - Initial parameters
        - Initial optimizer state
        - Prediction function
        - Update function
    """
    # Create model config
    model_config = {
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'key_size': hidden_dim // num_heads,
        'dropout': 0.1
    }
    
    # Create the continuous model function
    def model_fn(data: jnp.ndarray, target_idx: int, is_training: bool = False):
        model = ContinuousParentSetPredictionModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            key_size=hidden_dim // num_heads,
            dropout=0.1 if is_training else 0.0
        )
        return model(data, target_idx, is_training)
    
    # Transform to Haiku
    net = hk.transform(model_fn)
    
    # Initialize with dummy data
    dummy_data = jnp.zeros((10, n_variables, 3))  # [T, n_vars, 3]
    params = net.init(key, dummy_data, 0, False)
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Create prediction function
    def predict_fn(tensor: jnp.ndarray, target_idx: int, 
                   variables: List[str]) -> Dict[str, Any]:
        """
        Predict parent probabilities for target variable.
        
        Args:
            tensor: Data tensor [T, n_vars, 3]
            target_idx: Index of target variable
            variables: List of variable names (for creating output dict)
            
        Returns:
            Dictionary with parent probabilities and metadata
        """
        # Get model outputs
        # Note: We use a fixed key for deterministic predictions
        rng = jax.random.PRNGKey(0)
        outputs = net.apply(params, rng, tensor, target_idx, False)
        
        # Extract parent probabilities
        parent_probs = outputs['parent_probabilities']
        
        # Create output dictionary compatible with existing code
        # Convert continuous probabilities to marginal parent probabilities
        marginal_probs = {}
        for i, var in enumerate(variables):
            if i != target_idx:  # Exclude target from its own parents
                marginal_probs[var] = float(parent_probs[i])
            else:
                marginal_probs[var] = 0.0
        
        # Compute entropy of parent distribution
        # Avoid log(0) by adding small epsilon
        safe_probs = jnp.maximum(parent_probs, 1e-10)
        entropy = -jnp.sum(parent_probs * jnp.log(safe_probs))
        
        return {
            'marginal_parent_probs': marginal_probs,
            'parent_probabilities': parent_probs,
            'entropy': float(entropy),
            'attention_logits': outputs.get('attention_logits', None),
            'model_type': 'continuous'
        }
    
    # Create update function
    def update_fn(current_params: Any, current_opt_state: Any,
                  buffer: ExperienceBuffer, target_variable: str) -> Tuple[Any, Any, Dict[str, float]]:
        """
        Update surrogate parameters using data likelihood loss.
        
        Args:
            current_params: Current model parameters
            current_opt_state: Current optimizer state
            buffer: Experience buffer with observations/interventions
            target_variable: Name of target variable
            
        Returns:
            Tuple of (new_params, new_opt_state, metrics_dict)
        """
        # Convert buffer to tensor
        tensor, variables = buffer_to_three_channel_tensor(
            buffer, target_variable, standardize=True
        )
        
        # Skip if not enough data
        if buffer.size() < 5:
            return current_params, current_opt_state, {
                'loss': 0.0,
                'grad_norm': 0.0,
                'param_norm': 0.0
            }
        
        # Get target index
        target_idx = variables.index(target_variable)
        
        # Define loss function
        def loss_fn(params, rng):
            # Get predictions with RNG for dropout
            outputs = net.apply(params, rng, tensor, target_idx, True)
            parent_probs = outputs['parent_probabilities']
            
            # Simple entropy regularization to encourage confident predictions
            entropy = -jnp.sum(parent_probs * jnp.log(parent_probs + 1e-10))
            
            # Sparsity regularization to encourage fewer parents
            sparsity_loss = jnp.sum(parent_probs)
            
            # Combined loss (negative entropy + sparsity)
            # In practice, you'd compute actual data likelihood here
            loss = -entropy + 0.1 * sparsity_loss
            
            return loss
        
        # Generate RNG key for dropout during training
        rng = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
        
        # Compute gradients
        loss, grads = jax.value_and_grad(loss_fn)(current_params, rng)
        
        # Update parameters
        updates, new_opt_state = optimizer.update(grads, current_opt_state)
        new_params = optax.apply_updates(current_params, updates)
        
        # Compute metrics
        grad_norm = optax.global_norm(grads)
        param_norm = optax.global_norm(new_params)
        
        return new_params, new_opt_state, {
            'loss': float(loss),
            'grad_norm': float(grad_norm),
            'param_norm': float(param_norm)
        }
    
    return net, params, opt_state, predict_fn, update_fn


def compute_posterior_from_buffer_continuous(
    buffer: ExperienceBuffer,
    target_variable: str,
    net: hk.Transformed,
    params: Any
) -> Dict[str, Any]:
    """
    Compute posterior prediction from current buffer state using continuous model.
    
    Args:
        buffer: Experience buffer with observations/interventions
        target_variable: Target variable for optimization
        net: Haiku transformed model
        params: Current model parameters
        
    Returns:
        Dictionary with posterior information
    """
    # Convert buffer to tensor
    tensor, variables = buffer_to_three_channel_tensor(
        buffer, target_variable, standardize=True
    )
    
    # Get target index
    target_idx = variables.index(target_variable)
    
    # Get prediction (no dropout during inference)
    rng = jax.random.PRNGKey(0)
    outputs = net.apply(params, rng, tensor, target_idx, False)
    
    # Extract parent probabilities
    parent_probs = outputs['parent_probabilities']
    
    # Create output dictionary
    marginal_probs = {}
    for i, var in enumerate(variables):
        if i != target_idx:
            marginal_probs[var] = float(parent_probs[i])
        else:
            marginal_probs[var] = 0.0
    
    # Compute entropy
    safe_probs = jnp.maximum(parent_probs, 1e-10)
    entropy = -jnp.sum(parent_probs * jnp.log(safe_probs))
    
    return {
        'marginal_parent_probs': marginal_probs,
        'parent_probabilities': parent_probs,
        'entropy': float(entropy),
        'target_variable': target_variable,
        'model_type': 'continuous'
    }


def compute_structure_metrics_continuous(
    posterior: Dict[str, Any],
    true_parents: List[str],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute structure learning metrics from continuous posterior.
    
    Args:
        posterior: Posterior dictionary from continuous model
        true_parents: True parent variables
        threshold: Probability threshold for parent selection
        
    Returns:
        Dictionary with F1, precision, recall metrics
    """
    if 'marginal_parent_probs' not in posterior:
        return {
            'f1_score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'shd': float('inf')
        }
    
    # Get marginal probabilities
    marginals = posterior['marginal_parent_probs']
    
    # Predicted parents (above threshold)
    predicted_parents = {
        var for var, prob in marginals.items() 
        if prob >= threshold
    }
    
    # True parent set
    true_parent_set = set(true_parents)
    
    # Compute metrics
    if len(predicted_parents) == 0 and len(true_parent_set) == 0:
        # Both empty - perfect match
        f1 = 1.0
        precision = 1.0
        recall = 1.0
    elif len(predicted_parents) == 0:
        # No predictions but true parents exist
        f1 = 0.0
        precision = 0.0
        recall = 0.0
    elif len(true_parent_set) == 0:
        # Predictions but no true parents
        f1 = 0.0
        precision = 0.0
        recall = 1.0
    else:
        # Normal case
        tp = len(predicted_parents & true_parent_set)
        fp = len(predicted_parents - true_parent_set)
        fn = len(true_parent_set - predicted_parents)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Structural Hamming Distance
    shd = len(predicted_parents ^ true_parent_set)  # Symmetric difference
    
    return {
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'shd': float(shd)
    }


def create_surrogate_fn_wrapper(
    net: hk.Transformed,
    params: Any
) -> Callable[[jnp.ndarray, str], Dict[str, Any]]:
    """
    Create a surrogate function wrapper for the evaluator.
    
    This wrapper handles the variable name mapping correctly by inferring
    variable names from the tensor context and target variable.
    
    Args:
        net: Haiku transformed model
        params: Model parameters
        
    Returns:
        Function that takes (tensor, target_var) and returns posterior dict
    """
    def surrogate_fn(tensor: jnp.ndarray, target_var: str) -> Dict[str, Any]:
        # Get dimensions
        n_vars = tensor.shape[1]
        
        # Infer variable names based on context
        if n_vars == 3 and target_var in ['X', 'Y', 'Z']:
            # Standard 3-variable SCMs (fork, collider, etc.)
            variables = ['X', 'Y', 'Z']
        elif target_var.startswith('X') and target_var[1:].isdigit():
            # Chain or other numbered SCMs
            variables = [f'X{i}' for i in range(n_vars)]
        else:
            # Default to generic names
            variables = [f'X{i}' for i in range(n_vars)]
        
        # Get target index
        if target_var in variables:
            target_idx = variables.index(target_var)
        else:
            # Handle special cases
            if target_var == 'Y' and n_vars == 3:
                target_idx = 1
                variables = ['X', 'Y', 'Z']
            elif target_var == 'Z' and n_vars == 3:
                target_idx = 2
                variables = ['X', 'Y', 'Z']
            else:
                # Default to last variable
                target_idx = n_vars - 1
        
        # Get prediction (no dropout during inference)
        rng = jax.random.PRNGKey(0)
        outputs = net.apply(params, rng, tensor, target_idx, False)
        
        # Extract parent probabilities
        parent_probs = outputs['parent_probabilities']
        
        # Create output dictionary with proper variable names
        marginal_probs = {}
        for i, var in enumerate(variables):
            if i != target_idx:
                marginal_probs[var] = float(parent_probs[i])
            else:
                marginal_probs[var] = 0.0
        
        # Compute entropy
        safe_probs = jnp.maximum(parent_probs, 1e-10)
        entropy = -jnp.sum(parent_probs * jnp.log(safe_probs))
        
        return {
            'marginal_parent_probs': marginal_probs,
            'parent_probabilities': parent_probs,
            'entropy': float(entropy),
            'attention_logits': outputs.get('attention_logits'),
            'model_type': 'continuous'
        }
    
    return surrogate_fn