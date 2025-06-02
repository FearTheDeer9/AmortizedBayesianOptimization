"""
Inference utilities for parent set prediction.
"""

import jax
import jax.numpy as jnp
import jax.random as random
import optax
from typing import Dict, List, FrozenSet, Optional

from .posterior import ParentSetPosterior, create_parent_set_posterior


def predict_parent_posterior(
    net, params, x, variable_order, target_variable, metadata=None
) -> ParentSetPosterior:
    """
    Predict parent set posterior distribution.
    
    This is the main function for getting parent set predictions from the model.
    Returns a structured ParentSetPosterior with rich analysis capabilities.
    
    Args:
        net: Transformed Haiku model
        params: Model parameters
        x: Input data [N, d, 3]
        variable_order: Variable names in order
        target_variable: Target variable name
        metadata: Optional metadata to include in posterior
        
    Returns:
        ParentSetPosterior object with probabilities and utilities
        
    Example:
        >>> posterior = predict_parent_posterior(net, params, data, vars, 'Y')
        >>> most_likely = posterior.top_k_sets[0][0]  # Most likely parent set
        >>> summary = summarize_posterior(posterior)   # Rich analysis
        >>> marginals = get_marginal_parent_probabilities(posterior, vars)
    """
    # Get model output
    output = net.apply(
        params, random.PRNGKey(0), x, variable_order, target_variable, False
    )
    
    # Convert logits to probabilities
    logits = output['parent_set_logits']
    probabilities = jax.nn.softmax(logits)
    parent_sets = output['parent_sets'][:output['k']]
    
    # Add model metadata
    if metadata is None:
        metadata = {}
    
    metadata.update({
        'model_k': output['k'],
        'all_variables': variable_order,
        'prediction_method': 'ParentSetPredictionModel'
    })
    
    # Create formal posterior
    return create_parent_set_posterior(
        target_variable=target_variable,
        parent_sets=parent_sets,
        probabilities=probabilities[:output['k']],
        metadata=metadata
    )


def compute_loss(net, params, x, variable_order, target_variable, true_parent_set, is_training=True):
    """
    Compute loss for training with improved handling of missing true parent sets.
    
    Args:
        net: Transformed Haiku model
        params: Model parameters
        x: Input data
        variable_order: Variable names in order
        target_variable: Target variable name
        true_parent_set: Ground truth parent set (frozenset)
        is_training: Training mode
        
    Returns:
        Loss value
    """
    output = net.apply(
        params, random.PRNGKey(0), x, variable_order, target_variable, is_training
    )
    
    logits = output['parent_set_logits']
    parent_sets = output['parent_sets']
    
    # Find the true parent set in the predictions
    true_idx = None
    for i, ps in enumerate(parent_sets):
        if ps == true_parent_set:
            true_idx = i
            break
    
    if true_idx is None:
        # IMPROVED: When true parent set not in top-k, use margin loss
        # This encourages the model to include correct parent sets in future predictions
        
        # Find the closest parent set (by set distance) as a proxy
        min_distance = float('inf')
        closest_idx = 0
        for i, ps in enumerate(parent_sets):
            # Use Jaccard distance: 1 - |intersection| / |union|
            intersection = len(true_parent_set.intersection(ps))
            union = len(true_parent_set.union(ps))
            distance = 1 - (intersection / union if union > 0 else 1)
            
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
        
        # Use cross-entropy loss with closest match + penalty for missing true set
        log_probs = jax.nn.log_softmax(logits)
        proxy_loss = -log_probs[closest_idx]
        
        # Add penalty term proportional to distance from true parent set
        penalty = min_distance * jnp.log(len(parent_sets))
        
        return proxy_loss + penalty
    else:
        # Standard cross-entropy loss when true parent set is in predictions
        log_probs = jax.nn.log_softmax(logits)
        cross_entropy_loss = -log_probs[true_idx]
        return cross_entropy_loss


def compute_loss_with_regularization(net, params, x, variable_order, target_variable, 
                                   true_parent_set, is_training=True, reg_weight=1e-4):
    """
    Compute loss with L2 regularization on embeddings.
    """
    # Standard loss
    loss = compute_loss(net, params, x, variable_order, target_variable, true_parent_set, is_training)
    
    # Add L2 regularization on parameters
    l2_reg = 0.0
    for param in jax.tree_util.tree_leaves(params):
        l2_reg += jnp.sum(param ** 2)
    
    return loss + reg_weight * l2_reg


def create_train_step(net, optimizer):
    """
    Create a training step function.
    
    Args:
        net: Transformed Haiku model
        optimizer: Optax optimizer
        
    Returns:
        Training step function
    """
    def train_step(params, opt_state, x, variable_order, target_variable, true_parent_set):
        """Single training step."""
        def loss_fn(params):
            return compute_loss(net, params, x, variable_order, target_variable, true_parent_set, is_training=True)
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss
    
    return train_step
