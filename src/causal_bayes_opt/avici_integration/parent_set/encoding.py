"""
Parent set encoding utilities.
"""

import jax.numpy as jnp


def encode_parent_set(parent_set_indicators: jnp.ndarray, 
                     variable_embeddings: jnp.ndarray) -> jnp.ndarray:
    """
    Encode a parent set using variable embeddings.
    
    FIXED: Empty parent sets get normalized mean embedding - no special bias.
    
    Args:
        parent_set_indicators: [n_vars] binary indicators (1 if in parent set)
        variable_embeddings: [n_vars, dim] variable embeddings
        
    Returns:
        [dim] parent set embedding
    """
    parent_set_size = jnp.sum(parent_set_indicators)
    
    # Check if this is an empty parent set
    is_empty = parent_set_size == 0
    
    if is_empty:
        # FIXED: Use normalized mean - gives empty sets a fair chance
        # This removes the bias of always getting dot product = 0
        empty_embedding = jnp.mean(variable_embeddings, axis=0)
        return empty_embedding
    else:
        # Non-empty sets: use weighted mean as before
        weighted_embeddings = variable_embeddings * parent_set_indicators[..., None]
        parent_set_embedding = jnp.sum(weighted_embeddings, axis=-2)
        return parent_set_embedding / parent_set_size


def create_parent_set_indicators(parent_set: frozenset, 
                                variable_order: list, 
                                n_vars: int) -> jnp.ndarray:
    """
    Create binary indicators for a parent set.
    
    Args:
        parent_set: Set of parent variable names
        variable_order: Ordered list of all variable names
        n_vars: Total number of variables
        
    Returns:
        [n_vars] binary indicator array
    """
    indicators = jnp.zeros(n_vars)
    for var in parent_set:
        if var in variable_order:
            var_idx = variable_order.index(var)
            indicators = indicators.at[var_idx].set(1.0)
    return indicators
