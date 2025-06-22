"""
Target-aware conditioning utilities.

Implements the target conditioning from the modular model to enhance
the proven transformer architecture from the original model.
"""

from typing import List
import jax.numpy as jnp
import haiku as hk


def add_target_conditioning(
    embeddings: jnp.ndarray,  # [N, d, hidden_dim]
    target_idx: int,
    variable_order: List[str],
    config
) -> jnp.ndarray:
    """
    Add target-aware conditioning to variable embeddings.
    
    This enhances the proven transformer architecture with target awareness
    from the modular model, improving causal structure learning.
    
    Args:
        embeddings: Variable embeddings [N, d, hidden_dim] 
        target_idx: Index of target variable
        variable_order: List of variable names
        config: Configuration with target conditioning parameters
        
    Returns:
        Enhanced embeddings with target conditioning [N, d, hidden_dim + target_dim]
    """
    if not config.enable_target_conditioning:
        return embeddings
        
    N, d, hidden_dim = embeddings.shape
    
    # Create target indicator mask
    target_mask = jnp.zeros((d,))
    target_mask = target_mask.at[target_idx].set(1.0)
    
    # Learn target-specific embedding
    target_embedding = hk.Linear(
        config.target_embedding_dim, 
        name="target_conditioning_embedding"
    )(target_mask)
    
    # Broadcast target embedding to match batch and variable dimensions
    target_broadcast = jnp.broadcast_to(
        target_embedding[None, None, :],  # [1, 1, target_dim]
        (N, d, config.target_embedding_dim)  # [N, d, target_dim]
    )
    
    # Concatenate with original embeddings
    enhanced_embeddings = jnp.concatenate([embeddings, target_broadcast], axis=-1)
    
    return enhanced_embeddings


def create_positional_encoding(
    variable_order: List[str],
    target_variable: str,
    hidden_dim: int
) -> jnp.ndarray:
    """
    Create positional encoding that incorporates target variable position.
    
    This provides an alternative/additional way to incorporate target awareness
    through positional information.
    
    Args:
        variable_order: List of variable names
        target_variable: Name of target variable  
        hidden_dim: Dimension for positional encoding
        
    Returns:
        Positional encoding [d, hidden_dim]
    """
    d = len(variable_order)
    target_idx = variable_order.index(target_variable)
    
    # Standard sinusoidal positional encoding
    position = jnp.arange(d)[:, None]  # [d, 1]
    div_term = jnp.exp(jnp.arange(0, hidden_dim, 2) * -(jnp.log(10000.0) / hidden_dim))
    
    pos_encoding = jnp.zeros((d, hidden_dim))
    pos_encoding = pos_encoding.at[:, 0::2].set(jnp.sin(position * div_term))
    pos_encoding = pos_encoding.at[:, 1::2].set(jnp.cos(position * div_term))
    
    # Enhance target position with special encoding
    if hidden_dim > 4:  # Ensure we have space for target enhancement
        target_enhancement = jnp.zeros((d, 4))
        target_enhancement = target_enhancement.at[target_idx, :].set(1.0)  # Target gets [1,1,1,1]
        
        # Concatenate or add to existing encoding
        pos_encoding = pos_encoding.at[:, -4:].set(target_enhancement)
    
    return pos_encoding


def compute_target_attention_mask(
    variable_order: List[str],
    target_variable: str,
    attention_type: str = "causal"
) -> jnp.ndarray:
    """
    Compute attention mask that respects causal constraints and target focus.
    
    Args:
        variable_order: List of variable names
        target_variable: Name of target variable
        attention_type: Type of masking ("causal", "target_focused", "bidirectional")
        
    Returns:
        Attention mask [d, d] where 1=attend, 0=mask
    """
    d = len(variable_order)
    target_idx = variable_order.index(target_variable)
    
    if attention_type == "causal":
        # Lower triangular mask (variable can only attend to previous variables)
        mask = jnp.tril(jnp.ones((d, d)))
        
    elif attention_type == "target_focused":
        # All variables can attend to target, target can attend to all
        mask = jnp.zeros((d, d))
        mask = mask.at[:, target_idx].set(1.0)  # All attend to target
        mask = mask.at[target_idx, :].set(1.0)  # Target attends to all
        mask = mask.at[jnp.diag_indices(d)].set(1.0)  # Self-attention
        
    elif attention_type == "bidirectional":
        # Full bidirectional attention
        mask = jnp.ones((d, d))
        
    else:
        raise ValueError(f"Unknown attention_type: {attention_type}")
    
    return mask


def apply_target_aware_dropout(
    x: jnp.ndarray,  # [N, d, hidden_dim]
    target_idx: int,
    dropout_rate: float,
    key: jnp.ndarray,
    protect_target: bool = True
) -> jnp.ndarray:
    """
    Apply dropout while optionally protecting target variable features.
    
    Args:
        x: Input tensor [N, d, hidden_dim]
        target_idx: Index of target variable
        dropout_rate: Standard dropout rate
        key: Random key for dropout
        protect_target: If True, apply reduced dropout to target variable
        
    Returns:
        Tensor with target-aware dropout applied
    """
    if not protect_target:
        return hk.dropout(key, dropout_rate, x)
    
    N, d, hidden_dim = x.shape
    
    # Create variable-specific dropout rates
    dropout_rates = jnp.full((d,), dropout_rate)
    
    # Reduce dropout for target variable (retain more information)
    target_dropout_rate = dropout_rate * 0.5  # Half the normal rate
    dropout_rates = dropout_rates.at[target_idx].set(target_dropout_rate)
    
    # Apply variable-specific dropout
    dropped_x = x
    for var_idx in range(d):
        var_key, key = jax.random.split(key)
        var_dropout = hk.dropout(var_key, dropout_rates[var_idx], x[:, var_idx:var_idx+1, :])
        dropped_x = dropped_x.at[:, var_idx:var_idx+1, :].set(var_dropout)
    
    return dropped_x