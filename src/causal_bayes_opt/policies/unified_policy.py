"""
Unified policy model for ACBO.

Since BC and GRPO policies are functionally identical (both map states to 
intervention distributions), this module provides a single unified implementation
that can be used for both training methods.

The only difference is in the training objective:
- BC: Supervised learning from demonstrations
- GRPO: Policy gradient with group advantages

The architecture and forward pass are identical.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def create_unified_policy(
    hidden_dim: int = 256,
    num_layers: int = 3,
    use_layer_norm: bool = True,
    dropout_rate: float = 0.1
) -> callable:
    """
    Create unified policy function for ACBO.
    
    This policy accepts 5-channel input tensors and outputs intervention
    distributions. It can be used for both BC and GRPO training.
    
    Args:
        hidden_dim: Hidden dimension for the network
        num_layers: Number of hidden layers
        use_layer_norm: Whether to use layer normalization
        dropout_rate: Dropout rate (only applied during training)
        
    Returns:
        Policy function ready for hk.transform
    """
    def policy_fn(
        tensor_input: jnp.ndarray, 
        target_idx: int = 0,
        is_training: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """
        Unified policy network for intervention selection.
        
        Args:
            tensor_input: [T, n_vars, C] tensor where C is 5 (or 3 for legacy)
            target_idx: Index of target variable to mask from selection
            is_training: Whether in training mode (affects dropout)
            
        Returns:
            Dictionary with:
            - variable_logits: [n_vars] logits for variable selection
            - value_params: [n_vars, 2] mean and log_std for each variable
            - attention_weights: [T, n_vars] attention over history (optional)
        """
        T, n_vars, n_channels = tensor_input.shape
        
        # Handle legacy 3-channel input
        if n_channels == 3:
            padded = jnp.zeros((T, n_vars, 5))
            padded = padded.at[:, :, :3].set(tensor_input)
            tensor_input = padded
            n_channels = 5
        elif n_channels != 5:
            raise ValueError(f"Expected 3 or 5 channels, got {n_channels}")
        
        # Extract channels for clarity
        values = tensor_input[:, :, 0]  # Variable values
        target_indicators = tensor_input[:, :, 1]  # Target indicators
        intervention_indicators = tensor_input[:, :, 2]  # Intervention history
        parent_probs = tensor_input[:, :, 3]  # Surrogate predictions
        recency = tensor_input[:, :, 4]  # Intervention recency
        
        # Input projection with channel-specific processing
        # Process each channel type appropriately
        channel_features = []
        
        # Values channel - normalize and project
        values_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(values)
        values_feat = hk.Linear(hidden_dim // 4, name="value_projection")(values_norm)
        channel_features.append(values_feat)
        
        # Binary channels - simple projection
        for i, (channel, name) in enumerate([
            (target_indicators, "target"),
            (intervention_indicators, "intervention")
        ]):
            feat = hk.Linear(hidden_dim // 8, name=f"{name}_projection")(channel[..., None])
            channel_features.append(feat.squeeze(-1))
        
        # Parent probability channel - emphasize structure information
        parent_feat = hk.Linear(hidden_dim // 4, name="parent_projection")(parent_probs[..., None])
        channel_features.append(parent_feat.squeeze(-1))
        
        # Recency channel - temporal awareness
        recency_feat = hk.Linear(hidden_dim // 8, name="recency_projection")(recency[..., None])
        channel_features.append(recency_feat.squeeze(-1))
        
        # Combine channel features
        combined = jnp.concatenate([f[..., None] if f.ndim == 2 else f for f in channel_features], axis=-1)
        x = hk.Linear(hidden_dim, name="channel_fusion")(combined)
        x = jax.nn.relu(x)
        
        # Apply dropout if training
        if is_training and dropout_rate > 0:
            x = hk.dropout(hk.next_rng_key(), dropout_rate, x)
        
        # Process through hidden layers
        for i in range(num_layers):
            # Save for residual
            residual = x
            
            # Layer norm
            if use_layer_norm:
                x = hk.LayerNorm(
                    axis=-1,
                    create_scale=True,
                    create_offset=True,
                    name=f"layer_{i}_norm"
                )(x)
            
            # MLP block
            x = hk.Linear(hidden_dim * 2, name=f"layer_{i}_up")(x)
            x = jax.nn.gelu(x)  # GELU activation
            
            if is_training and dropout_rate > 0:
                x = hk.dropout(hk.next_rng_key(), dropout_rate, x)
            
            x = hk.Linear(hidden_dim, name=f"layer_{i}_down")(x)
            
            # Residual connection
            x = x + residual
        
        # Temporal attention mechanism
        # Compute attention weights over history
        attention_query = hk.Linear(hidden_dim // 4, name="attention_query")(x)
        attention_key = hk.Linear(hidden_dim // 4, name="attention_key")(x)
        
        # Compute attention scores [T, n_vars]
        attention_scores = jnp.sum(attention_query * attention_key, axis=-1) / jnp.sqrt(hidden_dim // 4)
        attention_weights = jax.nn.softmax(attention_scores, axis=0)
        
        # Apply attention to aggregate temporal information
        x_attended = jnp.sum(x * attention_weights[..., None], axis=0)  # [n_vars, hidden_dim]
        
        # Final layer norm
        if use_layer_norm:
            x_attended = hk.LayerNorm(
                axis=-1,
                create_scale=True,
                create_offset=True,
                name="output_norm"
            )(x_attended)
        
        # Output heads
        # Variable selection head - influenced by structure predictions
        var_features = x_attended
        
        # Incorporate parent probabilities more directly for variable selection
        # Higher parent probability should increase selection likelihood
        parent_influence = hk.Linear(hidden_dim // 2, name="parent_influence")(
            parent_probs[-1, :, None]  # Use most recent predictions
        )
        var_features = var_features + parent_influence.squeeze(-1)
        
        # Variable logits
        variable_head = hk.Linear(1, name="variable_head")(var_features)
        variable_logits = variable_head.squeeze(-1)  # [n_vars]
        
        # Mask out target variable
        variable_logits = jnp.where(
            jnp.arange(n_vars) == target_idx,
            -jnp.inf,
            variable_logits
        )
        
        # Value prediction head
        value_head = hk.Linear(2, name="value_head")(x_attended)  # [n_vars, 2]
        
        # Constrain log_std to reasonable range
        value_mean = value_head[:, 0]
        value_log_std = jnp.clip(value_head[:, 1], -5.0, 2.0)
        value_params = jnp.stack([value_mean, value_log_std], axis=1)
        
        return {
            'variable_logits': variable_logits,
            'value_params': value_params,
            'attention_weights': attention_weights  # For interpretability
        }
    
    return policy_fn


def create_lightweight_policy(hidden_dim: int = 128) -> callable:
    """
    Create a lightweight version of the unified policy.
    
    Fewer parameters but same interface. Good for faster experiments.
    """
    return create_unified_policy(
        hidden_dim=hidden_dim,
        num_layers=2,
        use_layer_norm=True,
        dropout_rate=0.0  # No dropout in lightweight version
    )


def verify_unified_parameter_compatibility(
    params: Any,
    transformed_fn: hk.Transformed,
    dummy_input: jnp.ndarray
) -> bool:
    """
    Verify that parameters are compatible with the unified policy.
    
    Args:
        params: Parameters to check
        transformed_fn: Transformed policy function
        dummy_input: Dummy input tensor for shape checking
        
    Returns:
        True if parameters are compatible
    """
    try:
        # Try to apply the function
        _ = transformed_fn.apply(params, None, dummy_input, 0)
        return True
    except Exception as e:
        logger.warning(f"Parameter compatibility check failed: {e}")
        return False