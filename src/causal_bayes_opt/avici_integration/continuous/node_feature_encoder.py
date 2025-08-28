"""
Node Feature Encoder - Implements alternating attention mechanism inspired by AVICI.

This encoder alternates attention between observation and variable axes,
providing strong inductive bias for causal discovery.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional


class NodeFeatureEncoder(hk.Module):
    """
    Encoder that implements alternating attention between observations and variables.
    
    Key principles:
    - Alternates attention between observation axis (N) and variable axis (d)
    - Provides strong inductive bias through architectural design
    - Captures both local patterns (within variables) and global patterns (across variables)
    - Based on AVICI architecture (Lorch et al.)
    """
    
    def __init__(self,
                 hidden_dim: int = 128,
                 num_layers: int = 8,  # AVICI uses 8 layers
                 num_heads: int = 8,
                 key_size: int = 32,
                 dropout_rate: float = 0.1,
                 widening_factor: int = 4,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers * 2  # Double for alternating (like AVICI)
        self.num_heads = num_heads
        self.key_size = key_size
        self.dropout_rate = dropout_rate
        self.widening_factor = widening_factor
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
        
    def _attention_block(self, z: jnp.ndarray, dropout_rate: float, name_suffix: str) -> jnp.ndarray:
        """Single attention block with residual connection and FFN."""
        # Multi-head attention with residual
        z_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                              name=f"ln_attn_{name_suffix}")(z)
        
        z_attn = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.key_size,
            w_init_scale=2.0,
            model_size=self.hidden_dim,
            name=f"mha_{name_suffix}"
        )(z_norm, z_norm, z_norm)
        
        z = z + hk.dropout(hk.next_rng_key(), dropout_rate, z_attn)
        
        # Feed-forward network with residual
        z_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                              name=f"ln_ffn_{name_suffix}")(z)
        
        z_ffn = hk.Sequential([
            hk.Linear(self.widening_factor * self.hidden_dim, w_init=self.w_init,
                     name=f"ffn_up_{name_suffix}"),
            jax.nn.relu,
            hk.Linear(self.hidden_dim, w_init=self.w_init,
                     name=f"ffn_down_{name_suffix}"),
        ])(z_norm)
        
        z = z + hk.dropout(hk.next_rng_key(), dropout_rate, z_ffn)
        return z
    
    def __call__(self,
                 data: jnp.ndarray,
                 is_training: bool = True) -> jnp.ndarray:
        """
        Encode intervention data into variable embeddings using alternating attention.
        
        Args:
            data: [N, d, 3] or [batch_size, N, d, 3] tensor (values, target_indicator, intervention_indicator)
            is_training: Whether in training mode
            
        Returns:
            embeddings: [d, hidden_dim] or [batch_size, d, hidden_dim]
        """
        # Handle both single and batch inputs
        is_batched = data.ndim == 4
        if is_batched:
            batch_size, N, d, channels = data.shape
        else:
            N, d, channels = data.shape
            data = data[None, ...]  # Add batch dimension: [1, N, d, 3]
            batch_size = 1
        
        dropout_rate = self.dropout_rate if is_training else 0.0
        
        # Initial projection to hidden dimension
        z = hk.Linear(self.hidden_dim, w_init=self.w_init, name="input_projection")(data)
        # z shape: [batch_size, N, d, hidden_dim]
        
        # Apply alternating attention blocks
        for layer_idx in range(self.num_layers):
            # Apply attention block
            z = self._attention_block(z, dropout_rate, name_suffix=str(layer_idx))
            
            # Swap axes to alternate between observation and variable attention
            # Even layers: attention over observations (axis -3)
            # Odd layers: attention over variables (axis -2)
            z = jnp.swapaxes(z, -3, -2)
        
        # Final layer norm
        z = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                        name="final_ln")(z)
        
        # Max pooling over observations to get variable embeddings
        # After alternating, we need to determine which axis is observations
        # If num_layers is even, observations are back in axis -3
        # If num_layers is odd, observations are in axis -2
        if self.num_layers % 2 == 0:
            # Observations in axis -3 (original position)
            embeddings = jnp.max(z, axis=-3)  # [batch_size, d, hidden_dim]
        else:
            # Observations in axis -2 (swapped position)
            embeddings = jnp.max(z, axis=-2)  # [batch_size, N, hidden_dim]
            # Need to swap back to get [batch_size, d, hidden_dim]
            embeddings = jnp.swapaxes(embeddings, -2, -1)
        
        # Remove batch dimension if input wasn't batched
        if not is_batched:
            embeddings = embeddings[0]  # [d, hidden_dim]
        
        return embeddings