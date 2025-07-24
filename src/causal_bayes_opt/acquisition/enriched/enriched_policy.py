"""
Enriched attention encoder for enhanced context architecture.

This module implements the enriched transformer that processes multi-channel
input with temporal context evolution, replacing post-transformer feature
concatenation with learned attention over enriched input.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def layer_norm(axis: int = -1, name: Optional[str] = None) -> hk.LayerNorm:
    """Helper function for creating layer normalization."""
    return hk.LayerNorm(axis=axis, create_scale=True, create_offset=True, name=name)


class EnrichedAttentionEncoder(hk.Module):
    """
    Enriched attention encoder for processing multi-channel temporal input.
    
    This encoder replaces the post-transformer feature concatenation approach
    with a design that processes ALL context information through transformer
    attention, enabling temporal learning of context evolution.
    """
    
    def __init__(self,
                 num_layers: int = 4,
                 num_heads: int = 8, 
                 hidden_dim: int = 128,
                 key_size: int = 32,
                 widening_factor: int = 4,
                 dropout: float = 0.1,
                 name: str = "EnrichedAttentionEncoder"):
        super().__init__(name=name)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.key_size = key_size
        self.widening_factor = widening_factor
        self.dropout = dropout
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
    
    def __call__(self, 
                 enriched_history: jnp.ndarray,  # [max_history_size, n_vars, num_channels]
                 is_training: bool = True        # Training mode flag
                 ) -> jnp.ndarray:               # [n_vars, hidden_dim]
        """
        Process enriched history through transformer attention.
        
        Args:
            enriched_history: Multi-channel temporal input [T, n_vars, C]
            is_training: Whether model is in training mode
            
        Returns:
            Variable embeddings [n_vars, hidden_dim]
        """
        max_history_size, n_vars, num_channels = enriched_history.shape
        dropout_rate = self.dropout if is_training else 0.0
        
        # Project enriched input to hidden dimension
        x = self._project_enriched_input(enriched_history)  # [T, n_vars, hidden_dim]
        
        # Add positional encoding for temporal information
        x = self._add_positional_encoding(x)  # [T, n_vars, hidden_dim]
        
        # Apply transformer layers with alternating attention patterns
        x = self._apply_transformer_layers(x, dropout_rate)  # [T, n_vars, hidden_dim]
        
        # Aggregate temporal information to get final variable embeddings
        variable_embeddings = self._aggregate_temporal_features(x)  # [n_vars, hidden_dim]
        
        return variable_embeddings
    
    def _project_enriched_input(self, enriched_history: jnp.ndarray) -> jnp.ndarray:
        """
        Project multi-channel input to hidden dimension.
        
        Args:
            enriched_history: Input [T, n_vars, num_channels]
            
        Returns:
            Projected input [T, n_vars, hidden_dim]
        """
        T, n_vars, num_channels = enriched_history.shape
        
        # Flatten for projection
        flattened = enriched_history.reshape(T * n_vars, num_channels)
        
        # Project to hidden dimension with residual connection and normalization
        projected = hk.Linear(self.hidden_dim, w_init=self.w_init, name="input_projection")(flattened)
        
        # Add channel-wise attention to weight different types of information
        channel_weights = self._compute_channel_attention(flattened)  # [T*n_vars, num_channels]
        weighted_input = jnp.sum(
            flattened[:, :, None] * channel_weights[:, :, None], axis=1
        )  # [T*n_vars, 1] -> [T*n_vars, hidden_dim]
        
        # Combine projected features with weighted input
        combined = projected + hk.Linear(
            self.hidden_dim, w_init=self.w_init, name="weighted_input_projection"
        )(weighted_input)
        
        # Reshape back to temporal structure
        x = combined.reshape(T, n_vars, self.hidden_dim)
        
        # Apply layer normalization
        x = layer_norm(axis=-1, name="input_layer_norm")(x)
        
        return x
    
    def _compute_channel_attention(self, flattened_input: jnp.ndarray) -> jnp.ndarray:
        """
        Compute attention weights for different input channels.
        
        Args:
            flattened_input: Input [T*n_vars, num_channels]
            
        Returns:
            Channel attention weights [T*n_vars, num_channels]
        """
        num_channels = flattened_input.shape[-1]
        
        # Compute channel importance scores
        channel_scores = hk.Linear(
            num_channels, w_init=self.w_init, name="channel_attention"
        )(flattened_input)  # [T*n_vars, num_channels]
        
        # Apply softmax to get attention weights
        channel_weights = jax.nn.softmax(channel_scores, axis=-1)
        
        return channel_weights
    
    def _add_positional_encoding(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Variable-agnostic: No positional encoding for variables.
        
        Variables should learn relationships through data patterns in alternating attention,
        not through imposed positional structure. This enables variable-agnostic processing
        for SCMs with 3-8+ variables.
        
        Args:
            x: Input [T, n_vars, hidden_dim]
            
        Returns:
            Input without variable positional encoding [T, n_vars, hidden_dim]
        """
        # Keep only minimal temporal information if needed for recency
        # But no variable positional embeddings - relationships learned through attention
        return x
    
    def _apply_transformer_layers(self, x: jnp.ndarray, dropout_rate: float) -> jnp.ndarray:
        """
        Apply transformer layers with alternating attention patterns.
        
        Args:
            x: Input [T, n_vars, hidden_dim]
            dropout_rate: Dropout rate
            
        Returns:
            Transformed features [T, n_vars, hidden_dim]
        """
        for layer_idx in range(self.num_layers):
            # Alternate between temporal and variable attention
            if layer_idx % 2 == 0:
                # Temporal attention: attend across time for each variable
                x = self._temporal_attention_layer(x, dropout_rate, layer_idx)
            else:
                # Variable attention: attend across variables for each time step
                x = self._variable_attention_layer(x, dropout_rate, layer_idx)
        
        return x
    
    def _temporal_attention_layer(self, 
                                x: jnp.ndarray, 
                                dropout_rate: float, 
                                layer_idx: int) -> jnp.ndarray:
        """
        Temporal attention layer: attend across time for each variable.
        
        Args:
            x: Input [T, n_vars, hidden_dim]
            dropout_rate: Dropout rate
            layer_idx: Layer index for naming
            
        Returns:
            Output [T, n_vars, hidden_dim]
        """
        T, n_vars, hidden_dim = x.shape
        
        # Reshape for temporal attention: [n_vars, T, hidden_dim]
        x_reshaped = jnp.transpose(x, (1, 0, 2))
        
        # Apply attention across time dimension for each variable
        def temporal_attention_for_variable(var_sequence):
            # var_sequence: [T, hidden_dim]
            q_in = layer_norm(axis=-1, name=f"temporal_attn_{layer_idx}_q_norm")(var_sequence)
            k_in = layer_norm(axis=-1, name=f"temporal_attn_{layer_idx}_k_norm")(var_sequence)
            v_in = layer_norm(axis=-1, name=f"temporal_attn_{layer_idx}_v_norm")(var_sequence)
            
            attn_out = hk.MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.key_size,
                w_init_scale=2.0,
                model_size=self.hidden_dim,
                name=f"temporal_attention_{layer_idx}"
            )(q_in, k_in, v_in)
            
            return var_sequence + hk.dropout(hk.next_rng_key(), dropout_rate, attn_out)
        
        # Apply to all variables
        attended = jax.vmap(temporal_attention_for_variable)(x_reshaped)  # [n_vars, T, hidden_dim]
        
        # Reshape back and apply FFN
        x_attended = jnp.transpose(attended, (1, 0, 2))  # [T, n_vars, hidden_dim]
        
        # Feed-forward network
        x_out = self._apply_ffn(x_attended, dropout_rate, f"temporal_ffn_{layer_idx}")
        
        return x_out
    
    def _variable_attention_layer(self, 
                                x: jnp.ndarray, 
                                dropout_rate: float, 
                                layer_idx: int) -> jnp.ndarray:
        """
        Variable attention layer: attend across variables for each time step.
        
        Args:
            x: Input [T, n_vars, hidden_dim]
            dropout_rate: Dropout rate
            layer_idx: Layer index for naming
            
        Returns:
            Output [T, n_vars, hidden_dim]
        """
        T, n_vars, hidden_dim = x.shape
        
        # Apply attention across variable dimension for each time step
        def variable_attention_for_timestep(timestep_vars):
            # timestep_vars: [n_vars, hidden_dim]
            q_in = layer_norm(axis=-1, name=f"variable_attn_{layer_idx}_q_norm")(timestep_vars)
            k_in = layer_norm(axis=-1, name=f"variable_attn_{layer_idx}_k_norm")(timestep_vars)
            v_in = layer_norm(axis=-1, name=f"variable_attn_{layer_idx}_v_norm")(timestep_vars)
            
            attn_out = hk.MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.key_size,
                w_init_scale=2.0,
                model_size=self.hidden_dim,
                name=f"variable_attention_{layer_idx}"
            )(q_in, k_in, v_in)
            
            return timestep_vars + hk.dropout(hk.next_rng_key(), dropout_rate, attn_out)
        
        # Apply to all time steps
        attended = jax.vmap(variable_attention_for_timestep)(x)  # [T, n_vars, hidden_dim]
        
        # Feed-forward network
        x_out = self._apply_ffn(attended, dropout_rate, f"variable_ffn_{layer_idx}")
        
        return x_out
    
    def _apply_ffn(self, x: jnp.ndarray, dropout_rate: float, name_prefix: str) -> jnp.ndarray:
        """
        Apply feed-forward network with residual connection.
        
        Args:
            x: Input [T, n_vars, hidden_dim]
            dropout_rate: Dropout rate
            name_prefix: Prefix for layer names
            
        Returns:
            Output [T, n_vars, hidden_dim]
        """
        # Layer normalization
        x_in = layer_norm(axis=-1, name=f"{name_prefix}_norm")(x)
        
        # Feed-forward layers
        x_ffn = hk.Sequential([
            hk.Linear(self.widening_factor * self.hidden_dim, w_init=self.w_init, name=f"{name_prefix}_linear1"),
            jax.nn.relu,
            hk.Linear(self.hidden_dim, w_init=self.w_init, name=f"{name_prefix}_linear2"),
        ], name=f"{name_prefix}_ffn")(x_in)
        
        # Residual connection with dropout
        return x + hk.dropout(hk.next_rng_key(), dropout_rate, x_ffn)
    
    def _aggregate_temporal_features(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Aggregate temporal information to get final variable embeddings.
        
        Uses a learnable query vector for attention-based aggregation that is
        independent of the temporal dimension T, allowing variable sequence lengths.
        
        Args:
            x: Input [T, n_vars, hidden_dim]
            
        Returns:
            Variable embeddings [n_vars, hidden_dim]
        """
        T, n_vars, hidden_dim = x.shape
        
        # Apply final layer normalization
        x = layer_norm(axis=-1, name="final_layer_norm")(x)
        
        # Create learnable query vector for temporal aggregation
        # This is T-independent, allowing variable sequence lengths
        query = hk.get_parameter(
            "temporal_aggregation_query",
            shape=[hidden_dim],
            dtype=x.dtype,
            init=self.w_init
        )
        
        # Compute attention scores between query and temporal features
        # Shape: [T, n_vars] - dot product of query with each temporal position
        attention_scores = jnp.einsum('tvi,i->tv', x, query)
        
        # Apply softmax across time dimension to get attention weights
        attention_weights = jax.nn.softmax(attention_scores / jnp.sqrt(hidden_dim), axis=0)
        
        # Weighted sum across time using attention weights
        aggregated = jnp.einsum('tvi,tv->vi', x, attention_weights)  # [n_vars, hidden_dim]
        
        # Optional: Add max pooling as auxiliary information
        max_pooled = jnp.max(x, axis=0)  # [n_vars, hidden_dim]
        
        # Combine weighted attention and max pooling
        combined = aggregated + 0.1 * max_pooled  # Small weight for max pooling
        
        # Final projection and normalization
        final_embeddings = hk.Linear(
            self.hidden_dim, w_init=self.w_init, name="final_projection"
        )(combined)
        
        final_embeddings = layer_norm(axis=-1, name="final_embedding_norm")(final_embeddings)
        
        return final_embeddings


class EnrichedTransformerBlock(hk.Module):
    """Individual transformer block for enriched attention encoder."""
    
    def __init__(self,
                 num_heads: int = 8,
                 hidden_dim: int = 128,
                 key_size: int = 32,
                 widening_factor: int = 4,
                 attention_type: str = "temporal",  # "temporal" or "variable"
                 name: str = "EnrichedTransformerBlock"):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.key_size = key_size
        self.widening_factor = widening_factor
        self.attention_type = attention_type
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
    
    def __call__(self, x: jnp.ndarray, dropout_rate: float = 0.0) -> jnp.ndarray:
        """
        Apply transformer block with specified attention type.
        
        Args:
            x: Input [T, n_vars, hidden_dim]
            dropout_rate: Dropout rate
            
        Returns:
            Output [T, n_vars, hidden_dim]
        """
        if self.attention_type == "temporal":
            return self._temporal_block(x, dropout_rate)
        elif self.attention_type == "variable":
            return self._variable_block(x, dropout_rate)
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")
    
    def _temporal_block(self, x: jnp.ndarray, dropout_rate: float) -> jnp.ndarray:
        """Temporal attention block."""
        # Multi-head attention across time
        attn_input = layer_norm(axis=-1, name="temporal_attn_norm")(x)
        
        # Reshape for temporal attention
        T, n_vars, hidden_dim = x.shape
        attn_input_reshaped = attn_input.reshape(T, n_vars * hidden_dim)
        
        attn_out = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.key_size,
            w_init_scale=2.0,
            model_size=n_vars * hidden_dim,
            name="temporal_attention"
        )(attn_input_reshaped, attn_input_reshaped, attn_input_reshaped)
        
        attn_out = attn_out.reshape(T, n_vars, hidden_dim)
        x = x + hk.dropout(hk.next_rng_key(), dropout_rate, attn_out)
        
        # Feed-forward
        ffn_input = layer_norm(axis=-1, name="temporal_ffn_norm")(x)
        ffn_out = hk.Sequential([
            hk.Linear(self.widening_factor * self.hidden_dim, w_init=self.w_init),
            jax.nn.relu,
            hk.Linear(self.hidden_dim, w_init=self.w_init),
        ], name="temporal_ffn")(ffn_input)
        
        return x + hk.dropout(hk.next_rng_key(), dropout_rate, ffn_out)
    
    def _variable_block(self, x: jnp.ndarray, dropout_rate: float) -> jnp.ndarray:
        """Variable attention block."""
        # Multi-head attention across variables
        attn_input = layer_norm(axis=-1, name="variable_attn_norm")(x)
        
        # Reshape for variable attention
        T, n_vars, hidden_dim = x.shape
        attn_input_reshaped = attn_input.transpose(0, 2, 1).reshape(T * hidden_dim, n_vars)
        
        attn_out = hk.MultiHeadAttention(
            num_heads=min(self.num_heads, n_vars),  # Can't have more heads than variables
            key_size=self.key_size,
            w_init_scale=2.0,
            model_size=n_vars,
            name="variable_attention"
        )(attn_input_reshaped, attn_input_reshaped, attn_input_reshaped)
        
        attn_out = attn_out.reshape(T, hidden_dim, n_vars).transpose(0, 2, 1)
        x = x + hk.dropout(hk.next_rng_key(), dropout_rate, attn_out)
        
        # Feed-forward
        ffn_input = layer_norm(axis=-1, name="variable_ffn_norm")(x)
        ffn_out = hk.Sequential([
            hk.Linear(self.widening_factor * self.hidden_dim, w_init=self.w_init),
            jax.nn.relu,
            hk.Linear(self.hidden_dim, w_init=self.w_init),
        ], name="variable_ffn")(ffn_input)
        
        return x + hk.dropout(hk.next_rng_key(), dropout_rate, ffn_out)