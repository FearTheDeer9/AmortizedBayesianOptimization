"""
Node Feature Encoder - Creates individual node representations without inter-node attention.

This encoder focuses on creating good per-variable representations from the data,
leaving relationship modeling to the ParentAttentionLayer.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional


class NodeFeatureEncoder(hk.Module):
    """
    Encoder that creates node embeddings from individual variable features.
    
    Key principles:
    - No inter-node attention (avoid uniformity)
    - Rich per-variable features
    - Maintain diversity through independent processing
    - Let ParentAttentionLayer handle relationships
    """
    
    def __init__(self,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout_rate: float = 0.1,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
    def __call__(self,
                 data: jnp.ndarray,
                 is_training: bool = True) -> jnp.ndarray:
        """
        Encode intervention data into variable embeddings.
        
        Args:
            data: [N, d, 3] tensor (values, target_indicator, intervention_indicator)
            is_training: Whether in training mode
            
        Returns:
            embeddings: [d, hidden_dim]
        """
        N, d, channels = data.shape
        
        # Extract channels
        values = data[:, :, 0]  # [N, d]
        target_indicators = data[:, :, 1]  # [N, d]
        intervention_indicators = data[:, :, 2]  # [N, d]
        
        # Compute per-variable features
        features_list = []
        
        # 1. Basic statistics
        features_list.append(jnp.mean(values, axis=0))  # [d]
        features_list.append(jnp.std(values, axis=0))   # [d]
        features_list.append(jnp.min(values, axis=0))   # [d]
        features_list.append(jnp.max(values, axis=0))   # [d]
        
        # 2. Quantiles
        features_list.append(jnp.percentile(values, 25, axis=0))  # [d]
        features_list.append(jnp.percentile(values, 50, axis=0))  # [d] median
        features_list.append(jnp.percentile(values, 75, axis=0))  # [d]
        
        # 3. Higher moments
        centered = values - jnp.mean(values, axis=0, keepdims=True)
        features_list.append(jnp.mean(centered**3, axis=0) / (jnp.std(values, axis=0)**3 + 1e-8))  # Skewness
        features_list.append(jnp.mean(centered**4, axis=0) / (jnp.std(values, axis=0)**4 + 1e-8) - 3)  # Kurtosis
        
        # 4. Intervention information
        features_list.append(jnp.mean(intervention_indicators, axis=0))  # Intervention rate
        features_list.append(jnp.sum(intervention_indicators, axis=0))   # Total interventions
        
        # 5. Value dynamics (useful for time series aspects)
        value_diffs = values[1:] - values[:-1]
        features_list.append(jnp.mean(value_diffs, axis=0))  # Mean change
        features_list.append(jnp.std(value_diffs, axis=0))   # Volatility
        
        # 6. Extreme value statistics
        features_list.append(jnp.percentile(values, 5, axis=0))   # Lower tail
        features_list.append(jnp.percentile(values, 95, axis=0))  # Upper tail
        
        # Stack all features
        features = jnp.stack(features_list, axis=1)  # [d, num_features]
        
        # Handle NaN values
        features = jnp.nan_to_num(features, 0.0)
        
        # Add positional encoding to break symmetry
        # This helps distinguish variables even if they have similar statistics
        position_encoding = self.get_position_encoding(d, features.shape[1])
        features = features + 0.1 * position_encoding  # Small contribution
        
        # Project to hidden dimension through MLP
        embeddings = features
        
        for layer_idx in range(self.num_layers):
            # Layer with residual connection
            layer = hk.Linear(
                self.hidden_dim if layer_idx == 0 else self.hidden_dim,
                w_init=hk.initializers.VarianceScaling(2.0, "fan_in", "uniform"),
                name=f"layer_{layer_idx}"
            )
            
            if layer_idx == 0:
                embeddings = layer(embeddings)
            else:
                # Residual connection for deeper layers
                residual = embeddings
                embeddings = layer(embeddings)
                embeddings = embeddings + residual
            
            # Activation
            embeddings = jax.nn.gelu(embeddings)
            
            # Layer normalization
            embeddings = hk.LayerNorm(
                axis=-1,
                create_scale=True,
                create_offset=True,
                name=f"ln_{layer_idx}"
            )(embeddings)
            
            # Dropout
            if is_training and self.dropout_rate > 0:
                embeddings = hk.dropout(hk.next_rng_key(), self.dropout_rate, embeddings)
        
        # Final projection if needed
        if embeddings.shape[-1] != self.hidden_dim:
            final_proj = hk.Linear(
                self.hidden_dim,
                w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
                name="final_projection"
            )
            embeddings = final_proj(embeddings)
        
        return embeddings
    
    def get_position_encoding(self, n_positions: int, dim: int) -> jnp.ndarray:
        """
        Create sinusoidal position encodings to break symmetry.
        
        Args:
            n_positions: Number of positions (variables)
            dim: Dimension of encoding
            
        Returns:
            position_encoding: [n_positions, dim]
        """
        positions = jnp.arange(n_positions)[:, None]
        dimensions = jnp.arange(dim)[None, :]
        
        # Sinusoidal encoding
        angles = positions / jnp.power(10000, 2 * (dimensions // 2) / dim)
        
        encoding = jnp.zeros((n_positions, dim))
        encoding = encoding.at[:, 0::2].set(jnp.sin(angles[:, 0::2]))
        encoding = encoding.at[:, 1::2].set(jnp.cos(angles[:, 1::2]))
        
        return encoding