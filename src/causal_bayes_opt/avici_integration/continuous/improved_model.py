"""
Improved Continuous Parent Set Prediction Model with Relationship-Aware Aggregation.

This module fixes the aggregation issue in the original model by preserving
cross-variable relationships during the encoding process.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional


class ImprovedNodeEncoder(hk.Module):
    """
    Improved encoder that preserves cross-variable relationships.
    
    Key improvements:
    1. Cross-sample attention to capture correlations
    2. Information-aware aggregation weighting
    3. Explicit relationship encoding
    """
    
    def __init__(self,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_attention_heads: int = 4,
                 key_size: int = 32,
                 name: str = "ImprovedNodeEncoder"):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.key_size = key_size
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
    
    def __call__(self, data: jnp.ndarray) -> jnp.ndarray:
        """
        Encode intervention data into node representations with relationship awareness.
        
        Args:
            data: Intervention data [N, d, 3] where:
                  [:, :, 0] = variable values
                  [:, :, 1] = intervention indicators (1 if intervened)
                  [:, :, 2] = target indicators (1 for target variable, 0 otherwise)
                  
        Returns:
            Node embeddings [d, hidden_dim] that preserve relationships
        """
        N, d, channels = data.shape
        
        # Step 1: Initial per-element embedding
        flattened = data.reshape(N * d, channels)
        x = hk.Linear(self.hidden_dim, w_init=self.w_init)(flattened)
        x = jax.nn.relu(x)
        
        # Step 2: Process with residual blocks (same as original)
        for _ in range(self.num_layers - 1):
            residual = x
            x = hk.Linear(self.hidden_dim, w_init=self.w_init)(x)
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            x = jax.nn.relu(x)
            x = x + residual
        
        # Reshape back to samples x variables
        x = x.reshape(N, d, self.hidden_dim)  # [N, d, hidden_dim]
        
        # Step 3: Cross-sample attention to capture relationships
        # This is the KEY improvement - let each variable see its pattern across samples
        for layer_idx in range(2):  # 2 layers of cross-sample attention
            # Layer norm
            x_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            
            # Multi-head attention across samples for each variable
            # We transpose to [d, N, hidden_dim] so each variable can attend to its samples
            x_transposed = jnp.transpose(x_norm, (1, 0, 2))  # [d, N, hidden_dim]
            
            # Create attention module
            attn = hk.MultiHeadAttention(
                num_heads=self.num_attention_heads,
                key_size=self.key_size,
                w_init_scale=2.0,
                model_size=self.hidden_dim,
                name=f"cross_sample_attention_{layer_idx}"
            )
            
            # Each variable attends to its values across all samples
            # This captures temporal/sample patterns and correlations
            attended = jax.vmap(
                lambda var_samples: attn(var_samples, var_samples, var_samples),
                in_axes=0,
                out_axes=0
            )(x_transposed)  # [d, N, hidden_dim]
            
            # Transpose back and add residual
            x_attn = jnp.transpose(attended, (1, 0, 2))  # [N, d, hidden_dim]
            x = x + x_attn
        
        # Step 4: Compute relationship-aware aggregation weights
        # Key insight: Observational samples are more informative for learning correlations
        # Interventional samples break natural correlations
        
        # Extract intervention indicators
        intervention_mask = data[:, :, 1]  # [N, d], 1 if intervened
        target_mask = data[:, :, 2]   # [N, d], 1 if target variable
        
        # Compute informativeness score for each sample
        # A sample is informative if it has many observed (non-intervened) variables
        # Note: target_mask not used here - it just indicates which variable is being predicted
        n_observed_per_sample = jnp.sum((1 - intervention_mask), axis=1)  # [N]
        informativeness = n_observed_per_sample / d  # Normalize by number of variables
        
        # Convert to attention weights using softmax
        # Temperature parameter controls sharpness (lower = sharper)
        temperature = 0.5
        attention_weights = jax.nn.softmax(informativeness / temperature)  # [N]
        
        # Step 5: Weighted aggregation that preserves relationships
        # Instead of simple mean, use informativeness-weighted average
        node_embeddings = jnp.einsum('n,ndi->di', attention_weights, x)  # [d, hidden_dim]
        
        # Step 6: Add explicit correlation features
        # Compute correlation matrix from observed data
        values = data[:, :, 0]  # [N, d]
        
        # Mask out intervened values for correlation computation
        masked_values = jnp.where(
            observation_mask * (1 - intervention_mask),
            values,
            jnp.nan
        )
        
        # Compute correlation matrix (handling NaN values)
        # This is a simplified version - in practice would use more robust correlation estimation
        centered = masked_values - jnp.nanmean(masked_values, axis=0, keepdims=True)
        cov = jnp.nanmean(centered[:, :, None] * centered[:, None, :], axis=0)  # [d, d]
        std = jnp.sqrt(jnp.diag(cov))
        correlation = cov / (std[:, None] * std[None, :] + 1e-8)  # [d, d]
        
        # Encode correlation information
        correlation_features = hk.Linear(
            self.hidden_dim // 4,
            w_init=self.w_init,
            name="correlation_encoder"
        )(correlation)  # [d, hidden_dim // 4]
        
        # Combine node embeddings with correlation features
        combined = jnp.concatenate([
            node_embeddings[:, :3 * self.hidden_dim // 4],
            correlation_features
        ], axis=1)  # [d, hidden_dim]
        
        # Final projection to ensure correct dimensionality
        node_embeddings = hk.Linear(
            self.hidden_dim,
            w_init=self.w_init,
            name="final_projection"
        )(combined)
        
        return node_embeddings


class ImprovedParentAttentionLayer(hk.Module):
    """Parent attention layer - same as original but included for completeness."""
    
    def __init__(self, 
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 key_size: int = 32,
                 name: str = "ImprovedParentAttentionLayer"):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.key_size = key_size
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
    
    def __call__(self, 
                 query: jnp.ndarray,      # [hidden_dim] - target node embedding
                 key_value: jnp.ndarray   # [n_vars, hidden_dim] - all node embeddings
                 ) -> jnp.ndarray:        # [n_vars] - parent attention scores
        """Compute attention scores between target node and all potential parents."""
        n_vars = key_value.shape[0]
        
        # Expand query to match batch dimension
        query_expanded = jnp.tile(query[None, :], (n_vars, 1))  # [n_vars, hidden_dim]
        
        # Multi-head attention computation
        attention = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.key_size,
            w_init_scale=2.0,
            model_size=self.hidden_dim,
        )
        
        # Compute attention: query attends to all potential parents
        attended = attention(
            query=query_expanded,    # [n_vars, hidden_dim]
            key=key_value,          # [n_vars, hidden_dim]  
            value=key_value         # [n_vars, hidden_dim]
        )  # [n_vars, hidden_dim]
        
        # Project to scalar scores
        parent_logits = hk.Linear(1, w_init=self.w_init)(attended).squeeze(-1)  # [n_vars]
        
        return parent_logits


class ImprovedContinuousParentSetPredictionModel(hk.Module):
    """
    Improved continuous parent set prediction model with relationship-aware encoding.
    
    This model fixes the aggregation issue by:
    1. Using cross-sample attention to capture correlations
    2. Weighting samples by informativeness
    3. Explicitly encoding correlation structure
    4. Preserving all relationship information through aggregation
    """
    
    def __init__(self,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 key_size: int = 32,
                 dropout: float = 0.1,
                 name: str = "ImprovedContinuousParentSetPredictionModel"):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.key_size = key_size
        self.dropout = dropout
    
    def __call__(self, 
                 data: jnp.ndarray,         # [N, d, 3] intervention data
                 target_variable: int,      # Target variable index
                 is_training: bool = True   # Training mode flag
                 ) -> dict[str, jnp.ndarray]:
        """
        Predict parent probabilities for target variable with improved encoding.
        
        Returns same format as original for compatibility.
        """
        N, d, channels = data.shape
        dropout_rate = self.dropout if is_training else 0.0
        
        # Use improved encoder that preserves relationships
        node_encoder = ImprovedNodeEncoder(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_attention_heads=self.num_heads // 2,  # Fewer heads for cross-sample attention
            key_size=self.key_size
        )
        node_embeddings = node_encoder(data)  # [d, hidden_dim]
        
        # Apply dropout for regularization
        if is_training:
            node_embeddings = hk.dropout(hk.next_rng_key(), dropout_rate, node_embeddings)
        
        # Get target node embedding
        target_embedding = node_embeddings[target_variable]  # [hidden_dim]
        
        # Compute parent attention scores (same as original)
        parent_attention = ImprovedParentAttentionLayer(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            key_size=self.key_size
        )
        parent_logits = parent_attention(target_embedding, node_embeddings)  # [d]
        
        # Mask target variable (cannot be its own parent)
        masked_logits = jnp.where(
            jnp.arange(d) == target_variable,
            -1e9,
            parent_logits
        )
        
        # Convert to probabilities
        parent_probs = jax.nn.softmax(masked_logits)  # [d]
        
        return {
            'node_embeddings': node_embeddings,
            'target_embedding': target_embedding,
            'attention_logits': parent_logits,
            'parent_probabilities': parent_probs
        }