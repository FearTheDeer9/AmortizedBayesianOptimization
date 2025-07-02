"""
Continuous Parent Set Prediction Model.

This module implements a continuous alternative to discrete parent set enumeration,
using attention mechanisms and probability distributions for scalable causal discovery.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional


class ParentAttentionLayer(hk.Module):
    """Attention layer for learning parent relationships."""
    
    def __init__(self, 
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 key_size: int = 32,
                 name: str = "ParentAttentionLayer"):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.key_size = key_size
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
    
    def __call__(self, 
                 query: jnp.ndarray,      # [hidden_dim] - target node embedding
                 key_value: jnp.ndarray   # [n_vars, hidden_dim] - all node embeddings
                 ) -> jnp.ndarray:        # [n_vars] - parent attention scores
        """
        Compute attention scores between target node and all potential parents.
        
        Args:
            query: Target node embedding [hidden_dim]
            key_value: All node embeddings [n_vars, hidden_dim]
            
        Returns:
            Parent attention logits [n_vars]
        """
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


class NodeEncoder(hk.Module):
    """Encoder for learning node representations from intervention data."""
    
    def __init__(self,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 name: str = "NodeEncoder"):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
    
    def __call__(self, data: jnp.ndarray) -> jnp.ndarray:
        """
        Encode intervention data into node representations.
        
        Args:
            data: Intervention data [N, d, 3] where:
                  [:, :, 0] = variable values
                  [:, :, 1] = intervention indicators
                  [:, :, 2] = target indicators
                  
        Returns:
            Node embeddings [d, hidden_dim]
        """
        N, d, channels = data.shape
        assert channels == 3, f"Expected 3 channels, got {channels}"
        
        # Flatten across samples for processing: [N*d, 3]
        flattened = data.reshape(N * d, channels)
        
        # Initial embedding
        x = hk.Linear(self.hidden_dim, w_init=self.w_init)(flattened)  # [N*d, hidden_dim]
        x = jax.nn.relu(x)
        
        # Additional layers for more complex representations
        for _ in range(self.num_layers - 1):
            residual = x
            x = hk.Linear(self.hidden_dim, w_init=self.w_init)(x)
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            x = jax.nn.relu(x)
            x = x + residual  # Residual connection
        
        # Reshape back and aggregate across samples: [N, d, hidden_dim] -> [d, hidden_dim]
        x = x.reshape(N, d, self.hidden_dim)
        node_embeddings = jnp.mean(x, axis=0)  # Average across samples
        
        return node_embeddings


class ContinuousParentSetPredictionModel(hk.Module):
    """
    Continuous parent set prediction model using attention mechanisms.
    
    This model replaces discrete parent set enumeration with continuous
    probability distributions over parent relationships, enabling:
    - Natural JAX compatibility (no lookup tables or enumeration)
    - Linear scaling with number of variables (vs exponential)
    - End-to-end differentiability
    - Natural uncertainty quantification
    """
    
    def __init__(self,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 key_size: int = 32,
                 dropout: float = 0.1,
                 name: str = "ContinuousParentSetPredictionModel"):
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
                 ) -> jnp.ndarray:          # [d] parent probabilities
        """
        Predict parent probabilities for target variable.
        
        Args:
            data: Intervention data [N, d, 3]
            target_variable: Index of target variable (0 <= target_variable < d)
            is_training: Whether model is in training mode
            
        Returns:
            Parent probabilities [d] where probabilities sum to 1.0
            and target_variable has probability 0.0
        """
        N, d, channels = data.shape
        # Note: Cannot use assert on traced values in JAX compilation context
        # Bounds checking is handled by JAX indexing operations
        
        dropout_rate = self.dropout if is_training else 0.0
        
        # Encode intervention data into node representations
        node_encoder = NodeEncoder(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )
        node_embeddings = node_encoder(data)  # [d, hidden_dim]
        
        # Apply dropout for regularization
        if is_training:
            node_embeddings = hk.dropout(hk.next_rng_key(), dropout_rate, node_embeddings)
        
        # Get target node embedding
        target_embedding = node_embeddings[target_variable]  # [hidden_dim]
        
        # Compute parent attention scores
        parent_attention = ParentAttentionLayer(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            key_size=self.key_size
        )
        parent_logits = parent_attention(target_embedding, node_embeddings)  # [d]
        
        # Mask target variable (cannot be its own parent)
        # Use a large negative value instead of log(0) for better numerics
        mask = jnp.ones(d)
        masked_logits = jnp.where(
            jnp.arange(d) == target_variable,
            -1e9,  # Large negative value for target variable
            parent_logits
        )
        
        # Convert to probabilities using masked softmax
        parent_probs = jax.nn.softmax(masked_logits)  # [d]
        # Note: target variable will automatically have ~0 probability due to -1e9 logit
        
        return parent_probs
    
    def compute_uncertainty(self, parent_probs: jnp.ndarray) -> float:
        """
        Compute uncertainty measure from parent probabilities.
        
        Args:
            parent_probs: Parent probabilities [d]
            
        Returns:
            Entropy-based uncertainty measure
        """
        # Entropy as uncertainty measure
        entropy = -jnp.sum(parent_probs * jnp.log(parent_probs + 1e-8))
        return entropy
    
    def get_top_k_parents(self, 
                         parent_probs: jnp.ndarray, 
                         k: int = 3) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Extract top-k parent variables for backward compatibility.
        
        Args:
            parent_probs: Parent probabilities [d]
            k: Number of top parents to return
            
        Returns:
            Tuple of (top_k_indices, top_k_probabilities)
        """
        k = min(k, parent_probs.shape[0])
        top_k_indices = jnp.argsort(parent_probs)[-k:][::-1]  # Descending order
        top_k_probs = parent_probs[top_k_indices]
        return top_k_indices, top_k_probs


def layer_norm(axis: int = -1) -> hk.LayerNorm:
    """Helper function for creating layer normalization."""
    return hk.LayerNorm(axis=axis, create_scale=True, create_offset=True)