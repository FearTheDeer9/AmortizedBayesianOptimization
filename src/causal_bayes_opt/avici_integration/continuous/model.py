"""
Fixed Continuous Parent Set Prediction Model.

This fixes the attention mechanism to properly compute parent scores.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional


class ParentAttentionLayer(hk.Module):
    """Fixed attention layer that properly computes parent relationships."""
    
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
        
        Key fix: Use proper attention where the target queries each potential parent,
        not where identical queries are used for all parents.
        
        Args:
            query: Target node embedding [hidden_dim]
            key_value: All node embeddings [n_vars, hidden_dim]
            
        Returns:
            Parent attention logits [n_vars]
        """
        n_vars = key_value.shape[0]
        
        # Method 1: Simple dot product attention
        # This directly measures similarity between target and each potential parent
        # parent_logits = jnp.dot(key_value, query)  # [n_vars]
        
        # Method 2: Learned attention with proper query-key interaction
        # Transform query and keys to attention space
        query_projection = hk.Linear(self.key_size, w_init=self.w_init, name="query_proj")
        key_projection = hk.Linear(self.key_size, w_init=self.w_init, name="key_proj")
        
        q = query_projection(query)  # [key_size]
        k = key_projection(key_value)  # [n_vars, key_size]
        
        # Compute attention scores
        scores = jnp.dot(k, q) / jnp.sqrt(self.key_size)  # [n_vars]
        
        # Return scores directly without variable-specific bias
        # This ensures the model remains truly variable-agnostic
        return scores


class NodeEncoder(hk.Module):
    """Encoder that preserves variable-specific information."""
    
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
                  [:, :, 2] = observation indicators
                  
        Returns:
            Node embeddings [d, hidden_dim]
        """
        N, d, channels = data.shape
        assert channels == 3, f"Expected 3 channels, got {channels}"
        
        # Create shared layers once (outside the loop)
        layers = []
        for i in range(self.num_layers):
            layers.append(hk.Linear(self.hidden_dim, w_init=self.w_init, name=f"encoder_layer_{i}"))
        
        # Process all variables in parallel using vmap for true variable-agnostic behavior
        def process_single_variable(var_data):
            """Process a single variable's data."""
            # var_data shape: [N, 3]
            values = var_data[:, 0]  # [N]
            interventions = var_data[:, 1]  # [N]
            observations = var_data[:, 2]  # [N]
            
            # Only use observational data for statistics
            obs_mask = observations * (1 - interventions)
            masked_values = jnp.where(obs_mask, values, 0.0)
            
            # Compute features
            count = jnp.sum(obs_mask) + 1e-8
            mean = jnp.sum(masked_values) / count
            variance = jnp.sum(masked_values**2 * obs_mask) / count - mean**2
            std = jnp.sqrt(jnp.maximum(variance, 0.0))
            
            # Intervention statistics
            n_interventions = jnp.sum(interventions)
            intervention_rate = n_interventions / N
            
            # Create feature vector
            features = jnp.array([
                mean,
                std,
                intervention_rate,
                jnp.min(values),
                jnp.max(values),
                jnp.mean(values),  # Including intervened values
                jnp.std(values)
            ])
            
            # Pad to hidden_dim
            x = jnp.pad(features, (0, self.hidden_dim - len(features)))
            
            # Process through shared MLP
            for i, layer in enumerate(layers):
                x = layer(x)
                if i < len(layers) - 1:  # Apply ReLU to all but last layer
                    x = jax.nn.relu(x)
            
            return x
        
        # Apply to all variables in parallel
        # Transpose to [d, N, 3] for vmap over first dimension
        data_transposed = jnp.transpose(data, (1, 0, 2))
        node_embeddings = jax.vmap(process_single_variable)(data_transposed)  # [d, hidden_dim]
        
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
                 ) -> dict[str, jnp.ndarray]:  # Dictionary with embeddings and probabilities
        """
        Predict parent probabilities for target variable.
        
        Args:
            data: Intervention data [N, d, 3]
            target_variable: Index of target variable (0 <= target_variable < d)
            is_training: Whether model is in training mode
            
        Returns:
            Dictionary containing:
            - 'node_embeddings': Node representations [d, hidden_dim]
            - 'target_embedding': Target node representation [hidden_dim]
            - 'attention_logits': Raw attention scores [d]
            - 'parent_probabilities': Parent probabilities [d] (sum to 1.0, target has prob 0.0)
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
        
        return {
            'node_embeddings': node_embeddings,
            'target_embedding': target_embedding,
            'attention_logits': parent_logits,
            'parent_probabilities': parent_probs
        }
    
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