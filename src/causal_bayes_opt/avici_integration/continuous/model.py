"""
Continuous Parent Set Prediction Model.

This model uses:
- NodeFeatureEncoder for individual node representations (default)
- EnhancedParentAttentionLayer that computes pairwise features
- Clean separation between node encoding and relationship modeling

Key improvements over previous version:
- Avoids embedding collapse through separated concerns
- Better gradient flow and learning dynamics
- More diverse predictions
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional
import os

# Import our improved components
from .node_feature_encoder import NodeFeatureEncoder
from .parent_attention import ParentAttentionLayer
# Keep compatibility with other encoders
# from .deprecated.simple_working_encoder import SimpleWorkingEncoder
# from .deprecated.alternating_attention_encoder import AlternatingAttentionEncoder
# from .deprecated.data_aware_encoder import DataAwareEncoder
# from .deprecated.relational_encoder_v1 import RelationalEncoder


class ContinuousParentSetPredictionModel(hk.Module):
    """
    Improved parent set prediction model using enhanced architecture.
    
    Key improvements:
    - Encoder focuses on node features without inter-node attention
    - Parent attention layer uses pairwise statistical features
    - Cleaner separation of concerns
    """
    
    def __init__(self,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 key_size: int = 32,
                 dropout: float = 0.1,
                 encoder_type: str = "node_feature",
                 use_temperature_scaling: bool = False,
                 temperature_init: float = 0.0):
        super().__init__(name="ContinuousParentSetPrediction")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.key_size = key_size
        self.dropout = dropout
        self.encoder_type = encoder_type
        self.use_temperature_scaling = use_temperature_scaling
        self.temperature_init = temperature_init
    
    def __call__(self, 
                 data: jnp.ndarray,                    # [batch_size, N, d, 3] or [N, d, 3] intervention data
                 target_variable,                      # int or [batch_size] target variable index(es)
                 is_training: bool = True              # Training mode flag
                 ) -> dict[str, jnp.ndarray]:
        """
        Predict parent probabilities for target variable(s).
        
        Args:
            data: Intervention data [batch_size, N, d, 3] or [N, d, 3] where:
                  - [..., :, :, 0] = variable values (standardized)
                  - [..., :, :, 1] = target indicators (1 if target variable)
                  - [..., :, :, 2] = intervention indicators (1 if intervened)
            target_variable: Index of target variable (int) or indices [batch_size]
            is_training: Whether model is in training mode
            
        Returns:
            Dictionary containing:
            - 'node_embeddings': Node representations [batch_size, d, hidden_dim] or [d, hidden_dim]
            - 'target_embedding': Target node representation [batch_size, hidden_dim] or [hidden_dim]
            - 'attention_logits': Raw attention scores [batch_size, d] or [d]
            - 'parent_probabilities': Parent probabilities [batch_size, d] or [d]
        """
        # Handle both single and batch inputs
        is_batched = data.ndim == 4
        if is_batched:
            batch_size, N, d, channels = data.shape
            # Ensure target_variable is array for batch processing
            if isinstance(target_variable, int):
                target_variable = jnp.full(batch_size, target_variable)
        else:
            N, d, channels = data.shape
            # Add batch dimension for unified processing
            data = data[None, ...]  # [1, N, d, 3]
            batch_size = 1
            if isinstance(target_variable, int):
                target_variable = jnp.array([target_variable])
            else:
                target_variable = jnp.array(target_variable)[None]
        
        dropout_rate = self.dropout if is_training else 0.0
        
        # Select encoder based on encoder_type
        if self.encoder_type == "node_feature":
            encoder = NodeFeatureEncoder(
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                key_size=self.key_size,
                dropout_rate=dropout_rate,
                name="encoder"
            )
        elif self.encoder_type == "simple":
            encoder = SimpleWorkingEncoder(
                hidden_dim=self.hidden_dim,
                name="encoder"
            )
        elif self.encoder_type == "alternating":
            encoder = AlternatingAttentionEncoder(
                hidden_dim=self.hidden_dim,
                num_blocks=self.num_layers,
                num_heads=self.num_heads,
                dropout_rate=dropout_rate,
                name="encoder"
            )
        elif self.encoder_type == "data_aware":
            encoder = DataAwareEncoder(
                hidden_dim=self.hidden_dim,
                num_blocks=self.num_layers,
                num_heads=self.num_heads,
                dropout_rate=dropout_rate,
                name="encoder"
            )
        elif self.encoder_type == "relational":
            encoder = RelationalEncoder(
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout_rate=dropout_rate,
                name="encoder"
            )
        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")
        
        # Encode intervention data into node representations
        # Now encoder supports batch processing natively
        node_embeddings = encoder(data, is_training=is_training)  # [batch_size, d, hidden_dim] or [d, hidden_dim]
        
        # Ensure we have batch dimension for unified processing
        if not is_batched and node_embeddings.ndim == 2:
            node_embeddings = node_embeddings[None, ...]  # [1, d, hidden_dim]
        
        # Apply dropout for regularization
        if is_training:
            node_embeddings = hk.dropout(hk.next_rng_key(), dropout_rate, node_embeddings)
        
        # Get target node embeddings for each item in batch
        batch_indices = jnp.arange(batch_size)
        target_embedding = node_embeddings[batch_indices, target_variable]  # [batch_size, hidden_dim]
        
        # Extract raw values for computing pairwise features
        values = data[:, :, :, 0]  # [batch_size, N, d]
        
        # Compute parent attention scores using enhanced layer
        if self.encoder_type in ["node_feature", "simple"]:
            # Use enhanced attention with pairwise features
            parent_attention = ParentAttentionLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                key_size=self.key_size
            )
            parent_logits = parent_attention(
                target_embedding,     # [batch_size, hidden_dim]
                node_embeddings,      # [batch_size, d, hidden_dim] 
                values,               # [batch_size, N, d]
                target_variable       # [batch_size]
            )  # [batch_size, d]
        else:
            # Fallback to original attention for other encoders
            parent_attention = OriginalParentAttentionLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                key_size=self.key_size
            )
            parent_logits = parent_attention(target_embedding, node_embeddings)  # [batch_size, d]
        
        # Apply temperature scaling if enabled
        if self.use_temperature_scaling:
            # Learn a temperature parameter similar to AVICI
            temperature = hk.get_parameter(
                "temperature", 
                shape=(1,), 
                dtype=parent_logits.dtype,
                init=hk.initializers.Constant(self.temperature_init)
            )
            # Apply temperature scaling: logits * exp(temperature)
            # This allows the model to learn to sharpen or soften predictions
            parent_logits = parent_logits * jnp.exp(temperature)
        
        # Mask target variables (cannot be their own parent)
        target_mask = jnp.arange(d)[None, :] == target_variable[:, None]  # [batch_size, d]
        masked_logits = jnp.where(
            target_mask,
            -1e9,  # Large negative value for target variables
            parent_logits
        )
        
        
        
        # Convert to probabilities using sigmoid (NOT softmax!)
        # Each variable independently has a probability of being a parent
        parent_probs = jax.nn.sigmoid(masked_logits)  # [batch_size, d]
        
        # If input wasn't batched, remove batch dimension from outputs
        if not is_batched:
            node_embeddings = node_embeddings[0]      # [d, hidden_dim]
            target_embedding = target_embedding[0]    # [hidden_dim] 
            parent_logits = parent_logits[0]          # [d]
            parent_probs = parent_probs[0]            # [d]
        
        return {
            'node_embeddings': node_embeddings,
            'target_embedding': target_embedding,
            'attention_logits': parent_logits,
            'parent_probabilities': parent_probs
        }


class OriginalParentAttentionLayer(hk.Module):
    """Original attention layer for compatibility."""
    
    def __init__(self, 
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 key_size: int = 32,
                 name: str = "ParentAttentionLayer"):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.key_size = key_size
        self.w_init = hk.initializers.Orthogonal()
    
    def __call__(self, 
                 query: jnp.ndarray,      # [hidden_dim] - target node embedding
                 key_value: jnp.ndarray   # [n_vars, hidden_dim] - all node embeddings
                 ) -> jnp.ndarray:        # [n_vars] - parent attention scores
        """Compute attention scores between target node and all potential parents."""
        n_vars = key_value.shape[0]
        
        # Transform query and keys to attention space
        query_projection = hk.Linear(self.key_size, w_init=self.w_init, name="query_proj")
        key_projection = hk.Linear(self.key_size, w_init=self.w_init, name="key_proj")
        
        q = query_projection(query)  # [key_size]
        k = key_projection(key_value)  # [n_vars, key_size]
        
        # Compute attention scores
        scores = jnp.dot(k, q) / jnp.sqrt(self.key_size)  # [n_vars]
        
        return scores