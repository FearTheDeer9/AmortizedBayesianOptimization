"""
Configurable Continuous Parent Set Prediction Model.

This module provides an enhanced version of ContinuousParentSetPredictionModel
that supports different encoder and attention architectures through configuration.

This allows for easy experimentation with different architectural choices
while maintaining backward compatibility.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional, Dict, Any

from .encoder_factory import create_encoder, create_attention_layer


class ConfigurableContinuousParentSetPredictionModel(hk.Module):
    """
    Configurable continuous parent set prediction model.
    
    This model extends the original ContinuousParentSetPredictionModel with:
    - Configurable encoder architecture
    - Configurable attention mechanism
    - Support for pairwise features
    - Backward compatibility with original API
    """
    
    def __init__(self,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 key_size: int = 32,
                 dropout: float = 0.1,
                 encoder_type: str = "node_feature",
                 attention_type: str = "pairwise",
                 encoder_config: Optional[Dict[str, Any]] = None,
                 attention_config: Optional[Dict[str, Any]] = None,
                 name: str = "ConfigurableContinuousParentSetPredictionModel"):
        """
        Initialize configurable model.
        
        Args:
            hidden_dim: Hidden dimension for embeddings
            num_layers: Number of layers in encoder
            num_heads: Number of attention heads
            key_size: Size of attention keys
            dropout: Dropout rate
            encoder_type: Type of encoder to use ("node_feature", "node", "simple", "improved")
            attention_type: Type of attention to use ("pairwise", "simple", "original")
            encoder_config: Additional encoder configuration
            attention_config: Additional attention configuration
            name: Module name
        """
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.key_size = key_size
        self.dropout = dropout
        self.encoder_type = encoder_type
        self.attention_type = attention_type
        
        # Prepare configurations
        self.encoder_config = encoder_config or {}
        self.encoder_config.update({
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout
        })
        
        self.attention_config = attention_config or {}
        self.attention_config.update({
            'hidden_dim': hidden_dim,
            'num_heads': num_heads,
            'key_size': key_size,
            'dropout': dropout
        })
    
    def __call__(self, 
                 data: jnp.ndarray,         # [N, d, 3] intervention data
                 target_variable: int,      # Target variable index
                 is_training: bool = True   # Training mode flag
                 ) -> Dict[str, jnp.ndarray]:
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
        
        dropout_rate = self.dropout if is_training else 0.0
        
        # Create and apply encoder
        encoder = create_encoder(self.encoder_type, self.encoder_config)
        node_embeddings = encoder(data, is_training)  # [d, hidden_dim]
        
        # Apply dropout for regularization
        if is_training and dropout_rate > 0:
            node_embeddings = hk.dropout(hk.next_rng_key(), dropout_rate, node_embeddings)
        
        # Get target node embedding
        target_embedding = node_embeddings[target_variable]  # [hidden_dim]
        
        # Create and apply attention layer
        attention_layer = create_attention_layer(self.attention_type, self.attention_config)
        
        # Check if attention layer supports pairwise features
        if self.attention_type == "pairwise":
            # Pass data for pairwise feature computation
            parent_logits = attention_layer(
                target_embedding, 
                node_embeddings,
                data=data,
                target_idx=target_variable,
                is_training=is_training
            )
        else:
            # Standard attention without pairwise features
            parent_logits = attention_layer(
                target_embedding,
                node_embeddings
            )
        
        # Mask target variable (cannot be its own parent)
        mask = jnp.ones(d)
        masked_logits = jnp.where(
            jnp.arange(d) == target_variable,
            -1e9,  # Large negative value for target variable
            parent_logits
        )
        
        # Convert to probabilities using masked softmax
        parent_probs = jax.nn.softmax(masked_logits)  # [d]
        
        return {
            'node_embeddings': node_embeddings,
            'target_embedding': target_embedding,
            'attention_logits': parent_logits,
            'parent_probabilities': parent_probs,
            'encoder_type': self.encoder_type,
            'attention_type': self.attention_type
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
        Extract top-k parent variables.
        
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


def create_model_with_config(model_config: Dict[str, Any]) -> ConfigurableContinuousParentSetPredictionModel:
    """
    Create a configurable model from a configuration dictionary.
    
    Args:
        model_config: Configuration dictionary with keys:
            - encoder_type: Type of encoder (default: "node_feature")
            - attention_type: Type of attention (default: "pairwise")
            - hidden_dim: Hidden dimension (default: 128)
            - num_layers: Number of layers (default: 4)
            - num_heads: Number of heads (default: 8)
            - key_size: Key size (default: 32)
            - dropout: Dropout rate (default: 0.1)
            
    Returns:
        Configured model instance
    """
    return ConfigurableContinuousParentSetPredictionModel(
        hidden_dim=model_config.get('hidden_dim', 128),
        num_layers=model_config.get('num_layers', 4),
        num_heads=model_config.get('num_heads', 8),
        key_size=model_config.get('key_size', 32),
        dropout=model_config.get('dropout', 0.1),
        encoder_type=model_config.get('encoder_type', 'node_feature'),
        attention_type=model_config.get('attention_type', 'pairwise'),
        encoder_config=model_config.get('encoder_config'),
        attention_config=model_config.get('attention_config')
    )


# Backward compatibility: Create a version that matches the original interface
def create_continuous_parent_set_model(hidden_dim: int = 128,
                                     num_layers: int = 4,
                                     num_heads: int = 8,
                                     key_size: int = 32,
                                     dropout: float = 0.1,
                                     encoder_type: str = "node_feature") -> ConfigurableContinuousParentSetPredictionModel:
    """
    Create a continuous parent set model with the specified encoder.
    
    This function provides backward compatibility with the original API
    while allowing encoder selection.
    """
    # Auto-select compatible attention type
    attention_type = "pairwise" if encoder_type == "node_feature" else "original"
    
    return ConfigurableContinuousParentSetPredictionModel(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        key_size=key_size,
        dropout=dropout,
        encoder_type=encoder_type,
        attention_type=attention_type
    )