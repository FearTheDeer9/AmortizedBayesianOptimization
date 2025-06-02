"""
Core parent set prediction model architecture.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Dict, List

from .enumeration import enumerate_possible_parent_sets, compute_adaptive_k
from .encoding import encode_parent_set, create_parent_set_indicators


def layer_norm(*, axis, name=None):
    """Layer normalization."""
    return hk.LayerNorm(axis=axis, create_scale=True, create_offset=True, name=name)


class ParentSetPredictionModel(hk.Module):
    """
    Model that predicts top-k parent sets for a target variable.
    
    Uses MLP for scoring instead of simple dot product.
    """
    
    def __init__(self,
                 layers=8,
                 dim=128,
                 key_size=32,
                 num_heads=8,
                 widening_factor=4,
                 dropout=0.1,
                 max_parent_size=3,
                 name="ParentSetPredictionModel"):
        """Initialize parent set prediction model."""
        super().__init__(name=name)
        self.dim = dim
        self.layers = 2 * layers
        self.dropout = dropout
        self.ln_axis = -1
        self.widening_factor = widening_factor
        self.num_heads = num_heads
        self.key_size = key_size
        self.max_parent_size = max_parent_size
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")

    @hk.transparent
    def _transformer_block(self, z, dropout_rate):
        """Transformer block - identical to AVICI."""
        # Multi-head attention
        q_in = layer_norm(axis=self.ln_axis)(z)
        k_in = layer_norm(axis=self.ln_axis)(z)
        v_in = layer_norm(axis=self.ln_axis)(z)
        z_attn = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.key_size,
            w_init_scale=2.0,
            model_size=self.dim,
        )(q_in, k_in, v_in)
        z = z + hk.dropout(hk.next_rng_key(), dropout_rate, z_attn)

        # Feed-forward network
        z_in = layer_norm(axis=self.ln_axis)(z)
        z_ffn = hk.Sequential([
            hk.Linear(self.widening_factor * self.dim, w_init=self.w_init),
            jax.nn.relu,
            hk.Linear(self.dim, w_init=self.w_init),
        ])(z_in)
        z = z + hk.dropout(hk.next_rng_key(), dropout_rate, z_ffn)
        return z

    @hk.transparent
    def _apply_transformer_layers(self, z, dropout_rate):
        """Apply transformer layers with swapping."""
        assert not self.layers % 2, "Number of layers must be even"

        for _ in range(self.layers):
            z = self._transformer_block(z, dropout_rate)
            z = jnp.swapaxes(z, -3, -2)

        z = layer_norm(axis=self.ln_axis)(z)
        z = jnp.max(z, axis=-3)  # [..., n_vars, dim]
        return z

    @hk.transparent  
    def _score_parent_set(self, target_embedding, parent_set_embedding, dropout_rate):
        """
        Use MLP to score target-parent compatibility instead of dot product.
        
        This gives the model more expressive power to learn complex relationships.
        """
        # Concatenate target and parent set embeddings
        combined = jnp.concatenate([target_embedding, parent_set_embedding], axis=-1)
        
        # MLP to compute compatibility score - apply dropout 
        # First layer
        x = hk.Linear(self.dim, w_init=self.w_init)(combined)
        x = jax.nn.relu(x)
        x = hk.dropout(hk.next_rng_key(), dropout_rate, x)
        
        # Second layer
        x = hk.Linear(self.dim // 2, w_init=self.w_init)(x)
        x = jax.nn.relu(x)
        
        # Output layer (no dropout on final output)
        score = hk.Linear(1, w_init=self.w_init)(x)
        return jnp.squeeze(score, axis=-1)  # Remove last dimension

    def __call__(self, x, variable_order, target_variable, is_training=True):
        """
        Forward pass to predict parent set probabilities.
        
        Args:
            x: Input data [N, d, 3]
            variable_order: List of variable names in order
            target_variable: Name of target variable
            is_training: Training mode flag
            
        Returns:
            Dictionary containing:
            - 'parent_set_logits': [k] logits for top-k parent sets
            - 'parent_sets': List of k parent sets
            - 'k': Number of parent sets
        """
        n_vars = len(variable_order)
        dropout_rate = self.dropout if is_training else 0.0
        
        # Extract target variable index
        target_idx = variable_order.index(target_variable)
        
        # Initial embedding
        z = hk.Linear(self.dim)(x)  # [N, d, 3] -> [N, d, dim]
        
        # Apply transformer layers
        variable_embeddings = self._apply_transformer_layers(z, dropout_rate)  # [d, dim]
        
        # Get target-specific embedding
        target_embedding = variable_embeddings[target_idx]  # [dim]
        
        # Enumerate possible parent sets
        possible_parent_sets = enumerate_possible_parent_sets(
            variable_order, target_variable, self.max_parent_size
        )
        
        # Compute adaptive k
        k = min(len(possible_parent_sets), compute_adaptive_k(n_vars, self.max_parent_size))
        
        # For each possible parent set, compute compatibility with target
        parent_set_logits = []
        
        for parent_set in possible_parent_sets:
            # Create indicators for this parent set
            indicators = create_parent_set_indicators(parent_set, variable_order, n_vars)
            
            # Encode parent set
            parent_set_emb = encode_parent_set(indicators, variable_embeddings)
            
            # IMPROVED: Use MLP scoring instead of dot product
            score = self._score_parent_set(target_embedding, parent_set_emb, dropout_rate)
            parent_set_logits.append(score)
        
        parent_set_logits = jnp.array(parent_set_logits)
        
        # Get top-k parent sets in descending order of logits
        if len(possible_parent_sets) > k:
            # Sort indices by logits in descending order
            sorted_indices = jnp.argsort(parent_set_logits)[::-1]  # Descending order
            top_k_indices = sorted_indices[:k]
            top_k_logits = parent_set_logits[top_k_indices]
            top_k_parent_sets = [possible_parent_sets[i] for i in top_k_indices]
        else:
            # Fewer parent sets than k - use all but sort by logits
            sorted_indices = jnp.argsort(parent_set_logits)[::-1]  # Descending order
            top_k_logits = parent_set_logits[sorted_indices]
            top_k_parent_sets = [possible_parent_sets[i] for i in sorted_indices]
            
            # Update k to actual number of parent sets
            k = len(possible_parent_sets)
        
        return {
            'parent_set_logits': top_k_logits,
            'parent_sets': top_k_parent_sets,
            'k': k,
            'all_possible_parent_sets': possible_parent_sets
        }


def create_parent_set_model(model_kwargs=None, max_parent_size=3):
    """
    Create a parent set prediction model.
    
    Args:
        model_kwargs: Model configuration
        max_parent_size: Maximum parent set size
        
    Returns:
        Transformed Haiku model
    """
    if model_kwargs is None:
        model_kwargs = {}
    
    def model_fn(x, variable_order, target_variable, is_training=True):
        model = ParentSetPredictionModel(
            max_parent_size=max_parent_size,
            **model_kwargs
        )
        return model(x, variable_order, target_variable, is_training)
    
    return hk.transform(model_fn)
