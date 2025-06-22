"""
JAX-Native Vectorized Attention for Policy Networks

This module provides a complete rewrite of the alternating attention mechanism
using JAX vmap operations instead of Python loops. This enables true JAX compilation
and eliminates the critical performance blockers in the policy network.

Key innovations:
1. vmap-based attention processing (no for loops)
2. Vectorized variable and sample processing
3. Static shape operations for JAX compilation  
4. Identical functionality to original but JAX-native
"""

import warnings

warnings.warn(
    "This module is deprecated as of Phase 1.5. "
    "Use JAX vmap operations in jax_native module instead. "
    "See docs/migration/MIGRATION_GUIDE.md for migration instructions. "
    "This module will be removed on 2024-02-01.",
    DeprecationWarning,
    stacklevel=2
)


import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional


def layer_norm(*, axis, name=None):
    """Layer normalization - identical to original."""
    return hk.LayerNorm(axis=axis, create_scale=True, create_offset=True, name=name)


class VectorizedAlternatingAttentionEncoder(hk.Module):
    """
    JAX-native vectorized alternating attention encoder.
    
    This is a complete rewrite of AlternatingAttentionEncoder that eliminates
    all Python for loops and uses JAX vmap operations for true compilation.
    
    Performance improvements:
    - All operations are JAX-compilable
    - Vectorized processing over variables and samples
    - Static tensor shapes throughout
    - No dynamic loops that block compilation
    """
    
    def __init__(self,
                 num_layers: int = 4,
                 num_heads: int = 8, 
                 hidden_dim: int = 128,
                 dropout: float = 0.1,
                 name: str = "VectorizedAlternatingAttentionEncoder"):
        super().__init__(name=name)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
    
    def __call__(self,
                 history: jnp.ndarray,  # [MAX_HISTORY_SIZE, n_vars, 3]
                 is_training: bool = True) -> jnp.ndarray:
        """
        Apply vectorized alternating attention.
        
        Args:
            history: Fixed-size intervention history [MAX_HISTORY_SIZE, n_vars, 3]
            is_training: Training mode flag
            
        Returns:
            State embedding [n_vars, hidden_dim]
        """
        dropout_rate = self.dropout if is_training else 0.0
        
        # Input projection: [MAX_HISTORY_SIZE, n_vars, 3] -> [MAX_HISTORY_SIZE, n_vars, hidden_dim]
        x = hk.Linear(self.hidden_dim, w_init=self.w_init)(history)
        
        # Apply alternating attention layers using vectorized operations
        for layer_idx in range(self.num_layers):
            # Sample attention: each variable attends over its samples
            x = self._vectorized_sample_attention(x, dropout_rate, f"sample_attn_{layer_idx}")
            
            # Variable attention: each sample attends over variables  
            x = self._vectorized_variable_attention(x, dropout_rate, f"var_attn_{layer_idx}")
        
        # Final pooling: [MAX_HISTORY_SIZE, n_vars, hidden_dim] -> [n_vars, hidden_dim]
        state_embedding = jnp.max(x, axis=0)
        
        return state_embedding
    
    @hk.transparent
    def _vectorized_sample_attention(self,
                                   x: jnp.ndarray,  # [MAX_HISTORY_SIZE, n_vars, hidden_dim]
                                   dropout_rate: float,
                                   layer_name: str) -> jnp.ndarray:
        """
        Vectorized sample attention using vmap (no for loops).
        
        Each variable independently attends over its samples using vmap.
        """
        with hk.experimental.name_scope(layer_name):
            # x shape: [MAX_HISTORY_SIZE, n_vars, hidden_dim]
            # Transpose to: [n_vars, MAX_HISTORY_SIZE, hidden_dim] for vmap processing
            x_vars = jnp.transpose(x, (1, 0, 2))
            
            def process_single_variable(var_samples):
                """Process samples for a single variable."""
                # var_samples: [MAX_HISTORY_SIZE, hidden_dim]
                
                # Layer norm before attention
                var_samples_norm = layer_norm(axis=-1)(var_samples)
                
                # Multi-head self-attention over samples
                attn_output = hk.MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_size=self.hidden_dim // self.num_heads,
                    w_init_scale=2.0,
                    model_size=self.hidden_dim
                )(var_samples_norm, var_samples_norm, var_samples_norm)
                
                # Residual connection with dropout
                var_attended = var_samples + hk.dropout(hk.next_rng_key(), dropout_rate, attn_output)
                
                # Feed-forward network
                var_norm = layer_norm(axis=-1)(var_attended)
                var_ff = hk.Sequential([
                    hk.Linear(4 * self.hidden_dim, w_init=self.w_init),
                    jax.nn.relu,
                    hk.Linear(self.hidden_dim, w_init=self.w_init),
                ])(var_norm)
                
                # Final residual connection
                return var_attended + hk.dropout(hk.next_rng_key(), dropout_rate, var_ff)
            
            # Vectorize over variables dimension (axis=0)
            x_processed = jax.vmap(process_single_variable, in_axes=0, out_axes=0)(x_vars)
            
            # Transpose back: [n_vars, MAX_HISTORY_SIZE, hidden_dim] -> [MAX_HISTORY_SIZE, n_vars, hidden_dim]
            return jnp.transpose(x_processed, (1, 0, 2))
    
    @hk.transparent  
    def _vectorized_variable_attention(self,
                                     x: jnp.ndarray,  # [MAX_HISTORY_SIZE, n_vars, hidden_dim]
                                     dropout_rate: float,
                                     layer_name: str) -> jnp.ndarray:
        """
        Vectorized variable attention using vmap (no for loops).
        
        Each sample independently attends over variables using vmap.
        """
        with hk.experimental.name_scope(layer_name):
            # x shape: [MAX_HISTORY_SIZE, n_vars, hidden_dim]
            
            def process_single_sample(sample_vars):
                """Process variables for a single sample."""
                # sample_vars: [n_vars, hidden_dim]
                
                # Layer norm before attention
                sample_vars_norm = layer_norm(axis=-1)(sample_vars)
                
                # Multi-head self-attention over variables
                attn_output = hk.MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_size=self.hidden_dim // self.num_heads,
                    w_init_scale=2.0,
                    model_size=self.hidden_dim
                )(sample_vars_norm, sample_vars_norm, sample_vars_norm)
                
                # Residual connection with dropout
                sample_attended = sample_vars + hk.dropout(hk.next_rng_key(), dropout_rate, attn_output)
                
                # Feed-forward network
                sample_norm = layer_norm(axis=-1)(sample_attended)
                sample_ff = hk.Sequential([
                    hk.Linear(4 * self.hidden_dim, w_init=self.w_init),
                    jax.nn.relu,
                    hk.Linear(self.hidden_dim, w_init=self.w_init),
                ])(sample_norm)
                
                # Final residual connection
                return sample_attended + hk.dropout(hk.next_rng_key(), dropout_rate, sample_ff)
            
            # Vectorize over samples dimension (axis=0)
            return jax.vmap(process_single_sample, in_axes=0, out_axes=0)(x)


class VectorizedAcquisitionPolicyNetwork(hk.Module):
    """
    JAX-native vectorized policy network.
    
    Complete rewrite of AcquisitionPolicyNetwork using the vectorized attention
    encoder and tensor-based mechanism features. Eliminates all JAX compilation
    blockers while maintaining identical functionality.
    """
    
    def __init__(self,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 name: str = "VectorizedAcquisitionPolicyNetwork"):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
    
    def __call__(self, state_data: dict, is_training: bool = True) -> dict:
        """
        Vectorized forward pass for intervention selection.
        
        Args:
            state_data: Dictionary containing:
                - 'history': [MAX_HISTORY_SIZE, n_vars, 3] tensor
                - 'mechanism_features': MechanismFeaturesTensor  
                - 'marginal_probs': [n_vars] array
                - 'context_features': [n_vars, k] array
            is_training: Training mode flag
            
        Returns:
            Dictionary with policy outputs
        """
        # Extract pre-computed tensor data
        history = state_data['history']
        mechanism_features = state_data['mechanism_features']
        marginal_probs = state_data['marginal_probs']
        context_features = state_data['context_features']
        
        # Vectorized attention encoding
        encoder = VectorizedAlternatingAttentionEncoder(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        )
        
        # Get state embedding: [n_vars, hidden_dim]
        state_embedding = encoder(history, is_training)
        
        # Use tensor-based mechanism features (JAX-compiled)
        from .tensor_features import compute_mechanism_features_tensor
        mech_features = compute_mechanism_features_tensor(mechanism_features)  # [n_vars, 3]
        
        # Variable selection head (vectorized)
        variable_logits = self._vectorized_variable_selection(
            state_embedding, marginal_probs, context_features, mech_features
        )
        
        # Value selection head (vectorized)
        value_params = self._vectorized_value_selection(
            state_embedding, context_features, mech_features
        )
        
        # State value head (vectorized)
        state_value = self._vectorized_state_value(state_embedding, context_features)
        
        return {
            'variable_logits': variable_logits,
            'value_params': value_params,
            'state_value': state_value
        }
    
    @hk.transparent
    def _vectorized_variable_selection(self,
                                     state_emb: jnp.ndarray,      # [n_vars, hidden_dim]
                                     marginal_probs: jnp.ndarray, # [n_vars]
                                     context_features: jnp.ndarray, # [n_vars, k] 
                                     mech_features: jnp.ndarray   # [n_vars, 3]
                                     ) -> jnp.ndarray:
        """Vectorized variable selection head."""
        n_vars = state_emb.shape[0]
        
        # Compute uncertainty features (vectorized)
        uncertainty_features = 1.0 - 2.0 * jnp.abs(marginal_probs - 0.5)  # [n_vars]
        
        # Stack all features: [n_vars, hidden_dim + k + 3 + 1]
        all_features = jnp.concatenate([
            state_emb,                                    # [n_vars, hidden_dim]
            context_features,                             # [n_vars, k] 
            mech_features,                                # [n_vars, 3]
            uncertainty_features[:, None]                 # [n_vars, 1]
        ], axis=1)
        
        # MLP for variable selection (vectorized over all variables)
        x = hk.Linear(self.hidden_dim, w_init=self.w_init)(all_features)
        x = layer_norm(axis=-1)(x)
        x = jax.nn.relu(x)
        
        x = hk.Linear(self.hidden_dim // 2, w_init=self.w_init)(x)
        x = layer_norm(axis=-1)(x)
        x = jax.nn.relu(x)
        
        # Output logits [n_vars]
        variable_logits = hk.Linear(1, w_init=self.w_init)(x).squeeze(-1)
        
        return variable_logits
    
    @hk.transparent
    def _vectorized_value_selection(self,
                                   state_emb: jnp.ndarray,       # [n_vars, hidden_dim]
                                   context_features: jnp.ndarray, # [n_vars, k]
                                   mech_features: jnp.ndarray    # [n_vars, 3]
                                   ) -> jnp.ndarray:
        """Vectorized value selection head."""
        # Combine features: [n_vars, hidden_dim + k + 3]
        combined_features = jnp.concatenate([
            state_emb, context_features, mech_features
        ], axis=1)
        
        # MLP for value parameters (vectorized)
        x = hk.Linear(self.hidden_dim, w_init=self.w_init)(combined_features)
        x = layer_norm(axis=-1)(x) 
        x = jax.nn.relu(x)
        
        x = hk.Linear(self.hidden_dim // 2, w_init=self.w_init)(x)
        x = jax.nn.relu(x)
        
        # Output parameters [n_vars, 2] (mean, log_std)
        value_params = hk.Linear(2, w_init=self.w_init)(x)
        
        # Clip log_stds for stability
        means = value_params[:, 0]
        log_stds = jnp.clip(value_params[:, 1], -2.0, 2.0)
        
        return jnp.stack([means, log_stds], axis=1)
    
    @hk.transparent
    def _vectorized_state_value(self,
                              state_emb: jnp.ndarray,       # [n_vars, hidden_dim]
                              context_features: jnp.ndarray  # [n_vars, k]
                              ) -> jnp.ndarray:
        """Vectorized state value head."""
        # Global pooling over variables
        global_features = jnp.mean(state_emb, axis=0)  # [hidden_dim]
        
        # Add global context  
        global_context = jnp.mean(context_features, axis=0)  # [k]
        combined = jnp.concatenate([global_features, global_context])
        
        # MLP for state value
        x = hk.Linear(self.hidden_dim // 2, w_init=self.w_init)(combined)
        x = jax.nn.relu(x)
        
        x = hk.Linear(self.hidden_dim // 4, w_init=self.w_init)(x)
        x = jax.nn.relu(x)
        
        state_value = hk.Linear(1, w_init=self.w_init)(x).squeeze(-1)
        
        return state_value


# Factory function for backward compatibility
def create_vectorized_acquisition_policy(config, example_state_data):
    """Create vectorized policy network with JAX compilation."""
    
    def policy_fn(state_data, is_training=True):
        policy = VectorizedAcquisitionPolicyNetwork(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        return policy(state_data, is_training)
    
    return hk.transform(policy_fn)