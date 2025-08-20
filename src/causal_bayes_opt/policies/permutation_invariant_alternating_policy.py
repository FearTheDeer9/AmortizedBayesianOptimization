#!/usr/bin/env python3
"""
Permutation-invariant alternating attention policy.

This combines the alternating attention architecture with permutation invariance,
avoiding uniform embeddings by using channel statistics and proper normalization.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Dict, Callable


def create_permutation_invariant_alternating_policy(
    hidden_dim: int = 256,
    use_fixed_std: bool = False,
    fixed_std: float = 0.5
) -> Callable:
    """
    Create permutation-invariant alternating attention policy.
    
    Key features:
    1. Alternating attention over time and variables (like CAASL)
    2. NO positional encodings (maintains permutation invariance)
    3. Uses channel statistics to create distinguishing features
    4. Proper pre-norm architecture with correct residuals
    5. Lower initialization scales to prevent saturation
    """
    
    def policy_fn(tensor_input: jnp.ndarray, target_idx: int = 0) -> Dict[str, jnp.ndarray]:
        """
        Permutation-invariant alternating attention policy network.
        
        Args:
            tensor_input: [T, n_vars, C] tensor where C is 3 or 5
            target_idx: Index of target variable to mask
            
        Returns:
            Dictionary with variable_logits and value_params
        """
        T, n_vars, n_channels = tensor_input.shape
        
        # Handle both 3 and 5 channel inputs
        if n_channels == 3:
            padded = jnp.zeros((T, n_vars, 5))
            padded = padded.at[:, :, :3].set(tensor_input)
            tensor_input = padded
            n_channels = 5
        elif n_channels != 5:
            raise ValueError(f"Expected 3 or 5 channels, got {n_channels}")
        
        # Extract channel statistics to create distinguishing features
        # These are permutation-invariant within the variable dimension
        channel_mean = jnp.mean(tensor_input, axis=0)  # [n_vars, 5]
        channel_std = jnp.std(tensor_input, axis=0) + 1e-8  # [n_vars, 5]
        channel_max = jnp.max(tensor_input, axis=0)  # [n_vars, 5]
        channel_min = jnp.min(tensor_input, axis=0)  # [n_vars, 5]
        
        # Combine original features with statistics
        # This helps break symmetry without positional encoding
        stats_features = jnp.stack([channel_mean, channel_std, channel_max, channel_min], axis=0)  # [4, n_vars, 5]
        stats_features = stats_features.reshape(4, n_vars * 5).T  # [n_vars * 5, 4]
        stats_features = stats_features.reshape(n_vars, 20)  # [n_vars, 20]
        
        # Initial projection with channel-aware processing
        x_flat = tensor_input.reshape(-1, 5)
        x_proj = hk.Linear(hidden_dim, name="input_projection")(x_flat)
        x = x_proj.reshape(T, n_vars, hidden_dim)
        
        # Add processed statistics to each timestep
        stats_proj = hk.Linear(hidden_dim // 4, name="stats_projection")(stats_features)  # [n_vars, hidden_dim//4]
        stats_proj = stats_proj[None, :, :].repeat(T, axis=0)  # [T, n_vars, hidden_dim//4]
        
        # Combine with gating mechanism to control information flow
        gate = hk.Linear(hidden_dim, name="stats_gate")(
            jnp.concatenate([x[:, :, :hidden_dim//4], stats_proj], axis=-1)
        )
        gate = jax.nn.sigmoid(gate)
        x = x * gate  # Gated combination
        
        # Alternating attention layers with proper pre-norm
        num_layers = 4
        num_heads = 4
        
        for layer_idx in range(num_layers):
            if layer_idx % 2 == 0:
                # Attention over time dimension (for each variable)
                x = _permutation_invariant_sample_attention(
                    x, num_heads, hidden_dim, f"time_attn_{layer_idx}"
                )
            else:
                # Attention over variables dimension (for each timestep)
                x = _permutation_invariant_variable_attention(
                    x, num_heads, hidden_dim, f"var_attn_{layer_idx}"
                )
        
        # Attention-based pooling over time
        # Use self-attention to determine importance of each timestep
        temporal_query = hk.Linear(hidden_dim // 4, name="temporal_query")(x)  # [T, n_vars, hidden_dim//4]
        temporal_key = hk.Linear(hidden_dim // 4, name="temporal_key")(x)  # [T, n_vars, hidden_dim//4]
        
        # Average across variables for temporal attention
        temporal_query_avg = jnp.mean(temporal_query, axis=1)  # [T, hidden_dim//4]
        temporal_key_avg = jnp.mean(temporal_key, axis=1)  # [T, hidden_dim//4]
        
        # Compute attention scores
        scores = jnp.dot(temporal_query_avg, temporal_key_avg.T) / jnp.sqrt(hidden_dim // 4)  # [T, T]
        
        # Use last timestep query to attend to all timesteps
        final_scores = scores[-1, :]  # [T]
        attention_weights = jax.nn.softmax(final_scores)  # [T]
        
        # Weighted sum over time
        x_pooled = jnp.einsum('t,tvh->vh', attention_weights, x)  # [n_vars, hidden_dim]
        
        # Final processing with channel statistics
        combined = jnp.concatenate([x_pooled, stats_proj[0]], axis=-1)  # [n_vars, hidden_dim + hidden_dim//4]
        
        # Output heads with layer norm
        x_final = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                               name="output_norm")(combined)
        
        # Variable selection head
        var_hidden = hk.Sequential([
            hk.Linear(hidden_dim // 2, name="var_mlp_1"),
            jax.nn.gelu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="var_norm"),
            hk.Linear(hidden_dim // 4, name="var_mlp_2"),
            jax.nn.gelu,
            hk.Linear(1, name="var_mlp_output")
        ])(x_final)
        
        variable_logits = var_hidden.squeeze(-1)
        
        # Mask target variable
        variable_logits = jnp.where(
            jnp.arange(n_vars) == target_idx,
            -jnp.inf,
            variable_logits
        )
        
        # Value prediction head
        if use_fixed_std:
            # Only predict mean
            val_hidden = hk.Sequential([
                hk.Linear(hidden_dim // 2, name="val_mlp_1"),
                jax.nn.gelu,
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="val_norm"),
                hk.Linear(hidden_dim // 4, name="val_mlp_2"),
                jax.nn.gelu,
                hk.Linear(1, name="val_mlp_output",
                         w_init=hk.initializers.VarianceScaling(0.01, "fan_avg", "truncated_normal"))
            ])(x_final)
            
            # Expand to [n_vars, 2] with fixed std
            value_mean = val_hidden.squeeze(-1)
            value_log_std = jnp.ones(n_vars) * jnp.log(fixed_std)
            value_params = jnp.stack([value_mean, value_log_std], axis=-1)
        else:
            # Predict both mean and log_std
            val_hidden = hk.Sequential([
                hk.Linear(hidden_dim // 2, name="val_mlp_1"),
                jax.nn.gelu,
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="val_norm"),
                hk.Linear(hidden_dim // 4, name="val_mlp_2"),
                jax.nn.gelu,
                hk.Linear(2, name="val_mlp_output",
                         w_init=hk.initializers.VarianceScaling(0.01, "fan_avg", "truncated_normal"))
            ])(x_final)
            
            value_params = val_hidden  # [n_vars, 2]
        
        return {
            'variable_logits': variable_logits,
            'value_params': value_params
        }
    
    return policy_fn


@hk.transparent
def _permutation_invariant_sample_attention(x, num_heads, hidden_dim, layer_name):
    """
    Apply self-attention over samples dimension using vmap.
    Maintains permutation invariance by not using positional encodings.
    """
    with hk.experimental.name_scope(layer_name):
        # Transpose for attention: [n_vars, T, hidden_dim]
        x_transposed = jnp.transpose(x, (1, 0, 2))
        
        def attend_single_var(var_samples):
            """Process one variable's samples. var_samples: [T, hidden_dim]"""
            # Store residual
            residual = var_samples
            
            # Pre-norm
            var_samples_norm = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True,
                name="sample_norm"
            )(var_samples)
            
            # Multi-head self-attention with lower init scale
            attn_output = hk.MultiHeadAttention(
                num_heads=num_heads,
                key_size=hidden_dim // num_heads,
                w_init_scale=1.0,  # Lower than 2.0
                model_size=hidden_dim,
                name="mha"
            )(var_samples_norm, var_samples_norm, var_samples_norm)
            
            # Residual connection
            var_attended = residual + attn_output
            
            # Feed-forward network with pre-norm
            residual2 = var_attended
            var_norm = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True,
                name="ff_norm"
            )(var_attended)
            
            var_ff = hk.Sequential([
                hk.Linear(4 * hidden_dim, name="ff_up"),
                jax.nn.gelu,
                hk.Linear(hidden_dim, name="ff_down"),
            ], name="ff")(var_norm)
            
            # Final residual
            return residual2 + var_ff
        
        # Apply to all variables using vmap
        x_attended = jax.vmap(attend_single_var)(x_transposed)
        
        # Transpose back: [T, n_vars, hidden_dim]
        return jnp.transpose(x_attended, (1, 0, 2))


@hk.transparent
def _permutation_invariant_variable_attention(x, num_heads, hidden_dim, layer_name):
    """
    Apply self-attention over variables dimension using vmap.
    Maintains permutation invariance.
    """
    with hk.experimental.name_scope(layer_name):
        def attend_single_timestep(timestep_vars):
            """Process one timestep's variables. timestep_vars: [n_vars, hidden_dim]"""
            # Store residual
            residual = timestep_vars
            
            # Pre-norm
            timestep_norm = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True,
                name="var_norm"
            )(timestep_vars)
            
            # Multi-head self-attention with lower init scale
            attn_output = hk.MultiHeadAttention(
                num_heads=num_heads,
                key_size=hidden_dim // num_heads,
                w_init_scale=1.0,  # Lower than 2.0
                model_size=hidden_dim,
                name="mha"
            )(timestep_norm, timestep_norm, timestep_norm)
            
            # Residual connection
            timestep_attended = residual + attn_output
            
            # Feed-forward network with pre-norm
            residual2 = timestep_attended
            timestep_norm2 = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True,
                name="ff_norm"
            )(timestep_attended)
            
            timestep_ff = hk.Sequential([
                hk.Linear(4 * hidden_dim, name="ff_up"),
                jax.nn.gelu,
                hk.Linear(hidden_dim, name="ff_down"),
            ], name="ff")(timestep_norm2)
            
            # Final residual
            return residual2 + timestep_ff
        
        # Apply to all timesteps using vmap
        return jax.vmap(attend_single_timestep)(x)