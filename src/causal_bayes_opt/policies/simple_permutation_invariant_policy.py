#!/usr/bin/env python3
"""
Simplified permutation-invariant policy without feature extraction.

This removes the channel statistics extraction to test if simpler is better.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Dict, Callable


def create_simple_permutation_invariant_policy(
    hidden_dim: int = 256,
    use_fixed_std: bool = True,
    fixed_std: float = 0.5
) -> Callable:
    """
    Create simplified permutation-invariant policy.
    
    Key differences from original:
    1. NO channel statistics extraction
    2. NO gating mechanism
    3. Simpler direct processing
    4. Option for fixed std instead of learned
    
    Args:
        hidden_dim: Hidden dimension size
        use_fixed_std: If True, use fixed std for exploration
        fixed_std: Fixed std value if use_fixed_std=True
    """
    
    def policy_fn(tensor_input: jnp.ndarray, target_idx: int = 0) -> Dict[str, jnp.ndarray]:
        """
        Simplified permutation-invariant policy network.
        
        Args:
            tensor_input: [T, n_vars, C] tensor where C is 3 or 5
            target_idx: Index of target variable to mask
            
        Returns:
            Dictionary with variable_logits and value_params
        """
        T, n_vars, n_channels = tensor_input.shape
        
        # Handle different channel inputs and normalize to 4 channels
        if n_channels == 3:
            # No surrogate case - pad channel 3 with uniform 0.5 parent probs
            padded = jnp.zeros((T, n_vars, 4))
            padded = padded.at[:, :, :3].set(tensor_input)
            padded = padded.at[:, :, 3].set(0.5)  # Uniform parent probability
            tensor_input = padded
        elif n_channels == 5:
            # Has surrogate - drop duplicative 5th channel (intervention_recency)
            # Channel 4 is redundant with channel 2 (intervened_on_indicator)
            tensor_input = tensor_input[:, :, :4]
        elif n_channels != 4:
            raise ValueError(f"Expected 3, 4, or 5 channels, got {n_channels}")
        
        # Simple initial projection (no statistics) - now always 4 channels
        x_flat = tensor_input.reshape(-1, 4)
        x_proj = hk.Linear(hidden_dim, name="input_projection")(x_flat)
        x = x_proj.reshape(T, n_vars, hidden_dim)
        
        # Alternating attention layers with proper pre-norm
        num_layers = 4
        num_heads = 4
        
        for layer_idx in range(num_layers):
            if layer_idx % 2 == 0:
                # Attention over time dimension (for each variable)
                x = _simple_sample_attention(
                    x, num_heads, hidden_dim, f"time_attn_{layer_idx}"
                )
            else:
                # Attention over variables dimension (for each timestep)
                x = _simple_variable_attention(
                    x, num_heads, hidden_dim, f"var_attn_{layer_idx}"
                )
        
        # Simple mean pooling over time
        x_pooled = jnp.mean(x, axis=0)  # [n_vars, hidden_dim]
        
        # Final processing
        x_final = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True,
            name="output_norm"
        )(x_pooled)
        
        # Variable selection head
        var_hidden = hk.Sequential([
            hk.Linear(hidden_dim // 2, name="var_mlp_1"),
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
def _simple_sample_attention(x, num_heads, hidden_dim, layer_name):
    """
    Simplified attention over samples dimension.
    """
    with hk.experimental.name_scope(layer_name):
        # Transpose for attention: [n_vars, T, hidden_dim]
        x_transposed = jnp.transpose(x, (1, 0, 2))
        
        def attend_single_var(var_samples):
            """Process one variable's samples."""
            # Store residual
            residual = var_samples
            
            # Pre-norm
            var_samples_norm = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True,
                name="sample_norm"
            )(var_samples)
            
            # Multi-head self-attention
            attn_output = hk.MultiHeadAttention(
                num_heads=num_heads,
                key_size=hidden_dim // num_heads,
                w_init_scale=1.0,
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
def _simple_variable_attention(x, num_heads, hidden_dim, layer_name):
    """
    Simplified attention over variables dimension.
    """
    with hk.experimental.name_scope(layer_name):
        def attend_single_timestep(timestep_vars):
            """Process one timestep's variables."""
            # Store residual
            residual = timestep_vars
            
            # Pre-norm
            timestep_norm = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True,
                name="var_norm"
            )(timestep_vars)
            
            # Multi-head self-attention
            attn_output = hk.MultiHeadAttention(
                num_heads=num_heads,
                key_size=hidden_dim // num_heads,
                w_init_scale=1.0,
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