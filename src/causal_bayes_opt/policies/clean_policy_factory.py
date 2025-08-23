"""
Shared policy definitions for clean ACBO implementation.

This module ensures that the same function definitions are used
for both training and inference, preventing Haiku module path mismatches.

Key insight: Haiku creates different module hierarchies based on WHERE
functions are defined, not just HOW they're structured. This factory
ensures consistent module paths across training and inference.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Dict, Any, Callable
# Import handling for permutation_invariant_alternating_policy
try:
    from .permutation_invariant_alternating_policy import create_permutation_invariant_alternating_policy
    from .simple_permutation_invariant_policy import create_simple_permutation_invariant_policy
except ImportError:
    # Fallback for when module is imported directly
    try:
        from permutation_invariant_alternating_policy import create_permutation_invariant_alternating_policy
        from simple_permutation_invariant_policy import create_simple_permutation_invariant_policy
    except ImportError:
        pass  # Will be handled when used


def create_clean_grpo_policy(
    hidden_dim: int = 256,
    architecture: str = "permutation_invariant",
    use_fixed_std: bool = False,
    fixed_std: float = 0.5
) -> Callable:
    """
    Create GRPO policy function with consistent module paths.
    
    This factory ensures the same function is used for both
    training and inference, preventing Haiku parameter mismatches.
    
    Args:
        hidden_dim: Hidden dimension for the network
        architecture: Architecture type - "simple", "attention", "alternating_attention", 
                    "permutation_invariant", or "simple_permutation_invariant"
        use_fixed_std: If True, use fixed std for value sampling
        fixed_std: Fixed std value if use_fixed_std=True
        
    Returns:
        Policy function that maps tensor inputs to action distributions
    """
    # IMPORTANT: "permutation_invariant" now maps to the simple version (4-channel standard)
    # The old alternating version is deprecated
    if architecture == "permutation_invariant":
        # Map to the new standard (simple_permutation_invariant)
        return create_simple_permutation_invariant_policy(hidden_dim, use_fixed_std, fixed_std)
    elif architecture == "simple_permutation_invariant":
        return create_simple_permutation_invariant_policy(hidden_dim, use_fixed_std, fixed_std)
    elif architecture == "deprecated_permutation_invariant_alternating":
        # Old 5-channel version - DEPRECATED, kept only for backward compatibility
        return create_permutation_invariant_alternating_policy(hidden_dim, use_fixed_std, fixed_std)
    elif architecture == "alternating_attention":
        return create_alternating_attention_policy(hidden_dim)
    elif architecture == "attention":
        return create_attention_policy(hidden_dim)
    elif architecture == "simple":
        return create_simple_policy(hidden_dim)
    elif architecture == "simple_mlp":
        return create_simple_mlp_policy(hidden_dim)
    elif architecture == "simplified_permutation_invariant":
        return create_simplified_permutation_invariant_policy(hidden_dim, use_fixed_std, fixed_std)
    elif architecture == "quantile":
        return create_quantile_policy(hidden_dim, use_fixed_std, fixed_std)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def create_simple_policy(hidden_dim: int = 256) -> Callable:
    """Create the simple MLP-based policy (current implementation)."""
    def policy_fn(tensor_input: jnp.ndarray, target_idx: int = 0) -> Dict[str, jnp.ndarray]:
        """
        GRPO policy network processing 5-channel tensor input.
        
        Args:
            tensor_input: [T, n_vars, C] tensor where C can be 3 or 5
            target_idx: Index of target variable to mask
            
        Returns:
            Dictionary with:
            - variable_logits: [n_vars] logits for variable selection
            - value_params: [n_vars, 2] mean and log_std for each variable
        """
        T, n_vars, n_channels = tensor_input.shape
        
        # Handle both 3 and 5 channel inputs
        if n_channels == 3:
            # Legacy 3-channel format - pad with zeros
            padded = jnp.zeros((T, n_vars, 5))
            padded = padded.at[:, :, :3].set(tensor_input)
            tensor_input = padded
            n_channels = 5
        elif n_channels != 5:
            raise ValueError(f"Expected 3 or 5 channels, got {n_channels}")
        
        # Project to hidden dimension (now from 5 channels)
        flat_input = tensor_input.reshape(T * n_vars, 5)
        x = hk.Linear(hidden_dim, name="input_projection")(flat_input)
        x = jax.nn.relu(x)
        
        # Reshape for temporal processing
        x = x.reshape(T, n_vars, hidden_dim)
        
        # Process each timestep independently
        def process_timestep(timestep_data):
            """Process single timestep with residual connection."""
            # Layer normalization
            x_norm = hk.LayerNorm(
                axis=-1, 
                create_scale=True, 
                create_offset=True, 
                name="timestep_norm"
            )(timestep_data)
            
            # MLP with expansion
            x_hidden = hk.Linear(hidden_dim * 2, name="timestep_hidden")(x_norm)
            x_hidden = jax.nn.relu(x_hidden)
            x_out = hk.Linear(hidden_dim, name="timestep_output")(x_hidden)
            
            # Residual connection
            return x_out + timestep_data
        
        # Process all timesteps
        x = jax.vmap(process_timestep)(x)
        
        # Aggregate over time (simple mean pooling)
        x_agg = jnp.mean(x, axis=0)  # [n_vars, hidden_dim]
        
        # Final layer norm before output heads
        x_agg = hk.LayerNorm(
            axis=-1,
            create_scale=True,
            create_offset=True,
            name="output_norm"
        )(x_agg)
        
        # Variable selection head
        variable_head = hk.Linear(1, name="variable_head")(x_agg)
        variable_logits = variable_head.squeeze(-1)  # [n_vars]
        
        # Mask out target variable (FIXED: use large negative instead of -inf)
        variable_logits = jnp.where(
            jnp.arange(n_vars) == target_idx,
            -1e10,  # Large negative instead of -inf to avoid infinite loss
            variable_logits
        )
        
        # Value prediction head (mean and log_std for each variable)
        value_head = hk.Linear(2, name="value_head")(x_agg)  # [n_vars, 2]
        
        return {
            'variable_logits': variable_logits,
            'value_params': value_head
        }
    
    return policy_fn


def create_simple_mlp_policy(hidden_dim: int = 256) -> Callable:
    """
    Create the simplest possible MLP policy for GRPO comparison.
    
    This architecture:
    1. Flattens all input to single vector (no temporal/structural assumptions)
    2. Uses standard MLP layers (no residuals, no layer norm complexity)
    3. Same input/output interface as existing policies
    4. Tests if architectural complexity is limiting learning
    
    Args:
        hidden_dim: Hidden dimension for MLP layers
        
    Returns:
        Policy function with same interface as create_clean_grpo_policy
    """
    def simple_mlp_policy_fn(tensor_input: jnp.ndarray, target_idx: int = 0) -> Dict[str, jnp.ndarray]:
        """
        Simple MLP policy network - no assumptions about structure.
        
        Args:
            tensor_input: [T, n_vars, C] tensor where C can be 3 or 5
            target_idx: Index of target variable to mask
            
        Returns:
            Dictionary with:
            - variable_logits: [n_vars] logits for variable selection
            - value_params: [n_vars, 2] mean and log_std for each variable
        """
        T, n_vars, n_channels = tensor_input.shape
        
        # Handle both 3 and 5 channel inputs (compatibility)
        if n_channels == 3:
            # Legacy 3-channel format - pad with zeros
            padded = jnp.zeros((T, n_vars, 5))
            padded = padded.at[:, :, :3].set(tensor_input)
            tensor_input = padded
            n_channels = 5
        elif n_channels != 5:
            raise ValueError(f"Expected 3 or 5 channels, got {n_channels}")
        
        # SIMPLE: Flatten everything to single vector
        # No temporal processing, no permutation invariance, no complex aggregation
        flat_input = tensor_input.flatten()  # [T * n_vars * 5]
        
        # Standard MLP stack - just learn the mapping
        x = hk.Linear(hidden_dim, name="mlp_layer1")(flat_input)
        x = jax.nn.relu(x)
        
        x = hk.Linear(hidden_dim, name="mlp_layer2")(x)
        x = jax.nn.relu(x)
        
        x = hk.Linear(hidden_dim, name="mlp_layer3")(x)
        x = jax.nn.relu(x)
        
        # Output heads - predict for each variable
        # Variable selection head
        var_logits = hk.Linear(n_vars, name="variable_output")(x)  # [n_vars]
        
        # Value prediction head  
        value_flat = hk.Linear(n_vars * 2, name="value_output")(x)  # [n_vars * 2]
        value_params = value_flat.reshape(n_vars, 2)  # [n_vars, 2]
        
        # Mask out target variable (FIXED: use large negative instead of -inf)
        variable_logits = jnp.where(
            jnp.arange(n_vars) == target_idx,
            -1e10,  # Large negative instead of -inf to avoid infinite loss
            var_logits
        )
        
        return {
            'variable_logits': variable_logits,
            'value_params': value_params
        }
    
    return simple_mlp_policy_fn


def create_simplified_permutation_invariant_policy(
    hidden_dim: int = 256,
    use_fixed_std: bool = True,
    fixed_std: float = 0.5
) -> Callable:
    """
    Create simplified version of permutation invariant policy.
    
    Based on simple_permutation_invariant_policy.py but without complex attention:
    1. Uses 4 channels (like current)
    2. Pads 3→4 with uniform 0.5 parent probs (like current) 
    3. SIMPLIFIED: No alternating attention layers
    4. SIMPLIFIED: Simple pooling instead of complex temporal processing
    5. Same output heads as current
    """
    def simplified_policy_fn(tensor_input: jnp.ndarray, target_idx: int = 0) -> Dict[str, jnp.ndarray]:
        """Simplified permutation invariant policy without complex attention."""
        T, n_vars, n_channels = tensor_input.shape
        
        # Handle channel conversion (same as current architecture)
        if n_channels == 3:
            # No surrogate case - pad channel 3 with uniform 0.5 parent probs
            padded = jnp.zeros((T, n_vars, 4))
            padded = padded.at[:, :, :3].set(tensor_input)
            padded = padded.at[:, :, 3].set(0.5)  # Uniform parent probability
            tensor_input = padded
        elif n_channels == 5:
            # Has surrogate - drop duplicative 5th channel
            tensor_input = tensor_input[:, :, :4]
        elif n_channels != 4:
            raise ValueError(f"Expected 3, 4, or 5 channels, got {n_channels}")
        
        # Simple initial projection (same as current)
        x_flat = tensor_input.reshape(-1, 4)
        x_proj = hk.Linear(hidden_dim, name="input_projection")(x_flat)
        x = x_proj.reshape(T, n_vars, hidden_dim)
        
        # SIMPLIFIED: Replace 4 alternating attention layers with simple processing
        # Option 1: Single attention layer + pooling
        x_attended = _simple_attention_layer(x, hidden_dim, "single_attention")
        
        # Simple mean pooling over time (like original)
        x_pooled = jnp.mean(x_attended, axis=0)  # [n_vars, hidden_dim]
        
        # Final processing (like original)
        x_final = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True,
            name="output_norm"
        )(x_pooled)
        
        # ORIGINAL GENERALIZATION PATTERN: Per-variable heads (works for any variable count)
        # Variable selection head - applies same function to each variable
        var_hidden = hk.Sequential([
            hk.Linear(hidden_dim // 2, name="var_mlp_1"),
            jax.nn.gelu,
            hk.Linear(1, name="var_mlp_output")  # ✅ Output size = 1 (not n_vars!)
        ])(x_final)
        
        variable_logits = var_hidden.squeeze(-1)  # [n_vars, 1] → [n_vars]
        
        # TODAY'S GRADIENT FIX: Proper target masking
        variable_logits = jnp.where(
            jnp.arange(n_vars) == target_idx,
            -1e10,  # Large negative instead of -inf to avoid infinite loss
            variable_logits
        )
        
        # Value prediction head: MEAN-ONLY (fixed std handled in training loop)
        val_hidden = hk.Sequential([
            hk.Linear(hidden_dim // 2, name="val_mlp_1"),
            jax.nn.gelu,
            hk.Linear(1, name="val_mlp_output",
                     w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal"))  # Larger init for exploration
        ])(x_final)
        
        value_means = val_hidden.squeeze(-1)  # [n_vars] - only means!
        
        # For interface compatibility, create value_params with fixed std
        if use_fixed_std:
            value_log_std = jnp.ones(n_vars) * jnp.log(fixed_std)
            value_params = jnp.stack([value_means, value_log_std], axis=-1)  # [n_vars, 2]
        else:
            # If not using fixed std, still only predict means (std becomes fixed default)
            default_std = 0.5
            value_log_std = jnp.ones(n_vars) * jnp.log(default_std)
            value_params = jnp.stack([value_means, value_log_std], axis=-1)  # [n_vars, 2]
        
        return {
            'variable_logits': variable_logits,
            'value_params': value_params
        }
    
    return simplified_policy_fn


@hk.transparent
def _simple_attention_layer(x, hidden_dim, layer_name):
    """Single attention layer based on working create_attention_policy pattern."""
    with hk.experimental.name_scope(layer_name):
        T, n_vars, _ = x.shape
        
        # Pre-norm (like working version)
        x_norm = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True,
            name="pre_norm"
        )(x)
        
        # Attention over time dimension (like working version)
        x_transposed = jnp.transpose(x_norm, (1, 0, 2))  # [n_vars, T, hidden_dim]
        
        def attend_single_var(var_sequence):
            """Attend over time for single variable (like working version)."""
            return hk.MultiHeadAttention(
                num_heads=4,
                key_size=hidden_dim // 4,
                model_size=hidden_dim,
                w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),  # ✅ PROPER INIT
                name="time_attention"
            )(var_sequence, var_sequence, var_sequence)
        
        # Apply attention to each variable
        attended_vars = jax.vmap(attend_single_var)(x_transposed)
        x_attended = jnp.transpose(attended_vars, (1, 0, 2))  # [T, n_vars, hidden_dim]
        
        # Add & norm (like working version)
        x_attended = x + x_attended
        
        # FFN layer (like working version)
        x_norm2 = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True,
            name="ffn_norm"
        )(x_attended)
        
        # Flatten for FFN
        x_flat = x_norm2.reshape(-1, hidden_dim)
        ffn_output = hk.Sequential([
            hk.Linear(hidden_dim * 4, name="ffn_up", 
                     w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")),
            jax.nn.gelu,
            hk.Linear(hidden_dim, name="ffn_down",
                     w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"))
        ])(x_flat)
        
        # Reshape and add residual
        x_ffn = ffn_output.reshape(T, n_vars, hidden_dim)
        return x_attended + x_ffn


def create_quantile_policy(
    hidden_dim: int = 256,
    use_fixed_std: bool = True,
    fixed_std: float = 0.5
) -> Callable:
    """
    Create quantile-based policy that unifies variable selection and value prediction.
    
    Brilliant architecture idea:
    - Output: [n_vars, 3] quantile scores (25%, 50%, 75% for each variable)
    - Selection: argmax across ALL n_vars×3 scores 
    - Value mapping: Selected quantile → percentile from buffer history
    - Gradient flow: Policy gradients handle non-differentiable selection perfectly
    
    Advantages:
    - Single head (no gradient conflicts)
    - Historical awareness (uses actual buffer statistics)
    - Adaptive strategy (learns optimal percentile choices)
    - Joint optimization (variable + value learned together)
    """
    
    def quantile_policy_fn(tensor_input: jnp.ndarray, target_idx: int = 0) -> Dict[str, jnp.ndarray]:
        """Quantile-based policy with unified variable and value prediction."""
        T, n_vars, n_channels = tensor_input.shape
        
        # Handle channel conversion (same as successful simplified policy)
        if n_channels == 3:
            # No surrogate - pad with uniform parent probs
            padded = jnp.zeros((T, n_vars, 4))
            padded = padded.at[:, :, :3].set(tensor_input)
            padded = padded.at[:, :, 3].set(0.5)
            tensor_input = padded
        elif n_channels == 5:
            # Drop 5th channel
            tensor_input = tensor_input[:, :, :4]
        elif n_channels != 4:
            raise ValueError(f"Expected 3, 4, or 5 channels, got {n_channels}")
        
        # Enhanced processing with alternating attention for sample/variable information sharing
        x_flat = tensor_input.reshape(-1, 4)
        x_proj = hk.Linear(hidden_dim, name="input_projection")(x_flat)
        x = x_proj.reshape(T, n_vars, hidden_dim)
        
        # ALTERNATING ATTENTION: Share information through both samples and variables
        num_layers = 4
        num_heads = 4
        
        for layer_idx in range(num_layers):
            if layer_idx % 2 == 0:
                # Attention over time dimension (for each variable) - learn sample patterns
                x = _sample_attention_layer_vmap(x, num_heads, hidden_dim, f"time_attn_{layer_idx}")
            else:
                # Attention over variables dimension (for each timestep) - learn variable relationships
                x = _variable_attention_layer_vmap(x, num_heads, hidden_dim, f"var_attn_{layer_idx}")
        
        # Advanced pooling with learned attention weights (like working attention policy)
        pooling_query = hk.get_parameter(
            "pooling_query", [1, hidden_dim],
            init=hk.initializers.TruncatedNormal()
        )
        pooling_query = jnp.broadcast_to(pooling_query, (n_vars, hidden_dim))
        
        # Compute attention weights over time
        scores = jnp.einsum('vh,tvh->tv', pooling_query, x) / jnp.sqrt(hidden_dim)
        attention_weights = jax.nn.softmax(scores, axis=0)  # [T, n_vars]
        
        # Weighted sum over time (preserves more information than mean)
        x_pooled = jnp.einsum('tv,tvh->vh', attention_weights, x)  # [n_vars, hidden_dim]
        
        # Final processing
        x_final = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True,
            name="output_norm"
        )(x_pooled)
        
        # UNIFIED QUANTILE HEAD: 3 scores per variable (25%, 50%, 75%)
        quantile_scores = hk.Sequential([
            hk.Linear(hidden_dim // 2, name="quantile_mlp_1"),
            jax.nn.gelu,
            hk.Linear(3, name="quantile_output")  # [n_vars, 3] - 3 quantile scores per variable
        ])(x_final)
        
        # Mask target variable (FIXED: use -infinity for true masking)
        target_mask = jnp.full(3, -jnp.inf)  # True masking to prevent any selection
        quantile_scores = quantile_scores.at[target_idx, :].set(target_mask)
        
        # For interface compatibility, create standard outputs
        # Variable logits: Use max quantile score per variable
        variable_logits = jnp.max(quantile_scores, axis=1)  # [n_vars]
        
        # Value params: Placeholder (will be filled by training loop based on quantile selection)
        value_params = jnp.zeros((n_vars, 2))
        
        return {
            'quantile_scores': quantile_scores,  # [n_vars, 3] - primary output
            'variable_logits': variable_logits,  # [n_vars] - compatibility
            'value_params': value_params         # [n_vars, 2] - compatibility placeholder
        }
    
    return quantile_policy_fn


def create_attention_policy(hidden_dim: int = 256) -> Callable:
    """Create the alternating attention-based policy."""
    def policy_fn(tensor_input: jnp.ndarray, target_idx: int = 0) -> Dict[str, jnp.ndarray]:
        """
        GRPO policy network with alternating attention over time and variables.
        
        This follows a simplified version of the CAASL architecture:
        - Alternates attention over timesteps and variables
        - Uses multi-head attention for richer representations
        - Better at capturing complex dependencies
        
        Args:
            tensor_input: [T, n_vars, 5] tensor
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
        
        # Initial projection
        x = tensor_input  # [T, n_vars, 5]
        
        # Project to hidden dimension
        x_flat = x.reshape(-1, 5)  # [T*n_vars, 5]
        x_proj = hk.Linear(hidden_dim, name="input_projection")(x_flat)
        x = x_proj.reshape(T, n_vars, hidden_dim)
        
        # Alternating attention layers
        num_layers = 4
        num_heads = 4
        
        for layer_idx in range(num_layers):
            # Layer norm before attention
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                            name=f"pre_norm_{layer_idx}")(x)
            
            if layer_idx % 2 == 0:
                # Attention over time dimension (for each variable)
                x_transposed = jnp.transpose(x, (1, 0, 2))  # [n_vars, T, hidden_dim]
                
                # Multi-head attention over time
                attn_output = hk.MultiHeadAttention(
                    num_heads=num_heads,
                    key_size=hidden_dim // num_heads,
                    model_size=hidden_dim,
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
                    name=f"time_attention_{layer_idx}"
                )(x_transposed, x_transposed, x_transposed)
                
                # Add & norm
                x_transposed = x_transposed + attn_output
                x = jnp.transpose(x_transposed, (1, 0, 2))  # Back to [T, n_vars, hidden_dim]
                
            else:
                # Attention over variables dimension (for each timestep)
                # Multi-head attention over variables
                attn_output = hk.MultiHeadAttention(
                    num_heads=num_heads,
                    key_size=hidden_dim // num_heads,
                    model_size=hidden_dim,
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
                    name=f"var_attention_{layer_idx}"
                )(x, x, x)
                
                # Add & norm
                x = x + attn_output
            
            # FFN after attention
            x_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                                 name=f"ffn_norm_{layer_idx}")(x)
            
            # Flatten for FFN
            x_flat = x_norm.reshape(-1, hidden_dim)
            ffn_output = hk.Sequential([
                hk.Linear(hidden_dim * 4, name=f"ffn_up_{layer_idx}"),
                jax.nn.gelu,
                hk.Linear(hidden_dim, name=f"ffn_down_{layer_idx}")
            ])(x_flat)
            
            # Reshape and residual
            ffn_output = ffn_output.reshape(T, n_vars, hidden_dim)
            x = x + ffn_output
        
        # Global pooling over time with learned weights
        # Use attention pooling instead of mean
        pooling_query = hk.get_parameter("pooling_query", [1, hidden_dim], 
                                        init=hk.initializers.TruncatedNormal())
        pooling_query = jnp.broadcast_to(pooling_query, (n_vars, hidden_dim))
        
        # Compute attention weights over time
        scores = jnp.einsum('vh,tvh->tv', pooling_query, x) / jnp.sqrt(hidden_dim)
        attention_weights = jax.nn.softmax(scores, axis=0)  # [T, n_vars]
        
        # Weighted sum over time
        x_pooled = jnp.einsum('tv,tvh->vh', attention_weights, x)  # [n_vars, hidden_dim]
        
        # Final layer norm
        x_pooled = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                               name="output_norm")(x_pooled)
        
        # Output heads with more capacity
        # Variable selection head with 2-layer MLP
        var_hidden = hk.Linear(hidden_dim // 2, name="var_mlp_hidden")(x_pooled)
        var_hidden = jax.nn.relu(var_hidden)
        variable_logits = hk.Linear(1, name="var_mlp_output")(var_hidden).squeeze(-1)
        
        # Mask target variable
        variable_logits = jnp.where(
            jnp.arange(n_vars) == target_idx,
            -jnp.inf,
            variable_logits
        )
        
        # Value prediction head with 2-layer MLP
        val_hidden = hk.Linear(hidden_dim // 2, name="val_mlp_hidden")(x_pooled)
        val_hidden = jax.nn.relu(val_hidden)
        value_params = hk.Linear(2, name="val_mlp_output")(val_hidden)
        
        return {
            'variable_logits': variable_logits,
            'value_params': value_params
        }
    
    return policy_fn


def create_alternating_attention_policy(hidden_dim: int = 256) -> Callable:
    """
    Create policy with alternating attention architecture from CAASL.
    
    This architecture alternates between attending over samples (time) and
    variables, which is particularly effective for causal discovery tasks.
    """
    def policy_fn(tensor_input: jnp.ndarray, target_idx: int = 0) -> Dict[str, jnp.ndarray]:
        """
        Alternating attention policy network.
        
        Args:
            tensor_input: [T, n_vars, C] tensor
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
        
        # Initial projection
        x_flat = tensor_input.reshape(-1, 5)
        x_proj = hk.Linear(hidden_dim, name="input_projection")(x_flat)
        x = x_proj.reshape(T, n_vars, hidden_dim)
        
        # Alternating attention layers
        num_layers = 4
        num_heads = 4
        
        for layer_idx in range(num_layers):
            if layer_idx % 2 == 0:
                # Attention over time dimension (for each variable)
                # Use vmap instead of for loop for JAX compatibility
                x = _sample_attention_layer_vmap(
                    x, num_heads, hidden_dim, f"time_attn_{layer_idx}"
                )
            else:
                # Attention over variables dimension (for each timestep)
                x = _variable_attention_layer_vmap(
                    x, num_heads, hidden_dim, f"var_attn_{layer_idx}"
                )
        
        # Pooling over time with learned attention weights
        pooling_query = hk.get_parameter(
            "pooling_query", [1, hidden_dim],
            init=hk.initializers.TruncatedNormal()
        )
        pooling_query = jnp.broadcast_to(pooling_query, (n_vars, hidden_dim))
        
        # Compute attention weights over time
        scores = jnp.einsum('vh,tvh->tv', pooling_query, x) / jnp.sqrt(hidden_dim)
        attention_weights = jax.nn.softmax(scores, axis=0)  # [T, n_vars]
        
        # Weighted sum over time
        x_pooled = jnp.einsum('tv,tvh->vh', attention_weights, x)  # [n_vars, hidden_dim]
        
        # Final layer norm
        x_pooled = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                               name="output_norm")(x_pooled)
        
        # Output heads with more capacity
        # Variable selection head
        var_hidden = hk.Linear(hidden_dim // 2, name="var_mlp_hidden")(x_pooled)
        var_hidden = jax.nn.relu(var_hidden)
        variable_logits = hk.Linear(1, name="var_mlp_output")(var_hidden).squeeze(-1)
        
        # Mask target variable
        variable_logits = jnp.where(
            jnp.arange(n_vars) == target_idx,
            -jnp.inf,
            variable_logits
        )
        
        # Value prediction head
        val_hidden = hk.Linear(hidden_dim // 2, name="val_mlp_hidden")(x_pooled)
        val_hidden = jax.nn.relu(val_hidden)
        value_params = hk.Linear(2, name="val_mlp_output")(val_hidden)
        
        return {
            'variable_logits': variable_logits,
            'value_params': value_params
        }
    
    return policy_fn


@hk.transparent
def _sample_attention_layer_vmap(x, num_heads, hidden_dim, layer_name):
    """
    Apply self-attention over samples dimension using vmap.
    
    Args:
        x: [T, n_vars, hidden_dim]
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension
        layer_name: Layer name for parameter scoping
        
    Returns:
        Updated x with same shape
    """
    with hk.experimental.name_scope(layer_name):
        # Transpose for attention: [n_vars, T, hidden_dim]
        x_transposed = jnp.transpose(x, (1, 0, 2))
        
        # Define single variable attention function
        def attend_single_var(var_samples):
            """Process one variable's samples. var_samples: [T, hidden_dim]"""
            # Pre-norm: Layer norm before attention
            var_samples_norm = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True,
                name="sample_norm"
            )(var_samples)
            
            # Multi-head self-attention
            attn_output = hk.MultiHeadAttention(
                num_heads=num_heads,
                key_size=hidden_dim // num_heads,
                w_init_scale=1.0,  # Reduced from 2.0
                model_size=hidden_dim,
                name="mha"
            )(var_samples_norm, var_samples_norm, var_samples_norm)
            
            # Residual connection (pre-norm: add to original input)
            var_attended = var_samples + attn_output
            
            # Feed-forward network
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
            return var_attended + var_ff
        
        # Apply to all variables using vmap
        x_attended = jax.vmap(attend_single_var)(x_transposed)
        
        # Transpose back: [T, n_vars, hidden_dim]
        return jnp.transpose(x_attended, (1, 0, 2))


@hk.transparent
def _variable_attention_layer_vmap(x, num_heads, hidden_dim, layer_name):
    """
    Apply self-attention over variables dimension using vmap.
    
    Args:
        x: [T, n_vars, hidden_dim]
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension
        layer_name: Layer name for parameter scoping
        
    Returns:
        Updated x with same shape
    """
    with hk.experimental.name_scope(layer_name):
        # Define single timestep attention function
        def attend_single_timestep(timestep_vars):
            """Process one timestep's variables. timestep_vars: [n_vars, hidden_dim]"""
            # Pre-norm: Layer norm before attention
            timestep_norm = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True,
                name="var_norm"
            )(timestep_vars)
            
            # Multi-head self-attention
            attn_output = hk.MultiHeadAttention(
                num_heads=num_heads,
                key_size=hidden_dim // num_heads,
                w_init_scale=1.0,  # Reduced from 2.0
                model_size=hidden_dim,
                name="mha"
            )(timestep_norm, timestep_norm, timestep_norm)
            
            # Residual connection (pre-norm: add to original input)
            timestep_attended = timestep_vars + attn_output
            
            # Feed-forward network
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
            return timestep_attended + timestep_ff
        
        # Apply to all timesteps using vmap
        return jax.vmap(attend_single_timestep)(x)


def verify_parameter_compatibility(
    saved_params: Dict[str, Any],
    model_fn: hk.Transformed,
    dummy_input: jnp.ndarray,
    target_idx: int = 0
) -> bool:
    """
    Verify that saved parameters match expected model structure.
    
    This is crucial for catching Haiku module path mismatches early.
    
    Args:
        saved_params: Parameters loaded from checkpoint
        model_fn: Haiku transformed function
        dummy_input: Example input for initialization
        target_idx: Target variable index
        
    Returns:
        True if parameters are compatible, False otherwise
    """
    import jax.tree_util as tree
    
    try:
        # Get expected parameter structure
        rng = jax.random.PRNGKey(0)
        expected_params = model_fn.init(rng, dummy_input, target_idx)
        
        # Get parameter keys
        saved_flat, saved_tree = tree.tree_flatten(saved_params)
        expected_flat, expected_tree = tree.tree_flatten(expected_params)
        
        # Extract keys (parameter paths)
        def get_keys(params):
            """Extract all parameter paths."""
            keys = []
            
            def traverse(path, node):
                if isinstance(node, dict):
                    for k, v in node.items():
                        traverse(path + "/" + k if path else k, v)
                else:
                    keys.append(path)
            
            traverse("", params)
            return set(keys)
        
        saved_keys = get_keys(saved_params)
        expected_keys = get_keys(expected_params)
        
        # Check compatibility
        if saved_keys != expected_keys:
            missing = expected_keys - saved_keys
            extra = saved_keys - expected_keys
            
            print("Parameter mismatch detected!")
            if missing:
                print(f"Missing parameters: {missing}")
            if extra:
                print(f"Extra parameters: {extra}")
            
            return False
        
        # Check shapes match
        for key in saved_keys:
            saved_shape = tree.tree_map(lambda x: x.shape, saved_params)
            expected_shape = tree.tree_map(lambda x: x.shape, expected_params)
            
            if saved_shape != expected_shape:
                print(f"Shape mismatch for {key}")
                return False
        
        print("Parameters are compatible!")
        return True
        
    except Exception as e:
        print(f"Error verifying parameters: {e}")
        return False


def create_parameter_migration_util() -> Dict[str, Callable]:
    """
    Create utilities for migrating parameters between different module structures.
    
    Returns:
        Dictionary of migration utilities
    """
    def extract_leaf_params(params: Dict) -> Dict[str, jnp.ndarray]:
        """Extract all leaf parameters with their paths."""
        leaves = {}
        
        def traverse(path, node):
            if isinstance(node, dict):
                for k, v in node.items():
                    new_path = f"{path}/{k}" if path else k
                    traverse(new_path, v)
            else:
                leaves[path] = node
        
        traverse("", params)
        return leaves
    
    def remap_parameters(
        old_params: Dict,
        old_to_new_mapping: Dict[str, str]
    ) -> Dict:
        """Remap parameters from old structure to new structure."""
        # Extract leaves
        old_leaves = extract_leaf_params(old_params)
        
        # Build new structure
        new_params = {}
        for old_path, new_path in old_to_new_mapping.items():
            if old_path in old_leaves:
                # Navigate and create nested structure
                parts = new_path.split('/')
                current = new_params
                
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                current[parts[-1]] = old_leaves[old_path]
        
        return new_params
    
    return {
        'extract_leaves': extract_leaf_params,
        'remap': remap_parameters
    }