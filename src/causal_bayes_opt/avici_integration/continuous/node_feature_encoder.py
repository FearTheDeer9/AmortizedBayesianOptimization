"""
Node Feature Encoder for BC Surrogate Model.

This encoder computes per-variable features WITHOUT inter-node attention,
which is crucial for preventing embedding collapse and maintaining prediction diversity.

Key Design Principles:
- No information mixing between nodes during encoding
- Each variable gets its own feature representation
- Features computed from the variable's own data only
- Maintains node identity throughout encoding
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import List, Optional


class NodeFeatureEncoder(hk.Module):
    """
    Encoder that computes per-variable features without cross-variable attention.
    
    This encoder addresses the uniformity issue by:
    1. Computing features independently for each variable
    2. Avoiding shared projections that lead to collapse
    3. Preserving variable-specific information
    4. Not mixing information between nodes during encoding
    """
    
    def __init__(self,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 name: str = "NodeFeatureEncoder"):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
    
    def __call__(self, 
                 data: jnp.ndarray,
                 is_training: bool = False) -> jnp.ndarray:
        """
        Encode intervention data into node representations.
        
        Args:
            data: Intervention data [N, d, 3] where:
                  [:, :, 0] = variable values
                  [:, :, 1] = intervention indicators
                  [:, :, 2] = target indicators (1 for target variable, 0 otherwise)
            is_training: Whether in training mode (for dropout)
                  
        Returns:
            Node embeddings [d, hidden_dim] - one per variable
        """
        N, d, channels = data.shape
        assert channels == 3, f"Expected 3 channels, got {channels}"
        
        # Process each variable independently to avoid collapse
        def encode_single_variable(var_data):
            """
            Encode a single variable's data into features.
            
            Args:
                var_data: Single variable data [N, 3]
                
            Returns:
                Variable embedding [hidden_dim]
            """
            # Extract channels
            values = var_data[:, 0]  # [N]
            interventions = var_data[:, 1]  # [N]
            target_indicator = var_data[:, 2]  # [N] - 1 if this is the target variable
            
            # Compute observational statistics (excluding interventions)
            # Note: target_indicator not used in feature computation
            obs_mask = (1 - interventions)
            n_obs = jnp.sum(obs_mask) + 1e-8  # Avoid division by zero
            
            # Masked statistics for observational data
            masked_values = jnp.where(obs_mask, values, 0.0)
            obs_mean = jnp.sum(masked_values) / n_obs
            obs_var = jnp.sum(masked_values**2 * obs_mask) / n_obs - obs_mean**2
            obs_std = jnp.sqrt(jnp.maximum(obs_var, 0.0))
            
            # Intervention statistics
            n_interventions = jnp.sum(interventions)
            intervention_rate = n_interventions / N
            
            # Value range statistics (from all data)
            value_min = jnp.min(values)
            value_max = jnp.max(values)
            value_range = value_max - value_min
            
            # Additional raw moments
            raw_mean = jnp.mean(values)  # Including interventions
            raw_std = jnp.std(values)
            
            # Higher-order statistics
            centered_values = values - raw_mean
            raw_skewness = jnp.mean(centered_values**3) / (raw_std**3 + 1e-8)
            raw_kurtosis = jnp.mean(centered_values**4) / (raw_std**4 + 1e-8) - 3.0
            
            # Percentiles for better distribution characterization
            percentile_25 = jnp.percentile(values, 25)
            percentile_50 = jnp.percentile(values, 50)  # median
            percentile_75 = jnp.percentile(values, 75)
            percentile_5 = jnp.percentile(values, 5)   # lower tail
            percentile_95 = jnp.percentile(values, 95)  # upper tail
            
            # Value dynamics (temporal patterns)
            if N > 1:
                value_diffs = values[1:] - values[:-1]
                mean_change = jnp.mean(value_diffs)
                volatility = jnp.std(value_diffs)
            else:
                mean_change = 0.0
                volatility = 0.0
            
            # Create feature vector with all statistics
            features = jnp.array([
                obs_mean,
                obs_std,
                intervention_rate,
                value_min,
                value_max,
                value_range,
                raw_mean,
                raw_std,
                raw_skewness,
                raw_kurtosis,
                n_interventions / 10.0,  # Normalized intervention count
                n_obs / N,  # Observation rate
                percentile_5,
                percentile_25,
                percentile_50,
                percentile_75,
                percentile_95,
                mean_change,
                volatility,
            ])
            
            # Project to hidden dimension
            # Use variable-specific MLP (no shared weights across variables)
            x = hk.Linear(self.hidden_dim, w_init=self.w_init, name="initial_projection")(features)
            x = jax.nn.gelu(x)
            
            # Apply dropout if training
            if is_training and self.dropout > 0:
                x = hk.dropout(hk.next_rng_key(), self.dropout, x)
            
            # Additional layers for more expressive features
            for i in range(self.num_layers - 1):
                residual = x
                x = hk.Linear(self.hidden_dim, w_init=self.w_init, name=f"layer_{i}")(x)
                x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=f"ln_{i}")(x)
                x = jax.nn.gelu(x)
                if is_training and self.dropout > 0:
                    x = hk.dropout(hk.next_rng_key(), self.dropout, x)
                # Residual connection
                x = x + residual
            
            # Final projection
            x = hk.Linear(self.hidden_dim, w_init=self.w_init, name="final_projection")(x)
            
            return x
        
        # Apply encoding to each variable independently using vmap
        # This ensures true variable-agnostic processing without information mixing
        data_transposed = jnp.transpose(data, (1, 0, 2))  # [d, N, 3]
        
        # Use vmap to apply the same encoding function to all variables
        # This shares parameters across variables but processes them independently
        node_embeddings = jax.vmap(encode_single_variable)(data_transposed)  # [d, hidden_dim]
        
        return node_embeddings


class SimpleNodeFeatureEncoder(hk.Module):
    """
    Simplified version of NodeFeatureEncoder for testing and comparison.
    
    This version uses shared parameters but still avoids cross-variable attention.
    """
    
    def __init__(self,
                 hidden_dim: int = 128,
                 name: str = "SimpleNodeFeatureEncoder"):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
    
    def __call__(self, 
                 data: jnp.ndarray,
                 is_training: bool = False) -> jnp.ndarray:
        """
        Simple encoding that still maintains variable separation.
        """
        N, d, channels = data.shape
        
        # Compute features for all variables at once
        values = data[:, :, 0]  # [N, d]
        interventions = data[:, :, 1]  # [N, d]
        target_indicators = data[:, :, 2]  # [N, d]
        
        # Observational mask (when not intervened)
        # Note: target_indicators not used in feature computation
        obs_mask = (1 - interventions)  # [N, d]
        
        # Per-variable statistics (computed along N axis)
        obs_means = jnp.sum(values * obs_mask, axis=0) / (jnp.sum(obs_mask, axis=0) + 1e-8)  # [d]
        intervention_rates = jnp.mean(interventions, axis=0)  # [d]
        value_ranges = jnp.ptp(values, axis=0)  # [d] - peak to peak
        
        # Stack features
        features = jnp.stack([
            obs_means,
            intervention_rates,
            value_ranges,
            jnp.mean(values, axis=0),  # Raw means
            jnp.std(values, axis=0),   # Raw stds
        ], axis=1)  # [d, 5]
        
        # Pad to hidden dimension
        padding = self.hidden_dim - features.shape[1]
        if padding > 0:
            features = jnp.pad(features, ((0, 0), (0, padding)))
        
        # Simple linear projection (shared across variables)
        embeddings = hk.Linear(self.hidden_dim, w_init=self.w_init)(features)  # [d, hidden_dim]
        embeddings = jax.nn.gelu(embeddings)
        
        return embeddings