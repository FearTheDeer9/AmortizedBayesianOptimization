"""
Parent Attention Layer - Uses pairwise features for parent prediction.

This layer computes relationship features between the target and potential parents,
combining node embeddings with pairwise statistical features.

Key features:
- Computes pairwise statistical features (correlation, lag correlation, etc.)
- Combines node embeddings with relationship features
- Uses these rich features to predict parent relationships
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional, Tuple



class ParentAttentionLayer(hk.Module):
    """
    Attention layer that uses both node embeddings and pairwise features.
    
    Key improvements:
    - Computes pairwise statistical features (correlation, lag correlation, etc.)
    - Combines node embeddings with relationship features
    - Uses these rich features to predict parent relationships
    """
    
    def __init__(self, 
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 key_size: int = 32,
                 name: str = "ParentAttentionLayer"):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.key_size = key_size
        self.w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
    
    def compute_pairwise_features(self,
                                 values: jnp.ndarray,
                                 target_idx: int) -> jnp.ndarray:
        """
        Compute statistical features between target and all variables.
        
        Args:
            values: [N, d] observation values
            target_idx: Index of target variable
            
        Returns:
            features: [d, num_features] pairwise features
        """
        N, d = values.shape
        target_values = values[:, target_idx]
        
        features_list = []
        
        for j in range(d):
            if j == target_idx:
                # Self-features
                features = jnp.array([
                    1.0,  # Is target indicator
                    0.0,  # Correlation (self)
                    0.0,  # Lag correlation
                    0.0,  # Reverse lag correlation
                    0.0,  # Conditional variance ratio
                    0.0,  # Mutual information proxy
                ])
            else:
                # Compute relationship features
                other_values = values[:, j]
                
                # 1. Standard correlation
                corr = jnp.corrcoef(target_values, other_values)[0, 1]
                
                # 2. Lag correlation (does j predict target?)
                lag_corr = jnp.corrcoef(other_values[:-1], target_values[1:])[0, 1]
                
                # 3. Reverse lag correlation (does target predict j?)
                rev_lag_corr = jnp.corrcoef(target_values[:-1], other_values[1:])[0, 1]
                
                # 4. Conditional variance ratio
                # If j is parent of target, var(target|j) < var(target)
                var_target = jnp.var(target_values)
                # Simple linear regression coefficient
                coef = corr * jnp.std(target_values) / (jnp.std(other_values) + 1e-8)
                residual = target_values - coef * (other_values - jnp.mean(other_values))
                var_target_given_j = jnp.var(residual)
                var_ratio = var_target_given_j / (var_target + 1e-8)
                
                # 5. Mutual information proxy (using correlation)
                # MI ≈ -0.5 * log(1 - corr^2) for Gaussian
                mi_proxy = -0.5 * jnp.log(1 - corr**2 + 1e-8)
                
                features = jnp.array([
                    0.0,  # Not target
                    corr,
                    lag_corr,
                    rev_lag_corr,
                    var_ratio,
                    mi_proxy,
                ])
            
            # Handle NaN values
            features = jnp.nan_to_num(features, 0.0)
            features_list.append(features)
        
        return jnp.stack(features_list)  # [d, 6]
    
    def __call__(self, 
                 query: jnp.ndarray,        # [batch_size, hidden_dim] or [hidden_dim] - target node embedding(s)
                 keys: jnp.ndarray,         # [batch_size, n_vars, hidden_dim] or [n_vars, hidden_dim] - all node embeddings  
                 values_data: jnp.ndarray,  # [batch_size, N, d] or [N, d] - raw observation values
                 target_idx                 # [batch_size] or int - Index of target variable(s)
                 ) -> jnp.ndarray:          # [batch_size, n_vars] or [n_vars] - parent attention scores
        """
        Compute attention scores using both embeddings and pairwise features.
        
        Args:
            query: Target node embedding(s) [batch_size, hidden_dim] or [hidden_dim]
            keys: All node embeddings [batch_size, n_vars, hidden_dim] or [n_vars, hidden_dim]
            values_data: Raw observation values [batch_size, N, d] or [N, d]
            target_idx: Index of target variable(s) [batch_size] or int
            
        Returns:
            Parent attention logits [batch_size, n_vars] or [n_vars]
        """
        # Handle both single and batch inputs
        is_batched = keys.ndim == 3
        if is_batched:
            batch_size, n_vars, _ = keys.shape
            if isinstance(target_idx, int):
                target_idx = jnp.full(batch_size, target_idx)
        else:
            n_vars = keys.shape[0]
            # Add batch dimension for unified processing
            query = query[None, ...]
            keys = keys[None, ...]
            values_data = values_data[None, ...]
            target_idx = jnp.array([target_idx])
            batch_size = 1
        
        # 1. Compute pairwise statistical features - COMMENTED OUT TO TEST PURE NN
        # pairwise_features = self.compute_pairwise_features(values_data, target_idx)  # [d, 6]
        
        # 2. Project pairwise features to embedding space - COMMENTED OUT
        # feature_projection = hk.Linear(
        #     self.hidden_dim // 2,
        #     w_init=self.w_init,
        #     name="feature_projection"
        # )
        # projected_features = feature_projection(pairwise_features)  # [d, hidden_dim//2]
        
        # 3. Combine with node embeddings - SIMPLIFIED TO JUST USE EMBEDDINGS
        # Create combined representation for each potential parent
        # For batched: query=[batch_size, hidden_dim], keys=[batch_size, n_vars, hidden_dim]
        query_expanded = query[:, None, :]  # [batch_size, 1, hidden_dim]
        query_tiled = jnp.tile(query_expanded, (1, n_vars, 1))  # [batch_size, n_vars, hidden_dim]
        
        # Combine target and candidate embeddings
        combined_keys = jnp.concatenate([
            query_tiled,  # [batch_size, n_vars, hidden_dim] - target for each candidate
            keys          # [batch_size, n_vars, hidden_dim] - candidate embeddings
        ], axis=-1)  # [batch_size, n_vars, hidden_dim * 2]
        
        # 4. Multi-layer scoring network
        # This learns to predict parent probability from combined features
        score_net = hk.Sequential([
            hk.Linear(self.hidden_dim, w_init=self.w_init),
            jax.nn.relu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            hk.Linear(self.hidden_dim // 2, w_init=self.w_init),
            jax.nn.relu,
            hk.Linear(1, w_init=self.w_init)  # Single score per variable
        ], name="score_network")
        
        scores = score_net(combined_keys).squeeze(-1)  # [batch_size, n_vars]
        
        # 5. Add direct feature influence - COMMENTED OUT TO TEST PURE NN
        # Some features are strong indicators (e.g., high lag correlation)
        # feature_direct = hk.Linear(
        #     1,
        #     w_init=self.w_init,
        #     with_bias=False,
        #     name="feature_direct"
        # )
        
        # Use most informative features directly - REMOVED
        # direct_features = pairwise_features[:, [1, 2, 4]]  # corr, lag_corr, var_ratio
        # direct_scores = feature_direct(direct_features).squeeze(-1)  # [n_vars]
        
        # Combine learned scores with direct feature scores - JUST USE SCORES
        # final_scores = scores + 0.5 * direct_scores
        final_scores = scores  # Pure NN output, no statistical feature hacks
        
        # Remove batch dimension if input wasn't batched
        if not is_batched:
            final_scores = final_scores[0]  # [n_vars]
        
        # DEBUG: Check 2-variable attention computation
        # if n_vars == 2:
        #     import pdb
        #     pdb.set_trace()
        #     # When breakpoint hits, investigate:
        #     # (Pdb) p pairwise_features
        #     # (Pdb) p scores
        #     # (Pdb) p direct_scores
        #     # (Pdb) p final_scores
        #     # (Pdb) p jnp.std(final_scores)
        #     # (Pdb) p combined_keys.shape
        #     # (Pdb) c
        
        return final_scores