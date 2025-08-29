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
        
        # Normalize embeddings before processing (like AVICI)
        query_norm = query / (jnp.linalg.norm(query, axis=-1, keepdims=True) + 1e-8)
        keys_norm = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)
        
        # Use normalized embeddings
        query_expanded_norm = query_norm[:, None, :]  # [batch_size, 1, hidden_dim]
        query_tiled_norm = jnp.tile(query_expanded_norm, (1, n_vars, 1))  # [batch_size, n_vars, hidden_dim]
        
        # Combine normalized target and candidate embeddings
        combined_keys = jnp.concatenate([
            query_tiled_norm,  # [batch_size, n_vars, hidden_dim] - normalized target for each candidate
            keys_norm          # [batch_size, n_vars, hidden_dim] - normalized candidate embeddings
        ], axis=-1)  # [batch_size, n_vars, hidden_dim * 2]
        
        # Multi-layer scoring network with bias initialization
        score_net = hk.Sequential([
            hk.Linear(self.hidden_dim, w_init=self.w_init),
            jax.nn.relu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            hk.Linear(self.hidden_dim // 2, w_init=self.w_init),
            jax.nn.relu,
            hk.Linear(1, w_init=self.w_init, b_init=hk.initializers.Constant(-3.0))  # Sparse prior bias like AVICI
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