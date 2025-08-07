"""
Parent Attention Layer for BC Surrogate Model.

This module implements attention mechanisms specifically designed for parent prediction,
using pairwise statistical features to capture parent-child relationships.

Key Design Principles:
- Attention computed ONLY between target and potential parents
- Uses relational features that capture statistical dependencies
- No shared transformations that could cause collapse
- Preserves the ability to distinguish between variables
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional, Tuple


class ParentAttentionLayer(hk.Module):
    """
    Attention layer that uses pairwise statistical features for parent prediction.
    
    This layer computes attention scores between a target variable and all potential
    parents using statistical dependency measures rather than just embedding similarity.
    """
    
    def __init__(self, 
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 key_size: int = 32,
                 use_pairwise_features: bool = True,
                 dropout: float = 0.1,
                 name: str = "ParentAttentionLayer"):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.key_size = key_size
        self.use_pairwise_features = use_pairwise_features
        self.dropout = dropout
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
    
    def compute_pairwise_features(self,
                                 target_data: jnp.ndarray,
                                 candidate_data: jnp.ndarray) -> jnp.ndarray:
        """
        Compute pairwise statistical features between target and candidate parent.
        
        Args:
            target_data: Target variable data [N, 3]
            candidate_data: Candidate parent data [N, 3]
            
        Returns:
            Pairwise features [feature_dim]
        """
        # Extract values and masks
        target_values = target_data[:, 0]
        target_interventions = target_data[:, 1]
        target_indicators = target_data[:, 2]  # 1 if this is the target variable
        
        candidate_values = candidate_data[:, 0]
        candidate_interventions = candidate_data[:, 1]
        candidate_indicators = candidate_data[:, 2]  # 1 if this is the target variable
        
        # Compute mask for when both are observed (not intervened)
        # Note: target_indicators not used here - they indicate which variable is the prediction target
        both_observed = ((1 - target_interventions) * (1 - candidate_interventions))
        n_both_obs = jnp.sum(both_observed) + 1e-8
        
        # Correlation coefficient
        target_mean = jnp.sum(target_values * both_observed) / n_both_obs
        candidate_mean = jnp.sum(candidate_values * both_observed) / n_both_obs
        
        target_centered = (target_values - target_mean) * both_observed
        candidate_centered = (candidate_values - candidate_mean) * both_observed
        
        covariance = jnp.sum(target_centered * candidate_centered) / n_both_obs
        target_std = jnp.sqrt(jnp.sum(target_centered**2) / n_both_obs + 1e-8)
        candidate_std = jnp.sqrt(jnp.sum(candidate_centered**2) / n_both_obs + 1e-8)
        
        correlation = covariance / (target_std * candidate_std + 1e-8)
        
        # Mutual information approximation (using binning)
        # Simple approximation based on correlation for continuous variables
        # MI ≈ -0.5 * log(1 - correlation^2) for Gaussian variables
        mi_approx = -0.5 * jnp.log(1 - jnp.minimum(correlation**2, 0.99) + 1e-8)
        
        # Value range overlap
        target_min, target_max = jnp.min(target_values), jnp.max(target_values)
        candidate_min, candidate_max = jnp.min(candidate_values), jnp.max(candidate_values)
        
        overlap_min = jnp.maximum(target_min, candidate_min)
        overlap_max = jnp.minimum(target_max, candidate_max)
        overlap_ratio = jnp.maximum(0.0, (overlap_max - overlap_min) / 
                                   (jnp.maximum(target_max - target_min, 
                                               candidate_max - candidate_min) + 1e-8))
        
        # Lag correlations (important for causal relationships)
        N_lag = len(target_values) - 1
        if N_lag > 0:
            # Lag correlation: does candidate predict target?
            lag_corr_matrix = jnp.corrcoef(candidate_values[:-1], target_values[1:])
            lag_corr = jnp.nan_to_num(lag_corr_matrix[0, 1], 0.0)
            # Reverse lag correlation: does target predict candidate?
            rev_lag_corr_matrix = jnp.corrcoef(target_values[:-1], candidate_values[1:])
            rev_lag_corr = jnp.nan_to_num(rev_lag_corr_matrix[0, 1], 0.0)
        else:
            lag_corr = 0.0
            rev_lag_corr = 0.0
        
        # Conditional variance ratio
        # If candidate is parent of target, var(target|candidate) < var(target)
        var_target = jnp.var(target_values)
        # Simple linear regression coefficient
        coef = correlation * jnp.std(target_values) / (jnp.std(candidate_values) + 1e-8)
        residual = target_values - coef * (candidate_values - jnp.mean(candidate_values))
        var_target_given_candidate = jnp.var(residual)
        var_ratio = var_target_given_candidate / (var_target + 1e-8)
        
        # Intervention consistency
        # Check if intervening on candidate affects target
        candidate_intervened = candidate_interventions > 0.5
        target_when_candidate_intervened = jnp.sum(target_values * candidate_intervened) / (jnp.sum(candidate_intervened) + 1e-8)
        target_when_candidate_not_intervened = jnp.sum(target_values * (1 - candidate_intervened)) / (jnp.sum(1 - candidate_intervened) + 1e-8)
        intervention_effect = jnp.abs(target_when_candidate_intervened - target_when_candidate_not_intervened)
        
        # Additional statistical measures
        # Rank correlation approximation
        target_ranks = jnp.argsort(jnp.argsort(target_values))
        candidate_ranks = jnp.argsort(jnp.argsort(candidate_values))
        rank_correlation = jnp.corrcoef(target_ranks, candidate_ranks)[0, 1]
        
        # Create feature vector
        features = jnp.array([
            correlation,
            jnp.abs(correlation),  # Absolute correlation
            correlation**2,  # R-squared
            mi_approx,
            overlap_ratio,
            intervention_effect,
            rank_correlation,
            jnp.sign(correlation),  # Direction of relationship
            lag_corr,  # Lag correlation (candidate → target)
            rev_lag_corr,  # Reverse lag correlation (target → candidate)
            var_ratio,  # Conditional variance ratio
        ])
        
        # Handle any NaN values that might have slipped through
        features = jnp.nan_to_num(features, 0.0)
        
        return features
    
    def __call__(self, 
                 target_embedding: jnp.ndarray,      # [hidden_dim]
                 node_embeddings: jnp.ndarray,       # [n_vars, hidden_dim]
                 data: Optional[jnp.ndarray] = None,  # [N, n_vars, 3] - for pairwise features
                 target_idx: Optional[int] = None,   # Target variable index
                 is_training: bool = False) -> jnp.ndarray:  # [n_vars]
        """
        Compute attention scores between target node and all potential parents.
        
        Args:
            target_embedding: Target node embedding [hidden_dim]
            node_embeddings: All node embeddings [n_vars, hidden_dim]
            data: Original data for computing pairwise features (optional)
            target_idx: Index of target variable (required if data provided)
            is_training: Whether in training mode
            
        Returns:
            Parent attention logits [n_vars]
        """
        n_vars = node_embeddings.shape[0]
        
        if self.use_pairwise_features and data is not None and target_idx is not None:
            # Compute pairwise features between target and each candidate
            target_data = data[:, target_idx, :]  # [N, 3]
            
            pairwise_features = []
            for i in range(n_vars):
                if i == target_idx:
                    # Target cannot be its own parent - use zero features
                    features = jnp.zeros(11)  # Match feature dimension (updated with lag correlations)
                else:
                    candidate_data = data[:, i, :]  # [N, 3]
                    features = self.compute_pairwise_features(target_data, candidate_data)
                pairwise_features.append(features)
            
            pairwise_features = jnp.stack(pairwise_features, axis=0)  # [n_vars, 11]
            
            # Project pairwise features to attention dimension
            pairwise_projection = hk.Linear(
                self.key_size, 
                w_init=self.w_init, 
                name="pairwise_projection"
            )(pairwise_features)  # [n_vars, key_size]
            
            # Combine with embedding-based attention
            # This allows the model to use both learned representations and statistical features
        else:
            pairwise_projection = 0.0
        
        # Standard attention mechanism
        query_projection = hk.Linear(self.key_size, w_init=self.w_init, name="query_proj")
        key_projection = hk.Linear(self.key_size, w_init=self.w_init, name="key_proj")
        
        q = query_projection(target_embedding)  # [key_size]
        k = key_projection(node_embeddings)  # [n_vars, key_size]
        
        # Add pairwise features if available
        if self.use_pairwise_features and data is not None:
            k = k + pairwise_projection
        
        # Compute initial attention scores
        scores = jnp.dot(k, q) / jnp.sqrt(self.key_size)  # [n_vars]
        
        # Sophisticated scoring network (from other implementation)
        if self.use_pairwise_features and data is not None:
            # Create combined representation for each potential parent
            # Combines: [target_embedding, candidate_embedding, pairwise_features]
            combined_features = []
            for j in range(n_vars):
                combined = jnp.concatenate([
                    target_embedding,          # Target embedding [hidden_dim]
                    node_embeddings[j],        # Candidate parent embedding [hidden_dim]
                    pairwise_features[j] * 10  # Scaled pairwise features [11]
                ])
                combined_features.append(combined)
            
            combined_features = jnp.stack(combined_features)  # [n_vars, hidden_dim * 2 + 11]
            
            # Multi-layer scoring network
            score_net = hk.Sequential([
                hk.Linear(self.hidden_dim, w_init=self.w_init),
                jax.nn.gelu,
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                hk.Linear(self.hidden_dim // 2, w_init=self.w_init),
                jax.nn.gelu,
                hk.Linear(1, w_init=self.w_init)  # Single score per variable
            ], name="score_network")
            
            network_scores = score_net(combined_features).squeeze(-1)  # [n_vars]
            
            # Add direct feature influence
            # Some features are strong indicators (e.g., high lag correlation)
            feature_direct = hk.Linear(
                1,
                w_init=self.w_init,
                with_bias=False,
                name="feature_direct"
            )
            
            # Use most informative features directly
            # correlation, lag_corr, var_ratio
            direct_features = pairwise_features[:, [0, 8, 10]]  # indices for corr, lag_corr, var_ratio
            direct_scores = feature_direct(direct_features).squeeze(-1)  # [n_vars]
            
            # Combine all score components
            scores = scores + network_scores + 0.5 * direct_scores
        
        # Apply dropout if training
        if is_training and self.dropout > 0:
            scores = hk.dropout(hk.next_rng_key(), self.dropout, scores)
        
        return scores


class SimpleParentAttentionLayer(hk.Module):
    """
    Simplified parent attention layer for comparison.
    
    This version uses only embedding similarity without pairwise features.
    """
    
    def __init__(self, 
                 hidden_dim: int = 128,
                 key_size: int = 32,
                 name: str = "SimpleParentAttentionLayer"):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.key_size = key_size
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
    
    def __call__(self, 
                 target_embedding: jnp.ndarray,
                 node_embeddings: jnp.ndarray,
                 **kwargs) -> jnp.ndarray:
        """
        Simple dot-product attention.
        """
        # Direct similarity computation
        scores = jnp.dot(node_embeddings, target_embedding)  # [n_vars]
        
        # Normalize by dimension
        scores = scores / jnp.sqrt(self.hidden_dim)
        
        return scores