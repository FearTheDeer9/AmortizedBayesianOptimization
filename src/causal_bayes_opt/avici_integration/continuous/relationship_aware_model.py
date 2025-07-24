"""
Relationship-Aware Parent Set Prediction Model.

This model fundamentally changes the architecture to preserve cross-variable
relationships throughout the computation.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional, Dict


class RelationshipAwareEncoder(hk.Module):
    """
    Encoder that processes the entire data matrix to preserve relationships.
    
    Key insight: We need to process the ENTIRE [N, d] data matrix jointly,
    not each variable independently.
    """
    
    def __init__(self,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 name: str = "RelationshipAwareEncoder"):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
    
    def __call__(self, data: jnp.ndarray, target_idx: int) -> Dict[str, jnp.ndarray]:
        """
        Process intervention data while preserving relationships.
        
        Args:
            data: [N, d, 3] intervention data
            target_idx: Index of target variable
            
        Returns:
            Dictionary with parent probabilities and embeddings
        """
        N, d, channels = data.shape
        
        # Step 1: Extract the data components
        values = data[:, :, 0]  # [N, d] - actual values
        interventions = data[:, :, 1]  # [N, d] - intervention indicators
        observations = data[:, :, 2]  # [N, d] - observation indicators
        
        # Step 2: Compute sufficient statistics that preserve relationships
        # These are the key features for causal discovery
        
        # 2a. Correlation matrix (the most important feature!)
        # Only use observational data for correlations
        obs_mask = observations * (1 - interventions)  # [N, d]
        masked_values = jnp.where(obs_mask, values, 0.0)
        
        # Center the data
        counts = jnp.sum(obs_mask, axis=0, keepdims=True) + 1e-8  # [1, d]
        means = jnp.sum(masked_values, axis=0, keepdims=True) / counts  # [1, d]
        centered = (masked_values - means) * obs_mask  # [N, d]
        
        # Compute covariance
        cov_matrix = (centered.T @ centered) / (jnp.sum(obs_mask[:, 0]) + 1e-8)  # [d, d]
        
        # Normalize to correlation
        std = jnp.sqrt(jnp.diag(cov_matrix) + 1e-8)
        corr_matrix = cov_matrix / (std[:, None] * std[None, :])  # [d, d]
        
        # 2b. Intervention effects
        # When we intervene on Xi, how does Xj change?
        intervention_effects = jnp.zeros((d, d))
        for i in range(d):
            # Samples where Xi was intervened
            intervened_mask = interventions[:, i]  # [N]
            if jnp.sum(intervened_mask) > 0:
                # Compare values of other variables when Xi is/isn't intervened
                for j in range(d):
                    if i != j:
                        effect = jnp.mean(values[intervened_mask, j]) - means[0, j]
                        intervention_effects = intervention_effects.at[i, j].set(effect)
        
        # 2c. Sample diversity score (how informative is our data)
        sample_diversity = jnp.std(values, axis=0)  # [d]
        
        # Step 3: Encode these relationship features
        # Flatten correlation matrix to use as features
        corr_features = corr_matrix.reshape(-1)  # [d*d]
        effect_features = intervention_effects.reshape(-1)  # [d*d]
        
        # Combine all features
        all_features = jnp.concatenate([
            corr_features,
            effect_features,
            sample_diversity,
            means[0],  # Variable means
        ])  # [2*d*d + 2*d]
        
        # Step 4: Process with MLP to get parent scores
        x = hk.Linear(self.hidden_dim)(all_features)
        x = jax.nn.relu(x)
        
        for _ in range(self.num_layers - 1):
            x = hk.Linear(self.hidden_dim)(x)
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            x = jax.nn.relu(x)
        
        # Step 5: Extract parent-specific features for target
        # The key insight: parent relationships are in the correlation/effect matrices
        
        # Get target's row from correlation matrix (how target correlates with others)
        target_correlations = corr_matrix[target_idx, :]  # [d]
        
        # Get intervention effects on target
        target_effects = intervention_effects[:, target_idx]  # [d]
        
        # Combine with learned features
        target_features = jnp.concatenate([
            target_correlations,
            target_effects,
            x[:d]  # First d elements of learned features
        ])  # [3*d]
        
        # Step 6: Compute parent scores directly from relationships
        # Use a simple linear layer - the features already contain the information!
        parent_scores = hk.Linear(d)(target_features)  # [d]
        
        # Mask target (can't be its own parent)
        masked_scores = jnp.where(
            jnp.arange(d) == target_idx,
            -1e9,
            parent_scores
        )
        
        # Convert to probabilities
        parent_probs = jax.nn.softmax(masked_scores)
        
        return {
            'parent_probabilities': parent_probs,
            'parent_scores': parent_scores,
            'correlation_matrix': corr_matrix,
            'intervention_effects': intervention_effects,
            'target_correlations': target_correlations
        }


class RelationshipAwareParentSetModel(hk.Module):
    """
    Complete model using relationship-aware encoding.
    
    This is a complete redesign that:
    1. Processes the data matrix jointly (not per-variable)
    2. Explicitly computes and uses correlations
    3. Incorporates intervention effects
    4. Directly maps relationships to parent probabilities
    """
    
    def __init__(self,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 name: str = "RelationshipAwareParentSetModel"):
        super().__init__(name=name)
        self.encoder = RelationshipAwareEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        self.dropout = dropout
    
    def __call__(self,
                 data: jnp.ndarray,
                 target_variable: int,
                 is_training: bool = False) -> Dict[str, jnp.ndarray]:
        """
        Predict parent probabilities using relationship-aware encoding.
        
        Args:
            data: [N, d, 3] intervention data
            target_variable: Target variable index
            is_training: Training flag
            
        Returns:
            Dictionary with predictions and intermediate features
        """
        # Get predictions from encoder
        output = self.encoder(data, target_variable)
        
        # Apply dropout if training
        if is_training and self.dropout > 0:
            output['parent_probabilities'] = hk.dropout(
                hk.next_rng_key(),
                self.dropout,
                output['parent_probabilities']
            )
        
        return output


def create_simple_correlation_based_model():
    """
    Create an even simpler model that directly uses correlations.
    
    This is for testing - it should definitely produce varied outputs!
    """
    def correlation_model(data: jnp.ndarray, target_idx: int) -> Dict[str, jnp.ndarray]:
        """Ultra-simple: parent probability proportional to correlation."""
        N, d, _ = data.shape
        
        # Get values
        values = data[:, :, 0]  # [N, d]
        
        # Compute correlations
        corr_matrix = jnp.corrcoef(values.T)  # [d, d]
        
        # Get absolute correlations with target
        target_corrs = jnp.abs(corr_matrix[target_idx, :])  # [d]
        
        # Mask target itself
        masked_corrs = jnp.where(
            jnp.arange(d) == target_idx,
            0.0,
            target_corrs
        )
        
        # Convert to probabilities (with temperature for sharpness)
        temperature = 0.5
        parent_probs = jax.nn.softmax(masked_corrs / temperature)
        
        return {
            'parent_probabilities': parent_probs,
            'correlations': target_corrs,
            'correlation_matrix': corr_matrix
        }
    
    return correlation_model