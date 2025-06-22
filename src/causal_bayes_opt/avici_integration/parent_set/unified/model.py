"""
Unified Parent Set Prediction Model

Combines the proven AVICI-style transformer architecture from the original model
with target-aware conditioning and optional mechanism prediction capabilities.

This addresses the key architectural concerns:
1. Uses proven transformer architecture (lower risk)
2. Adds target-aware conditioning for better performance  
3. Provides adaptive max_parents (fixes hard-coded scalability issue)
4. Optional mechanism prediction via feature flags
5. Modular design for maintainability
"""

from typing import Dict, List, Any, Optional
import jax
import jax.numpy as jnp
import haiku as hk

from .config import TargetAwareConfig
from .target_conditioning import add_target_conditioning
from .mechanism_heads import MechanismPredictionHeads
from .utils import (
    compute_adaptive_max_parents, 
    validate_unified_config,
    validate_model_outputs
)
from ..enumeration import enumerate_possible_parent_sets, compute_adaptive_k
from ..encoding import encode_parent_set, create_parent_set_indicators


def layer_norm(*, axis, name=None):
    """Layer normalization - identical to original model."""
    return hk.LayerNorm(axis=axis, create_scale=True, create_offset=True, name=name)


class UnifiedParentSetModel(hk.Module):
    """
    Unified parent set prediction model that combines proven architecture
    with enhanced capabilities.
    
    Key features:
    - Proven AVICI-style transformer from original model  
    - Target-aware conditioning from modular model
    - Adaptive max_parents (fixes scalability issue)
    - Optional mechanism prediction via feature flags
    - Backward compatible with existing code
    """
    
    def __init__(self, config: TargetAwareConfig, name="UnifiedParentSetModel"):
        super().__init__(name=name)
        self.config = config
        self.ln_axis = -1
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
        
        # Initialize mechanism heads if needed
        if config.predict_mechanisms:
            self.mechanism_heads = MechanismPredictionHeads(config)
    
    def __call__(self, 
                 x: jnp.ndarray,  # [N, d, 3] 
                 variable_order: List[str],
                 target_variable: str,
                 is_training: bool = True) -> Dict[str, Any]:
        """
        Forward pass combining structure prediction with optional mechanism prediction.
        
        Args:
            x: Input data in AVICI format [N, d, 3] where:
                - [:, :, 0] = variable values
                - [:, :, 1] = intervention indicators  
                - [:, :, 2] = target indicators (1 for target variable, 0 otherwise)
            variable_order: List of variable names in order
            target_variable: Name of target variable
            is_training: Training mode flag
            
        Returns:
            Dictionary containing:
            - 'parent_set_logits': [k] logits for top-k parent sets
            - 'parent_sets': List of k parent sets (frozensets)
            - 'k': Number of parent sets
            - 'mechanism_predictions': (if enabled) Dict with mechanism info
        """
        n_vars = len(variable_order)
        # JAX-compatible target index computation using tensor operations
        target_idx = jnp.argmax(x[0, :, 2])  # Find target from indicator channel
        dropout_rate = self.config.dropout if is_training else 0.0
        
        # Validate configuration for this graph
        validate_unified_config(self.config, n_vars)
        
        # Compute adaptive max_parents (fixes hard-coded issue)
        adaptive_max_parents = compute_adaptive_max_parents(n_vars, self.config)
        
        # SIMPLIFIED: Use proven transformer architecture with [N, d, 3] format maintained
        
        # Initial embedding - process all 3 channels directly
        z = hk.Linear(self.config.dim, name="input_embedding")(x)  # [N, d, 3] -> [N, d, dim]
        
        # PROVEN: Apply original transformer layers with swapping
        variable_embeddings = self._apply_transformer_layers(z, dropout_rate)  # [d, dim]
        
        # Extract target-specific embedding
        target_embedding = variable_embeddings[target_idx]  # [dim]
        
        # Pre-compute parent sets using static enumeration based on concrete values
        # For JAX compatibility, use the variable order index directly
        target_idx_concrete = variable_order.index(target_variable)
        possible_parent_sets = self._enumerate_parent_sets_static(
            n_vars, target_idx_concrete, adaptive_max_parents
        )
        
        # Compute adaptive k
        k = min(len(possible_parent_sets), compute_adaptive_k(n_vars, adaptive_max_parents))
        
        # PROVEN: Use original MLP-based parent set scoring (JAX-compatible)
        parent_set_logits, pooled_features = self._score_parent_sets_jax_compatible(
            target_embedding, variable_embeddings, possible_parent_sets, n_vars, dropout_rate
        )
        
        # Get top-k parent sets (JAX-compatible)
        top_k_logits, top_k_indices = self._select_top_k_indices_jax_compatible(
            parent_set_logits, k
        )
        
        # Convert indices back to frozensets (outside JAX-compiled region)
        top_k_parent_sets = self._convert_indices_to_parent_sets(
            top_k_indices, possible_parent_sets
        )
        
        # Base output (always present)
        outputs = {
            'parent_set_logits': top_k_logits,
            'parent_sets': top_k_parent_sets,
            'k': len(top_k_parent_sets),
            'all_possible_parent_sets': possible_parent_sets
        }
        
        # ENHANCEMENT: Add mechanism predictions if enabled
        if self.config.predict_mechanisms:
            mechanism_predictions = self.mechanism_heads(
                pooled_features, top_k_logits, top_k_parent_sets
            )
            outputs['mechanism_predictions'] = mechanism_predictions
        
        # Validate outputs before returning
        expected_keys = ['parent_set_logits', 'parent_sets', 'k']
        validate_model_outputs(outputs, expected_keys, top_k_parent_sets, self.config)
        
        return outputs
    
    @hk.transparent
    def _apply_transformer_layers(self, z: jnp.ndarray, dropout_rate: float) -> jnp.ndarray:
        """
        Apply transformer layers - IDENTICAL to original proven architecture.
        
        This preserves the proven AVICI-style transformer that has been validated.
        """
        layers = 2 * self.config.layers  # Original used 2 * layers
        assert not layers % 2, "Number of layers must be even"

        for _ in range(layers):
            z = self._transformer_block(z, dropout_rate)
            z = jnp.swapaxes(z, -3, -2)  # PROVEN: swapaxes pattern from original

        z = layer_norm(axis=self.ln_axis)(z)
        z = jnp.max(z, axis=-3)  # [..., n_vars, enhanced_dim]
        return z
    
    @hk.transparent
    def _transformer_block(self, z: jnp.ndarray, dropout_rate: float) -> jnp.ndarray:
        """Transformer block - IDENTICAL to original proven architecture."""
        # Multi-head attention
        q_in = layer_norm(axis=self.ln_axis)(z)
        k_in = layer_norm(axis=self.ln_axis)(z)
        v_in = layer_norm(axis=self.ln_axis)(z)
        z_attn = hk.MultiHeadAttention(
            num_heads=self.config.num_heads,
            key_size=self.config.key_size,
            w_init_scale=2.0,
            model_size=z.shape[-1],  # Adapt to enhanced dimension
        )(q_in, k_in, v_in)
        z = z + hk.dropout(hk.next_rng_key(), dropout_rate, z_attn)

        # Feed-forward network
        z_in = layer_norm(axis=self.ln_axis)(z)
        z_ffn = hk.Sequential([
            hk.Linear(self.config.widening_factor * z.shape[-1], w_init=self.w_init),
            jax.nn.relu,
            hk.Linear(z.shape[-1], w_init=self.w_init),
        ])(z_in)
        z = z + hk.dropout(hk.next_rng_key(), dropout_rate, z_ffn)
        return z
    
    
    def _create_parent_set_scorer(self, dropout_rate: float) -> hk.Sequential:
        """Create parent set scorer MLP - shared across all parent sets."""
        def dropout_fn(x):
            return hk.dropout(hk.next_rng_key(), dropout_rate, x)
            
        return hk.Sequential([
            hk.Linear(self.config.dim, w_init=self.w_init, name="score_1"),
            jax.nn.relu,
            dropout_fn,
            hk.Linear(self.config.dim // 2, w_init=self.w_init, name="score_2"),
            jax.nn.relu,
            hk.Linear(1, w_init=self.w_init, name="score_out"),
            lambda x: jnp.squeeze(x, axis=-1)
        ], name="parent_set_scorer")
    
    @hk.transparent
    def _score_parent_set(self, 
                         scorer: hk.Sequential,
                         target_embedding: jnp.ndarray, 
                         parent_set_embedding: jnp.ndarray) -> jnp.ndarray:
        """
        Score target-parent compatibility using shared MLP.
        """
        # Concatenate target and parent set embeddings
        combined = jnp.concatenate([target_embedding, parent_set_embedding], axis=-1)
        
        # Use shared scorer
        return scorer(combined)
    
    
    def _enumerate_parent_sets_static(self, 
                                    n_vars: int, 
                                    target_idx: int, 
                                    max_parents: int) -> List[jnp.ndarray]:
        """
        Static parent set enumeration (computed outside JAX compilation).
        Returns parent sets as binary indicator arrays.
        """
        # Convert JAX scalars to Python ints to avoid tracing issues
        n_vars = int(n_vars)
        target_idx = int(target_idx)
        max_parents = int(max_parents)
        
        possible_parent_sets = []
        
        # Empty parent set
        empty_set = jnp.zeros(n_vars, dtype=bool)
        possible_parent_sets.append(empty_set)
        
        # Single parent sets
        for i in range(n_vars):
            if i != target_idx:  # Can't be parent of itself
                parent_set = jnp.zeros(n_vars, dtype=bool)
                parent_set = parent_set.at[i].set(True)
                possible_parent_sets.append(parent_set)
        
        # For larger parent sets, enumerate combinations up to max_parents
        if max_parents > 1:
            from itertools import combinations
            non_target_indices = [i for i in range(n_vars) if i != target_idx]
            
            for size in range(2, min(max_parents + 1, len(non_target_indices) + 1)):
                for parent_indices in combinations(non_target_indices, size):
                    parent_set = jnp.zeros(n_vars, dtype=bool)
                    for idx in parent_indices:
                        parent_set = parent_set.at[idx].set(True)
                    possible_parent_sets.append(parent_set)
        
        return possible_parent_sets
    
    @hk.transparent  
    def _score_parent_sets_jax_compatible(self, 
                                        target_embedding: jnp.ndarray,
                                        variable_embeddings: jnp.ndarray,
                                        possible_parent_sets: List[jnp.ndarray],
                                        n_vars: int,
                                        dropout_rate: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        JAX-compatible parent set scoring using tensor operations.
        """
        # Create shared scorer (only creates modules once)
        scorer = self._create_parent_set_scorer(dropout_rate)
        
        parent_set_logits = []
        
        for parent_set_indicators in possible_parent_sets:
            # Encode parent set using tensor operations
            parent_set_emb = self._encode_parent_set_jax_compatible(
                parent_set_indicators, variable_embeddings
            )
            
            # PROVEN: Use MLP scoring from original model
            score = self._score_parent_set(scorer, target_embedding, parent_set_emb)
            parent_set_logits.append(score)
        
        parent_set_logits = jnp.array(parent_set_logits)
        
        # Create pooled features for mechanism prediction
        pooled_features = jnp.mean(variable_embeddings, axis=0)  # Global representation
        
        return parent_set_logits, pooled_features
    
    def _encode_parent_set_jax_compatible(self, 
                                        parent_set_indicators: jnp.ndarray,
                                        variable_embeddings: jnp.ndarray) -> jnp.ndarray:
        """
        JAX-compatible parent set encoding using tensor operations.
        """
        # Weighted sum of parent variable embeddings
        weighted_embeddings = variable_embeddings * parent_set_indicators[:, None]
        parent_set_emb = jnp.sum(weighted_embeddings, axis=0)
        
        # If empty parent set, use zero embedding
        num_parents = jnp.sum(parent_set_indicators)
        parent_set_emb = jnp.where(
            num_parents > 0,
            parent_set_emb / num_parents,  # Average of parent embeddings
            jnp.zeros_like(parent_set_emb)  # Zero for empty set
        )
        
        return parent_set_emb
        
    def _select_top_k_indices_jax_compatible(self, 
                                           parent_set_logits: jnp.ndarray,
                                           k: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        JAX-compatible top-k selection returning indices only.
        """
        # Sort indices by logits in descending order
        sorted_indices = jnp.argsort(parent_set_logits)[::-1]
        
        # Take top-k (or all if fewer than k)
        actual_k = jnp.minimum(k, len(parent_set_logits))
        top_k_indices = sorted_indices[:actual_k]
        top_k_logits = parent_set_logits[top_k_indices]
        
        return top_k_logits, top_k_indices
    
    def _convert_indices_to_parent_sets(self,
                                      top_k_indices: jnp.ndarray,
                                      possible_parent_sets: List[jnp.ndarray]) -> List[frozenset]:
        """
        Convert indices back to frozensets (non-JAX compiled).
        """
        top_k_parent_sets = []
        for idx in top_k_indices:
            indicators = possible_parent_sets[int(idx)]
            # Convert binary indicators back to frozenset of indices
            parent_indices = jnp.where(indicators)[0]
            parent_set = frozenset(int(i) for i in parent_indices)
            top_k_parent_sets.append(parent_set)
        
        return top_k_parent_sets


def create_unified_parent_set_model(config: Optional[TargetAwareConfig] = None) -> hk.Transformed:
    """
    Factory function to create unified parent set model.
    
    Args:
        config: Configuration for the model. If None, uses default structure-only config.
        
    Returns:
        Transformed Haiku model ready for training/inference
    """
    if config is None:
        from .config import create_structure_only_config
        config = create_structure_only_config()
    
    def model_fn(x: jnp.ndarray, 
                 variable_order: List[str], 
                 target_variable: str, 
                 is_training: bool = True):
        model = UnifiedParentSetModel(config)
        return model(x, variable_order, target_variable, is_training)
    
    return hk.transform(model_fn)


# Backward compatibility functions
def create_structure_only_model(**kwargs) -> hk.Transformed:
    """Create model in structure-only mode for backward compatibility."""
    from .config import create_structure_only_config
    config = create_structure_only_config(**kwargs)
    return create_unified_parent_set_model(config)


def create_mechanism_aware_model(mechanism_types: List[str] = None, **kwargs) -> hk.Transformed:
    """Create model with mechanism prediction enabled."""
    from .config import create_mechanism_aware_config
    config = create_mechanism_aware_config(mechanism_types, **kwargs)
    return create_unified_parent_set_model(config)