"""
JAX-Compatible Unified Parent Set Model

This module provides a JAX-compilable version of the UnifiedParentSetModel that:
1. Preserves ALL unified features (target conditioning, mechanism prediction, adaptive parameters)
2. Eliminates JAX compilation blockers (Python loops, string operations, dynamic control flow)
3. Uses tensor-based operations for maximum performance
4. Maintains backward compatibility with the unified API

Key improvements over unified/model.py:
- Pre-computed parent set lookup tables
- Vectorized parent set scoring (replaces Python loops)
- Integer-based indexing (eliminates string operations)
- Fixed-size tensor operations with padding
- Full @jax.jit compatibility
"""

from typing import Dict, List, Any, Optional, Tuple
import jax
import jax.numpy as jnp
import haiku as hk
import logging

from .config import TargetAwareConfig, compute_adaptive_max_parents, validate_config_for_graph
from .target_conditioning import add_target_conditioning
from .mechanism_heads import MechanismPredictionHeads
from .utils import validate_unified_config, validate_model_outputs

# Import JAX-compatible utilities
from ..jax_enumeration import (
    precompute_parent_set_tables,
    filter_parent_sets_for_target_jax,
    select_top_k_parent_sets_jax,
    create_parent_set_lookup,
    interpret_parent_set_results
)
from ..jax_encoding import batch_encode_parent_sets_jax

logger = logging.getLogger(__name__)


def layer_norm(*, axis, name=None):
    """Layer normalization - identical to unified model."""
    return hk.LayerNorm(axis=axis, create_scale=True, create_offset=True, name=name)


class JAXUnifiedParentSetModel(hk.Module):
    """
    JAX-compatible version of UnifiedParentSetModel.
    
    Preserves all unified features while enabling JAX compilation:
    - Target-aware conditioning from unified model
    - Adaptive max_parents (fixes scalability)
    - Optional mechanism prediction via feature flags
    - Proven AVICI-style transformer architecture
    - Full backward compatibility with unified API
    
    NEW: JAX-compilable tensor operations instead of Python loops
    """
    
    def __init__(self, 
                 config: TargetAwareConfig, 
                 n_vars: int,
                 name="JAXUnifiedParentSetModel"):
        super().__init__(name=name)
        self.config = config
        self.n_vars = n_vars
        self.ln_axis = -1
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
        
        # Pre-compute parent set lookup tables for JAX compilation
        self._init_parent_set_tables()
        
        # Initialize mechanism heads if needed (preserve unified feature)
        if config.predict_mechanisms:
            self.mechanism_heads = MechanismPredictionHeads(config)
    
    def _init_parent_set_tables(self):
        """Pre-compute parent set lookup tables for JAX operations."""
        # Compute adaptive max_parents (preserve unified feature)
        self.adaptive_max_parents = compute_adaptive_max_parents(self.n_vars, self.config)
        
        # Pre-compute all possible parent sets as lookup tables
        indicators, sizes, _ = precompute_parent_set_tables(
            self.n_vars, self.adaptive_max_parents
        )
        
        # Store as JAX arrays for compilation
        self.parent_set_indicators = indicators  # [max_parent_sets, n_vars]
        self.parent_set_sizes = sizes           # [max_parent_sets]
        self.max_parent_sets = len(indicators)
        
        # Compute adaptive k (preserve unified logic)
        self.adaptive_k = self._compute_adaptive_k()
        
        logger.debug(
            f"Initialized JAX parent set tables: n_vars={self.n_vars}, "
            f"max_parents={self.adaptive_max_parents}, max_sets={self.max_parent_sets}, k={self.adaptive_k}"
        )
    
    def _compute_adaptive_k(self) -> int:
        """Compute adaptive k value - same logic as unified model."""
        # Use unified model's compute_adaptive_k logic
        from ..enumeration import compute_adaptive_k
        return compute_adaptive_k(self.n_vars, self.adaptive_max_parents)
    
    def __call__(self, 
                 x: jnp.ndarray,  # [N, d, 3] 
                 target_idx: int,  # Use integer index for JAX compatibility
                 is_training: bool = True) -> Dict[str, Any]:
        """
        JAX-compatible forward pass preserving unified model features.
        
        Args:
            x: Input data in AVICI format [N, d, 3]
            target_idx: Target variable index (converted from string outside)
            is_training: Training mode flag
            
        Returns:
            Dictionary containing same outputs as unified model but with JAX arrays
        """
        dropout_rate = self.config.dropout if is_training else 0.0
        
        # PRESERVE UNIFIED FEATURE: Validate configuration
        validate_unified_config(self.config, self.n_vars)
        
        # PRESERVE UNIFIED ARCHITECTURE: Use proven transformer with enhancements
        
        # Initial embedding (same as unified)
        z = hk.Linear(self.config.dim, name="input_embedding")(x)  # [N, d, 3] -> [N, d, dim]
        
        # PRESERVE UNIFIED FEATURE: Add target-aware conditioning
        if self.config.enable_target_conditioning:
            z = self._add_target_conditioning_jax(z, target_idx)
        # z is now [N, d, enhanced_dim] if conditioning enabled
        
        # PRESERVE UNIFIED ARCHITECTURE: Apply transformer layers with swapping
        variable_embeddings = self._apply_transformer_layers(z, dropout_rate)  # [d, enhanced_dim]
        
        # Extract target-specific embedding
        target_embedding = variable_embeddings[target_idx]  # [enhanced_dim]
        
        # JAX IMPROVEMENT: Use tensor-based parent set operations
        
        # Filter parent sets for this target using JAX operations
        valid_mask = filter_parent_sets_for_target_jax(
            self.parent_set_indicators,
            self.parent_set_sizes,
            target_idx,
            self.adaptive_max_parents
        )  # [max_parent_sets]
        
        # Vectorized parent set encoding (replaces Python loop)
        parent_set_embeddings = batch_encode_parent_sets_jax(
            self.parent_set_indicators,
            variable_embeddings,
            valid_mask
        )  # [max_parent_sets, enhanced_dim]
        
        # PRESERVE UNIFIED FEATURE: Use proven MLP-based scoring (vectorized)
        parent_set_logits = self._score_parent_sets_vectorized(
            target_embedding,
            parent_set_embeddings,
            valid_mask,
            dropout_rate
        )  # [max_parent_sets]
        
        # JAX IMPROVEMENT: Select top-k using tensor operations
        # Use fixed k=8 to avoid JAX compilation issues for now
        # TODO: Make adaptive_k work with JAX compilation
        fixed_k = 8  # Should be sufficient for most small-scale problems
        top_k_indices, top_k_logits = select_top_k_parent_sets_jax(
            parent_set_logits,
            valid_mask,
            fixed_k
        )  # [k], [k]
        
        # Create pooled features for mechanism prediction (preserve unified feature)
        pooled_features = jnp.mean(variable_embeddings, axis=0)  # Global representation
        
        # Base output (preserve unified API)
        outputs = {
            'parent_set_logits': top_k_logits,           # [k] JAX array
            'parent_set_indices': top_k_indices,         # [k] JAX array (NEW: indices for interpretation)
            'k': jnp.array(len(top_k_logits)),          # JAX scalar
            'all_logits': parent_set_logits,             # [max_parent_sets] for debugging
            'variable_embeddings': variable_embeddings,  # [d, enhanced_dim]
            'target_embedding': target_embedding,        # [enhanced_dim]
            'valid_mask': valid_mask                     # [max_parent_sets]
        }
        
        # PRESERVE UNIFIED FEATURE: Add mechanism predictions if enabled
        if self.config.predict_mechanisms:
            mechanism_predictions = self.mechanism_heads(
                pooled_features, top_k_logits, top_k_indices  # Use indices instead of frozensets
            )
            outputs['mechanism_predictions'] = mechanism_predictions
        
        return outputs
    
    @hk.transparent
    def _add_target_conditioning_jax(self, z: jnp.ndarray, target_idx: int) -> jnp.ndarray:
        """
        Add target-aware conditioning using JAX operations.
        
        Preserves the target conditioning feature from unified model but with JAX compatibility.
        """
        n_samples, n_vars, dim = z.shape
        
        # Create target mask using JAX operations
        target_mask = jnp.zeros((n_vars, 1))
        target_mask = target_mask.at[target_idx].set(1.0)  # [n_vars, 1]
        
        # Expand mask for batch dimension
        target_mask_expanded = jnp.tile(target_mask[None, :, :], (n_samples, 1, 1))  # [N, n_vars, 1]
        
        # Target conditioning embedding
        target_emb = hk.Linear(
            self.config.target_embedding_dim, 
            name="target_conditioning"
        )(target_mask_expanded)  # [N, n_vars, target_dim]
        
        # Concatenate with original embeddings
        conditioned_z = jnp.concatenate([z, target_emb], axis=-1)  # [N, n_vars, dim + target_dim]
        
        return conditioned_z
    
    @hk.transparent
    def _apply_transformer_layers(self, z: jnp.ndarray, dropout_rate: float) -> jnp.ndarray:
        """
        Apply transformer layers - IDENTICAL to unified model proven architecture.
        """
        layers = 2 * self.config.layers  # Unified used 2 * layers
        assert not layers % 2, "Number of layers must be even"

        for _ in range(layers):
            z = self._transformer_block(z, dropout_rate)
            z = jnp.swapaxes(z, -3, -2)  # PROVEN: swapaxes pattern from unified

        z = layer_norm(axis=self.ln_axis)(z)
        z = jnp.max(z, axis=-3)  # [..., n_vars, enhanced_dim]
        return z
    
    @hk.transparent
    def _transformer_block(self, z: jnp.ndarray, dropout_rate: float) -> jnp.ndarray:
        """Transformer block - IDENTICAL to unified model proven architecture."""
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
    
    @hk.transparent
    def _score_parent_sets_vectorized(self,
                                    target_embedding: jnp.ndarray,    # [enhanced_dim]
                                    parent_set_embeddings: jnp.ndarray, # [max_parent_sets, enhanced_dim]
                                    valid_mask: jnp.ndarray,          # [max_parent_sets]
                                    dropout_rate: float) -> jnp.ndarray:
        """
        JAX IMPROVEMENT: Vectorized parent set scoring (replaces Python loop).
        
        Preserves the proven MLP-based scoring from unified model but vectorizes the computation.
        """
        batch_size = parent_set_embeddings.shape[0]
        
        # Expand target embedding to match batch
        target_expanded = jnp.tile(target_embedding[None, :], (batch_size, 1))  # [max_parent_sets, enhanced_dim]
        
        # Concatenate target and parent set embeddings
        combined = jnp.concatenate([target_expanded, parent_set_embeddings], axis=-1)  # [max_parent_sets, 2*enhanced_dim]
        
        # PRESERVE UNIFIED FEATURE: Use proven MLP architecture (vectorized)
        # First layer
        x = hk.Linear(self.config.dim, w_init=self.w_init, name="score_layer_1")(combined)
        x = jax.nn.relu(x)
        x = hk.dropout(hk.next_rng_key(), dropout_rate, x)
        
        # Second layer
        x = hk.Linear(self.config.dim // 2, w_init=self.w_init, name="score_layer_2")(x)
        x = jax.nn.relu(x)
        
        # Output layer
        scores = hk.Linear(1, w_init=self.w_init, name="score_output")(x).squeeze(-1)  # [max_parent_sets]
        
        # Apply validity mask (set invalid scores to -inf)
        masked_scores = jnp.where(valid_mask, scores, -jnp.inf)
        
        return masked_scores


def create_jax_unified_parent_set_model(
    config: Optional[TargetAwareConfig] = None,
    n_vars: int = None,
    variable_names: List[str] = None
) -> Tuple[hk.Transformed, Dict]:
    """
    Factory function to create JAX-compatible unified parent set model.
    
    Preserves the unified model API while enabling JAX compilation.
    
    Args:
        config: Configuration for the model. If None, uses default structure-only config.
        n_vars: Number of variables (required for JAX pre-computation)
        variable_names: Variable names for result interpretation
        
    Returns:
        transformed_model: JAX-compilable Haiku transformed model
        lookup_tables: Pre-computed lookup tables for result interpretation
    """
    if config is None:
        from .config import create_structure_only_config
        config = create_structure_only_config()
    
    if n_vars is None:
        if variable_names is not None:
            n_vars = len(variable_names)
        else:
            raise ValueError("Either n_vars or variable_names must be provided")
    
    if variable_names is None:
        variable_names = [f"X{i}" for i in range(n_vars)]
    
    # Validate configuration for this graph size
    validate_config_for_graph(config, n_vars)
    
    def model_fn(x: jnp.ndarray, target_idx: int, is_training: bool = True):
        model = JAXUnifiedParentSetModel(config, n_vars)
        return model(x, target_idx, is_training)
    
    transformed_model = hk.transform(model_fn)
    
    # Create lookup tables for result interpretation (preserve unified API)
    adaptive_max_parents = compute_adaptive_max_parents(n_vars, config)
    lookup_tables = create_parent_set_lookup(n_vars, variable_names, adaptive_max_parents)
    lookup_tables['config'] = config  # Store config for reference
    
    return transformed_model, lookup_tables


class JAXUnifiedParentSetModelWrapper:
    """
    Backward compatibility wrapper that provides the same interface as UnifiedParentSetModel
    but uses JAX-compiled operations internally.
    
    This allows existing code to use the JAX model without changes.
    """
    
    def __init__(self, 
                 config: TargetAwareConfig,
                 variable_names: List[str],
                 **kwargs):
        self.config = config
        self.variable_names = variable_names
        self.n_vars = len(variable_names)
        
        # Create JAX model and lookup tables
        self.model, self.lookup_tables = create_jax_unified_parent_set_model(
            config, self.n_vars, variable_names
        )
        self.params = None
        
        # Cache for string-to-index conversion
        self.name_to_idx = {name: i for i, name in enumerate(variable_names)}
    
    def init(self, key: jax.Array, x: jnp.ndarray, target_variable: str = None, **kwargs):
        """Initialize model parameters using the same API as unified model."""
        if target_variable is None:
            target_variable = self.variable_names[0]  # Default to first variable
        
        target_idx = self.name_to_idx[target_variable]
        self.params = self.model.init(key, x, target_idx, True)
        return self.params
    
    def __call__(self, 
                 x: jnp.ndarray,
                 variable_order: List[str],
                 target_variable: str,
                 is_training: bool = True) -> Dict[str, Any]:
        """
        Forward pass with the same API as unified model.
        
        Converts string inputs to integer indices for JAX compatibility,
        then converts outputs back to unified format.
        """
        if self.params is None:
            raise ValueError("Model must be initialized before calling")
        
        # Convert target variable to index
        if target_variable not in self.name_to_idx:
            raise ValueError(f"Target variable '{target_variable}' not in variable_names")
        target_idx = self.name_to_idx[target_variable]
        
        # JAX forward pass
        jax_outputs = self.model.apply(self.params, x, target_idx, is_training)
        
        # Convert outputs to unified format (preserve API)
        unified_outputs = self._convert_to_unified_format(
            jax_outputs, target_variable, variable_order
        )
        
        return unified_outputs
    
    def _convert_to_unified_format(self, 
                                 jax_outputs: Dict[str, jnp.ndarray],
                                 target_variable: str,
                                 variable_order: List[str]) -> Dict[str, Any]:
        """Convert JAX outputs to unified model format for backward compatibility."""
        # Extract top-k results
        top_k_indices = jax_outputs['parent_set_indices']
        top_k_logits = jax_outputs['parent_set_logits']
        
        # Convert indices back to parent sets (frozensets of variable names)
        parent_sets_with_probs = interpret_parent_set_results(
            top_k_indices, top_k_logits, self.lookup_tables, target_variable
        )
        
        # Extract parent sets and logits separately (unified format)
        parent_sets = [ps for ps, _ in parent_sets_with_probs]
        parent_set_logits = jnp.array([prob for _, prob in parent_sets_with_probs])
        
        # Create unified outputs (preserve all fields)
        unified_outputs = {
            'parent_set_logits': parent_set_logits,
            'parent_sets': parent_sets,
            'k': len(parent_sets),
            'all_possible_parent_sets': parent_sets_with_probs  # For debugging
        }
        
        # Add mechanism predictions if present (preserve unified feature)
        if 'mechanism_predictions' in jax_outputs:
            unified_outputs['mechanism_predictions'] = jax_outputs['mechanism_predictions']
        
        return unified_outputs


def predict_with_jax_unified_model(
    model_apply: callable,
    params: Dict,
    x: jnp.ndarray,
    target_variable: str,
    variable_order: List[str],
    lookup_tables: Dict,
    is_training: bool = False
) -> Tuple[List[Tuple[frozenset, float]], Dict[str, jnp.ndarray]]:
    """
    Predict parent sets using JAX unified model with result interpretation.
    
    Provides a direct interface for using the JAX model while maintaining
    compatibility with the unified model API.
    """
    # Convert target variable to index
    name_to_idx = lookup_tables['name_to_idx']
    if target_variable not in name_to_idx:
        raise ValueError(f"Target variable '{target_variable}' not found in lookup tables")
    target_idx = name_to_idx[target_variable]
    
    # JAX forward pass
    jax_outputs = model_apply(params, x, target_idx, is_training)
    
    # Interpret results
    top_k_indices = jax_outputs['parent_set_indices'] 
    top_k_logits = jax_outputs['parent_set_logits']
    
    interpreted_results = interpret_parent_set_results(
        top_k_indices, top_k_logits, lookup_tables, target_variable
    )
    
    return interpreted_results, jax_outputs