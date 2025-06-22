#!/usr/bin/env python3
"""
⚠️  DEPRECATED: Mechanism-Aware Parent Set Prediction Model

This module is deprecated in favor of the unified implementation.
Use src/causal_bayes_opt/avici_integration/parent_set/unified/model.py instead.

The unified model provides:
- Better JAX compatibility and performance
- Simplified [N, d, 3] format compatibility  
- Same mechanism-aware functionality with cleaner architecture
- Improved integration with the rest of the ACBO framework

Migration Guide:
- Replace imports: from ..mechanism_aware import → from ..unified.model import
- Use create_unified_parent_set_model() instead of create_modular_parent_set_model()
- Configuration remains similar: MechanismAwareConfig → TargetAwareConfig

This file is maintained for backward compatibility during transition period.

---

ORIGINAL DESCRIPTION:
Mechanism-Aware Parent Set Prediction Model

This module implements the ModularParentSetModel with configurable mechanism prediction
capabilities. The model can operate in two modes:

1. Structure-only mode (predict_mechanisms=False): 
   - Backward compatible with existing infrastructure
   - Only predicts parent set topology
   - Fallback mode for validation and comparison

2. Enhanced mechanism-aware mode (predict_mechanisms=True):
   - Predicts both topology AND mechanism information
   - Mechanism type classification (linear, polynomial, gaussian, neural)
   - Parameter regression (coefficients, effect magnitudes, uncertainties)
   - Enables targeted high-impact intervention selection

The modular design allows easy switching between modes via feature flags and
supports scientific comparison of when added complexity is worthwhile.

Architecture Enhancement Pivot - Part A: Modular Model Architecture
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, FrozenSet, Tuple
from enum import Enum

import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
import pyrsistent as pyr

from .posterior import ParentSetPosterior, create_parent_set_posterior
from .enumeration import enumerate_possible_parent_sets
from .encoding import encode_parent_set, create_parent_set_indicators

logger = logging.getLogger(__name__)


class MechanismType:
    """Supported mechanism types for causal relationships."""
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    GAUSSIAN = "gaussian"
    NEURAL = "neural"


@dataclass(frozen=True)
class MechanismAwareConfig:
    """Configuration for mechanism-aware parent set prediction."""
    
    # Core feature flag
    predict_mechanisms: bool = False
    
    # Mechanism type configuration
    mechanism_types: List[str] = None
    
    # Model architecture
    max_parents: int = 5
    hidden_dim: int = 128
    n_layers: int = 8
    dropout: float = 0.1
    
    # Advanced options
    use_attention: bool = True
    parameter_dim: int = 32  # Dimension for mechanism parameter predictions
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Set default mechanism types if not provided
        if self.mechanism_types is None:
            object.__setattr__(self, 'mechanism_types', ["linear"])
        
        # Validate mechanism types
        if not self.mechanism_types:
            raise ValueError("mechanism_types cannot be empty")
        
        valid_types = get_all_mechanism_types()
        for mech_type in self.mechanism_types:
            if mech_type not in valid_types:
                raise ValueError(f"Unknown mechanism type: {mech_type}. Valid types: {valid_types}")
        
        # Validate other parameters
        if self.max_parents <= 0:
            raise ValueError("max_parents must be positive")
        
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")


@dataclass(frozen=True)
class MechanismPrediction:
    """Prediction for a specific parent set's mechanism."""
    
    parent_set: FrozenSet[str]
    mechanism_type: str
    parameters: Dict[str, Any]
    confidence: float  # Confidence in this mechanism prediction [0, 1]


class ModularParentSetModel(hk.Module):
    """
    Modular parent set prediction model with configurable mechanism awareness.
    
    This model can operate in two modes controlled by the config.predict_mechanisms flag:
    
    1. Structure-only mode (predict_mechanisms=False):
       - Only predicts parent set topology (which variables are parents)
       - Backward compatible with existing infrastructure
       - Returns only parent_set_logits
    
    2. Enhanced mechanism-aware mode (predict_mechanisms=True):
       - Predicts topology AND mechanism information
       - Mechanism type classification for each parent set
       - Parameter regression for effect magnitudes
       - Returns both parent_set_logits and mechanism_predictions
    
    The modular design allows easy switching between modes and scientific comparison.
    """
    
    def __init__(self, config: MechanismAwareConfig, name: str = "ModularParentSetModel"):
        super().__init__(name=name)
        self.config = config
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
    
    def __call__(self,
                 x: jnp.ndarray,  # [N, d, 3] AVICI format
                 variable_order: List[str],
                 target_variable: str) -> Dict[str, Any]:
        """
        Forward pass for parent set and optionally mechanism prediction.
        
        Args:
            x: Input data in AVICI format [N, d, 3]
            variable_order: Ordered list of variable names
            target_variable: Name of target variable
            
        Returns:
            Dictionary containing:
            - 'parent_set_logits': [k] logits over possible parent sets
            - 'mechanism_predictions': (if predict_mechanisms=True) Dict with:
                - 'mechanism_type_logits': [k, n_mechanism_types] 
                - 'mechanism_parameters': [k, n_mechanism_types, param_dim]
        """
        target_idx = variable_order.index(target_variable)
        
        # Always predict parent set structure (core functionality)
        structure_outputs = self._predict_structure(x, target_idx)
        
        if not self.config.predict_mechanisms:
            # Structure-only mode: return only parent set logits
            return {
                "parent_set_logits": structure_outputs["parent_set_logits"]
            }
        
        # Enhanced mode: add mechanism predictions
        mechanism_outputs = self._predict_mechanisms(
            structure_outputs["features"], 
            structure_outputs["parent_set_logits"],
            variable_order,
            target_variable
        )
        
        return {
            "parent_set_logits": structure_outputs["parent_set_logits"],
            "mechanism_predictions": mechanism_outputs
        }
    
    def _predict_structure(self, x: jnp.ndarray, target_idx: int) -> Dict[str, jnp.ndarray]:
        """
        Predict parent set structure (core functionality used in both modes).
        
        This implements the core parent set prediction logic similar to AVICI
        but adapted for our target-aware setting.
        """
        N, d, channels = x.shape
        
        # Input embedding: [N, d, 3] -> [N, d, hidden_dim]
        embedded = hk.Linear(self.config.hidden_dim, w_init=self.w_init)(x)
        
        # Target-aware conditioning: highlight target variable
        target_mask = jnp.zeros((d,))
        target_mask = target_mask.at[target_idx].set(1.0)
        target_embedding = hk.Linear(self.config.hidden_dim // 4, w_init=self.w_init)(target_mask)
        
        # Broadcast target embedding: [hidden_dim//4] -> [N, d, hidden_dim//4]
        target_broadcast = jnp.broadcast_to(
            target_embedding[None, None, :], 
            (N, d, self.config.hidden_dim // 4)
        )
        
        # Concatenate with input embeddings: [N, d, hidden_dim + hidden_dim//4]
        embedded_with_target = jnp.concatenate([embedded, target_broadcast], axis=-1)
        
        # Transformer layers for processing intervention data
        features = embedded_with_target
        for layer_idx in range(self.config.n_layers):
            features = self._transformer_layer(features, f"layer_{layer_idx}")
        
        # Global pooling to get dataset-level representation: [N, d, hidden_dim] -> [hidden_dim]
        pooled_features = jnp.mean(features, axis=(0, 1))
        
        # Enumerate possible parent sets for target variable
        max_parents = min(self.config.max_parents, d - 1)  # Can't include target itself
        possible_parent_sets = self._enumerate_possible_parent_sets(d, target_idx, max_parents)
        k = len(possible_parent_sets)
        
        # Parent set scoring: [hidden_dim] -> [k]
        parent_set_logits = self._score_parent_sets(pooled_features, possible_parent_sets, k)
        
        return {
            "parent_set_logits": parent_set_logits,
            "features": pooled_features,  # For mechanism prediction
            "possible_parent_sets": possible_parent_sets
        }
    
    def _predict_mechanisms(self,
                           features: jnp.ndarray,  # [hidden_dim] 
                           parent_set_logits: jnp.ndarray,  # [k]
                           variable_order: List[str],
                           target_variable: str) -> Dict[str, jnp.ndarray]:
        """
        Predict mechanism information for each parent set.
        
        This is the enhanced functionality that predicts HOW parents influence
        the target variable, not just WHICH variables are parents.
        """
        k = parent_set_logits.shape[0]
        n_mechanism_types = len(self.config.mechanism_types)
        
        # Mechanism type classification for each parent set
        # Features -> [k, n_mechanism_types] logits
        mechanism_classifier = hk.Sequential([
            hk.Linear(self.config.hidden_dim, w_init=self.w_init),
            jax.nn.relu,
            hk.Linear(self.config.hidden_dim // 2, w_init=self.w_init),
            jax.nn.relu,
            hk.Linear(k * n_mechanism_types, w_init=self.w_init)
        ])
        
        mechanism_type_logits = mechanism_classifier(features)
        mechanism_type_logits = jnp.reshape(mechanism_type_logits, (k, n_mechanism_types))
        
        # Mechanism parameter regression for each parent set and mechanism type
        # Features -> [k, n_mechanism_types, param_dim] parameters
        parameter_regressor = hk.Sequential([
            hk.Linear(self.config.hidden_dim, w_init=self.w_init),
            jax.nn.relu,
            hk.Linear(self.config.hidden_dim // 2, w_init=self.w_init),
            jax.nn.relu,
            hk.Linear(k * n_mechanism_types * self.config.parameter_dim, w_init=self.w_init)
        ])
        
        mechanism_parameters = parameter_regressor(features)
        mechanism_parameters = jnp.reshape(
            mechanism_parameters, 
            (k, n_mechanism_types, self.config.parameter_dim)
        )
        
        return {
            "mechanism_type_logits": mechanism_type_logits,
            "mechanism_parameters": mechanism_parameters
        }
    
    def _transformer_layer(self, x: jnp.ndarray, name: str) -> jnp.ndarray:
        """Transformer layer for processing intervention data."""
        with hk.experimental.name_scope(name):
            N, d, hidden_dim = x.shape
            
            # Layer normalization
            x_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            
            # Multi-head attention over variables (axis=1) for each sample
            def attention_over_vars(sample_data):
                # sample_data: [d, hidden_dim]
                attn_output = hk.MultiHeadAttention(
                    num_heads=8,
                    key_size=hidden_dim // 8,
                    w_init_scale=2.0,
                    model_size=hidden_dim
                )(sample_data, sample_data, sample_data)
                return attn_output
            
            # Apply attention to each sample independently
            attn_outputs = jax.vmap(attention_over_vars)(x_norm)
            
            # Residual connection
            x = x + hk.dropout(hk.next_rng_key(), self.config.dropout, attn_outputs)
            
            # Feed-forward network
            x_norm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            
            ff_output = hk.Sequential([
                hk.Linear(4 * hidden_dim, w_init=self.w_init),
                jax.nn.relu,
                hk.Linear(hidden_dim, w_init=self.w_init)
            ])(x_norm2)
            
            # Residual connection
            x = x + hk.dropout(hk.next_rng_key(), self.config.dropout, ff_output)
            
            return x
    
    def _enumerate_possible_parent_sets(self, 
                                       d: int, 
                                       target_idx: int, 
                                       max_parents: int) -> List[FrozenSet[int]]:
        """
        Enumerate possible parent sets for the target variable.
        
        Returns list of parent sets as frozensets of variable indices.
        Excludes the target variable itself.
        """
        possible_parents = list(range(d))
        possible_parents.remove(target_idx)  # Can't be parent of itself
        
        parent_sets = []
        
        # Add empty parent set
        parent_sets.append(frozenset())
        
        # Add parent sets of increasing size
        from itertools import combinations
        for size in range(1, min(max_parents + 1, len(possible_parents) + 1)):
            for parent_combo in combinations(possible_parents, size):
                parent_sets.append(frozenset(parent_combo))
        
        return parent_sets
    
    def _score_parent_sets(self, 
                          features: jnp.ndarray,  # [hidden_dim]
                          possible_parent_sets: List[FrozenSet[int]], 
                          k: int) -> jnp.ndarray:
        """Score each possible parent set using learned features."""
        # Simple MLP-based scoring for now
        # In practice, this could use more sophisticated architectures
        
        parent_set_scorer = hk.Sequential([
            hk.Linear(self.config.hidden_dim, w_init=self.w_init),
            jax.nn.relu,
            hk.Linear(self.config.hidden_dim // 2, w_init=self.w_init),
            jax.nn.relu,
            hk.Linear(k, w_init=self.w_init)
        ])
        
        parent_set_logits = parent_set_scorer(features)
        return parent_set_logits


# ============================================================================
# Factory Functions and High-Level API
# ============================================================================

def create_modular_parent_set_model(config: MechanismAwareConfig) -> hk.Transformed:
    """
    Create and initialize modular parent set model.
    
    Args:
        config: Model configuration specifying mode and parameters
        
    Returns:
        Transformed Haiku model ready for training/inference
    """
    def model_fn(x: jnp.ndarray, variable_order: List[str], target_variable: str):
        model = ModularParentSetModel(config)
        return model(x, variable_order, target_variable)
    
    return hk.transform(model_fn)


def predict_with_mechanisms(
    net: hk.Transformed,
    params: Any,
    x: jnp.ndarray,
    variable_order: List[str],
    target_variable: str,
    config: Optional[MechanismAwareConfig] = None,
    key: Optional[jax.Array] = None
) -> ParentSetPosterior:
    """
    High-level prediction function with mechanism awareness.
    
    This function works with both structure-only and enhanced models,
    returning a ParentSetPosterior with optional mechanism information.
    
    Args:
        net: Transformed model from create_modular_parent_set_model
        params: Model parameters
        x: Input data [N, d, 3]
        variable_order: Variable names in order
        target_variable: Target variable name
        config: Optional model config (needed for mechanism type mapping)
        key: Optional random key
        
    Returns:
        ParentSetPosterior with structure and optionally mechanism predictions
    """
    if key is None:
        key = random.PRNGKey(42)
    
    if config is None:
        config = MechanismAwareConfig()
    
    # Forward pass
    output = net.apply(params, key, x, variable_order, target_variable)
    
    parent_set_logits = output["parent_set_logits"]
    parent_set_probs = jax.nn.softmax(parent_set_logits)
    
    # Create parent sets from indices (this is simplified - real implementation
    # would need to map from indices back to variable names)
    target_idx = variable_order.index(target_variable)
    d = len(variable_order)
    max_parents = min(5, d - 1)  # Match model's enumeration
    
    # Enumerate same parent sets as model
    possible_parents = [i for i in range(d) if i != target_idx]
    parent_sets = [frozenset()]  # Empty set
    
    from itertools import combinations
    for size in range(1, min(max_parents + 1, len(possible_parents) + 1)):
        for parent_combo in combinations(possible_parents, size):
            parent_sets.append(frozenset(parent_combo))
    
    # Convert indices to variable names
    parent_sets_named = []
    for parent_set_indices in parent_sets[:len(parent_set_probs)]:
        parent_set_names = frozenset(variable_order[i] for i in parent_set_indices)
        parent_sets_named.append(parent_set_names)
    
    # Create basic posterior
    metadata = pyr.m()
    
    # Add mechanism predictions if available
    if "mechanism_predictions" in output:
        mechanism_preds = output["mechanism_predictions"]
        mechanism_type_logits = mechanism_preds["mechanism_type_logits"]
        mechanism_parameters = mechanism_preds["mechanism_parameters"]
        
        # Convert to MechanismPrediction objects
        mechanism_predictions = []
        
        for i, (parent_set, prob) in enumerate(zip(parent_sets_named, parent_set_probs)):
            if i >= len(mechanism_type_logits):
                break
                
            # Get most likely mechanism type for this parent set
            mech_type_probs = jax.nn.softmax(mechanism_type_logits[i])
            best_mech_idx = jnp.argmax(mech_type_probs)
            best_mech_type = config.mechanism_types[int(best_mech_idx)]
            mech_confidence = float(mech_type_probs[best_mech_idx])
            
            # Extract parameters (simplified - real implementation would 
            # parse parameters based on mechanism type)
            raw_params = mechanism_parameters[i, best_mech_idx]
            parsed_params = {"raw_parameters": raw_params.tolist()}
            
            mech_pred = MechanismPrediction(
                parent_set=parent_set,
                mechanism_type=best_mech_type,
                parameters=parsed_params,
                confidence=mech_confidence
            )
            mechanism_predictions.append(mech_pred)
        
        metadata = metadata.set("mechanism_predictions", mechanism_predictions)
    
    return create_parent_set_posterior(
        target_variable=target_variable,
        parent_sets=parent_sets_named,
        probabilities=parent_set_probs,
        metadata=dict(metadata)
    )


# ============================================================================
# Utility Functions
# ============================================================================

def get_all_mechanism_types() -> List[str]:
    """Get all supported mechanism types."""
    return [
        MechanismType.LINEAR,
        MechanismType.POLYNOMIAL, 
        MechanismType.GAUSSIAN,
        MechanismType.NEURAL
    ]


def validate_mechanism_types(mechanism_types: List[str]) -> bool:
    """Validate a list of mechanism types."""
    if not mechanism_types:
        return False
    
    valid_types = set(get_all_mechanism_types())
    return all(mech_type in valid_types for mech_type in mechanism_types)


def create_structure_only_config(**kwargs) -> MechanismAwareConfig:
    """Create configuration for structure-only mode (backward compatibility)."""
    return MechanismAwareConfig(predict_mechanisms=False, **kwargs)


def create_enhanced_config(mechanism_types: List[str] = None, **kwargs) -> MechanismAwareConfig:
    """Create configuration for enhanced mechanism-aware mode."""
    if mechanism_types is None:
        mechanism_types = ["linear", "polynomial"]
    
    return MechanismAwareConfig(
        predict_mechanisms=True,
        mechanism_types=mechanism_types,
        **kwargs
    )


def compare_model_outputs(structure_output: Dict[str, Any], 
                         enhanced_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare outputs from structure-only and enhanced models for analysis.
    
    Args:
        structure_output: Output from structure-only model
        enhanced_output: Output from enhanced model
        
    Returns:
        Comparison metrics and analysis
    """
    comparison = {
        "structure_only_keys": list(structure_output.keys()),
        "enhanced_keys": list(enhanced_output.keys()),
        "shared_keys": list(set(structure_output.keys()) & set(enhanced_output.keys())),
        "enhanced_only_keys": list(set(enhanced_output.keys()) - set(structure_output.keys()))
    }
    
    # Compare parent set predictions if both present
    both_outputs = structure_output.keys() & enhanced_output.keys()
    if "parent_set_logits" in both_outputs:
        struct_logits = structure_output["parent_set_logits"]
        enhanced_logits = enhanced_output["parent_set_logits"]
        
        # Compute similarity metrics
        if struct_logits.shape == enhanced_logits.shape:
            cosine_sim = jnp.dot(struct_logits, enhanced_logits) / (
                jnp.linalg.norm(struct_logits) * jnp.linalg.norm(enhanced_logits)
            )
            l2_distance = jnp.linalg.norm(struct_logits - enhanced_logits)
            
            comparison.update({
                "parent_set_cosine_similarity": float(cosine_sim),
                "parent_set_l2_distance": float(l2_distance),
                "logits_shape_match": True
            })
        else:
            comparison["logits_shape_match"] = False
    
    return comparison