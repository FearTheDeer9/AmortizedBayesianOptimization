"""
JAX-Native Tensor-Based Mechanism Features

This module provides a completely tensor-based mechanism feature system that eliminates
all dictionary lookups and Python loops for true JAX compilation compatibility.

Key innovations:
1. MechanismFeaturesTensor: Pure JAX array representation
2. Vectorized feature extraction with no loops
3. Integer-based indexing for all mechanism types
4. Direct tensor operations for confidence and scaling
"""

import warnings

warnings.warn(
    "This module is deprecated as of Phase 1.5. "
    "Use causal_bayes_opt.jax_native.operations.compute_policy_features_jax instead. "
    "See docs/migration/MIGRATION_GUIDE.md for migration instructions. "
    "This module will be removed on 2024-02-01.",
    DeprecationWarning,
    stacklevel=2
)


from dataclasses import dataclass
from typing import List, Dict, Optional, Any, FrozenSet
import jax
import jax.numpy as jnp
import pyrsistent as pyr


# Mechanism type encoding (same as before but now central)
MECHANISM_TYPE_ENCODING = {
    'linear': 0,
    'polynomial': 1, 
    'gaussian': 2,
    'neural': 3,
    'unknown': 4
}

# Reverse mapping for interpretation
MECHANISM_TYPE_NAMES = {v: k for k, v in MECHANISM_TYPE_ENCODING.items()}

# Mechanism scaling factors as JAX array (indexed by mechanism type)
MECHANISM_SCALING_FACTORS = jnp.array([
    1.0,   # linear: Standard scaling
    0.5,   # polynomial: Smaller interventions for nonlinear effects  
    1.2,   # gaussian: Slightly larger for Gaussian processes
    0.8,   # neural: Moderate scaling for neural network mechanisms
    1.0    # unknown: Default scaling
])


@dataclass(frozen=True)
class MechanismFeaturesTensor:
    """
    Pure tensor representation of mechanism features for JAX compilation.
    
    All data is stored as JAX arrays indexed by variable position, eliminating
    the need for dictionary lookups or string operations.
    """
    # Core tensor data (all JAX arrays)
    predicted_effects: jnp.ndarray      # [n_vars] - Predicted coefficient magnitudes
    mechanism_types: jnp.ndarray        # [n_vars] - Integer mechanism type indices
    confidences: jnp.ndarray           # [n_vars] - Confidence scores [0, 1]
    uncertainties: jnp.ndarray          # [n_vars] - Uncertainty scores [0, 1]
    
    # Variable mapping for interpretation (not used in JAX-compiled code)
    variable_order: List[str]
    
    def __post_init__(self):
        """Validate tensor shapes and contents."""
        n_vars = len(self.variable_order)
        
        # Check all tensors have correct shape
        for field_name in ['predicted_effects', 'mechanism_types', 'confidences', 'uncertainties']:
            tensor = getattr(self, field_name)
            if tensor.shape != (n_vars,):
                raise ValueError(f"{field_name} has shape {tensor.shape}, expected ({n_vars},)")
        
        # Check mechanism types are valid indices
        max_type_idx = len(MECHANISM_TYPE_ENCODING) - 1
        if jnp.any(self.mechanism_types < 0) or jnp.any(self.mechanism_types > max_type_idx):
            raise ValueError(f"Mechanism types must be in range [0, {max_type_idx}]")
    
    @property
    def n_vars(self) -> int:
        """Number of variables."""
        return len(self.variable_order)


def create_mechanism_features_tensor(
    variable_order: List[str],
    mechanism_insights: Dict[str, Any],
    mechanism_confidence: Dict[str, float]
) -> MechanismFeaturesTensor:
    """
    Create tensor representation from traditional dictionary format.
    
    This function handles the conversion from legacy dict format to pure tensors.
    It will be called once during data preparation, not in JAX-compiled code.
    """
    n_vars = len(variable_order)
    
    # Initialize arrays with defaults
    predicted_effects = jnp.ones(n_vars)  # Default effect magnitude
    mechanism_types = jnp.full(n_vars, MECHANISM_TYPE_ENCODING['unknown'], dtype=jnp.int32)
    confidences = jnp.full(n_vars, 0.5)  # Default medium confidence
    uncertainties = jnp.full(n_vars, 0.5)  # Default medium uncertainty
    
    # Extract data from dictionaries (this happens outside JAX compilation)
    effects_dict = mechanism_insights.get('predicted_effects', {})
    types_dict = mechanism_insights.get('mechanism_types', {})
    
    # Convert to arrays using efficient vectorized operations where possible
    for i, var_name in enumerate(variable_order):
        # Predicted effects
        if var_name in effects_dict:
            effect = effects_dict[var_name]
            predicted_effects = predicted_effects.at[i].set(abs(effect) if effect != 0 else 1.0)
        
        # Mechanism types  
        if var_name in types_dict:
            mech_type = types_dict[var_name]
            type_idx = MECHANISM_TYPE_ENCODING.get(mech_type, MECHANISM_TYPE_ENCODING['unknown'])
            mechanism_types = mechanism_types.at[i].set(type_idx)
        
        # Confidence scores
        if var_name in mechanism_confidence:
            confidence = mechanism_confidence[var_name]
            confidences = confidences.at[i].set(confidence)
            uncertainties = uncertainties.at[i].set(1.0 - confidence)
    
    return MechanismFeaturesTensor(
        predicted_effects=predicted_effects,
        mechanism_types=mechanism_types,
        confidences=confidences,
        uncertainties=uncertainties,
        variable_order=variable_order
    )


def compute_mechanism_features_tensor(features_tensor: MechanismFeaturesTensor) -> jnp.ndarray:
    """
    Compute mechanism features from tensor representation (NOT JAX-compiled).
    
    This function extracts arrays from the MechanismFeaturesTensor dataclass
    and delegates to the JAX-compiled version for actual computation.
    
    Args:
        features_tensor: Tensor representation of mechanism features
        
    Returns:
        JAX array of shape [n_vars, 3] with:
        [:, 0] = predicted_effects
        [:, 1] = uncertainties  
        [:, 2] = mechanism_scaling_factors
    """
    # Delegate to JAX-compiled version with pure arrays
    return compute_mechanism_features_from_arrays_jit(
        features_tensor.predicted_effects,
        features_tensor.mechanism_types,
        features_tensor.uncertainties
    )


# Create a separate JAX-compiled version that works with pure arrays
@jax.jit
def compute_mechanism_features_from_arrays_jit(
    predicted_effects: jnp.ndarray,
    mechanism_types: jnp.ndarray,
    uncertainties: jnp.ndarray
) -> jnp.ndarray:
    """JAX-compiled function using pure arrays instead of dataclass."""
    scaling_factors = MECHANISM_SCALING_FACTORS[mechanism_types]
    return jnp.stack([predicted_effects, uncertainties, scaling_factors], axis=1)


@jax.jit
def extract_high_impact_variables_jit(
    features_tensor: MechanismFeaturesTensor,
    impact_threshold: float = 0.5
) -> jnp.ndarray:
    """
    JAX-compiled function to identify high-impact variables.
    
    Args:
        features_tensor: Mechanism features
        impact_threshold: Minimum effect magnitude for high impact
        
    Returns:
        Boolean array [n_vars] indicating high-impact variables
    """
    return features_tensor.predicted_effects > impact_threshold


@jax.jit  
def extract_uncertain_mechanisms_jit(
    features_tensor: MechanismFeaturesTensor,
    marginal_parent_probs: jnp.ndarray,
    uncertainty_threshold: float = 0.5,
    parent_prob_threshold: float = 0.3
) -> jnp.ndarray:
    """
    JAX-compiled function to identify uncertain mechanisms.
    
    Args:
        features_tensor: Mechanism features
        marginal_parent_probs: [n_vars] probability each variable is a parent
        uncertainty_threshold: Minimum uncertainty for uncertain mechanism
        parent_prob_threshold: Minimum parent probability to consider
        
    Returns:
        Boolean array [n_vars] indicating uncertain mechanisms
    """
    high_uncertainty = features_tensor.uncertainties > uncertainty_threshold
    likely_parent = marginal_parent_probs > parent_prob_threshold
    return high_uncertainty & likely_parent


def create_target_mask_tensor_jit(n_vars: int, target_idx: int) -> jnp.ndarray:
    """
    JAX-compiled function to create target variable mask.
    
    Args:
        n_vars: Number of variables
        target_idx: Index of target variable
        
    Returns:
        Mask array with 0 for non-target, -inf for target
    """
    mask = jnp.zeros(n_vars)
    return mask.at[target_idx].set(-jnp.inf)


# Compile the target mask function
create_target_mask_tensor_jit = jax.jit(create_target_mask_tensor_jit, static_argnums=(0,))


def convert_mechanism_insights_to_tensor(
    mechanism_insights: Dict[str, Any],
    mechanism_confidence: Dict[str, float],
    variable_order: List[str]
) -> MechanismFeaturesTensor:
    """
    Convert legacy mechanism insights to tensor format.
    
    This handles the conversion from the current dictionary-based format
    to the new tensor-based format for JAX compilation.
    """
    return create_mechanism_features_tensor(
        variable_order=variable_order,
        mechanism_insights=mechanism_insights, 
        mechanism_confidence=mechanism_confidence
    )


def interpret_mechanism_tensor_results(
    features_tensor: MechanismFeaturesTensor,
    high_impact_mask: jnp.ndarray,
    uncertain_mask: jnp.ndarray
) -> Dict[str, Any]:
    """
    Convert tensor results back to interpretable format.
    
    This function provides backward compatibility by converting tensor results
    back to the dictionary format expected by existing code.
    """
    variable_order = features_tensor.variable_order
    
    # Convert masks to variable lists
    high_impact_variables = [
        var for i, var in enumerate(variable_order) 
        if high_impact_mask[i]
    ]
    
    uncertain_mechanisms = [
        var for i, var in enumerate(variable_order)
        if uncertain_mask[i]
    ]
    
    # Convert effects and types back to dictionaries
    predicted_effects = {
        var: float(features_tensor.predicted_effects[i])
        for i, var in enumerate(variable_order)
    }
    
    mechanism_types = {
        var: MECHANISM_TYPE_NAMES[int(features_tensor.mechanism_types[i])]
        for i, var in enumerate(variable_order)
    }
    
    return {
        'mechanism_aware': True,
        'high_impact_variables': high_impact_variables,
        'uncertain_mechanisms': uncertain_mechanisms,
        'predicted_effects': predicted_effects,
        'mechanism_types': mechanism_types,
        'mechanism_confidence_avg': float(jnp.mean(features_tensor.confidences))
    }


# Backward compatibility function
def create_mechanism_features_vectorized(
    variable_order: List[str],
    mechanism_insights: Dict,
    mechanism_confidence: Dict[str, float]
) -> jnp.ndarray:
    """
    Backward compatibility function that uses the new tensor system.
    
    This replaces the old dictionary-based function with the new tensor approach.
    """
    # Convert to tensor format
    tensor_features = create_mechanism_features_tensor(
        variable_order, mechanism_insights, mechanism_confidence
    )
    
    # Use JAX-compiled computation
    return compute_mechanism_features_tensor(tensor_features)