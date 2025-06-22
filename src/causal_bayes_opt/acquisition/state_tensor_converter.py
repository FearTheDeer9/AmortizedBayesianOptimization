"""
State to Tensor Conversion for JAX-Native Policy Networks

⚠️  DEPRECATED: This module is deprecated as of Phase 1.5. Use the JAX-native
    architecture in `causal_bayes_opt.jax_native` for optimal performance.

    Migration path:
    - Replace with `causal_bayes_opt.jax_native.state.JAXAcquisitionState`
    - Use direct tensor operations instead of conversion functions
    - See docs/migration/MIGRATION_GUIDE.md for details

This module handles the conversion from AcquisitionState (which contains dictionaries
and complex data structures) to pure tensor representations suitable for JAX compilation.

Key functions:
1. Convert AcquisitionState to tensor format
2. Prepare data for vectorized policy networks
3. Maintain backward compatibility with existing interfaces
"""

from typing import Dict, List, Any, Optional
import warnings
import jax.numpy as jnp
import pyrsistent as pyr

from .tensor_features import (
    MechanismFeaturesTensor, 
    create_mechanism_features_tensor,
    convert_mechanism_insights_to_tensor
)
from .jax_utils import prepare_samples_for_history_jax


def convert_acquisition_state_to_tensors(
    state,  # AcquisitionState (avoiding import for now)
    max_history_size: int = 100
) -> Dict[str, Any]:
    """
    Convert AcquisitionState to tensor format for vectorized policy networks.
    
    DEPRECATED: Use JAXAcquisitionState from causal_bayes_opt.jax_native instead.
    
    This function bridges the gap between the current dictionary-based state
    representation and the pure tensor format needed for JAX compilation.
    
    Args:
        state: AcquisitionState object with mechanism predictions
        max_history_size: Fixed size for history tensor
        
    Returns:
        Dictionary with tensor representations suitable for JAX compilation
    """
    warnings.warn(
        "convert_acquisition_state_to_tensors() is deprecated as of Phase 1.5. "
        "Use JAXAcquisitionState from causal_bayes_opt.jax_native for optimal performance. "
        "See docs/migration/MIGRATION_GUIDE.md for migration instructions.",
        DeprecationWarning,
        stacklevel=2
    )
    # Get basic information
    variable_order = sorted(state.buffer.get_variable_coverage())
    n_vars = len(variable_order)
    
    if n_vars == 0:
        # Handle empty case
        return _create_empty_tensor_data(max_history_size)
    
    # Convert history to fixed-size tensor
    all_samples = state.buffer.get_all_samples()
    history_tensor = prepare_samples_for_history_jax(
        all_samples, variable_order, state.current_target, max_history_size
    )
    
    # Convert mechanism features to tensor format
    mechanism_features = create_mechanism_features_tensor(
        variable_order=variable_order,
        mechanism_insights=state.get_mechanism_insights(),
        mechanism_confidence=state.mechanism_confidence
    )
    
    # Convert marginal probabilities to tensor
    marginal_probs = jnp.array([
        state.marginal_parent_probs.get(var, 0.0) 
        for var in variable_order
    ])
    
    # Create context features tensor
    context_features = _create_context_features_tensor(state, variable_order)
    
    # Find target index for masking
    try:
        target_idx = variable_order.index(state.current_target)
    except ValueError:
        target_idx = -1  # Invalid target
    
    return {
        'history': history_tensor,                    # [max_history_size, n_vars, 3]
        'mechanism_features': mechanism_features,     # MechanismFeaturesTensor
        'marginal_probs': marginal_probs,            # [n_vars]
        'context_features': context_features,         # [n_vars, k]
        'target_idx': target_idx,                    # int
        'variable_order': variable_order,            # List[str] (for interpretation)
        'n_vars': n_vars                             # int
    }


def _create_context_features_tensor(state, variable_order: List[str]) -> jnp.ndarray:
    """
    Create context features tensor from state information.
    
    Combines various scalar state features into a tensor format suitable
    for vectorized processing.
    
    Returns:
        JAX array of shape [n_vars, k] with context features
    """
    n_vars = len(variable_order)
    
    # Global context features (broadcast to all variables)
    global_features = jnp.array([
        state.best_value,                           # Current best target value
        state.uncertainty_bits,                     # Posterior uncertainty 
        float(state.step),                          # Current step number
        float(state.buffer_statistics.total_samples),    # Total samples
        float(state.buffer_statistics.num_interventions) # Number of interventions
    ])
    
    # Broadcast global features to all variables: [n_vars, 5]
    global_broadcasted = jnp.tile(global_features[None, :], (n_vars, 1))
    
    # Variable-specific features
    optimization_progress = state.get_optimization_progress()
    exploration_coverage = state.get_exploration_coverage()
    
    # Create variable-specific features [n_vars, 3]
    var_specific_features = jnp.array([
        [
            optimization_progress['improvement_from_start'],
            optimization_progress['optimization_rate'], 
            exploration_coverage['target_coverage_rate']
        ] for _ in range(n_vars)
    ])
    
    # Combine: [n_vars, 8]
    return jnp.concatenate([global_broadcasted, var_specific_features], axis=1)


def _create_empty_tensor_data(max_history_size: int) -> Dict[str, Any]:
    """Create empty tensor data for edge cases."""
    return {
        'history': jnp.zeros((max_history_size, 1, 3)),
        'mechanism_features': MechanismFeaturesTensor(
            predicted_effects=jnp.array([1.0]),
            mechanism_types=jnp.array([4], dtype=jnp.int32),  # 'unknown'
            confidences=jnp.array([0.5]),
            uncertainties=jnp.array([0.5]),
            variable_order=['dummy']
        ),
        'marginal_probs': jnp.array([0.5]),
        'context_features': jnp.zeros((1, 8)),
        'target_idx': 0,
        'variable_order': ['dummy'],
        'n_vars': 1
    }


def apply_target_mask_to_logits(
    variable_logits: jnp.ndarray,
    target_idx: int,
    mask_value: float = -jnp.inf
) -> jnp.ndarray:
    """
    Apply target variable mask to logits (JAX-compiled compatible).
    
    Args:
        variable_logits: [n_vars] logits for variable selection
        target_idx: Index of target variable to mask
        mask_value: Value to use for masking (default: -inf)
        
    Returns:
        Masked logits with target variable set to mask_value
    """
    if target_idx < 0 or target_idx >= len(variable_logits):
        # Invalid target index, return unchanged
        return variable_logits
    
    return variable_logits.at[target_idx].set(mask_value)


def create_policy_input_from_state(state, config=None) -> Dict[str, Any]:
    """
    High-level function to create policy input from AcquisitionState.
    
    This is the main interface function that existing code can use to convert
    from the current state format to the new tensor format.
    
    Args:
        state: AcquisitionState object
        config: Optional policy configuration
        
    Returns:
        Dictionary suitable for vectorized policy network
    """
    max_history_size = getattr(config, 'max_history_size', 100) if config else 100
    
    return convert_acquisition_state_to_tensors(state, max_history_size)


# Backward compatibility wrapper
class TensorizedAcquisitionState:
    """
    Wrapper that provides tensor-based interface while maintaining compatibility.
    
    This allows existing code to gradually migrate to the tensor-based approach
    without breaking existing functionality.
    """
    
    def __init__(self, original_state, max_history_size: int = 100):
        self.original_state = original_state
        self.tensor_data = convert_acquisition_state_to_tensors(
            original_state, max_history_size
        )
    
    @property
    def history_tensor(self) -> jnp.ndarray:
        """Get history as JAX tensor."""
        return self.tensor_data['history']
    
    @property  
    def mechanism_features_tensor(self) -> MechanismFeaturesTensor:
        """Get mechanism features as tensor."""
        return self.tensor_data['mechanism_features']
    
    @property
    def marginal_probs_tensor(self) -> jnp.ndarray:
        """Get marginal probabilities as JAX tensor."""
        return self.tensor_data['marginal_probs']
    
    @property
    def context_features_tensor(self) -> jnp.ndarray:
        """Get context features as JAX tensor."""
        return self.tensor_data['context_features']
    
    def get_policy_input(self) -> Dict[str, Any]:
        """Get input suitable for vectorized policy network."""
        return self.tensor_data
    
    # Delegate other attributes to original state
    def __getattr__(self, name):
        return getattr(self.original_state, name)