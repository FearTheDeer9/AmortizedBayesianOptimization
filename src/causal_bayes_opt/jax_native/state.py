"""
JAX-Native Acquisition State

Provides immutable acquisition state using pure JAX tensors with all
computations performed by JAX-compiled functions.

Key features:
- Pure tensor fields with no dictionary operations
- Immutable state following functional programming principles
- All state computations via JAX-compiled functions
- Type safety with comprehensive hints
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import jax
import jax.numpy as jnp

from .config import JAXConfig
from .sample_buffer import JAXSampleBuffer


@dataclass(frozen=True)
class JAXAcquisitionState:
    """
    Immutable acquisition state using pure JAX tensors.
    
    All state information is stored as JAX arrays or primitives to enable
    optimal compilation. State computations are performed by pure JAX functions.
    
    Args:
        sample_buffer: Circular buffer with sample history
        mechanism_features: Mechanism feature tensor [n_vars, feature_dim]
        marginal_probs: Parent probability estimates [n_vars]
        confidence_scores: Mechanism confidence scores [n_vars]
        best_value: Current best target value
        current_step: Current optimization step
        uncertainty_bits: Posterior uncertainty estimate
        config: Static configuration
    """
    
    # Core tensor fields
    sample_buffer: JAXSampleBuffer
    mechanism_features: jnp.ndarray      # [n_vars, feature_dim]
    marginal_probs: jnp.ndarray          # [n_vars] - parent probabilities  
    confidence_scores: jnp.ndarray       # [n_vars] - mechanism confidence
    
    # Scalar state
    best_value: float
    current_step: int
    uncertainty_bits: float
    
    # Static configuration
    config: JAXConfig
    
    def __post_init__(self) -> None:
        """Validate state consistency."""
        # Shape validation
        expected_mech_shape = (self.config.n_vars, self.config.feature_dim)
        if self.mechanism_features.shape != expected_mech_shape:
            raise ValueError(
                f"mechanism_features shape {self.mechanism_features.shape} != {expected_mech_shape}"
            )
        
        expected_prob_shape = (self.config.n_vars,)
        if self.marginal_probs.shape != expected_prob_shape:
            raise ValueError(
                f"marginal_probs shape {self.marginal_probs.shape} != {expected_prob_shape}"
            )
        
        if self.confidence_scores.shape != expected_prob_shape:
            raise ValueError(
                f"confidence_scores shape {self.confidence_scores.shape} != {expected_prob_shape}"
            )
        
        # Value validation
        if not jnp.isfinite(self.best_value):
            raise ValueError(f"best_value must be finite, got {self.best_value}")
        
        if self.current_step < 0:
            raise ValueError(f"current_step must be non-negative, got {self.current_step}")
        
        if not jnp.isfinite(self.uncertainty_bits) or self.uncertainty_bits < 0:
            raise ValueError(f"uncertainty_bits must be non-negative and finite")
        
        # Buffer config consistency
        if self.sample_buffer.config != self.config:
            raise ValueError("Sample buffer config must match state config")
    
    def get_target_name(self) -> str:
        """Get target variable name."""
        return self.config.get_target_name()
    
    def get_n_samples(self) -> int:
        """Get number of samples in buffer."""
        return self.sample_buffer.n_samples
    
    def is_buffer_empty(self) -> bool:
        """Check if sample buffer is empty."""
        return self.sample_buffer.is_empty()


def create_jax_state(
    config: JAXConfig,
    sample_buffer: Optional[JAXSampleBuffer] = None,
    mechanism_features: Optional[jnp.ndarray] = None,
    marginal_probs: Optional[jnp.ndarray] = None,
    confidence_scores: Optional[jnp.ndarray] = None,
    best_value: float = 0.0,
    current_step: int = 0,
    uncertainty_bits: float = 1.0
) -> JAXAcquisitionState:
    """
    Create JAX acquisition state with default values.
    
    Args:
        config: Static configuration
        sample_buffer: Optional sample buffer (creates empty if None)
        mechanism_features: Optional mechanism features (creates defaults if None)
        marginal_probs: Optional marginal probabilities (creates defaults if None)
        confidence_scores: Optional confidence scores (creates defaults if None)
        best_value: Initial best value
        current_step: Initial step number
        uncertainty_bits: Initial uncertainty estimate
        
    Returns:
        Initialized JAX acquisition state
    """
    from .sample_buffer import create_empty_jax_buffer
    
    # Create defaults if not provided
    if sample_buffer is None:
        sample_buffer = create_empty_jax_buffer(config)
    
    if mechanism_features is None:
        # Default mechanism features: [effect=1.0, uncertainty=0.5, confidence=0.5]
        mechanism_features = jnp.ones((config.n_vars, config.feature_dim)) * 0.5
        mechanism_features = mechanism_features.at[:, 0].set(1.0)  # Default effect size
    
    if marginal_probs is None:
        # Default marginal probabilities (uniform, target excluded)
        marginal_probs = jnp.ones(config.n_vars) * 0.5
        marginal_probs = marginal_probs.at[config.target_idx].set(0.0)
    
    if confidence_scores is None:
        # Default confidence scores (medium confidence, target excluded)
        confidence_scores = jnp.ones(config.n_vars) * 0.5
        confidence_scores = confidence_scores.at[config.target_idx].set(0.0)
    
    return JAXAcquisitionState(
        sample_buffer=sample_buffer,
        mechanism_features=mechanism_features,
        marginal_probs=marginal_probs,
        confidence_scores=confidence_scores,
        best_value=best_value,
        current_step=current_step,
        uncertainty_bits=uncertainty_bits,
        config=config
    )


@jax.jit
def update_jax_state(
    state: JAXAcquisitionState,
    new_sample_buffer: Optional[JAXSampleBuffer] = None,
    new_mechanism_features: Optional[jnp.ndarray] = None,
    new_marginal_probs: Optional[jnp.ndarray] = None,
    new_confidence_scores: Optional[jnp.ndarray] = None,
    new_best_value: Optional[float] = None,
    new_step: Optional[int] = None,
    new_uncertainty: Optional[float] = None
) -> JAXAcquisitionState:
    """
    Update JAX acquisition state with new values (JAX-compiled).
    
    Args:
        state: Current state
        new_*: Optional new values for each field
        
    Returns:
        Updated state with new values
    """
    return JAXAcquisitionState(
        sample_buffer=new_sample_buffer if new_sample_buffer is not None else state.sample_buffer,
        mechanism_features=new_mechanism_features if new_mechanism_features is not None else state.mechanism_features,
        marginal_probs=new_marginal_probs if new_marginal_probs is not None else state.marginal_probs,
        confidence_scores=new_confidence_scores if new_confidence_scores is not None else state.confidence_scores,
        best_value=new_best_value if new_best_value is not None else state.best_value,
        current_step=new_step if new_step is not None else state.current_step,
        uncertainty_bits=new_uncertainty if new_uncertainty is not None else state.uncertainty_bits,
        config=state.config
    )


@jax.jit
def add_sample_to_state_jax(
    state: JAXAcquisitionState,
    variable_values: jnp.ndarray,   # [n_vars]
    intervention_mask: jnp.ndarray, # [n_vars] boolean
    target_value: float
) -> JAXAcquisitionState:
    """
    Add a sample to the state buffer (JAX-compiled).
    
    Args:
        state: Current state
        variable_values: Values for all variables
        intervention_mask: Boolean intervention indicators
        target_value: Target variable value
        
    Returns:
        State with updated buffer and best value
    """
    from .sample_buffer import add_sample_jax
    
    # Add sample to buffer
    new_buffer = add_sample_jax(
        state.sample_buffer, variable_values, intervention_mask, target_value
    )
    
    # Update best value if this target is better
    new_best_value = jnp.maximum(state.best_value, target_value)
    
    # Increment step
    new_step = state.current_step + 1
    
    return JAXAcquisitionState(
        sample_buffer=new_buffer,
        mechanism_features=state.mechanism_features,
        marginal_probs=state.marginal_probs,
        confidence_scores=state.confidence_scores,
        best_value=float(new_best_value),
        current_step=new_step,
        uncertainty_bits=state.uncertainty_bits,
        config=state.config
    )


@jax.jit
def get_policy_input_tensor_jax(state: JAXAcquisitionState) -> jnp.ndarray:
    """
    Extract policy input tensor from state (JAX-compiled).
    
    Combines all relevant state information into a single tensor suitable
    for policy network input.
    
    Args:
        state: JAX acquisition state
        
    Returns:
        Policy input tensor [n_vars, total_features]
    """
    # Get basic features
    n_vars = state.config.n_vars
    
    # Mechanism features: [n_vars, feature_dim]
    mech_features = state.mechanism_features
    
    # Marginal probabilities: [n_vars, 1]
    marginal_features = state.marginal_probs[:, None]
    
    # Confidence scores: [n_vars, 1]
    confidence_features = state.confidence_scores[:, None]
    
    # Global context features (broadcast to all variables)
    global_features = jnp.array([
        state.best_value,
        state.uncertainty_bits,
        float(state.current_step),
        float(state.sample_buffer.n_samples)
    ])
    global_broadcasted = jnp.tile(global_features[None, :], (n_vars, 1))
    
    # Combine all features: [n_vars, total_dim]
    policy_input = jnp.concatenate([
        mech_features,           # [n_vars, feature_dim]
        marginal_features,       # [n_vars, 1]
        confidence_features,     # [n_vars, 1]
        global_broadcasted       # [n_vars, 4]
    ], axis=1)
    
    return policy_input


def create_test_state() -> JAXAcquisitionState:
    """Create a test state for unit testing."""
    from .config import create_test_config
    from .sample_buffer import create_test_buffer
    
    config = create_test_config()
    buffer = create_test_buffer()
    
    return create_jax_state(
        config=config,
        sample_buffer=buffer,
        best_value=1.5,
        current_step=5,
        uncertainty_bits=0.8
    )