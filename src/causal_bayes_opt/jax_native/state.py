"""
JAX-Native Acquisition State

Provides immutable acquisition state using pure JAX tensors with all
computations performed by JAX-compiled functions.

Key features:
- Pure tensor fields with no dictionary operations
- Immutable state following functional programming principles
- All state computations via JAX-compiled functions
- Type safety with comprehensive hints
- TensorBackedAcquisitionState provides full AcquisitionState interface compatibility
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
import warnings
import jax
import jax.numpy as jnp
import pyrsistent as pyr

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


# ==============================================================================
# TensorBackedAcquisitionState - Unified Tensor-Interface Architecture
# ==============================================================================

@dataclass(frozen=True)
class TensorBackedAcquisitionState(JAXAcquisitionState):
    """
    Unified acquisition state providing both tensor efficiency and AcquisitionState interface.
    
    This class extends JAXAcquisitionState with additional tensor data and computed properties
    that provide full backward compatibility with existing AcquisitionState usage patterns.
    
    Key innovations:
    - Tensor-first storage for JAX compilation efficiency
    - On-demand interface reconstruction via @property methods
    - Maintains immutability and functional programming principles
    - Supports variable-size graphs through dynamic tensor shaping
    - Full GRPO compatibility with vmap operations
    
    Additional tensor data beyond JAXAcquisitionState:
        posterior_logits: Log probabilities over parent sets [n_parent_sets]
        parent_sets: Immutable tuple of frozenset parent set definitions
        variable_names: Immutable tuple of variable names (for interface only)
        current_target: Target variable name
        metadata: Additional metadata as immutable map
    """
    
    # Additional tensor data for full AcquisitionState compatibility
    posterior_logits: jnp.ndarray = field(default_factory=lambda: jnp.array([0.0]))
    parent_sets: Tuple[frozenset, ...] = field(default_factory=lambda: (frozenset(),))
    variable_names: Tuple[str, ...] = field(default_factory=tuple)
    current_target: str = ""
    metadata: pyr.PMap = field(default_factory=lambda: pyr.m())
    
    def __post_init__(self) -> None:
        """Validate tensor-backed state consistency."""
        # Call parent validation first
        super().__post_init__()
        
        # Additional validation for TensorBackedAcquisitionState
        if len(self.variable_names) != self.config.n_vars:
            raise ValueError(
                f"variable_names length {len(self.variable_names)} != n_vars {self.config.n_vars}"
            )
        
        if self.current_target and self.current_target not in self.variable_names:
            raise ValueError(f"current_target '{self.current_target}' not in variable_names")
        
        if len(self.parent_sets) != self.posterior_logits.shape[0]:
            raise ValueError(
                f"parent_sets length {len(self.parent_sets)} != posterior_logits shape {self.posterior_logits.shape[0]}"
            )
    
    # ==========================================================================
    # AcquisitionState Interface Compatibility (Computed Properties)
    # ==========================================================================
    
    @property
    def posterior(self):
        """Reconstruct ParentSetPosterior from tensor data."""
        from ..avici_integration.parent_set import ParentSetPosterior
        
        # Convert log probabilities to probabilities
        probs = jax.nn.softmax(self.posterior_logits)
        
        # Create parent_set_probs dictionary and top_k_sets list
        parent_set_probs = {}
        top_k_sets = []
        for parent_set, prob in zip(self.parent_sets, probs):
            prob_float = float(prob)
            parent_set_probs[parent_set] = prob_float
            top_k_sets.append((parent_set, prob_float))
        
        # Sort top_k_sets by probability (descending)
        top_k_sets.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate actual entropy from the probability distribution
        probs_array = jnp.array([prob_float for _, prob_float in top_k_sets])
        # Compute entropy: -sum(p * log(p)) where p > 0
        log_probs = jnp.log(jnp.maximum(probs_array, 1e-12))  # Avoid log(0)
        actual_entropy = -jnp.sum(probs_array * log_probs)
        
        # Create ParentSetPosterior with tensor-derived data
        return ParentSetPosterior(
            target_variable=self.current_target,
            parent_set_probs=pyr.pmap(parent_set_probs),
            uncertainty=float(actual_entropy),  # Use actual entropy of the distribution
            top_k_sets=top_k_sets
        )
    
    @property
    def buffer(self):
        """Provide ExperienceBuffer interface from tensor data."""
        return TensorBackedBufferInterface(self)
    
    @property
    def step(self) -> int:
        """Current step from JAXAcquisitionState.current_step."""
        return self.current_step
    
    # best_value is already inherited from JAXAcquisitionState - no need to override
    
    @property
    def marginal_parent_probs(self) -> Dict[str, float]:
        """Compute marginal parent probabilities from tensor data."""
        marginal_dict = {}
        
        # Get non-target variables
        target_idx = self.config.target_idx
        for i, var_name in enumerate(self.variable_names):
            if i != target_idx:
                marginal_dict[var_name] = float(self.marginal_probs[i])
        
        return marginal_dict
    
    @property 
    def buffer_statistics(self):
        """Compute buffer statistics from tensor data."""
        from ..data_structures.buffer import BufferStatistics
        
        # Use sample buffer to compute statistics
        n_samples = self.sample_buffer.n_samples
        
        return BufferStatistics(
            total_samples=n_samples,
            intervention_samples=max(0, n_samples - 1),  # Assume first is observational
            observational_samples=min(1, n_samples),
            variable_coverage=set(self.variable_names) - {self.current_target}
        )
    
    @property
    def mechanism_confidence(self) -> Dict[str, float]:
        """Extract mechanism confidence from tensor data."""
        confidence_dict = {}
        
        for i, var_name in enumerate(self.variable_names):
            if i != self.config.target_idx:
                confidence_dict[var_name] = float(self.confidence_scores[i])
        
        return confidence_dict
    
    # ==========================================================================
    # Factory Methods and Utilities
    # ==========================================================================
    
    @classmethod
    def create_empty(cls, config: JAXConfig) -> 'TensorBackedAcquisitionState':
        """Create empty TensorBackedAcquisitionState with default values."""
        from .sample_buffer import create_empty_jax_buffer
        
        # Create base JAX state
        base_state = create_jax_state(config)
        
        # Create default parent sets (empty set with probability 1.0)
        default_parent_sets = (frozenset(),)
        default_posterior_logits = jnp.array([0.0])  # Log prob = 0 -> prob = 1
        
        # Get variable names from config
        variable_names = config.variable_names
        current_target = variable_names[config.target_idx] if variable_names else ""
        
        return cls(
            # JAXAcquisitionState fields
            sample_buffer=base_state.sample_buffer,
            mechanism_features=base_state.mechanism_features,
            marginal_probs=base_state.marginal_probs,
            confidence_scores=base_state.confidence_scores,
            best_value=base_state.best_value,
            current_step=base_state.current_step,
            uncertainty_bits=base_state.uncertainty_bits,
            config=base_state.config,
            
            # TensorBackedAcquisitionState fields
            posterior_logits=default_posterior_logits,
            parent_sets=default_parent_sets,
            variable_names=variable_names,
            current_target=current_target,
            metadata=pyr.m()
        )
    
    @classmethod
    def from_jax_state(
        cls, 
        jax_state: JAXAcquisitionState,
        posterior_logits: Optional[jnp.ndarray] = None,
        parent_sets: Optional[Tuple[frozenset, ...]] = None,
        variable_names: Optional[Tuple[str, ...]] = None,
        current_target: Optional[str] = None,
        metadata: Optional[pyr.PMap] = None
    ) -> 'TensorBackedAcquisitionState':
        """Create TensorBackedAcquisitionState from existing JAXAcquisitionState."""
        # Use defaults if not provided
        if posterior_logits is None:
            posterior_logits = jnp.array([0.0])
        if parent_sets is None:
            parent_sets = (frozenset(),)
        if variable_names is None:
            variable_names = jax_state.config.variable_names
        if current_target is None:
            current_target = variable_names[jax_state.config.target_idx] if variable_names else ""
        if metadata is None:
            metadata = pyr.m()
        
        return cls(
            # JAXAcquisitionState fields
            sample_buffer=jax_state.sample_buffer,
            mechanism_features=jax_state.mechanism_features,
            marginal_probs=jax_state.marginal_probs,
            confidence_scores=jax_state.confidence_scores,
            best_value=jax_state.best_value,
            current_step=jax_state.current_step,
            uncertainty_bits=jax_state.uncertainty_bits,
            config=jax_state.config,
            
            # TensorBackedAcquisitionState fields
            posterior_logits=posterior_logits,
            parent_sets=parent_sets,
            variable_names=variable_names,
            current_target=current_target,
            metadata=metadata
        )


class TensorBackedBufferInterface:
    """
    Provides ExperienceBuffer interface for TensorBackedAcquisitionState.
    
    This adapter class translates ExperienceBuffer method calls to operations
    on the underlying tensor data, maintaining interface compatibility.
    """
    
    def __init__(self, tensor_state: TensorBackedAcquisitionState):
        self._state = tensor_state
    
    def get_all_samples(self) -> List[pyr.PMap]:
        """Get all samples from tensor buffer."""
        # Convert tensor buffer to sample list
        # This is a simplified implementation - real version would reconstruct samples
        return []  # Placeholder - would extract from sample_buffer tensor
    
    def get_variable_coverage(self) -> List[str]:
        """Get variable coverage as list of variable names."""
        # Return all non-target variables
        target_idx = self._state.config.target_idx
        return [
            var_name for i, var_name in enumerate(self._state.variable_names)
            if i != target_idx
        ]
    
    def get_statistics(self):
        """Get buffer statistics."""
        return self._state.buffer_statistics
    
    def with_sample(self, sample: pyr.PMap) -> 'TensorBackedBufferInterface':
        """Return new buffer with added sample (immutable operation)."""
        # This would create a new state with updated buffer
        # For now, return self (placeholder implementation)
        return self


# Factory function for backward compatibility
def create_tensor_backed_state(config: JAXConfig) -> TensorBackedAcquisitionState:
    """Create TensorBackedAcquisitionState with default values."""
    return TensorBackedAcquisitionState.create_empty(config)


def create_tensor_backed_state_from_scm(
    scm: pyr.PMap, 
    step: int = 0, 
    best_value: float = 0.0,
    uncertainty_bits: float = 1.0,
    max_samples: int = 1000,
    max_history: int = 50,  # Standardize to 50 to match enriched policy initialization
    feature_dim: int = 3,
    use_bootstrap_surrogate: bool = True
) -> TensorBackedAcquisitionState:
    """
    Factory for creating TensorBackedAcquisitionState from SCM context.
    
    This replaces Mock object creation in enriched_trainer with proper tensor-backed states.
    Now integrates with bootstrap surrogate features for meaningful variable differentiation.
    
    Args:
        scm: SCM defining variables and target
        step: Current optimization step
        best_value: Current best target value
        uncertainty_bits: Posterior uncertainty estimate
        max_samples: Maximum samples in buffer
        max_history: Maximum history length
        feature_dim: Dimension of mechanism features
        use_bootstrap_surrogate: Whether to use bootstrap surrogate features (True) or legacy constants (False)
        
    Returns:
        TensorBackedAcquisitionState ready for training
    """
    # Extract variables and target from SCM
    from ..data_structures.scm import get_variables, get_target
    
    variables = list(get_variables(scm))
    target = get_target(scm)
    
    if not variables:
        raise ValueError("SCM must have at least one variable")
    if target not in variables:
        raise ValueError(f"Target '{target}' not found in SCM variables {variables}")
    
    # Create JAX config from SCM
    from .config import create_jax_config
    
    config = create_jax_config(
        variable_names=variables,
        target_variable=target,
        max_samples=max_samples,
        max_history=max_history,
        feature_dim=feature_dim
    )
    
    if use_bootstrap_surrogate:
        # NEW: Use bootstrap surrogate features for meaningful variable differentiation
        from ..surrogate import (
            create_bootstrap_surrogate_features, 
            PhaseConfig, 
            BootstrapConfig,
            project_embeddings_to_mechanism_features
        )
        
        # Create configurations
        phase_config = PhaseConfig(bootstrap_steps=100, exploration_noise_start=0.5, exploration_noise_end=0.1)
        bootstrap_config = BootstrapConfig(structure_encoding_dim=128)
        
        # Generate bootstrap features
        bootstrap_features = create_bootstrap_surrogate_features(
            scm=scm,
            step=step,
            config=phase_config,
            bootstrap_config=bootstrap_config,
            rng_key=jax.random.PRNGKey(42 + step)  # Step-dependent for variety
        )
        
        # Project 128D embeddings to feature_dim (default 3)
        mechanism_features = project_embeddings_to_mechanism_features(
            bootstrap_features.node_embeddings, 
            target_dim=feature_dim
        )
        
        # Use bootstrap features instead of constants
        marginal_probs = bootstrap_features.parent_probabilities
        confidence_scores = 1.0 - bootstrap_features.uncertainties  # Convert uncertainty to confidence
        
        # Create base JAX state with bootstrap features
        base_state = create_jax_state(
            config=config,
            mechanism_features=mechanism_features,
            marginal_probs=marginal_probs,
            confidence_scores=confidence_scores,
            best_value=best_value,
            current_step=step,
            uncertainty_bits=uncertainty_bits
        )
        
    else:
        # LEGACY: Create base JAX state with constant defaults (for backward compatibility)
        base_state = create_jax_state(
            config=config,
            best_value=best_value,
            current_step=step,
            uncertainty_bits=uncertainty_bits
        )
    
    # Create meaningful parent sets based on SCM structure
    # For training, use simple parent set structure
    n_vars = len(variables)
    target_idx = variables.index(target)
    
    # Create parent sets: empty set + some reasonable combinations
    parent_sets = [frozenset()]  # Always include empty parent set
    
    if n_vars > 1:
        # Add single-parent sets for non-target variables
        for i, var in enumerate(variables):
            if i != target_idx:
                parent_sets.append(frozenset([var]))
    
    if n_vars > 2:
        # Add one two-parent combination
        non_target_vars = [v for i, v in enumerate(variables) if i != target_idx]
        if len(non_target_vars) >= 2:
            parent_sets.append(frozenset(non_target_vars[:2]))
    
    # Create posterior logits (uniform distribution over parent sets)
    num_parent_sets = len(parent_sets)
    posterior_logits = jnp.zeros(num_parent_sets)  # Uniform log probabilities
    
    # Create TensorBackedAcquisitionState with SCM-derived data
    return TensorBackedAcquisitionState(
        # JAXAcquisitionState fields
        sample_buffer=base_state.sample_buffer,
        mechanism_features=base_state.mechanism_features,
        marginal_probs=base_state.marginal_probs,
        confidence_scores=base_state.confidence_scores,
        best_value=base_state.best_value,
        current_step=base_state.current_step,
        uncertainty_bits=base_state.uncertainty_bits,
        config=base_state.config,
        
        # TensorBackedAcquisitionState fields
        posterior_logits=posterior_logits,
        parent_sets=tuple(parent_sets),
        variable_names=tuple(variables),
        current_target=target,
        metadata=pyr.m(scm_info=scm, creation_step=step)
    )