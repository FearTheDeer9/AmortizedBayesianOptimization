"""
Bootstrap to Trained Surrogate Transition

This module handles the smooth transition from bootstrap surrogate features
to fully trained surrogate model outputs during the transition phase.

Key functions:
- get_mixed_surrogate_features(): Interpolate between bootstrap and trained features
- create_trained_surrogate_features(): Extract features from trained surrogate model
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
import jax
import jax.numpy as jnp

from .bootstrap import BootstrapSurrogateOutputs
from .phase_manager import PhaseConfig, compute_transition_weight


@dataclass(frozen=True)
class TrainedSurrogateOutputs:
    """
    Output format for fully trained surrogate model features.
    
    This represents the rich learned information from a trained surrogate model,
    providing the target format for the transition process.
    
    Attributes:
        node_embeddings: Learned variable representations [n_vars, 128]
        parent_probabilities: Learned parent probabilities [n_vars]
        uncertainties: Prediction uncertainties from model [n_vars]
        attention_weights: Optional attention patterns from model
        metadata: Model confidence and training metrics
    """
    node_embeddings: jnp.ndarray                # [n_vars, 128] - Learned variable representations
    parent_probabilities: jnp.ndarray           # [n_vars] - Learned parent probabilities
    uncertainties: jnp.ndarray                  # [n_vars] - Prediction uncertainties
    attention_weights: Optional[jnp.ndarray]    # Optional attention patterns from model
    metadata: Dict[str, Any]                    # Model confidence and training metrics
    
    def __post_init__(self):
        """Validate trained surrogate outputs."""
        n_vars = self.node_embeddings.shape[0]
        
        # Check shape consistency
        if self.node_embeddings.shape[1] != 128:
            raise ValueError(f"node_embeddings must be [n_vars, 128], got {self.node_embeddings.shape}")
        
        if self.parent_probabilities.shape != (n_vars,):
            raise ValueError(f"parent_probabilities must be [n_vars], got {self.parent_probabilities.shape}")
        
        if self.uncertainties.shape != (n_vars,):
            raise ValueError(f"uncertainties must be [n_vars], got {self.uncertainties.shape}")
        
        # Check value ranges
        if jnp.any(self.parent_probabilities < 0) or jnp.any(self.parent_probabilities > 1):
            raise ValueError("parent_probabilities must be in [0, 1]")
        
        if jnp.any(self.uncertainties < 0) or jnp.any(self.uncertainties > 1):
            raise ValueError("uncertainties must be in [0, 1]")


@dataclass(frozen=True)
class MixedSurrogateOutputs:
    """
    Mixed surrogate outputs during transition phase.
    
    This combines bootstrap and trained features using weighted interpolation,
    providing a smooth transition between the two phases.
    
    Attributes:
        node_embeddings: Mixed embeddings [n_vars, 128]
        parent_probabilities: Mixed parent probabilities [n_vars]
        uncertainties: Mixed uncertainties [n_vars]
        metadata: Transition parameters and mixing weights
    """
    node_embeddings: jnp.ndarray        # [n_vars, 128] - Mixed embeddings
    parent_probabilities: jnp.ndarray   # [n_vars] - Mixed parent probabilities
    uncertainties: jnp.ndarray          # [n_vars] - Mixed uncertainties
    metadata: Dict[str, Any]            # Transition parameters and mixing weights
    
    def __post_init__(self):
        """Validate mixed surrogate outputs."""
        n_vars = self.node_embeddings.shape[0]
        
        # Check shape consistency
        if self.node_embeddings.shape[1] != 128:
            raise ValueError(f"node_embeddings must be [n_vars, 128], got {self.node_embeddings.shape}")
        
        if self.parent_probabilities.shape != (n_vars,):
            raise ValueError(f"parent_probabilities must be [n_vars], got {self.parent_probabilities.shape}")
        
        if self.uncertainties.shape != (n_vars,):
            raise ValueError(f"uncertainties must be [n_vars], got {self.uncertainties.shape}")


def get_mixed_surrogate_features(
    bootstrap_features: BootstrapSurrogateOutputs,
    trained_features: TrainedSurrogateOutputs,
    step: int,
    config: PhaseConfig
) -> MixedSurrogateOutputs:
    """
    Smoothly transition from bootstrap to trained surrogate features.
    
    This function performs weighted interpolation between bootstrap and trained
    features based on the current step and transition schedule, ensuring no
    sudden jumps in policy behavior during the transition.
    
    Args:
        bootstrap_features: Bootstrap surrogate outputs
        trained_features: Trained surrogate outputs  
        step: Current training step
        config: Phase configuration with transition schedule
        
    Returns:
        Mixed surrogate outputs with interpolated features
        
    Raises:
        ValueError: If feature shapes are incompatible
        
    Example:
        >>> mixed = get_mixed_surrogate_features(bootstrap, trained, step=125, config)
        >>> mixed.metadata['bootstrap_weight']  # Should be 0.5 at mid-transition
        0.5
    """
    # Validate input shapes
    if bootstrap_features.node_embeddings.shape != trained_features.node_embeddings.shape:
        raise ValueError("Bootstrap and trained embeddings must have same shape")
    
    if bootstrap_features.parent_probabilities.shape != trained_features.parent_probabilities.shape:
        raise ValueError("Bootstrap and trained parent probabilities must have same shape")
    
    if bootstrap_features.uncertainties.shape != trained_features.uncertainties.shape:
        raise ValueError("Bootstrap and trained uncertainties must have same shape")
    
    # Compute transition weights
    weights = compute_transition_weight(step, config)
    bootstrap_weight = weights['bootstrap_weight']
    trained_weight = weights['trained_weight']
    
    # Linear interpolation between features
    mixed_embeddings = (
        bootstrap_weight * bootstrap_features.node_embeddings +
        trained_weight * trained_features.node_embeddings
    )
    
    mixed_parent_probs = (
        bootstrap_weight * bootstrap_features.parent_probabilities +
        trained_weight * trained_features.parent_probabilities
    )
    
    mixed_uncertainties = (
        bootstrap_weight * bootstrap_features.uncertainties +
        trained_weight * trained_features.uncertainties
    )
    
    # Create metadata combining information from both sources
    metadata = {
        'mixed': True,
        'step': step,
        'bootstrap_weight': float(bootstrap_weight),
        'trained_weight': float(trained_weight),
        'transition_progress': weights.get('transition_progress'),
        'bootstrap_metadata': bootstrap_features.metadata,
        'trained_metadata': trained_features.metadata,
        'transition_schedule': config.transition_schedule,
        'transition_steps': config.transition_steps
    }
    
    return MixedSurrogateOutputs(
        node_embeddings=mixed_embeddings,
        parent_probabilities=mixed_parent_probs,
        uncertainties=mixed_uncertainties,
        metadata=metadata
    )


def create_trained_surrogate_features(
    surrogate_model: Any,
    surrogate_params: Any,
    batch_data: Dict[str, jnp.ndarray],
    target_variable: str,
    variable_order: List[str]
) -> TrainedSurrogateOutputs:
    """
    Extract features from fully trained surrogate model.
    
    This function runs inference on the trained surrogate model and extracts
    the rich learned features (embeddings, probabilities, uncertainties) that
    will replace the bootstrap features.
    
    Args:
        surrogate_model: Trained surrogate model (JAX/Haiku)
        surrogate_params: Model parameters
        batch_data: Batch data for inference [N, d, 3] format
        target_variable: Target variable name
        variable_order: Ordered list of variable names
        
    Returns:
        TrainedSurrogateOutputs with learned features
        
    Raises:
        ValueError: If model outputs are invalid or missing
        
    Example:
        >>> trained = create_trained_surrogate_features(model, params, batch, 'Y', ['X', 'Y', 'Z'])
        >>> trained.node_embeddings.shape
        (3, 128)
    """
    # Validate inputs
    if not variable_order:
        raise ValueError("variable_order cannot be empty")
    
    if target_variable not in variable_order:
        raise ValueError(f"target_variable '{target_variable}' not in variable_order {variable_order}")
    
    target_idx = variable_order.index(target_variable)
    
    # Run surrogate model inference
    try:
        if hasattr(surrogate_model, 'apply'):
            # JAX/Haiku model
            rng_key = jax.random.PRNGKey(42)  # Use fixed key for inference
            # Try with is_training parameter first, fallback without if it fails
            try:
                outputs = surrogate_model.apply(
                    surrogate_params,
                    rng_key,
                    batch_data['intervention_data'] if 'intervention_data' in batch_data else batch_data,
                    target_idx,
                    is_training=False
                )
            except TypeError as te:
                if 'is_training' in str(te):
                    # Model doesn't accept is_training parameter
                    outputs = surrogate_model.apply(
                        surrogate_params,
                        rng_key,
                        batch_data['intervention_data'] if 'intervention_data' in batch_data else batch_data,
                        target_idx
                    )
                else:
                    raise te
        else:
            # Legacy model interface
            outputs = surrogate_model(batch_data, target_variable, is_training=False)
        
    except Exception as e:
        raise ValueError(f"Failed to run surrogate model inference: {e}")
    
    # Extract features from model outputs
    if isinstance(outputs, dict):
        # Modern model with structured outputs
        node_embeddings = outputs.get('node_embeddings')
        parent_probs = outputs.get('parent_probabilities')
        attention_weights = outputs.get('attention_weights')
        
        if node_embeddings is None:
            raise ValueError("Model outputs missing 'node_embeddings'")
        if parent_probs is None:
            raise ValueError("Model outputs missing 'parent_probabilities'")
            
    else:
        # Legacy model - assume outputs are just parent probabilities
        parent_probs = outputs
        node_embeddings = None
        attention_weights = None
    
    # Validate and process node embeddings
    n_vars = len(variable_order)
    
    if node_embeddings is None:
        # Create placeholder embeddings if model doesn't provide them
        # Use parent probabilities to create meaningful embeddings
        node_embeddings = jnp.zeros((n_vars, 128))
        for i in range(n_vars):
            # Encode parent probability and position information
            node_embeddings = node_embeddings.at[i, 0].set(parent_probs[i] if i < len(parent_probs) else 0.0)
            node_embeddings = node_embeddings.at[i, 1].set(float(i == target_idx))  # Target indicator
            node_embeddings = node_embeddings.at[i, 2].set(float(i) / n_vars)      # Position encoding
    
    # Validate embeddings shape
    if node_embeddings.shape[0] != n_vars:
        raise ValueError(f"node_embeddings has {node_embeddings.shape[0]} variables, expected {n_vars}")
    
    if node_embeddings.shape[1] != 128:
        # Pad or truncate to 128 dimensions
        current_dim = node_embeddings.shape[1]
        if current_dim < 128:
            # Pad with zeros
            padding = jnp.zeros((n_vars, 128 - current_dim))
            node_embeddings = jnp.concatenate([node_embeddings, padding], axis=1)
        else:
            # Truncate to 128
            node_embeddings = node_embeddings[:, :128]
    
    # Validate and process parent probabilities
    if len(parent_probs) != n_vars:
        # Extend or truncate to match variable count
        if len(parent_probs) < n_vars:
            # Pad with zeros
            padding = jnp.zeros(n_vars - len(parent_probs))
            parent_probs = jnp.concatenate([parent_probs, padding])
        else:
            # Truncate
            parent_probs = parent_probs[:n_vars]
    
    # Ensure target has zero parent probability
    parent_probs = parent_probs.at[target_idx].set(0.0)
    
    # Renormalize probabilities
    prob_sum = jnp.sum(parent_probs)
    if prob_sum > 0:
        parent_probs = parent_probs / prob_sum
    
    # Compute per-variable uncertainty from probability distributions
    uncertainties = compute_prediction_uncertainty(parent_probs, attention_weights)
    
    # Create metadata
    model_confidence = float(jnp.mean(1.0 - uncertainties))
    metadata = {
        'fully_trained': True,
        'model_type': type(surrogate_model).__name__,
        'model_confidence': model_confidence,
        'n_variables': n_vars,
        'target_variable': target_variable,
        'target_idx': target_idx,
        'variable_order': variable_order,
        'has_attention_weights': attention_weights is not None,
        'mean_parent_probability': float(jnp.mean(parent_probs)),
        'max_parent_probability': float(jnp.max(parent_probs)),
        'entropy': float(-jnp.sum(parent_probs * jnp.log(parent_probs + 1e-8)))
    }
    
    return TrainedSurrogateOutputs(
        node_embeddings=node_embeddings,
        parent_probabilities=parent_probs,
        uncertainties=uncertainties,
        attention_weights=attention_weights,
        metadata=metadata
    )


def compute_prediction_uncertainty(
    parent_probs: jnp.ndarray,
    attention_weights: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Compute per-variable uncertainty from model predictions.
    
    Args:
        parent_probs: Parent probabilities [n_vars]
        attention_weights: Optional attention weights from model
        
    Returns:
        Uncertainty scores [n_vars] in range [0, 1]
    """
    n_vars = len(parent_probs)
    
    # Method 1: Entropy-based uncertainty from parent probabilities
    # Higher entropy = higher uncertainty
    max_entropy = jnp.log(n_vars)  # Maximum possible entropy
    
    # Compute entropy for each variable (treating it as a categorical distribution)
    uncertainties = jnp.zeros(n_vars)
    
    for i in range(n_vars):
        # Create a simple binary distribution: [prob, 1-prob] for parent probability
        prob = parent_probs[i]
        if prob > 0 and prob < 1:
            binary_entropy = -prob * jnp.log(prob) - (1 - prob) * jnp.log(1 - prob)
            uncertainty = binary_entropy / jnp.log(2)  # Normalize by max binary entropy
        else:
            uncertainty = 0.0  # No uncertainty for extreme probabilities
        
        uncertainties = uncertainties.at[i].set(uncertainty)
    
    # Method 2: If attention weights available, use attention entropy
    if attention_weights is not None:
        # Add attention-based uncertainty (higher attention spread = higher uncertainty)
        # This is model-specific and would need to be implemented based on the model architecture
        pass
    
    # Ensure uncertainties are in [0, 1] range
    uncertainties = jnp.clip(uncertainties, 0.0, 1.0)
    
    return uncertainties


def validate_transition_compatibility(
    bootstrap_features: BootstrapSurrogateOutputs,
    trained_features: TrainedSurrogateOutputs
) -> bool:
    """
    Validate that bootstrap and trained features are compatible for transition.
    
    Args:
        bootstrap_features: Bootstrap surrogate outputs
        trained_features: Trained surrogate outputs
        
    Returns:
        True if features are compatible
        
    Raises:
        ValueError: If features are incompatible
    """
    # Check shape compatibility
    if bootstrap_features.node_embeddings.shape != trained_features.node_embeddings.shape:
        raise ValueError("Bootstrap and trained embeddings have incompatible shapes")
    
    if bootstrap_features.parent_probabilities.shape != trained_features.parent_probabilities.shape:
        raise ValueError("Bootstrap and trained parent probabilities have incompatible shapes")
    
    if bootstrap_features.uncertainties.shape != trained_features.uncertainties.shape:
        raise ValueError("Bootstrap and trained uncertainties have incompatible shapes")
    
    # Check that target variables match
    bootstrap_target = bootstrap_features.metadata.get('target_variable')
    trained_target = trained_features.metadata.get('target_variable')
    
    if bootstrap_target != trained_target:
        raise ValueError(f"Target variables don't match: bootstrap='{bootstrap_target}', trained='{trained_target}'")
    
    # Check that variable ordering is consistent
    bootstrap_vars = bootstrap_features.metadata.get('variables', [])
    trained_vars = trained_features.metadata.get('variable_order', [])
    
    if bootstrap_vars and trained_vars and bootstrap_vars != trained_vars:
        raise ValueError(f"Variable orders don't match: bootstrap={bootstrap_vars}, trained={trained_vars}")
    
    return True