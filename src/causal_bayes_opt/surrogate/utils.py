"""
Surrogate Integration Utilities

This module provides utility functions for integrating surrogate model outputs
with the state creation pipeline and policy networks.

Key functions:
- project_embeddings_to_mechanism_features(): Project high-D embeddings to mechanism features
- generate_surrogate_features(): Main dispatcher for feature generation based on phase
"""

from typing import Dict, Any, Optional, Union
import jax
import jax.numpy as jnp
import pyrsistent as pyr

from .bootstrap import BootstrapSurrogateOutputs, create_bootstrap_surrogate_features
from .transition import (
    TrainedSurrogateOutputs, 
    MixedSurrogateOutputs,
    create_trained_surrogate_features,
    get_mixed_surrogate_features
)
from .phase_manager import PhaseConfig, BootstrapConfig, get_current_phase, TrainingPhase


def project_embeddings_to_mechanism_features(
    embeddings: jnp.ndarray, 
    target_dim: int = 3,
    method: str = "pca"
) -> jnp.ndarray:
    """
    Project high-dimensional embeddings to mechanism feature space.
    
    This function compresses 128D node embeddings to the target dimensionality
    (typically 3) while preserving maximum information content for variable
    differentiation.
    
    Args:
        embeddings: Input embeddings [n_vars, 128]
        target_dim: Target feature dimension (default 3)
        method: Projection method ("pca", "linear", "nonlinear")
        
    Returns:
        Projected features [n_vars, target_dim]
        
    Example:
        >>> embeddings = jnp.ones((5, 128))  # 5 variables, 128D embeddings
        >>> features = project_embeddings_to_mechanism_features(embeddings, target_dim=3)
        >>> features.shape
        (5, 3)
    """
    n_vars, embed_dim = embeddings.shape
    
    if embed_dim < target_dim:
        # Pad with zeros if embedding dimension is smaller than target
        padding = jnp.zeros((n_vars, target_dim - embed_dim))
        return jnp.concatenate([embeddings, padding], axis=1)
    
    if embed_dim == target_dim:
        # Already correct dimension
        return embeddings
    
    # Project high-dimensional embeddings to target dimension
    if method == "pca":
        # Simple PCA-like projection using SVD
        # Center the data
        mean_embedding = jnp.mean(embeddings, axis=0, keepdims=True)
        centered_embeddings = embeddings - mean_embedding
        
        # SVD for dimensionality reduction
        U, s, Vt = jnp.linalg.svd(centered_embeddings, full_matrices=False)
        
        # Project to target dimension using top components
        projection_matrix = Vt[:target_dim, :].T  # [embed_dim, target_dim]
        projected = jnp.dot(centered_embeddings, projection_matrix)
        
        # Add back a scaled version of the mean to maintain interpretability
        projected = projected + jnp.mean(projected, axis=0, keepdims=True) * 0.1
        
        return projected
        
    elif method == "linear":
        # Simple linear projection using first target_dim dimensions
        return embeddings[:, :target_dim]
        
    elif method == "nonlinear":
        # Nonlinear projection using a simple MLP-like transformation
        # Create a fixed random projection matrix (deterministic based on shapes)
        key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
        
        # First layer: embed_dim -> target_dim * 2
        W1 = jax.random.normal(key, (embed_dim, target_dim * 2)) * 0.1
        hidden = jax.nn.tanh(jnp.dot(embeddings, W1))
        
        # Second layer: target_dim * 2 -> target_dim  
        key, subkey = jax.random.split(key)
        W2 = jax.random.normal(subkey, (target_dim * 2, target_dim)) * 0.1
        projected = jnp.dot(hidden, W2)
        
        return projected
        
    else:
        raise ValueError(f"Unknown projection method: {method}")


def generate_surrogate_features(
    scm: pyr.PMap,
    step: int,
    surrogate_model: Optional[Any] = None,
    surrogate_params: Optional[Any] = None,
    sample_buffer: Optional[Any] = None,
    phase_config: Optional[PhaseConfig] = None,
    bootstrap_config: Optional[BootstrapConfig] = None
) -> Union[BootstrapSurrogateOutputs, TrainedSurrogateOutputs, MixedSurrogateOutputs]:
    """
    Generate appropriate surrogate features based on training phase.
    
    This is the main dispatcher function that determines which type of surrogate
    features to generate based on the current training step and available models.
    
    Args:
        scm: Structural causal model
        step: Current training step
        surrogate_model: Optional trained surrogate model
        surrogate_params: Optional model parameters
        sample_buffer: Optional sample buffer for trained model inference
        phase_config: Phase configuration (creates default if None)
        bootstrap_config: Bootstrap configuration (creates default if None)
        
    Returns:
        Appropriate surrogate features for current phase
        
    Example:
        >>> features = generate_surrogate_features(scm, step=50)  # Bootstrap phase
        >>> type(features)
        <class 'BootstrapSurrogateOutputs'>
    """
    # Create default configurations if not provided
    if phase_config is None:
        phase_config = PhaseConfig()
    
    if bootstrap_config is None:
        bootstrap_config = BootstrapConfig()
    
    # Determine current phase
    current_phase = get_current_phase(step, phase_config)
    
    if current_phase == TrainingPhase.BOOTSTRAP:
        # Generate bootstrap features
        return create_bootstrap_surrogate_features(
            scm=scm,
            step=step,
            config=phase_config,
            bootstrap_config=bootstrap_config
        )
        
    elif current_phase == TrainingPhase.TRANSITION:
        # Generate mixed features if trained model is available
        if surrogate_model is not None and surrogate_params is not None and sample_buffer is not None:
            try:
                # Generate both bootstrap and trained features
                bootstrap_features = create_bootstrap_surrogate_features(
                    scm=scm,
                    step=step,
                    config=phase_config,
                    bootstrap_config=bootstrap_config
                )
                
                # Create batch data from sample buffer
                batch_data = _prepare_batch_from_buffer(sample_buffer, scm)
                target_variable = _get_target_from_scm(scm)
                variable_order = _get_variables_from_scm(scm)
                
                trained_features = create_trained_surrogate_features(
                    surrogate_model=surrogate_model,
                    surrogate_params=surrogate_params,
                    batch_data=batch_data,
                    target_variable=target_variable,
                    variable_order=variable_order
                )
                
                # Mix features based on transition schedule
                return get_mixed_surrogate_features(
                    bootstrap_features=bootstrap_features,
                    trained_features=trained_features,
                    step=step,
                    config=phase_config
                )
                
            except Exception as e:
                # Fallback to bootstrap if trained model fails
                print(f"Warning: Failed to create trained features, falling back to bootstrap: {e}")
                return create_bootstrap_surrogate_features(
                    scm=scm,
                    step=step,
                    config=phase_config,
                    bootstrap_config=bootstrap_config
                )
        else:
            # No trained model available - use bootstrap
            return create_bootstrap_surrogate_features(
                scm=scm,
                step=step,
                config=phase_config,
                bootstrap_config=bootstrap_config
            )
            
    elif current_phase == TrainingPhase.TRAINED:
        # Use fully trained model if available
        if surrogate_model is not None and surrogate_params is not None and sample_buffer is not None:
            try:
                batch_data = _prepare_batch_from_buffer(sample_buffer, scm)
                target_variable = _get_target_from_scm(scm)
                variable_order = _get_variables_from_scm(scm)
                
                return create_trained_surrogate_features(
                    surrogate_model=surrogate_model,
                    surrogate_params=surrogate_params,
                    batch_data=batch_data,
                    target_variable=target_variable,
                    variable_order=variable_order
                )
                
            except Exception as e:
                # Fallback to bootstrap if trained model fails
                print(f"Warning: Trained model failed, falling back to bootstrap: {e}")
                return create_bootstrap_surrogate_features(
                    scm=scm,
                    step=step,
                    config=phase_config,
                    bootstrap_config=bootstrap_config
                )
        else:
            # No trained model available - use bootstrap even in trained phase
            return create_bootstrap_surrogate_features(
                scm=scm,
                step=step,
                config=phase_config,
                bootstrap_config=bootstrap_config
            )
    
    else:
        raise ValueError(f"Unknown training phase: {current_phase}")


def create_tensor_backed_state_with_surrogate(
    scm: pyr.PMap,
    step: int,
    surrogate_model: Optional[Any] = None,
    surrogate_params: Optional[Any] = None,
    sample_buffer: Optional[Any] = None,
    bootstrap_config: Optional[BootstrapConfig] = None,
    phase_config: Optional[PhaseConfig] = None,
    **kwargs
) -> Any:
    """
    Create state using surrogate outputs (bootstrap or trained).
    
    This is the main factory function for creating tensor-backed states
    using the appropriate surrogate features for the current training phase.
    
    Args:
        scm: Structural causal model
        step: Current training step
        surrogate_model: Optional trained surrogate model
        surrogate_params: Optional model parameters
        sample_buffer: Optional sample buffer
        bootstrap_config: Bootstrap configuration
        phase_config: Phase configuration
        **kwargs: Additional arguments passed to state creation
        
    Returns:
        TensorBackedAcquisitionState with surrogate-derived features
    """
    # Import here to avoid circular imports
    from ..jax_native.state import create_tensor_backed_state_from_scm
    
    # Generate appropriate surrogate features
    surrogate_features = generate_surrogate_features(
        scm=scm,
        step=step,
        surrogate_model=surrogate_model,
        surrogate_params=surrogate_params,
        sample_buffer=sample_buffer,
        phase_config=phase_config,
        bootstrap_config=bootstrap_config
    )
    
    # Create state using the factory with bootstrap surrogate enabled
    return create_tensor_backed_state_from_scm(
        scm=scm,
        step=step,
        use_bootstrap_surrogate=True,
        **kwargs
    )


# Helper functions for internal use

def _prepare_batch_from_buffer(sample_buffer, scm):
    """Prepare batch data from sample buffer for model inference."""
    # Placeholder implementation - would need to be implemented based on buffer format
    # This would convert the sample buffer to the [N, d, 3] format expected by models
    return {'intervention_data': jnp.ones((10, 3, 3))}  # Dummy data


def _get_target_from_scm(scm):
    """Extract target variable from SCM."""
    from ..data_structures.scm import get_target
    return get_target(scm)


def _get_variables_from_scm(scm):
    """Extract variable list from SCM."""
    from ..data_structures.scm import get_variables
    return list(get_variables(scm))