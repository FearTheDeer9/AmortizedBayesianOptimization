"""
Bootstrap Surrogate Implementation

This module implements bootstrap surrogate features that provide meaningful variable
differentiation during early training when the actual surrogate model is not yet trained.

The bootstrap approach uses SCM structure knowledge to create realistic surrogate-like
features, solving the chicken-and-egg problem of needing a trained surrogate for good
policy and good policy for training the surrogate.

Key function:
- create_bootstrap_surrogate_features(): Main bootstrap function with exploration schedule
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import jax
import jax.numpy as jnp
import pyrsistent as pyr

from .structure_encoding import encode_causal_structure, compute_structural_parent_probabilities
from .phase_manager import PhaseConfig, BootstrapConfig, compute_exploration_factor
from ..data_structures.scm import get_variables, get_target, get_edges


@dataclass(frozen=True)
class BootstrapSurrogateOutputs:
    """
    Output format for bootstrap surrogate features.
    
    This matches the interface expected by the state creation pipeline,
    providing the same rich information as a trained surrogate model.
    
    Attributes:
        node_embeddings: Structure-aware variable embeddings [n_vars, 128]
        parent_probabilities: Prior parent probabilities based on structure [n_vars]
        uncertainties: Exploration-based uncertainty measures [n_vars]
        metadata: Bootstrap parameters and exploration state
    """
    node_embeddings: jnp.ndarray        # [n_vars, 128] - Structure-aware embeddings
    parent_probabilities: jnp.ndarray   # [n_vars] - Prior parent probabilities  
    uncertainties: jnp.ndarray          # [n_vars] - Exploration-based uncertainties
    metadata: Dict[str, Any]            # Bootstrap parameters and exploration state
    
    def __post_init__(self):
        """Validate bootstrap outputs."""
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
        
        # Check probability normalization (allowing some numerical tolerance)
        prob_sum = float(jnp.sum(self.parent_probabilities))
        if not (0.95 <= prob_sum <= 1.05):
            raise ValueError(f"parent_probabilities should sum to ~1.0, got {prob_sum}")


def create_bootstrap_surrogate_features(
    scm: pyr.PMap,
    step: int,
    config: PhaseConfig,
    bootstrap_config: BootstrapConfig,
    rng_key: Optional[jnp.ndarray] = None
) -> BootstrapSurrogateOutputs:
    """
    Generate structure-aware bootstrap features for early training.
    
    This function creates meaningful surrogate-like features based on the known
    SCM structure, providing variable differentiation from day one of training.
    Features evolve from high exploration (noisy) to more structured over time.
    
    Args:
        scm: Structural causal model containing variables, edges, and target
        step: Current training step (controls exploration schedule)
        config: Phase configuration for exploration schedule
        bootstrap_config: Bootstrap-specific configuration
        rng_key: JAX random key for noise generation (creates if None)
        
    Returns:
        BootstrapSurrogateOutputs with structure-derived features
        
    Raises:
        ValueError: If SCM is invalid or missing required information
        
    Example:
        >>> scm = pyr.pmap({'variables': ['X', 'Y', 'Z'], 'target': 'Z', 'edges': [('X', 'Y'), ('Y', 'Z')]})
        >>> config = PhaseConfig(bootstrap_steps=100)
        >>> bootstrap_config = BootstrapConfig()
        >>> features = create_bootstrap_surrogate_features(scm, step=50, config, bootstrap_config)
        >>> features.node_embeddings.shape
        (3, 128)
    """
    # Validate inputs
    if not isinstance(scm, pyr.PMap):
        raise ValueError("SCM must be a pyrsistent PMap")
    
    if step < 0:
        raise ValueError("Step must be non-negative")
    
    # Extract SCM components
    try:
        variables = list(get_variables(scm))
        target = get_target(scm)
        edges = get_edges(scm)
    except Exception as e:
        raise ValueError(f"Failed to extract SCM components: {e}")
    
    if not variables:
        raise ValueError("SCM must contain at least one variable")
    
    if target not in variables:
        raise ValueError(f"Target '{target}' not found in SCM variables {variables}")
    
    # Create random key if not provided
    if rng_key is None:
        rng_key = jax.random.PRNGKey(42 + step)  # Step-dependent seed for reproducibility
    
    # Split random key for different noise components
    key1, key2 = jax.random.split(rng_key)
    
    # 1. Generate structure-aware node embeddings
    structural_embeddings = encode_causal_structure(
        variables=variables,
        edges=edges, 
        target=target,
        dim=bootstrap_config.structure_encoding_dim
    )
    
    # 2. Add decreasing exploration noise
    exploration_factor = compute_exploration_factor(step, config, bootstrap_config)
    
    # Generate noise with same shape as embeddings
    exploration_noise = jax.random.normal(key1, structural_embeddings.shape) * exploration_factor
    noisy_embeddings = structural_embeddings + exploration_noise
    
    # 3. Compute structural parent probabilities
    structural_parent_probs = compute_structural_parent_probabilities(
        variables=variables,
        edges=edges,
        target=target
    )
    
    # 4. Mix structural priors with uniform exploration
    n_vars = len(variables)
    target_idx = variables.index(target)
    
    # Create uniform distribution over non-target variables
    uniform_parent_probs = jnp.ones(n_vars) / max(1, n_vars - 1)  # Exclude target
    uniform_parent_probs = uniform_parent_probs.at[target_idx].set(0.0)  # Target can't be its own parent
    
    # Mix structural and uniform probabilities based on exploration factor
    mixed_parent_probs = (
        exploration_factor * uniform_parent_probs + 
        (1 - exploration_factor) * structural_parent_probs
    )
    
    # Ensure target has zero probability
    mixed_parent_probs = mixed_parent_probs.at[target_idx].set(0.0)
    
    # Renormalize to ensure valid probability distribution
    prob_sum = jnp.sum(mixed_parent_probs)
    if prob_sum > 0:
        mixed_parent_probs = mixed_parent_probs / prob_sum
    
    # 5. Compute per-variable uncertainties (high during exploration)
    base_uncertainties = jnp.ones(n_vars) * exploration_factor
    
    # Add small amount of variable-specific noise to uncertainties
    uncertainty_noise = jax.random.normal(key2, (n_vars,)) * 0.1 * exploration_factor
    uncertainties = jnp.clip(base_uncertainties + uncertainty_noise, 0.0, 1.0)
    
    # Target variable has zero uncertainty (we know it's the target)
    uncertainties = uncertainties.at[target_idx].set(0.0)
    
    # 6. Create metadata
    metadata = {
        'bootstrap': True,
        'step': step,
        'exploration_factor': float(exploration_factor),
        'n_variables': n_vars,
        'target_variable': target,
        'variables': variables,
        'structural_edges': edges,
        'config': {
            'bootstrap_steps': config.bootstrap_steps,
            'exploration_noise_start': config.exploration_noise_start,
            'exploration_noise_end': config.exploration_noise_end,
            'structure_encoding_dim': bootstrap_config.structure_encoding_dim,
            'noise_schedule': bootstrap_config.noise_schedule
        }
    }
    
    return BootstrapSurrogateOutputs(
        node_embeddings=noisy_embeddings,
        parent_probabilities=mixed_parent_probs,
        uncertainties=uncertainties,
        metadata=metadata
    )


def validate_bootstrap_features(features: BootstrapSurrogateOutputs) -> bool:
    """
    Validate that bootstrap features are reasonable and meaningful.
    
    Args:
        features: Bootstrap surrogate outputs to validate
        
    Returns:
        True if features pass validation
        
    Raises:
        ValueError: If features are invalid
    """
    n_vars = features.node_embeddings.shape[0]
    
    # Check that variables have different embeddings (no constant channels)
    for channel in range(min(10, features.node_embeddings.shape[1])):  # Check first 10 channels
        channel_values = features.node_embeddings[:, channel]
        if jnp.allclose(channel_values, channel_values[0], atol=1e-6):
            raise ValueError(f"Channel {channel} has constant values - no variable differentiation")
    
    # Check that parent probabilities are different across variables (excluding target)
    target_var = features.metadata.get('target_variable')
    variables = features.metadata.get('variables', [])
    
    if target_var and variables:
        target_idx = variables.index(target_var)
        non_target_probs = jnp.concatenate([
            features.parent_probabilities[:target_idx],
            features.parent_probabilities[target_idx+1:]
        ])
        
        if len(non_target_probs) > 1 and jnp.allclose(non_target_probs, non_target_probs[0], atol=1e-6):
            raise ValueError("All non-target variables have identical parent probabilities")
    
    # Check that uncertainties are reasonable
    if jnp.all(features.uncertainties == 0.0):
        raise ValueError("All uncertainties are zero - no exploration signal")
    
    if jnp.all(features.uncertainties == 1.0):
        raise ValueError("All uncertainties are one - no structure signal")
    
    return True


def create_test_bootstrap_features(
    n_vars: int = 3,
    target_idx: int = 2,
    step: int = 0
) -> BootstrapSurrogateOutputs:
    """
    Create test bootstrap features for unit testing.
    
    Args:
        n_vars: Number of variables
        target_idx: Index of target variable
        step: Training step
        
    Returns:
        Test bootstrap features
    """
    # Create simple test SCM
    variables = [f"X{i}" for i in range(n_vars)]
    target = variables[target_idx]
    edges = [(variables[i], variables[i+1]) for i in range(n_vars-1)]  # Simple chain
    
    scm = pyr.pmap({
        'variables': variables,
        'target': target,
        'edges': edges
    })
    
    config = PhaseConfig(bootstrap_steps=100)
    bootstrap_config = BootstrapConfig()
    
    return create_bootstrap_surrogate_features(
        scm=scm,
        step=step,
        config=config,
        bootstrap_config=bootstrap_config,
        rng_key=jax.random.PRNGKey(42)
    )