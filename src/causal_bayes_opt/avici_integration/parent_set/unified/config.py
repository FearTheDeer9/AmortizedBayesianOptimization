"""
Configuration system for unified parent set model.

Addresses the hard-coded max_parents=5 scalability issue by providing
adaptive configuration based on graph characteristics.
"""

from dataclasses import dataclass
from typing import List, Optional
import jax.numpy as jnp


@dataclass(frozen=True)
class TargetAwareConfig:
    """
    Configuration for unified parent set model with adaptive parameters.
    
    This addresses the scalability concerns by making max_parents adaptive
    and providing clear feature flag control for mechanism prediction.
    """
    
    # Core architecture (from original model)
    layers: int = 8
    dim: int = 128
    key_size: int = 32
    num_heads: int = 8
    widening_factor: int = 4
    dropout: float = 0.1
    
    # FIXED: Adaptive max_parents instead of hard-coded
    max_parent_size: Optional[int] = None  # If None, compute adaptively
    max_parent_density: float = 0.3        # Max density for adaptive computation
    min_max_parents: int = 2               # Minimum max_parents regardless of graph size
    max_max_parents: int = 10              # Maximum max_parents for computational limits
    
    # Target-aware conditioning (from modular model)
    enable_target_conditioning: bool = True
    target_embedding_dim: Optional[int] = None  # If None, use dim // 4
    
    # Mechanism prediction (optional extension)
    predict_mechanisms: bool = False
    mechanism_types: List[str] = None
    mechanism_param_dim: int = 32
    
    # Advanced options
    use_mla_scoring: bool = True            # Use MLA-based parent set scoring
    enable_uncertainty_quantification: bool = True
    
    def __post_init__(self):
        """Validate and set default values."""
        # Set default mechanism types
        if self.mechanism_types is None:
            object.__setattr__(self, 'mechanism_types', ['linear'])
            
        # Set default target embedding dimension
        if self.target_embedding_dim is None:
            object.__setattr__(self, 'target_embedding_dim', self.dim // 4)
            
        # Validate parameters
        if self.max_parent_density <= 0 or self.max_parent_density > 1:
            raise ValueError("max_parent_density must be in (0, 1]")
            
        if self.min_max_parents < 1:
            raise ValueError("min_max_parents must be >= 1")
            
        if self.max_max_parents < self.min_max_parents:
            raise ValueError("max_max_parents must be >= min_max_parents")
            
        if self.target_embedding_dim <= 0:
            raise ValueError("target_embedding_dim must be positive")
            
        # Validate mechanism types if mechanism prediction enabled
        if self.predict_mechanisms:
            valid_types = {'linear', 'polynomial', 'gaussian', 'neural'}
            for mech_type in self.mechanism_types:
                if mech_type not in valid_types:
                    raise ValueError(f"Unknown mechanism type: {mech_type}. Valid: {valid_types}")


def create_structure_only_config(**kwargs) -> TargetAwareConfig:
    """
    Create configuration for structure-only mode (backward compatibility).
    
    This maintains compatibility with existing code while providing
    the adaptive max_parents functionality.
    """
    defaults = {
        'predict_mechanisms': False,
        'enable_target_conditioning': True,  # Still beneficial for structure learning
    }
    defaults.update(kwargs)
    return TargetAwareConfig(**defaults)


def create_mechanism_aware_config(
    mechanism_types: List[str] = None,
    **kwargs
) -> TargetAwareConfig:
    """
    Create configuration for mechanism-aware mode.
    
    Args:
        mechanism_types: List of mechanism types to predict
        **kwargs: Additional configuration options
    """
    if mechanism_types is None:
        mechanism_types = ['linear', 'polynomial']
        
    defaults = {
        'predict_mechanisms': True,
        'mechanism_types': mechanism_types,
        'enable_target_conditioning': True,
    }
    defaults.update(kwargs)
    return TargetAwareConfig(**defaults)


def compute_adaptive_max_parents(n_variables: int, config: TargetAwareConfig) -> int:
    """
    Compute adaptive max_parents based on graph size and density constraints.
    
    This addresses the scalability issue by automatically adjusting max_parents
    based on the number of variables in the graph.
    
    Args:
        n_variables: Number of variables in the graph
        config: Configuration containing density constraints
        
    Returns:
        Adaptive max_parents value
    """
    if config.max_parent_size is not None:
        # Explicit override - respect user choice but validate bounds
        return max(
            config.min_max_parents,
            min(config.max_parent_size, config.max_max_parents, n_variables - 1)
        )
    
    # Adaptive computation based on density
    # For n variables, maximum possible parents is n-1 (excluding target)
    max_possible = n_variables - 1
    
    # Compute based on density constraint
    density_based = int(config.max_parent_density * max_possible)
    
    # Apply bounds
    adaptive_max = max(
        config.min_max_parents,
        min(density_based, config.max_max_parents, max_possible)
    )
    
    return adaptive_max


def validate_config_for_graph(config: TargetAwareConfig, n_variables: int) -> bool:
    """
    Validate configuration is appropriate for given graph size.
    
    Args:
        config: Configuration to validate
        n_variables: Number of variables in the graph
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid for graph size
    """
    if n_variables < 2:
        raise ValueError("Graph must have at least 2 variables")
        
    max_parents = compute_adaptive_max_parents(n_variables, config)
    
    if max_parents >= n_variables:
        raise ValueError(
            f"Computed max_parents ({max_parents}) must be < n_variables ({n_variables})"
        )
        
    # Check computational feasibility
    from math import comb
    total_parent_sets = sum(
        comb(n_variables - 1, k) for k in range(max_parents + 1)
    )
    
    # Warn if extremely large number of parent sets
    if total_parent_sets > 10000:
        import warnings
        warnings.warn(
            f"Large number of parent sets ({total_parent_sets}) may impact performance. "
            f"Consider reducing max_parent_density from {config.max_parent_density}"
        )
    
    return True


# Preset configurations for common use cases
SMALL_GRAPH_CONFIG = create_structure_only_config(
    max_parent_density=0.5,  # Allow higher density for small graphs
    min_max_parents=2,
    max_max_parents=5
)

MEDIUM_GRAPH_CONFIG = create_structure_only_config(
    max_parent_density=0.3,  # Balanced density
    min_max_parents=2,
    max_max_parents=8
)

LARGE_GRAPH_CONFIG = create_structure_only_config(
    max_parent_density=0.2,  # Lower density for large graphs
    min_max_parents=2,
    max_max_parents=10
)

MECHANISM_AWARE_CONFIG = create_mechanism_aware_config(
    mechanism_types=['linear', 'polynomial', 'gaussian'],
    max_parent_density=0.25,  # Slightly lower for computational efficiency
    min_max_parents=2,
    max_max_parents=8
)