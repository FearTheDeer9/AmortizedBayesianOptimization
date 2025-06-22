"""
JAX-Native Configuration System

Provides static configuration for JAX-optimized causal Bayesian optimization.
All configuration is immutable and established at initialization to enable
optimal JAX compilation with static shapes.

Key features:
- Immutable configuration with comprehensive validation
- Static tensor shapes throughout the pipeline
- Integer indexing for JAX compatibility
- Variable names stored for interpretation only (not used in JAX operations)
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import jax.numpy as jnp


@dataclass(frozen=True)
class JAXConfig:
    """
    Immutable configuration for JAX-native causal Bayesian optimization.
    
    This configuration establishes all static parameters needed for JAX compilation
    with fixed tensor shapes. Variable names are stored for interpretation but
    never used in JAX-compiled operations.
    
    Args:
        n_vars: Total number of variables in the problem
        target_idx: Index of the target variable (0 <= target_idx < n_vars)
        max_samples: Maximum number of samples to store in buffer
        max_history: Maximum history length for attention mechanisms
        variable_names: Names of variables (for interpretation only)
        mechanism_types: Type indices for each variable's mechanism
        feature_dim: Dimension of mechanism feature vectors
        
    Raises:
        ValueError: If configuration parameters are invalid
    """
    
    n_vars: int
    target_idx: int
    max_samples: int
    max_history: int
    variable_names: Tuple[str, ...]
    mechanism_types: Tuple[int, ...]
    feature_dim: int = 3  # Default: [effect, uncertainty, confidence]
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        # Basic parameter validation
        if self.n_vars <= 0:
            raise ValueError(f"n_vars must be positive, got {self.n_vars}")
        
        if not (0 <= self.target_idx < self.n_vars):
            raise ValueError(f"target_idx must be in [0, {self.n_vars}), got {self.target_idx}")
        
        if self.max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {self.max_samples}")
        
        if self.max_history <= 0:
            raise ValueError(f"max_history must be positive, got {self.max_history}")
        
        if self.feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {self.feature_dim}")
        
        # Tuple length validation
        if len(self.variable_names) != self.n_vars:
            raise ValueError(
                f"variable_names length ({len(self.variable_names)}) "
                f"must equal n_vars ({self.n_vars})"
            )
        
        if len(self.mechanism_types) != self.n_vars:
            raise ValueError(
                f"mechanism_types length ({len(self.mechanism_types)}) "
                f"must equal n_vars ({self.n_vars})"
            )
        
        # Variable name uniqueness
        if len(set(self.variable_names)) != len(self.variable_names):
            raise ValueError("Variable names must be unique")
        
        # Mechanism type validation
        if any(mt < 0 for mt in self.mechanism_types):
            raise ValueError("All mechanism types must be non-negative")
    
    def get_variable_name(self, idx: int) -> str:
        """Get variable name by index (for interpretation only)."""
        if not (0 <= idx < self.n_vars):
            raise ValueError(f"Index {idx} out of range [0, {self.n_vars})")
        return self.variable_names[idx]
    
    def get_target_name(self) -> str:
        """Get target variable name."""
        return self.variable_names[self.target_idx]
    
    def get_non_target_indices(self) -> Tuple[int, ...]:
        """Get indices of all non-target variables."""
        return tuple(i for i in range(self.n_vars) if i != self.target_idx)
    
    def create_target_mask(self) -> jnp.ndarray:
        """Create boolean mask for target variable (JAX-compatible)."""
        mask = jnp.zeros(self.n_vars, dtype=bool)
        return mask.at[self.target_idx].set(True)
    
    def create_non_target_mask(self) -> jnp.ndarray:
        """Create boolean mask for non-target variables (JAX-compatible)."""
        return ~self.create_target_mask()


def create_jax_config(
    variable_names: List[str],
    target_variable: str,
    max_samples: int = 1000,
    max_history: int = 100,
    mechanism_types: Optional[List[int]] = None,
    feature_dim: int = 3
) -> JAXConfig:
    """
    Create JAX configuration from variable specifications.
    
    Args:
        variable_names: List of variable names
        target_variable: Name of target variable
        max_samples: Maximum samples in buffer
        max_history: Maximum history length
        mechanism_types: Optional mechanism type indices
        feature_dim: Dimension of mechanism features
        
    Returns:
        Validated JAX configuration
        
    Raises:
        ValueError: If target variable not in variable names
    """
    if target_variable not in variable_names:
        raise ValueError(f"Target variable '{target_variable}' not found in {variable_names}")
    
    target_idx = variable_names.index(target_variable)
    n_vars = len(variable_names)
    
    # Default mechanism types if not provided
    if mechanism_types is None:
        mechanism_types = [4] * n_vars  # Default to 'unknown' type
    
    return JAXConfig(
        n_vars=n_vars,
        target_idx=target_idx,
        max_samples=max_samples,
        max_history=max_history,
        variable_names=tuple(variable_names),
        mechanism_types=tuple(mechanism_types),
        feature_dim=feature_dim
    )


def validate_jax_config(config: JAXConfig) -> None:
    """
    Validate JAX configuration for consistency.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # The dataclass __post_init__ already handles validation
    # This function is for additional runtime validation if needed
    
    # Check for reasonable parameter ranges
    if config.max_samples > 100000:
        raise ValueError(f"max_samples too large: {config.max_samples} > 100000")
    
    if config.max_history > config.max_samples:
        raise ValueError(
            f"max_history ({config.max_history}) cannot exceed "
            f"max_samples ({config.max_samples})"
        )
    
    if config.n_vars > 1000:
        raise ValueError(f"n_vars too large: {config.n_vars} > 1000")


def create_test_config() -> JAXConfig:
    """Create a test configuration for unit testing."""
    return create_jax_config(
        variable_names=['X', 'Y', 'Z'],
        target_variable='Y',
        max_samples=100,
        max_history=50,
        mechanism_types=[0, 1, 2],  # linear, polynomial, exponential
        feature_dim=3
    )