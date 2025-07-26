"""
Linear mechanism implementations for causal models.

This module provides functions for creating linear causal mechanisms
following functional programming principles.
"""

# Standard library imports
import logging
from typing import Dict, List, Callable, Any

# Third-party imports
import jax
import jax.numpy as jnp
import jax.random as random
import pyrsistent as pyr

# Local imports (relative)
from ..data_structures.scm import (
    get_parents, topological_sort, 
    get_variables, get_mechanisms, validate_mechanisms
)
from ..data_structures.sample import create_sample
from .serializable_mechanisms import LinearMechanism, RootMechanism

# Type aliases
MechanismFunction = Callable[[Dict[str, float], jax.Array], float]
SampleList = List[pyr.PMap]

# Constants
DEFAULT_NOISE_SCALE = 1.0
DEFAULT_INTERCEPT = 0.0

logger = logging.getLogger(__name__)


# Private implementation functions
def _create_linear_mechanism_impl(
    coefficients: Dict[str, float],
    intercept: float,
    noise_scale: float
) -> MechanismFunction:
    """
    Core implementation for creating a linear mechanism function.
    
    Args:
        coefficients: Mapping from parent variable names to coefficients
        intercept: Constant term in the linear equation
        noise_scale: Standard deviation of Gaussian noise
        
    Returns:
        A mechanism function that computes: intercept + sum(coef * parent) + noise
    """
    # Convert to JAX arrays for efficient computation
    parent_names = list(coefficients.keys())
    coeff_array = jnp.array([coefficients[name] for name in parent_names])
    
    def mechanism(parent_values: Dict[str, float], noise_key: jax.Array) -> float:
        """
        Linear mechanism: Y = intercept + sum(coeff_i * X_i) + noise.
        
        Args:
            parent_values: Dictionary mapping parent names to their values
            noise_key: JAX random key for noise generation
            
        Returns:
            Computed value for this variable
        """
        # Extract parent values in the same order as coefficients
        parent_vals = jnp.array([parent_values[name] for name in parent_names])
        
        # Linear combination: sum(coeff_i * X_i)
        linear_term = jnp.dot(coeff_array, parent_vals)
        
        # Add intercept
        result = linear_term + intercept
        
        # Add Gaussian noise
        if noise_scale > 0:
            noise = random.normal(noise_key) * noise_scale
            result = result + noise
            
        return float(result)
    
    return mechanism


def _create_root_mechanism_impl(
    mean: float,
    noise_scale: float
) -> MechanismFunction:
    """
    Create a mechanism for root variables (variables with no parents).
    
    Args:
        mean: Mean value for the root variable
        noise_scale: Standard deviation of Gaussian noise
        
    Returns:
        A mechanism function that generates: mean + noise
    """
    def mechanism(parent_values: Dict[str, float], noise_key: jax.Array) -> float:
        """Root mechanism: Y = mean + noise."""
        if parent_values:
            logger.warning(f"Root mechanism called with non-empty parent values: {parent_values}")
        
        noise = random.normal(noise_key) * noise_scale
        return float(mean + noise)
    
    return mechanism


def _sample_variable_value(
    variable: str,
    mechanism: MechanismFunction,
    parent_values: Dict[str, float],
    noise_key: jax.Array
) -> float:
    """
    Sample a value for a single variable using its mechanism.
    
    Args:
        variable: Name of the variable being sampled
        mechanism: The mechanism function for this variable
        parent_values: Values of parent variables
        noise_key: Random key for noise generation
        
    Returns:
        Sampled value for the variable
    """
    try:
        return mechanism(parent_values, noise_key)
    except Exception as e:
        raise RuntimeError(
            f"Failed to sample variable '{variable}' with parents {list(parent_values.keys())}: {e}"
        ) from e


# Validation functions
def _validate_mechanism_inputs(
    parents: List[str],
    coefficients: Dict[str, float]
) -> None:
    """
    Validate inputs for linear mechanism creation.
    
    Args:
        parents: List of parent variable names
        coefficients: Dictionary mapping parent names to coefficients
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not parents and coefficients:
        raise ValueError("Cannot specify coefficients for a variable with no parents")
    
    if parents and not coefficients:
        raise ValueError("Must provide coefficients for variables with parents")
    
    # Check that all parents have coefficients
    missing_coeffs = set(parents) - set(coefficients.keys())
    if missing_coeffs:
        raise ValueError(f"Missing coefficients for parents: {sorted(missing_coeffs)}")
    
    # Check that no extra coefficients are provided
    extra_coeffs = set(coefficients.keys()) - set(parents)
    if extra_coeffs:
        raise ValueError(f"Coefficients provided for non-parent variables: {sorted(extra_coeffs)}")
    
    # Validate coefficient values
    for parent, coeff in coefficients.items():
        if not isinstance(coeff, (int, float)) or not jnp.isfinite(coeff):
            raise ValueError(f"Invalid coefficient for parent '{parent}': {coeff}")


def _validate_noise_scale(noise_scale: float) -> None:
    """
    Validate noise scale parameter.
    
    Args:
        noise_scale: Standard deviation of noise
        
    Raises:
        ValueError: If noise scale is invalid
    """
    if not isinstance(noise_scale, (int, float)) or not jnp.isfinite(noise_scale):
        raise ValueError(f"Noise scale must be a finite number, got: {noise_scale}")
    
    if noise_scale < 0:
        raise ValueError(f"Noise scale must be non-negative, got: {noise_scale}")


def _validate_scm_for_sampling(scm: pyr.PMap) -> None:
    """
    Validate that an SCM is ready for sampling.
    
    Args:
        scm: The SCM to validate
        
    Raises:
        ValueError: If SCM is invalid for sampling
    """
    if not validate_mechanisms(scm):
        missing = set(get_variables(scm)) - set(get_mechanisms(scm).keys())
        raise ValueError(f"SCM missing mechanisms for variables: {sorted(missing)}")
    
    try:
        topological_sort(scm)
    except ValueError as e:
        raise ValueError(f"SCM is not valid for sampling: {e}") from e


# Public API functions
def create_linear_mechanism(
    parents: List[str],
    coefficients: Dict[str, float],
    intercept: float = DEFAULT_INTERCEPT,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    _return_descriptor: bool = False
) -> MechanismFunction:
    """
    Create a linear mechanism function for a variable.
    
    Creates a mechanism that computes: Y = intercept + sum(coeff_i * X_i) + noise,
    where X_i are the parent variables and coeff_i are their coefficients.
    
    Args:
        parents: List of parent variable names (can be empty for root variables)
        coefficients: Mapping from parent variable names to their coefficients
        intercept: Constant term in the linear equation
        noise_scale: Standard deviation of Gaussian noise (>= 0)
        _return_descriptor: If True, return (function, descriptor) tuple for serialization
        
    Returns:
        A mechanism function that takes (parent_values, noise_key) and returns a scalar.
        If _return_descriptor=True, returns (function, descriptor) tuple.
        
    Raises:
        ValueError: If inputs are inconsistent or invalid
        
    Example:
        >>> # Variable Y with parents X and Z: Y = 1.0 + 2.0*X - 1.5*Z + noise
        >>> mechanism = create_linear_mechanism(
        ...     parents=['X', 'Z'],
        ...     coefficients={'X': 2.0, 'Z': -1.5},
        ...     intercept=1.0,
        ...     noise_scale=0.1
        ... )
        >>> key = random.PRNGKey(42)
        >>> value = mechanism({'X': 1.0, 'Z': 2.0}, key)
    """
    # Validate inputs
    _validate_mechanism_inputs(parents, coefficients)
    _validate_noise_scale(noise_scale)
    
    if not isinstance(intercept, (int, float)) or not jnp.isfinite(intercept):
        raise ValueError(f"Intercept must be a finite number, got: {intercept}")
    
    # Handle root variables (no parents)
    if not parents:
        logger.debug(f"Creating root mechanism with mean={intercept}, noise_scale={noise_scale}")
        mechanism = RootMechanism(intercept, noise_scale)
        
        if _return_descriptor:
            from .descriptors import RootMechanismDescriptor
            descriptor = RootMechanismDescriptor(
                mean=intercept,
                noise_scale=noise_scale
            )
            return mechanism, descriptor
        
        return mechanism
    
    # Create linear mechanism
    logger.debug(f"Creating linear mechanism with {len(parents)} parents, intercept={intercept}")
    mechanism = LinearMechanism(coefficients, intercept, noise_scale)
    
    if _return_descriptor:
        from .descriptors import LinearMechanismDescriptor
        descriptor = LinearMechanismDescriptor(
            parents=parents,
            coefficients=coefficients,
            intercept=intercept,
            noise_scale=noise_scale
        )
        return mechanism, descriptor
    
    return mechanism


def create_root_mechanism(
    mean: float = 0.0,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    _return_descriptor: bool = False
) -> MechanismFunction:
    """
    Create a mechanism for root variables (variables with no parents).
    
    Args:
        mean: Mean value for the root variable
        noise_scale: Standard deviation of Gaussian noise
        _return_descriptor: If True, return (function, descriptor) tuple for serialization
        
    Returns:
        A mechanism function for a root variable.
        If _return_descriptor=True, returns (function, descriptor) tuple.
        
    Raises:
        ValueError: If parameters are invalid
        
    Example:
        >>> mechanism = create_root_mechanism(mean=5.0, noise_scale=1.0)
        >>> key = random.PRNGKey(42)
        >>> value = mechanism({}, key)  # Empty parent_values for root
    """
    _validate_noise_scale(noise_scale)
    
    if not isinstance(mean, (int, float)) or not jnp.isfinite(mean):
        raise ValueError(f"Mean must be a finite number, got: {mean}")
    
    mechanism = RootMechanism(mean, noise_scale)
    
    if _return_descriptor:
        from .descriptors import RootMechanismDescriptor
        descriptor = RootMechanismDescriptor(
            mean=mean,
            noise_scale=noise_scale
        )
        return mechanism, descriptor
    
    return mechanism


def sample_from_linear_scm(
    scm: pyr.PMap,
    n_samples: int,
    seed: int = 42
) -> SampleList:
    """
    Generate observational samples from a linear SCM.
    
    Samples variables in topological order to respect causal dependencies.
    Each sample is generated independently with proper random key threading.
    
    Args:
        scm: The structural causal model to sample from
        n_samples: Number of samples to generate (must be positive)
        seed: Random seed for reproducible sampling
        
    Returns:
        List of Sample objects containing the generated data
        
    Raises:
        ValueError: If SCM is invalid or n_samples is not positive
        
    Example:
        >>> from ..experiments.test_scms import create_simple_test_scm
        >>> scm = create_simple_test_scm()
        >>> samples = sample_from_linear_scm(scm, n_samples=100, seed=42)
        >>> len(samples)
        100
    """
    # Validate inputs
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive integer, got: {n_samples}")
    
    _validate_scm_for_sampling(scm)
    
    # Get sampling order and mechanisms
    variables = get_variables(scm)
    variable_order = topological_sort(scm)
    mechanisms = get_mechanisms(scm)
    
    logger.debug(f"Sampling {n_samples} samples from SCM with variables: {variable_order}")
    
    # Initialize random key
    key = random.PRNGKey(seed)
    samples = []
    
    for sample_idx in range(n_samples):
        # Split key for this sample
        key, sample_key = random.split(key)
        
        # Initialize values for this sample
        sample_values = {}
        
        # Sample variables in topological order
        variable_keys = random.split(sample_key, len(variable_order))
        
        for var, var_key in zip(variable_order, variable_keys):
            # Get parent values for this variable
            parents = get_parents(scm, var)
            parent_values = {parent: sample_values[parent] for parent in parents}
            
            # Sample this variable
            mechanism = mechanisms[var]
            value = _sample_variable_value(var, mechanism, parent_values, var_key)
            sample_values[var] = value
        
        # Create Sample object
        sample = create_sample(values=sample_values)
        samples.append(sample)
    
    logger.debug(f"Successfully generated {len(samples)} samples")
    return samples
