"""
Test SCM factories for experimental validation and testing.

This module provides factory functions for creating common test SCMs with
known structure and parameters, useful for validation and development.
"""

# Standard library imports
import logging
from typing import Dict, List, Optional, Tuple

# Third-party imports
import pyrsistent as pyr

# Local imports
from ..mechanisms.linear import create_linear_mechanism, create_root_mechanism
from ..data_structures.scm import create_scm

# Constants
DEFAULT_NOISE_SCALE = 1.0
DEFAULT_INTERCEPT = 0.0

logger = logging.getLogger(__name__)


# Private validation functions (shared across factories)
def _validate_noise_scale(noise_scale: float) -> None:
    """
    Validate noise scale parameter.
    
    Args:
        noise_scale: Standard deviation of noise
        
    Raises:
        ValueError: If noise scale is invalid
    """
    import jax.numpy as jnp
    
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
    from ..data_structures.scm import validate_mechanisms, topological_sort, get_variables, get_mechanisms
    
    if not validate_mechanisms(scm):
        missing = set(get_variables(scm)) - set(get_mechanisms(scm).keys())
        raise ValueError(f"SCM missing mechanisms for variables: {sorted(missing)}")
    
    try:
        topological_sort(scm)
    except ValueError as e:
        raise ValueError(f"SCM is not valid for sampling: {e}") from e


# Public factory functions
def create_simple_linear_scm(
    variables: List[str],
    edges: List[Tuple[str, str]],
    coefficients: Dict[Tuple[str, str], float],
    noise_scales: Dict[str, float],
    target: str,
    intercepts: Optional[Dict[str, float]] = None,
    variable_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    output_bounds: Optional[Tuple[float, float]] = None
) -> pyr.PMap:
    """
    Factory function for creating simple linear SCMs with validation.
    
    Creates an SCM where each variable follows a linear mechanism with Gaussian noise.
    All validation is performed to ensure the resulting SCM is consistent and acyclic.
    
    Args:
        variables: Variable names in the SCM
        edges: List of (parent, child) pairs defining the causal graph
        coefficients: Mapping from (parent, child) edges to coefficients
        noise_scales: Mapping from variable names to noise standard deviations
        target: Target variable name (must be in variables)
        intercepts: Optional intercepts for each variable (default: 0.0)
        variable_ranges: Optional ranges for intervention values (default: (-10, 10))
        output_bounds: Optional (min, max) bounds for mechanism outputs
        
    Returns:
        A validated SCM ready for sampling and analysis
        
    Raises:
        ValueError: If the specification is invalid or results in cycles
        
    Example:
        >>> # Create SCM: X → Y ← Z with linear relationships
        >>> scm = create_simple_linear_scm(
        ...     variables=['X', 'Y', 'Z'],
        ...     edges=[('X', 'Y'), ('Z', 'Y')],
        ...     coefficients={('X', 'Y'): 2.0, ('Z', 'Y'): -1.5},
        ...     noise_scales={'X': 1.0, 'Y': 0.5, 'Z': 1.0},
        ...     target='Y'
        ... )
    """
    # Input validation
    if not variables:
        raise ValueError("Variables list cannot be empty")
    
    if target not in variables:
        raise ValueError(f"Target variable '{target}' not in variables list: {variables}")
    
    variables_set = set(variables)
    for parent, child in edges:
        if parent not in variables_set:
            raise ValueError(f"Edge parent '{parent}' not in variables: {variables}")
        if child not in variables_set:
            raise ValueError(f"Edge child '{child}' not in variables: {variables}")
    
    # Validate coefficients
    edges_set = set(edges)
    missing_coeffs = edges_set - set(coefficients.keys())
    if missing_coeffs:
        raise ValueError(f"Missing coefficients for edges: {sorted(missing_coeffs)}")
    
    extra_coeffs = set(coefficients.keys()) - edges_set
    if extra_coeffs:
        raise ValueError(f"Coefficients provided for non-existent edges: {sorted(extra_coeffs)}")
    
    # Validate noise scales
    missing_noise = set(variables) - set(noise_scales.keys())
    if missing_noise:
        raise ValueError(f"Missing noise scales for variables: {sorted(missing_noise)}")
    
    for var, scale in noise_scales.items():
        if var not in variables_set:
            raise ValueError(f"Noise scale provided for unknown variable: {var}")
        _validate_noise_scale(scale)
    
    # Set default intercepts
    if intercepts is None:
        intercepts = {var: DEFAULT_INTERCEPT for var in variables}
    else:
        # Fill in missing intercepts
        intercepts = {var: intercepts.get(var, DEFAULT_INTERCEPT) for var in variables}
    
    # Set default variable ranges with some variety
    import random
    if variable_ranges is None:
        # Create varied but reasonable ranges for each variable
        range_options = [(-5.0, 5.0), (-10.0, 10.0), (-2.0, 2.0), (-3.0, 3.0)]
        rng = random.Random(42)  # Fixed seed for reproducibility
        variable_ranges = {}
        for var in variables:
            # Select a range from options based on variable index
            idx = hash(var) % len(range_options)
            variable_ranges[var] = range_options[idx]
    else:
        # Fill in missing ranges with default
        for var in variables:
            if var not in variable_ranges:
                variable_ranges[var] = (-10.0, 10.0)
    
    # Build mechanisms for each variable
    mechanisms = {}
    
    for var in variables:
        # Find parents and their coefficients
        parents = [parent for parent, child in edges if child == var]
        var_coefficients = {parent: coefficients[(parent, var)] for parent in parents}
        
        # Create mechanism with variable-specific bounds from ranges
        var_bounds = variable_ranges.get(var, output_bounds)
        mechanism = create_linear_mechanism(
            parents=parents,
            coefficients=var_coefficients,
            intercept=intercepts[var],
            noise_scale=noise_scales[var],
            output_bounds=var_bounds,  # Use variable-specific bounds
            variable_name=var,
            scm_metadata={'variable_ranges': variable_ranges}
        )
        mechanisms[var] = mechanism
    
    # Create SCM with variable ranges in metadata
    scm = create_scm(
        variables=frozenset(variables),
        edges=frozenset(edges),
        mechanisms=pyr.pmap(mechanisms),  # Convert dict to pmap for consistency
        target=target,
        metadata={
            'mechanism_type': 'linear',
            'created_by': 'test_factory',
            'variable_ranges': variable_ranges
        }
    )
    
    # Final validation
    _validate_scm_for_sampling(scm)
    
    logger.info(f"Created linear SCM with {len(variables)} variables, {len(edges)} edges, target='{target}'")
    return scm


def create_simple_test_scm(
    noise_scale: float = 1.0,
    target: str = "Y"
) -> pyr.PMap:
    """
    Create a simple 3-variable linear SCM for testing: X → Y ← Z.
    
    This is the standard test SCM used throughout Phase 1 development.
    The relationships are:
    - X ~ Normal(0, noise_scale)
    - Z ~ Normal(0, noise_scale)  
    - Y = 2.0 * X - 1.5 * Z + Normal(0, noise_scale)
    
    Args:
        noise_scale: Standard deviation for all noise terms
        target: Target variable name
        
    Returns:
        A simple linear SCM suitable for testing
        
    Example:
        >>> scm = create_simple_test_scm(noise_scale=0.5, target="Y")
        >>> # This SCM is ready for use in sampling and AVICI conversion tests
    """
    return create_simple_linear_scm(
        variables=['X', 'Y', 'Z'],
        edges=[('X', 'Y'), ('Z', 'Y')],
        coefficients={('X', 'Y'): 2.0, ('Z', 'Y'): -1.5},
        noise_scales={'X': noise_scale, 'Y': noise_scale, 'Z': noise_scale},
        target=target
    )


def create_chain_test_scm(
    chain_length: int = 3,
    coefficient: float = 1.5,
    noise_scale: float = 1.0,
    target: Optional[str] = None
) -> pyr.PMap:
    """
    Create a linear chain SCM for testing: X0 → X1 → X2 → ... → X_{n-1}.
    
    Each relationship is: X_{i+1} = coefficient * X_i + noise
    Useful for testing topological ordering and longer causal chains.
    
    Args:
        chain_length: Length of the chain (must be >= 2)
        coefficient: Coefficient for each link in the chain
        noise_scale: Standard deviation for all noise terms
        target: Target variable name (default: last variable in chain)
        
    Returns:
        A linear chain SCM
        
    Raises:
        ValueError: If chain_length < 2
        
    Example:
        >>> scm = create_chain_test_scm(chain_length=4, coefficient=0.8)
        >>> # Creates: X0 → X1 → X2 → X3 with coefficient 0.8
    """
    if chain_length < 2:
        raise ValueError("Chain length must be at least 2")
    
    # Create variable names
    variables = [f"X{i}" for i in range(chain_length)]
    
    # Create edges: X0 → X1 → X2 → ...
    edges = [(f"X{i}", f"X{i+1}") for i in range(chain_length - 1)]
    
    # Create coefficients
    coefficients = {edge: coefficient for edge in edges}
    
    # Create noise scales
    noise_scales = {var: noise_scale for var in variables}
    
    # Set target to last variable if not specified
    if target is None:
        target = f"X{chain_length - 1}"
    
    return create_simple_linear_scm(
        variables=variables,
        edges=edges,
        coefficients=coefficients,
        noise_scales=noise_scales,
        target=target
    )


def create_collider_test_scm(
    noise_scale: float = 1.0,
    target: str = "Z"
) -> pyr.PMap:
    """
    Create a collider structure SCM for testing: X → Z ← Y.
    
    This tests a different causal structure than the simple test SCM.
    The relationships are:
    - X ~ Normal(0, noise_scale)
    - Y ~ Normal(0, noise_scale)
    - Z = 1.5 * X + 2.0 * Y + Normal(0, noise_scale)
    
    Args:
        noise_scale: Standard deviation for all noise terms
        target: Target variable name
        
    Returns:
        A collider structure SCM suitable for testing
        
    Example:
        >>> scm = create_collider_test_scm(noise_scale=0.8, target="Z")
        >>> # This tests collider (common effect) structures
    """
    return create_simple_linear_scm(
        variables=['X', 'Y', 'Z'],
        edges=[('X', 'Z'), ('Y', 'Z')],
        coefficients={('X', 'Z'): 1.5, ('Y', 'Z'): 2.0},
        noise_scales={'X': noise_scale, 'Y': noise_scale, 'Z': noise_scale},
        target=target
    )


def create_fork_test_scm(
    noise_scale: float = 1.0,
    target: str = "Y"
) -> pyr.PMap:
    """
    Create a fork structure SCM for testing: X ← Z → Y.
    
    This tests a confounding variable structure.
    The relationships are:
    - Z ~ Normal(0, noise_scale)
    - X = 1.0 * Z + Normal(0, noise_scale)
    - Y = 2.0 * Z + Normal(0, noise_scale)
    
    Args:
        noise_scale: Standard deviation for all noise terms
        target: Target variable name
        
    Returns:
        A fork structure SCM suitable for testing
        
    Example:
        >>> scm = create_fork_test_scm(noise_scale=0.5, target="Y") 
        >>> # This tests common cause (confounding) structures
    """
    return create_simple_linear_scm(
        variables=['X', 'Y', 'Z'],
        edges=[('Z', 'X'), ('Z', 'Y')],
        coefficients={('Z', 'X'): 1.0, ('Z', 'Y'): 2.0},
        noise_scales={'X': noise_scale, 'Y': noise_scale, 'Z': noise_scale},
        target=target
    )


# Utility functions for test SCMs
def get_scm_summary(scm: pyr.PMap) -> Dict[str, any]:
    """
    Get a summary of a test SCM's structure and properties.
    
    Args:
        scm: The SCM to summarize
        
    Returns:
        Dictionary containing SCM summary information
    """
    from ..data_structures.scm import get_variables, get_edges, topological_sort
    
    variables = get_variables(scm)
    edges = get_edges(scm)
    
    # Count root and leaf variables
    all_parents = {child for parent, child in edges}
    all_children = {parent for parent, child in edges}
    
    root_vars = variables - all_parents
    leaf_vars = variables - all_children
    
    return {
        'num_variables': len(variables),
        'num_edges': len(edges),
        'variables': sorted(variables),
        'edges': sorted(edges),
        'root_variables': sorted(root_vars),
        'leaf_variables': sorted(leaf_vars),
        'target': scm.get('target'),
        'topological_order': topological_sort(scm),
        'mechanism_type': scm.get('metadata', {}).get('mechanism_type', 'unknown')
    }
