"""
Benchmark SCM Suite for Systematic Evaluation

This module provides a comprehensive suite of well-defined SCMs with known
causal structures for systematic evaluation of causal discovery methods.
Each SCM has documented characteristics and coefficients for reproducible research.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import pyrsistent as pyr

from .test_scms import create_simple_linear_scm
from ..data_structures.scm import create_scm
from ..mechanisms.linear import create_linear_mechanism, create_root_mechanism

logger = logging.getLogger(__name__)


def create_fork_scm(noise_scale: float = 1.0, target: str = "Y") -> pyr.PMap:
    """
    Create a simple fork structure: X → Y ← Z
    
    This is the classic "common effect" structure where two causes influence
    one target variable. Tests method ability to identify multiple parents.
    
    Structure: X ~ N(0,1), Z ~ N(0,1), Y = 2.0*X - 1.5*Z + N(0,1)
    
    Args:
        noise_scale: Standard deviation for noise terms
        target: Target variable name
        
    Returns:
        SCM with fork structure
    """
    scm = create_simple_linear_scm(
        variables=['X', 'Y', 'Z'],
        edges=[('X', 'Y'), ('Z', 'Y')],
        coefficients={('X', 'Y'): 2.0, ('Z', 'Y'): -1.5},
        noise_scales={'X': noise_scale, 'Y': noise_scale, 'Z': noise_scale},
        target=target
    )
    
    # Add metadata for identification
    metadata = scm.get('metadata', pyr.pmap({}))
    metadata = metadata.update({
        'structure_type': 'fork',
        'complexity': 'simple',
        'num_parents': 2,
        'coefficients': {('X', 'Y'): 2.0, ('Z', 'Y'): -1.5},
        'description': 'Simple fork: X -> Y <- Z'
    })
    scm = scm.set('metadata', metadata)
    
    logger.info("Created fork SCM: X -> Y <- Z")
    return scm


def create_chain_scm(chain_length: int = 3, coefficient: float = 1.5, 
                    noise_scale: float = 1.0) -> pyr.PMap:
    """
    Create a linear chain structure: X0 → X1 → X2 → ... → X_{n-1}
    
    Tests method ability to handle sequential causal relationships
    and distinguish direct vs indirect effects.
    
    Args:
        chain_length: Number of variables in chain (minimum 3)
        coefficient: Coefficient for each link
        noise_scale: Standard deviation for noise terms
        
    Returns:
        SCM with chain structure
    """
    if chain_length < 3:
        raise ValueError("Chain length must be at least 3")
    
    variables = [f"X{i}" for i in range(chain_length)]
    edges = [(f"X{i}", f"X{i+1}") for i in range(chain_length - 1)]
    coefficients = {edge: coefficient for edge in edges}
    noise_scales = {var: noise_scale for var in variables}
    target = f"X{chain_length - 1}"
    
    scm = create_simple_linear_scm(
        variables=variables,
        edges=edges,
        coefficients=coefficients,
        noise_scales=noise_scales,
        target=target
    )
    
    # Add metadata
    metadata = scm.get('metadata', pyr.pmap({}))
    metadata = metadata.update({
        'structure_type': 'chain',
        'complexity': 'simple',
        'chain_length': chain_length,
        'coefficients': coefficients,
        'description': f'Linear chain: X0 -> X1 -> ... -> X{chain_length-1}'
    })
    scm = scm.set('metadata', metadata)
    
    logger.info(f"Created chain SCM with {chain_length} variables")
    return scm


def create_collider_scm(noise_scale: float = 1.0) -> pyr.PMap:
    """
    Create a collider structure: X → Z ← Y
    
    Tests method ability to handle collider bias and identify
    when variables should NOT be conditioned on.
    
    Structure: X ~ N(0,1), Y ~ N(0,1), Z = 1.2*X + 0.8*Y + N(0,1)
    
    Args:
        noise_scale: Standard deviation for noise terms
        
    Returns:
        SCM with collider structure
    """
    scm = create_simple_linear_scm(
        variables=['X', 'Y', 'Z'],
        edges=[('X', 'Z'), ('Y', 'Z')],
        coefficients={('X', 'Z'): 1.2, ('Y', 'Z'): 0.8},
        noise_scales={'X': noise_scale, 'Y': noise_scale, 'Z': noise_scale},
        target='Z'
    )
    
    # Add metadata
    metadata = scm.get('metadata', pyr.pmap({}))
    metadata = metadata.update({
        'structure_type': 'collider',
        'complexity': 'simple',
        'num_parents': 2,
        'coefficients': {('X', 'Z'): 1.2, ('Y', 'Z'): 0.8},
        'description': 'Collider: X -> Z <- Y'
    })
    scm = scm.set('metadata', metadata)
    
    logger.info("Created collider SCM: X -> Z <- Y")
    return scm


def create_diamond_scm(noise_scale: float = 1.0) -> pyr.PMap:
    """
    Create a diamond structure: X → Y → W ← Z ← X
    
    Tests method ability to handle multiple paths and indirect effects.
    
    Structure:
    - X ~ N(0,1)
    - Y = 1.5*X + N(0,1)  
    - Z = -1.0*X + N(0,1)
    - W = 2.0*Y + 1.0*Z + N(0,1)
    
    Args:
        noise_scale: Standard deviation for noise terms
        
    Returns:
        SCM with diamond structure
    """
    scm = create_simple_linear_scm(
        variables=['X', 'Y', 'Z', 'W'],
        edges=[('X', 'Y'), ('X', 'Z'), ('Y', 'W'), ('Z', 'W')],
        coefficients={
            ('X', 'Y'): 1.5,
            ('X', 'Z'): -1.0,
            ('Y', 'W'): 2.0,
            ('Z', 'W'): 1.0
        },
        noise_scales={'X': noise_scale, 'Y': noise_scale, 'Z': noise_scale, 'W': noise_scale},
        target='W'
    )
    
    # Add metadata
    metadata = scm.get('metadata', pyr.pmap({}))
    metadata = metadata.update({
        'structure_type': 'diamond',
        'complexity': 'medium',
        'num_variables': 4,
        'coefficients': {
            ('X', 'Y'): 1.5, ('X', 'Z'): -1.0,
            ('Y', 'W'): 2.0, ('Z', 'W'): 1.0
        },
        'description': 'Diamond: X -> Y -> W <- Z <- X'
    })
    scm = scm.set('metadata', metadata)
    
    logger.info("Created diamond SCM: X -> Y -> W <- Z <- X")
    return scm


def create_butterfly_scm(noise_scale: float = 1.0) -> pyr.PMap:
    """
    Create a complex butterfly structure with 5 variables.
    
    Tests method ability to handle complex, realistic causal structures
    with multiple paths and varying coefficient magnitudes.
    
    Structure:
    - A ~ N(0,1), B ~ N(0,1)
    - C = 0.8*A + 1.2*B + N(0,1)
    - D = -0.5*A + 0.9*C + N(0,1)  
    - E = 1.1*C + 0.7*D + N(0,1)
    
    Args:
        noise_scale: Standard deviation for noise terms
        
    Returns:
        SCM with butterfly structure
    """
    scm = create_simple_linear_scm(
        variables=['A', 'B', 'C', 'D', 'E'],
        edges=[('A', 'C'), ('B', 'C'), ('A', 'D'), ('C', 'D'), ('C', 'E'), ('D', 'E')],
        coefficients={
            ('A', 'C'): 0.8,
            ('B', 'C'): 1.2,
            ('A', 'D'): -0.5,
            ('C', 'D'): 0.9,
            ('C', 'E'): 1.1,
            ('D', 'E'): 0.7
        },
        noise_scales={'A': noise_scale, 'B': noise_scale, 'C': noise_scale, 
                     'D': noise_scale, 'E': noise_scale},
        target='E'
    )
    
    # Add metadata
    metadata = scm.get('metadata', pyr.pmap({}))
    metadata = metadata.update({
        'structure_type': 'butterfly',
        'complexity': 'high',
        'num_variables': 5,
        'coefficients': {
            ('A', 'C'): 0.8, ('B', 'C'): 1.2, ('A', 'D'): -0.5,
            ('C', 'D'): 0.9, ('C', 'E'): 1.1, ('D', 'E'): 0.7
        },
        'description': 'Complex butterfly structure with multiple paths'
    })
    scm = scm.set('metadata', metadata)
    
    logger.info("Created butterfly SCM with 5 variables")
    return scm


def create_dense_scm(num_vars: int = 4, edge_prob: float = 0.7, 
                    noise_scale: float = 1.0) -> pyr.PMap:
    """
    Create a dense SCM with high edge connectivity.
    
    Tests method performance on densely connected graphs where
    many variables influence the target.
    
    Args:
        num_vars: Number of variables
        edge_prob: Probability of edge between variables
        noise_scale: Standard deviation for noise terms
        
    Returns:
        SCM with dense structure
    """
    from .benchmark_graphs import create_erdos_renyi_scm
    
    scm = create_erdos_renyi_scm(
        n_nodes=num_vars,
        edge_prob=edge_prob,
        noise_scale=noise_scale,
        seed=123  # Fixed seed for reproducibility
    )
    
    # Update metadata to indicate this is a dense benchmark
    metadata = scm.get('metadata', pyr.pmap({}))
    metadata = metadata.update({
        'structure_type': 'dense',
        'complexity': 'high',
        'edge_density': edge_prob,
        'description': f'Dense graph with {edge_prob:.1%} edge probability'
    })
    scm = scm.set('metadata', metadata)
    
    logger.info(f"Created dense SCM with {num_vars} variables, density={edge_prob:.2f}")
    return scm


def create_sparse_scm(num_vars: int = 5, edge_prob: float = 0.2,
                     noise_scale: float = 1.0) -> pyr.PMap:
    """
    Create a sparse SCM with low edge connectivity.
    
    Tests method performance on sparsely connected graphs where
    few variables influence the target.
    
    Args:
        num_vars: Number of variables
        edge_prob: Probability of edge between variables
        noise_scale: Standard deviation for noise terms
        
    Returns:
        SCM with sparse structure
    """
    from .benchmark_graphs import create_erdos_renyi_scm
    
    scm = create_erdos_renyi_scm(
        n_nodes=num_vars,
        edge_prob=edge_prob,
        noise_scale=noise_scale,
        seed=456  # Fixed seed for reproducibility
    )
    
    # Update metadata to indicate this is a sparse benchmark
    metadata = scm.get('metadata', pyr.pmap({}))
    metadata = metadata.update({
        'structure_type': 'sparse',
        'complexity': 'medium',
        'edge_density': edge_prob,
        'description': f'Sparse graph with {edge_prob:.1%} edge probability'
    })
    scm = scm.set('metadata', metadata)
    
    logger.info(f"Created sparse SCM with {num_vars} variables, density={edge_prob:.2f}")
    return scm


def create_mixed_coeff_scm(noise_scale: float = 1.0) -> pyr.PMap:
    """
    Create SCM with mixed positive/negative coefficients.
    
    Tests method ability to handle both positive and negative
    causal relationships and their interaction effects.
    
    Structure:
    - A ~ N(0,1), B ~ N(0,1), C ~ N(0,1)
    - D = 2.0*A - 1.5*B + 0.5*C + N(0,1)
    
    Args:
        noise_scale: Standard deviation for noise terms
        
    Returns:
        SCM with mixed coefficient signs
    """
    scm = create_simple_linear_scm(
        variables=['A', 'B', 'C', 'D'],
        edges=[('A', 'D'), ('B', 'D'), ('C', 'D')],
        coefficients={('A', 'D'): 2.0, ('B', 'D'): -1.5, ('C', 'D'): 0.5},
        noise_scales={'A': noise_scale, 'B': noise_scale, 'C': noise_scale, 'D': noise_scale},
        target='D'
    )
    
    # Add metadata
    metadata = scm.get('metadata', pyr.pmap({}))
    metadata = metadata.update({
        'structure_type': 'mixed_coefficients',
        'complexity': 'medium',
        'num_parents': 3,
        'coefficients': {('A', 'D'): 2.0, ('B', 'D'): -1.5, ('C', 'D'): 0.5},
        'description': 'Mixed positive/negative coefficients: +A -B +C -> D'
    })
    scm = scm.set('metadata', metadata)
    
    logger.info("Created mixed coefficient SCM: A(+2.0), B(-1.5), C(+0.5) -> D")
    return scm


def create_scm_suite() -> Dict[str, pyr.PMap]:
    """
    Create comprehensive suite of benchmark SCMs for systematic evaluation.
    
    Returns:
        Dictionary mapping SCM names to SCM objects with metadata
    """
    suite = {
        # Simple structures (3 variables)
        "fork_3var": create_fork_scm(),
        "chain_3var": create_chain_scm(3),
        "collider_3var": create_collider_scm(),
        
        # Medium complexity (4 variables)
        "diamond_4var": create_diamond_scm(),
        "mixed_coeffs_4var": create_mixed_coeff_scm(),
        "dense_4var": create_dense_scm(4, 0.7),
        
        # Higher complexity (5 variables)
        "butterfly_5var": create_butterfly_scm(),
        "sparse_5var": create_sparse_scm(5, 0.2),
        "chain_5var": create_chain_scm(5),
    }
    
    logger.info(f"Created SCM suite with {len(suite)} benchmark structures")
    return suite


def get_scm_characteristics(scm: pyr.PMap) -> Dict[str, Any]:
    """
    Extract characteristics from an SCM for analysis and logging.
    
    Args:
        scm: SCM to analyze
        
    Returns:
        Dictionary with SCM characteristics
    """
    from ..data_structures.scm import get_variables, get_edges, get_target, get_parents
    
    variables = get_variables(scm)
    edges = get_edges(scm)
    target = get_target(scm)
    metadata = scm.get('metadata', pyr.pmap({}))
    
    # Compute derived characteristics
    num_vars = len(variables)
    num_edges = len(edges)
    edge_density = num_edges / (num_vars * (num_vars - 1)) if num_vars > 1 else 0
    
    # Get target parents
    target_parents = list(get_parents(scm, target)) if target else []
    
    characteristics = {
        'num_variables': num_vars,
        'num_edges': num_edges,
        'edge_density': edge_density,
        'target_variable': target,
        'num_target_parents': len(target_parents),
        'target_parents': target_parents,
        'structure_type': metadata.get('structure_type', 'unknown'),
        'complexity': metadata.get('complexity', 'unknown'),
        'description': metadata.get('description', 'No description'),
        'coefficients': dict(metadata.get('coefficients', {})),
    }
    
    return characteristics


# Export functions for easy import
__all__ = [
    'create_scm_suite',
    'create_fork_scm',
    'create_chain_scm', 
    'create_collider_scm',
    'create_diamond_scm',
    'create_butterfly_scm',
    'create_dense_scm',
    'create_sparse_scm',
    'create_mixed_coeff_scm',
    'get_scm_characteristics'
]