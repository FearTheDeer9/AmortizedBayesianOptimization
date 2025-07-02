"""
Benchmark Graph Generation using NetworkX

This module provides functions for generating standard benchmark graphs
for causal discovery evaluation, following functional programming principles.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, FrozenSet
import warnings

# Standard numerical libraries
import jax.numpy as jnp
import jax.random as random
import numpy as onp  # For I/O only, following CLAUDE.md
import pyrsistent as pyr

# NetworkX integration (available via PARENT_SCALE)
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    warnings.warn("NetworkX not available. Some graph generation functions will not work.")
    NETWORKX_AVAILABLE = False

# Local imports
from ..data_structures.scm import create_scm
from ..mechanisms.linear import create_linear_mechanism, create_root_mechanism

logger = logging.getLogger(__name__)

# Type aliases
GraphStructure = Tuple[List[str], List[Tuple[str, str]]]
CoefficientsDict = Dict[Tuple[str, str], float]


def check_networkx_availability() -> bool:
    """
    Check if NetworkX is available for graph generation.
    
    Returns:
        bool: True if NetworkX is available, False otherwise
    """
    return NETWORKX_AVAILABLE


def _validate_graph_size(n_nodes: int) -> None:
    """
    Validate graph size parameters.
    
    Args:
        n_nodes: Number of nodes in the graph
        
    Raises:
        ValueError: If graph size is invalid
    """
    if not isinstance(n_nodes, int) or n_nodes < 2:
        raise ValueError(f"n_nodes must be an integer >= 2, got: {n_nodes}")
    
    if n_nodes > 1000:
        warnings.warn(f"Large graph size {n_nodes} may cause performance issues")


def _validate_probability(prob: float, name: str) -> None:
    """
    Validate probability parameters.
    
    Args:
        prob: Probability value to validate
        name: Parameter name for error messages
        
    Raises:
        ValueError: If probability is invalid
    """
    if not isinstance(prob, (int, float)):
        raise ValueError(f"{name} must be a number, got: {type(prob)}")
    
    if not (0.0 <= prob <= 1.0):
        raise ValueError(f"{name} must be between 0 and 1, got: {prob}")


def _networkx_to_structure(
    nx_graph: Any,
    variable_prefix: str = "X"
) -> GraphStructure:
    """
    Convert NetworkX graph to variable names and edge list.
    
    Args:
        nx_graph: NetworkX DiGraph object
        variable_prefix: Prefix for variable names
        
    Returns:
        Tuple of (variables, edges) where variables are strings and edges are (parent, child) pairs
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError("NetworkX not available for graph conversion")
    
    # Create variable names
    n_nodes = nx_graph.number_of_nodes()
    variables = [f"{variable_prefix}{i}" for i in range(n_nodes)]
    
    # Convert edges from node indices to variable names
    edges = []
    for source, target in nx_graph.edges():
        parent_var = f"{variable_prefix}{source}"
        child_var = f"{variable_prefix}{target}"
        edges.append((parent_var, child_var))
    
    return variables, edges


def _generate_random_coefficients(
    edges: List[Tuple[str, str]],
    coeff_range: Tuple[float, float] = (-2.0, 2.0),
    avoid_zero_range: Tuple[float, float] = (-0.1, 0.1),
    seed: int = 42
) -> CoefficientsDict:
    """
    Generate random coefficients for edges, avoiding near-zero values.
    
    Args:
        edges: List of (parent, child) edge pairs
        coeff_range: Range for coefficient values
        avoid_zero_range: Range around zero to avoid
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping edges to coefficients
    """
    rng = onp.random.RandomState(seed)
    coefficients = {}
    
    min_coeff, max_coeff = coeff_range
    avoid_min, avoid_max = avoid_zero_range
    
    for edge in edges:
        # Generate coefficient avoiding near-zero values
        while True:
            coeff = rng.uniform(min_coeff, max_coeff)
            if not (avoid_min <= coeff <= avoid_max):
                break
        
        coefficients[edge] = float(coeff)
    
    return coefficients


def create_erdos_renyi_scm(
    n_nodes: int,
    edge_prob: float,
    target_variable: Optional[str] = None,
    coeff_range: Tuple[float, float] = (-2.0, 2.0),
    noise_scale: float = 1.0,
    seed: int = 42,
    variable_prefix: str = "X"
) -> pyr.PMap:
    """
    Create a linear SCM from an Erdos-Renyi random directed graph.
    
    Args:
        n_nodes: Number of variables/nodes
        edge_prob: Probability of edge between any two nodes
        target_variable: Target variable name (default: last variable)
        coeff_range: Range for random edge coefficients
        noise_scale: Standard deviation for noise terms
        seed: Random seed for reproducibility
        variable_prefix: Prefix for variable names
        
    Returns:
        An immutable SCM with linear mechanisms
        
    Raises:
        ValueError: If parameters are invalid
        ImportError: If NetworkX is not available
        
    Example:
        >>> scm = create_erdos_renyi_scm(n_nodes=10, edge_prob=0.3, seed=42)
        >>> variables = get_variables(scm)
        >>> len(variables)
        10
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError("NetworkX required for Erdos-Renyi graph generation")
    
    # Validate inputs
    _validate_graph_size(n_nodes)
    _validate_probability(edge_prob, "edge_prob")
    
    if noise_scale <= 0:
        raise ValueError(f"noise_scale must be positive, got: {noise_scale}")
    
    # Generate directed acyclic graph using NetworkX
    rng = onp.random.RandomState(seed)
    
    # Generate Erdos-Renyi graph and make it DAG by removing back edges
    G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=seed, directed=True)
    
    # Ensure it's a DAG by removing edges that would create cycles
    # Use simple node ordering: only keep edges i -> j where i < j
    edges_to_remove = []
    for source, target in G.edges():
        if source >= target:  # Remove back edges and self-loops
            edges_to_remove.append((source, target))
    
    G.remove_edges_from(edges_to_remove)
    
    # Verify it's now a DAG
    if not nx.is_directed_acyclic_graph(G):
        # If still not a DAG, create a simple chain structure
        G.clear_edges()
        # Add a few edges to create a simple DAG
        for i in range(min(n_nodes-1, int(edge_prob * n_nodes * (n_nodes-1) / 2))):
            if i + 1 < n_nodes:
                G.add_edge(i, i + 1)
    
    # Convert to our format
    variables, edges = _networkx_to_structure(G, variable_prefix)
    
    # Generate random coefficients
    coefficients = _generate_random_coefficients(edges, coeff_range, seed=seed)
    
    # Set target variable
    if target_variable is None:
        target_variable = variables[-1]  # Last variable by default
    elif target_variable not in variables:
        raise ValueError(f"target_variable '{target_variable}' not in generated variables")
    
    # Create noise scales for all variables
    noise_scales = {var: noise_scale for var in variables}
    
    # Use existing factory function
    from .test_scms import create_simple_linear_scm
    
    scm = create_simple_linear_scm(
        variables=variables,
        edges=edges,
        coefficients=coefficients,
        noise_scales=noise_scales,
        target=target_variable
    )
    
    # Add metadata including coefficients for oracle access
    metadata = {
        'graph_type': 'erdos_renyi',
        'n_nodes': n_nodes,
        'edge_prob': edge_prob,
        'actual_edges': len(edges),
        'seed': seed,
        'coeff_range': coeff_range,
        'noise_scale': noise_scale,
        'coefficients': coefficients  # Store coefficients for oracle access
    }
    
    scm = scm.set('metadata', pyr.pmap(scm['metadata'].update(metadata)))
    
    logger.info(f"Created Erdos-Renyi SCM: {n_nodes} nodes, {len(edges)} edges, target='{target_variable}'")
    return scm


def create_scale_free_scm(
    n_nodes: int,
    alpha: float = 0.41,
    beta: float = 0.54,
    gamma: float = 0.05,
    target_variable: Optional[str] = None,
    coeff_range: Tuple[float, float] = (-2.0, 2.0),
    noise_scale: float = 1.0,
    seed: int = 42,
    variable_prefix: str = "X"
) -> pyr.PMap:
    """
    Create a linear SCM from a scale-free directed graph.
    
    Scale-free networks have degree distributions that follow a power law,
    common in many real-world networks including biological systems.
    
    Args:
        n_nodes: Number of variables/nodes
        alpha: Probability for adding a new node connected to existing nodes
        beta: Probability for adding an edge between existing nodes
        gamma: Probability for adding a new node with no connections
        target_variable: Target variable name (default: highest degree node)
        coeff_range: Range for random edge coefficients
        noise_scale: Standard deviation for noise terms
        seed: Random seed for reproducibility
        variable_prefix: Prefix for variable names
        
    Returns:
        An immutable SCM with linear mechanisms
        
    Raises:
        ValueError: If parameters are invalid
        ImportError: If NetworkX is not available
        
    Example:
        >>> scm = create_scale_free_scm(n_nodes=20, seed=42)
        >>> # Scale-free networks often have hub nodes with many connections
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError("NetworkX required for scale-free graph generation")
    
    # Validate inputs
    _validate_graph_size(n_nodes)
    
    if not jnp.isclose(alpha + beta + gamma, 1.0, atol=1e-6):
        raise ValueError(f"alpha + beta + gamma must equal 1.0, got: {alpha + beta + gamma}")
    
    if any(x < 0 for x in [alpha, beta, gamma]):
        raise ValueError("alpha, beta, gamma must be non-negative")
    
    # Generate scale-free directed graph
    G = nx.scale_free_graph(
        n_nodes,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        seed=seed
    )
    
    # Ensure it's a DAG by topological sorting
    if not nx.is_directed_acyclic_graph(G):
        # Remove edges to make it a DAG
        edges_to_remove = []
        for source, target in G.edges():
            if source >= target:  # Simple heuristic: higher index -> lower index
                edges_to_remove.append((source, target))
        G.remove_edges_from(edges_to_remove)
    
    # Convert to our format
    variables, edges = _networkx_to_structure(G, variable_prefix)
    
    # Generate random coefficients
    coefficients = _generate_random_coefficients(edges, coeff_range, seed=seed)
    
    # Set target variable (choose node with highest in-degree as interesting target)
    if target_variable is None:
        in_degrees = dict(G.in_degree())
        if in_degrees:
            target_node = max(in_degrees, key=in_degrees.get)
            target_variable = f"{variable_prefix}{target_node}"
        else:
            target_variable = variables[-1]
    elif target_variable not in variables:
        raise ValueError(f"target_variable '{target_variable}' not in generated variables")
    
    # Create noise scales for all variables
    noise_scales = {var: noise_scale for var in variables}
    
    # Create SCM
    from .test_scms import create_simple_linear_scm
    
    scm = create_simple_linear_scm(
        variables=variables,
        edges=edges,
        coefficients=coefficients,
        noise_scales=noise_scales,
        target=target_variable
    )
    
    # Add metadata
    metadata = {
        'graph_type': 'scale_free',
        'n_nodes': n_nodes,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'actual_edges': len(edges),
        'seed': seed,
        'coeff_range': coeff_range,
        'noise_scale': noise_scale
    }
    
    scm = scm.set('metadata', pyr.pmap(scm['metadata'].update(metadata)))
    
    logger.info(f"Created scale-free SCM: {n_nodes} nodes, {len(edges)} edges, target='{target_variable}'")
    return scm


def create_small_world_scm(
    n_nodes: int,
    k: int = 4,
    p: float = 0.3,
    target_variable: Optional[str] = None,
    coeff_range: Tuple[float, float] = (-2.0, 2.0),
    noise_scale: float = 1.0,
    seed: int = 42,
    variable_prefix: str = "X"
) -> pyr.PMap:
    """
    Create a linear SCM from a small-world directed graph.
    
    Small-world networks have high clustering but short path lengths,
    inspired by the Watts-Strogatz model.
    
    Args:
        n_nodes: Number of variables/nodes
        k: Each node is connected to k nearest neighbors in ring topology
        p: Probability of rewiring each edge
        target_variable: Target variable name (default: middle node)
        coeff_range: Range for random edge coefficients
        noise_scale: Standard deviation for noise terms
        seed: Random seed for reproducibility
        variable_prefix: Prefix for variable names
        
    Returns:
        An immutable SCM with linear mechanisms
        
    Raises:
        ValueError: If parameters are invalid
        ImportError: If NetworkX is not available
        
    Example:
        >>> scm = create_small_world_scm(n_nodes=30, k=6, p=0.3, seed=42)
        >>> # Small-world networks have local clustering with some long-range connections
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError("NetworkX required for small-world graph generation")
    
    # Validate inputs
    _validate_graph_size(n_nodes)
    _validate_probability(p, "p")
    
    if not isinstance(k, int) or k < 2 or k >= n_nodes:
        raise ValueError(f"k must be an integer with 2 <= k < n_nodes, got: {k}")
    
    if k % 2 != 0:
        raise ValueError(f"k must be even for small-world graph generation, got: {k}")
    
    # Generate small-world graph (undirected first)
    G_undirected = nx.watts_strogatz_graph(n_nodes, k, p, seed=seed)
    
    # Convert to directed by choosing direction randomly for each edge
    rng = onp.random.RandomState(seed)
    G = nx.DiGraph()
    G.add_nodes_from(G_undirected.nodes())
    
    for u, v in G_undirected.edges():
        # Randomly choose direction, with bias towards lower->higher index for DAG property
        if rng.random() < 0.7:  # Bias towards forward edges
            if u < v:
                G.add_edge(u, v)
            else:
                G.add_edge(v, u)
        else:
            if u > v:
                G.add_edge(u, v)
            else:
                G.add_edge(v, u)
    
    # Ensure it's a DAG
    if not nx.is_directed_acyclic_graph(G):
        edges_to_remove = []
        for source, target in G.edges():
            if source >= target:
                edges_to_remove.append((source, target))
        G.remove_edges_from(edges_to_remove)
    
    # Convert to our format
    variables, edges = _networkx_to_structure(G, variable_prefix)
    
    # Generate random coefficients
    coefficients = _generate_random_coefficients(edges, coeff_range, seed=seed)
    
    # Set target variable (middle node often interesting in small-world networks)
    if target_variable is None:
        target_variable = variables[n_nodes // 2]
    elif target_variable not in variables:
        raise ValueError(f"target_variable '{target_variable}' not in generated variables")
    
    # Create noise scales for all variables
    noise_scales = {var: noise_scale for var in variables}
    
    # Create SCM
    from .test_scms import create_simple_linear_scm
    
    scm = create_simple_linear_scm(
        variables=variables,
        edges=edges,
        coefficients=coefficients,
        noise_scales=noise_scales,
        target=target_variable
    )
    
    # Add metadata
    metadata = {
        'graph_type': 'small_world',
        'n_nodes': n_nodes,
        'k': k,
        'p': p,
        'actual_edges': len(edges),
        'seed': seed,
        'coeff_range': coeff_range,
        'noise_scale': noise_scale
    }
    
    scm = scm.set('metadata', pyr.pmap(scm['metadata'].update(metadata)))
    
    logger.info(f"Created small-world SCM: {n_nodes} nodes, {len(edges)} edges, target='{target_variable}'")
    return scm


def create_random_dag_scm(
    n_nodes: int,
    expected_edges: int,
    target_variable: Optional[str] = None,
    coeff_range: Tuple[float, float] = (-2.0, 2.0),
    noise_scale: float = 1.0,
    seed: int = 42,
    variable_prefix: str = "X"
) -> pyr.PMap:
    """
    Create a linear SCM from a random DAG with approximately the specified number of edges.
    
    This function provides more direct control over the number of edges compared to
    Erdos-Renyi graphs, which specify edge probability.
    
    Args:
        n_nodes: Number of variables/nodes
        expected_edges: Approximate number of edges to include
        target_variable: Target variable name (default: last variable)
        coeff_range: Range for random edge coefficients
        noise_scale: Standard deviation for noise terms
        seed: Random seed for reproducibility
        variable_prefix: Prefix for variable names
        
    Returns:
        An immutable SCM with linear mechanisms
        
    Raises:
        ValueError: If parameters are invalid
        
    Example:
        >>> scm = create_random_dag_scm(n_nodes=15, expected_edges=20, seed=42)
        >>> # Creates a random DAG with approximately 20 edges
    """
    # Validate inputs
    _validate_graph_size(n_nodes)
    
    max_edges = n_nodes * (n_nodes - 1) // 2  # Maximum edges in DAG
    if expected_edges < 0 or expected_edges > max_edges:
        raise ValueError(f"expected_edges must be between 0 and {max_edges}, got: {expected_edges}")
    
    # Generate random DAG by sampling edges from upper triangular matrix
    rng = onp.random.RandomState(seed)
    variables = [f"{variable_prefix}{i}" for i in range(n_nodes)]
    
    # Create all possible edges (i -> j where i < j)
    possible_edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            possible_edges.append((variables[i], variables[j]))
    
    # Sample approximately expected_edges edges
    if expected_edges >= len(possible_edges):
        selected_edges = possible_edges
    else:
        selected_edges = rng.choice(
            len(possible_edges),
            size=expected_edges,
            replace=False
        )
        selected_edges = [possible_edges[i] for i in selected_edges]
    
    # Generate random coefficients
    coefficients = _generate_random_coefficients(selected_edges, coeff_range, seed=seed)
    
    # Set target variable
    if target_variable is None:
        target_variable = variables[-1]
    elif target_variable not in variables:
        raise ValueError(f"target_variable '{target_variable}' not in generated variables")
    
    # Create noise scales for all variables
    noise_scales = {var: noise_scale for var in variables}
    
    # Create SCM
    from .test_scms import create_simple_linear_scm
    
    scm = create_simple_linear_scm(
        variables=variables,
        edges=selected_edges,
        coefficients=coefficients,
        noise_scales=noise_scales,
        target=target_variable
    )
    
    # Add metadata
    metadata = {
        'graph_type': 'random_dag',
        'n_nodes': n_nodes,
        'expected_edges': expected_edges,
        'actual_edges': len(selected_edges),
        'seed': seed,
        'coeff_range': coeff_range,
        'noise_scale': noise_scale
    }
    
    scm = scm.set('metadata', pyr.pmap(scm['metadata'].update(metadata)))
    
    logger.info(f"Created random DAG SCM: {n_nodes} nodes, {len(selected_edges)} edges, target='{target_variable}'")
    return scm


def get_benchmark_graph_summary(scm: pyr.PMap) -> Dict[str, Any]:
    """
    Get a summary of a benchmark graph's properties.
    
    Args:
        scm: The SCM to analyze
        
    Returns:
        Dictionary containing graph properties and statistics
    """
    from ..data_structures.scm import get_variables, get_edges, get_target
    
    variables = get_variables(scm)
    edges = get_edges(scm)
    target = get_target(scm)
    metadata = scm.get('metadata', {})
    
    # Compute graph statistics
    n_nodes = len(variables)
    n_edges = len(edges)
    edge_density = n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0.0
    
    # Count in-degrees
    in_degrees = {}
    for var in variables:
        in_degree = sum(1 for parent, child in edges if child == var)
        in_degrees[var] = in_degree
    
    # Count out-degrees  
    out_degrees = {}
    for var in variables:
        out_degree = sum(1 for parent, child in edges if parent == var)
        out_degrees[var] = out_degree
    
    summary = {
        'graph_type': metadata.get('graph_type', 'unknown'),
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'edge_density': edge_density,
        'target_variable': target,
        'max_in_degree': max(in_degrees.values()) if in_degrees else 0,
        'max_out_degree': max(out_degrees.values()) if out_degrees else 0,
        'avg_in_degree': sum(in_degrees.values()) / len(in_degrees) if in_degrees else 0,
        'avg_out_degree': sum(out_degrees.values()) / len(out_degrees) if out_degrees else 0,
        'root_variables': [var for var, deg in in_degrees.items() if deg == 0],
        'leaf_variables': [var for var, deg in out_degrees.items() if deg == 0],
        'generation_seed': metadata.get('seed'),
        'generation_params': {k: v for k, v in metadata.items() 
                            if k not in ['graph_type', 'seed', 'mechanism_type', 'created_by']}
    }
    
    return summary


# Export public functions
__all__ = [
    'check_networkx_availability',
    'create_erdos_renyi_scm',
    'create_scale_free_scm', 
    'create_small_world_scm',
    'create_random_dag_scm',
    'get_benchmark_graph_summary'
]