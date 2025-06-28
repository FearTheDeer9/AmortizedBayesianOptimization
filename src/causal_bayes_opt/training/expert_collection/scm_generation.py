#!/usr/bin/env python3
"""
SCM Generation for Expert Demonstration Collection

Generates diverse SCM problems for demonstration collection with various
graph structures and complexities.
"""

from typing import List, Tuple, FrozenSet, Optional

import numpy as onp
import jax
import jax.random as random
import pyrsistent as pyr

from causal_bayes_opt.data_structures.scm import create_scm_with_descriptors
from causal_bayes_opt.mechanisms.linear import create_linear_mechanism, create_root_mechanism
from causal_bayes_opt.mechanisms.descriptors import RootMechanismDescriptor, LinearMechanismDescriptor


def generate_scm_problems(
    n_problems: int = 100,
    node_sizes: List[int] = [3, 5, 8, 10, 15, 20],
    graph_types: List[str] = ["chain", "star", "fork", "collider"],
    key: Optional[jax.Array] = None
) -> List[Tuple[pyr.PMap, str]]:
    """
    Generate diverse SCM problems for demonstration collection.
    
    Args:
        n_problems: Number of problems to generate
        node_sizes: List of graph sizes to sample from
        graph_types: List of graph structures to sample from
        key: Random key for reproducible generation
        
    Returns:
        List of (scm, graph_type) tuples
    """
    if key is None:
        key = random.PRNGKey(42)
    
    problems = []
    
    for i in range(n_problems):
        key, subkey = random.split(key)
        
        # Sample problem characteristics
        n_nodes = onp.random.choice(node_sizes)
        graph_type = onp.random.choice(graph_types)
        
        # Generate SCM
        scm = generate_scm(n_nodes, graph_type, subkey)
        problems.append((scm, graph_type))
    
    return problems


def generate_scm(n_nodes: int, graph_type: str, key: jax.Array) -> pyr.PMap:
    """Generate a single SCM of specified type and size."""
    variables = [f"X{i}" for i in range(n_nodes)]
    target = variables[-1]  # Last variable is target
    
    # Generate edges based on graph type
    if graph_type == "chain":
        edges = frozenset([(variables[i], variables[i+1]) for i in range(n_nodes-1)])
    elif graph_type == "star":
        # First 3 variables point to target
        edges = frozenset([(variables[i], target) for i in range(min(3, n_nodes-1))])
    elif graph_type == "fork":
        # Root variable affects multiple children
        edges = frozenset([(variables[0], variables[i]) for i in range(1, min(4, n_nodes))])
    elif graph_type == "collider":
        # Multiple variables affect target
        parents = variables[:-1] if n_nodes <= 4 else variables[:3]
        edges = frozenset([(parent, target) for parent in parents])
    else:
        # Random sparse graph
        edges = generate_random_edges(variables, key)
    
    # Generate linear mechanisms with descriptors for serialization
    mechanism_descriptors = {}
    key, *subkeys = random.split(key, len(variables) + 1)
    
    for i, var in enumerate(variables):
        # Get parents
        parents = [p for p, c in edges if c == var]
        
        if not parents:
            # Root variable
            mean_val = random.normal(subkeys[i]) * 0.5
            mechanism_descriptors[var] = RootMechanismDescriptor(
                mean=float(mean_val), 
                noise_scale=0.1
            )
        else:
            # Variable with parents
            coefficients = {}
            for parent in parents:
                coeff = random.normal(subkeys[i]) * 0.8 + 0.5  # Avoid zero coefficients
                coefficients[parent] = float(coeff)
            
            intercept = random.normal(subkeys[i]) * 0.2
            mechanism_descriptors[var] = LinearMechanismDescriptor(
                parents=parents,
                coefficients=coefficients,
                intercept=float(intercept),
                noise_scale=0.1
            )
    
    return create_scm_with_descriptors(
        variables=frozenset(variables),
        edges=edges,
        mechanism_descriptors=mechanism_descriptors,
        target=target
    )


def generate_random_edges(variables: List[str], key: jax.Array) -> FrozenSet[Tuple[str, str]]:
    """Generate random sparse edges with topological ordering."""
    n_vars = len(variables)
    target_density = 0.3  # Sparse graphs
    max_edges = int(target_density * n_vars * (n_vars - 1) / 2)
    
    edges = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if random.uniform(key) < target_density and len(edges) < max_edges:
                edges.append((variables[i], variables[j]))
                key, _ = random.split(key)
    
    return frozenset(edges)