"""
SCM Structure Encoding Utilities

This module provides functions to encode causal structure information into meaningful
embeddings and prior probabilities, forming the foundation of the bootstrap surrogate.

Key functions:
- encode_causal_structure(): Convert SCM graph properties to 128D embeddings
- compute_structural_parent_probabilities(): Prior beliefs from SCM structure
- Helper functions for graph analysis and distance computation

This ensures variables have different, meaningful features based on their causal roles.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import jax
import jax.numpy as jnp
import pyrsistent as pyr
from ..data_structures.scm import get_variables, get_target, get_edges


def encode_causal_structure(
    variables: List[str], 
    edges: List[Tuple[str, str]], 
    target: str,
    dim: int = 128
) -> jnp.ndarray:
    """
    Encode structural properties of SCM variables into embedding space.
    
    This function converts graph properties (degrees, distances, etc.) into
    dense embeddings that provide meaningful differentiation between variables
    based on their causal roles.
    
    Args:
        variables: List of variable names in the SCM
        edges: List of directed edges as (parent, child) tuples
        target: Target variable name
        dim: Embedding dimension (default 128)
        
    Returns:
        Embeddings array [n_vars, dim] with structural features
        
    Example:
        >>> variables = ['X', 'Y', 'Z']
        >>> edges = [('X', 'Y'), ('Y', 'Z')]
        >>> embeddings = encode_causal_structure(variables, edges, 'Z')
        >>> embeddings.shape
        (3, 128)
    """
    n_vars = len(variables)
    var_to_idx = {var: i for i, var in enumerate(variables)}
    target_idx = var_to_idx[target] if target in var_to_idx else -1
    
    # Initialize embeddings
    embeddings = jnp.zeros((n_vars, dim))
    
    # Build adjacency structures
    parents_of = {var: set() for var in variables}
    children_of = {var: set() for var in variables}
    
    for parent, child in edges:
        if parent in parents_of and child in children_of:
            parents_of[child].add(parent)
            children_of[parent].add(child)
    
    # Compute graph distances to target
    distances_to_target = _compute_distances_to_target(variables, edges, target)
    
    # Encode features for each variable
    for i, var in enumerate(variables):
        # Basic graph features (channels 0-9)
        embeddings = embeddings.at[i, 0].set(len(parents_of[var]))        # In-degree
        embeddings = embeddings.at[i, 1].set(len(children_of[var]))       # Out-degree
        embeddings = embeddings.at[i, 2].set(float(len(parents_of[var]) == 0))  # Is root
        embeddings = embeddings.at[i, 3].set(float(len(children_of[var]) == 0)) # Is leaf
        embeddings = embeddings.at[i, 4].set(float(var == target))        # Is target
        embeddings = embeddings.at[i, 5].set(distances_to_target.get(var, float('inf')))  # Distance to target
        
        # Normalized features (channels 6-9) 
        total_degree = len(parents_of[var]) + len(children_of[var])
        embeddings = embeddings.at[i, 6].set(total_degree)                # Total degree
        embeddings = embeddings.at[i, 7].set(total_degree / max(1, n_vars - 1))  # Normalized degree
        
        # Structural importance (channels 8-9)
        embeddings = embeddings.at[i, 8].set(_compute_betweenness_centrality(var, variables, edges))
        embeddings = embeddings.at[i, 9].set(_compute_closeness_centrality(var, variables, edges))
        
        # Path-based features (channels 10-19)
        paths_through_var = _count_paths_through_variable(var, variables, edges, target)
        embeddings = embeddings.at[i, 10].set(paths_through_var)
        
        # Reachability features (channels 11-15)
        reachable_from_var = len(_get_reachable_variables(var, children_of))
        reachable_to_var = len(_get_reachable_variables(var, parents_of, reverse=True))
        embeddings = embeddings.at[i, 11].set(reachable_from_var)
        embeddings = embeddings.at[i, 12].set(reachable_to_var)
        embeddings = embeddings.at[i, 13].set(reachable_from_var / max(1, n_vars - 1))  # Normalized
        embeddings = embeddings.at[i, 14].set(reachable_to_var / max(1, n_vars - 1))    # Normalized
        
        # Ancestor/descendant relationships with target (channels 15-19)
        is_ancestor_of_target = _is_ancestor_of(var, target, children_of)
        is_descendant_of_target = _is_ancestor_of(target, var, children_of)
        embeddings = embeddings.at[i, 15].set(float(is_ancestor_of_target))
        embeddings = embeddings.at[i, 16].set(float(is_descendant_of_target))
        
        # Shortest path features (channels 17-19)
        shortest_path_length = distances_to_target.get(var, float('inf'))
        if shortest_path_length != float('inf'):
            embeddings = embeddings.at[i, 17].set(1.0 / (shortest_path_length + 1))  # Inverse distance
            embeddings = embeddings.at[i, 18].set(jnp.exp(-shortest_path_length))     # Exponential decay
        
        # Local neighborhood features (channels 20-29)
        local_clustering = _compute_local_clustering(var, parents_of, children_of)
        embeddings = embeddings.at[i, 20].set(local_clustering)
        
        # Second-order features (channels 21-29)
        second_order_neighbors = _count_second_order_neighbors(var, parents_of, children_of)
        embeddings = embeddings.at[i, 21].set(second_order_neighbors)
        
        # Remaining channels (30-127): Reserved for future structural features
        # For now, fill with positional encoding to ensure no two variables are identical
        for j in range(30, min(dim, 50)):
            # Simple positional encoding based on variable index
            embeddings = embeddings.at[i, j].set(jnp.sin(i * jnp.pi / n_vars + j * 0.1))
    
    return embeddings


def compute_structural_parent_probabilities(
    variables: List[str], 
    edges: List[Tuple[str, str]], 
    target: str
) -> jnp.ndarray:
    """
    Compute reasonable parent probabilities from causal structure.
    
    Variables closer to the target in the causal graph receive higher probabilities,
    reflecting their likely causal relevance for intervention selection.
    
    Args:
        variables: List of variable names
        edges: List of directed edges as (parent, child) tuples  
        target: Target variable name
        
    Returns:
        Array [n_vars] of parent probabilities (sums to 1.0, target excluded)
        
    Example:
        >>> variables = ['X', 'Y', 'Z']
        >>> edges = [('X', 'Y'), ('Y', 'Z')]
        >>> probs = compute_structural_parent_probabilities(variables, edges, 'Z')
        >>> probs  # Y should have higher probability than X (closer to Z)
        Array([0.33, 0.67, 0.0])  # X, Y, Z (target excluded)
    """
    n_vars = len(variables)
    target_idx = variables.index(target) if target in variables else -1
    
    # Compute distances to target
    distances = _compute_distances_to_target(variables, edges, target)
    
    # Initialize probabilities
    probs = jnp.zeros(n_vars)
    
    for i, var in enumerate(variables):
        if var != target:  # Target cannot be its own parent
            distance = distances.get(var, float('inf'))
            
            if distance == float('inf'):
                # Not connected to target - low but non-zero probability
                prob = 0.1
            else:
                # Inverse relationship: closer = higher probability
                prob = 1.0 / (distance + 1.0)
            
            probs = probs.at[i].set(prob)
    
    # Set target probability to 0
    if target_idx >= 0:
        probs = probs.at[target_idx].set(0.0)
    
    # Normalize to valid probability distribution
    prob_sum = jnp.sum(probs)
    if prob_sum > 0:
        probs = probs / prob_sum
    else:
        # Fallback: uniform over non-target variables
        non_target_count = n_vars - (1 if target_idx >= 0 else 0)
        if non_target_count > 0:
            uniform_prob = 1.0 / non_target_count
            for i, var in enumerate(variables):
                if var != target:
                    probs = probs.at[i].set(uniform_prob)
    
    return probs


# Helper functions for graph analysis

def _compute_distances_to_target(
    variables: List[str], 
    edges: List[Tuple[str, str]], 
    target: str
) -> Dict[str, float]:
    """Compute shortest path distances to target variable."""
    # Build adjacency list (both directions for undirected distance)
    adj = {var: set() for var in variables}
    for parent, child in edges:
        if parent in adj and child in adj:
            adj[parent].add(child)
            adj[child].add(parent)  # Treat as undirected for distance computation
    
    # BFS from target
    distances = {target: 0.0}
    queue = [(target, 0.0)]
    visited = {target}
    
    while queue:
        current, dist = queue.pop(0)
        
        for neighbor in adj[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = dist + 1.0
                queue.append((neighbor, dist + 1.0))
    
    return distances


def _compute_betweenness_centrality(
    var: str, 
    variables: List[str], 
    edges: List[Tuple[str, str]]
) -> float:
    """Compute simplified betweenness centrality for a variable."""
    # Simplified implementation - count how many shortest paths pass through this variable
    # For efficiency, we approximate this with the number of edges incident to the variable
    incident_edges = 0
    for parent, child in edges:
        if parent == var or child == var:
            incident_edges += 1
    
    # Normalize by maximum possible edges
    max_edges = 2 * (len(variables) - 1)  # Maximum edges for a variable
    return incident_edges / max(1, max_edges)


def _compute_closeness_centrality(
    var: str, 
    variables: List[str], 
    edges: List[Tuple[str, str]]
) -> float:
    """Compute simplified closeness centrality for a variable."""
    distances = _compute_distances_to_target(variables, edges, var)
    
    # Sum of distances to all other variables
    total_distance = sum(dist for v, dist in distances.items() if v != var and dist != float('inf'))
    
    if total_distance > 0:
        return 1.0 / total_distance
    else:
        return 0.0


def _count_paths_through_variable(
    var: str, 
    variables: List[str], 
    edges: List[Tuple[str, str]], 
    target: str
) -> float:
    """Count approximate number of paths from any variable to target that pass through var."""
    if var == target:
        return 0.0
    
    # Simplified: count variables that can reach var * variables var can reach
    parents_of = {v: set() for v in variables}
    children_of = {v: set() for v in variables}
    
    for parent, child in edges:
        if parent in parents_of and child in children_of:
            parents_of[child].add(parent)
            children_of[parent].add(child)
    
    can_reach_var = len(_get_reachable_variables(var, parents_of, reverse=True))
    var_can_reach = len(_get_reachable_variables(var, children_of))
    
    return float(can_reach_var * var_can_reach)


def _get_reachable_variables(
    start_var: str, 
    adjacency: Dict[str, Set[str]], 
    reverse: bool = False
) -> Set[str]:
    """Get all variables reachable from start_var using BFS."""
    reachable = set()
    queue = [start_var]
    visited = {start_var}
    
    while queue:
        current = queue.pop(0)
        
        neighbors = adjacency.get(current, set())
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                reachable.add(neighbor)
                queue.append(neighbor)
    
    return reachable


def _is_ancestor_of(
    ancestor: str, 
    descendant: str, 
    children_of: Dict[str, Set[str]]
) -> bool:
    """Check if ancestor is an ancestor of descendant in the DAG."""
    if ancestor == descendant:
        return False
    
    # BFS from ancestor to see if we can reach descendant
    queue = [ancestor]
    visited = {ancestor}
    
    while queue:
        current = queue.pop(0)
        
        for child in children_of.get(current, set()):
            if child == descendant:
                return True
            if child not in visited:
                visited.add(child)
                queue.append(child)
    
    return False


def _compute_local_clustering(
    var: str, 
    parents_of: Dict[str, Set[str]], 
    children_of: Dict[str, Set[str]]
) -> float:
    """Compute local clustering coefficient for a variable."""
    neighbors = parents_of.get(var, set()) | children_of.get(var, set())
    
    if len(neighbors) <= 1:
        return 0.0
    
    # Count edges between neighbors
    neighbor_edges = 0
    neighbors_list = list(neighbors)
    
    for i, n1 in enumerate(neighbors_list):
        for j in range(i + 1, len(neighbors_list)):
            n2 = neighbors_list[j]
            # Check if n1 and n2 are connected
            if (n2 in children_of.get(n1, set()) or 
                n1 in children_of.get(n2, set())):
                neighbor_edges += 1
    
    # Maximum possible edges between neighbors
    max_edges = len(neighbors) * (len(neighbors) - 1) / 2
    
    return neighbor_edges / max(1, max_edges)


def _count_second_order_neighbors(
    var: str, 
    parents_of: Dict[str, Set[str]], 
    children_of: Dict[str, Set[str]]
) -> float:
    """Count second-order neighbors (neighbors of neighbors)."""
    first_order = parents_of.get(var, set()) | children_of.get(var, set())
    second_order = set()
    
    for neighbor in first_order:
        second_order.update(parents_of.get(neighbor, set()))
        second_order.update(children_of.get(neighbor, set()))
    
    # Remove first-order neighbors and the variable itself
    second_order -= first_order
    second_order.discard(var)
    
    return float(len(second_order))