import networkx as nx
import numpy as np
from typing import Optional, Tuple, List


def generate_erdos_renyi(n_nodes: int, edge_prob: float, seed: Optional[int] = None) -> nx.DiGraph:
    """Generate Erdos-Renyi random graph with n_nodes and probability edge_prob.

    Args:
        n_nodes: Number of nodes in the graph
        edge_prob: Probability of an edge between any two nodes
        seed: Random seed for reproducibility

    Returns:
        A directed acyclic graph (DAG) with the specified properties
    """
    rng = np.random.default_rng(seed)

    # Create a random node ordering to ensure acyclicity
    node_ordering = rng.permutation(n_nodes)

    # Create empty directed graph
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))

    # Add edges based on probability, respecting the node ordering
    edges = []
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if rng.random() < edge_prob:
                source = node_ordering[i]
                target = node_ordering[j]
                edges.append((source, target))

    G.add_edges_from(edges)

    return G


def generate_scale_free(n_nodes: int, attachment_param: float, seed: Optional[int] = None) -> nx.DiGraph:
    """Generate scale-free network with preferential attachment.

    Args:
        n_nodes: Number of nodes in the graph
        attachment_param: Parameter controlling the preferential attachment
        seed: Random seed for reproducibility

    Returns:
        A directed acyclic graph (DAG) with scale-free properties
    """
    rng = np.random.default_rng(seed)

    # Start with a small initial graph
    G = nx.DiGraph()
    G.add_nodes_from(range(2))
    G.add_edge(0, 1)

    # Add remaining nodes
    for new_node in range(2, n_nodes):
        # Calculate attachment probabilities
        degrees = np.array([G.out_degree(n) for n in G.nodes()])
        probs = (degrees + attachment_param) / \
            (degrees.sum() + attachment_param * len(G))

        # Select parent nodes
        num_parents = rng.integers(1, min(4, len(G) + 1))
        parents = rng.choice(
            list(G.nodes()), size=num_parents, p=probs, replace=False)

        # Add edges from parents to new node
        for parent in parents:
            G.add_edge(parent, new_node)

    return G


def generate_small_world(n_nodes: int, k_neighbors: int, rewire_prob: float,
                         seed: Optional[int] = None) -> nx.DiGraph:
    """Generate small-world network with rewiring probability.

    Args:
        n_nodes: Number of nodes in the graph
        k_neighbors: Number of nearest neighbors to connect
        rewire_prob: Probability of rewiring each edge
        seed: Random seed for reproducibility

    Returns:
        A directed acyclic graph (DAG) with small-world properties
    """
    rng = np.random.default_rng(seed)

    # Start with a ring lattice
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))

    # Connect each node to its k nearest neighbors
    for i in range(n_nodes):
        for j in range(1, k_neighbors + 1):
            # Connect to neighbors in both directions
            G.add_edge(i, (i + j) % n_nodes)
            G.add_edge(i, (i - j) % n_nodes)

    # Rewire edges with probability p
    edges = list(G.edges())
    for u, v in edges:
        if rng.random() < rewire_prob:
            # Remove the edge
            G.remove_edge(u, v)
            # Add new edge to random node
            new_v = rng.choice(list(set(range(n_nodes)) - {u, v}))
            G.add_edge(u, new_v)

    # Ensure it's a DAG by removing cycles
    while not nx.is_directed_acyclic_graph(G):
        cycles = list(nx.simple_cycles(G))
        if cycles:
            # Remove a random edge from the first cycle found
            cycle = cycles[0]
            G.remove_edge(cycle[0], cycle[1])

    return G


def verify_graph_properties(G: nx.DiGraph) -> Tuple[bool, List[str]]:
    """Verify that a graph meets our requirements.

    Args:
        G: The graph to verify

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check if it's a DAG
    if not nx.is_directed_acyclic_graph(G):
        issues.append("Graph contains cycles")

    # Check if it's connected
    if not nx.is_weakly_connected(G):
        issues.append("Graph is not weakly connected")

    # Check for self-loops
    if any(G.has_edge(n, n) for n in G.nodes()):
        issues.append("Graph contains self-loops")

    return len(issues) == 0, issues
