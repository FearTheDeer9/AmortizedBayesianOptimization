import networkx as nx
from typing import Any, Dict

def calculate_graph_edit_distance(g1: nx.DiGraph, g2: nx.DiGraph, **kwargs) -> float:
    """Calculate the graph edit distance between two NetworkX directed graphs.

    Graph edit distance is a measure of similarity between two graphs.
    It is defined as the minimum cost sequence of edit operations
    (node/edge substitutions, insertions, deletions)
    that transform one graph into another.

    Note: This can be computationally expensive for large graphs.

    Args:
        g1: The first directed graph.
        g2: The second directed graph.
        **kwargs: Additional keyword arguments passed to `nx.graph_edit_distance`.
                  Common arguments include `node_match`, `edge_match`,
                  `node_subst_cost`, `node_del_cost`, `node_ins_cost`,
                  `edge_subst_cost`, `edge_del_cost`, `edge_ins_cost`,
                  `timeout`.

    Returns:
        The graph edit distance between g1 and g2.

    Example:
        >>> g1 = nx.DiGraph()
        >>> g1.add_edges_from([(1, 2), (2, 3)])
        >>> g2 = nx.DiGraph()
        >>> g2.add_edges_from([(1, 2), (2, 4)]) # Different edge
        >>> calculate_graph_edit_distance(g1, g2)
        2.0 # Cost might vary based on default costs (e.g., 1 del + 1 ins)
    """
    # Set default costs if not provided, assuming unit costs
    default_kwargs = {
        'node_subst_cost': kwargs.pop('node_subst_cost', lambda n1, n2: 0 if n1 == n2 else 1),
        'node_del_cost': kwargs.pop('node_del_cost', lambda n: 1),
        'node_ins_cost': kwargs.pop('node_ins_cost', lambda n: 1),
        'edge_subst_cost': kwargs.pop('edge_subst_cost', lambda e1, e2: 0 if e1 == e2 else 1),
        'edge_del_cost': kwargs.pop('edge_del_cost', lambda e: 1),
        'edge_ins_cost': kwargs.pop('edge_ins_cost', lambda e: 1),
    }
    kwargs.update(default_kwargs)

    # Ensure graphs are directed (though the function often works for undirected too)
    if not isinstance(g1, nx.DiGraph) or not isinstance(g2, nx.DiGraph):
        # Consider warning or converting, but for now assume input is correct type
        pass

    try:
        # Consider adding a timeout as this can be slow
        ged = nx.graph_edit_distance(g1, g2, **kwargs)
        return float(ged) if ged is not None else float('inf')
    except Exception as e:
        # Log the error appropriately
        print(f"Error calculating graph edit distance: {e}")
        # Return infinity or raise a custom exception
        return float('inf')

# Potential additions:
# - Functions for other similarity metrics (e.g., Jaccard similarity on edges/nodes)
# - Utilities for graph visualization comparison
# - Functions to extract specific graph properties relevant for comparison
