"""
Random Directed Acyclic Graph (DAG) generator for causal graph structure learning.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple, Dict, Any

from causal_meta.graph import CausalGraph, DirectedGraph
from causal_meta.graph.generators.factory import GraphFactory


class RandomDAGGenerator:
    """
    Generator for random Directed Acyclic Graphs (DAGs).

    This class provides a simplified interface for generating random DAGs,
    leveraging the existing GraphFactory implementation.
    """

    @staticmethod
    def generate_random_dag(
        num_nodes: int,
        edge_probability: float,
        as_adjacency_matrix: bool = True,
        seed: Optional[int] = None,
    ) -> Union[np.ndarray, CausalGraph]:
        """
        Generate a random Directed Acyclic Graph (DAG).

        Args:
            num_nodes (int): Number of nodes in the DAG
            edge_probability (float): Probability of edge creation between valid node pairs
            as_adjacency_matrix (bool): If True, returns the adjacency matrix representation
                                        If False, returns a CausalGraph object
            seed (Optional[int]): Random seed for reproducibility

        Returns:
            Union[np.ndarray, CausalGraph]: A random DAG as either an adjacency matrix or CausalGraph
        """
        # Generate the random DAG using GraphFactory
        graph = GraphFactory.create_random_dag(
            num_nodes=num_nodes,
            edge_probability=edge_probability,
            is_causal=True,
            seed=seed,
        )

        # Return the requested representation
        if as_adjacency_matrix:
            return np.array(graph.get_adjacency_matrix())
        else:
            return graph

    @staticmethod
    def validate_acyclicity(adj_matrix: np.ndarray) -> bool:
        """
        Check if a given adjacency matrix represents a Directed Acyclic Graph (DAG).

        Args:
            adj_matrix (np.ndarray): Adjacency matrix representation of a graph

        Returns:
            bool: True if the graph is acyclic, False otherwise
        """
        # Convert to NetworkX graph for validation
        nx_graph = nx.DiGraph()
        n = adj_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                if adj_matrix[i, j] > 0:
                    nx_graph.add_edge(i, j)

        # Check if the graph is a DAG
        return nx.is_directed_acyclic_graph(nx_graph)

    @staticmethod
    def visualize_dag(
        adj_matrix: np.ndarray,
        node_names: Optional[Dict[int, str]] = None,
        figsize: Tuple[int, int] = (8, 6),
        title: str = "Random Directed Acyclic Graph",
    ) -> None:
        """
        Visualize a Directed Acyclic Graph (DAG) from its adjacency matrix.

        Args:
            adj_matrix (np.ndarray): Adjacency matrix representation of a DAG
            node_names (Optional[Dict[int, str]]): Mapping of node indices to names
            figsize (Tuple[int, int]): Figure size
            title (str): Plot title
        """
        # Convert to NetworkX graph for visualization
        nx_graph = nx.DiGraph()
        n = adj_matrix.shape[0]
        
        # Add nodes
        if node_names is None:
            node_names = {i: str(i) for i in range(n)}
        for i in range(n):
            nx_graph.add_node(i, label=node_names.get(i, str(i)))
        
        # Add edges
        for i in range(n):
            for j in range(n):
                if adj_matrix[i, j] > 0:
                    nx_graph.add_edge(i, j)

        # Ensure the graph is a DAG
        if not nx.is_directed_acyclic_graph(nx_graph):
            raise ValueError("The adjacency matrix does not represent a DAG")

        # Draw the graph
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(nx_graph, seed=42)  # For reproducible layout
        
        # Draw nodes
        nx.draw_networkx_nodes(nx_graph, pos, node_size=500, node_color="lightblue")
        
        # Draw edges
        nx.draw_networkx_edges(nx_graph, pos, arrowsize=20, width=1.5)
        
        # Draw node labels
        node_labels = {node: nx_graph.nodes[node].get("label", str(node)) for node in nx_graph.nodes}
        nx.draw_networkx_labels(nx_graph, pos, labels=node_labels)
        
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show() 