"""
Random graph generators for the causal_meta library.

This module provides implementations for generating different types of random graphs,
including Erdős–Rényi model, random DAGs, and other random graph structures.
"""
from typing import Optional, Union, Dict, List, Tuple, Any
import random
import numpy as np

from causal_meta.graph import CausalGraph, DirectedGraph
from causal_meta.graph.generators.factory import GraphGenerationError


class RandomGraphGenerator:
    """
    Generator class for random graph structures.

    This class provides methods for generating various types of random graphs,
    including Erdős–Rényi graphs, random DAGs, and random causal models.
    """

    @staticmethod
    def erdos_renyi(
        num_nodes: int,
        edge_probability: float,
        directed: bool = True,
        is_causal: bool = True,
        ensure_dag: bool = False,
        seed: Optional[int] = None
    ) -> Union[CausalGraph, DirectedGraph]:
        """
        Generate a random graph using the Erdős–Rényi model (G(n,p) model).

        In this model, each possible edge is included with probability p,
        independent of all other edges.

        Args:
            num_nodes: Number of nodes in the graph
            edge_probability: Probability of edge creation between any two nodes
            directed: Whether to create a directed graph (default: True)
            is_causal: Whether to return a CausalGraph (default: True)
            ensure_dag: If True, ensures the graph is a DAG by enforcing a topological ordering
            seed: Random seed for reproducibility (default: None)

        Returns:
            A random graph instance

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        # Validate parameters
        if num_nodes < 1:
            raise GraphGenerationError("Number of nodes must be at least 1")
        if edge_probability < 0.0 or edge_probability > 1.0:
            raise GraphGenerationError(
                "Edge probability must be between 0.0 and 1.0")

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Create the appropriate graph type
        if is_causal:
            graph = CausalGraph()
        else:
            graph = DirectedGraph()

        # Add nodes to the graph
        for i in range(num_nodes):
            graph.add_node(i)

        # For DAGs, we need to ensure a topological ordering
        if ensure_dag:
            # Only consider edges from lower index to higher index
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if random.random() < edge_probability:
                        graph.add_edge(i, j)
        else:
            # Add edges based on probability
            for i in range(num_nodes):
                for j in range(num_nodes):
                    # Skip self-loops
                    if i == j:
                        continue

                    # In undirected graphs, only add one edge between each pair of nodes
                    if not directed and i > j:
                        continue

                    # Add edge with the specified probability
                    if random.random() < edge_probability:
                        graph.add_edge(i, j)

                        # For undirected graphs, add the reverse edge
                        if not directed:
                            graph.add_edge(j, i)

        return graph

    @staticmethod
    def random_dag(
        num_nodes: int,
        edge_probability: float,
        is_causal: bool = True,
        seed: Optional[int] = None
    ) -> Union[CausalGraph, DirectedGraph]:
        """
        Generate a random Directed Acyclic Graph (DAG).

        This method creates a random DAG by enforcing a topological ordering
        of nodes and only adding edges from lower-indexed nodes to higher-indexed
        nodes, which ensures acyclicity.

        Args:
            num_nodes: Number of nodes in the graph
            edge_probability: Probability of edge creation between valid node pairs
            is_causal: Whether to return a CausalGraph (default: True)
            seed: Random seed for reproducibility (default: None)

        Returns:
            A random DAG instance

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        return RandomGraphGenerator.erdos_renyi(
            num_nodes=num_nodes,
            edge_probability=edge_probability,
            directed=True,  # DAGs are always directed
            is_causal=is_causal,
            ensure_dag=True,
            seed=seed
        )

    @staticmethod
    def random_tree(
        num_nodes: int,
        is_causal: bool = True,
        balanced: bool = False,
        seed: Optional[int] = None
    ) -> Union[CausalGraph, DirectedGraph]:
        """
        Generate a random directed tree.

        A tree is a connected graph with no cycles. This method generates a
        random directed tree where each node (except the root) has exactly
        one parent.

        Args:
            num_nodes: Number of nodes in the tree
            is_causal: Whether to return a CausalGraph (default: True)
            balanced: If True, attempts to create a more balanced tree structure
            seed: Random seed for reproducibility (default: None)

        Returns:
            A random directed tree

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        if num_nodes < 1:
            raise GraphGenerationError("Number of nodes must be at least 1")

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Create the appropriate graph type
        if is_causal:
            graph = CausalGraph()
        else:
            graph = DirectedGraph()

        # Add all nodes
        for i in range(num_nodes):
            graph.add_node(i)

        if num_nodes == 1:
            return graph  # Single node tree

        # For a tree with n nodes, we need exactly n-1 edges
        if balanced:
            # Create a more balanced tree by assigning nodes to levels
            levels = []
            nodes_per_level = 1  # Start with 1 node at level 0
            remaining_nodes = num_nodes

            # Distribute nodes across levels
            while remaining_nodes > 0:
                level_size = min(nodes_per_level, remaining_nodes)
                levels.append(level_size)
                remaining_nodes -= level_size
                nodes_per_level *= 2  # Double the capacity for the next level

            # Assign node indices to levels
            node_idx = 0
            nodes_by_level = []
            for level_size in levels:
                nodes_by_level.append(
                    list(range(node_idx, node_idx + level_size)))
                node_idx += level_size

            # Connect nodes across adjacent levels
            for i in range(len(levels) - 1):
                parent_level = nodes_by_level[i]
                child_level = nodes_by_level[i + 1]

                # Assign parents to children
                for j, child in enumerate(child_level):
                    parent_idx = j // 2 if len(parent_level) > 1 else 0
                    parent = parent_level[min(
                        parent_idx, len(parent_level) - 1)]
                    graph.add_edge(parent, child)
        else:
            # Simple random tree: each node (except root) connects to a random preceding node
            for i in range(1, num_nodes):
                parent = random.randint(0, i - 1)
                graph.add_edge(parent, i)

        return graph

    @staticmethod
    def random_erdos_renyi_with_noise(
        num_nodes: int,
        base_edge_probability: float,
        noise_probability: float = 0.1,
        directed: bool = True,
        is_causal: bool = True,
        ensure_dag: bool = False,
        seed: Optional[int] = None
    ) -> Union[CausalGraph, DirectedGraph]:
        """
        Generate an Erdős–Rényi random graph with additional noise edges.

        This method first creates a base random graph using the Erdős–Rényi model,
        then adds random noise edges based on a separate noise probability.

        Args:
            num_nodes: Number of nodes in the graph
            base_edge_probability: Probability for initial edge creation
            noise_probability: Probability of adding each possible noise edge
            directed: Whether to create a directed graph (default: True)
            is_causal: Whether to return a CausalGraph (default: True)
            ensure_dag: If True, ensures the graph is a DAG by enforcing a topological ordering
            seed: Random seed for reproducibility (default: None)

        Returns:
            A random graph with noise

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        # Create the base graph
        graph = RandomGraphGenerator.erdos_renyi(
            num_nodes=num_nodes,
            edge_probability=base_edge_probability,
            directed=directed,
            is_causal=is_causal,
            ensure_dag=ensure_dag,
            seed=seed
        )

        # Set seed for noise edges (different from the base graph if seed is provided)
        if seed is not None:
            random.seed(seed + 1)
            np.random.seed(seed + 1)

        # Add noise edges
        for i in range(num_nodes):
            for j in range(num_nodes):
                # Skip self-loops
                if i == j:
                    continue

                # Skip existing edges
                if graph.has_edge(i, j):
                    continue

                # For DAGs, respect the topological ordering
                if ensure_dag and i >= j:
                    continue

                # Add noise edge with the specified probability
                if random.random() < noise_probability:
                    graph.add_edge(i, j)

                    # For undirected graphs, add the reverse edge if it doesn't exist
                    if not directed and not graph.has_edge(j, i):
                        graph.add_edge(j, i)

        return graph

    @staticmethod
    def uniform_random_dag(
        num_nodes: int,
        expected_edges: Optional[int] = None,
        edge_probability: Optional[float] = None,
        is_causal: bool = True,
        seed: Optional[int] = None
    ) -> Union[CausalGraph, DirectedGraph]:
        """
        Generate a uniformly random DAG.

        This method samples a DAG uniformly from the space of all possible DAGs
        with the given number of nodes. It first generates a random permutation
        of nodes and then adds edges with the specified probability.

        Args:
            num_nodes: Number of nodes in the graph
            expected_edges: Expected number of edges in the graph (alternative to edge_probability)
            edge_probability: Probability of edge creation (default: calculated from expected_edges)
            is_causal: Whether to return a CausalGraph (default: True)
            seed: Random seed for reproducibility (default: None)

        Returns:
            A uniformly random DAG

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        if num_nodes < 1:
            raise GraphGenerationError("Number of nodes must be at least 1")

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Calculate edge probability if expected_edges is provided
        if edge_probability is None:
            if expected_edges is None:
                # Default to an average of 2 edges per node
                expected_edges = min(
                    2 * num_nodes, num_nodes * (num_nodes - 1) // 2)

            # Calculate probability based on expected number of edges
            max_possible_edges = num_nodes * (num_nodes - 1) // 2
            edge_probability = min(1.0, expected_edges / max_possible_edges)

        # Create the appropriate graph type
        if is_causal:
            graph = CausalGraph()
        else:
            graph = DirectedGraph()

        # Add all nodes
        for i in range(num_nodes):
            graph.add_node(i)

        # Generate a random permutation of nodes to ensure a valid topological order
        node_permutation = list(range(num_nodes))
        random.shuffle(node_permutation)

        # Add edges based on the permutation and probability
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # Only add edges from earlier to later nodes in the permutation
                u = node_permutation[i]
                v = node_permutation[j]

                if random.random() < edge_probability:
                    graph.add_edge(u, v)

        return graph

    @staticmethod
    def random_bipartite(
        n1: int,
        n2: int,
        edge_probability: float = 0.5,
        directed: bool = True,
        is_causal: bool = True,
        seed: Optional[int] = None
    ) -> Union[CausalGraph, DirectedGraph]:
        """
        Generate a random bipartite graph.

        A bipartite graph has two sets of nodes, with edges only between nodes
        from different sets (not within the same set).

        Args:
            n1: Number of nodes in the first set
            n2: Number of nodes in the second set
            edge_probability: Probability of edge creation between nodes in different sets
            directed: Whether to create a directed graph (default: True)
            is_causal: Whether to return a CausalGraph (default: True)
            seed: Random seed for reproducibility (default: None)

        Returns:
            A random bipartite graph

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        if n1 < 1 or n2 < 1:
            raise GraphGenerationError(
                "Number of nodes in each set must be at least 1")
        if edge_probability < 0.0 or edge_probability > 1.0:
            raise GraphGenerationError(
                "Edge probability must be between 0.0 and 1.0")

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Create the appropriate graph type
        if is_causal:
            graph = CausalGraph()
        else:
            graph = DirectedGraph()

        # Add nodes to the graph
        # First set: 0 to n1-1
        # Second set: n1 to n1+n2-1
        for i in range(n1 + n2):
            graph.add_node(i)

        # Add edges only between nodes from different sets
        for i in range(n1):
            for j in range(n1, n1 + n2):
                if random.random() < edge_probability:
                    if directed:
                        # Add directed edge
                        graph.add_edge(i, j)
                    else:
                        # For undirected, randomly choose direction but ensure edge exists both ways
                        if random.random() < 0.5:
                            graph.add_edge(i, j)
                            graph.add_edge(j, i)
                        else:
                            graph.add_edge(j, i)
                            graph.add_edge(i, j)

        return graph
