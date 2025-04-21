"""
Graph generator factory for the causal_meta library.

This module provides a factory pattern implementation for generating different types
of graph structures, including random graphs, scale-free networks, and predefined structures.
"""
from typing import Dict, Any, Optional, Union, List, Tuple
import random
from abc import ABC, abstractmethod

from causal_meta.graph import CausalGraph, DirectedGraph


class GraphGenerationError(Exception):
    """Exception raised for errors during graph generation."""
    pass


class GraphFactory:
    """
    A factory class for generating different types of graph structures.

    This class provides methods to create various graph types including:
    - Random graphs (Erdős–Rényi model)
    - Scale-free networks (Barabási–Albert model)
    - Predefined causal structures (chains, forks, colliders, etc.)
    """

    @classmethod
    def create_graph(cls,
                     graph_type: str,
                     **kwargs) -> Union[CausalGraph, DirectedGraph]:
        """
        Factory method to create a graph based on the specified type and parameters.

        Args:
            graph_type: The type of graph to create. Options include:
                        'random', 'scale_free', 'predefined'
            **kwargs: Parameters specific to the graph type

        Returns:
            A graph instance of the specified type

        Raises:
            GraphGenerationError: If the graph_type is invalid or if required parameters are missing
        """
        # Route to the appropriate creation method based on the graph type
        if graph_type.lower() == 'random':
            return cls.create_random_graph(**kwargs)
        elif graph_type.lower() == 'scale_free':
            return cls.create_scale_free_graph(**kwargs)
        elif graph_type.lower() == 'predefined':
            return cls.create_predefined_graph(**kwargs)
        else:
            raise GraphGenerationError(f"Unknown graph type: {graph_type}")

    @classmethod
    def create_random_graph(cls,
                            num_nodes: int,
                            edge_probability: float,
                            directed: bool = True,
                            is_causal: bool = True,
                            seed: Optional[int] = None,
                            **kwargs) -> Union[CausalGraph, DirectedGraph]:
        """
        Create a random graph using the Erdős–Rényi model.

        Args:
            num_nodes: Number of nodes in the graph
            edge_probability: Probability of creating an edge between any two nodes
            directed: Whether to create a directed graph (default: True)
            is_causal: Whether to return a CausalGraph (default: True)
            seed: Random seed for reproducibility (default: None)
            **kwargs: Additional parameters

        Returns:
            A random graph instance

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        # Validate parameters
        cls._validate_parameters({
            'num_nodes': (num_nodes, (1, float('inf'))),
            'edge_probability': (edge_probability, (0.0, 1.0))
        })

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)

        # Create the appropriate graph type
        if is_causal:
            graph = CausalGraph()
        else:
            graph = DirectedGraph()

        # Add nodes to the graph
        for i in range(num_nodes):
            graph.add_node(i)

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

    @classmethod
    def create_scale_free_graph(cls,
                                num_nodes: int,
                                m: int = 2,
                                directed: bool = True,
                                is_causal: bool = True,
                                seed: Optional[int] = None,
                                **kwargs) -> Union[CausalGraph, DirectedGraph]:
        """
        Create a scale-free graph using the Barabási–Albert model.

        Args:
            num_nodes: Number of nodes in the graph
            m: Number of edges to attach from a new node to existing nodes
            directed: Whether to create a directed graph (default: True)
            is_causal: Whether to return a CausalGraph (default: True)
            seed: Random seed for reproducibility (default: None)
            **kwargs: Additional parameters

        Returns:
            A scale-free graph instance

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        # Validate parameters
        cls._validate_parameters({
            'num_nodes': (num_nodes, (m + 1, float('inf'))),
            'm': (m, (1, num_nodes - 1))
        })

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)

        # Create the appropriate graph type
        if is_causal:
            graph = CausalGraph()
        else:
            graph = DirectedGraph()

        # Add initial nodes to the graph (fully connected)
        for i in range(m + 1):
            graph.add_node(i)

        # Connect the initial nodes (create a complete graph of m+1 nodes)
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                graph.add_edge(i, j)
                if not directed:
                    graph.add_edge(j, i)

        # Add remaining nodes using preferential attachment
        for i in range(m + 1, num_nodes):
            graph.add_node(i)

            # Calculate node degrees and create a probability distribution
            node_degrees = {}
            total_degree = 0

            for node in range(i):
                # For directed graphs, use out-degree; for undirected, use total degree
                if directed:
                    degree = len(graph.get_successors(node))
                else:
                    degree = len(graph.get_successors(node)) + \
                        len(graph.get_predecessors(node))

                node_degrees[node] = degree
                total_degree += degree

            # Select m distinct nodes based on their degree probability
            targets = []
            while len(targets) < m:
                # Choose a random number between 0 and the total degree
                r = random.uniform(0, total_degree)

                # Find the node corresponding to this position
                cumulative = 0
                for node, degree in node_degrees.items():
                    if node in targets:
                        continue

                    cumulative += degree
                    if cumulative >= r:
                        targets.append(node)
                        break

                # If we didn't find a node (due to rounding errors), pick a random one
                if len(targets) == 0 or (len(targets) < m and targets[-1] not in targets):
                    available_nodes = [node for node in range(
                        i) if node not in targets]
                    if available_nodes:
                        targets.append(random.choice(available_nodes))

            # Add edges from the new node to the selected targets
            for target in targets:
                graph.add_edge(i, target)
                if not directed:
                    graph.add_edge(target, i)

        return graph

    @classmethod
    def create_predefined_graph(cls,
                                structure_type: str,
                                num_nodes: Optional[int] = None,
                                noise_probability: float = 0.0,
                                is_causal: bool = True,
                                seed: Optional[int] = None,
                                **kwargs) -> Union[CausalGraph, DirectedGraph]:
        """
        Create a predefined graph structure.

        Args:
            structure_type: Type of structure to create (e.g., 'chain', 'fork', 'collider', 
                           'complete', 'tree', 'bipartite')
            num_nodes: Number of nodes (for scalable structures like chains, trees)
            noise_probability: Probability of adding random noise edges (default: 0.0)
            is_causal: Whether to return a CausalGraph (default: True)
            seed: Random seed for reproducibility (default: None)
            **kwargs: Additional parameters specific to the structure type
                      - For 'bipartite': n1, n2 (sizes of the two node sets)
                      - For 'tree': branching_factor (default: 2)
                      - For 'from_matrix': adjacency_matrix (2D list/array)
                      - For 'from_edges': edge_list (list of tuples)

        Returns:
            A graph with the specified structure

        Raises:
            GraphGenerationError: If structure_type is invalid or parameters are invalid
        """
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)

        # Create the appropriate graph type
        if is_causal:
            graph = CausalGraph()
        else:
            graph = DirectedGraph()

        # Create the specified structure
        if structure_type.lower() == 'chain':
            cls._create_chain_structure(graph, num_nodes, **kwargs)
        elif structure_type.lower() == 'fork':
            cls._create_fork_structure(graph, num_nodes, **kwargs)
        elif structure_type.lower() == 'collider':
            cls._create_collider_structure(graph, num_nodes, **kwargs)
        elif structure_type.lower() == 'complete':
            cls._create_complete_structure(graph, num_nodes, **kwargs)
        elif structure_type.lower() == 'tree':
            cls._create_tree_structure(graph, num_nodes, **kwargs)
        elif structure_type.lower() == 'bipartite':
            cls._create_bipartite_structure(graph, **kwargs)
        elif structure_type.lower() == 'from_matrix':
            cls._create_from_adjacency_matrix(graph, **kwargs)
        elif structure_type.lower() == 'from_edges':
            edge_list = kwargs.pop('edge_list', None)
            cls._create_from_edge_list(
                graph, edge_list=edge_list, num_nodes=num_nodes, **kwargs)
        else:
            raise GraphGenerationError(
                f"Unknown structure type: {structure_type}")

        # Add noise edges if specified
        if noise_probability > 0:
            cls._add_noise_edges(graph, noise_probability)

        return graph

    @staticmethod
    def _create_chain_structure(graph: Union[CausalGraph, DirectedGraph],
                                num_nodes: int,
                                **kwargs) -> None:
        """
        Create a chain structure (A -> B -> C -> ...)

        Args:
            graph: Graph to add the structure to
            num_nodes: Number of nodes in the chain
            **kwargs: Additional parameters

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        if num_nodes is None or num_nodes < 2:
            raise GraphGenerationError("Chain requires at least 2 nodes")

        # Add nodes
        for i in range(num_nodes):
            graph.add_node(i)

        # Add edges to form a chain
        for i in range(num_nodes - 1):
            graph.add_edge(i, i + 1)

    @staticmethod
    def _create_fork_structure(graph: Union[CausalGraph, DirectedGraph],
                               num_nodes: int,
                               **kwargs) -> None:
        """
        Create a fork structure (A <- B -> C)

        Args:
            graph: Graph to add the structure to
            num_nodes: Number of child nodes (default: 2) plus 1 parent node
            **kwargs: Additional parameters

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        if num_nodes is None or num_nodes < 3:
            raise GraphGenerationError(
                "Fork requires at least 3 nodes (1 parent + 2 children)")

        # Add nodes
        for i in range(num_nodes):
            graph.add_node(i)

        # Node 0 is the parent, connecting to all other nodes
        for i in range(1, num_nodes):
            graph.add_edge(0, i)

    @staticmethod
    def _create_collider_structure(graph: Union[CausalGraph, DirectedGraph],
                                   num_nodes: int,
                                   **kwargs) -> None:
        """
        Create a collider structure (A -> C <- B)

        Args:
            graph: Graph to add the structure to
            num_nodes: Number of parent nodes (default: 2) plus 1 child node
            **kwargs: Additional parameters

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        if num_nodes is None or num_nodes < 3:
            raise GraphGenerationError(
                "Collider requires at least 3 nodes (2 parents + 1 child)")

        # Add nodes
        for i in range(num_nodes):
            graph.add_node(i)

        # Node 0 is the child, receiving edges from all other nodes
        for i in range(1, num_nodes):
            graph.add_edge(i, 0)

    @staticmethod
    def _create_complete_structure(graph: Union[CausalGraph, DirectedGraph],
                                   num_nodes: int,
                                   directed: bool = True,
                                   **kwargs) -> None:
        """
        Create a complete graph where every node is connected to every other node.

        Args:
            graph: Graph to add the structure to
            num_nodes: Number of nodes in the complete graph
            directed: Whether to create directed edges
            **kwargs: Additional parameters

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        if num_nodes is None or num_nodes < 2:
            raise GraphGenerationError(
                "Complete graph requires at least 2 nodes")

        # Add nodes
        for i in range(num_nodes):
            graph.add_node(i)

        # Add edges to form a complete graph
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # Skip self-loops
                    if not directed and i > j:  # For undirected, avoid duplicate edges
                        continue
                    graph.add_edge(i, j)
                    if not directed:
                        graph.add_edge(j, i)

    @classmethod
    def _create_tree_structure(cls,
                               graph: Union[CausalGraph, DirectedGraph],
                               num_nodes: int,
                               branching_factor: int = 2,
                               **kwargs) -> None:
        """
        Create a directed tree structure where each node has at most 'branching_factor' children.

        Args:
            graph: Graph to add the structure to
            num_nodes: Total number of nodes in the tree
            branching_factor: Maximum number of children per node
            **kwargs: Additional parameters

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        if num_nodes is None or num_nodes < 1:
            raise GraphGenerationError("Tree requires at least 1 node")

        if branching_factor < 1:
            raise GraphGenerationError("Branching factor must be at least 1")

        # Add nodes
        for i in range(num_nodes):
            graph.add_node(i)

        # Skip further processing if there's only one node
        if num_nodes == 1:
            return

        # Add edges to form a tree
        # For each node (except the root), add an edge from its parent
        nodes_added = 1  # Root node is already added
        current_level_start = 0
        current_level_end = 0

        while nodes_added < num_nodes:
            next_level_start = nodes_added
            # Process all nodes in the current level
            for parent in range(current_level_start, current_level_end + 1):
                # Add up to branching_factor children for this parent
                for _ in range(branching_factor):
                    if nodes_added >= num_nodes:
                        break
                    graph.add_edge(parent, nodes_added)
                    nodes_added += 1

            # Update level boundaries
            current_level_start = next_level_start
            current_level_end = nodes_added - 1

    @staticmethod
    def _create_bipartite_structure(graph: Union[CausalGraph, DirectedGraph],
                                    n1: int = None,
                                    n2: int = None,
                                    directed: bool = True,
                                    cross_probability: float = 1.0,
                                    **kwargs) -> None:
        """
        Create a bipartite graph with two sets of nodes, and edges between the sets.

        Args:
            graph: Graph to add the structure to
            n1: Number of nodes in the first set
            n2: Number of nodes in the second set
            directed: Whether to create directed edges (from set 1 to set 2)
            cross_probability: Probability of creating an edge between nodes from different sets
            **kwargs: Additional parameters

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        if n1 is None or n1 < 1 or n2 is None or n2 < 1:
            raise GraphGenerationError(
                "Bipartite graph requires positive values for n1 and n2")

        if not 0 <= cross_probability <= 1:
            raise GraphGenerationError(
                "Edge probability must be between 0 and 1")

        # Add nodes for both sets
        total_nodes = n1 + n2
        for i in range(total_nodes):
            graph.add_node(i)

        # Add edges between nodes in different sets
        for i in range(n1):
            for j in range(n1, total_nodes):
                if random.random() < cross_probability:
                    graph.add_edge(i, j)
                    if not directed:
                        graph.add_edge(j, i)

    @staticmethod
    def _create_from_adjacency_matrix(graph: Union[CausalGraph, DirectedGraph],
                                      adjacency_matrix: List[List[int]],
                                      **kwargs) -> None:
        """
        Create a graph from an adjacency matrix.

        Args:
            graph: Graph to add the structure to
            adjacency_matrix: 2D matrix where adjacency_matrix[i][j] indicates an edge from i to j
            **kwargs: Additional parameters

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        if not adjacency_matrix:
            raise GraphGenerationError("Adjacency matrix cannot be empty")

        n = len(adjacency_matrix)

        # Check that matrix is square
        for row in adjacency_matrix:
            if len(row) != n:
                raise GraphGenerationError("Adjacency matrix must be square")

        # Add nodes
        for i in range(n):
            graph.add_node(i)

        # Add edges according to the adjacency matrix
        for i in range(n):
            for j in range(n):
                if adjacency_matrix[i][j]:
                    graph.add_edge(i, j)

    @staticmethod
    def _create_from_edge_list(graph: Union[CausalGraph, DirectedGraph],
                               edge_list: List[Tuple[int, int]],
                               num_nodes: Optional[int] = None,
                               **kwargs) -> None:
        """
        Create a graph from a list of edges.

        Args:
            graph: Graph to add the structure to
            edge_list: List of tuples (source, target) representing edges
            num_nodes: Optional number of nodes (useful if isolated nodes should be included)
            **kwargs: Additional parameters

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        if not edge_list and (num_nodes is None or num_nodes <= 0):
            raise GraphGenerationError(
                "Either edge list or num_nodes must be provided")

        # Determine the number of nodes needed
        node_set = set()
        if edge_list:
            for source, target in edge_list:
                node_set.add(source)
                node_set.add(target)

        # If num_nodes is provided, use it (even if edge_list suggests fewer nodes)
        # If not, use the maximum node ID from edge_list + 1
        required_nodes = max(node_set) + 1 if node_set else 0
        if num_nodes is None:
            num_nodes = required_nodes
        elif num_nodes < required_nodes:
            # Validate that all nodes in edge_list are within the range of num_nodes
            for source, target in edge_list:
                if source >= num_nodes or target >= num_nodes:
                    raise GraphGenerationError(
                        f"Edge ({source}, {target}) references a node outside range [0, {num_nodes-1}]")

        # Add nodes
        for i in range(num_nodes):
            graph.add_node(i)

        # Add edges from the edge list
        for source, target in edge_list:
            graph.add_edge(source, target)

    @staticmethod
    def _add_noise_edges(graph: Union[CausalGraph, DirectedGraph],
                         noise_probability: float) -> None:
        """
        Add random noise edges to the graph with the given probability.

        Args:
            graph: Graph to add noise edges to
            noise_probability: Probability of adding an edge between any two nodes

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        if not 0 <= noise_probability <= 1:
            raise GraphGenerationError(
                "Noise probability must be between 0 and 1")

        nodes = list(graph._nodes)
        n = len(nodes)

        for i in range(n):
            for j in range(n):
                if i != j and (i, j) not in graph._edges:
                    if random.random() < noise_probability:
                        graph.add_edge(i, j)

    @staticmethod
    def _validate_parameters(params_dict: Dict[str, Any]) -> None:
        """
        Validate the parameters for graph generation.

        Args:
            params_dict: Dictionary of parameter names and their valid ranges or values

        Raises:
            GraphGenerationError: If any parameter is invalid
        """
        for param_name, (param_value, valid_range) in params_dict.items():
            if isinstance(valid_range, tuple) and len(valid_range) == 2:
                min_val, max_val = valid_range
                if not (min_val <= param_value <= max_val):
                    raise GraphGenerationError(
                        f"Parameter '{param_name}' value {param_value} is outside valid range [{min_val}, {max_val}]")
            elif isinstance(valid_range, list) and param_value not in valid_range:
                raise GraphGenerationError(
                    f"Parameter '{param_name}' value {param_value} is not in valid values {valid_range}")
            elif callable(valid_range) and not valid_range(param_value):
                raise GraphGenerationError(
                    f"Parameter '{param_name}' value {param_value} is invalid")
