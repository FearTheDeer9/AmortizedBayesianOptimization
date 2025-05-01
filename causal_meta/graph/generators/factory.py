"""
Graph generator factory for the causal_meta library.

This module provides a factory pattern implementation for generating different types
of graph structures, including random graphs, scale-free networks, and predefined structures.
"""
from typing import Dict, Any, Optional, Union, List, Tuple
import random
from abc import ABC, abstractmethod
import logging
import networkx as nx

from causal_meta.graph import CausalGraph, DirectedGraph
from causal_meta.graph.generators.errors import GraphGenerationError
from causal_meta.graph.generators.scale_free import ScaleFreeNetworkGenerator
from causal_meta.graph.generators.predefined import PredefinedGraphStructureGenerator

logger = logging.getLogger(__name__)

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
                                alpha: float = 1.0,
                                beta: float = 0.0,
                                gamma: float = 0.0,
                                directed: bool = True,
                                is_causal: bool = True,
                                seed: Optional[int] = None,
                                **kwargs) -> Union[CausalGraph, DirectedGraph]:
        """
        Create a scale-free graph using the Barabási-Albert model or its variants.

        Args:
            num_nodes: Number of nodes in the graph
            m: Number of edges to attach from a new node to existing nodes
            alpha: Preferential attachment exponent (default: 1.0)
                   - alpha > 1: Increases the "rich get richer" effect
                   - alpha < 1: Decreases the effect, making the degree distribution less skewed
                   - alpha = 0: Random attachment (equivalent to an Erdős–Rényi model)
            beta: Probability of random attachment vs. preferential attachment (default: 0.0)
                  - beta=0: Pure preferential attachment (standard BA model)
                  - beta=1: Pure random attachment
                  - 0<beta<1: Mixture of both mechanisms
            gamma: Aging factor for older nodes (default: 0.0)
                  - gamma=0: No aging effect
                  - gamma>0: Older nodes become less attractive over time
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

        if alpha != 1.0 or beta != 0.0 or gamma != 0.0:
            # Use extended model if any of the additional parameters are non-default
            # Delegate to the ScaleFreeNetworkGenerator implementation
            return ScaleFreeNetworkGenerator.extended_barabasi_albert(
                num_nodes=num_nodes,
                m=m,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                directed=directed,
                is_causal=is_causal,
                seed=seed
            )
        elif alpha == 1.0:
            # Use standard Barabási-Albert model
            return ScaleFreeNetworkGenerator.barabasi_albert(
                num_nodes=num_nodes,
                m=m,
                directed=directed,
                is_causal=is_causal,
                seed=seed
            )

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
        # Use the PredefinedGraphStructureGenerator for supported structures
        if structure_type.lower() == 'chain':
            if not num_nodes:
                raise GraphGenerationError(
                    "Number of nodes must be specified for chain structure")
            return PredefinedGraphStructureGenerator.chain(
                num_nodes=num_nodes,
                noise_probability=noise_probability,
                is_causal=is_causal,
                seed=seed
            )
        elif structure_type.lower() == 'fork':
            if not num_nodes:
                raise GraphGenerationError(
                    "Number of nodes must be specified for fork structure")
            return PredefinedGraphStructureGenerator.fork(
                num_nodes=num_nodes,
                noise_probability=noise_probability,
                is_causal=is_causal,
                seed=seed
            )
        elif structure_type.lower() == 'collider':
            if not num_nodes:
                raise GraphGenerationError(
                    "Number of nodes must be specified for collider structure")
            return PredefinedGraphStructureGenerator.collider(
                num_nodes=num_nodes,
                noise_probability=noise_probability,
                is_causal=is_causal,
                seed=seed
            )
        elif structure_type.lower() == 'mediator':
            return PredefinedGraphStructureGenerator.mediator(
                noise_probability=noise_probability,
                is_causal=is_causal,
                seed=seed
            )
        elif structure_type.lower() == 'confounder':
            return PredefinedGraphStructureGenerator.confounder(
                num_nodes=num_nodes if num_nodes else 4,
                noise_probability=noise_probability,
                is_causal=is_causal,
                seed=seed
            )
        elif structure_type.lower() == 'diamond':
            return PredefinedGraphStructureGenerator.diamond(
                noise_probability=noise_probability,
                is_causal=is_causal,
                seed=seed
            )
        elif structure_type.lower() == 'm_structure':
            return PredefinedGraphStructureGenerator.m_structure(
                noise_probability=noise_probability,
                is_causal=is_causal,
                seed=seed
            )
        elif structure_type.lower() == 'instrumental_variable':
            return PredefinedGraphStructureGenerator.instrumental_variable(
                noise_probability=noise_probability,
                is_causal=is_causal,
                seed=seed
            )
        elif structure_type.lower() == 'complete':
            if not num_nodes:
                raise GraphGenerationError(
                    "Number of nodes must be specified for complete structure")
            directed = kwargs.get('directed', True)
            return PredefinedGraphStructureGenerator.complete(
                num_nodes=num_nodes,
                directed=directed,
                noise_probability=noise_probability,
                is_causal=is_causal,
                seed=seed
            )
        elif structure_type.lower() == 'tree':
            if not num_nodes:
                raise GraphGenerationError(
                    "Number of nodes must be specified for tree structure")
            branching_factor = kwargs.get('branching_factor', 2)
            return PredefinedGraphStructureGenerator.tree(
                num_nodes=num_nodes,
                branching_factor=branching_factor,
                noise_probability=noise_probability,
                is_causal=is_causal,
                seed=seed
            )
        elif structure_type.lower() == 'bipartite':
            n1 = kwargs.get('n1')
            n2 = kwargs.get('n2')
            if n1 is None or n2 is None:
                raise GraphGenerationError(
                    "n1 and n2 must be specified for bipartite structure")
            directed = kwargs.get('directed', True)
            cross_probability = kwargs.get('cross_probability', 1.0)
            return PredefinedGraphStructureGenerator.bipartite(
                n1=n1,
                n2=n2,
                directed=directed,
                cross_probability=cross_probability,
                noise_probability=noise_probability,
                is_causal=is_causal,
                seed=seed
            )
        elif structure_type.lower() == 'from_matrix':
            adjacency_matrix = kwargs.pop('adjacency_matrix', None)
            if not adjacency_matrix:
                raise GraphGenerationError(
                    "Adjacency matrix must be specified for from_matrix structure")
            return PredefinedGraphStructureGenerator.from_adjacency_matrix(
                adjacency_matrix=adjacency_matrix,
                is_causal=is_causal
            )
        elif structure_type.lower() == 'from_edges':
            edge_list = kwargs.pop('edge_list', None)
            if not edge_list:
                raise GraphGenerationError(
                    "Edge list must be specified for from_edges structure")
            return PredefinedGraphStructureGenerator.from_edge_list(
                edge_list=edge_list,
                num_nodes=num_nodes,
                is_causal=is_causal
            )
        else:
            raise GraphGenerationError(
                f"Unknown structure type: {structure_type}")

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

    @classmethod
    def create_random_dag(cls,
                          num_nodes: int,
                          edge_probability: float,
                          is_causal: bool = True,
                          seed: Optional[int] = None,
                          **kwargs) -> Union[CausalGraph, DirectedGraph]:
        """
        Create a random Directed Acyclic Graph (DAG).

        Ensures acyclicity by only allowing edges from lower-indexed nodes to
        higher-indexed nodes based on a fixed node ordering.

        Args:
            num_nodes: Number of nodes in the DAG.
            edge_probability: Probability (0 to 1) of creating an edge between
                              any two valid node pairs (respecting DAG constraints).
            is_causal: Whether to return a CausalGraph instance (default: True).
                       If False, returns a DirectedGraph.
            seed: Random seed for reproducibility (default: None).
            **kwargs: Additional unused parameters (for potential future extensions).

        Returns:
            A random DAG instance (CausalGraph or DirectedGraph).

        Raises:
            GraphGenerationError: If parameters are invalid (e.g., num_nodes <= 0,
                                edge_probability outside [0, 1]).
        """
        # Parameter Validation
        if not isinstance(num_nodes, int) or num_nodes <= 0:
            raise GraphGenerationError("Number of nodes must be a positive integer.")
        if not isinstance(edge_probability, (float, int)) or not (0.0 <= edge_probability <= 1.0):
            raise GraphGenerationError("Edge probability must be between 0.0 and 1.0.")

        # Set random seed if provided for reproducibility
        if seed is not None:
            random.seed(seed)

        # Create the appropriate graph type
        if is_causal:
            graph = CausalGraph()
        else:
            graph = DirectedGraph()

        # Add nodes (0 to num_nodes - 1)
        for i in range(num_nodes):
            graph.add_node(i)
        assert len(graph.get_nodes()) == num_nodes

        # Core loop structure for potential edges (i -> j where i < j)
        edge_count = 0
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes): # Ensure j > i
                # Logic for adding edge based on edge_probability goes here (Subtask 1.3)
                if random.random() < edge_probability:
                    graph.add_edge(i, j)
                    edge_count += 1
                #pass # No edges added in Subtask 1.2

        # Log edge count
        logger.debug(f"Generated DAG with {num_nodes} nodes and {edge_count} edges.")

        # Validation logic will be added in subtask 1.4
        # Convert to NetworkX graph for validation
        nx_graph = nx.DiGraph()
        nx_graph.add_nodes_from(graph.get_nodes())
        nx_graph.add_edges_from(graph.get_edges())
        # Validate acyclicity (should always be true by construction)
        assert nx.is_directed_acyclic_graph(nx_graph), "Generated graph is not a DAG!"

        return graph
