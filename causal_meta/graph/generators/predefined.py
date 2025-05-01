"""
Predefined graph structure generators for the causal_meta library.

This module provides implementations for generating common predefined graph structures,
including chains, forks, colliders, and various specialized DAG patterns commonly 
used in causal inference research.
"""
from typing import Optional, Union, Dict, List, Tuple, Any
import random
import numpy as np

from causal_meta.graph import CausalGraph, DirectedGraph
from causal_meta.graph.generators.errors import GraphGenerationError


class PredefinedGraphStructureGenerator:
    """
    Generator class for predefined graph structures.

    This class provides methods for generating common causal graph structures and patterns
    that are frequently used in causal inference research, including chains, forks, colliders,
    and more complex DAG structures.
    """

    @staticmethod
    def chain(
        num_nodes: int,
        noise_probability: float = 0.0,
        is_causal: bool = True,
        seed: Optional[int] = None
    ) -> Union[CausalGraph, DirectedGraph]:
        """
        Generate a chain structure (causal chain) where each node causes the next node.

        Pattern: X₁ → X₂ → X₃ → ... → Xₙ

        Args:
            num_nodes: Number of nodes in the chain
            noise_probability: Probability of adding random noise edges
            is_causal: Whether to return a CausalGraph (default: True)
            seed: Random seed for reproducibility (default: None)

        Returns:
            A graph with a chain structure

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        # Validate parameters
        if num_nodes < 2:
            raise GraphGenerationError(
                "Chain structure requires at least 2 nodes")
        if not 0 <= noise_probability <= 1:
            raise GraphGenerationError(
                "Noise probability must be between 0 and 1")

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

        # Create the chain structure
        for i in range(num_nodes - 1):
            graph.add_edge(i, i + 1)

        # Add noise edges if specified
        if noise_probability > 0:
            for i in range(num_nodes):
                for j in range(num_nodes):
                    # Skip self-loops and existing edges
                    if i == j or graph.has_edge(i, j):
                        continue
                    # Skip edges that would create cycles
                    if j <= i:  # Assuming nodes are in topological order
                        continue
                    # Add edge with the specified noise probability
                    if random.random() < noise_probability:
                        graph.add_edge(i, j)

        return graph

    @staticmethod
    def fork(
        num_nodes: int,
        noise_probability: float = 0.0,
        is_causal: bool = True,
        seed: Optional[int] = None
    ) -> Union[CausalGraph, DirectedGraph]:
        """
        Generate a fork structure (common cause) where a central node is the cause of all other nodes.

        Pattern: X₁ ← X₀ → X₂, X₀ → X₃, ..., X₀ → Xₙ₋₁

        Args:
            num_nodes: Number of nodes in the fork (including the common cause)
            noise_probability: Probability of adding random noise edges
            is_causal: Whether to return a CausalGraph (default: True)
            seed: Random seed for reproducibility (default: None)

        Returns:
            A graph with a fork structure

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        # Validate parameters
        if num_nodes < 3:
            raise GraphGenerationError(
                "Fork structure requires at least 3 nodes (1 parent and 2 children)")
        if not 0 <= noise_probability <= 1:
            raise GraphGenerationError(
                "Noise probability must be between 0 and 1")

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

        # Create the fork structure
        # Node 0 is the common cause of all other nodes
        for i in range(1, num_nodes):
            graph.add_edge(0, i)

        # Add noise edges if specified
        if noise_probability > 0:
            for i in range(num_nodes):
                for j in range(num_nodes):
                    # Skip self-loops and existing edges
                    if i == j or graph.has_edge(i, j):
                        continue
                    # Skip edges that would create cycles
                    if j <= i and not (i == 0):  # Allow edges from node 0
                        continue
                    # Add edge with the specified noise probability
                    if random.random() < noise_probability:
                        graph.add_edge(i, j)

        return graph

    @staticmethod
    def collider(
        num_nodes: int,
        noise_probability: float = 0.0,
        is_causal: bool = True,
        seed: Optional[int] = None
    ) -> Union[CausalGraph, DirectedGraph]:
        """
        Generate a collider structure (common effect) where all nodes cause a central node.

        Pattern: X₁ → X₀ ← X₂, X₃ → X₀, ..., Xₙ₋₁ → X₀

        Args:
            num_nodes: Number of nodes in the collider (including the common effect)
            noise_probability: Probability of adding random noise edges
            is_causal: Whether to return a CausalGraph (default: True)
            seed: Random seed for reproducibility (default: None)

        Returns:
            A graph with a collider structure

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        # Validate parameters
        if num_nodes < 3:
            raise GraphGenerationError(
                "Collider structure requires at least 3 nodes (2 parents and 1 child)")
        if not 0 <= noise_probability <= 1:
            raise GraphGenerationError(
                "Noise probability must be between 0 and 1")

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

        # Create the collider structure
        # Node 0 is the common effect of all other nodes
        for i in range(1, num_nodes):
            graph.add_edge(i, 0)

        # Add noise edges if specified
        if noise_probability > 0:
            for i in range(num_nodes):
                for j in range(num_nodes):
                    # Skip self-loops and existing edges
                    if i == j or graph.has_edge(i, j):
                        continue
                    # Skip edges that would create cycles
                    if j <= i and j != 0:  # Allow edges to node 0
                        continue
                    # Add edge with the specified noise probability
                    if random.random() < noise_probability:
                        graph.add_edge(i, j)

        return graph

    @staticmethod
    def mediator(
        noise_probability: float = 0.0,
        is_causal: bool = True,
        seed: Optional[int] = None
    ) -> Union[CausalGraph, DirectedGraph]:
        """
        Generate a mediator structure where one variable mediates the effect of 
        another on a third variable.

        Pattern: X → M → Y, potentially with direct effect X → Y

        Args:
            noise_probability: Probability of adding the direct effect edge (X → Y)
            is_causal: Whether to return a CausalGraph (default: True)
            seed: Random seed for reproducibility (default: None)

        Returns:
            A graph with a mediator structure

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        # Validate parameters
        if not 0 <= noise_probability <= 1:
            raise GraphGenerationError(
                "Noise probability must be between 0 and 1")

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
        for i in range(3):  # X, M, Y
            graph.add_node(i)

        # Create the mediator structure
        # X → M
        graph.add_edge(0, 1)
        # M → Y
        graph.add_edge(1, 2)

        # Add direct effect X → Y with the specified probability
        if random.random() < noise_probability:
            graph.add_edge(0, 2)

        return graph

    @staticmethod
    def confounder(
        num_nodes: int = 4,
        noise_probability: float = 0.0,
        is_causal: bool = True,
        seed: Optional[int] = None
    ) -> Union[CausalGraph, DirectedGraph]:
        """
        Generate a confounder structure where unobserved variables confound the relationship
        between observed variables.

        Pattern (for 4 nodes): Z → X, Z → Y (where Z is unobserved)
        Implementation: X ← Z → Y (represented as X ← 0 → Y)

        Args:
            num_nodes: Total number of nodes (including the confounder)
            noise_probability: Probability of adding random noise edges
            is_causal: Whether to return a CausalGraph (default: True)
            seed: Random seed for reproducibility (default: None)

        Returns:
            A graph with a confounder structure

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        # Validate parameters
        if num_nodes < 3:
            raise GraphGenerationError(
                "Confounder structure requires at least 3 nodes (1 confounder and 2 observed variables)")
        if not 0 <= noise_probability <= 1:
            raise GraphGenerationError(
                "Noise probability must be between 0 and 1")

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

        # Node 0 is the confounder that affects all other nodes
        for i in range(1, num_nodes):
            graph.add_edge(0, i)

        # Add noise edges if specified
        if noise_probability > 0:
            for i in range(1, num_nodes):  # Start from 1 to exclude confounder
                for j in range(1, num_nodes):
                    # Skip self-loops and existing edges
                    if i == j or graph.has_edge(i, j):
                        continue
                    # Add edge with the specified noise probability
                    if random.random() < noise_probability:
                        graph.add_edge(i, j)

        return graph

    @staticmethod
    def diamond(
        noise_probability: float = 0.0,
        is_causal: bool = True,
        seed: Optional[int] = None
    ) -> Union[CausalGraph, DirectedGraph]:
        """
        Generate a diamond structure with two parallel paths from a source to a sink.

        Pattern: A → B → D, A → C → D

        Args:
            noise_probability: Probability of adding random noise edges
            is_causal: Whether to return a CausalGraph (default: True)
            seed: Random seed for reproducibility (default: None)

        Returns:
            A graph with a diamond structure

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        # Validate parameters
        if not 0 <= noise_probability <= 1:
            raise GraphGenerationError(
                "Noise probability must be between 0 and 1")

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Create the appropriate graph type
        if is_causal:
            graph = CausalGraph()
        else:
            graph = DirectedGraph()

        # Add nodes to the graph (A, B, C, D)
        for i in range(4):
            graph.add_node(i)

        # Create the diamond structure
        # A → B
        graph.add_edge(0, 1)
        # A → C
        graph.add_edge(0, 2)
        # B → D
        graph.add_edge(1, 3)
        # C → D
        graph.add_edge(2, 3)

        # Add noise edges if specified
        if noise_probability > 0:
            # Only consider adding B → C or C → B to maintain DAG property
            if random.random() < noise_probability:
                # Randomly decide the direction
                if random.random() < 0.5:
                    graph.add_edge(1, 2)  # B → C
                else:
                    graph.add_edge(2, 1)  # C → B

            # Consider adding A → D directly
            if random.random() < noise_probability:
                graph.add_edge(0, 3)

        return graph

    @staticmethod
    def m_structure(
        noise_probability: float = 0.0,
        is_causal: bool = True,
        seed: Optional[int] = None
    ) -> Union[CausalGraph, DirectedGraph]:
        """
        Generate an M-structure, which is used in causal discovery to illustrate d-separation.

        Pattern: X₁ → X₃ ← X₂, X₂ → X₄ ← X₁

        Args:
            noise_probability: Probability of adding random noise edges
            is_causal: Whether to return a CausalGraph (default: True)
            seed: Random seed for reproducibility (default: None)

        Returns:
            A graph with an M-structure

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        # Validate parameters
        if not 0 <= noise_probability <= 1:
            raise GraphGenerationError(
                "Noise probability must be between 0 and 1")

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
        for i in range(5):  # Include an extra node for noise
            graph.add_node(i)

        # Create the M-structure
        # X₁ → X₃
        graph.add_edge(0, 2)
        # X₂ → X₃
        graph.add_edge(1, 2)
        # X₁ → X₄
        graph.add_edge(0, 3)
        # X₂ → X₄
        graph.add_edge(1, 3)

        # Add noise edges if specified
        if noise_probability > 0:
            # Consider adding X₁ → X₂ or X₂ → X₁
            if random.random() < noise_probability:
                # Randomly decide the direction
                if random.random() < 0.5:
                    graph.add_edge(0, 1)  # X₁ → X₂
                else:
                    graph.add_edge(1, 0)  # X₂ → X₁

            # Consider adding edges to/from the extra node
            for i in range(4):
                if random.random() < noise_probability:
                    graph.add_edge(i, 4)  # Xi → X₅
                if random.random() < noise_probability:
                    graph.add_edge(4, i)  # X₅ → Xi

        return graph

    @staticmethod
    def instrumental_variable(
        noise_probability: float = 0.0,
        is_causal: bool = True,
        seed: Optional[int] = None
    ) -> Union[CausalGraph, DirectedGraph]:
        """
        Generate an instrumental variable structure used in causal inference for handling confounding.

        Pattern: Z → X → Y, with (potentially unobserved) confounder U affecting both X and Y.
        Implementation: Z → X → Y, with a direct U → X and U → Y (U is node 3)

        Args:
            noise_probability: Probability of adding the direct effect edge (Z → Y)
            is_causal: Whether to return a CausalGraph (default: True)
            seed: Random seed for reproducibility (default: None)

        Returns:
            A graph with an instrumental variable structure

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        # Validate parameters
        if not 0 <= noise_probability <= 1:
            raise GraphGenerationError(
                "Noise probability must be between 0 and 1")

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Create the appropriate graph type
        if is_causal:
            graph = CausalGraph()
        else:
            graph = DirectedGraph()

        # Add nodes to the graph: Z, X, Y, U
        for i in range(4):
            graph.add_node(i)

        # Create the instrumental variable structure
        # Z → X
        graph.add_edge(0, 1)
        # X → Y
        graph.add_edge(1, 2)
        # U → X
        graph.add_edge(3, 1)
        # U → Y
        graph.add_edge(3, 2)

        # Add direct effect Z → Y with the specified probability
        if random.random() < noise_probability:
            graph.add_edge(0, 2)

        return graph

    @staticmethod
    def from_adjacency_matrix(
        adjacency_matrix: List[List[int]],
        is_causal: bool = True
    ) -> Union[CausalGraph, DirectedGraph]:
        """
        Generate a graph from a specified adjacency matrix.

        Args:
            adjacency_matrix: A 2D matrix where adjacency_matrix[i][j] indicates an edge from i to j
            is_causal: Whether to return a CausalGraph (default: True)

        Returns:
            A graph based on the provided adjacency matrix

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

        # Create the appropriate graph type
        if is_causal:
            graph = CausalGraph()
        else:
            graph = DirectedGraph()

        # Add nodes to the graph
        for i in range(n):
            graph.add_node(i)

        # Add edges according to the adjacency matrix
        for i in range(n):
            for j in range(n):
                if adjacency_matrix[i][j]:
                    graph.add_edge(i, j)

        return graph

    @staticmethod
    def from_edge_list(
        edge_list: List[Tuple[int, int]],
        num_nodes: Optional[int] = None,
        is_causal: bool = True
    ) -> Union[CausalGraph, DirectedGraph]:
        """
        Generate a graph from a specified list of edges.

        Args:
            edge_list: List of tuples (source, target) representing edges
            num_nodes: Number of nodes to include in the graph (optional, inferred from edge_list if not provided)
            is_causal: Whether to return a CausalGraph (default: True)

        Returns:
            A graph based on the provided edge list

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

        # Create the appropriate graph type
        if is_causal:
            graph = CausalGraph()
        else:
            graph = DirectedGraph()

        # Add nodes to the graph
        for i in range(num_nodes):
            graph.add_node(i)

        # Add edges from the edge list
        for source, target in edge_list:
            graph.add_edge(source, target)

        return graph

    @staticmethod
    def complete(
        num_nodes: int,
        directed: bool = True,
        noise_probability: float = 0.0,
        is_causal: bool = True,
        seed: Optional[int] = None
    ) -> Union[CausalGraph, DirectedGraph]:
        """
        Generate a complete graph where every node is connected to every other node.

        Args:
            num_nodes: Number of nodes in the complete graph
            directed: Whether to create directed edges
            noise_probability: Probability of removing edges (acts as an anti-noise parameter)
            is_causal: Whether to return a CausalGraph (default: True)
            seed: Random seed for reproducibility (default: None)

        Returns:
            A complete graph

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        # Validate parameters
        if num_nodes < 2:
            raise GraphGenerationError(
                "Complete graph requires at least 2 nodes")
        if not 0 <= noise_probability <= 1:
            raise GraphGenerationError(
                "Noise probability must be between 0 and 1")

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

        # Add edges to form a complete graph
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # Skip self-loops
                    if not directed and i > j:  # For undirected, avoid duplicate edges
                        continue

                    # Add edge unless we remove it based on noise probability
                    if random.random() >= noise_probability:
                        graph.add_edge(i, j)
                        if not directed:
                            graph.add_edge(j, i)

        return graph

    @staticmethod
    def tree(
        num_nodes: int,
        branching_factor: int = 2,
        noise_probability: float = 0.0,
        is_causal: bool = True,
        seed: Optional[int] = None
    ) -> Union[CausalGraph, DirectedGraph]:
        """
        Generate a tree structure with a specified branching factor.

        Args:
            num_nodes: Total number of nodes in the tree
            branching_factor: Maximum number of children for each node
            noise_probability: Probability of adding random noise edges
            is_causal: Whether to return a CausalGraph (default: True)
            seed: Random seed for reproducibility (default: None)

        Returns:
            A tree-structured graph

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        # Validate parameters
        if num_nodes < 1:
            raise GraphGenerationError("Tree requires at least 1 node")
        if branching_factor < 1:
            raise GraphGenerationError("Branching factor must be at least 1")
        if not 0 <= noise_probability <= 1:
            raise GraphGenerationError(
                "Noise probability must be between 0 and 1")

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

        # Create the tree structure
        # Start with node 0 as the root
        # For each node, determine its parent based on its index
        for i in range(1, num_nodes):
            # The parent index is (i-1) // branching_factor
            parent = (i - 1) // branching_factor
            graph.add_edge(parent, i)

        # Add noise edges if specified
        if noise_probability > 0:
            for i in range(num_nodes):
                for j in range(num_nodes):
                    # Skip self-loops and existing edges
                    if i == j or graph.has_edge(i, j):
                        continue
                    # Skip edges that would create cycles
                    if j <= i:
                        continue
                    # Add edge with the specified noise probability
                    if random.random() < noise_probability:
                        graph.add_edge(i, j)

        return graph

    @staticmethod
    def bipartite(
        n1: int,
        n2: int,
        directed: bool = True,
        cross_probability: float = 1.0,
        noise_probability: float = 0.0,
        is_causal: bool = True,
        seed: Optional[int] = None
    ) -> Union[CausalGraph, DirectedGraph]:
        """
        Generate a bipartite graph with two sets of nodes and edges between them.

        Args:
            n1: Number of nodes in the first set
            n2: Number of nodes in the second set
            directed: Whether to create directed edges (from set 1 to set 2)
            cross_probability: Probability of creating an edge between nodes from different sets
            noise_probability: Probability of adding random noise edges within the same set
            is_causal: Whether to return a CausalGraph (default: True)
            seed: Random seed for reproducibility (default: None)

        Returns:
            A bipartite graph

        Raises:
            GraphGenerationError: If parameters are invalid
        """
        # Validate parameters
        if n1 < 1 or n2 < 1:
            raise GraphGenerationError(
                "Both node sets must have at least 1 node")
        if not 0 <= cross_probability <= 1:
            raise GraphGenerationError(
                "Cross probability must be between 0 and 1")
        if not 0 <= noise_probability <= 1:
            raise GraphGenerationError(
                "Noise probability must be between 0 and 1")

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Create the appropriate graph type
        if is_causal:
            graph = CausalGraph()
        else:
            graph = DirectedGraph()

        # Total number of nodes
        num_nodes = n1 + n2

        # Add nodes to the graph
        for i in range(num_nodes):
            graph.add_node(i)

        # Create edges between the two sets based on cross_probability
        for i in range(n1):
            for j in range(n1, num_nodes):
                if random.random() < cross_probability:
                    graph.add_edge(i, j)
                    if not directed:
                        graph.add_edge(j, i)

        # Add noise edges within the same set if specified
        if noise_probability > 0:
            # Edges within the first set
            for i in range(n1):
                for j in range(n1):
                    if i != j and not graph.has_edge(i, j):
                        if random.random() < noise_probability:
                            graph.add_edge(i, j)
                            if not directed:
                                graph.add_edge(j, i)

            # Edges within the second set
            for i in range(n1, num_nodes):
                for j in range(n1, num_nodes):
                    if i != j and not graph.has_edge(i, j):
                        if random.random() < noise_probability:
                            graph.add_edge(i, j)
                            if not directed:
                                graph.add_edge(j, i)

        return graph
