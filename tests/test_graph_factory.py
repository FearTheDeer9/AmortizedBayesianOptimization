"""
Tests for the graph factory module.
"""
import unittest
from unittest.mock import patch, MagicMock
import random
import math
from collections import Counter
import networkx as nx

from causal_meta.graph import CausalGraph, DirectedGraph
from causal_meta.graph.generators import GraphFactory, GraphGenerationError


class TestGraphFactory(unittest.TestCase):
    """Tests for the GraphFactory class."""

    def test_factory_initialization(self):
        """Test that the GraphFactory can be initialized."""
        # GraphFactory is a static class with class methods, so no initialization is needed
        self.assertIsNotNone(GraphFactory)

    def test_create_graph_with_invalid_type(self):
        """Test that creating a graph with an invalid type raises an error."""
        with self.assertRaises(GraphGenerationError):
            GraphFactory.create_graph("invalid_type")

    def test_create_graph_routes_to_random(self):
        """Test that create_graph routes to create_random_graph for random type."""
        with patch.object(GraphFactory, 'create_random_graph', return_value=MagicMock(spec=CausalGraph)) as mock_method:
            GraphFactory.create_graph(
                "random", num_nodes=10, edge_probability=0.3)
            mock_method.assert_called_once_with(
                num_nodes=10, edge_probability=0.3)

    def test_create_graph_routes_to_scale_free(self):
        """Test that create_graph routes to create_scale_free_graph for scale_free type."""
        with patch.object(GraphFactory, 'create_scale_free_graph', return_value=MagicMock(spec=CausalGraph)) as mock_method:
            GraphFactory.create_graph("scale_free", num_nodes=10, m=2)
            mock_method.assert_called_once_with(num_nodes=10, m=2)

    def test_create_graph_routes_to_predefined(self):
        """Test that create_graph routes to create_predefined_graph for predefined type."""
        with patch.object(GraphFactory, 'create_predefined_graph', return_value=MagicMock(spec=CausalGraph)) as mock_method:
            GraphFactory.create_graph(
                "predefined", structure_type="chain", num_nodes=5)
            mock_method.assert_called_once_with(
                structure_type="chain", num_nodes=5)

    def test_parameter_validation(self):
        """Test the parameter validation method."""
        # Test valid parameter within range
        GraphFactory._validate_parameters({"param1": (5, (1, 10))})

        # Test invalid parameter outside range
        with self.assertRaises(GraphGenerationError):
            GraphFactory._validate_parameters({"param1": (15, (1, 10))})

        # Test valid parameter in list
        GraphFactory._validate_parameters(
            {"param1": ("value1", ["value1", "value2"])})

        # Test invalid parameter not in list
        with self.assertRaises(GraphGenerationError):
            GraphFactory._validate_parameters(
                {"param1": ("value3", ["value1", "value2"])})

        # Test valid parameter with custom validation function
        GraphFactory._validate_parameters(
            {"param1": (5, lambda x: x % 5 == 0)})

        # Test invalid parameter with custom validation function
        with self.assertRaises(GraphGenerationError):
            GraphFactory._validate_parameters(
                {"param1": (7, lambda x: x % 5 == 0)})

    def test_create_random_graph_validates_parameters(self):
        """Test that create_random_graph validates its parameters."""
        # Test with invalid num_nodes
        with self.assertRaises(GraphGenerationError):
            GraphFactory.create_random_graph(num_nodes=0, edge_probability=0.5)

        # Test with invalid edge_probability (too high)
        with self.assertRaises(GraphGenerationError):
            GraphFactory.create_random_graph(
                num_nodes=10, edge_probability=1.5)

        # Test with invalid edge_probability (too low)
        with self.assertRaises(GraphGenerationError):
            GraphFactory.create_random_graph(
                num_nodes=10, edge_probability=-0.5)

    def test_create_random_graph_returns_correct_type(self):
        """Test that create_random_graph returns the correct graph type."""
        # Test with is_causal=True
        graph = GraphFactory.create_random_graph(
            num_nodes=5, edge_probability=0.5, is_causal=True)
        self.assertIsInstance(graph, CausalGraph)

        # Test with is_causal=False
        graph = GraphFactory.create_random_graph(
            num_nodes=5, edge_probability=0.5, is_causal=False)
        self.assertIsInstance(graph, DirectedGraph)

    def test_create_random_graph_has_correct_nodes(self):
        """Test that create_random_graph creates a graph with the correct number of nodes."""
        num_nodes = 10
        graph = GraphFactory.create_random_graph(
            num_nodes=num_nodes, edge_probability=0.5)
        self.assertEqual(len(graph._nodes), num_nodes)

        # Check that nodes are numbered 0 to num_nodes-1
        for i in range(num_nodes):
            self.assertIn(i, graph._nodes)

    def test_create_random_graph_edge_probability(self):
        """Test that create_random_graph respects the edge probability."""
        # Set a fixed seed for reproducibility
        seed = 42
        random.seed(seed)

        # Create a graph with 100 nodes and edge probability of 0.3
        num_nodes = 100
        edge_probability = 0.3
        graph = GraphFactory.create_random_graph(
            num_nodes=num_nodes,
            edge_probability=edge_probability,
            seed=seed
        )

        # Calculate the expected number of edges (approximately)
        # In a directed graph without self-loops, there are n(n-1) possible edges
        possible_edges = num_nodes * (num_nodes - 1)
        expected_edges = possible_edges * edge_probability

        # Allow for some deviation due to randomness (within 10%)
        acceptable_error = 0.1
        lower_bound = expected_edges * (1 - acceptable_error)
        upper_bound = expected_edges * (1 + acceptable_error)

        actual_edges = len(graph._edges)
        self.assertTrue(
            lower_bound <= actual_edges <= upper_bound,
            f"Expected {expected_edges} edges (±{acceptable_error*100}%), got {actual_edges}"
        )

    def test_create_random_graph_undirected(self):
        """Test that create_random_graph handles undirected graphs correctly."""
        # Create an undirected graph
        num_nodes = 10
        graph = GraphFactory.create_random_graph(
            num_nodes=num_nodes,
            edge_probability=0.5,
            directed=False
        )

        # Check that for each edge (i,j), the edge (j,i) also exists
        for i, j in graph._edges:
            self.assertIn((j, i), graph._edges)

    def test_create_random_graph_seed_reproducibility(self):
        """Test that create_random_graph generates the same graph when using the same seed."""
        seed = 42
        num_nodes = 20
        edge_probability = 0.3

        # Create two graphs with the same seed
        graph1 = GraphFactory.create_random_graph(
            num_nodes=num_nodes,
            edge_probability=edge_probability,
            seed=seed
        )

        graph2 = GraphFactory.create_random_graph(
            num_nodes=num_nodes,
            edge_probability=edge_probability,
            seed=seed
        )

        # Both graphs should have the same edges
        self.assertEqual(graph1._edges, graph2._edges)

        # Create a third graph with a different seed
        graph3 = GraphFactory.create_random_graph(
            num_nodes=num_nodes,
            edge_probability=edge_probability,
            seed=seed+1
        )

        # It should have different edges
        self.assertNotEqual(graph1._edges, graph3._edges)

    def test_create_scale_free_graph_validates_parameters(self):
        """Test that create_scale_free_graph validates its parameters."""
        # Test with invalid num_nodes (too small)
        with self.assertRaises(GraphGenerationError):
            GraphFactory.create_scale_free_graph(num_nodes=2, m=2)

        # Test with invalid m (too large)
        with self.assertRaises(GraphGenerationError):
            GraphFactory.create_scale_free_graph(num_nodes=10, m=10)

        # Test with invalid m (too small)
        with self.assertRaises(GraphGenerationError):
            GraphFactory.create_scale_free_graph(num_nodes=10, m=0)

    def test_create_scale_free_graph_returns_correct_type(self):
        """Test that create_scale_free_graph returns the correct graph type."""
        # Test with is_causal=True
        graph = GraphFactory.create_scale_free_graph(
            num_nodes=10, m=2, is_causal=True)
        self.assertIsInstance(graph, CausalGraph)

        # Test with is_causal=False
        graph = GraphFactory.create_scale_free_graph(
            num_nodes=10, m=2, is_causal=False)
        self.assertIsInstance(graph, DirectedGraph)

    def test_create_scale_free_graph_has_correct_nodes(self):
        """Test that create_scale_free_graph creates a graph with the correct number of nodes."""
        num_nodes = 20
        m = 3
        graph = GraphFactory.create_scale_free_graph(num_nodes=num_nodes, m=m)
        self.assertEqual(len(graph._nodes), num_nodes)

        # Check that nodes are numbered 0 to num_nodes-1
        for i in range(num_nodes):
            self.assertIn(i, graph._nodes)

    def test_create_scale_free_graph_has_correct_edges(self):
        """Test that create_scale_free_graph creates a graph with the correct number of edges."""
        num_nodes = 100
        m = 2
        graph = GraphFactory.create_scale_free_graph(num_nodes=num_nodes, m=m)

        # In the Barabási-Albert model:
        # - Initial nodes (m+1) form a complete graph with m(m+1)/2 edges
        # - Each new node (n-m-1 of them) adds m edges
        expected_edges = (m * (m + 1)) // 2 + m * (num_nodes - m - 1)

        # For a directed graph, the edge count should match exactly
        self.assertEqual(len(graph._edges), expected_edges)

    def test_create_scale_free_graph_undirected(self):
        """Test that create_scale_free_graph handles undirected graphs correctly."""
        num_nodes = 20
        m = 2
        graph = GraphFactory.create_scale_free_graph(
            num_nodes=num_nodes,
            m=m,
            directed=False
        )

        # Check that for each edge (i,j), the edge (j,i) also exists
        for i, j in graph._edges:
            self.assertIn((j, i), graph._edges)

    def test_create_scale_free_graph_seed_reproducibility(self):
        """Test that create_scale_free_graph generates the same graph when using the same seed."""
        seed = 42
        num_nodes = 30
        m = 2

        # Create two graphs with the same seed
        graph1 = GraphFactory.create_scale_free_graph(
            num_nodes=num_nodes,
            m=m,
            seed=seed
        )

        graph2 = GraphFactory.create_scale_free_graph(
            num_nodes=num_nodes,
            m=m,
            seed=seed
        )

        # Both graphs should have the same edges
        self.assertEqual(graph1._edges, graph2._edges)

        # Create a third graph with a different seed
        graph3 = GraphFactory.create_scale_free_graph(
            num_nodes=num_nodes,
            m=m,
            seed=seed+1
        )

        # It should have different edges
        self.assertNotEqual(graph1._edges, graph3._edges)

    def test_create_scale_free_graph_follows_power_law(self):
        """Test that create_scale_free_graph generates a graph with power-law degree distribution."""
        num_nodes = 1000
        m = 2
        seed = 42

        graph = GraphFactory.create_scale_free_graph(
            num_nodes=num_nodes,
            m=m,
            seed=seed
        )

        # Calculate degree distribution
        degrees = []
        for node in range(num_nodes):
            # In a directed scale-free graph, we look at in-degree
            degree = len(graph.get_predecessors(node))
            degrees.append(degree)

        degree_counts = Counter(degrees)

        # Calculate the power-law exponent using linear regression on log-log plot
        # P(k) ~ k^(-gamma) where gamma is the power-law exponent
        x_values = []  # log(degree)
        y_values = []  # log(frequency)

        for degree, count in sorted(degree_counts.items()):
            if degree == 0:
                continue  # Skip degree 0 for log calculations

            x_values.append(math.log(degree))
            y_values.append(math.log(count))

        # Calculate slope using simple linear regression
        if len(x_values) > 1:
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x*y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x*x for x in x_values)

            # Slope of the regression line
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

            # The power-law exponent is the negative of the slope
            gamma = -slope

            # In Barabási-Albert model, gamma should be close to 3
            # Allow for some deviation due to finite size effects
            self.assertTrue(
                2.0 <= gamma <= 4.0,
                f"Power-law exponent {gamma} outside expected range [2.0, 4.0]"
            )
        else:
            self.fail("Not enough unique degrees to calculate power-law exponent")

    def test_create_predefined_graph_chain(self):
        """Test creating a chain graph structure."""
        num_nodes = 5
        graph = GraphFactory.create_predefined_graph(
            structure_type="chain",
            num_nodes=num_nodes,
            is_causal=True
        )

        # Check number of nodes
        self.assertEqual(len(graph._nodes), num_nodes)

        # Check edges - should be a simple chain
        expected_edges = [(i, i+1) for i in range(num_nodes-1)]
        self.assertEqual(sorted(graph._edges), sorted(expected_edges))

    def test_create_predefined_graph_fork(self):
        """Test creating a fork graph structure."""
        num_nodes = 4  # 1 parent, 3 children
        graph = GraphFactory.create_predefined_graph(
            structure_type="fork",
            num_nodes=num_nodes,
            is_causal=True
        )

        # Check number of nodes
        self.assertEqual(len(graph._nodes), num_nodes)

        # Check edges - node 0 should connect to all others
        expected_edges = [(0, i) for i in range(1, num_nodes)]
        self.assertEqual(sorted(graph._edges), sorted(expected_edges))

    def test_create_predefined_graph_collider(self):
        """Test creating a collider graph structure."""
        num_nodes = 4  # 1 child, 3 parents
        graph = GraphFactory.create_predefined_graph(
            structure_type="collider",
            num_nodes=num_nodes,
            is_causal=True
        )

        # Check number of nodes
        self.assertEqual(len(graph._nodes), num_nodes)

        # Check edges - all nodes should point to node 0
        expected_edges = [(i, 0) for i in range(1, num_nodes)]
        self.assertEqual(sorted(graph._edges), sorted(expected_edges))

    def test_create_predefined_graph_complete(self):
        """Test creating a complete graph structure."""
        num_nodes = 4
        graph = GraphFactory.create_predefined_graph(
            structure_type="complete",
            num_nodes=num_nodes,
            is_causal=True
        )

        # Check number of nodes
        self.assertEqual(len(graph._nodes), num_nodes)

        # Check edges - every node should connect to every other node
        expected_edges = [(i, j) for i in range(num_nodes)
                          for j in range(num_nodes) if i != j]
        self.assertEqual(sorted(graph._edges), sorted(expected_edges))

    def test_create_predefined_graph_tree(self):
        """Test creating a tree graph structure."""
        num_nodes = 7  # Root + 2 children + 4 grandchildren
        branching_factor = 2
        graph = GraphFactory.create_predefined_graph(
            structure_type="tree",
            num_nodes=num_nodes,
            branching_factor=branching_factor,
            is_causal=True
        )

        # Check number of nodes
        self.assertEqual(len(graph._nodes), num_nodes)

        # Check edges - should follow the tree structure
        # Node 0 connects to 1,2
        # Node 1 connects to 3,4
        # Node 2 connects to 5,6
        expected_edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
        self.assertEqual(sorted(graph._edges), sorted(expected_edges))

    def test_create_predefined_graph_bipartite(self):
        """Test creating a bipartite graph structure."""
        n1 = 3  # First set size
        n2 = 2  # Second set size
        graph = GraphFactory.create_predefined_graph(
            structure_type="bipartite",
            n1=n1,
            n2=n2,
            cross_probability=1.0,  # Ensure all possible edges are created
            is_causal=True
        )

        # Check number of nodes
        self.assertEqual(len(graph._nodes), n1 + n2)

        # Check edges - nodes from first set should only connect to second set
        expected_edges = [(i, j) for i in range(n1)
                          for j in range(n1, n1 + n2)]
        self.assertEqual(sorted(graph._edges), sorted(expected_edges))

    def test_create_predefined_graph_from_matrix(self):
        """Test creating a graph from an adjacency matrix."""
        adjacency_matrix = [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ]
        graph = GraphFactory.create_predefined_graph(
            structure_type="from_matrix",
            adjacency_matrix=adjacency_matrix,
            is_causal=True
        )

        # Check number of nodes
        self.assertEqual(len(graph._nodes), 3)

        # Check edges match the adjacency matrix
        expected_edges = [(0, 1), (1, 2), (2, 0)]
        self.assertEqual(sorted(graph._edges), sorted(expected_edges))

    def test_create_predefined_graph_from_edges(self):
        """Test creating a graph from an edge list."""
        edge_list = [(0, 1), (1, 2), (2, 0)]
        graph = GraphFactory.create_predefined_graph(
            structure_type="from_edges",
            edge_list=edge_list,
            is_causal=True
        )

        # Check edges match the provided list
        self.assertEqual(sorted(graph._edges), sorted(edge_list))

        # Test with explicit node count (including isolated nodes)
        graph2 = GraphFactory.create_predefined_graph(
            structure_type="from_edges",
            edge_list=edge_list,
            num_nodes=5,  # Creates 2 isolated nodes
            is_causal=True
        )
        self.assertEqual(len(graph2._nodes), 5)
        self.assertEqual(sorted(graph2._edges), sorted(edge_list))

    def test_create_predefined_graph_with_noise(self):
        """Test that adding noise edges works correctly."""
        num_nodes = 5
        # Create a chain with high noise probability
        graph = GraphFactory.create_predefined_graph(
            structure_type="chain",
            num_nodes=num_nodes,
            noise_probability=0.8,  # High probability to ensure noise edges
            seed=42,
            is_causal=True
        )

        # Original chain has num_nodes-1 edges
        # With noise, we expect significantly more
        self.assertGreater(len(graph._edges), num_nodes - 1)

    def test_create_predefined_graph_invalid_structure(self):
        """Test that creating an invalid structure raises an error."""
        with self.assertRaises(GraphGenerationError):
            GraphFactory.create_predefined_graph(
                structure_type="nonexistent_structure",
                num_nodes=5
            )

    def test_create_predefined_graph_invalid_parameters(self):
        """Test that invalid parameters for predefined structures raise errors."""
        # Test chain with too few nodes
        with self.assertRaises(GraphGenerationError):
            GraphFactory.create_predefined_graph(
                structure_type="chain",
                num_nodes=1
            )

        # Test fork with too few nodes
        with self.assertRaises(GraphGenerationError):
            GraphFactory.create_predefined_graph(
                structure_type="fork",
                num_nodes=2
            )

        # Test collider with too few nodes
        with self.assertRaises(GraphGenerationError):
            GraphFactory.create_predefined_graph(
                structure_type="collider",
                num_nodes=2
            )

        # Test tree with invalid branching factor
        with self.assertRaises(GraphGenerationError):
            GraphFactory.create_predefined_graph(
                structure_type="tree",
                num_nodes=5,
                branching_factor=0
            )

        # Test bipartite with missing parameters
        with self.assertRaises(GraphGenerationError):
            GraphFactory.create_predefined_graph(
                structure_type="bipartite",
                n1=None,
                n2=None
            )

        # Test from_matrix with non-square matrix
        with self.assertRaises(GraphGenerationError):
            GraphFactory.create_predefined_graph(
                structure_type="from_matrix",
                adjacency_matrix=[[0, 1], [0, 1], [1, 0]]
            )

        # Test from_edges with invalid node reference
        with self.assertRaises(GraphGenerationError):
            GraphFactory.create_predefined_graph(
                structure_type="from_edges",
                edge_list=[(0, 1), (1, 5)],
                num_nodes=3
            )

    def test_create_predefined_graph_returns_correct_type(self):
        """Test that create_predefined_graph returns the correct graph type."""
        # Test with is_causal=True
        graph = GraphFactory.create_predefined_graph(
            structure_type="chain",
            num_nodes=5,
            is_causal=True
        )
        self.assertIsInstance(graph, CausalGraph)

        # Test with is_causal=False
        graph = GraphFactory.create_predefined_graph(
            structure_type="chain",
            num_nodes=5,
            is_causal=False
        )
        self.assertIsInstance(graph, DirectedGraph)

    # ---- Tests for create_random_dag ----

    def test_create_random_dag_validates_parameters(self):
        """Test that create_random_dag validates parameters."""
        with self.assertRaises(GraphGenerationError):
            GraphFactory.create_random_dag(num_nodes=0, edge_probability=0.5)
        with self.assertRaises(GraphGenerationError):
            GraphFactory.create_random_dag(num_nodes=10, edge_probability=-0.1)
        with self.assertRaises(GraphGenerationError):
            GraphFactory.create_random_dag(num_nodes=10, edge_probability=1.1)

    def test_create_random_dag_basic_cases(self):
        """Test basic DAG generation with various parameters."""
        test_cases = [
            {"num_nodes": 5, "edge_probability": 0.5, "seed": 1, "is_causal": True, "expected_type": CausalGraph},
            {"num_nodes": 50, "edge_probability": 0.2, "seed": 2, "is_causal": True, "expected_type": CausalGraph},
            {"num_nodes": 1, "edge_probability": 0.5, "seed": 3, "is_causal": True, "expected_type": CausalGraph},
            {"num_nodes": 2, "edge_probability": 0.0, "seed": 4, "is_causal": True, "expected_type": CausalGraph},
            {"num_nodes": 2, "edge_probability": 1.0, "seed": 5, "is_causal": True, "expected_type": CausalGraph},
            {"num_nodes": 10, "edge_probability": 0.3, "seed": 6, "is_causal": False, "expected_type": DirectedGraph},
        ]

        for params in test_cases:
            with self.subTest(params=params):
                num_nodes = params['num_nodes']
                graph = GraphFactory.create_random_dag(
                    num_nodes=num_nodes,
                    edge_probability=params['edge_probability'],
                    seed=params['seed'],
                    is_causal=params['is_causal']
                )

                # 1. Assert Type
                self.assertIsInstance(graph, params['expected_type'])

                # 2. Assert Node Count
                self.assertEqual(graph.num_nodes, num_nodes)

                # 3. Assert Acyclicity using NetworkX
                nx_graph = nx.DiGraph()
                nx_graph.add_nodes_from(graph.get_nodes())
                nx_graph.add_edges_from(graph.get_edges())
                self.assertTrue(nx.is_directed_acyclic_graph(nx_graph), 
                                f"Graph generated with params {params} is not acyclic")

    def test_create_random_dag_reproducibility(self):
        """Test reproducibility of create_random_dag with the same seed."""
        params = {'num_nodes': 20, 'edge_probability': 0.4, 'seed': 42}
        graph1 = GraphFactory.create_random_dag(**params)
        graph2 = GraphFactory.create_random_dag(**params)
        graph3 = GraphFactory.create_random_dag(**params, seed=43)

        self.assertEqual(graph1.get_edges(), graph2.get_edges())
        self.assertNotEqual(graph1.get_edges(), graph3.get_edges())

    # ---- End Tests for create_random_dag ----


if __name__ == "__main__":
    unittest.main()
