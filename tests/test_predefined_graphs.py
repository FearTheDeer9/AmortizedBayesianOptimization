import unittest
import numpy as np
import random

from causal_meta.graph import CausalGraph, DirectedGraph
from causal_meta.graph.generators.factory import GraphFactory
from causal_meta.graph.generators.predefined import PredefinedGraphStructureGenerator


class TestPredefinedGraphs(unittest.TestCase):
    """
    Tests for predefined graph structures.
    """

    def setUp(self):
        # Set a fixed seed for reproducibility
        random.seed(42)
        np.random.seed(42)

    def test_chain_structure(self):
        """Test the chain structure generator."""
        # Create a chain with 5 nodes
        graph = PredefinedGraphStructureGenerator.chain(num_nodes=5)

        # Check that it's a causal graph
        self.assertIsInstance(graph, CausalGraph)

        # Check the number of nodes and edges
        self.assertEqual(graph.num_nodes, 5)
        self.assertEqual(len(graph.get_edges()), 4)

        # Check that the chain structure is correct
        self.assertTrue(graph.has_edge(0, 1))
        self.assertTrue(graph.has_edge(1, 2))
        self.assertTrue(graph.has_edge(2, 3))
        self.assertTrue(graph.has_edge(3, 4))

        # Check with noise
        noise_graph = PredefinedGraphStructureGenerator.chain(
            num_nodes=5, noise_probability=1.0)
        # With noise_probability=1.0, all possible forward edges should be added
        self.assertTrue(noise_graph.has_edge(0, 2))
        self.assertTrue(noise_graph.has_edge(0, 3))
        self.assertTrue(noise_graph.has_edge(0, 4))
        self.assertTrue(noise_graph.has_edge(1, 3))
        self.assertTrue(noise_graph.has_edge(1, 4))
        self.assertTrue(noise_graph.has_edge(2, 4))

    def test_fork_structure(self):
        """Test the fork structure generator."""
        # Create a fork with 5 nodes
        graph = PredefinedGraphStructureGenerator.fork(num_nodes=5)

        # Check that it's a causal graph
        self.assertIsInstance(graph, CausalGraph)

        # Check the number of nodes and edges
        self.assertEqual(graph.num_nodes, 5)
        self.assertEqual(len(graph.get_edges()), 4)

        # Check that the fork structure is correct
        self.assertTrue(graph.has_edge(0, 1))
        self.assertTrue(graph.has_edge(0, 2))
        self.assertTrue(graph.has_edge(0, 3))
        self.assertTrue(graph.has_edge(0, 4))

        # Verify there are no edges between the child nodes
        self.assertFalse(graph.has_edge(1, 2))
        self.assertFalse(graph.has_edge(2, 3))
        self.assertFalse(graph.has_edge(3, 4))

    def test_collider_structure(self):
        """Test the collider structure generator."""
        # Create a collider with 5 nodes
        graph = PredefinedGraphStructureGenerator.collider(num_nodes=5)

        # Check that it's a causal graph
        self.assertIsInstance(graph, CausalGraph)

        # Check the number of nodes and edges
        self.assertEqual(graph.num_nodes, 5)
        self.assertEqual(len(graph.get_edges()), 4)

        # Check that the collider structure is correct
        self.assertTrue(graph.has_edge(1, 0))
        self.assertTrue(graph.has_edge(2, 0))
        self.assertTrue(graph.has_edge(3, 0))
        self.assertTrue(graph.has_edge(4, 0))

        # Verify there are no edges between the parent nodes
        self.assertFalse(graph.has_edge(1, 2))
        self.assertFalse(graph.has_edge(2, 3))
        self.assertFalse(graph.has_edge(3, 4))

    def test_mediator_structure(self):
        """Test the mediator structure generator."""
        # Create a mediator
        graph = PredefinedGraphStructureGenerator.mediator()

        # Check that it's a causal graph
        self.assertIsInstance(graph, CausalGraph)

        # Check the number of nodes and edges
        self.assertEqual(graph.num_nodes, 3)
        self.assertEqual(len(graph.get_edges()), 2)

        # Check that the mediator structure is correct
        self.assertTrue(graph.has_edge(0, 1))  # X → M
        self.assertTrue(graph.has_edge(1, 2))  # M → Y

        # With noise, the direct effect should be added
        noise_graph = PredefinedGraphStructureGenerator.mediator(
            noise_probability=1.0)
        self.assertTrue(noise_graph.has_edge(0, 2))  # X → Y

    def test_complete_structure(self):
        """Test the complete graph structure generator."""
        # Create a complete graph with 5 nodes
        graph = PredefinedGraphStructureGenerator.complete(num_nodes=5)

        # Check that it's a causal graph
        self.assertIsInstance(graph, CausalGraph)

        # Check the number of nodes and edges
        self.assertEqual(graph.num_nodes, 5)
        # For a directed complete graph with n nodes, there should be n*(n-1) edges
        self.assertEqual(len(graph.get_edges()), 5 * (5 - 1))

        # Check that every possible edge exists
        for i in range(5):
            for j in range(5):
                if i != j:
                    self.assertTrue(graph.has_edge(i, j))

        # Test undirected complete graph
        undirected_graph = PredefinedGraphStructureGenerator.complete(
            num_nodes=5, directed=False)
        # For an undirected graph represented as directed, each edge appears twice (i→j and j→i)
        self.assertEqual(len(undirected_graph.get_edges()), 5 * (5 - 1))

        # Test with noise (as anti-noise in this case)
        noise_graph = PredefinedGraphStructureGenerator.complete(
            num_nodes=5, noise_probability=1.0)
        # With noise_probability=1.0, all edges should be removed
        self.assertEqual(len(noise_graph.get_edges()), 0)

    def test_tree_structure(self):
        """Test the tree structure generator."""
        # Create a tree with 7 nodes and branching factor 2
        graph = PredefinedGraphStructureGenerator.tree(
            num_nodes=7, branching_factor=2)

        # Check that it's a causal graph
        self.assertIsInstance(graph, CausalGraph)

        # Check the number of nodes and edges
        self.assertEqual(graph.num_nodes, 7)
        # For a tree with n nodes, there should be n-1 edges
        self.assertEqual(len(graph.get_edges()), 7 - 1)

        # Check that the tree structure is correct
        # Level 1
        self.assertTrue(graph.has_edge(0, 1))
        self.assertTrue(graph.has_edge(0, 2))
        # Level 2
        self.assertTrue(graph.has_edge(1, 3))
        self.assertTrue(graph.has_edge(1, 4))
        self.assertTrue(graph.has_edge(2, 5))
        self.assertTrue(graph.has_edge(2, 6))

        # Test with branching factor 3
        bf3_graph = PredefinedGraphStructureGenerator.tree(
            num_nodes=10, branching_factor=3)
        # Check specific edges
        self.assertTrue(bf3_graph.has_edge(0, 1))
        self.assertTrue(bf3_graph.has_edge(0, 2))
        self.assertTrue(bf3_graph.has_edge(0, 3))
        self.assertTrue(bf3_graph.has_edge(1, 4))
        self.assertTrue(bf3_graph.has_edge(1, 5))
        self.assertTrue(bf3_graph.has_edge(1, 6))

    def test_bipartite_structure(self):
        """Test the bipartite graph structure generator."""
        # Create a bipartite graph with 3 nodes in the first set and 2 in the second
        graph = PredefinedGraphStructureGenerator.bipartite(n1=3, n2=2)

        # Check that it's a causal graph
        self.assertIsInstance(graph, CausalGraph)

        # Check the number of nodes and edges
        self.assertEqual(graph.num_nodes, 5)
        # With cross_probability=1.0 (default), all possible edges should exist
        self.assertEqual(len(graph.get_edges()), 3 * 2)

        # Check that the bipartite structure is correct
        # Edges from first set to second set
        self.assertTrue(graph.has_edge(0, 3))
        self.assertTrue(graph.has_edge(0, 4))
        self.assertTrue(graph.has_edge(1, 3))
        self.assertTrue(graph.has_edge(1, 4))
        self.assertTrue(graph.has_edge(2, 3))
        self.assertTrue(graph.has_edge(2, 4))

        # Verify there are no edges within sets
        self.assertFalse(graph.has_edge(0, 1))
        self.assertFalse(graph.has_edge(1, 2))
        self.assertFalse(graph.has_edge(3, 4))

        # Test with noise
        noise_graph = PredefinedGraphStructureGenerator.bipartite(
            n1=3, n2=2, noise_probability=1.0
        )
        # With noise_probability=1.0, all possible edges within sets should exist
        self.assertTrue(noise_graph.has_edge(0, 1))
        self.assertTrue(noise_graph.has_edge(0, 2))
        self.assertTrue(noise_graph.has_edge(1, 2))
        self.assertTrue(noise_graph.has_edge(3, 4))

    def test_factory_integration(self):
        """Test that the factory correctly uses predefined graph structures."""
        # Test each predefined structure through the factory

        # Chain
        chain = GraphFactory.create_predefined_graph(
            structure_type='chain', num_nodes=5
        )
        self.assertEqual(chain.num_nodes, 5)
        self.assertEqual(len(chain.get_edges()), 4)

        # Fork
        fork = GraphFactory.create_predefined_graph(
            structure_type='fork', num_nodes=5
        )
        self.assertEqual(fork.num_nodes, 5)
        self.assertEqual(len(fork.get_edges()), 4)

        # Complete
        complete = GraphFactory.create_predefined_graph(
            structure_type='complete', num_nodes=5
        )
        self.assertEqual(complete.num_nodes, 5)
        self.assertEqual(len(complete.get_edges()), 5 * (5 - 1))

        # Tree
        tree = GraphFactory.create_predefined_graph(
            structure_type='tree', num_nodes=7, branching_factor=2
        )
        self.assertEqual(tree.num_nodes, 7)
        self.assertEqual(len(tree.get_edges()), 6)

        # Bipartite
        bipartite = GraphFactory.create_predefined_graph(
            structure_type='bipartite', n1=3, n2=2
        )
        self.assertEqual(bipartite.num_nodes, 5)
        self.assertEqual(len(bipartite.get_edges()), 6)


if __name__ == '__main__':
    unittest.main()
