"""
Tests for the directed graph implementation.
"""
import unittest
import numpy as np
from causal_meta.graph.directed_graph import DirectedGraph


class TestDirectedGraph(unittest.TestCase):
    """Tests for the DirectedGraph class."""

    def setUp(self):
        """Set up a directed graph for testing."""
        self.graph = DirectedGraph()

        # Add some nodes
        self.graph.add_node(1, name="Node 1")
        self.graph.add_node(2, name="Node 2")
        self.graph.add_node(3, name="Node 3")
        self.graph.add_node(4, name="Node 4")
        self.graph.add_node(5, name="Node 5")

        # Add some edges
        self.graph.add_edge(1, 2, weight=0.5)
        self.graph.add_edge(2, 3, weight=0.8)
        self.graph.add_edge(3, 4, weight=0.6)
        self.graph.add_edge(1, 3, weight=0.7)
        self.graph.add_edge(4, 5, weight=0.9)

    def test_get_successors(self):
        """Test the get_successors method."""
        successors = self.graph.get_successors(1)
        self.assertEqual(len(successors), 2)
        self.assertIn(2, successors)
        self.assertIn(3, successors)

        successors = self.graph.get_successors(2)
        self.assertEqual(len(successors), 1)
        self.assertIn(3, successors)

        successors = self.graph.get_successors(5)
        self.assertEqual(len(successors), 0)

    def test_get_predecessors(self):
        """Test the get_predecessors method."""
        predecessors = self.graph.get_predecessors(3)
        self.assertEqual(len(predecessors), 2)
        self.assertIn(1, predecessors)
        self.assertIn(2, predecessors)

        predecessors = self.graph.get_predecessors(1)
        self.assertEqual(len(predecessors), 0)

        predecessors = self.graph.get_predecessors(5)
        self.assertEqual(len(predecessors), 1)
        self.assertIn(4, predecessors)

    def test_get_adjacency_matrix(self):
        """Test the get_adjacency_matrix method."""
        adj_matrix = self.graph.get_adjacency_matrix()

        # Expected adjacency matrix
        expected = np.zeros((5, 5), dtype=int)
        expected[0, 1] = 1  # Edge 1->2
        expected[1, 2] = 1  # Edge 2->3
        expected[2, 3] = 1  # Edge 3->4
        expected[0, 2] = 1  # Edge 1->3
        expected[3, 4] = 1  # Edge 4->5

        np.testing.assert_array_equal(adj_matrix, expected)

    def test_has_path(self):
        """Test the has_path method."""
        self.assertTrue(self.graph.has_path(1, 5))
        self.assertTrue(self.graph.has_path(1, 4))
        self.assertTrue(self.graph.has_path(2, 5))
        self.assertFalse(self.graph.has_path(5, 1))
        self.assertFalse(self.graph.has_path(4, 2))

    def test_get_all_paths(self):
        """Test the get_all_paths method."""
        paths = self.graph.get_all_paths(1, 4)
        expected_paths = [[1, 2, 3, 4], [1, 3, 4]]

        # Check that the number of paths is correct
        self.assertEqual(len(paths), len(expected_paths))

        # Check that each expected path is in the result
        for path in expected_paths:
            self.assertIn(path, paths)

    def test_has_cycle(self):
        """Test the has_cycle method."""
        # The original graph has no cycles
        self.assertFalse(self.graph.has_cycle())

        # Add an edge to create a cycle
        self.graph.add_edge(5, 1)

        # Now the graph has a cycle
        self.assertTrue(self.graph.has_cycle())

    def test_get_cycles(self):
        """Test the get_cycles method."""
        # Add edges to create cycles
        self.graph.add_edge(5, 1)

        cycles = self.graph.get_cycles()

        # Print cycles for debugging
        print(f"Found cycles: {cycles}")

        # There should be cycles
        self.assertTrue(len(cycles) > 0)

        # Check if one of the expected cycles is in the result
        # One cycle should be 1->2->3->4->5->1
        expected_cycle = [1, 2, 3, 4, 5, 1]

        # Simply check if the exact expected cycle is in the results
        self.assertIn(expected_cycle, cycles)

    def test_topological_sort(self):
        """Test the topological_sort method."""
        # The original graph is a DAG
        topo_order = self.graph.topological_sort()

        # Check that the order is valid
        self.assertEqual(len(topo_order), 5)

        # Check that each node comes before its successors in the order
        for i in range(len(topo_order)):
            node = topo_order[i]
            for successor in self.graph.get_successors(node):
                # Get the index of the successor in the order
                successor_idx = topo_order.index(successor)
                # Verify that the successor comes after the node
                self.assertGreater(successor_idx, i)

        # Adding a cycle should make topological sort raise an error
        self.graph.add_edge(5, 1)
        with self.assertRaises(ValueError):
            self.graph.topological_sort()

    def test_get_sources_and_sinks(self):
        """Test the get_sources and get_sinks methods."""
        sources = self.graph.get_sources()
        self.assertEqual(len(sources), 1)
        self.assertIn(1, sources)

        sinks = self.graph.get_sinks()
        self.assertEqual(len(sinks), 1)
        self.assertIn(5, sinks)

    def test_get_ancestors_and_descendants(self):
        """Test the get_ancestors and get_descendants methods."""
        ancestors = self.graph.get_ancestors(4)
        self.assertEqual(len(ancestors), 3)
        self.assertIn(1, ancestors)
        self.assertIn(2, ancestors)
        self.assertIn(3, ancestors)

        descendants = self.graph.get_descendants(2)
        self.assertEqual(len(descendants), 3)
        self.assertIn(3, descendants)
        self.assertIn(4, descendants)
        self.assertIn(5, descendants)

    def test_is_dag(self):
        """Test the is_dag method."""
        # The original graph is a DAG
        self.assertTrue(self.graph.is_dag())

        # Add an edge to create a cycle
        self.graph.add_edge(5, 1)

        # Now the graph is not a DAG
        self.assertFalse(self.graph.is_dag())


if __name__ == "__main__":
    unittest.main()
