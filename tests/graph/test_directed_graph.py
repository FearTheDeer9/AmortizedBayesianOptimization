import unittest
import numpy as np

from causal_meta.graph.directed_graph import DirectedGraph


class TestDirectedGraph(unittest.TestCase):
    """Test cases for DirectedGraph class."""

    def setUp(self):
        """Set up a test graph for each test case."""
        self.graph = DirectedGraph()
        # Create a simple directed graph
        self.graph.add_node("A")
        self.graph.add_node("B")
        self.graph.add_node("C")
        self.graph.add_node("D")
        self.graph.add_edge("A", "B")
        self.graph.add_edge("B", "C")
        self.graph.add_edge("C", "D")
        self.graph.add_edge("A", "D")

    def test_get_nodes(self):
        """Test that get_nodes returns a list of nodes."""
        nodes = self.graph.get_nodes()
        self.assertIsInstance(nodes, list)
        self.assertEqual(set(nodes), {"A", "B", "C", "D"})

    def test_get_edges(self):
        """Test that get_edges returns a list of edges."""
        edges = self.graph.get_edges()
        self.assertIsInstance(edges, list)
        self.assertEqual(set(edges), {("A", "B"), ("B", "C"), ("C", "D"), ("A", "D")})

    def test_get_successors(self):
        """Test that get_successors returns a set of successor nodes."""
        successors = self.graph.get_successors("A")
        self.assertIsInstance(successors, set)
        self.assertEqual(successors, {"B", "D"})

    def test_get_predecessors(self):
        """Test that get_predecessors returns a set of predecessor nodes."""
        predecessors = self.graph.get_predecessors("D")
        self.assertIsInstance(predecessors, set)
        self.assertEqual(predecessors, {"C", "A"})

    def test_get_sources(self):
        """Test that get_sources returns a set of source nodes."""
        sources = self.graph.get_sources()
        self.assertIsInstance(sources, set)
        self.assertEqual(sources, {"A"})

    def test_get_sinks(self):
        """Test that get_sinks returns a set of sink nodes."""
        sinks = self.graph.get_sinks()
        self.assertIsInstance(sinks, set)
        self.assertEqual(sinks, {"D"})

    def test_get_ancestors(self):
        """Test that get_ancestors returns a set of ancestor nodes."""
        ancestors = self.graph.get_ancestors("D")
        self.assertIsInstance(ancestors, set)
        self.assertEqual(ancestors, {"A", "B", "C"})

    def test_get_descendants(self):
        """Test that get_descendants returns a set of descendant nodes."""
        descendants = self.graph.get_descendants("A")
        self.assertIsInstance(descendants, set)
        self.assertEqual(descendants, {"B", "C", "D"})

    def test_get_adjacency_matrix(self):
        """Test that get_adjacency_matrix returns correct adjacency matrix."""
        adj_matrix = self.graph.get_adjacency_matrix()
        self.assertIsInstance(adj_matrix, np.ndarray)
        self.assertEqual(adj_matrix.shape, (4, 4))
        # Convert node labels to indices for validation
        nodes = self.graph.get_nodes()
        idx = {node: i for i, node in enumerate(nodes)}
        # Check specific connections
        self.assertEqual(adj_matrix[idx["A"], idx["B"]], 1)
        self.assertEqual(adj_matrix[idx["B"], idx["C"]], 1)
        self.assertEqual(adj_matrix[idx["C"], idx["D"]], 1)
        self.assertEqual(adj_matrix[idx["A"], idx["D"]], 1)
        # Check non-connections
        self.assertEqual(adj_matrix[idx["B"], idx["A"]], 0)
        self.assertEqual(adj_matrix[idx["D"], idx["A"]], 0)


if __name__ == "__main__":
    unittest.main() 