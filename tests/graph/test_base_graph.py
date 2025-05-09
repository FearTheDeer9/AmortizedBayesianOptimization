import unittest

from causal_meta.graph.base import Graph


class TestGraph(unittest.TestCase):
    """Test cases for the base Graph class."""

    def setUp(self):
        """Set up a test graph for each test case."""
        self.graph = Graph()
        # Create a simple graph
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

    def test_get_neighbors(self):
        """Test that get_neighbors returns a set of neighbor nodes."""
        neighbors = self.graph.get_neighbors("A")
        self.assertIsInstance(neighbors, set)
        self.assertEqual(neighbors, {"B", "D"})

    def test_has_node(self):
        """Test that has_node correctly checks node existence."""
        self.assertTrue(self.graph.has_node("A"))
        self.assertFalse(self.graph.has_node("Z"))

    def test_has_edge(self):
        """Test that has_edge correctly checks edge existence."""
        self.assertTrue(self.graph.has_edge("A", "B"))
        self.assertFalse(self.graph.has_edge("A", "C"))
        self.assertFalse(self.graph.has_edge("Z", "A"))

    def test_add_node_with_attributes(self):
        """Test adding a node with attributes."""
        self.graph.add_node("E", color="red", weight=10)
        self.assertTrue(self.graph.has_node("E"))
        attrs = self.graph.get_node_attributes("E")
        self.assertEqual(attrs, {"color": "red", "weight": 10})

    def test_add_edge_with_attributes(self):
        """Test adding an edge with attributes."""
        self.graph.add_edge("A", "C", weight=5, type="direct")
        self.assertTrue(self.graph.has_edge("A", "C"))
        attrs = self.graph.get_edge_attributes("A", "C")
        self.assertEqual(attrs, {"weight": 5, "type": "direct"})

    def test_remove_node(self):
        """Test removing a node and its associated edges."""
        self.graph.remove_node("B")
        self.assertFalse(self.graph.has_node("B"))
        self.assertFalse(self.graph.has_edge("A", "B"))
        self.assertFalse(self.graph.has_edge("B", "C"))
        # Other edges should still exist
        self.assertTrue(self.graph.has_edge("C", "D"))
        self.assertTrue(self.graph.has_edge("A", "D"))

    def test_remove_edge(self):
        """Test removing an edge."""
        self.graph.remove_edge("A", "B")
        self.assertFalse(self.graph.has_edge("A", "B"))
        # Node should still exist
        self.assertTrue(self.graph.has_node("A"))
        self.assertTrue(self.graph.has_node("B"))
        # Other edges should still exist
        self.assertTrue(self.graph.has_edge("B", "C"))


if __name__ == "__main__":
    unittest.main() 