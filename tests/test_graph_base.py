"""
Tests for the base graph class.
"""
import unittest
from causal_meta.graph.base import Graph


class TestGraph(unittest.TestCase):
    """Tests for the Graph class."""

    def setUp(self):
        """Set up a graph for testing."""
        self.graph = Graph()

        # Add some nodes
        self.graph.add_node(1, name="Node 1")
        self.graph.add_node(2, name="Node 2")
        self.graph.add_node(3, name="Node 3")

        # Add some edges
        self.graph.add_edge(1, 2, weight=0.5)
        self.graph.add_edge(2, 3, weight=0.8)

    def test_has_node(self):
        """Test the has_node method."""
        self.assertTrue(self.graph.has_node(1))
        self.assertTrue(self.graph.has_node(2))
        self.assertTrue(self.graph.has_node(3))
        self.assertFalse(self.graph.has_node(4))

    def test_has_edge(self):
        """Test the has_edge method."""
        self.assertTrue(self.graph.has_edge(1, 2))
        self.assertTrue(self.graph.has_edge(2, 3))
        self.assertFalse(self.graph.has_edge(1, 3))
        self.assertFalse(self.graph.has_edge(3, 1))

    def test_get_nodes(self):
        """Test the get_nodes method."""
        nodes = self.graph.get_nodes()
        self.assertEqual(len(nodes), 3)
        self.assertIn(1, nodes)
        self.assertIn(2, nodes)
        self.assertIn(3, nodes)

    def test_get_edges(self):
        """Test the get_edges method."""
        edges = self.graph.get_edges()
        self.assertEqual(len(edges), 2)
        self.assertIn((1, 2), edges)
        self.assertIn((2, 3), edges)

    def test_get_node_attributes(self):
        """Test the get_node_attributes method."""
        attr = self.graph.get_node_attributes(1)
        self.assertEqual(attr.get("name"), "Node 1")

        attr = self.graph.get_node_attributes(2)
        self.assertEqual(attr.get("name"), "Node 2")

    def test_get_edge_attributes(self):
        """Test the get_edge_attributes method."""
        attr = self.graph.get_edge_attributes(1, 2)
        self.assertEqual(attr.get("weight"), 0.5)

        attr = self.graph.get_edge_attributes(2, 3)
        self.assertEqual(attr.get("weight"), 0.8)

    def test_remove_node(self):
        """Test the remove_node method."""
        self.graph.remove_node(2)

        self.assertFalse(self.graph.has_node(2))
        self.assertFalse(self.graph.has_edge(1, 2))
        self.assertFalse(self.graph.has_edge(2, 3))

        # Verify that node 1 and 3 still exist
        self.assertTrue(self.graph.has_node(1))
        self.assertTrue(self.graph.has_node(3))

    def test_remove_edge(self):
        """Test the remove_edge method."""
        self.graph.remove_edge(1, 2)

        self.assertFalse(self.graph.has_edge(1, 2))
        self.assertTrue(self.graph.has_edge(2, 3))

        # Verify that all nodes still exist
        self.assertTrue(self.graph.has_node(1))
        self.assertTrue(self.graph.has_node(2))
        self.assertTrue(self.graph.has_node(3))

    def test_get_neighbors(self):
        """Test the get_neighbors method."""
        neighbors = self.graph.get_neighbors(1)
        self.assertEqual(len(neighbors), 1)
        self.assertIn(2, neighbors)

        neighbors = self.graph.get_neighbors(2)
        self.assertEqual(len(neighbors), 2)
        self.assertIn(1, neighbors)
        self.assertIn(3, neighbors)

        neighbors = self.graph.get_neighbors(3)
        self.assertEqual(len(neighbors), 1)
        self.assertIn(2, neighbors)


if __name__ == "__main__":
    unittest.main()
