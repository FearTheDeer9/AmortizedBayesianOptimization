import unittest
import numpy as np
import networkx as nx

from causal_meta.graph.causal_graph import CausalGraph


class TestCausalGraph(unittest.TestCase):
    """Test cases for CausalGraph class."""

    def setUp(self):
        """Set up a test causal graph for each test case."""
        self.graph = CausalGraph()
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

    def test_get_parents(self):
        """Test that get_parents returns a set of parent nodes."""
        parents = self.graph.get_parents("D")
        self.assertIsInstance(parents, set)
        self.assertEqual(parents, {"C", "A"})

    def test_get_children(self):
        """Test that get_children returns a set of child nodes."""
        children = self.graph.get_children("A")
        self.assertIsInstance(children, set)
        self.assertEqual(children, {"B", "D"})

    def test_get_markov_blanket(self):
        """Test that get_markov_blanket returns correct nodes."""
        # B's Markov blanket should be {A, C}
        mb = self.graph.get_markov_blanket("B")
        self.assertIsInstance(mb, set)
        self.assertEqual(mb, {"A", "C"})

        # C's Markov blanket should be {B, D, A} because A is a parent of D (which is C's child)
        mb = self.graph.get_markov_blanket("C")
        self.assertIsInstance(mb, set)
        self.assertEqual(mb, {"B", "D", "A"})

    def test_predecessors_parents_equivalence(self):
        """Test that get_predecessors and get_parents return the same results."""
        # For each node, verify that predecessors and parents are the same
        for node in self.graph.get_nodes():
            with self.subTest(node=node):
                predecessors = self.graph.get_predecessors(node)
                parents = self.graph.get_parents(node)
                self.assertEqual(predecessors, parents)

    def test_successors_children_equivalence(self):
        """Test that get_successors and get_children return the same results."""
        # For each node, verify that successors and children are the same
        for node in self.graph.get_nodes():
            with self.subTest(node=node):
                successors = self.graph.get_successors(node)
                children = self.graph.get_children(node)
                self.assertEqual(successors, children)

    def test_copy(self):
        """Test that copy creates a proper deep copy."""
        copy_graph = self.graph.copy()
        self.assertIsInstance(copy_graph, CausalGraph)
        # Same nodes and edges
        self.assertEqual(set(copy_graph.get_nodes()), set(self.graph.get_nodes()))
        self.assertEqual(set(copy_graph.get_edges()), set(self.graph.get_edges()))
        # Modifying copy should not affect original
        copy_graph.add_node("E")
        copy_graph.add_edge("D", "E")
        self.assertNotIn("E", self.graph.get_nodes())
        self.assertNotIn(("D", "E"), self.graph.get_edges())

    def test_to_networkx(self):
        """Test that to_networkx creates a proper NetworkX graph."""
        nx_graph = self.graph.to_networkx()
        self.assertIsInstance(nx_graph, nx.DiGraph)
        # Same nodes and edges
        self.assertEqual(set(nx_graph.nodes()), set(self.graph.get_nodes()))
        self.assertEqual(set(nx_graph.edges()), set(self.graph.get_edges()))


if __name__ == "__main__":
    unittest.main() 