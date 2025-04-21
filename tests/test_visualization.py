"""
Tests for graph visualization module.
"""
import unittest
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numbers import Number

from causal_meta.graph import CausalGraph, DirectedGraph
from causal_meta.graph.visualization import (
    plot_graph,
    plot_causal_graph,
    plot_graph_adjacency_matrix,
    plot_intervention,
    _convert_to_networkx,
    _get_layout
)


class TestVisualization(unittest.TestCase):
    """Tests for graph visualization functions."""

    def setUp(self):
        """Create test graphs for visualization."""
        # Create a simple directed graph
        self.directed_graph = DirectedGraph()
        for i in range(5):
            self.directed_graph.add_node(i)
        self.directed_graph.add_edge(0, 1)
        self.directed_graph.add_edge(0, 2)
        self.directed_graph.add_edge(1, 3)
        self.directed_graph.add_edge(2, 3)
        self.directed_graph.add_edge(3, 4)

        # Create a simple causal graph (fork structure)
        self.causal_graph = CausalGraph()
        for i in range(3):
            self.causal_graph.add_node(i)
        self.causal_graph.add_edge(0, 1)  # 0 -> 1
        self.causal_graph.add_edge(0, 2)  # 0 -> 2

        # Create a slightly more complex causal graph (collider structure)
        self.complex_causal_graph = CausalGraph()
        for i in range(5):
            self.complex_causal_graph.add_node(i)
        self.complex_causal_graph.add_edge(0, 2)  # 0 -> 2
        self.complex_causal_graph.add_edge(1, 2)  # 1 -> 2
        self.complex_causal_graph.add_edge(2, 3)  # 2 -> 3
        self.complex_causal_graph.add_edge(2, 4)  # 2 -> 4

        # Create a causal graph with a backdoor path
        self.backdoor_graph = CausalGraph()
        nodes = ['X', 'Y', 'Z', 'W']
        for node in nodes:
            self.backdoor_graph.add_node(node)
        # X -> Y (direct effect)
        self.backdoor_graph.add_edge('X', 'Y')
        # X <- Z -> Y (backdoor path)
        self.backdoor_graph.add_edge('Z', 'X')
        self.backdoor_graph.add_edge('Z', 'Y')
        # W -> X (another path)
        self.backdoor_graph.add_edge('W', 'X')

    def test_convert_to_networkx(self):
        """Test conversion of DirectedGraph to networkx DiGraph."""
        # Convert the directed graph to networkx
        nx_graph = _convert_to_networkx(self.directed_graph)

        # Verify node count
        self.assertEqual(len(nx_graph.nodes), len(self.directed_graph._nodes))

        # Verify edge count
        self.assertEqual(len(nx_graph.edges), len(self.directed_graph._edges))

        # Verify nodes and edges
        for node in self.directed_graph._nodes:
            self.assertIn(node, nx_graph.nodes)

        for edge in self.directed_graph._edges:
            self.assertIn(edge, nx_graph.edges)

    def test_get_layout(self):
        """Test getting layout for a networkx graph."""
        # Convert the directed graph to networkx
        nx_graph = _convert_to_networkx(self.directed_graph)

        # Test different layouts
        layouts = ['spring', 'circular', 'kamada_kawai',
                   'random', 'shell', 'spectral']

        for layout in layouts:
            pos = _get_layout(nx_graph, layout)

            # Verify that positions are returned for all nodes
            self.assertEqual(len(pos), len(nx_graph.nodes))

            # Verify that positions are 2D coordinates
            for node, coords in pos.items():
                self.assertEqual(len(coords), 2)
                self.assertTrue(isinstance(
                    coords[0], Number), f"Expected numeric type, got {type(coords[0])}")
                self.assertTrue(isinstance(
                    coords[1], Number), f"Expected numeric type, got {type(coords[1])}")

    def test_plot_graph(self):
        """Test basic graph plotting."""
        # Test plotting with default parameters
        fig, ax = plt.subplots()
        result_ax = plot_graph(self.directed_graph, ax=ax)

        # The function should return the axes object
        self.assertEqual(result_ax, ax)

        # Test highlighting nodes and edges
        highlight_nodes = {1, 3}
        highlight_edges = [(0, 1), (3, 4)]

        fig, ax = plt.subplots()
        result_ax = plot_graph(
            self.directed_graph,
            ax=ax,
            highlight_nodes=highlight_nodes,
            highlight_edges=highlight_edges
        )

        # The function should return the axes object
        self.assertEqual(result_ax, ax)

        # Close all figures to avoid memory leaks
        plt.close('all')

    def test_plot_causal_graph(self):
        """Test causal graph plotting with treatment and outcome."""
        # Test plotting with treatment and outcome
        treatment = 0
        outcome = 2

        fig, ax = plt.subplots()
        result_ax = plot_causal_graph(
            self.causal_graph,
            treatment=treatment,
            outcome=outcome,
            ax=ax
        )

        # The function should return the axes object
        self.assertEqual(result_ax, ax)

        # Test plotting with conditioning set
        conditioning_set = {1}

        fig, ax = plt.subplots()
        result_ax = plot_causal_graph(
            self.complex_causal_graph,
            treatment=0,
            outcome=4,
            conditioning_set=conditioning_set,
            ax=ax
        )

        # The function should return the axes object
        self.assertEqual(result_ax, ax)

        # Test plotting with backdoor paths
        fig, ax = plt.subplots()
        result_ax = plot_causal_graph(
            self.backdoor_graph,
            treatment='X',
            outcome='Y',
            show_backdoor_paths=True,
            ax=ax
        )

        # The function should return the axes object
        self.assertEqual(result_ax, ax)

        # Close all figures to avoid memory leaks
        plt.close('all')

    def test_plot_graph_adjacency_matrix(self):
        """Test plotting the adjacency matrix of a graph."""
        # Test plotting adjacency matrix with default parameters
        fig, ax = plt.subplots()
        result_ax = plot_graph_adjacency_matrix(self.directed_graph, ax=ax)

        # The function should return the axes object
        self.assertEqual(result_ax, ax)

        # Test with custom node ordering
        node_order = [4, 3, 2, 1, 0]  # Reverse order

        fig, ax = plt.subplots()
        result_ax = plot_graph_adjacency_matrix(
            self.directed_graph,
            ax=ax,
            node_order=node_order,
            show_colorbar=False
        )

        # The function should return the axes object
        self.assertEqual(result_ax, ax)

        # Close all figures to avoid memory leaks
        plt.close('all')

    def test_plot_intervention(self):
        """Test plotting a before-and-after comparison of a graph intervention."""
        # Create an intervened graph by applying do(Z=z) to the backdoor graph
        intervention_node = 'Z'
        intervened_graph = self.backdoor_graph.do_intervention(
            intervention_node)

        # Test plotting intervention with default parameters
        ax1, ax2 = plot_intervention(
            self.backdoor_graph,
            intervened_graph,
            intervention_node
        )

        # The function should return two axes objects
        self.assertIsInstance(ax1, plt.Axes)
        self.assertIsInstance(ax2, plt.Axes)

        # Close all figures to avoid memory leaks
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
