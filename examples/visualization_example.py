"""
Example script demonstrating the graph visualization utilities.

This script creates various types of graphs and visualizes them using the
visualization utilities from the causal_meta library.
"""
from causal_meta.graph.visualization import (
    plot_graph,
    plot_causal_graph,
    plot_graph_adjacency_matrix,
    plot_intervention
)
from causal_meta.graph import CausalGraph, DirectedGraph
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the Python path to allow direct imports
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def main():
    """Run the visualization example."""
    # Create a simple directed graph
    print("Creating a directed graph...")
    directed_graph = create_directed_graph()

    # Create a simple causal graph (fork structure)
    print("Creating a fork causal graph...")
    fork_graph = create_fork_structure()

    # Create a collider causal graph
    print("Creating a collider causal graph...")
    collider_graph = create_collider_structure()

    # Create a more complex causal graph with backdoor paths
    print("Creating a complex causal graph with backdoor paths...")
    backdoor_graph = create_backdoor_graph()

    # Basic visualization examples
    print("Visualizing basic graph structures...")

    # Example 1: Basic directed graph visualization
    plt.figure(figsize=(8, 6))
    plot_graph(directed_graph, title="Directed Graph Example")
    plt.tight_layout()
    plt.savefig("directed_graph_example.png")

    # Example 2: Visualize with highlighted nodes and edges
    plt.figure(figsize=(8, 6))
    highlighted_nodes = {1, 3}
    highlighted_edges = [(0, 1), (3, 4)]
    plot_graph(
        directed_graph,
        highlight_nodes=highlighted_nodes,
        highlight_edges=highlighted_edges,
        title="Directed Graph with Highlights"
    )
    plt.tight_layout()
    plt.savefig("directed_graph_highlights.png")

    # Example 3: Different layout algorithms
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    layouts = ['spring', 'circular', 'kamada_kawai',
               'shell', 'spectral', 'random']

    for i, layout in enumerate(layouts):
        row, col = i // 3, i % 3
        plot_graph(
            directed_graph,
            ax=axes[row, col],
            layout=layout,
            title=f"{layout.capitalize()} Layout"
        )

    plt.tight_layout()
    plt.savefig("layout_algorithms.png")

    # Example 4: Causal graph visualization
    plt.figure(figsize=(8, 6))
    plot_causal_graph(
        fork_graph,
        treatment=0,  # The common cause
        outcome=2,    # One of the effects
        title="Fork Structure (Causal Graph)"
    )
    plt.tight_layout()
    plt.savefig("fork_causal_graph.png")

    # Example 5: Collider with conditioning
    plt.figure(figsize=(8, 6))
    plot_causal_graph(
        collider_graph,
        treatment=0,
        outcome=1,
        conditioning_set={2},  # Conditioning on the collider
        title="Collider with Conditioning"
    )
    plt.tight_layout()
    plt.savefig("collider_conditioning.png")

    # Example 6: Complex graph with backdoor paths
    plt.figure(figsize=(10, 8))
    plot_causal_graph(
        backdoor_graph,
        treatment='X',
        outcome='Y',
        show_backdoor_paths=True,
        title="Causal Graph with Backdoor Paths"
    )
    plt.tight_layout()
    plt.savefig("backdoor_paths.png")

    # Example 7: Adjacency matrix visualization
    plt.figure(figsize=(8, 8))
    plot_graph_adjacency_matrix(
        directed_graph,
        title="Adjacency Matrix Visualization"
    )
    plt.tight_layout()
    plt.savefig("adjacency_matrix.png")

    # Example 8: Intervention visualization
    intervention_node = 'Z'
    intervened_graph = backdoor_graph.do_intervention(intervention_node)

    # Create a figure with two subplots side by side
    plot_intervention(
        backdoor_graph,
        intervened_graph,
        intervention_node,
        figsize=(15, 6)
    )
    plt.tight_layout()
    plt.savefig("intervention_comparison.png")

    print("Visualization examples completed. All images saved to files.")


def create_directed_graph():
    """Create a simple directed graph for visualization."""
    graph = DirectedGraph()

    # Add nodes
    for i in range(5):
        graph.add_node(i)

    # Add edges to create a DAG
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 3)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)

    return graph


def create_fork_structure():
    """Create a causal graph with a fork structure (common cause)."""
    graph = CausalGraph()

    # Add nodes
    for i in range(3):
        graph.add_node(i)

    # Create a fork: 0 -> 1, 0 -> 2
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)

    return graph


def create_collider_structure():
    """Create a causal graph with a collider structure."""
    graph = CausalGraph()

    # Add nodes
    for i in range(3):
        graph.add_node(i)

    # Create a collider: 0 -> 2, 1 -> 2
    graph.add_edge(0, 2)
    graph.add_edge(1, 2)

    return graph


def create_backdoor_graph():
    """Create a causal graph with backdoor paths."""
    graph = CausalGraph()

    # Add nodes
    nodes = ['X', 'Y', 'Z', 'W', 'U']
    for node in nodes:
        graph.add_node(node)

    # X -> Y (direct effect)
    graph.add_edge('X', 'Y')

    # X <- Z -> Y (backdoor path)
    graph.add_edge('Z', 'X')
    graph.add_edge('Z', 'Y')

    # X <- W -> U -> Y (another backdoor path)
    graph.add_edge('W', 'X')
    graph.add_edge('W', 'U')
    graph.add_edge('U', 'Y')

    return graph


if __name__ == "__main__":
    main()
