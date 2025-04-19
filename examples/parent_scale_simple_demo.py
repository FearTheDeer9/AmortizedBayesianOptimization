#!/usr/bin/env python3
"""
Simple demonstration of the PARENT_SCALE algorithm using the new graphs framework.
"""

# fmt: off
# Set up path FIRST - before any custom imports
import sys
import os
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import OrderedDict

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Now import project-specific modules
from graphs import PARENT_SCALE, CausalEnvironmentAdapter
from graphs.graph_generators import generate_erdos_renyi, generate_scale_free
from graphs.scm_generators import (
    generate_linear_scm,
    generate_mixed_scm,
    sample_observational,
    sample_interventional
)
# fmt: on

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("parent_scale_demo")


def run_simple_demo():
    """Run a simple PARENT_SCALE demonstration."""
    print("\n=== PARENT_SCALE Simple Demo ===")

    # Create a simple graph
    num_nodes = 5
    print(f"Creating Erdos-Renyi graph with {num_nodes} nodes...")
    graph = generate_erdos_renyi(num_nodes, edge_prob=0.3, seed=42)

    # Print node names
    nodes = list(graph.nodes())
    print(f"Graph nodes: {nodes}")

    # Define target node (last node by default)
    target_node = nodes[-1]
    print(f"Target node: {target_node}")

    # Plot the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph, seed=42)

    # Draw all nodes except target
    non_target_nodes = [n for n in nodes if n != target_node]
    nx.draw_networkx_nodes(
        graph, pos,
        nodelist=non_target_nodes,
        node_color='skyblue',
        node_size=800
    )

    # Draw target node with different color
    nx.draw_networkx_nodes(
        graph, pos,
        nodelist=[target_node],
        node_color='red',
        node_size=800
    )

    # Draw edges and labels
    nx.draw_networkx_edges(graph, pos, width=2, arrowsize=20)
    nx.draw_networkx_labels(graph, pos, font_weight='bold')

    plt.title("Causal Graph", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("parent_scale_simple_graph.png")
    plt.close()

    # Generate SCM
    print("Generating mixed mechanism SCM...")
    mechanism_map = {
        str(i): ["linear", "polynomial", "exponential", "sinusoidal"][i % 4]
        for i in range(num_nodes)
    }
    scm = generate_mixed_scm(graph, mechanism_map=mechanism_map, seed=42)

    # Generate data
    print("Generating observational data...")
    n_obs = 500
    obs_data = sample_observational(scm, n_obs, seed=42)

    # Generate interventional data
    print("Generating interventional data...")
    n_int = 5
    int_data = {}

    # Get intervention nodes (all except target)
    intervention_nodes = [n for n in nodes if n != target_node]
    print(f"Intervention nodes: {intervention_nodes}")

    # Sample interventions
    for node in intervention_nodes:
        # Choose intervention values
        values = np.linspace(-2, 2, n_int)

        for value in values:
            int_samples = sample_interventional(
                scm, node, value, n_samples=100, seed=42, graph=graph
            )
            int_key = (node,)
            if int_key not in int_data:
                int_data[int_key] = {}
            int_data[int_key][value] = int_samples

    # Define exploration set (single-node interventions)
    exploration_set = [(node,) for node in intervention_nodes]

    # Create PARENT_SCALE instance
    print("Creating PARENT_SCALE algorithm...")
    ps = PARENT_SCALE(
        graph=graph,
        nonlinear=True,
        causal_prior=True,
        noiseless=False,
        scale_data=True,
        seed=42
    )

    # Set data
    ps.set_values(obs_data, int_data, exploration_set)

    # Run the algorithm
    print("Running PARENT_SCALE algorithm...")
    result = ps.run_algorithm(T=10)

    # Print results
    print("\nPosterior probabilities for parent sets:")
    for parents, prob in ps.prior_probabilities.items():
        print(f"Parents {parents}: {prob:.4f}")

    # Check true parents
    true_parents = {}
    for node in graph.nodes():
        if node == target_node:
            true_parents[node] = list(graph.predecessors(node))

    print("\nTrue parent sets:")
    for node, parents in true_parents.items():
        print(f"Node {node} has parents: {parents}")

    print("\nDemo completed! Check parent_scale_simple_graph.png for visualization.")


if __name__ == "__main__":
    run_simple_demo()
