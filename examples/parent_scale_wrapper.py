#!/usr/bin/env python3
"""
Simplified wrapper for using the PARENT_SCALE algorithm.
"""

# fmt: off
# Set up path FIRST - before any custom imports
import sys
import os

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Now import project-specific modules
from graphs import PARENT_SCALE, CausalEnvironmentAdapter
from graphs.graph_generators import generate_erdos_renyi, generate_scale_free, generate_small_world
from graphs.scm_generators import (
    generate_linear_scm,
    generate_polynomial_scm,
    generate_mixed_scm,
    sample_observational,
    sample_interventional
)

# Standard library imports
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
# fmt: on


def run_parent_scale(
    graph=None,
    scm=None,
    num_nodes=5,
    edge_prob=0.3,
    mechanism_type="mixed",
    n_obs=1000,
    n_int=10,
    seed=42,
    nonlinear=True,
    scale_data=True,
):
    """
    A simplified interface for running PARENT_SCALE.

    Args:
        graph: An existing graph to use (if None, a new one will be generated)
        scm: An existing SCM to use (if None, a new one will be generated)
        num_nodes: Number of nodes if generating a new graph
        edge_prob: Edge probability if generating a new Erdos-Renyi graph
        mechanism_type: Type of mechanisms to use ('mixed', 'linear', 'polynomial')
        n_obs: Number of observational samples
        n_int: Number of intervention values per node
        seed: Random seed
        nonlinear: Whether to use nonlinear mechanism models
        scale_data: Whether to scale the data

    Returns:
        Dictionary containing algorithm results
    """
    # Create a graph if not provided
    if graph is None:
        print(
            f"Creating Erdos-Renyi graph with {num_nodes} nodes and edge probability {edge_prob}")
        graph = generate_erdos_renyi(num_nodes, edge_prob=edge_prob, seed=seed)

    # Create SCM if not provided
    if scm is None:
        print(f"Creating {mechanism_type} SCM")
        if mechanism_type == "mixed":
            mechanism_map = {
                str(i): ["linear", "polynomial", "exponential", "sinusoidal"][i % 4]
                for i in range(num_nodes)
            }
            scm = generate_mixed_scm(
                graph, mechanism_map=mechanism_map, seed=seed)
        elif mechanism_type == "linear":
            scm = generate_linear_scm(graph, seed=seed)
        elif mechanism_type == "polynomial":
            scm = generate_polynomial_scm(graph, max_degree=3, seed=seed)
        else:
            raise ValueError(f"Unknown mechanism type: {mechanism_type}")

    # Generate data
    print(
        f"Generating {n_obs} observational samples and {n_int} intervention values per node")
    obs_data = sample_observational(scm, n_obs, seed=seed)

    # Generate interventional data
    int_data = {}

    # Get all nodes except the target node (assumed to be the last node)
    nodes = list(graph.nodes())
    target_node = nodes[-1]
    intervention_nodes = [n for n in nodes if n != target_node]

    # Create interventional data for each node
    for node in intervention_nodes:
        # Choose intervention values
        values = np.linspace(-2, 2, n_int)

        for value in values:
            int_samples = sample_interventional(
                scm, node, value, n_samples=100, seed=seed, graph=graph
            )
            int_key = (node,)
            if int_key not in int_data:
                int_data[int_key] = {}
            int_data[int_key][value] = int_samples

    # Define exploration set (all single-node interventions)
    exploration_set = [(node,) for node in intervention_nodes]

    # Create the PARENT_SCALE algorithm instance
    print("Creating PARENT_SCALE algorithm")
    ps = PARENT_SCALE(
        graph=graph,
        nonlinear=nonlinear,
        causal_prior=True,
        noiseless=False,
        scale_data=scale_data,
        seed=seed
    )

    # Set the data and exploration set
    ps.set_values(obs_data, int_data, exploration_set)

    # Run the algorithm
    print("Running PARENT_SCALE algorithm")
    result = ps.run_algorithm(T=10)

    # Plot the original graph
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(graph, seed=seed)
    nx.draw(
        graph, pos, with_labels=True,
        node_color='skyblue', node_size=800,
        font_weight='bold', arrowsize=20, width=2
    )
    plt.title("Causal Graph")
    plt.savefig("parent_scale_graph.png")
    plt.close()

    print("\nPosterior probabilities for parent sets:")
    for parents, prob in ps.prior_probabilities.items():
        print(f"Parents {parents}: {prob:.4f}")

    print("\nCheck parent_scale_graph.png for the causal graph visualization.")

    return {
        "graph": graph,
        "scm": scm,
        "probabilities": ps.prior_probabilities,
        "parent_scale": ps
    }


if __name__ == "__main__":
    # Example usage
    run_parent_scale(num_nodes=5, edge_prob=0.3,
                     mechanism_type="mixed", seed=42)
