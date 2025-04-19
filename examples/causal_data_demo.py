#!/usr/bin/env python3
"""
Comprehensive demo script for the causal data generation framework.
Demonstrates graph generation, SCM mechanisms, interventions, and dataset utilities.
"""

# fmt: off
# Set up path FIRST - before any custom imports
import sys
import os

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Now import project-specific modules
from graphs.graph_generators import (
    generate_erdos_renyi,
    generate_scale_free,
    generate_small_world
)
from graphs.scm_generators import (
    NoiseConfig,
    generate_linear_scm,
    generate_polynomial_scm,
    generate_exponential_scm,
    generate_sinusoidal_scm,
    generate_mixed_scm,
    sample_observational,
    sample_interventional
)
from diffcbed.envs.causal_dataset import CausalDataset

# Standard library imports
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
# fmt: on


def plot_graph(G, title=None):
    """Plot a directed graph with clear node labels."""
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color='skyblue',
            node_size=800, arrowsize=20, width=2,
            font_weight='bold', font_size=14)
    plt.title(title or "Causal Graph Structure", fontsize=16)
    plt.tight_layout()


def plot_node_distributions(samples, title="Node Distributions", figsize=(15, 10)):
    """Plot distribution of all nodes in the samples."""
    num_nodes = len(samples)
    rows = (num_nodes + 2) // 3  # 3 columns
    fig, axes = plt.subplots(nrows=rows, ncols=3, figsize=figsize)
    axes = axes.flatten()

    for i, (node, data) in enumerate(sorted(samples.items(), key=lambda x: int(x[0]))):
        ax = axes[i]
        ax.hist(data, bins=30, alpha=0.7)
        ax.set_title(f"Node {node}")
        ax.grid(alpha=0.3)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)


def plot_parent_child_relationships(G, samples, title="Parent-Child Relationships", figsize=(15, 10)):
    """Plot scatter plots showing relationships between parent-child node pairs."""
    # Find all parent-child pairs
    edges = list(G.edges())

    if not edges:
        print("No edges found in graph to plot parent-child relationships")
        return

    # Create subplots
    rows = (len(edges) + 2) // 3  # 3 columns
    fig, axes = plt.subplots(nrows=rows, ncols=3, figsize=figsize)
    axes = axes.flatten()

    for i, (parent, child) in enumerate(edges):
        if i >= len(axes):
            break

        ax = axes[i]
        parent_str, child_str = str(parent), str(child)

        # Plot scatter with parent values on x-axis and child values on y-axis
        ax.scatter(samples[parent_str], samples[child_str], alpha=0.5, s=20)
        ax.set_xlabel(f"Node {parent_str} (Parent)")
        ax.set_ylabel(f"Node {child_str} (Child)")
        ax.set_title(f"{parent_str} → {child_str}")
        ax.grid(alpha=0.3)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)


def plot_intervention_comparison(obs_samples, int_samples, node_idx, int_value, figsize=(15, 10)):
    """Plot comparison between observational and interventional distributions."""
    # Determine how many nodes to plot
    num_nodes = len(obs_samples)
    rows = (num_nodes + 2) // 3  # 3 columns
    fig, axes = plt.subplots(nrows=rows, ncols=3, figsize=figsize)
    axes = axes.flatten()

    for i, node in enumerate(sorted(obs_samples.keys(), key=lambda x: int(x))):
        if i >= len(axes):
            break

        ax = axes[i]
        ax.hist(obs_samples[node], bins=20, alpha=0.6,
                label='Observational', color='blue')
        ax.hist(int_samples[node], bins=20, alpha=0.6,
                label='Interventional', color='red')

        # Add vertical line for intervention
        if node == str(node_idx):
            ax.axvline(x=int_value, color='red', linestyle='--', linewidth=2,
                       label=f'Intervention value = {int_value}')
            ax.set_title(f"Node {node} (Intervened)")
        else:
            ax.set_title(f"Node {node}")

        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize='small')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Effect of Intervention on Node {node_idx} = {int_value}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)


def plot_intervention_parent_child(graph, scm, node_to_intervene, int_value, seed=42, figsize=(15, 10)):
    """
    Plot parent-child relationships before and after intervention to clearly 
    show the effect of different mechanisms.
    """
    # Sample observational data
    obs_samples = sample_observational(scm, n_samples=1000, seed=seed)

    # Sample interventional data using the CORRECT method with graph parameter
    int_samples = sample_interventional(
        scm,
        node=node_to_intervene,
        value=int_value,
        n_samples=1000,
        seed=seed,
        graph=graph  # Pass the graph for correct causal structure
    )

    # Get children of the intervened node
    node_str = str(node_to_intervene)
    children = list(graph.successors(node_to_intervene))

    # If there are no children, there's nothing to show
    if not children:
        print(
            f"Node {node_to_intervene} has no children to show intervention effects.")
        return

    # Create a figure
    fig, axes = plt.subplots(len(children), 2, figsize=figsize)

    # Handle case with only one child
    if len(children) == 1:
        axes = np.array([axes])

    # For each child, plot parent-child relationship before and after intervention
    for i, child in enumerate(children):
        child_str = str(child)

        # Observational (before intervention)
        ax1 = axes[i, 0]
        ax1.scatter(obs_samples[node_str], obs_samples[child_str], alpha=0.5)
        ax1.set_title(f"Observational: {node_str} → {child_str}")
        ax1.set_xlabel(f"Node {node_str}")
        ax1.set_ylabel(f"Node {child_str}")
        ax1.grid(True, alpha=0.3)

        # Interventional (after intervention)
        ax2 = axes[i, 1]
        ax2.scatter(int_samples[node_str], int_samples[child_str], alpha=0.5)
        ax2.set_title(
            f"Intervention do({node_str}={int_value}): {node_str} → {child_str}")
        ax2.set_xlabel(f"Node {node_str}")
        ax2.set_ylabel(f"Node {child_str}")
        ax2.axvline(x=int_value, color='red', linestyle='--', linewidth=2)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle(
        f"Intervention Effects on Children of Node {node_to_intervene}", fontsize=16)


def demo_graph_types(seed=42):
    """Demonstrate different graph types."""
    print("\n=== Demonstrating Graph Types ===")

    # Generate different graph types
    er_graph = generate_erdos_renyi(5, edge_prob=0.3, seed=seed)
    sf_graph = generate_scale_free(5, attachment_param=1.0, seed=seed)
    sw_graph = generate_small_world(
        10, k_neighbors=2, rewire_prob=0.2, seed=seed)

    print(
        f"Erdos-Renyi: {len(er_graph.nodes())} nodes, {len(er_graph.edges())} edges")
    print(
        f"Scale-Free: {len(sf_graph.nodes())} nodes, {len(sf_graph.edges())} edges")
    print(
        f"Small-World: {len(sw_graph.nodes())} nodes, {len(sw_graph.edges())} edges")

    # Plot graphs
    plt.figure(figsize=(18, 6))

    plt.subplot(131)
    plot_graph(er_graph, "Erdos-Renyi")

    plt.subplot(132)
    plot_graph(sf_graph, "Scale-Free")

    plt.subplot(133)
    plot_graph(sw_graph, "Small-World")

    plt.tight_layout()
    plt.savefig("plots/graph_types.png")
    plt.close()

    return sw_graph  # Return the small-world graph for further demos


def demo_mechanism_diversity(graph, seed=42):
    """Demonstrate different mechanism types."""
    print("\n=== Demonstrating Mechanism Diversity ===")

    print(
        f"Graph has {len(graph.nodes())} nodes and {len(graph.edges())} edges")

    # Generate different types of SCMs
    mechanisms = {
        "Linear": generate_linear_scm(graph, seed=seed),
        "Polynomial": generate_polynomial_scm(graph, max_degree=3, seed=seed),
        "Exponential": generate_exponential_scm(graph, seed=seed),
        "Sinusoidal": generate_sinusoidal_scm(graph, seed=seed),
    }

    # Also create a mixed SCM with different mechanism types
    mechanism_map = {
        str(i): ["linear", "polynomial", "exponential", "sinusoidal"][i % 4] for i in range(10)
    }
    mixed_scm = generate_mixed_scm(
        graph, mechanism_map=mechanism_map, seed=seed)
    mechanisms["Mixed"] = mixed_scm

    # Sample from each SCM and plot both distributions and relationships
    for name, scm in mechanisms.items():
        print(f"- Generating {name} mechanism samples")
        samples = sample_observational(scm, n_samples=1000, seed=seed)

        # Plot distributions
        plot_node_distributions(
            samples, title=f"{name} Mechanism - Node Distributions")
        plt.savefig(f"plots/mechanism_{name.lower()}_distributions.png")
        plt.close()

        # Plot parent-child relationships
        plot_parent_child_relationships(
            graph, samples, title=f"{name} Mechanism - Causal Relationships")
        plt.savefig(f"plots/mechanism_{name.lower()}_relationships.png")
        plt.close()

    # Plot the graph structure
    plot_graph(graph)
    plt.savefig("plots/causal_graph.png")
    plt.close()

    return mechanisms, mechanism_map


def demo_intervention_effects(graph, mechanisms, seed=42):
    """Demonstrate intervention effects with different mechanisms."""
    print("\n=== Demonstrating Intervention Effects ===")

    # Focus on a node with children to show intervention effects
    try:
        # Find a node with at least 2 children
        parent_node = None
        for node in graph.nodes():
            children = list(graph.successors(node))
            if len(children) >= 2:
                parent_node = node
                break

        # If no parent with 2+ children, just use first node with any children
        if parent_node is None:
            for node in graph.nodes():
                children = list(graph.successors(node))
                if children:
                    parent_node = node
                    break

        # If still no suitable node, just use the first node
        if parent_node is None:
            parent_node = sorted(graph.nodes())[0]

        print(f"- Using node {parent_node} for interventions")
    except:
        parent_node = 0
        print("- Defaulting to node 0 for interventions")

    # Intervention value
    int_value = 2.0

    # For each mechanism, show intervention effects
    for name, scm in mechanisms.items():
        print(f"- Demonstrating intervention effects with {name} mechanism")

        # Sample observational data
        obs_samples = sample_observational(scm, n_samples=1000, seed=seed)

        # Sample interventional data using the CORRECT method with graph parameter
        int_samples = sample_interventional(
            scm,
            node=parent_node,
            value=int_value,
            n_samples=1000,
            seed=seed,
            graph=graph  # Pass the graph for correct causal propagation
        )

        # Plot standard intervention comparison
        plot_intervention_comparison(
            obs_samples, int_samples, node_idx=parent_node, int_value=int_value)
        plt.savefig(f"plots/intervention_{name.lower()}.png")
        plt.close()

        # Plot parent-child relationships to better show mechanism differences
        plot_intervention_parent_child(
            graph, scm, parent_node, int_value, seed=seed)
        plt.savefig(f"plots/intervention_{name.lower()}_parent_child.png")
        plt.close()


def demo_noise_configurations(graph, seed=42):
    """Demonstrate different noise configurations."""
    print("\n=== Demonstrating Noise Configurations ===")

    # Create different noise configurations
    noise_configs = {
        "Gaussian (σ=0.5)": NoiseConfig("gaussian", {"std": 0.5}, seed=seed),
        "Gaussian (σ=2.0)": NoiseConfig("gaussian", {"std": 2.0}, seed=seed),
        "Uniform": NoiseConfig("uniform", {"low": -2.0, "high": 2.0}, seed=seed),
        "Heteroskedastic": NoiseConfig("heteroskedastic",
                                       {"base_std": 0.3, "scale_factor": 0.5},
                                       seed=seed)
    }

    # Generate linear SCMs with different noise configurations
    for name, noise_config in noise_configs.items():
        print(f"- Generating samples with {name} noise")
        scm = generate_linear_scm(graph, noise_config=noise_config, seed=seed)
        samples = sample_observational(scm, n_samples=1000, seed=seed)

        # Format filename
        filename_base = name.lower().replace(' ', '_').replace(
            '(', '').replace(')', '').replace('=', '')

        # Plot distributions
        plot_node_distributions(samples, title=f"Linear SCM with {name} Noise")
        plt.savefig(f"plots/noise_{filename_base}.png")
        plt.close()

        # Plot relationships
        plot_parent_child_relationships(
            graph, samples, title=f"Parent-Child Relationships with {name} Noise")
        plt.savefig(f"plots/noise_{filename_base}_relationships.png")
        plt.close()


def demo_serialization(graph, mechanisms, mechanism_map, seed=42):
    """Demonstrate serialization and loading."""
    print("\n=== Demonstrating Serialization ===")

    # Create a dataset with the mixed mechanism
    dataset = CausalDataset(
        graph=graph,
        scm=mechanisms["Mixed"],
        n_obs=1000,
        n_int=500,
        metadata={"mechanism_types": mechanism_map},
        seed=seed
    )

    # Add some interventions
    dataset.add_intervention(node=0, value=1.0)
    dataset.add_intervention(node=2, value=-1.0)

    # Get data
    obs_data = dataset.get_obs_data()
    int_data_0 = dataset.get_intervention_samples(0, 1.0)
    int_data_2 = dataset.get_intervention_samples(2, -1.0)

    # Print dataset info
    print(f"Original dataset: {dataset}")

    # Save the dataset
    os.makedirs("saved_data", exist_ok=True)
    save_path = "saved_data/mixed_mechanism_dataset.pkl"
    dataset.save(save_path)
    print(f"- Saved dataset to {save_path}")

    # Load the dataset back
    loaded_dataset = CausalDataset.load(save_path)
    print(f"- Loaded dataset: {loaded_dataset}")

    # Verify data integrity
    loaded_obs_data = loaded_dataset.get_obs_data()
    assert all(np.array_equal(obs_data[node], loaded_obs_data[node])
               for node in obs_data.keys())
    print("- Data integrity verified: original and loaded data match")

    # Get standardized data format
    std_data = dataset.get_standardized_data()
    print(f"- Standardized data keys: {std_data.keys()}")


def main():
    """Run the full demo."""
    seed = 42
    np.random.seed(seed)

    # Create output directories
    os.makedirs("plots", exist_ok=True)
    os.makedirs("saved_data", exist_ok=True)

    print("=== Causal Data Generation Framework Demo ===")
    print("This demo showcases the key features of the causal data generation framework.")

    # First, demonstrate different graph types
    graph = demo_graph_types(seed)
    plt.close('all')  # Close figures to save memory

    # Demonstrate mechanism diversity
    mechanisms, mechanism_map = demo_mechanism_diversity(graph, seed)
    plt.close('all')  # Close figures to save memory

    # Demonstrate intervention effects
    demo_intervention_effects(graph, mechanisms, seed)
    plt.close('all')  # Close figures to save memory

    # Demonstrate noise configurations
    demo_noise_configurations(graph, seed)
    plt.close('all')  # Close figures to save memory

    # Demonstrate serialization
    demo_serialization(graph, mechanisms, mechanism_map, seed)
    plt.close('all')  # Close figures to save memory

    print("\nDemo completed! Check the 'plots' directory for visualization results.")


if __name__ == "__main__":
    main()
