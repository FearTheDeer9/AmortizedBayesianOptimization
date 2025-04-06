# fmt: off

# Add the project root directory to Python path
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from diffcbed.envs.causal_environment import CausalEnvironment
from diffcbed.envs.causal_dataset import CausalDataset
from diffcbed.envs.erdos_renyi import ErdosRenyi
from graphs.graph_generators import (
    generate_erdos_renyi,
    generate_scale_free,
    generate_small_world,
    verify_graph_properties
)
from graphs.scm_generators import (
    generate_linear_scm,
    generate_nonlinear_scm,
    sample_observational,
    sample_interventional
)


# fmt: on

# Now import everything else

# Set random seed for reproducibility
n_nodes = 5
seed = 42


class Args:
    def __init__(self):
        self.old_er_logic = True
        self.scm_bias = 0.0
        self.noise_bias = 0.0


def plot_graph(G, title):
    """Plot a directed graph with clear node labels and edges."""
    # Increase spring constant and iterations
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Draw nodes with higher contrast
    nx.draw_networkx_nodes(G, pos,
                           node_color='skyblue',
                           node_size=1000,
                           edgecolors='black',
                           linewidths=2)

    # Draw edges with curved arrows
    nx.draw_networkx_edges(G, pos,
                           edge_color='black',
                           width=2,
                           arrowsize=20,
                           arrowstyle='->',
                           connectionstyle='arc3,rad=0.2')

    # Draw labels with better visibility
    nx.draw_networkx_labels(G, pos,
                            font_size=14,
                            font_weight='bold')

    plt.title(title, fontsize=16, pad=20)
    plt.axis('off')  # Turn off axis
    plt.tight_layout()


def plot_distributions(data, title):
    """Plot distribution of data without showing it immediately."""
    plt.hist(data, bins=50, alpha=0.7, label=title)
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def main(silent=False):
    """Main function to demonstrate causal data generation."""
    print("No GPU automatically detected. Setting SETTINGS.GPU to 0, and SETTINGS.NJOBS to cpu_count.")

    # Create args object
    args = Args()

    print("\n1. Graph Generation")
    print("-----------------")

    # Generate and validate different graph types
    er_graph = generate_erdos_renyi(5, edge_prob=0.3, seed=42)
    sf_graph = generate_scale_free(5, attachment_param=1.0, seed=42)
    sw_graph = generate_small_world(5, k_neighbors=2, rewire_prob=0.1, seed=42)

    # Validate graphs
    er_valid, er_issues = verify_graph_properties(er_graph)
    sf_valid, sf_issues = verify_graph_properties(sf_graph)
    sw_valid, sw_issues = verify_graph_properties(sw_graph)

    print(f"Erdos-Renyi graph valid: {er_valid}")
    if not er_valid:
        print(f"Issues: {er_issues}")
    print(f"\nScale-free graph valid: {sf_valid}")
    if not sf_valid:
        print(f"Issues: {sf_issues}")
    print(f"\nSmall-world graph valid: {sw_valid}")
    if not sw_valid:
        print(f"Issues: {sw_issues}")

    if not silent:
        # Plot graphs
        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        plt.title("Erdos-Renyi", fontsize=16, pad=20)
        plot_graph(er_graph, "Erdos-Renyi")

        plt.subplot(132)
        plt.title("Scale-Free", fontsize=16, pad=20)
        plot_graph(sf_graph, "Scale-Free")

        plt.subplot(133)
        plt.title("Small-World", fontsize=16, pad=20)
        plot_graph(sw_graph, "Small-World")

        plt.tight_layout()
        plt.show()

    print("\n2. SCM Generation and Sampling")
    print("----------------------------")

    # Generate linear and nonlinear SCMs
    linear_scm = generate_linear_scm(er_graph, seed=42)
    nonlinear_scm = generate_nonlinear_scm(er_graph, seed=42)

    # Sample from SCMs
    linear_samples = sample_observational(linear_scm, 1000, seed=42)
    nonlinear_samples = sample_observational(nonlinear_scm, 1000, seed=42)

    print("Linear SCM distributions:")
    if not silent:
        plt.figure(figsize=(10, 4))
        plot_distributions(linear_samples, "Linear SCM")
        plt.show()

    print("\nNonlinear SCM distributions:")
    if not silent:
        plt.figure(figsize=(10, 4))
        plot_distributions(nonlinear_samples, "Nonlinear SCM")
        plt.show()

    print("\n3. Using CausalDataset")
    print("--------------------")

    # Create and use CausalDataset
    dataset = CausalDataset(er_graph, linear_scm,
                            n_obs=1000, n_int=100, seed=42)

    # Get observational data
    obs_data = dataset.get_obs_data()

    # Add and get intervention data
    # Add intervention first
    dataset.add_intervention("0", 1.0, n_samples=1000)
    int_data = dataset.get_intervention_samples("0", 1.0)  # Then retrieve it

    if not silent:
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plot_distributions(obs_data, "Observational")
        plt.subplot(122)
        plot_distributions(int_data, "Interventional (node 0)")
        plt.tight_layout()
        plt.show()

    print("\n4. Integration with CausalEnvironment")
    print("---------------------------------")

    # Create and use CausalEnvironment
    env = ErdosRenyi(
        args=args,
        num_nodes=5,
        exp_edges=2,  # Expected number of edges per node
        noise_type="isotropic-gaussian",
        num_samples=1000,
        node_range=(-1, 1),
        seed=42
    )

    # Sample from environment
    print("\nSampling from environment:")
    obs_samples = env.sample(1000)

    # Example 1: Single node intervention
    print("\nPerforming single node intervention:")
    int_samples1 = env.intervene(
        nodes=0,      # Node index
        values=1.0,   # Intervention value
        num_samples=1000
    )

    # Example 2: Multiple node intervention
    print("\nPerforming multiple node intervention:")
    int_samples2 = env.intervene(
        nodes=[0, 2],      # List of node indices
        values=[1.0, -1.0],  # List of intervention values
        num_samples=1000
    )

    # Example 3: Vector-valued intervention
    print("\nPerforming vector-valued intervention:")
    node_vector = np.zeros(env.num_nodes)
    node_vector[0] = 1.0
    node_vector[2] = 1.0
    value_vector = np.zeros(env.num_nodes)
    value_vector[0] = 1.0
    value_vector[2] = -1.0
    int_samples3 = env.intervene(
        nodes=node_vector,    # Vector indicating nodes to intervene on
        values=value_vector,  # Vector of intervention values
        num_samples=1000
    )

    if not silent:
        # Plot all intervention comparisons
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plot_distributions(obs_samples.samples, "Observational")
        plt.subplot(132)
        plot_distributions(int_samples1.samples, "Single Node Intervention")
        plt.subplot(133)
        plot_distributions(int_samples2.samples, "Multiple Node Intervention")
        plt.tight_layout()
        plt.show()

        # Plot comparison of intervention methods
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plot_distributions(int_samples2.samples, "Multiple Node (List)")
        plt.subplot(122)
        plot_distributions(int_samples3.samples, "Multiple Node (Vector)")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Run in non-silent mode to visualize results
    main(silent=False)
