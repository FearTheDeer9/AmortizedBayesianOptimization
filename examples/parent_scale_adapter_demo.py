#!/usr/bin/env python3
"""
Demonstration of the PARENT_SCALE algorithm using the CausalEnvironmentAdapter.
This script shows how to:
1. Create a causal graph
2. Generate a structural causal model (SCM)
3. Set up the CausalEnvironmentAdapter to bridge the graph and algorithm
4. Run the PARENT_SCALE algorithm
5. Visualize and evaluate results
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
from graphs.graph_generators import (
    generate_erdos_renyi,
    generate_scale_free,
    generate_small_world
)
from graphs.scm_generators import (
    NoiseConfig,
    generate_linear_scm,
    generate_polynomial_scm,
    generate_mixed_scm,
    sample_observational,
    sample_interventional
)
from diffcbed.envs.causal_dataset import CausalDataset
# fmt: on

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("parent_scale_adapter_demo")

# Ensure output directories exist
os.makedirs("plots", exist_ok=True)
os.makedirs("saved_data", exist_ok=True)


def plot_graph(G, target_node=None, title=None, filename=None):
    """Plot a directed graph with clear node labels, highlighting the target node."""
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)

    # Draw all nodes except target
    if target_node is not None:
        non_target_nodes = [n for n in G.nodes() if n != target_node]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=non_target_nodes,
            node_color='skyblue',
            node_size=800
        )
        # Draw target node with different color
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[target_node],
            node_color='red',
            node_size=800
        )
    else:
        # Draw all nodes the same color
        nx.draw_networkx_nodes(
            G, pos,
            node_color='skyblue',
            node_size=800
        )

    # Draw edges and labels
    nx.draw_networkx_edges(G, pos, width=2, arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_weight='bold')

    plt.title(title or "Causal Graph", fontsize=16)
    plt.axis('off')
    plt.tight_layout()

    if filename:
        plt.savefig(f"plots/{filename}")

    plt.close()


def plot_results(global_opt, current_y, current_cost, title=None, filename=None):
    """Plot optimization results from PARENT_SCALE."""
    plt.figure(figsize=(12, 8))

    # Create two subplots
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)

    # Plot optimization progress
    iterations = range(1, len(global_opt) + 1)
    ax1.plot(iterations, global_opt, 'b-', linewidth=2, label='Global Optimum')
    ax1.plot(iterations, current_y, 'r--',
             linewidth=1.5, label='Current Target Value')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Target Value')
    ax1.set_title('Optimization Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot cumulative cost
    ax2.plot(iterations, current_cost, 'g-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cumulative Cost')
    ax2.set_title('Intervention Cost')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title or "PARENT_SCALE Results", fontsize=16)
    plt.tight_layout()

    if filename:
        plt.savefig(f"plots/{filename}")

    plt.close()


def plot_posterior_probabilities(posterior_probs, true_parents=None, filename=None):
    """Plot posterior probabilities of parent sets."""
    # Sort parent sets by probability
    sorted_parents = sorted(posterior_probs.items(),
                            key=lambda x: x[1], reverse=True)

    # Take top 10 or fewer if less than 10
    top_parents = sorted_parents[:min(10, len(sorted_parents))]

    # Create labels and values
    labels = [str(p[0]) for p in top_parents]
    values = [p[1] for p in top_parents]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, values, color='skyblue')

    # Highlight true parents if provided
    if true_parents is not None:
        for i, label in enumerate(labels):
            if set(eval(label) if isinstance(label, str) else label) == set(true_parents):
                bars[i].set_color('green')
                bars[i].set_hatch('//')

    plt.xlabel('Parent Sets')
    plt.ylabel('Posterior Probability')
    plt.title('Posterior Probabilities of Parent Sets')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if filename:
        plt.savefig(f"plots/{filename}")

    plt.close()


def demo_parent_scale_with_adapter():
    """Run a demonstration of PARENT_SCALE with the CausalEnvironmentAdapter."""
    logger.info("Starting PARENT_SCALE with CausalEnvironmentAdapter demo")

    # 1. Create a causal graph
    num_nodes = 5
    logger.info(f"Creating Erdos-Renyi graph with {num_nodes} nodes")
    graph = generate_erdos_renyi(num_nodes, edge_prob=0.4, seed=42)

    # Get nodes and set target node (last node by default)
    nodes = list(graph.nodes())
    target_node = nodes[-1]
    logger.info(f"Graph nodes: {nodes}")
    logger.info(f"Target node: {target_node}")

    # Plot the graph
    plot_graph(graph, target_node,
               title=f"Causal Graph (Target: Node {target_node})",
               filename="parent_scale_adapter_graph.png")

    # 2. Generate an SCM with mixed mechanisms
    logger.info("Generating mixed mechanism SCM")
    mechanism_map = {
        str(i): ["linear", "polynomial", "exponential", "sinusoidal"][i % 4]
        for i in range(num_nodes)
    }
    scm = generate_mixed_scm(graph, mechanism_map=mechanism_map, seed=42)

    # 3. Create the CausalEnvironmentAdapter
    logger.info("Creating CausalEnvironmentAdapter")
    adapter = CausalEnvironmentAdapter(
        graph=graph,
        scm=scm,
        seed=42
    )

    # We need to set the target node for PARENT_SCALE
    target_variable = str(target_node)

    # Print graph properties through the adapter
    logger.info(f"Graph nodes via adapter: {adapter.variables}")
    logger.info(f"Number of nodes via adapter: {adapter.num_nodes}")
    logger.info(f"Target parents: {list(graph.predecessors(target_node))}")

    # 4. Generate observational data
    logger.info("Generating observational data")
    n_obs = 500
    obs_data_result = adapter.sample(num_samples=n_obs)

    # Convert samples to dictionary format for PARENT_SCALE
    obs_data = {}
    for i, var in enumerate(adapter.variables):
        obs_data[var] = obs_data_result.samples[:, i]

    # Log observational data structure
    logger.info("Observational data structure:")
    logger.info(f"  Variables: {list(obs_data.keys())}")
    logger.info(
        f"  Sample shape for target: {obs_data[target_variable].shape}")

    # 5. Generate some initial interventional data
    logger.info("Generating initial interventional data")
    n_int_values = 3
    int_data = {}

    # Get intervention nodes (all except target)
    intervention_nodes = [
        var for var in adapter.variables if var != target_variable]
    logger.info(f"Intervention nodes: {intervention_nodes}")

    # Sample a few interventions for each node
    for node in intervention_nodes:
        # Choose intervention values in a reasonable range
        values = np.linspace(-2, 2, n_int_values)

        for value in values:
            # Sample interventional data
            int_samples_result = adapter.intervene(
                nodes=int(node),
                values=value,
                num_samples=100
            )

            # Convert to dictionary format
            int_samples = {}
            for i, var in enumerate(adapter.variables):
                int_samples[var] = int_samples_result.samples[:, i]

            # Store the data
            int_key = (node,)
            if int_key not in int_data:
                int_data[int_key] = {}
            int_data[int_key][value] = int_samples

    # Log int_data structure to debug
    logger.info("Interventional data structure:")
    for int_set, values_dict in int_data.items():
        logger.info(f"  Intervention set: {int_set}")
        for value, samples in values_dict.items():
            sample_keys = list(samples.keys())
            logger.info(f"    Value {value}: Sample keys: {sample_keys}")
            if target_variable in samples:
                logger.info(
                    f"      Target '{target_variable}' sample shape: {samples[target_variable].shape}")
            else:
                logger.info(
                    f"      ⚠️ Target '{target_variable}' MISSING in keys")

    # 6. Define exploration set (single-node interventions)
    exploration_set = [(node,) for node in intervention_nodes]

    # Need to create a helper class to give the graph a target property
    class GraphAdapter:
        def __init__(self, graph, target):
            self.G = graph
            self._target = target
            self._nodes = [str(n) for n in graph.nodes()]
            self._variables = [str(n) for n in graph.nodes()]
            self._manipulative_variables = [
                v for v in self._variables if v != target]

            # Build parent-child relationships
            self._parents, self._children = self._build_relationships()

        def nodes(self):
            """Make nodes callable to match nx.DiGraph interface."""
            return self._nodes

        @property
        def variables(self):
            return self._variables

        @property
        def target(self):
            return self._target

        @property
        def manipulative_variables(self):
            return self._manipulative_variables

        @property
        def parents(self):
            return self._parents

        @property
        def children(self):
            return self._children

        @property
        def edges(self):
            return [(str(u), str(v)) for u, v in self.G.edges()]

        def _build_relationships(self):
            """Build parent and child dictionaries for the graph."""
            parents = {}
            children = {}

            for node in self._nodes:
                node_int = int(node)
                parents[node] = [str(p) for p in self.G.predecessors(node_int)]
                children[node] = [str(c) for c in self.G.successors(node_int)]

            return parents, children

        def get_cost_structure(self, cost_num):
            """Return a simple cost structure."""
            from collections import OrderedDict

            costs = OrderedDict()
            for var in self._manipulative_variables:
                costs[var] = lambda x: 1.0  # Constant cost

            return costs

    # Create our adapter with the target property
    graph_adapter = GraphAdapter(graph, target_variable)

    # Subclass PARENT_SCALE to add debugging
    class DebugPARENT_SCALE(PARENT_SCALE):
        def data_and_prior_setup(self):
            """Override to add debug information."""
            logger.info(f"PARENT_SCALE target: {self.target}")
            logger.info("PARENT_SCALE interventional data structure:")
            for int_set, values_dict in self.D_I.items():
                logger.info(f"  Intervention set: {int_set}")
                logger.info(f"  Values: {list(values_dict.keys())}")
                # Check first value's data structure
                first_value = list(values_dict.keys())[0]
                first_data = values_dict[first_value]
                logger.info(
                    f"  First value data keys: {list(first_data.keys())}")

            # Normalize datasets for posterior probability calculations
            self.standardize_all_data()

            # Check structure of standardized data
            logger.info("PARENT_SCALE scaled interventional data structure:")
            for int_set, values_dict in self.D_I_scaled.items():
                logger.info(f"  Intervention set: {int_set}")
                for value, samples in values_dict.items():
                    logger.info(
                        f"    Value {value}: Keys: {list(samples.keys())}")

            # Set initial probabilities
            self.prior_probabilities = self.determine_initial_probabilities()

            # Create SCM model for posterior updates
            if self.nonlinear:
                self.posterior_model = NonLinearSCMModel(
                    self.prior_probabilities, self.graph)
            else:
                self.posterior_model = LinearSCMModel(
                    self.prior_probabilities, self.graph)

            # Set initial data
            self.posterior_model.set_data(self.D_O_scaled)

            # Create graphs based on priors
            self.define_all_possible_graphs()

            # Override the rest with a try-except to catch exactly where it fails
            try:
                # Add interventional data to posterior
                logger.info("Processing interventional data for posterior:")
                for intervention in self.D_I_scaled:
                    logger.info(
                        f"  Processing intervention set: {intervention}")
                    D_I_sample = self.D_I_scaled[intervention]
                    logger.info(
                        f"    D_I_sample keys: {list(D_I_sample.keys())}")

                    # Check if the target is present
                    if self.target not in D_I_sample:
                        logger.error(
                            f"    ⚠️ Target '{self.target}' missing from D_I_sample keys")

                        # Try to fix by adding target from first value
                        int_key = intervention
                        first_val = list(self.D_I[int_key].keys())[0]
                        if self.target in self.D_I[int_key][first_val]:
                            logger.info(
                                f"    Attempting to fix by copying target values")
                            D_I_sample[self.target] = self.D_I[int_key][first_val][self.target]

                    # Check again
                    if self.target in D_I_sample:
                        logger.info(
                            f"    Target key present, sample count: {len(D_I_sample[self.target])}")
                        num_samples = len(D_I_sample[self.target])

                        for n in range(num_samples):
                            x_dict = {
                                obs_key: D_I_sample[obs_key][n]
                                for obs_key in D_I_sample
                                if obs_key != self.target
                            }
                            y = D_I_sample[self.target][n]
                            self.posterior_model.update_all(x_dict, y)
                    else:
                        logger.error(
                            f"    ⚠️ Still missing target in D_I_sample after fix attempt")

                # Update prior probabilities with posterior
                self.prior_probabilities = self.posterior_model.prior_probabilities.copy()
            except Exception as e:
                logger.error(f"Error in data processing: {e}")
                # Continue without updating posterior if there's an error
                pass

    # 7. Create PARENT_SCALE instance with debug
    logger.info("Creating PARENT_SCALE algorithm")
    ps = DebugPARENT_SCALE(
        graph=graph_adapter,  # Use our adapter with target property
        nonlinear=True,
        causal_prior=True,
        noiseless=False,
        scale_data=True,
        seed=42
    )

    # Set data
    ps.set_values(obs_data, int_data, exploration_set)

    # 8. Run the algorithm
    logger.info("Running PARENT_SCALE algorithm for 15 iterations")
    T = 15  # Number of iterations
    start_time = time.time()
    global_opt, current_y, current_cost, int_sets, int_values, posterior_probs = ps.run_algorithm(
        T=T)
    end_time = time.time()

    logger.info(f"Algorithm completed in {end_time - start_time:.2f} seconds")

    # 9. Plot and analyze results
    # Plot optimization progress
    plot_results(
        global_opt, current_y, current_cost,
        title="PARENT_SCALE Optimization Progress",
        filename="parent_scale_adapter_progress.png"
    )

    # Plot posterior probabilities
    true_parents = list(graph.predecessors(target_node))
    plot_posterior_probabilities(
        posterior_probs,
        true_parents=true_parents,
        filename="parent_scale_adapter_posterior.png"
    )

    # 10. Print results and analysis
    logger.info("\n--- PARENT_SCALE Results ---")

    # Best intervention found
    best_index = np.argmax(current_y)
    best_intervention_set = int_sets[best_index]
    best_intervention_values = int_values[best_index]
    best_target_value = current_y[best_index]

    logger.info(
        f"Best intervention: {best_intervention_set} = {best_intervention_values}")
    logger.info(f"Best target value: {best_target_value}")

    # Most likely parent set
    sorted_parents = sorted(posterior_probs.items(),
                            key=lambda x: x[1], reverse=True)
    most_likely_parents = sorted_parents[0][0]
    most_likely_prob = sorted_parents[0][1]

    logger.info(
        f"Most likely parent set: {most_likely_parents} (prob: {most_likely_prob:.4f})")
    logger.info(f"True parent set: {true_parents}")

    # Check if true parents are correctly identified
    correctly_identified = set(most_likely_parents) == set(true_parents)
    logger.info(f"Parents correctly identified: {correctly_identified}")

    # Print information about all interventions performed
    logger.info("\nAll interventions performed:")
    for i in range(len(int_sets)):
        logger.info(
            f"Iteration {i+1}: {int_sets[i]} = {int_values[i]}, Target value: {current_y[i]}")

    logger.info(
        "\nDemo completed! Check the 'plots' directory for visualizations.")


if __name__ == "__main__":
    demo_parent_scale_with_adapter()
