#!/usr/bin/env python3
"""
Comprehensive demo script for the PARENT_SCALE algorithm.
Demonstrates:
1. Graph generation and setup
2. Causal dataset creation
3. Algorithm initialization
4. Posterior calculation over parent sets (Theorem 5.3.2)
5. Surrogate model definition and updates
6. Acquisition function optimization
7. Doubly robust method for larger graphs
8. Validation of structure recovery
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
import pandas as pd
from collections import OrderedDict
from copy import deepcopy

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Now import project-specific modules
from graphs import PARENT_SCALE, CausalEnvironmentAdapter
from graphs.graph_generators import (
    generate_erdos_renyi,
    generate_scale_free,
    generate_small_world,
    verify_graph_properties
)
from graphs.scm_generators import (
    NoiseConfig,
    generate_linear_scm,
    generate_mixed_scm,
    sample_observational,
    sample_interventional
)
from diffcbed.envs.causal_dataset import CausalDataset
from graphs.graph_erdos_renyi import ErdosRenyiGraph
from graphs.data_setup import setup_observational_interventional
from algorithms.PARENT_SCALE_algorithm import PARENT_SCALE
from posterior_model.model import (
    DoublyRobustModel,
    LinearSCMModel,
    NonLinearSCMModel
)
from utils.sem_sampling import sample_model
from graphs.parent_scale import PARENT_SCALE
from graphs.graph_generators import generate_erdos_renyi
from graphs.scm_generators import (
    generate_linear_scm,
    generate_polynomial_scm,
    generate_mixed_scm,
    NoiseConfig,
    sample_observational,
    sample_interventional
)
from graphs.causal_env_adapter import CausalEnvironmentAdapter
# fmt: on

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("parent_scale_demo")


def create_robust_dag(num_nodes, exp_edges, seed=42):
    """Create a directed acyclic graph (DAG) using topological ordering.

    This guarantees that the graph is acyclic because edges only go from 
    earlier nodes to later nodes in the ordering.

    Args:
        num_nodes: Number of nodes in the graph
        exp_edges: Expected number of edges per node
        seed: Random seed

    Returns:
        NetworkX DiGraph
    """
    logger.info(
        f"Creating robust DAG with {num_nodes} nodes and ~{exp_edges} edges per node")

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Create a random ordering of nodes (topological order)
    node_order = np.random.permutation(num_nodes)

    # Calculate edge probability to get expected number of edges
    # For each node, we can connect to nodes later in the ordering
    # On average, each node can connect to (num_nodes-1)/2 other nodes
    # To get exp_edges per node, probability = exp_edges/((num_nodes-1)/2)
    p = min(1.0, 2 * exp_edges / (num_nodes - 1))
    logger.info(f"Using edge probability {p:.4f}")

    # Initialize empty adjacency matrix
    adj_matrix = np.zeros((num_nodes, num_nodes))

    # Add edges according to the topological ordering
    edge_count = 0
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            # Add edge with probability p, only from earlier to later nodes
            if np.random.random() < p:
                source = node_order[i]
                target = node_order[j]
                adj_matrix[source, target] = 1
                edge_count += 1

    # Create NetworkX graph from adjacency matrix
    graph = nx.DiGraph(adj_matrix)

    logger.info(f"Created DAG with {edge_count} edges")

    return graph


def create_erdos_renyi_graph(num_nodes, exp_edges=1, seed=42):
    """Creates a graph compatible with the PARENT_SCALE algorithm."""
    # Use the graph_generators directly instead of our custom function
    nx_graph = generate_erdos_renyi(
        n_nodes=num_nodes,
        # Convert edge expectation to probability
        edge_prob=exp_edges/(num_nodes-1),
        seed=seed
    )

    # Verify the graph properties
    is_valid, issues = verify_graph_properties(nx_graph)
    if not is_valid:
        logger.warning(f"Generated graph has issues: {issues}")
        # Try again with a different seed if needed
        if "Graph contains cycles" in issues:
            logger.info("Attempting to remove cycles...")
            # Remove cycles by creating a DAG
            nx_graph = create_robust_dag(num_nodes, exp_edges, seed)

        if "Graph is not weakly connected" in issues:
            logger.info("Graph is not connected, using robust DAG instead...")
            nx_graph = create_robust_dag(num_nodes, exp_edges, seed)

    # Create the graph structure directly
    from graphs.graph import GraphStructure

    class CustomGraph(GraphStructure):
        def __init__(self, nx_graph, num_nodes, seed):
            self._nodes = [str(i) for i in range(num_nodes)]
            self._variables = [str(i) for i in range(num_nodes)]
            self._edges = [(str(edge[0]), str(edge[1]))
                           for edge in nx_graph.edges]
            self._parents, self._children = self.build_relationships()
            self._G = self.make_graphical_model()
            self._target = None
            self._manipulative_variables = self._nodes.copy()
            self._interventional_ranges = None
            self._population_mean_variance = {}
            self._SEM = None
            self._task = "max"  # Default task
            self._parameters = {}

        def make_graphical_model(self):
            G = nx.DiGraph()
            G.add_nodes_from(self._nodes)
            G.add_edges_from(self._edges)
            return G

        @property
        def nodes(self):
            return self._nodes

        @property
        def variables(self):
            return self._variables

        @property
        def edges(self):
            return self._edges

        @property
        def G(self):
            return self._G

        @property
        def parents(self):
            return self._parents

        @property
        def children(self):
            return self._children

        @property
        def target(self):
            return self._target

        @property
        def manipulative_variables(self):
            return self._manipulative_variables

        @property
        def SEM(self):
            return self._SEM

        @property
        def task(self):
            return self._task

        @property
        def population_mean_variance(self):
            return self._population_mean_variance

        def set_target(self, target):
            self._target = target
            if target in self._manipulative_variables:
                self._manipulative_variables.remove(target)
            return self

        def get_sets(self):
            target_set = [self._target]
            parents_set = self._parents[self._target]
            manipulative_set = [
                var for var in self._nodes if var != self._target]
            return target_set, parents_set, manipulative_set

        def initialize_sem(self):
            """Initialize a linear Structural Equation Model (SEM)"""
            from graphs.scm_generators import generate_linear_scm, sample_observational
            self._SEM = generate_linear_scm(self._G, seed=seed)
            return self

        def get_interventional_range(self):
            """Get range of possible interventions for each variable"""
            if self._interventional_ranges is None:
                self._interventional_ranges = {
                    var: (-5.0, 5.0) for var in self._variables}
            return self._interventional_ranges

        def set_interventional_range_data(self, data):
            """Set interventional ranges based on observational data"""
            self._interventional_ranges = {}
            for var in self._variables:
                if var in data:
                    mean = np.mean(data[var])
                    std = np.std(data[var])
                    self._interventional_ranges[var] = (
                        mean - 2*std, mean + 2*std)
                else:
                    self._interventional_ranges[var] = (-5.0, 5.0)
            return self

        def get_parameter_space(self, variables):
            """Get parameter space for optimization"""
            from emukit.core import ContinuousParameter, ParameterSpace

            params = []
            for var in variables:
                if var in self._interventional_ranges:
                    low, high = self._interventional_ranges[var]
                    params.append(ContinuousParameter(var, low, high))
                else:
                    params.append(ContinuousParameter(var, -5.0, 5.0))
            return ParameterSpace(params)

        def get_cost_structure(self, cost_num=1):
            """Create cost functions for interventions"""
            from utils.cbo_classes import Cost

            cost_dict = {}
            for var in self._variables:
                if var != self.target:
                    cost_dict[var] = Cost(cost_type=cost_num).cost
            return cost_dict

        def mispecify_graph(self, edges):
            """Create a mispecified version of the graph with given edges to target"""
            # Keep only edges to target
            self._edges = [(parent, self._target)
                           for parent, child in edges if child == self._target]
            # Rebuild graph
            self._parents, self._children = self.build_relationships()
            self._G = self.make_graphical_model()
            return self

        def fit_samples_to_graph(self, samples, parameters=None, set_priors=False):
            """Fit Gaussian processes to the data"""
            import numpy as np
            from GPy.kern import RBF
            from GPy.models.gp_regression import GPRegression
            from collections import OrderedDict

            # Create an OrderedDict to store GPs
            self._functions = OrderedDict()

            for child, parents in self._parents.items():
                if parents:
                    Y = np.array(samples[child]).reshape(-1, 1)
                    X = np.hstack([samples[parent].reshape(-1, 1)
                                  for parent in parents])

                    # Create and fit GP
                    kernel = RBF(input_dim=len(parents),
                                 variance=1.0, lengthscale=1.0)
                    gp = GPRegression(X=X, Y=Y, kernel=kernel)
                    gp.optimize()
                    self._functions[child] = gp

            return self

        @property
        def functions(self):
            if hasattr(self, '_functions'):
                return self._functions
            return {}

    # Create and initialize our custom graph
    graph = CustomGraph(nx_graph, num_nodes, seed)

    # Set target node (last node)
    target_node = str(num_nodes - 1)
    graph.set_target(target_node)

    # Initialize SEM
    graph.initialize_sem()

    return graph


def plot_graph(G, title=None):
    """Plot a directed graph with clear node labels."""
    # Use G._G for our custom graph, or G directly if it's already a networkx graph
    nx_graph = G.G if hasattr(G, 'G') else G

    pos = nx.spring_layout(nx_graph, k=2, iterations=50, seed=42)
    plt.figure(figsize=(10, 8))

    # Get target if available
    target = G.target if hasattr(G, 'target') else None

    # Draw nodes, with target highlighted
    node_colors = ['red' if node ==
                   target else 'skyblue' for node in nx_graph.nodes]

    nx.draw(nx_graph, pos, with_labels=True, node_color=node_colors,
            node_size=800, arrowsize=20, width=2,
            font_weight='bold', font_size=14)

    # Add legend
    plt.legend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=15),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15)],
               ['Regular Node', 'Target Node'])

    plt.title(title or "Causal Graph Structure", fontsize=16)
    plt.tight_layout()


def plot_posterior_evolution(posterior_history, true_parents):
    """Plot the evolution of posterior probabilities over iterations."""
    plt.figure(figsize=(12, 8))

    # Identify all parent sets that ever had non-zero probability
    all_parent_sets = set()
    for posterior in posterior_history:
        all_parent_sets.update(posterior.keys())

    # Sort parent sets by whether they are true and their max probability
    max_probs = {ps: max([posterior.get(ps, 0) for posterior in posterior_history])
                 for ps in all_parent_sets}
    parent_sets = sorted(all_parent_sets,
                         key=lambda ps: (ps == true_parents, max_probs[ps]),
                         reverse=True)

    # Plot each parent set's probability evolution
    for i, ps in enumerate(parent_sets):
        probs = [posterior.get(ps, 0) for posterior in posterior_history]
        label = f"{ps} (TRUE)" if ps == true_parents else str(ps)

        # Use bold line for true parents
        if ps == true_parents:
            plt.plot(probs, 'r-', linewidth=3, label=label)
        else:
            plt.plot(probs, '-o', linewidth=1.5,
                     markersize=4, label=label, alpha=0.7)

    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Posterior Probability", fontsize=14)
    plt.title("Evolution of Posterior Probabilities Over Parent Sets", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()


def plot_optimization_performance(global_opt, current_y, current_cost):
    """Plot the optimization performance metrics."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Optimal value plot
    ax1.plot(global_opt, 'b-', linewidth=2, label='Global Optimum')
    ax1.scatter(range(len(current_y)), current_y, color='r',
                marker='o', s=50, label='Current Sample')
    ax1.set_ylabel('Target Value', fontsize=14)
    ax1.set_title('Optimization Progress', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Cost plot
    ax2.plot(current_cost, 'g-', linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=14)
    ax2.set_ylabel('Cumulative Cost', fontsize=14)
    ax2.set_title('Cumulative Intervention Cost', fontsize=16)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()


def plot_edge_recovery(graph, posterior_history):
    """Plot the accuracy of edge recovery over iterations."""
    # Get the true edges to the target
    true_edges = set()
    for parent in graph.parents[graph.target]:
        true_edges.add((parent, graph.target))

    # Calculate precision and recall at each iteration
    precision = []
    recall = []
    f1_scores = []

    for posterior in posterior_history:
        # Get predicted parents (with prob > 0.5)
        predicted_parents = set()
        for parents, prob in posterior.items():
            if prob > 0.5:
                for parent in parents:
                    predicted_parents.add((parent, graph.target))

        # Calculate metrics
        true_positives = len(predicted_parents.intersection(true_edges))

        # Handle edge cases to avoid division by zero
        if len(predicted_parents) > 0:
            prec = true_positives / len(predicted_parents)
        else:
            prec = 0.0

        if len(true_edges) > 0:
            rec = true_positives / len(true_edges)
        else:
            rec = 0.0

        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0

        precision.append(prec)
        recall.append(rec)
        f1_scores.append(f1)

    # Plot metrics
    plt.figure(figsize=(12, 6))
    plt.plot(precision, 'b-', linewidth=2, label='Precision')
    plt.plot(recall, 'g-', linewidth=2, label='Recall')
    plt.plot(f1_scores, 'r-', linewidth=2, label='F1 Score')

    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Causal Edge Recovery Performance', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


def create_example_graph(num_nodes=5, seed=42):
    """Create an example causal graph for the demo."""
    # Generate a random DAG
    graph = generate_erdos_renyi(num_nodes, edge_prob=0.3, seed=seed)

    # Generate SCM with mixed mechanisms
    mechanism_map = {
        str(i): ["linear", "polynomial", "exponential", "sinusoidal"][i % 4]
        for i in range(num_nodes)
    }
    scm = generate_mixed_scm(graph, mechanism_map=mechanism_map, seed=seed)

    return graph, scm


def generate_data(graph, scm, n_obs=1000, n_int=10, seed=42):
    """Generate observational and interventional data."""
    # Generate observational data
    obs_data = sample_observational(scm, n_obs, seed=seed)

    # Generate interventional data
    int_data = {}

    # Get all nodes except the target node (last node)
    nodes = list(graph.nodes())
    target_node = nodes[-1]
    intervention_nodes = [n for n in nodes if n != target_node]

    # Create interventional data for each node
    for node in intervention_nodes:
        # Choose a set of intervention values
        values = np.linspace(-2, 2, n_int)

        for value in values:
            int_samples = sample_interventional(
                scm, node, value, n_samples=100, seed=seed, graph=graph)
            int_key = (node,)
            if int_key not in int_data:
                int_data[int_key] = {}
            int_data[int_key][value] = int_samples

    # Define exploration set (all single-node interventions)
    exploration_set = [(node,) for node in intervention_nodes]

    return obs_data, int_data, exploration_set


def run_parent_scale_demo(
    num_nodes=6,
    exp_edges=1,
    n_obs=200,
    n_int=5,
    n_trials=15,
    nonlinear=False,
    use_doubly_robust=True,
    show_graphics=True,
    seed=42
):
    """Run the full PARENT_SCALE demo with validation."""
    np.random.seed(seed)
    start_time = time.time()

    # Create output directories
    os.makedirs("plots", exist_ok=True)
    os.makedirs("saved_data", exist_ok=True)

    print("=== PARENT_SCALE Algorithm Demo ===")
    print("This demo showcases how PARENT_SCALE learns causal structure and optimizes interventions.")

    # 1. Create a graph for demonstration
    logger.info("Creating an Erdos-Renyi graph...")
    graph = create_erdos_renyi_graph(num_nodes, exp_edges, seed)
    logger.info(f"Graph created with {len(graph.edges)} edges")
    logger.info(f"Target variable is: {graph.target}")

    # Plot the true graph
    if show_graphics:
        plot_graph(graph, title="True Causal Graph")
        plt.savefig("plots/true_causal_graph.png")
        plt.close()

    # 2. Generate observational and interventional data
    logger.info("Generating observational and interventional data...")
    D_O, D_I, exploration_set = setup_observational_interventional(
        graph_type=None,
        noiseless=True,
        seed=seed,
        n_obs=n_obs,
        n_int=n_int,
        graph=graph,
    )
    logger.info(f"Generated {len(D_O[graph.target])} observational samples")
    logger.info(f"Generated data for {len(D_I)} interventions")

    # 3. Initialize the PARENT_SCALE algorithm
    logger.info("Initializing PARENT_SCALE algorithm...")
    parent_scale = PARENT_SCALE(
        graph=graph,
        nonlinear=nonlinear,
        causal_prior=True,
        noiseless=True,
        scale_data=True,
        individual=False,
        use_doubly_robust=use_doubly_robust,
        use_iscm=False
    )

    # Set values to the algorithm
    parent_scale.set_values(D_O, D_I, exploration_set)

    # 4. Run the algorithm
    logger.info("Running the PARENT_SCALE algorithm...")
    run_start = time.time()

    # Store posterior history for analysis
    posterior_history = []

    # Override the run_algorithm method to capture posterior probabilities
    original_run_algorithm = parent_scale.run_algorithm

    def instrumented_run_algorithm(*args, **kwargs):
        # Store initial posterior
        parent_scale.data_and_prior_setup()
        parent_scale.define_all_possible_graphs()
        parent_scale.fit_samples_to_graphs()
        posterior_history.append(deepcopy(parent_scale.prior_probabilities))

        # Run the standard algorithm with hooks for capturing posteriors
        original_update_all = parent_scale.posterior_model.update_all

        def instrumented_update_all(*args, **kwargs):
            result = original_update_all(*args, **kwargs)
            posterior_history.append(
                deepcopy(parent_scale.posterior_model.prior_probabilities))
            return result

        # Replace the method
        parent_scale.posterior_model.update_all = instrumented_update_all

        # Run the original algorithm
        result = original_run_algorithm(*args, **kwargs)

        # Restore original method
        parent_scale.posterior_model.update_all = original_update_all

        return result

    # Replace the method
    parent_scale.run_algorithm = instrumented_run_algorithm

    # Run the algorithm and collect results
    global_opt, current_y, current_cost, intervention_set, intervention_values, _ = parent_scale.run_algorithm(
        T=n_trials,
        show_graphics=False  # We'll create our own visualizations
    )

    # Restore original method
    parent_scale.run_algorithm = original_run_algorithm

    logger.info(
        f"Algorithm completed in {time.time() - run_start:.2f} seconds")

    # 5. Validate the results
    logger.info("Validating results...")

    # 5.1. Determine true parents of the target
    true_parents = tuple(sorted(graph.parents[graph.target]))
    logger.info(f"True parents of target node {graph.target}: {true_parents}")

    # Check if the algorithm found the correct structure
    final_posterior = posterior_history[-1]
    best_parent_set = max(final_posterior.items(), key=lambda x: x[1])[
        0] if final_posterior else None
    logger.info(f"Best parent set according to algorithm: {best_parent_set}")
    logger.info(f"Final posterior probabilities: {final_posterior}")

    structure_correct = (best_parent_set == true_parents)
    logger.info(f"Structure correctly identified: {structure_correct}")

    # 5.2. Test if the optimization found a good solution
    best_y = global_opt[-1]
    logger.info(f"Best target value found: {best_y}")

    # Determine the theoretical optimum
    optimal_intervention = {}
    for var in graph.manipulative_variables:
        # For this demo, we assume the best intervention is at the bounds
        min_val, max_val = graph.get_interventional_range()[var]
        # Try both bounds to see which gives better result
        samples_min = sample_model(graph.SEM, {var: min_val}, 100, graph)
        samples_max = sample_model(graph.SEM, {var: max_val}, 100, graph)

        if np.mean(samples_min[graph.target]) > np.mean(samples_max[graph.target]):
            optimal_intervention[var] = min_val
        else:
            optimal_intervention[var] = max_val

    # Sample with optimal intervention
    optimal_samples = sample_model(
        graph.SEM, optimal_intervention, 1000, graph)
    optimal_y = np.mean(optimal_samples[graph.target])

    logger.info(f"Theoretical optimum value: {optimal_y}")
    logger.info(f"Relative performance: {best_y / optimal_y * 100:.2f}%")

    # 6. Visualize the results
    if show_graphics:
        # 6.1. Plot posterior evolution
        plot_posterior_evolution(posterior_history, true_parents)
        plt.savefig("plots/posterior_evolution.png")
        plt.close()

        # 6.2. Plot optimization performance
        plot_optimization_performance(global_opt, current_y, current_cost)
        plt.savefig("plots/optimization_performance.png")
        plt.close()

        # 6.3. Plot edge recovery performance
        plot_edge_recovery(graph, posterior_history)
        plt.savefig("plots/edge_recovery.png")
        plt.close()

    # 7. Run regression test to validate compatibility
    logger.info("Running regression test...")
    # Let's create a simpler graph and run a quick test
    test_graph = create_erdos_renyi_graph(4, 1, seed=seed+1)
    test_D_O, test_D_I, test_exploration_set = setup_observational_interventional(
        graph_type=None,
        noiseless=True,
        seed=seed+1,
        n_obs=100,
        n_int=2,
        graph=test_graph,
    )

    # Initialize and run a short version
    test_parent_scale = PARENT_SCALE(
        graph=test_graph,
        nonlinear=nonlinear,
        use_doubly_robust=use_doubly_robust
    )
    test_parent_scale.set_values(test_D_O, test_D_I, test_exploration_set)

    # Run a short test
    test_results = test_parent_scale.run_algorithm(T=3, show_graphics=False)

    logger.info(f"Regression test completed successfully")

    print("\nDemo completed! Check the 'plots' directory for visualization results.")
    return {
        'graph': graph,
        'posterior_history': posterior_history,
        'true_parents': true_parents,
        'best_parent_set': best_parent_set,
        'structure_correct': structure_correct,
        'optimization_results': {
            'global_opt': global_opt,
            'current_y': current_y,
            'current_cost': current_cost,
            'intervention_set': intervention_set,
            'intervention_values': intervention_values,
        },
        'theoretical_optimum': optimal_y,
        'relative_performance': best_y / optimal_y
    }


if __name__ == "__main__":
    results = run_parent_scale_demo(
        num_nodes=6,        # Small graph for demonstration
        exp_edges=1,        # ~1 edge per node
        n_obs=200,          # 200 observational samples
        n_int=5,            # 5 initial interventional samples
        n_trials=15,        # Run for 15 iterations
        nonlinear=False,    # Use linear SCMs for simplicity
        use_doubly_robust=True,  # Use doubly robust method
        show_graphics=True,      # Generate visualizations
        seed=42                  # Fixed seed for reproducibility
    )
