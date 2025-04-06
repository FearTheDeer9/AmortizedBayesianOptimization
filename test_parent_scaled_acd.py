from graphs.graph_erdos_renyi import ErdosRenyiGraph
from graphs.data_setup import setup_observational_interventional
from algorithms.PARENT_SCALE_algorithm import PARENT_SCALE
from algorithms.PARENT_SCALE_ACD import PARENT_SCALE_ACD
import logging
import os
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
import torch.nn.functional as F
import time
import traceback
import networkx as nx
import argparse

os.chdir("..")
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

# Set up more detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

# Create a logger for this script
logger = logging.getLogger("test_parent_scaled_acd")
logger.setLevel(logging.DEBUG)


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


def create_custom_erdos_renyi_graph(num_nodes, exp_edges=1, seed=42):
    """Creates a custom ErdosRenyiGraph with guaranteed acyclicity.

    Args:
        num_nodes: Number of nodes in the graph
        exp_edges: Expected number of edges per node
        seed: Random seed

    Returns:
        ErdosRenyiGraph instance
    """
    # Create NetworkX graph using our robust method
    nx_graph = create_robust_dag(num_nodes, exp_edges, seed)

    # Create an args object for compatibility with ErdosRenyi
    args = argparse.Namespace(scm_bias=0.0, noise_bias=0.0, old_er_logic=True)

    # Create ErdosRenyiGraph with our pre-made graph
    from diffcbed.envs.erdos_renyi import ErdosRenyi
    from diffcbed.envs.causal_environment import CausalEnvironment

    # Create an ErdosRenyi instance with dummy parameters
    causal_env = ErdosRenyi(
        args=args,
        num_nodes=num_nodes,
        binary_nodes=True,
        nonlinear=False,
        seed=seed,
        exp_edges=exp_edges,
    )

    # Replace its graph with our own
    causal_env.graph = nx_graph
    causal_env.adjacency_matrix = nx.to_numpy_array(nx_graph)
    causal_env.weighted_adjacency_matrix = causal_env.adjacency_matrix.copy()

    # Create ErdosRenyiGraph and replace its causal_env
    graph = ErdosRenyiGraph(
        num_nodes=num_nodes,
        nonlinear=False,
        exp_edges=exp_edges,
        seed=seed
    )
    graph.causal_env = causal_env

    # Update internal structures
    graph._edges = [(str(edge[0]), str(edge[1])) for edge in nx_graph.edges]
    graph._parents, graph._children = graph.build_relationships()
    graph._G = graph.make_graphical_model()

    return graph


def train_and_evaluate_acd_models(
    num_training_graphs: int = 1,
    graph_size: int = 6,
    save_interval: int = 10,
    n_obs: int = 200,
    n_int: int = 2,
    n_trials: int = 10,
    noiseless: bool = True,
    nonlinear: bool = False,
    seed: int = 42
):
    """Train ACD models on multiple Erdos-Renyi graphs and evaluate their performance."""
    start_time = time.time()
    logger.info("\nCreating Erdos-Renyi graph...")

    try:
        # Create a single Erdos-Renyi graph using our robust method
        logger.debug(
            f"Starting graph creation with size={graph_size}, seed={seed}")

        # Use our custom robust graph generation
        graph = create_custom_erdos_renyi_graph(
            num_nodes=graph_size,
            exp_edges=1,  # 1 edge per node is a reasonable starting point
            seed=seed
        )

        graph.set_target(str(graph_size-1))  # Set last node as target
        # Ensure the task is set
        graph.task = "max"  # Assuming we want to maximize the target variable

        logger.info(f"Graph created with {len(graph.edges)} edges")
        logger.info(f"Target variable is: {graph.target}")
        logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")

        # Detailed logging for graph structure
        logger.debug(f"Graph edges: {graph.edges}")
        logger.debug(f"Graph nodes: {graph.nodes}")
        logger.debug(f"Graph variables: {graph.variables}")

        # Get data
        data_start = time.time()
        logger.info("\nGenerating observational and interventional data...")
        D_O, D_I, exploration_set = setup_observational_interventional(
            graph_type=None,
            noiseless=noiseless,
            seed=seed,
            n_obs=n_obs,
            n_int=n_int,
            graph=graph,
        )
        logger.info(f"Generated {len(D_O)} observational samples")
        logger.info(f"Generated {len(D_I)} interventional samples")
        logger.info(f"Time taken: {time.time() - data_start:.2f} seconds")

        # Set population statistics based on observational data
        stats_start = time.time()
        logger.info("\nComputing population statistics...")
        for var in graph.variables:
            if var != graph.target:
                mean_val = np.mean(D_O[var])
                std_val = np.std(D_O[var])
                graph.population_mean_variance[var] = {
                    "mean": mean_val, "std": std_val}
                logger.info(
                    f"Variable {var}: mean={mean_val:.3f}, std={std_val:.3f}")
        logger.info(f"Time taken: {time.time() - stats_start:.2f} seconds")

        # Initialize and run ACD model
        model_start = time.time()
        logger.info("\nInitializing PARENT_SCALE_ACD...")
        acd_model = PARENT_SCALE_ACD(
            graph=graph,
            noiseless=noiseless,
            device="cpu"
        )
        logger.info("Setting values...")
        acd_model.set_values(D_O, D_I, exploration_set)
        logger.info(
            f"Time taken for initialization: {time.time() - model_start:.2f} seconds")

        logger.info("\nRunning algorithm...")
        algo_start = time.time()
        results = acd_model.run_algorithm(T=n_trials, show_graphics=True)
        logger.info("Algorithm completed successfully!")
        logger.info(
            f"Time taken for algorithm: {time.time() - algo_start:.2f} seconds")

        logger.info("\nResults:")
        logger.info(f"Final best Y: {results[0][-1]:.3f}")
        logger.info(f"Final cumulative cost: {results[2][-1]:.3f}")
        logger.info(
            f"\nTotal time taken: {time.time() - start_time:.2f} seconds")

        return results

    except Exception as e:
        logger.error(f"Error in test script: {str(e)}")
        logger.error(traceback.format_exc())
        raise e


if __name__ == "__main__":
    # Run the simplified test
    train_and_evaluate_acd_models(
        num_training_graphs=1,
        graph_size=6,
        save_interval=10,
        n_obs=200,
        n_int=2,
        n_trials=10,
        noiseless=True,
        nonlinear=False,
        seed=42
    )
