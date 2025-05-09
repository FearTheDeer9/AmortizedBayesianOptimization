#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MAML-Based Adaptive Causal Discovery Demo

This script demonstrates how to use Model-Agnostic Meta-Learning (MAML) for
adaptive causal discovery with interventions. Unlike the standard approach,
this demo shows how the model's understanding of the causal structure
improves with each intervention through adaptation.

The demo uses components from the causal_meta package according to the Component Registry.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any
import networkx as nx

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utilities and components from refactored utils
from demos.refactored_utils import (
    get_assets_dir,
    get_checkpoints_dir,
    standardize_tensor_shape,
    get_node_name,
    get_node_id,
    format_interventions,
    create_causal_graph_from_adjacency,
    visualize_graph,
    compare_graphs,
    load_model,
    infer_adjacency_matrix,
    convert_to_structural_equation_model,
    select_intervention_target_by_parent_count,
    GraphFactory,
    StructuralCausalModel,
    CausalGraph,
    AmortizedCausalDiscovery,
    MAMLForCausalDiscovery,
    TaskEmbedding,
    logger
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MAML-Based Adaptive Causal Discovery Demo')
    parser.add_argument('--num_nodes', type=int, default=5,
                        help='Number of nodes in the synthetic graph')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples for synthetic data generation')
    parser.add_argument('--max_interventions', type=int, default=3,
                        help='Maximum number of interventions to perform')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Enable visualization')
    parser.add_argument('--inner_lr', type=float, default=0.01,
                        help='Inner learning rate for MAML adaptation')
    parser.add_argument('--num_inner_steps', type=int, default=3,
                        help='Number of inner adaptation steps for MAML')
    parser.add_argument('--pretrained_model_path', type=str, 
                        default=os.path.join(get_checkpoints_dir(), 'acd_model.pt'),
                        help='Path to pretrained model')
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def generate_synthetic_data(num_nodes, num_samples, seed):
    """
    Generate synthetic causal graph and data for demonstration.
    
    Args:
        num_nodes: Number of nodes in the causal graph
        num_samples: Number of observational samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        graph: CausalGraph object
        scm: StructuralCausalModel object
        data: Numpy array of observational data
    """
    np.random.seed(seed)
    logger.info(f"Generating synthetic graph with {num_nodes} nodes...")
    
    # Create node names as strings
    node_names = [get_node_name(i) for i in range(num_nodes)]
    
    # Create a synthetic graph using GraphFactory
    if GraphFactory is not None:
        # First create a graph with integer IDs
        temp_graph = GraphFactory.create_random_dag(
            num_nodes=num_nodes,
            edge_probability=0.3,  # Moderate density
            seed=seed
        )
        
        # Create a new CausalGraph with string node names
        if CausalGraph is not None:
            graph = CausalGraph()
            
            # Add nodes with string names
            for node in node_names:
                graph.add_node(node)
            
            # Add edges, translating from integer nodes to string nodes
            for i, j in temp_graph.get_edges():
                graph.add_edge(node_names[i], node_names[j])
            
            logger.info(f"Graph generated with {len(graph.get_edges())} edges")
        else:
            # Fallback if CausalGraph is not available
            graph = create_causal_graph_from_adjacency(
                temp_graph.get_adjacency_matrix(),
                node_names=node_names
            )
    else:
        # Fallback if GraphFactory is not available
        logger.warning("GraphFactory not available, creating a simple graph manually")
        # Create a simple chain graph X_0 -> X_1 -> ... -> X_n-1
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes - 1):
            adj_matrix[i, i + 1] = 1
            
        graph = create_causal_graph_from_adjacency(adj_matrix, node_names=node_names)
    
    # Create SCM from graph using the improved utility function
    scm = convert_to_structural_equation_model(graph, node_names=node_names)
    
    # Generate observational data
    data = scm.sample_data(sample_size=num_samples)
    logger.info(f"Generated {num_samples} observational samples")
    
    # Convert from DataFrame to numpy array if needed
    if hasattr(data, 'to_numpy'):
        data = data.to_numpy()
    
    return graph, scm, data


def calculate_structural_hamming_distance(graph1, graph2):
    """
    Calculate the Structural Hamming Distance (SHD) between two graphs.
    
    Args:
        graph1: First graph (CausalGraph or similar)
        graph2: Second graph (CausalGraph or similar)
        
    Returns:
        SHD value (int)
    """
    # Extract adjacency matrices
    if hasattr(graph1, 'get_adjacency_matrix'):
        adj1 = graph1.get_adjacency_matrix()
    elif hasattr(graph1, 'adj_matrix'):
        adj1 = graph1.adj_matrix
    else:
        raise ValueError("graph1 must have get_adjacency_matrix method or adj_matrix attribute")
        
    if hasattr(graph2, 'get_adjacency_matrix'):
        adj2 = graph2.get_adjacency_matrix()
    elif hasattr(graph2, 'adj_matrix'):
        adj2 = graph2.adj_matrix
    else:
        raise ValueError("graph2 must have get_adjacency_matrix method or adj_matrix attribute")
    
    # Convert to binary matrices if needed
    adj1_binary = (adj1 > 0.5).astype(int)
    adj2_binary = (adj2 > 0.5).astype(int)
    
    # Calculate SHD (missing edges + extra edges)
    diff = np.abs(adj1_binary - adj2_binary)
    shd = np.sum(diff)
    
    return shd


def maml_adaptive_causal_discovery(model, scm, obs_data, max_interventions, device, 
                                inner_lr=0.01, num_inner_steps=3, visualize=False):
    """
    Implementation of adaptive causal discovery with MAML.
    
    Args:
        model: Neural network model (AmortizedCausalDiscovery)
        scm: Structural causal model
        obs_data: Observational data
        max_interventions: Maximum number of interventions to perform
        device: PyTorch device
        inner_lr: Learning rate for MAML inner loop adaptation
        num_inner_steps: Number of inner loop steps for adaptation
        visualize: Whether to visualize intermediate results
        
    Returns:
        inferred_graph: Final inferred causal graph
        intervention_history: List of interventions performed
        shd_history: History of Structural Hamming Distance values
    """
    # Get node names from SCM
    if hasattr(scm, 'get_causal_graph') and hasattr(scm.get_causal_graph(), 'get_nodes'):
        true_graph = scm.get_causal_graph()
        node_names = true_graph.get_nodes()
    else:
        # Fallback to using adjacency matrix
        if hasattr(scm, 'get_adjacency_matrix'):
            adj_matrix = scm.get_adjacency_matrix()
            num_nodes = adj_matrix.shape[0]
            node_names = [get_node_name(i) for i in range(num_nodes)]
            true_graph = create_causal_graph_from_adjacency(adj_matrix, node_names=node_names)
        else:
            raise ValueError("SCM does not have required methods")
    
    num_nodes = len(node_names)
    
    # Determine number of samples in the observational data
    if isinstance(obs_data, np.ndarray):
        num_samples = obs_data.shape[0]
    else:
        num_samples = len(obs_data)
    
    # Convert observational data to tensor
    if isinstance(obs_data, np.ndarray):
        obs_tensor = torch.tensor(obs_data, dtype=torch.float32).to(device)
    else:
        obs_tensor = obs_data.to(device)
    
    # Prepare data for graph encoder
    encoder_input = standardize_tensor_shape(
        obs_tensor, 
        for_component='graph_encoder'
    )
    
    # Create MAML wrapper for the model
    logger.info("Initializing MAML for adaptive causal discovery...")
    maml_model = MAMLForCausalDiscovery(
        model=model,
        inner_lr=inner_lr,
        num_inner_steps=num_inner_steps,
        device=device
    )
    
    logger.info("Starting adaptive causal discovery with MAML...")
    
    # Perform initial inference from observational data
    start_time = time.time()
    with torch.no_grad():
        # Get predicted adjacency matrix
        adj_matrix = infer_adjacency_matrix(model, encoder_input)
        # Get probabilities for clearer visualization (not thresholded)
        adj_probs = adj_matrix.cpu().numpy()
    
    # Convert inferred adjacency matrix to graph
    inferred_graph = create_causal_graph_from_adjacency(
        adj_probs, 
        node_names=node_names,
        threshold=0.5
    )
    
    logger.info(f"Initial inference completed in {time.time() - start_time:.2f} seconds")
    
    # Calculate initial SHD
    initial_shd = calculate_structural_hamming_distance(true_graph, inferred_graph)
    logger.info(f"Initial Structural Hamming Distance (SHD): {initial_shd}")
    
    # Visualize initial inferred graph if requested
    if visualize:
        plt.figure(figsize=(10, 6))
        visualize_graph(
            inferred_graph,
            title="Initial Inferred Graph (Observational Data)",
            figsize=(10, 6)
        )
        plt.show()
        
        # Compare true and inferred graphs
        compare_graphs(
            true_graph, 
            inferred_graph,
            left_title="True Graph",
            right_title="Initial Inferred Graph",
            save_path=os.path.join(get_assets_dir(), "maml_initial_comparison.png")
        )
        plt.show()
    
    # Track intervention history and SHD
    intervention_history = []
    shd_history = [initial_shd]
    
    # Current model for adaptation
    current_model = model
    
    # Perform sequential interventions with adaptation
    for intervention_idx in range(max_interventions):
        logger.info(f"\nPerforming intervention {intervention_idx + 1}/{max_interventions}")
        
        # Select intervention target based on parent count in the inferred graph
        target_node = select_intervention_target_by_parent_count(inferred_graph)
        intervention_value = 2.0  # Fixed value for simplicity
        
        # Log the intervention
        intervention = {target_node: intervention_value}
        intervention_history.append(intervention)
        logger.info(f"Intervening on node {target_node} with value {intervention_value}")
        
        # Prepare formatted interventions for the model
        formatted_interventions = format_interventions(
            intervention, 
            for_tensor=True,
            num_nodes=num_nodes,
            device=device
        )
        
        # Generate interventional data from SCM
        int_data = scm.sample_interventional_data(
            interventions=intervention,
            sample_size=num_samples
        )
        
        # Convert to tensor
        if hasattr(int_data, 'to_numpy'):
            # Convert pandas DataFrame to numpy array
            int_data_np = int_data.to_numpy()
        else:
            # If already numpy array
            int_data_np = int_data
            
        int_tensor = torch.tensor(int_data_np, dtype=torch.float32).to(device)
        
        # Prepare data for the graph encoder
        encoder_input = standardize_tensor_shape(
            int_tensor, 
            for_component='graph_encoder'
        )
        
        # Prepare support data for MAML adaptation
        # We use both the input tensor and expected output (which is the same tensor in this case)
        # to adapt the model through interventional data
        support_data = (encoder_input, encoder_input)
        
        # Adapt the model using MAML
        logger.info("Adapting model with interventional data...")
        adapted_model = maml_model.adapt(
            graph=inferred_graph,  # Current understanding of the graph
            support_data=support_data  # Interventional data for adaptation
        )
        
        # Update the current model
        current_model = adapted_model
        
        # Perform inference with the adapted model
        start_time = time.time()
        with torch.no_grad():
            # Get predicted adjacency matrix using the adapted model
            adj_matrix = infer_adjacency_matrix(current_model, encoder_input)
            # Get probabilities for clearer visualization (not thresholded)
            adj_probs = adj_matrix.cpu().numpy()
        
        # Convert inferred adjacency matrix to graph
        inferred_graph = create_causal_graph_from_adjacency(
            adj_probs, 
            node_names=node_names,
            threshold=0.5
        )
        
        # Calculate updated SHD
        current_shd = calculate_structural_hamming_distance(true_graph, inferred_graph)
        shd_history.append(current_shd)
        logger.info(f"Structural Hamming Distance after intervention {intervention_idx + 1}: {current_shd}")
        
        logger.info(f"Inference with adapted model completed in {time.time() - start_time:.2f} seconds")
        
        # Visualize updated inferred graph if requested
        if visualize:
            plt.figure(figsize=(10, 6))
            visualize_graph(
                inferred_graph,
                title=f"Inferred Graph After Intervention {intervention_idx + 1}",
                highlight_nodes=[target_node],
                figsize=(10, 6)
            )
            plt.show()
            
            # Compare true and inferred graphs
            compare_graphs(
                true_graph, 
                inferred_graph,
                left_title="True Graph",
                right_title=f"Inferred Graph (Intervention {intervention_idx + 1})",
                save_path=os.path.join(get_assets_dir(), f"maml_comparison_intervention_{intervention_idx + 1}.png")
            )
            plt.show()
    
    # Plot SHD history
    if visualize:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(shd_history)), shd_history, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Intervention Number', fontsize=12)
        plt.ylabel('Structural Hamming Distance', fontsize=12)
        plt.title('Improvement in Graph Structure Understanding', fontsize=14)
        plt.xticks(range(len(shd_history)), ['Initial'] + [f'Int {i+1}' for i in range(max_interventions)])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(get_assets_dir(), "maml_shd_history.png"))
        plt.show()
    
    logger.info("\nMAML-based adaptive causal discovery completed successfully.")
    return inferred_graph, intervention_history, shd_history


def main():
    """Main function to run the MAML-based adaptive causal discovery demo."""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Generate synthetic data
    graph, scm, obs_data = generate_synthetic_data(
        num_nodes=args.num_nodes,
        num_samples=args.num_samples,
        seed=args.seed
    )
    
    # Load ACD model with improved loader
    model = load_model(
        path=args.pretrained_model_path, 
        model_class=AmortizedCausalDiscovery,
        device=device, 
        hidden_dim=64,
        input_dim=1,
        attention_heads=2,
        num_layers=2,
        dropout=0.1,
        sparsity_weight=0.1,
        acyclicity_weight=1.0
    )
    
    # Visualize the true graph for reference
    if args.visualize:
        graph_path = os.path.join(get_assets_dir(), 'maml_true_causal_graph.png')
        visualize_graph(graph, title="True Causal Graph", save_path=graph_path)
        logger.info(f"Saved true causal graph visualization to {graph_path}")
        plt.show()
    
    # Run the algorithm
    inferred_graph, intervention_history, shd_history = maml_adaptive_causal_discovery(
        model=model,
        scm=scm,
        obs_data=obs_data,
        max_interventions=args.max_interventions,
        device=device,
        inner_lr=args.inner_lr,
        num_inner_steps=args.num_inner_steps,
        visualize=args.visualize
    )
    
    # Print summary
    logger.info("\nMAML-Based Adaptive Causal Discovery Demo Summary:")
    logger.info(f"Number of nodes: {args.num_nodes}")
    logger.info(f"Number of samples: {args.num_samples}")
    logger.info(f"Number of interventions: {args.max_interventions}")
    logger.info(f"MAML inner learning rate: {args.inner_lr}")
    logger.info(f"MAML inner steps: {args.num_inner_steps}")
    
    logger.info("\nIntervention History:")
    for i, intervention in enumerate(intervention_history):
        logger.info(f"Intervention {i+1}: {intervention}")
    
    logger.info("\nStructural Hamming Distance (SHD) History:")
    logger.info(f"Initial SHD: {shd_history[0]}")
    for i in range(1, len(shd_history)):
        logger.info(f"After Intervention {i}: {shd_history[i]}")
    
    # Final comparison visualization
    if args.visualize:
        final_comparison_path = os.path.join(get_assets_dir(), 'maml_final_comparison.png')
        compare_graphs(
            graph, 
            inferred_graph,
            left_title="True Graph",
            right_title="Final Inferred Graph",
            save_path=final_comparison_path
        )
        logger.info(f"Saved final comparison to {final_comparison_path}")
        plt.show()


if __name__ == "__main__":
    main() 