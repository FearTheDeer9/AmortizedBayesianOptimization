#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Parent-Scaled ACD Demo with Neural Networks

This script demonstrates how to use neural networks as a drop-in replacement for 
traditional surrogate models in the Parent-Scaled ACD algorithm. It uses pre-trained
models for quick inference without extensive training time.

This version uses the Component Registry components from causal_meta package.
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
    logger
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Parent-Scaled ACD Demo')
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
    parser.add_argument('--quick', action='store_true', 
                        help='Run in quick mode with minimal settings')
    parser.add_argument('--pretrained_model_path', type=str, 
                        default=os.path.join(get_checkpoints_dir(), 'acd_model.pt'),
                        help='Path to pretrained model')
    parser.add_argument('--use_full_algorithm', action='store_true',
                        help='Use the full PARENT_SCALE_ACD algorithm implementation')
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


def parent_scaled_acd(model, scm, obs_data, max_interventions, device, visualize=False):
    """
    Implementation of Parent-Scaled ACD algorithm with neural networks.
    
    Args:
        model: Neural network model (AmortizedCausalDiscovery)
        scm: Structural causal model
        obs_data: Observational data
        max_interventions: Maximum number of interventions to perform
        device: PyTorch device
        visualize: Whether to visualize intermediate results
        
    Returns:
        inferred_graph: Inferred causal graph
        intervention_history: List of interventions performed
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
    
    logger.info("Starting Parent-Scaled ACD algorithm with neural networks...")
    
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
            save_path=os.path.join(get_assets_dir(), "initial_comparison.png")
        )
        plt.show()
    
    # Track intervention history
    intervention_history = []
    
    # Perform sequential interventions
    for intervention_idx in range(max_interventions):
        logger.info(f"\nPerforming intervention {intervention_idx + 1}/{max_interventions}")
        
        # Select intervention target based on parent count
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
        
        # Perform inference with the interventional data using the enhanced utility
        start_time = time.time()
        with torch.no_grad():
            # Get predicted adjacency matrix
            adj_matrix = infer_adjacency_matrix(
                model, 
                encoder_input, 
                interventions=formatted_interventions
            )
            # Get probabilities for clearer visualization (not thresholded)
            adj_probs = adj_matrix.cpu().numpy()
        
        # Convert inferred adjacency matrix to graph
        inferred_graph = create_causal_graph_from_adjacency(
            adj_probs, 
            node_names=node_names,
            threshold=0.5
        )
        
        logger.info(f"Inference with intervention completed in {time.time() - start_time:.2f} seconds")
        
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
                save_path=os.path.join(get_assets_dir(), f"comparison_intervention_{intervention_idx + 1}.png")
            )
            plt.show()
    
    logger.info("\nParent-Scaled ACD algorithm completed successfully.")
    return inferred_graph, intervention_history


def main():
    """Main function to run the Parent-Scaled ACD demo."""
    # Parse command line arguments
    args = parse_args()
    
    # Apply quick mode settings if requested
    if args.quick:
        args.num_nodes = 3
        args.num_samples = 50
        args.max_interventions = 1
    
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
        graph_path = os.path.join(get_assets_dir(), 'true_causal_graph.png')
        visualize_graph(graph, title="True Causal Graph", save_path=graph_path)
        logger.info(f"Saved true causal graph visualization to {graph_path}")
        plt.show()
    
    # Run the algorithm
    inferred_graph, intervention_history = parent_scaled_acd(
        model=model,
        scm=scm,
        obs_data=obs_data,
        max_interventions=args.max_interventions,
        device=device,
        visualize=args.visualize
    )
    
    # Print summary
    logger.info("\nParent-Scaled ACD Demo Summary:")
    logger.info(f"Number of nodes: {args.num_nodes}")
    logger.info(f"Number of samples: {args.num_samples}")
    logger.info(f"Number of interventions: {args.max_interventions}")
    
    logger.info("\nIntervention History:")
    for i, intervention in enumerate(intervention_history):
        logger.info(f"Intervention {i+1}: {intervention}")
    
    # Final comparison visualization
    if args.visualize:
        final_comparison_path = os.path.join(get_assets_dir(), 'final_comparison.png')
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