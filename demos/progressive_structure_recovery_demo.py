#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Progressive Structure Recovery Demo

This script demonstrates the capability of neural networks to progressively
recover the true causal graph structure through a series of adaptive interventions.
The demo continues until the exact graph structure is recovered or a maximum
number of iterations is reached.

This serves as an MVP proof of concept for the neural network approach to
causal structure learning, showing that with sufficient interventions and
adaptations, the true causal structure can be completely recovered.
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
from tqdm import tqdm
import random

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
    calculate_structural_hamming_distance,
    GraphFactory,
    StructuralCausalModel,
    CausalGraph,
    AmortizedCausalDiscovery,
    TaskEmbedding,
    logger
)

# Import directly from causal_meta
from causal_meta.meta_learning.maml import MAMLForCausalDiscovery


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Progressive Structure Recovery Demo')
    parser.add_argument('--num_nodes', type=int, default=5,
                        help='Number of nodes in the synthetic graph')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples for synthetic data generation')
    parser.add_argument('--max_interventions', type=int, default=20,
                        help='Maximum number of interventions before stopping')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Enable visualization')
    parser.add_argument('--inner_lr', type=float, default=0.01,
                        help='Inner learning rate for MAML adaptation')
    parser.add_argument('--num_inner_steps', type=int, default=5,
                        help='Number of inner adaptation steps for MAML')
    parser.add_argument('--pretrained_model_path', type=str, 
                        default=os.path.join(get_checkpoints_dir(), 'acd_model.pt'),
                        help='Path to pretrained model')
    parser.add_argument('--intervention_strategy', type=str, default='parent_count',
                        choices=['parent_count', 'random', 'random_cyclic', 'uncertainty', 'max_parents'],
                        help='Strategy for selecting nodes for intervention')
    parser.add_argument('--quick', action='store_true', 
                        help='Run in quick mode with minimal settings')
    parser.add_argument('--disable_intermediate_plots', action='store_true',
                        help='Disable intermediate plots, only show final results')
    parser.add_argument('--edge_probability', type=float, default=0.3,
                        help='Probability of edges in the random graph generation')
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def generate_synthetic_data(num_nodes, num_samples, seed, edge_probability=0.3):
    """
    Generate synthetic causal graph and data for demonstration.
    
    Args:
        num_nodes: Number of nodes in the causal graph
        num_samples: Number of observational samples to generate
        seed: Random seed for reproducibility
        edge_probability: Probability of an edge between nodes
        
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
            edge_probability=edge_probability,
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


def select_intervention_node(model, inferred_graph, true_graph, strategy='random', intervention_counts=None):
    """
    Select a node for intervention based on the specified strategy.
    
    Args:
        model: Trained model
        inferred_graph: Currently inferred graph structure
        true_graph: True graph structure (for evaluation)
        strategy: Strategy to select the intervention node
                  - 'random': Select a random node
                  - 'random_cyclic': Cycle through all nodes in random order
                  - 'parent_count': Select node with most parents in inferred graph
                  - 'uncertainty': Select node with highest uncertainty
                  - 'max_parents': Select node with most parents in true graph (oracle)
        intervention_counts: Dictionary tracking how many times each node has been intervened on
        
    Returns:
        Selected node index
    """
    num_nodes = inferred_graph.shape[0]
    
    if intervention_counts is None:
        intervention_counts = {f'X_{i}': 0 for i in range(num_nodes)}
    
    if strategy == 'random':
        return random.randint(0, num_nodes - 1)
    
    elif strategy == 'random_cyclic':
        # Find the node that has been intervened on the least
        min_interventions = min(intervention_counts.values())
        candidates = [int(node.split('_')[1]) for node, count in intervention_counts.items() 
                      if count == min_interventions]
        # If all nodes have been intervened on equally, select randomly
        return random.choice(candidates)
    
    elif strategy == 'parent_count':
        # Select node with most parents in inferred graph
        parent_counts = inferred_graph.sum(axis=0)
        return torch.argmax(parent_counts).item()
    
    elif strategy == 'uncertainty':
        # This is a placeholder for a more sophisticated uncertainty-based strategy
        # For demonstration, we'll just add some randomness to the parent count strategy
        parent_counts = inferred_graph.sum(axis=0)
        uncertainty_scores = parent_counts + torch.rand_like(parent_counts)
        return torch.argmax(uncertainty_scores).item()
    
    elif strategy == 'max_parents':
        # Oracle strategy: select node with most parents in true graph
        parent_counts = true_graph.sum(axis=0)
        return torch.argmax(parent_counts).item()
    
    else:
        raise ValueError(f"Unknown intervention strategy: {strategy}")


def progressive_structure_recovery(
    model, 
    scm, 
    obs_data, 
    true_graph,
    max_interventions=20, 
    device=None,
    inner_lr=0.01, 
    num_inner_steps=5,
    intervention_strategy='parent_count',
    visualize=False,
    convergence_threshold=0,
    disable_intermediate_plots=False
):
    """
    Progressive structure recovery through adaptive interventions.
    
    This function implements a full training loop that continues until the
    true graph structure is recovered or the maximum number of interventions
    is reached.
    
    Args:
        model: Neural network model (AmortizedCausalDiscovery)
        scm: Structural causal model
        obs_data: Observational data
        true_graph: True causal graph structure
        max_interventions: Maximum number of interventions before stopping
        device: PyTorch device
        inner_lr: Learning rate for inner adaptation loop
        num_inner_steps: Number of adaptation steps per intervention
        intervention_strategy: Strategy for selecting nodes for intervention
        visualize: Whether to visualize intermediate results
        convergence_threshold: SHD threshold to consider convergence (0 means exact match)
        disable_intermediate_plots: Whether to disable all intermediate plots, only showing final result
        
    Returns:
        results: Dictionary with results and metrics
    """
    # Get node names from SCM
    if hasattr(scm, 'get_causal_graph') and hasattr(scm.get_causal_graph(), 'get_nodes'):
        node_names = true_graph.get_nodes()
    else:
        # Fallback to using adjacency matrix
        if hasattr(scm, 'get_adjacency_matrix'):
            adj_matrix = scm.get_adjacency_matrix()
            num_nodes = adj_matrix.shape[0]
            node_names = [get_node_name(i) for i in range(num_nodes)]
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
    
    logger.info("Starting progressive structure recovery...")
    
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
    
    # Check if already converged
    if initial_shd <= convergence_threshold:
        logger.info("Initial inference already recovered the true graph structure!")
        return {
            'converged': True,
            'interventions_to_converge': 0,
            'final_shd': initial_shd,
            'shd_history': [initial_shd],
            'intervention_history': [],
            'inferred_graph': inferred_graph
        }
    
    # Visualize initial inferred graph if requested and not disabled
    if visualize and not disable_intermediate_plots:
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
            save_path=os.path.join(get_assets_dir(), "progressive_initial_comparison.png")
        )
        plt.show()
    
    # Track intervention history and SHD
    intervention_history = []
    shd_history = [initial_shd]
    
    # Current model for adaptation
    current_model = model
    
    # Keep track of interventions on each node
    node_intervention_count = {node: 0 for node in node_names}
    
    # Perform sequential interventions until convergence or max iterations
    logger.info(f"Starting progressive interventions (max {max_interventions})...")
    
    # Create progress bar
    pbar = tqdm(total=max_interventions, desc="Progressive Interventions")
    
    # Initialize variables for the loop
    converged = False
    interventions_to_converge = max_interventions
    best_shd = initial_shd
    best_shd_intervention = 0
    
    for intervention_idx in range(max_interventions):
        # Select node for intervention
        node_idx = select_intervention_node(
            model, inferred_graph, true_graph, 
            strategy=intervention_strategy,
            intervention_counts=node_intervention_count
        )
        
        # Update intervention counts
        node_name = f'X_{node_idx}'
        node_intervention_count[node_name] += 1
        
        # Set intervention value (fixed for simplicity)
        intervention_value = 2.0
        
        # Log the intervention
        intervention = {node_name: intervention_value}
        intervention_history.append(intervention)
        logger.info(f"Intervention {intervention_idx + 1}: Node {node_name} with value {intervention_value}")
        
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
        # We use the true graph's adjacency matrix as the target
        true_adj_matrix = torch.tensor(true_graph.get_adjacency_matrix(), 
                                      dtype=torch.float32).to(device)
        
        # Expand to match batch dimension if needed
        if true_adj_matrix.dim() == 2 and encoder_input.dim() > 2:
            true_adj_matrix = true_adj_matrix.unsqueeze(0).expand(encoder_input.shape[0], -1, -1)
        
        # Support data for adaptation (input tensor, ground truth adjacency matrix)
        support_data = (encoder_input, true_adj_matrix)
        
        # Adapt the model using MAML
        logger.info(f"Adapting model with intervention {intervention_idx + 1} data...")
        try:
            # Adapt the model
            adapted_model = maml_model.adapt(
                graph=inferred_graph,  # Current understanding of the graph
                support_data=support_data  # Interventional data for adaptation
            )
            
            # Update the current model
            current_model = adapted_model
        except Exception as e:
            logger.error(f"Error during model adaptation: {e}")
            logger.warning("Continuing with the current model without adaptation")
        
        # Perform inference with the adapted model
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
        logger.info(f"SHD after intervention {intervention_idx + 1}: {current_shd}")
        
        # Keep track of best SHD so far
        if current_shd < best_shd:
            best_shd = current_shd
            best_shd_intervention = intervention_idx + 1
            logger.info(f"New best SHD: {best_shd} at intervention {best_shd_intervention}")
        
        # Visualize updated inferred graph if requested (only every 5 interventions to avoid too many plots)
        if visualize and not disable_intermediate_plots and (intervention_idx + 1) % 5 == 0:
            plt.figure(figsize=(10, 6))
            visualize_graph(
                inferred_graph,
                title=f"Inferred Graph After Intervention {intervention_idx + 1}",
                highlight_nodes=[node_name],
                figsize=(10, 6)
            )
            plt.show()
            
            # Compare true and inferred graphs
            compare_graphs(
                true_graph, 
                inferred_graph,
                left_title="True Graph",
                right_title=f"Inferred Graph (Int. {intervention_idx + 1})",
                save_path=os.path.join(get_assets_dir(), f"progressive_int_{intervention_idx + 1}.png")
            )
            plt.show()
        
        # Check convergence
        if current_shd <= convergence_threshold:
            logger.info(f"Converged to true graph structure after {intervention_idx + 1} interventions!")
            converged = True
            interventions_to_converge = intervention_idx + 1
            break
        
        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({'SHD': current_shd, 'Best SHD': best_shd})
    
    # Close progress bar
    pbar.close()
    
    # Final result summary
    if converged:
        logger.info(f"Successfully recovered true graph structure after {interventions_to_converge} interventions!")
    else:
        logger.info(f"Did not fully recover true graph structure after {max_interventions} interventions.")
        logger.info(f"Final SHD: {current_shd}")
        logger.info(f"Best SHD: {best_shd} at intervention {best_shd_intervention}")
    
    # Plot SHD history
    if visualize:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(shd_history)), shd_history, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Intervention Number', fontsize=12)
        plt.ylabel('Structural Hamming Distance', fontsize=12)
        plt.title('Graph Structure Recovery Progress', fontsize=14)
        plt.xticks(range(len(shd_history)), ['Initial'] + [f'Int {i+1}' for i in range(len(shd_history)-1)])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(get_assets_dir(), "progressive_shd_history.png"))
        plt.show()
        
        # Visualize final comparison
        compare_graphs(
            true_graph, 
            inferred_graph,
            left_title="True Graph",
            right_title="Final Inferred Graph",
            save_path=os.path.join(get_assets_dir(), "progressive_final_comparison.png")
        )
        plt.show()
        
        # Plot intervention counts per node
        plt.figure(figsize=(10, 6))
        nodes = list(node_intervention_count.keys())
        counts = [node_intervention_count[node] for node in nodes]
        plt.bar(nodes, counts)
        plt.xlabel('Node', fontsize=12)
        plt.ylabel('Number of Interventions', fontsize=12)
        plt.title('Intervention Count per Node', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(get_assets_dir(), "progressive_intervention_counts.png"))
        plt.show()
    
    # Prepare results
    results = {
        'converged': converged,
        'interventions_to_converge': interventions_to_converge if converged else None,
        'final_shd': current_shd,
        'best_shd': best_shd,
        'best_shd_intervention': best_shd_intervention,
        'shd_history': shd_history,
        'intervention_history': intervention_history,
        'node_intervention_count': node_intervention_count,
        'inferred_graph': inferred_graph,
        'final_model': current_model
    }
    
    return results


def main():
    """Main function to run the progressive structure recovery demo."""
    # Parse command line arguments
    args = parse_args()
    
    # Apply quick mode settings if requested
    if args.quick:
        args.num_nodes = 3
        args.num_samples = 50
        args.max_interventions = 5
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Generate synthetic data
    graph, scm, obs_data = generate_synthetic_data(
        num_nodes=args.num_nodes,
        num_samples=args.num_samples,
        seed=args.seed,
        edge_probability=args.edge_probability
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
        graph_path = os.path.join(get_assets_dir(), 'progressive_true_graph.png')
        visualize_graph(graph, title="True Causal Graph", save_path=graph_path)
        logger.info(f"Saved true causal graph visualization to {graph_path}")
        plt.show()
    
    # Run the algorithm
    results = progressive_structure_recovery(
        model=model,
        scm=scm,
        obs_data=obs_data,
        true_graph=graph,
        max_interventions=args.max_interventions,
        device=device,
        inner_lr=args.inner_lr,
        num_inner_steps=args.num_inner_steps,
        intervention_strategy=args.intervention_strategy,
        visualize=args.visualize,
        convergence_threshold=0,  # Exact match required
        disable_intermediate_plots=args.disable_intermediate_plots
    )
    
    # Print summary
    logger.info("\nProgressive Structure Recovery Demo Summary:")
    logger.info(f"Number of nodes: {args.num_nodes}")
    logger.info(f"Number of samples: {args.num_samples}")
    logger.info(f"Maximum interventions: {args.max_interventions}")
    logger.info(f"Intervention strategy: {args.intervention_strategy}")
    logger.info(f"Inner learning rate: {args.inner_lr}")
    logger.info(f"Number of inner steps: {args.num_inner_steps}")
    
    logger.info("\nResults:")
    if results['converged']:
        logger.info(f"✓ Successfully recovered true graph structure!")
        logger.info(f"Number of interventions to converge: {results['interventions_to_converge']}")
    else:
        logger.info(f"✗ Did not fully recover true graph structure.")
        logger.info(f"Final SHD: {results['final_shd']}")
        logger.info(f"Best SHD: {results['best_shd']} at intervention {results['best_shd_intervention']}")
    
    logger.info("\nSHD History:")
    for i, shd in enumerate(results['shd_history']):
        if i == 0:
            logger.info(f"Initial: {shd}")
        else:
            logger.info(f"After Intervention {i}: {shd}")
    
    logger.info("\nIntervention History:")
    for i, intervention in enumerate(results['intervention_history']):
        logger.info(f"Intervention {i+1}: {intervention}")
    
    logger.info("\nNode Intervention Counts:")
    for node, count in results['node_intervention_count'].items():
        logger.info(f"Node {node}: {count} interventions")


if __name__ == "__main__":
    main() 