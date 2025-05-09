#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fixed Causal Structure Recovery

A streamlined and more robust implementation for causal structure discovery
using meta-learning and strategic interventions. This script efficiently
recovers true causal graph structures through adaptive intervention selection.
"""

import os
import sys
import time
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import argparse
from typing import Dict, List, Tuple, Union, Any, Optional
from pathlib import Path

# Add the parent directory to the path if running from demos directory
if Path(__file__).parent.name == 'demos' and str(Path(__file__).parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent))

# Import utilities
from demos.refactored_utils import (
    infer_adjacency_matrix,
    structural_hamming_distance,
    visualize_graph_comparison,
    compare_graphs,
    DummyGraph,
    get_assets_dir,
    create_causal_graph_from_adjacency,
    visualize_graph,
    convert_to_structural_equation_model,
    get_node_name
)

# Import the SimplifiedCausalDiscovery model
from demos.simplified_causal_discovery import SimplifiedCausalDiscovery

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizedMAML:
    """
    Optimized Model-Agnostic Meta-Learning for causal discovery.
    
    This implements an enhanced MAML algorithm specifically designed to 
    efficiently incorporate interventional data for causal discovery.
    """
    
    def __init__(
        self, 
        model, 
        inner_lr=0.01, 
        num_inner_steps=5,
        l2_reg_weight=0.01,
        sparsity_reg_weight=0.1,
        acyclicity_reg_weight=0.5,
        device=None
    ):
        """Initialize the optimized MAML model."""
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.l2_reg_weight = l2_reg_weight
        self.sparsity_reg_weight = sparsity_reg_weight
        self.acyclicity_reg_weight = acyclicity_reg_weight
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize edge uncertainty tracking
        self.edge_uncertainties = []
    
    def _compute_acyclicity_loss(self, adj_matrix):
        """
        Compute acyclicity constraint loss to ensure DAG structure.
        
        Args:
            adj_matrix: Adjacency matrix [batch_size, num_nodes, num_nodes]
            
        Returns:
            acyclicity_loss: Loss term minimized when graph is acyclic
        """
        batch_size, n, _ = adj_matrix.shape
        loss = 0.0
        
        for b in range(batch_size):
            A = adj_matrix[b]
            A_squared = A * A
            M = torch.matrix_exp(A_squared)
            trace = torch.trace(M)
            h_A = trace - n
            loss += h_A
        
        return loss / batch_size
    
    def _encode_interventions(self, x, intervention_mask):
        """
        Encode intervention information into input features.
        
        Args:
            x: Input tensor [batch_size, num_nodes]
            intervention_mask: Binary mask for interventions [num_nodes]
            
        Returns:
            Enhanced input features
        """
        # Expand intervention mask to match batch dimension
        batch_size = x.shape[0]
        mask_expanded = intervention_mask.expand(batch_size, -1)
        
        # Create intervention marker features (1.0 for intervened nodes, 0.0 otherwise)
        intervention_features = mask_expanded.clone().float()
        
        # Concatenate along feature dimension to create enhanced input
        enhanced_input = torch.cat([x, intervention_features], dim=1)
        
        return enhanced_input
    
    def _compute_edge_uncertainties(self, pred_adj):
        """
        Compute uncertainty for each edge in the predicted adjacency matrix.
        
        Args:
            pred_adj: Predicted adjacency matrix [batch_size, num_nodes, num_nodes]
            
        Returns:
            Uncertainty values for each edge
        """
        # Uncertainty is highest when probability is close to 0.5
        # We use 4*p*(1-p) which peaks at p=0.5 with value 1.0
        uncertainties = 4 * pred_adj * (1 - pred_adj)
        
        # Store for later use
        if len(self.edge_uncertainties) >= 3:
            self.edge_uncertainties.pop(0)  # Keep only recent history
        self.edge_uncertainties.append(uncertainties.detach().cpu())
        
        return uncertainties
    
    def adapt(self, data, intervention_mask=None, intervention_values=None):
        """
        Adapt the model to new data with interventions.
        
        Args:
            data: Observation data [batch_size, num_nodes]
            intervention_mask: Binary mask for interventions [num_nodes]
            intervention_values: Values of interventions [num_nodes]
            
        Returns:
            Updated model
        """
        # Create a copy of the model for adaptation
        adapted_model = copy.deepcopy(self.model)
        adapted_model.train()
        
        # Create optimizer for inner loop optimization
        optimizer = optim.Adam(adapted_model.parameters(), lr=self.inner_lr)
        
        # Track original parameters for regularization
        orig_params = {name: param.clone() for name, param in adapted_model.named_parameters()}
        
        # Enhance input with intervention information if provided
        if intervention_mask is not None:
            enhanced_data = self._encode_interventions(data, intervention_mask)
        else:
            enhanced_data = data
        
        # Perform adaptation steps
        for step in range(self.num_inner_steps):
            # Forward pass
            optimizer.zero_grad()
            
            # Get adjacency matrix prediction
            pred_adj = adapted_model(enhanced_data)
            
            if isinstance(pred_adj, dict) and 'adjacency' in pred_adj:
                pred_adj = pred_adj['adjacency']
            
            # We don't have ground truth, so use regularization to guide learning
            
            # Compute L2 regularization
            l2_reg = 0.0
            for name, param in adapted_model.named_parameters():
                if param.requires_grad:
                    l2_reg += ((param - orig_params[name]) ** 2).sum()
            
            # Compute sparsity regularization (prefer sparse graphs)
            sparsity_reg = torch.mean(pred_adj)
            
            # Compute acyclicity regularization (enforce DAG constraint)
            acyclicity_reg = self._compute_acyclicity_loss(pred_adj)
            
            # Combine all losses
            loss = self.l2_reg_weight * l2_reg + self.sparsity_reg_weight * sparsity_reg + self.acyclicity_reg_weight * acyclicity_reg
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Log progress
            if (step + 1) % 2 == 0:
                logger.info(f"  Step {step+1}/{self.num_inner_steps}, Loss: {loss.item():.6f}")
        
        # Compute edge uncertainties
        with torch.no_grad():
            pred_adj = adapted_model(enhanced_data)
            if isinstance(pred_adj, dict) and 'adjacency' in pred_adj:
                pred_adj = pred_adj['adjacency']
            uncertainties = self._compute_edge_uncertainties(pred_adj)
            adapted_model.uncertainties = uncertainties
        
        return adapted_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fixed Causal Structure Recovery')
    
    # Data parameters
    parser.add_argument('--num-nodes', type=int, default=5,
                        help='Number of nodes in the causal graph')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of observational samples')
    parser.add_argument('--edge-probability', type=float, default=0.3,
                        help='Probability of edges in random graph')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Model parameters
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension size')
    parser.add_argument('--attention-heads', type=int, default=4,
                        help='Number of attention heads for larger graphs')
    
    # Training parameters
    parser.add_argument('--inner-lr', type=float, default=0.01,
                        help='Learning rate for adaptation')
    parser.add_argument('--num-inner-steps', type=int, default=5,
                        help='Number of adaptation steps')
    parser.add_argument('--l2-reg-weight', type=float, default=0.01,
                        help='L2 regularization weight')
    parser.add_argument('--sparsity-reg-weight', type=float, default=0.1,
                        help='Sparsity regularization weight')
    parser.add_argument('--acyclicity-reg-weight', type=float, default=0.5,
                        help='Acyclicity regularization weight')
    
    # Intervention parameters
    parser.add_argument('--max-interventions', type=int, default=10,
                        help='Maximum number of interventions')
    parser.add_argument('--intervention-value', type=float, default=2.0,
                        help='Value to set for interventions')
    parser.add_argument('--intervention-samples', type=int, default=500,
                        help='Samples per intervention')
    
    # Visualization and output
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Directory for output files')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def generate_synthetic_data(num_nodes, num_samples, seed, edge_probability=0.3):
    """
    Generate synthetic causal graph and data.
    
    Args:
        num_nodes: Number of nodes in the causal graph
        num_samples: Number of observational samples to generate
        seed: Random seed for reproducibility
        edge_probability: Probability of edges in the graph
        
    Returns:
        graph: Generated causal graph
        scm: Structural causal model
        data: Generated observational data
    """
    from causal_meta.graph.generators.factory import GraphFactory
    from causal_meta.graph.causal_graph import CausalGraph
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Create node names
    node_names = [f"X_{i}" for i in range(num_nodes)]
    
    try:
        # Generate a random DAG using GraphFactory
        logger.info(f"Generating random DAG with {num_nodes} nodes...")
        temp_graph = GraphFactory.create_random_dag(num_nodes, edge_probability, seed=seed)
        
        # Create a proper CausalGraph with string node names
        graph = CausalGraph()
        
        # Add nodes with string names
        for node in node_names:
            graph.add_node(node)
        
        # Add edges, translating from integer nodes to string nodes
        for i, j in temp_graph.get_edges():
            graph.add_edge(node_names[i], node_names[j])
        
        logger.info(f"Generated graph with {len(graph.get_edges())} edges")
    except Exception as e:
        logger.error(f"Error generating graph with GraphFactory: {e}")
        logger.info("Falling back to manual graph creation")
        
        # Create a manual random DAG
        adj_matrix = np.zeros((num_nodes, num_nodes))
        
        # Add edges ensuring DAG property (i->j only if i<j)
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if np.random.random() < edge_probability:
                    adj_matrix[i, j] = 1
        
        # Create graph from adjacency matrix
        graph = create_causal_graph_from_adjacency(adj_matrix, node_names=node_names)
    
    # Create structural causal model
    scm = convert_to_structural_equation_model(graph, node_names=node_names, noise_scale=0.1)
    
    # Generate observational data
    logger.info(f"Generating {num_samples} observational samples...")
    data = scm.sample_data(sample_size=num_samples)
    
    # Convert from DataFrame to numpy array if needed
    if hasattr(data, 'to_numpy'):
        data = data.to_numpy()
    
    return graph, scm, data


def select_intervention_node(model, graph, edge_uncertainties=None, intervention_counts=None):
    """
    Select the next node for intervention based on uncertainty and intervention history.
    
    Args:
        model: Current model
        graph: Current graph estimate
        edge_uncertainties: Tensor of edge uncertainties
        intervention_counts: Dictionary tracking intervention counts per node
        
    Returns:
        Selected node index
    """
    # Get number of nodes
    if hasattr(graph, 'get_nodes'):
        num_nodes = len(graph.get_nodes())
    else:
        num_nodes = graph.shape[0]
    
    # Initialize intervention counts if not provided
    if intervention_counts is None:
        intervention_counts = {i: 0 for i in range(num_nodes)}
    
    # Three-phase strategy:
    # 1. Initial exploration phase: target high-degree nodes first
    # 2. Uncertainty-based phase: focus on the most uncertain edges
    # 3. Refinement phase: verify edges with highest impact
    
    # Determine total interventions so far
    total_interventions = sum(intervention_counts.values())
    
    # Phase 1: Initial exploration (first few interventions)
    if total_interventions < 2:
        # Get node degrees from the graph
        if hasattr(graph, 'get_adjacency_matrix'):
            adj_matrix = graph.get_adjacency_matrix()
        else:
            adj_matrix = graph
            
        # Calculate total degree (in + out) for each node
        if isinstance(adj_matrix, np.ndarray):
            in_degree = adj_matrix.sum(axis=0)
            out_degree = adj_matrix.sum(axis=1)
        else:
            in_degree = adj_matrix.sum(dim=0)
            out_degree = adj_matrix.sum(dim=1)
            
        total_degree = in_degree + out_degree
        
        # Adjust by intervention counts
        adjusted_degree = [total_degree[i] / (intervention_counts[i] + 1) for i in range(num_nodes)]
        
        # Select node with highest adjusted degree
        return np.argmax(adjusted_degree)
    
    # Phase 2: Uncertainty-based selection
    elif edge_uncertainties is not None:
        # Calculate uncertainty score for each node
        node_scores = np.zeros(num_nodes)
        
        # Get uncertainty matrix as numpy array
        if torch.is_tensor(edge_uncertainties):
            uncertainty_matrix = edge_uncertainties.cpu().numpy()
        else:
            uncertainty_matrix = edge_uncertainties
        
        # For each node, sum uncertainties of incoming and outgoing edges
        for i in range(num_nodes):
            # Incoming edges (node is target)
            in_uncertainty = uncertainty_matrix[:, i].sum()
            # Outgoing edges (node is source)
            out_uncertainty = uncertainty_matrix[i, :].sum()
            
            # Combined score
            node_scores[i] = in_uncertainty + out_uncertainty
            
            # Penalize based on previous interventions
            node_scores[i] /= (intervention_counts[i] + 1)
        
        # Return node with highest score
        return np.argmax(node_scores)
    
    # Phase 3: Refinement or fallback to round-robin
    else:
        # Find nodes with minimum intervention count
        min_count = min(intervention_counts.values())
        candidates = [i for i, count in intervention_counts.items() if count == min_count]
        
        # Select randomly from candidates
        return random.choice(candidates)


def recover_causal_structure(args):
    """
    Main function for causal structure recovery.
    
    This implements an efficient loop that adapts a model through
    strategic interventions to recover the true causal graph structure.
    
    Args:
        args: Command line arguments
        
    Returns:
        Results dictionary
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Generate synthetic data for testing
    logger.info("Generating synthetic data...")
    true_graph, scm, obs_data = generate_synthetic_data(
        num_nodes=args.num_nodes,
        num_samples=args.num_samples,
        seed=args.seed,
        edge_probability=args.edge_probability
    )
    
    # Create model
    logger.info("Creating model...")
    model = SimplifiedCausalDiscovery(
        input_dim=args.num_nodes,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        dropout=0.1,
        sparsity_weight=args.sparsity_reg_weight,
        acyclicity_weight=args.acyclicity_reg_weight,
        use_attention=(args.num_nodes > 3),
        num_heads=min(args.attention_heads, args.num_nodes)
    ).to(device)
    
    # Create MAML wrapper
    maml = OptimizedMAML(
        model=model,
        inner_lr=args.inner_lr,
        num_inner_steps=args.num_inner_steps,
        l2_reg_weight=args.l2_reg_weight,
        sparsity_reg_weight=args.sparsity_reg_weight,
        acyclicity_reg_weight=args.acyclicity_reg_weight,
        device=device
    )
    
    # Convert observational data to tensor
    if isinstance(obs_data, np.ndarray):
        obs_tensor = torch.tensor(obs_data, dtype=torch.float32, device=device)
    else:
        obs_tensor = obs_data.to(device)
    
    # Initial inference with observational data
    logger.info("Performing initial inference with observational data...")
    with torch.no_grad():
        initial_adj_matrix = model(obs_tensor)
        
        if isinstance(initial_adj_matrix, dict) and 'adjacency' in initial_adj_matrix:
            initial_adj_matrix = initial_adj_matrix['adjacency']
    
    # Create graph from adjacency matrix
    initial_graph = create_causal_graph_from_adjacency(
        initial_adj_matrix.cpu().numpy(), 
        node_names=[f"X_{i}" for i in range(args.num_nodes)]
    )
    
    # Calculate initial SHD
    initial_shd = structural_hamming_distance(true_graph, initial_graph)
    logger.info(f"Initial SHD: {initial_shd}")
    
    # Show initial comparison if visualization is enabled
    if args.visualize:
        compare_graphs(
            true_graph, 
            initial_graph,
            left_title="True Graph", 
            right_title="Initial Inference"
        )
        plt.savefig(os.path.join(args.output_dir, "initial_comparison.png"))
        plt.show()
    
    # Initialize tracking
    current_graph = initial_graph
    current_model = model
    intervention_history = []
    shd_history = [initial_shd]
    edge_uncertainties = None
    
    # Track intervention counts for all nodes
    intervention_counts = {i: 0 for i in range(args.num_nodes)}
    
    # Progressive intervention loop
    logger.info(f"Starting progressive interventions (max {args.max_interventions})...")
    pbar = tqdm(total=args.max_interventions, desc="Interventions")
    
    for i in range(args.max_interventions):
        # Select node for intervention
        node_idx = select_intervention_node(
            current_model, 
            current_graph, 
            edge_uncertainties, 
            intervention_counts
        )
        
        # Update intervention count
        intervention_counts[node_idx] += 1
        
        # Create node name for intervention
        node_name = f"X_{node_idx}"
        
        # Create intervention dictionary
        intervention = {node_name: args.intervention_value}
        intervention_history.append(intervention)
        
        # Generate interventional data
        int_data = scm.sample_interventional_data(
            interventions=intervention,
            sample_size=args.intervention_samples
        )
        
        # Convert to tensor
        if hasattr(int_data, 'to_numpy'):
            int_data = int_data.to_numpy()
        int_tensor = torch.tensor(int_data, dtype=torch.float32, device=device)
        
        # Create intervention mask
        intervention_mask = torch.zeros(args.num_nodes, device=device)
        intervention_mask[node_idx] = 1.0
        
        # Log intervention
        logger.info(f"Intervention {i+1}: Node {node_name} with value {args.intervention_value}")
        
        # Adapt model with interventional data
        logger.info(f"Adapting model with intervention data...")
        current_model = maml.adapt(
            data=int_tensor,
            intervention_mask=intervention_mask,
            intervention_values=None  # Values handled in the SCM
        )
        
        # Get edge uncertainties for next intervention selection
        edge_uncertainties = current_model.uncertainties
        
        # Infer updated graph structure using adapted model
        with torch.no_grad():
            adj_matrix = current_model(obs_tensor)
            
            if isinstance(adj_matrix, dict) and 'adjacency' in adj_matrix:
                adj_matrix = adj_matrix['adjacency']
        
        # Create graph from adjacency matrix
        current_graph = create_causal_graph_from_adjacency(
            adj_matrix.cpu().numpy(),
            node_names=[f"X_{i}" for i in range(args.num_nodes)]
        )
        
        # Calculate updated SHD
        current_shd = structural_hamming_distance(true_graph, current_graph)
        shd_history.append(current_shd)
        
        # Log progress
        logger.info(f"SHD after intervention {i+1}: {current_shd}")
        
        # Check for convergence (perfect recovery)
        if current_shd == 0:
            logger.info(f"Converged to true graph after {i+1} interventions!")
            pbar.update(args.max_interventions - i - 1)  # Update progress bar to completion
            break
        
        # Visualize if requested
        if args.visualize and (i+1) % 2 == 0:
            compare_graphs(
                true_graph, 
                current_graph,
                left_title="True Graph", 
                right_title=f"After Intervention {i+1}"
            )
            plt.savefig(os.path.join(args.output_dir, f"comparison_int_{i+1}.png"))
            plt.show()
            
            # Also visualize edge uncertainties
            if edge_uncertainties is not None:
                plt.figure(figsize=(8, 6))
                plt.imshow(edge_uncertainties.cpu().numpy().squeeze(), cmap='hot')
                plt.colorbar(label='Uncertainty')
                plt.title(f"Edge Uncertainties after Intervention {i+1}")
                plt.xticks(range(args.num_nodes), [f"X_{i}" for i in range(args.num_nodes)])
                plt.yticks(range(args.num_nodes), [f"X_{i}" for i in range(args.num_nodes)])
                plt.savefig(os.path.join(args.output_dir, f"uncertainty_int_{i+1}.png"))
                plt.show()
        
        # Update progress bar
        pbar.update(1)
    
    # Close progress bar
    pbar.close()
    
    # Always show final comparison
    if args.visualize:
        compare_graphs(
            true_graph, 
            current_graph,
            left_title="True Graph", 
            right_title="Final Inference"
        )
        plt.savefig(os.path.join(args.output_dir, "final_comparison.png"))
        plt.show()
        
        # Plot SHD history
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(shd_history)), shd_history, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Intervention Number', fontsize=12)
        plt.ylabel('Structural Hamming Distance', fontsize=12)
        plt.title('Graph Structure Recovery Progress', fontsize=14)
        plt.xticks(range(len(shd_history)), ['Initial'] + [f'Int {i+1}' for i in range(len(shd_history)-1)])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "shd_history.png"))
        plt.show()
    
    # Prepare results
    results = {
        'true_graph': true_graph,
        'initial_graph': initial_graph,
        'final_graph': current_graph,
        'initial_shd': initial_shd,
        'final_shd': shd_history[-1],
        'shd_history': shd_history,
        'intervention_history': intervention_history,
        'intervention_counts': intervention_counts,
        'converged': shd_history[-1] == 0,
        'num_interventions': len(intervention_history)
    }
    
    return results


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run structure recovery
    logger.info("Starting causal structure recovery...")
    start_time = time.time()
    results = recover_causal_structure(args)
    elapsed_time = time.time() - start_time
    
    # Print summary
    logger.info("\n=== Results Summary ===")
    logger.info(f"Number of nodes: {args.num_nodes}")
    logger.info(f"Initial SHD: {results['initial_shd']}")
    logger.info(f"Final SHD: {results['final_shd']}")
    if results['converged']:
        logger.info(f"Successfully recovered true graph structure in {results['num_interventions']} interventions!")
    else:
        logger.info(f"Did not fully recover true graph structure after {results['num_interventions']} interventions")
        improvement_pct = 100 * (1 - results['final_shd'] / results['initial_shd']) if results['initial_shd'] > 0 else 0
        logger.info(f"Improvement: {improvement_pct:.1f}%")
    logger.info(f"Total runtime: {elapsed_time:.2f} seconds")
    
    return results


if __name__ == "__main__":
    main() 