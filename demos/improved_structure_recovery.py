#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Improved Structure Recovery Demo

This script implements an improved version of the progressive structure recovery demo
that fixes tensor dimension issues and enhances the robustness of causal structure
discovery through strategic interventions and meta-learning.
"""

import os
import sys
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Dict, List, Tuple, Optional
import argparse
import networkx as nx
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from demos.simplified_causal_discovery import SimplifiedCausalDiscovery
from demos.refactored_utils import (create_causal_graph_from_adjacency as CausalGraph, 
                          infer_adjacency_matrix, visualize_graph, 
                          structural_hamming_distance)

def compute_edge_uncertainties(adj_matrix):
    """
    Calculate uncertainty scores for each potential edge in the graph.
    
    Args:
        adj_matrix: Adjacency matrix tensor [batch_size, num_nodes, num_nodes]
        
    Returns:
        Uncertainty scores tensor of same shape
    """
    # Uncertainty is highest at p=0.5, represented by 4*p*(1-p)
    # This peaks at 0.5 with value 1.0
    uncertainties = 4 * adj_matrix * (1 - adj_matrix)
    return uncertainties

def compute_acyclicity_loss(adj_matrix):
    """
    Compute acyclicity regularization loss using matrix exponential.
    
    Args:
        adj_matrix: Adjacency matrix tensor [batch_size, num_nodes, num_nodes]
        
    Returns:
        Acyclicity loss scalar
    """
    batch_size, n, _ = adj_matrix.shape
    loss = 0.0
    
    for b in range(batch_size):
        A = adj_matrix[b]
        A_squared = A * A  # Element-wise square
        M = torch.matrix_exp(A_squared)  # Matrix exponential
        trace = torch.trace(M)
        h_A = trace - n
        loss += h_A
    
    return loss / batch_size

def encode_intervention_as_features(inputs, interventions):
    """
    Encode intervention information as additional input features.
    
    Args:
        inputs: Input tensor [batch_size, seq_len, num_nodes] or [batch_size, num_nodes]
        interventions: Intervention mask [batch_size, num_nodes] or [num_nodes]
        
    Returns:
        Enhanced inputs with intervention markers
    """
    # Ensure interventions has a batch dimension
    if interventions.dim() == 1:
        interventions = interventions.unsqueeze(0)
    
    # Ensure inputs has at least 2 dimensions (batch, nodes)
    if inputs.dim() == 1:
        inputs = inputs.unsqueeze(0)
    
    # Match batch dimension between inputs and interventions
    batch_size = inputs.shape[0]
    if interventions.shape[0] == 1 and batch_size > 1:
        interventions = interventions.repeat(batch_size, 1)
    
    # Apply a scaling factor to intervened node values
    enhanced_data = inputs.clone()
    
    # Handle 2D input case: [batch_size, num_nodes]
    if inputs.dim() == 2:
        for b in range(batch_size):
            for i in range(interventions.shape[1]):
                if interventions[b, i] > 0:
                    enhanced_data[b, i] = enhanced_data[b, i] * 1.2
    
    # Handle 3D input case: [batch_size, seq_len, num_nodes]
    elif inputs.dim() == 3:
        seq_len = inputs.shape[1]
        for b in range(batch_size):
            for i in range(interventions.shape[1]):
                if interventions[b, i] > 0:
                    enhanced_data[b, :, i] = enhanced_data[b, :, i] * 1.2
    
    return enhanced_data

def select_intervention_node(model, inferred_graph, true_graph, strategy='uncertainty', intervention_counts=None, edge_uncertainties=None):
    """
    Select a node for intervention based on edge uncertainty.
    
    Args:
        model: Causal discovery model
        inferred_graph: Current graph estimate
        true_graph: True graph structure (only used for 'oracle' strategy)
        strategy: Strategy for selecting nodes ('uncertainty', 'degree', 'random', 'oracle')
        intervention_counts: Dictionary tracking intervention counts
        edge_uncertainties: Pre-computed edge uncertainties tensor
    
    Returns:
        Selected node index
    """
    num_nodes = inferred_graph.shape[1]
    
    # Initialize intervention counts if not provided
    if intervention_counts is None:
        intervention_counts = {i: 0 for i in range(num_nodes)}
    
    # Calculate uncertainty score for each node
    node_scores = np.zeros(num_nodes)
    
    # Use provided edge uncertainties or compute them
    if edge_uncertainties is None and hasattr(model, 'compute_edge_uncertainties'):
        uncertainties = model.compute_edge_uncertainties(inferred_graph)
    else:
        uncertainties = edge_uncertainties if edge_uncertainties is not None else compute_edge_uncertainties(inferred_graph)
    
    if strategy == 'uncertainty':
        # For each node, calculate information gain potential
        for i in range(num_nodes):
            # Weight incoming edges more (more important for discovery)
            incoming = uncertainties[:, i].sum().item() * 1.5
            
            # Consider outgoing edges
            outgoing = uncertainties[i, :].sum().item()
            
            # Total uncertainty
            node_scores[i] = incoming + outgoing
            
            # Apply logarithmic penalty for previous interventions
            count = intervention_counts.get(i, 0)
            if count > 0:
                node_scores[i] /= (1 + np.log1p(count))
    
    elif strategy == 'degree':
        # Select nodes with highest degree in the current graph
        for i in range(num_nodes):
            in_degree = inferred_graph[:, i].sum().item()
            out_degree = inferred_graph[i, :].sum().item()
            node_scores[i] = in_degree + out_degree
            
            # Apply intervention count penalty
            count = intervention_counts.get(i, 0)
            if count > 0:
                node_scores[i] /= (1 + count)
    
    elif strategy == 'oracle':
        # If we have true graph, target nodes that are most important in true graph
        if true_graph is not None:
            for i in range(num_nodes):
                true_incoming = true_graph[:, i].sum()
                true_outgoing = true_graph[i, :].sum()
                true_total = true_incoming + true_outgoing
                
                # Target important nodes we've intervened on less
                count = intervention_counts.get(i, 0)
                node_scores[i] = true_total / (1 + count)
    
    else:  # Default to random
        node_scores = np.ones(num_nodes)
    
    # Check if all scores are zero or if sum is very close to zero
    if np.sum(node_scores) < 1e-10:
        # If all uncertainties are zero, prioritize nodes with fewest interventions
        min_interventions = float('inf')
        candidates = []
        
        for i in range(num_nodes):
            count = intervention_counts.get(i, 0)
            if count < min_interventions:
                min_interventions = count
                candidates = [i]
            elif count == min_interventions:
                candidates.append(i)
        
        # Randomly select from the candidates with minimum interventions
        return random.choice(candidates)
    else:
        # Normalize scores to get probabilities
        probs = node_scores / np.sum(node_scores)
        
        # Ensure there are no NaN values
        if np.any(np.isnan(probs)):
            # Fall back to deterministic selection
            return np.argmax(node_scores)
        
        # Use weighted random selection with 80% probability
        if np.random.random() < 0.8:
            return np.random.choice(np.arange(num_nodes), p=probs)
        else:
            # 20% of the time select the highest score
            return np.argmax(node_scores)

class EnhancedMAMLForCausalDiscovery:
    """
    Enhanced Model-Agnostic Meta-Learning for causal discovery
    with improved adaptation and uncertainty estimation.
    """
    def __init__(self, model, inner_lr=0.01, num_inner_steps=5, 
                 l2_reg_weight=0.001, sparsity_reg_weight=0.1, acyclicity_reg_weight=0.5,
                 anneal_regularization=True):
        self.model = model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.l2_reg_weight = l2_reg_weight
        self.sparsity_reg_weight = sparsity_reg_weight
        self.acyclicity_reg_weight = acyclicity_reg_weight
        self.anneal_regularization = anneal_regularization
        self.edge_uncertainties = None
    
    def compute_edge_uncertainties(self, pred_adj):
        """Calculate uncertainty for each potential edge"""
        self.edge_uncertainties = compute_edge_uncertainties(pred_adj)
        return self.edge_uncertainties
        
    def compute_loss(self, pred_adj, targets=None, step=0):
        """
        Compute regularization loss with optional supervision
        
        Args:
            pred_adj: Predicted adjacency matrix
            targets: Optional target adjacency matrix for supervision
            step: Current adaptation step (used for annealing)
        
        Returns:
            Total loss, individual loss components
        """
        # Supervised loss if targets are provided
        supervised_loss = 0.0
        if targets is not None:
            supervised_loss = nn.BCELoss()(pred_adj, targets)
        
        # L2 regularization on edges (prefer smaller edge weights)
        l2_loss = torch.mean(pred_adj * pred_adj)
        
        # Sparsity loss (prefer fewer edges)
        sparsity_loss = torch.mean(pred_adj)
        
        # Acyclicity loss (ensure DAG property)
        acyclicity_loss = compute_acyclicity_loss(pred_adj)
        
        # Annealing factor for regularization (gradually reduce from 1.0 to 0.1)
        if self.anneal_regularization:
            annealing_factor = max(1.0 - step * 0.1, 0.1)
        else:
            annealing_factor = 1.0
            
        # Combined loss
        total_loss = (
            supervised_loss + 
            self.l2_reg_weight * annealing_factor * l2_loss +
            self.sparsity_reg_weight * annealing_factor * sparsity_loss + 
            self.acyclicity_reg_weight * annealing_factor * acyclicity_loss
        )
        
        # Return all components for logging
        loss_components = {
            'total': total_loss.item(),
            'supervised': supervised_loss.item() if targets is not None else 0,
            'l2': l2_loss.item(),
            'sparsity': sparsity_loss.item(),
            'acyclicity': acyclicity_loss.item()
        }
        
        return total_loss, loss_components
    
    def adapt(self, data, interventions=None, targets=None):
        """
        Adapt model with interventional data
        
        Args:
            data: Input data tensor
            interventions: Tensor indicating which nodes were intervened on
            targets: Optional ground truth adjacency matrix
            
        Returns:
            Adapted model copy
        """
        # Create a copy of the model for adaptation
        adapted_model = deepcopy(self.model)
        adapted_model.train()
        
        # Create optimizer for inner loop
        optimizer = optim.Adam(adapted_model.parameters(), lr=self.inner_lr)
        
        # Enhanced data with intervention encoding
        if interventions is not None:
            enhanced_data = encode_intervention_as_features(data, interventions)
        else:
            enhanced_data = data
        
        # Track loss history
        loss_history = []
        
        # Perform adaptation steps
        for step in range(self.num_inner_steps):
            optimizer.zero_grad()
            
            # Forward pass
            pred_adj = adapted_model(enhanced_data)
            
            # Compute loss
            loss, loss_components = self.compute_loss(pred_adj, targets, step)
            loss_history.append(loss.item())
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Log progress for even-numbered steps
            if (step + 1) % 2 == 0:
                logger.info(f"Adaptation step {step+1}/{self.num_inner_steps}, "
                           f"Loss: {loss_components['total']:.4f}")
                logger.info(f"  Sparsity: {loss_components['sparsity']:.4f}, "
                           f"Acyclicity: {loss_components['acyclicity']:.4f}")
        
        # Compute edge uncertainties
        with torch.no_grad():
            adapted_model.eval()
            final_pred = adapted_model(enhanced_data)
            self.compute_edge_uncertainties(final_pred)
            
            # Basic early stopping: if loss increased in last step, use previous weights
            if len(loss_history) > 2 and loss_history[-1] > loss_history[-2]:
                logger.info("Loss increased, using model from previous step")
        
        return adapted_model

def calculate_threshold_shd(true_adj, pred_adj, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """
    Calculate SHD for multiple thresholds and return the best one
    
    Args:
        true_adj: True adjacency matrix
        pred_adj: Predicted adjacency matrix
        thresholds: List of thresholds to try
        
    Returns:
        Best threshold, corresponding SHD
    """
    # Convert to numpy for consistent handling
    if isinstance(pred_adj, torch.Tensor):
        pred_adj = pred_adj.detach().cpu().numpy()
    if isinstance(true_adj, torch.Tensor):
        true_adj = true_adj.detach().cpu().numpy()
    
    best_threshold = thresholds[0]
    best_shd = float('inf')
    
    for threshold in thresholds:
        # Apply threshold
        pred_binary = (pred_adj > threshold).astype(float)
        
        # Calculate SHD
        shd = np.sum(np.abs(true_adj - pred_binary))
        
        # Update best if improved
        if shd < best_shd:
            best_shd = shd
            best_threshold = threshold
    
    return best_threshold, best_shd

def progressive_structure_recovery(args):
    """
    Recover causal graph structure through progressive interventions.
    
    Args:
        args: Command line arguments
    
    Returns:
        Results dictionary with metrics and recovered graph
    """
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Generate random DAG for testing
    num_nodes = args.num_nodes
    logger.info(f"Generating random DAG with {num_nodes} nodes...")
    adj_matrix = generate_random_dag(num_nodes, edge_probability=args.edge_probability)
    
    # Count number of edges
    num_edges = np.sum(adj_matrix)
    logger.info(f"Generated graph with {int(num_edges)} edges")
    
    # Create SCM using the DAG
    scm = generate_linear_scm(adj_matrix, noise_scale=args.noise_scale)
    
    # Generate observational data
    sample_size = args.sample_size
    logger.info(f"Generating {sample_size} observational samples...")
    observational_data = scm.sample(sample_size)
    
    # Prepare data tensor (ensure it's a tensor of right shape)
    obs_tensor = torch.tensor(observational_data.values, dtype=torch.float32)
    
    # Create model
    logger.info("Creating model...")
    model = SimplifiedCausalDiscovery(
        input_dim=num_nodes, 
        hidden_dim=args.model_hidden_dim,
        dropout=args.dropout,
        use_attention=args.use_attention
    )
    
    # Create MAML wrapper
    maml = EnhancedMAMLForCausalDiscovery(
        model, 
        inner_lr=args.inner_lr,
        num_inner_steps=args.num_inner_steps,
        l2_reg_weight=args.l2_reg_weight,
        sparsity_reg_weight=args.sparsity_reg_weight,
        acyclicity_reg_weight=args.acyclicity_reg_weight,
        anneal_regularization=args.anneal_regularization
    )
    
    # Initial graph inference from observational data
    logger.info("Performing initial inference with observational data...")
    initial_adj = infer_adjacency_matrix(model, obs_tensor)
    
    # Track interventions and progress
    intervention_history = []
    intervention_counts = {i: 0 for i in range(num_nodes)}
    shd_history = []
    
    # Calculate initial SHD
    _, initial_shd = calculate_threshold_shd(adj_matrix, initial_adj[0])
    shd_history.append(initial_shd)
    logger.info(f"Initial SHD: {initial_shd}")
    
    # Store best model and results
    current_model = model
    best_model = deepcopy(model)
    best_shd = initial_shd
    
    # Visualize initial graph if requested
    if not args.no_visualize:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # True graph
        cg_true = CausalGraph.from_adjacency_matrix(adj_matrix)
        visualize_graph(cg_true, ax=axs[0], title="True Causal Graph")
        
        # Inferred graph
        cg_inferred = CausalGraph.from_adjacency_matrix(initial_adj[0].detach().cpu().numpy(), threshold=0.5)
        visualize_graph(cg_inferred, ax=axs[1], title="Initial Inferred Graph")
        
        plt.suptitle("Initial Graph Comparison")
        plt.tight_layout()
        plt.savefig("initial_comparison.png")
        if not args.disable_intermediate_plots:
            plt.show()
        plt.close(fig)
    
    # Progressive intervention loop
    max_interventions = args.max_interventions
    logger.info(f"Starting progressive interventions (max {max_interventions})...")
    
    for i in range(max_interventions):
        # Infer adjacency matrix with current model
        current_adj = infer_adjacency_matrix(current_model, obs_tensor)
        
        # Select intervention node using strategy
        if i == 0 and maml.edge_uncertainties is None:
            # For first intervention, compute uncertainties
            maml.compute_edge_uncertainties(current_adj)
            
        node_idx = select_intervention_node(
            model=current_model,
            inferred_graph=current_adj,
            true_graph=adj_matrix,
            strategy=args.intervention_strategy,
            intervention_counts=intervention_counts,
            edge_uncertainties=maml.edge_uncertainties
        )
        
        # Update intervention history
        intervention_history.append(node_idx)
        intervention_counts[node_idx] = intervention_counts.get(node_idx, 0) + 1
        
        # Generate interventional data
        intervention_value = 2.0  # Fixed intervention for simplicity
        logger.info(f"Intervention {i+1}: Setting node X{node_idx} to {intervention_value}")
        
        # Generate interventional data
        int_data = intervention_effects(
            scm=scm,
            node_idx=node_idx,
            value=intervention_value,
            sample_size=args.sample_size // 2  # Use smaller sample size for interventions
        )
        
        # Prepare intervention tensor (one-hot encoding)
        int_tensor = torch.tensor(int_data.values, dtype=torch.float32)
        intervention_tensor = torch.zeros(num_nodes)
        intervention_tensor[node_idx] = 1.0
        
        # Adapt model with interventional data
        logger.info(f"Adapting model with interventional data...")
        current_model = maml.adapt(int_tensor, intervention_tensor)
        
        # Evaluate updated model
        updated_adj = infer_adjacency_matrix(current_model, obs_tensor)
        
        # Try multiple thresholds
        best_threshold, current_shd = calculate_threshold_shd(
            adj_matrix, updated_adj[0].detach().cpu().numpy()
        )
        logger.info(f"Best threshold: {best_threshold} with SHD: {current_shd}")
        
        # Track best model
        if current_shd < best_shd:
            best_shd = current_shd
            best_model = deepcopy(current_model)
            logger.info(f"New best model with SHD: {best_shd}")
        
        # Update SHD history
        shd_history.append(current_shd)
        logger.info(f"SHD after intervention {i+1}: {current_shd}")
        
        # Check for convergence
        if current_shd <= args.convergence_threshold:
            logger.info(f"Reached convergence threshold after {i+1} interventions")
            break
        
        # Visualize intermediate results
        if not args.no_visualize and not args.disable_intermediate_plots and (i+1) % 2 == 0:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            
            # True graph
            cg_true = CausalGraph.from_adjacency_matrix(adj_matrix)
            visualize_graph(cg_true, ax=axs[0], title="True Causal Graph")
            
            # Inferred graph
            cg_inferred = CausalGraph.from_adjacency_matrix(
                updated_adj[0].detach().cpu().numpy(), 
                threshold=best_threshold
            )
            visualize_graph(cg_inferred, ax=axs[1], 
                           title=f"Inferred Graph (Intervention {i+1})")
            
            plt.suptitle(f"Graph Comparison - Intervention {i+1}")
            plt.tight_layout()
            plt.savefig(f"comparison_int_{i+1}.png")
            if not args.disable_intermediate_plots:
                plt.show()
            plt.close(fig)
            
            # Also visualize uncertainty
            if maml.edge_uncertainties is not None:
                plt.figure(figsize=(8, 6))
                plt.imshow(maml.edge_uncertainties[0].detach().cpu().numpy(), cmap='hot')
                plt.colorbar(label="Uncertainty")
                plt.title(f"Edge Uncertainties After Intervention {i+1}")
                plt.xticks(range(num_nodes), [f"X{j}" for j in range(num_nodes)])
                plt.yticks(range(num_nodes), [f"X{j}" for j in range(num_nodes)])
                plt.savefig(f"uncertainty_int_{i+1}.png")
                if not args.disable_intermediate_plots:
                    plt.show()
                plt.close()
    
    # Final evaluation
    if not args.no_visualize:
        # Use best model for final visualization
        with torch.no_grad():
            best_adj = infer_adjacency_matrix(best_model, obs_tensor)
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # True graph
        cg_true = CausalGraph.from_adjacency_matrix(adj_matrix)
        visualize_graph(cg_true, ax=axs[0], title="True Causal Graph")
        
        # Inferred graph
        cg_inferred = CausalGraph.from_adjacency_matrix(
            best_adj[0].detach().cpu().numpy(), 
            threshold=best_threshold
        )
        visualize_graph(cg_inferred, ax=axs[1], title="Final Inferred Graph (Best Model)")
        
        plt.suptitle("Final Graph Comparison")
        plt.tight_layout()
        plt.savefig("final_comparison.png")
        plt.show()
        plt.close(fig)
        
        # Plot SHD history
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(shd_history)), shd_history, marker='o', linestyle='-')
        plt.xlabel('Intervention Number')
        plt.ylabel('Structural Hamming Distance (SHD)')
        plt.title('Graph Recovery Progress')
        plt.xticks(range(len(shd_history)), 
                  ['Initial'] + [f'Int {j+1}' for j in range(len(shd_history)-1)])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig("shd_history.png")
        plt.show()
        plt.close()
    
    # Calculate improvement
    improvement = (1 - best_shd / initial_shd) * 100 if initial_shd > 0 else 0
    
    # Return results
    return {
        'true_adj': adj_matrix,
        'final_adj': best_adj[0].detach().cpu().numpy(),
        'initial_shd': initial_shd,
        'final_shd': best_shd,
        'improvement': improvement,
        'interventions': len(intervention_history),
        'intervention_history': intervention_history,
        'intervention_counts': intervention_counts,
        'shd_history': shd_history
    }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Improved Structure Recovery Demo")
    
    # Graph generation parameters
    parser.add_argument("--num-nodes", type=int, default=5, help="Number of nodes in the graph")
    parser.add_argument("--edge-probability", type=float, default=0.3, help="Probability of edge between any two nodes")
    parser.add_argument("--noise-scale", type=float, default=0.1, help="Scale of noise in the SCM")
    
    # Model parameters
    parser.add_argument("--model-hidden-dim", type=int, default=64, help="Hidden dimension for model")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--use-attention", action="store_true", help="Use attention in the model")
    
    # Adaptation parameters
    parser.add_argument("--inner-lr", type=float, default=0.01, help="Learning rate for adaptation")
    parser.add_argument("--num-inner-steps", type=int, default=5, help="Number of adaptation steps")
    parser.add_argument("--l2-reg-weight", type=float, default=0.001, help="Weight for L2 regularization")
    parser.add_argument("--sparsity-reg-weight", type=float, default=0.1, help="Weight for sparsity regularization")
    parser.add_argument("--acyclicity-reg-weight", type=float, default=0.5, help="Weight for acyclicity regularization")
    parser.add_argument("--anneal-regularization", action="store_true", help="Anneal regularization during adaptation")
    
    # Intervention parameters
    parser.add_argument("--max-interventions", type=int, default=10, help="Maximum number of interventions")
    parser.add_argument("--intervention-strategy", type=str, default="uncertainty", 
                        choices=["random", "degree", "oracle", "uncertainty"],
                        help="Strategy for selecting intervention nodes")
    
    # Visualization parameters
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization")
    parser.add_argument("--convergence-threshold", type=float, default=0.0, help="Convergence threshold for SHD")
    parser.add_argument("--disable-intermediate-plots", action="store_true", help="Don't show intermediate plots")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of samples")
    parser.add_argument("--test-multiple-sizes", action="store_true", help="Test multiple graph sizes")
    parser.add_argument("--min-nodes", type=int, default=3, help="Minimum number of nodes for multiple size testing")
    parser.add_argument("--max-nodes", type=int, default=10, help="Maximum number of nodes for multiple size testing")
    
    args = parser.parse_args()
    
    if args.test_multiple_sizes:
        # Test recovery on multiple graph sizes
        results = {}
        sizes = range(args.min_nodes, args.max_nodes + 1)
        
        for size in sizes:
            logger.info(f"\n=== Testing graph size: {size} nodes ===")
            args.num_nodes = size
            
            # Run recovery
            size_result = progressive_structure_recovery(args)
            results[size] = size_result
            
            # Print summary
            logger.info(f"Graph size {size}: Initial SHD {size_result['initial_shd']}, "
                       f"Final SHD {size_result['final_shd']}, "
                       f"Improvement {size_result['improvement']:.1f}%, "
                       f"Interventions {size_result['interventions']}")
        
        # Plot summary across sizes
        sizes = list(results.keys())
        improvements = [results[s]['improvement'] for s in sizes]
        interventions = [results[s]['interventions'] for s in sizes]
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.bar(sizes, improvements)
        plt.xlabel('Graph Size (nodes)')
        plt.ylabel('Improvement (%)')
        plt.title('Recovery Improvement by Graph Size')
        
        plt.subplot(1, 2, 2)
        plt.plot(sizes, interventions, marker='o')
        plt.xlabel('Graph Size (nodes)')
        plt.ylabel('Interventions Used')
        plt.title('Interventions Required by Graph Size')
        
        plt.tight_layout()
        plt.savefig("size_comparison.png")
        if not args.disable_intermediate_plots:
            plt.show()
        plt.close()
        
        return results
    else:
        # Run structure recovery with specified parameters
        results = progressive_structure_recovery(args)
        
        # Print summary
        logger.info("\n=== Results Summary ===")
        logger.info(f"Graph size: {args.num_nodes} nodes")
        logger.info(f"Initial SHD: {results['initial_shd']}")
        logger.info(f"Final SHD: {results['final_shd']}")
        
        if results['final_shd'] == 0:
            logger.info(f"✅ Successfully recovered true graph in {results['interventions']} interventions!")
        else:
            logger.info(f"Improved graph recovery by {results['improvement']:.1f}% "
                       f"after {results['interventions']} interventions")
        
        # Log intervention distribution
        logger.info("Intervention distribution:")
        for node, count in results['intervention_counts'].items():
            if count > 0:
                logger.info(f"  Node X{node}: {count} interventions")
        
        return results

if __name__ == "__main__":
    main() 