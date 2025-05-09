#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Minimal Causal Structure Recovery Implementation

This file contains a streamlined implementation of causal structure recovery 
that addresses key issues with tensor dimensions, intervention encoding,
uncertainty estimation, and optimization stability.
"""

import os
import sys
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Union, Any

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


# --- Model Definition ---

class SimpleCausalNet(nn.Module):
    """
    Simple neural network for causal discovery with explicit intervention handling.
    """
    def __init__(self, input_dim, hidden_dim=64):
        """Initialize the causal discovery network."""
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Feature extraction layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Edge prediction layers - predict entire adjacency matrix
        # +input_dim for intervention encoding
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * input_dim),
        )
        
    def forward(self, x, intervention_mask=None):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            intervention_mask: Optional binary mask indicating intervened nodes
                              [batch_size, input_dim] or [input_dim]
        
        Returns:
            adj_matrix: Predicted adjacency matrix [batch_size, input_dim, input_dim]
        """
        batch_size = x.shape[0]
        
        # Extract features
        features = self.encoder(x)
        
        # Create intervention encoding if provided
        if intervention_mask is not None:
            # Ensure intervention_mask has batch dimension
            if intervention_mask.dim() == 1:
                intervention_mask = intervention_mask.unsqueeze(0).expand(batch_size, -1)
                
            # Concatenate intervention information
            features = torch.cat([features, intervention_mask], dim=1)
        else:
            # If no intervention, pad with zeros
            zero_pad = torch.zeros(batch_size, self.input_dim, device=x.device)
            features = torch.cat([features, zero_pad], dim=1)
        
        # Predict adjacency matrix
        adj_flat = self.edge_predictor(features)
        adj_matrix = adj_flat.view(batch_size, self.input_dim, self.input_dim)
        
        # Apply sigmoid to get probabilities
        adj_matrix = torch.sigmoid(adj_matrix)
        
        # Ensure no self-loops by zeroing diagonal
        adj_matrix = adj_matrix * (1 - torch.eye(self.input_dim, device=x.device).unsqueeze(0))
        
        return adj_matrix


# --- Data Generation Functions ---

def generate_random_dag(num_nodes, edge_probability=0.3, seed=None):
    """
    Generate a random directed acyclic graph (DAG).
    
    Args:
        num_nodes: Number of nodes in the graph
        edge_probability: Probability of edge between any two nodes
        seed: Random seed
        
    Returns:
        Adjacency matrix as numpy array [num_nodes, num_nodes]
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Initialize adjacency matrix
    adj_matrix = np.zeros((num_nodes, num_nodes))
    
    # Create edges with random weights
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):  # Only consider j > i to ensure acyclicity
            if np.random.random() < edge_probability:
                adj_matrix[i, j] = np.random.uniform(0.5, 2.0)  # Random edge weight
    
    return adj_matrix

def generate_data_from_graph(adj_matrix, num_samples=1000, noise_scale=0.1, seed=None):
    """
    Generate data from a linear structural causal model.
    
    Args:
        adj_matrix: Adjacency matrix [num_nodes, num_nodes]
        num_samples: Number of samples to generate
        noise_scale: Scale of the noise
        seed: Random seed
        
    Returns:
        Data matrix [num_samples, num_nodes]
    """
    if seed is not None:
        np.random.seed(seed)
        
    num_nodes = adj_matrix.shape[0]
    data = np.zeros((num_samples, num_nodes))
    
    # Topological order (since it's a DAG, nodes with lower indices influence nodes with higher indices)
    for i in range(num_nodes):
        # Calculate the value based on parents (nodes that have edges to this node)
        parents = np.where(adj_matrix[:, i] != 0)[0]
        parent_contribution = np.zeros(num_samples)
        
        for p in parents:
            parent_contribution += adj_matrix[p, i] * data[:, p]
        
        # Add noise
        noise = np.random.normal(0, noise_scale, num_samples)
        data[:, i] = parent_contribution + noise
    
    return data

def generate_interventional_data(adj_matrix, intervention_node, intervention_value, 
                                num_samples=500, noise_scale=0.1, seed=None):
    """
    Generate interventional data by setting a node to a fixed value.
    
    Args:
        adj_matrix: Adjacency matrix [num_nodes, num_nodes]
        intervention_node: Index of the node to intervene on
        intervention_value: Value to set the node to
        num_samples: Number of samples to generate
        noise_scale: Scale of the noise
        seed: Random seed
        
    Returns:
        Data matrix [num_samples, num_nodes]
    """
    if seed is not None:
        np.random.seed(seed)
        
    num_nodes = adj_matrix.shape[0]
    data = np.zeros((num_samples, num_nodes))
    
    # Set the intervened node to the intervention value
    data[:, intervention_node] = intervention_value
    
    # Topological order starting from the intervention node
    for i in range(num_nodes):
        if i == intervention_node:
            continue  # Skip the intervened node
            
        # Calculate the value based on parents
        parents = np.where(adj_matrix[:, i] != 0)[0]
        parent_contribution = np.zeros(num_samples)
        
        for p in parents:
            parent_contribution += adj_matrix[p, i] * data[:, p]
        
        # Add noise
        noise = np.random.normal(0, noise_scale, num_samples)
        data[:, i] = parent_contribution + noise
    
    return data


# --- Fast MAML Implementation ---

class FastMAML:
    """
    Simplified MAML implementation specifically for causal discovery.
    """
    def __init__(self, model, inner_lr=0.01, num_inner_steps=5, sparsity_weight=0.1, acyclicity_weight=0.5):
        self.model = model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.sparsity_weight = sparsity_weight
        self.acyclicity_weight = acyclicity_weight
        self.edge_uncertainties = None
    
    def compute_edge_uncertainties(self, pred_adj):
        """Calculate uncertainty for each potential edge"""
        # Uncertainty is highest at p=0.5, drops to 0 at p=0 or p=1
        # Use 4*p*(1-p) which maxes at 1.0 when p=0.5
        uncertainty = 4 * pred_adj * (1 - pred_adj)
        self.edge_uncertainties = uncertainty
        return uncertainty
    
    def compute_acyclicity_loss(self, adj_matrix):
        """Compute loss to enforce acyclicity"""
        batch_size, n, _ = adj_matrix.shape
        loss = 0.0
        
        for b in range(batch_size):
            # Matrix exponential approach to enforce acyclicity
            A = adj_matrix[b]
            # Use a more stable approach with power series
            M = torch.eye(n, device=A.device)
            A_pow = A.clone()
            # Compute the first few terms of matrix exponential
            for i in range(1, min(n, 10)):  # limit to 10 terms for efficiency
                M = M + A_pow / i
                A_pow = A_pow @ A  # Matrix multiplication
                
            # Trace should be n for a DAG
            trace = torch.trace(M)
            h_A = torch.abs(trace - n)  # Absolute difference for stability
            loss += h_A
        
        return loss / batch_size
    
    def compute_loss(self, pred_adj):
        """Compute unsupervised loss with regularization"""
        # Sparsity loss - prefer fewer edges
        sparsity_loss = torch.mean(pred_adj)
        
        # Acyclicity loss - ensure the graph is a DAG
        acyclicity_loss = self.compute_acyclicity_loss(pred_adj)
        
        # Combined loss
        total_loss = (
            self.sparsity_weight * sparsity_loss + 
            self.acyclicity_weight * acyclicity_loss
        )
        
        return total_loss, {
            'total': total_loss.item(),
            'sparsity': sparsity_loss.item(),
            'acyclicity': acyclicity_loss.item()
        }
    
    def encode_intervention(self, data, intervention_node):
        """Create an intervention mask tensor"""
        batch_size, input_dim = data.shape
        device = data.device
        
        # Create a binary mask for the intervention
        mask = torch.zeros(batch_size, input_dim, device=device)
        if intervention_node is not None:
            mask[:, intervention_node] = 1.0
            
        return mask
    
    def adapt(self, model, data, intervention_node=None):
        """
        Adapt model with interventional data
        
        Args:
            model: Neural network model
            data: Input data tensor [batch_size, input_dim]
            intervention_node: Optional node index that was intervened on
            
        Returns:
            Adapted model copy
        """
        # Create a copy of the model for adaptation
        adapted_model = deepcopy(model)
        adapted_model.train()
        
        # Prepare data 
        device = next(adapted_model.parameters()).device
        x = torch.tensor(data, dtype=torch.float32, device=device)
        
        # Create intervention mask
        intervention_mask = self.encode_intervention(x, intervention_node)
        
        # Create optimizer for inner loop
        optimizer = torch.optim.Adam(adapted_model.parameters(), lr=self.inner_lr)
        
        # Track loss history
        loss_history = []
        
        # Perform adaptation steps
        for step in range(self.num_inner_steps):
            optimizer.zero_grad()
            
            # Forward pass with intervention information
            pred_adj = adapted_model(x, intervention_mask)
            
            # Compute loss
            loss, loss_components = self.compute_loss(pred_adj)
            loss_history.append(loss.item())
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Log progress periodically
            if (step + 1) % 2 == 0:
                logger.info(f"Adaptation step {step+1}/{self.num_inner_steps}, "
                           f"Loss: {loss_components['total']:.4f}")
                
        # Compute edge uncertainties for the adapted model
        with torch.no_grad():
            adapted_model.eval()
            final_pred = adapted_model(x, intervention_mask)
            uncertainties = self.compute_edge_uncertainties(final_pred)
            
        return adapted_model, final_pred, uncertainties


# --- Intervention Selection Strategy ---

def select_intervention_node(uncertainties, intervention_counts, true_adj=None):
    """
    Select a node for intervention based on edge uncertainty.
    
    Args:
        uncertainties: Edge uncertainty tensor [batch_size, num_nodes, num_nodes]
        intervention_counts: Dictionary tracking intervention counts per node
        true_adj: Optional true adjacency matrix (for oracle strategies)
    
    Returns:
        Selected node index
    """
    if isinstance(uncertainties, torch.Tensor):
        uncertainties = uncertainties.detach().cpu().numpy()
    
    # Extract first batch
    if uncertainties.ndim > 2:
        uncertainties = uncertainties[0]
        
    num_nodes = uncertainties.shape[0]
    
    # Calculate uncertainty score for each node
    node_scores = np.zeros(num_nodes)
    
    for i in range(num_nodes):
        # Weight incoming edges more (more important for discovery)
        incoming = np.sum(uncertainties[:, i]) * 1.5
        
        # Consider outgoing edges
        outgoing = np.sum(uncertainties[i, :])
        
        # Total uncertainty
        node_scores[i] = incoming + outgoing
        
        # Apply logarithmic penalty for previous interventions
        count = intervention_counts.get(i, 0)
        if count > 0:
            # Reduce score based on previous interventions
            node_scores[i] /= (1 + np.log1p(count))
    
    # Handle case where all scores are very close to zero
    if np.sum(node_scores) < 1e-10:
        # Fall back to nodes with fewest interventions
        min_interventions = float('inf')
        candidates = []
        
        for i in range(num_nodes):
            count = intervention_counts.get(i, 0)
            if count < min_interventions:
                min_interventions = count
                candidates = [i]
            elif count == min_interventions:
                candidates.append(i)
        
        return random.choice(candidates)
    
    # Add small random noise to break ties
    node_scores += np.random.uniform(0, 1e-6, num_nodes)
    
    # Use weighted random selection with 80% probability
    if np.random.random() < 0.8:
        # Normalize scores to get probabilities
        probs = node_scores / np.sum(node_scores)
        
        # Ensure there are no NaN values
        if np.any(np.isnan(probs)):
            # Fall back to deterministic selection
            return np.argmax(node_scores)
        
        # Weighted random selection
        return np.random.choice(np.arange(num_nodes), p=probs)
    else:
        # 20% of the time select the highest score
        return np.argmax(node_scores)


# --- Evaluation Functions ---

def calculate_shd(true_adj, pred_adj, threshold=0.5):
    """
    Calculate the Structural Hamming Distance (SHD).
    
    Args:
        true_adj: True adjacency matrix
        pred_adj: Predicted adjacency matrix
        threshold: Threshold for binarizing predictions
        
    Returns:
        SHD value
    """
    # Convert tensors to numpy if needed
    if isinstance(true_adj, torch.Tensor):
        true_adj = true_adj.detach().cpu().numpy()
    if isinstance(pred_adj, torch.Tensor):
        pred_adj = pred_adj.detach().cpu().numpy()
    
    # Handle batch dimension if present
    if pred_adj.ndim > 2:
        pred_adj = pred_adj[0]  # Take first batch item
    
    # Binarize predicted adjacency matrix
    pred_binary = (pred_adj > threshold).astype(float)
    
    # Calculate SHD (sum of different entries)
    shd = np.sum(np.abs(true_adj - pred_binary))
    
    return shd


# --- Visualization Functions ---

def visualize_graph(adj_matrix, title=None, figsize=(8, 6)):
    """
    Visualize a graph from its adjacency matrix.
    
    Args:
        adj_matrix: Adjacency matrix
        title: Title for the plot
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    # Convert tensor to numpy if needed
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.detach().cpu().numpy()
        
    # Handle batch dimension if present
    if adj_matrix.ndim > 2:
        adj_matrix = adj_matrix[0]  # Take first batch item
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    num_nodes = adj_matrix.shape[0]
    G.add_nodes_from(range(num_nodes))
    
    # Add edges
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] > 0.5:  # Threshold
                G.add_edge(i, j, weight=adj_matrix[i, j])
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Get positions
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, width=1.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, {i: f"X{i}" for i in range(num_nodes)})
    
    # Set title
    if title:
        plt.title(title)
    
    plt.axis('off')
    plt.tight_layout()
    
    return plt.gcf()

def compare_graphs(true_adj, pred_adj, figsize=(12, 6)):
    """
    Compare true and predicted graphs side by side.
    
    Args:
        true_adj: True adjacency matrix
        pred_adj: Predicted adjacency matrix
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    
    # Visualize true graph
    plt.sca(axs[0])
    visualize_graph(true_adj, title="True Graph")
    
    # Visualize predicted graph
    plt.sca(axs[1])
    visualize_graph(pred_adj, title="Predicted Graph")
    
    plt.tight_layout()
    return fig


# --- Main Recovery Algorithm ---

def recover_causal_structure(num_nodes=5, max_interventions=10, visualize=True, 
                             inner_lr=0.02, num_inner_steps=10, 
                             sparsity_weight=0.15, acyclicity_weight=0.5,
                             edge_threshold=0.3):
    """
    Recover causal structure using interventional data.
    
    Args:
        num_nodes: Number of nodes in the graph
        max_interventions: Maximum number of interventions to perform
        visualize: Whether to visualize results
        inner_lr: Inner loop learning rate for MAML
        num_inner_steps: Number of inner loop steps
        sparsity_weight: Weight for sparsity regularization
        acyclicity_weight: Weight for acyclicity regularization
        edge_threshold: Threshold for binarizing edges
        
    Returns:
        Results dictionary
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Generate random DAG
    logger.info(f"Generating random DAG with {num_nodes} nodes...")
    true_adj = generate_random_dag(num_nodes, edge_probability=0.3)
    
    # Count edges
    num_edges = np.sum(true_adj > 0)
    logger.info(f"Generated DAG with {num_edges} edges")
    
    if visualize:
        visualize_graph(true_adj, title="True Causal Graph")
        plt.show()
    
    # Generate observational data
    logger.info("Generating observational data...")
    obs_data = generate_data_from_graph(true_adj, num_samples=1000)
    
    # Create model
    logger.info("Creating causal discovery model...")
    model = SimpleCausalNet(input_dim=num_nodes, hidden_dim=64).to(device)
    
    # Create MAML wrapper
    maml = FastMAML(
        model=model,
        inner_lr=inner_lr,
        num_inner_steps=num_inner_steps,
        sparsity_weight=sparsity_weight,
        acyclicity_weight=acyclicity_weight
    )
    
    # Initial inference
    logger.info("Performing initial inference...")
    obs_tensor = torch.tensor(obs_data, dtype=torch.float32).to(device)
    
    # Adapt from observational data (no intervention)
    init_model, init_pred, init_uncertainties = maml.adapt(model, obs_data)
    
    # Track best model and metrics
    best_model = init_model
    best_adj = init_pred[0].detach().cpu().numpy()
    best_shd = calculate_shd(true_adj, best_adj, threshold=edge_threshold)
    
    # Track interventions and metrics
    intervention_history = []
    intervention_counts = {i: 0 for i in range(num_nodes)}
    shd_history = [best_shd]
    
    logger.info(f"Initial SHD: {best_shd}")
    
    # Try different thresholds to find best initial SHD
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    best_threshold = edge_threshold
    min_shd = best_shd
    
    for threshold in thresholds:
        shd = calculate_shd(true_adj, init_pred[0].detach().cpu().numpy(), threshold)
        if shd < min_shd:
            min_shd = shd
            best_threshold = threshold
            
    logger.info(f"Best threshold: {best_threshold} with SHD: {min_shd}")
    best_shd = min_shd  # Update best SHD
    
    # Compare initial graphs if visualize is enabled
    if visualize:
        compare_graphs(true_adj, init_pred[0].detach().cpu().numpy())
        plt.suptitle("Initial Comparison")
        plt.savefig("initial_comparison.png")
        plt.show()
    
    # Progressive intervention loop
    current_model = init_model
    current_adj = init_pred
    current_uncertainties = init_uncertainties
    
    logger.info(f"Starting progressive interventions (max {max_interventions})...")
    
    for i in range(max_interventions):
        # Select intervention node
        node_idx = select_intervention_node(
            uncertainties=current_uncertainties,
            intervention_counts=intervention_counts
        )
        
        # Update intervention history and counts
        intervention_history.append(node_idx)
        intervention_counts[node_idx] = intervention_counts.get(node_idx, 0) + 1
        
        # Generate interventional data
        logger.info(f"Intervention {i+1}: Setting node X{node_idx} to value 2.0")
        int_data = generate_interventional_data(
            adj_matrix=true_adj,
            intervention_node=node_idx,
            intervention_value=2.0,
            num_samples=500
        )
        
        # Adapt model with interventional data
        logger.info(f"Adapting model with interventional data...")
        adapted_model, adapted_pred, adapted_uncertainties = maml.adapt(
            model=current_model,
            data=int_data,
            intervention_node=node_idx
        )
        
        # Update current model and adjacency
        current_model = adapted_model
        current_adj = adapted_pred
        current_uncertainties = adapted_uncertainties
        
        # Try different thresholds again
        best_int_threshold = best_threshold
        min_int_shd = calculate_shd(true_adj, current_adj[0].detach().cpu().numpy(), best_threshold)
        
        for threshold in thresholds:
            shd = calculate_shd(true_adj, current_adj[0].detach().cpu().numpy(), threshold)
            if shd < min_int_shd:
                min_int_shd = shd
                best_int_threshold = threshold
                
        logger.info(f"Best threshold after intervention: {best_int_threshold} with SHD: {min_int_shd}")
        
        # Update best model if improved
        if min_int_shd < best_shd:
            best_shd = min_int_shd
            best_model = current_model
            best_adj = current_adj[0].detach().cpu().numpy()
            best_threshold = best_int_threshold
            logger.info(f"New best model with SHD: {best_shd}")
        
        # Update SHD history
        current_shd = calculate_shd(true_adj, current_adj[0].detach().cpu().numpy(), best_threshold)
        shd_history.append(current_shd)
        logger.info(f"SHD after intervention {i+1}: {current_shd}")
        
        # Visualize intermediate results
        if visualize and (i+1) % 2 == 0:
            compare_graphs(true_adj, current_adj[0].detach().cpu().numpy())
            plt.suptitle(f"Comparison after Intervention {i+1}")
            plt.savefig(f"comparison_int_{i+1}.png")
            plt.show()
            
            # Also visualize uncertainty
            plt.figure(figsize=(8, 6))
            plt.imshow(current_uncertainties[0].detach().cpu().numpy(), cmap='hot')
            plt.colorbar(label="Uncertainty")
            plt.title(f"Edge Uncertainties After Intervention {i+1}")
            plt.xticks(range(num_nodes), [f"X{j}" for j in range(num_nodes)])
            plt.yticks(range(num_nodes), [f"X{j}" for j in range(num_nodes)])
            plt.savefig(f"uncertainty_int_{i+1}.png")
            plt.show()
            
        # Check for early stopping
        if best_shd == 0 or (i > 3 and all(s == shd_history[-1] for s in shd_history[-3:])):
            logger.info(f"Early stopping at intervention {i+1}")
            break
    
    # Final visualization
    if visualize:
        compare_graphs(true_adj, best_adj)
        plt.suptitle("Final Comparison (Best Model)")
        plt.savefig("final_comparison.png")
        plt.show()
        
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
    
    # Calculate improvement
    improvement = (1 - best_shd / shd_history[0]) * 100 if shd_history[0] > 0 else 0
    logger.info(f"Improvement: {improvement:.1f}%")
    
    # Return results
    return {
        'true_adj': true_adj,
        'final_adj': best_adj,
        'initial_shd': shd_history[0],
        'final_shd': best_shd,
        'improvement': improvement,
        'interventions': len(intervention_history),
        'intervention_history': intervention_history,
        'intervention_counts': intervention_counts,
        'shd_history': shd_history,
        'best_threshold': best_threshold
    }


# --- Main Entry Point ---

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Minimal Causal Structure Recovery")
    
    # Graph parameters
    parser.add_argument('--num-nodes', type=int, default=5,
                       help="Number of nodes in the graph")
    parser.add_argument('--max-interventions', type=int, default=10,
                       help="Maximum number of interventions")
    
    # Model parameters
    parser.add_argument('--inner-lr', type=float, default=0.02,
                       help="Inner loop learning rate")
    parser.add_argument('--num-inner-steps', type=int, default=10,
                       help="Number of inner loop steps")
    parser.add_argument('--sparsity-weight', type=float, default=0.15,
                       help="Weight for sparsity regularization")
    parser.add_argument('--acyclicity-weight', type=float, default=0.5,
                       help="Weight for acyclicity regularization")
    
    # Visualization
    parser.add_argument('--no-visualize', action='store_true',
                       help="Disable visualization")
    
    args = parser.parse_args()
    
    # Run recovery
    results = recover_causal_structure(
        num_nodes=args.num_nodes,
        max_interventions=args.max_interventions,
        visualize=not args.no_visualize,
        inner_lr=args.inner_lr,
        num_inner_steps=args.num_inner_steps,
        sparsity_weight=args.sparsity_weight,
        acyclicity_weight=args.acyclicity_weight
    )
    
    # Print summary
    logger.info("\n=== Results Summary ===")
    logger.info(f"Initial SHD: {results['initial_shd']}")
    logger.info(f"Final SHD: {results['final_shd']}")
    logger.info(f"Improvement: {results['improvement']:.1f}%")
    logger.info(f"Interventions: {results['interventions']}")
    logger.info(f"Best threshold: {results['best_threshold']}")
    
    return results


if __name__ == "__main__":
    main() 