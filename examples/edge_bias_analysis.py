"""
Edge Prediction Bias Analysis

This script provides a minimal implementation to analyze the "no edge" prediction bias
in the SimpleGraphLearner model. It compares the original model with enhanced settings
that address the bias.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime

from causal_meta.structure_learning.simple_graph_learner import SimpleGraphLearner
from causal_meta.graph.generators.random_graphs import RandomGraphGenerator

# Create output directory for results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"results/edge_bias_analysis_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Configuration
NUM_NODES = 5
EDGE_PROBABILITY = 0.3
NUM_SAMPLES = 1000
HIDDEN_DIM = 64
NUM_LAYERS = 2
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_random_graph():
    """Generate a random DAG and convert to adjacency matrix."""
    print("Generating random DAG...")
    causal_graph = RandomGraphGenerator.random_dag(
        num_nodes=NUM_NODES,
        edge_probability=EDGE_PROBABILITY,
        seed=SEED
    )
    
    adjacency_matrix = np.zeros((NUM_NODES, NUM_NODES))
    for i in range(NUM_NODES):
        for j in causal_graph.get_children(i):
            adjacency_matrix[i, j] = 1
    
    # Calculate graph statistics
    total_possible_edges = NUM_NODES * (NUM_NODES - 1)  # Exclude self-loops
    actual_edges = adjacency_matrix.sum()
    edge_density = actual_edges / total_possible_edges
    
    print(f"Generated graph with {NUM_NODES} nodes and {int(actual_edges)} edges")
    print(f"Edge density: {edge_density:.4f} ({int(actual_edges)} / {total_possible_edges})")
    
    return adjacency_matrix

def generate_synthetic_data(adjacency_matrix, n_samples=1000):
    """Generate synthetic data based on a linear SCM."""
    print("Generating synthetic data...")
    n_nodes = adjacency_matrix.shape[0]
    
    # Create random linear weights for the SCM
    weights = np.random.uniform(0.5, 1.5, size=adjacency_matrix.shape)
    weights = weights * adjacency_matrix  # Zero out non-edges
    
    # Generate data
    data = np.zeros((n_samples, n_nodes))
    for i in range(n_samples):
        # Generate in topological order
        for j in range(n_nodes):
            # Compute the value based on parents
            parent_contribution = np.sum(data[i] * weights[:, j])
            # Add noise
            noise = np.random.normal(0, 0.1)
            data[i, j] = parent_contribution + noise
    
    return data

def calculate_metrics(pred_adj, true_adj):
    """Calculate evaluation metrics for predicted graph."""
    # Remove diagonal (self-loops)
    mask = ~np.eye(true_adj.shape[0], dtype=bool)
    pred_adj_flat = pred_adj[mask]
    true_adj_flat = true_adj[mask]
    
    # Calculate metrics
    tp = np.sum((pred_adj_flat == 1) & (true_adj_flat == 1))
    fp = np.sum((pred_adj_flat == 1) & (true_adj_flat == 0))
    tn = np.sum((pred_adj_flat == 0) & (true_adj_flat == 0))
    fn = np.sum((pred_adj_flat == 0) & (true_adj_flat == 1))
    
    # Derived metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Structural Hamming Distance
    shd = fp + fn
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'shd': shd,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }

def train_model(model, data, true_adj, epochs=100):
    """Train the model and collect diagnostics."""
    print(f"Training model: {type(model).__name__}")
    
    # Convert data to tensor
    data_tensor = torch.tensor(data, dtype=torch.float32)
    true_adj_tensor = torch.tensor(true_adj, dtype=torch.float32)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training diagnostics
    diagnostics = {
        'loss_history': [],
        'edge_prob_history': [],
        'loss_components': [],
        'metrics_history': []
    }
    
    # Move to device
    model.to(DEVICE)
    data_tensor = data_tensor.to(DEVICE)
    true_adj_tensor = true_adj_tensor.to(DEVICE)
    
    # Training loop
    for epoch in range(epochs):
        # Forward pass
        edge_probs = model(data_tensor)
        
        # Calculate loss
        loss, loss_components = model.calculate_loss(edge_probs, true_adj_tensor)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record diagnostics every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            # Get current edge probabilities
            with torch.no_grad():
                current_probs = model(data_tensor)
                current_probs_np = current_probs.cpu().numpy()
                
                # Apply threshold
                thresholded_adj = (current_probs_np > 0.5).astype(float)
                
                # Calculate metrics
                metrics = calculate_metrics(thresholded_adj, true_adj)
                
                # Calculate mean edge probability
                mean_prob = current_probs_np.mean()
                median_prob = np.median(current_probs_np)
                above_threshold = (current_probs_np > 0.5).mean()
                
                # Record components
                loss_dict = {k: v.item() for k, v in loss_components.items()}
                diagnostics['loss_history'].append(loss.item())
                diagnostics['edge_prob_history'].append({
                    'epoch': epoch,
                    'mean': mean_prob,
                    'median': median_prob,
                    'above_threshold': above_threshold
                })
                diagnostics['loss_components'].append({
                    'epoch': epoch,
                    **loss_dict
                })
                diagnostics['metrics_history'].append({
                    'epoch': epoch,
                    **metrics
                })
                
                # Print progress
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Mean Edge Prob = {mean_prob:.4f}, "
                      f"Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['f1']:.4f}")
                
                if epoch % 20 == 0 or epoch == epochs - 1:
                    # Save edge probability heatmap
                    plt.figure(figsize=(8, 6))
                    plt.imshow(current_probs_np, cmap='Blues', vmin=0, vmax=1)
                    plt.colorbar(label='Edge Probability')
                    plt.title(f"Edge Probabilities at Epoch {epoch}")
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"{model.model_type}_edge_probs_epoch_{epoch}.png"))
                    plt.close()
    
    return model, diagnostics

def plot_training_diagnostics(diagnostics_orig, diagnostics_enh):
    """Plot comparison of training diagnostics between original and enhanced models."""
    # Extract data
    epochs_orig = [d['epoch'] for d in diagnostics_orig['edge_prob_history']]
    mean_probs_orig = [d['mean'] for d in diagnostics_orig['edge_prob_history']]
    above_thresh_orig = [d['above_threshold'] for d in diagnostics_orig['edge_prob_history']]
    
    epochs_enh = [d['epoch'] for d in diagnostics_enh['edge_prob_history']]
    mean_probs_enh = [d['mean'] for d in diagnostics_enh['edge_prob_history']]
    above_thresh_enh = [d['above_threshold'] for d in diagnostics_enh['edge_prob_history']]
    
    # Plot mean edge probability
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_orig, mean_probs_orig, 'b-', label='Original Model')
    plt.plot(epochs_enh, mean_probs_enh, 'g-', label='Enhanced Model')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Edge Probability')
    plt.title('Evolution of Mean Edge Probability')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "mean_edge_prob_comparison.png"))
    plt.close()
    
    # Plot percentage above threshold
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_orig, above_thresh_orig, 'b-', label='Original Model')
    plt.plot(epochs_enh, above_thresh_enh, 'g-', label='Enhanced Model')
    plt.xlabel('Epoch')
    plt.ylabel('Fraction of Edges Above Threshold')
    plt.title('Evolution of Edges Above Threshold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "above_threshold_comparison.png"))
    plt.close()
    
    # Plot F1 score
    f1_orig = [d['f1'] for d in diagnostics_orig['metrics_history']]
    f1_enh = [d['f1'] for d in diagnostics_enh['metrics_history']]
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_orig, f1_orig, 'b-', label='Original Model')
    plt.plot(epochs_enh, f1_enh, 'g-', label='Enhanced Model')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Evolution of F1 Score')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "f1_score_comparison.png"))
    plt.close()
    
    # Plot loss components
    for model_type, diagnostics in [('original', diagnostics_orig), ('enhanced', diagnostics_enh)]:
        plt.figure(figsize=(12, 6))
        for component in ['supervised', 'sparsity', 'acyclicity']:
            if any(component in d for d in diagnostics['loss_components']):
                values = [d.get(component, 0) for d in diagnostics['loss_components']]
                epochs = [d['epoch'] for d in diagnostics['loss_components']]
                plt.plot(epochs, values, label=component)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss Component Value')
        plt.yscale('log')  # Log scale for better visualization
        plt.title(f'{model_type.capitalize()} Model Loss Components')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"{model_type}_loss_components.png"))
        plt.close()

def compare_final_distributions(model_orig, model_enh, data, true_adj):
    """Compare the final edge probability distributions of both models."""
    # Get final predictions
    with torch.no_grad():
        data_tensor = torch.tensor(data, dtype=torch.float32).to(DEVICE)
        
        probs_orig = model_orig(data_tensor).cpu().numpy()
        probs_enh = model_enh(data_tensor).cpu().numpy()
        
        # Calculate metrics
        thresh_orig = (probs_orig > 0.5).astype(float)
        thresh_enh = (probs_enh > 0.5).astype(float)
        
        metrics_orig = calculate_metrics(thresh_orig, true_adj)
        metrics_enh = calculate_metrics(thresh_enh, true_adj)
    
    # Plot distributions side by side
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Create masks for true edges and non-edges
    edge_mask = (true_adj == 1)
    non_edge_mask = (true_adj == 0) & ~np.eye(true_adj.shape[0], dtype=bool)  # Exclude diagonal
    
    # Original model
    true_edge_probs_orig = probs_orig[edge_mask]
    non_edge_probs_orig = probs_orig[non_edge_mask]
    
    if len(true_edge_probs_orig) > 0:
        axes[0].hist(true_edge_probs_orig, bins=10, alpha=0.7, color='green', 
                    label=f'True Edges (n={len(true_edge_probs_orig)})')
    
    if len(non_edge_probs_orig) > 0:
        axes[0].hist(non_edge_probs_orig, bins=10, alpha=0.7, color='red', 
                    label=f'Non-Edges (n={len(non_edge_probs_orig)})')
    
    axes[0].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    axes[0].set_title('Original Model')
    axes[0].set_xlabel('Edge Probability')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    
    # Enhanced model
    true_edge_probs_enh = probs_enh[edge_mask]
    non_edge_probs_enh = probs_enh[non_edge_mask]
    
    if len(true_edge_probs_enh) > 0:
        axes[1].hist(true_edge_probs_enh, bins=10, alpha=0.7, color='green', 
                    label=f'True Edges (n={len(true_edge_probs_enh)})')
    
    if len(non_edge_probs_enh) > 0:
        axes[1].hist(non_edge_probs_enh, bins=10, alpha=0.7, color='red', 
                    label=f'Non-Edges (n={len(non_edge_probs_enh)})')
    
    axes[1].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    axes[1].set_title('Enhanced Model')
    axes[1].set_xlabel('Edge Probability')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "edge_prob_distribution_comparison.png"))
    plt.close()
    
    # Print final metrics comparison
    print("\n=== Final Metrics Comparison ===")
    print("Metric\t\tOriginal\tEnhanced")
    print("-" * 40)
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'shd']:
        orig_val = metrics_orig[metric]
        enh_val = metrics_enh[metric]
        print(f"{metric.capitalize()}\t\t{orig_val:.4f}\t\t{enh_val:.4f}")
    
    print("\n=== Confusion Matrix Elements ===")
    print("Element\t\tOriginal\tEnhanced")
    print("-" * 40)
    for element in ['tp', 'fp', 'tn', 'fn']:
        orig_val = metrics_orig[element]
        enh_val = metrics_enh[element]
        print(f"{element.upper()}\t\t{int(orig_val)}\t\t{int(enh_val)}")
    
    # Save metrics to file
    with open(os.path.join(output_dir, "final_metrics.txt"), "w") as f:
        f.write("=== Final Metrics Comparison ===\n")
        f.write("Metric\t\tOriginal\tEnhanced\n")
        f.write("-" * 40 + "\n")
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'shd']:
            orig_val = metrics_orig[metric]
            enh_val = metrics_enh[metric]
            f.write(f"{metric.capitalize()}\t\t{orig_val:.4f}\t\t{enh_val:.4f}\n")
        
        f.write("\n=== Confusion Matrix Elements ===\n")
        f.write("Element\t\tOriginal\tEnhanced\n")
        f.write("-" * 40 + "\n")
        for element in ['tp', 'fp', 'tn', 'fn']:
            orig_val = metrics_orig[element]
            enh_val = metrics_enh[element]
            f.write(f"{element.upper()}\t\t{int(orig_val)}\t\t{int(enh_val)}\n")
    
    # Plot comparison heatmaps of adjacency matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # True adjacency matrix
    im0 = axes[0].imshow(true_adj, cmap='Blues', vmin=0, vmax=1)
    axes[0].set_title('True Graph')
    axes[0].set_xlabel('Target Node')
    axes[0].set_ylabel('Source Node')
    
    # Original model prediction
    im1 = axes[1].imshow(thresh_orig, cmap='Blues', vmin=0, vmax=1)
    axes[1].set_title('Original Model Prediction')
    axes[1].set_xlabel('Target Node')
    axes[1].set_ylabel('Source Node')
    
    # Enhanced model prediction
    im2 = axes[2].imshow(thresh_enh, cmap='Blues', vmin=0, vmax=1)
    axes[2].set_title('Enhanced Model Prediction')
    axes[2].set_xlabel('Target Node')
    axes[2].set_ylabel('Source Node')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "adjacency_matrix_comparison.png"))
    plt.close()


def main():
    """Run the edge prediction bias analysis experiment."""
    print(f"Starting edge prediction bias analysis... (Results in {output_dir})")
    
    # Generate random DAG and data
    adj_matrix = generate_random_graph()
    synthetic_data = generate_synthetic_data(adj_matrix)
    
    # Create original model (with bias toward no edges)
    print("\nCreating original model (with no-edge bias)...")
    model_orig = SimpleGraphLearner(
        input_dim=NUM_NODES,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        sparsity_weight=0.1,  # Original strong sparsity regularization
        acyclicity_weight=1.0,
        pos_weight=1.0,  # No class balancing
        consistency_weight=0.0,
        edge_prob_bias=0.0,
        expected_density=None
    )
    model_orig.model_type = "original"
    
    # Create enhanced model (with bias mitigation)
    print("\nCreating enhanced model (with bias mitigation)...")
    model_enh = SimpleGraphLearner(
        input_dim=NUM_NODES,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        sparsity_weight=0.05,  # Reduced sparsity weight
        acyclicity_weight=1.0,
        pos_weight=5.0,  # Weight positive examples 5x more
        consistency_weight=0.1,
        edge_prob_bias=0.1,  # Bias toward some edge predictions
        expected_density=0.3
    )
    model_enh.model_type = "enhanced"
    
    # Train both models
    model_orig, diagnostics_orig = train_model(model_orig, synthetic_data, adj_matrix, EPOCHS)
    model_enh, diagnostics_enh = train_model(model_enh, synthetic_data, adj_matrix, EPOCHS)
    
    # Plot training diagnostics
    print("\nGenerating diagnostic visualizations...")
    plot_training_diagnostics(diagnostics_orig, diagnostics_enh)
    
    # Compare final distributions
    compare_final_distributions(model_orig, model_enh, synthetic_data, adj_matrix)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main() 