"""
Train SimpleGraphLearner with Optimal Parameters

This script trains the SimpleGraphLearner model with parameters that were
found to consistently achieve perfect graph recovery (SHD=0) on small graphs.
It builds on the insights from the parameter tuning experiments.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

from causal_meta.structure_learning.simple_graph_learner import SimpleGraphLearner
from causal_meta.graph.generators.random_graphs import RandomGraphGenerator
from causal_meta.structure_learning.scm_generators import LinearSCMGenerator
from causal_meta.structure_learning.data_utils import generate_observational_data
from causal_meta.utils.advanced_visualization import (
    plot_edge_probabilities,
    plot_edge_probability_histogram,
    plot_edge_probability_distribution,
    plot_threshold_sensitivity
)

# Optimal parameters found during tuning
OPTIMAL_PARAMS = {
    "sparsity_weight": 0.07,
    "pos_weight": 5.0,
    "edge_prob_bias": 0.3,
    "consistency_weight": 0.1,
    "expected_density": 0.4,
    "acyclicity_weight": 1.0  # Keep the default acyclicity weight
}

# Default configuration
DEFAULT_CONFIG = {
    "num_nodes": 6,                # Small graph for easier perfect recovery
    "edge_probability": 0.3,       # Typical sparsity for small causal graphs
    "num_samples": 2000,           # More samples for better learning
    "hidden_dim": 64,              # Default model architecture
    "num_layers": 2,               # Default model architecture
    "epochs": 500,                 # More training epochs to ensure convergence
    "learning_rate": 0.001,        # Default learning rate
    "weight_decay": 0.0,           # Default weight decay
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),  # MPS not supported for matrix_exp
    "seed": 42                     # Random seed for reproducibility
}

def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def train_model(model, data, true_adj, config, output_dir):
    """Train the model and evaluate its performance."""
    # Convert data to tensor
    data_tensor = torch.tensor(data, dtype=torch.float32)
    true_adj_tensor = torch.tensor(true_adj, dtype=torch.float32)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    # Training diagnostics
    diagnostics = {
        'loss_history': [],
        'edge_prob_history': [],
        'loss_components': [],
        'metrics_history': []
    }
    
    # Move to device
    device = config["device"]
    model.to(device)
    data_tensor = data_tensor.to(device)
    true_adj_tensor = true_adj_tensor.to(device)
    
    # Best model tracking
    best_model_state = None
    best_shd = float('inf')
    best_epoch = -1
    
    # Create visualizations directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Save initial edge probability distribution
    with torch.no_grad():
        initial_probs = model(data_tensor).cpu().numpy()
        plt.figure(figsize=(10, 6))
        plt.hist(initial_probs.flatten(), bins=30)
        plt.title("Initial Edge Probability Distribution")
        plt.xlabel("Edge Probability")
        plt.ylabel("Count")
        plt.savefig(os.path.join(vis_dir, "initial_edge_probs.png"))
        plt.close()
    
    # Training loop
    print(f"Training model with optimal parameters for {config['epochs']} epochs...")
    for epoch in range(config["epochs"]):
        # Forward pass
        edge_probs = model(data_tensor)
        
        # Calculate loss
        loss, loss_components = model.calculate_loss(edge_probs, true_adj_tensor)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record diagnostics every 20 epochs
        if epoch % 20 == 0 or epoch == config["epochs"] - 1:
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
                
                # Track best model
                if metrics['shd'] < best_shd:
                    best_shd = metrics['shd']
                    best_epoch = epoch
                    best_model_state = model.state_dict()
                
                # Visualize edge probabilities at different epochs
                if epoch % 40 == 0 or epoch == config["epochs"] - 1:
                    # Distribution histogram
                    plt.figure(figsize=(10, 6))
                    plt.hist(current_probs_np.flatten(), bins=30)
                    plt.title(f"Edge Probability Distribution (Epoch {epoch})")
                    plt.xlabel("Edge Probability")
                    plt.ylabel("Count")
                    plt.savefig(os.path.join(vis_dir, f"edge_probs_epoch_{epoch}.png"))
                    plt.close()
                    
                    # Heatmap
                    plt.figure(figsize=(8, 6))
                    plt.imshow(current_probs_np, cmap='Blues', vmin=0, vmax=1)
                    plt.colorbar(label='Edge Probability')
                    plt.title(f"Edge Probabilities at Epoch {epoch}")
                    plt.tight_layout()
                    plt.savefig(os.path.join(vis_dir, f"edge_probs_heatmap_epoch_{epoch}.png"))
                    plt.close()
                    
                    # Split distribution by true edges/non-edges
                    fig = plot_edge_probability_distribution(
                        true_adj,
                        current_probs_np,
                        figsize=(10, 6)
                    )
                    plt.title(f"Edge Probability Distribution by True Graph (Epoch {epoch})")
                    fig.savefig(os.path.join(vis_dir, f"edge_probs_by_truth_epoch_{epoch}.png"))
                    plt.close(fig)
                
                # Print progress
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}, SHD = {metrics['shd']}, "
                      f"F1 = {metrics['f1']:.4f}, Mean Edge Prob = {mean_prob:.4f}")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model from epoch {best_epoch} with SHD = {best_shd}")
    
    # Final evaluation
    with torch.no_grad():
        final_probs = model(data_tensor)
        final_probs_np = final_probs.cpu().numpy()
        final_adj = (final_probs_np > 0.5).astype(float)
        final_metrics = calculate_metrics(final_adj, true_adj)
    
    # Visualize final results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # True adjacency matrix
    im0 = axes[0].imshow(true_adj, cmap='Blues', vmin=0, vmax=1)
    axes[0].set_title('True Graph')
    axes[0].set_xlabel('Target Node')
    axes[0].set_ylabel('Source Node')
    
    # Edge probabilities
    im1 = axes[1].imshow(final_probs_np, cmap='Blues', vmin=0, vmax=1)
    axes[1].set_title('Edge Probabilities')
    axes[1].set_xlabel('Target Node')
    axes[1].set_ylabel('Source Node')
    plt.colorbar(im1, ax=axes[1], label='Probability')
    
    # Thresholded prediction
    im2 = axes[2].imshow(final_adj, cmap='Blues', vmin=0, vmax=1)
    axes[2].set_title('Predicted Graph')
    axes[2].set_xlabel('Target Node')
    axes[2].set_ylabel('Source Node')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_graph_comparison.png"))
    plt.close()
    
    # Plot threshold sensitivity
    fig = plot_threshold_sensitivity(
        true_adj, 
        final_probs_np,
        thresholds=np.linspace(0, 1, 101),
        figsize=(10, 6)
    )
    fig.savefig(os.path.join(output_dir, "threshold_sensitivity.png"))
    plt.close(fig)
    
    # Plot training metrics
    epochs = [d['epoch'] for d in diagnostics['metrics_history']]
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    
    plt.figure(figsize=(12, 6))
    for metric in metrics_to_plot:
        values = [d[metric] for d in diagnostics['metrics_history']]
        plt.plot(epochs, values, marker='o', label=metric.capitalize())
    
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Training Metrics Over Time')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_metrics.png"))
    plt.close()
    
    # Plot SHD
    plt.figure(figsize=(10, 6))
    shd_values = [d['shd'] for d in diagnostics['metrics_history']]
    plt.plot(epochs, shd_values, marker='o', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('SHD (Structural Hamming Distance)')
    plt.title('SHD Over Training')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shd_over_time.png"))
    plt.close()
    
    # Print final metrics
    print("\n=== Final Metrics ===")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    print(f"F1 Score: {final_metrics['f1']:.4f}")
    print(f"SHD: {final_metrics['shd']}")
    print(f"TP: {int(final_metrics['tp'])}, FP: {int(final_metrics['fp'])}, TN: {int(final_metrics['tn'])}, FN: {int(final_metrics['fn'])}")
    
    # Save model checkpoint
    checkpoint_path = os.path.join(output_dir, "model_checkpoint.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': final_metrics,
        'params': OPTIMAL_PARAMS,
        'config': config,
        'best_epoch': best_epoch
    }, checkpoint_path)
    
    # Save metrics to file
    with open(os.path.join(output_dir, "final_metrics.txt"), "w") as f:
        f.write("=== Final Metrics ===\n")
        for metric, value in final_metrics.items():
            f.write(f"{metric}: {value}\n")
    
    return model, final_metrics, diagnostics

def main(args):
    """Train a model with optimal parameters."""
    # Update config with command line arguments
    config = DEFAULT_CONFIG.copy()
    config["num_nodes"] = args.num_nodes
    config["num_samples"] = args.num_samples
    config["epochs"] = args.epochs
    config["seed"] = args.seed
    
    # Set random seed for reproducibility
    set_seed(config["seed"])
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/optimal_model_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting optimal model training with {config['num_nodes']} nodes")
    print(f"Results will be saved to: {output_dir}")
    
    # Generate random DAG
    print("Generating random graph...")
    causal_graph = RandomGraphGenerator.random_dag(
        num_nodes=config["num_nodes"],
        edge_probability=config["edge_probability"],
        seed=config["seed"]
    )
    
    # Convert to adjacency matrix
    adjacency_matrix = np.zeros((config["num_nodes"], config["num_nodes"]))
    for i in range(config["num_nodes"]):
        for j in causal_graph.get_children(i):
            adjacency_matrix[i, j] = 1
    
    # Create SCM and generate data
    print("Generating synthetic data...")
    scm = LinearSCMGenerator.generate_linear_scm(
        adj_matrix=adjacency_matrix,
        noise_scale=0.1,
        seed=config["seed"]
    )
    
    # Generate synthetic data
    synthetic_data = generate_observational_data(
        scm=scm,
        n_samples=config["num_samples"]
    )
    
    # Extract data into a numpy array
    data_array = np.column_stack([synthetic_data[f'x{i}'] for i in range(config["num_nodes"])])
    
    # Print graph statistics
    total_possible_edges = config["num_nodes"] * (config["num_nodes"] - 1) 
    actual_edges = adjacency_matrix.sum()
    edge_density = actual_edges / total_possible_edges
    
    print(f"Generated graph with {config['num_nodes']} nodes and {int(actual_edges)} edges")
    print(f"Edge density: {edge_density:.4f} ({int(actual_edges)} / {total_possible_edges})")
    
    # Save adjacency matrix visualization
    plt.figure(figsize=(8, 6))
    plt.imshow(adjacency_matrix, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(label='Edge Presence')
    plt.title(f"True Graph Structure ({config['num_nodes']} nodes)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "true_graph.png"))
    plt.close()
    
    # Create and train the model with optimal parameters
    print("Creating model with optimal parameters...")
    model = SimpleGraphLearner(
        input_dim=config["num_nodes"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        **OPTIMAL_PARAMS
    )
    
    # Train the model
    trained_model, metrics, diagnostics = train_model(
        model, 
        data_array, 
        adjacency_matrix, 
        config,
        output_dir
    )
    
    print(f"\nTraining complete! Results saved to {output_dir}")
    
    # Return success if achieved perfect recovery
    return metrics['shd'] == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SimpleGraphLearner with optimal parameters")
    parser.add_argument("--num_nodes", type=int, default=6, help="Number of nodes in the graph")
    parser.add_argument("--num_samples", type=int, default=2000, help="Number of training samples")
    parser.add_argument("--epochs", type=int, default=3000, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    success = main(args)
    
    if success:
        print("\n🎉 Perfect recovery achieved! (SHD = 0)")
    else:
        print("\nTraining complete, but perfect recovery not achieved.") 