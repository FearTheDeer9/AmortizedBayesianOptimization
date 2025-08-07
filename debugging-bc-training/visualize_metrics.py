"""
Visualization tools for BC training metrics.

This module provides functions to visualize training metrics, embeddings,
and performance patterns.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_training_curves(metrics_history: List[Dict], output_dir: Path):
    """
    Plot training and validation curves over epochs.
    
    Args:
        metrics_history: List of epoch metrics
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = [m['epoch'] for m in metrics_history]
    
    # Plot accuracy curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training accuracy
    if 'train_accuracy' in metrics_history[0]:
        train_acc = [m.get('train_accuracy', 0) for m in metrics_history]
        axes[0, 0].plot(epochs, train_acc, 'b-', label='Train')
        
        if 'val_accuracy' in metrics_history[0]:
            val_acc = [m.get('val_accuracy', 0) for m in metrics_history]
            axes[0, 0].plot(epochs, val_acc, 'r-', label='Validation')
        
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Variable Selection Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # F1 Score
    if 'val_f1' in metrics_history[0]:
        val_f1 = [m.get('val_f1', 0) for m in metrics_history]
        axes[0, 1].plot(epochs, val_f1, 'g-')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('Validation F1 Score')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Precision and Recall
    if 'val_precision' in metrics_history[0]:
        val_precision = [m.get('val_precision', 0) for m in metrics_history]
        val_recall = [m.get('val_recall', 0) for m in metrics_history]
        axes[1, 0].plot(epochs, val_precision, 'b-', label='Precision')
        axes[1, 0].plot(epochs, val_recall, 'r-', label='Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Precision and Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Value prediction MSE (if available)
    if 'train_value_mse' in metrics_history[0]:
        train_mse = [m.get('train_value_mse', 0) for m in metrics_history]
        axes[1, 1].plot(epochs, train_mse, 'b-')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MSE')
        axes[1, 1].set_title('Value Prediction MSE')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training curves to {output_dir / 'training_curves.png'}")


def plot_confusion_matrix(confusion_matrices: Dict[int, Dict], output_dir: Path):
    """
    Plot confusion matrices for different epochs.
    
    Args:
        confusion_matrices: Dictionary of confusion matrices by epoch
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not confusion_matrices:
        print("No confusion matrices to plot")
        return
    
    # Get the latest confusion matrix
    latest_epoch = max(confusion_matrices.keys())
    cm_data = confusion_matrices[latest_epoch]
    cm = cm_data['matrix']
    labels = cm_data.get('labels', [f'Var{i}' for i in range(cm.shape[0])])
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={'label': 'Proportion'})
    
    ax.set_xlabel('Predicted Variable')
    ax.set_ylabel('True Variable')
    ax.set_title(f'Confusion Matrix (Epoch {latest_epoch})')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_matrix_epoch_{latest_epoch}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confusion matrix to {output_dir / f'confusion_matrix_epoch_{latest_epoch}.png'}")


def plot_per_variable_performance(per_variable_stats: Dict[str, Dict], output_dir: Path):
    """
    Plot per-variable performance statistics.
    
    Args:
        per_variable_stats: Dictionary of stats per variable
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not per_variable_stats:
        print("No per-variable statistics to plot")
        return
    
    # Extract variable names and accuracies
    var_names = list(per_variable_stats.keys())
    accuracies = []
    attempts = []
    
    for var_name in var_names:
        stats = per_variable_stats[var_name]
        if stats['attempts'] > 0:
            accuracies.append(stats['correct'] / stats['attempts'])
            attempts.append(stats['attempts'])
        else:
            accuracies.append(0)
            attempts.append(0)
    
    # Create bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy per variable
    colors = ['green' if acc >= 0.8 else 'orange' if acc >= 0.5 else 'red' 
              for acc in accuracies]
    bars1 = ax1.bar(range(len(var_names)), accuracies, color=colors)
    ax1.set_xticks(range(len(var_names)))
    ax1.set_xticklabels(var_names, rotation=45, ha='right')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Per-Variable Selection Accuracy')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylim([0, 1])
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.2f}', ha='center', va='bottom')
    
    # Attempts per variable
    bars2 = ax2.bar(range(len(var_names)), attempts, color='steelblue')
    ax2.set_xticks(range(len(var_names)))
    ax2.set_xticklabels(var_names, rotation=45, ha='right')
    ax2.set_ylabel('Number of Attempts')
    ax2.set_title('Selection Attempts per Variable')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_variable_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved per-variable performance to {output_dir / 'per_variable_performance.png'}")


def plot_embedding_evolution(embeddings_history: Dict[int, List], output_dir: Path,
                            method: str = 'tsne', perplexity: int = 30):
    """
    Visualize embedding evolution over training epochs.
    
    Args:
        embeddings_history: Dictionary of embeddings by epoch
        output_dir: Directory to save plots
        method: Dimensionality reduction method ('tsne' or 'pca')
        perplexity: Perplexity parameter for t-SNE
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not embeddings_history:
        print("No embeddings to visualize")
        return
    
    # Select epochs to visualize (up to 4)
    epochs = sorted(embeddings_history.keys())
    selected_epochs = epochs[::max(1, len(epochs)//4)][:4]
    
    fig, axes = plt.subplots(1, len(selected_epochs), figsize=(5*len(selected_epochs), 5))
    if len(selected_epochs) == 1:
        axes = [axes]
    
    for idx, epoch in enumerate(selected_epochs):
        embeddings_list = embeddings_history[epoch]
        if not embeddings_list:
            continue
        
        # Concatenate embeddings
        all_embeddings = np.concatenate(embeddings_list, axis=0)
        
        # Limit number of points for visualization
        n_samples = min(500, len(all_embeddings))
        if n_samples < len(all_embeddings):
            indices = np.random.choice(len(all_embeddings), n_samples, replace=False)
            all_embeddings = all_embeddings[indices]
        
        # Reduce dimensionality
        if method == 'tsne' and all_embeddings.shape[0] > perplexity:
            reducer = TSNE(n_components=2, perplexity=min(perplexity, n_samples-1), 
                          random_state=42)
        else:
            reducer = PCA(n_components=2)
        
        try:
            embeddings_2d = reducer.fit_transform(all_embeddings)
            
            # Plot
            scatter = axes[idx].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                      c=range(len(embeddings_2d)), cmap='viridis',
                                      alpha=0.6, s=20)
            axes[idx].set_title(f'Epoch {epoch}')
            axes[idx].set_xlabel('Component 1')
            axes[idx].set_ylabel('Component 2')
            
            # Add colorbar for the last plot
            if idx == len(selected_epochs) - 1:
                plt.colorbar(scatter, ax=axes[idx], label='Sample Index')
        except Exception as e:
            print(f"Error visualizing embeddings for epoch {epoch}: {e}")
            axes[idx].text(0.5, 0.5, 'Error in visualization', 
                          ha='center', va='center', transform=axes[idx].transAxes)
    
    plt.suptitle(f'Embedding Evolution ({method.upper()})')
    plt.tight_layout()
    plt.savefig(output_dir / f'embedding_evolution_{method}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved embedding evolution to {output_dir / f'embedding_evolution_{method}.png'}")


def plot_metrics_comparison(metrics_file_paths: List[Path], labels: List[str], output_dir: Path):
    """
    Compare metrics from multiple training runs.
    
    Args:
        metrics_file_paths: List of paths to metrics files
        labels: Labels for each run
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for path, label in zip(metrics_file_paths, labels):
        if not path.exists():
            print(f"Metrics file not found: {path}")
            continue
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        metrics_history = data['epoch_metrics']
        if not metrics_history:
            continue
        
        epochs = [m['epoch'] for m in metrics_history]
        
        # Plot different metrics
        if 'val_accuracy' in metrics_history[0]:
            val_acc = [m.get('val_accuracy', 0) for m in metrics_history]
            axes[0, 0].plot(epochs, val_acc, label=label)
        
        if 'val_f1' in metrics_history[0]:
            val_f1 = [m.get('val_f1', 0) for m in metrics_history]
            axes[0, 1].plot(epochs, val_f1, label=label)
        
        if 'val_precision' in metrics_history[0]:
            val_precision = [m.get('val_precision', 0) for m in metrics_history]
            axes[1, 0].plot(epochs, val_precision, label=label)
        
        if 'val_recall' in metrics_history[0]:
            val_recall = [m.get('val_recall', 0) for m in metrics_history]
            axes[1, 1].plot(epochs, val_recall, label=label)
    
    # Configure subplots
    axes[0, 0].set_title('Validation Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Validation F1 Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Validation Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Validation Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training Runs Comparison')
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved metrics comparison to {output_dir / 'metrics_comparison.png'}")


def generate_all_visualizations(metrics_file: Path, output_dir: Path):
    """
    Generate all available visualizations from a metrics file.
    
    Args:
        metrics_file: Path to metrics pickle file
        output_dir: Directory to save all plots
    """
    print(f"Loading metrics from {metrics_file}")
    
    with open(metrics_file, 'rb') as f:
        data = pickle.load(f)
    
    print("Generating visualizations...")
    
    # Training curves
    if 'epoch_metrics' in data and data['epoch_metrics']:
        plot_training_curves(data['epoch_metrics'], output_dir)
    
    # Confusion matrices
    if 'confusion_matrices' in data and data['confusion_matrices']:
        plot_confusion_matrix(data['confusion_matrices'], output_dir)
    
    # Per-variable performance
    if 'per_variable_stats' in data and data['per_variable_stats']:
        plot_per_variable_performance(data['per_variable_stats'], output_dir)
    
    # Embedding evolution
    if 'embeddings_history' in data and data['embeddings_history']:
        plot_embedding_evolution(data['embeddings_history'], output_dir, method='pca')
        if len(next(iter(data['embeddings_history'].values()))[0]) > 30:
            plot_embedding_evolution(data['embeddings_history'], output_dir, method='tsne')
    
    print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize BC training metrics')
    parser.add_argument('--metrics_file', type=str, required=True,
                       help='Path to metrics pickle file')
    parser.add_argument('--output_dir', type=str, 
                       default='debugging-bc-training/results/plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    metrics_file = Path(args.metrics_file)
    output_dir = Path(args.output_dir)
    
    if not metrics_file.exists():
        print(f"Error: Metrics file not found: {metrics_file}")
        exit(1)
    
    generate_all_visualizations(metrics_file, output_dir)