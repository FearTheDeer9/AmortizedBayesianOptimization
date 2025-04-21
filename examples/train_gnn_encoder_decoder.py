#!/usr/bin/env python
"""
Example script for training a GNN encoder-decoder model.

This script demonstrates how to create, train, and evaluate a GNN-based
encoder-decoder model for graph representation learning.
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from causal_meta.inference.models.wrapper import create_model
from causal_meta.inference.training.data_loader import RandomGraphGenerator, create_data_loaders
from causal_meta.inference.training.trainer import train_model
from causal_meta.inference.training.visualization import (
    plot_training_history,
    plot_graph_reconstruction_examples,
    visualize_latent_space,
    plot_confusion_matrix
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train GNN encoder-decoder model')

    # Model parameters
    parser.add_argument('--architecture', type=str, default='gcn',
                        choices=['gcn', 'gat', 'gin'],
                        help='GNN architecture to use')
    parser.add_argument('--input-dim', type=int, default=3,
                        help='Input node feature dimension')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden layer dimension')
    parser.add_argument('--latent-dim', type=int, default=32,
                        help='Latent representation dimension')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of GNN layers')

    # Dataset parameters
    parser.add_argument('--num-graphs', type=int, default=100,
                        help='Number of graphs to generate')
    parser.add_argument('--min-nodes', type=int, default=10,
                        help='Minimum number of nodes per graph')
    parser.add_argument('--max-nodes', type=int, default=20,
                        help='Maximum number of nodes per graph')
    parser.add_argument('--graph-type', type=str, default='erdos_renyi',
                        choices=['erdos_renyi',
                                 'barabasi_albert', 'watts_strogatz'],
                        help='Type of graph to generate')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--loss-type', type=str, default='weighted_bce',
                        choices=['bce', 'mse', 'weighted_bce'],
                        help='Loss function for edge prediction')
    parser.add_argument('--edge-weight', type=float, default=1.0,
                        help='Weight for edge prediction loss')
    parser.add_argument('--feature-weight', type=float, default=0.5,
                        help='Weight for node feature reconstruction loss')
    parser.add_argument('--l2-weight', type=float, default=0.0,
                        help='Weight for L2 regularization')
    parser.add_argument('--early-stopping', type=int, default=10,
                        help='Patience for early stopping')

    # Output parameters
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training')

    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()

    # Setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if use_cuda:
        torch.cuda.manual_seed(42)

    print(f"Using device: {device}")
    print(f"Generating {args.num_graphs} {args.graph_type} graphs...")

    # Generate random graphs
    graphs = RandomGraphGenerator.generate_dataset(
        num_graphs=args.num_graphs,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        graph_type=args.graph_type,
        node_features_dim=args.input_dim
    )

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset=graphs,
        batch_size=args.batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        shuffle=True
    )

    print(f"Created data loaders with {len(train_loader.dataset)} training, "
          f"{len(val_loader.dataset)} validation, and {len(test_loader.dataset)} test graphs")

    # Determine maximum number of nodes for the decoder
    max_nodes = args.max_nodes

    # Create model
    model = create_model(
        architecture=args.architecture,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        output_dim=args.input_dim,  # Output dim = input dim for reconstruction
        num_nodes=max_nodes,
        # Encoder parameters
        encoder_num_layers=args.num_layers,
        encoder_dropout=0.1,
        encoder_activation='relu',
        encoder_batch_norm=True,
        # Decoder parameters
        decoder_num_layers=args.num_layers,
        decoder_dropout=0.1,
        decoder_activation='relu',
        decoder_edge_prediction_method='inner_product',
        decoder_batch_norm=True,
        # Wrapper parameters
        loss_type=args.loss_type,
        edge_weight=args.edge_weight,
        feature_weight=args.feature_weight,
        l2_weight=args.l2_weight
    )

    print(f"Created {args.architecture.upper()} model with:"
          f"\n  - {args.input_dim} input dimensions"
          f"\n  - {args.hidden_dim} hidden dimensions"
          f"\n  - {args.latent_dim} latent dimensions"
          f"\n  - {args.num_layers} layers")

    # Setup training directories
    log_dir = output_dir / 'logs'
    checkpoint_dir = output_dir / 'checkpoints'
    plot_dir = output_dir / 'plots'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Train model
    print(f"Training model for {args.epochs} epochs...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
        device=device,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir
    )

    # Plot training history
    print("Generating training plots...")
    history_fig = plot_training_history(
        history=history,
        metrics=['loss', 'edge_f1', 'reconstruction_accuracy'],
        save_path=plot_dir / 'training_history.png'
    )
    plt.close(history_fig)

    # Plot reconstruction examples
    examples_fig = plot_graph_reconstruction_examples(
        model=model,
        dataset=test_loader.dataset,
        num_examples=3,
        save_path=plot_dir / 'reconstruction_examples.png'
    )
    plt.close(examples_fig)

    # Visualize latent space
    latent_fig = visualize_latent_space(
        model=model,
        dataset=test_loader.dataset,
        method='pca',
        save_path=plot_dir / 'latent_space.png'
    )
    plt.close(latent_fig)

    # Plot confusion matrix
    cm_fig = plot_confusion_matrix(
        model=model,
        dataset=test_loader.dataset,
        save_path=plot_dir / 'confusion_matrix.png'
    )
    plt.close(cm_fig)

    print(f"Training complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
