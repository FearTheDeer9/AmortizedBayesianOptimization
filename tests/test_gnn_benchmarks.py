"""
Benchmarking tests for GNN encoder-decoder architectures.

This module provides performance benchmarks for different GNN architectures,
comparing reconstruction quality, training time, and latent space properties.
"""

import pytest
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader

from causal_meta.inference.models.wrapper import EncoderDecoderWrapper, create_model
from causal_meta.inference.models.encoder import edge_f1_score, reconstruction_accuracy


# Custom collate function for PyTorch Geometric Data objects
def pyg_collate(batch):
    return Batch.from_data_list(batch)


# ================ Fixtures ================

@pytest.fixture
def test_dir():
    """Create a test directory for saving benchmark outputs."""
    test_dir = Path("benchmark_outputs")
    test_dir.mkdir(exist_ok=True)
    yield test_dir
    # Uncomment to clean up after tests
    # import shutil
    # shutil.rmtree(test_dir)


@pytest.fixture
def synthetic_benchmark_dataset():
    """Create a synthetic dataset of graphs for benchmarking."""
    num_graphs = 50
    feature_dim = 4
    min_nodes, max_nodes = 10, 20
    graphs = []

    for i in range(num_graphs):
        # Create random graph with varying number of nodes
        num_nodes = np.random.randint(min_nodes, max_nodes + 1)
        x = torch.randn(num_nodes, feature_dim)

        # Create random edges (with controlled density)
        edge_index = []
        edge_density = np.random.uniform(0.1, 0.3)  # 10-30% of possible edges

        for src in range(num_nodes):
            for dst in range(num_nodes):
                if src != dst and np.random.rand() < edge_density:
                    edge_index.append([src, dst])

        if not edge_index:  # Ensure at least one edge
            i, j = np.random.choice(num_nodes, 2, replace=False)
            edge_index.append([i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()

        # Add graph-level class label
        y = torch.tensor([i % 3], dtype=torch.float)  # 3 classes

        # Create graph
        graph = Data(x=x, edge_index=edge_index, y=y)
        graphs.append(graph)

    return graphs


@pytest.fixture
def architecture_configurations():
    """Define configurations for different architecture benchmarks."""
    return {
        'gcn': {
            'encoder_kwargs': {
                'num_layers': 3,
                'dropout': 0.1,
                'activation': 'relu',
                'pooling': 'mean',
                'batch_norm': True
            },
            'decoder_kwargs': {
                'num_layers': 3,
                'dropout': 0.1,
                'activation': 'relu',
                'edge_prediction_method': 'mlp',
                'batch_norm': True
            }
        },
        'gat': {
            'encoder_kwargs': {
                'num_layers': 3,
                'dropout': 0.1,
                'activation': 'relu',
                'pooling': 'mean',
                'batch_norm': True,
                'heads': 4,
                'concat_heads': True
            },
            'decoder_kwargs': {
                'num_layers': 3,
                'dropout': 0.1,
                'activation': 'relu',
                'edge_prediction_method': 'mlp',
                'batch_norm': True,
                'heads': 4,
                'concat_heads': True
            }
        },
        'gin': {
            'encoder_kwargs': {
                'num_layers': 3,
                'dropout': 0.1,
                'activation': 'relu',
                'pooling': 'mean',
                'batch_norm': True,
                'epsilon': 0.0,
                'train_epsilon': False
            },
            'decoder_kwargs': {
                'num_layers': 3,
                'dropout': 0.1,
                'activation': 'relu',
                'edge_prediction_method': 'mlp',
                'batch_norm': True,
                'epsilon': 0.0,
                'train_epsilon': False
            }
        }
    }


# ================ Benchmark Tests ================

@pytest.mark.benchmark
def test_architecture_reconstruction_quality(synthetic_benchmark_dataset, architecture_configurations, test_dir):
    """Benchmark the reconstruction quality of different architectures."""
    # Setup data
    batch_size = 16
    dataloader = DataLoader(
        synthetic_benchmark_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pyg_collate
    )

    # Get data dimensions
    sample_graph = synthetic_benchmark_dataset[0]
    input_dim = sample_graph.x.shape[1]
    max_nodes = max(g.num_nodes for g in synthetic_benchmark_dataset)

    # Hyperparameters
    hidden_dim = 64
    latent_dim = 32

    # Results storage
    results = {}

    # Benchmark each architecture
    for arch in ['gcn', 'gat', 'gin']:
        print(f"\nBenchmarking {arch.upper()} architecture...")

        # Create model
        model = create_model(
            architecture=arch,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            output_dim=input_dim,
            num_nodes=max_nodes,
            **architecture_configurations[arch]['encoder_kwargs'],
            **{f"decoder_{k}": v for k, v in architecture_configurations[arch]['decoder_kwargs'].items()}
        )

        # Evaluate on all graphs
        precision_values = []
        recall_values = []
        f1_values = []
        accuracy_values = []

        for batch in dataloader:
            with torch.no_grad():
                # Get reconstructions
                reconstructed_batch, _ = model(batch)

                # Process graphs individually for metrics
                orig_graphs = batch.to_data_list()
                recon_graphs = reconstructed_batch.to_data_list()

                for orig, recon in zip(orig_graphs, recon_graphs):
                    precision, recall, f1 = edge_f1_score(orig, recon)
                    accuracy = reconstruction_accuracy(orig, recon)

                    precision_values.append(precision)
                    recall_values.append(recall)
                    f1_values.append(f1)
                    accuracy_values.append(accuracy)

        # Compute statistics
        results[arch] = {
            'precision': {
                'mean': np.mean(precision_values),
                'std': np.std(precision_values)
            },
            'recall': {
                'mean': np.mean(recall_values),
                'std': np.std(recall_values)
            },
            'f1': {
                'mean': np.mean(f1_values),
                'std': np.std(f1_values)
            },
            'accuracy': {
                'mean': np.mean(accuracy_values),
                'std': np.std(accuracy_values)
            }
        }

        print(
            f"  Precision: {results[arch]['precision']['mean']:.4f} ± {results[arch]['precision']['std']:.4f}")
        print(
            f"  Recall: {results[arch]['recall']['mean']:.4f} ± {results[arch]['recall']['std']:.4f}")
        print(
            f"  F1 Score: {results[arch]['f1']['mean']:.4f} ± {results[arch]['f1']['std']:.4f}")
        print(
            f"  Accuracy: {results[arch]['accuracy']['mean']:.4f} ± {results[arch]['accuracy']['std']:.4f}")

    # Assert that results exist for all architectures
    assert len(results) == 3
    for arch in ['gcn', 'gat', 'gin']:
        assert arch in results
        assert 'precision' in results[arch]
        assert 'recall' in results[arch]
        assert 'f1' in results[arch]
        assert 'accuracy' in results[arch]

    # Visualize benchmark results
    metrics = ['precision', 'recall', 'f1', 'accuracy']

    # Create a grouped bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set width of bar
    bar_width = 0.2
    index = np.arange(len(metrics))

    for i, arch in enumerate(['gcn', 'gat', 'gin']):
        means = [results[arch][metric]['mean'] for metric in metrics]
        stds = [results[arch][metric]['std'] for metric in metrics]

        ax.bar(
            index + i*bar_width,
            means,
            bar_width,
            yerr=stds,
            label=arch.upper(),
            capsize=5
        )

    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Reconstruction Quality Comparison')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = test_dir / "reconstruction_quality_benchmark.png"
    plt.savefig(plot_path)
    plt.close()

    # Check that the plot was saved
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


@pytest.mark.benchmark
def test_architecture_training_time(synthetic_benchmark_dataset, architecture_configurations, test_dir):
    """Benchmark the training time of different architectures."""
    # Setup data
    batch_size = 16
    dataloader = DataLoader(
        synthetic_benchmark_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pyg_collate
    )

    # Get data dimensions
    sample_graph = synthetic_benchmark_dataset[0]
    input_dim = sample_graph.x.shape[1]
    max_nodes = max(g.num_nodes for g in synthetic_benchmark_dataset)

    # Hyperparameters
    hidden_dim = 64
    latent_dim = 32
    num_epochs = 3  # Small number for testing

    # Results storage
    results = {}

    # Benchmark each architecture
    for arch in ['gcn', 'gat', 'gin']:
        print(f"\nBenchmarking {arch.upper()} training time...")

        # Create model
        model = create_model(
            architecture=arch,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            output_dim=input_dim,
            num_nodes=max_nodes,
            **architecture_configurations[arch]['encoder_kwargs'],
            **{f"decoder_{k}": v for k, v in architecture_configurations[arch]['decoder_kwargs'].items()}
        )

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Record training time
        epoch_times = []
        forward_times = []
        backward_times = []
        total_loss = 0

        for epoch in range(num_epochs):
            epoch_start = time.time()
            batch_losses = []

            for batch in dataloader:
                # Forward pass timing
                forward_start = time.time()
                reconstructed_batch, latent = model(batch)
                losses = model.compute_loss(batch, reconstructed_batch, latent)
                loss = losses['total_loss']
                forward_end = time.time()

                # Backward pass timing
                backward_start = time.time()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                backward_end = time.time()

                # Record times
                forward_times.append(forward_end - forward_start)
                backward_times.append(backward_end - backward_start)

                # Record loss
                batch_losses.append(loss.item())

            # Record epoch time
            epoch_end = time.time()
            epoch_times.append(epoch_end - epoch_start)

            # Print epoch stats
            print(
                f"  Epoch {epoch+1}/{num_epochs} - Time: {epoch_times[-1]:.4f}s, Loss: {np.mean(batch_losses):.4f}")

            # Update total loss
            total_loss += np.mean(batch_losses)

        # Store results
        results[arch] = {
            'epoch_time': {
                'mean': np.mean(epoch_times),
                'std': np.std(epoch_times)
            },
            'forward_time': {
                'mean': np.mean(forward_times),
                'std': np.std(forward_times)
            },
            'backward_time': {
                'mean': np.mean(backward_times),
                'std': np.std(backward_times)
            },
            'total_loss': total_loss / num_epochs
        }

        print(
            f"  Average epoch time: {results[arch]['epoch_time']['mean']:.4f} ± {results[arch]['epoch_time']['std']:.4f}s")
        print(
            f"  Average forward pass time: {results[arch]['forward_time']['mean']:.4f} ± {results[arch]['forward_time']['std']:.4f}s")
        print(
            f"  Average backward pass time: {results[arch]['backward_time']['mean']:.4f} ± {results[arch]['backward_time']['std']:.4f}s")

    # Assert that results exist for all architectures
    assert len(results) == 3
    for arch in ['gcn', 'gat', 'gin']:
        assert arch in results
        assert 'epoch_time' in results[arch]
        assert 'forward_time' in results[arch]
        assert 'backward_time' in results[arch]

    # Visualize benchmark results
    metrics = ['epoch_time', 'forward_time', 'backward_time']
    titles = ['Epoch Time', 'Forward Pass Time', 'Backward Pass Time']

    # Create a grouped bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set width of bar
    bar_width = 0.3
    index = np.arange(len(metrics))

    for i, arch in enumerate(['gcn', 'gat', 'gin']):
        means = [results[arch][metric]['mean'] for metric in metrics]
        stds = [results[arch][metric]['std'] for metric in metrics]

        ax.bar(
            index + i*bar_width,
            means,
            bar_width,
            yerr=stds,
            label=arch.upper(),
            capsize=5
        )

    ax.set_xlabel('Timing Metric')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Training Time Comparison')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(titles)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = test_dir / "training_time_benchmark.png"
    plt.savefig(plot_path)
    plt.close()

    # Check that the plot was saved
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0

    # Plot loss comparison
    plt.figure(figsize=(8, 5))
    architectures = ['gcn', 'gat', 'gin']
    losses = [results[arch]['total_loss'] for arch in architectures]

    plt.bar(architectures, losses, color=['blue', 'green', 'orange'])
    plt.xlabel('Architecture')
    plt.ylabel('Average Loss')
    plt.title('Loss Comparison Across Architectures')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    loss_plot_path = test_dir / "loss_comparison_benchmark.png"
    plt.savefig(loss_plot_path)
    plt.close()

    # Check that the loss plot was saved
    assert loss_plot_path.exists()
    assert loss_plot_path.stat().st_size > 0


@pytest.mark.benchmark
def test_latent_space_properties(synthetic_benchmark_dataset, architecture_configurations, test_dir):
    """Benchmark latent space properties of different architectures."""
    # Setup data
    # Create a batch from all graphs for consistent comparison
    batch = Batch.from_data_list(synthetic_benchmark_dataset)

    # Get data dimensions
    sample_graph = synthetic_benchmark_dataset[0]
    input_dim = sample_graph.x.shape[1]
    max_nodes = max(g.num_nodes for g in synthetic_benchmark_dataset)

    # Extract labels
    y = torch.cat([graph.y for graph in synthetic_benchmark_dataset])

    # Hyperparameters
    hidden_dim = 64
    latent_dim = 8  # Larger latent dim for this test

    # Results storage
    latent_vectors = {}

    # Benchmark each architecture
    for arch in ['gcn', 'gat', 'gin']:
        print(f"\nAnalyzing {arch.upper()} latent space...")

        # Create model
        model = create_model(
            architecture=arch,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            output_dim=input_dim,
            num_nodes=max_nodes,
            **architecture_configurations[arch]['encoder_kwargs'],
            **{f"decoder_{k}": v for k, v in architecture_configurations[arch]['decoder_kwargs'].items()}
        )

        # Get embeddings
        with torch.no_grad():
            _, encodings = model(batch)
            latent_vectors[arch] = encodings

    # Visualize latent spaces using PCA for dimensionality reduction
    from sklearn.decomposition import PCA

    # Create a 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Plot for each architecture
    for i, arch in enumerate(['gcn', 'gat', 'gin']):
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_vectors[arch].detach().numpy())

        # Create scatter plot
        scatter = axes[i].scatter(
            latent_2d[:, 0],
            latent_2d[:, 1],
            c=y.detach().numpy(),
            cmap='viridis',
            alpha=0.7,
            s=80
        )

        # Add labels and legend
        axes[i].set_title(f"{arch.upper()} Latent Space (PCA)")
        axes[i].set_xlabel("PC1")
        axes[i].set_ylabel("PC2")
        axes[i].grid(True, linestyle='--', alpha=0.7)

    # Add a combined plot
    # Apply PCA to all latent vectors combined
    all_latents = torch.cat([
        latent_vectors['gcn'],
        latent_vectors['gat'],
        latent_vectors['gin']
    ], dim=0).detach().numpy()

    # Create labels identifying the architecture for each point
    architecture_labels = np.concatenate([
        np.zeros(len(synthetic_benchmark_dataset)),
        np.ones(len(synthetic_benchmark_dataset)),
        np.ones(len(synthetic_benchmark_dataset)) * 2
    ])

    # Apply PCA
    combined_pca = PCA(n_components=2)
    combined_latent_2d = combined_pca.fit_transform(all_latents)

    # Create scatter plot
    scatter = axes[3].scatter(
        combined_latent_2d[:, 0],
        combined_latent_2d[:, 1],
        c=architecture_labels,
        cmap='Set1',
        alpha=0.7,
        s=80
    )

    # Add labels and legend
    axes[3].set_title("Architecture Comparison (PCA)")
    axes[3].set_xlabel("PC1")
    axes[3].set_ylabel("PC2")
    axes[3].grid(True, linestyle='--', alpha=0.7)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[3])
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(['GCN', 'GAT', 'GIN'])

    plt.tight_layout()
    plot_path = test_dir / "latent_space_comparison.png"
    plt.savefig(plot_path)
    plt.close()

    # Check that the plot was saved
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0

    # Calculate and visualize latent space statistics
    stats = {}

    for arch in ['gcn', 'gat', 'gin']:
        # Get latent vectors
        latents = latent_vectors[arch].detach().numpy()

        # Calculate statistics
        stats[arch] = {
            'mean': np.mean(latents),
            'std': np.std(latents),
            'min': np.min(latents),
            'max': np.max(latents),
            'l2_norm': np.mean(np.sqrt(np.sum(latents**2, axis=1))),
            'variance_explained': pca.explained_variance_ratio_.sum() if arch != 'combined' else None
        }

    # Plot statistics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Basic statistics
    basic_stats = ['mean', 'std', 'min', 'max']
    axes[0].bar(
        np.arange(len(basic_stats)),
        [stats['gcn'][stat] for stat in basic_stats],
        width=0.25,
        label='GCN'
    )
    axes[0].bar(
        np.arange(len(basic_stats)) + 0.25,
        [stats['gat'][stat] for stat in basic_stats],
        width=0.25,
        label='GAT'
    )
    axes[0].bar(
        np.arange(len(basic_stats)) + 0.5,
        [stats['gin'][stat] for stat in basic_stats],
        width=0.25,
        label='GIN'
    )
    axes[0].set_xticks(np.arange(len(basic_stats)) + 0.25)
    axes[0].set_xticklabels(basic_stats)
    axes[0].set_title("Basic Latent Space Statistics")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # L2 norm
    axes[1].bar(
        ['GCN', 'GAT', 'GIN'],
        [stats[arch]['l2_norm'] for arch in ['gcn', 'gat', 'gin']],
        color=['blue', 'green', 'orange']
    )
    axes[1].set_title("Average L2 Norm of Latent Vectors")
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # Variance explained by PCA
    axes[2].bar(
        ['GCN', 'GAT', 'GIN'],
        [stats[arch]['variance_explained'] for arch in ['gcn', 'gat', 'gin']],
        color=['blue', 'green', 'orange']
    )
    axes[2].set_title("Variance Explained by First 2 PCA Components")
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, linestyle='--', alpha=0.7)

    # Inter-class distance (using y labels)
    from sklearn.metrics import pairwise_distances

    class_distances = {}
    for arch in ['gcn', 'gat', 'gin']:
        latents = latent_vectors[arch].detach().numpy()

        # Group by class
        unique_classes = np.unique(y.detach().numpy())
        class_centroids = []

        for c in unique_classes:
            class_mask = (y.detach().numpy() == c)
            class_centroid = np.mean(latents[class_mask], axis=0)
            class_centroids.append(class_centroid)

        # Calculate pairwise distances between class centroids
        distances = pairwise_distances(class_centroids)
        # Average inter-class distance
        avg_distance = np.mean(
            distances[np.triu_indices(len(unique_classes), k=1)])
        class_distances[arch] = avg_distance

    axes[3].bar(
        ['GCN', 'GAT', 'GIN'],
        [class_distances[arch] for arch in ['gcn', 'gat', 'gin']],
        color=['blue', 'green', 'orange']
    )
    axes[3].set_title("Average Inter-Class Distance")
    axes[3].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    stats_path = test_dir / "latent_space_statistics.png"
    plt.savefig(stats_path)
    plt.close()

    # Check that the plot was saved
    assert stats_path.exists()
    assert stats_path.stat().st_size > 0
