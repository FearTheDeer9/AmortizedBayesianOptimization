"""
Comprehensive tests for GNN encoder-decoder models.

This module provides extensive testing for all aspects of the GNN encoder-decoder
implementations, including unit tests, integration tests, benchmark comparisons, 
and visualization checks.
"""

import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
import io
import os
from pathlib import Path
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import TUDataset
from torch.utils.data import DataLoader

from causal_meta.inference.models.wrapper import EncoderDecoderWrapper, create_model
from causal_meta.inference.models.gcn_encoder import GCNEncoder
from causal_meta.inference.models.gcn_decoder import GCNDecoder
from causal_meta.inference.models.gat_encoder import GATEncoder
from causal_meta.inference.models.gat_decoder import GATDecoder
from causal_meta.inference.models.gin_encoder import GINEncoder
from causal_meta.inference.models.gin_decoder import GINDecoder
from causal_meta.inference.models.encoder import edge_f1_score, reconstruction_accuracy


# Custom collate function for PyTorch Geometric Data objects
def pyg_collate(batch):
    return Batch.from_data_list(batch)


# ================ Fixtures ================

@pytest.fixture
def test_dir():
    """Create a test directory for saving visualization outputs."""
    test_dir = Path("test_outputs")
    test_dir.mkdir(exist_ok=True)
    yield test_dir
    # Uncomment to clean up after tests
    # import shutil
    # shutil.rmtree(test_dir)


@pytest.fixture
def simple_graph():
    """Create a simple graph for testing."""
    # Create a simple graph with 4 nodes and 5 edges
    x = torch.randn(4, 3)  # 4 nodes with 3 features each
    edge_index = torch.tensor(
        [[0, 0, 1, 2, 3], [1, 2, 3, 3, 2]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


@pytest.fixture
def batch_graphs():
    """Create a batch of simple graphs for testing."""
    graphs = []
    for _ in range(3):  # Create 3 graphs
        num_nodes = np.random.randint(3, 6)
        x = torch.randn(num_nodes, 3)

        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and np.random.rand() > 0.5:
                    edge_index.append([i, j])

        if not edge_index:  # Ensure at least one edge
            i, j = np.random.choice(num_nodes, 2, replace=False)
            edge_index.append([i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        graphs.append(Data(x=x, edge_index=edge_index))

    return Batch.from_data_list(graphs)


@pytest.fixture
def model_configs():
    """Define configurations for different model architectures."""
    return {
        'gcn': {
            'encoder_kwargs': {
                'activation': 'relu',
                'pooling': 'mean',
                'batch_norm': True,
                'dropout': 0.1
            },
            'decoder_kwargs': {
                'activation': 'relu',
                'edge_prediction_method': 'mlp',
                'batch_norm': True,
                'dropout': 0.1
            }
        },
        'gat': {
            'encoder_kwargs': {
                'activation': 'relu',
                'pooling': 'mean',
                'batch_norm': True,
                'dropout': 0.1,
                'heads': 4,
                'concat_heads': True
            },
            'decoder_kwargs': {
                'activation': 'relu',
                'edge_prediction_method': 'mlp',
                'batch_norm': True,
                'dropout': 0.1,
                'heads': 4,
                'concat_heads': True
            }
        },
        'gin': {
            'encoder_kwargs': {
                'activation': 'relu',
                'pooling': 'mean',
                'batch_norm': True,
                'dropout': 0.1,
                'epsilon': 0.0,
                'train_epsilon': False
            },
            'decoder_kwargs': {
                'activation': 'relu',
                'edge_prediction_method': 'mlp',
                'batch_norm': True,
                'dropout': 0.1,
                'epsilon': 0.0,
                'train_epsilon': False
            }
        }
    }


@pytest.fixture
def synthetic_graph_dataset():
    """Create a synthetic dataset of graphs for testing data loading."""
    num_graphs = 10
    graphs = []

    for i in range(num_graphs):
        # Create random graph with 5-10 nodes
        num_nodes = np.random.randint(5, 11)
        x = torch.randn(num_nodes, 3)

        # Create random edges (with controlled density)
        edge_index = []
        edge_density = np.random.uniform(0.2, 0.5)  # 20-50% of possible edges

        for src in range(num_nodes):
            for dst in range(num_nodes):
                if src != dst and np.random.rand() < edge_density:
                    edge_index.append([src, dst])

        if not edge_index:  # Ensure at least one edge
            i, j = np.random.choice(num_nodes, 2, replace=False)
            edge_index.append([i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()

        # Add y values (for supervised tasks)
        y = torch.tensor([i % 2], dtype=torch.float)  # Binary labels

        # Create graph with node features and edge indices
        graph = Data(x=x, edge_index=edge_index, y=y)
        graphs.append(graph)

    return graphs


# ================ Unit Tests ================

def test_encoder_base_implementations():
    """Test that all encoder implementations have required functionality."""
    input_dim, hidden_dim, latent_dim = 3, 16, 8

    # List of encoder classes to test
    encoder_classes = [GCNEncoder, GATEncoder, GINEncoder]

    for EncoderClass in encoder_classes:
        # Get class-specific kwargs
        kwargs = {}
        if EncoderClass == GATEncoder:
            kwargs = {'heads': 2, 'concat_heads': True}
        elif EncoderClass == GINEncoder:
            kwargs = {'epsilon': 0.0, 'train_epsilon': False}

        # Initialize the encoder
        encoder = EncoderClass(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            **kwargs
        )

        # Check that required attributes are present
        assert hasattr(encoder, 'encode')
        assert hasattr(encoder, 'forward')
        assert hasattr(encoder, 'preprocess_graph')
        assert hasattr(encoder, 'get_config')

        # Check that get_config returns expected attributes
        config = encoder.get_config()
        assert config['input_dim'] == input_dim
        assert config['hidden_dim'] == hidden_dim
        assert config['latent_dim'] == latent_dim
        assert config['type'] == EncoderClass.__name__


def test_decoder_base_implementations():
    """Test that all decoder implementations have required functionality."""
    latent_dim, hidden_dim, output_dim, num_nodes = 8, 16, 3, 4

    # List of decoder classes to test
    decoder_classes = [GCNDecoder, GATDecoder, GINDecoder]

    for DecoderClass in decoder_classes:
        # Get class-specific kwargs
        kwargs = {}
        if DecoderClass == GATDecoder:
            kwargs = {'heads': 2, 'concat_heads': True}
        elif DecoderClass == GINDecoder:
            kwargs = {'epsilon': 0.0, 'train_epsilon': False}

        # Initialize the decoder
        decoder = DecoderClass(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_nodes=num_nodes,
            **kwargs
        )

        # Check that required attributes are present
        assert hasattr(decoder, 'decode')
        assert hasattr(decoder, 'forward')
        assert hasattr(decoder, 'predict_node_features')
        assert hasattr(decoder, 'predict_edges')
        assert hasattr(decoder, 'get_config')

        # Check that get_config returns expected attributes
        config = decoder.get_config()
        assert config['latent_dim'] == latent_dim
        assert config['hidden_dim'] == hidden_dim
        assert config['output_dim'] == output_dim
        assert config['num_nodes'] == num_nodes
        assert config['type'] == DecoderClass.__name__


def test_model_factory():
    """Test the create_model factory function."""
    input_dim, hidden_dim, latent_dim = 3, 16, 8
    output_dim, num_nodes = 3, 4

    # Test GCN creation
    gcn_model = create_model(
        architecture='gcn',
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        output_dim=output_dim,
        num_nodes=num_nodes
    )
    assert isinstance(gcn_model, EncoderDecoderWrapper)
    assert isinstance(gcn_model.encoder, GCNEncoder)
    assert isinstance(gcn_model.decoder, GCNDecoder)

    # Test GAT creation
    gat_model = create_model(
        architecture='gat',
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        output_dim=output_dim,
        num_nodes=num_nodes
    )
    assert isinstance(gat_model, EncoderDecoderWrapper)
    assert isinstance(gat_model.encoder, GATEncoder)
    assert isinstance(gat_model.decoder, GATDecoder)

    # Test GIN creation
    gin_model = create_model(
        architecture='gin',
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        output_dim=output_dim,
        num_nodes=num_nodes
    )
    assert isinstance(gin_model, EncoderDecoderWrapper)
    assert isinstance(gin_model.encoder, GINEncoder)
    assert isinstance(gin_model.decoder, GINDecoder)

    # Test with custom parameters
    custom_model = create_model(
        architecture='gcn',
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        output_dim=output_dim,
        num_nodes=num_nodes,
        encoder_num_layers=4,
        decoder_edge_prediction_method='bilinear',
        loss_type='mse',
        edge_weight=0.7
    )
    assert custom_model.encoder.num_layers == 4
    assert custom_model.decoder.edge_prediction_method == 'bilinear'
    assert custom_model.loss_type == 'mse'
    assert custom_model.edge_weight == 0.7


# ================ Integration Tests ================

def test_encoder_decoder_pipeline_all_architectures(simple_graph, model_configs):
    """Test the full encoder-decoder pipeline with all architectures."""
    input_dim = simple_graph.x.shape[1]
    num_nodes = simple_graph.x.shape[0]
    hidden_dim, latent_dim = 16, 8

    for arch in ['gcn', 'gat', 'gin']:
        # Create model with the specified architecture
        model = create_model(
            architecture=arch,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            output_dim=input_dim,
            num_nodes=num_nodes,
            **model_configs[arch]['encoder_kwargs'],
            **{f"decoder_{k}": v for k, v in model_configs[arch]['decoder_kwargs'].items()}
        )

        # Run forward pass
        reconstructed_graph, latent = model(simple_graph)

        # Basic checks
        assert isinstance(reconstructed_graph, Data)
        assert isinstance(latent, torch.Tensor)
        assert latent.shape[-1] == latent_dim
        assert reconstructed_graph.x.shape == simple_graph.x.shape

        # Calculate metrics
        precision, recall, f1 = edge_f1_score(
            simple_graph, reconstructed_graph)
        accuracy = reconstruction_accuracy(simple_graph, reconstructed_graph)

        # We're not checking specific values here as they depend on initialization,
        # but we ensure calculations don't error and return sensible values
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1
        assert 0 <= accuracy <= 1


def test_batch_processing_all_architectures(batch_graphs, model_configs):
    """Test batch processing with all architectures."""
    input_dim = batch_graphs.x.shape[1]
    max_nodes = max([g.num_nodes for g in batch_graphs.to_data_list()])
    hidden_dim, latent_dim = 16, 8

    for arch in ['gcn', 'gat', 'gin']:
        # Create model
        model = create_model(
            architecture=arch,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            output_dim=input_dim,
            num_nodes=max_nodes,
            **model_configs[arch]['encoder_kwargs'],
            **{f"decoder_{k}": v for k, v in model_configs[arch]['decoder_kwargs'].items()}
        )

        # Run forward pass
        reconstructed_batch, latent = model(batch_graphs)

        # Check batch processing
        assert isinstance(reconstructed_batch, Batch)
        assert latent.shape == (batch_graphs.num_graphs, latent_dim)

        # Check individual graph reconstructions
        recon_graphs = reconstructed_batch.to_data_list()
        orig_graphs = batch_graphs.to_data_list()

        assert len(recon_graphs) == len(orig_graphs)
        for i, (orig, recon) in enumerate(zip(orig_graphs, recon_graphs)):
            assert recon.x.shape == orig.x.shape


def test_different_loss_functions(simple_graph):
    """Test different loss functions for training."""
    input_dim = simple_graph.x.shape[1]
    num_nodes = simple_graph.x.shape[0]
    hidden_dim, latent_dim = 16, 8

    loss_types = ['bce', 'mse', 'weighted_bce']

    for loss_type in loss_types:
        # Create model with specific loss type
        model = create_model(
            architecture='gcn',
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            output_dim=input_dim,
            num_nodes=num_nodes,
            loss_type=loss_type
        )

        # Run forward pass
        reconstructed_graph, latent = model(simple_graph)

        # Compute losses
        losses = model.compute_loss(simple_graph, reconstructed_graph, latent)

        # Check loss computation
        assert 'edge_loss' in losses
        assert 'total_loss' in losses
        assert torch.isfinite(losses['edge_loss'])
        assert torch.isfinite(losses['total_loss'])


def test_regularization_options(simple_graph):
    """Test regularization options for training."""
    input_dim = simple_graph.x.shape[1]
    num_nodes = simple_graph.x.shape[0]
    hidden_dim, latent_dim = 16, 8

    # Test with different regularization weights
    model_configs = [
        {'feature_weight': 0.5, 'edge_weight': 1.0,
            'kl_weight': 0.0, 'l2_weight': 0.0},
        {'feature_weight': 0.3, 'edge_weight': 0.7,
            'kl_weight': 0.1, 'l2_weight': 0.0},
        {'feature_weight': 0.2, 'edge_weight': 0.8,
            'kl_weight': 0.0, 'l2_weight': 0.1},
        {'feature_weight': 0.2, 'edge_weight': 0.7,
            'kl_weight': 0.05, 'l2_weight': 0.05}
    ]

    for config in model_configs:
        # Create model with specific regularization
        model = create_model(
            architecture='gcn',
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            output_dim=input_dim,
            num_nodes=num_nodes,
            **config
        )

        # Run forward pass
        reconstructed_graph, latent = model(simple_graph)

        # Compute losses
        losses = model.compute_loss(simple_graph, reconstructed_graph, latent)

        # Check that all expected loss components are present
        expected_keys = ['edge_loss', 'feature_loss', 'total_loss']
        if config['kl_weight'] > 0:
            expected_keys.append('kl_loss')
        if config['l2_weight'] > 0:
            expected_keys.append('l2_loss')

        for key in expected_keys:
            assert key in losses
            assert torch.isfinite(losses[key])


# ================ Benchmark Tests ================

def test_architecture_performance_comparison(batch_graphs, model_configs):
    """Compare reconstruction performance across architectures."""
    input_dim = batch_graphs.x.shape[1]
    max_nodes = max([g.num_nodes for g in batch_graphs.to_data_list()])
    hidden_dim, latent_dim = 16, 8

    results = {}

    for arch in ['gcn', 'gat', 'gin']:
        # Create model
        model = create_model(
            architecture=arch,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            output_dim=input_dim,
            num_nodes=max_nodes,
            **model_configs[arch]['encoder_kwargs'],
            **{f"decoder_{k}": v for k, v in model_configs[arch]['decoder_kwargs'].items()}
        )

        # Run evaluation
        metrics = model.evaluate(batch_graphs)

        # Store results
        results[arch] = {
            'edge_precision': metrics['edge_precision'],
            'edge_recall': metrics['edge_recall'],
            'edge_f1': metrics['edge_f1'],
            'reconstruction_accuracy': metrics['reconstruction_accuracy']
        }

    # Assert that results are collected for all architectures
    assert len(results) == 3

    # We're not asserting specific performance as models are randomly initialized,
    # but we verify that metrics are computed successfully


# ================ Data Loading Utilities Tests ================

def test_data_loading_and_batching(synthetic_graph_dataset):
    """Test loading and batching graph data."""
    # Create DataLoader
    batch_size = 4
    dataloader = DataLoader(
        synthetic_graph_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pyg_collate
    )

    # Check loader creation
    assert len(dataloader) == (
        len(synthetic_graph_dataset) + batch_size - 1) // batch_size

    # Check batch structure
    for batch in dataloader:
        assert isinstance(batch, Batch)
        assert hasattr(batch, 'x')
        assert hasattr(batch, 'edge_index')
        assert hasattr(batch, 'y')
        assert hasattr(batch, 'batch')

        # Check batch size (could be smaller for the last batch)
        assert batch.num_graphs <= batch_size

        # Check that individual graphs can be recovered
        graphs = batch.to_data_list()
        assert len(graphs) == batch.num_graphs

        for graph in graphs:
            assert isinstance(graph, Data)
            assert hasattr(graph, 'x')
            assert hasattr(graph, 'edge_index')
            assert hasattr(graph, 'y')
            break  # Only need to check one batch

        break  # Only need to check one batch


def test_model_training_loop(synthetic_graph_dataset):
    """Test a basic training loop with the encoder-decoder models."""
    # Setup data
    batch_size = 4
    dataloader = DataLoader(
        synthetic_graph_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pyg_collate
    )

    # Get data dimensions
    sample_graph = synthetic_graph_dataset[0]
    input_dim = sample_graph.x.shape[1]
    max_nodes = max(g.num_nodes for g in synthetic_graph_dataset)

    # Create model
    model = create_model(
        architecture='gcn',
        input_dim=input_dim,
        hidden_dim=32,
        latent_dim=16,
        output_dim=input_dim,
        num_nodes=max_nodes
    )

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Run a few training iterations
    initial_loss = None
    final_loss = None

    for epoch in range(2):  # Just 2 epochs for testing
        epoch_loss = 0.0

        for batch in dataloader:
            # Forward pass
            reconstructed_batch, latent = model(batch)

            # Compute loss
            losses = model.compute_loss(batch, reconstructed_batch, latent)
            loss = losses['total_loss']

            if initial_loss is None:
                initial_loss = loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Store final loss
        final_loss = epoch_loss / len(dataloader)

    # Check that loss exists and is finite
    assert initial_loss is not None
    assert final_loss is not None
    assert np.isfinite(initial_loss)
    assert np.isfinite(final_loss)


# ================ Visualization Tests ================

def visualize_graph(graph, title="Graph", ax=None):
    """
    Helper function to visualize a graph.
    """
    import networkx as nx
    from torch_geometric.utils import to_networkx

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Convert to networkx graph
    G = to_networkx(graph, to_undirected=True)

    # Draw the graph
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='skyblue',
            node_size=400, edge_color='gray', linewidths=1, alpha=0.7)

    ax.set_title(title)
    return ax


def test_visualization_utilities(simple_graph, test_dir):
    """Test graph visualization utilities."""
    input_dim = simple_graph.x.shape[1]
    num_nodes = simple_graph.x.shape[0]
    hidden_dim, latent_dim = 16, 8

    # Create model
    model = create_model(
        architecture='gcn',
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        output_dim=input_dim,
        num_nodes=num_nodes
    )

    # Run forward pass
    reconstructed_graph, _ = model(simple_graph)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    visualize_graph(simple_graph, "Original Graph", ax1)
    visualize_graph(reconstructed_graph, "Reconstructed Graph", ax2)

    # Save figure to verify visualization works
    plt.tight_layout()
    plot_path = test_dir / "graph_reconstruction.png"
    plt.savefig(plot_path)
    plt.close()

    # Check that file was created
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_latent_space_visualization(synthetic_graph_dataset, test_dir):
    """Test visualization of latent space embeddings."""
    # Create model
    sample_graph = synthetic_graph_dataset[0]
    input_dim = sample_graph.x.shape[1]
    max_nodes = max(g.num_nodes for g in synthetic_graph_dataset)

    # Use a 2D latent space for easy visualization
    latent_dim = 2

    model = create_model(
        architecture='gin',  # Using GIN for this test
        input_dim=input_dim,
        hidden_dim=32,
        latent_dim=latent_dim,
        output_dim=input_dim,
        num_nodes=max_nodes
    )

    # Create a batch from all graphs
    batch = Batch.from_data_list(synthetic_graph_dataset)

    # Get embeddings
    _, latent_vectors = model(batch)

    # Extract labels
    y = torch.cat([graph.y for graph in synthetic_graph_dataset])

    # Create scatter plot of latent space
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        latent_vectors[:, 0].detach().numpy(),
        latent_vectors[:, 1].detach().numpy(),
        c=y.detach().numpy(),
        cmap='coolwarm',
        alpha=0.7,
        s=100
    )
    plt.colorbar(scatter, label="Graph Class")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("2D Latent Space Visualization")
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save figure
    plt.tight_layout()
    plot_path = test_dir / "latent_space.png"
    plt.savefig(plot_path)
    plt.close()

    # Check that file was created
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


# ================ Robustness Tests ================

def test_model_robustness_to_input_noise(simple_graph):
    """Test model robustness to input noise."""
    input_dim = simple_graph.x.shape[1]
    num_nodes = simple_graph.x.shape[0]

    # Create model
    model = create_model(
        architecture='gcn',
        input_dim=input_dim,
        hidden_dim=32,
        latent_dim=16,
        output_dim=input_dim,
        num_nodes=num_nodes
    )

    # Get baseline reconstruction
    original_recon, original_latent = model(simple_graph)
    original_metrics = model.evaluate(simple_graph)

    # Add different levels of noise
    noise_levels = [0.05, 0.1, 0.2]
    metrics_with_noise = []

    for noise_level in noise_levels:
        # Create noisy version of input
        noisy_features = simple_graph.x + noise_level * \
            torch.randn_like(simple_graph.x)
        noisy_graph = Data(
            x=noisy_features, edge_index=simple_graph.edge_index)

        # Get reconstruction and metrics
        noisy_metrics = model.evaluate(noisy_graph)
        metrics_with_noise.append(noisy_metrics)

    # We expect some degradation with noise, but the model should still produce valid outputs
    for metrics in metrics_with_noise:
        assert 0 <= metrics['edge_precision'] <= 1
        assert 0 <= metrics['edge_recall'] <= 1
        assert 0 <= metrics['edge_f1'] <= 1
        assert 0 <= metrics['reconstruction_accuracy'] <= 1
