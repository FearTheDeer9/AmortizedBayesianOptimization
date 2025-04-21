"""
Test module for GIN encoder and decoder implementations.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, Batch

from causal_meta.inference.models.gin_encoder import GINEncoder
from causal_meta.inference.models.gin_decoder import GINDecoder


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
        # Random number of nodes between 3 and 5
        num_nodes = np.random.randint(3, 6)
        x = torch.randn(num_nodes, 3)  # Node features

        # Create random edges (ensuring no self-loops)
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and np.random.rand() > 0.5:  # 50% chance of edge
                    edge_index.append([i, j])

        if not edge_index:  # Ensure at least one edge
            i, j = np.random.choice(num_nodes, 2, replace=False)
            edge_index.append([i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        graphs.append(Data(x=x, edge_index=edge_index))

    return Batch.from_data_list(graphs)


def test_gin_encoder_init():
    """Test GIN encoder initialization."""
    encoder = GINEncoder(
        input_dim=3,
        hidden_dim=16,
        latent_dim=8,
        num_layers=2,
        dropout=0.1,
        activation="relu",
        pooling="mean",
        batch_norm=True,
        epsilon=0.0,
        train_epsilon=False
    )

    assert encoder.input_dim == 3
    assert encoder.hidden_dim == 16
    assert encoder.latent_dim == 8
    assert encoder.num_layers == 2
    assert encoder.dropout_rate == 0.1
    assert encoder.activation_name == "relu"
    assert encoder.pooling_method == "mean"
    assert encoder.use_batch_norm is True
    assert encoder.epsilon == 0.0
    assert encoder.train_epsilon is False


def test_gin_decoder_init():
    """Test GIN decoder initialization."""
    decoder = GINDecoder(
        latent_dim=8,
        hidden_dim=16,
        output_dim=3,
        num_nodes=4,
        edge_prediction_method="mlp",
        num_layers=2,
        dropout=0.1,
        activation="relu",
        threshold=0.5,
        batch_norm=True,
        epsilon=0.0,
        train_epsilon=False
    )

    assert decoder.latent_dim == 8
    assert decoder.hidden_dim == 16
    assert decoder.output_dim == 3
    assert decoder.num_nodes == 4
    assert decoder.edge_prediction_method == "mlp"
    assert decoder.num_layers == 2
    assert decoder.dropout_rate == 0.1
    assert decoder.activation_name == "relu"
    assert decoder.threshold == 0.5
    assert decoder.use_batch_norm is True
    assert decoder.epsilon == 0.0
    assert decoder.train_epsilon is False


def test_gin_encoder_forward(simple_graph):
    """Test GIN encoder forward pass."""
    encoder = GINEncoder(
        input_dim=3,
        hidden_dim=16,
        latent_dim=8,
        num_layers=2
    )

    # Test forward pass
    z = encoder(simple_graph)

    # Output should be a latent vector
    assert isinstance(z, torch.Tensor)
    assert z.shape == (8,)  # latent_dim = 8

    # Test encode method
    z2 = encoder.encode(simple_graph)
    assert torch.allclose(z, z2)

    # Test get_config method
    config = encoder.get_config()
    assert config["input_dim"] == 3
    assert config["hidden_dim"] == 16
    assert config["latent_dim"] == 8


def test_gin_decoder_forward():
    """Test GIN decoder forward pass."""
    decoder = GINDecoder(
        latent_dim=8,
        hidden_dim=16,
        output_dim=3,
        num_nodes=4,
        edge_prediction_method="mlp"
    )

    # Create a latent vector
    z = torch.randn(8)

    # Test forward pass
    graph = decoder(z)

    # Output should be a graph
    assert isinstance(graph, Data)
    assert graph.x.shape[0] == 4  # num_nodes = 4
    assert graph.x.shape[1] == 3  # output_dim = 3
    assert graph.edge_index.shape[0] == 2  # [2, num_edges]

    # Test decode method
    graph2 = decoder.decode(z)
    assert torch.allclose(graph.x, graph2.x)

    # Test with batched input
    batch_z = torch.randn(2, 8)  # Batch of 2 latent vectors
    batch_graphs = decoder(batch_z)

    # Output should be a batch of graphs
    assert isinstance(batch_graphs, Batch)


def test_encoder_decoder_pipeline(simple_graph):
    """Test the full encoder-decoder pipeline."""
    # Initialize encoder and decoder
    latent_dim = 8
    input_dim = simple_graph.x.shape[1]
    num_nodes = simple_graph.x.shape[0]

    encoder = GINEncoder(
        input_dim=input_dim,
        hidden_dim=16,
        latent_dim=latent_dim,
        num_layers=2
    )

    decoder = GINDecoder(
        latent_dim=latent_dim,
        hidden_dim=16,
        output_dim=input_dim,
        num_nodes=num_nodes,
        edge_prediction_method="mlp"
    )

    # Encode
    z = encoder(simple_graph)

    # Decode
    reconstructed_graph = decoder(z)

    # Check outputs
    assert isinstance(reconstructed_graph, Data)
    assert reconstructed_graph.x.shape == simple_graph.x.shape
    assert reconstructed_graph.edge_index.shape[0] == 2


def test_gin_encoder_with_batched_graphs(batch_graphs):
    """Test GIN encoder with batched graphs."""
    input_dim = batch_graphs.x.shape[1]

    encoder = GINEncoder(
        input_dim=input_dim,
        hidden_dim=16,
        latent_dim=8,
        num_layers=2
    )

    # Get batch size
    batch_size = batch_graphs.num_graphs

    # Test forward pass with batched graphs
    z = encoder(batch_graphs)

    # Output should be a batch of latent vectors
    assert isinstance(z, torch.Tensor)
    assert z.shape == (batch_size, 8)  # [batch_size, latent_dim]


def test_different_activation_functions():
    """Test different activation functions in GIN encoder and decoder."""
    activations = ["relu", "leaky_relu", "elu", "tanh"]

    for activation in activations:
        # Test encoder
        encoder = GINEncoder(
            input_dim=3,
            hidden_dim=16,
            latent_dim=8,
            activation=activation
        )

        # Test decoder
        decoder = GINDecoder(
            latent_dim=8,
            hidden_dim=16,
            output_dim=3,
            num_nodes=4,
            activation=activation
        )

        # Simple forward pass to ensure no errors
        z = torch.randn(8)
        graph = decoder(z)

        assert isinstance(graph, Data)


def test_different_edge_prediction_methods(simple_graph):
    """Test different edge prediction methods in GIN decoder."""
    methods = ["mlp", "bilinear", "attention"]

    # First encode the graph to get a latent representation
    encoder = GINEncoder(
        input_dim=simple_graph.x.shape[1],
        hidden_dim=16,
        latent_dim=8
    )

    z = encoder(simple_graph)

    for method in methods:
        decoder = GINDecoder(
            latent_dim=8,
            hidden_dim=16,
            output_dim=simple_graph.x.shape[1],
            num_nodes=simple_graph.x.shape[0],
            edge_prediction_method=method
        )

        # Simple forward pass to ensure no errors
        graph = decoder(z)

        assert isinstance(graph, Data)
        assert graph.x.shape == simple_graph.x.shape
        assert graph.edge_index.shape[0] == 2


def test_epsilon_parameter_impact():
    """Test the impact of different epsilon values in GIN."""
    # Create a simple graph
    x = torch.randn(4, 3)
    edge_index = torch.tensor(
        [[0, 0, 1, 2, 3], [1, 2, 3, 3, 2]], dtype=torch.long)
    graph = Data(x=x, edge_index=edge_index)

    # Test with different epsilon values
    epsilon_values = [0.0, 0.5, 1.0]

    for eps in epsilon_values:
        # Create encoder with specific epsilon
        encoder = GINEncoder(
            input_dim=3,
            hidden_dim=16,
            latent_dim=8,
            epsilon=eps,
            train_epsilon=False
        )

        # Encode the graph
        z1 = encoder(graph)

        # Create another encoder with trainable epsilon
        encoder_train = GINEncoder(
            input_dim=3,
            hidden_dim=16,
            latent_dim=8,
            epsilon=eps,
            train_epsilon=True
        )

        # Encode the graph
        z2 = encoder_train(graph)

        # Results should be different tensors
        assert not torch.allclose(z1, z2)
