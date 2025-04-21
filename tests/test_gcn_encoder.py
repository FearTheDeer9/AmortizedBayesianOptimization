"""
Tests for the GCN-based encoder implementation.
"""

from causal_meta.inference.models.gcn_encoder import GCNEncoder
import unittest
import torch
import numpy as np
from torch_geometric.data import Data, Batch

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGCNEncoder(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        # Common test parameters
        self.input_dim = 3
        self.hidden_dim = 16
        self.latent_dim = 32

        # Create a simple graph for testing
        self.num_nodes = 5
        # 5 nodes with 3 features each
        self.x = torch.randn(self.num_nodes, self.input_dim)
        self.edge_index = torch.tensor(
            [[0, 1, 1, 2, 3], [1, 0, 2, 3, 4]], dtype=torch.long)  # Some edges
        self.graph = Data(x=self.x, edge_index=self.edge_index)

        # Create a batch of graphs for testing batch processing
        self.graphs = [
            Data(x=torch.randn(4, self.input_dim),
                 edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)),
            Data(x=torch.randn(5, self.input_dim),
                 edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)),
            Data(x=torch.randn(3, self.input_dim),
                 edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long))
        ]
        self.batch = Batch.from_data_list(self.graphs)

    def test_initialization(self):
        """Test that the GCN encoder can be initialized with different parameters."""
        # Default parameters
        encoder = GCNEncoder(self.input_dim, self.hidden_dim, self.latent_dim)
        self.assertEqual(encoder.input_dim, self.input_dim)
        self.assertEqual(encoder.hidden_dim, self.hidden_dim)
        self.assertEqual(encoder.latent_dim, self.latent_dim)
        self.assertEqual(encoder.num_layers, 3)  # Default
        self.assertEqual(encoder.pooling_method, "mean")  # Default

        # Custom parameters
        encoder = GCNEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=4,
            dropout=0.2,
            activation="leaky_relu",
            pooling="max",
            batch_norm=False
        )
        self.assertEqual(encoder.num_layers, 4)
        self.assertEqual(encoder.dropout_rate, 0.2)
        self.assertEqual(encoder.activation_name, "leaky_relu")
        self.assertEqual(encoder.pooling_method, "max")
        self.assertEqual(encoder.use_batch_norm, False)

    def test_forward_pass(self):
        """Test the forward pass of the GCN encoder."""
        # Create encoder
        encoder = GCNEncoder(self.input_dim, self.hidden_dim, self.latent_dim)

        # Run forward pass
        latent = encoder(self.graph)

        # Check output shape
        self.assertEqual(latent.shape, torch.Size([self.latent_dim]))

        # Ensure output values are finite
        self.assertTrue(torch.all(torch.isfinite(latent)))

    def test_batch_processing(self):
        """Test that the encoder can process batches of graphs."""
        # Set a fixed seed for reproducibility
        torch.manual_seed(42)

        # Create encoder
        encoder = GCNEncoder(self.input_dim, self.hidden_dim, self.latent_dim)

        # Force the model to evaluation mode to ensure consistent behavior
        encoder.eval()

        # Process batch
        with torch.no_grad():
            latent = encoder(self.batch)

        # Check output shape - should be [batch_size, latent_dim]
        self.assertEqual(latent.shape, torch.Size(
            [len(self.graphs), self.latent_dim]))

        # Test that each graph in the batch is processed independently
        # Process each graph individually
        individual_latents = []
        with torch.no_grad():
            for graph in self.graphs:
                individual_latents.append(encoder(graph))

        # Stack individual results
        stacked_latents = torch.stack(individual_latents)

        # Compare with batch processing result
        self.assertTrue(torch.allclose(
            latent, stacked_latents, rtol=1e-4, atol=1e-4))

    def test_different_pooling_methods(self):
        """Test different pooling methods."""
        pooling_methods = ["mean", "max", "sum"]

        for method in pooling_methods:
            encoder = GCNEncoder(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim,
                pooling=method
            )

            # Process graph
            latent = encoder(self.graph)

            # Check output shape
            self.assertEqual(latent.shape, torch.Size([self.latent_dim]))

            # Check batch processing
            batch_latent = encoder(self.batch)
            self.assertEqual(batch_latent.shape, torch.Size(
                [len(self.graphs), self.latent_dim]))

    def test_feature_preprocessing(self):
        """Test preprocessing of features with different dimensions."""
        # Create encoder
        encoder = GCNEncoder(
            input_dim=5, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim)

        # Test with fewer features than expected
        graph_small = Data(x=torch.randn(self.num_nodes, 2),
                           edge_index=self.edge_index)
        latent = encoder(graph_small)
        self.assertEqual(latent.shape, torch.Size([self.latent_dim]))

        # Test with more features than expected
        graph_large = Data(x=torch.randn(self.num_nodes, 8),
                           edge_index=self.edge_index)
        latent = encoder(graph_large)
        self.assertEqual(latent.shape, torch.Size([self.latent_dim]))

    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        # Create encoder with a small number of parameters for faster testing
        encoder = GCNEncoder(self.input_dim, 8, 8, num_layers=2)

        # Set requires_grad for input
        x = self.x.clone().detach().requires_grad_(True)
        graph = Data(x=x, edge_index=self.edge_index)

        # Forward pass
        latent = encoder(graph)

        # Compute loss (arbitrary)
        loss = latent.sum()

        # Backward pass
        loss.backward()

        # Check that gradients flow back to the input
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.all(x.grad == 0))

        # Check that model parameters have gradients
        for name, param in encoder.named_parameters():
            self.assertIsNotNone(param.grad)
            # Some parameters might be unused and have zero gradients
            # We expect at least some parameters to have non-zero gradients

        # Check at least one parameter in each layer has non-zero gradients
        layers_have_grad = [False] * len(encoder.convs)
        for name, param in encoder.named_parameters():
            for i, layer_name in enumerate([f"convs.{i}" for i in range(len(encoder.convs))]):
                if layer_name in name and param.grad is not None and not torch.all(param.grad == 0):
                    layers_have_grad[i] = True

        # Check all layers have at least some non-zero gradients
        self.assertTrue(all(layers_have_grad))

    def test_config_retrieval(self):
        """Test the configuration retrieval method."""
        # Create encoder with custom parameters
        encoder = GCNEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=4,
            dropout=0.2,
            activation="elu",
            pooling="sum",
            batch_norm=True
        )

        # Get configuration
        config = encoder.get_config()

        # Check configuration values
        self.assertEqual(config["input_dim"], self.input_dim)
        self.assertEqual(config["hidden_dim"], self.hidden_dim)
        self.assertEqual(config["latent_dim"], self.latent_dim)
        self.assertEqual(config["num_layers"], 4)
        self.assertEqual(config["dropout"], 0.2)
        self.assertEqual(config["activation"], "elu")
        self.assertEqual(config["pooling"], "sum")
        self.assertEqual(config["batch_norm"], True)
        self.assertEqual(config["type"], "GCNEncoder")


if __name__ == "__main__":
    unittest.main()
