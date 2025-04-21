"""
Tests for the EncoderDecoderWrapper class.
"""

import unittest
import torch
import numpy as np
import os
import shutil
import tempfile
from torch_geometric.data import Data, Batch

from causal_meta.inference.models.wrapper import EncoderDecoderWrapper, create_model
from causal_meta.inference.models.gcn_encoder import GCNEncoder
from causal_meta.inference.models.gcn_decoder import GCNDecoder
from causal_meta.inference.models.gat_encoder import GATEncoder
from causal_meta.inference.models.gat_decoder import GATDecoder


class TestEncoderDecoderWrapper(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        # Common test parameters
        self.input_dim = 3
        self.hidden_dim = 16
        self.latent_dim = 32
        self.output_dim = 3
        self.num_nodes = 5

        # Create a simple graph for testing
        self.x = torch.randn(self.num_nodes, self.input_dim)
        self.edge_index = torch.tensor(
            [[0, 1, 1, 2, 3], [1, 0, 2, 3, 4]], dtype=torch.long)
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

        # Create encoder and decoder for testing
        self.encoder = GCNEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim
        )
        self.decoder = GCNDecoder(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_nodes=self.num_nodes
        )

        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test that the wrapper can be initialized with different parameters."""
        # Default parameters
        wrapper = EncoderDecoderWrapper(self.encoder, self.decoder)
        self.assertEqual(wrapper.edge_weight, 1.0)
        self.assertEqual(wrapper.feature_weight, 0.5)
        self.assertEqual(wrapper.kl_weight, 0.0)
        self.assertEqual(wrapper.l2_weight, 0.0)
        self.assertEqual(wrapper.loss_type, "bce")

        # Custom parameters
        wrapper = EncoderDecoderWrapper(
            encoder=self.encoder,
            decoder=self.decoder,
            loss_type="mse",
            edge_weight=0.8,
            feature_weight=0.2,
            kl_weight=0.1,
            l2_weight=0.01
        )
        self.assertEqual(wrapper.edge_weight, 0.8)
        self.assertEqual(wrapper.feature_weight, 0.2)
        self.assertEqual(wrapper.kl_weight, 0.1)
        self.assertEqual(wrapper.l2_weight, 0.01)
        self.assertEqual(wrapper.loss_type, "mse")

    def test_forward_pass(self):
        """Test the forward pass of the wrapper."""
        # Create wrapper
        wrapper = EncoderDecoderWrapper(self.encoder, self.decoder)

        # Run forward pass
        reconstructed_graph, latent = wrapper(self.graph)

        # Check output types
        self.assertIsInstance(reconstructed_graph, Data)
        self.assertIsInstance(latent, torch.Tensor)

        # Check output shapes
        self.assertEqual(latent.shape, torch.Size([self.latent_dim]))
        self.assertEqual(reconstructed_graph.x.shape[1], self.output_dim)

    def test_batch_processing(self):
        """Test that the wrapper can process batches of graphs."""
        # Create wrapper with appropriately sized decoder for varying node counts
        batch_decoder = GCNDecoder(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_nodes=5  # Maximum nodes in the batch
        )
        wrapper = EncoderDecoderWrapper(self.encoder, batch_decoder)

        # Process individual graphs
        individual_latents = []
        for graph in self.graphs:
            _, latent = wrapper(graph)
            individual_latents.append(latent)

        # Process batch
        _, batch_latents = wrapper(self.batch)

        # Check output shape
        self.assertEqual(batch_latents.shape, torch.Size(
            [len(self.graphs), self.latent_dim]))

        # Compare with individual processing results - note that exact values may differ
        # due to batch normalization, so we just check shapes
        self.assertEqual(len(individual_latents), batch_latents.shape[0])
        for i, latent in enumerate(individual_latents):
            self.assertEqual(latent.shape, batch_latents[i].shape)

    def test_loss_computation(self):
        """Test loss computation methods."""
        # Create wrapper
        wrapper = EncoderDecoderWrapper(
            encoder=self.encoder,
            decoder=self.decoder,
            edge_weight=1.0,
            feature_weight=1.0,
            l2_weight=0.1
        )

        # Get reconstructed graph
        reconstructed_graph, latent = wrapper(self.graph)

        # Compute losses
        losses = wrapper.compute_loss(self.graph, reconstructed_graph, latent)

        # Check that all expected loss components are present
        self.assertIn('edge_loss', losses)
        self.assertIn('feature_loss', losses)
        self.assertIn('l2_loss', losses)
        self.assertIn('total_loss', losses)

        # Check loss values are finite
        for loss_name, loss_value in losses.items():
            self.assertTrue(torch.isfinite(loss_value),
                            f"{loss_name} is not finite")

        # Check that total loss is correctly computed
        expected_total = (losses['edge_loss'] + losses['feature_loss'] +
                          0.1 * losses['l2_loss'])
        self.assertTrue(torch.allclose(losses['total_loss'], expected_total))

    def test_edge_loss_types(self):
        """Test different edge loss types."""
        loss_types = ['bce', 'mse', 'weighted_bce']

        for loss_type in loss_types:
            # Create wrapper with specific loss type
            wrapper = EncoderDecoderWrapper(
                encoder=self.encoder,
                decoder=self.decoder,
                loss_type=loss_type
            )

            # Get reconstructed graph
            reconstructed_graph, latent = wrapper(self.graph)

            # Compute losses
            losses = wrapper.compute_loss(
                self.graph, reconstructed_graph, latent)

            # Check loss value
            self.assertTrue(torch.isfinite(losses['edge_loss']))
            self.assertTrue(losses['edge_loss'] >= 0)

    def test_evaluation_metrics(self):
        """Test evaluation metrics computation."""
        # Create wrapper
        wrapper = EncoderDecoderWrapper(self.encoder, self.decoder)

        # Get metrics
        metrics = wrapper.evaluate(self.graph)

        # Check metrics
        self.assertIn('loss', metrics)
        self.assertIn('edge_precision', metrics)
        self.assertIn('edge_recall', metrics)
        self.assertIn('edge_f1', metrics)
        self.assertIn('reconstruction_accuracy', metrics)

        # Check metric values are reasonable
        self.assertTrue(0 <= metrics['edge_precision'] <= 1)
        self.assertTrue(0 <= metrics['edge_recall'] <= 1)
        self.assertTrue(0 <= metrics['edge_f1'] <= 1)
        self.assertTrue(0 <= metrics['reconstruction_accuracy'] <= 1)

    def test_configuration(self):
        """Test configuration retrieval."""
        # Create wrapper
        wrapper = EncoderDecoderWrapper(
            encoder=self.encoder,
            decoder=self.decoder,
            loss_type="weighted_bce",
            edge_weight=0.7,
            feature_weight=0.3
        )

        # Get configuration
        config = wrapper.get_config()

        # Check configuration values
        self.assertEqual(config["encoder_type"], "GCNEncoder")
        self.assertEqual(config["decoder_type"], "GCNDecoder")
        self.assertEqual(config["loss_type"], "weighted_bce")
        self.assertEqual(config["edge_weight"], 0.7)
        self.assertEqual(config["feature_weight"], 0.3)
        self.assertIn("encoder_config", config)
        self.assertIn("decoder_config", config)

    def test_create_model_factory(self):
        """Test the create_model factory function."""
        # Test with different architectures
        architectures = ['gcn', 'gat']

        for arch in architectures:
            # Create model
            model = create_model(
                architecture=arch,
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim,
                output_dim=self.output_dim,
                num_nodes=self.num_nodes,
                loss_type="bce",
                edge_weight=0.8,
                feature_weight=0.2,
                encoder_dropout=0.1,
                decoder_edge_prediction_method="inner_product"
            )

            # Check model type
            self.assertIsInstance(model, EncoderDecoderWrapper)

            # Check encoder type
            if arch == 'gcn':
                self.assertIsInstance(model.encoder, GCNEncoder)
                self.assertIsInstance(model.decoder, GCNDecoder)
            elif arch == 'gat':
                self.assertIsInstance(model.encoder, GATEncoder)
                self.assertIsInstance(model.decoder, GATDecoder)

            # Check model configuration
            config = model.get_config()
            self.assertEqual(config["loss_type"], "bce")
            self.assertEqual(config["edge_weight"], 0.8)
            self.assertEqual(config["feature_weight"], 0.2)
            self.assertEqual(config["encoder_config"]
                             ["input_dim"], self.input_dim)
            self.assertEqual(config["encoder_config"]
                             ["hidden_dim"], self.hidden_dim)
            self.assertEqual(config["encoder_config"]
                             ["latent_dim"], self.latent_dim)
            self.assertEqual(config["decoder_config"]
                             ["output_dim"], self.output_dim)
            self.assertEqual(config["decoder_config"]
                             ["num_nodes"], self.num_nodes)

            # Test forward pass
            reconstructed_graph, latent = model(self.graph)
            self.assertIsInstance(reconstructed_graph, Data)
            self.assertIsInstance(latent, torch.Tensor)

    def test_regularization_options(self):
        """Test different regularization options."""
        # Create wrapper with KL divergence regularization
        wrapper_kl = EncoderDecoderWrapper(
            encoder=self.encoder,
            decoder=self.decoder,
            kl_weight=0.1,
            l2_weight=0.0
        )

        # Create wrapper with L2 regularization
        wrapper_l2 = EncoderDecoderWrapper(
            encoder=self.encoder,
            decoder=self.decoder,
            kl_weight=0.0,
            l2_weight=0.1
        )

        # Create wrapper with both regularizations
        wrapper_both = EncoderDecoderWrapper(
            encoder=self.encoder,
            decoder=self.decoder,
            kl_weight=0.1,
            l2_weight=0.1
        )

        # Get reconstructed graphs
        reconstructed_graph_kl, latent_kl = wrapper_kl(self.graph)
        reconstructed_graph_l2, latent_l2 = wrapper_l2(self.graph)
        reconstructed_graph_both, latent_both = wrapper_both(self.graph)

        # Compute losses
        losses_kl = wrapper_kl.compute_loss(
            self.graph, reconstructed_graph_kl, latent_kl)
        losses_l2 = wrapper_l2.compute_loss(
            self.graph, reconstructed_graph_l2, latent_l2)
        losses_both = wrapper_both.compute_loss(
            self.graph, reconstructed_graph_both, latent_both)

        # Check regularization losses
        self.assertIn('kl_loss', losses_kl)
        self.assertIn('l2_loss', losses_l2)
        self.assertIn('kl_loss', losses_both)
        self.assertIn('l2_loss', losses_both)

        # Check that regularization is applied correctly
        self.assertTrue(losses_kl['kl_loss'] > 0)
        self.assertEqual(losses_kl['l2_loss'].item(), 0.0)
        self.assertEqual(losses_l2['kl_loss'].item(), 0.0)
        self.assertTrue(losses_l2['l2_loss'] > 0)
        self.assertTrue(losses_both['kl_loss'] > 0)
        self.assertTrue(losses_both['l2_loss'] > 0)

    def test_unsupported_architecture(self):
        """Test error handling for unsupported architectures."""
        with self.assertRaises(ValueError):
            create_model(
                architecture='unsupported',
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim,
                output_dim=self.output_dim,
                num_nodes=self.num_nodes
            )


if __name__ == "__main__":
    unittest.main()
