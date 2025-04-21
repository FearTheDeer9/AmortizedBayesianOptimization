import torch
import unittest
import numpy as np
from torch_geometric.data import Data, Batch

from causal_meta.inference.models.graph_utils import GraphBatcher
from causal_meta.inference.models.gcn_encoder import GCNEncoder
from causal_meta.inference.models.gcn_decoder import GCNDecoder
from causal_meta.inference.models.gat_encoder import GATEncoder
from causal_meta.inference.models.gat_decoder import GATDecoder
from causal_meta.inference.models.wrapper import EncoderDecoderWrapper


def generate_random_graph(num_nodes, feature_dim=5, edge_prob=0.3, seed=None):
    """Generate a random graph with specified number of nodes."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Create node features
    x = torch.randn(num_nodes, feature_dim)

    # Create random edges
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and np.random.random() < edge_prob:
                edges.append([i, j])

    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


class TestBatchProcessing(unittest.TestCase):
    """Tests for batch processing with different graph sizes and structures."""

    def setUp(self):
        """Set up common test variables."""
        self.feature_dim = 5
        self.hidden_dim = 16
        self.latent_dim = 8

        # Create encoder and decoder
        self.encoder = GCNEncoder(
            input_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=2
        )

        self.decoder = GCNDecoder(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.feature_dim,
            num_nodes=1000  # Large enough for our tests
        )

        # Create wrapper with edge validation
        self.wrapper = EncoderDecoderWrapper(
            encoder=self.encoder,
            decoder=self.decoder,
            validate_edges=True,
            pad_graphs=True
        )

        # Create graph batcher
        self.batcher = GraphBatcher(pad_to_max=True)

    def test_extreme_size_variations(self):
        """Test processing graphs with extreme size variations."""
        # Create graphs with vastly different sizes
        graphs = [
            generate_random_graph(
                1, feature_dim=self.feature_dim),      # 1 node
            generate_random_graph(
                5, feature_dim=self.feature_dim),      # 5 nodes
            generate_random_graph(
                50, feature_dim=self.feature_dim),     # 50 nodes
            generate_random_graph(
                200, feature_dim=self.feature_dim)     # 200 nodes
        ]

        # Process through the wrapper
        results = self.wrapper.process_batch(graphs)

        # Check all graphs were processed
        self.assertEqual(len(results['reconstructed_graphs']), len(graphs))

        # Check sizes were preserved
        for i, graph in enumerate(results['reconstructed_graphs']):
            original_size = graphs[i].num_nodes
            self.assertEqual(graph.num_nodes, original_size)

            # Check all edge indices are valid
            if graph.edge_index.size(1) > 0:
                self.assertTrue((graph.edge_index[0] < original_size).all())
                self.assertTrue((graph.edge_index[1] < original_size).all())

    def test_dynamic_batching_mechanism(self):
        """Test the dynamic batching mechanism with node feature padding."""
        # Create graphs with different sizes
        graphs = [
            generate_random_graph(3, feature_dim=self.feature_dim),
            generate_random_graph(7, feature_dim=self.feature_dim),
            generate_random_graph(5, feature_dim=self.feature_dim)
        ]

        # Create batch with padding
        batch, batch_info = self.batcher.batch_graphs(graphs)

        # Check batch properties
        self.assertEqual(batch.x.size(0), 7 * len(graphs)
                         )  # Max nodes * num graphs
        # Should have batch assignments
        self.assertTrue(hasattr(batch, 'batch'))
        self.assertTrue(hasattr(batch, 'ptr'))    # Should have pointer array

        # Check ptr values
        self.assertEqual(len(batch.ptr), len(graphs) + 1)
        self.assertEqual(batch.ptr[0], 0)
        self.assertEqual(batch.ptr[-1], batch.x.size(0))

        # Check mask tracking (if implemented)
        if hasattr(batch, 'node_mask'):
            self.assertEqual(batch.node_mask.size(0), batch.x.size(0))

            # First graph (3 nodes)
            self.assertTrue(batch.node_mask[:3].all())
            self.assertTrue((~batch.node_mask[3:7]).all())  # Padding

            # Second graph (7 nodes)
            self.assertTrue(batch.node_mask[7:14].all())

            # Third graph (5 nodes)
            self.assertTrue(batch.node_mask[14:19].all())
            self.assertTrue((~batch.node_mask[19:21]).all())  # Padding

    def test_edge_index_offset_calculation(self):
        """Test edge index offset calculation in batched graphs."""
        # Create two simple graphs with known edge structure
        g1 = Data(
            x=torch.randn(3, self.feature_dim),
            edge_index=torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long)
        )

        g2 = Data(
            x=torch.randn(4, self.feature_dim),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        )

        # Create batch
        batch = Batch.from_data_list([g1, g2])

        # Check edge indices are correctly offset
        g1_edges = batch.edge_index[:, :3]  # First 3 edges from g1
        g2_edges = batch.edge_index[:, 3:]  # Remaining edges from g2

        # First graph edges should be unchanged
        self.assertTrue(torch.equal(g1_edges, g1.edge_index))

        # Second graph edges should be offset by first graph node count
        expected_g2_edges = g2.edge_index + torch.tensor([[3], [3]])
        self.assertTrue(torch.equal(g2_edges, expected_g2_edges))

    def test_batch_aware_loss_functions(self):
        """Test batch-aware loss computation that respects graph boundaries."""
        # Create graphs with different structures
        graphs = [
            generate_random_graph(4, feature_dim=self.feature_dim, seed=42),
            generate_random_graph(6, feature_dim=self.feature_dim, seed=43)
        ]

        # Forward pass through model
        reconstructed, latent = self.wrapper(graphs)

        # Compute loss
        loss_dict = self.wrapper.compute_loss(graphs, reconstructed, latent)

        # Should compute losses correctly
        self.assertIn('edge_loss', loss_dict)
        self.assertIn('feature_loss', loss_dict)
        self.assertIn('total_loss', loss_dict)

        # Losses should be finite
        self.assertTrue(torch.isfinite(loss_dict['edge_loss']))
        self.assertTrue(torch.isfinite(loss_dict['feature_loss']))
        self.assertTrue(torch.isfinite(loss_dict['total_loss']))

    def test_scatter_based_operations(self):
        """Test scatter-based operations for aggregating node features."""
        # This test requires inspecting the model internals

        # Create graphs with different sizes
        graphs = [
            generate_random_graph(5, feature_dim=self.feature_dim),
            generate_random_graph(8, feature_dim=self.feature_dim),
            generate_random_graph(3, feature_dim=self.feature_dim)
        ]

        # Create batch
        batch = Batch.from_data_list(graphs)

        # Process through encoder-decoder
        with torch.no_grad():
            # Should not raise errors related to scatter operations
            latent = self.encoder(batch)

            # Check latent dimension is correct
            self.assertEqual(latent.shape[0], len(graphs))
            self.assertEqual(latent.shape[1], self.latent_dim)


class TestEdgeIndexValidation(unittest.TestCase):
    """Tests for edge index validation across graph boundaries."""

    def setUp(self):
        """Set up common test variables."""
        self.feature_dim = 5
        self.hidden_dim = 16
        self.latent_dim = 8

        # Create encoder and decoder
        self.encoder = GCNEncoder(
            input_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=2
        )

        self.decoder = GCNDecoder(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.feature_dim,
            num_nodes=1000  # Large enough for our tests
        )

        # Create wrapper with edge validation
        self.wrapper = EncoderDecoderWrapper(
            encoder=self.encoder,
            decoder=self.decoder,
            validate_edges=True,
            pad_graphs=True
        )

        # Create graph batcher
        self.batcher = GraphBatcher(pad_to_max=True)

    def test_edge_index_boundary_validation(self):
        """Test validation of edge indices to ensure they stay within graph boundaries."""
        # Create graphs with different sizes
        graphs = [
            generate_random_graph(1, feature_dim=self.feature_dim),   # 1 node
            generate_random_graph(5, feature_dim=self.feature_dim),   # 5 nodes
            generate_random_graph(
                50, feature_dim=self.feature_dim),  # 50 nodes
            generate_random_graph(3, feature_dim=self.feature_dim)    # 3 nodes
        ]

        # Create batch
        batch, batch_info = self.batcher.batch_graphs(graphs)

        # Create some invalid edge indices that cross graph boundaries
        invalid_edges = torch.tensor([
            # Within first graph
            [0, 0],
            # Cross from graph 1 to graph 2
            [0, 10],
            # Cross from graph 2 to graph 3
            [7, 60],
            # Within third graph but invalid node index
            [55, 56],
            # Within fourth graph
            [155, 156]
        ], dtype=torch.long).t()

        # Validate edges
        valid_edges = self.batcher.validate_edge_indices(
            invalid_edges,
            batch.ptr,
            batch_info['num_nodes_per_graph']
        )

        # Only the valid edges should remain
        # First graph has only 1 node, so self-loop [0,0] is valid
        # Fourth graph: [155,156] crosses 3rd graph boundary (only 3 nodes)
        self.assertEqual(valid_edges.shape[1], 1)  # Only one valid edge: [0,0]

        if valid_edges.shape[1] > 0:
            self.assertEqual(valid_edges[0, 0].item(), 0)
            self.assertEqual(valid_edges[1, 0].item(), 0)

    def test_encoder_decoder_with_validation(self):
        """Test entire encoder-decoder pipeline with edge validation."""
        # Create graphs with different sizes
        graphs = [
            generate_random_graph(4, feature_dim=self.feature_dim),
            generate_random_graph(7, feature_dim=self.feature_dim),
            generate_random_graph(2, feature_dim=self.feature_dim)
        ]

        # Process through model
        results = self.wrapper.process_batch(graphs)
        reconstructed_graphs = results['reconstructed_graphs']

        # Check that reconstructed graphs have valid edge indices
        for i, graph in enumerate(reconstructed_graphs):
            original_size = graphs[i].num_nodes

            # All edges should reference valid nodes
            if graph.edge_index.size(1) > 0:
                self.assertTrue((graph.edge_index[0] < original_size).all())
                self.assertTrue((graph.edge_index[1] < original_size).all())

    def test_validate_edge_indices_implementation(self):
        """Test the implementation of validate_edge_indices method."""
        # Create batch with known structure
        g1 = Data(x=torch.randn(3, self.feature_dim),
                  edge_index=torch.zeros((2, 0), dtype=torch.long))
        g2 = Data(x=torch.randn(5, self.feature_dim),
                  edge_index=torch.zeros((2, 0), dtype=torch.long))

        batch, batch_info = self.batcher.batch_graphs([g1, g2])

        # Create edge indices with various validity scenarios
        edge_indices = torch.tensor([
            # Valid edges for g1
            [0, 1], [1, 2], [2, 0],
            # Invalid cross-boundary edges
            [2, 3], [4, 7],
            # Valid edges for g2
            [3, 4], [4, 5], [5, 6], [6, 7], [7, 3]
        ], dtype=torch.long).t()

        # Validate edges
        valid_indices = self.batcher.validate_edge_indices(
            edge_indices,
            batch.ptr,
            batch_info['num_nodes_per_graph']
        )

        # Should have 3 valid edges from g1 and 5 valid edges from g2
        self.assertEqual(valid_indices.shape[1], 8)

        # Check first 3 edges belong to g1
        g1_edges = valid_indices[:, :3]
        self.assertTrue((g1_edges[0] >= 0).all())
        self.assertTrue((g1_edges[0] < 3).all())
        self.assertTrue((g1_edges[1] >= 0).all())
        self.assertTrue((g1_edges[1] < 3).all())

        # Check remaining 5 edges belong to g2
        g2_edges = valid_indices[:, 3:]
        self.assertTrue((g2_edges[0] >= 3).all())
        self.assertTrue((g2_edges[0] < 8).all())
        self.assertTrue((g2_edges[1] >= 3).all())
        self.assertTrue((g2_edges[1] < 8).all())

    def test_batch_boundary_tracking(self):
        """Test batch boundary tracking properties for efficient graph boundary lookup."""
        # Create graphs with different sizes
        graphs = [
            generate_random_graph(2, feature_dim=self.feature_dim),
            generate_random_graph(4, feature_dim=self.feature_dim),
            generate_random_graph(3, feature_dim=self.feature_dim)
        ]

        # Create batch
        batch, batch_info = self.batcher.batch_graphs(graphs)

        # Check ptr property
        self.assertTrue(hasattr(batch, 'ptr'))
        self.assertEqual(len(batch.ptr), len(graphs) + 1)
        self.assertEqual(batch.ptr[0], 0)

        # Check ptr values match expected boundaries
        expected_ptr = [0, 2, 6, 9]  # 0, 0+2, 2+4, 6+3
        for i, val in enumerate(expected_ptr):
            self.assertEqual(batch.ptr[i].item(), val)

        # Check ptr_pairs property if implemented
        if hasattr(batch, 'ptr_pairs'):
            self.assertEqual(len(batch.ptr_pairs), len(graphs))

            # First pair should be (0, 2)
            self.assertEqual(batch.ptr_pairs[0][0].item(), 0)
            self.assertEqual(batch.ptr_pairs[0][1].item(), 2)

            # Second pair should be (2, 6)
            self.assertEqual(batch.ptr_pairs[1][0].item(), 2)
            self.assertEqual(batch.ptr_pairs[1][1].item(), 6)

            # Third pair should be (6, 9)
            self.assertEqual(batch.ptr_pairs[2][0].item(), 6)
            self.assertEqual(batch.ptr_pairs[2][1].item(), 9)


class TestGATBatchProcessing(unittest.TestCase):
    """Tests for batch processing with GAT models that have attention mechanisms."""

    def setUp(self):
        """Set up common test variables."""
        self.feature_dim = 5
        self.hidden_dim = 16
        self.latent_dim = 8

        # Create GAT encoder and decoder
        self.encoder = GATEncoder(
            input_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=2,
            heads=2
        )

        self.decoder = GATDecoder(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.feature_dim,
            num_nodes=50,
            heads=2
        )

        # Create wrapper
        self.wrapper = EncoderDecoderWrapper(
            encoder=self.encoder,
            decoder=self.decoder,
            validate_edges=True,
            pad_graphs=True
        )

    def test_batch_aware_attention_mechanisms(self):
        """Test batch-aware attention mechanisms in GAT models."""
        # Create graphs with different sizes
        graphs = [
            generate_random_graph(5, feature_dim=self.feature_dim),
            generate_random_graph(8, feature_dim=self.feature_dim),
            generate_random_graph(3, feature_dim=self.feature_dim)
        ]

        # Process through model
        with torch.no_grad():
            batch = Batch.from_data_list(graphs)

            # Should not raise errors related to attention mechanisms
            latent = self.encoder(batch)

            # Check latent dimension is correct
            self.assertEqual(latent.shape[0], len(graphs))
            self.assertEqual(latent.shape[1], self.latent_dim)

            # Forward through decoder
            reconstructed = self.decoder(latent, batch)

            # Should have valid edge indices
            if hasattr(reconstructed, 'edge_index') and reconstructed.edge_index.shape[1] > 0:
                for i, (start, end) in enumerate(zip(batch.ptr[:-1], batch.ptr[1:])):
                    # Get edges for this graph
                    mask = ((reconstructed.edge_index[0] >= start) &
                            (reconstructed.edge_index[0] < end))
                    graph_edges = reconstructed.edge_index[:, mask]

                    if graph_edges.shape[1] > 0:
                        # All edges should be within this graph's boundaries
                        self.assertTrue((graph_edges[0] >= start).all())
                        self.assertTrue((graph_edges[0] < end).all())
                        self.assertTrue((graph_edges[1] >= start).all())
                        self.assertTrue((graph_edges[1] < end).all())


if __name__ == "__main__":
    unittest.main()
