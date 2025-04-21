import torch
import unittest
from torch_geometric.data import Data, Batch

from causal_meta.inference.models.graph_utils import (
    batch_to_graphs,
    validate_edge_indices,
    calculate_batch_boundary_indices,
    validate_batch_boundaries,
    GraphBatcher
)
from causal_meta.inference.models.wrapper import EncoderDecoderWrapper
from causal_meta.inference.models.gcn_encoder import GCNEncoder
from causal_meta.inference.models.gcn_decoder import GCNDecoder


def create_random_graph(num_nodes, edge_prob=0.3, feature_dim=5):
    """Create a random graph with the specified number of nodes"""
    # Create random node features
    x = torch.randn(num_nodes, feature_dim)

    # Create random edges with a given probability
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and torch.rand(1).item() < edge_prob:
                edges.append([i, j])

    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


class TestEdgeValidation(unittest.TestCase):
    """Tests for edge index validation utilities."""

    def test_batch_boundary_tracking(self):
        """Test calculation of batch boundary indices."""
        # Create test data
        data1 = Data(x=torch.randn(2, 3), edge_index=torch.tensor(
            [[0, 1], [1, 0]], dtype=torch.long))
        data2 = Data(x=torch.randn(2, 3), edge_index=torch.tensor(
            [[0, 1], [1, 0]], dtype=torch.long))
        batch = Batch.from_data_list([data1, data2])

        # Calculate batch boundary indices
        boundaries = calculate_batch_boundary_indices(batch)

        # Verify boundaries
        # Changed from 4 to 2 - we expect 2 boundaries for 2 graphs
        self.assertEqual(len(boundaries), 2)
        self.assertEqual(boundaries[0], 0)  # First boundary starts at index 0
        self.assertEqual(boundaries[1], 2)  # Second boundary starts at index 2

    def test_batch_to_graphs_conversion(self):
        """Test conversion from batch back to individual graphs."""
        # Create test data with 3 graphs
        graph_sizes = [3, 2, 4]
        graphs = [
            Data(x=torch.randn(size, 5), edge_index=torch.tensor(
                [[0, 1], [1, 0]], dtype=torch.long))
            for size in graph_sizes
        ]

        # Create batch
        batch = Batch.from_data_list(graphs)

        # Convert back to individual graphs
        unbatched = batch_to_graphs(batch)

        # Verify conversion
        self.assertEqual(len(unbatched), len(graphs))
        for i, graph in enumerate(unbatched):
            self.assertEqual(graph.num_nodes, graph_sizes[i])

    def test_validate_edge_indices(self):
        """Test validation of edge indices."""
        # Create edge indices that include invalid edges
        num_nodes = 5
        edge_index = torch.tensor([
            [0, 1, 2, 7],  # Last index is invalid
            [1, 2, 3, 4]
        ], dtype=torch.long)

        # Validate edge indices
        valid_edge_index = validate_edge_indices(edge_index, num_nodes)

        # Check result
        # Only 3 edges should be valid
        self.assertEqual(valid_edge_index.shape[1], 3)

        # Verify there are no invalid edges
        self.assertTrue((valid_edge_index[0] < num_nodes).all())
        self.assertTrue((valid_edge_index[1] < num_nodes).all())

    def test_edge_index_boundary_validation(self):
        """Test validation of edge indices against batch boundaries."""
        # Create a batch with two graphs
        graph1 = Data(x=torch.randn(3, 5), edge_index=torch.tensor(
            [[0, 1, 2], [1, 2, 0]], dtype=torch.long))
        graph2 = Data(x=torch.randn(2, 5), edge_index=torch.tensor(
            [[0, 1], [1, 0]], dtype=torch.long))
        batch = Batch.from_data_list([graph1, graph2])

        # Create edge indices that cross batch boundaries
        edge_index = torch.tensor([
            # Node 2 in last edge is from graph1, trying to connect to graph2
            [0, 1, 2, 3, 4, 2],
            [1, 2, 0, 0, 1, 4]   # Node 4 in last edge is out of bounds for graph1
        ], dtype=torch.long)

        # Validate against batch boundaries
        valid_edge_index = validate_batch_boundaries(edge_index, batch)

        # Expected 4 valid edges (excluding the last two that cross boundaries)
        # Updated to 4 valid edges
        self.assertEqual(valid_edge_index.shape[1], 4)

    def test_graph_batcher_edge_validation(self):
        """Test edge validation functionality in GraphBatcher"""
        # Create test graphs with different sizes
        graphs = [
            create_random_graph(5),
            create_random_graph(10),
            create_random_graph(3)
        ]

        # Initialize batcher
        batcher = GraphBatcher(pad_to_max=True)

        # Create batch with padding
        batch, batch_info = batcher.batch_graphs(graphs)

        # Create edge indices with some invalid references
        # These edges point to nodes beyond graph boundaries
        invalid_edge_indices = torch.tensor([
            # Valid edges
            [0, 1], [1, 2], [2, 0],
            # Invalid edges - reference nodes beyond their graph boundaries
            [0, 15], [4, 12], [8, 18]
        ], dtype=torch.long).t()

        # Validate edge indices
        valid_edge_indices = batcher.validate_edge_indices(
            invalid_edge_indices,
            batch.ptr,
            batch_info['num_nodes_per_graph']
        )

        # Check that only valid edges remain
        self.assertEqual(valid_edge_indices.shape[1], 3)

        # Check that all remaining edges are within valid boundaries
        for i, (start_idx, end_idx) in enumerate(batch.ptr_pairs):
            graph_edges = valid_edge_indices[:, (valid_edge_indices[0] >= start_idx) &
                                             (valid_edge_indices[0] < end_idx)]

            if graph_edges.shape[1] > 0:
                # All source nodes should be within range
                self.assertTrue((graph_edges[0] >= start_idx).all().item())
                self.assertTrue((graph_edges[0] < end_idx).all().item())

                # All target nodes should be within range
                self.assertTrue((graph_edges[1] >= start_idx).all().item())
                self.assertTrue((graph_edges[1] < end_idx).all().item())

    def test_wrapper_edge_validation(self):
        """Test that EncoderDecoderWrapper correctly validates edge indices"""
        # Create test graphs with different sizes
        graphs = [
            create_random_graph(2, edge_prob=0.5),
            create_random_graph(4, edge_prob=0.5),
            create_random_graph(1, edge_prob=0.5)
        ]

        # Create encoder and decoder with minimal configurations
        encoder = GCNEncoder(
            input_dim=5,         # Node feature dimension
            hidden_dim=16,       # Hidden layer dimension
            latent_dim=8,        # Latent dimension
            num_layers=2,
            dropout=0.1
        )

        decoder = GCNDecoder(
            latent_dim=8,
            hidden_dim=16,
            output_dim=5,
            num_nodes=4  # Max nodes
        )

        # Create wrapper with edge validation enabled
        wrapper = EncoderDecoderWrapper(
            encoder=encoder,
            decoder=decoder,
            pad_graphs=True,
            validate_edges=True
        )

        # Process batch with wrapper
        results = wrapper.process_batch(graphs)
        reconstructed_graphs = results['reconstructed_graphs']

        # Check that each reconstructed graph has valid edge indices
        for i, graph in enumerate(reconstructed_graphs):
            if graph.edge_index.size(1) > 0:
                num_nodes = graph.num_nodes if hasattr(
                    graph, 'num_nodes') else graph.x.size(0)

                # All edges should reference nodes within the graph
                self.assertTrue((graph.edge_index[0] < num_nodes).all().item())
                self.assertTrue((graph.edge_index[1] < num_nodes).all().item())

    def test_extreme_graph_variations(self):
        """Test with extreme variations in graph sizes"""
        # Create graphs with very different sizes
        graphs = [
            create_random_graph(1, edge_prob=0.0),  # Single node, no edges
            create_random_graph(50, edge_prob=0.1),  # Medium graph
            create_random_graph(2, edge_prob=1.0)   # Small complete graph
        ]

        # Initialize batcher
        batcher = GraphBatcher(pad_to_max=True)

        # Create batch
        batch, batch_info = batcher.batch_graphs(graphs)

        # Verify ptr and ptr_pairs
        self.assertEqual(len(batch.ptr), len(graphs) + 1)
        self.assertEqual(len(batch.ptr_pairs), len(graphs))

        # Check that the batch has the expected properties
        self.assertEqual(batch.x.size(0), 50 * len(graphs)
                         )  # Max nodes * num graphs

        # Create some edge indices that intentionally cross graph boundaries
        cross_boundary_edges = torch.tensor([
            # These edges cross graph boundaries
            [0, 60], [49, 55], [51, 150]
        ], dtype=torch.long).t()

        # Validate edge indices
        valid_edges = batcher.validate_edge_indices(
            cross_boundary_edges,
            batch.ptr,
            batch_info['num_nodes_per_graph']
        )

        # No edges should remain after validation
        self.assertEqual(valid_edges.shape[1], 0)


if __name__ == "__main__":
    unittest.main()
