"""
Tests for base GNN encoder and decoder classes.
"""

from causal_meta.inference.models.decoder import BaseGNNDecoder
from causal_meta.inference.models.encoder import BaseGNNEncoder
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Minimal concrete implementation of BaseGNNEncoder for testing
class MinimalEncoder(BaseGNNEncoder):
    def __init__(self, input_dim=2, hidden_dim=8, latent_dim=16):
        super().__init__(input_dim, hidden_dim, latent_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, latent_dim)

    def encode(self, graph_data):
        x = graph_data.x
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        # Perform global mean pooling to get a graph-level representation
        graph_embedding = x.mean(dim=0)
        return graph_embedding

    def forward(self, graph_data):
        return self.encode(graph_data)


# Minimal concrete implementation of BaseGNNDecoder for testing
class MinimalDecoder(BaseGNNDecoder):
    def __init__(self, latent_dim=16, hidden_dim=8, output_dim=2, num_nodes=5):
        super().__init__(latent_dim, hidden_dim, output_dim, num_nodes)
        self.linear1 = nn.Linear(latent_dim, hidden_dim * num_nodes)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.edge_predictor = nn.Linear(hidden_dim * 2, 1)

    def decode(self, latent_representation):
        # Generate node features
        node_features = self.predict_node_features(latent_representation)

        # Generate edges
        edge_index, edge_attr = self.predict_edges(node_features)

        # Create graph
        graph = Data(x=node_features, edge_index=edge_index,
                     edge_attr=edge_attr)

        return graph

    def forward(self, latent_representation):
        return self.decode(latent_representation)

    def predict_node_features(self, latent_representation):
        # Transform latent rep to features for all nodes
        hidden = self.linear1(latent_representation).view(self.num_nodes, -1)
        features = self.linear2(hidden)
        return features

    def predict_edges(self, node_embeddings):
        # Create all possible node pairs
        num_nodes = node_embeddings.shape[0]

        # List of all source and target node combinations
        source_nodes = []
        target_nodes = []

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # Exclude self-loops
                    source_nodes.append(i)
                    target_nodes.append(j)

        edge_index = torch.tensor(
            [source_nodes, target_nodes], dtype=torch.long)

        # Get node features for each node in the pairs
        source_features = node_embeddings[edge_index[0]]
        target_features = node_embeddings[edge_index[1]]

        # Concatenate features and predict edge existence
        edge_features = torch.cat([source_features, target_features], dim=1)
        edge_logits = self.edge_predictor(edge_features)
        edge_probs = torch.sigmoid(edge_logits).squeeze()

        return edge_index, edge_probs


class TestGNNBaseClasses(unittest.TestCase):

    def test_encoder_initialization(self):
        """Test that the encoder can be initialized."""
        encoder = MinimalEncoder(input_dim=3, hidden_dim=10, latent_dim=20)
        self.assertEqual(encoder.input_dim, 3)
        self.assertEqual(encoder.hidden_dim, 10)
        self.assertEqual(encoder.latent_dim, 20)

    def test_decoder_initialization(self):
        """Test that the decoder can be initialized."""
        decoder = MinimalDecoder(
            latent_dim=20, hidden_dim=10, output_dim=3, num_nodes=7)
        self.assertEqual(decoder.latent_dim, 20)
        self.assertEqual(decoder.hidden_dim, 10)
        self.assertEqual(decoder.output_dim, 3)
        self.assertEqual(decoder.num_nodes, 7)

    def test_encoder_forward(self):
        """Test the encoder forward pass."""
        encoder = MinimalEncoder(input_dim=3, hidden_dim=10, latent_dim=20)

        # Create a simple graph
        num_nodes = 5
        x = torch.randn(num_nodes, 3)  # 5 nodes with 3 features each
        edge_index = torch.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)  # Simple path
        graph = Data(x=x, edge_index=edge_index)

        # Run forward pass
        embedding = encoder(graph)

        # Check output shape
        self.assertEqual(embedding.shape, torch.Size([20]))

    def test_decoder_forward(self):
        """Test the decoder forward pass."""
        decoder = MinimalDecoder(
            latent_dim=20, hidden_dim=10, output_dim=3, num_nodes=5)

        # Create a latent representation
        latent = torch.randn(20)

        # Run forward pass
        reconstructed_graph = decoder(latent)

        # Check output
        self.assertIsInstance(reconstructed_graph, Data)
        self.assertEqual(reconstructed_graph.x.shape,
                         torch.Size([5, 3]))  # 5 nodes with 3 features
        # Edge index has 2 rows
        self.assertEqual(reconstructed_graph.edge_index.shape[0], 2)

    def test_end_to_end(self):
        """Test encoder-decoder pipeline end-to-end."""
        encoder = MinimalEncoder(input_dim=3, hidden_dim=10, latent_dim=20)
        decoder = MinimalDecoder(
            latent_dim=20, hidden_dim=10, output_dim=3, num_nodes=5)

        # Create a simple graph
        num_nodes = 5
        x = torch.randn(num_nodes, 3)  # 5 nodes with 3 features each
        edge_index = torch.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)  # Simple path
        original_graph = Data(x=x, edge_index=edge_index)

        # Encode and decode
        latent = encoder(original_graph)
        reconstructed_graph = decoder(latent)

        # Check output
        self.assertIsInstance(reconstructed_graph, Data)
        self.assertEqual(reconstructed_graph.x.shape,
                         torch.Size([5, 3]))  # 5 nodes with 3 features
        # Edge index has 2 rows
        self.assertEqual(reconstructed_graph.edge_index.shape[0], 2)

    def test_config_retrieval(self):
        """Test that configurations can be retrieved."""
        encoder = MinimalEncoder(input_dim=3, hidden_dim=10, latent_dim=20)
        decoder = MinimalDecoder(
            latent_dim=20, hidden_dim=10, output_dim=3, num_nodes=5)

        encoder_config = encoder.get_config()
        decoder_config = decoder.get_config()

        self.assertEqual(encoder_config["input_dim"], 3)
        self.assertEqual(encoder_config["hidden_dim"], 10)
        self.assertEqual(encoder_config["latent_dim"], 20)
        self.assertEqual(encoder_config["type"], "MinimalEncoder")

        self.assertEqual(decoder_config["latent_dim"], 20)
        self.assertEqual(decoder_config["hidden_dim"], 10)
        self.assertEqual(decoder_config["output_dim"], 3)
        self.assertEqual(decoder_config["num_nodes"], 5)
        self.assertEqual(decoder_config["type"], "MinimalDecoder")


if __name__ == "__main__":
    unittest.main()
