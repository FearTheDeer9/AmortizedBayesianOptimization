"""
Tests for graph data loading utilities.
"""

import unittest
import torch
import numpy as np
import os
import shutil
import tempfile
from torch_geometric.data import Data, Batch, Dataset

from causal_meta.inference.training.data_loader import (
    GraphDataset, RandomGraphGenerator, create_data_loaders
)


class TestGraphDataLoading(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()

        # Sample graph parameters
        self.input_dim = 3
        self.num_nodes = 5

        # Create a simple graph for testing
        self.x = torch.randn(self.num_nodes, self.input_dim)
        self.edge_index = torch.tensor(
            [[0, 1, 1, 2, 3], [1, 0, 2, 3, 4]], dtype=torch.long)
        self.graph = Data(x=self.x, edge_index=self.edge_index)

        # Create sample graphs for dataset
        self.graphs = [
            Data(x=torch.randn(4, self.input_dim),
                 edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)),
            Data(x=torch.randn(5, self.input_dim),
                 edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)),
            Data(x=torch.randn(3, self.input_dim),
                 edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long))
        ]

    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)

    def test_graph_dataset_creation(self):
        """Test creating a graph dataset from a list of graphs."""
        # Create dataset
        dataset = GraphDataset(
            root=os.path.join(self.test_dir, 'test_dataset'),
            graphs=self.graphs
        )

        # Check dataset properties
        self.assertEqual(len(dataset), len(self.graphs))

        # Check that processed file was created
        processed_file = os.path.join(
            self.test_dir, 'test_dataset', 'processed', 'processed_data.pt')
        self.assertTrue(os.path.exists(processed_file))

        # Check accessing individual graphs
        for i in range(len(dataset)):
            graph = dataset[i]
            self.assertIsInstance(graph, Data)
            self.assertTrue(hasattr(graph, 'x'))
            self.assertTrue(hasattr(graph, 'edge_index'))

    def test_random_graph_generator_erdos_renyi(self):
        """Test generating random Erdos-Renyi graphs."""
        # Generate a single graph
        graph = RandomGraphGenerator.create_erdos_renyi_graph(
            num_nodes=10,
            edge_prob=0.3,
            node_features_dim=4
        )

        # Check graph properties
        self.assertIsInstance(graph, Data)
        self.assertEqual(graph.x.shape, (10, 4))
        # 2 rows for source/target
        self.assertEqual(graph.edge_index.shape[0], 2)

        # Generate multiple graphs
        num_graphs = 5
        graphs = RandomGraphGenerator.generate_dataset(
            num_graphs=num_graphs,
            min_nodes=5,
            max_nodes=10,
            graph_type='erdos_renyi',
            edge_prob=0.3,
            node_features_dim=4
        )

        # Check graphs
        self.assertEqual(len(graphs), num_graphs)
        for graph in graphs:
            self.assertIsInstance(graph, Data)
            self.assertTrue(5 <= graph.x.shape[0] <= 10)  # Node count in range
            self.assertEqual(graph.x.shape[1], 4)  # Feature dimension

    def test_random_graph_generator_barabasi_albert(self):
        """Test generating random Barabasi-Albert graphs."""
        # Generate a single graph
        graph = RandomGraphGenerator.create_barabasi_albert_graph(
            num_nodes=10,
            num_edges=2,
            node_features_dim=4
        )

        # Check graph properties
        self.assertIsInstance(graph, Data)
        self.assertEqual(graph.x.shape, (10, 4))
        # 2 rows for source/target
        self.assertEqual(graph.edge_index.shape[0], 2)

        # Generate multiple graphs
        num_graphs = 5
        graphs = RandomGraphGenerator.generate_dataset(
            num_graphs=num_graphs,
            min_nodes=5,
            max_nodes=10,
            graph_type='barabasi_albert',
            num_edges=3,
            node_features_dim=4
        )

        # Check graphs
        self.assertEqual(len(graphs), num_graphs)
        for graph in graphs:
            self.assertIsInstance(graph, Data)
            self.assertTrue(5 <= graph.x.shape[0] <= 10)  # Node count in range
            self.assertEqual(graph.x.shape[1], 4)  # Feature dimension

    def test_random_graph_generator_watts_strogatz(self):
        """Test generating random Watts-Strogatz graphs."""
        # Generate a single graph
        graph = RandomGraphGenerator.create_watts_strogatz_graph(
            num_nodes=10,
            k=4,
            p=0.1,
            node_features_dim=4
        )

        # Check graph properties
        self.assertIsInstance(graph, Data)
        self.assertEqual(graph.x.shape, (10, 4))
        # 2 rows for source/target
        self.assertEqual(graph.edge_index.shape[0], 2)

        # Generate multiple graphs
        num_graphs = 5
        graphs = RandomGraphGenerator.generate_dataset(
            num_graphs=num_graphs,
            min_nodes=5,
            max_nodes=10,
            graph_type='watts_strogatz',
            k=4,
            p=0.1,
            node_features_dim=4
        )

        # Check graphs
        self.assertEqual(len(graphs), num_graphs)
        for graph in graphs:
            self.assertIsInstance(graph, Data)
            self.assertTrue(5 <= graph.x.shape[0] <= 10)  # Node count in range
            self.assertEqual(graph.x.shape[1], 4)  # Feature dimension

    def test_create_data_loaders(self):
        """Test creating train/val/test data loaders."""
        # Create dataset
        dataset = GraphDataset(
            root=os.path.join(self.test_dir, 'test_dataset'),
            graphs=self.graphs
        )

        # Create data loaders
        batch_size = 2
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset=dataset,
            batch_size=batch_size,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )

        # Check data loaders
        # Note: With only 3 graphs, the split might result in empty loaders
        self.assertIsNotNone(train_loader)

        # Check batch properties if there are enough samples
        if len(train_loader) > 0:
            batch = next(iter(train_loader))
            self.assertIsInstance(batch, Batch)

        # Try with list of graphs
        train_loader2, val_loader2, test_loader2 = create_data_loaders(
            dataset=self.graphs,
            batch_size=batch_size,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )

        self.assertIsNotNone(train_loader2)
        if len(train_loader2) > 0:
            batch = next(iter(train_loader2))
            self.assertIsInstance(batch, Batch)

    def test_graph_dataset_with_more_graphs(self):
        """Test graph dataset with more graphs to ensure proper splitting."""
        # Generate more graphs for better testing
        num_graphs = 20
        graphs = RandomGraphGenerator.generate_dataset(
            num_graphs=num_graphs,
            min_nodes=5,
            max_nodes=10,
            graph_type='erdos_renyi'
        )

        # Create dataset
        dataset = GraphDataset(
            root=os.path.join(self.test_dir, 'large_dataset'),
            graphs=graphs
        )

        # Create data loaders with specific split
        batch_size = 4
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset=dataset,
            batch_size=batch_size,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            shuffle=True
        )

        # Check split sizes (approximate due to rounding)
        expected_train_size = int(0.7 * num_graphs)
        expected_val_size = int(0.15 * num_graphs)
        expected_test_size = num_graphs - expected_train_size - expected_val_size

        # Calculate actual sizes
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        # Check close to expected (might be off by 1 due to rounding)
        self.assertTrue(abs(train_size - expected_train_size) <= 1)
        self.assertTrue(abs(val_size - expected_val_size) <= 1)
        self.assertTrue(abs(test_size - expected_test_size) <= 1)

        # Check batch properties
        batch = next(iter(train_loader))
        self.assertIsInstance(batch, Batch)

        # Check that all samples are accounted for
        self.assertEqual(train_size + val_size + test_size, num_graphs)


if __name__ == "__main__":
    unittest.main()
