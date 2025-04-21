"""
Data loading utilities for graph datasets.

This module provides functions and classes for loading and preprocessing graph datasets
for training GNN-based models.
"""

import torch
import numpy as np
import os
import json
from typing import Dict, List, Optional, Union, Tuple, Callable, Any

import torch_geometric
from torch_geometric.data import Data, Batch, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform


class GraphDataset(InMemoryDataset):
    """
    Dataset class for loading and preprocessing graph data.

    This class handles loading graphs from various sources and formats,
    applying transformations, and generating batches for training.
    """

    def __init__(
        self,
        root: str,
        graphs: Optional[List[Data]] = None,
        transform: Optional[BaseTransform] = None,
        pre_transform: Optional[BaseTransform] = None,
        pre_filter: Optional[Callable] = None
    ):
        """
        Initialize the graph dataset.

        Args:
            root: Root directory where the dataset should be saved
            graphs: Optional list of graph data objects to use directly
            transform: PyTorch Geometric transform to apply at runtime
            pre_transform: PyTorch Geometric transform to apply once during preprocessing
            pre_filter: Filter to apply to graphs during preprocessing
        """
        self.graphs = graphs
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """
        Get the names of raw data files.

        Returns:
            List of raw file names
        """
        return ['graph_data.json'] if self.graphs is None else []

    @property
    def processed_file_names(self) -> List[str]:
        """
        Get the names of processed data files.

        Returns:
            List of processed file names
        """
        return ['processed_data.pt']

    def download(self):
        """
        Download the dataset if it doesn't exist.

        This method is required by the PyTorch Geometric Dataset base class,
        but in this case, we expect data to be provided.
        """
        # No download performed - data is assumed to be provided
        pass

    def process(self):
        """
        Process the raw data into a format suitable for training.

        This method converts raw graph data into PyTorch Geometric Data objects.
        """
        # If graphs are provided directly, use them
        if self.graphs is not None:
            data_list = self.graphs
        else:
            # Otherwise, load from raw files
            data_list = []

            if os.path.exists(self.raw_paths[0]):
                with open(self.raw_paths[0], 'r') as f:
                    raw_data = json.load(f)

                # Process each graph in the raw data
                for graph_data in raw_data:
                    # Extract graph components
                    x = torch.tensor(graph_data.get(
                        'node_features', []), dtype=torch.float)
                    edge_index = torch.tensor(graph_data.get(
                        'edge_index', []), dtype=torch.long)

                    # Create graph data object
                    data = Data(x=x, edge_index=edge_index)

                    # Add any additional attributes
                    for key, value in graph_data.items():
                        if key not in ['node_features', 'edge_index']:
                            data[key] = torch.tensor(value)

                    data_list.append(data)

        # Apply preprocessing transforms if provided
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class RandomGraphGenerator:
    """
    Generator for synthetic random graph datasets.

    This class provides methods to generate various types of random graph structures
    for testing and training GNN models.
    """

    @staticmethod
    def create_erdos_renyi_graph(
        num_nodes: int,
        edge_prob: float = 0.2,
        node_features_dim: int = 3
    ) -> Data:
        """
        Create a random Erdos-Renyi graph.

        Args:
            num_nodes: Number of nodes in the graph
            edge_prob: Probability of an edge between any two nodes
            node_features_dim: Dimension of node features

        Returns:
            PyTorch Geometric Data object representing the graph
        """
        # Create random node features
        x = torch.randn(num_nodes, node_features_dim)

        # Create edges with probability edge_prob
        edge_list = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and torch.rand(1).item() < edge_prob:
                    edge_list.append([i, j])

        # Convert edge list to edge index
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        else:
            # Ensure at least one edge to avoid empty graphs
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()

        # Create graph data object
        graph = Data(x=x, edge_index=edge_index)

        return graph

    @staticmethod
    def create_barabasi_albert_graph(
        num_nodes: int,
        num_edges: int = 2,
        node_features_dim: int = 3
    ) -> Data:
        """
        Create a random Barabasi-Albert graph (scale-free network).

        Args:
            num_nodes: Number of nodes in the graph
            num_edges: Number of edges to attach from a new node to existing nodes
            node_features_dim: Dimension of node features

        Returns:
            PyTorch Geometric Data object representing the graph
        """
        # Create random node features
        x = torch.randn(num_nodes, node_features_dim)

        # Initialize with a complete graph on num_edges nodes
        edge_list = []
        for i in range(min(num_edges, num_nodes)):
            for j in range(i + 1, min(num_edges, num_nodes)):
                edge_list.append([i, j])
                edge_list.append([j, i])  # Add both directions

        # Add remaining nodes using preferential attachment
        degrees = {i: 0 for i in range(num_nodes)}
        for edge in edge_list:
            degrees[edge[0]] += 1
            degrees[edge[1]] += 1

        for i in range(num_edges, num_nodes):
            # Select nodes to connect to based on their degree
            targets = []
            for _ in range(num_edges):
                # Simple preferential attachment approximation
                total_degree = sum(degrees[j] for j in range(i))
                if total_degree == 0:
                    # If all nodes have degree 0, choose uniformly
                    targets.append(torch.randint(0, i, (1,)).item())
                else:
                    # Choose based on degree
                    r = torch.rand(1).item() * total_degree
                    cumsum = 0
                    for j in range(i):
                        cumsum += degrees[j]
                        if cumsum >= r:
                            targets.append(j)
                            break

            # Add edges to target nodes
            for target in targets:
                edge_list.append([i, target])
                edge_list.append([target, i])  # Add both directions
                degrees[i] += 1
                degrees[target] += 1

        # Convert edge list to edge index
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()

        # Create graph data object
        graph = Data(x=x, edge_index=edge_index)

        return graph

    @staticmethod
    def create_watts_strogatz_graph(
        num_nodes: int,
        k: int = 4,
        p: float = 0.1,
        node_features_dim: int = 3
    ) -> Data:
        """
        Create a random Watts-Strogatz small world graph.

        Args:
            num_nodes: Number of nodes in the graph
            k: Each node is connected to k nearest neighbors in ring topology
            p: Probability of rewiring each edge
            node_features_dim: Dimension of node features

        Returns:
            PyTorch Geometric Data object representing the graph
        """
        # Create random node features
        x = torch.randn(num_nodes, node_features_dim)

        # Create ring lattice
        edge_list = []
        for i in range(num_nodes):
            for j in range(1, k // 2 + 1):
                # Connect to k/2 neighbors on each side
                edge_list.append([i, (i + j) % num_nodes])
                edge_list.append([i, (i - j) % num_nodes])

        # Rewire edges with probability p
        for i in range(len(edge_list)):
            if torch.rand(1).item() < p:
                # Keep source node
                source = edge_list[i][0]

                # Choose a random target node
                potential_targets = list(range(num_nodes))
                potential_targets.remove(source)

                # Remove existing connections to avoid multi-edges
                existing_targets = [edge[1]
                                    for edge in edge_list if edge[0] == source]
                for target in existing_targets:
                    if target in potential_targets:
                        potential_targets.remove(target)

                if potential_targets:
                    new_target = potential_targets[torch.randint(
                        0, len(potential_targets), (1,)).item()]
                    edge_list[i][1] = new_target

        # Convert edge list to edge index (ensuring directed edges)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()

        # Create graph data object
        graph = Data(x=x, edge_index=edge_index)

        return graph

    @staticmethod
    def generate_dataset(
        num_graphs: int,
        min_nodes: int = 5,
        max_nodes: int = 20,
        graph_type: str = 'erdos_renyi',
        **kwargs
    ) -> List[Data]:
        """
        Generate a dataset of random graphs.

        Args:
            num_graphs: Number of graphs to generate
            min_nodes: Minimum number of nodes per graph
            max_nodes: Maximum number of nodes per graph
            graph_type: Type of graph to generate ('erdos_renyi', 'barabasi_albert', 'watts_strogatz')
            **kwargs: Additional parameters for the specific graph generator

        Returns:
            List of PyTorch Geometric Data objects
        """
        graphs = []

        for _ in range(num_graphs):
            # Random number of nodes for this graph
            num_nodes = torch.randint(min_nodes, max_nodes + 1, (1,)).item()

            # Generate graph based on type
            if graph_type == 'erdos_renyi':
                graph = RandomGraphGenerator.create_erdos_renyi_graph(
                    num_nodes, **kwargs)
            elif graph_type == 'barabasi_albert':
                graph = RandomGraphGenerator.create_barabasi_albert_graph(
                    num_nodes, **kwargs)
            elif graph_type == 'watts_strogatz':
                graph = RandomGraphGenerator.create_watts_strogatz_graph(
                    num_nodes, **kwargs)
            else:
                raise ValueError(f"Unsupported graph type: {graph_type}")

            graphs.append(graph)

        return graphs


def create_data_loaders(
    dataset: Union[Dataset, List[Data]],
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.

    Args:
        dataset: Graph dataset or list of graph data objects
        batch_size: Batch size for data loaders
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        shuffle: Whether to shuffle the data
        num_workers: Number of subprocesses for data loading

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Convert list of Data objects to a dataset if needed
    if isinstance(dataset, list):
        dataset = GraphDataset(
            root='temp_dataset',
            graphs=dataset
        )

    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Split dataset
    if shuffle:
        indices = torch.randperm(total_size)
    else:
        indices = torch.arange(total_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
