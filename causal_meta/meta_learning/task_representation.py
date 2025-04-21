"""
Task representation models for embedding tasks into a common latent space.

This module provides classes and utilities for embedding tasks (graphs and their
associated data) into a common latent space that captures task similarity.
These representations can be used for tasks like meta-learning and task similarity
analysis.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

# Import necessary modules from the project
from causal_meta.inference.models.encoder import BaseGNNEncoder
from causal_meta.inference.models.gcn_encoder import GCNEncoder
from causal_meta.inference.models.gat_encoder import GATEncoder
from causal_meta.inference.models.gin_encoder import GINEncoder
from causal_meta.inference.models.graph_utils import preprocess_node_features

# Try to import visualization dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import MDS, TSNE
    HAS_VIZ_DEPS = True
except ImportError:
    HAS_VIZ_DEPS = False


class TaskRepresentation(ABC):
    """
    Abstract base class for task representation models.

    This class defines the interface that all task representation models must
    implement. Task representation models embed tasks (graphs and their associated
    data) into a common latent space that captures task similarity.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        similarity_metric: str = "cosine",
        device: Union[str, torch.device] = "cpu",
        **kwargs
    ):
        """
        Initialize the task representation model.

        Args:
            embedding_dim: Dimension of the task embedding vector
            similarity_metric: Metric to use for computing similarity between embeddings
                               (options: "cosine", "euclidean", "dot")
            device: Device to use for computations (cpu or cuda)
            **kwargs: Additional model-specific parameters
        """
        self.embedding_dim = embedding_dim
        self.similarity_metric = similarity_metric
        self.device = torch.device(device)

        # Validate similarity metric
        valid_metrics = ["cosine", "euclidean", "dot"]
        if similarity_metric not in valid_metrics:
            raise ValueError(
                f"Similarity metric must be one of {valid_metrics}")

    @abstractmethod
    def embed_task(
        self,
        graph: Union[Data, Dict[str, Any]],
        data: Optional[Union[np.ndarray, torch.Tensor, Dict[str, Any]]] = None
    ) -> torch.Tensor:
        """
        Embed a task into the latent space.

        Args:
            graph: Graph structure representing the task (PyTorch Geometric Data object
                  or dictionary with graph information)
            data: Optional data associated with the task (e.g., samples from the causal model,
                  intervention outcomes, etc.)

        Returns:
            Task embedding as a tensor of shape [embedding_dim]
        """
        pass

    def similarity(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor
    ) -> float:
        """
        Compute similarity between two task embeddings.

        Args:
            embedding1: First task embedding
            embedding2: Second task embedding

        Returns:
            Similarity score (higher means more similar)
        """
        # Ensure embeddings are on the same device
        if embedding1.device != embedding2.device:
            embedding2 = embedding2.to(embedding1.device)

        # Compute similarity based on the specified metric
        if self.similarity_metric == "cosine":
            sim = torch.nn.functional.cosine_similarity(
                embedding1.unsqueeze(0),
                embedding2.unsqueeze(0)
            )
        elif self.similarity_metric == "euclidean":
            # For euclidean distance, lower is more similar, so we negate
            sim = -torch.norm(embedding1 - embedding2)
        elif self.similarity_metric == "dot":
            sim = torch.dot(embedding1, embedding2)

        return sim.item()

    def batch_embed_tasks(
        self,
        graphs: List[Union[Data, Dict[str, Any]]],
        data_list: Optional[List[Union[np.ndarray,
                                       torch.Tensor, Dict[str, Any]]]] = None
    ) -> torch.Tensor:
        """
        Embed multiple tasks into the latent space.

        Args:
            graphs: List of graph structures representing the tasks
            data_list: Optional list of data associated with each task
                       (should be the same length as graphs)

        Returns:
            Batch of task embeddings as a tensor of shape [num_tasks, embedding_dim]
        """
        if data_list is not None and len(graphs) != len(data_list):
            raise ValueError("Number of graphs and data instances must match")

        embeddings = []
        for i, graph in enumerate(graphs):
            data = None if data_list is None else data_list[i]
            embedding = self.embed_task(graph, data)
            embeddings.append(embedding)

        return torch.stack(embeddings)

    def compute_pairwise_similarity(
        self,
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise similarity matrix from a batch of embeddings.

        Args:
            embeddings: Tensor of shape [num_tasks, embedding_dim]

        Returns:
            Similarity matrix of shape [num_tasks, num_tasks]
        """
        num_tasks = embeddings.size(0)
        sim_matrix = torch.zeros(
            (num_tasks, num_tasks), device=embeddings.device)

        for i in range(num_tasks):
            for j in range(num_tasks):
                sim_matrix[i, j] = self.similarity(
                    embeddings[i], embeddings[j])

        return sim_matrix


class GraphStructureRepresentation(TaskRepresentation):
    """
    Task representation model based on graph structure.

    This implementation focuses on embedding the structural properties of the causal
    graph (nodes, edges, connectivity patterns) without considering the associated data.
    It uses a Graph Neural Network (GNN) to create fixed-size embeddings of variable-sized
    graphs.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        input_dim: int = 1,
        hidden_dim: int = 32,
        architecture: str = "gcn",
        num_layers: int = 3,
        dropout: float = 0.1,
        pooling: str = "mean",
        similarity_metric: str = "cosine",
        device: Union[str, torch.device] = "cpu",
        node_feature_type: str = "degree_centrality",
        use_edge_attr: bool = False,
        **kwargs
    ):
        """
        Initialize the graph structure-based task representation model.

        Args:
            embedding_dim: Dimension of the task embedding vector
            input_dim: Dimension of input node features (default: 1 for degree centrality)
            hidden_dim: Dimension of hidden layers in the GNN
            architecture: GNN architecture to use ("gcn", "gat", "gin")
            num_layers: Number of GNN layers
            dropout: Dropout probability in the GNN
            pooling: Pooling method for graph-level embeddings ("mean", "max", "sum")
            similarity_metric: Metric for computing similarity between embeddings
            device: Device to use for computations
            node_feature_type: Type of node features to use if not provided
                              ("degree_centrality", "one_hot", "random", "constant")
            use_edge_attr: Whether to use edge attributes if available
            **kwargs: Additional parameters for the GNN
        """
        super().__init__(
            embedding_dim=embedding_dim,
            similarity_metric=similarity_metric,
            device=device
        )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.architecture = architecture
        self.node_feature_type = node_feature_type
        self.use_edge_attr = use_edge_attr

        # Create the GNN encoder based on the specified architecture
        if architecture.lower() == "gcn":
            self.encoder = GCNEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                latent_dim=embedding_dim,
                num_layers=num_layers,
                dropout=dropout,
                pooling=pooling,
                **kwargs
            )
        elif architecture.lower() == "gat":
            self.encoder = GATEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                latent_dim=embedding_dim,
                num_layers=num_layers,
                dropout=dropout,
                pooling=pooling,
                **kwargs
            )
        elif architecture.lower() == "gin":
            self.encoder = GINEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                latent_dim=embedding_dim,
                num_layers=num_layers,
                dropout=dropout,
                pooling=pooling,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported GNN architecture: {architecture}")

        # Move to device
        self.encoder = self.encoder.to(self.device)

    def _prepare_graph(self, graph: Union[Data, Dict[str, Any]]) -> Data:
        """
        Prepare a graph for embedding by ensuring it has proper node features.

        Args:
            graph: Graph structure (PyTorch Geometric Data or dictionary)

        Returns:
            PyTorch Geometric Data object with appropriate node features
        """
        # Convert dict to PyTorch Geometric Data if needed
        if isinstance(graph, dict):
            if 'edge_index' in graph:
                edge_index = graph['edge_index']
                if isinstance(edge_index, np.ndarray):
                    edge_index = torch.from_numpy(edge_index).long()

                # Get optional attributes
                x = graph.get('x', None)
                edge_attr = graph.get(
                    'edge_attr', None) if self.use_edge_attr else None

                num_nodes = graph.get('num_nodes', None)
                if num_nodes is None:
                    if edge_index.numel() > 0:
                        num_nodes = int(edge_index.max()) + 1
                    else:
                        num_nodes = 0

                # Create Data object
                graph = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    num_nodes=num_nodes
                )
            else:
                raise ValueError("Graph dictionary must contain 'edge_index'")

        # Check if node features are available, otherwise create them
        if not hasattr(graph, 'x') or graph.x is None:
            graph.x = self._create_node_features(graph)
        elif graph.x.size(1) != self.input_dim:
            # If feature dimensions don't match, we need to adjust
            # This can happen if the model was initialized with a different input_dim
            # than what's available in the graph
            graph.x = self._transform_node_features(graph.x)

        # Ensure graph is on the correct device
        graph = graph.to(self.device)

        return graph

    def _create_node_features(self, graph: Data) -> torch.Tensor:
        """
        Create node features based on the specified node_feature_type.

        Args:
            graph: PyTorch Geometric Data object

        Returns:
            Node feature tensor of shape [num_nodes, input_dim]
        """
        num_nodes = graph.num_nodes

        if self.node_feature_type == "degree_centrality":
            # Use node degree as a feature
            if graph.edge_index.numel() > 0:
                edge_index = graph.edge_index
                row, col = edge_index
                degree = torch.zeros(
                    num_nodes, dtype=torch.float, device=self.device)
                ones = torch.ones(
                    row.size(0), dtype=torch.float, device=self.device)
                degree = degree.scatter_add(0, row, ones)

                # Normalize degrees
                if degree.max() > 0:
                    degree = degree / degree.max()

                # Expand to input_dim if needed
                if self.input_dim > 1:
                    degree = degree.unsqueeze(1).repeat(1, self.input_dim)
                else:
                    degree = degree.unsqueeze(1)

                return degree
            else:
                # No edges, all nodes have degree 0
                return torch.zeros(num_nodes, self.input_dim, device=self.device)

        elif self.node_feature_type == "one_hot":
            # One-hot encoding of node indices
            # Limit the dimensionality to input_dim
            effective_num_nodes = min(num_nodes, self.input_dim)
            features = torch.zeros(
                num_nodes, self.input_dim, device=self.device)

            # Assign one-hot features up to effective_num_nodes
            for i in range(effective_num_nodes):
                features[i, i] = 1.0

            # For remaining nodes, give them the same feature as node effective_num_nodes-1
            if num_nodes > self.input_dim:
                for i in range(self.input_dim, num_nodes):
                    features[i, self.input_dim-1] = 1.0

            return features

        elif self.node_feature_type == "random":
            # Random node features
            return torch.randn(num_nodes, self.input_dim, device=self.device)

        elif self.node_feature_type == "constant":
            # Constant node features (all ones)
            return torch.ones(num_nodes, self.input_dim, device=self.device)

        else:
            # Default to constant features
            return torch.ones(num_nodes, self.input_dim, device=self.device)

    def _transform_node_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform existing node features to match the required input dimension.

        Args:
            x: Node feature tensor

        Returns:
            Transformed node feature tensor with input_dim columns
        """
        current_dim = x.size(1)
        num_nodes = x.size(0)

        if current_dim > self.input_dim:
            # If current dimension is larger, take first input_dim columns
            return x[:, :self.input_dim]
        else:
            # If current dimension is smaller, pad with zeros
            padded = torch.zeros(num_nodes, self.input_dim,
                                 dtype=x.dtype, device=x.device)
            padded[:, :current_dim] = x
            return padded

    def embed_task(
        self,
        graph: Union[Data, Dict[str, Any]],
        data: Optional[Union[np.ndarray, torch.Tensor, Dict[str, Any]]] = None
    ) -> torch.Tensor:
        """
        Embed a task into the latent space based on its graph structure.

        Args:
            graph: Graph structure representing the task
            data: Optional data associated with the task (not used in this implementation)

        Returns:
            Task embedding as a tensor of shape [embedding_dim]
        """
        # Prepare the graph for embedding
        graph_data = self._prepare_graph(graph)

        # Set encoder to evaluation mode
        self.encoder.eval()

        # Embed the graph
        with torch.no_grad():
            embedding = self.encoder(graph_data)

            # If we get a batch of embeddings (should be of size 1 in this case),
            # take the first one
            if len(embedding.shape) > 1 and embedding.shape[0] > 1:
                embedding = embedding[0]

        return embedding

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the task representation model.

        Returns:
            Dictionary containing the model configuration
        """
        config = {
            "type": self.__class__.__name__,
            "embedding_dim": self.embedding_dim,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "architecture": self.architecture,
            "node_feature_type": self.node_feature_type,
            "use_edge_attr": self.use_edge_attr,
            "similarity_metric": self.similarity_metric,
        }

        # Add encoder config if available
        if hasattr(self.encoder, 'get_config'):
            config["encoder_config"] = self.encoder.get_config()

        return config


class DataStatisticsRepresentation(TaskRepresentation):
    """
    Task representation model based on data statistics.

    This implementation focuses on embedding the statistical properties of the data
    associated with the task (variable distributions, correlations, etc.) without 
    considering the graph structure. It uses statistical methods to create fixed-size
    embeddings from variable data.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        feature_dim: int = 32,
        stats_type: str = "moments",
        use_correlations: bool = True,
        use_higher_moments: bool = True,
        use_neural_embedding: bool = False,
        similarity_metric: str = "cosine",
        device: Union[str, torch.device] = "cpu",
        **kwargs
    ):
        """
        Initialize the data statistics-based task representation model.

        Args:
            embedding_dim: Dimension of the task embedding vector
            feature_dim: Dimension of intermediate feature representations
            stats_type: Type of statistics to use ("moments", "quantiles", "histogram", "kernel")
            use_correlations: Whether to use correlation statistics
            use_higher_moments: Whether to use higher moments (skewness, kurtosis)
            use_neural_embedding: Whether to use a neural network for embedding
            similarity_metric: Metric for computing similarity between embeddings
            device: Device to use for computations
            **kwargs: Additional parameters
        """
        super().__init__(
            embedding_dim=embedding_dim,
            similarity_metric=similarity_metric,
            device=device
        )

        self.feature_dim = feature_dim
        self.stats_type = stats_type
        self.use_correlations = use_correlations
        self.use_higher_moments = use_higher_moments
        self.use_neural_embedding = use_neural_embedding

        # If using neural embedding, create a neural network
        if use_neural_embedding:
            # The input dimension depends on the stats we collect
            input_size = self._calculate_input_size()

            self.neural_embedder = nn.Sequential(
                nn.Linear(input_size, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, embedding_dim)
            ).to(device)
        else:
            # Otherwise, use a simple projection layer
            input_size = self._calculate_input_size()
            self.projector = nn.Linear(input_size, embedding_dim).to(device)

    def _calculate_input_size(self) -> int:
        """
        Calculate the input size for the neural embedder or projector based on the statistics we collect.

        Returns:
            Input size for the neural network
        """
        # Base size for first-order statistics (mean, variance, etc.)
        base_size = 0

        if self.stats_type == "moments":
            # Mean and variance for each variable
            base_size += 2

            # Higher moments if enabled
            if self.use_higher_moments:
                # Skewness and kurtosis
                base_size += 2

        elif self.stats_type == "quantiles":
            # Typically use 5 quantiles (min, 25%, median, 75%, max)
            base_size += 5

        elif self.stats_type == "histogram":
            # Use 10 bins by default
            base_size += 10

        elif self.stats_type == "kernel":
            # Kernel density estimation, simplified representation
            base_size += self.feature_dim

        # Add correlation size if enabled
        # For N variables, there are N*(N-1)/2 pairwise correlations
        # We'll estimate this based on feature_dim for now
        if self.use_correlations:
            # Use a fixed size for correlations to avoid dimensionality issues
            correlation_size = min(self.feature_dim, 20)
            base_size += correlation_size

        return base_size * self.feature_dim

    def _compute_moment_statistics(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute moment statistics (mean, variance, skewness, kurtosis) for the data.

        Args:
            data: Data tensor of shape [num_samples, num_variables]

        Returns:
            Statistics tensor
        """
        # Ensure data is a torch tensor
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float().to(self.device)
        else:
            data = data.float().to(self.device)

        # Check if data has at least 2 dimensions
        if data.dim() == 1:
            data = data.unsqueeze(1)

        # If data is 3D or higher, flatten all but the last dimension
        if data.dim() > 2:
            original_shape = data.shape
            data = data.view(-1, original_shape[-1])

        # Get number of variables
        num_variables = data.shape[1]

        # Initialize statistics tensor
        if self.use_higher_moments:
            # [mean, variance, skewness, kurtosis] for each variable
            stats = torch.zeros(num_variables, 4, device=self.device)
        else:
            # [mean, variance] for each variable
            stats = torch.zeros(num_variables, 2, device=self.device)

        # Compute statistics
        if data.shape[0] > 1:  # At least 2 samples needed for variance
            # Mean and variance
            stats[:, 0] = data.mean(dim=0)
            stats[:, 1] = data.var(dim=0, unbiased=True)

            # Higher moments if enabled
            if self.use_higher_moments and data.shape[0] > 2:
                # Compute centered and standardized data
                centered = data - stats[:, 0].unsqueeze(0)
                std = torch.sqrt(stats[:, 1]).unsqueeze(0)
                std = torch.clamp(std, min=1e-8)  # Avoid division by zero
                standardized = centered / std

                # Skewness: E[(X - μ)³] / σ³
                skewness = torch.mean(standardized**3, dim=0)
                stats[:, 2] = skewness

                # Kurtosis: E[(X - μ)⁴] / σ⁴ - 3
                # Subtracting 3 gives excess kurtosis (normal distribution has kurtosis of 3)
                kurtosis = torch.mean(standardized**4, dim=0) - 3.0
                stats[:, 3] = kurtosis

        # Flatten statistics into a single vector
        stats_flat = stats.flatten()

        # Compute correlations if enabled
        if self.use_correlations and data.shape[0] > 1 and num_variables > 1:
            # Compute correlation matrix
            centered = data - data.mean(dim=0, keepdim=True)
            std = torch.std(data, dim=0, keepdim=True)
            std = torch.clamp(std, min=1e-8)  # Avoid division by zero
            normalized = centered / std
            corr_matrix = torch.mm(
                normalized.t(), normalized) / (data.shape[0] - 1)

            # Extract upper triangular part (excluding diagonal)
            mask = torch.triu(torch.ones_like(corr_matrix), diagonal=1).bool()
            correlations = corr_matrix[mask]

            # If we have too many correlations, downsample
            max_corr = min(self.feature_dim, 20)
            if correlations.shape[0] > max_corr:
                # Take evenly spaced samples or just the first max_corr elements
                indices = torch.linspace(
                    0, correlations.shape[0]-1, max_corr).long()
                correlations = correlations[indices]
            elif correlations.shape[0] < max_corr:
                # Pad with zeros if we have fewer correlations
                padding = torch.zeros(
                    max_corr - correlations.shape[0], device=self.device)
                correlations = torch.cat([correlations, padding])

            # Concatenate with the other statistics
            stats_flat = torch.cat([stats_flat, correlations])

        return stats_flat

    def _compute_quantile_statistics(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute quantile statistics (min, 25%, median, 75%, max) for the data.

        Args:
            data: Data tensor of shape [num_samples, num_variables]

        Returns:
            Statistics tensor
        """
        # Similar implementation to moment statistics but using quantiles
        # Convert to numpy for quantile calculation if needed
        if isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
        else:
            data_np = data

        # Ensure data is 2D
        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)

        # If data is 3D or higher, flatten all but the last dimension
        if data_np.ndim > 2:
            original_shape = data_np.shape
            data_np = data_np.reshape(-1, original_shape[-1])

        # Get number of variables
        num_variables = data_np.shape[1]

        # Compute quantiles for each variable
        quantiles = np.zeros((num_variables, 5))
        for i in range(num_variables):
            if data_np.shape[0] > 0:
                quantiles[i] = np.quantile(
                    data_np[:, i], [0, 0.25, 0.5, 0.75, 1])

        # Convert back to tensor
        quantiles_tensor = torch.from_numpy(quantiles).float().to(self.device)

        # Flatten statistics into a single vector
        stats_flat = quantiles_tensor.flatten()

        # Add correlation if enabled (using moment statistics)
        if self.use_correlations and data_np.shape[0] > 1 and num_variables > 1:
            # Convert to tensor for correlation calculation
            data_tensor = torch.from_numpy(data_np).float().to(self.device)

            # Compute correlation matrix
            centered = data_tensor - data_tensor.mean(dim=0, keepdim=True)
            std = torch.std(data_tensor, dim=0, keepdim=True)
            std = torch.clamp(std, min=1e-8)  # Avoid division by zero
            normalized = centered / std
            corr_matrix = torch.mm(
                normalized.t(), normalized) / (data_tensor.shape[0] - 1)

            # Extract upper triangular part (excluding diagonal)
            mask = torch.triu(torch.ones_like(corr_matrix), diagonal=1).bool()
            correlations = corr_matrix[mask]

            # If we have too many correlations, downsample
            max_corr = min(self.feature_dim, 20)
            if correlations.shape[0] > max_corr:
                indices = torch.linspace(
                    0, correlations.shape[0]-1, max_corr).long()
                correlations = correlations[indices]
            elif correlations.shape[0] < max_corr:
                padding = torch.zeros(
                    max_corr - correlations.shape[0], device=self.device)
                correlations = torch.cat([correlations, padding])

            # Concatenate with the other statistics
            stats_flat = torch.cat([stats_flat, correlations])

        return stats_flat

    def embed_task(
        self,
        graph: Union[Data, Dict[str, Any]],
        data: Optional[Union[np.ndarray, torch.Tensor, Dict[str, Any]]] = None
    ) -> torch.Tensor:
        """
        Embed a task into the latent space based on its associated data.

        Args:
            graph: Graph structure representing the task (not used in this implementation)
            data: Data associated with the task (samples from the causal model)

        Returns:
            Task embedding as a tensor of shape [embedding_dim]
        """
        # Check if data is provided
        if data is None:
            # If no data is provided, return a zero embedding
            return torch.zeros(self.embedding_dim, device=self.device)

        # Handle different data formats
        if isinstance(data, dict):
            # If data is a dictionary, extract the main data array
            if 'samples' in data:
                data_tensor = data['samples']
            elif 'data' in data:
                data_tensor = data['data']
            elif 'X' in data:
                data_tensor = data['X']
            else:
                # Try to find a key with array-like values
                for key, value in data.items():
                    if isinstance(value, (np.ndarray, torch.Tensor)) and (
                        isinstance(value, np.ndarray) and value.ndim > 0 or
                        isinstance(value, torch.Tensor) and value.dim() > 0
                    ):
                        data_tensor = value
                        break
                else:
                    # No suitable data found
                    return torch.zeros(self.embedding_dim, device=self.device)
        else:
            # Directly use the provided data
            data_tensor = data

        # Compute statistics based on the specified type
        if self.stats_type == "moments":
            stats = self._compute_moment_statistics(data_tensor)
        elif self.stats_type == "quantiles":
            stats = self._compute_quantile_statistics(data_tensor)
        else:
            # Default to moment statistics
            stats = self._compute_moment_statistics(data_tensor)

        # Ensure stats has the right shape for the projector or neural embedder
        expected_size = self._calculate_input_size()
        if stats.shape[0] < expected_size:
            # Pad with zeros if we have fewer statistics
            padding = torch.zeros(
                expected_size - stats.shape[0], device=self.device)
            stats = torch.cat([stats, padding])
        elif stats.shape[0] > expected_size:
            # Truncate if we have more statistics
            stats = stats[:expected_size]

        # Project stats to embedding space
        with torch.no_grad():
            if self.use_neural_embedding:
                embedding = self.neural_embedder(stats)
            else:
                embedding = self.projector(stats)

        return embedding

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the task representation model.

        Returns:
            Dictionary containing the model configuration
        """
        return {
            "type": self.__class__.__name__,
            "embedding_dim": self.embedding_dim,
            "feature_dim": self.feature_dim,
            "stats_type": self.stats_type,
            "use_correlations": self.use_correlations,
            "use_higher_moments": self.use_higher_moments,
            "use_neural_embedding": self.use_neural_embedding,
            "similarity_metric": self.similarity_metric
        }
