"""
Meta-learning components for Amortized Causal Discovery.

This module implements task embeddings and MAML integration for meta-learning
with causal structures. It enables few-shot adaptation to new causal structures
with minimal data.

Sequential Analysis:
- Problem: Enable meta-learning for causal structures with task-specific adaptations
- Components: TaskEmbedding for encoding graph structures, MAML integration
- Approach: Build a TaskEmbedding class that extends GraphStructureRepresentation
- Challenges: Balancing task similarity with adaptation flexibility
- Solution: Graph-based embeddings with causal structure awareness
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

from causal_meta.graph.causal_graph import CausalGraph
from causal_meta.meta_learning.task_representation import GraphStructureRepresentation
from causal_meta.meta_learning.maml import MAML
from torch_geometric.data import Data


class TaskEmbedding(nn.Module):
    """
    Task embedding network for causal graphs.
    
    This class encodes causal graph structures into fixed-size representations
    that can be used for meta-learning and enables similarity computations
    between different causal structures.
    
    Args:
        input_dim: Dimension of node features (default: 1)
        embedding_dim: Dimension of the task embedding (default: 32)
        hidden_dim: Dimension of hidden layers in the GNN (default: 64)
        architecture: GNN architecture to use (default: "gat")
        num_layers: Number of GNN layers (default: 3)
        dropout: Dropout probability (default: 0.1)
        pooling: Pooling method for graph-level embeddings (default: "mean")
        similarity_metric: Method to compute embedding similarity (default: "cosine")
        device: Device to use for computations (default: "cpu")
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        architecture: str = "gat",
        num_layers: int = 3,
        dropout: float = 0.1,
        pooling: str = "mean",
        similarity_metric: str = "cosine",
        device: Union[str, torch.device] = "cpu"
    ):
        """Initialize the TaskEmbedding model."""
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.device = torch.device(device)
        
        # Use GraphStructureRepresentation for core embedding functionality
        self.graph_embedder = GraphStructureRepresentation(
            embedding_dim=embedding_dim,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            architecture=architecture,
            num_layers=num_layers,
            dropout=dropout,
            pooling=pooling,
            similarity_metric=similarity_metric,
            device=device,
            node_feature_type="degree_centrality"  # Use degree centrality as default node feature
        )
        
        # Additional layer to transform embeddings if needed
        self.embedding_transform = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for standard tensor inputs.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Embedding tensor of shape [batch_size, embedding_dim]
        """
        return self.embedding_transform(x)
    
    def encode_graph(self, graph: Union[CausalGraph, Data, Dict[str, Any]]) -> torch.Tensor:
        """
        Encode a causal graph into a fixed-size embedding.
        
        Args:
            graph: Causal graph to encode (CausalGraph, PyTorch Geometric Data, or dict)
            
        Returns:
            Embedding tensor of shape [embedding_dim]
        """
        # Convert CausalGraph to PyTorch Geometric Data if needed
        if isinstance(graph, CausalGraph):
            graph_data = self._convert_causal_graph_to_data(graph)
        else:
            graph_data = graph
            
        # Get the embedding from the underlying graph embedder
        embedding = self.graph_embedder.embed_task(graph_data)
        
        # Apply additional transformation
        embedding = self.embedding_transform(embedding)
        
        return embedding
    
    def compute_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """
        Compute similarity between two task embeddings.
        
        Args:
            embedding1: First task embedding
            embedding2: Second task embedding
            
        Returns:
            Similarity score (higher means more similar)
        """
        return self.graph_embedder.similarity(embedding1, embedding2)
    
    def _convert_causal_graph_to_data(self, causal_graph: CausalGraph) -> Data:
        """
        Convert a CausalGraph object to PyTorch Geometric Data.
        
        Args:
            causal_graph: CausalGraph object to convert
            
        Returns:
            PyTorch Geometric Data object
        """
        # Get nodes and edges
        nodes = list(causal_graph.get_nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Create edge index tensor
        edges = causal_graph.get_edges()
        edge_index = torch.tensor(
            [[node_to_idx[src], node_to_idx[tgt]] for src, tgt in edges],
            dtype=torch.long
        ).t().contiguous()
        
        # If edge index is empty, create a dummy self-loop to avoid errors
        if edge_index.numel() == 0:
            edge_index = torch.zeros((2, 1), dtype=torch.long)
        
        # Create node features based on degree centrality
        num_nodes = len(nodes)
        node_features = torch.zeros((num_nodes, self.input_dim))
        
        # Compute in-degree and out-degree for each node
        for i, node in enumerate(nodes):
            in_degree = len(causal_graph.get_predecessors(node))
            out_degree = len(causal_graph.get_successors(node))
            
            # Use normalized degree as feature
            if self.input_dim == 1:
                # If input_dim is 1, use combined degree
                node_features[i, 0] = (in_degree + out_degree) / (2 * (num_nodes - 1)) if num_nodes > 1 else 0
            elif self.input_dim >= 2:
                # If input_dim is at least 2, use separate in/out degree
                node_features[i, 0] = in_degree / (num_nodes - 1) if num_nodes > 1 else 0
                node_features[i, 1] = out_degree / (num_nodes - 1) if num_nodes > 1 else 0
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=node_features,
            edge_index=edge_index
        )
        
        return data
    
    def batch_encode_graphs(self, graphs: List[CausalGraph]) -> torch.Tensor:
        """
        Encode multiple causal graphs into a batch of embeddings.
        
        Args:
            graphs: List of CausalGraph objects
            
        Returns:
            Batch of embeddings with shape [batch_size, embedding_dim]
        """
        embeddings = []
        for graph in graphs:
            embedding = self.encode_graph(graph)
            embeddings.append(embedding)
        
        return torch.stack(embeddings)
    
    def compute_pairwise_similarities(self, graphs: List[CausalGraph]) -> torch.Tensor:
        """
        Compute pairwise similarity matrix for a list of causal graphs.
        
        Args:
            graphs: List of CausalGraph objects
            
        Returns:
            Similarity matrix with shape [num_graphs, num_graphs]
        """
        embeddings = self.batch_encode_graphs(graphs)
        return self.graph_embedder.compute_pairwise_similarity(embeddings)


class MAMLForCausalDiscovery(nn.Module):
    """
    MAML implementation for few-shot adaptation of causal discovery models.
    
    This class integrates the MAML algorithm with AmortizedCausalDiscovery for
    fast adaptation to new causal structures with minimal data.
    
    Args:
        model: Base model for causal discovery
        task_embedding: TaskEmbedding model for encoding causal structures
        inner_lr: Learning rate for inner loop adaptation
        outer_lr: Learning rate for outer loop optimization
        first_order: Whether to use first-order approximation
        num_inner_steps: Number of inner loop adaptation steps
        device: Device to use for computation
    """
    
    def __init__(
        self,
        model: nn.Module,
        task_embedding: Optional[TaskEmbedding] = None,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        first_order: bool = False,
        num_inner_steps: int = 5,
        device: Union[str, torch.device] = "cpu"
    ):
        """Initialize the MAML model for causal discovery."""
        super().__init__()
        
        self.model = model
        self.device = torch.device(device)
        
        # Create task embedding model if not provided
        if task_embedding is None:
            self.task_embedding = TaskEmbedding(
                embedding_dim=32,
                device=device
            )
        else:
            self.task_embedding = task_embedding
        
        # Initialize MAML algorithm
        self.maml = MAML(
            model=model,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            num_inner_steps=num_inner_steps,
            first_order=first_order,
            device=device
        )
        
    def meta_train(
        self,
        task_batch: List[Dict[str, Any]],
        optimizer: torch.optim.Optimizer,
        num_inner_steps: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Perform meta-training on a batch of tasks.
        
        Args:
            task_batch: List of task dictionaries, each containing:
                - 'graph': Causal graph structure
                - 'support': Support set for adaptation (inputs, targets)
                - 'query': Query set for evaluation (inputs, targets)
            optimizer: Optimizer for the outer loop
            num_inner_steps: Number of inner loop steps (overrides default if provided)
            
        Returns:
            Dictionary with meta-training metrics
        """
        # Format tasks for MAML
        tasks = []
        for task in task_batch:
            support_data = task['support']
            query_data = task['query']
            tasks.append((support_data, query_data))
        
        # Perform MAML outer loop
        metrics = self.maml.outer_loop(
            tasks=tasks,
            grad_clip=1.0,
            optimizer_step=True
        )
        
        return metrics
    
    def adapt(
        self,
        graph: CausalGraph,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        num_steps: Optional[int] = None
    ) -> nn.Module:
        """
        Adapt the model to a new causal graph using a small support set.
        
        Args:
            graph: Causal graph structure
            support_data: Tuple of (inputs, targets) for adaptation
            num_steps: Number of adaptation steps (overrides default if provided)
            
        Returns:
            Adapted model for the given causal graph
        """
        # Compute task embedding for the graph
        task_embedding = self.task_embedding.encode_graph(graph)
        
        # Adapt the model using MAML
        adapted_model = self.maml.adapt(
            batch=support_data,
            num_steps=num_steps
        )
        
        return adapted_model
    
    def forward(
        self,
        x: torch.Tensor,
        graph: Optional[CausalGraph] = None,
        adapted_model: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            graph: Optional causal graph (if adaptation is needed)
            adapted_model: Optional pre-adapted model (if already adapted)
            
        Returns:
            Model predictions
        """
        if adapted_model is not None:
            # Use the pre-adapted model
            return adapted_model(x)
        elif graph is not None:
            # Encode the graph and adapt on the fly (not recommended for efficiency)
            task_embedding = self.task_embedding.encode_graph(graph)
            return self.model(x)
        else:
            # Use the base model without adaptation
            return self.model(x) 