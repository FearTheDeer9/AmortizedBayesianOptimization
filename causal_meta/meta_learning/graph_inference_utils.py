import torch
import numpy as np
import networkx as nx
from typing import Tuple, Dict, Union, List, Optional

from causal_meta.graph.causal_graph import CausalGraph


def threshold_graph(edge_probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Convert a matrix of edge probabilities to a binary adjacency matrix using a threshold.
    
    Args:
        edge_probs: Tensor of shape [n_variables, n_variables] with edge probabilities
        threshold: Probability threshold for including an edge
        
    Returns:
        Binary adjacency matrix tensor of the same shape
    """
    # Apply threshold
    adj_matrix = (edge_probs > threshold).float()
    
    # Ensure no self-loops (diagonal elements should be 0)
    adj_matrix.fill_diagonal_(0)
    
    return adj_matrix


def sample_from_posterior(edge_probs: torch.Tensor, n_samples: int = 10, 
                         ensure_acyclic: bool = True) -> torch.Tensor:
    """
    Generate samples from the posterior distribution over graphs.
    
    Args:
        edge_probs: Tensor of shape [n_variables, n_variables] with edge probabilities
        n_samples: Number of graph samples to generate
        ensure_acyclic: Whether to enforce acyclicity in the sampled graphs
        
    Returns:
        Tensor of shape [n_samples, n_variables, n_variables] containing sampled adjacency matrices
    """
    n_variables = edge_probs.shape[0]
    device = edge_probs.device
    
    # Initialize samples tensor
    samples = torch.zeros((n_samples, n_variables, n_variables), device=device)
    
    # Generate samples
    for i in range(n_samples):
        # Sample edges according to their probabilities
        random_values = torch.rand((n_variables, n_variables), device=device)
        sample = (random_values < edge_probs).float()
        
        # Ensure no self-loops
        sample.fill_diagonal_(0)
        
        # Enforce acyclicity if requested
        if ensure_acyclic:
            sample = ensure_dag(sample)
        
        samples[i] = sample
    
    return samples


def ensure_dag(adj_matrix: torch.Tensor) -> torch.Tensor:
    """
    Enforce the directed acyclic graph (DAG) constraint on an adjacency matrix.
    
    Removes edges to break cycles while minimizing changes to the original matrix.
    
    Args:
        adj_matrix: Binary adjacency matrix tensor
        
    Returns:
        Modified adjacency matrix that represents a DAG
    """
    # Convert to numpy for easier handling with networkx
    if isinstance(adj_matrix, torch.Tensor):
        device = adj_matrix.device
        adj_numpy = adj_matrix.cpu().numpy()
    else:
        device = None
        adj_numpy = adj_matrix
    
    # Create networkx DiGraph
    G = nx.DiGraph()
    n_variables = adj_numpy.shape[0]
    
    # Add nodes
    for i in range(n_variables):
        G.add_node(i)
    
    # Create edge list ordered by weight (assuming binary weights here)
    edges = []
    for i in range(n_variables):
        for j in range(n_variables):
            if adj_numpy[i, j] > 0:
                edges.append((i, j))
    
    # Add edges one by one, checking for cycles
    dag_adj = np.zeros_like(adj_numpy)
    for source, target in edges:
        # Temporarily add the edge
        G.add_edge(source, target)
        
        # Check if the graph is still acyclic
        if nx.is_directed_acyclic_graph(G):
            # Keep the edge
            dag_adj[source, target] = 1
        else:
            # Remove the edge to maintain acyclicity
            G.remove_edge(source, target)
    
    # Convert back to torch if needed
    if device is not None:
        return torch.tensor(dag_adj, device=device, dtype=torch.float32)
    else:
        return dag_adj


def compute_shd(true_adj: torch.Tensor, pred_adj: torch.Tensor) -> int:
    """
    Compute the Structural Hamming Distance (SHD) between two graphs.
    
    SHD counts the number of edge additions, deletions, and reversals needed
    to transform the predicted graph into the true graph.
    
    Args:
        true_adj: Ground truth adjacency matrix
        pred_adj: Predicted adjacency matrix
        
    Returns:
        SHD value (lower is better, 0 is perfect)
    """
    if isinstance(true_adj, torch.Tensor):
        true_adj = true_adj.cpu().numpy()
    if isinstance(pred_adj, torch.Tensor):
        pred_adj = pred_adj.cpu().numpy()
    
    # Convert to binary matrices
    true_adj = (true_adj > 0).astype(np.int32)
    pred_adj = (pred_adj > 0).astype(np.int32)
    
    # Count edges in different categories
    missing = np.logical_and(true_adj == 1, pred_adj == 0).sum()  # False negatives
    extra = np.logical_and(true_adj == 0, pred_adj == 1).sum()  # False positives
    reversed_edges = 0
    
    # Count reversed edges (need to check element by element)
    for i in range(true_adj.shape[0]):
        for j in range(true_adj.shape[1]):
            if true_adj[i, j] == 1 and pred_adj[j, i] == 1:
                # Edge is reversed in prediction
                reversed_edges += 1
                # Adjust missing and extra counts to avoid double counting
                missing -= 1
                extra -= 1
    
    # Total SHD is the sum of all differences
    shd = missing + extra + reversed_edges
    
    return int(shd)


def compute_precision_recall(true_adj: torch.Tensor, pred_adj: torch.Tensor) -> Tuple[float, float]:
    """
    Compute precision and recall metrics for graph edge prediction.
    
    Args:
        true_adj: Ground truth adjacency matrix
        pred_adj: Predicted adjacency matrix
        
    Returns:
        Tuple of (precision, recall)
    """
    if isinstance(true_adj, torch.Tensor):
        true_adj = true_adj.cpu().numpy()
    if isinstance(pred_adj, torch.Tensor):
        pred_adj = pred_adj.cpu().numpy()
    
    # Convert to binary matrices
    true_adj = (true_adj > 0).astype(np.int32)
    pred_adj = (pred_adj > 0).astype(np.int32)
    
    # True positives: edges present in both graphs
    tp = np.logical_and(true_adj == 1, pred_adj == 1).sum()
    
    # Total predicted positive edges
    total_pred_edges = pred_adj.sum()
    
    # Total true edges
    total_true_edges = true_adj.sum()
    
    # Calculate precision: TP / (TP + FP)
    precision = float(tp) / float(total_pred_edges) if total_pred_edges > 0 else 1.0
    
    # Calculate recall: TP / (TP + FN)
    recall = float(tp) / float(total_true_edges) if total_true_edges > 0 else 1.0
    
    return precision, recall


class GraphMetrics:
    """
    Class for computing and tracking various graph recovery metrics.
    """
    
    def __init__(self, true_adj: Union[torch.Tensor, np.ndarray, CausalGraph]):
        """
        Initialize with a ground truth adjacency matrix or CausalGraph.
        
        Args:
            true_adj: Ground truth adjacency matrix or CausalGraph
        """
        # Convert CausalGraph to adjacency matrix if needed
        if isinstance(true_adj, CausalGraph):
            self.true_adj = torch.tensor(true_adj.get_adjacency_matrix())
        else:
            # Make a copy to avoid modifying the original
            self.true_adj = true_adj.clone() if isinstance(true_adj, torch.Tensor) else torch.tensor(true_adj)
    
    def compute_all_metrics(self, pred_probs: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute all available graph recovery metrics.
        
        Args:
            pred_probs: Predicted edge probabilities
            threshold: Threshold for converting probabilities to adjacency matrix
            
        Returns:
            Dictionary of metrics
        """
        # Threshold probabilities to get adjacency matrix
        pred_adj = threshold_graph(pred_probs, threshold)
        
        # Compute SHD
        shd = compute_shd(self.true_adj, pred_adj)
        
        # Compute precision and recall
        precision, recall = compute_precision_recall(self.true_adj, pred_adj)
        
        # Compute F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Return all metrics
        return {
            'shd': shd,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def to_causal_graph(self, pred_probs: torch.Tensor, threshold: float = 0.5) -> CausalGraph:
        """
        Convert edge probabilities to a CausalGraph object.
        
        Args:
            pred_probs: Predicted edge probabilities
            threshold: Threshold for including edges
            
        Returns:
            CausalGraph instance
        """
        # Threshold to get adjacency matrix
        adj_matrix = threshold_graph(pred_probs, threshold)
        
        # Ensure the result is a DAG
        dag_adj = ensure_dag(adj_matrix)
        
        # Convert to networkx DiGraph
        if isinstance(dag_adj, torch.Tensor):
            dag_adj = dag_adj.cpu().numpy()
        
        # Create CausalGraph
        causal_graph = CausalGraph()
        
        # Add nodes
        for i in range(dag_adj.shape[0]):
            causal_graph.add_node(i)
        
        # Add edges
        for i in range(dag_adj.shape[0]):
            for j in range(dag_adj.shape[1]):
                if dag_adj[i, j] > 0:
                    causal_graph.add_edge(i, j)
        
        return causal_graph 