import torch
import pytest
import numpy as np
import networkx as nx

from causal_meta.meta_learning.graph_inference_utils import (
    threshold_graph, compute_shd, compute_precision_recall,
    sample_from_posterior, ensure_dag, GraphMetrics
)
from causal_meta.graph.causal_graph import CausalGraph


class TestGraphInferenceUtils:
    @pytest.fixture
    def sample_matrices(self):
        # Create true adjacency matrix
        n_variables = 5
        true_adj = np.zeros((n_variables, n_variables))
        
        # Add edges 0->1, 0->2, 1->3, 2->3, 2->4
        true_adj[0, 1] = 1
        true_adj[0, 2] = 1
        true_adj[1, 3] = 1
        true_adj[2, 3] = 1
        true_adj[2, 4] = 1
        
        # Create predicted probabilities matrix with some errors
        pred_probs = np.zeros((n_variables, n_variables))
        # Correct edges with high probability
        pred_probs[0, 1] = 0.9  # True positive
        pred_probs[0, 2] = 0.8  # True positive
        pred_probs[2, 4] = 0.7  # True positive
        
        # Incorrect edges with low to medium probability
        pred_probs[3, 1] = 0.6  # False positive (wrong direction)
        pred_probs[1, 2] = 0.4  # False positive (doesn't exist)
        
        # Missing edge with low probability
        pred_probs[1, 3] = 0.3  # False negative (should be higher)
        pred_probs[2, 3] = 0.2  # False negative (should be higher)
        
        # Create CausalGraph from true adjacency matrix
        true_graph = CausalGraph()
        for i in range(n_variables):
            true_graph.add_node(i)
        
        # Add edges
        for i in range(n_variables):
            for j in range(n_variables):
                if true_adj[i, j] == 1:
                    true_graph.add_edge(i, j)
        
        return {
            'true_adj': torch.tensor(true_adj, dtype=torch.float32),
            'pred_probs': torch.tensor(pred_probs, dtype=torch.float32),
            'true_graph': true_graph,
            'n_variables': n_variables
        }
    
    def test_threshold_graph(self, sample_matrices):
        """Test the threshold_graph function with different thresholds."""
        pred_probs = sample_matrices['pred_probs']
        
        # Test with different thresholds
        high_threshold = threshold_graph(pred_probs, threshold=0.7)
        medium_threshold = threshold_graph(pred_probs, threshold=0.5)
        low_threshold = threshold_graph(pred_probs, threshold=0.3)
        
        # Check output shapes
        assert high_threshold.shape == pred_probs.shape
        assert medium_threshold.shape == pred_probs.shape
        assert low_threshold.shape == pred_probs.shape
        
        # Check binary values
        assert torch.all((high_threshold == 0) | (high_threshold == 1))
        assert torch.all((medium_threshold == 0) | (medium_threshold == 1))
        assert torch.all((low_threshold == 0) | (low_threshold == 1))
        
        # Check expected number of edges for each threshold
        assert high_threshold.sum() == 3  # Only the highest probabilities (0.7+)
        assert medium_threshold.sum() == 4  # Medium probabilities (0.5+)
        assert low_threshold.sum() == 7  # Low probabilities (0.3+)
    
    def test_compute_shd(self, sample_matrices):
        """Test the Structural Hamming Distance computation."""
        true_adj = sample_matrices['true_adj']
        pred_probs = sample_matrices['pred_probs']
        
        # Threshold with different values
        high_threshold = threshold_graph(pred_probs, threshold=0.7)
        medium_threshold = threshold_graph(pred_probs, threshold=0.5)
        low_threshold = threshold_graph(pred_probs, threshold=0.3)
        
        # Compute SHD
        shd_high = compute_shd(true_adj, high_threshold)
        shd_medium = compute_shd(true_adj, medium_threshold)
        shd_low = compute_shd(true_adj, low_threshold)
        
        # Check SHD values (compare with expected values)
        # High threshold: 2 missing edges (false negatives)
        assert shd_high == 2
        
        # Medium threshold: 1 wrong edge + 1 missing edge
        assert shd_medium == 2
        
        # Low threshold: 3 wrong edges + 0 missing edges
        assert shd_low == 3
    
    def test_compute_precision_recall(self, sample_matrices):
        """Test precision and recall computation."""
        true_adj = sample_matrices['true_adj']
        pred_probs = sample_matrices['pred_probs']
        
        # Threshold with different values
        high_threshold = threshold_graph(pred_probs, threshold=0.7)
        medium_threshold = threshold_graph(pred_probs, threshold=0.5)
        low_threshold = threshold_graph(pred_probs, threshold=0.3)
        
        # Compute precision and recall
        precision_high, recall_high = compute_precision_recall(true_adj, high_threshold)
        precision_medium, recall_medium = compute_precision_recall(true_adj, medium_threshold)
        precision_low, recall_low = compute_precision_recall(true_adj, low_threshold)
        
        # Check precision values
        # High threshold: 3 correct / 3 predicted
        assert precision_high == 1.0
        
        # Medium threshold: 3 correct / 4 predicted
        assert precision_medium == 0.75
        
        # Low threshold: 3 correct / 7 predicted
        assert precision_low == 3/7
        
        # Check recall values
        # High threshold: 3 correct / 5 total true edges
        assert recall_high == 0.6
        
        # Medium threshold: 3 correct / 5 total true edges
        assert recall_medium == 0.6
        
        # Low threshold: 3 correct / 5 total true edges
        assert recall_low == 0.6
    
    def test_sample_from_posterior(self, sample_matrices):
        """Test sampling from the posterior distribution."""
        pred_probs = sample_matrices['pred_probs']
        n_variables = sample_matrices['n_variables']
        n_samples = 10
        
        # Sample from posterior
        samples = sample_from_posterior(pred_probs, n_samples=n_samples)
        
        # Check output shape
        assert samples.shape == (n_samples, n_variables, n_variables)
        
        # Check binary values
        assert torch.all((samples == 0) | (samples == 1))
        
        # Check that different samples are generated
        sample_differences = 0
        for i in range(n_samples - 1):
            if not torch.all(samples[i] == samples[i+1]):
                sample_differences += 1
        
        # At least some samples should be different (might rarely fail due to randomness)
        assert sample_differences > 0
    
    def test_ensure_dag(self, sample_matrices):
        """Test the ensure_dag function for DAG constraints."""
        pred_probs = sample_matrices['pred_probs']
        
        # Create a graph with a cycle
        cyclic_graph = torch.zeros_like(pred_probs)
        # Add cycle 0->1->2->0
        cyclic_graph[0, 1] = 1
        cyclic_graph[1, 2] = 1
        cyclic_graph[2, 0] = 1
        
        # Apply DAG enforcement
        dag_graph = ensure_dag(cyclic_graph)
        
        # Check output shape
        assert dag_graph.shape == cyclic_graph.shape
        
        # Check binary values
        assert torch.all((dag_graph == 0) | (dag_graph == 1))
        
        # Create networkx graph to check acyclicity
        nx_graph = nx.DiGraph()
        for i in range(dag_graph.shape[0]):
            nx_graph.add_node(i)
        
        for i in range(dag_graph.shape[0]):
            for j in range(dag_graph.shape[1]):
                if dag_graph[i, j] == 1:
                    nx_graph.add_edge(i, j)
        
        # Ensure the result is acyclic
        assert nx.is_directed_acyclic_graph(nx_graph)
    
    def test_graph_metrics_class(self, sample_matrices):
        """Test the GraphMetrics class for computing multiple metrics."""
        true_adj = sample_matrices['true_adj']
        pred_probs = sample_matrices['pred_probs']
        
        # Create GraphMetrics instance
        metrics = GraphMetrics(true_adj)
        
        # Compute metrics with default threshold
        result = metrics.compute_all_metrics(pred_probs)
        
        # Check that all expected metrics are included
        assert 'shd' in result
        assert 'precision' in result
        assert 'recall' in result
        assert 'f1' in result
        
        # Check that F1 score is correctly computed from precision and recall
        precision = result['precision']
        recall = result['recall']
        expected_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        assert abs(result['f1'] - expected_f1) < 1e-6 