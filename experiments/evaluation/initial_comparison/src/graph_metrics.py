"""
Graph comparison metrics for evaluating learned causal structures.

This module provides functions to compute standard graph comparison metrics
including Structural Hamming Distance (SHD) and F1 score.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def adjacency_from_parents(parent_probs: np.ndarray, 
                          threshold: float = 0.5) -> np.ndarray:
    """
    Convert parent probabilities to adjacency matrix.
    
    Args:
        parent_probs: Matrix of parent probabilities [n_vars, n_vars]
                     where element [i, j] is P(i -> j)
        threshold: Probability threshold for edge inclusion
        
    Returns:
        Binary adjacency matrix
    """
    adj = (parent_probs > threshold).astype(int)
    # Ensure no self-loops
    np.fill_diagonal(adj, 0)
    return adj


def compute_shd(true_adj: np.ndarray, pred_adj: np.ndarray) -> int:
    """
    Compute Structural Hamming Distance between two adjacency matrices.
    
    SHD counts the number of edge additions, deletions, and reversals
    needed to transform the predicted graph into the true graph.
    
    Args:
        true_adj: True adjacency matrix [n_vars, n_vars]
        pred_adj: Predicted adjacency matrix [n_vars, n_vars]
        
    Returns:
        SHD value (lower is better, 0 is perfect)
    """
    # Ensure same shape
    assert true_adj.shape == pred_adj.shape, "Adjacency matrices must have same shape"
    
    # Count differences
    diff = np.abs(true_adj - pred_adj)
    
    # For directed graphs, each wrong edge counts once
    shd = np.sum(diff)
    
    # Check for reversed edges (both count as errors)
    # If true has i->j but pred has j->i, that's 2 errors total
    # But diff already counts both, so no additional computation needed
    
    return int(shd)


def compute_edge_metrics(true_adj: np.ndarray, 
                        pred_adj: np.ndarray) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score for edge prediction.
    
    Args:
        true_adj: True adjacency matrix [n_vars, n_vars]
        pred_adj: Predicted adjacency matrix [n_vars, n_vars]
        
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    # Flatten matrices to compute metrics
    true_flat = true_adj.flatten()
    pred_flat = pred_adj.flatten()
    
    # True positives: edges that exist in both
    tp = np.sum((true_flat == 1) & (pred_flat == 1))
    
    # False positives: edges in predicted but not in true
    fp = np.sum((true_flat == 0) & (pred_flat == 1))
    
    # False negatives: edges in true but not in predicted
    fn = np.sum((true_flat == 1) & (pred_flat == 0))
    
    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }


def compute_orientation_accuracy(true_adj: np.ndarray, 
                                pred_adj: np.ndarray) -> float:
    """
    Compute accuracy of edge orientation for correctly identified edges.
    
    Only considers edges that exist in both graphs and checks if they
    have the correct orientation.
    
    Args:
        true_adj: True adjacency matrix
        pred_adj: Predicted adjacency matrix
        
    Returns:
        Orientation accuracy (0 to 1)
    """
    print(f"DEBUG ORIENTATION: true_adj type: {type(true_adj)}, dtype: {true_adj.dtype}")
    print(f"DEBUG ORIENTATION: pred_adj type: {type(pred_adj)}, dtype: {pred_adj.dtype}")
    print(f"DEBUG ORIENTATION: About to compute true_skeleton = true_adj | true_adj.T")
    
    # Find skeleton (undirected edges) that match
    true_skeleton = true_adj | true_adj.T
    print(f"DEBUG ORIENTATION: true_skeleton computed successfully")
    pred_skeleton = pred_adj | pred_adj.T
    print(f"DEBUG ORIENTATION: pred_skeleton computed successfully")
    
    # Find edges that exist in both skeletons
    common_skeleton = true_skeleton & pred_skeleton
    
    # For each edge in common skeleton, check orientation
    correct_orientations = 0
    total_edges = 0
    
    n_vars = true_adj.shape[0]
    for i in range(n_vars):
        for j in range(i + 1, n_vars):  # Only check upper triangle
            if common_skeleton[i, j]:
                total_edges += 1
                # Check if orientation matches
                if true_adj[i, j] == pred_adj[i, j] and true_adj[j, i] == pred_adj[j, i]:
                    correct_orientations += 1
    
    if total_edges == 0:
        return 1.0  # No common edges to orient
    
    return correct_orientations / total_edges


def evaluate_graph_discovery(true_parents: Dict[str, List[str]],
                            predicted_probs: Dict[str, np.ndarray],
                            variables: List[str],
                            threshold: float = 0.5) -> Dict[str, any]:
    """
    Comprehensive evaluation of graph discovery performance.
    
    Args:
        true_parents: Dictionary mapping each variable to its true parents
        predicted_probs: Dictionary mapping each variable to parent probabilities
        variables: Ordered list of all variables (or frozenset)
        threshold: Probability threshold for edge inclusion
        
    Returns:
        Dictionary with all metrics
    """
    # Ensure variables is a list for indexing
    var_list = list(variables) if not isinstance(variables, list) else variables
    
    n_vars = len(var_list)
    var_to_idx = {var: i for i, var in enumerate(var_list)}
    
    # Build true adjacency matrix (use int dtype for bitwise operations)
    true_adj = np.zeros((n_vars, n_vars), dtype=int)
    print(f"DEBUG GRAPH: Building true adjacency matrix ({n_vars}x{n_vars})")
    print(f"DEBUG GRAPH: true_parents type: {type(true_parents)}")
    print(f"DEBUG GRAPH: Sample true_parents entry: {list(true_parents.items())[0] if true_parents else 'None'}")
    
    for child, parents in true_parents.items():
        print(f"DEBUG GRAPH: Processing child={child}, parents={parents}, parents_type={type(parents)}")
        if child in var_to_idx:
            child_idx = var_to_idx[child]
            for parent in parents:
                if parent in var_to_idx:
                    parent_idx = var_to_idx[parent]
                    true_adj[parent_idx, child_idx] = 1
                    print(f"DEBUG GRAPH: Set edge {parent}({parent_idx}) -> {child}({child_idx})")
    
    print(f"DEBUG GRAPH: True adjacency matrix dtype: {true_adj.dtype}, shape: {true_adj.shape}")
    
    # Build predicted adjacency matrix (use int dtype for bitwise operations)
    pred_adj = np.zeros((n_vars, n_vars), dtype=int)
    print(f"DEBUG GRAPH: Building predicted adjacency matrix")
    print(f"DEBUG GRAPH: predicted_probs type: {type(predicted_probs)}")
    print(f"DEBUG GRAPH: Sample predicted_probs entry: {list(predicted_probs.items())[0] if predicted_probs else 'None'}")
    
    for child, probs in predicted_probs.items():
        print(f"DEBUG GRAPH: Processing child={child}, probs_shape={probs.shape if hasattr(probs, 'shape') else 'no_shape'}, probs_type={type(probs)}")
        if child in var_to_idx:
            child_idx = var_to_idx[child]
            # probs should be a vector of length n_vars
            if len(probs) == n_vars:
                threshold_result = (probs > threshold)
                print(f"DEBUG GRAPH: threshold_result type: {type(threshold_result)}, dtype: {threshold_result.dtype if hasattr(threshold_result, 'dtype') else 'no_dtype'}")
                pred_adj[:, child_idx] = threshold_result.astype(int)
                pred_adj[child_idx, child_idx] = 0  # No self-loops
                print(f"DEBUG GRAPH: Set predicted edges for {child}")
    
    print(f"DEBUG GRAPH: Predicted adjacency matrix dtype: {pred_adj.dtype}, shape: {pred_adj.shape}")
    
    # Compute all metrics
    shd = compute_shd(true_adj, pred_adj)
    edge_metrics = compute_edge_metrics(true_adj, pred_adj)
    orientation_acc = compute_orientation_accuracy(true_adj, pred_adj)
    
    # Additional statistics
    n_true_edges = int(np.sum(true_adj))
    n_pred_edges = int(np.sum(pred_adj))
    
    results = {
        'shd': shd,
        'precision': edge_metrics['precision'],
        'recall': edge_metrics['recall'],
        'f1': edge_metrics['f1'],
        'orientation_accuracy': orientation_acc,
        'n_true_edges': n_true_edges,
        'n_predicted_edges': n_pred_edges,
        'true_positives': edge_metrics['true_positives'],
        'false_positives': edge_metrics['false_positives'],
        'false_negatives': edge_metrics['false_negatives']
    }
    
    logger.debug(f"Graph metrics: SHD={shd}, F1={edge_metrics['f1']:.3f}, "
                f"Precision={edge_metrics['precision']:.3f}, "
                f"Recall={edge_metrics['recall']:.3f}")
    
    return results


def compute_parent_accuracy_per_variable(true_parents: Dict[str, List[str]],
                                        predicted_probs: Dict[str, np.ndarray],
                                        variables: List[str],
                                        threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
    """
    Compute parent prediction accuracy for each variable separately.
    
    Args:
        true_parents: Dictionary mapping each variable to its true parents
        predicted_probs: Dictionary mapping each variable to parent probabilities  
        variables: Ordered list of all variables (or frozenset)
        threshold: Probability threshold for parent inclusion
        
    Returns:
        Dictionary mapping each variable to its parent prediction metrics
    """
    # Ensure variables is a list for indexing
    var_list = list(variables) if not isinstance(variables, list) else variables
    
    var_to_idx = {var: i for i, var in enumerate(var_list)}
    per_var_metrics = {}
    
    for var in var_list:
        if var not in predicted_probs:
            continue
            
        # Get true parents
        true_parent_set = set(true_parents.get(var, []))
        
        # Get predicted parents
        probs = predicted_probs[var]
        predicted_parent_set = set()
        for i, prob in enumerate(probs):
            if prob > threshold and i != var_to_idx[var]:
                predicted_parent_set.add(var_list[i])
        
        # Compute metrics for this variable
        tp = len(true_parent_set & predicted_parent_set)
        fp = len(predicted_parent_set - true_parent_set)
        fn = len(true_parent_set - predicted_parent_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_var_metrics[var] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_true_parents': len(true_parent_set),
            'n_predicted_parents': len(predicted_parent_set)
        }
    
    return per_var_metrics