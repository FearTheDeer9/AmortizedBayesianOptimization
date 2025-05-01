#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shared utility functions for demo scripts.

This module provides shared functionality for the demo scripts, including
error handling, path management, tensor shape standardization, and fallback
implementations when needed.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from causal_meta package with error handling
try:
    from causal_meta.graph.causal_graph import CausalGraph
    from causal_meta.graph.generators.factory import GraphFactory
    from causal_meta.environments.scm import StructuralCausalModel
    from causal_meta.graph.visualization import plot_graph, plot_causal_graph
    from causal_meta.meta_learning.acd_models import GraphEncoder
    from causal_meta.meta_learning.dynamics_decoder import DynamicsDecoder
    from causal_meta.meta_learning.amortized_causal_discovery import AmortizedCausalDiscovery
    from causal_meta.meta_learning.meta_learning import TaskEmbedding, MAMLForCausalDiscovery
    from causal_meta.meta_learning.amortized_cbo import AmortizedCBO
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    print("Some functionality may be limited. Using fallbacks where necessary.")

# Path management
def ensure_dir(directory: str) -> str:
    """Ensure a directory exists and return its path."""
    os.makedirs(directory, exist_ok=True)
    return directory

def get_assets_dir() -> str:
    """Get the assets directory path."""
    assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
    return ensure_dir(assets_dir)

def get_checkpoints_dir() -> str:
    """Get the model checkpoints directory path."""
    checkpoints_dir = os.path.join(os.path.dirname(__file__), '..', 'example_checkpoints')
    return ensure_dir(checkpoints_dir)

# Tensor handling
def standardize_tensor_shape(data: Union[np.ndarray, torch.Tensor], for_encoder: bool = True) -> torch.Tensor:
    """
    Standardize tensor shape for neural network processing.
    
    For GraphEncoder: [batch_size, seq_length, num_nodes]
    For DynamicsDecoder: [batch_size * num_nodes, feature_dim]
    
    Args:
        data: Input data as numpy array or torch tensor
        for_encoder: Whether the data is for GraphEncoder (True) or DynamicsDecoder (False)
        
    Returns:
        Properly shaped torch tensor
    """
    # Convert to torch tensor if numpy array
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
    
    # Handle different input shapes
    if for_encoder:
        # For GraphEncoder - shape should be [batch_size, seq_length, num_nodes]
        if len(data.shape) == 1:
            # 1D vector: [samples] -> [1, samples, 1]
            return data.unsqueeze(0).unsqueeze(2)
        elif len(data.shape) == 2:
            # 2D matrix: [samples, nodes] -> [1, samples, nodes]
            return data.unsqueeze(0)
        elif len(data.shape) == 3:
            # Already in correct shape
            return data
        else:
            raise ValueError(f"Unsupported data shape for encoder: {data.shape}")
    else:
        # For DynamicsDecoder - shape should be [batch_size * num_nodes, feature_dim]
        if len(data.shape) == 1:
            # 1D vector: [samples] -> [samples, 1]
            return data.unsqueeze(1)
        elif len(data.shape) == 2:
            # Already 2D, assume correct shape
            return data
        elif len(data.shape) == 3:
            # 3D tensor: [batch, seq, nodes] -> [batch * nodes, seq]
            batch_size, seq_len, num_nodes = data.shape
            return data.reshape(batch_size * num_nodes, seq_len)
        else:
            raise ValueError(f"Unsupported data shape for decoder: {data.shape}")

# Node naming utilities
def get_node_name(node_id: Union[int, str]) -> str:
    """Get standardized node name from node ID."""
    if isinstance(node_id, str) and node_id.startswith('X_'):
        return node_id
    return f"X_{node_id}"

def get_node_id(node_name: Union[int, str]) -> int:
    """Get node ID from node name."""
    if isinstance(node_name, int):
        return node_name
    elif isinstance(node_name, str) and node_name.startswith('X_'):
        return int(node_name.split('_')[1])
    else:
        raise ValueError(f"Invalid node name format: {node_name}")

def format_interventions(interventions: Dict[Union[int, str], float]) -> Dict[str, float]:
    """Format interventions to use standardized node names."""
    return {get_node_name(node): value for node, value in interventions.items()}

# Fallback implementations - only used when actual implementations are unavailable
class DummyGraph:
    """Minimal implementation of a graph structure when CausalGraph is unavailable."""
    
    def __init__(self, adj_matrix: np.ndarray):
        """Initialize with adjacency matrix."""
        self.adj_matrix = adj_matrix
        self.num_nodes = adj_matrix.shape[0]
        self._node_names = [f"X_{i}" for i in range(self.num_nodes)]
    
    def get_num_nodes(self) -> int:
        """Get number of nodes."""
        return self.num_nodes
    
    def has_edge(self, i: int, j: int) -> bool:
        """Check if there's an edge from i to j."""
        return self.adj_matrix[i, j] > 0
    
    def nodes(self) -> List[str]:
        """Get list of node names."""
        return self._node_names.copy()
    
    def get_nodes(self) -> List[str]:
        """Alternative method name for getting nodes."""
        return self.nodes()
    
    def edges(self) -> List[Tuple[int, int]]:
        """Get list of edges as (source, target) tuples."""
        edges = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.has_edge(i, j):
                    edges.append((i, j))
        return edges
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Get adjacency matrix."""
        return self.adj_matrix

class DummySCM:
    """Minimal implementation of a structural causal model when SCM is unavailable."""
    
    def __init__(self, graph: Any):
        """Initialize with a graph structure."""
        self.graph = graph
        # Handle different graph types
        if hasattr(graph, 'get_num_nodes'):
            self.num_nodes = graph.get_num_nodes()
        elif hasattr(graph, 'num_nodes'):
            self.num_nodes = graph.num_nodes
        elif hasattr(graph, '_nodes'):
            # CausalGraph implementation
            self.num_nodes = len(graph._nodes)
        elif hasattr(graph, 'nodes') and callable(graph.nodes):
            # For graph implementations with a nodes() method
            self.num_nodes = len(graph.nodes())
        elif hasattr(graph, 'get_nodes') and callable(graph.get_nodes):
            # Alternative node accessor
            self.num_nodes = len(graph.get_nodes())
        elif hasattr(graph, 'adj_matrix'):
            # For graph implementations with an adjacency matrix
            self.num_nodes = graph.adj_matrix.shape[0]
        else:
            raise ValueError("Cannot determine number of nodes in the graph")
        
        self.node_names = [f"X_{i}" for i in range(self.num_nodes)]
        self._interventions = {}
    
    def sample_data(self, sample_size: int) -> np.ndarray:
        """Generate observational data."""
        return np.random.normal(0, 1, (sample_size, self.num_nodes))
    
    def sample_interventional_data(self, interventions: Dict[Union[int, str], float], sample_size: int) -> np.ndarray:
        """Generate interventional data."""
        data = np.random.normal(0, 1, (sample_size, self.num_nodes))
        for node, value in interventions.items():
            # Convert node name to index if needed
            if isinstance(node, str) and node.startswith('X_'):
                index = int(node.split('_')[1])
            else:
                index = node
            
            if 0 <= index < self.num_nodes:
                data[:, index] = value
        
        return data
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Get adjacency matrix of the underlying graph."""
        if hasattr(self.graph, 'get_adjacency_matrix'):
            return self.graph.get_adjacency_matrix()
        return self.graph.adj_matrix
    
    def define_structural_equation(self, variable: Union[int, str], 
                                  equation_function: Callable, 
                                  exogenous_function: Optional[Callable] = None) -> None:
        """Define structural equation for a variable (dummy implementation)."""
        pass
    
    def set_node_names(self, names: List[str]) -> None:
        """Set node names."""
        if len(names) != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} names, got {len(names)}")
        self.node_names = names.copy()

# Visualization fallback
def fallback_plot_graph(adj_matrix: np.ndarray, ax=None, edge_color='blue') -> plt.Axes:
    """Simple graph visualization when the actual plotting function is unavailable."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    
    # Place nodes in a circle
    n = adj_matrix.shape[0]
    positions = {}
    
    for i in range(n):
        angle = 2 * np.pi * i / n
        positions[i] = [np.cos(angle), np.sin(angle)]
    
    # Draw nodes
    for i in range(n):
        circle = plt.Circle(positions[i], 0.1, color='lightblue', zorder=2)
        ax.add_patch(circle)
        ax.text(positions[i][0], positions[i][1], str(i), 
                ha='center', va='center', zorder=3)
    
    # Draw edges
    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] > 0:
                ax.arrow(positions[i][0], positions[i][1],
                        positions[j][0] - positions[i][0],
                        positions[j][1] - positions[i][1],
                        head_width=0.05, head_length=0.1, fc=edge_color, ec=edge_color,
                        length_includes_head=True, alpha=0.7, zorder=1)
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')
    
    return ax

# Error handling utilities
def safe_import(module_name: str, fallback_value: Any = None) -> Any:
    """Safely import a module with a fallback value if import fails."""
    try:
        module = __import__(module_name, fromlist=[''])
        return module
    except ImportError:
        print(f"Warning: Failed to import {module_name}. Using fallback.")
        return fallback_value

# Define safe versions of key components
safe_GraphFactory = GraphFactory if 'GraphFactory' in globals() else None
safe_StructuralCausalModel = StructuralCausalModel if 'StructuralCausalModel' in globals() else None
safe_plot_graph = plot_graph if 'plot_graph' in globals() else fallback_plot_graph
safe_plot_causal_graph = plot_causal_graph if 'plot_causal_graph' in globals() else None
safe_GraphEncoder = GraphEncoder if 'GraphEncoder' in globals() else None
safe_DynamicsDecoder = DynamicsDecoder if 'DynamicsDecoder' in globals() else None
safe_AmortizedCausalDiscovery = AmortizedCausalDiscovery if 'AmortizedCausalDiscovery' in globals() else None
safe_TaskEmbedding = TaskEmbedding if 'TaskEmbedding' in globals() else None
safe_MAMLForCausalDiscovery = MAMLForCausalDiscovery if 'MAMLForCausalDiscovery' in globals() else None
safe_AmortizedCBO = AmortizedCBO if 'AmortizedCBO' in globals() else None 