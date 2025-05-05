#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shared utility functions for demo scripts.

This module provides shared functionality for the demo scripts, including
error handling, path management, tensor shape standardization, and proper
integration with components from the Component Registry.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Type
import importlib
import inspect

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Use safe imports to handle potential import errors gracefully
def safe_import(module_path: str, fallback_value: Any = None) -> Any:
    """
    Safely import a module or class, with a fallback value if import fails.
    
    Args:
        module_path: Dot-separated path to the module or class
        fallback_value: Value to return if import fails
        
    Returns:
        The imported module/class or the fallback value
    """
    try:
        if '.' in module_path:
            module_parts = module_path.split('.')
            module_name = '.'.join(module_parts[:-1])
            class_name = module_parts[-1]
            
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        else:
            return importlib.import_module(module_path)
    except (ImportError, AttributeError) as e:
        print(f"Warning: Failed to import {module_path}: {e}")
        return fallback_value

# Import from causal_meta package with error handling
CausalGraph = safe_import('causal_meta.graph.causal_graph.CausalGraph')
GraphFactory = safe_import('causal_meta.graph.generators.factory.GraphFactory')
StructuralCausalModel = safe_import('causal_meta.environments.scm.StructuralCausalModel')
plot_graph = safe_import('causal_meta.graph.visualization.plot_graph')
plot_causal_graph = safe_import('causal_meta.graph.visualization.plot_causal_graph')
GraphEncoder = safe_import('causal_meta.meta_learning.acd_models.GraphEncoder')
DynamicsDecoder = safe_import('causal_meta.meta_learning.dynamics_decoder.DynamicsDecoder')
AmortizedCausalDiscovery = safe_import('causal_meta.meta_learning.amortized_causal_discovery.AmortizedCausalDiscovery')
TaskEmbedding = safe_import('causal_meta.meta_learning.meta_learning.TaskEmbedding')
MAMLForCausalDiscovery = safe_import('causal_meta.meta_learning.meta_learning.MAMLForCausalDiscovery')
AmortizedCBO = safe_import('causal_meta.meta_learning.amortized_cbo.AmortizedCBO')

# Path management functions
def ensure_dir(directory: str) -> str:
    """
    Ensure a directory exists and return its path.
    
    Args:
        directory: Path to the directory
        
    Returns:
        The path to the existing or newly created directory
    """
    os.makedirs(directory, exist_ok=True)
    return directory

def get_assets_dir() -> str:
    """
    Get the assets directory path.
    
    Returns:
        Path to the assets directory
    """
    assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
    return ensure_dir(assets_dir)

def get_checkpoints_dir() -> str:
    """
    Get the model checkpoints directory path.
    
    Returns:
        Path to the checkpoints directory
    """
    checkpoints_dir = os.path.join(os.path.dirname(__file__), '..', 'example_checkpoints')
    return ensure_dir(checkpoints_dir)

# Tensor handling utilities
def standardize_tensor_shape(
    data: Union[np.ndarray, torch.Tensor], 
    for_component: str = 'graph_encoder',
    batch_size: int = 1,
    num_nodes: Optional[int] = None
) -> torch.Tensor:
    """
    Standardize tensor shape for neural network components.
    
    Different components expect different tensor shapes:
    - GraphEncoder: [batch_size, seq_length, num_nodes]
    - DynamicsDecoder: [batch_size * num_nodes, feature_dim]
    
    Args:
        data: Input data as numpy array or torch tensor
        for_component: Which component this is for ('graph_encoder', 'dynamics_decoder')
        batch_size: Batch size to use
        num_nodes: Number of nodes in the graph (required for some conversions)
        
    Returns:
        Properly shaped torch tensor
    """
    # Convert to torch tensor if numpy array
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
    
    # Handle different input shapes based on target component
    if for_component.lower() == 'graph_encoder':
        # For GraphEncoder - shape should be [batch_size, seq_length, num_nodes]
        if len(data.shape) == 1:
            # 1D vector: [samples] -> [1, samples, 1]
            return data.unsqueeze(0).unsqueeze(2)
        elif len(data.shape) == 2:
            # 2D matrix: [samples, nodes] -> [batch_size, samples, nodes]
            return data.unsqueeze(0).expand(batch_size, *data.shape)
        elif len(data.shape) == 3:
            # Already in correct shape, but may need to adjust batch_size
            if data.shape[0] != batch_size:
                # Replicate across batch dimension
                return data.expand(batch_size, *data.shape[1:])
            return data
        else:
            raise ValueError(f"Unsupported data shape for graph_encoder: {data.shape}")
    
    elif for_component.lower() == 'dynamics_decoder':
        # For DynamicsDecoder - shape should be [batch_size * num_nodes, feature_dim]
        if num_nodes is None:
            raise ValueError("num_nodes must be provided for dynamics_decoder")
            
        if len(data.shape) == 1:
            # 1D vector: [samples] -> [num_nodes, 1]
            return data.unsqueeze(1).expand(num_nodes * batch_size, 1)
        elif len(data.shape) == 2:
            # Handle different 2D shapes appropriately
            if data.shape[1] == 1:
                # [samples, 1] -> [num_nodes * batch_size, 1]
                # We can't expand different dimensions directly, so repeat the data
                return data.repeat(num_nodes * batch_size // data.shape[0] + 1, 1)[:num_nodes * batch_size]
            elif data.shape[1] == num_nodes:
                # [samples, nodes] -> [nodes, samples] -> [nodes * batch_size, 1]
                # If shape is [samples, nodes], we need to transpose and reshape
                return data.transpose(0, 1).reshape(num_nodes * batch_size, -1)
            else:
                # Generic case: reshape and repeat as needed
                return data.repeat(num_nodes, 1)
        elif len(data.shape) == 3:
            # 3D tensor: [batch, seq, nodes] -> [batch * nodes, seq]
            batch, seq_len, nodes = data.shape
            # Permute dimensions to get [batch, nodes, seq], then reshape
            return data.permute(0, 2, 1).reshape(batch * nodes, seq_len)
        else:
            raise ValueError(f"Unsupported data shape for dynamics_decoder: {data.shape}")
    
    else:
        raise ValueError(f"Unknown component type: {for_component}")

# Node naming utilities
def get_node_name(node_id: Union[int, str]) -> str:
    """
    Get standardized node name from node ID.
    
    Args:
        node_id: Integer index or string name
        
    Returns:
        Standardized string node name (format: "X_index")
    """
    if isinstance(node_id, str) and node_id.startswith('X_'):
        return node_id
    return f"X_{node_id}"

def get_node_id(node_name: Union[int, str]) -> int:
    """
    Get node ID from node name.
    
    Args:
        node_name: String node name or integer index
        
    Returns:
        Integer node index
        
    Raises:
        ValueError: If the node_name format is invalid
    """
    if isinstance(node_name, int):
        return node_name
    elif isinstance(node_name, str) and node_name.startswith('X_'):
        try:
            return int(node_name.split('_')[1])
        except (IndexError, ValueError):
            raise ValueError(f"Invalid node name format: {node_name}")
    else:
        raise ValueError(f"Invalid node name format: {node_name}")

def format_interventions(
    interventions: Dict[Union[int, str], float],
    for_tensor: bool = False,
    num_nodes: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Union[Dict[str, float], Dict[str, torch.Tensor]]:
    """
    Format interventions to use standardized node names or tensor format.
    
    Args:
        interventions: Dictionary mapping node IDs or names to intervention values
        for_tensor: Whether to format for neural network tensor input
        num_nodes: Number of nodes in the graph (required if for_tensor=True)
        device: PyTorch device for tensor creation
        
    Returns:
        Formatted interventions dictionary
    """
    if not for_tensor:
        # Simple string name formatting
        return {get_node_name(node): value for node, value in interventions.items()}
    
    # Format for tensor input to neural networks
    if num_nodes is None:
        raise ValueError("num_nodes must be provided when for_tensor=True")
    
    # Extract intervention targets and values
    targets = []
    values = []
    
    for node, value in interventions.items():
        # Convert to node ID if it's a string name
        if isinstance(node, str):
            node_id = get_node_id(node)
        else:
            node_id = node
        
        targets.append(node_id)
        values.append(value)
    
    # Convert to tensors
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    values_tensor = torch.tensor(values, dtype=torch.float32)
    
    # Move to device if specified
    if device is not None:
        targets_tensor = targets_tensor.to(device)
        values_tensor = values_tensor.to(device)
    
    return {
        'targets': targets_tensor,
        'values': values_tensor
    }

# Graph conversion utilities
def create_causal_graph_from_adjacency(
    adj_matrix: Union[np.ndarray, torch.Tensor],
    node_names: Optional[List[str]] = None,
    threshold: float = 0.5
) -> Any:
    """
    Create a CausalGraph from an adjacency matrix.
    
    Args:
        adj_matrix: Adjacency matrix as numpy array or torch tensor
        node_names: Optional list of node names (default: "X_0", "X_1", etc.)
        threshold: Threshold for edge detection in probabilistic adjacency matrices
        
    Returns:
        CausalGraph instance or a fallback graph representation
    """
    # Convert torch tensor to numpy if needed
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.detach().cpu().numpy()
    
    # Apply threshold for probabilistic adjacency matrices
    binary_adj = (adj_matrix > threshold).astype(np.float32)
    
    # Get number of nodes
    num_nodes = adj_matrix.shape[0]
    
    # Create default node names if not provided
    if node_names is None:
        node_names = [get_node_name(i) for i in range(num_nodes)]
    
    # Check if we have the CausalGraph component
    if CausalGraph is not None:
        # Create CausalGraph instance
        graph = CausalGraph()
        
        # Add nodes
        for node in node_names:
            graph.add_node(node)
        
        # Add edges
        for i in range(num_nodes):
            for j in range(num_nodes):
                if binary_adj[i, j] > 0:
                    graph.add_edge(node_names[i], node_names[j])
        
        return graph
    else:
        # Fallback to DummyGraph if CausalGraph is not available
        print("Warning: CausalGraph not available, using fallback implementation")
        return DummyGraph(binary_adj, node_names)

# Model loading utilities
def load_model(path, model_class=None, device='cpu', **model_params):
    """
    Load a PyTorch model from a checkpoint file with improved error handling.
    
    Args:
        path: Path to model checkpoint
        model_class: Class to instantiate if checkpoint not found
        device: PyTorch device
        **model_params: Parameters for model instantiation
        
    Returns:
        Loaded or newly instantiated model
    """
    try:
        print(f"Loading acd model from {path}...")
        
        # Check if file exists first
        if not os.path.exists(path):
            print(f"Checkpoint file not found at {path}")
            raise FileNotFoundError(f"No checkpoint file at {path}")
            
        # Attempt to load the checkpoint
        checkpoint = torch.load(path, map_location=device)
        
        # Handle different checkpoint formats
        if hasattr(checkpoint, 'state_dict'):
            # Model was saved directly
            model = checkpoint
        elif isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                # Extract model from dictionary
                model = checkpoint['model']
            elif 'state_dict' in checkpoint:
                # We have a state dict, need to instantiate model
                if model_class is None:
                    raise ValueError("Cannot load state_dict without model_class")
                model = model_class(**model_params)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Assume it's a direct state dict
                if model_class is None:
                    raise ValueError("Cannot load state_dict without model_class")
                model = model_class(**model_params)
                try:
                    model.load_state_dict(checkpoint)
                except Exception as e:
                    print(f"Failed to load state dict: {str(e)}")
                    raise
        else:
            raise ValueError(f"Unrecognized checkpoint format: {type(checkpoint)}")
        
        # Ensure model is on the right device
        model = model.to(device)
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        
        if model_class is not None:
            # Create a new model instance as fallback
            print(f"Creating a new {model_class.__name__} model with parameters: {model_params}")
            model = model_class(**model_params)
            model = model.to(device)
            return model
        else:
            print("Cannot instantiate fallback model: model_class not provided")
            # Re-raise the exception
            raise

def create_new_model(
    model_type: str,
    model_params: Optional[Dict[str, Any]] = None,
    device: torch.device = torch.device('cpu')
) -> Optional[nn.Module]:
    """
    Create a new model instance.
    
    Args:
        model_type: Type of model to create ('acd', 'graph_encoder', 'dynamics_decoder')
        model_params: Parameters for model creation
        device: PyTorch device for the model
        
    Returns:
        New model instance or None if the model type is not available
    """
    if model_params is None:
        model_params = {}
    
    # Set default parameters if not provided
    default_params = {
        'hidden_dim': 64,
        'input_dim': 1,
        'attention_heads': 2,
        'num_layers': 2,
        'dropout': 0.1,
        'sparsity_weight': 0.1,
        'acyclicity_weight': 1.0
    }
    
    # Merge defaults with provided params
    params = {**default_params, **model_params}
    
    print(f"Creating a new {model_type} model with parameters: {params}")
    
    if model_type == 'acd':
        if AmortizedCausalDiscovery is not None:
            model = AmortizedCausalDiscovery(
                hidden_dim=params['hidden_dim'],
                input_dim=params['input_dim'],
                attention_heads=params['attention_heads'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                sparsity_weight=params['sparsity_weight'],
                acyclicity_weight=params['acyclicity_weight']
            )
        else:
            print("Warning: AmortizedCausalDiscovery not available")
            return None
    
    elif model_type == 'graph_encoder':
        if GraphEncoder is not None:
            model = GraphEncoder(
                hidden_dim=params['hidden_dim'],
                attention_heads=params['attention_heads'],
                num_layers=params['num_layers'],
                sparsity_weight=params['sparsity_weight'],
                acyclicity_weight=params['acyclicity_weight']
            )
        else:
            print("Warning: GraphEncoder not available")
            return None
    
    elif model_type == 'dynamics_decoder':
        if DynamicsDecoder is not None:
            model = DynamicsDecoder(
                input_dim=params['input_dim'],
                hidden_dim=params['hidden_dim'],
                num_layers=params['num_layers'],
                dropout=params['dropout']
            )
        else:
            print("Warning: DynamicsDecoder not available")
            return None
    
    else:
        print(f"Unknown model type: {model_type}")
        return None
    
    return model.to(device)

# Visualization utilities
def visualize_graph(
    graph: Any,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    layout: str = 'spring',
    highlight_nodes: Optional[List[Union[int, str]]] = None,
    save_path: Optional[str] = None
) -> plt.Axes:
    """
    Visualize a graph using appropriate visualization function.
    
    Args:
        graph: Graph to visualize (CausalGraph or adjacency matrix)
        ax: Matplotlib axes to plot on
        title: Title for the plot
        figsize: Figure size as (width, height)
        layout: Layout algorithm to use
        highlight_nodes: List of nodes to highlight
        save_path: Path to save the figure
        
    Returns:
        Matplotlib axes containing the plot
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # If we have the plot_graph function from causal_meta
    if plot_graph is not None and isinstance(graph, (CausalGraph, type(CausalGraph))):
        # Use the built-in visualization
        ax = plot_graph(
            graph,
            ax=ax,
            title=title,
            layout=layout,
            highlight_nodes=highlight_nodes
        )
    else:
        # Fallback to simple visualization
        ax = fallback_plot_graph(
            graph if isinstance(graph, np.ndarray) else graph.get_adjacency_matrix(),
            ax=ax,
            title=title
        )
    
    # Add title if provided
    if title:
        ax.set_title(title)
    
    # Save if path provided
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return ax

def compare_graphs(
    left_graph: Any,
    right_graph: Any,
    left_title: str = "Left Graph",
    right_title: str = "Right Graph",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare two graphs side by side.
    
    Args:
        left_graph: First graph to visualize
        right_graph: Second graph to visualize
        left_title: Title for the left plot
        right_title: Title for the right plot
        figsize: Figure size as (width, height)
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure containing the plots
    """
    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Visualize left graph
    visualize_graph(left_graph, ax=axes[0], title=left_title)
    
    # Visualize right graph
    visualize_graph(right_graph, ax=axes[1], title=right_title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

# Fallback implementations - only used when actual implementations are unavailable
class DummyGraph:
    """Minimal implementation of a graph structure when CausalGraph is unavailable."""
    
    def __init__(self, adj_matrix: np.ndarray, node_names: Optional[List[str]] = None):
        """Initialize with adjacency matrix and optional node names."""
        self.adj_matrix = np.array(adj_matrix)
        self.num_nodes = adj_matrix.shape[0]
        
        # Use provided node names or generate default names
        if node_names is None:
            self._node_names = [f"X_{i}" for i in range(self.num_nodes)]
        else:
            if len(node_names) != self.num_nodes:
                raise ValueError(f"Expected {self.num_nodes} node names, got {len(node_names)}")
            self._node_names = list(node_names)
    
    def get_num_nodes(self) -> int:
        """Get number of nodes."""
        return self.num_nodes
    
    def has_edge(self, i: int, j: int) -> bool:
        """Check if there's an edge from i to j."""
        if isinstance(i, str):
            i = self._node_names.index(i)
        if isinstance(j, str):
            j = self._node_names.index(j)
        return self.adj_matrix[i, j] > 0
    
    def get_nodes(self) -> List[str]:
        """Get list of node names."""
        return self._node_names.copy()
    
    def get_edges(self) -> List[Tuple[str, str]]:
        """Get list of edges as (source, target) tuples."""
        edges = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.has_edge(i, j):
                    edges.append((self._node_names[i], self._node_names[j]))
        return edges
    
    def add_node(self, node: str) -> None:
        """Add a node to the graph."""
        if node not in self._node_names:
            self._node_names.append(node)
            # Expand adjacency matrix
            new_size = len(self._node_names)
            new_adj = np.zeros((new_size, new_size))
            new_adj[:self.num_nodes, :self.num_nodes] = self.adj_matrix
            self.adj_matrix = new_adj
            self.num_nodes = new_size
    
    def add_edge(self, u: str, v: str) -> None:
        """Add an edge to the graph."""
        if u not in self._node_names:
            self.add_node(u)
        if v not in self._node_names:
            self.add_node(v)
        
        i = self._node_names.index(u)
        j = self._node_names.index(v)
        self.adj_matrix[i, j] = 1
    
    def get_adjacency_matrix(self, node_order: Optional[List[str]] = None) -> np.ndarray:
        """Get adjacency matrix with optional node ordering."""
        if node_order is None:
            return self.adj_matrix
        
        # Create new adjacency matrix with specified node order
        indices = [self._node_names.index(node) for node in node_order]
        return self.adj_matrix[np.ix_(indices, indices)]
    
    def get_parents(self, node: str) -> List[str]:
        """Get parents of a node."""
        if isinstance(node, int):
            node = self._node_names[node]
        
        j = self._node_names.index(node)
        parents = []
        
        for i in range(self.num_nodes):
            if self.adj_matrix[i, j] > 0:
                parents.append(self._node_names[i])
        
        return parents
    
    def get_children(self, node: str) -> List[str]:
        """Get children of a node."""
        if isinstance(node, int):
            node = self._node_names[node]
        
        i = self._node_names.index(node)
        children = []
        
        for j in range(self.num_nodes):
            if self.adj_matrix[i, j] > 0:
                children.append(self._node_names[j])
        
        return children

    def do_intervention(self, node_id, value=None):
        """
        Create a new graph reflecting a do-intervention.
        
        Args:
            node_id: Node to intervene on
            value: Intervention value (not used in graph structure)
            
        Returns:
            New graph with incoming edges to node_id removed
        """
        # Convert node_id to internal index
        if isinstance(node_id, str):
            if node_id in self._node_names:
                node_idx = self._node_names.index(node_id)
            else:
                try:
                    node_idx = int(node_id.split('_')[1])
                except (IndexError, ValueError):
                    raise ValueError(f"Invalid node name: {node_id}")
        else:
            node_idx = node_id
        
        # Create a new adjacency matrix
        new_adj = self.adj_matrix.copy()
        
        # Remove incoming edges to the intervened node
        new_adj[:, node_idx] = 0
        
        # Create a new graph with the modified adjacency matrix
        new_graph = DummyGraph(new_adj, self._node_names.copy())
        
        return new_graph

# Visualization fallback
def fallback_plot_graph(adj_matrix: np.ndarray, ax=None, title=None) -> plt.Axes:
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
                       head_width=0.05, head_length=0.1, fc='black', ec='black',
                       length_includes_head=True, zorder=1)
    
    # Set limits and title
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    if title:
        ax.set_title(title)
    
    return ax 

def infer_adjacency_matrix(model, data, interventions=None, threshold=None):
    """
    Wrapper for inferring adjacency matrix that works with both actual models and dummy implementations.
    
    Args:
        model: AmortizedCausalDiscovery model or similar
        data: Input data tensor
        interventions: Optional interventions dictionary
        threshold: Optional threshold for binarizing the matrix
        
    Returns:
        Adjacency matrix as tensor
    """
    # Check if the model has infer_adjacency_matrix method directly
    if hasattr(model, 'infer_adjacency_matrix'):
        return model.infer_adjacency_matrix(data, interventions=interventions)
        
    # Check if it's an AmortizedCausalDiscovery model
    elif hasattr(model, 'infer_causal_graph'):
        # Call infer_causal_graph and extract the adjacency matrix
        with torch.no_grad():
            return model.graph_encoder(data)
        
    # Fallback: generate a random adjacency matrix for testing
    else:
        print("Warning: Using fallback random adjacency matrix generation")
        if isinstance(data, torch.Tensor):
            # Infer the number of nodes from the data
            # Assume data is [batch_size, seq_len, num_nodes]
            if len(data.shape) == 3:
                num_nodes = data.shape[2]
            elif len(data.shape) == 2:
                num_nodes = data.shape[1]
            else:
                num_nodes = 5  # Default fallback
                
            # Generate random matrix with PyTorch
            adj_matrix = torch.rand((num_nodes, num_nodes), device=data.device)
            # Make it sparse and zero the diagonal (no self-loops)
            adj_matrix = (adj_matrix > 0.7).float() * (1 - torch.eye(num_nodes, device=data.device))
            return adj_matrix
        else:
            # Numpy fallback
            num_nodes = 5
            adj_matrix = np.random.random((num_nodes, num_nodes))
            adj_matrix = (adj_matrix > 0.7).astype(float)
            np.fill_diagonal(adj_matrix, 0)
            return torch.tensor(adj_matrix, dtype=torch.float32)

def convert_to_structural_equation_model(graph, node_names=None, num_samples=100):
    """
    Convert a graph to a StructuralCausalModel with appropriate structural equations.
    
    Args:
        graph: CausalGraph or similar with adjacency matrix
        node_names: Optional list of node names
        num_samples: Number of samples to use for testing
        
    Returns:
        StructuralCausalModel instance or fallback implementation
    """
    # Get adjacency matrix
    if hasattr(graph, 'get_adjacency_matrix'):
        adj_matrix = graph.get_adjacency_matrix()
    elif isinstance(graph, np.ndarray):
        adj_matrix = graph
    elif isinstance(graph, torch.Tensor):
        adj_matrix = graph.cpu().numpy()
    else:
        raise ValueError(f"Unsupported graph type: {type(graph)}")
    
    # Get node names if not provided
    if node_names is None:
        if hasattr(graph, 'get_nodes'):
            node_names = graph.get_nodes()
        else:
            num_nodes = adj_matrix.shape[0]
            node_names = [get_node_name(i) for i in range(num_nodes)]
    
    # Try to use the actual StructuralCausalModel
    if StructuralCausalModel is not None:
        # Create or convert graph to CausalGraph
        if CausalGraph is not None and not isinstance(graph, CausalGraph):
            causal_graph = CausalGraph()
            # Add nodes
            for node in node_names:
                causal_graph.add_node(node)
            # Add edges based on adjacency matrix
            num_nodes = adj_matrix.shape[0]
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if adj_matrix[i, j] > 0:
                        causal_graph.add_edge(node_names[i], node_names[j])
        else:
            causal_graph = graph
            
        # Create the SCM
        scm = StructuralCausalModel(
            causal_graph=causal_graph,
            variable_names=node_names
        )
        
        # Define structural equations
        for node in node_names:
            parents = list(causal_graph.get_parents(node))
            
            # Create dynamic structural equation function with proper signature
            parent_args = ", ".join(parents)
            noise_arg = ", noise" if parent_args else "noise"
            func_def = f"def equation_function({parent_args}{noise_arg}):\n"
            func_def += "    result = noise\n"
            
            # Add parent contributions
            for parent in parents:
                coef = np.random.uniform(-1.0, 1.0)
                func_def += f"    result += {coef} * {parent}\n"
                
            func_def += "    return result\n"
            
            # Create the function namespace
            namespace = {}
            
            # Execute the function definition
            exec(func_def, namespace)
            
            # Get the function from the namespace
            equation_function = namespace['equation_function']
            
            # Define noise function
            def noise_function(sample_size, random_state=None):
                # Instead of using seed directly, create a new RandomState if needed
                if random_state is not None and not isinstance(random_state, np.random.RandomState):
                    try:
                        # Try to convert to int for seeding
                        seed = int(random_state)
                        rng = np.random.RandomState(seed)
                    except (TypeError, ValueError):
                        # If conversion fails, use a default seed
                        print(f"Warning: Could not use provided random_state as seed, using default")
                        rng = np.random.RandomState(42)
                else:
                    # Either use the provided RandomState or the global state
                    rng = random_state if random_state is not None else np.random
                
                # Generate noise using the RNG
                return rng.normal(0, 0.5, size=sample_size)
            
            # Define the probabilistic structural equation
            scm.define_probabilistic_equation(
                variable=node,
                equation_function=equation_function,
                noise_distribution=noise_function
            )
            
        return scm
    else:
        # Create a dummy SCM object with essential methods
        class DummySCM:
            def __init__(self, graph, node_names):
                self.graph = graph
                self.node_names = node_names
                self.adj_matrix = adj_matrix
                
            def get_causal_graph(self):
                return self.graph
                
            def get_adjacency_matrix(self):
                return self.adj_matrix
                
            def sample_data(self, sample_size=100):
                # Generate random data with causal structure
                data = {}
                num_nodes = len(self.node_names)
                
                # Initialize with random noise
                for node in self.node_names:
                    data[node] = np.random.normal(0, 1, size=sample_size)
                
                # Propagate causal effects
                for i, node in enumerate(self.node_names):
                    for j, parent in enumerate(self.node_names):
                        if self.adj_matrix[j, i] > 0:
                            # Add effect from parent to node
                            effect = np.random.uniform(0.5, 1.5)
                            data[node] += effect * data[parent]
                
                return pd.DataFrame(data)
                
            def sample_interventional_data(self, interventions, sample_size=100):
                # Start with observational data
                data = self.sample_data(sample_size)
                
                # Apply interventions
                for node, value in interventions.items():
                    data[node] = value
                    
                    # Propagate effects to children
                    for i, child in enumerate(self.node_names):
                        if self.adj_matrix[self.node_names.index(node), i] > 0:
                            # Update child based on intervention
                            effect = np.random.uniform(0.5, 1.5)
                            data[child] = effect * value + np.random.normal(0, 0.5, size=sample_size)
                
                return data
                
        # Need to ensure pandas is available for the dummy implementation
        import pandas as pd
        
        return DummySCM(graph, node_names) 