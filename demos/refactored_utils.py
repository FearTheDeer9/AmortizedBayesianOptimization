#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Refactored utility functions for demo scripts.

This module provides improved implementations of utility functions that properly
leverage existing components from the causal_meta package according to the
Component Registry. It replaces duplicate implementations in the existing utils.py
with direct use of the official components.
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Type
import importlib
import inspect
import json
import pickle
import copy
import random
import networkx as nx

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import specialized components with fallbacks
AmortizedCausalDiscovery = None
try:
    from causal_meta.meta_learning.amortized_causal_discovery import AmortizedCausalDiscovery
except ImportError:
    logger.warning("Failed to import AmortizedCausalDiscovery. Some functionality may not be available.")

# Import handling with comprehensive error messages and fallbacks
def safe_import(module_path: str, fallback_value: Any = None) -> Any:
    """
    Safely import a module or class with improved error handling and fallback mechanisms.
    
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
        logger.warning(f"Failed to import {module_path}: {e}")
        
        # Check for common import issues and provide helpful advice
        if "No module named 'causal_meta'" in str(e):
            logger.error(
                "The causal_meta package is not installed. Please ensure the package "
                "is installed correctly or that the repository is in your PYTHONPATH."
            )
        elif "cannot import name" in str(e):
            logger.error(
                f"The component {module_path.split('.')[-1]} was not found. "
                "Check that you're using the correct import path according to "
                "the Component Registry."
            )
        
        return fallback_value

# Import all required components from the Component Registry
CausalGraph = safe_import('causal_meta.graph.causal_graph.CausalGraph')
GraphFactory = safe_import('causal_meta.graph.generators.factory.GraphFactory')
StructuralCausalModel = safe_import('causal_meta.environments.scm.StructuralCausalModel')
TaskFamily = safe_import('causal_meta.graph.task_family.TaskFamily')
plot_graph = safe_import('causal_meta.graph.visualization.plot_graph')
plot_graphs_comparison = safe_import('causal_meta.graph.visualization.plot_graphs_comparison')
GraphEncoder = safe_import('causal_meta.meta_learning.acd_models.GraphEncoder')
DynamicsDecoder = safe_import('causal_meta.meta_learning.dynamics_decoder.DynamicsDecoder')
AmortizedCausalDiscovery = safe_import('causal_meta.meta_learning.amortized_causal_discovery.AmortizedCausalDiscovery')
TaskEmbedding = safe_import('causal_meta.meta_learning.meta_learning.TaskEmbedding')

# First try to import from causal_meta, if not available use our custom implementation
MAMLForCausalDiscovery = safe_import('causal_meta.meta_learning.meta_learning.MAMLForCausalDiscovery')
if MAMLForCausalDiscovery is None:
    try:
        # Try to import from our custom implementation
        from causal_meta.meta_learning.maml_causal_discovery_demo import MAMLForCausalDiscovery
        logger.info("Using custom implementation of MAMLForCausalDiscovery")
    except ImportError as e:
        logger.warning(f"Could not import MAMLForCausalDiscovery: {e}")
        MAMLForCausalDiscovery = None

AmortizedCBO = safe_import('causal_meta.meta_learning.amortized_cbo.AmortizedCBO')
SyntheticDataGenerator = safe_import('causal_meta.meta_learning.data_generation.SyntheticDataGenerator')

# Path management functions with improved error handling
def ensure_dir(directory: str) -> str:
    """
    Ensure a directory exists and return its path.
    
    Args:
        directory: Path to the directory
        
    Returns:
        The path to the existing or newly created directory
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return directory
    except Exception as e:
        logger.error(f"Failed to create directory {directory}: {e}")
        raise

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

# Tensor handling utilities with improved error handling and shape validation
def standardize_tensor_shape(
    data: Union[np.ndarray, torch.Tensor], 
    for_component: str = 'graph_encoder',
    batch_size: int = 1,
    num_nodes: Optional[int] = None
) -> torch.Tensor:
    """
    Standardize tensor shape for neural network components with improved handling.
    
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
    # Input validation
    if data is None:
        raise ValueError("Input data cannot be None")
    
    # Convert to torch tensor if numpy array with proper type handling
    if isinstance(data, np.ndarray):
        if np.isnan(data).any():
            logger.warning("NaN values detected in input data. These may cause issues with neural components.")
        data = torch.tensor(data, dtype=torch.float32)
    
    # Infer num_nodes if not provided but needed
    if num_nodes is None and for_component.lower() == 'dynamics_decoder':
        if len(data.shape) == 2:
            num_nodes = data.shape[1]
            logger.info(f"Inferring num_nodes={num_nodes} from input data shape {data.shape}")
        elif len(data.shape) == 3:
            num_nodes = data.shape[2]
            logger.info(f"Inferring num_nodes={num_nodes} from input data shape {data.shape}")
        else:
            raise ValueError("Cannot infer num_nodes from input data shape. Please provide it explicitly.")
    
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
            if data.shape[0] != batch_size and batch_size > 1:
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
                return data.repeat(num_nodes * batch_size // data.shape[0] + 1, 1)[:num_nodes * batch_size]
            elif data.shape[1] == num_nodes:
                # [samples, nodes] -> [nodes, samples] -> [nodes * batch_size, 1]
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
        raise ValueError(f"Unknown component type: {for_component}. " +
                         "Expected 'graph_encoder' or 'dynamics_decoder'.")

# Node naming utilities with improved consistency
def get_node_name(node_id: Union[int, str]) -> str:
    """
    Get standardized node name (format: 'x{index}').
    Args:
        node_id: Integer index or string name
    Returns:
        Standardized string node name (format: 'x{index}')
    """
    if isinstance(node_id, str) and node_id.startswith('x'):
        return node_id
    return f"x{int(node_id)}"

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
    elif isinstance(node_name, str) and node_name.startswith('x'):
        try:
            return int(node_name[1:])
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

def create_causal_graph_from_adjacency(
    adj_matrix: Union[np.ndarray, torch.Tensor],
    node_names: Optional[List[str]] = None,
    threshold: float = 0.5
) -> Any:
    """
    Create a causal graph from an adjacency matrix, using CausalGraph when available.
    
    Args:
        adj_matrix: Adjacency matrix as numpy array or torch tensor
        node_names: List of node names (default: "x0", "x1", etc.)
        threshold: Threshold for binarizing probabilistic edges
        
    Returns:
        CausalGraph object or a fallback graph representation
    """
    # Convert torch tensor to numpy if needed
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.detach().cpu().numpy()
    
    # Apply threshold for probabilistic adjacency matrices
    if np.max(adj_matrix) <= 1.0 and adj_matrix.dtype in [np.float32, np.float64]:
        binary_adj = (adj_matrix > threshold).astype(float)
    else:
        binary_adj = adj_matrix
    
    # Determine number of nodes
    num_nodes = adj_matrix.shape[0]
    
    # Create node names if not provided
    if node_names is None:
        node_names = [get_node_name(i) for i in range(num_nodes)]
    
    # If CausalGraph is available, use it
    if CausalGraph is not None:
        try:
            # Create a CausalGraph object
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
        except Exception as e:
            logger.error(f"Error creating CausalGraph: {e}")
            logger.info("Falling back to DummyGraph implementation")
    
    # Fallback to DummyGraph if CausalGraph is not available
    return DummyGraph(binary_adj)

def load_model(
    path: str, 
    model_class: Any = None, 
    device: Union[str, torch.device] = 'cpu', 
    **model_params
) -> Optional[nn.Module]:
    """
    Load neural model from checkpoint with improved error handling.
    
    Args:
        path: Path to model checkpoint
        model_class: Model class to instantiate (defaults to AmortizedCausalDiscovery)
        device: PyTorch device
        **model_params: Additional parameters for model initialization
        
    Returns:
        Loaded model or None if loading failed
    """
    # Convert string device to torch.device
    if isinstance(device, str):
        device = torch.device(device)
    
    # Default to AmortizedCausalDiscovery if model_class not specified
    if model_class is None:
        model_class = AmortizedCausalDiscovery
        logger.info("No model class specified, defaulting to AmortizedCausalDiscovery")
    
    # Check if model_class is available
    if model_class is None:
        logger.error("Model class not available. Check that the causal_meta package is installed correctly.")
        return None
    
    # Check if checkpoint file exists
    if not os.path.exists(path):
        logger.warning(f"Model checkpoint not found at {path}")
        
        # Look for checkpoint in example_checkpoints directory
        checkpoints_dir = get_checkpoints_dir()
        alt_path = os.path.join(checkpoints_dir, os.path.basename(path))
        
        if os.path.exists(alt_path):
            logger.info(f"Found model checkpoint at alternative path: {alt_path}")
            path = alt_path
        else:
            logger.info("Creating new model instead of loading checkpoint")
            try:
                # Try to initialize model with provided parameters
                if model_class == AmortizedCausalDiscovery:
                    # Use the direct parameter approach for AmortizedCausalDiscovery
                    model = AmortizedCausalDiscovery(
                        hidden_dim=model_params.get('hidden_dim', 64),
                        input_dim=model_params.get('input_dim', 1),
                        attention_heads=model_params.get('attention_heads', 2),
                        num_layers=model_params.get('num_layers', 2),
                        dropout=model_params.get('dropout', 0.1),
                        sparsity_weight=model_params.get('sparsity_weight', 0.1),
                        acyclicity_weight=model_params.get('acyclicity_weight', 1.0),
                        dynamics_weight=model_params.get('dynamics_weight', 1.0),
                        structure_weight=model_params.get('structure_weight', 1.0),
                        uncertainty=model_params.get('uncertainty', False),
                        num_ensembles=model_params.get('num_ensembles', 5)
                    )
                else:
                    # For other model classes, pass parameters directly
                    model = model_class(**model_params)
                
                model.to(device)
                logger.info(f"Created new {model_class.__name__} model")
                return model
            except Exception as e:
                logger.error(f"Error creating model: {e}")
                return None
    
    # Try to load the model
    try:
        # Load the state dict
        checkpoint = torch.load(path, map_location=device)
        
        # Create a new model instance
        if model_class == AmortizedCausalDiscovery:
            # Use the direct parameter approach for AmortizedCausalDiscovery
            model = AmortizedCausalDiscovery(
                hidden_dim=model_params.get('hidden_dim', 64),
                input_dim=model_params.get('input_dim', 1),
                attention_heads=model_params.get('attention_heads', 2),
                num_layers=model_params.get('num_layers', 2),
                dropout=model_params.get('dropout', 0.1),
                sparsity_weight=model_params.get('sparsity_weight', 0.1),
                acyclicity_weight=model_params.get('acyclicity_weight', 1.0),
                dynamics_weight=model_params.get('dynamics_weight', 1.0),
                structure_weight=model_params.get('structure_weight', 1.0),
                uncertainty=model_params.get('uncertainty', False),
                num_ensembles=model_params.get('num_ensembles', 5)
            )
        else:
            # For other model classes, pass parameters directly
            model = model_class(**model_params)
        
        # Load state dict with error handling for mismatched keys
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        
        if missing:
            logger.warning(f"Missing keys in state dict: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys in state dict: {unexpected}")
        
        # Move model to device
        model.to(device)
        logger.info(f"Successfully loaded model from {path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {path}: {e}")
        logger.info("Creating new model instead")
        
        try:
            # Try to initialize model with provided parameters
            if model_class == AmortizedCausalDiscovery:
                # Use the direct parameter approach for AmortizedCausalDiscovery
                model = AmortizedCausalDiscovery(
                    hidden_dim=model_params.get('hidden_dim', 64),
                    input_dim=model_params.get('input_dim', 1),
                    attention_heads=model_params.get('attention_heads', 2),
                    num_layers=model_params.get('num_layers', 2),
                    dropout=model_params.get('dropout', 0.1),
                    sparsity_weight=model_params.get('sparsity_weight', 0.1),
                    acyclicity_weight=model_params.get('acyclicity_weight', 1.0),
                    dynamics_weight=model_params.get('dynamics_weight', 1.0),
                    structure_weight=model_params.get('structure_weight', 1.0),
                    uncertainty=model_params.get('uncertainty', False),
                    num_ensembles=model_params.get('num_ensembles', 5)
                )
            else:
                # For other model classes, pass parameters directly
                model = model_class(**model_params)
            
            model.to(device)
            logger.info(f"Created new {model_class.__name__} model")
            return model
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            return None

def infer_adjacency_matrix(model, encoder_input, interventions=None):
    """
    Infer adjacency matrix from encoder input using the given model.
    
    Args:
        model: Neural network model
        encoder_input: Input for the graph encoder component [batch_size, num_nodes] or [batch_size, seq_len, num_nodes]
        interventions: Optional intervention information to be encoded
        
    Returns:
        Inferred adjacency matrix as a tensor [batch_size, num_nodes, num_nodes] or [num_nodes, num_nodes]
    """
    # Set model to evaluation mode
    model.eval()
    
    # Standardize input dimensions
    with torch.no_grad():
        # Handle 1D input (single sample, multiple nodes)
        if encoder_input.dim() == 1:
            # [num_nodes] -> [1, num_nodes]
            encoder_input = encoder_input.unsqueeze(0)
        
        # Extract dimensions from input
        if encoder_input.dim() == 2:
            # [batch_size, num_nodes]
            batch_size, num_nodes = encoder_input.shape
            has_seq_dim = False
        elif encoder_input.dim() == 3:
            # [batch_size, seq_len, num_nodes]
            batch_size, seq_len, num_nodes = encoder_input.shape
            has_seq_dim = True
        else:
            raise ValueError(f"Unexpected encoder_input shape: {encoder_input.shape}. Expected 1D, 2D, or 3D tensor.")
        
        # Forward pass with appropriate method based on model type
        try:
            # Check if model has methods specifically for handling interventions
            if hasattr(model, 'forward_with_interventions') and interventions is not None:
                # Use the model's dedicated method for intervention-aware inference
                inferred_adj_matrix = model.forward_with_interventions(encoder_input, interventions)
            elif hasattr(model, 'encode_interventions') and interventions is not None:
                # Use the model's intervention encoding method
                inferred_adj_matrix = model(encoder_input, encode_interventions=interventions)
            elif isinstance(model, AmortizedCausalDiscovery):
                # Special handling for AmortizedCausalDiscovery model
                device = encoder_input.device
                
                # Create node features (just use 1D features for simplicity)
                node_features = torch.ones((batch_size * num_nodes, 1), device=device)
                
                # Create a fully connected edge index for initial inference
                edge_index = []
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if i != j:  # No self-loops
                            edge_index.append([i, j])
                edge_index = torch.tensor(edge_index, device=device).t()
                
                # Create batch assignment tensor
                batch = torch.repeat_interleave(
                    torch.arange(batch_size, device=device),
                    repeats=num_nodes
                )
                
                # Handle input shape for AmortizedCausalDiscovery
                x_input = encoder_input
                if not has_seq_dim and hasattr(model, 'expects_seq_dim') and model.expects_seq_dim:
                    # Add sequence dimension if model expects it
                    x_input = encoder_input.unsqueeze(1)
                
                # Call the model with all required parameters
                outputs = model(
                    x=x_input,
                    node_features=node_features,
                    edge_index=edge_index,
                    batch=batch,
                    interventions=interventions if interventions is not None else None,
                    return_uncertainty=False
                )
                
                # Extract adjacency matrix from outputs
                if isinstance(outputs, dict) and 'adjacency' in outputs:
                    inferred_adj_matrix = outputs['adjacency']
                else:
                    # Assume the output is the adjacency matrix
                    inferred_adj_matrix = outputs
            else:
                # Handle model specific requirements for input shape
                model_input = encoder_input
                
                # Check if model expects 3D input with sequence dimension
                if not has_seq_dim and hasattr(model, 'expects_3d_input') and model.expects_3d_input:
                    # Add a dummy sequence dimension [batch_size, 1, num_nodes]
                    model_input = encoder_input.unsqueeze(1)
                
                # Standard forward pass
                outputs = model(model_input)
                
                # Extract adjacency matrix depending on output format
                if isinstance(outputs, dict) and 'adjacency' in outputs:
                    inferred_adj_matrix = outputs['adjacency']
                elif isinstance(outputs, tuple) and len(outputs) > 0:
                    inferred_adj_matrix = outputs[0]
                else:
                    # Assume the output is the adjacency matrix
                    inferred_adj_matrix = outputs
            
            # Standardize the output shape
            # Expected output: [batch_size, num_nodes, num_nodes] or [num_nodes, num_nodes]
            if inferred_adj_matrix.dim() == 2:
                # Check if this is already a valid adjacency matrix
                if inferred_adj_matrix.shape[0] == inferred_adj_matrix.shape[1] == num_nodes:
                    # Valid [num_nodes, num_nodes] adjacency matrix
                    return inferred_adj_matrix
                else:
                    # Try to reshape flat tensor to proper adjacency shape
                    # This handles cases where the model outputs a flattened adjacency matrix
                    if inferred_adj_matrix.shape[0] == batch_size and inferred_adj_matrix.shape[1] == num_nodes * num_nodes:
                        # Reshape [batch_size, num_nodes*num_nodes] -> [batch_size, num_nodes, num_nodes]
                        return inferred_adj_matrix.view(batch_size, num_nodes, num_nodes)
                    else:
                        logger.warning(f"Unexpected adjacency shape: {inferred_adj_matrix.shape}, expected [{num_nodes}, {num_nodes}]")
                        # Try to reshape to match expected shape
                        if inferred_adj_matrix.numel() == num_nodes * num_nodes:
                            return inferred_adj_matrix.view(num_nodes, num_nodes)
                        else:
                            # Return as is if we can't reshape
                            return inferred_adj_matrix
            
            elif inferred_adj_matrix.dim() == 3:
                # Check if the output is already in the correct shape
                if inferred_adj_matrix.shape[1:] == (num_nodes, num_nodes):
                    # If batch_size is 1 and we need a 2D output, squeeze
                    if batch_size == 1 and inferred_adj_matrix.shape[0] == 1:
                        return inferred_adj_matrix.squeeze(0)
                    else:
                        return inferred_adj_matrix
                else:
                    logger.warning(f"Unexpected adjacency shape: {inferred_adj_matrix.shape}, expected [batch, {num_nodes}, {num_nodes}]")
                    # If dimensions are swapped, try to transpose
                    if inferred_adj_matrix.shape[1] == num_nodes and inferred_adj_matrix.shape[2] == num_nodes:
                        return inferred_adj_matrix
                    else:
                        # Return as is if we can't fix
                        return inferred_adj_matrix
            else:
                # Handle unexpected number of dimensions
                logger.warning(f"Unexpected adjacency dimensions: {inferred_adj_matrix.dim()}, shape: {inferred_adj_matrix.shape}")
                
                if inferred_adj_matrix.numel() == num_nodes * num_nodes:
                    # Try to reshape to [num_nodes, num_nodes]
                    return inferred_adj_matrix.view(num_nodes, num_nodes)
                elif inferred_adj_matrix.numel() == batch_size * num_nodes * num_nodes:
                    # Try to reshape to [batch_size, num_nodes, num_nodes]
                    return inferred_adj_matrix.view(batch_size, num_nodes, num_nodes)
                else:
                    # Return as is if we can't fix
                    return inferred_adj_matrix
                
        except Exception as e:
            logger.error(f"Error in adjacency matrix inference: {e}")
            # Create a safe fallback output
            return torch.zeros((batch_size, num_nodes, num_nodes), device=encoder_input.device)

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
    Visualize a causal graph, using the appropriate visualization function.
    
    Args:
        graph: CausalGraph or other graph representation
        ax: Matplotlib axes to plot on
        title: Plot title
        figsize: Figure size
        layout: Graph layout algorithm
        highlight_nodes: List of nodes to highlight
        save_path: Path to save the figure
        
    Returns:
        Matplotlib axes with the plot
    """
    # Create axes if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    
    # Process highlight nodes to ensure consistent format
    if highlight_nodes:
        highlight_nodes = [get_node_name(node) for node in highlight_nodes]
    
    try:
        # Try to use the appropriate visualization function based on graph type
        if plot_graph is not None and hasattr(graph, 'get_adjacency_matrix'):
            plot_graph(graph, ax=ax, title=title, layout=layout, 
                      highlight_nodes=highlight_nodes)
        elif hasattr(graph, 'plot') and callable(graph.plot):
            # Some graph objects might have their own plot method
            graph.plot(ax=ax, title=title)
        else:
            # Fallback to basic visualization
            adj_matrix = getattr(graph, 'get_adjacency_matrix', lambda: getattr(graph, 'adj_matrix', None))()
            if adj_matrix is None:
                logger.error("Cannot visualize graph: no adjacency matrix available")
                ax.text(0.5, 0.5, "Graph visualization not available", 
                       horizontalalignment='center', verticalalignment='center')
            else:
                # Use NetworkX for visualization
                import networkx as nx
                
                # Create NetworkX graph
                G = nx.DiGraph()
                
                # Get node names
                node_names = getattr(graph, 'get_nodes', lambda: [f"x{i}" for i in range(adj_matrix.shape[0])])()
                
                # Add nodes
                G.add_nodes_from(node_names)
                
                # Add edges
                for i in range(adj_matrix.shape[0]):
                    for j in range(adj_matrix.shape[0]):
                        if adj_matrix[i, j] > 0:
                            G.add_edge(node_names[i], node_names[j])
                
                # Determine node colors
                node_colors = ['lightblue' for _ in node_names]
                if highlight_nodes:
                    for i, node in enumerate(node_names):
                        if node in highlight_nodes:
                            node_colors[i] = 'orange'
                
                # Draw the graph
                pos = nx.spring_layout(G) if layout == 'spring' else nx.circular_layout(G)
                nx.draw(G, pos, with_labels=True, node_color=node_colors, 
                        node_size=800, font_size=10, ax=ax)
                
                # Add title if provided
                if title:
                    ax.set_title(title)
    except Exception as e:
        logger.error(f"Error visualizing graph: {e}")
        ax.text(0.5, 0.5, f"Error visualizing graph: {str(e)}", 
               horizontalalignment='center', verticalalignment='center')
    
    # Save the figure if a path is provided
    if save_path:
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved graph visualization to {save_path}")
        except Exception as e:
            logger.error(f"Error saving graph visualization: {e}")
    
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
    Compare two graphs side by side with improved visualization.
    
    Args:
        left_graph: First graph to compare
        right_graph: Second graph to compare
        left_title: Title for left graph
        right_title: Title for right graph
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure with the comparison
    """
    # Create figure and axes
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    
    # Visualize left graph
    visualize_graph(left_graph, ax=axs[0], title=left_title)
    
    # Visualize right graph
    visualize_graph(right_graph, ax=axs[1], title=right_title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved graph comparison to {save_path}")
        except Exception as e:
            logger.error(f"Error saving graph comparison: {e}")
    
    return fig

class DummyGraph:
    """A simple dummy graph class for when CausalGraph has issues."""
    
    def __init__(self, adjacency_matrix):
        """
        Initialize a dummy graph from an adjacency matrix.
        
        Args:
            adjacency_matrix: Adjacency matrix as a numpy array
        """
        self.adjacency_matrix = adjacency_matrix
        if isinstance(adjacency_matrix, torch.Tensor):
            self.adjacency_matrix = adjacency_matrix.detach().cpu().numpy()
        
    def get_adjacency_matrix(self):
        """Return the adjacency matrix."""
        return self.adjacency_matrix
        
    def get_nodes(self):
        """Return the node names."""
        return [f"x{i}" for i in range(self.adjacency_matrix.shape[0])]
        
    def get_edges(self):
        """Return the edges as a list of tuples."""
        edges = []
        for i in range(self.adjacency_matrix.shape[0]):
            for j in range(self.adjacency_matrix.shape[1]):
                if self.adjacency_matrix[i, j] > 0.5:  # Threshold for edge existence
                    edges.append((f"x{i}", f"x{j}"))
        return edges

def convert_to_structural_equation_model(
    graph: Any, 
    node_names: Optional[List[str]] = None,
    noise_scale: float = 0.5
) -> Any:
    """
    Convert a graph to a StructuralCausalModel with proper structural equations.
    
    Args:
        graph: CausalGraph or other graph representation
        node_names: List of node names (default: use graph.get_nodes())
        noise_scale: Scale of noise for structural equations
        
    Returns:
        StructuralCausalModel or a fallback SCM implementation
    """
    # Extract adjacency matrix and node names
    if isinstance(graph, np.ndarray):
        adj_matrix = graph
        node_names = node_names or [f"x{i}" for i in range(adj_matrix.shape[0])]
    elif isinstance(graph, torch.Tensor):
        adj_matrix = graph.detach().cpu().numpy()
        node_names = node_names or [f"x{i}" for i in range(adj_matrix.shape[0])]
    elif hasattr(graph, 'get_adjacency_matrix'):
        adj_matrix = graph.get_adjacency_matrix()
        node_names = node_names or (graph.get_nodes() if hasattr(graph, 'get_nodes') else [f"x{i}" for i in range(adj_matrix.shape[0])])
    else:
        logger.error("Unsupported graph type for SCM conversion")
        return None
    
    # Create structural equations and noise functions
    structural_equations = {}
    noise_functions = {}
    
    # Create random noise function
    def create_noise_function(scale=noise_scale):
        def noise_func(sample_size, random_state=None):
            if random_state is not None:
                np.random.seed(random_state)
            return np.random.normal(0, scale, size=sample_size)
        return noise_func
    
    # Check if StructuralCausalModel is available
    if StructuralCausalModel is not None:
        try:
            # Create a causal graph if needed
            if not isinstance(graph, CausalGraph) and CausalGraph is not None:
                cg = CausalGraph()
                for node in node_names:
                    cg.add_node(node)
                
                for i, parent in enumerate(node_names):
                    for j, child in enumerate(node_names):
                        if adj_matrix[i, j] > 0:
                            cg.add_edge(parent, child)
                
                causal_graph = cg
            else:
                causal_graph = graph
            
            # Create equations for each node
            for i, node in enumerate(node_names):
                # Get parent indices
                parent_indices = [j for j in range(len(node_names)) if adj_matrix[j, i] > 0]
                parent_nodes = [node_names[j] for j in parent_indices]
                
                # Create equation
                structural_equations[node] = create_structural_equation(node, parent_nodes)
                noise_functions[node] = create_noise_function()
            
            # Create SCM
            scm = StructuralCausalModel(
                causal_graph=causal_graph,
                structural_equations=structural_equations,
                noise_functions=noise_functions
            )
            
            return scm
            
        except Exception as e:
            logger.error(f"Error creating StructuralCausalModel: {e}")
            logger.info("Falling back to DummySCM implementation")
    
    # Fallback to DummySCM if StructuralCausalModel is not available
    return DummySCM(graph, node_names)

def create_structural_equation(node, parents):
    """
    Create a linear structural equation with random coefficients.
    
    Args:
        node: Target node name
        parents: List of parent node names
        
    Returns:
        equation_function: Function that implements the linear equation
    """
    # Generate random coefficients
    coefficients = {parent: np.random.uniform(-1.0, 1.0) for parent in parents}
    
    # Create a closure that captures the coefficients
    def equation_function(values, noise):
        """Linear structural equation with random coefficients."""
        # Start with the noise term
        result = noise
        
        # Add contribution from each parent
        for parent, coef in coefficients.items():
            if parent in values:
                result += coef * values[parent]
        
        return result
    
    return equation_function

def select_intervention_target_by_parent_count(graph: Any) -> str:
    """
    Select a node to intervene on based on parent count (choose node with most parents).
    
    Args:
        graph: CausalGraph or other graph representation
        
    Returns:
        Selected node name
    """
    # Check for proper graph object
    if graph is None:
        raise ValueError("Graph cannot be None")
    
    # Get nodes
    if hasattr(graph, 'get_nodes'):
        nodes = graph.get_nodes()
    elif hasattr(graph, 'node_names'):
        nodes = graph.node_names
    else:
        try:
            # Try to get adjacency matrix and infer nodes
            adj_matrix = graph.get_adjacency_matrix()
            nodes = [f"x{i}" for i in range(adj_matrix.shape[0])]
        except:
            raise ValueError("Cannot determine nodes in the graph")
    
    # Get parent counts for each node
    parent_counts = {}
    for node in nodes:
        if hasattr(graph, 'get_parents'):
            parents = graph.get_parents(node)
            parent_counts[node] = len(parents)
        elif hasattr(graph, 'adj_matrix'):
            # Use adjacency matrix to find parents
            idx = None
            if isinstance(node, str) and node.startswith('x'):
                try:
                    idx = int(node[1:])
                except:
                    idx = nodes.index(node) if node in nodes else None
            else:
                idx = nodes.index(node) if node in nodes else None
            
            if idx is not None:
                parents = sum(graph.adj_matrix[:, idx])
                parent_counts[node] = parents
        else:
            # Default to 0 parents if can't determine
            parent_counts[node] = 0
    
    # Select node with the most parents
    if not parent_counts:
        # Fallback to random selection
        return np.random.choice(nodes)
    
    max_parents = max(parent_counts.values())
    candidates = [node for node, count in parent_counts.items() if count == max_parents]
    
    # If multiple candidates, choose one randomly
    return np.random.choice(candidates)

class DummySCM:
    """Fallback implementation when StructuralCausalModel is not available."""
    
    def __init__(self, graph, node_names=None):
        """Initialize with graph and node names."""
        self.graph = graph
        
        # Extract adjacency matrix
        if hasattr(graph, 'get_adjacency_matrix'):
            self.adj_matrix = graph.get_adjacency_matrix()
        elif hasattr(graph, 'adj_matrix'):
            self.adj_matrix = graph.adj_matrix
        elif isinstance(graph, (np.ndarray, torch.Tensor)):
            self.adj_matrix = graph.detach().cpu().numpy() if isinstance(graph, torch.Tensor) else graph
        else:
            raise ValueError("Graph must have get_adjacency_matrix method or adj_matrix attribute")
        
        # Extract node names
        if node_names is not None:
            self.node_names = node_names
        elif hasattr(graph, 'get_nodes'):
            self.node_names = graph.get_nodes()
        elif hasattr(graph, 'node_names'):
            self.node_names = graph.node_names
        else:
            self.node_names = [f"x{i}" for i in range(self.adj_matrix.shape[0])]
        
        # Create structural equations with random coefficients
        self.equations = self._create_structural_equations()
    
    def get_causal_graph(self):
        """Return the causal graph."""
        return self.graph
    
    def get_adjacency_matrix(self):
        """Return the adjacency matrix."""
        return self.adj_matrix
    
    def _create_structural_equations(self):
        """Create structural equations with random coefficients."""
        equations = {}
        num_nodes = len(self.node_names)
        
        for i, node in enumerate(self.node_names):
            # Get parent indices
            parent_indices = np.where(self.adj_matrix[:, i] > 0)[0]
            parents = [self.node_names[j] for j in parent_indices]
            
            # Create coefficients
            coefficients = {parent: np.random.uniform(-1.0, 1.0) for parent in parents}
            equations[node] = (parents, coefficients)
        
        return equations
    
    def sample_data(self, sample_size=100):
        """Generate synthetic data according to the causal graph."""
        num_nodes = len(self.node_names)
        data = np.zeros((sample_size, num_nodes))
        
        # Topological sort
        visited = set()
        temp = set()
        order = []
        
        def dfs(node_idx):
            if node_idx in visited:
                return
            if node_idx in temp:
                # Cycle detected
                return
            
            temp.add(node_idx)
            
            # Visit parents
            for i in range(num_nodes):
                if self.adj_matrix[i, node_idx] > 0:
                    dfs(i)
            
            temp.remove(node_idx)
            visited.add(node_idx)
            order.append(node_idx)
        
        for i in range(num_nodes):
            if i not in visited:
                dfs(i)
        
        # Generate data in topological order
        order.reverse()  # We need to start from sources
        
        for node_idx in order:
            node = self.node_names[node_idx]
            parents, coefficients = self.equations[node]
            
            # Noise term
            data[:, node_idx] = np.random.normal(0, 0.5, size=sample_size)
            
            # Add parent contributions
            for parent in parents:
                parent_idx = self.node_names.index(parent)
                data[:, node_idx] += coefficients[parent] * data[:, parent_idx]
        
        return data
    
    def sample_interventional_data(self, interventions, sample_size=100):
        """Generate interventional data with the specified interventions."""
        # Start with observational data
        data = self.sample_data(sample_size)
        
        # Apply interventions
        for node, value in interventions.items():
            # Convert node to index
            if isinstance(node, str):
                try:
                    node_idx = self.node_names.index(node)
                except ValueError:
                    # Try extracting index from node name
                    if node.startswith('x'):
                        try:
                            node_idx = int(node[1:])
                        except:
                            logger.warning(f"Invalid node name: {node}")
                            continue
                    else:
                        logger.warning(f"Invalid node name: {node}")
                        continue
            else:
                node_idx = node
            
            # Set the value
            data[:, node_idx] = value
        
        return data

def fallback_plot_graph(adj_matrix, ax=None, title=None, layout='spring', highlight_nodes=None):
    """
    Fallback function to plot a graph from an adjacency matrix.
    
    Args:
        adj_matrix: Adjacency matrix
        ax: Matplotlib axes to plot on
        title: Plot title
        layout: Graph layout algorithm
        highlight_nodes: List of nodes to highlight
        
    Returns:
        Matplotlib axes with the plot
    """
    import networkx as nx
    
    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    num_nodes = adj_matrix.shape[0]
    node_names = [f"x{i}" for i in range(num_nodes)]
    G.add_nodes_from(node_names)
    
    # Add edges
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] > 0:
                G.add_edge(node_names[i], node_names[j])
    
    # Determine node colors
    node_colors = ['lightblue' for _ in range(num_nodes)]
    if highlight_nodes:
        highlight_indices = []
        for node in highlight_nodes:
            if isinstance(node, str):
                if node.startswith('x'):
                    try:
                        idx = int(node[1:])
                        if 0 <= idx < num_nodes:
                            highlight_indices.append(idx)
                    except:
                        pass
            elif isinstance(node, int) and 0 <= node < num_nodes:
                highlight_indices.append(node)
        
        for idx in highlight_indices:
            node_colors[idx] = 'orange'
    
    # Draw the graph
    if layout == 'spring':
        pos = nx.spring_layout(G)
    else:
        pos = nx.circular_layout(G)
    
    nx.draw(G, pos, with_labels=True, node_color=node_colors, 
            node_size=800, font_size=10, ax=ax)
    
    # Add title if provided
    if title:
        ax.set_title(title)
    
    return ax

def calculate_structural_hamming_distance(graph1, graph2):
    """
    Calculate the Structural Hamming Distance (SHD) between two graphs.
    
    Args:
        graph1: First graph (CausalGraph or similar)
        graph2: Second graph (CausalGraph or similar)
        
    Returns:
        SHD value (int)
    """
    # Extract adjacency matrices
    if hasattr(graph1, 'get_adjacency_matrix'):
        adj1 = graph1.get_adjacency_matrix()
    elif hasattr(graph1, 'adj_matrix'):
        adj1 = graph1.adj_matrix
    else:
        raise ValueError("graph1 must have get_adjacency_matrix method or adj_matrix attribute")
        
    if hasattr(graph2, 'get_adjacency_matrix'):
        adj2 = graph2.get_adjacency_matrix()
    elif hasattr(graph2, 'adj_matrix'):
        adj2 = graph2.adj_matrix
    else:
        raise ValueError("graph2 must have get_adjacency_matrix method or adj_matrix attribute")
    
    # Convert to binary matrices if needed
    adj1_binary = (adj1 > 0.5).astype(int)
    adj2_binary = (adj2 > 0.5).astype(int)
    
    # Calculate SHD (missing edges + extra edges)
    diff = np.abs(adj1_binary - adj2_binary)
    shd = np.sum(diff)
    
    return shd

def structural_hamming_distance(graph1, graph2):
    """
    Calculate the structural hamming distance between two graphs.
    
    Works with both CausalGraph instances and DummyGraph instances.
    
    Args:
        graph1: First graph
        graph2: Second graph
        
    Returns:
        Structural hamming distance
    """
    # Extract adjacency matrices
    if hasattr(graph1, 'get_adjacency_matrix'):
        adj1 = graph1.get_adjacency_matrix()
    elif hasattr(graph1, 'adjacency_matrix'):
        adj1 = graph1.adjacency_matrix
    else:
        raise ValueError("graph1 must have get_adjacency_matrix method or adjacency_matrix attribute")
        
    if hasattr(graph2, 'get_adjacency_matrix'):
        adj2 = graph2.get_adjacency_matrix()
    elif hasattr(graph2, 'adjacency_matrix'):
        adj2 = graph2.adjacency_matrix
    else:
        raise ValueError("graph2 must have get_adjacency_matrix method or adjacency_matrix attribute")
    
    # Convert to numpy arrays if they are tensors
    if isinstance(adj1, torch.Tensor):
        adj1 = adj1.detach().cpu().numpy()
    if isinstance(adj2, torch.Tensor):
        adj2 = adj2.detach().cpu().numpy()
    
    # Ensure boolean arrays for comparison
    if adj1.dtype != bool:
        adj1 = adj1 > 0.5
    if adj2.dtype != bool:
        adj2 = adj2 > 0.5
        
    # Calculate SHD (count of differing entries in the adjacency matrices)
    return np.sum(adj1 != adj2)

def visualize_graph_comparison(graph1, graph2, title="Graph Comparison"):
    """
    Visualize a comparison between two graphs.
    
    Args:
        graph1: First graph (typically the true graph)
        graph2: Second graph (typically the inferred graph)
        title: Title for the figure
    """
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Extract adjacency matrices
    if hasattr(graph1, 'get_adjacency_matrix'):
        adj1 = graph1.get_adjacency_matrix()
    elif hasattr(graph1, 'adjacency_matrix'):
        adj1 = graph1.adjacency_matrix
    else:
        raise ValueError("graph1 must have get_adjacency_matrix method or adjacency_matrix attribute")
        
    if hasattr(graph2, 'get_adjacency_matrix'):
        adj2 = graph2.get_adjacency_matrix()
    elif hasattr(graph2, 'adjacency_matrix'):
        adj2 = graph2.adjacency_matrix
    else:
        raise ValueError("graph2 must have get_adjacency_matrix method or adjacency_matrix attribute")
    
    # Convert to numpy arrays if they are tensors
    if isinstance(adj1, torch.Tensor):
        adj1 = adj1.detach().cpu().numpy()
    if isinstance(adj2, torch.Tensor):
        adj2 = adj2.detach().cpu().numpy()
    
    # Get number of nodes
    num_nodes = adj1.shape[0]
    
    # Get node names
    if hasattr(graph1, 'get_nodes'):
        node_names = graph1.get_nodes()
    else:
        node_names = [f"x{i}" for i in range(num_nodes)]
    
    # Create networkx graphs
    G1 = nx.DiGraph()
    G2 = nx.DiGraph()
    
    # Add nodes
    for i in range(num_nodes):
        G1.add_node(i, name=node_names[i])
        G2.add_node(i, name=node_names[i])
    
    # Add edges with threshold
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj1[i, j] > 0.5:
                G1.add_edge(i, j)
            if adj2[i, j] > 0.5:
                G2.add_edge(i, j)
    
    # Create layouts
    pos = nx.spring_layout(G1, seed=42)
    
    # Draw first graph
    nx.draw_networkx_nodes(G1, pos, ax=axes[0], node_color='lightblue', 
                         node_size=500, alpha=0.8)
    nx.draw_networkx_edges(G1, pos, ax=axes[0], arrowstyle='->', 
                         arrowsize=15, width=1.5, edge_color='blue')
    nx.draw_networkx_labels(G1, pos, ax=axes[0], 
                         labels={i: G1.nodes[i]['name'] for i in G1.nodes})
    axes[0].set_title("True Graph")
    axes[0].axis('off')
    
    # Draw second graph
    nx.draw_networkx_nodes(G2, pos, ax=axes[1], node_color='lightgreen', 
                         node_size=500, alpha=0.8)
    nx.draw_networkx_edges(G2, pos, ax=axes[1], arrowstyle='->', 
                         arrowsize=15, width=1.5, edge_color='green')
    nx.draw_networkx_labels(G2, pos, ax=axes[1], 
                         labels={i: G2.nodes[i]['name'] for i in G2.nodes})
    axes[1].set_title("Inferred Graph")
    axes[1].axis('off')
    
    # Add global title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig

# Initialize logging when the module is imported
logger.info("Refactored utilities module loaded successfully")

def test_get_node_name_and_id():
    # Integer input
    assert get_node_name(0) == 'x0'
    assert get_node_name(5) == 'x5'
    # String input (int)
    assert get_node_name('2') == 'x2'
    # Already correct
    assert get_node_name('x3') == 'x3'
    # get_node_id
    assert get_node_id('x0') == 0
    assert get_node_id('x12') == 12
    assert get_node_id(7) == 7
    # Error on bad input
    try:
        get_node_id('foo')
        assert False, 'Should raise ValueError'
    except ValueError:
        pass

def test_format_interventions():
    # Dict with int keys
    interventions = {0: 1.0, 1: 2.0}
    formatted = format_interventions(interventions)
    assert formatted == {'x0': 1.0, 'x1': 2.0, 0: 1.0, 1: 2.0}
    # Dict with string keys
    interventions = {'x0': 1.0, 'x1': 2.0}
    formatted = format_interventions(interventions)
    assert formatted == {'x0': 1.0, 'x1': 2.0, 0: 1.0, 1: 2.0}
    # Tensor format
    import torch
    mask = format_interventions({'x0': 1.0}, for_tensor=True, num_nodes=2)
    assert mask[0, 0] == 1.0
    assert mask[0, 1] == 0.0 