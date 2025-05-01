#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Parent-Scaled ACD Demo with Neural Networks

This script demonstrates how to use neural networks as a drop-in replacement for 
traditional surrogate models in the Parent-Scaled ACD algorithm. It uses pre-trained
models for quick inference without extensive training time.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import networkx as nx

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utilities
from demos.utils import (
    get_assets_dir, 
    get_checkpoints_dir,
    standardize_tensor_shape, 
    get_node_name,
    get_node_id,
    format_interventions
)

# Import from causal_meta (required, no fallbacks)
from causal_meta.graph.generators.factory import GraphFactory
from causal_meta.environments.scm import StructuralCausalModel
from causal_meta.graph.visualization import plot_graph
from causal_meta.meta_learning.acd_models import GraphEncoder
from causal_meta.meta_learning.dynamics_decoder import DynamicsDecoder
from causal_meta.meta_learning.amortized_causal_discovery import AmortizedCausalDiscovery
from causal_meta.graph.causal_graph import CausalGraph

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Parent-Scaled ACD Demo')
    parser.add_argument('--num_nodes', type=int, default=5,
                        help='Number of nodes in the synthetic graph')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples for synthetic data generation')
    parser.add_argument('--max_interventions', type=int, default=3,
                        help='Maximum number of interventions to perform')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Enable visualization')
    parser.add_argument('--quick', action='store_true', 
                        help='Run in quick mode with minimal settings')
    parser.add_argument('--pretrained_model_path', type=str, 
                        default=os.path.join(get_checkpoints_dir(), 'acd_model.pt'),
                        help='Path to pretrained model')
    parser.add_argument('--use_full_algorithm', action='store_true',
                        help='Use the full PARENT_SCALE_ACD algorithm implementation')
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_pretrained_model(model_path, num_nodes, device):
    """
    Load a pretrained neural network model or create a new one.
    
    Args:
        model_path: Path to the pretrained model (or None to create new one)
        num_nodes: Number of nodes in the causal graph
        device: PyTorch device to load the model on
        
    Returns:
        model: AmortizedCausalDiscovery model
    """
    if model_path and os.path.exists(model_path):
        print(f"Loading pretrained model from {model_path}...")
        try:
            # Try loading the full model
            loaded = torch.load(model_path, map_location=device)
            
            # Check if loaded is a state_dict (OrderedDict) or a full model
            if isinstance(loaded, dict):
                print("Loaded state dictionary, creating model instance...")
                try:
                    # Create a new model instance
                    model = AmortizedCausalDiscovery(
                        hidden_dim=64,
                        input_dim=1,  # Single feature per node
                        attention_heads=2,
                        num_layers=2,
                        dropout=0.1,
                        sparsity_weight=0.1,
                        acyclicity_weight=1.0
                    )
                    # Load the state dictionary
                    model.load_state_dict(loaded, strict=False)
                    print("Model loaded with some missing or unexpected keys (non-strict loading).")
                except Exception as e:
                    print(f"Error loading model state dictionary: {e}")
                    print("Creating a new model instead.")
                    model = create_new_model(num_nodes, device)
            else:
                # Loaded the full model
                model = loaded
                
            model = model.to(device)
            model.eval()  # Set to evaluation mode
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating a new model instead.")
    else:
        if model_path:
            print(f"Model file not found at {model_path}")
        print("Creating a new neural network model...")
    
    return create_new_model(num_nodes, device)


def create_new_model(num_nodes, device):
    """
    Create a new neural network model for demonstration.
    
    Args:
        num_nodes: Number of nodes in the causal graph
        device: PyTorch device to create the model on
        
    Returns:
        model: AmortizedCausalDiscovery model
    """
    print("Creating a new neural network model for demonstration...")
    
    # Neural network size configuration
    hidden_dim = 64
    num_layers = 2
    attention_heads = 2
    
    # Use the proper AmortizedCausalDiscovery class
    model = AmortizedCausalDiscovery(
        hidden_dim=hidden_dim,
        input_dim=1,  # Single feature per node (can be changed for multivariate)
        attention_heads=attention_heads,
        num_layers=num_layers,
        dropout=0.1,
        sparsity_weight=0.1,
        acyclicity_weight=1.0
    )
    
    # Move model to device
    model = model.to(device)
    
    # Set to evaluation mode (important for batch norm, dropout, etc.)
    model.eval()
    
    return model


def generate_synthetic_data(num_nodes, num_samples, seed):
    """
    Generate synthetic causal graph and data for demonstration.
    
    Args:
        num_nodes: Number of nodes in the causal graph
        num_samples: Number of observational samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        graph: CausalGraph object
        scm: StructuralCausalModel object
        data: Numpy array of observational data
    """
    np.random.seed(seed)
    print(f"Generating synthetic graph with {num_nodes} nodes...")
    
    # Create node names as strings
    node_names = [f"X_{i}" for i in range(num_nodes)]
    
    # Create a synthetic graph using GraphFactory
    graph_factory = GraphFactory()
    
    # First create a graph with integer IDs
    temp_graph = graph_factory.create_random_dag(
        num_nodes=num_nodes,
        edge_probability=0.3,  # Moderate density
        seed=seed
    )
    
    # Create a new CausalGraph with string node names
    graph = CausalGraph()
    
    # Add nodes with string names
    for i, node in enumerate(node_names):
        graph.add_node(node)
    
    # Add edges, translating from integer nodes to string nodes
    for i, j in temp_graph.get_edges():
        graph.add_edge(node_names[i], node_names[j])
    
    print(f"Graph generated with {len(graph.get_edges())} edges")
    
    # Create SCM from graph with explicit variable_names
    scm = StructuralCausalModel(
        causal_graph=graph,
        variable_names=node_names  # Explicitly provide variable names
    )
    
    # Define structural equations for each node
    for node in node_names:
        parents = list(graph.get_parents(node))
        
        # Create a structural equation for this node
        equation = create_linear_equation(node, parents)
        
        # Define the structural equation with noise
        scm.define_probabilistic_equation(
            variable=node,
            equation_function=equation,
            noise_distribution=noise_function
        )
    
    # Generate observational data
    data = scm.sample_data(sample_size=num_samples)
    print(f"Generated {num_samples} observational samples")
    
    return graph, scm, data


def create_linear_equation(node, parents):
    """
    Create a linear structural equation with random coefficients.
    
    The equation is of the form Y = a1*X1 + a2*X2 + ... + noise, where a1, a2, etc.
    are random coefficients between 0.5 and 2.0.
    
    Args:
        node: The node for which to create an equation
        parents: List of parent nodes
        
    Returns:
        equation_function: A function that calculates the node value based on parents
    """
    # Set a seed based on the node name (for reproducibility)
    node_idx = int(node.split('_')[1]) if '_' in node else hash(node)
    np.random.seed(node_idx)
    
    # Create random coefficients for each parent
    coefficients = {parent: 0.5 + 1.5 * np.random.rand() for parent in parents}
    
    # Create a string representation of the function definition with parameters matching parents
    if parents:
        # If there are parents, include them in the signature
        params = ', '.join(parents) + ', noise=0.0'
    else:
        # If no parents, just have the noise parameter
        params = 'noise=0.0'
    
    func_def = f"def equation({params}):\n"
    func_def += "    result = 0.0\n"
    
    # Add contribution from each parent
    for parent, coef in coefficients.items():
        func_def += f"    result += {coef} * {parent}\n"
    
    # Add noise
    func_def += "    result += noise\n"
    func_def += "    return result"
    
    # Create a local namespace to execute the function definition
    local_namespace = {}
    exec(func_def, globals(), local_namespace)
    
    # Return the dynamically created function
    return local_namespace["equation"]


def noise_function(sample_size, random_state=None):
    """Generate normal noise for structural equations."""
    # Handle random state properly
    if random_state is None:
        # Use default numpy random state
        rng = np.random
    elif isinstance(random_state, np.random.RandomState):
        # Use the provided RandomState object
        rng = random_state
    else:
        # Create a new RandomState with the provided seed
        rng = np.random.RandomState(random_state)
    
    # Normal noise with random variance between 0.1 and 0.5
    variance = 0.1 + 0.4 * rng.rand()
    return rng.normal(0, np.sqrt(variance), size=sample_size)


def create_graph_from_adjacency(adj_matrix, node_names=None):
    """Create a proper CausalGraph object from an adjacency matrix."""
    # Create node list if not provided
    if node_names is None:
        node_names = [f"X_{i}" for i in range(adj_matrix.shape[0])]
    
    # Create edges from adjacency matrix
    edges = []
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[0]):
            if adj_matrix[i, j] > 0:
                edges.append((node_names[i], node_names[j]))
    
    # Create CausalGraph with nodes and edges
    graph = CausalGraph()
    for node in node_names:
        graph.add_node(node)
    for src, tgt in edges:
        graph.add_edge(src, tgt)
    
    return graph


def graph_to_networkx(graph):
    """
    Convert a CausalGraph to a NetworkX DiGraph for visualization.
    
    Args:
        graph: CausalGraph object
        
    Returns:
        nx.DiGraph: A NetworkX directed graph
    """
    # Create a new networkx directed graph
    nx_graph = nx.DiGraph()
    
    # Add nodes
    for node in graph.get_nodes():
        nx_graph.add_node(node)
    
    # Add edges
    for edge in graph.get_edges():
        source, target = edge
        nx_graph.add_edge(source, target)
    
    return nx_graph


def enhanced_plot_graph(graph, ax=None, title=None, figsize=(8, 6), is_inferred=False, is_true=False):
    """
    Enhanced plotting of a causal graph with directional edges and clear labels.
    
    Args:
        graph: CausalGraph object or NetworkX DiGraph
        ax: Matplotlib axes to plot on (optional)
        title: Plot title
        figsize: Figure size as (width, height)
        is_inferred: Whether this is an inferred graph (affects node color)
        is_true: Whether this is a ground truth graph (affects node color)
        
    Returns:
        ax: Matplotlib axes with the plot
    """
    # Convert CausalGraph to NetworkX if needed
    if hasattr(graph, 'get_nodes'):
        nx_graph = graph_to_networkx(graph)
    else:
        nx_graph = graph
    
    # Create figure and axes if not provided
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    
    # Determine node color based on graph type
    if is_true:
        node_color = 'lightgreen'  # Use green for ground truth
    elif is_inferred:
        node_color = 'lightblue'   # Use blue for inferred
    else:
        node_color = 'lightgray'   # Default gray for unspecified
    
    # Create layout
    pos = nx.spring_layout(nx_graph, seed=42)
    
    # Draw the graph with nice styling
    nx.draw_networkx(
        nx_graph,
        pos=pos,
        ax=ax,
        with_labels=True,
        node_color=node_color,
        node_size=500,
        arrowsize=20,
        font_size=12,
        font_weight='bold'
    )
    
    # Add title if provided
    if title:
        ax.set_title(title)
    
    # Remove axes
    ax.set_axis_off()
    
    return ax


def parent_scaled_acd(model, scm, obs_data, max_interventions, device, visualize=False):
    """
    Run the Parent-Scaled ACD algorithm with neural network models.
    
    This algorithm:
    1. Infers causal structure from observational data using neural networks
    2. Selects intervention targets based on parent-count metric
    3. Performs interventions and updates causal structure
    4. Repeats until max interventions reached
    
    Args:
        model: AmortizedCausalDiscovery model
        scm: StructuralCausalModel for generating intervention data
        obs_data: Observational data array or DataFrame
        max_interventions: Maximum number of interventions to perform
        device: PyTorch device for computation
        visualize: Whether to visualize results
        
    Returns:
        Final inferred causal graph and intervention history
    """
    print("\nRunning Parent-Scaled ACD algorithm with neural inference...")
    print(f"Maximum interventions: {max_interventions}")
    
    # Convert DataFrame to numpy if needed
    if hasattr(obs_data, 'to_numpy'):
        obs_data = obs_data.to_numpy()
    
    # Convert observational data to tensor if needed
    if isinstance(obs_data, np.ndarray):
        obs_data = torch.tensor(obs_data, dtype=torch.float32)
    
    # Move data to the specified device
    obs_data = obs_data.to(device)
    
    # Initial shape processing for the neural network
    data_for_encoder = standardize_tensor_shape(obs_data, for_encoder=True)
    
    # Extract dimensions
    batch_size, num_samples, num_nodes = data_for_encoder.shape
    print(f"Data shape: batch_size={batch_size}, samples={num_samples}, nodes={num_nodes}")
    
    # 1. Initial causal structure inference from observational data using neural network
    print("Inferring initial causal structure from observational data using neural networks...")
    
    # Use the model to infer the causal graph
    with torch.no_grad():
        # Check if the model has a specific infer_causal_graph method
        if hasattr(model, 'infer_causal_graph'):
            adj_matrix = model.infer_causal_graph(data_for_encoder)
        # Alternative: if model is just a GraphEncoder
        elif hasattr(model, 'forward'):
            adj_matrix = model.forward(data_for_encoder)
        else:
            raise TypeError("Model doesn't have infer_causal_graph method or forward method")
    
    # Convert to numpy for processing
    if isinstance(adj_matrix, torch.Tensor):
        adj_np = adj_matrix.detach().cpu().numpy()
    else:
        adj_np = adj_matrix
    
    # Threshold adjacency matrix to get binary edges
    threshold = 0.5
    binary_adj = (adj_np > threshold).astype(float)
    
    print(f"Initial adjacency matrix shape: {adj_np.shape}")
    
    # Create a proper CausalGraph object with string node names
    node_names = [f"X_{i}" for i in range(num_nodes)]
    inferred_graph = create_graph_from_adjacency(binary_adj, node_names)
    
    # Visualize initial structure if requested
    if visualize:
        # Plot individual initial graph
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        enhanced_plot_graph(inferred_graph, ax=ax, title="Initial Inferred Structure", is_inferred=True)
        plt.tight_layout()
        
        # Save the initial structure plot
        init_save_path = os.path.join(get_assets_dir(), "initial_graph.png")
        os.makedirs(os.path.dirname(init_save_path), exist_ok=True)
        plt.savefig(init_save_path)
        print(f"Saved initial graph visualization to {init_save_path}")
        plt.show()
        
        # Compare initial inference to ground truth
        compare_save_path = os.path.join(get_assets_dir(), "initial_graph_comparison.png")
        fig = plot_graph_comparison(
            left_graph=scm.get_causal_graph(),
            right_graph=inferred_graph,
            left_title="Ground Truth Graph",
            right_title="Initial Inferred Graph",
            figsize=(12, 6),
            save_path=compare_save_path
        )
        plt.show()
    
    # Store intervention history
    interventions_history = []
    
    # 2. Iterative intervention loop
    for i in range(max_interventions):
        print(f"\nIntervention {i+1}/{max_interventions}")
        
        # Select intervention target using parent count heuristic
        target_node = select_intervention_target_by_parent_count(inferred_graph, adj_matrix=adj_np)
        # Generate a random intervention value
        target_value = np.random.uniform(-2.0, 2.0)
        
        print(f"Intervening on node {target_node} with value {target_value:.2f}")
        
        # Record intervention
        interventions_history.append((target_node, target_value))
        
        # Perform intervention and collect data
        try:
            scm.do_intervention(target_node, target_value)
            int_data = scm.sample_data(sample_size=num_samples)
            scm.reset()  # Reset the SCM
        except Exception as e:
            print(f"Error performing intervention: {e}")
            print("Skipping to next intervention")
            continue
        
        # Convert DataFrame to numpy if needed
        if hasattr(int_data, 'to_numpy'):
            int_data = int_data.to_numpy()
        
        # Convert to tensor if needed
        if isinstance(int_data, np.ndarray):
            int_data = torch.tensor(int_data, dtype=torch.float32)
        
        # Move to device
        int_data = int_data.to(device)
        
        # Prepare data for the neural network
        int_data_for_encoder = standardize_tensor_shape(int_data, for_encoder=True)
        
        # Format interventions for the model
        # Extract node index from string (e.g., "X_0" -> 0)
        try:
            target_idx = int(target_node.split('_')[1])
            formatted_interventions = format_interventions(
                [(target_idx, target_value)], 
                num_nodes=num_nodes, 
                device=device
            )
        except (ValueError, IndexError) as e:
            print(f"Error formatting intervention: {e}")
            target_idx = 0  # Default to first node
            formatted_interventions = format_interventions(
                [(target_idx, target_value)], 
                num_nodes=num_nodes, 
                device=device
            )
        
        # Update causal graph with the new interventional data using neural network
        print("Updating causal graph with interventional data...")
        
        # Use neural network for graph inference with interventional data
        with torch.no_grad():
            try:
                # Check model capabilities and type
                if hasattr(model, 'infer_causal_graph'):
                    # Try with interventions parameter if the method supports it
                    import inspect
                    sig = inspect.signature(model.infer_causal_graph)
                    if 'interventions' in sig.parameters:
                        updated_adj_matrix = model.infer_causal_graph(
                            int_data_for_encoder, 
                            interventions=formatted_interventions
                        )
                    else:
                        print("Model does not support interventions parameter, using data only.")
                        updated_adj_matrix = model.infer_causal_graph(int_data_for_encoder)
                else:
                    # Fallback for simpler models
                    print("Using model's forward method (no intervention support).")
                    updated_adj_matrix = model(int_data_for_encoder)
            except Exception as e:
                print(f"Error during causal graph inference: {e}")
                print("Using previous adjacency matrix.")
                updated_adj_matrix = adj_matrix
        
        # Convert to numpy for processing
        if isinstance(updated_adj_matrix, torch.Tensor):
            updated_adj_np = updated_adj_matrix.detach().cpu().numpy()
        else:
            updated_adj_np = updated_adj_matrix
        
        # Threshold to get binary adjacency matrix
        updated_binary_adj = (updated_adj_np > threshold).astype(float)
        
        # Create updated graph with string node names
        updated_graph = create_graph_from_adjacency(updated_binary_adj, node_names)
        
        # Visualize intermediate results if requested
        if visualize:
            # Plot individual updated graph
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
            enhanced_plot_graph(
                updated_graph, 
                ax=ax, 
                title=f"Inferred Graph Structure After Intervention {i+1} (on node {target_node})",
                is_inferred=True
            )
            plt.tight_layout()
            # Save the intermediate plot
            int_save_path = os.path.join(get_assets_dir(), f"graph_after_intervention_{i+1}.png")
            os.makedirs(os.path.dirname(int_save_path), exist_ok=True)
            plt.savefig(int_save_path)
            print(f"Saved intervention {i+1} graph visualization to {int_save_path}")
            plt.show()
            
            # Also show a before/after comparison
            compare_save_path = os.path.join(get_assets_dir(), f"intervention_{i+1}_comparison.png")
            # Use the previous version of the graph for before/after comparison
            previous_graph = inferred_graph if i == 0 else create_graph_from_adjacency(
                (adj_np > threshold).astype(float), node_names
            )
            
            fig = plot_graph_comparison(
                left_graph=previous_graph,
                right_graph=updated_graph,
                left_title=f"Inferred Graph Before Intervention {i+1}",
                right_title=f"Inferred Graph After Intervention {i+1} (on {target_node})",
                figsize=(12, 6),
                save_path=compare_save_path
            )
            plt.show()
        
        # Update current graph and adjacency matrix for next iteration
        inferred_graph = updated_graph
        adj_np = updated_adj_np
    
    print("\nParent-Scaled ACD completed.")
    
    # Final visualization comparing ground truth (from SCM) to our inferred graph
    if visualize:
        # Use the comparison plotting function for better visualization
        final_save_path = os.path.join(get_assets_dir(), "parent_scaled_acd_results.png")
        
        # Create comparison between ground truth and final inferred graph
        fig = plot_graph_comparison(
            left_graph=scm.get_causal_graph(),
            right_graph=inferred_graph,
            left_title="Ground Truth Causal Graph",
            right_title=f"Final Inferred Graph (After {max_interventions} Interventions)",
            figsize=(12, 6),
            save_path=final_save_path
        )
        plt.suptitle(f"Final Results After {max_interventions} Interventions", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the super title
        plt.show()
        print(f"Saved final comparison to {final_save_path}")
    
    return inferred_graph, interventions_history


def select_intervention_target_by_parent_count(graph, adj_matrix=None):
    """
    Select an intervention target based on parent count.
    
    This function selects nodes with more parents with higher probability.
    The intuition is that nodes with more parents have more complex causal
    relationships that might benefit from direct intervention.
    
    Args:
        graph: CausalGraph object
        adj_matrix: Optional adjacency matrix if not extracted from graph
        
    Returns:
        selected_node: The selected intervention target node name
    """
    # Extract adjacency matrix if not provided
    if adj_matrix is None:
        adj_matrix = graph.get_adjacency_matrix()
    
    # Ensure adj_matrix is numpy array
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.detach().cpu().numpy()
    
    # Get number of nodes
    num_nodes = len(graph.get_nodes())
    
    # Count incoming edges (parents) for each node using the adjacency matrix
    parent_counts = np.zeros(num_nodes)
    
    # Get the list of nodes
    nodes = list(graph.get_nodes())
    
    # Count parents for each node
    for i, node in enumerate(nodes):
        parents = graph.get_parents(node)
        parent_counts[i] = len(parents)
    
    # Use parent counts as weights, but avoid zeros
    # We add a small constant to ensure all nodes have some chance of selection
    weights = parent_counts + 0.1
    
    # Normalize weights to get probabilities
    probabilities = weights / np.sum(weights)
    
    # Select a node based on these probabilities
    selected_idx = np.random.choice(num_nodes, p=probabilities)
    selected_node = nodes[selected_idx]
    
    print(f"Selected node {selected_node} with {parent_counts[selected_idx]:.0f} parents")
    return selected_node


def format_interventions(interventions, num_nodes, device):
    """
    Format interventions for the neural network model.
    
    Args:
        interventions: List of (node, value) tuples where node can be name or index
        num_nodes: Number of nodes in the graph
        device: PyTorch device
        
    Returns:
        Formatted interventions tensor in the shape [batch_size, num_nodes, 2]
        where values are (is_intervened, intervention_value)
    """
    # Initialize intervention tensor: [batch_size=1, num_nodes, 2]
    formatted = torch.zeros((1, num_nodes, 2), device=device)
    
    # For each intervention target and value
    for node, value in interventions:
        # If node is a string, extract index from it (e.g., "X_0" -> 0)
        if isinstance(node, str) and '_' in node:
            try:
                node_idx = int(node.split('_')[1])
            except (ValueError, IndexError):
                print(f"Warning: Could not extract index from node name {node}, skipping")
                continue
        else:
            # Otherwise, node is already an index
            node_idx = node
        
        # Set is_intervened flag to 1.0
        formatted[0, node_idx, 0] = 1.0
        # Set intervention value
        formatted[0, node_idx, 1] = float(value)
    
    return formatted


def standardize_tensor_shape(tensor, for_encoder=False):
    """
    Standardize tensor shape for neural network processing.
    
    Args:
        tensor: Input tensor [batch_size, seq_length, num_nodes]
        for_encoder: If True, prepare for GraphEncoder, else for DynamicsDecoder
        
    Returns:
        Tensor with appropriate shape
    """
    # Make sure it's a tensor
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, dtype=torch.float32)
    
    # Extract dimensions
    shape = tensor.shape
    
    # Handle different input shapes
    if len(shape) == 2:
        # [samples, nodes] -> [1, samples, nodes]
        tensor = tensor.unsqueeze(0)
    elif len(shape) != 3:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {shape}")
    
    # For graph encoder the shape should be [batch_size, seq_length, num_nodes]
    if for_encoder:
        return tensor
    
    # For dynamics decoder, may need additional processing
    # E.g., reshaping, adding features, etc.
    return tensor


def plot_graph_comparison(left_graph, right_graph, left_title="Left Graph", right_title="Right Graph", figsize=(12, 6), save_path=None):
    """
    Plot a side-by-side comparison of two graphs with consistent node positions.
    
    Args:
        left_graph: The graph to display on the left panel
        right_graph: The graph to display on the right panel
        left_title: Title for the left graph
        right_title: Title for the right graph
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
        
    Returns:
        fig: Matplotlib figure
    """
    # Convert both graphs to NetworkX format
    if hasattr(left_graph, 'get_nodes'):
        nx_left = graph_to_networkx(left_graph)
    else:
        nx_left = left_graph
        
    if hasattr(right_graph, 'get_nodes'):
        nx_right = graph_to_networkx(right_graph)
    else:
        nx_right = right_graph
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Get all unique nodes from both graphs
    all_nodes = set(nx_left.nodes()).union(set(nx_right.nodes()))
    
    # Create a combined graph for layout calculation to ensure consistent positions
    combined = nx.DiGraph()
    combined.add_nodes_from(all_nodes)
    combined.add_edges_from(nx_left.edges())
    combined.add_edges_from(nx_right.edges())
    
    # Generate a consistent layout for both graphs
    pos = nx.spring_layout(combined, seed=42)
    
    # Plot left graph
    nx.draw_networkx(
        nx_left,
        pos=pos,
        ax=axes[0],
        with_labels=True,
        node_color='lightgreen',
        node_size=500,
        arrowsize=20,
        font_size=12,
        font_weight='bold'
    )
    axes[0].set_title(left_title)
    axes[0].axis('off')
    
    # Plot right graph
    nx.draw_networkx(
        nx_right,
        pos=pos,
        ax=axes[1],
        with_labels=True,
        node_color='lightblue',
        node_size=500,
        arrowsize=20,
        font_size=12,
        font_weight='bold'
    )
    axes[1].set_title(right_title)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"Saved graph comparison to {save_path}")
    
    return fig


def main():
    """Run the Parent-Scaled ACD demo with neural networks for inference."""
    # Parse arguments
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set device (CPU/GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate synthetic data
    graph, scm, obs_data = generate_synthetic_data(
        num_nodes=args.num_nodes,
        num_samples=args.num_samples,
        seed=args.seed
    )
    
    # Load or create model
    model = load_pretrained_model(
        model_path=args.pretrained_model_path,
        num_nodes=args.num_nodes,
        device=device
    )
    
    # Run in quick mode with fewer samples if requested
    if args.quick:
        print("Running in quick mode with minimal settings")
        args.num_samples = min(args.num_samples, 50)
        args.max_interventions = min(args.max_interventions, 2)
        obs_data = obs_data[:args.num_samples]
    
    # Run parent-scaled ACD algorithm
    # First, visualize the true graph to provide context
    if args.visualize:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        enhanced_plot_graph(
            scm.get_causal_graph(), 
            ax=ax, 
            title="True Underlying Causal Graph",
            is_true=True
        )
        plt.tight_layout()
        
        # Save the true graph plot
        true_graph_path = os.path.join(get_assets_dir(), "true_causal_graph.png")
        os.makedirs(os.path.dirname(true_graph_path), exist_ok=True)
        plt.savefig(true_graph_path)
        print(f"Saved true causal graph visualization to {true_graph_path}")
        plt.show()
    
    # Now run the actual ACD algorithm
    inferred_graph, intervention_history = parent_scaled_acd(
        model=model,
        scm=scm,
        obs_data=obs_data,
        max_interventions=args.max_interventions,
        device=device,
        visualize=args.visualize
    )
    
    # Print summary
    print("\nParent-Scaled ACD Demo Summary:")
    print(f"Number of nodes: {args.num_nodes}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Number of interventions: {args.max_interventions}")
    
    # Print intervention history
    print("\nIntervention History:")
    for i, intervention in enumerate(intervention_history):
        print(f"Intervention {i+1}: {intervention}")
    
    return inferred_graph, intervention_history


if __name__ == "__main__":
    main() 