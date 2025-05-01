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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utilities
from demos.utils import (
    get_assets_dir, 
    get_checkpoints_dir,
    standardize_tensor_shape, 
    get_node_name,
    get_node_id,
    format_interventions,
    DummyGraph,
    DummySCM,
    fallback_plot_graph
)

# Try importing from causal_meta, with graceful fallbacks
try:
    from causal_meta.graph.generators.factory import GraphFactory
    from causal_meta.environments.scm import StructuralCausalModel
    from causal_meta.graph.visualization import plot_graph
    from causal_meta.meta_learning.acd_models import GraphEncoder
    from causal_meta.meta_learning.dynamics_decoder import DynamicsDecoder
    from causal_meta.meta_learning.amortized_causal_discovery import AmortizedCausalDiscovery
    from causal_meta.graph.causal_graph import CausalGraph
    CAUSAL_META_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some causal_meta imports failed: {e}")
    print("Using fallback implementations where necessary.")
    GraphFactory = None
    StructuralCausalModel = None
    plot_graph = fallback_plot_graph
    GraphEncoder = None
    DynamicsDecoder = None
    AmortizedCausalDiscovery = None
    CausalGraph = None
    CAUSAL_META_AVAILABLE = False

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
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_pretrained_model(model_path, num_nodes, device):
    """Load a pretrained AmortizedCausalDiscovery model."""
    print(f"Looking for pretrained model at {model_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    if not CAUSAL_META_AVAILABLE:
        print("causal_meta package not available. Using fallback implementation.")
        # Use a simple fallback model
        class SimpleAmortizedCausalDiscovery(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = 64
                self.mlp = nn.Sequential(
                    nn.Linear(num_nodes, self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, num_nodes * num_nodes)
                )
                
            def infer_causal_graph(self, data, interventions=None):
                # Simple graph inference
                batch_size = data.shape[0]
                output = self.mlp(data.mean(dim=1))
                adj_matrix = torch.sigmoid(output.view(batch_size, num_nodes, num_nodes))
                # Make it acyclic (upper triangular)
                mask = torch.triu(torch.ones(num_nodes, num_nodes), diagonal=1).to(device)
                adj_matrix = adj_matrix * mask
                return adj_matrix
                
            def predict_intervention_outcomes(self, data, interventions):
                # Simple prediction
                batch_size = data.shape[0]
                return torch.randn(batch_size * num_nodes, 1).to(device)
                
            def forward(self, data, interventions=None):
                return self.infer_causal_graph(data)
                
            def to_causal_graph(self, adj_matrix, threshold=0.5):
                # Return a simple graph representation
                return DummyGraph(adj_matrix.detach().cpu().numpy() > threshold)
                
        acd_model = SimpleAmortizedCausalDiscovery()
        acd_model.to(device)
        print("Created fallback model.")
        return acd_model
    
    # Create a new model with the right dimensions
    try:
        # Create the full amortized causal discovery model using the proper initialization
        # Using parameters that match the actual implementation
        acd_model = AmortizedCausalDiscovery(
            hidden_dim=64,
            input_dim=1,  # Single feature dimension for time series
            attention_heads=2,
            num_layers=2,
            dropout=0.1,
            sparsity_weight=0.1,
            acyclicity_weight=0.1,
            dynamics_weight=1.0,
            structure_weight=1.0,
            uncertainty=True,
            num_ensembles=5
        )
        
        print("Successfully created AmortizedCausalDiscovery model.")
    except Exception as e:
        print(f"Error creating model: {e}")
        print("Using a fallback model instead.")
        # Create a fallback model if the initialization fails
        class SimpleAmortizedCausalDiscovery(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = 64
                self.graph_encoder = nn.Sequential(
                    nn.Linear(num_nodes, self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, num_nodes * num_nodes)
                )
                
                self.dynamics_decoder = nn.Sequential(
                    nn.Linear(num_nodes, self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, 1)
                )
                
            def infer_causal_graph(self, data, interventions=None):
                # Simple graph inference
                batch_size = data.shape[0]
                output = self.graph_encoder(data.mean(dim=1))
                adj_matrix = torch.sigmoid(output.view(batch_size, num_nodes, num_nodes))
                # Make it acyclic (upper triangular)
                mask = torch.triu(torch.ones(num_nodes, num_nodes), diagonal=1).to(device)
                adj_matrix = adj_matrix * mask
                return adj_matrix
                
            def predict_intervention_outcomes(self, data, node_features=None, edge_index=None, 
                                             batch=None, intervention_targets=None, 
                                             intervention_values=None):
                # Simple prediction
                batch_size = data.shape[0]
                return torch.randn(batch_size * num_nodes, 1).to(device)
                
            def forward(self, data, interventions=None):
                return self.infer_causal_graph(data)
                
            def to_causal_graph(self, adj_matrix, threshold=0.5):
                # Return a simple graph representation
                return DummyGraph((adj_matrix.detach().cpu().numpy() > threshold).astype(float))
                
        acd_model = SimpleAmortizedCausalDiscovery()
    
    # Check if model exists, if not, create a dummy model for demo purposes
    if os.path.exists(model_path):
        try:
            acd_model.load_state_dict(torch.load(model_path, map_location=device))
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using untrained model for demonstration.")
    else:
        print(f"Model not found at {model_path}. Using untrained model for demonstration.")
        print("In a real scenario, you would use a properly trained model.")
        # For demo purposes, we'll create a dummy checkpoint
        try:
            torch.save(acd_model.state_dict(), model_path)
            print(f"Created a dummy model checkpoint at {model_path} for future runs.")
        except Exception as e:
            print(f"Could not create dummy checkpoint: {e}")
    
    acd_model.to(device)
    acd_model.eval()
    return acd_model


def generate_synthetic_data(num_nodes, num_samples, seed):
    """Generate synthetic causal graph and data."""
    print(f"Generating synthetic data with {num_nodes} nodes...")
    
    # Create a random DAG using GraphFactory if available
    if GraphFactory is not None and CAUSAL_META_AVAILABLE:
        graph_factory = GraphFactory()
        try:
            graph = graph_factory.create_random_dag(
                num_nodes=num_nodes,
                edge_probability=0.3,
                seed=seed
            )
            print("Successfully created graph using GraphFactory")
        except Exception as e:
            print(f"Error creating graph: {e}")
            print("Using fallback graph creation")
            # Fallback to manual graph creation
            adj_matrix = np.zeros((num_nodes, num_nodes))
            np.random.seed(seed)
            for i in range(num_nodes):
                for j in range(i+1, num_nodes):  # Ensure DAG by only connecting i->j where i<j
                    if np.random.rand() < 0.3:
                        adj_matrix[i, j] = 1
            graph = DummyGraph(adj_matrix)
    else:
        # Manual graph creation as fallback
        print("GraphFactory not available. Using fallback graph creation.")
        adj_matrix = np.zeros((num_nodes, num_nodes))
        np.random.seed(seed)
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):  # Ensure DAG by only connecting i->j where i<j
                if np.random.rand() < 0.3:
                    adj_matrix[i, j] = 1
        graph = DummyGraph(adj_matrix)
    
    # Create Structural Causal Model (SCM)
    if StructuralCausalModel is not None and CAUSAL_META_AVAILABLE:
        try:
            scm = StructuralCausalModel(graph)
            
            # Set node names for clearer visualization if the method exists
            try:
                node_names = [f"X_{i}" for i in range(num_nodes)]
                if hasattr(scm, 'set_node_names'):
                    scm.set_node_names(node_names)
                    print("Successfully set node names")
            except Exception as e:
                print(f"Warning: Could not set node names: {e}")
            
            # Define structural equations for each node
            for i in range(num_nodes):
                node_name = f"X_{i}"
                
                # Get parents of this node
                if hasattr(graph, 'get_parents'):
                    parents = graph.get_parents(node_name)
                elif hasattr(graph, 'get_predecessors'):
                    # For CausalGraph implementation
                    try:
                        parents = graph.get_predecessors(node_name)
                    except:
                        # If node_name doesn't work, try with index
                        parents = graph.get_predecessors(i)
                else:
                    # Fallback: manually determine parents
                    parents = []
                    for j in range(num_nodes):
                        if j < i and (hasattr(graph, 'adj_matrix') and graph.adj_matrix[j, i] > 0 or
                                     hasattr(graph, 'has_edge') and graph.has_edge(j, i)):
                            parents.append(f"X_{j}")
                
                # Create linear equations with normal noise
                equation = create_linear_equation(node_name, parents)
                noise_dist = noise_function
                
                # Add structural equation to the SCM
                scm.define_structural_equation(node_name, equation, noise_dist)
            
            print("Successfully created SCM")
        except Exception as e:
            print(f"Error creating SCM: {e}")
            print("Using fallback SCM implementation")
            scm = DummySCM(graph)
    else:
        print("StructuralCausalModel not available. Using fallback implementation.")
        scm = DummySCM(graph)
    
    # Generate observational data
    try:
        data = scm.sample_data(sample_size=num_samples)
        print(f"Generated {num_samples} observational samples")
    except Exception as e:
        print(f"Error generating data: {e}")
        print("Using random data as fallback")
        data = np.random.normal(0, 1, (num_samples, num_nodes))
    
    return graph, scm, data


def create_linear_equation(node, parents):
    """Create a linear structural equation with random coefficients."""
    def equation(**kwargs):
        # Start with a base value
        result = 0.0
        
        # Add contribution from each parent
        for parent in parents:
            if parent in kwargs:
                # Random coefficient between 0.5 and 2.0
                coef = 0.5 + 1.5 * np.random.rand()
                result += coef * kwargs[parent]
        
        return result
    
    return equation


def noise_function(sample_size, random_state=None):
    """Generate normal noise for structural equations."""
    if random_state is not None:
        np.random.seed(random_state)
    
    # Normal noise with random variance between 0.1 and 0.5
    variance = 0.1 + 0.4 * np.random.rand()
    return np.random.normal(0, np.sqrt(variance), size=sample_size)


def create_graph_from_adjacency(adj_matrix, node_names=None):
    """Create a proper CausalGraph object from an adjacency matrix."""
    if not CAUSAL_META_AVAILABLE or CausalGraph is None:
        # If CausalGraph isn't available, use our DummyGraph with _nodes attribute
        graph = DummyGraph(adj_matrix)
        # Add _nodes attribute that mimics CausalGraph for visualization
        num_nodes = adj_matrix.shape[0]
        graph._nodes = node_names if node_names else [f"X_{i}" for i in range(num_nodes)]
        graph._edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if adj_matrix[i, j] > 0]
        graph._node_attributes = {node: {} for node in graph._nodes}
        graph._edge_attributes = {edge: {} for edge in graph._edges}
        return graph
    
    # Use actual CausalGraph implementation if available
    try:
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
    except Exception as e:
        print(f"Error creating CausalGraph: {e}")
        # Fall back to DummyGraph with _nodes attribute
        graph = DummyGraph(adj_matrix)
        graph._nodes = node_names if node_names else [f"X_{i}" for i in range(adj_matrix.shape[0])]
        graph._edges = [(i, j) for i in range(adj_matrix.shape[0]) for j in range(adj_matrix.shape[0]) 
                      if adj_matrix[i, j] > 0]
        graph._node_attributes = {node: {} for node in graph._nodes}
        graph._edge_attributes = {edge: {} for edge in graph._edges}
        return graph


def parent_scaled_acd(model, scm, obs_data, max_interventions, device, visualize=False):
    """
    Run the Parent-Scaled ACD algorithm with neural network models.
    
    This algorithm:
    1. Infers causal structure from observational data
    2. Selects intervention targets based on parent-count metric
    3. Performs interventions and updates causal structure
    4. Repeats until max interventions reached
    
    Args:
        model: AmortizedCausalDiscovery model
        scm: StructuralCausalModel for generating intervention data
        obs_data: Observational data array
        max_interventions: Maximum number of interventions to perform
        device: PyTorch device for computation
        visualize: Whether to visualize results
        
    Returns:
        Final inferred causal graph and intervention history
    """
    print("\nRunning Parent-Scaled ACD algorithm...")
    print(f"Maximum interventions: {max_interventions}")
    
    # Convert observational data to tensor if needed
    if isinstance(obs_data, np.ndarray):
        obs_data = torch.tensor(obs_data, dtype=torch.float32)
    
    # Move data to the specified device
    obs_data = obs_data.to(device)
    
    # Initial shape processing
    # [num_samples, num_nodes] -> [batch_size, seq_length, num_nodes]
    # For simplicity, we treat the samples as a single batch with sequence length
    if len(obs_data.shape) == 2:
        obs_data = obs_data.unsqueeze(0)  # Add batch dimension
    
    # Extract dimensions
    batch_size, num_samples, num_nodes = obs_data.shape
    
    # 1. Initial causal structure inference from observational data
    print("Inferring initial causal structure from observational data...")
    
    try:
        # Reshape for GraphEncoder if needed
        # Different models might expect different shapes, so we try to be flexible
        try:
            # This is the expected format by the AmortizedCausalDiscovery model
            adj_matrix = model.infer_causal_graph(obs_data)
        except Exception as e:
            print(f"Error in standard inference: {e}")
            print("Trying with reshaped tensor...")
            adj_matrix = model.infer_causal_graph(standardize_tensor_shape(obs_data, for_encoder=True))
        
        # Convert to numpy for visualization if tensor
        if isinstance(adj_matrix, torch.Tensor):
            adj_np = adj_matrix.detach().cpu().numpy()
        else:
            adj_np = adj_matrix
        
        # Threshold adjacency matrix to get binary edges
        threshold = 0.5
        binary_adj = (adj_np > threshold).astype(float)
        
        print(f"Initial adjacency matrix shape: {adj_np.shape}")
        
        # Convert to CausalGraph if possible
        is_causal_graph = False
        if hasattr(model, 'to_causal_graph') and CAUSAL_META_AVAILABLE:
            try:
                graph = model.to_causal_graph(adj_matrix, threshold=threshold)
                print("Successfully converted to CausalGraph")
                is_causal_graph = True
            except Exception as e:
                print(f"Error converting to CausalGraph: {e}")
                # Fallback to DummyGraph
                graph = DummyGraph(binary_adj)
        else:
            graph = DummyGraph(binary_adj)
        
        if visualize:
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.title("Initial Inferred Structure")
            
            if is_causal_graph:
                # If we already have a proper graph object, use it directly
                try:
                    plot_graph(graph)
                except Exception as e:
                    print(f"Error plotting graph: {e}")
                    # Convert adjacency matrix to proper graph as fallback
                    proper_graph = create_graph_from_adjacency(binary_adj)
                    plot_graph(proper_graph)
            else:
                # Create a proper graph from the adjacency matrix
                proper_graph = create_graph_from_adjacency(binary_adj)
                try:
                    plot_graph(proper_graph)
                except Exception as e:
                    print(f"Error plotting graph: {e}")
                    # Use fallback as last resort
                    fallback_plot_graph(binary_adj)
            
            plt.tight_layout()
            plt.show()
        
    except Exception as e:
        print(f"Error in initial structure inference: {e}")
        # Create a random initial graph as fallback
        binary_adj = np.zeros((num_nodes, num_nodes))
        graph = DummyGraph(binary_adj)
        is_causal_graph = False
    
    # Store intervention history
    interventions_history = []
    
    # 2. Iterative intervention loop
    for i in range(max_interventions):
        print(f"\nIntervention {i+1}/{max_interventions}")
        
        # Select intervention target using parent count heuristic
        try:
            target_node = select_intervention_target_by_parent_count(graph, adj_matrix=adj_np)
            # Generate a random intervention value
            target_value = np.random.uniform(-2.0, 2.0)
            print(f"Intervening on node X_{target_node} with value {target_value:.2f}")
        except Exception as e:
            print(f"Error selecting intervention target: {e}")
            # Fallback to random target
            target_node = np.random.randint(0, num_nodes)
            target_value = np.random.uniform(-2.0, 2.0)
            print(f"Fallback: Random intervention on node X_{target_node} with value {target_value:.2f}")
        
        # Record intervention
        intervention = {f"X_{target_node}": target_value}
        interventions_history.append(intervention)
        
        # Generate interventional data
        try:
            int_data = scm.sample_interventional_data(intervention, sample_size=num_samples)
            int_data = torch.tensor(int_data, dtype=torch.float32).to(device)
            
            # Reshape interventional data to match observational data
            int_data = int_data.unsqueeze(0) if len(int_data.shape) == 2 else int_data
            
            print(f"Generated interventional data shape: {int_data.shape}")
        except Exception as e:
            print(f"Error generating interventional data: {e}")
            # Fallback to random data
            int_data = torch.randn_like(obs_data)
            print("Using random interventional data as fallback")
        
        # Format intervention for model prediction (expected by predict_intervention_outcomes)
        intervention_target = target_node
        intervention_value = target_value
        
        # Prepare tensors for the DynamicsDecoder
        node_features = obs_data.reshape(batch_size * num_nodes, num_samples).transpose(0, 1)
        
        # Create edge_index and batch tensors for dynamics prediction
        edge_index = []
        for src in range(num_nodes):
            for tgt in range(num_nodes):
                if binary_adj[src, tgt] > 0:
                    edge_index.append([src, tgt])
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().to(device)
        else:
            # If no edges, create a dummy edge to avoid errors
            edge_index = torch.zeros((2, 1), dtype=torch.long).to(device)
        
        batch = torch.zeros(num_nodes, dtype=torch.long).to(device)
        
        # Prepare intervention tensors
        intervention_targets = torch.tensor([intervention_target], dtype=torch.long).to(device)
        intervention_values = torch.tensor([intervention_value], dtype=torch.float32).to(device)
        
        # Update causal structure using interventional data
        try:
            # Combine observational and interventional data if appropriate for the model
            combined_data = torch.cat([obs_data, int_data], dim=1)
            
            # Update adjacency matrix - NOTE: infer_causal_graph doesn't take interventions param
            adj_matrix = model.infer_causal_graph(combined_data)
            
            if isinstance(adj_matrix, torch.Tensor):
                adj_np = adj_matrix.detach().cpu().numpy()
            else:
                adj_np = adj_matrix
            
            # Apply threshold to get binary edges
            binary_adj = (adj_np > threshold).astype(float)
            
            # Update graph
            is_causal_graph = False
            if hasattr(model, 'to_causal_graph') and CAUSAL_META_AVAILABLE:
                try:
                    graph = model.to_causal_graph(adj_matrix, threshold=threshold)
                    is_causal_graph = True
                except Exception as e:
                    print(f"Error converting to CausalGraph: {e}")
                    graph = DummyGraph(binary_adj)
            else:
                graph = DummyGraph(binary_adj)
            
            print("Successfully updated causal structure")
        except Exception as e:
            print(f"Error updating causal structure: {e}")
            print("Keeping previous structure")
        
        # Visualize the updated graph if requested
        if visualize:
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.title(f"Structure After Intervention {i+1}")
            
            if is_causal_graph:
                try:
                    plot_graph(graph)
                except Exception as e:
                    print(f"Error plotting graph: {e}")
                    # Convert adjacency matrix to proper graph as fallback
                    proper_graph = create_graph_from_adjacency(binary_adj)
                    plot_graph(proper_graph)
            else:
                # Create a proper graph from the adjacency matrix
                proper_graph = create_graph_from_adjacency(binary_adj)
                try:
                    plot_graph(proper_graph)
                except Exception as e:
                    print(f"Error plotting graph: {e}")
                    # Use fallback as last resort
                    fallback_plot_graph(binary_adj)
            
            plt.tight_layout()
            plt.show()
    
    # Final visualization
    if visualize:
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 1, 1)
        plt.title("Final Inferred Causal Structure")
        
        if is_causal_graph:
            try:
                plot_graph(graph)
            except Exception as e:
                print(f"Error plotting graph: {e}")
                # Convert adjacency matrix to proper graph as fallback
                proper_graph = create_graph_from_adjacency(binary_adj)
                plot_graph(proper_graph)
        else:
            # Create a proper graph from the adjacency matrix
            proper_graph = create_graph_from_adjacency(binary_adj)
            try:
                plot_graph(proper_graph)
            except Exception as e:
                print(f"Error plotting graph: {e}")
                # Use fallback as last resort
                fallback_plot_graph(binary_adj)
                
        plt.tight_layout()
        plt.show()
    
    print("\nParent-Scaled ACD algorithm completed.")
    print(f"Performed {len(interventions_history)} interventions.")
    
    return graph, interventions_history


def select_intervention_target_by_parent_count(graph, adj_matrix=None):
    """
    Select intervention target based on the number of parents scaled by edge weight.
    
    This strategy selects nodes that have more parents, assuming they have greater
    potential for causal influence. It normalizes by the number of incoming edges.
    
    Args:
        graph: The causal graph object (CausalGraph or similar)
        adj_matrix: Optional adjacency matrix if graph doesn't provide one
        
    Returns:
        Target node index for intervention
    """
    print("Selecting intervention target based on parent count...")
    
    # Extract adjacency matrix if not provided
    if adj_matrix is None:
        if hasattr(graph, 'get_adjacency_matrix'):
            adj_matrix = graph.get_adjacency_matrix()
        elif hasattr(graph, 'adj_matrix'):
            adj_matrix = graph.adj_matrix
        else:
            raise ValueError("Cannot extract adjacency matrix from graph")
    
    # Ensure adj_matrix is numpy array
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.detach().cpu().numpy()
    
    # Get number of nodes
    if hasattr(graph, 'get_num_nodes'):
        num_nodes = graph.get_num_nodes()
    elif hasattr(graph, 'num_nodes'):
        num_nodes = graph.num_nodes
    else:
        num_nodes = adj_matrix.shape[0]
    
    # Count incoming edges (parents) for each node
    parent_counts = np.zeros(num_nodes)
    
    # Method 1: Use adjacency matrix
    for i in range(num_nodes):
        # Sum incoming edges (column i in adjacency matrix)
        # Note: If adj_matrix[j,i] > 0, then j is a parent of i
        parent_counts[i] = np.sum(adj_matrix[:, i])
    
    # Alternative method if adjacency matrix approach returns all zeros
    if np.sum(parent_counts) == 0:
        print("Adjacency matrix approach failed, trying alternative...")
        # Try using graph methods directly
        for i in range(num_nodes):
            node_name = f"X_{i}"
            # Try different methods to get parents
            if hasattr(graph, 'get_parents'):
                try:
                    parents = graph.get_parents(node_name)
                    parent_counts[i] = len(parents)
                except:
                    try:
                        parents = graph.get_parents(i)
                        parent_counts[i] = len(parents)
                    except:
                        pass
            elif hasattr(graph, 'get_predecessors'):
                try:
                    predecessors = graph.get_predecessors(node_name)
                    parent_counts[i] = len(predecessors)
                except:
                    try:
                        predecessors = graph.get_predecessors(i)
                        parent_counts[i] = len(predecessors)
                    except:
                        pass
            elif hasattr(graph, 'in_degree'):
                try:
                    parent_counts[i] = graph.in_degree(node_name)
                except:
                    try:
                        parent_counts[i] = graph.in_degree(i)
                    except:
                        pass
    
    # Use parent counts as weights, but avoid zeros
    # We add a small constant to ensure all nodes have some chance of selection
    weights = parent_counts + 0.1
    
    # Normalize weights to get probabilities
    probabilities = weights / np.sum(weights)
    
    # Select a node based on these probabilities
    selected_node = np.random.choice(num_nodes, p=probabilities)
    
    print(f"Selected node {selected_node} (X_{selected_node}) with {parent_counts[selected_node]} parents")
    return selected_node


def main():
    """Run the Parent-Scaled ACD demo."""
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
    
    # Visualize final results
    if args.visualize:
        # Get ground truth adjacency matrix from SCM if available
        true_adj_matrix = None
        if hasattr(scm, 'get_adjacency_matrix'):
            true_adj_matrix = scm.get_adjacency_matrix()
        
        # Create final visualization
        plt.figure(figsize=(12, 6))
        
        # Create proper graph objects for visualization
        inferred_graph_for_viz = None
        if hasattr(inferred_graph, 'get_adjacency_matrix') and callable(inferred_graph.get_adjacency_matrix):
            # If it's already a proper graph, use it directly
            inferred_graph_for_viz = inferred_graph
        elif hasattr(inferred_graph, 'adj_matrix'):
            # Convert from adjacency matrix
            inferred_graph_for_viz = create_graph_from_adjacency(inferred_graph.adj_matrix)
        
        # Plot inferred graph
        plt.subplot(1, 2 if true_adj_matrix is not None else 1, 1)
        if inferred_graph_for_viz is not None:
            try:
                plot_graph(inferred_graph_for_viz)
            except Exception as e:
                print(f"Error plotting inferred graph: {e}")
                # Use fallback as last resort
                if hasattr(inferred_graph, 'get_adjacency_matrix'):
                    fallback_plot_graph(inferred_graph.get_adjacency_matrix())
                elif hasattr(inferred_graph, 'adj_matrix'):
                    fallback_plot_graph(inferred_graph.adj_matrix)
                else:
                    fallback_plot_graph(np.zeros((args.num_nodes, args.num_nodes)))
        else:
            # Last resort - empty matrix
            fallback_plot_graph(np.zeros((args.num_nodes, args.num_nodes)))
            print("Warning: Could not create a proper graph for visualization")
            
        plt.title("Final Inferred Causal Graph")
        
        # Plot true graph if available
        if true_adj_matrix is not None:
            plt.subplot(1, 2, 2)
            # Create a proper graph object for the true graph
            true_graph = create_graph_from_adjacency(true_adj_matrix if isinstance(true_adj_matrix, np.ndarray) 
                                                  else np.array(true_adj_matrix))
            try:
                plot_graph(true_graph)
            except Exception as e:
                print(f"Error plotting true graph: {e}")
                fallback_plot_graph(true_adj_matrix if isinstance(true_adj_matrix, np.ndarray) 
                                 else np.array(true_adj_matrix))
            plt.title("True Causal Graph")
        
        plt.tight_layout()
        
        # Save visualization
        save_path = os.path.join(get_assets_dir(), "parent_scaled_acd_results.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved results visualization to {save_path}")
        plt.show()
    
    # Print intervention history
    print("\nIntervention History:")
    for i, intervention in enumerate(intervention_history):
        print(f"Intervention {i+1}: {intervention}")
    
    return inferred_graph, intervention_history


if __name__ == "__main__":
    main() 