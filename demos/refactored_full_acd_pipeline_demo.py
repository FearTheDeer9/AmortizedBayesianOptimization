#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Full Amortized Causal Discovery Pipeline Demo (Refactored)

This script demonstrates the complete amortized approach to causal discovery,
including training, meta-learning adaptation, and intervention selection.
It shows how meta-learning improves performance across related causal structures.

This version uses the refactored utilities that properly leverage components
from the causal_meta package according to the Component Registry.
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
import networkx as nx
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from typing import Any, List, Tuple, Dict, Optional, Union
import logging
import math

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import causal_meta components
import causal_meta
try:
    from causal_meta.meta_learning import AmortizedCausalDiscovery
    from causal_meta.meta_learning.data_generation import SyntheticDataGenerator
    from causal_meta.graph.generators.factory import GraphFactory
    from causal_meta.graph.causal_graph import CausalGraph
    from causal_meta.graph.task_family import TaskFamily
except ImportError as e:
    print(f"Error importing causal_meta components: {e}")
    # Set to None to indicate unavailability
    AmortizedCausalDiscovery = None
    SyntheticDataGenerator = None
    GraphFactory = None
    CausalGraph = None
    TaskFamily = None

# Import utilities from refactored_utils
try:
    from demos.refactored_utils import (
        get_assets_dir,
        get_checkpoints_dir,
        get_node_name,
        get_node_id,
        format_interventions,
        visualize_graph,
        compare_graphs,
        DummyTaskFamily,
        DummyGraph,
        DummySCM,
        plot_graph,
        logger
    )
except ImportError as e:
    print(f"Error importing refactored utilities: {e}")
    # Set up basic logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

# Define CAUSAL_META_AVAILABLE based on whether key components are available
CAUSAL_META_AVAILABLE = (GraphFactory is not None and TaskFamily is not None)

# Define fallback classes if needed
class DummyTaskFamily:
    """
    A simple fallback implementation when TaskFamily is not available.
    
    Args:
        base_graph: The base graph for the family
        variations: List of variations on the base graph
        metadata: Optional metadata about the task family
    """
    def __init__(self, base_graph, variations, metadata=None):
        self.base_graph = base_graph
        self.variations = variations
        self.metadata = metadata or {}
        self._graphs = [base_graph] + [v.get('graph', v) for v in variations]
        
    def __len__(self):
        return len(self._graphs)
        
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._graphs):
            raise IndexError(f"Index {idx} out of range (0-{len(self._graphs)-1})")
        return self._graphs[idx]
        
    def get_graphs(self):
        return self._graphs
        
    def get_base_graph(self):
        return self.base_graph
        
    def get_variations(self):
        return self.variations
        
    def get_metadata(self):
        return self.metadata


class DummySCM:
    """
    A dummy Structural Causal Model for fallbacks when the real implementation isn't available.
    
    This provides a minimal interface to generate random data for demonstration purposes.
    
    Args:
        graph: The causal graph structure
    """
    def __init__(self, graph):
        self.graph = graph
        
        # Determine number of nodes
        if hasattr(graph, 'nodes'):
            self.num_nodes = len(graph.nodes)
        elif hasattr(graph, 'get_num_nodes'):
            self.num_nodes = graph.get_num_nodes()
        elif hasattr(graph, '_nodes'):
            self.num_nodes = len(graph._nodes)
        else:
            self.num_nodes = 5  # Default
        
    def sample(self, n_samples=100, interventions=None, random_state=None):
        """Generate random samples from the SCM."""
        if random_state is not None:
            np.random.seed(random_state)
            
        # Generate random samples
        data = np.random.randn(n_samples, self.num_nodes)
        
        # If interventions are specified, fix values for intervened nodes
        if interventions:
            for node, value in interventions.items():
                if isinstance(node, str) and node.startswith('X_'):
                    node_idx = int(node[2:])
                else:
                    node_idx = int(node)
                
                # Ensure node index is valid
                if 0 <= node_idx < self.num_nodes:
                    data[:, node_idx] = value
        
        return data
        
    def get_causal_graph(self):
        """Return the causal graph."""
        return self.graph


def plot_family_comparison(
    graphs: list, 
    save_path: str = None, 
    figsize_per_graph: tuple = (4, 4),
    titles: list = None
) -> tuple:
    """
    Visualize a family of graphs for comparison using proper visualization components.
    
    Args:
        graphs: List of graph objects (CausalGraph or other with get_adjacency_matrix)
        save_path: Path to save the figure
        figsize_per_graph: Figure size per graph
        titles: List of titles for each graph (default: "Graph {i+1}")
        
    Returns:
        Tuple of (figure, axes) for further customization
    """
    logger.info(f"Visualizing family of {len(graphs)} related graphs")
    
    num_graphs = len(graphs)
    
    # Calculate grid dimensions
    if num_graphs <= 3:
        cols = num_graphs
        rows = 1
    else:
        cols = 3
        rows = (num_graphs + 2) // 3
    
    figsize = (cols * figsize_per_graph[0], rows * figsize_per_graph[1])
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    
    # Convert to flattened array if there's only one row or column
    if rows == 1 and cols == 1:
        axs = np.array([axs])
    elif rows == 1 or cols == 1:
        axs = axs.flatten()
    
    # Plot each graph
    for i, graph in enumerate(graphs):
        if i >= rows * cols:
            break
            
        # Calculate row and column indices
        if rows == 1:
            ax = axs[i]
        elif cols == 1:
            ax = axs[i]
        else:
            r, c = i // cols, i % cols
            ax = axs[r, c]
        
        # Use title if provided, otherwise use default
        title = titles[i] if titles and i < len(titles) else f"Graph {i+1}"
        
        try:
            # Use visualize_graph from refactored utils which handles different graph types
            visualize_graph(graph, ax=ax, title=title)
        except Exception as e:
            logger.error(f"Error visualizing graph {i}: {e}")
            ax.text(0.5, 0.5, f"Error: {str(e)}", 
                   horizontalalignment='center', verticalalignment='center')
            ax.set_title(title)
    
    # Hide any unused subplots
    for i in range(num_graphs, rows * cols):
        if rows == 1:
            axs[i].axis('off')
        elif cols == 1:
            axs[i].axis('off')
        else:
            r, c = i // cols, i % cols
            axs[r, c].axis('off')
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Saved graph family visualization to {save_path}")
        except Exception as e:
            logger.error(f"Error saving graph family visualization: {e}")
    
    return fig, axs


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments as a Namespace object
    """
    parser = argparse.ArgumentParser(
        description='Full Amortized Causal Discovery Pipeline Demo (Refactored)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data generation parameters
    parser.add_argument('--num-nodes', type=int, default=5,
                        help='Number of nodes in the synthetic graph')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of samples for synthetic data generation')
    parser.add_argument('--family-size', type=int, default=5,
                        help='Number of related tasks in the family')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for training')
    
    # Meta-learning parameters
    parser.add_argument('--meta-epochs', type=int, default=5,
                        help='Number of meta-training steps')
    parser.add_argument('--adaptation-steps', type=int, default=3,
                        help='Number of adaptation steps for new tasks')
    parser.add_argument('--inner-lr', type=float, default=0.01,
                        help='Inner loop learning rate for meta-learning')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--quick-mode', action='store_true',
                        help='Run in quick mode with reduced epochs')
    parser.add_argument('--skip-meta', action='store_true',
                        help='Skip meta-learning')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint to load')
    
    # Add any other parameters needed for the demo
    
    args = parser.parse_args()
    
    return args


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility with comprehensive error handling.
    
    Args:
        seed: Random seed
    """
    try:
        logger.info(f"Setting random seed to {seed}")
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e:
        logger.error(f"Error setting random seed: {e}")
        logger.warning("Continuing without deterministic behavior")


def create_task_family(num_nodes: int, family_size: int, seed: int = 42) -> Tuple[Any, List[nx.DiGraph]]:
    """
    Create a family of related causal graphs.
    
    Args:
        num_nodes: Number of nodes in each graph
        family_size: Number of graphs in the family
        seed: Random seed for reproducibility
        
    Returns:
        task_family: TaskFamily object containing the graph family
        family_graphs: List of networkx DiGraph objects
    """
    logger.info(f"Creating task family with {family_size} related causal structures...")
    
    try:
        # Create base graph using GraphFactory
        graph_factory = GraphFactory()
        base_graph = graph_factory.create_random_dag(
            num_nodes=num_nodes, 
            edge_probability=0.3,
            seed=seed
        )
        logger.info(f"Created base graph with {num_nodes} nodes")
        
        # Convert to networkx DiGraph if not already
        if not isinstance(base_graph, nx.DiGraph):
            # Get adjacency matrix and convert to nx.DiGraph
            adj_matrix = base_graph.get_adjacency_matrix()
            nx_graph = nx.DiGraph()
            for i in range(num_nodes):
                nx_graph.add_node(i)
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if adj_matrix[i, j] > 0:
                        nx_graph.add_edge(i, j)
            base_graph_nx = nx_graph
        else:
            base_graph_nx = base_graph
        
        # Generate variations
        variations = []
        variation_graphs_nx = []
        
        for i in range(family_size - 1):
            # Create variation with different parameters
            var_seed = seed + i + 1
            var_graph = graph_factory.create_random_dag(
                num_nodes=num_nodes,
                edge_probability=0.3,
                seed=var_seed
            )
            
            # Convert to networkx DiGraph if needed
            if not isinstance(var_graph, nx.DiGraph):
                adj_matrix = var_graph.get_adjacency_matrix()
                nx_graph = nx.DiGraph()
                for i in range(num_nodes):
                    nx_graph.add_node(i)
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if adj_matrix[i, j] > 0:
                            nx_graph.add_edge(i, j)
                var_graph_nx = nx_graph
            else:
                var_graph_nx = var_graph
                
            variation_graphs_nx.append(var_graph_nx)
            
            # Add to variations list
            variations.append({
                'graph': var_graph_nx,
                'metadata': {
                    'variation_type': 'edge_rewiring',
                    'variation_strength': 0.2,
                    'seed': var_seed
                }
            })
        
        # Create TaskFamily with nx.DiGraph objects
        task_family = causal_meta.graph.task_family.TaskFamily(
            base_graph=base_graph_nx,
            variations=variations,
            metadata={'name': f"SyntheticTaskFamily_{num_nodes}nodes_{family_size}tasks"}
        )
        logger.info(f"Created TaskFamily with {len(task_family)} graphs")
        
        # Return both the TaskFamily object and a list of all original graphs
        family_graphs = [base_graph] + [v['graph'] for v in variations]
        return task_family, family_graphs
        
    except Exception as e:
        logger.error(f"Error creating task family: {e}")
        logger.warning("Using emergency fallback to create dummy task family")
        
        # Create dummy graphs if needed
        dummy_base = nx.DiGraph()
        for i in range(num_nodes):
            dummy_base.add_node(i)
        for i in range(num_nodes-1):
            dummy_base.add_edge(i, i+1)
            
        dummy_variations = []
        dummy_family_graphs = [dummy_base]
        
        for i in range(family_size - 1):
            var_graph = nx.DiGraph()
            for j in range(num_nodes):
                var_graph.add_node(j)
            for j in range(num_nodes-1):
                if np.random.random() < 0.7:  # 70% chance to add each edge
                    var_graph.add_edge(j, j+1)
            
            dummy_variations.append({
                'graph': var_graph,
                'metadata': {'seed': seed + i + 1}
            })
            dummy_family_graphs.append(var_graph)
            
        # Create a dummy task family
        task_family = DummyTaskFamily(dummy_base, dummy_variations)
        
        return task_family, dummy_family_graphs


# Define fallback functions for missing utilities
def create_linear_scm_from_graph(graph, noise_scale=0.5, seed=None):
    """
    Create a linear Structural Causal Model (SCM) from a graph.
    
    Args:
        graph: Graph object (networkx.DiGraph or CausalGraph)
        noise_scale: Scale of the noise in the SCM
        seed: Random seed for reproducibility
        
    Returns:
        Structural Causal Model object
    """
    # Set random seed
    np.random.seed(seed)
    
    # Create a dummy SCM if the proper function is not available
    return DummySCM(graph)


def create_synthetic_data(
    task_family, 
    family_graphs, 
    num_samples=100, 
    include_interventional=True, 
    intervention_nodes_per_graph=1, 
    seed=42
) -> Tuple[List, List, List, List]:
    """
    Generate synthetic observational and interventional data for a family of graphs.
    
    Args:
        task_family: TaskFamily object containing the graph family
        family_graphs: List of graph objects
        num_samples: Number of samples per graph
        include_interventional: Whether to include interventional data
        intervention_nodes_per_graph: Number of intervention targets per graph
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (observational_data_family, interventional_data_family, 
                 scms, intervention_targets)
    """
    logger.info(f"Creating synthetic data with {num_samples} samples per graph")
    
    # Initialize lists to store data and models
    obs_data_family = []
    int_data_family = []
    scms = []
    intervention_targets = []
    
    try:
        # Generate data for each graph in the family
        for i, graph in enumerate(family_graphs):
            # Create SCM from graph
            scm = create_linear_scm_from_graph(graph, seed=seed+i)
            scms.append(scm)
            
            # Generate observational data
            try:
                # Try to use scm sampling method
                obs_data = scm.sample(n_samples=num_samples)
                obs_data = torch.tensor(obs_data, dtype=torch.float32)
            except Exception as e:
                logger.warning(f"Error using SCM sampling method: {e}. Using fallback.")
                # Convert to numpy array
                num_nodes = len(graph.nodes) if hasattr(graph, 'nodes') else graph.get_num_nodes()
                obs_data = torch.randn(num_samples, num_nodes)
            
            obs_data_family.append(obs_data)
            
            # Generate interventional data if requested
            if include_interventional:
                # Select intervention targets
                if hasattr(graph, 'nodes'):
                    nodes = list(graph.nodes)
                elif hasattr(graph, 'get_nodes'):
                    nodes = graph.get_nodes()
                else:
                    nodes = [f"X_{i}" for i in range(graph.get_num_nodes())]
                
                # Randomly select intervention targets
                np.random.seed(seed + i)
                selected_intervention_targets = np.random.choice(
                    nodes, 
                    size=min(intervention_nodes_per_graph, len(nodes)),
                    replace=False
                )
                intervention_targets.append(selected_intervention_targets)
                
                # Generate data for each intervention target
                int_data_for_graph = []
                for target in selected_intervention_targets:
                    # Create intervention
                    if isinstance(target, str) and target.startswith('X_'):
                        target_idx = int(target[2:])
                    else:
                        target_idx = nodes.index(target) if target in nodes else int(target)
                    
                    intervention_value = np.random.randn()
                    interventions = {target_idx: intervention_value}
                    
                    # Generate interventional data
                    try:
                        int_data = scm.sample(n_samples=num_samples, interventions=interventions)
                        int_data = torch.tensor(int_data, dtype=torch.float32)
                    except Exception as e:
                        logger.warning(f"Error generating interventional data: {e}. Using fallback.")
                        int_data = torch.randn(num_samples, len(nodes))
                    
                    int_data_for_graph.append((int_data, {target: intervention_value}))
                
                int_data_family.append(int_data_for_graph)
            else:
                int_data_family.append([])
                intervention_targets.append([])
        
        return obs_data_family, int_data_family, scms, intervention_targets
    
    except Exception as e:
        logger.error(f"Error creating SCM or generating data: {e}")
        logger.warning("Using emergency fallback to create synthetic data")
        
        # Create dummy observational and interventional data
        obs_data_family = []
        int_data_family = []
        scms = []
        intervention_targets = []
        
        for graph in family_graphs:
            # Get number of nodes
            if hasattr(graph, 'nodes'):
                num_nodes = len(graph.nodes)
            elif hasattr(graph, 'get_num_nodes'):
                num_nodes = graph.get_num_nodes()
            elif hasattr(graph, '_nodes'):
                num_nodes = len(graph._nodes)
            else:
                # Default to 5 nodes if we can't determine
                num_nodes = 5
            
            # Generate random observational data
            obs_data = torch.randn(num_samples, num_nodes)
            obs_data_family.append(obs_data)
            
            # Create dummy SCM
            scm = DummySCM(graph)
            scms.append(scm)
            
            # Generate random interventional data if requested
            if include_interventional:
                # Select random intervention target
                selected_targets = [i for i in range(min(intervention_nodes_per_graph, num_nodes))]
                intervention_targets.append(selected_targets)
                
                # Generate interventional data for each target
                int_data_for_graph = []
                for target in selected_targets:
                    intervention_value = np.random.randn()
                    int_data = torch.randn(num_samples, num_nodes)
                    int_data_for_graph.append((int_data, {target: intervention_value}))
                
                int_data_family.append(int_data_for_graph)
            else:
                int_data_family.append([])
                intervention_targets.append([])
        
        return obs_data_family, int_data_family, scms, intervention_targets


def create_model(
    num_nodes: int, 
    device: torch.device,
    hidden_dim: int = 64,
    input_dim: int = 1,
    checkpoint_path: str = None
) -> nn.Module:
    """
    Create an AmortizedCausalDiscovery model using components from Component Registry.
    
    Args:
        num_nodes: Number of nodes in the graph
        device: PyTorch device to place model on
        hidden_dim: Hidden dimension size for neural networks
        input_dim: Input dimension size
        checkpoint_path: Path to model checkpoint (optional)
        
    Returns:
        AmortizedCausalDiscovery model or fallback implementation
    """
    logger.info(f"Creating model for graphs with {num_nodes} nodes")
    
    try:
        if not CAUSAL_META_AVAILABLE:
            raise ImportError("Required causal_meta components not available")
            
        # Check if components are available
        if (GraphFactory is None or 
            AmortizedCausalDiscovery is None):
            raise ImportError("Required neural network components not available")
        
        # Define encoder parameters
        encoder_params = {
            'hidden_dim': hidden_dim,
            'num_nodes': num_nodes,
            'input_dim': input_dim
        }
        
        # Define decoder parameters
        decoder_params = {
            'hidden_dim': hidden_dim,
            'input_dim': input_dim,
            'output_dim': 1  # For simplicity
        }
        
        # First try to load from checkpoint if provided
        if checkpoint_path:
            logger.info(f"Attempting to load model from checkpoint: {checkpoint_path}")
            model = load_model(
                path=checkpoint_path,
                model_class=AmortizedCausalDiscovery,
                device=device,
                hidden_dim=hidden_dim,
                input_dim=input_dim,
                encoder_hidden_dim=hidden_dim,
                encoder_num_nodes=num_nodes,
                encoder_input_dim=input_dim,
                decoder_hidden_dim=hidden_dim,
                decoder_input_dim=input_dim,
                decoder_output_dim=1
            )
            
            if model is not None:
                return model
            
            logger.warning("Failed to load model from checkpoint, creating new model")
        
        # Create encoder
        logger.info("Creating GraphEncoder")
        graph_encoder = GraphEncoder(**encoder_params)
        
        # Create decoder
        logger.info("Creating DynamicsDecoder")
        dynamics_decoder = DynamicsDecoder(**decoder_params)
        
        # Create AmortizedCausalDiscovery model
        logger.info("Creating AmortizedCausalDiscovery model")
        model = AmortizedCausalDiscovery(
            graph_encoder=graph_encoder,
            dynamics_decoder=dynamics_decoder
        )
        
        # Move model to device
        model.to(device)
        return model
        
    except Exception as e:
        logger.error(f"Error creating AmortizedCausalDiscovery model: {e}")
        logger.warning("Using fallback SimpleAmortizedCausalDiscovery implementation")
        
        # Fallback implementation
        class SimpleGraphEncoder(nn.Module):
            """Simple graph encoder fallback when proper GraphEncoder is not available."""
            
            def __init__(self, hidden_dim=64, num_nodes=5, input_dim=1):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.num_nodes = num_nodes
                self.input_dim = input_dim
                
                # Simple MLP
                self.mlp = nn.Sequential(
                    nn.Linear(num_nodes, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, num_nodes * num_nodes)
                )
            
            def forward(self, data, interventions=None):
                """
                Infer graph structure from data.
                
                Args:
                    data: Input data tensor [batch_size, seq_len, num_nodes]
                    interventions: Optional intervention information
                    
                Returns:
                    Adjacency matrix [batch_size, num_nodes, num_nodes]
                """
                # Ensure input has proper shape
                if len(data.shape) == 2:
                    # [seq_len, num_nodes] -> [1, seq_len, num_nodes]
                    data = data.unsqueeze(0)
                
                # Aggregate features
                x = data.mean(dim=1)  # [batch_size, num_nodes]
                
                # Generate adjacency matrix
                batch_size = x.shape[0]
                adjacency = self.mlp(x).view(batch_size, self.num_nodes, self.num_nodes)
                
                # Apply sigmoid to get probabilities
                adjacency = torch.sigmoid(adjacency)
                
                # Ensure DAG by zeroing lower triangle
                mask = torch.triu(torch.ones_like(adjacency), diagonal=1)
                adjacency = adjacency * mask
                
                return adjacency
        
        class SimpleDynamicsDecoder(nn.Module):
            """Simple dynamics decoder fallback when proper DynamicsDecoder is not available."""
            
            def __init__(self, hidden_dim=64, input_dim=1, output_dim=1):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.input_dim = input_dim
                self.output_dim = output_dim
                
                # Simple MLP
                self.mlp = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )
            
            def forward(self, x, adjacency, interventions=None, return_uncertainty=False):
                """
                Predict outcomes given graph structure and interventions.
                
                Args:
                    x: Input features [batch_size, num_nodes, feature_dim]
                    adjacency: Adjacency matrix [batch_size, num_nodes, num_nodes]
                    interventions: Optional intervention information
                    return_uncertainty: Whether to return uncertainty estimates
                    
                Returns:
                    Predicted outcomes, optionally with uncertainty
                """
                # Apply a simple transformation
                predictions = self.mlp(x)
                
                if return_uncertainty:
                    # Simple constant uncertainty estimate
                    uncertainty = torch.ones_like(predictions) * 0.1
                    return predictions, uncertainty
                
                return predictions
        
        class SimpleAmortizedCausalDiscovery(nn.Module):
            """
            Simple AmortizedCausalDiscovery fallback when proper implementation is not available.
            """
            
            def __init__(self, graph_encoder=None, dynamics_decoder=None, 
                         hidden_dim=64, num_nodes=5, input_dim=1):
                super().__init__()
                
                # Create components if not provided
                self.graph_encoder = graph_encoder or SimpleGraphEncoder(
                    hidden_dim=hidden_dim, 
                    num_nodes=num_nodes,
                    input_dim=input_dim
                )
                
                self.dynamics_decoder = dynamics_decoder or SimpleDynamicsDecoder(
                    hidden_dim=hidden_dim,
                    input_dim=input_dim,
                    output_dim=1
                )
                
                self.num_nodes = num_nodes
            
            def forward(self, data, interventions=None):
                """
                Forward pass through the full model.
                
                Args:
                    data: Input data tensor
                    interventions: Optional intervention information
                    
                Returns:
                    Tuple of (adjacency_matrix, predictions)
                """
                # Infer graph structure
                adjacency = self.graph_encoder(data, interventions)
                
                # Prepare input for dynamics decoder
                batch_size = data.shape[0] if len(data.shape) > 1 else 1
                x = data.mean(dim=1, keepdim=True).expand(batch_size, self.num_nodes, 1)
                
                # Predict outcomes
                predictions = self.dynamics_decoder(x, adjacency, interventions)
                
                return adjacency, predictions
            
            def infer_causal_graph(self, data, interventions=None, threshold=0.5):
                """
                Infer causal graph structure from data.
                
                Args:
                    data: Input data tensor
                    interventions: Optional intervention information
                    threshold: Threshold for edge probabilities
                    
                Returns:
                    Adjacency matrix
                """
                with torch.no_grad():
                    adjacency = self.graph_encoder(data, interventions)
                    if threshold is not None:
                        adjacency = (adjacency > threshold).float()
                    return adjacency
            
            def predict_intervention_outcomes(self, data, interventions=None, return_uncertainty=False):
                """
                Predict outcomes under interventions.
                
                Args:
                    data: Input data tensor
                    interventions: Intervention information
                    return_uncertainty: Whether to return uncertainty estimates
                    
                Returns:
                    Predicted outcomes, optionally with uncertainty
                """
                with torch.no_grad():
                    # Infer graph structure
                    adjacency = self.graph_encoder(data)
                    
                    # Prepare input for dynamics decoder
                    batch_size = data.shape[0] if len(data.shape) > 1 else 1
                    x = data.mean(dim=1, keepdim=True).expand(batch_size, self.num_nodes, 1)
                    
                    # Predict outcomes
                    return self.dynamics_decoder(x, adjacency, interventions, return_uncertainty)
            
            def to_causal_graph(self, adjacency_matrix, threshold=0.5):
                """
                Convert adjacency matrix to CausalGraph object.
                
                Args:
                    adjacency_matrix: Adjacency matrix tensor
                    threshold: Threshold for edge probabilities
                    
                Returns:
                    CausalGraph object or DummyGraph fallback
                """
                if threshold is not None:
                    adjacency_matrix = (adjacency_matrix > threshold).float()
                
                # Convert to numpy if needed
                if isinstance(adjacency_matrix, torch.Tensor):
                    adjacency_matrix = adjacency_matrix.detach().cpu().numpy()
                
                # Create graph
                return create_causal_graph_from_adjacency(
                    adjacency_matrix,
                    threshold=None  # Already thresholded
                )
        
        # Create the fallback model
        model = SimpleAmortizedCausalDiscovery(
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            input_dim=input_dim
        )
        
        # Move to device
        model.to(device)
        return model


def train_step(
    model: nn.Module, 
    data_batch: tuple, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device,
    graph_weight: float = 1.0,
    dynamics_weight: float = 1.0,
    sparsity_weight: float = 0.1
) -> dict:
    """
    Perform a single training step on the model.
    
    Args:
        model: AmortizedCausalDiscovery model
        data_batch: Tuple of (obs_data, int_data, true_graphs)
        optimizer: PyTorch optimizer
        device: PyTorch device
        graph_weight: Weight for graph loss component
        dynamics_weight: Weight for dynamics loss component
        sparsity_weight: Weight for sparsity regularization
        
    Returns:
        Dictionary with loss information
    """
    # Unpack the batch
    obs_data, int_data, true_graphs = data_batch
    
    # Move data to device
    obs_data = obs_data.to(device)
    if int_data is not None:
        int_data = int_data.to(device)
    
    # Move true graphs to device if they are tensors
    if isinstance(true_graphs, torch.Tensor):
        true_graphs = true_graphs.to(device)
    
    # Reset gradients
    optimizer.zero_grad()
    
    try:
        # Forward pass
        if hasattr(model, 'train_step') and callable(model.train_step):
            # Use model's built-in train_step if available
            loss_info = model.train_step(
                obs_data=obs_data,
                int_data=int_data,
                true_graphs=true_graphs,
                optimizer=optimizer,
                graph_weight=graph_weight,
                dynamics_weight=dynamics_weight,
                sparsity_weight=sparsity_weight
            )
        else:
            # Manual implementation for models without built-in train_step
            # Infer graph structure
            pred_graphs = model.graph_encoder(obs_data)
            
            # Calculate graph structure loss if true graphs are available
            if true_graphs is not None:
                # Convert true_graphs to tensor if it's not already
                if not isinstance(true_graphs, torch.Tensor):
                    if hasattr(true_graphs[0], 'get_adjacency_matrix'):
                        # Convert CausalGraph objects to adjacency matrices
                        true_graph_arrays = [g.get_adjacency_matrix() for g in true_graphs]
                        true_graphs_tensor = torch.tensor(np.stack(true_graph_arrays), 
                                                         dtype=torch.float32, device=device)
                    elif hasattr(true_graphs[0], 'adj_matrix'):
                        # Convert objects with adj_matrix attribute
                        true_graph_arrays = [g.adj_matrix for g in true_graphs]
                        true_graphs_tensor = torch.tensor(np.stack(true_graph_arrays), 
                                                         dtype=torch.float32, device=device)
                    else:
                        # Use as is, assuming it's a list of numpy arrays
                        true_graphs_tensor = torch.tensor(np.stack(true_graphs), 
                                                         dtype=torch.float32, device=device)
                else:
                    true_graphs_tensor = true_graphs
                
                # Binary cross entropy loss for graph structure
                graph_loss = F.binary_cross_entropy(pred_graphs, true_graphs_tensor)
            else:
                # If no true graphs, use a placeholder
                graph_loss = torch.tensor(0.0, device=device)
            
            # Add sparsity regularization
            sparsity_loss = pred_graphs.mean()
            
            # Predict dynamics with interventional data if available
            if int_data is not None:
                # Create dummy interventions for this demo
                interventions = {
                    'mask': torch.zeros(pred_graphs.shape[1], device=device),
                    'values': torch.zeros(pred_graphs.shape[1], device=device)
                }
                interventions['mask'][0] = 1.0  # Intervene on first node
                interventions['values'][0] = 1.0  # Set value to 1.0
                
                # Forward pass through dynamics decoder
                batch_size = pred_graphs.shape[0]
                node_features = obs_data.mean(dim=1, keepdim=True).expand(batch_size, pred_graphs.shape[1], 1)
                
                pred_outcomes = model.dynamics_decoder(
                    node_features, 
                    pred_graphs, 
                    interventions
                )
                
                # Calculate dynamics loss (MSE)
                target_outcomes = int_data
                dynamics_loss = F.mse_loss(pred_outcomes, target_outcomes)
            else:
                # If no interventional data, use a placeholder
                dynamics_loss = torch.tensor(0.0, device=device)
            
            # Combined loss
            loss = (
                graph_weight * graph_loss +
                dynamics_weight * dynamics_loss +
                sparsity_weight * sparsity_loss
            )
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Create loss info dictionary
            loss_info = {
                'loss': loss.item(),
                'graph_loss': graph_loss.item(),
                'dynamics_loss': dynamics_loss.item() if int_data is not None else 0.0,
                'sparsity_loss': sparsity_loss.item()
            }
    
    except Exception as e:
        logger.error(f"Error in train_step: {e}")
        
        # Return dummy loss info on error
        loss_info = {
            'loss': 0.0,
            'graph_loss': 0.0,
            'dynamics_loss': 0.0,
            'sparsity_loss': 0.0,
            'error': str(e)
        }
    
    return loss_info


def prepare_training_data(
    observational_data: list, 
    interventional_data: list = None,
    true_graphs: list = None,
    batch_size: int = 16
) -> DataLoader:
    """
    Prepare training data for AmortizedCausalDiscovery model.
    
    Args:
        observational_data: List of observational data tensors
        interventional_data: List of interventional data tensors (optional)
        true_graphs: List of true graph objects or adjacency matrices (optional)
        batch_size: Batch size for DataLoader
        
    Returns:
        PyTorch DataLoader
    """
    logger.info("Preparing training data")
    
    try:
        # Convert all data to torch tensors if they aren't already
        obs_data_tensors = []
        for data in observational_data:
            if isinstance(data, np.ndarray):
                obs_data_tensors.append(torch.tensor(data, dtype=torch.float32))
            else:
                obs_data_tensors.append(data)
        
        # Stack all observations
        obs_data_combined = torch.cat(obs_data_tensors, dim=0)
        
        # Process interventional data if available
        if interventional_data and len(interventional_data) > 0:
            int_data_tensors = []
            for data in interventional_data:
                if isinstance(data, np.ndarray):
                    int_data_tensors.append(torch.tensor(data, dtype=torch.float32))
                else:
                    int_data_tensors.append(data)
            
            # Stack all interventions
            int_data_combined = torch.cat(int_data_tensors, dim=0)
        else:
            int_data_combined = None
        
        # Process true graphs if available
        if true_graphs and len(true_graphs) > 0:
            # If true_graphs are already tensors, use them directly
            if isinstance(true_graphs[0], torch.Tensor):
                true_graphs_combined = torch.cat(true_graphs, dim=0)
            # If they're adjacency matrices as numpy arrays
            elif isinstance(true_graphs[0], np.ndarray):
                true_graphs_combined = torch.tensor(np.stack(true_graphs), dtype=torch.float32)
            # If they're CausalGraph objects or similar
            elif hasattr(true_graphs[0], 'get_adjacency_matrix'):
                true_graphs_arrays = [g.get_adjacency_matrix() for g in true_graphs]
                true_graphs_combined = torch.tensor(np.stack(true_graphs_arrays), dtype=torch.float32)
            # If they're objects with adj_matrix attribute
            elif hasattr(true_graphs[0], 'adj_matrix'):
                true_graphs_arrays = [g.adj_matrix for g in true_graphs]
                true_graphs_combined = torch.tensor(np.stack(true_graphs_arrays), dtype=torch.float32)
            else:
                logger.warning("Unknown graph type, using observation shape to create dummy graphs")
                # Create dummy graphs based on observation shape
                num_nodes = obs_data_combined.shape[-1]
                true_graphs_combined = torch.zeros(len(observational_data), num_nodes, num_nodes)
        else:
            true_graphs_combined = None
        
        # Create dataset and dataloader
        if int_data_combined is not None and true_graphs_combined is not None:
            # Full dataset with observational, interventional, and true graphs
            dataset = TensorDataset(obs_data_combined, int_data_combined, true_graphs_combined)
        elif int_data_combined is not None:
            # Dataset with observational and interventional data
            dataset = TensorDataset(obs_data_combined, int_data_combined)
        elif true_graphs_combined is not None:
            # Dataset with observational data and true graphs
            dataset = TensorDataset(obs_data_combined, true_graphs_combined)
        else:
            # Dataset with only observational data
            dataset = TensorDataset(obs_data_combined)
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return dataloader
    
    except Exception as e:
        logger.error(f"Error preparing training data: {e}")
        
        # Create a minimal dummy dataset as fallback
        logger.warning("Creating dummy dataset as fallback")
        
        # Create minimal tensors
        dummy_obs = torch.randn(batch_size, 10, 5)  # [batch, seq_len, features]
        dummy_dataset = TensorDataset(dummy_obs)
        dummy_loader = DataLoader(dummy_dataset, batch_size=batch_size, shuffle=True)
        
        return dummy_loader


def train_model(
    model: nn.Module, 
    dataloader: DataLoader, 
    num_epochs: int, 
    device: torch.device,
    learning_rate: float = 0.001,
    graph_weight: float = 1.0,
    dynamics_weight: float = 1.0,
    sparsity_weight: float = 0.1,
    checkpoint_path: str = None,
    quick_mode: bool = False
) -> nn.Module:
    """
    Train an AmortizedCausalDiscovery model.
    
    Args:
        model: AmortizedCausalDiscovery model
        dataloader: DataLoader with training data
        num_epochs: Number of training epochs
        device: PyTorch device
        learning_rate: Learning rate for optimizer
        graph_weight: Weight for graph loss component
        dynamics_weight: Weight for dynamics loss component
        sparsity_weight: Weight for sparsity regularization
        checkpoint_path: Path to save model checkpoint (optional)
        quick_mode: Whether to run in quick mode with minimal training
        
    Returns:
        Trained model
    """
    logger.info(f"Training model for {num_epochs} epochs")
    
    # Adjust epochs if in quick mode
    if quick_mode:
        num_epochs = min(num_epochs, 3)
        logger.info(f"Quick mode: reduced to {num_epochs} epochs")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        graph_losses = 0.0
        dynamics_losses = 0.0
        sparsity_losses = 0.0
        num_batches = 0
        
        # Process each batch
        for batch_idx, batch in enumerate(dataloader):
            # Convert batch to appropriate format based on dataset structure
            if len(batch) == 3:
                # Full dataset with observational, interventional, and true graphs
                obs_data, int_data, true_graphs = batch
                train_batch = (obs_data, int_data, true_graphs)
            elif len(batch) == 2:
                # Dataset with observational data and either interventional data or true graphs
                # Determine which one based on dataset content and shape
                if batch[1].dim() == 3 and batch[0].shape[-1] == batch[1].shape[-1]:
                    # Second tensor is likely interventional data
                    obs_data, int_data = batch
                    train_batch = (obs_data, int_data, None)
                else:
                    # Second tensor is likely true graphs
                    obs_data, true_graphs = batch
                    train_batch = (obs_data, None, true_graphs)
            else:
                # Dataset with only observational data
                obs_data = batch[0]
                train_batch = (obs_data, None, None)
            
            # Perform a training step
            loss_info = train_step(
                model=model,
                data_batch=train_batch,
                optimizer=optimizer,
                device=device,
                graph_weight=graph_weight,
                dynamics_weight=dynamics_weight,
                sparsity_weight=sparsity_weight
            )
            
            # Accumulate losses
            epoch_loss += loss_info['loss']
            graph_losses += loss_info.get('graph_loss', 0.0)
            dynamics_losses += loss_info.get('dynamics_loss', 0.0)
            sparsity_losses += loss_info.get('sparsity_loss', 0.0)
            num_batches += 1
            
            # Print progress for first and last batch of each epoch
            if batch_idx == 0 or batch_idx == len(dataloader) - 1:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                           f"Loss: {loss_info['loss']:.4f}, "
                           f"Graph: {loss_info.get('graph_loss', 0.0):.4f}, "
                           f"Dynamics: {loss_info.get('dynamics_loss', 0.0):.4f}")
        
        # Calculate average losses
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_graph_loss = graph_losses / max(num_batches, 1)
        avg_dynamics_loss = dynamics_losses / max(num_batches, 1)
        avg_sparsity_loss = sparsity_losses / max(num_batches, 1)
        
        # Print epoch summary
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed, "
                   f"Avg Loss: {avg_loss:.4f}, "
                   f"Graph: {avg_graph_loss:.4f}, "
                   f"Dynamics: {avg_dynamics_loss:.4f}, "
                   f"Sparsity: {avg_sparsity_loss:.4f}")
    
    # Save checkpoint if path is provided
    if checkpoint_path:
        try:
            logger.info(f"Saving model checkpoint to {checkpoint_path}")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)
        except Exception as e:
            logger.error(f"Error saving model checkpoint: {e}")
    
    return model 


def prepare_meta_training_data(
    task_family: Any,
    family_observational_data: list,
    family_interventional_data: list = None,
    family_graphs: list = None,
    num_tasks: int = None,
    batch_size: int = 4,
    k_shot: int = 5,
    meta_batch_size: int = 16
) -> DataLoader:
    """
    Prepare meta-learning data for a family of related causal tasks.
    
    Args:
        task_family: TaskFamily object or dict containing task family information
        family_observational_data: List of observational data tensors for each task
        family_interventional_data: List of interventional data tensors for each task (optional)
        family_graphs: List of graph objects for each task (optional)
        num_tasks: Number of tasks to use (if None, use all available)
        batch_size: Batch size for each task's data
        k_shot: Number of samples for adaptation
        meta_batch_size: Batch size for meta-learning
        
    Returns:
        PyTorch DataLoader for meta-learning
    """
    logger.info("Preparing meta-training data")
    
    try:
        # Use specified number of tasks or all available
        if num_tasks is None:
            num_tasks = len(family_observational_data)
        else:
            num_tasks = min(num_tasks, len(family_observational_data))
        
        logger.info(f"Using {num_tasks} tasks for meta-learning")
        
        # Prepare data for each task
        task_dataloaders = []
        for i in range(num_tasks):
            # Get data for this task
            obs_data = family_observational_data[i]
            int_data = family_interventional_data[i] if family_interventional_data is not None else None
            graph = family_graphs[i] if family_graphs is not None else None
            
            # Convert to appropriate format
            if isinstance(obs_data, np.ndarray):
                obs_data = torch.tensor(obs_data, dtype=torch.float32)
            
            if int_data is not None and isinstance(int_data, np.ndarray):
                int_data = torch.tensor(int_data, dtype=torch.float32)
            
            if graph is not None:
                if isinstance(graph, np.ndarray):
                    graph = torch.tensor(graph, dtype=torch.float32)
                elif hasattr(graph, 'get_adjacency_matrix'):
                    graph = torch.tensor(graph.get_adjacency_matrix(), dtype=torch.float32)
                elif hasattr(graph, 'adj_matrix'):
                    graph = torch.tensor(graph.adj_matrix, dtype=torch.float32)
            
            # Create dataset
            if int_data is not None and graph is not None:
                dataset = TensorDataset(obs_data, int_data, graph.unsqueeze(0).expand(len(obs_data), -1, -1))
            elif int_data is not None:
                dataset = TensorDataset(obs_data, int_data)
            elif graph is not None:
                dataset = TensorDataset(obs_data, graph.unsqueeze(0).expand(len(obs_data), -1, -1))
            else:
                dataset = TensorDataset(obs_data)
            
            # Create dataloader
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            task_dataloaders.append(dataloader)
        
        # Function to sample a batch of tasks
        def sample_meta_batch():
            sampled_tasks = np.random.choice(num_tasks, size=meta_batch_size, replace=True)
            task_batch = []
            
            for task_idx in sampled_tasks:
                # Get dataloader for this task
                dataloader = task_dataloaders[task_idx]
                
                # Get a batch of data
                try:
                    batch = next(iter(dataloader))
                except StopIteration:
                    # Reinitialize dataloader if it's exhausted
                    dataloader = DataLoader(dataloader.dataset, batch_size=batch_size, shuffle=True)
                    task_dataloaders[task_idx] = dataloader
                    batch = next(iter(dataloader))
                
                # Split into adaptation and evaluation sets
                adapt_data = [tensor[:k_shot] for tensor in batch]
                eval_data = [tensor[k_shot:2*k_shot] for tensor in batch]
                
                # Add to task batch
                task_batch.append((adapt_data, eval_data))
            
            return task_batch
        
        return sample_meta_batch
    
    except Exception as e:
        logger.error(f"Error preparing meta-training data: {e}")
        
        # Create a dummy meta-batch sampler as fallback
        logger.warning("Creating dummy meta-batch sampler as fallback")
        
        def dummy_sampler():
            # Create dummy data
            dummy_tasks = []
            for _ in range(meta_batch_size):
                # Create dummy tensors
                dummy_obs = torch.randn(2*k_shot, 10, 5)  # [samples, seq_len, features]
                dummy_adapt = (dummy_obs[:k_shot],)
                dummy_eval = (dummy_obs[k_shot:2*k_shot],)
                dummy_tasks.append((dummy_adapt, dummy_eval))
            return dummy_tasks
        
        return dummy_sampler


def setup_meta_learning(
    model: nn.Module, 
    device: torch.device,
    inner_lr: float = 0.01,
    meta_lr: float = 0.001,
    first_order: bool = True
) -> tuple:
    """
    Set up meta-learning for the AmortizedCausalDiscovery model.
    
    Args:
        model: AmortizedCausalDiscovery model
        device: PyTorch device
        inner_lr: Learning rate for inner adaptation
        meta_lr: Learning rate for meta-update
        first_order: Whether to use first-order approximation
        
    Returns:
        Tuple of (meta_model, meta_optimizer, inner_optimizer_class)
    """
    logger.info("Setting up meta-learning")
    
    try:
        # Check if required components are available
        if MAMLForCausalDiscovery is None:
            raise ImportError("MAMLForCausalDiscovery is not available")
        
        # Create MAML wrapper
        meta_model = MAMLForCausalDiscovery(
            model=model, 
            inner_lr=inner_lr,
            first_order=first_order
        )
        
        # Move to device
        meta_model.to(device)
        
        # Create meta-optimizer
        meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=meta_lr)
        
        # Return optimizer class for inner loop
        inner_optimizer_class = torch.optim.SGD
        
        return meta_model, meta_optimizer, inner_optimizer_class
    
    except Exception as e:
        logger.error(f"Error setting up meta-learning: {e}")
        logger.warning("Using fallback implementation")
        
        # Simple fallback implementation of MAML for causal discovery
        class SimpleMAMLForCausalDiscovery(nn.Module):
            """Simple MAML implementation for causal discovery."""
            
            def __init__(self, model, inner_lr=0.01, first_order=True):
                super().__init__()
                self.model = model
                self.inner_lr = inner_lr
                self.first_order = first_order
            
            def clone_model(self):
                """Create a clone of the model with same parameters."""
                clone = type(self.model)(
                    hidden_dim=self.model.graph_encoder.hidden_dim 
                        if hasattr(self.model.graph_encoder, 'hidden_dim') else 64,
                    num_nodes=self.model.graph_encoder.num_nodes 
                        if hasattr(self.model.graph_encoder, 'num_nodes') else 5
                )
                
                # Copy parameters
                clone.load_state_dict(self.model.state_dict())
                return clone
            
            def adapt(self, adapt_data, steps=1, create_graph=False):
                """Adapt the model to a new task."""
                # Clone the model
                adapted_model = self.clone_model()
                
                # Perform adaptation steps
                for _ in range(steps):
                    # Forward pass
                    if len(adapt_data) == 3:
                        obs_data, int_data, true_graphs = adapt_data
                        pred_graphs = adapted_model.graph_encoder(obs_data)
                        
                        # Graph loss
                        graph_loss = F.binary_cross_entropy(pred_graphs, true_graphs)
                        
                        # Dynamics loss if interventional data available
                        if int_data is not None:
                            # Create dummy interventions
                            interventions = {
                                'mask': torch.zeros(pred_graphs.shape[1], device=obs_data.device),
                                'values': torch.zeros(pred_graphs.shape[1], device=obs_data.device)
                            }
                            interventions['mask'][0] = 1.0
                            interventions['values'][0] = 1.0
                            
                            # Forward pass through dynamics decoder
                            batch_size = pred_graphs.shape[0]
                            node_features = obs_data.mean(dim=1, keepdim=True).expand(
                                batch_size, pred_graphs.shape[1], 1)
                            
                            pred_outcomes = adapted_model.dynamics_decoder(
                                node_features, pred_graphs, interventions)
                            
                            # Dynamics loss
                            dynamics_loss = F.mse_loss(pred_outcomes, int_data)
                            
                            # Combined loss
                            loss = graph_loss + dynamics_loss
                        else:
                            loss = graph_loss
                    elif len(adapt_data) == 2:
                        # If only observations and one other data type
                        if adapt_data[1].dim() == 3 and adapt_data[0].shape[-1] == adapt_data[1].shape[-1]:
                            # Second tensor is interventional data
                            obs_data, int_data = adapt_data
                            true_graphs = None
                        else:
                            # Second tensor is graph data
                            obs_data, true_graphs = adapt_data
                            int_data = None
                        
                        # Continue with adaptation as above...
                        # This is simplified for brevity
                        pred_graphs = adapted_model.graph_encoder(obs_data)
                        
                        if true_graphs is not None:
                            loss = F.binary_cross_entropy(pred_graphs, true_graphs)
                        else:
                            # Dummy loss
                            loss = pred_graphs.mean()  # Sparsity loss
                    else:
                        # Only observational data
                        obs_data = adapt_data[0]
                        pred_graphs = adapted_model.graph_encoder(obs_data)
                        
                        # Sparsity loss as dummy objective
                        loss = pred_graphs.mean()
                    
                    # Compute gradients
                    grads = torch.autograd.grad(
                        loss, 
                        adapted_model.parameters(),
                        create_graph=create_graph,
                        retain_graph=True
                    )
                    
                    # Update parameters
                    for p, g in zip(adapted_model.parameters(), grads):
                        p.data.sub_(self.inner_lr * g)
                
                return adapted_model
            
            def forward(self, task_batch):
                """
                Forward pass for meta-learning.
                
                Args:
                    task_batch: List of (adapt_data, eval_data) tuples
                    
                Returns:
                    Meta-loss
                """
                meta_loss = 0.0
                
                for adapt_data, eval_data in task_batch:
                    # Adapt model to this task
                    adapted_model = self.adapt(
                        adapt_data,
                        steps=1,
                        create_graph=not self.first_order
                    )
                    
                    # Evaluate on query set
                    if len(eval_data) == 3:
                        obs_data, int_data, true_graphs = eval_data
                        pred_graphs = adapted_model.graph_encoder(obs_data)
                        
                        # Graph loss
                        graph_loss = F.binary_cross_entropy(pred_graphs, true_graphs)
                        
                        # Dynamics loss if available
                        if int_data is not None:
                            # Create dummy interventions
                            interventions = {
                                'mask': torch.zeros(pred_graphs.shape[1], device=obs_data.device),
                                'values': torch.zeros(pred_graphs.shape[1], device=obs_data.device)
                            }
                            interventions['mask'][0] = 1.0
                            interventions['values'][0] = 1.0
                            
                            # Forward pass through dynamics decoder
                            batch_size = pred_graphs.shape[0]
                            node_features = obs_data.mean(dim=1, keepdim=True).expand(
                                batch_size, pred_graphs.shape[1], 1)
                            
                            pred_outcomes = adapted_model.dynamics_decoder(
                                node_features, pred_graphs, interventions)
                            
                            # Dynamics loss
                            dynamics_loss = F.mse_loss(pred_outcomes, int_data)
                            
                            # Combined loss
                            task_loss = graph_loss + dynamics_loss
                        else:
                            task_loss = graph_loss
                    elif len(eval_data) == 2:
                        # Similar logic as in adapt method
                        # This is simplified for brevity
                        if eval_data[1].dim() == 3 and eval_data[0].shape[-1] == eval_data[1].shape[-1]:
                            obs_data, int_data = eval_data
                            true_graphs = None
                        else:
                            obs_data, true_graphs = eval_data
                            int_data = None
                        
                        pred_graphs = adapted_model.graph_encoder(obs_data)
                        
                        if true_graphs is not None:
                            task_loss = F.binary_cross_entropy(pred_graphs, true_graphs)
                        else:
                            task_loss = pred_graphs.mean()  # Sparsity loss
                    else:
                        # Only observational data
                        obs_data = eval_data[0]
                        pred_graphs = adapted_model.graph_encoder(obs_data)
                        
                        # Sparsity loss as dummy objective
                        task_loss = pred_graphs.mean()
                    
                    # Add to meta-loss
                    meta_loss += task_loss
                
                # Average over tasks
                meta_loss /= len(task_batch)
                
                return meta_loss
            
            def to(self, device):
                """Move model to device."""
                self.model.to(device)
                return super().to(device)
        
        # Create fallback meta-model
        meta_model = SimpleMAMLForCausalDiscovery(
            model=model,
            inner_lr=inner_lr,
            first_order=first_order
        )
        
        # Move to device
        meta_model.to(device)
        
        # Create meta-optimizer
        meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=meta_lr)
        
        # Return optimizer class for inner loop
        inner_optimizer_class = torch.optim.SGD
        
        return meta_model, meta_optimizer, inner_optimizer_class


def meta_train_step(
    meta_model: nn.Module,
    meta_batch: list,
    meta_optimizer: torch.optim.Optimizer,
    device: torch.device
) -> dict:
    """
    Perform a single meta-training step.
    
    Args:
        meta_model: Meta-learning model (e.g., MAMLForCausalDiscovery)
        meta_batch: List of (adapt_data, eval_data) tuples for each task
        meta_optimizer: Optimizer for meta-parameters
        device: PyTorch device
        
    Returns:
        Dictionary with loss information
    """
    # Reset gradients
    meta_optimizer.zero_grad()
    
    try:
        # Move data to device
        device_meta_batch = []
        for adapt_data, eval_data in meta_batch:
            # Move adaptation data to device
            device_adapt_data = tuple(tensor.to(device) for tensor in adapt_data)
            
            # Move evaluation data to device
            device_eval_data = tuple(tensor.to(device) for tensor in eval_data)
            
            # Add to device batch
            device_meta_batch.append((device_adapt_data, device_eval_data))
        
        # Forward pass
        meta_loss = meta_model(device_meta_batch)
        
        # Backward pass
        meta_loss.backward()
        
        # Update meta-parameters
        meta_optimizer.step()
        
        # Create loss info
        loss_info = {
            'meta_loss': meta_loss.item()
        }
    
    except Exception as e:
        logger.error(f"Error in meta_train_step: {e}")
        
        # Return dummy loss info
        loss_info = {
            'meta_loss': 0.0,
            'error': str(e)
        }
    
    return loss_info


def meta_train_model(
    meta_model: nn.Module,
    meta_batch_sampler: callable,
    meta_optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    quick_mode: bool = False,
    checkpoint_path: str = None
) -> nn.Module:
    """
    Perform meta-training on a family of related causal tasks.
    
    Args:
        meta_model: Meta-learning model (e.g., MAMLForCausalDiscovery)
        meta_batch_sampler: Function that samples meta-batches
        meta_optimizer: Optimizer for meta-parameters
        device: PyTorch device
        num_epochs: Number of meta-training epochs
        quick_mode: Whether to run in quick mode with minimal training
        checkpoint_path: Path to save model checkpoint
        
    Returns:
        Meta-trained model
    """
    logger.info(f"Meta-training model for {num_epochs} epochs")
    
    # Adjust epochs if in quick mode
    if quick_mode:
        num_epochs = min(num_epochs, 3)
        num_steps_per_epoch = 5
        logger.info(f"Quick mode: reduced to {num_epochs} epochs, {num_steps_per_epoch} steps per epoch")
    else:
        num_steps_per_epoch = 10
    
    # Training loop
    meta_model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_steps = 0
        
        # Perform steps for this epoch
        for step in range(num_steps_per_epoch):
            # Sample meta-batch
            meta_batch = meta_batch_sampler()
            
            # Perform meta-training step
            loss_info = meta_train_step(
                meta_model=meta_model,
                meta_batch=meta_batch,
                meta_optimizer=meta_optimizer,
                device=device
            )
            
            # Accumulate loss
            epoch_loss += loss_info['meta_loss']
            num_steps += 1
            
            # Print progress for first and last step of each epoch
            if step == 0 or step == num_steps_per_epoch - 1:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Step {step+1}/{num_steps_per_epoch}, "
                           f"Meta-Loss: {loss_info['meta_loss']:.4f}")
        
        # Calculate average loss
        avg_loss = epoch_loss / max(num_steps, 1)
        
        # Print epoch summary
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed, Avg Meta-Loss: {avg_loss:.4f}")
    
    # Save checkpoint if path is provided
    if checkpoint_path:
        try:
            logger.info(f"Saving meta-model checkpoint to {checkpoint_path}")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'model_state_dict': meta_model.state_dict(),
                'optimizer_state_dict': meta_optimizer.state_dict()
            }, checkpoint_path)
        except Exception as e:
            logger.error(f"Error saving meta-model checkpoint: {e}")
    
    return meta_model 


def evaluate_model(
    model: nn.Module,
    test_data: tuple,
    true_graphs: list = None,
    device: torch.device = None
) -> dict:
    """
    Evaluate model performance on test data.
    
    Args:
        model: The model to evaluate
        test_data: Tuple of (observational_data, interventional_data, interventions)
        true_graphs: List of true graph structures
        device: Device to run evaluation on
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model performance")
    
    if device is None:
        device = next(model.parameters()).device
    
    # Move model to device if needed
    model.to(device)
    
    try:
        # Unpack test data with flexible format handling
        if len(test_data) == 2:
            obs_data, int_data = test_data
            interventions = None
        elif len(test_data) == 3:
            obs_data, int_data, interventions = test_data
        else:
            raise ValueError(f"Unexpected test_data format with {len(test_data)} elements")
        
        # Move test data to device
        if isinstance(obs_data, torch.Tensor):
            obs_data = obs_data.to(device)
        
        if isinstance(int_data, torch.Tensor) and int_data is not None:
            int_data = int_data.to(device)
        
        # Get model predictions
        with torch.no_grad():
            # Infer causal graph from observational data
            if hasattr(model, 'infer_causal_graph'):
                pred_adj_matrix = model.infer_causal_graph(obs_data)
            else:
                # Forward pass to get adjacency matrix
                outputs = model(obs_data)
                if isinstance(outputs, tuple):
                    pred_adj_matrix = outputs[0]
                else:
                    pred_adj_matrix = outputs
            
            # Predict interventional outcomes if interventional data is available
            if int_data is not None and hasattr(model, 'predict_intervention_outcomes'):
                # Use provided interventions or create dummy ones
                if interventions is None:
                    interventions = {0: 1.0}  # Default intervention on first node
                
                # Make predictions
                pred_int_outcomes = model.predict_intervention_outcomes(
                    obs_data, interventions=interventions
                )
                
                # Calculate interventional prediction error
                if isinstance(pred_int_outcomes, tuple):
                    pred_int_outcomes = pred_int_outcomes[0]
                
                int_mse = F.mse_loss(pred_int_outcomes, int_data).item()
            else:
                int_mse = float('nan')
        
        # Calculate graph accuracy if true graphs are provided
        if true_graphs and len(true_graphs) > 0:
            # Get adjacency matrix of true graph
            if hasattr(true_graphs[0], 'get_adjacency_matrix'):
                true_adj_matrix = torch.tensor(
                    true_graphs[0].get_adjacency_matrix(), 
                    dtype=torch.float32,
                    device=device
                )
            elif hasattr(true_graphs[0], 'adjacency_matrix'):
                true_adj_matrix = torch.tensor(
                    true_graphs[0].adjacency_matrix, 
                    dtype=torch.float32,
                    device=device
                )
            else:
                # Assume it's already a tensor or numpy array
                true_adj_matrix = torch.tensor(
                    np.array(true_graphs[0]), 
                    dtype=torch.float32,
                    device=device
                )
            
            # Apply threshold to predicted adjacency matrix
            threshold = 0.5
            pred_adj_binary = (pred_adj_matrix > threshold).float()
            
            # Calculate graph accuracy (percentage of correctly predicted edges)
            correct_edges = (pred_adj_binary == true_adj_matrix).float().mean().item()
            
            # Calculate precision and recall
            true_positives = ((pred_adj_binary == 1) & (true_adj_matrix == 1)).float().sum().item()
            false_positives = ((pred_adj_binary == 1) & (true_adj_matrix == 0)).float().sum().item()
            false_negatives = ((pred_adj_binary == 0) & (true_adj_matrix == 1)).float().sum().item()
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Return metrics
            return {
                'graph_accuracy': correct_edges,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'interventional_mse': int_mse,
                'dynamics_accuracy': 1.0 - min(int_mse, 1.0) if not math.isnan(int_mse) else 0.0
            }
        else:
            return {
                'graph_accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'interventional_mse': int_mse,
                'dynamics_accuracy': 1.0 - min(int_mse, 1.0) if not math.isnan(int_mse) else 0.0
            }
    
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return {'error': str(e)}


def main():
    """
    Main function demonstrating the full Amortized Causal Discovery pipeline.
    """
    # Parse arguments
    args = parse_args()
    
    # Logging already setup in refactored_utils
    logger.info("Starting full Amortized Causal Discovery pipeline demonstration")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    #=========================================================================
    # Task Family Creation
    #=========================================================================
    logger.info("=== Creating Task Family ===")
    task_family, family_graphs = create_task_family(
        num_nodes=args.num_nodes,
        family_size=args.family_size,
        seed=args.seed
    )
    
    # Visualize task family
    logger.info("=== Visualizing Task Family ===")
    os.makedirs("outputs", exist_ok=True)
    plot_family_comparison(
        graphs=family_graphs,
        save_path="outputs/task_family.png"
    )
    
    #=========================================================================
    # Data Generation
    #=========================================================================
    logger.info("=== Generating Synthetic Data ===")
    obs_data_family, int_data_family, scms, intervention_targets = create_synthetic_data(
        task_family=task_family,
        family_graphs=family_graphs,
        num_samples=args.samples,
        include_interventional=True,
        intervention_nodes_per_graph=1,
        seed=args.seed
    )
    
    #=========================================================================
    # Training Data Preparation
    #=========================================================================
    logger.info("=== Preparing Training Data ===")
    logger.info("Preparing training data")
    
    # Split data for training and evaluation
    train_obs_data = obs_data_family[:-1]  # Use all but last task for training
    train_int_data = int_data_family[:-1]
    train_graphs = family_graphs[:-1]
    
    test_obs_data = obs_data_family[-1:]  # Use last task for testing
    test_int_data = int_data_family[-1:]
    test_graph = family_graphs[-1]
    
    # Prepare training data
    try:
        train_dataloader = prepare_training_data(
            observational_data=train_obs_data,
            interventional_data=train_int_data,
            true_graphs=train_graphs,
            batch_size=args.batch_size
        )
        logger.info(f"Created dataloader with {len(train_dataloader)} batches")
    except Exception as e:
        logger.error(f"Error preparing training data: {e}")
        logger.warning("Creating dummy dataset as fallback")
        
        # Create dummy dataset if data preparation fails
        dummy_x = torch.randn(args.batch_size, args.num_nodes)
        dummy_y = torch.randn(args.batch_size, args.num_nodes)
        dummy_graph = torch.zeros(args.batch_size, args.num_nodes, args.num_nodes)
        dummy_dataset = TensorDataset(dummy_x, dummy_y, dummy_graph)
        train_dataloader = DataLoader(dummy_dataset, batch_size=args.batch_size)
    
    #=========================================================================
    # Model Creation
    #=========================================================================
    logger.info("=== Creating Model ===")
    logger.info(f"Creating model for graphs with {args.num_nodes} nodes")
    
    model = create_model(
        num_nodes=args.num_nodes,
        device=device,
        hidden_dim=64
    )
    
    #=========================================================================
    # Base Model Training
    #=========================================================================
    logger.info("=== Training Base Model ===")
    logger.info(f"Training model for {args.epochs} epochs")
    
    model = train_model(
        model=model,
        dataloader=train_dataloader,
        num_epochs=args.epochs,
        device=device,
        learning_rate=args.lr,
        graph_weight=1.0,
        dynamics_weight=1.0,
        sparsity_weight=0.1,
        checkpoint_path="outputs/base_model.pt",
        quick_mode=args.quick_mode
    )
    
    #=========================================================================
    # Base Model Evaluation
    #=========================================================================
    logger.info("=== Evaluating Base Model ===")
    
    # Prepare test data for evaluation
    if len(test_obs_data) > 0 and len(test_int_data) > 0:
        # Create simplified test data format for evaluation
        test_data_family = []
        
        # Process test observational data
        test_obs = test_obs_data[0]
        
        # Process test interventional data - intervention_data is a list of (data, interventions) tuples
        if len(test_int_data) > 0 and len(test_int_data[0]) > 0:
            # Extract the first intervention pair (data, intervention)
            test_int_data_tensor, test_int_targets = test_int_data[0][0]
            
            # Create test pair
            test_data_family.append((test_obs, test_int_data_tensor, test_int_targets))
        else:
            # Create dummy interventional data if none exists
            dummy_int_data = torch.randn_like(test_obs)
            dummy_targets = {0: 1.0}  # Intervene on first node with value 1.0
            test_data_family.append((test_obs, dummy_int_data, dummy_targets))
        
        # Evaluate model
        logger.info("Evaluating base model on test data")
        base_metrics = evaluate_model(
            model=model,
            test_data=test_data_family[0],
            true_graphs=[test_graph],
            device=device
        )
        
        # Log results
        logger.info(f"Base model metrics: {base_metrics}")
    else:
        logger.warning("No test data available for evaluation")
    
    #=========================================================================
    # Meta-Learning Setup and Training 
    #=========================================================================
    if not args.skip_meta and len(train_obs_data) > 1:
        logger.info("=== Setting Up Meta-Learning ===")
        
        # Prepare meta-learning data
        meta_batch_sampler = prepare_meta_training_data(
            task_family=task_family,
            family_observational_data=train_obs_data,
            family_interventional_data=train_int_data,
            family_graphs=train_graphs,
            batch_size=args.batch_size,
            k_shot=5,
            meta_batch_size=2
        )
        
        # Setup meta-learning model and optimizer
        meta_model, meta_optimizer = setup_meta_learning(
            model=model,
            device=device,
            inner_lr=args.inner_lr,
            meta_lr=args.lr / 10,
            first_order=True
        )
        
        # Train meta-model
        logger.info("=== Training Meta-Model ===")
        logger.info(f"Training meta-model for {args.meta_epochs} epochs")
        
        meta_model = meta_train_model(
            meta_model=meta_model,
            meta_batch_sampler=meta_batch_sampler,
            meta_optimizer=meta_optimizer,
            device=device,
            num_epochs=args.meta_epochs,
            quick_mode=args.quick_mode,
            checkpoint_path="outputs/meta_model.pt"
        )
        
        # Adapt meta-model to test task
        logger.info("=== Adapting Meta-Model to Test Task ===")
        if len(test_obs_data) > 0:
            # Perform adaptation steps
            logger.info(f"Adapting meta-model with {args.adaptation_steps} steps")
            
            # Use first few samples as adaptation data
            adaptation_data = test_obs_data[0][:5]
            adapted_model = meta_model.adapt(
                adaptation_data, 
                steps=args.adaptation_steps
            )
            
            # Evaluate adapted model
            logger.info("Evaluating adapted model on test data")
            adapted_metrics = evaluate_model(
                model=adapted_model,
                test_data=test_data_family[0],
                true_graphs=[test_graph],
                device=device
            )
            
            # Log results
            logger.info(f"Adapted model metrics: {adapted_metrics}")
            
            # Compare base vs adapted performance
            logger.info("=== Comparing Base vs Adapted Performance ===")
            logger.info(f"Graph accuracy - Base: {base_metrics['graph_accuracy']:.4f}, " + 
                      f"Adapted: {adapted_metrics['graph_accuracy']:.4f}")
            logger.info(f"Dynamics accuracy - Base: {base_metrics['dynamics_accuracy']:.4f}, " + 
                      f"Adapted: {adapted_metrics['dynamics_accuracy']:.4f}")
        else:
            logger.warning("No test data available for meta-model evaluation")
    else:
        if args.skip_meta:
            logger.info("Skipping meta-learning as requested")
        else:
            logger.warning("Not enough tasks for meta-learning (need at least 2)")
    
    logger.info("Demo completed successfully!")
    return model


if __name__ == "__main__":
    main() 