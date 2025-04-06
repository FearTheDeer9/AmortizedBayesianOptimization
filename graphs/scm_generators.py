import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import functional as F
import re


class MLP(nn.Module):
    """Simple MLP for nonlinear SCM mechanisms."""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int = 1):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def generate_linear_scm(graph: nx.DiGraph, weight_range: Tuple[float, float] = (-2, 2),
                        noise_std: float = 1.0, seed: Optional[int] = None) -> Dict[str, callable]:
    """Generate linear SCM with specified graph structure.

    Args:
        graph: Directed acyclic graph representing the causal structure
        weight_range: Range for the edge weights
        noise_std: Standard deviation of the Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping node names to their structural equations
    """
    rng = np.random.default_rng(seed)
    mechanisms = {}

    # Get all nodes in sorted order
    nodes = sorted(graph.nodes(), key=lambda x: int(x))

    for node in nodes:
        parents = list(graph.predecessors(node))
        node_str = str(node)

        if not parents:
            # Root node - just noise
            def root_mechanism(n_samples: int = 1, parent_values: Optional[np.ndarray] = None) -> np.ndarray:
                return rng.normal(0, noise_std, size=(n_samples, 1))
            mechanisms[node_str] = root_mechanism
        else:
            # Generate random weights for parents
            weights = rng.uniform(
                weight_range[0], weight_range[1], size=len(parents))

            def mechanism(n_samples: int = 1, parent_values: Optional[np.ndarray] = None) -> np.ndarray:
                if parent_values is None:
                    # If no parent values provided, try to sample without parents
                    return rng.normal(0, noise_std, size=(n_samples, 1))
                if parent_values.shape[1] != len(parents):
                    # If wrong number of parents, use what we have
                    actual_parents = min(parent_values.shape[1], len(parents))
                    return parent_values[:, :actual_parents] @ weights[:actual_parents].reshape(-1, 1) + rng.normal(0, noise_std, size=(n_samples, 1))
                # Normal case: use all parents
                return parent_values @ weights.reshape(-1, 1) + rng.normal(0, noise_std, size=(n_samples, 1))

            mechanisms[node_str] = mechanism

    return mechanisms


def generate_nonlinear_scm(graph: nx.DiGraph, mechanism_type: str = 'mlp',
                           hidden_dims: List[int] = [50, 20], noise_std: float = 1.0,
                           seed: Optional[int] = None) -> Dict[str, callable]:
    """Generate nonlinear SCM with specified graph structure using MLPs.

    Args:
        graph: Directed acyclic graph representing the causal structure
        mechanism_type: Type of nonlinear mechanism ('mlp' or 'polynomial')
        hidden_dims: Hidden layer dimensions for MLP
        noise_std: Standard deviation of the Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping node names to their structural equations
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed if seed is not None else rng.integers(0, 2**32))
    mechanisms = {}

    # Get all nodes in sorted order
    nodes = sorted(graph.nodes(), key=lambda x: int(x))

    for node in nodes:
        parents = list(graph.predecessors(node))
        node_str = str(node)

        if not parents:
            # Root node - just noise
            def root_mechanism(n_samples: int = 1, parent_values: Optional[np.ndarray] = None) -> np.ndarray:
                return rng.normal(0, noise_std, size=(n_samples, 1))
            mechanisms[node_str] = root_mechanism
        else:
            if mechanism_type == 'mlp':
                # Create MLP for this node
                mlp = MLP(len(parents), hidden_dims)

                def mechanism(n_samples: int = 1, parent_values: Optional[np.ndarray] = None) -> np.ndarray:
                    if parent_values is None:
                        # If no parent values provided, try to sample without parents
                        return rng.normal(0, noise_std, size=(n_samples, 1))
                    if parent_values.shape[1] != len(parents):
                        # If wrong number of parents, use what we have
                        actual_parents = min(
                            parent_values.shape[1], len(parents))
                        parent_values = parent_values[:, :actual_parents]
                        # Pad with zeros if needed
                        if actual_parents < len(parents):
                            padding = np.zeros(
                                (parent_values.shape[0], len(parents) - actual_parents))
                            parent_values = np.hstack([parent_values, padding])
                    # Convert to tensor and pass through MLP
                    with torch.no_grad():
                        output = mlp(torch.FloatTensor(parent_values))
                    # Add noise
                    return output.numpy() + rng.normal(0, noise_std, size=(n_samples, 1))

            elif mechanism_type == 'polynomial':
                # Generate random polynomial coefficients
                coeffs = rng.normal(0, 1, size=(
                    len(parents), 3))  # Quadratic terms

                def mechanism(n_samples: int = 1, parent_values: Optional[np.ndarray] = None) -> np.ndarray:
                    if parent_values is None:
                        # If no parent values provided, try to sample without parents
                        return rng.normal(0, noise_std, size=(n_samples, 1))
                    if parent_values.shape[1] != len(parents):
                        # If wrong number of parents, use what we have
                        actual_parents = min(
                            parent_values.shape[1], len(parents))
                        parent_values = parent_values[:, :actual_parents]
                        # Pad with zeros if needed
                        if actual_parents < len(parents):
                            padding = np.zeros(
                                (parent_values.shape[0], len(parents) - actual_parents))
                            parent_values = np.hstack([parent_values, padding])
                    # Compute polynomial terms
                    linear = parent_values @ coeffs[:, 0:1]
                    quadratic = (parent_values ** 2) @ coeffs[:, 1:2]
                    cubic = (parent_values ** 3) @ coeffs[:, 2:3]
                    # Combine terms and add noise
                    return linear + quadratic + cubic + rng.normal(0, noise_std, size=(n_samples, 1))

            else:
                raise ValueError(f"Unknown mechanism type: {mechanism_type}")

            mechanisms[node_str] = mechanism

    return mechanisms


def sample_observational(scm: Dict[str, callable], n_samples: int,
                         seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Generate observational samples from the SCM.

    Args:
        scm: Dictionary of structural equations
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping node names to their samples
    """
    rng = np.random.default_rng(seed)
    samples = {}

    # Get topological order from node names
    nodes = sorted(scm.keys(), key=lambda x: int(x))  # Sort by numeric value

    # Create a dependency graph based on parent requirements
    dep_graph = nx.DiGraph()
    dep_graph.add_nodes_from(nodes)

    # First pass: identify root nodes (those that don't need parents)
    root_nodes = []
    for node in nodes:
        try:
            scm[node](1)  # Try sampling without parents
            root_nodes.append(node)
        except ValueError:
            pass

    # Second pass: identify dependencies for non-root nodes
    for node in nodes:
        if node not in root_nodes:
            try:
                scm[node](1)  # This should fail
            except ValueError as e:
                match = re.search(r"requires (\d+) parent values", str(e))
                if match:
                    num_parents = int(match.group(1))
                    # Add edges from previous nodes as parents
                    node_idx = nodes.index(node)
                    # Get all possible parent nodes
                    possible_parents = nodes[:node_idx]
                    # If we have more possible parents than needed, take the most recent ones
                    if len(possible_parents) > num_parents:
                        possible_parents = possible_parents[-num_parents:]
                    for parent in possible_parents:
                        dep_graph.add_edge(parent, node)

    # Sample in topological order
    for node in nx.topological_sort(dep_graph):
        if node in root_nodes:
            samples[node] = scm[node](n_samples)
        else:
            parent_nodes = list(dep_graph.predecessors(node))
            if parent_nodes:
                parent_values = np.hstack([samples[p] for p in parent_nodes])
                samples[node] = scm[node](n_samples, parent_values)
            else:
                # If no parents found but node is not a root, try sampling without parents
                samples[node] = scm[node](n_samples)

    return samples


def sample_interventional(scm, node, value, n_samples=1000, seed=None):
    """Generate interventional samples for a specific node and value.

    Args:
        scm: Structural causal model
        node: Node index to intervene on
        value: Value to set the node to
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping node names to their samples under the intervention
    """
    # Convert node index to string for SCM compatibility
    node_str = str(node)

    # Get all nodes in sorted order
    nodes = sorted(scm.keys(), key=lambda x: int(x))

    # Create dependency graph based on parent requirements
    dependency_graph = nx.DiGraph()
    dependency_graph.add_nodes_from(nodes)

    # First pass: identify root nodes (those that don't need parents)
    root_nodes = []
    for curr_node in nodes:
        try:
            scm[curr_node](1)  # Try sampling without parents
            root_nodes.append(curr_node)
        except (ValueError, TypeError):
            pass

    # Second pass: identify dependencies for non-root nodes
    for curr_node in nodes:
        if curr_node not in root_nodes:
            try:
                scm[curr_node](1)  # This should fail
            except (ValueError, TypeError) as e:
                # Add edges from previous nodes as parents
                node_idx = nodes.index(curr_node)
                # Get all possible parent nodes
                possible_parents = nodes[:node_idx]
                # Add edges from all possible parents
                for parent in possible_parents:
                    dependency_graph.add_edge(parent, curr_node)

    # Initialize samples dictionary
    samples = {}

    # Set intervened node to specified value
    samples[node_str] = np.full(n_samples, value)

    # Sample remaining nodes in topological order
    for curr_node in nx.topological_sort(dependency_graph):
        if curr_node == node_str:
            continue  # Skip intervened node

        # Get parent values
        parent_nodes = list(dependency_graph.predecessors(curr_node))
        if curr_node in root_nodes:
            samples[curr_node] = scm[curr_node](n_samples)
        elif parent_nodes:
            parent_values = np.hstack([samples[p] for p in parent_nodes])
            samples[curr_node] = scm[curr_node](n_samples, parent_values)
        else:
            # If no parents found but node is not a root, try sampling without parents
            samples[curr_node] = scm[curr_node](n_samples)

    return samples
