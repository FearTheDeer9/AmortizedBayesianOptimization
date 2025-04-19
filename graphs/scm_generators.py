import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
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


class NoiseConfig:
    """Configuration for noise distributions in SCMs."""

    def __init__(self, noise_type: str = 'gaussian',
                 params: Dict[str, Any] = None,
                 seed: Optional[int] = None):
        """Initialize noise configuration.

        Args:
            noise_type: Type of noise ('gaussian', 'uniform', or 'heteroskedastic')
            params: Parameters for the noise distribution
            seed: Random seed for reproducibility
        """
        self.noise_type = noise_type
        self.params = params or {}
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Set default parameters if not provided
        if noise_type == 'gaussian' and 'std' not in self.params:
            self.params['std'] = 1.0
        elif noise_type == 'uniform' and ('low' not in self.params or 'high' not in self.params):
            self.params['low'] = -1.0
            self.params['high'] = 1.0
        elif noise_type == 'heteroskedastic' and 'base_std' not in self.params:
            self.params['base_std'] = 0.5
            self.params['scale_factor'] = 0.5

    def generate(self, size: Tuple[int, ...], parent_values: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate noise samples based on the configured distribution.

        Args:
            size: Size of the noise tensor to generate
            parent_values: Parent values for heteroskedastic noise

        Returns:
            Noise samples of the specified size
        """
        if self.noise_type == 'gaussian':
            return self.rng.normal(0, self.params['std'], size=size)

        elif self.noise_type == 'uniform':
            return self.rng.uniform(self.params['low'], self.params['high'], size=size)

        elif self.noise_type == 'heteroskedastic':
            if parent_values is None:
                return self.rng.normal(0, self.params['base_std'], size=size)

            # Scale noise standard deviation based on parent values
            base_std = self.params['base_std']
            scale_factor = self.params['scale_factor']

            # Use the mean of absolute parent values to scale noise
            if len(parent_values.shape) > 1 and parent_values.shape[1] > 0:
                parent_scale = scale_factor * \
                    np.mean(np.abs(parent_values), axis=1,
                            keepdims=True) + base_std
                return self.rng.normal(0, 1, size=size) * parent_scale
            else:
                return self.rng.normal(0, base_std, size=size)

        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")


def generate_linear_scm(graph: nx.DiGraph, weight_range: Tuple[float, float] = (-2, 2),
                        noise_config: Optional[NoiseConfig] = None,
                        seed: Optional[int] = None) -> Dict[str, callable]:
    """Generate linear SCM with specified graph structure.

    Args:
        graph: Directed acyclic graph representing the causal structure
        weight_range: Range for the edge weights
        noise_config: Configuration for noise distribution
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping node names to their structural equations
    """
    rng = np.random.default_rng(seed)
    mechanisms = {}

    # Use default Gaussian noise if not specified
    if noise_config is None:
        noise_config = NoiseConfig('gaussian', {'std': 1.0}, seed)

    # Get all nodes in sorted order
    nodes = sorted(graph.nodes(), key=lambda x: int(x))

    for node in nodes:
        parents = list(graph.predecessors(node))
        node_str = str(node)

        if not parents:
            # Root node - just noise
            def root_mechanism(n_samples: int = 1, parent_values: Optional[np.ndarray] = None) -> np.ndarray:
                return noise_config.generate((n_samples, 1), None)
            mechanisms[node_str] = root_mechanism
        else:
            # Generate random weights for parents
            weights = rng.uniform(
                weight_range[0], weight_range[1], size=len(parents))

            def mechanism(n_samples: int = 1, parent_values: Optional[np.ndarray] = None) -> np.ndarray:
                if parent_values is None:
                    # If no parent values provided, try to sample without parents
                    return noise_config.generate((n_samples, 1), None)
                if parent_values.shape[1] != len(parents):
                    # If wrong number of parents, use what we have
                    actual_parents = min(parent_values.shape[1], len(parents))
                    linear_component = parent_values[:,
                                                     :actual_parents] @ weights[:actual_parents].reshape(-1, 1)
                    return linear_component + noise_config.generate((n_samples, 1), parent_values[:, :actual_parents])
                # Normal case: use all parents
                linear_component = parent_values @ weights.reshape(-1, 1)
                return linear_component + noise_config.generate((n_samples, 1), parent_values)

            mechanisms[node_str] = mechanism

    return mechanisms


def generate_polynomial_scm(graph: nx.DiGraph, max_degree: int = 3,
                            noise_config: Optional[NoiseConfig] = None,
                            seed: Optional[int] = None) -> Dict[str, callable]:
    """Generate polynomial SCM with specified graph structure.

    Args:
        graph: Directed acyclic graph representing the causal structure
        max_degree: Maximum polynomial degree
        noise_config: Configuration for noise distribution
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping node names to their structural equations
    """
    rng = np.random.default_rng(seed)
    mechanisms = {}

    # Use default Gaussian noise if not specified
    if noise_config is None:
        noise_config = NoiseConfig('gaussian', {'std': 1.0}, seed)

    # Get all nodes in sorted order
    nodes = sorted(graph.nodes(), key=lambda x: int(x))

    for node in nodes:
        parents = list(graph.predecessors(node))
        node_str = str(node)

        if not parents:
            # Root node - just noise
            def root_mechanism(n_samples: int = 1, parent_values: Optional[np.ndarray] = None) -> np.ndarray:
                return noise_config.generate((n_samples, 1), None)
            mechanisms[node_str] = root_mechanism
        else:
            # Generate random polynomial coefficients for each parent and each degree
            coeffs = {}
            for degree in range(1, max_degree + 1):
                coeffs[degree] = rng.normal(0, 1.0/degree, size=len(parents))

            def mechanism(n_samples: int = 1, parent_values: Optional[np.ndarray] = None) -> np.ndarray:
                if parent_values is None:
                    # If no parent values provided, try to sample without parents
                    return noise_config.generate((n_samples, 1), None)

                if parent_values.shape[1] != len(parents):
                    # If wrong number of parents, use what we have
                    actual_parents = min(parent_values.shape[1], len(parents))
                    parent_values = parent_values[:, :actual_parents]
                    # Initialize result
                    result = np.zeros((n_samples, 1))
                    # Add polynomial terms
                    for degree in range(1, max_degree + 1):
                        result += (parent_values **
                                   degree) @ coeffs[degree][:actual_parents].reshape(-1, 1)
                    # Add noise
                    return result + noise_config.generate((n_samples, 1), parent_values)

                # Normal case: use all parents
                result = np.zeros((n_samples, 1))
                # Add polynomial terms
                for degree in range(1, max_degree + 1):
                    result += (parent_values **
                               degree) @ coeffs[degree].reshape(-1, 1)
                # Add noise
                return result + noise_config.generate((n_samples, 1), parent_values)

            mechanisms[node_str] = mechanism

    return mechanisms


def generate_exponential_scm(graph: nx.DiGraph, base_range: Tuple[float, float] = (0.5, 1.5),
                             scale_range: Tuple[float, float] = (0.1, 0.5),
                             noise_config: Optional[NoiseConfig] = None,
                             seed: Optional[int] = None) -> Dict[str, callable]:
    """Generate SCM with exponential relationships between nodes.

    Args:
        graph: Directed acyclic graph representing the causal structure
        base_range: Range for the exponential base
        scale_range: Range for the scaling factor
        noise_config: Configuration for noise distribution
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping node names to their structural equations
    """
    rng = np.random.default_rng(seed)
    mechanisms = {}

    # Use default Gaussian noise if not specified
    if noise_config is None:
        noise_config = NoiseConfig('gaussian', {'std': 1.0}, seed)

    # Get all nodes in sorted order
    nodes = sorted(graph.nodes(), key=lambda x: int(x))

    for node in nodes:
        parents = list(graph.predecessors(node))
        node_str = str(node)

        if not parents:
            # Root node - just noise
            def root_mechanism(n_samples: int = 1, parent_values: Optional[np.ndarray] = None) -> np.ndarray:
                return noise_config.generate((n_samples, 1), None)
            mechanisms[node_str] = root_mechanism
        else:
            # Generate random exponential parameters
            bases = rng.uniform(
                base_range[0], base_range[1], size=len(parents))
            scales = rng.uniform(
                scale_range[0], scale_range[1], size=len(parents))
            # Random sign for each parent
            signs = rng.choice([-1, 1], size=len(parents))

            def mechanism(n_samples: int = 1, parent_values: Optional[np.ndarray] = None) -> np.ndarray:
                if parent_values is None:
                    # If no parent values provided, try to sample without parents
                    return noise_config.generate((n_samples, 1), None)

                if parent_values.shape[1] != len(parents):
                    # If wrong number of parents, use what we have
                    actual_parents = min(parent_values.shape[1], len(parents))
                    parent_values = parent_values[:, :actual_parents]
                    # Apply exponential function to each parent
                    exp_values = np.zeros((n_samples, 1))
                    for i in range(actual_parents):
                        parent_val = parent_values[:, i:i+1]
                        exp_values += signs[i] * scales[i] * \
                            (bases[i] ** parent_val - 1.0)
                    # Add noise
                    return exp_values + noise_config.generate((n_samples, 1), parent_values)

                # Normal case: use all parents
                exp_values = np.zeros((n_samples, 1))
                for i in range(len(parents)):
                    parent_val = parent_values[:, i:i+1]
                    exp_values += signs[i] * scales[i] * \
                        (bases[i] ** parent_val - 1.0)
                # Add noise
                return exp_values + noise_config.generate((n_samples, 1), parent_values)

            mechanisms[node_str] = mechanism

    return mechanisms


def generate_sinusoidal_scm(graph: nx.DiGraph,
                            freq_range: Tuple[float, float] = (0.5, 2.0),
                            amp_range: Tuple[float, float] = (0.5, 1.5),
                            noise_config: Optional[NoiseConfig] = None,
                            seed: Optional[int] = None) -> Dict[str, callable]:
    """Generate SCM with sinusoidal relationships between nodes.

    Args:
        graph: Directed acyclic graph representing the causal structure
        freq_range: Range for frequency parameters
        amp_range: Range for amplitude parameters
        noise_config: Configuration for noise distribution
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping node names to their structural equations
    """
    rng = np.random.default_rng(seed)
    mechanisms = {}

    # Use default Gaussian noise if not specified
    if noise_config is None:
        noise_config = NoiseConfig('gaussian', {'std': 1.0}, seed)

    # Get all nodes in sorted order
    nodes = sorted(graph.nodes(), key=lambda x: int(x))

    for node in nodes:
        parents = list(graph.predecessors(node))
        node_str = str(node)

        if not parents:
            # Root node - just noise
            def root_mechanism(n_samples: int = 1, parent_values: Optional[np.ndarray] = None) -> np.ndarray:
                return noise_config.generate((n_samples, 1), None)
            mechanisms[node_str] = root_mechanism
        else:
            # Generate random sinusoidal parameters
            frequencies = rng.uniform(
                freq_range[0], freq_range[1], size=len(parents))
            amplitudes = rng.uniform(
                amp_range[0], amp_range[1], size=len(parents))
            phases = rng.uniform(0, 2 * np.pi, size=len(parents))

            def mechanism(n_samples: int = 1, parent_values: Optional[np.ndarray] = None) -> np.ndarray:
                if parent_values is None:
                    # If no parent values provided, try to sample without parents
                    return noise_config.generate((n_samples, 1), None)

                if parent_values.shape[1] != len(parents):
                    # If wrong number of parents, use what we have
                    actual_parents = min(parent_values.shape[1], len(parents))
                    parent_values = parent_values[:, :actual_parents]
                    # Apply sinusoidal function to each parent
                    sin_values = np.zeros((n_samples, 1))
                    for i in range(actual_parents):
                        parent_val = parent_values[:, i:i+1]
                        sin_values += amplitudes[i] * \
                            np.sin(frequencies[i] * parent_val + phases[i])
                    # Add noise
                    return sin_values + noise_config.generate((n_samples, 1), parent_values)

                # Normal case: use all parents
                sin_values = np.zeros((n_samples, 1))
                for i in range(len(parents)):
                    parent_val = parent_values[:, i:i+1]
                    sin_values += amplitudes[i] * \
                        np.sin(frequencies[i] * parent_val + phases[i])
                # Add noise
                return sin_values + noise_config.generate((n_samples, 1), parent_values)

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

    # Create noise configuration
    noise_config = NoiseConfig('gaussian', {'std': noise_std}, seed)

    # Get all nodes in sorted order
    nodes = sorted(graph.nodes(), key=lambda x: int(x))

    for node in nodes:
        parents = list(graph.predecessors(node))
        node_str = str(node)

        if not parents:
            # Root node - just noise
            def root_mechanism(n_samples: int = 1, parent_values: Optional[np.ndarray] = None) -> np.ndarray:
                return noise_config.generate((n_samples, 1), None)
            mechanisms[node_str] = root_mechanism
        else:
            if mechanism_type == 'mlp':
                # Create MLP for this node
                mlp = MLP(len(parents), hidden_dims)

                def mechanism(n_samples: int = 1, parent_values: Optional[np.ndarray] = None) -> np.ndarray:
                    if parent_values is None:
                        # If no parent values provided, try to sample without parents
                        return noise_config.generate((n_samples, 1), None)
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
                    return output.numpy() + noise_config.generate((n_samples, 1), parent_values)

            elif mechanism_type == 'polynomial':
                # Generate random polynomial coefficients
                coeffs = rng.normal(0, 1, size=(
                    len(parents), 3))  # Quadratic terms

                def mechanism(n_samples: int = 1, parent_values: Optional[np.ndarray] = None) -> np.ndarray:
                    if parent_values is None:
                        # If no parent values provided, try to sample without parents
                        return noise_config.generate((n_samples, 1), None)
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
                    return linear + quadratic + cubic + noise_config.generate((n_samples, 1), parent_values)

            else:
                raise ValueError(f"Unknown mechanism type: {mechanism_type}")

            mechanisms[node_str] = mechanism

    return mechanisms


def generate_mixed_scm(graph: nx.DiGraph,
                       mechanism_map: Optional[Dict[str, str]] = None,
                       noise_config: Optional[NoiseConfig] = None,
                       seed: Optional[int] = None) -> Dict[str, callable]:
    """Generate SCM with mixed mechanism types for different nodes.

    Args:
        graph: Directed acyclic graph representing the causal structure
        mechanism_map: Dictionary mapping node IDs to mechanism types
                      ('linear', 'polynomial', 'exponential', 'sinusoidal', 'mlp')
        noise_config: Configuration for noise distribution
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping node names to their structural equations
    """
    rng = np.random.default_rng(seed)
    mechanisms = {}

    # Use default Gaussian noise if not specified
    if noise_config is None:
        noise_config = NoiseConfig('gaussian', {'std': 1.0}, seed)

    # Get all nodes in sorted order
    nodes = sorted(graph.nodes(), key=lambda x: int(x))

    # If no mechanism map is provided, randomly assign mechanisms
    if mechanism_map is None:
        mechanism_types = ['linear', 'polynomial',
                           'exponential', 'sinusoidal', 'mlp']
        mechanism_map = {str(node): rng.choice(mechanism_types)
                         for node in nodes}

    # Generate temporary SCMs for each mechanism type
    linear_scm = generate_linear_scm(
        graph, seed=seed, noise_config=noise_config)
    poly_scm = generate_polynomial_scm(
        graph, seed=seed, noise_config=noise_config)
    exp_scm = generate_exponential_scm(
        graph, seed=seed, noise_config=noise_config)
    sin_scm = generate_sinusoidal_scm(
        graph, seed=seed, noise_config=noise_config)
    mlp_scm = generate_nonlinear_scm(graph, mechanism_type='mlp', seed=seed)

    # Assign mechanisms based on the mechanism map
    for node in nodes:
        node_str = str(node)
        mech_type = mechanism_map.get(node_str, 'linear')  # Default to linear

        if mech_type == 'linear':
            mechanisms[node_str] = linear_scm[node_str]
        elif mech_type == 'polynomial':
            mechanisms[node_str] = poly_scm[node_str]
        elif mech_type == 'exponential':
            mechanisms[node_str] = exp_scm[node_str]
        elif mech_type == 'sinusoidal':
            mechanisms[node_str] = sin_scm[node_str]
        elif mech_type == 'mlp':
            mechanisms[node_str] = mlp_scm[node_str]
        else:
            raise ValueError(f"Unknown mechanism type: {mech_type}")

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


def sample_interventional(scm, node, value, n_samples=1000, seed=None, graph=None):
    """Generate interventional samples for a specific node and value.

    Args:
        scm: Structural causal model
        node: Node index to intervene on
        value: Value to set the node to
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        graph: Optional causal graph (networkx.DiGraph) defining the DAG structure.
               If not provided, will try to infer from the SCM (not recommended).

    Returns:
        Dictionary mapping node names to their samples under the intervention
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Convert node index to string for SCM compatibility
    node_str = str(node)

    # Initialize samples dictionary
    samples = {}

    # Set intervened node to specified value
    samples[node_str] = np.full((n_samples, 1), value)

    if graph is not None:
        # Use the provided graph structure (recommended approach)
        # Get topological ordering from the graph
        try:
            topo_order = list(nx.topological_sort(graph))
        except:
            raise ValueError("Graph must be a directed acyclic graph (DAG)")

        # Sample remaining nodes in topological order
        for curr_node in topo_order:
            curr_node_str = str(curr_node)

            # Skip intervened node
            if curr_node_str == node_str:
                continue

            # Get parent nodes from the graph
            parent_nodes = list(graph.predecessors(curr_node))
            parent_str = [str(p) for p in parent_nodes]

            if not parent_nodes:
                # Node has no parents, sample from its mechanism
                samples[curr_node_str] = scm[curr_node_str](n_samples)
            else:
                # Node has parents, get their values
                parent_values = np.hstack([samples[p] for p in parent_str])
                # Apply mechanism with parent values
                samples[curr_node_str] = scm[curr_node_str](
                    n_samples, parent_values)
    else:
        # LEGACY APPROACH: Try to infer the dependency graph (not recommended)
        # Warning: This may not correctly identify the true causal structure
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

        # Print warning about legacy approach
        warnings_msg = (
            "WARNING: Using legacy approach to infer causal graph structure. "
            "This may not correctly identify the true causal dependencies. "
            "For correct interventional effects, provide the 'graph' parameter."
        )
        print(warnings_msg)

    return samples
