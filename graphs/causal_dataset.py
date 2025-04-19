import numpy as np
import os
import json
import pickle
from typing import Dict, Optional, Tuple, List, Any, Union
import networkx as nx
from graphs.scm_generators import sample_observational, sample_interventional, NoiseConfig


class CausalDataset:
    """Dataset class that holds observational and interventional data.

    This class handles the storage and organization of both observational and interventional data.
    """

    def __init__(self, graph: nx.DiGraph, scm: Dict[str, callable],
                 n_obs: int = 1000, n_int: int = 100,
                 noise_config: Optional[NoiseConfig] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 seed: Optional[int] = None):
        """Initialize with graph, SCM, and generate initial data.

        Args:
            graph: Directed acyclic graph representing the causal structure
            scm: Dictionary of structural equations
            n_obs: Number of observational samples to generate
            n_int: Number of interventional samples per intervention
            noise_config: Configuration for the noise distribution
            metadata: Additional metadata about the dataset
            seed: Random seed for reproducibility
        """
        self.graph = graph
        self.scm = scm
        self.n_obs = n_obs
        self.n_int = n_int
        self.seed = seed
        self.noise_config = noise_config
        self.metadata = metadata or {}

        # Store mechanism types and parameters if available
        if 'mechanism_types' not in self.metadata:
            self.metadata['mechanism_types'] = {}

        # Initialize data storage
        self.obs_data = {}
        self.int_data = {}  # {node_idx: {value: samples}}

        # Generate initial observational data if SCM is provided
        if scm is not None:
            self.obs_data = sample_observational(scm, n_obs, seed=seed)

    def add_intervention(self, node, value, n_samples=None):
        """Add intervention samples to the dataset.

        Args:
            node: Node index to intervene on (integer)
            value: Value to set the node to (float)
            n_samples: Number of samples to generate (defaults to self.n_int)
        """
        if n_samples is None:
            n_samples = self.n_int

        # Convert node index to string for internal storage
        node_str = str(node)

        # Generate samples using the SCM
        samples = sample_interventional(
            self.scm, node, value, n_samples, self.seed)

        # Store samples in int_data dictionary
        if node_str not in self.int_data:
            self.int_data[node_str] = {}
        self.int_data[node_str][value] = samples

    def add_multiple_interventions(self, nodes, values, n_samples=None):
        """Add multiple intervention samples to the dataset.

        Args:
            nodes: List of node indices to intervene on
            values: List of values to set the nodes to
            n_samples: Number of samples to generate (defaults to self.n_int)
        """
        if n_samples is None:
            n_samples = self.n_int

        if len(nodes) != len(values):
            raise ValueError("Number of nodes must match number of values")

        # Convert nodes to strings
        node_strs = [str(node) for node in nodes]

        # Create a composite key for the multi-intervention
        intervention_key = tuple(zip(node_strs, values))

        # TODO: Implement multiple intervention sampling
        # For now, we'll just do a simple approach of intervening on each node separately
        for node, value in zip(nodes, values):
            self.add_intervention(node, value, n_samples)

    def get_obs_data(self) -> Dict[str, np.ndarray]:
        """Return observational data."""
        return self.obs_data

    def get_int_data(self) -> Dict[str, Dict[float, Dict[str, np.ndarray]]]:
        """Return interventional data."""
        return self.int_data

    def get_all_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[float, Dict[str, np.ndarray]]]]:
        """Return both observational and interventional data."""
        return self.obs_data, self.int_data

    def get_standardized_data(self) -> Dict[str, Any]:
        """Return data in a standardized format.

        Returns:
            Dictionary with standardized data format containing:
            - observational_data: Dict mapping node names to samples
            - interventional_data: Dict mapping (node, value) tuples to samples
            - graph: Networkx graph structure
            - metadata: Additional dataset information
        """
        # Convert graph to adjacency list for serialization
        if hasattr(self.graph, 'adjacency'):
            adjacency_list = dict(self.graph.adjacency())
            # Convert node objects to strings
            graph_data = {str(node): {str(neighbor): data for neighbor, data in neighbors.items()}
                          for node, neighbors in adjacency_list.items()}
        else:
            graph_data = None

        # Create standardized data structure
        return {
            'observational_data': self.obs_data,
            'interventional_data': self.int_data,
            'graph': graph_data,
            'metadata': self.metadata,
            'noise_config': self.noise_config.__dict__ if self.noise_config else None,
            'n_obs': self.n_obs,
            'n_int': self.n_int,
            'seed': self.seed
        }

    def get_node_data(self, node: str) -> Tuple[np.ndarray, Dict[float, np.ndarray]]:
        """Get all data for a specific node.

        Args:
            node: Node to get data for

        Returns:
            Tuple of (observational_data, interventional_data)
            where interventional_data maps intervention values to samples
        """
        obs_data = self.obs_data[node]
        int_data = {}

        # Extract intervention data for the specific node
        if node in self.int_data:
            int_data = {value: data[node]
                        for value, data in self.int_data[node].items()}

        return obs_data, int_data

    def get_intervention_values(self, node: str) -> List[float]:
        """Get all intervention values that have been tried for a node.

        Args:
            node: Node to get intervention values for

        Returns:
            List of intervention values
        """
        if node not in self.int_data:
            return []
        return list(self.int_data[node].keys())

    def get_intervention_samples(self, node, value):
        """Get intervention samples from the dataset.

        Args:
            node: Node index to get samples for (integer)
            value: Intervention value to get samples for (float)

        Returns:
            Dictionary mapping node names to their samples under the intervention
        """
        # Convert node index to string for internal storage
        node_str = str(node)

        if node_str not in self.int_data or value not in self.int_data[node_str]:
            raise KeyError(
                f"No samples found for node {node} with value {value}")

        return self.int_data[node_str][value]

    def get_valid_interventions(self) -> List[str]:
        """Get list of nodes that can be intervened on.

        Returns:
            List of node names that can be intervened on
        """
        return [str(node) for node in self.graph.nodes()]

    def get_num_nodes(self) -> int:
        """Get the number of nodes in the graph.

        Returns:
            Number of nodes
        """
        return len(self.graph.nodes())

    def get_node_range(self) -> Tuple[float, float]:
        """Get the range of values for nodes.

        Returns:
            Tuple of (min_value, max_value)
        """
        # This could be made more sophisticated based on the SCM
        return (-10.0, 10.0)

    def get_held_out_data(self, n_samples: int = 1000) -> Dict[str, np.ndarray]:
        """Generate held-out data for evaluation.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Dictionary mapping node names to their samples
        """
        return sample_observational(self.scm, n_samples, self.seed)

    def save(self, path: str, save_scm: bool = False) -> None:
        """Save dataset to disk.

        Args:
            path: Path to save the dataset
            save_scm: Whether to save the SCM (functions not serializable by default)
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Create a dictionary of all serializable data
        data_to_save = {
            'obs_data': self.obs_data,
            'int_data': self.int_data,
            'n_obs': self.n_obs,
            'n_int': self.n_int,
            'seed': self.seed,
            'metadata': self.metadata,
            'noise_config': self.noise_config.__dict__ if self.noise_config else None
        }

        # Save graph separately as adjacency list
        if hasattr(self.graph, 'adjacency'):
            data_to_save['graph_adjacency'] = dict(self.graph.adjacency())

        # Save with pickle format
        with open(path, 'wb') as f:
            pickle.dump(data_to_save, f)

        # If requested, try to save SCM in a separate file
        if save_scm:
            scm_path = f"{os.path.splitext(path)[0]}_scm.pkl"
            try:
                with open(scm_path, 'wb') as f:
                    pickle.dump(self.scm, f)
            except Exception as e:
                print(f"Warning: Could not save SCM: {e}")

    @classmethod
    def load(cls, path: str, load_scm: bool = False) -> 'CausalDataset':
        """Load dataset from disk.

        Args:
            path: Path to load the dataset from
            load_scm: Whether to try loading the SCM

        Returns:
            Loaded CausalDataset object
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        # Reconstruct graph from adjacency list
        if 'graph_adjacency' in data:
            graph = nx.DiGraph(data['graph_adjacency'])
        else:
            graph = nx.DiGraph()

        # Try to load SCM if requested
        scm = None
        if load_scm:
            scm_path = f"{os.path.splitext(path)[0]}_scm.pkl"
            if os.path.exists(scm_path):
                try:
                    with open(scm_path, 'rb') as f:
                        scm = pickle.load(f)
                except Exception as e:
                    print(f"Warning: Could not load SCM: {e}")

        # Reconstruct noise config if available
        noise_config = None
        if 'noise_config' in data and data['noise_config']:
            noise_config = NoiseConfig(
                noise_type=data['noise_config'].get('noise_type', 'gaussian'),
                params=data['noise_config'].get('params', {}),
                seed=data['noise_config'].get('seed')
            )

        # Create new instance without generating new data
        dataset = cls.__new__(cls)
        dataset.graph = graph
        dataset.scm = scm
        dataset.n_obs = data.get('n_obs', 1000)
        dataset.n_int = data.get('n_int', 100)
        dataset.noise_config = noise_config
        dataset.metadata = data.get('metadata', {})
        dataset.seed = data.get('seed')
        dataset.obs_data = data.get('obs_data', {})
        dataset.int_data = data.get('int_data', {})

        return dataset

    def __str__(self) -> str:
        """String representation of the dataset."""
        nodes = list(self.graph.nodes())
        edges = list(self.graph.edges())
        int_nodes = list(self.int_data.keys())

        return (f"CausalDataset: {len(nodes)} nodes, {len(edges)} edges\n"
                f"Observational samples: {self.n_obs}\n"
                f"Interventional nodes: {len(int_nodes)}\n"
                f"Metadata: {self.metadata}")
