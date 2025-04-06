import numpy as np
from typing import Dict, Optional, Tuple, List
import networkx as nx
from graphs.scm_generators import sample_observational, sample_interventional


class CausalDataset:
    """Dataset class that holds observational and interventional data.

    This class is designed to be used as a data management component within CausalEnvironment.
    It handles the storage and organization of both observational and interventional data,
    while CausalEnvironment handles the broader experimental setup and graph structure.
    """

    def __init__(self, graph: nx.DiGraph, scm: Dict[str, callable],
                 n_obs: int = 1000, n_int: int = 100, seed: Optional[int] = None):
        """Initialize with graph, SCM, and generate initial data.

        Args:
            graph: Directed acyclic graph representing the causal structure
            scm: Dictionary of structural equations
            n_obs: Number of observational samples to generate
            n_int: Number of interventional samples per intervention
            seed: Random seed for reproducibility
        """
        self.graph = graph
        self.scm = scm
        self.n_obs = n_obs
        self.n_int = n_int
        self.seed = seed

        # Generate initial observational data
        self.obs_data = sample_observational(scm, n_obs, seed=seed)

        # Initialize interventional data storage
        self.int_data = {}  # {node_idx: {value: samples}}

    def add_intervention(self, node, value, n_samples=1000):
        """Add intervention samples to the dataset.

        Args:
            node: Node index to intervene on (integer)
            value: Value to set the node to (float)
            n_samples: Number of samples to generate
        """
        # Convert node index to string for internal storage
        node_str = str(node)

        # Generate samples using the SCM
        samples = sample_interventional(
            self.scm, node, value, n_samples, self.seed)

        # Store samples in int_data dictionary
        if node_str not in self.int_data:
            self.int_data[node_str] = {}
        self.int_data[node_str][value] = samples

    def get_obs_data(self) -> Dict[str, np.ndarray]:
        """Return observational data."""
        return self.obs_data

    def get_int_data(self) -> Dict[Tuple[str, float], Dict[str, np.ndarray]]:
        """Return interventional data."""
        return self.int_data

    def get_all_data(self) -> Tuple[Dict[str, np.ndarray], Dict[Tuple[str, float], Dict[str, np.ndarray]]]:
        """Return both observational and interventional data."""
        return self.obs_data, self.int_data

    def get_node_data(self, node: str) -> Tuple[np.ndarray, Dict[float, np.ndarray]]:
        """Get all data for a specific node.

        Args:
            node: Node to get data for

        Returns:
            Tuple of (observational_data, interventional_data)
            where interventional_data maps intervention values to samples
        """
        obs_data = self.obs_data[node]
        int_data = {value: data[node] for (n, value), data in self.int_data.items()
                    if n == node}
        return obs_data, int_data

    def get_intervention_values(self, node: str) -> np.ndarray:
        """Get all intervention values that have been tried for a node.

        Args:
            node: Node to get intervention values for

        Returns:
            Array of intervention values
        """
        return np.array([value for (n, value) in self.int_data.keys() if n == node])

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
        return list(self.graph.nodes())

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
