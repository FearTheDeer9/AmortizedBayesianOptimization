import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Union, Optional, Any

from graphs.graph import GraphStructure
from graphs.causal_dataset import CausalDataset
from graphs.scm_generators import sample_observational, sample_interventional


class CausalEnvironmentAdapter:
    """Adapter class that provides CausalEnvironment-like interface using GraphStructure and CausalDataset.

    This is a compatibility layer to ease migration from diffcbed to graphs framework.
    """

    def __init__(
        self,
        graph: Union[GraphStructure, nx.DiGraph],
        scm: Dict[str, callable] = None,
        num_samples: int = 1000,
        seed: Optional[int] = None,
        nonlinear: bool = False,
        noise_type: str = "isotropic-gaussian",
        **kwargs
    ):
        """Initialize the causal environment adapter.

        Args:
            graph: Graph structure (either GraphStructure or nx.DiGraph)
            scm: Dictionary of structural equation models
            num_samples: Number of samples to generate
            seed: Random seed for reproducibility
            nonlinear: Whether to use nonlinear mechanisms
            noise_type: Type of noise to use
            **kwargs: Additional arguments for compatibility
        """
        # Accept either GraphStructure or nx.DiGraph
        if isinstance(graph, GraphStructure):
            self.graph_structure = graph
            self.graph = graph.G if hasattr(graph, 'G') else nx.DiGraph()
        else:
            self.graph = graph
            # Don't try to create a GraphStructure from nx.DiGraph automatically
            # as GraphStructure is an abstract class
            self.graph_structure = None

        self.scm = scm
        self.num_samples = num_samples
        self.seed = seed
        self.nonlinear = nonlinear
        self.noise_type = noise_type

        # Store additional args for compatibility
        self.args = kwargs.get('args', {})

        # Set up random number generator
        self.rng = np.random.default_rng(seed)

        # Create or use existing CausalDataset
        if self.graph_structure is not None and hasattr(self.graph_structure, 'dataset') and self.graph_structure.dataset is not None:
            self.dataset = self.graph_structure.dataset
        else:
            self.dataset = CausalDataset(
                graph=self.graph,
                scm=self.scm,
                n_obs=num_samples,
                n_int=100,  # Default for interventional samples
                seed=seed
            )
            # Attach dataset to graph structure if possible
            if self.graph_structure is not None:
                self.graph_structure.dataset = self.dataset

        # Store variables and num_nodes for compatibility
        if self.graph_structure is not None and hasattr(self.graph_structure, 'variables'):
            self.variables = self.graph_structure.variables
        else:
            self.variables = [str(n) for n in self.graph.nodes()]

        self.num_nodes = len(self.graph.nodes())

        # Initialize held-out data
        self.held_out_data = None
        self.held_out_interventions = []

        # Generate observational data if we have an SCM
        if self.scm is not None:
            self._generate_data()

    def _generate_data(self):
        """Generate observational data using the provided SCM."""
        # Get observational samples from the dataset
        self.obs_data = self.dataset.get_obs_data()

        # Convert to samples format expected by legacy code
        self.samples = np.zeros((self.num_samples, self.num_nodes))
        for i, node in enumerate(sorted(self.variables, key=lambda x: int(x) if x.isdigit() else x)):
            if node in self.obs_data:
                self.samples[:, i] = self.obs_data[node].flatten()

        # Generate held-out data
        self.held_out_data = sample_observational(
            self.scm, self.num_samples, self.seed)

    def sample(self, num_samples=None):
        """Sample from the observational distribution.

        Args:
            num_samples: Number of samples to generate (defaults to self.num_samples)

        Returns:
            Named tuple with samples and intervention_node fields
        """
        from collections import namedtuple
        Data = namedtuple("Data", ["samples", "intervention_node"])

        if num_samples is None:
            num_samples = self.num_samples

        # Generate observational data if not already done
        if self.scm is not None:
            samples = sample_observational(self.scm, num_samples, self.seed)

            # Convert to numpy array format expected by legacy code
            samples_array = np.zeros((num_samples, self.num_nodes))
            for i, node in enumerate(sorted(self.variables, key=lambda x: int(x) if x.isdigit() else x)):
                if node in samples:
                    samples_array[:, i] = samples[node].flatten()

            return Data(samples=samples_array, intervention_node=None)

        # If no SCM is available, return random samples
        return Data(
            samples=np.random.normal(0, 1, size=(num_samples, self.num_nodes)),
            intervention_node=None
        )

    def intervene(self, nodes, values, num_samples=1000):
        """Intervene on one or more nodes with given values.

        Args:
            nodes: Node(s) to intervene on. Can be:
                - Single integer: Intervene on one node
                - List of integers: Intervene on multiple nodes simultaneously
            values: Value(s) to set the node(s) to. Can be:
                - Single float: Set one node to this value
                - List of floats: Set corresponding nodes to these values
            num_samples: Number of samples to generate

        Returns:
            Named tuple with samples and intervention_node fields
        """
        from collections import namedtuple
        Data = namedtuple("Data", ["samples", "intervention_node"])

        # Handle different input formats
        if isinstance(nodes, int):
            nodes = [nodes]
        if not isinstance(values, (list, tuple, np.ndarray)):
            values = [values]

        # Ensure equal number of nodes and values
        if len(nodes) != len(values):
            raise ValueError("Number of nodes must match number of values")

        # Convert nodes to strings if needed
        node_strs = [str(node) for node in nodes]

        # Generate interventional samples
        all_samples = {}
        for node, value in zip(nodes, values):
            if len(nodes) == 1:
                # Single intervention
                int_samples = sample_interventional(
                    self.scm, node, value, num_samples, self.seed, graph=self.graph
                )
                all_samples = int_samples
            else:
                # Multi-intervention - need to handle differently
                # For now, we'll just do a simple approach of intervening on each node separately
                # This should be improved in the future
                raise NotImplementedError(
                    "Multi-interventions not yet implemented in adapter")

        # Convert to numpy array format expected by legacy code
        samples_array = np.zeros((num_samples, self.num_nodes))
        for i, node in enumerate(sorted(self.variables, key=lambda x: int(x) if x.isdigit() else x)):
            if node in all_samples:
                samples_array[:, i] = all_samples[node].flatten()

        return Data(samples=samples_array, intervention_node=nodes)

    def get_adjacency_matrix(self):
        """Get the adjacency matrix of the graph.

        Returns:
            Adjacency matrix as numpy array
        """
        return nx.to_numpy_array(self.graph, nodelist=sorted(self.graph.nodes()))

    def get_edge_weights(self):
        """Get the edge weights of the graph.

        Returns:
            Edge weight matrix as numpy array
        """
        adj_matrix = self.get_adjacency_matrix()
        # In NetworkX, edges might have weight attributes
        # For now, we'll just return binary adjacency matrix
        return adj_matrix

    @property
    def weighted_adjacency_matrix(self):
        """Get the weighted adjacency matrix for compatibility."""
        return self.get_edge_weights()

    def plot_graph(self, ax=None, highlight_nodes=None, **kwargs):
        """Plot the graph using NetworkX and matplotlib.

        Args:
            ax: Matplotlib axis to plot on
            highlight_nodes: Nodes to highlight
            **kwargs: Additional arguments to pass to nx.draw

        Returns:
            Matplotlib axis
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))

        pos = nx.spring_layout(self.graph, seed=self.seed)

        # Draw nodes and edges
        nx.draw_networkx_nodes(
            self.graph, pos, node_color='skyblue', node_size=700, ax=ax)
        nx.draw_networkx_edges(self.graph, pos, width=2, arrowsize=20, ax=ax)

        # Highlight specific nodes if requested
        if highlight_nodes is not None:
            nx.draw_networkx_nodes(
                self.graph, pos,
                nodelist=highlight_nodes,
                node_color='red',
                node_size=700,
                ax=ax
            )

        # Draw labels
        nx.draw_networkx_labels(
            self.graph, pos, font_weight='bold', font_size=14, ax=ax)

        return ax
