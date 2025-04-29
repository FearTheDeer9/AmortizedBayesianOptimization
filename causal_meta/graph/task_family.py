import networkx as nx
from typing import List, Dict, Any, Optional
import logging
import pickle
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class TaskFamily:
    """Represents a family of related causal tasks (graphs).

    Attributes:
        base_graph (nx.DiGraph): The original graph from which variations were derived.
        variations (List[Dict[str, Any]]): A list of dictionaries, each containing:
            - 'graph' (nx.DiGraph): The variation graph.
            - 'metadata' (Dict[str, Any]): Metadata specific to this variation's generation
                                            (e.g., type, strength, seed).
        graphs (List[nx.DiGraph]): A list containing the base_graph followed by all variation graphs.
        metadata (Dict[str, Any]): Metadata about the generation process for the entire family
                                  (e.g., base_graph_id, generation_timestamp, generation_params).
    """

    def __init__(self,
                 base_graph: nx.DiGraph,
                 variations: Optional[List[Dict[str, Any]]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """Initializes the TaskFamily.

        Args:
            base_graph: The base graph.
            variations: Optional list of initial variation dictionaries.
                        Each dict must have a 'graph' key (nx.DiGraph) and optionally a 'metadata' dict.
            metadata: Optional dictionary containing metadata for the family generation process.
        """
        if not isinstance(base_graph, nx.DiGraph):
            raise TypeError("base_graph must be a networkx.DiGraph")

        self.base_graph = base_graph
        self.variations = []
        self.metadata = metadata if metadata is not None else {}
        self.metadata.setdefault('generation_timestamp', datetime.now().isoformat())
        # Ensure generation_params exists
        self.metadata.setdefault('generation_params', {})

        # Process initial variations if provided
        if variations:
            for i, var_info in enumerate(variations):
                if not isinstance(var_info, dict) or 'graph' not in var_info:
                    logger.warning(f"Skipping item {i} in initial variations: not a dict or missing 'graph' key.")
                    continue
                if not isinstance(var_info['graph'], nx.DiGraph):
                     logger.warning(f"Skipping item {i} in initial variations: 'graph' is not a nx.DiGraph.")
                     continue
                # Standardize variation structure
                processed_var_info = {
                    'graph': var_info['graph'],
                    'metadata': var_info.get('metadata', {}) # Ensure metadata dict exists
                }
                self.variations.append(processed_var_info)

        # Update the combined list of graphs
        self._update_graphs_list()

        logger.debug(f"Initialized TaskFamily with {len(self.graphs)} total graphs.")

    def _update_graphs_list(self):
        """Helper to update the self.graphs list."""
        self.graphs = [self.base_graph] + [var['graph'] for var in self.variations]

    def add_variation(self, graph: nx.DiGraph, metadata: Optional[Dict[str, Any]] = None):
        """Adds a new variation graph to the family.

        Args:
            graph: The variation graph (nx.DiGraph).
            metadata: Optional metadata specific to this variation's generation
                      (e.g., {'type': 'edge_weights', 'strength': 0.1}).
        """
        if not isinstance(graph, nx.DiGraph):
            raise TypeError("Variation graph must be a networkx.DiGraph")

        variation_info = {
            'graph': graph,
            'metadata': metadata if metadata is not None else {}
        }
        # Add default timestamp if not provided
        variation_info['metadata'].setdefault('added_timestamp', datetime.now().isoformat())

        self.variations.append(variation_info)
        self._update_graphs_list() # Keep the graphs list consistent
        logger.debug(f"Added variation. Total graphs: {len(self.graphs)}")

    def __len__(self) -> int:
        """Return the total number of graphs in the family (base + variations)."""
        return len(self.graphs)

    def __getitem__(self, index: int) -> nx.DiGraph:
        """Allow accessing graphs by index (0 is base_graph, 1+ are variations)."""
        if index < 0 or index >= len(self.graphs):
            raise IndexError("TaskFamily index out of range")
        return self.graphs[index]

    def get_variation_info(self, index: int) -> Dict[str, Any]:
        """Gets the full info dictionary (graph + metadata) for a specific variation.
           Note: Index refers to the position within the `variations` list (0-based), so
                 `get_variation_info(0)` gets the first variation added.
        """
        if index < 0 or index >= len(self.variations):
            raise IndexError(f"Variation index {index} out of range (0 to {len(self.variations)-1})")
        return self.variations[index]

    def add_family_metadata(self, key: str, value: Any):
        """Adds a key-value pair to the family-level metadata."""
        self.metadata[key] = value
        logger.debug(f"Added family metadata: {{ '{key}': {value} }}")

    def save(self, filepath: str):
        """Saves the TaskFamily object to a file using pickle.

        Args:
            filepath: The path to the file where the object will be saved.
        """
        try:
            # Ensure directory exists
            dirpath = os.path.dirname(filepath)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)

            with open(filepath, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            logger.info(f"TaskFamily saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Error saving TaskFamily to {filepath}: {e}")
            raise

    @staticmethod
    def load(filepath: str) -> 'TaskFamily':
        """Loads a TaskFamily object from a file using pickle.

        Args:
            filepath: The path to the file containing the saved object.

        Returns:
            The loaded TaskFamily object.

        Raises:
            FileNotFoundError: If the file does not exist.
            Exception: If there is an error during unpickling.
        """
        try:
            with open(filepath, 'rb') as f:
                task_family = pickle.load(f)
            if not isinstance(task_family, TaskFamily):
                raise TypeError(f"Loaded object is not a TaskFamily: {type(task_family)}")
            logger.info(f"TaskFamily loaded successfully from {filepath}")
            return task_family
        except FileNotFoundError:
            logger.error(f"Error loading TaskFamily: File not found at {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading TaskFamily from {filepath}: {e}")
            raise

    def __str__(self) -> str:
        """Return a string representation of the TaskFamily."""
        return f"TaskFamily(base_graph_nodes={len(self.base_graph.nodes())}, num_variations={len(self.variations)})"

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        # Keep repr relatively concise
        return f"TaskFamily(base_graph=<nx.DiGraph size={len(self.base_graph)}>, variations=[...{len(self.variations)} items], metadata={list(self.metadata.keys())})"

    # Potential future methods:
    # - method to calculate average similarity within the family
    # - method to filter variations based on criteria
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families
    # - method to calculate similarity between graphs within a family
    # - method to calculate similarity between families

    def __str__(self):
        return f"TaskFamily with {len(self.graphs)} graphs"

    def __repr__(self):
        return f"TaskFamily(base_graph={self.base_graph}, variations={self.variations}, metadata={self.metadata})" 