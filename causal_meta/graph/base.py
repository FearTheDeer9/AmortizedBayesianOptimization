"""
Base graph classes for the causal_meta library.

This module contains the core base graph class that serves as the foundation
for all graph implementations in the causal_meta library.
"""
from typing import List, Dict, Set, Optional, Union, Any, Tuple


class Graph:
    """Base class for all graph implementations."""

    def __init__(self):
        """Initialize an empty graph."""
        # Core data structures for the graph
        self._nodes = set()  # Set of node IDs
        self._edges = set()  # Set of tuples (source, target)
        self._node_attributes = {}  # Dictionary of node ID to attributes
        # Dictionary of (source, target) to attributes
        self._edge_attributes = {}

    def add_node(self, node_id: Any, **attributes) -> None:
        """
        Add a node to the graph.

        Args:
            node_id: The identifier for the node
            **attributes: Optional attributes for the node
        """
        self._nodes.add(node_id)
        if attributes:
            self._node_attributes[node_id] = attributes

    def remove_node(self, node_id: Any) -> None:
        """
        Remove a node and all its incident edges from the graph.

        Args:
            node_id: The identifier for the node to remove
        """
        if node_id in self._nodes:
            # Remove all edges connected to this node
            self._edges = {(s, t) for s, t in self._edges
                           if s != node_id and t != node_id}

            # Remove node attributes
            if node_id in self._node_attributes:
                del self._node_attributes[node_id]

            # Remove edge attributes for any edges connected to this node
            keys_to_remove = []
            for edge in self._edge_attributes:
                if edge[0] == node_id or edge[1] == node_id:
                    keys_to_remove.append(edge)

            for key in keys_to_remove:
                del self._edge_attributes[key]

            # Remove the node itself
            self._nodes.remove(node_id)

    def add_edge(self, source: Any, target: Any, **attributes) -> None:
        """
        Add an edge to the graph.

        Args:
            source: The source node identifier
            target: The target node identifier
            **attributes: Optional attributes for the edge

        Raises:
            ValueError: If either the source or target node doesn't exist
        """
        if source not in self._nodes:
            raise ValueError(
                f"Source node {source} does not exist in the graph")

        if target not in self._nodes:
            raise ValueError(
                f"Target node {target} does not exist in the graph")

        self._edges.add((source, target))
        if attributes:
            self._edge_attributes[(source, target)] = attributes

    def remove_edge(self, source: Any, target: Any) -> None:
        """
        Remove an edge from the graph.

        Args:
            source: The source node identifier
            target: The target node identifier
        """
        if (source, target) in self._edges:
            self._edges.remove((source, target))

            # Remove edge attributes
            if (source, target) in self._edge_attributes:
                del self._edge_attributes[(source, target)]

    def has_node(self, node_id: Any) -> bool:
        """
        Check if a node exists in the graph.

        Args:
            node_id: The node identifier to check

        Returns:
            bool: True if the node exists, False otherwise
        """
        return node_id in self._nodes

    def has_edge(self, source: Any, target: Any) -> bool:
        """
        Check if an edge exists in the graph.

        Args:
            source: The source node identifier
            target: The target node identifier

        Returns:
            bool: True if the edge exists, False otherwise
        """
        return (source, target) in self._edges

    def get_nodes(self) -> List[Any]:
        """
        Get all nodes in the graph.

        Returns:
            List[Any]: A list of all node identifiers
        """
        return list(self._nodes)

    def get_edges(self) -> List[Tuple[Any, Any]]:
        """
        Get all edges in the graph.

        Returns:
            List[Tuple[Any, Any]]: A list of (source, target) tuples representing edges
        """
        return list(self._edges)

    def get_node_attributes(self, node_id: Any) -> Dict:
        """
        Get the attributes for a node.

        Args:
            node_id: The node identifier

        Returns:
            Dict: A dictionary of node attributes

        Raises:
            ValueError: If the node doesn't exist
        """
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} does not exist in the graph")

        return self._node_attributes.get(node_id, {}).copy()

    def get_edge_attributes(self, source: Any, target: Any) -> Dict:
        """
        Get the attributes for an edge.

        Args:
            source: The source node identifier
            target: The target node identifier

        Returns:
            Dict: A dictionary of edge attributes

        Raises:
            ValueError: If the edge doesn't exist
        """
        if (source, target) not in self._edges:
            raise ValueError(
                f"Edge ({source}, {target}) does not exist in the graph")

        return self._edge_attributes.get((source, target), {}).copy()

    def get_neighbors(self, node_id: Any) -> Set:
        """
        Get all neighbors of a node (both incoming and outgoing).

        Args:
            node_id: The node identifier

        Returns:
            Set: A set of node identifiers that are neighbors

        Raises:
            ValueError: If the node doesn't exist
        """
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} does not exist in the graph")

        neighbors = set()
        for s, t in self._edges:
            if s == node_id:
                neighbors.add(t)
            if t == node_id:
                neighbors.add(s)

        return neighbors

    def __len__(self) -> int:
        """
        Get the number of nodes in the graph.

        Returns:
            int: The number of nodes
        """
        return len(self._nodes)

    def __str__(self) -> str:
        """
        Get a string representation of the graph.

        Returns:
            str: A string representation
        """
        return f"Graph(nodes={len(self._nodes)}, edges={len(self._edges)})"
