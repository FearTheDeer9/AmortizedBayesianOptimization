"""
Directed graph implementation for the causal_meta library.

This module contains the DirectedGraph class that extends the base Graph class
with functionality specific to directed graphs, such as adjacency matrix
representation, path finding, cycle detection, and topological sorting.
"""
from typing import List, Dict, Set, Optional, Union, Any, Tuple, Iterator
import numpy as np

from causal_meta.graph.base import Graph


class DirectedGraph(Graph):
    """A directed graph implementation that extends the base Graph class."""

    def __init__(self):
        """Initialize an empty directed graph."""
        super().__init__()
        # Directed graph specific data structures
        self._out_edges = {}  # node -> set of outgoing neighbors
        self._in_edges = {}   # node -> set of incoming neighbors

    def add_node(self, node_id: Any, **attributes) -> None:
        """
        Add a node to the directed graph.

        Args:
            node_id: The identifier for the node
            **attributes: Optional attributes for the node
        """
        super().add_node(node_id, **attributes)

        # Initialize the edge mappings for the new node
        if node_id not in self._out_edges:
            self._out_edges[node_id] = set()

        if node_id not in self._in_edges:
            self._in_edges[node_id] = set()

    def remove_node(self, node_id: Any) -> None:
        """
        Remove a node and all its incident edges from the directed graph.

        Args:
            node_id: The identifier for the node to remove
        """
        if node_id in self._nodes:
            # Remove outgoing edges
            if node_id in self._out_edges:
                for target in self._out_edges[node_id]:
                    if target in self._in_edges:
                        self._in_edges[target].remove(node_id)
                del self._out_edges[node_id]

            # Remove incoming edges
            if node_id in self._in_edges:
                for source in self._in_edges[node_id]:
                    if source in self._out_edges:
                        self._out_edges[source].remove(node_id)
                del self._in_edges[node_id]

            # Call the parent method to remove the node and its attributes
            super().remove_node(node_id)

    def add_edge(self, source: Any, target: Any, **attributes) -> None:
        """
        Add a directed edge from source to target.

        Args:
            source: The source node identifier
            target: The target node identifier
            **attributes: Optional attributes for the edge

        Raises:
            ValueError: If either the source or target node doesn't exist
        """
        # Call the parent method to add the edge
        super().add_edge(source, target, **attributes)

        # Update the directed edge mappings
        self._out_edges[source].add(target)
        self._in_edges[target].add(source)

    def remove_edge(self, source: Any, target: Any) -> None:
        """
        Remove a directed edge from source to target.

        Args:
            source: The source node identifier
            target: The target node identifier
        """
        if (source, target) in self._edges:
            # Call the parent method to remove the edge
            super().remove_edge(source, target)

            # Update the directed edge mappings
            if target in self._out_edges[source]:
                self._out_edges[source].remove(target)

            if source in self._in_edges[target]:
                self._in_edges[target].remove(source)

    def get_successors(self, node_id: Any) -> Set:
        """
        Get all successor nodes (outgoing neighbors) of a node.

        Args:
            node_id: The node identifier

        Returns:
            Set: A set of node identifiers that are successors

        Raises:
            ValueError: If the node doesn't exist
        """
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} does not exist in the graph")

        return self._out_edges.get(node_id, set()).copy()

    def get_predecessors(self, node_id: Any) -> Set:
        """
        Get all predecessor nodes (incoming neighbors) of a node.

        Args:
            node_id: The node identifier

        Returns:
            Set: A set of node identifiers that are predecessors

        Raises:
            ValueError: If the node doesn't exist
        """
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} does not exist in the graph")

        return self._in_edges.get(node_id, set()).copy()

    def get_adjacency_matrix(self, node_order: Optional[List] = None) -> np.ndarray:
        """
        Get the adjacency matrix representation of the directed graph.

        Args:
            node_order: Optional list specifying the order of nodes.
                        If None, nodes will be ordered by their first appearance.

        Returns:
            np.ndarray: A binary adjacency matrix where entry [i, j] is 1 if there's
                       an edge from node i to node j, and 0 otherwise.
        """
        # If no order is specified, use the nodes in the order they were added
        if node_order is None:
            node_order = list(self._nodes)

        # Create a mapping from node IDs to indices
        node_to_idx = {node: idx for idx, node in enumerate(node_order)}

        # Initialize the adjacency matrix with zeros
        n = len(node_order)
        adj_matrix = np.zeros((n, n), dtype=int)

        # Fill in the adjacency matrix
        for source, target in self._edges:
            i = node_to_idx[source]
            j = node_to_idx[target]
            adj_matrix[i, j] = 1

        return adj_matrix

    def has_path(self, source: Any, target: Any) -> bool:
        """
        Check if there is a directed path from source to target.

        Args:
            source: The source node identifier
            target: The target node identifier

        Returns:
            bool: True if there is a path, False otherwise

        Raises:
            ValueError: If either the source or target node doesn't exist
        """
        if source not in self._nodes:
            raise ValueError(
                f"Source node {source} does not exist in the graph")

        if target not in self._nodes:
            raise ValueError(
                f"Target node {target} does not exist in the graph")

        # Breadth-first search
        visited = set()
        queue = [source]

        while queue:
            node = queue.pop(0)

            if node == target:
                return True

            if node in visited:
                continue

            visited.add(node)
            queue.extend(self.get_successors(node) - visited)

        return False

    def get_all_paths(self, source: Any, target: Any) -> List[List[Any]]:
        """
        Find all directed paths from source to target.

        Args:
            source: The source node identifier
            target: The target node identifier

        Returns:
            List[List[Any]]: A list of paths, where each path is a list of node IDs

        Raises:
            ValueError: If either the source or target node doesn't exist
        """
        if source not in self._nodes:
            raise ValueError(
                f"Source node {source} does not exist in the graph")

        if target not in self._nodes:
            raise ValueError(
                f"Target node {target} does not exist in the graph")

        # Use depth-first search to find all paths
        def dfs(current, path, paths):
            if current == target:
                paths.append(path.copy())
                return

            for successor in self.get_successors(current):
                if successor not in path:  # Avoid cycles
                    path.append(successor)
                    dfs(successor, path, paths)
                    path.pop()

        all_paths = []
        dfs(source, [source], all_paths)
        return all_paths

    def has_cycle(self) -> bool:
        """
        Check if the directed graph contains any cycles.

        Returns:
            bool: True if the graph contains at least one cycle, False otherwise
        """
        # Keep track of visited and currently exploring nodes
        visited = set()
        exploring = set()

        def dfs_cycle_check(node):
            visited.add(node)
            exploring.add(node)

            for successor in self.get_successors(node):
                if successor in exploring:
                    return True

                if successor not in visited:
                    if dfs_cycle_check(successor):
                        return True

            exploring.remove(node)
            return False

        # Check each node that hasn't been visited yet
        for node in self._nodes:
            if node not in visited:
                if dfs_cycle_check(node):
                    return True

        return False

    def get_cycles(self) -> List[List[Any]]:
        """
        Find all cycles in the directed graph.

        Returns:
            List[List[Any]]: A list of cycles, where each cycle is a list of node IDs
        """
        cycles = []
        visited = set()
        path = []
        path_set = set()  # For O(1) lookups

        def dfs_find_cycles(node):
            # If we've already completely explored this node, skip it
            if node in visited:
                return

            # If the node is already in our current path, we found a cycle
            if node in path_set:
                # Find index of the node in the current path to create cycle
                start_idx = path.index(node)
                # Create a cycle (adding the node again to close the cycle)
                cycle = path[start_idx:] + [node]
                cycles.append(cycle)
                return

            # Add node to current path
            path.append(node)
            path_set.add(node)

            # Explore all outgoing edges
            for successor in self.get_successors(node):
                dfs_find_cycles(successor)

            # Finished exploring this node
            path.pop()
            path_set.remove(node)
            visited.add(node)

        # Start DFS from each node to find all cycles
        for node in self._nodes:
            if node not in visited:
                dfs_find_cycles(node)

        return cycles

    def topological_sort(self) -> List[Any]:
        """
        Perform a topological sort of the directed graph.

        Returns:
            List[Any]: A topologically sorted list of node IDs

        Raises:
            ValueError: If the graph contains a cycle
        """
        if self.has_cycle():
            raise ValueError(
                "Cannot perform topological sort on a graph with cycles")

        # Keep track of visited nodes and the result list
        visited = set()
        result = []

        def dfs_topo_sort(node):
            visited.add(node)

            for successor in self.get_successors(node):
                if successor not in visited:
                    dfs_topo_sort(successor)

            result.append(node)

        # Visit each node that hasn't been visited yet
        for node in self._nodes:
            if node not in visited:
                dfs_topo_sort(node)

        # Reverse the result to get topological order
        return result[::-1]

    def get_sources(self) -> Set[Any]:
        """
        Get all source nodes (nodes with no incoming edges).

        Returns:
            Set[Any]: A set of node IDs that are sources
        """
        return {node for node in self._nodes if not self.get_predecessors(node)}

    def get_sinks(self) -> Set[Any]:
        """
        Get all sink nodes (nodes with no outgoing edges).

        Returns:
            Set[Any]: A set of node IDs that are sinks
        """
        return {node for node in self._nodes if not self.get_successors(node)}

    def get_ancestors(self, node_id: Any) -> Set[Any]:
        """
        Get all ancestors of a node (nodes that have a path to the given node).

        Args:
            node_id: The node identifier

        Returns:
            Set[Any]: A set of node IDs that are ancestors

        Raises:
            ValueError: If the node doesn't exist
        """
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} does not exist in the graph")

        ancestors = set()

        def dfs_ancestors(current):
            for pred in self.get_predecessors(current):
                if pred not in ancestors:
                    ancestors.add(pred)
                    dfs_ancestors(pred)

        dfs_ancestors(node_id)
        return ancestors

    def get_descendants(self, node_id: Any) -> Set[Any]:
        """
        Get all descendants of a node (nodes that the given node has a path to).

        Args:
            node_id: The node identifier

        Returns:
            Set[Any]: A set of node IDs that are descendants

        Raises:
            ValueError: If the node doesn't exist
        """
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} does not exist in the graph")

        descendants = set()

        def dfs_descendants(current):
            for succ in self.get_successors(current):
                if succ not in descendants:
                    descendants.add(succ)
                    dfs_descendants(succ)

        dfs_descendants(node_id)
        return descendants

    def is_dag(self) -> bool:
        """
        Check if the directed graph is a directed acyclic graph (DAG).

        Returns:
            bool: True if the graph is a DAG, False otherwise
        """
        return not self.has_cycle()

    def __str__(self) -> str:
        """
        Get a string representation of the directed graph.

        Returns:
            str: A string representation
        """
        return f"DirectedGraph(nodes={len(self._nodes)}, edges={len(self._edges)})"
