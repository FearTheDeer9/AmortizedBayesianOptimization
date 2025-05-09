"""
Causal graph implementation for the causal_meta library.

This module contains the CausalGraph class that extends the DirectedGraph class
with causal semantics, including d-separation checking, Markov blanket
identification, and intervention operations.
"""
from typing import List, Dict, Set, Optional, Union, Any, Tuple, FrozenSet
import copy
import numpy as np

from causal_meta.graph.directed_graph import DirectedGraph
from causal_meta.graph.utils import deprecated

# Import networkx for the new method
import networkx as nx


class CausalGraph(DirectedGraph):
    """
    A causal graph implementation that extends the DirectedGraph class.

    Causal graphs represent causal relationships between variables, with
    edges indicating direct causal effects. This class provides methods for
    causal reasoning, including d-separation, Markov blanket identification,
    and simulation of interventions.
    """

    def __init__(self):
        """Initialize an empty causal graph."""
        super().__init__()

    def copy(self) -> 'CausalGraph':
        """
        Create and return a deep copy of this causal graph.
        
        Returns:
            CausalGraph: A new graph with the same nodes, edges, and attributes
        """
        new_graph = CausalGraph()
        
        # Copy nodes and their attributes
        for node in self.get_nodes():
            new_graph.add_node(node, **self.get_node_attributes(node))
            
        # Copy edges and their attributes
        for source, target in self.get_edges():
            edge_attrs = self.get_edge_attributes(source, target) or {}
            new_graph.add_edge(source, target, **edge_attrs)
            
        return new_graph

    def get_parents(self, node_id: Any) -> Set:
        """
        Get the parents of a node (direct causes).

        Args:
            node_id: The node identifier

        Returns:
            Set: A set of node identifiers that are parents

        Raises:
            ValueError: If the node doesn't exist
        """
        return self.get_predecessors(node_id)

    def get_children(self, node_id: Any) -> Set:
        """
        Get the children of a node (direct effects).

        Args:
            node_id: The node identifier

        Returns:
            Set: A set of node identifiers that are children

        Raises:
            ValueError: If the node doesn't exist
        """
        return self.get_successors(node_id)

    def get_markov_blanket(self, node_id: Any) -> Set:
        """
        Get the Markov blanket of a node.

        The Markov blanket of a node X consists of X's parents, X's children,
        and the parents of X's children.

        Args:
            node_id: The node identifier

        Returns:
            Set: A set of node identifiers in the Markov blanket

        Raises:
            ValueError: If the node doesn't exist
        """
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} does not exist in the graph")

        # Get parents and children
        parents = self.get_parents(node_id)
        children = self.get_children(node_id)

        # Get parents of children (spouses)
        spouses = set()
        for child in children:
            child_parents = self.get_parents(child)
            spouses.update(child_parents)

        # Remove the node itself from spouses if present
        if node_id in spouses:
            spouses.remove(node_id)

        # Combine all sets
        markov_blanket = parents.union(children).union(spouses)

        return markov_blanket

    def is_collider(self, node_id: Any) -> bool:
        """
        Check if a node is a collider (has two or more parents).

        Args:
            node_id: The node identifier

        Returns:
            bool: True if the node is a collider, False otherwise

        Raises:
            ValueError: If the node doesn't exist
        """
        return len(self.get_parents(node_id)) >= 2

    def is_confounder(self, node_a: Any, node_b: Any) -> bool:
        """
        Check if there's a confounding variable for node_a and node_b.

        A confounding variable is a common cause of both node_a and node_b.

        Args:
            node_a: The first node identifier
            node_b: The second node identifier

        Returns:
            bool: True if there's a confounder, False otherwise

        Raises:
            ValueError: If either node doesn't exist
        """
        if node_a not in self._nodes:
            raise ValueError(f"Node {node_a} does not exist in the graph")
        if node_b not in self._nodes:
            raise ValueError(f"Node {node_b} does not exist in the graph")

        # Get ancestors of both nodes
        ancestors_a = self.get_ancestors(node_a)
        ancestors_b = self.get_ancestors(node_b)

        # Check if there's a common ancestor
        return bool(ancestors_a.intersection(ancestors_b))

    def _is_collider_on_path(self, path: List[Any], index: int) -> bool:
        """
        Check if the node at the given index in the path is a collider.

        A node is a collider on a path if the arrows come into the node from
        both the preceding and succeeding nodes on the path.

        Args:
            path: A list of nodes representing a path
            index: The index of the node to check

        Returns:
            bool: True if the node is a collider on the path, False otherwise
        """
        if index <= 0 or index >= len(path) - 1:
            return False

        prev_node = path[index - 1]
        curr_node = path[index]
        next_node = path[index + 1]

        # Check if the edges are pointing towards the current node
        return (prev_node in self.get_parents(curr_node) and
                next_node in self.get_parents(curr_node))

    def _is_active_trail(self, path: List[Any], conditioning_set: Set[Any]) -> bool:
        """
        Check if a path is an active trail given a conditioning set.

        A path is active if:
        1. For any chain (A -> B -> C) or fork (A <- B -> C) on the path, 
           B is not in the conditioning set.
        2. For any collider (A -> B <- C) on the path, B or any of its 
           descendants is in the conditioning set.

        Args:
            path: A list of nodes representing a path
            conditioning_set: A set of nodes we're conditioning on

        Returns:
            bool: True if the path is active, False otherwise
        """
        if len(path) < 3:
            return True

        for i in range(1, len(path) - 1):
            curr_node = path[i]

            if self._is_collider_on_path(path, i):
                # For colliders, the path is active if the collider or any of its
                # descendants is in the conditioning set
                collider_and_descendants = {curr_node}.union(
                    self.get_descendants(curr_node))

                # If none of the collider or its descendants are in the conditioning set,
                # the path is blocked (inactive)
                if not collider_and_descendants.intersection(conditioning_set):
                    return False
            else:
                # For chains and forks, the path is active if the middle node
                # is not in the conditioning set
                if curr_node in conditioning_set:
                    return False

        return True

    def is_d_separated(self, nodes_x: Union[Any, Set[Any]],
                       nodes_y: Union[Any, Set[Any]],
                       conditioning_set: Optional[Set[Any]] = None) -> bool:
        """
        Check if nodes_x and nodes_y are d-separated given conditioning_set.

        Two sets of nodes X and Y are d-separated by a set Z if every path from 
        any node in X to any node in Y is blocked by Z.

        Args:
            nodes_x: A node or set of nodes
            nodes_y: A node or set of nodes
            conditioning_set: A set of nodes to condition on (default: empty set)

        Returns:
            bool: True if X and Y are d-separated given Z, False otherwise

        Raises:
            ValueError: If any node doesn't exist
        """
        # Convert single nodes to sets
        if not isinstance(nodes_x, set):
            nodes_x = {nodes_x}
        if not isinstance(nodes_y, set):
            nodes_y = {nodes_y}
        if conditioning_set is None:
            conditioning_set = set()

        # Validate nodes
        for node in nodes_x.union(nodes_y).union(conditioning_set):
            if node not in self._nodes:
                raise ValueError(f"Node {node} does not exist in the graph")

        # Check for d-separation for each pair of nodes
        for x in nodes_x:
            for y in nodes_y:
                # Skip if x and y are the same node
                if x == y:
                    continue

                # Find all possible paths between x and y (in both directions)
                all_paths = []
                for node in self._nodes:
                    # Check paths through node from x to y
                    if node not in {x, y}:
                        x_to_node_paths = self.get_all_paths(x, node)
                        node_to_y_paths = self.get_all_paths(node, y)

                        for path1 in x_to_node_paths:
                            for path2 in node_to_y_paths:
                                # Combine paths and remove duplicates
                                if path1[-1] == path2[0]:
                                    combined_path = path1 + path2[1:]
                                    all_paths.append(combined_path)

                # Add direct paths
                all_paths.extend(self.get_all_paths(x, y))
                all_paths.extend(self.get_all_paths(y, x))

                # If there's any active path, nodes are not d-separated
                for path in all_paths:
                    if self._is_active_trail(path, conditioning_set):
                        return False

        # If no active path was found, nodes are d-separated
        return True

    def do_intervention(self, node_id: Any, value: Optional[Any] = None) -> 'CausalGraph':
        """
        Create a new causal graph reflecting an intervention on a node.

        This implements the do-operation: do(X=x), which removes all incoming
        edges to X and sets X to the given value.

        Args:
            node_id: The node to intervene on
            value: Optional value to set the node to

        Returns:
            CausalGraph: A new causal graph reflecting the intervention

        Raises:
            ValueError: If the node doesn't exist
        """
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} does not exist in the graph")

        # Create a copy of the current graph
        intervened_graph = copy.deepcopy(self)

        # Remove all incoming edges to the node
        parents = list(intervened_graph.get_parents(node_id))
        for parent in parents:
            intervened_graph.remove_edge(parent, node_id)

        # If a value is provided, store it as a node attribute
        if value is not None:
            intervened_graph._node_attributes[node_id] = {
                'intervention_value': value,
                **intervened_graph.get_node_attributes(node_id)
            }

        return intervened_graph

    def soft_intervention(self, node_id: Any,
                          mechanism_change: Dict[str, Any]) -> 'CausalGraph':
        """
        Create a new causal graph reflecting a soft intervention on a node.

        Soft intervention changes the causal mechanism (function) of a node
        without removing its parents.

        Args:
            node_id: The node to intervene on
            mechanism_change: Dictionary describing the change to the causal mechanism

        Returns:
            CausalGraph: A new causal graph reflecting the soft intervention

        Raises:
            ValueError: If the node doesn't exist
        """
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} does not exist in the graph")

        # Create a copy of the current graph
        intervened_graph = copy.deepcopy(self)

        # Store the mechanism change as a node attribute
        intervened_graph._node_attributes[node_id] = {
            'mechanism_change': mechanism_change,
            **intervened_graph.get_node_attributes(node_id)
        }

        return intervened_graph

    def get_backdoor_paths(self, treatment: Any, outcome: Any) -> List[List[Any]]:
        """
        Find all backdoor paths from treatment to outcome.

        A backdoor path is a path that starts with an arrow pointing to the
        treatment and ends at the outcome.

        Args:
            treatment: The treatment node
            outcome: The outcome node

        Returns:
            List[List[Any]]: A list of backdoor paths

        Raises:
            ValueError: If any node doesn't exist
        """
        if treatment not in self._nodes:
            raise ValueError(
                f"Treatment node {treatment} does not exist in the graph")
        if outcome not in self._nodes:
            raise ValueError(
                f"Outcome node {outcome} does not exist in the graph")

        # Get all parents of the treatment
        treatment_parents = self.get_parents(treatment)

        # For each parent, find paths to the outcome that don't go through treatment
        backdoor_paths = []

        # Create a copy of the graph without the treatment->outcome edge
        # This is to prevent paths that go through the direct effect
        temp_graph = copy.deepcopy(self)
        if temp_graph.has_edge(treatment, outcome):
            temp_graph.remove_edge(treatment, outcome)

        # Find paths from each parent of the treatment to the outcome
        for parent in treatment_parents:
            # Find all paths from the parent to the outcome in the modified graph
            parent_to_outcome_paths = temp_graph.get_all_paths(parent, outcome)

            for path in parent_to_outcome_paths:
                # Check if the treatment is in the path
                if treatment not in path:
                    # Add the treatment at the beginning to form the backdoor path
                    backdoor_path = [treatment] + path
                    backdoor_paths.append(backdoor_path)

        return backdoor_paths

    def is_valid_adjustment_set(self, treatment: Any, outcome: Any,
                                adjustment_set: Set[Any]) -> bool:
        """
        Check if a set is a valid adjustment set for estimating the causal effect.

        An adjustment set is valid if it blocks all backdoor paths between
        treatment and outcome, while not including any descendant of treatment.

        Args:
            treatment: The treatment node
            outcome: The outcome node
            adjustment_set: The adjustment set to check

        Returns:
            bool: True if the adjustment set is valid, False otherwise

        Raises:
            ValueError: If any node doesn't exist
        """
        if treatment not in self._nodes:
            raise ValueError(
                f"Treatment node {treatment} does not exist in the graph")
        if outcome not in self._nodes:
            raise ValueError(
                f"Outcome node {outcome} does not exist in the graph")

        # Validate adjustment set nodes
        for node in adjustment_set:
            if node not in self._nodes:
                raise ValueError(f"Node {node} does not exist in the graph")

        # Get all backdoor paths
        backdoor_paths = self.get_backdoor_paths(treatment, outcome)

        # Check if the adjustment set blocks all backdoor paths
        for path in backdoor_paths:
            # If the path has fewer than 3 nodes, it can't be blocked
            if len(path) < 3:
                return False

            # Check if the path is blocked by the adjustment set
            is_blocked = False
            for i in range(1, len(path) - 1):  # Skip first (treatment) and last (outcome)
                node = path[i]

                # If the node is in the adjustment set, it might block the path
                if node in adjustment_set:
                    # If it's a collider, it doesn't block the path
                    if self._is_collider_on_path(path, i):
                        # Actually, conditioning on a collider opens the path
                        is_blocked = False
                        break
                    else:
                        # If it's not a collider, it blocks the path
                        is_blocked = True
                        break
                # If the node is not in the adjustment set but is a collider,
                # it blocks the path if neither it nor any of its descendants
                # are in the adjustment set
                elif self._is_collider_on_path(path, i):
                    # Get the collider and all its descendants
                    collider_and_descendants = {node}.union(
                        self.get_descendants(node))

                    # If none are in the adjustment set, the path is blocked
                    if not collider_and_descendants.intersection(adjustment_set):
                        is_blocked = True
                        break

            # If this path isn't blocked, the adjustment set is not valid
            if not is_blocked:
                return False

        # If all backdoor paths are blocked, the adjustment set is valid
        return True

    def __str__(self) -> str:
        """
        Get a string representation of the causal graph.

        Returns:
            str: A string representation
        """
        return f"CausalGraph(nodes={len(self._nodes)}, edges={len(self._edges)})"

    # --- Edge Attribute Handling --- 
        
    def set_edge_attribute(self, u: Any, v: Any, name: str, value: Any):
        """
        Set an attribute for a specific edge.

        Args:
            u: The source node identifier.
            v: The target node identifier.
            name: The name of the attribute to set.
            value: The value of the attribute.

        Raises:
            ValueError: If the edge (u, v) does not exist.
        """
        if not self.has_edge(u, v):
            raise ValueError(f"Edge ({u}, {v}) does not exist.")
            
        edge_key = (u, v)
        # Ensure the attribute dictionary exists for this edge
        if edge_key not in self._edge_attributes:
            self._edge_attributes[edge_key] = {}
        self._edge_attributes[edge_key][name] = value

    def get_edge_attribute(self, u: Any, v: Any, name: str, default: Optional[Any] = None) -> Any:
        """
        Get an attribute for a specific edge.

        Args:
            u: The source node identifier.
            v: The target node identifier.
            name: The name of the attribute to get.
            default: The value to return if the edge or attribute doesn't exist.

        Returns:
            The value of the attribute, or the default value.
        """
        edge_key = (u, v)
        edge_data = self._edge_attributes.get(edge_key)
        if edge_data is None:
             # Check if edge exists at all before returning default
             # This prevents returning default if edge is valid but has no attributes
             if not self.has_edge(u,v):
                 return default
             else:
                 # Edge exists but has no attributes dict or specific attribute
                 return default
        return edge_data.get(name, default)

    def get_edge_attributes(self, u: Any, v: Any) -> Optional[Dict[str, Any]]:
        """
        Get all attributes for a specific edge.

        Args:
            u: The source node identifier.
            v: The target node identifier.

        Returns:
            A dictionary of attributes (or an empty dict if none), 
            or None if the edge doesn't exist.
        """
        edge_key = (u, v)
        if not self.has_edge(u, v):
            return None
        # Get the attribute dictionary, defaulting to empty if none exists for the edge
        edge_data = self._edge_attributes.get(edge_key, {})
        # Return a copy to prevent external modification
        return copy.deepcopy(edge_data)

    # has_path is inherited from DirectedGraph

    # --- Causal Specific Methods --- 

    def to_networkx(self, include_attributes: bool = True) -> nx.DiGraph:
        """Convert the CausalGraph to a NetworkX DiGraph object.

        Args:
            include_attributes (bool): If True, copy node and edge attributes
                                       from the CausalGraph to the NetworkX graph.
                                       Defaults to True.

        Returns:
            nx.DiGraph: A NetworkX directed graph representation.
        """
        nx_graph = nx.DiGraph()

        # Add nodes
        for node_id in self.get_nodes():
            if include_attributes:
                node_attrs = self.get_node_attributes(node_id) or {}
                nx_graph.add_node(node_id, **node_attrs)
            else:
                nx_graph.add_node(node_id)

        # Add edges
        for u, v in self.get_edges():
            if include_attributes:
                edge_attrs = self.get_edge_attributes(u, v) or {}
                nx_graph.add_edge(u, v, **edge_attrs)
            else:
                nx_graph.add_edge(u, v)

        return nx_graph

    @classmethod
    def from_networkx(cls, nx_graph: nx.DiGraph, include_attributes: bool = True) -> 'CausalGraph':
        """Create a CausalGraph from a NetworkX DiGraph.

        Args:
            nx_graph (nx.DiGraph): A NetworkX directed graph to convert
            include_attributes (bool): If True, copy node and edge attributes
                                       from the NetworkX graph to the CausalGraph.
                                       Defaults to True.

        Returns:
            CausalGraph: A new CausalGraph with the same nodes, edges, and optionally attributes
        """
        causal_graph = cls()
        
        # Add nodes
        for node in nx_graph.nodes():
            if include_attributes:
                attrs = dict(nx_graph.nodes[node])
                causal_graph.add_node(node, **attrs)
            else:
                causal_graph.add_node(node)
                
        # Add edges
        for u, v in nx_graph.edges():
            if include_attributes:
                attrs = dict(nx_graph.edges[u, v])
                causal_graph.add_edge(u, v, **attrs)
            else:
                causal_graph.add_edge(u, v)
                
        return causal_graph
