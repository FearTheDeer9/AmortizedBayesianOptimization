#!/usr/bin/env python3
"""
Playground script demonstrating different ways to check the number of nodes
in the graph classes from the causal_meta library.
"""

from causal_meta.graph import DirectedGraph, CausalGraph
from causal_meta.graph.generators.predefined import PredefinedGraphStructureGenerator
from causal_meta.graph.generators.scale_free import ScaleFreeNetworkGenerator


def main():
    print("===== DIRECTED GRAPH NODE COUNT METHODS =====")
    # Create a simple directed graph manually
    dg = DirectedGraph()

    # Add some nodes
    for i in range(5):
        dg.add_node(i)

    # Add some edges
    dg.add_edge(0, 1)
    dg.add_edge(1, 2)
    dg.add_edge(2, 3)
    dg.add_edge(3, 4)

    # Different ways to check the number of nodes in a DirectedGraph

    # Method 1: Using len() - RECOMMENDED - uses the __len__ method inherited from Graph
    print(f"Number of nodes using len(): {len(dg)}")

    # Method 2: Using get_nodes() method - RECOMMENDED - returns a copy of the nodes set
    print(f"Number of nodes using get_nodes(): {len(dg.get_nodes())}")

    # Method 3: Directly accessing the _nodes set - NOT RECOMMENDED - uses internal implementation
    print(f"Number of nodes using _nodes: {len(dg._nodes)}")

    # Check if num_nodes property exists (it doesn't in the base implementation)
    # Looking at the string representation gives us a hint:
    print(f"String representation: {str(dg)}")

    # We can create a property accessor for node count based on the string representation
    @property
    def num_nodes(self):
        return len(self._nodes)

    # Attach the property to the instance (not typical usage, just for demonstration)
    DirectedGraph.num_nodes = num_nodes
    print(f"Number of nodes using added property: {dg.num_nodes}")

    print("\n===== CAUSAL GRAPH NODE COUNT METHODS =====")
    # Create a causal graph using a predefined structure
    cg = PredefinedGraphStructureGenerator.fork(num_nodes=5, is_causal=True)

    # Same methods work for CausalGraph since it inherits from DirectedGraph
    print(f"Number of nodes using len(): {len(cg)}")
    print(f"Number of nodes using get_nodes(): {len(cg.get_nodes())}")
    print(f"String representation: {str(cg)}")

    print("\n===== COMPLEX GRAPH EXAMPLE =====")
    # Create a more complex graph
    complex_graph = PredefinedGraphStructureGenerator.diamond(is_causal=True)

    # Print information about the complex graph
    print(
        f"Diamond graph has {len(complex_graph)} nodes and {len(complex_graph.get_edges())} edges")

    # List all nodes and their relationships
    print("\nNode relationships in the diamond graph:")
    for node in complex_graph.get_nodes():
        children = complex_graph.get_children(node)
        parents = complex_graph.get_parents(node)
        print(f"Node {node}: Parents={parents}, Children={children}")

    print("\n===== SCALE-FREE GRAPH EXAMPLE =====")
    # Create a scale-free graph using the Barabási-Albert model
    sf_graph = ScaleFreeNetworkGenerator.barabasi_albert(
        num_nodes=10,
        m=2,
        directed=True,
        is_causal=True,
        seed=42
    )

    # Get node count
    print(
        f"Scale-free graph has {len(sf_graph)} nodes and {len(sf_graph.get_edges())} edges")

    # Calculate node degrees (in a directed graph)
    in_degrees = {}
    out_degrees = {}

    for node in sf_graph.get_nodes():
        in_degrees[node] = len(sf_graph.get_predecessors(node))
        out_degrees[node] = len(sf_graph.get_successors(node))

    print("\nNode degrees in the scale-free graph:")
    for node in sorted(sf_graph.get_nodes()):
        print(
            f"Node {node}: In-degree={in_degrees[node]}, Out-degree={out_degrees[node]}")

    print("\nSummary of best practices for counting nodes:")
    print("1. Use len(graph) - Most Pythonic method")
    print("2. Use len(graph.get_nodes()) - Explicit and safe")
    print("3. Avoid using len(graph._nodes) in production code - relies on implementation details")


if __name__ == "__main__":
    main()
