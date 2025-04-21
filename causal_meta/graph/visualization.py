"""
Graph visualization utilities for the causal_meta library.

This module provides functions for visualizing different types of graphs,
including directed graphs and causal graphs. It uses matplotlib for basic
visualization and integrates with networkx for advanced layout algorithms.
"""
from typing import Dict, Set, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import to_rgba

from causal_meta.graph import CausalGraph, DirectedGraph


def plot_graph(graph: Union[DirectedGraph, CausalGraph],
               ax: Optional[plt.Axes] = None,
               layout: str = 'spring',
               node_size: int = 300,
               node_color: str = '#1f78b4',
               edge_color: str = '#000000',
               font_size: int = 10,
               font_color: str = '#000000',
               alpha: float = 0.9,
               width: float = 1.5,
               arrowsize: int = 15,
               with_labels: bool = True,
               title: Optional[str] = None,
               highlight_nodes: Optional[Set] = None,
               highlight_edges: Optional[List[Tuple]] = None,
               highlight_node_color: str = '#ff7f00',
               highlight_edge_color: str = '#ff0000',
               node_labels: Optional[Dict] = None,
               layout_kwargs: Optional[Dict] = None,
               figsize: Tuple[int, int] = (8, 6),
               dpi: int = 100,
               **kwargs) -> plt.Axes:
    """
    Plot a directed or causal graph using matplotlib and networkx.

    Args:
        graph: The DirectedGraph or CausalGraph to visualize
        ax: Optional matplotlib Axes object to plot on (if None, a new figure is created)
        layout: The layout algorithm to use ('spring', 'circular', 'kamada_kawai',
                'planar', 'random', 'shell', 'spectral', 'spiral')
        node_size: Size of the nodes
        node_color: Color of the nodes
        edge_color: Color of the edges
        font_size: Size of the node labels
        font_color: Color of the node labels
        alpha: Transparency of nodes
        width: Width of the edges
        arrowsize: Size of the arrow heads
        with_labels: Whether to display node labels
        title: Optional title for the plot
        highlight_nodes: Optional set of nodes to highlight
        highlight_edges: Optional list of edges to highlight
        highlight_node_color: Color for highlighted nodes
        highlight_edge_color: Color for highlighted edges
        node_labels: Optional dictionary mapping node ids to custom labels
        layout_kwargs: Optional dictionary of keyword arguments for the layout algorithm
        figsize: Figure size as (width, height) in inches
        dpi: Dots per inch for the figure
        **kwargs: Additional keyword arguments passed to nx.draw_networkx

    Returns:
        matplotlib.pyplot.Axes: The axes object containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Convert the graph to a networkx graph
    G = _convert_to_networkx(graph)

    # Determine the layout
    pos = _get_layout(G, layout, layout_kwargs)

    # Draw the graph
    nx.draw_networkx(
        G,
        pos=pos,
        ax=ax,
        node_size=node_size,
        node_color=node_color,
        edge_color=edge_color,
        font_size=font_size,
        font_color=font_color,
        alpha=alpha,
        width=width,
        arrowsize=arrowsize,
        arrows=True,
        with_labels=with_labels,
        labels=node_labels,
        **kwargs
    )

    # Highlight specific nodes if requested
    if highlight_nodes:
        nx.draw_networkx_nodes(
            G,
            pos=pos,
            nodelist=list(highlight_nodes),
            node_color=highlight_node_color,
            ax=ax,
            node_size=node_size,
            alpha=alpha
        )

    # Highlight specific edges if requested
    if highlight_edges:
        nx.draw_networkx_edges(
            G,
            pos=pos,
            edgelist=highlight_edges,
            edge_color=highlight_edge_color,
            ax=ax,
            width=width * 1.5,
            arrowsize=arrowsize * 1.2,
            arrows=True,
            alpha=1.0
        )

    # Set title if provided
    if title:
        ax.set_title(title)

    # Remove axes
    ax.set_axis_off()

    return ax


def plot_causal_graph(causal_graph: CausalGraph,
                      treatment: Optional[Any] = None,
                      outcome: Optional[Any] = None,
                      conditioning_set: Optional[Set] = None,
                      show_backdoor_paths: bool = False,
                      ax: Optional[plt.Axes] = None,
                      layout: str = 'spring',
                      figsize: Tuple[int, int] = (10, 8),
                      dpi: int = 100,
                      **kwargs) -> plt.Axes:
    """
    Plot a causal graph with highlighting for causal analysis.

    This function extends plot_graph with causal-specific visualizations,
    such as highlighting treatment, outcome, and conditioning variables,
    as well as visualizing backdoor paths.

    Args:
        causal_graph: The CausalGraph to visualize
        treatment: Optional treatment (cause) node
        outcome: Optional outcome (effect) node
        conditioning_set: Optional set of conditioning variables
        show_backdoor_paths: Whether to highlight backdoor paths between
                             treatment and outcome (only if both are specified)
        ax: Optional matplotlib Axes object to plot on
        layout: The layout algorithm to use
        figsize: Figure size as (width, height) in inches
        dpi: Dots per inch for the figure
        **kwargs: Additional keyword arguments passed to plot_graph

    Returns:
        matplotlib.pyplot.Axes: The axes object containing the plot
    """
    # Initialize highlighted nodes and edges
    highlight_nodes = set()
    highlight_edges = []

    # Node colors for different types of variables
    if 'node_color' not in kwargs:
        kwargs['node_color'] = '#aec7e8'  # Light blue for regular nodes

    # Create color map for different node types
    if treatment is not None:
        highlight_nodes.add(treatment)

    if outcome is not None:
        highlight_nodes.add(outcome)

    # Add conditioning set to highlighted nodes
    if conditioning_set:
        highlight_nodes.update(conditioning_set)

    # Highlight backdoor paths if requested
    backdoor_paths = []
    if show_backdoor_paths and treatment is not None and outcome is not None:
        backdoor_paths = causal_graph.get_backdoor_paths(treatment, outcome)
        for path in backdoor_paths:
            for i in range(len(path) - 1):
                highlight_edges.append((path[i], path[i + 1]))

    # Prepare a node color map
    node_color_map = {}
    if treatment is not None:
        node_color_map[treatment] = '#d62728'  # Red for treatment
    if outcome is not None:
        node_color_map[outcome] = '#2ca02c'  # Green for outcome
    if conditioning_set:
        for node in conditioning_set:
            if node not in [treatment, outcome]:
                # Orange for conditioning variables
                node_color_map[node] = '#ff7f0e'

    # Create custom node colors list
    if node_color_map:
        node_colors = [
            node_color_map.get(node, kwargs['node_color'])
            for node in causal_graph._nodes
        ]
        kwargs['node_color'] = node_colors

    # Call the base plot_graph function with causal-specific settings
    ax = plot_graph(
        causal_graph,
        ax=ax,
        layout=layout,
        highlight_edges=highlight_edges if highlight_edges else None,
        highlight_edge_color='#9467bd',  # Purple for backdoor paths
        figsize=figsize,
        dpi=dpi,
        **kwargs
    )

    # Add a legend
    if treatment is not None or outcome is not None or conditioning_set:
        handles = []
        labels = []

        if treatment is not None:
            handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor='#d62728', markersize=10))
            labels.append('Treatment')

        if outcome is not None:
            handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor='#2ca02c', markersize=10))
            labels.append('Outcome')

        if conditioning_set:
            handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor='#ff7f0e', markersize=10))
            labels.append('Conditioning')

        if show_backdoor_paths and backdoor_paths:
            handles.append(plt.Line2D([0], [0], color='#9467bd', lw=2))
            labels.append('Backdoor Path')

        ax.legend(handles, labels, loc='upper right', framealpha=0.7)

    return ax


def plot_graph_adjacency_matrix(graph: Union[DirectedGraph, CausalGraph],
                                ax: Optional[plt.Axes] = None,
                                cmap: str = 'Blues',
                                node_order: Optional[List] = None,
                                figsize: Tuple[int, int] = (8, 8),
                                dpi: int = 100,
                                show_colorbar: bool = True,
                                title: Optional[str] = None) -> plt.Axes:
    """
    Plot the adjacency matrix of a graph.

    Args:
        graph: The DirectedGraph or CausalGraph to visualize
        ax: Optional matplotlib Axes object to plot on
        cmap: Colormap to use for the heatmap
        node_order: Optional list specifying the order of nodes
        figsize: Figure size as (width, height) in inches
        dpi: Dots per inch for the figure
        show_colorbar: Whether to show a colorbar
        title: Optional title for the plot

    Returns:
        matplotlib.pyplot.Axes: The axes object containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Get the adjacency matrix
    adjacency_matrix = graph.get_adjacency_matrix(node_order)

    # If no node order specified, use the graph nodes
    if node_order is None:
        node_order = list(graph._nodes)

    # Plot the adjacency matrix as a heatmap
    im = ax.imshow(adjacency_matrix, cmap=cmap, interpolation='none')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(node_order)))
    ax.set_yticks(np.arange(len(node_order)))
    ax.set_xticklabels(node_order)
    ax.set_yticklabels(node_order)

    # Rotate the x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    # Add a colorbar if requested
    if show_colorbar:
        plt.colorbar(im, ax=ax)

    # Set title if provided
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Adjacency Matrix')

    # Add labels
    ax.set_xlabel('Target Node')
    ax.set_ylabel('Source Node')

    # Ensure the plot is tight
    plt.tight_layout()

    return ax


def plot_intervention(graph: CausalGraph,
                      intervened_graph: CausalGraph,
                      intervention_node: Any,
                      ax: Optional[plt.Axes] = None,
                      layout: str = 'spring',
                      figsize: Tuple[int, int] = (12, 6),
                      dpi: int = 100,
                      **kwargs) -> plt.Axes:
    """
    Plot a before-and-after comparison of a graph intervention.

    Args:
        graph: The original CausalGraph before intervention
        intervened_graph: The CausalGraph after intervention
        intervention_node: The node on which the intervention was performed
        ax: Optional matplotlib Axes object to plot on
        layout: The layout algorithm to use
        figsize: Figure size as (width, height) in inches
        dpi: Dots per inch for the figure
        **kwargs: Additional keyword arguments passed to plot_graph

    Returns:
        matplotlib.pyplot.Axes: The axes object containing the plot
    """
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    else:
        # If a single axis is provided, create a new figure with two axes
        fig = plt.gcf()
        ax1, ax2 = fig.subplots(1, 2)

    # Convert the graphs to networkx for layout consistency
    G_orig = _convert_to_networkx(graph)

    # Use the same layout for both graphs
    layout_kwargs = kwargs.pop('layout_kwargs', {})
    pos = _get_layout(G_orig, layout, layout_kwargs)

    # Plot the original graph
    plot_graph(
        graph,
        ax=ax1,
        layout=layout,
        layout_kwargs=layout_kwargs,
        highlight_nodes={intervention_node},
        title="Original Graph",
        **kwargs
    )

    # Plot the intervened graph
    plot_graph(
        intervened_graph,
        ax=ax2,
        layout=layout,
        layout_kwargs=layout_kwargs,
        highlight_nodes={intervention_node},
        title=f"After do({intervention_node})",
        **kwargs
    )

    plt.tight_layout()

    return ax1, ax2


def _convert_to_networkx(graph: Union[DirectedGraph, CausalGraph]) -> nx.DiGraph:
    """
    Convert a DirectedGraph or CausalGraph to a networkx DiGraph.

    Args:
        graph: The graph to convert

    Returns:
        nx.DiGraph: The equivalent networkx graph
    """
    # Create a new networkx directed graph
    G = nx.DiGraph()

    # Add nodes with attributes
    for node in graph._nodes:
        G.add_node(node, **graph._node_attributes.get(node, {}))

    # Add edges with attributes
    for edge in graph._edges:
        source, target = edge
        G.add_edge(source, target, **graph._edge_attributes.get(edge, {}))

    return G


def _get_layout(G: nx.DiGraph,
                layout_name: str,
                layout_kwargs: Optional[Dict] = None) -> Dict:
    """
    Compute node positions using the specified layout algorithm.

    Args:
        G: The networkx graph
        layout_name: The name of the layout algorithm to use
        layout_kwargs: Optional arguments for the layout algorithm

    Returns:
        Dict: A dictionary of node positions
    """
    if layout_kwargs is None:
        layout_kwargs = {}

    # Map layout names to networkx layout functions
    layouts = {
        'spring': nx.spring_layout,
        'circular': nx.circular_layout,
        'kamada_kawai': nx.kamada_kawai_layout,
        'planar': nx.planar_layout,
        'random': nx.random_layout,
        'shell': nx.shell_layout,
        'spectral': nx.spectral_layout,
        'spiral': nx.spiral_layout
    }

    # Get the layout function
    layout_func = layouts.get(layout_name.lower(), nx.spring_layout)

    # Compute the layout
    return layout_func(G, **layout_kwargs)
