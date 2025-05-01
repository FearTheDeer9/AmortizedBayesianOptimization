import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import List, Callable, Optional, Any, Union, Tuple
import logging

logger = logging.getLogger(__name__)

# Assuming TaskFamily exists, e.g., in causal_meta.graph.task_family
# from causal_meta.graph.task_family import TaskFamily 
# Using a placeholder for now if the actual class isn't defined
try:
    from causal_meta.graph.task_family import TaskFamily
except ImportError:
    logger.warning("TaskFamily class not found, using placeholder.")
    class TaskFamily:
        def __init__(self, base_graph: nx.DiGraph, variations: List[nx.DiGraph]):
            self.base_graph = base_graph
            self.variations = variations
            self.graphs = [base_graph] + variations

# Add import for CausalGraph
try:
    from causal_meta.graph.causal_graph import CausalGraph
except ImportError:
    logger.warning("CausalGraph class not found, using placeholder.")
    class CausalGraph:
        def get_nodes(self):
            return []
        def get_edges(self):
            return []

def plot_graph_comparison(
    inferred_graph: Union[nx.DiGraph, CausalGraph], 
    true_graph: Union[nx.DiGraph, CausalGraph], 
    title: str = 'Inferred vs. True Causal Graph',
    figsize: Tuple[int, int] = (12, 6),
    node_color: str = 'lightblue',
    edge_color: str = 'gray',
    node_size: int = 500,
    font_size: int = 12,
    arrowsize: int = 20,
    seed: int = 42
) -> plt.Figure:
    """
    Plot a comparison between inferred and true causal graphs side by side.
    
    Args:
        inferred_graph: The inferred causal graph (CausalGraph or NetworkX DiGraph)
        true_graph: The true causal graph (CausalGraph or NetworkX DiGraph)
        title: Title for the overall plot
        figsize: Figure size as (width, height)
        node_color: Color for nodes
        edge_color: Color for edges
        node_size: Size of nodes in the plot
        font_size: Font size for node labels
        arrowsize: Size of arrows on edges
        seed: Random seed for layout consistency
        
    Returns:
        Matplotlib figure containing the graph comparison
    """
    # Convert CausalGraph to NetworkX DiGraph if needed
    def convert_to_nx(graph):
        if isinstance(graph, nx.DiGraph):
            return graph
        elif hasattr(graph, 'get_nodes') and hasattr(graph, 'get_edges'):
            # Assume it's a CausalGraph
            nx_graph = nx.DiGraph()
            nx_graph.add_nodes_from(graph.get_nodes())
            nx_graph.add_edges_from(graph.get_edges())
            return nx_graph
        else:
            raise TypeError("Graph must be either a NetworkX DiGraph or a CausalGraph")
    
    nx_inferred = convert_to_nx(inferred_graph)
    nx_true = convert_to_nx(true_graph)
    
    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Get positions - use same seed for consistent layouts
    # Get all unique nodes from both graphs
    all_nodes = set(nx_inferred.nodes()).union(set(nx_true.nodes()))
    
    # Create a combined graph for layout calculation
    combined = nx.DiGraph()
    combined.add_nodes_from(all_nodes)
    combined.add_edges_from(nx_inferred.edges())
    combined.add_edges_from(nx_true.edges())
    
    # Generate layout for consistent node positions
    pos = nx.spring_layout(combined, seed=seed)
    
    # Plot inferred graph on the left
    axes[0].set_title('Inferred Graph')
    nx.draw_networkx(
        nx_inferred, 
        pos=pos,
        ax=axes[0],
        with_labels=True, 
        node_color=node_color,
        edge_color=edge_color,
        node_size=node_size,
        font_size=font_size,
        arrowsize=arrowsize
    )
    axes[0].axis('off')
    
    # Plot true graph on the right
    axes[1].set_title('True Graph')
    nx.draw_networkx(
        nx_true, 
        pos=pos,
        ax=axes[1],
        with_labels=True, 
        node_color=node_color,
        edge_color=edge_color,
        node_size=node_size,
        font_size=font_size,
        arrowsize=arrowsize
    )
    axes[1].axis('off')
    
    # Add super title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig

class TaskFamilyVisualizer:
    """Provides methods to visualize TaskFamily objects."""

    def __init__(self, task_family: TaskFamily):
        """Initializes the TaskFamilyVisualizer.

        Args:
            task_family: The TaskFamily object to visualize.
        """
        if not isinstance(task_family, TaskFamily):
             raise TypeError("Input must be a TaskFamily object.")
        self.task_family = task_family
        logger.debug("TaskFamilyVisualizer initialized.")

    def plot_family_comparison(
        self,
        output_dir: Optional[str] = None,
        display: bool = True,
        highlight_differences: bool = True,
        layout_func: Callable[..., dict] = nx.spring_layout,
        filename: str = "family_comparison.png",
        figsize_base: tuple = (6, 5),
        node_color="lightblue",
        edge_color="gray",
        node_size=500,
        font_size=8,
        arrowsize=10
    ):
        """Plots a comparison of graphs within the task family.

        Args:
            output_dir: Directory to save the plot. If None, plot is not saved.
            display: Whether to display the plot using plt.show().
            highlight_differences: If True, attempt to highlight added/removed edges (Not fully implemented).
            layout_func: NetworkX layout function to use.
            filename: Name of the file to save the plot as.
            figsize_base: Tuple setting the base figure size per subplot.
            node_color: Color for nodes.
            edge_color: Color for edges.
            node_size: Size of nodes.
            font_size: Font size for labels.
            arrowsize: Size of arrows on edges.
        """
        task_family = self.task_family
        num_graphs = len(task_family.graphs)
        if num_graphs == 0:
            logger.warning("No graphs in the task family to plot.")
            return

        fig_width = figsize_base[0] * num_graphs
        fig_height = figsize_base[1]
        fig = plt.figure(figsize=(fig_width, fig_height))
        logger.debug(f"Creating figure for {num_graphs} graphs with size {fig_width}x{fig_height}")

        try:
            pos = layout_func(task_family.base_graph)
        except Exception as e:
            logger.warning(f"Layout function failed on base graph: {e}. Using default spring_layout.")
            pos = nx.spring_layout(task_family.base_graph)

        for i, graph in enumerate(task_family.graphs):
            ax = fig.add_subplot(1, num_graphs, i + 1)
            title = f"Graph {i} (Base)" if i == 0 else f"Graph {i}"
            ax.set_title(title)

            current_pos = {node: pos[node] for node in graph.nodes() if node in pos}
            if len(current_pos) != len(graph.nodes()):
                 logger.warning(f"Graph {i} has nodes not in the base layout's position dictionary. Recalculating layout for this graph.")
                 current_pos = layout_func(graph)

            nx.draw(
                graph,
                pos=current_pos,
                ax=ax,
                with_labels=True,
                node_color=node_color,
                edge_color=edge_color,
                node_size=node_size,
                font_size=font_size,
                arrows=True,
                arrowsize=arrowsize
            )

        plt.tight_layout()

        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath)
                logger.info(f"Comparison plot saved to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save comparison plot to {output_dir}: {e}")

        if display:
            logger.debug("Displaying comparison plot.")
            plt.show()
        else:
            logger.debug("Closing comparison plot figure (display=False).")
            plt.close(fig)

    def generate_difficulty_heatmap(
        self,
        difficulty_metric: Callable[[nx.DiGraph], float],
        output_dir: Optional[str] = None,
        display: bool = True,
        filename: str = "difficulty_heatmap.png",
        figsize_scale: float = 0.8,
        min_fig_width: float = 8.0,
        cmap: str = "viridis",
        annot: bool = True,
        fmt: str = ".2f"
    ):
        """Generates a heatmap visualizing the difficulty of tasks in the family.

        Args:
            difficulty_metric: A function that takes a graph and returns a scalar difficulty.
            output_dir: Directory to save the plot. If None, plot is not saved.
            display: Whether to display the plot using plt.show().
            filename: Name of the file to save the plot as.
            figsize_scale: Scaling factor for heatmap width based on number of graphs.
            min_fig_width: Minimum width of the heatmap figure.
            cmap: Colormap for the heatmap.
            annot: Whether to annotate cells with difficulty values.
            fmt: String format for annotations.
        """
        task_family = self.task_family
        num_graphs = len(task_family.graphs)
        if num_graphs == 0:
            logger.warning("No graphs in the task family for heatmap.")
            return

        try:
            difficulties = np.array([[difficulty_metric(g) for g in task_family.graphs]])
        except Exception as e:
            logger.error(f"Difficulty metric failed: {e}")
            return

        fig_width = max(min_fig_width, num_graphs * figsize_scale)
        fig, ax = plt.subplots(figsize=(fig_width, 2))
        logger.debug(f"Creating heatmap for {num_graphs} graphs with size {fig_width}x2")

        try:
            sns.heatmap(
                difficulties,
                annot=annot,
                fmt=fmt,
                cmap=cmap,
                cbar=True,
                ax=ax,
                linewidths=.5
            )
        except Exception as e:
             logger.error(f"Seaborn heatmap generation failed: {e}")
             plt.close(fig)
             return

        ax.set_title("Task Difficulty Heatmap")
        ax.set_yticks([])
        ax.set_xticks(np.arange(num_graphs) + 0.5)
        ax.set_xticklabels([f"G{i}" for i in range(num_graphs)], rotation=0)

        plt.tight_layout()

        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath)
                logger.info(f"Heatmap saved to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save heatmap to {output_dir}: {e}")

        if display:
            logger.debug("Displaying heatmap.")
            plt.show()
        else:
            logger.debug("Closing heatmap figure (display=False).")
            plt.close(fig)

    def plot(self, ax: Optional[plt.Axes] = None, graph_index: int = 0, **kwargs):
        """Plots a single graph from the task family.

        Helper method, potentially used by other plotting functions or directly.

        Args:
            ax: Matplotlib Axes to plot on. If None, creates a new figure.
            graph_index: Index of the graph to plot (0 for base, 1+ for variations).
            **kwargs: Additional arguments passed to nx.draw().
        """
        if graph_index < 0 or graph_index >= len(self.task_family.graphs):
             raise IndexError(f"graph_index {graph_index} out of bounds.")

        graph_to_plot = self.task_family.graphs[graph_index]
        title = f"Graph {graph_index} (Base)" if graph_index == 0 else f"Graph {graph_index}"

        if ax is None:
             fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 8)))

        try:
             pos = nx.spring_layout(graph_to_plot, seed=42) # Consistent layout
             nx.draw(
                 graph_to_plot,
                 pos=pos,
                 ax=ax,
                 with_labels=True,
                 node_color=kwargs.get("node_color", "lightblue"),
                 edge_color=kwargs.get("edge_color", "gray"),
                 node_size=kwargs.get("node_size", 500),
                 font_size=kwargs.get("font_size", 8),
                 arrows=True,
                 arrowsize=kwargs.get("arrowsize", 10),
                 **{k:v for k,v in kwargs.items() if k not in ["figsize", "node_color", "edge_color", "node_size", "font_size", "arrowsize"]}
             )
             ax.set_title(title)
        except Exception as e:
             logger.error(f"Error drawing graph {graph_index}: {e}")
             ax.text(0.5, 0.5, f"Error plotting graph {graph_index}", ha='center', va='center')

        return ax 