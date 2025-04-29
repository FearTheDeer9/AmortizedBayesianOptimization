import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import List, Callable, Optional, Any
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

class TaskFamilyVisualizer:
    """Provides methods to visualize TaskFamily objects."""

    def __init__(self):
        """Initializes the TaskFamilyVisualizer."""
        # Basic initialization, can be extended later
        pass

    def plot_family_comparison(
        self,
        task_family: TaskFamily,
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
        """Plots a comparison of graphs within a task family.

        Args:
            task_family: The TaskFamily object containing graphs to compare.
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
        task_family: TaskFamily,
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
        """Generates a heatmap visualizing the difficulty of tasks in a family.

        Args:
            task_family: The TaskFamily object.
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