import logging
from typing import Callable, Dict, OrderedDict

import networkx as nx
import numpy as np

from graphs.graph import GraphStructure
from graphs.graph_4_nodes import Graph4Nodes
from graphs.graph_5_nodes import Graph5Nodes
from graphs.graph_6_nodes import Graph6Nodes
from graphs.toy_graph import ToyGraph
from utils.sem_sampling import (
    create_grid_interventions,
    draw_interventional_samples_sem,
    sample_model,
)


def setup_observational_interventional(
    graph_type: str,
    n_obs: int = 100,
    n_int: int = 2,
    noiseless: bool = True,
    seed: int = 42,
    graph: GraphStructure = None,
):
    """
    Setup the graph based on the structure we are using
    """
    if not graph:
        assert graph_type in ["Toy", "Graph4", "Graph5", "Graph6", "ErdosRenyi"]
        if graph_type == "Toy":
            graph = ToyGraph()
        elif graph_type == "Graph4":
            graph = Graph4Nodes()
        elif graph_type == "Graph5":
            graph = Graph5Nodes()
        elif graph_type == "Graph6":
            graph = Graph6Nodes()

    logging.info("Sampling the observational data")

    D_O: Dict[str, np.ndarray] = sample_model(
        graph.SEM, sample_count=n_obs, graph=graph
    )

    exploration_set = graph.get_exploration_set()
    # getting the interventional data in two different formats
    logging.info("Sampling the interventional data")
    D_I = draw_interventional_samples_sem(
        exploration_set, graph, n_int=n_int, seed=seed, noiseless=noiseless
    )

    return D_O, D_I, exploration_set
