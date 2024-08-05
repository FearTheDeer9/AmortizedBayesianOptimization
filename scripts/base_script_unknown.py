import argparse
import logging
import pickle
from copy import deepcopy
from typing import List, Tuple

from algorithms.PARENT_2_algorithm import PARENT
from graphs.data_setup import setup_observational_interventional
from graphs.graph import GraphStructure
from graphs.graph_4_nodes import Graph4Nodes
from graphs.graph_5_nodes import Graph5Nodes
from graphs.graph_6_nodes import Graph6Nodes
from graphs.toy_graph import ToyGraph

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Set the format of log messages
    datefmt="%m/%d/%Y %I:%M:%S %p",  # Set the date format
)


def set_graph(graph_type: str) -> GraphStructure:
    assert graph_type in ["Toy", "Graph4", "Graph5", "Graph6"]
    if graph_type == "Toy":
        graph = ToyGraph()
    elif graph_type == "Graph4":
        graph = Graph4Nodes()
    elif graph_type == "Graph5":
        graph = Graph5Nodes()
    elif graph_type == "Graph6":
        graph = Graph6Nodes()
    return graph


def run_script_unknown(
    graph_type: str,
    run_num: int,
    noiseless: bool,
    noisy_string: str,
    seeds_int_data: int,
    n_obs: int,
    n_int: int,
    n_trials: int,
    filename: str,
    scale_data: bool = False,
):

    graph = set_graph(graph_type)
    D_O, D_I, exploration_set = setup_observational_interventional(
        graph_type=None,
        noiseless=noiseless,
        seed=seeds_int_data,
        n_obs=n_obs,
        n_int=n_int,
        graph=graph,
    )

    model = PARENT(graph=graph, nonlinear=True)
    model.set_values(D_O, D_I, exploration_set)
    (
        best_y_array,
        current_y_array,
        cost_array,
        intervention_set,
        intervention_value,
        average_uncertainty,
    ) = model.run_algorithm(T=n_trials, show_graphics=False)

    cbo_unknown_results_dict = {
        "Best_Y": best_y_array,
        "Per_trial_Y": current_y_array,
        "Cost": cost_array,
        "Intervention_Set": intervention_set,
        "Intervention_Value": intervention_value,
        "Uncertainty": average_uncertainty,
    }
    filename_cbo_unknown = f"results/{filename}/run{run_num}_cbo_unknown_results_{n_obs}_{n_int}{noisy_string}.pickle"
    with open(filename_cbo_unknown, "wb") as file:
        pickle.dump(cbo_unknown_results_dict, file)
