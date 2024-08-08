import logging
import os
import pickle
import sys
from copy import deepcopy
from typing import List, Tuple

os.chdir("..")
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from algorithms.PARENT_SCALE_algorithm import PARENT_SCALE
from graphs.data_setup import setup_observational_interventional
from graphs.graph import GraphStructure
from graphs.graph_4_nodes import Graph4Nodes
from graphs.graph_5_nodes import Graph5Nodes
from graphs.graph_6_nodes import Graph6Nodes
from graphs.graph_chain import ChainGraph
from graphs.toy_graph import ToyGraph
from scripts.base_script import parse_args

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Set the format of log messages
    datefmt="%m/%d/%Y %I:%M:%S %p",  # Set the date format
)


def set_graph(graph_type: str) -> GraphStructure:
    assert graph_type in ["Toy", "Graph4", "Graph5", "Graph6", "Graph10"]
    if graph_type == "Toy":
        graph = ToyGraph()
    elif graph_type == "Graph4":
        graph = Graph4Nodes()
    elif graph_type == "Graph5":
        graph = Graph5Nodes()
    elif graph_type == "Graph6":
        graph = Graph6Nodes()
    elif graph_type == "Graph10":
        graph = ChainGraph(num_nodes=10)
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
    nonlinear: bool,
    filename: str,
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

    model = PARENT_SCALE(graph=graph, nonlinear=nonlinear)
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


args = parse_args()
n_int = 2
seeds_int_data = args.seeds_replicate

n_anchor_points = args.n_anchor_points
n_trials = args.n_trials
n_obs = args.n_observational
run_num = args.run_num
noiseless = args.noiseless
noisy_string = "" if noiseless else "_noisy"

run_script_unknown(
    graph_type="Graph10",
    run_num=run_num,
    noiseless=noiseless,
    noisy_string=noisy_string,
    seeds_int_data=seeds_int_data,
    n_obs=n_obs,
    n_int=n_int,
    n_trials=n_trials,
    nonlinear=False,
    filename="Graph10Unknown",
)
