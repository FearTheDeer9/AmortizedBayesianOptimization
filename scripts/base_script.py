import logging
import os
import pickle
import sys
from copy import deepcopy
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from algorithms.BO_algorithm import BO
from algorithms.CBO_algorithm import CBO
from algorithms.CEO_algorithm import CEO
from config import RUN_BO, RUN_CBO, RUN_CEO, SAVE_RUN
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


def run_script(
    graph_type: str,
    run_num: int,
    all_graph_edges: List[List[Tuple]],
    noiseless: bool,
    noisy_string: str,
    seeds_int_data: int,
    n_obs: int,
    n_int: int,
    n_anchor_points: int,
    n_trials: int,
    filename: str,
):
    if run_num <= 5:
        safe_optimization = False
    else:
        safe_optimization = True
    # run the CEO method using all the edges and then the CBO for each of the e

    # using this as the interventional and observational data
    D_O, D_I, exploration_set = setup_observational_interventional(
        graph_type=graph_type,
        noiseless=noiseless,
        seed=seeds_int_data,
        n_obs=n_obs,
        n_int=n_int,
    )
    # print(D_O)
    print(D_I)
    # using this as the interventional and observational data

    filename_D_O = f"data/{filename}/run{run_num}_D_O{noisy_string}.pickle"
    filename_D_I = f"data/{filename}/run{run_num}_D_I{noisy_string}.pickle"
    filename_es = f"data/{filename}/run{run_num}_es{noisy_string}.pickle"

    if SAVE_RUN:
        with open(filename_D_O, "wb") as file:
            pickle.dump(D_O, file)

        with open(filename_D_I, "wb") as file:
            pickle.dump(D_I, file)

        with open(filename_es, "wb") as file:
            pickle.dump(exploration_set, file)

    if RUN_CEO:
        model: CEO = CEO(
            graph_type=graph_type,
            all_graph_edges=all_graph_edges,
            n_obs=n_obs,
            n_int=n_int,
            n_anchor_points=n_anchor_points,
            seed=seeds_int_data,
            noiseless=noiseless,
        )

        model.set_values(deepcopy(D_O), deepcopy(D_I), exploration_set)
        (
            best_y_array,
            current_y_array,
            cost_array,
            intervention_set,
            intervention_value,
        ) = model.run_algorithm(T=n_trials, safe_optimization=True)

        ceo_result_dict = {
            "Best_Y": best_y_array,
            "Per_trial_Y": current_y_array,
            "Cost": cost_array,
            "Intervention_Set": intervention_set,
            "Intervention_Value": intervention_value,
        }

        filename_ceo = f"results/{filename}/run_ceo_{run_num}_results_{n_obs}_{n_int}_{noisy_string}.pickle"

        if SAVE_RUN:
            with open(filename_ceo, "wb") as file:
                pickle.dump(ceo_result_dict, file)

    if RUN_CBO:
        # now for the CBO algorithm
        for i, edges in enumerate(all_graph_edges):
            graph = set_graph(graph_type)
            graph.mispecify_graph(edges)
            cbo_model = CBO(graph=graph, noiseless=noiseless)
            cbo_model.set_values(deepcopy(D_O), deepcopy(D_I), exploration_set)

            (
                best_y_array,
                current_y_array,
                cost_array,
                intervention_set,
                intervention_value,
            ) = cbo_model.run_algorithm(T=n_trials)

            cbo_results_dict = {
                "Best_Y": best_y_array,
                "Per_trial_Y": current_y_array,
                "Cost": cost_array,
                "Intervention_Set": intervention_set,
                "Intervention_Value": intervention_value,
            }

            filename_cbo = f"results/{filename}/run{run_num}_cbo_results_{n_obs}_{n_int}_graph_{i}{noisy_string}.pickle"

            if SAVE_RUN:
                with open(filename_cbo, "wb") as file:
                    pickle.dump(cbo_results_dict, file)

    # now for the BO implementation

    if RUN_BO:
        graph = set_graph(graph_type)
        graph.break_dependency_structure()
        bo_model = BO(graph=graph, noiseless=noiseless)
        bo_model.set_values(deepcopy(D_O))
        best_y_array, current_y_array, cost_array = bo_model.run_algorithm(T=n_trials)
        cbo_results_dict = {
            "Best_Y": best_y_array,
            "Per_trial_Y": current_y_array,
            "Cost": cost_array,
        }

        filename_bo = f"results/{filename}/run{run_num}_bo_results_{n_obs}_{n_int}_{noisy_string}.pickle"
        if SAVE_RUN:
            with open(filename_bo, "wb") as file:
                pickle.dump(cbo_results_dict, file)
