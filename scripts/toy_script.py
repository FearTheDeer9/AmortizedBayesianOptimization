import argparse
import logging
import os
import pickle
import sys
from copy import deepcopy

os.chdir("..")

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np

from algorithms.BO_algorithm import BO
from algorithms.CBO_algorithm import CBO
from algorithms.CEO_algorithm import CEO
from config import RUN_BO, RUN_CBO, RUN_CEO
from graphs.data_setup import setup_observational_interventional
from graphs.toy_graph import ToyGraph

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Set the format of log messages
    datefmt="%m/%d/%Y %I:%M:%S %p",  # Set the date format
)


# this is the arguments for running the script
parser = argparse.ArgumentParser()
parser.add_argument(
    "--seeds_replicate",
    type=int,
    # nargs="+",
    # help="seed for replicate: list of seeds for diff int. data",
)
parser.add_argument("--n_observational", type=int)
parser.add_argument("--n_trials", type=int)
parser.add_argument("--n_anchor_points", type=int)
parser.add_argument("--run_num", type=int)
parser.add_argument("--noiseless", type=bool)

# using the arguments
args = parser.parse_args()
n_int = 2
seeds_int_data = args.seeds_replicate

n_anchor_points = args.n_anchor_points
n_trials = args.n_trials
n_obs = args.n_observational
run_num = args.run_num
noiseless = args.noiseless
noisy_string = "" if noiseless else "_noisy"
if run_num <= 5:
    safe_optimization = False
else:
    safe_optimization = True
# run the CEO method using all the edges and then the CBO for each of the e

all_graph_edges = [
    [("X", "Z"), ("Z", "Y")],
    [("X", "Z"), ("X", "Y")],
    [("X", "Z"), ("Z", "Y"), ("X", "Y")],
    [("Z", "X"), ("Z", "Y")],
    [("X", "Y"), ("Z", "Y")],
    [("Z", "X"), ("X", "Y"), ("Z", "Y")],
    [("Z", "X"), ("X", "Y")],
]

# using this as the interventional and observational data
# D_O, D_I, exploration_set = setup_observational_interventional(
#     graph_type="Toy", noiseless=noiseless, seed=seeds_int_data
# )

print(os.getcwd())
with open("/Users/jeandurand/Documents/Masters Thesis/CEO/D_O.pickle", "rb") as file:
    D_O = pickle.load(file)

with open("/Users/jeandurand/Documents/Masters Thesis/CEO/D_I.pickle", "rb") as file:
    D_I = pickle.load(file)

print("The observational dataset")
print(D_O)
print("The interventional dataset")
print(D_I)
exploration_set = [("X",), ("Z",), ("X", "Z")]
# using this as the interventional and observational data


filename_D_O = f"data/ToyGraph/run{run_num}_D_O{noisy_string}.pickle"
filename_D_I = f"data/ToyGraph/run{run_num}_D_I{noisy_string}.pickle"
filename_es = f"data/ToyGraph/run{run_num}_es{noisy_string}.pickle"

with open(filename_D_O, "wb") as file:
    pickle.dump(D_O, file)

with open(filename_D_I, "wb") as file:
    pickle.dump(D_I, file)

with open(filename_es, "wb") as file:
    pickle.dump(exploration_set, file)


if RUN_CEO:
    model: CEO = CEO(
        graph_type="Toy",
        all_graph_edges=all_graph_edges,
        n_obs=n_obs,
        n_int=n_int,
        n_anchor_points=n_anchor_points,
        seed=seeds_int_data,
        noiseless=noiseless,
    )
    model.set_values(deepcopy(D_O), deepcopy(D_I), exploration_set)
    best_y_array, current_y_array, cost_array = model.run_algorithm(
        T=n_trials, safe_optimization=safe_optimization
    )
    ceo_result_dict = {
        "Best_Y": best_y_array,
        "Per_trial_Y": current_y_array,
        "Cost": cost_array,
    }

    filename_ceo = f"results/ToyGraph/run_ceo_{run_num}_results{noisy_string}.pickle"
    with open(filename_ceo, "wb") as file:
        pickle.dump(ceo_result_dict, file)


if RUN_CBO:
    # now for the CBO algorithm
    for i, edges in enumerate(all_graph_edges):
        graph = ToyGraph()
        graph.mispecify_graph(edges)
        cbo_model = CBO(graph=graph)
        cbo_model.set_values(deepcopy(D_O), deepcopy(D_I), exploration_set)
        best_y_array, current_y_array, cost_array = cbo_model.run_algorithm(T=n_trials)
        cbo_results_dict = {
            "Best_Y": best_y_array,
            "Per_trial_Y": current_y_array,
            "Cost": cost_array,
        }
        filename_cbo = (
            f"results/ToyGraph/run{run_num}_cbo_results_graph_{i}{noisy_string}.pickle"
        )
        with open(filename_cbo, "wb") as file:
            pickle.dump(cbo_results_dict, file)

    # # adding the CBO for the real graph
    # graph = ToyGraph()
    # cbo_model = CBO(graph=graph)
    # cbo_model.set_values(deepcopy(D_O), deepcopy(D_I), exploration_set)
    # best_y_array, current_y_array, cost_array = cbo_model.run_algorithm(T=n_trials)
    # cbo_results_dict = {
    #     "Best_Y": best_y_array,
    #     "Per_trial_Y": current_y_array,
    #     "Cost": cost_array,
    # }
    # filename_cbo = (
    #     f"results/ToyGraph/run{run_num}_cbo_results_true_graph{noisy_string}.pickle"
    # )
    # with open(filename_cbo, "wb") as file:
    #     pickle.dump(cbo_results_dict, file)


# now for the BO implementation

if RUN_BO:
    graph = ToyGraph()
    graph.break_dependency_structure()
    bo_model = BO(graph=graph)
    bo_model.set_values(deepcopy(D_O))
    best_y_array, current_y_array, cost_array = bo_model.run_algorithm(T=n_trials)
    cbo_results_dict = {
        "Best_Y": best_y_array,
        "Per_trial_Y": current_y_array,
        "Cost": cost_array,
    }

    filename_bo = f"results/ToyGraph/run{run_num}_bo_results{noisy_string}.pickle"
    with open(filename_bo, "wb") as file:
        pickle.dump(cbo_results_dict, file)
