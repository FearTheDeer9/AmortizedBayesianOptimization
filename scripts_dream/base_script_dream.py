import logging
import os
import pickle
import sys
from copy import deepcopy
from typing import List, Tuple

os.chdir("..")
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from algorithms.CBO_algorithm import CBO
from algorithms.PARENT_SCALE_algorithm import PARENT_SCALE
from algorithms.RANDOM_SCALE_algorithm import RANDOM_SCALE
from graphs.data_setup import setup_observational_interventional
from graphs.graph import GraphStructure
from graphs.graph_dream import Dream4Graph
from scripts.base_script import parse_args

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Set the format of log messages
    datefmt="%m/%d/%Y %I:%M:%S %p",  # Set the date format
)


def run_script_unknown(
    dream_env_name: str,
    run_num: int,
    noiseless: bool,
    noisy_string: str,
    seeds_int_data: int,
    n_obs: int,
    n_int: int,
    n_trials: int,
    nonlinear: bool,
    filename: str,
    run_cbo_unknown_dr_1: bool = False,
    run_cbo_unknown_dr_2: bool = False,
    run_cbo_unknown_all: bool = False,
    run_cbo_parents: bool = False,
    run_misspecified: bool = False,
    run_random: bool = False,
    use_iscm: bool = False,
):
    nonlinear_string = "_nonlinear" if nonlinear else ""
    use_iscm_string = "_iscm_" if use_iscm else ""
    if dream_env_name == "Ecoli1":
        graph = Dream4Graph(yml_name=f"InSilicoSize10-{dream_env_name}")
        graph.set_target("3")
        graph.set_seed(seeds_int_data)
    elif dream_env_name == "Ecoli2":
        graph = Dream4Graph(yml_name=f"InSilicoSize10-{dream_env_name}", seed=21)
        graph.set_target("6")
        graph.set_seed(seeds_int_data)

    D_O, D_I, exploration_set = setup_observational_interventional(
        graph_type=None,
        noiseless=noiseless,
        seed=seeds_int_data,
        n_obs=n_obs,
        n_int=n_int,
        graph=graph,
        use_iscm=use_iscm,
    )

    if run_cbo_unknown_dr_2:
        model = PARENT_SCALE(
            graph=graph,
            nonlinear=nonlinear,
            individual=True,
            use_doubly_robust=True,
            use_iscm=use_iscm,
        )
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
        filename_cbo_unknown = f"results/{dream_env_name}/run{run_num}_cbo_unknown_dr2_iscm_results_{n_obs}_{n_int}{nonlinear_string}.pickle"
        with open(filename_cbo_unknown, "wb") as file:
            pickle.dump(cbo_unknown_results_dict, file)

    if run_cbo_parents:
        parents = graph.parents[graph.target]
        edges = [(parent, graph.target) for parent in parents]
        graph.mispecify_graph(edges)
        graph.set_interventional_range_data(D_O)
        exploration_set = [(parent,) for parent in parents]
        model = CBO(graph=graph, use_iscm=use_iscm)
        model.set_values(D_O, D_I, exploration_set)
        model.run_algorithm(T=n_trials)
        (
            best_y_array,
            current_y_array,
            cost_array,
            intervention_set,
            intervention_value,
            average_uncertainty,
        ) = model.run_algorithm(T=n_trials)
        filename_cbo = f"results/{filename}/run{run_num}_cbo_results{use_iscm_string}{n_obs}_{n_int}{nonlinear_string}.pickle"
        cbo_results_dict = {
            "Best_Y": best_y_array,
            "Per_trial_Y": current_y_array,
            "Cost": cost_array,
            "Intervention_Set": intervention_set,
            "Intervention_Value": intervention_value,
            "Uncertainty": average_uncertainty,
        }
        with open(filename_cbo, "wb") as file:
            pickle.dump(cbo_results_dict, file)

    if run_random:
        model = RANDOM_SCALE(graph, use_iscm=use_iscm)
        model.set_values(D_O, D_I, exploration_set)
        (
            best_y_array,
            current_y_array,
            cost_array,
            intervention_set,
            intervention_value,
            average_uncertainty,
        ) = model.run_algorithm(T=n_trials)
        filename_cbo = f"results/{filename}/run{run_num}_cbo_results_random{use_iscm_string}{n_obs}_{n_int}{nonlinear_string}.pickle"
        cbo_results_dict = {
            "Best_Y": best_y_array,
            "Per_trial_Y": current_y_array,
            "Cost": cost_array,
            "Intervention_Set": intervention_set,
            "Intervention_Value": intervention_value,
            "Uncertainty": average_uncertainty,
        }
        with open(filename_cbo, "wb") as file:
            pickle.dump(cbo_results_dict, file)

    if run_misspecified:
        if dream_env_name == "Ecoli1":
            graph.misspecify_graph_random(seed=42)
        else:
            graph.misspecify_graph_random()
        parents = graph.parents[graph.target]
        edges = [(parent, graph.target) for parent in parents]
        graph.mispecify_graph(edges)
        graph.set_interventional_range_data(D_O)
        exploration_set = [(parent,) for parent in parents]
        model = CBO(graph=graph, use_iscm=use_iscm)
        model.set_values(D_O, D_I, exploration_set)
        model.run_algorithm(T=n_trials)
        (
            best_y_array,
            current_y_array,
            cost_array,
            intervention_set,
            intervention_value,
            average_uncertainty,
        ) = model.run_algorithm(T=n_trials)
        filename_cbo = f"results/{filename}/run{run_num}_cbo_misspecified_results{use_iscm_string}{n_obs}_{n_int}{nonlinear_string}.pickle"
        cbo_results_dict = {
            "Best_Y": best_y_array,
            "Per_trial_Y": current_y_array,
            "Cost": cost_array,
            "Intervention_Set": intervention_set,
            "Intervention_Value": intervention_value,
            "Uncertainty": average_uncertainty,
        }
        with open(filename_cbo, "wb") as file:
            pickle.dump(cbo_results_dict, file)


args = parse_args()
n_int = 1
seeds_int_data = args.seeds_replicate

n_anchor_points = args.n_anchor_points
n_trials = args.n_trials
n_obs = args.n_observational
run_num = args.run_num
noiseless = args.noiseless
nonlinear = args.nonlinear
noisy_string = "" if noiseless else "_noisy"

parent_method = args.parent_method
graph_type = args.graph_type

print(graph_type, parent_method, nonlinear)

if parent_method == "dr2":
    run_script_unknown(
        dream_env_name=graph_type,
        run_num=run_num,
        noiseless=noiseless,
        noisy_string=noisy_string,
        seeds_int_data=seeds_int_data,
        n_obs=n_obs,
        n_int=n_int,
        n_trials=n_trials,
        nonlinear=nonlinear,
        filename=graph_type,
        run_cbo_unknown_dr_2=True,
        run_cbo_parents=True,
        run_random=True,
        use_iscm=True,
    )
elif parent_method == "random":
    run_script_unknown(
        dream_env_name=graph_type,
        run_num=run_num,
        noiseless=noiseless,
        noisy_string=noisy_string,
        seeds_int_data=seeds_int_data,
        n_obs=n_obs,
        n_int=n_int,
        n_trials=n_trials,
        nonlinear=nonlinear,
        filename=graph_type,
        run_cbo_parents=True,
        run_random=False,
        run_misspecified=True,
        use_iscm=True,
    )
elif parent_method == "misspecified":
    run_script_unknown(
        dream_env_name=graph_type,
        run_num=run_num,
        noiseless=noiseless,
        noisy_string=noisy_string,
        seeds_int_data=seeds_int_data,
        n_obs=n_obs,
        n_int=n_int,
        n_trials=n_trials,
        nonlinear=nonlinear,
        filename=graph_type,
        run_cbo_parents=True,
        run_random=True,
        run_misspecified=True,
        use_iscm=True,
    )
