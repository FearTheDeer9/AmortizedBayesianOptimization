import argparse
import itertools
import os
import pickle
import sys

os.chdir("..")
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np

from graphs.data_setup import setup_observational_interventional
from graphs.graph import GraphStructure
from graphs.graph_erdos_renyi import ErdosRenyiGraph
from posterior_model.model import LinearSCMModel, NonLinearSCMModel


def set_graph(graph_type: str, nonlinear: bool = False) -> ErdosRenyiGraph:
    assert graph_type in [
        "Toy",
        "Graph4",
        "Graph5",
        "Graph6",
        "Graph10",
        "Erdos5",
        "Erdos10",
        "Erdos15",
        "Erdos20",
    ]
    if graph_type == "Erdos5":
        graph = ErdosRenyiGraph(num_nodes=5, nonlinear=nonlinear)
    elif graph_type == "Erdos10":
        graph = ErdosRenyiGraph(num_nodes=10, nonlinear=nonlinear)
        graph.set_target("1")
    elif graph_type == "Erdos15":
        graph = ErdosRenyiGraph(num_nodes=15, nonlinear=nonlinear)
        graph.set_target("8")
    elif graph_type == "Erdos20":
        graph = ErdosRenyiGraph(num_nodes=20, nonlinear=nonlinear)
        graph.set_target("18")
    return graph


# Function to standardize data
def standardize(data, mean, std):
    return (data - mean) / std


# Function to reverse the standardization
def reverse_standardize(data, mean, std):
    return (data * std) + mean


def scale_data(D_O, D_I, graph, include_nint):
    input_keys = [key for key in D_O.keys() if key != graph.target]
    means = {key: np.mean(D_O[key]) for key in input_keys}
    stds = {key: np.std(D_O[key]) for key in input_keys}

    D_O_scaled = {}
    for key in D_O:
        if key in input_keys:
            D_O_scaled[key] = standardize(D_O[key], means[key], stds[key])
        else:
            D_O_scaled[key] = D_O[key]

    interventions = D_I.keys()
    if include_nint:
        D_I_scaled = {intervention: {} for intervention in interventions}
        for intervention in interventions:
            for key in D_I[intervention]:
                if key in input_keys:
                    D_I_scaled[intervention][key] = standardize(
                        D_I[intervention][key], means[key], stds[key]
                    )
                else:
                    D_I_scaled[intervention][key] = D_I[intervention][key]
    else:
        D_I_scaled = {}
        for key in D_I:
            if key in input_keys:
                D_I_scaled[key] = standardize(D_I[key], means[key], stds[key])
            else:
                D_I_scaled[key] = D_I[key]

    D_O_scaled = D_O_scaled
    D_I_scaled = D_I_scaled
    return D_O_scaled, D_I_scaled


def run_posterior(
    graph_type: str,
    run_num: int,
    seed: int,
    nonlinear: bool,
    noise: float,
    n_obs: int,
    include_nint,
    D: int,
    p: float,
    filename: str,
):
    graph = set_graph(graph_type=graph_type, nonlinear=nonlinear)
    if noise:
        graph.set_noise(noise)

    n_int = 5
    D_O, D_I, _ = setup_observational_interventional(
        graph_type=None, n_obs=n_obs, n_int=n_int, graph=graph, seed=seed
    )

    if not include_nint:
        D_I, _, _ = setup_observational_interventional(
            graph_type=None, n_obs=20, n_int=n_int, graph=graph, seed=seed + 2
        )

    D_O_scaled, D_I_scaled = scale_data(D_O, D_I, graph, include_nint)
    variables = [var for var in graph.variables if var != graph.target]

    combinations = []
    for r in range(1, len(variables) + 1):
        combinations.extend(itertools.combinations(variables, r))

    probabilities = {combo: 1 / len(combinations) for combo in combinations}
    if nonlinear:
        model = NonLinearSCMModel(prior_probabilities=probabilities, graph=graph, D=D)
    else:
        model = LinearSCMModel(prior_probabilities=probabilities, graph=graph)

    model.set_data(D_O_scaled)
    if include_nint:
        for key in D_I_scaled:
            D_I_sample = D_I_scaled[key]
            for n in range(n_int):
                x_dict = {
                    obs_key: D_I_sample[obs_key][n]
                    for obs_key in D_I_sample
                    if obs_key != graph.target
                }
                y = D_I_sample[graph.target][n]
                model.update_all(x_dict, y)
                D_I = {key: np.array([D_I_sample[key][n]]) for key in D_I_sample}
                model.add_data(D_I)
                print(model.prior_probabilities)
    else:
        for n in range(20):
            x_dict = {
                obs_key: D_I_scaled[obs_key][n]
                for obs_key in D_I_scaled
                if obs_key != graph.target
            }
            y = D_I_scaled[graph.target][n]
            model.update_all(x_dict, y)
            D_I = {key: np.array([D_I_scaled[key][n]]) for key in D_I_scaled}
            model.add_data(D_I)

    results = {
        "accuracy": model.accuracy,
        "precision": model.precision,
        "recall": model.recall,
        "f1_score": model.f1_score,
    }
    print(results)
    nonlinear_string = "nonlinear" if nonlinear else "linear"
    include_int = "_nint" if include_nint else ""
    saved_file = f"results/posterior/{filename}/run{run_num}_{nonlinear_string}_{n_obs}{include_int}_D_{D}_noise_.pickle"
    with open(saved_file, "wb") as file:
        pickle.dump(results, file)


parser = argparse.ArgumentParser(description="Posterior Distribution Evaluation")
parser.add_argument("--graph_type", type=str, default="Erdos5")
parser.add_argument("--run_num", type=int)
parser.add_argument("--nonlinear", action="store_true")
parser.add_argument("--noise", type=float, default=None)
parser.add_argument("--n_obs", type=int, default=200)
parser.add_argument("--D", type=int, default=1000)
parser.add_argument("--seeds_replicate", type=int)
parser.add_argument("--include_nint", action="store_true")
parser.add_argument(
    "--p", type=float, default=1.0, help="Expected degree for the graphs"
)

args = parser.parse_args()
graph_type = args.graph_type
run_num = args.run_num
seed = args.seeds_replicate
nonlinear = args.nonlinear
noise = args.noise
n_obs = args.n_obs
include_nint = args.include_nint
D = args.D
p = args.p

run_posterior(
    graph_type=graph_type,
    run_num=run_num,
    seed=seed,
    nonlinear=nonlinear,
    noise=noise,
    n_obs=n_obs,
    include_nint=include_nint,
    p=p,
    D=D,
    filename=graph_type,
)
