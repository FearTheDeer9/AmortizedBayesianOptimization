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
from avici.synthetic import Beta, RandInt, Uniform
from avici.synthetic.gene import GRNSergio

from diffcbed.replay_buffer import ReplayBuffer
from graphs.data_setup import setup_observational_interventional
from graphs.graph import GraphStructure
from graphs.graph_dream import Dream4Graph
from posterior_model.model import DoublyRobustModel, LinearSCMModel, NonLinearSCMModel
from utils.sem_sampling import (
    change_obs_data_format_to_bo_sergio,
    change_obs_data_format_to_mi,
)


def set_graph(
    graph_type: str, nonlinear: bool = False, seeds_int_data=13
) -> Dream4Graph:
    print(graph_type)
    assert graph_type in ["Ecoli1", "Ecoli2"]
    if graph_type == "Ecoli1":
        graph = Dream4Graph(yml_name=f"InSilicoSize10-{graph_type}")
        graph.set_target("3")
        graph.set_seed(seeds_int_data)
    elif graph_type == "Ecoli2":
        graph = Dream4Graph(yml_name=f"InSilicoSize10-{graph_type}")
        graph.set_target("3")
        graph.set_seed(seeds_int_data)

    print(f"THE ACTUAL PARENTS ARE {graph.parents[graph.target]}")
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
    p: int,
    filename: str,
    use_dr: bool = False,
):
    dr_string = "_dr" if use_dr else ""
    graph = set_graph(graph_type=graph_type, nonlinear=nonlinear)
    if noise:
        graph.set_noise(noise)

    b = Uniform(low=0.5, high=1.5)  # Basal reproduction rates
    k_param = Uniform(low=0.1, high=0.5)  # Interaction strengths (non-negative)
    k_sign_p = Beta(a=2.0, b=2.0)  # Probability of positive interaction signs
    cell_types = RandInt(low=1, high=2)  # Number of cell types
    # Just normally simulating the data
    hill = 2.0
    decays = 0.8
    noise_params = 0.1
    args = argparse.Namespace(scm_bias=0.0, noise_bias=0.0, old_er_logic=True)

    # Initialize the GRNSergio simulator
    simulator = GRNSergio(
        b=b,
        k_param=k_param,
        k_sign_p=k_sign_p,
        hill=hill,
        decays=decays,
        noise_params=noise_params,
        cell_types=cell_types,
        noise_type="dpd",
        sampling_state=10,
        dt=0.01,
        tech_noise_config=None,
        add_outlier_effect=False,
        add_lib_size_effect=False,
        add_dropout_effect=False,
        return_count_data=False,
        n_ko_genes=9,  # No knockouts; observational data only
    )
    rng = np.random.default_rng(seed=seed)
    n_int = 25
    graph.set_seed(seed)
    n_vars = 10
    g = graph.causal_env.adjacency_matrix
    n_observations_obs = 200  # Number of observational samples
    n_observations_int = 40

    # Run the simulator
    data = simulator(rng, g, n_observations_obs, n_observations_int)
    x_obs = data.x_obs  # Observational data
    x_int = data.x_int
    D_O = change_obs_data_format_to_bo_sergio(x_obs, graph.variables)

    D_I = {str(var): None for var in graph.variables}

    # Iterate over each element in x_int
    for x in x_int:
        intervention = x[:, 1]  # Select the second column (as per your structure)
        bo_temp = change_obs_data_format_to_bo_sergio(
            np.expand_dims(x, axis=0), graph.variables
        )  # Get formatted data
        # print(bo_temp)
        node = next(
            (i for i, val in enumerate(intervention) if val == 1), None
        )  # Find the node index where value is 1
        node = str(node)  # Convert node to string (since D_I keys are strings)

        # Check if D_I[node] is None (i.e., first time accessing this node)
        if D_I[node] is not None:
            # If D_I[node] exists, vertically stack bo_temp's values for the corresponding intervention_value
            for intervention_value in graph.variables:
                D_I[node][intervention_value] = np.hstack(
                    [
                        D_I[node][intervention_value].flatten(),
                        bo_temp[intervention_value].flatten(),
                    ]
                )
        else:
            # If this is the first time accessing D_I[node], initialize it with bo_temp values
            # Reshape bo_temp[intervention_value] if necessary (e.g., if it's a scalar)
            D_I[node] = {
                var: (
                    bo_temp[var].reshape(1, -1)
                    if bo_temp[var].ndim == 1
                    else bo_temp[var]
                )
                for var in graph.variables
            }

    # D_O_scaled, D_I_scaled = scale_data(D_O, D_I, graph, include_nint)
    D_O_scaled, D_I_scaled = D_O, D_I
    variables = [var for var in graph.variables if var != graph.target]

    if use_dr:
        topological_order = list(D_O.keys())
        D_O_mi = change_obs_data_format_to_mi(
            D_O,
            graph_variables=graph.variables,
            intervention_node=np.zeros(shape=len(graph.variables)),
        )
        robust_model = DoublyRobustModel(
            graph=graph,
            topological_order=topological_order,
            target=graph.target,
            indivdual=True,
            num_bootstraps=30,
        )
        buffer = ReplayBuffer(binary=True)
        buffer.update(D_O_mi)
        robust_model.run_method(buffer.data())
        probabilities = robust_model.prob_estimate
        if () in probabilities:
            del probabilities[()]
    else:
        combinations = []
        max_len = min(len(variables) + 1, 10)
        for r in range(1, max_len):
            combinations.extend(itertools.combinations(variables, r))
        probabilities = {combo: 1 / len(combinations) for combo in combinations}

    if nonlinear:
        model = NonLinearSCMModel(prior_probabilities=probabilities, graph=graph)
    else:
        model = LinearSCMModel(prior_probabilities=probabilities, graph=graph)

    model.set_data(D_O_scaled)
    model.calculate_metrics()
    if include_nint:
        for key in D_I_scaled:
            if D_I_scaled[key]:
                D_I_sample = D_I_scaled[key]
                int_len = len(D_I_scaled[key][graph.target])
                for n in range(int_len):
                    x_dict = {
                        obs_key: D_I_sample[obs_key][n]
                        for obs_key in D_I_sample
                        if obs_key != graph.target
                    }
                    y = D_I_sample[graph.target][n]
                    model.update_all(x_dict, y)
                    D_I = {key: np.array([D_I_sample[key][n]]) for key in D_I_sample}
                    model.add_data(D_I)
                    model.redefine_prior_probabilities()
    else:
        for n in range(50):
            x_dict = {
                obs_key: float(D_I_scaled[obs_key][n])
                for obs_key in D_I_scaled
                if obs_key != graph.target
            }
            y = float(D_I_scaled[graph.target][n])
            model.update_all(x_dict, y)
            D_I = {key: np.array([D_I_scaled[key][n]]) for key in D_I_scaled}
            model.redefine_prior_probabilities()

    results = {
        "accuracy": model.accuracy,
        "precision": model.precision,
        "recall": model.recall,
        "f1_score": model.f1_score,
    }
    print(results)
    nonlinear_string = "nonlinear" if nonlinear else "linear"
    include_int = "_nint" if include_nint else ""
    saved_file = f"results/posterior/{filename}/run{run_num}_{nonlinear_string}_{n_obs}{include_int}{dr_string}_D_{D}_p_{p}_noise_{noise}.pickle"
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
parser.add_argument("--use_dr", action="store_true")
parser.add_argument("--p", type=int, default=1, help="Expected degree for the graphs")

args = parser.parse_args()
graph_type = args.graph_type
run_num = args.run_num
seed = args.seeds_replicate
nonlinear = args.nonlinear
noise = args.noise
n_obs = args.n_obs
include_nint = args.include_nint
use_dr = args.use_dr
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
    use_dr=use_dr,
)
