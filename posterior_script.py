import itertools

import numpy as np

from algorithms.PARENT_2_algorithm import PARENT
from diffcbed.replay_buffer import ReplayBuffer
from graphs.data_setup import setup_observational_interventional
from graphs.graph_6_nodes import Graph6Nodes
from graphs.graph_chain import ChainGraph
from graphs.graph_erdos_renyi import ErdosRenyiGraph
from graphs.toy_graph import ToyGraph
from posterior_model.model import DoublyRobustModel, LinearSCMModel, NonLinearSCMModel
from utils.sem_sampling import (
    change_int_data_format_to_mi,
    change_obs_data_format_to_mi,
)


# Function to standardize data
def standardize(data, mean, std):
    return (data - mean) / std


# Function to reverse the standardization
def reverse_standardize(data, mean, std):
    return (data * std) + mean


# graph = ErdosRenyiGraph(num_nodes=6, nonlinear=False)
# graph = Graph6Nodes()
# # graph = ChainGraph(num_nodes=5, nonlinear=False)
# n_obs = 200
# n_int = 3
# manipulative_variables = graph.get_sets()[2]
# manipulative_variables = graph.variables
# manipulative_variables = [var for var in manipulative_variables if var != graph.target]
# manipulative_variables = [item[0] for item in manipulative_variables]

# # random_seed = np.random.randint(1, 10000)
# D_O, D_I, exploration_set = setup_observational_interventional(
#     graph_type=None, graph=graph, n_obs=n_obs, n_int=n_int
# )

# # set everything up so that it works better with your calculated likelihood
# input_keys = [key for key in D_O.keys() if key != graph.target]
# means = {key: np.mean(D_O[key]) for key in input_keys}
# std = {key: np.std(D_O[key]) for key in input_keys}

# D_O_scaled = {}
# for key in D_O:
#     if key in input_keys:
#         D_O_scaled[key] = standardize(D_O[key], means[key], std[key])
#         # D_O_scaled[key] = D_O[key]
#     else:
#         D_O_scaled[key] = D_O[key]

# interventions = D_I.keys()
# D_I_scaled = {intervention: {} for intervention in interventions}
# for intervention in interventions:
#     for key in D_I[intervention]:
#         if key in input_keys:
#             D_I_scaled[intervention][key] = standardize(
#                 D_I[intervention][key], means[key], std[key]
#             )
#             # D_I_scaled[intervention][key] = D_I[intervention][key]
#         else:
#             D_I_scaled[intervention][key] = D_I[intervention][key]

# # Generate all unique combinations
# combinations = []
# for r in range(1, len(manipulative_variables) + 1):
#     combinations.extend(itertools.combinations(manipulative_variables, r))

# # Print the combinations
# # print(D_O_scaled)

# prior_probabilities = {combo: 1 / len(combinations) for combo in combinations}
# print(prior_probabilities)
# model = NonLinearSCMModel(prior_probabilities, graph)
# model.set_data(D_O_scaled)

# for n in range(5):
#     x_dict = {key: D_O_scaled[key][n] for key in D_O}
#     y = D_O_scaled[graph.target][n]
#     model.update_all(x_dict, y)


# intervention = ("Z",)
# for intervention in D_I_scaled:
#     print(f"-----------------{intervention}-------------------")
#     for n in range(n_int):
#         x_dict = {
#             key: D_I_scaled[intervention][key][n] for key in D_O if key != graph.target
#         }
#         y = D_I_scaled[intervention][graph.target][n]
#         model.update_all(x_dict, y)
#         D_I = {
#             key: np.array([D_I_scaled[intervention][key][n]])
#             for key in D_I_scaled[intervention]
#         }
# print(D_I)
# model.add_data(D_I)
# print(model.prior_probabilities)

# print(graph.parents[graph.target])
# compare how it change for obs data vs int data

n_obs = 200
n_int = 2
seed = np.random.randint(1, 10000)
noiseless = True
# graph = ChainGraph(num_nodes=5, nonlinear=False)
# graph = Graph6Nodes()
graph = ToyGraph()

D_O, D_I, exploration_set = setup_observational_interventional(
    graph_type="Toy",
    noiseless=noiseless,
    seed=seed,
    n_obs=n_obs,
    n_int=n_int,
    graph=graph,
)

# exploration_set = [("X",), ("Z",)]
# exploration_set = [("Z",)]
model = PARENT(graph, scale_data=False)
model.set_values(D_O, D_I, exploration_set)
model.run_algorithm()

# THE DOUBLY ROBUST METHOD
# n_obs = 200
# n_int = 2
# seed = np.random.randint(1, 10000)
# noiseless = True
# graph = ToyGraph()
# buffer = ReplayBuffer(binary=True)
# D_O, D_I, exploration_set = setup_observational_interventional(
#     graph_type="Graph6",
#     noiseless=noiseless,
#     seed=seed,
#     n_obs=n_obs,
#     n_int=n_int,
#     graph=graph,
# )
# topological_order = list(D_O.keys())
# D_O = change_obs_data_format_to_mi(
#     D_O,
#     graph_variables=graph.variables,
#     intervention_node=np.zeros(shape=len(graph.variables)),
# )

# robust_model = DoublyRobustModel(
#     graph=graph,
#     topological_order=topological_order,
#     target=graph.target,
#     num_bootstraps=30,
# )

# robust_model.covariance_matrix = np.cov(D_O.samples.T)
# buffer.update(D_O)
# robust_model.run_method(buffer.data())
