import argparse
import itertools

import numpy as np

from algorithms.PARENT_2_algorithm import PARENT
from diffcbed.envs.erdos_renyi import ErdosRenyi
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
    sample_model,
)


# Function to standardize data
def standardize(data, mean, std):
    return (data - mean) / std


# Function to reverse the standardization
def reverse_standardize(data, mean, std):
    return (data * std) + mean


# graph = ErdosRenyiGraph(num_nodes=10, nonlinear=True)
# interventions = {"2": 500}
# samples = sample_model(
#     static_sem=graph.SEM, sample_count=500, graph=graph, interventions=interventions
# )
# print(samples)
# print(samples["1"].mean())

# samples = sample_model(
#     static_sem=graph.SEM, sample_count=500, graph=graph, interventions=interventions
# )
# print(samples["1"].mean())


# samples = sample_model(
#     static_sem=graph.SEM, sample_count=500, graph=graph, interventions=interventions
# )
# print(samples["1"].mean())

# samples = sample_model(
#     static_sem=graph.SEM, sample_count=500, graph=graph, interventions=interventions
# )
# print(samples["1"].mean())
graph = ToyGraph()
D_O, D_I, exploration_set = setup_observational_interventional(
    graph_type=None,
    noiseless=True,
    seed=11,
    n_obs=200,
    n_int=2,
    graph=graph,
)

model = PARENT(
    graph=graph,
    nonlinear=True,
    use_doubly_robust=False,
    acquisition="PES",
    n_anchor_points=25,
)
model.set_values(D_O, D_I, exploration_set)
model.run_algorithm(T=10)
