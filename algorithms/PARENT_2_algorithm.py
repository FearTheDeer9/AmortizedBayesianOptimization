import itertools
from copy import deepcopy
from typing import Dict, Tuple

import numpy as np

from algorithms.BASE_algorithm import BASE
from diffcbed.envs.causal_environment import CausalEnvironment
from diffcbed.replay_buffer import ReplayBuffer
from graphs.graph import GraphStructure
from posterior_model.model import DoublyRobustModel, LinearSCMModel
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


class PARENT(BASE):
    """
    This is the class of my developed methodology
    """

    def __init__(self, graph: GraphStructure, graph_env: CausalEnvironment):
        self.graph = graph
        self.num_nodes = len(self.graph.variables)
        self.variables = self.graph.variables
        self.target = self.graph.target
        # self.acquisition_strategy = acquisition_strategy

        # setting up some more variables
        self.graph_env = graph_env
        self.buffer = ReplayBuffer(binary=True)

    def set_values(self, D_O, D_I, exploration_set):
        self.D_O = deepcopy(D_O)
        self.D_I = deepcopy(D_I)
        self.topological_order = list(self.D_O_bo_format.keys())
        # self.D_O = change_obs_data_format_to_mi(
        #     D_O,
        #     graph_variables=self.variables,
        #     intervention_node=np.zeros(shape=len(self.variables)),
        # )

        # self.D_I = change_int_data_format_to_mi(D_I, graph_variables=self.variables)
        # # just using the observational data for now
        # self.buffer.update(self.D_O)
        # for intervention in self.D_I:
        #     self.buffer.update(intervention)

    def determine_initial_probabilities(
        self, use_doubly_robust: bool = False
    ) -> Dict[Tuple, float]:
        if use_doubly_robust:
            """
            Still need to determine if I need this
            """
            pass
        else:
            manipulative_variables = self.graph.get_sets()[2]

            combinations = []
            for r in range(1, len(manipulative_variables) + 1):
                combinations.extend(itertools.combinations(manipulative_variables, r))

            probabilities = {combo: 1 / len(combinations) for combo in combinations}
        return probabilities

    def standardize_all_data(self):
        """
        This one just standardises the dataset
        """
        input_keys = [key for key in self.D_O.keys() if key != self.graph.target]
        means = {key: np.mean(self.D_O[key]) for key in input_keys}
        std = {key: np.std(self.D_O[key]) for key in input_keys}

        D_O_scaled = {}
        for key in self.D_O:
            if key in input_keys:
                D_O_scaled[key] = standardize(self.D_O[key], means[key], std[key])
            else:
                D_O_scaled[key] = self.D_O[key]

        interventions = self.D_I.keys()
        D_I_scaled = {intervention: {} for intervention in interventions}
        for intervention in interventions:
            for key in self.D_I[intervention]:
                if key in input_keys:
                    D_I_scaled[intervention][key] = standardize(
                        self.D_I[intervention][key], means[key], std[key]
                    )
                else:
                    D_I_scaled[intervention][key] = self.D_I[intervention][key]

        self.D_O_scaled = D_O_scaled
        self.D_I_scaled = D_I_scaled

    def run_algorithm(self, T: int = 30):
        self.prior_probabilities = self.determine_initial_probabilities()
        self.posterior_model = LinearSCMModel(self.prior_probabilities, self.graph)

        # normalize the datasets
        self.standardize_all_data()
        self.posterior_model.set_data(self.D_O_scaled)

        # update the posterior probabilities now with the interventional data
        for key in self.D_I_scaled:
            num_samples = len(self.D_I_scaled[key])
            for n in range(num_samples):
                x_dict = {
                    key: self.D_I_scaled[key][n]
                    for key in self.D_I_scaled
                    if key != self.graph.target
                }
                y = self.D_I_scaled[key][self.graph.target][n]
                self.posterior_model.update_all(x_dict, y)
