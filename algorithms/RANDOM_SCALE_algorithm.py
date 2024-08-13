# this is a random baseline, to see how the algorithm comparse to randomly selecting interventions
# second version which manages the exploration set in a more flexible way
# maybe use this if the manipulative variables are greater than 4
import itertools
import logging
import random
from copy import deepcopy
from typing import Dict, List, OrderedDict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper

import utils.cbo_functions as cbo_functions
import utils.ceo_utils as ceo_utils
from algorithms.BASE_algorithm import BASE
from diffcbed.replay_buffer import ReplayBuffer
from graphs.graph import GraphStructure
from utils.cbo_classes import TargetClass
from utils.sem_sampling import sample_model


# Function to standardize data
def standardize(data, mean, std):
    return (data - mean) / std


# Function to reverse the standardization
def reverse_standardize(data, mean, std):
    return (data * std) + mean


def sample_from_parameter_space(parameter_space: ParameterSpace):
    sample = {}
    for parameter in parameter_space.parameters:
        if isinstance(parameter, ContinuousParameter):
            sample[parameter.name] = random.uniform(parameter.min, parameter.max)
    return sample


class RANDOM_SCALE(BASE):
    """
    This is the class of my developed methodology
    """

    def __init__(
        self,
        graph: GraphStructure,
        noiseless: bool = True,
        cost_num: int = 1,
    ):
        self.graph = graph
        self.num_nodes = len(self.graph.variables)
        self.variables = self.graph.variables
        self.target = self.graph.target
        self.buffer = ReplayBuffer(binary=True)

        self.manipulative_variables = self.graph.get_sets()[2]
        self.noiseless = noiseless
        self.cost_num = cost_num

    def set_values(self, D_O, D_I, exploration_set):
        self.D_O = deepcopy(D_O)
        self.D_I = deepcopy(D_I)
        self.graph.set_interventional_range_data(self.D_O)
        self.topological_order = list(self.D_O.keys())
        # this is too much now, so we are continuously going to update it
        self.exploration_set = exploration_set
        self.es_to_n_mapping = {
            tuple(es): i for i, es in enumerate(self.exploration_set)
        }

    def run_algorithm(self, T: int = 30):

        # starting the setup for the CBO algorithm
        current_best_y = {
            tuple(interventions): [] for interventions in self.exploration_set
        }
        parameter_spaces: List[ParameterSpace] = [None] * len(self.exploration_set)
        target_classes: List[TargetClass] = [None] * len(self.exploration_set)
        self.model_list_overall: List[GPyModelWrapper] = [None] * len(
            self.exploration_set
        )

        for i in range(len(self.exploration_set)):
            parameter_spaces[i] = self.graph.get_parameter_space(
                self.exploration_set[i]
            )
            target_classes[i] = TargetClass(
                sem_model=self.graph.SEM,
                interventions=self.exploration_set[i],
                variables=self.graph.variables,
                graph=self.graph,
                noiseless=self.noiseless,
            )

        # STARTING THE ALGORITHM
        current_cost: List[int] = []
        global_opt: List[float] = []
        current_y: List[float] = []
        average_uncertainty: List[float] = []
        intervention_set: List[Tuple[str]] = []
        intervention_values: List[Tuple[float]] = []
        # global_opt.append(current_global_min)
        current_cost.append(0.0)
        cost_functions = self.graph.get_cost_structure(self.cost_num)

        for i in range(T):
            logging.info(f"----------------------ITERATION {i}----------------------")
            intervention_index = random.randint(0, len(self.exploration_set) - 1)
            intervention_set.append(self.exploration_set[intervention_index])
            sample = sample_from_parameter_space(parameter_spaces[intervention_index])

            x_new_list_intervention = np.array([sample[key] for key in sample]).reshape(
                1, -1
            )
            print(x_new_list_intervention)
            print(intervention_index)
            print(self.exploration_set[intervention_index])
            y_new = target_classes[intervention_index].compute_target(
                x_new_list_intervention
            )
            logging.info(
                f"The random interventional sample is {sample} with target {y_new}"
            )

            current_y.append(y_new[0][0])
            best_y = global_opt[i - 1] if global_opt else y_new[0][0]
            min_y = np.min(current_y)
            global_opt.append(min_y if min_y < best_y else best_y)
            logging.info(f"The current minimum {best_y}")
            total_cost = 0
            for j, val in enumerate(self.exploration_set[intervention_index]):
                total_cost += cost_functions[val](x_new_list_intervention[0, j])
            current_cost.append(current_cost[i] + total_cost)

        return (
            global_opt,
            current_y,
            current_cost,
            intervention_set,
            intervention_values,
            average_uncertainty,
        )
