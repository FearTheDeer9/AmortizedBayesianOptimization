import itertools
import logging
from copy import deepcopy
from typing import Dict, List, OrderedDict, Tuple

import numpy as np
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from GPy.models.gp_regression import GPRegression

import utils.cbo_functions as cbo_functions
import utils.ceo_utils as ceo_utils
from algorithms.BASE_algorithm import BASE
from diffcbed.envs.causal_environment import CausalEnvironment
from diffcbed.replay_buffer import ReplayBuffer
from graphs.graph import GraphStructure
from posterior_model.model import (
    DoublyRobustModel,
    LinearSCMModel,
    NonLinearSCMModel,
    SCMModel,
)
from utils.cbo_classes import DoFunctions, TargetClass
from utils.sem_sampling import (
    change_int_data_format_to_mi,
    change_intervention_list_format,
    change_obs_data_format_to_mi,
    sample_model,
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

    def __init__(
        self,
        graph: GraphStructure,
        # graph_env: CausalEnvironment,
        nonlinear: bool = True,
        causal_prior: bool = True,
        noiseless: bool = True,
        cost_num: int = 1,
        scale_data: bool = True,
    ):
        self.graph = graph
        self.num_nodes = len(self.graph.variables)
        self.variables = self.graph.variables
        self.target = self.graph.target
        self.nonlinear = nonlinear
        # self.acquisition_strategy = acquisition_strategy

        # setting up some more variables
        # self.graph_env = graph_env
        self.buffer = ReplayBuffer(binary=True)

        self.manipulative_variables = self.graph.get_sets()[2]
        self.causal_prior = causal_prior
        self.noiseless = noiseless
        self.cost_num = cost_num
        self.scale_data = scale_data

    def set_values(self, D_O, D_I, exploration_set):
        self.D_O = deepcopy(D_O)
        self.D_I = deepcopy(D_I)
        self.topological_order = list(self.D_O.keys())
        self.exploration_set = exploration_set
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
            variables = [
                var for var in self.graph.variables if var != self.graph.target
            ]

            combinations = []
            for r in range(1, len(variables) + 1):
                combinations.extend(itertools.combinations(variables, r))

            probabilities = {combo: 1 / len(combinations) for combo in combinations}
        return probabilities

    def standardize_all_data(self):
        """
        This one just standardises the dataset
        """
        input_keys = [key for key in self.D_O.keys() if key != self.graph.target]
        self.means = {key: np.mean(self.D_O[key]) for key in input_keys}
        self.stds = {key: np.std(self.D_O[key]) for key in input_keys}

        D_O_scaled = {}
        for key in self.D_O:
            if key in input_keys:
                D_O_scaled[key] = standardize(
                    self.D_O[key], self.means[key], self.stds[key]
                )
            else:
                D_O_scaled[key] = self.D_O[key]

        interventions = self.D_I.keys()
        D_I_scaled = {intervention: {} for intervention in interventions}
        for intervention in interventions:
            for key in self.D_I[intervention]:
                if key in input_keys:
                    D_I_scaled[intervention][key] = standardize(
                        self.D_I[intervention][key], self.means[key], self.stds[key]
                    )
                else:
                    D_I_scaled[intervention][key] = self.D_I[intervention][key]

        if self.scale_data:
            self.graph.set_data_standardised_flag(
                standardised=True, means=self.means, stds=self.stds
            )
            self.D_O_scaled = D_O_scaled
            self.D_I_scaled = D_I_scaled
        else:
            self.D_O_scaled = self.D_O
            self.D_I_scaled = self.D_I

        # setting data up to use for CBO algorithm
        self.observational_samples = np.hstack(
            ([self.D_O_scaled[var] for var in self.variables])
        )

        self.interventional_samples = change_intervention_list_format(
            self.D_I_scaled, self.exploration_set, target=self.graph.target
        )

    def define_all_possible_graphs(self, error_tol=1e-5):
        self.graphs: Dict[Tuple, GraphStructure] = {}
        self.posterior: List[float] = []
        parents_to_remove = []
        for parents in self.prior_probabilities:
            if self.prior_probabilities[parents] < error_tol:
                continue

            graph: GraphStructure = deepcopy(self.graph)
            edges = [(parent, graph.target) for parent in parents]
            graph.mispecify_graph(edges)
            self.graphs[parents] = graph
            self.posterior.append(self.prior_probabilities[parents])

    def redefine_all_possible_graphs(self, error_tol=1e-4):
        self.posterior: List[float] = []
        parents_to_remove = []
        for parents in self.prior_probabilities:
            if self.prior_probabilities[parents] < error_tol:
                # remove from self.graphs[parents] from self.graphs
                parents_to_remove.append(parents)
                continue
            self.posterior.append(self.prior_probabilities[parents])

        for parents in parents_to_remove:
            if parents in self.graphs:
                del self.graphs[parents]

    def calculate_do_statistics(self):
        do_effects_functions: List[List[DoFunctions]] = []
        for i, graph_parents in enumerate(self.graphs):
            # this is the mean and variance for each graph for each element in the exploration set
            logging.info(f"----Computing do function for graph {i}------")
            graph = self.graphs[graph_parents]
            do_effects_functions.append(
                cbo_functions.update_all_do_functions(
                    graph, self.observational_samples, self.exploration_set
                )
            )
        self.do_effects_functions: List[List[DoFunctions]] = do_effects_functions

    def fit_samples_to_graphs(self):
        sem_emit_fncs: List[OrderedDict[str, GPRegression]] = []
        for key, graph in self.graphs.items():
            graph.fit_samples_to_graph(self.D_O_scaled, set_priors=False)
            sem_emit_fncs.append(graph.functions)
        return sem_emit_fncs

    def data_and_prior_setup(self):

        # normalize the datasets
        self.standardize_all_data()

        self.prior_probabilities = self.determine_initial_probabilities()
        if self.nonlinear:
            self.posterior_model: SCMModel = NonLinearSCMModel(
                self.prior_probabilities, self.graph
            )
        else:
            self.posterior_model: SCMModel = LinearSCMModel(
                self.prior_probabilities, self.graph
            )
        self.posterior_model.set_data(self.D_O_scaled)

        # update the posterior probabilities now with the interventional data
        for intervention in self.D_I_scaled:

            D_I_sample = self.D_I_scaled[intervention]
            num_samples = len(D_I_sample[self.graph.target])
            for n in range(num_samples):
                x_dict = {
                    obs_key: D_I_sample[obs_key][n]
                    for obs_key in D_I_sample
                    if obs_key != self.graph.target
                }
                y = D_I_sample[self.graph.target][n]
                self.posterior_model.update_all(x_dict, y)
                D_I = {key: np.array([D_I_sample[key][n]]) for key in D_I_sample}
                self.posterior_model.add_data(D_I)

        self.prior_probabilities = self.posterior_model.prior_probabilities.copy()

    def run_algorithm(self, T: int = 30):

        self.data_and_prior_setup()
        self.define_all_possible_graphs()
        self.fit_samples_to_graphs()

        # starting the setup for the CBO algorithm
        (
            data_x_list,
            data_y_list,
            best_intervention_value,
            current_global_min,
            best_variable,
        ) = cbo_functions.define_initial_data_CBO(
            self.interventional_samples,
            self.exploration_set,
            self.manipulative_variables,
            self.target,
        )

        input_space = [len(vars) for vars in self.exploration_set]
        current_best_x = {
            tuple(interventions): [] for interventions in self.exploration_set
        }
        current_best_y = {
            tuple(interventions): [] for interventions in self.exploration_set
        }
        parameter_spaces = [None] * len(self.exploration_set)
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

        # define the initial surrogate models
        self.calculate_do_statistics()
        self.model_list_overall = ceo_utils.update_posterior_model_aggregate_2(
            self.exploration_set,
            True,
            self.model_list_overall,
            data_x_list,
            data_y_list,
            self.causal_prior,
            best_variable,
            input_space,
            self.do_effects_functions,
            self.posterior,
        )

        for i in range(T):
            logging.info(f"----------------------ITERATION {i}----------------------")
            logging.info(f"Updated posterior distribution {self.posterior}")
            logging.info(f"The corresponding parents are {self.graphs.keys()}")

            # get the next sample
            y_acquisition_list, x_new_list = cbo_functions.get_new_x_y_list(
                self.exploration_set,
                self.graph,
                current_global_min,
                self.model_list_overall,
                cost_functions,
            )

            # find the optimal intervention, which maximises the acquisition function
            target_index = np.argmax(y_acquisition_list)
            var_to_intervene = tuple(self.exploration_set[target_index])
            intervention_set.append(var_to_intervene)

            # Setting the data after the new point was found
            intervention_values.append(tuple(x_new_list[target_index][0]))

            print("------------RUN INFO---------------")
            print(f"The acquisition values are {y_acquisition_list}")
            print(f"The selected intervention {x_new_list[target_index][0]}")
            if self.scale_data:
                x_new_list_intervention = np.array(
                    [
                        reverse_standardize(
                            x_new_list[target_index][0, j],
                            self.means[var],
                            self.stds[var],
                        )
                        for j, var in enumerate(var_to_intervene)
                    ]
                ).reshape(1, -1)
            else:
                x_new_list_intervention = np.array(
                    [
                        x_new_list[target_index][0, j]
                        for j, var in enumerate(var_to_intervene)
                    ]
                ).reshape(1, -1)
            print(f"Back to the original range {x_new_list_intervention}")
            y_new = target_classes[target_index].compute_target(x_new_list_intervention)
            print(f"The outcome is {y_new}")
            data_x_list[target_index] = np.vstack(
                (data_x_list[target_index], x_new_list[target_index])
            )

            data_y_list[target_index] = np.concatenate(
                (data_y_list[target_index], y_new.reshape(-1))
            )

            print(data_x_list)
            print(data_y_list)
            # set the new best variable
            best_variable = target_index

            ## Update the dict storing the current optimal solution

            intervention = {
                var: x_new_list_intervention[0][j]
                for j, var in enumerate(var_to_intervene)
            }
            sample = sample_model(
                static_sem=self.graph.SEM,
                interventions=intervention,
                sample_count=1,
                graph=self.graph,
            )

            sample[self.graph.target] = y_new
            print(f"The interventional sample {sample}")
            current_best_x[var_to_intervene].append(x_new_list_intervention[0])
            current_best_y[var_to_intervene].append(y_new[0][0])

            current_y.append(y_new[0][0])
            best_y = global_opt[i - 1] if global_opt else y_new[0][0]
            all_values = [
                value for values in current_best_y.values() for value in values
            ]
            min_y = np.min(all_values)
            global_opt.append(min_y if min_y < best_y else best_y)

            logging.info(
                f"Selected intervention {var_to_intervene} at {x_new_list[target_index]} with y = {y_new}"
            )
            logging.info(f"Current global optimum {global_opt[i]}")

            total_cost = 0
            for j, val in enumerate(self.exploration_set[target_index]):
                total_cost += cost_functions[val](x_new_list[target_index][0, j])
            current_cost.append(current_cost[i] + total_cost)

            # now update the probabilities
            if self.scale_data:
                for key in sample:
                    if key != self.graph.target:
                        sample[key] = standardize(
                            sample[key], self.means[key], self.stds[key]
                        )
            x_dict = {
                key: val[0][0]
                for key, val in sample.items()
                if key != self.graph.target
            }
            y = sample[self.graph.target][0][0]
            self.posterior_model.update_all(x_dict, y)
            print(f"The standardized sample {sample}")
            D_I = {key: val.reshape(-1) for key, val in sample.items()}
            self.posterior_model.add_data(D_I)
            self.prior_probabilities = self.posterior_model.prior_probabilities

            self.redefine_all_possible_graphs()
            non_null_parents = list(self.graphs.keys())
            self.posterior_model.update_probabilities(non_null_parents)
            self.calculate_do_statistics()
            self.posterior = [
                self.prior_probabilities[parents] for parents in self.graphs
            ]
            self.model_list_overall = ceo_utils.update_posterior_model_aggregate_2(
                self.exploration_set,
                False,
                self.model_list_overall,
                data_x_list,
                data_y_list,
                self.causal_prior,
                best_variable,
                input_space,
                self.do_effects_functions,
                self.posterior,
            )
            current_global_min = global_opt[i]

        return (
            global_opt,
            current_y,
            current_cost,
            intervention_set,
            intervention_values,
            average_uncertainty,
        )
