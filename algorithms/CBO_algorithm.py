import logging
from typing import Dict, List

import numpy as np

import utils.cbo_functions as cbo_functions
from algorithms.BASE_algorithm import BASE
from graphs.graph import GraphStructure
from graphs.graph_functions import graph_setup
from utils.cbo_classes import TargetClass
from utils.sem_sampling import (
    change_intervention_list_format,
    draw_interventional_samples_sem,
    sample_model,
)

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Set the format of log messages
    datefmt="%m/%d/%Y %I:%M:%S %p",  # Set the date format
)


class CBO(BASE):

    def __init__(
        self,
        graph_type: str = "Toy",
        graph: GraphStructure = None,
        observational_samples: Dict = None,
        interventional_samples: Dict = None,
        causal_prior: bool = True,
        cost_num: int = 1,
        task: str = "min",
    ):
        self._graph_type = graph_type
        if graph is not None:
            assert observational_samples is not None
            self.graph = graph
            self.observational_samples = observational_samples
            self.interventional_samples = interventional_samples

        else:
            assert graph_type in ["Toy", "Synthetic", "Graph6", "Graph5", "Graph4"]
            # defining the initial variables
            (
                self.graph,
                self.exploration_set,
                self.manipulative_variables,
                self.target,
                self.samples,
                self.observational_samples,
                self.interventional_samples,
            ) = graph_setup(graph_type=graph_type)

        self.es_to_n_mapping = {
            tuple(es): i for i, es in enumerate(self.exploration_set)
        }
        self.causal_prior = causal_prior
        self.cost_num = cost_num
        self.task = task

    def set_values(self, D_O: Dict, D_I: Dict, exploration_set: List[List[str]]):
        logging.info("Using predefined values for the optimization algorithm")
        self.exploration_set = exploration_set
        # create mappings for the exploration set
        self.es_to_n_mapping = {
            tuple(es): i for i, es in enumerate(self.exploration_set)
        }
        self.D_O = D_O
        self.observational_samples = np.hstack(
            ([self.D_O[var] for var in self.variables])
        )
        self.D_I = D_I
        self.interventional_samples = change_intervention_list_format(
            self.D_I, self.exploration_set
        )

        self.arm_distribution = np.array(
            [1 / len(self.exploration_set)] * len(self.exploration_set)
        )

    def run_algorithm(self, T: int = 10):
        self.graph.refit_models(self.samples)
        num_interventions = len(self.exploration_set)

        # setting up the data for the rest of the algorithm
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

        # parameter in the algorithm
        # current_global_min = np.min(samples[target])
        input_space = [len(vars) for vars in self.exploration_set]
        objective = np.inf if self.task == "min" else -np.inf
        current_best_x = {
            tuple(interventions): [] for interventions in self.exploration_set
        }
        current_best_y = {
            tuple(interventions): [] for interventions in self.exploration_set
        }
        parameter_spaces = [None] * len(self.exploration_set)
        target_classes: List[TargetClass] = [None] * len(self.exploration_set)
        model_list = [None] * len(self.exploration_set)
        trial_observed = [True] * T

        for i in range(len(self.exploration_set)):
            parameter_spaces[i] = self.graph.get_parameter_space(
                self.exploration_set[i]
            )
            target_classes[i] = TargetClass(
                sem_model=self.graph.SEM,
                interventions=self.exploration_set[i],
                variables=self.graph.variables,
                graph=self.graph,
            )

        # defining some variables necessary for the algorithm
        alpha_coverage, hull_obs, coverage_total = cbo_functions.compute_coverage(
            self.observational_samples,
            self.manipulative_variables,
            self.graph.get_interventional_range(),
        )
        max_N = 200

        # some counters for the algorithm
        observed = 0
        intervened = 0

        # STARTING THE ALGORITHM
        current_cost = []
        global_opt = []
        global_opt.append(objective)
        current_cost.append(0.0)
        cost_functions = self.graph.get_cost_structure(self.cost_num)

        for i in range(T):
            coverage_obs = cbo_functions.update_hull(
                self.observational_samples, self.manipulative_variables
            )
            rescale = self.observational_samples.shape[0] / max_N
            epsilon_coverage = (coverage_obs / coverage_total) / rescale
            u = np.random.uniform()

            # ensure one observation and one intervention (at least)
            u = 0 if i == 0 else u
            u = 1 if i == 1 else u
            observe = u < epsilon_coverage

            if i == 0:
                observed += 1
                logging.info(
                    f"------ Iteration {i}: Observed {observed}, where epsilon = {epsilon_coverage} ------"
                )
                trial_observed[i] = True
                # 1. Observe new observations (xt, ct, yt)
                observed_sample = sample_model(self.graph.SEM, sample_count=1)
                # 2. Augment D_O
                self.observational_samples = np.vstack(
                    (
                        self.observational_samples,
                        [observed_sample[var][0, 0] for var in self.graph.variables],
                    )
                )

                # 3. Update the prior of the causal GP
                # update the interventional expectation and the interventional variance
                do_function_list = cbo_functions.update_all_do_functions(
                    self.graph, self.observational_samples, self.exploration_set
                )

                # update the optimal values, if observed, it is the same as the previous round
                global_opt.append(global_opt[i])
                current_cost.append(current_cost[i])
            else:
                # intervene
                # 1. compute the expected improvement for each element in the exploration set
                # 2. obtain the optimal interventional set-value pair
                # 3. intervene on the system
                # 4. Update the posterior of the causal GP
                intervened += 1
                logging.info(
                    f"------ Iteration {i}: Intervened {intervened} where epsilon = {epsilon_coverage}  ------"
                )
                trial_observed[i] = False

                # updating the model based on the previous trial
                model_list = cbo_functions.update_posterior_model(
                    self.exploration_set,
                    trial_observed[i - 1],
                    model_list,
                    data_x_list,
                    data_y_list,
                    self.causal_prior,
                    best_variable,
                    input_space,
                    do_function_list,
                )
                for es in self.exploration_set:
                    self.plot_model_list(model_list, es)

                # get the new optimal value based on all the elements in the exploration set
                y_acquisition_list, x_new_list = cbo_functions.get_new_x_y_list(
                    self.exploration_set,
                    self.graph,
                    current_global_min,
                    model_list,
                    cost_functions,
                )

                # find the optimal intervention, which maximises the acquisition function
                target_index = np.argmax(y_acquisition_list)
                var_to_intervene = tuple(self.exploration_set[target_index])
                y_new = target_classes[target_index].compute_target(
                    x_new_list[target_index]
                )

                data_x_list[target_index] = np.vstack(
                    (data_x_list[target_index], x_new_list[target_index])
                )

                data_y_list[target_index] = np.concatenate(
                    (data_y_list[target_index], y_new.reshape(-1))
                )
                # set the new best variable
                best_variable = target_index

                ## Update the dict storing the current optimal solution

                current_best_x[var_to_intervene].append(x_new_list[target_index][0][0])
                current_best_y[var_to_intervene].append(y_new[0][0])
                # maybe need to update the model -> i don't think so as this is done at the start of each intervention loop

                best_y = global_opt[i]
                all_values = [
                    value for values in current_best_y.values() for value in values
                ]
                min_y = np.min(all_values)
                global_opt.append(min_y if min_y < best_y else best_y)

                # compute the cost of the intervention
                # get the of the current intervention
                total_cost = 0
                for j, val in enumerate(self.exploration_set[target_index]):
                    total_cost += cost_functions[val](x_new_list[target_index][0, j])

                current_cost.append(current_cost[i] + total_cost)

                logging.info(
                    f"Selected intervention {var_to_intervene} at {x_new_list[target_index]} with y = {y_new}"
                )
                logging.info(f"Current global optimum {global_opt[i+1]}")
