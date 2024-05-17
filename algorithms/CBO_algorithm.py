import logging
import pickle
from copy import deepcopy
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from emukit.model_wrappers import GPyModelWrapper

import utils.cbo_functions as cbo_functions
from algorithms.BASE_algorithm import BASE
from config import SHOW_GRAPHICS
from graphs.graph import GraphStructure
from graphs.graph_functions import graph_setup
from utils.cbo_classes import TargetClass
from utils.sem_sampling import (
    change_intervention_list_format,
    draw_interventional_samples_sem,
    sample_model,
)

logging.basicConfig(
    level=logging.DEBUG,  # Set the loggingand level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Set the format of log messages
    datefmt="%m/%d/%Y %I:%M:%S %p",  # Set the date format
    filename="logfile.log",  # Specify the file to write the logs to
    filemode="w",  # Set the file mode to 'a' to append to the file (use 'w' to overwrite each time)
)


class CBO(BASE):

    def __init__(
        self,
        graph_type: str = "Toy",
        graph: GraphStructure = None,
        causal_prior: bool = True,
        cost_num: int = 1,
        task: str = "min",
        noiseless: bool = True,
    ):
        self._graph_type = graph_type
        self.noiseless = noiseless
        if graph is not None:
            self.graph = graph
        else:
            assert graph_type in ["Toy", "Synthetic", "Graph6", "Graph5", "Graph4"]
            # defining the initial variables
            self.graph = self.chosen_structure()

        self.exploration_set = self.graph.get_exploration_set()
        self.manipulative_variables = self.graph.get_sets()[2]
        self.target = self.graph.target
        self.es_to_n_mapping = {
            tuple(es): i for i, es in enumerate(self.exploration_set)
        }
        self.variables = self.graph.variables
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
        # just fitting the observational data to the graph
        self.graph.fit_samples_to_graph(self.D_O)
        self.D_I = D_I
        self.interventional_samples = change_intervention_list_format(
            self.D_I, self.exploration_set
        )

        self.do_function_list = cbo_functions.update_all_do_functions(
            self.graph, self.observational_samples, self.exploration_set
        )

    def do_function_graph(
        self,
        es: Tuple,
        size: int = 200,
        edge_num: int = 0,
        save_path: str = None,
        **kwargs,
    ):
        # setting up the plotting stuff
        true_vals = np.zeros(shape=size)
        predictions = np.zeros(shape=size)
        var = np.zeros(shape=size)
        es_num = self.es_to_n_mapping[es]
        interventions = {}

        # getting the number of entries in the exploration set
        intervention_domain = self.graph.get_interventional_range()
        min_intervention, max_intervention = intervention_domain[es[0]]
        intervention_vals = np.linspace(
            start=min_intervention, stop=max_intervention, num=size
        )

        for i in range(1, len(es)):
            min_i, max_i = intervention_domain[es[i]]
            interventions[es[i]] = (min_i + max_i) / 2

        for i, intervention_val in enumerate(intervention_vals):
            interventions[es[0]] = intervention_val
            true_vals[i] = np.mean(
                sample_model(
                    self.graph.SEM,
                    interventions=interventions,
                    sample_count=500,
                    graph=self.graph,
                    noiseless=True,
                )["Y"]
            )

            value = np.array([interventions[var] for var in es]).reshape(1, -1)
            predictions[i] = self.do_function_list[es_num].mean_function_do(value)
            var[i] = self.do_function_list[es_num].var_function_do(value)

        # Apply custom plot styles from kwargs
        plt.plot(
            intervention_vals,
            true_vals,
            label="True",
            **kwargs.get("true_vals_style", {}),
        )
        plt.plot(
            intervention_vals,
            predictions,
            label=f"Do {es[0] if len(es) == 1 else es}",
            **kwargs.get("predictions_style", {}),
        )
        plt.fill_between(
            intervention_vals,
            [p - 1.0 * np.sqrt(e) for p, e in zip(predictions, var)],
            [p + 1.0 * np.sqrt(e) for p, e in zip(predictions, var)],
            color=kwargs.get("fill_color", "gray"),
            alpha=kwargs.get("fill_alpha", 0.5),
        )
        plt.legend()
        plt.xlabel(kwargs.get("xlabel", "Intervention Value"))
        plt.ylabel(kwargs.get("ylabel", "Y"))
        plt.title(kwargs.get("title", "Do-Function Graph"))

        # Save the figure if save_path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        plt.show()

    def run_algorithm(self, T: int = 10, file: str = None):
        self.graph.fit_samples_to_graph(self.D_O)

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
        input_space = [len(vars) for vars in self.exploration_set]
        objective = (
            np.min(self.D_O[self.target])
            if self.task == "min"
            else np.max(self.D_O[self.target])
        )
        current_best_x = {
            tuple(interventions): [] for interventions in self.exploration_set
        }
        current_best_y = {
            tuple(interventions): [] for interventions in self.exploration_set
        }
        parameter_spaces = [None] * len(self.exploration_set)
        target_classes: List[TargetClass] = [None] * len(self.exploration_set)
        model_list = [[None] * len(self.exploration_set)]
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
                noiseless=self.noiseless,
            )

        # defining some variables necessary for the algorithm
        # alpha_coverage, hull_obs, coverage_total = cbo_functions.compute_coverage(
        #     self.D_O,
        #     self.manipulative_variables,
        #     self.graph.get_interventional_range(),
        # )

        current_global_min = np.mean(self.D_O[self.target])
        # some counters for the algorithm
        observed = 0
        intervened = 0

        # STARTING THE ALGORITHM
        current_cost: List[int] = []
        global_opt: List[float] = []
        current_y: List[float] = []
        intervention_set: List[Tuple[str]] = []
        intervention_values: List[Tuple[float]] = []
        global_opt.append(current_global_min)
        current_cost.append(0.0)
        cost_functions = self.graph.get_cost_structure(self.cost_num)

        for i in range(T):
            # coverage_obs = cbo_functions.update_hull(
            #     self.observational_samples, self.manipulative_variables
            # )
            # rescale = self.observational_samples.shape[0] / max_N
            # epsilon_coverage = (coverage_obs / coverage_total) / rescale
            # u = np.random.uniform()

            # ensure one observation and one intervention (at least)
            u = 0 if i == 0 else u
            u = 1 if i == 1 else u
            # observe = u < epsilon_coverage

            # this is changed to make it more comparable to the CEO method
            if i == 0:
                observed += 1
                logging.info(
                    f"------ Iteration {i}: Observed {observed}, where epsilon =  ------"
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
                self.do_function_list = cbo_functions.update_all_do_functions(
                    self.graph, self.observational_samples, self.exploration_set
                )

                # update the optimal values, if observed, it is the same as the previous round
                if global_opt:
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
                    f"------ Iteration {i}: Intervened {intervened} where epsilon =  ------"
                )
                trial_observed[i] = False

                # updating the model based on the previous trial
                model_list.append(
                    cbo_functions.update_posterior_model(
                        self.exploration_set,
                        trial_observed[i - 1],
                        model_list[-1],
                        data_x_list,
                        data_y_list,
                        self.causal_prior,
                        best_variable,
                        input_space,
                        self.do_function_list,
                    )
                )

                if SHOW_GRAPHICS:
                    for es in self.exploration_set:
                        fig, ax = self.plot_model_list(model_list[-1], es)
                        for i, intervention in enumerate(intervention_set):
                            if intervention == es:
                                ax.scatter(
                                    intervention_values[i],
                                    current_y[i],
                                    marker="x",
                                    color="black",
                                    s=100,
                                    label="Intervention Points",
                                )
                        if file:
                            filename = f"{file}_{es[0]}_iter_{i+1}"
                            plt.savefig(filename, bbox_inches="tight")
                        else:
                            plt.show()

                # get the new optimal value based on all the elements in the exploration set
                y_acquisition_list, x_new_list = cbo_functions.get_new_x_y_list(
                    self.exploration_set,
                    self.graph,
                    current_global_min,
                    model_list[-1],
                    cost_functions,
                )

                # find the optimal intervention, which maximises the acquisition function
                target_index = np.argmax(y_acquisition_list)
                var_to_intervene = tuple(self.exploration_set[target_index])
                intervention_set.append(var_to_intervene)

                # Setting the data after the new point was found
                intervention_values.append(tuple(x_new_list[target_index][0]))
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

                current_y.append(y_new[0][0])
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

        return (
            global_opt,
            current_y,
            current_cost,
            intervention_set,
            intervention_values,
        )
