import logging
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper

import utils.cbo_functions as cbo_functions
from algorithms.BASE_algorithm import BASE
from config import SHOW_GRAPHICS
from graphs.graph import GraphStructure
from graphs.graph_functions import graph_setup
from graphs.toy_graph import ToyGraph
from utils.cbo_classes import DoFunctions, TargetClass
from utils.sem_sampling import sample_model

logging.basicConfig(
    level=logging.DEBUG,  # Set the loggingand level
    # Set the format of log messages
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",  # Set the date format
    filename="logfile.log",  # Specify the file to write the logs to
    # Set the file mode to 'a' to append to the file (use 'w' to overwrite each time)
    filemode="w",
)


class BO(BASE):

    def __init__(
        self,
        graph_type: str = "Toy",
        graph: GraphStructure = None,
        causal_prior: bool = False,
        n_obs: int = 100,
        cost_num: int = 1,
        noiseless: int = True,
    ):

        self._graph_type = graph_type
        self.noiseless = noiseless
        if graph is not None:
            self.graph = graph
        else:
            self.graph = self.chosen_structure()

        self.manipulative_variables = self.graph.get_sets()[2]
        self.n_obs = n_obs
        self.variables = self.graph.variables
        self.var_mapping = {var: i for i, var in enumerate(self.variables)}

        self.D_O: Dict[str, np.ndarray] = sample_model(
            self.graph.SEM, sample_count=self.n_obs, graph=graph
        )

        self.observational_samples = np.hstack(
            ([self.D_O[var] for var in self.graph.variables])
        )
        # this is to change the edges so that the method works for BO
        self.graph.break_dependency_structure()
        self.causal_prior = causal_prior
        self.cost_num = cost_num
        self.target = self.graph.target
        self.es_to_n_mapping = {tuple(self.manipulative_variables): 0}

    def set_values(self, D_O: Dict):
        logging.info("Using predefined values for the optimization algorithm")
        self.D_O = D_O
        self.observational_samples = np.hstack(
            ([self.D_O[var] for var in self.variables])
        )

    def run_algorithm(self, T: int = 30):
        manipulative_index = [
            self.var_mapping[var]
            for var in self.variables
            if var in self.manipulative_variables
        ]
        self.exploration_set = [tuple(self.manipulative_variables)]
        self.graph.refit_models(self.D_O)
        X = np.hstack([self.D_O[var]
                      for var in self.variables if var != self.target])
        X = X[:, manipulative_index]
        Y = self.D_O[self.target]

        # for the BayesOpt algorithm
        input_space = len(self.manipulative_variables)
        best_x = np.zeros(shape=(T + 1, input_space))

        # setting up the variables
        best_y = []
        current_y = []
        current_cost = np.zeros(shape=T)
        average_uncertainty = []
        # current_best = np.argmin(Y)
        best_y.append(np.mean(self.D_O[self.target]))
        # best_x[0, :] = X[current_best, :]

        do_effects = DoFunctions(
            self.graph.get_all_do(), self.D_O, self.manipulative_variables
        )
        space = self.graph.get_parameter_space(self.manipulative_variables)

        target_class = TargetClass(
            sem_model=self.graph.SEM,
            interventions=self.manipulative_variables,
            variables=self.graph.variables,
            graph=self.graph,
            noiseless=self.noiseless,
        )

        emukit_model: GPyModelWrapper = cbo_functions.set_up_GP(
            self.causal_prior,
            input_space,
            do_effects.mean_function_do,
            do_effects.var_function_do,
            X,
            Y,
        )

        cummulative_cost = 0
        # this can be 1, 2, 3 or 4
        costs_functions = self.graph.get_cost_structure(self.cost_num)
        for i in range(T):
            logging.info(f"-------- Iteration {i} --------")
            emukit_model.optimize()
            model_list = [emukit_model]
            self.model_list_overall = model_list
            # total_uncertainty = self.quantify_total_uncertainty()
            # average_uncertainty.append(total_uncertainty["average"])

            if SHOW_GRAPHICS:
                self.plot_model_list(model_list, tuple(
                    self.manipulative_variables))

            acquisition = ExpectedImprovement(emukit_model)
            optimzer = GradientAcquisitionOptimizer(space)
            x_new, _ = optimzer.optimize(acquisition)
            y_new = target_class.compute_target(x_new)
            logging.info(f"The global optimum was {best_y[i]}")
            logging.info(
                f"The optimal point found in the optimizer is {y_new} for {x_new}"
            )

            # adding the new data point
            X = np.vstack([X, x_new])
            Y = np.vstack([Y, y_new])
            emukit_model.set_data(X, Y)

            # get the cost for these values
            total_cost = 0
            for j, val in enumerate(self.manipulative_variables):
                total_cost += costs_functions[val](x_new[0, j])

            cummulative_cost += total_cost
            current_cost[i] = cummulative_cost

            # get the optimum
            current_y.append(y_new[0][0])
            current_best = np.argmin(current_y)
            best_y.append(current_y[current_best])

            logging.info(
                f"Total cost - {total_cost}: Cummulative cost - {cummulative_cost}: Best Y - {best_y[i + 1]}"
            )

        return best_y, current_y, current_cost, average_uncertainty
