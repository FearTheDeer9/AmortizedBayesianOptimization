import logging
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper

import utils.cbo_functions as cbo_functions
from graphs.graph import GraphStructure
from graphs.graph_functions import graph_setup
from graphs.toy_graph import ToyGraph
from utils.cbo_classes import DoFunctions, TargetClass
from utils.sem_sampling import sample_model

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Set the format of log messages
    datefmt="%m/%d/%Y %I:%M:%S %p",  # Set the date format
)


class BO:

    def __init__(
        self,
        graph_type: str = "Toy",
        graph: GraphStructure = None,
        samples: Dict = None,
        causal_prior: bool = True,
        cost_num: int = 1,
    ):

        if graph is not None:
            self.graph = graph
            self.observational_samples = samples
        else:
            (
                self.graph,
                _,
                manipulative_variables,
                target,
                self.samples,
                _,
                _,
            ) = graph_setup(graph_type=graph_type)

        # this is to change the edges so that the method works for BO
        self.graph.break_dependency_structure()
        print(self.graph.edges)
        self.causal_prior = causal_prior
        self.cost_num = cost_num
        self.target = self.graph.target
        _, _, self.manipulative_variables = self.graph.get_sets()

    def run_algorithm(self, T: int = 30):
        self.graph.refit_models(self.samples)
        X = np.hstack([self.samples[var] for var in self.manipulative_variables])
        Y = self.samples[self.target]

        # trying to define the target function for the interventions
        interventions = self.manipulative_variables.copy()
        model = self.graph.SEM

        # for the BayesOpt algorithm
        input_space = len(interventions)
        best_x = np.zeros(shape=(T + 1, input_space))
        best_y = np.zeros(shape=T + 1)
        current_best = np.argmin(
            Y
        )  # can maybe change this so that it can be max as well
        best_y[0] = Y[current_best]
        best_x[0, :] = X[current_best, :]

        do_effects = DoFunctions(self.graph.get_all_do(), self.samples, interventions)
        space = self.graph.get_parameter_space(interventions)

        target_class = TargetClass(model, interventions)

        emukit_model = cbo_functions.set_up_GP(
            self.causal_prior, input_space, do_effects, X, Y
        )

        cummulative_cost = 0
        costs_functions = self.graph.get_cost_structure(
            self.cost_num
        )  # this can be 1, 2, 3 or 4
        for i in range(T):
            logging.info(f"-------- Iteration {i} --------")
            emukit_model.optimize()
            acquisition = ExpectedImprovement(emukit_model)
            optimzer = GradientAcquisitionOptimizer(space)
            x_new, _ = optimzer.optimize(acquisition)
            y_new = target_class.compute_target(x_new)
            logging.info(
                f"The optimal point found in the optimizer is {y_new} for {x_new}"
            )
            logging.info(
                f"The corresponding target is {target_class.compute_target(best_x[i].reshape(1, 2))}"
            )
            logging.info(f"The global optimum was {best_y[i]} for {best_x[i]}")

            # adding the new data point
            X = np.vstack([X, x_new])
            Y = np.vstack([Y, y_new])
            emukit_model.set_data(X, Y)

            # get the cost for these values
            total_cost = 0
            for j, val in enumerate(interventions):
                total_cost += costs_functions[val](x_new[0, j])

            cummulative_cost += total_cost

            # get the optimum
            results_X, results_Y = emukit_model.X, emukit_model.Y
            current_best = np.argmin(results_Y)
            best_y[i + 1] = results_Y[current_best]
            best_x[i + 1, :] = results_X[current_best, :]

            logging.info(
                f"Total cost - {total_cost}: Cummulative cost - {cummulative_cost}: Best Y - {best_y[i + 1], current_best}"
            )
