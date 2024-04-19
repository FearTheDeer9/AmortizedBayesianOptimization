import logging

import matplotlib.pyplot as plt
import numpy as np
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper

import utils.utils_functions as utils_functions
from utils.graph_utils.graph_functions import graph_setup
from utils.graph_utils.toy_graph import ToyGraph
from utils.sem_sampling import sample_model
from utils.utils_classes import DoFunctions, TargetClass

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Set the format of log messages
    datefmt="%m/%d/%Y %I:%M:%S %p",  # Set the date format
)


def BO(
    graph_type: str,
    causal_prior: bool = True,
    cost_num: int = 1,
    task: str = "min",
    T: int = 50,
):
    # Set some of the variables
    assert graph_type in ["Toy", "Synthetic"]
    # defining the initial variables
    (
        graph,
        _,
        manipulative_variables,
        target,
        samples,
        observational_samples,
        interventional_samples,
    ) = graph_setup(graph_type=graph_type)

    graph.refit_models(samples)
    T = 35
    causal_prior = True
    input_space = 2
    X = np.hstack([samples[var] for var in manipulative_variables])
    Y = samples[target]

    # trying to define the target function for the interventions
    interventions = manipulative_variables.copy()
    model = graph.SEM()

    # for the BayesOpt algorithm
    best_x = np.zeros(shape=(T + 1, input_space))
    best_y = np.zeros(shape=T + 1)
    current_best = np.argmin(Y)  # can maybe change this so that it can be max as well
    best_y[0] = Y[current_best]
    best_x[0, :] = X[current_best, :]

    do_effects = DoFunctions(graph.get_all_do(), samples, interventions)
    space = graph.get_parameter_space(interventions)

    target_class = TargetClass(model, interventions)

    emukit_model = utils_functions.set_up_GP(
        causal_prior, input_space, do_effects, X, Y
    )

    cummulative_cost = 0
    costs_functions = graph.get_cost_structure(3)  # this can be 1, 2, 3 or 4
    for i in range(T):
        logging.info(f"-------- Iteration {i} --------")
        emukit_model.optimize()
        acquisition = ExpectedImprovement(emukit_model)
        optimzer = GradientAcquisitionOptimizer(space)
        x_new, _ = optimzer.optimize(acquisition)
        y_new = target_class.compute_target(x_new)
        logging.debug(
            f"The optimal point found in the optimizer is {y_new} for {x_new}"
        )
        logging.debug(
            f"The corresponding target is {target_class.compute_target(best_x[i].reshape(1, 2))}"
        )
        logging.debug(f"The global optimum was {best_y[i]} for {best_x[i]}")

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


if __name__ == "__main__":
    BO("Toy")
