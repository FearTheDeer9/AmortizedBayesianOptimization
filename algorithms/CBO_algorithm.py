import logging
from typing import List

import numpy as np

import utils.utils_functions as utils_functions
from utils.graph_utils.graph_functions import graph_setup
from utils.sem_sampling import sample_model
from utils.utils_classes import TargetClass

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Set the format of log messages
    datefmt="%m/%d/%Y %I:%M:%S %p",  # Set the date format
)


def CBO(
    graph_type: str,
    causal_prior: bool = True,
    cost_num: int = 1,
    task: str = "min",
    T: int = 50,
):
    """
    This is the CBO
    """
    assert graph_type in ["Toy", "Synthetic"]
    # defining the initial variables
    (
        graph,
        exploration_set,
        manipulative_variables,
        target,
        samples,
        observational_samples,
        interventional_samples,
    ) = graph_setup(graph_type=graph_type)

    graph.refit_models(samples)
    num_interventions = len(exploration_set)

    # setting up the data for the rest of the algorithm
    data_x_list, data_y_list, best_variable, current_global_min, best_variable = (
        utils_functions.define_initial_data_CBO(
            interventional_samples,
            num_interventions,
            exploration_set,
            5,
            manipulative_variables,
            target,
        )
    )

    # parameter in the algorithm
    # current_global_min = np.min(samples[target])
    input_space = [len(vars) for vars in exploration_set]
    objective = np.inf if task == "min" else -np.inf
    current_best_x = {tuple(interventions): [] for interventions in exploration_set}
    current_best_y = {tuple(interventions): [] for interventions in exploration_set}
    parameter_spaces = [None] * len(exploration_set)
    target_classes: List[TargetClass] = [None] * len(exploration_set)
    model_list = [None] * len(exploration_set)
    trial_observed = [True] * T

    for i in range(len(exploration_set)):
        parameter_spaces[i] = graph.get_parameter_space(exploration_set[i])
        target_classes[i] = TargetClass(graph.SEM(), exploration_set[i])

    # defining some variables necessary for the algorithm
    alpha_coverage, hull_obs, coverage_total = utils_functions.compute_coverage(
        observational_samples, manipulative_variables, graph.get_interventional_range()
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
    cost_functions = graph.get_cost_structure(cost_num)

    for i in range(T):
        coverage_obs = utils_functions.update_hull(
            observational_samples, manipulative_variables
        )
        rescale = observational_samples.shape[0] / max_N
        epsilon_coverage = (coverage_obs / coverage_total) / rescale
        u = np.random.uniform()

        # ensure one observation and one intervention (at least)
        u = 0 if i == 0 else u
        u = 1 if i == 1 else u

        if u < epsilon_coverage:
            observed += 1
            logging.info(
                f"------ Iteration {i}: Observed {observed}, where epsilon = {epsilon_coverage} ------"
            )
            trial_observed[i] = True
            # 1. Observe new observations (xt, ct, yt)
            observed_sample = sample_model(graph.define_SEM(), sample_count=1)
            # 2. Augment D_O
            observational_samples = np.vstack(
                (
                    observational_samples,
                    [observed_sample[var][0, 0] for var in graph.get_variables()],
                )
            )

            # 3. Update the prior of the causal GP
            # update the interventional expectation and the interventional variance
            do_function_list = utils_functions.update_all_do_functions(
                graph, observational_samples, exploration_set
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
            model_list = utils_functions.update_posterior_model(
                exploration_set,
                trial_observed[i - 1],
                model_list,
                data_x_list,
                data_y_list,
                causal_prior,
                best_variable,
                input_space,
                do_function_list,
            )

            # get the new optimal value based on all the elements in the exploration set
            y_acquisition_list, x_new_list = utils_functions.get_new_x_y_list(
                exploration_set, graph, current_global_min, model_list, cost_functions
            )

            # find the optimal intervention, which maximises the acquisition function
            target_index = np.argmax(y_acquisition_list)
            var_to_intervene = tuple(exploration_set[target_index])
            y_new = target_classes[target_index].compute_target(
                x_new_list[target_index]
            )

            data_x_list[target_index] = np.vstack(
                (data_x_list[target_index], x_new_list[target_index])
            )
            data_y_list[target_index] = np.vstack((data_y_list[target_index], y_new))
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
            for j, val in enumerate(exploration_set[target_index]):
                total_cost += cost_functions[val](x_new_list[target_index][0, j])

            current_cost.append(current_cost[i] + total_cost)

            logging.info(
                f"Selected intervention {var_to_intervene} at {x_new_list[target_index]} with y = {y_new}"
            )
            logging.info(f"Current global optimum {global_opt[i+1]}")


if __name__ == "__main__":
    CBO_algorithm("Toy")
