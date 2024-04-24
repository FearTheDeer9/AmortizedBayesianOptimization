import logging
from copy import deepcopy
from typing import Callable, Dict, List, Tuple

import numpy as np

import utils.utils_functions as utils_functions
from utils.graph_utils.graph import GraphStructure
from utils.graph_utils.graph_functions import create_grid_interventions, graph_setup
from utils.sem_sampling import sample_model
from utils.utils_classes import TargetClass


def get_monte_carlo_expectation(intervention_samples):
    assert isinstance(intervention_samples, dict)
    new = {k: None for k in intervention_samples.keys()}
    for es in new.keys():
        new[es] = intervention_samples[es].mean(axis=0)

    # Returns the expected value of the intervention via MC sampling
    return new


def optimal_sequence_of_interventions(
    exploration_sets: list,
    interventional_grids: dict,
    graph: GraphStructure,
    random_state: int = 42,
    task: str = "min",
) -> Tuple:

    # setting up the noise model
    model_variables = graph.variables
    static_noise_model = {k: 0 for k in model_variables}

    target_variable = graph.target
    assert target_variable is not None
    assert target_variable in model_variables

    best_s_sequence = []
    best_s_values = []
    best_objective_values = []

    optimal_interventions = {setx: [] for setx in exploration_sets}

    y_stars = deepcopy(optimal_interventions)
    all_CE = []
    blank_intervention_blanket = {node: [] for node in graph.nodes}

    CE = {es: [] for es in exploration_sets}

    for s in exploration_sets:

        # Reset blanket so as to not carry over levels from previous exploration set
        intervention_blanket = {}
        for level in interventional_grids[s]:

            # Univariate intervention
            if len(s) == 1:
                intervention_blanket[s[0]] = float(level[0])

            # Multivariate intervention
            else:
                for var, val in zip(s, level):
                    intervention_blanket[var] = val

            intervention_samples = sample_model(
                SEM,
                interventions=intervention_blanket,
                sample_count=10,
                epsilon=static_noise_model,
            )

            out = get_monte_carlo_expectation(intervention_samples)
            CE[s].append((out[target_variable][0]))

    local_target_values = []
    for s in exploration_sets:
        if task == "min":
            idx = np.array(CE[s]).argmin()
        else:
            idx = np.array(CE[s]).argmax()
        local_target_values.append((s, idx, CE[s][idx]))
        y_stars[s] = CE[s][idx]
        optimal_interventions[s] = interventional_grids[s][idx]

    # Find best intervention at time t
    best_s, best_idx, best_objective_value = min(
        local_target_values, key=lambda t: t[2]
    )
    best_s_value = interventional_grids[best_s][best_idx]

    best_s_sequence.append(best_s)
    best_s_values.append(best_s_value)
    best_objective_values.append(best_objective_value)
    all_CE.append(CE)

    return (
        best_s_values,
        best_s_sequence,
        best_objective_values,
        y_stars,
        optimal_interventions,
        all_CE,
    )


(
    graph,
    exploration_set,
    manipulative_variables,
    target,
    samples,
    observational_samples,
    interventional_samples,
) = graph_setup("Toy")

SEM = graph.SEM
true_edges = graph.edges
interventional_range = graph.get_interventional_range()
intervention_grid = create_grid_interventions(
    interventional_range, get_list_format=True
)


# Original grid
# grid = [{"X": 3.5}, {"Z": 2}, {"X": 3.2, "Z": 2}]

# Initialize the new grid format
# new_grid = {}

# Loop through each dictionary in the original grid
# for entry in intervention_grid:
#     keys = tuple(
#         sorted(entry.keys())
#     )  # Sort keys to ensure consistent order for tuples
#     values = tuple(
#         entry[key] for key in keys
#     )  # Get the corresponding values in tuple form

#     if keys in new_grid:
#         new_grid[keys].append(values)
#     else:
#         new_grid[keys] = [values]

print(optimal_sequence_of_interventions(exploration_set, intervention_grid, graph, 42))
