import logging
from copy import deepcopy
from typing import Callable, Dict, List, OrderedDict, Tuple

import numpy as np
from GPy.models.gp_regression import GPRegression
from scipy.special import logsumexp

import utils.utils_functions as utils_functions
from utils.graph_utils.graph import GraphStructure
from utils.graph_utils.graph_functions import create_grid_interventions, graph_setup
from utils.graph_utils.toy_graph import ToyGraph
from utils.sem_sampling import draw_interventional_samples_sem, sample_model
from utils.utils_classes import TargetClass


def update_posterior_interventional(
    graphs: List[GraphStructure],
    posterior: List[float],
    intervened_var: Tuple,
    all_emission_fncs: List[OrderedDict[str, Callable]],
    interventional_samples: Dict[Tuple, Dict[str, np.ndarray]],
    lr=0.05,
):

    interventional_samples = interventional_samples[intervened_var]
    for graph_idx, emission_fncs in enumerate(
        all_emission_fncs
    ):  # as many emission_fncs dicts as graphs

        graph = graphs[graph_idx]
        print(f"Graph {graph_idx}")
        print(emission_fncs)
        for var in emission_fncs:
            parents = graph.parents[var]
            print(var, parents)
            xx = np.hstack(
                [interventional_samples[parent].reshape(1, -1) for parent in parents]
            )
            yy = interventional_samples[var].reshape(1, -1)

            # if isinstance(output, list):
            #     assert len(output) == 1
            #     output = output[0]

            # XXX LOOK AT THIS AGAIN, NOT SURE I COMPLETELY UNDERSTAND
            if var in intervened_var:
                # Here the truncated assumption comes in. Dont compute posterior
                continue
            # else:
            #     print(var, parents)

            posterior[graph_idx] += lr * log_likelihood(
                emission_fncs[var], xx, yy  # the model
            )

    return posterior


def log_likelihood(model: GPRegression, X_test: np.ndarray, y_test: np.ndarray):
    if y_test is not None:
        log_lik = model.log_predictive_density(x_test=X_test, y_test=y_test)
        res = log_lik.flatten().sum()
    else:
        # Marginal
        log_lik = model.score_samples(X_test)
        res = log_lik.flatten().sum()

    return res


# TODO: this is really a repetition since I needed this here outside the classs of CEO/CBO. I copied this from root.
def get_sem_emit_obs(
    G,
    sem_emit_fncs,
    observational_samples,
    t: int,
    pa: tuple,
    t_index_data: int = None,
) -> object:

    if len(pa) == 2 and pa[0] == None:
        # Source node
        pa_y = pa[1].split("_")[0]
        xx = observational_samples[pa_y].reshape(1, -1)

        return (xx, None, "Source", pa_y)  # Changed for CEO

    else:
        # Loop over all parents / explanatory variables
        xx = []
        vv = []
        outputvar = pa[1].split("_")[0]

        for v in pa[0]:
            temp_v = v.split("_")[0]
            vv.append(temp_v)
            x = observational_samples[temp_v].reshape(1, -1)
            xx.append(x)
        xx = np.hstack(xx)

        yy = observational_samples[outputvar].reshape(1, -1)

    assert len(xx.shape) == 2
    assert len(yy.shape) == 2
    assert xx.shape[0] == yy.shape[0]  # Column arrays

    if xx.shape[0] != yy.shape[0]:
        min_rows = np.min((xx.shape[0], yy.shape[0]))
        xx = xx[: int(min_rows)]
        yy = yy[: int(min_rows)]

    return xx, yy, vv, [outputvar]


def normalize_log(l):
    return np.exp(l - logsumexp(l))


def get_monte_carlo_expectation(intervention_samples: dict):
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


# defining parameters here for now
n_observational = 100

# setting up the incorrect toygraphs
graphs: List[GraphStructure] = []

graph1 = ToyGraph()
graph1.mispecify_graph([("Z", "X"), ("Z", "Y")])
graphs.append(graph1)

graph2 = ToyGraph()
graph2.mispecify_graph([("X", "Z"), ("X", "Y")])
graphs.append(graph2)

graph3 = ToyGraph()
graph3.mispecify_graph([("X", "Z"), ("X", "Y"), ("Z", "Y")])
graphs.append(graph3)

graph4 = ToyGraph()
graph4.mispecify_graph([("X", "Y"), ("Z", "Y")])
graphs.append(graph4)

graph5 = ToyGraph()
graph5.mispecify_graph([("Z", "X"), ("Z", "Y"), ("X", "Y")])
graphs.append(graph5)

graph6 = ToyGraph()
graph6.mispecify_graph([("Z", "X"), ("X", "Y")])
graphs.append(graph6)

graph7 = ToyGraph()
graphs.append(graph7)

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

(
    _,
    _,
    best_objective_value,
    _,
    _,
    ground_truth,  # this can also be thought of as the ground truth
) = optimal_sequence_of_interventions(exploration_set, intervention_grid, graph, 42)

D_O = sample_model(SEM, sample_count=n_observational)

# exploration_set = ["X", "Z", ("X", "Z")]
exploration_set = ["X"]
# now define interventional samples
interventional_ranges = graph.get_interventional_range()
interventions = create_grid_interventions(interventional_ranges, num_points=10)
D_I = draw_interventional_samples_sem(interventions, exploration_set, graph)

# stuff needed at this point, graph, true_objective_value, all_CE

# get the surrogate model for each of the graphs
sem_emit_fncs = []
for graph in graphs:
    graph.fit_samples_to_graph(samples, set_priors=False)
    sem_emit_fncs.append(graph.functions)

# get the initial posterior model in log form
posterior = np.log(np.asarray([1 / len(graphs)] * len(graphs)))
all_posteriors = []
all_posteriors.append(normalize_log(deepcopy(posterior)))

for es in exploration_set:
    posterior = update_posterior_interventional(
        graphs, posterior, tuple(es), sem_emit_fncs, D_I
    )
    print(normalize_log(deepcopy(posterior)))
