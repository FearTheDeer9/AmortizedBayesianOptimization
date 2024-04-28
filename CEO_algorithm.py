import logging
import random
from copy import deepcopy
from typing import Callable, Dict, List, OrderedDict

import numpy as np
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from GPy.models.gp_regression import GPRegression

import utils.ceo_utils as ceo_utils
import utils.utils_functions as utils_functions
from utils.graph_utils.graph import GraphStructure
from utils.graph_utils.graph_functions import create_grid_interventions, graph_setup
from utils.graph_utils.toy_graph import ToyGraph
from utils.sem_sampling import draw_interventional_samples_sem, sample_model
from utils.utils_classes import DoFunctions

# defining parameters here for now
n_observational = 100
T = 10
trial_observed: List[bool] = []
all_sem_hat = []

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
num_interventions = len(exploration_set)
interventional_range = graph.get_interventional_range()
intervention_grid = create_grid_interventions(
    interventional_range, get_list_format=True
)
arm_distribution = np.array([1 / len(exploration_set)] * len(exploration_set))

# (
#     _,
#     best_es_initial,
#     best_objective_value_initial,
#     _,
#     _,
#     ground_truth,  # this can also be thought of as the ground truth
# ) = optimal_sequence_of_interventions(exploration_set, intervention_grid, graph)
# setting up the data for the rest of the algorithm

data_x_list, data_y_list, best_intervention_value, current_global_min, best_variable = (
    utils_functions.define_initial_data_CBO(
        interventional_samples,
        exploration_set,
        manipulative_variables,
        target,
    )
)

print(data_x_list)


D_O: Dict[str, np.ndarray] = sample_model(SEM, sample_count=n_observational)

# exploration_set = ["X", "Z", ("X", "Z")]
# exploration_set = ["X"]
# now define interventional samples
interventional_ranges = graph.get_interventional_range()
interventions = create_grid_interventions(interventional_ranges, num_points=10)
D_I = draw_interventional_samples_sem(interventions, exploration_set, graph)

# stuff needed at this point, graph, true_objective_value, all_CE

# get the surrogate model for each of the graphs
sem_emit_fncs: List[OrderedDict[str, Callable]] = []
do_effects_functions: List[List[DoFunctions]] = []
for graph in graphs:
    graph.fit_samples_to_graph(samples, set_priors=False)
    sem_emit_fncs.append(graph.functions)

# get the initial posterior model in log form
posterior = np.log(np.asarray([1 / len(graphs)] * len(graphs)))
all_posteriors = []
all_posteriors.append(ceo_utils.normalize_log(deepcopy(posterior)))

for es in exploration_set:
    posterior = ceo_utils.update_posterior_interventional(
        graphs, posterior, tuple(es), sem_emit_fncs, D_I
    )

all_posteriors.append(ceo_utils.normalize_log(deepcopy(posterior)))

for key in D_O:
    D_O[key] = D_O[key].T.tolist()

for graph, emission_fncs in zip(graphs, sem_emit_fncs):
    all_sem_hat.append(emission_fncs)

for graph in graphs:
    # this is the mean and variance for each graph for each element in the exploration set
    do_effects_functions.append(
        utils_functions.update_all_do_functions(
            graph, observational_samples, exploration_set
        )
    )

# for i in range(len(exploration_set)):
#     es = tuple(exploration_set[i])
#     x = np.array([interventional_samples[i][es[0]][0]])
#     print(all_posteriors[-1])
#     print(
#         ceo_utils.aggregate_var_function(
#             i,
#             do_effects_functions,
#             all_posteriors[-1],
#             x,
#         )
#     )

input_space = [len(es) for es in exploration_set]
causal_prior = True
model_list: List[GPyModelWrapper] = [None] * len(exploration_set)
model_list_overall: List[GPyModelWrapper] = [None] * len(exploration_set)
n_es_mapping = {i: es for i, es in enumerate(exploration_set)}

print(intervention_grid)
print(interventional_range)
for i in range(T):
    if i == 0:
        # # update the prior of all the Gaussian Processes for each graph
        # trial_observed.append(True)

        # for g in range(len(graphs)):
        #     model_list[g] = utils_functions.update_posterior_model(
        #         exploration_set,
        #         trial_observed[i - 1],
        #         model_list[g],
        #         data_x_list,
        #         data_y_list,
        #         causal_prior,
        #         best_variable,
        #         input_space,
        #         do_effects_functions[g],
        #     )

        # for s, es in enumerate(exploration_set):
        model_list_overall = ceo_utils.update_posterior_model_aggregate(
            exploration_set,
            True,
            model_list_overall,
            data_x_list,
            data_y_list,
            causal_prior,
            best_variable,
            input_space,
            do_effects_functions,
            all_posteriors[-1],
        )

        # model_list = utils_functions.update_posterior_model(
        #     exploration_set,
        #     True,
        #     model_list,
        #     data_x_list,
        #     data_y_list,
        #     causal_prior,
        #     best_variable,
        #     input_space,
        #     do_effects_functions[0],
        # )
    else:

        # update the arm distribution, i.e. that each intervention in the exploration set is optimal
        arm_distribution = ceo_utils.update_arm_distribution(
            arm_distribution, model_list_overall, data_x_list, n_es_mapping
        )

        py_star_samples, p_x_star_samples = ceo_utils.build_p_y_star(
            exploration_set, model_list_overall, interventional_range, intervention_grid
        )

        samples_global_ystar, samples_global_xstar = ceo_utils.sample_global_xystar(
            n_samples_mixture=1000, all_ystar=py_star_samples, arm_dist=arm_distribution
        )

        kde_global = ceo_utils.MyKDENew(samples_global_xstar)
        try:
            kde_global.fit()
        except RuntimeError:
            kde_global.fit(bw=0.5)

        print(kde_global)
        break
