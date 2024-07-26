# import os

# # set the directory to the root
# # os.chdir("/Users/jeandurand/Documents/Masters Thesis/causal_bayes_opt")
# os.chdir("/vol/bitbucket/jd123/causal_bayes_opt")
# from algorithms.BOED_algorithm import BOED
# from algorithms.PARENT_algorithm import PARENT
# from diffcbed.envs.graph_to_env import GraphStructureEnv
# from diffcbed.models import DagBootstrap
# from diffcbed.strategies import PolicyOptNMC
# from graphs.data_setup import setup_observational_interventional
# from graphs.graph_6_nodes import Graph6Nodes
# from graphs.graph_chain import ChainGraph
# from graphs.graph_erdos_renyi import ErdosRenyiGraph
# from graphs.toy_graph import ToyGraph
# from scripts.base_script import parse_args

# args = parse_args()
# graph = ToyGraph()
# # graph = Graph6Nodes()
# graph = ChainGraph(num_nodes=10)
# # graph = ErdosRenyiGraph(num_nodes=10)
# # args.num_nodes = len(graph.variables)
# graph_env = GraphStructureEnv(graph, args)
# graph_variables = graph_env.nodes
# node_ranges = []
# interventional_ranges = graph.get_interventional_range()
# for node in graph.variables:
#     if node in interventional_ranges.keys():
#         node_ranges.append(tuple(interventional_ranges[node]))
#     else:
#         node_ranges.append((0, 0))
# args.node_range = node_ranges
# graph_env = GraphStructureEnv(graph, args)

# strategy_name = "policyoptnmc"
# model = DagBootstrap(graph_env, args)
# strategy = PolicyOptNMC(model, graph_env, args)
# parent = PARENT(graph, graph_env, model, strategy)
# D_O, D_I, _ = setup_observational_interventional(None, graph=graph, n_int=5, n_obs=200)
# parent.set_values(D_O, D_I)
# parent.run_algorithm(python_code=True)


# import jax
# import jax.numpy as jnp

# # Define values and ranges
# values = jnp.array([1, 2, 3]).reshape(-1, 1)
# val_range_single = [-10, 10]
# val_range_multiple = [(-10, 10), (-5, 5), (-2, 1)]

# # Apply original soft_tanh (assuming it's vectorized)
# original_soft_tanh = (
#     jax.nn.hard_sigmoid(
#         (values - val_range_single[0]) / (val_range_single[1] - val_range_single[0])
#     )
#     * (val_range_single[1] - val_range_single[0])
#     + val_range_single[0]
# )

# # Apply new element-wise soft_tanh
# element_wise_soft_tanh = jnp.stack(
#     [
#         jax.nn.hard_sigmoid((values[:, i] - vr[0]) / (vr[1] - vr[0])) * (vr[1] - vr[0])
#         + vr[0]
#         for i, vr in enumerate(val_range_multiple)
#     ],
#     axis=1,
# )

# print("Original soft_tanh output:", original_soft_tanh)
# print("Element-wise soft_tanh output:", element_wise_soft_tanh)

# import jax
# import jax.random as random

# key = random.PRNGKey(123)
# print(f"JAX backend: {jax.default_backend()}")

# from dibs.target import make_linear_gaussian_model, make_nonlinear_gaussian_model
# from dibs.utils import visualize_ground_truth

# key, subk = random.split(key)
# data, graph_model, likelihood_model = make_linear_gaussian_model(
#     key=subk, n_vars=20, graph_prior_str="sf"
# )

# attributes = [
#     "g",
#     "n_ho_observations",
#     "n_observations",
#     "n_vars",
#     "passed_key",
#     "theta",
#     "x",
#     "x_ho",
#     "x_interv",
# ]

# for attr in attributes:
#     print(f"{attr}:")
#     print(getattr(data, attr))
#     print()

# print("DATA")
# print(dir(data))
# print(data.x)
# print(data.g)
# print("GRAPH MODEL")
# print(graph_model)
# print("LIKELIHOOD MODEL")
# print(likelihood_model)

# from dibs.inference import JointDiBS

# dibs = JointDiBS(
#     x=data.x,
#     interv_mask=None,
#     graph_model=graph_model,
#     likelihood_model=likelihood_model,
# )
# key, subk = random.split(key)
# gs, thetas = dibs.sample(
#     key=subk,
#     n_particles=20,
#     steps=2000,
#     callback_every=100,
#     callback=dibs.visualize_callback(),
# )

# dibs_empirical = dibs.get_empirical(gs, thetas)
# dibs_mixture = dibs.get_mixture(gs, thetas)

# from dibs.metrics import expected_shd, neg_ave_log_likelihood, threshold_metrics

# for descr, dist in [("DiBS ", dibs_empirical), ("DiBS+", dibs_mixture)]:

#     eshd = expected_shd(dist=dist, g=data.g)
#     auroc = threshold_metrics(dist=dist, g=data.g)["roc_auc"]
#     negll = neg_ave_log_likelihood(
#         dist=dist,
#         eltwise_log_likelihood=dibs.eltwise_log_likelihood_observ,
#         x=data.x_ho,
#     )

#     print(
#         f"{descr} |  E-SHD: {eshd:4.1f}    AUROC: {auroc:5.2f}    neg. LL {negll:5.2f}"
#     )


# ChatGPT's code for active inference

import jax
import jax.random as random
from dibs.inference import JointDiBS
from dibs.target import make_nonlinear_gaussian_model

# Initialize the random key
key = random.PRNGKey(123)

# Generate initial data and models
key, subk = random.split(key)
data, graph_model, likelihood_model = make_nonlinear_gaussian_model(
    key=subk, n_vars=20, graph_prior_str="sf"
)

# Initialize DiBS sampler
dibs = JointDiBS(
    x=data.x,
    interv_mask=None,
    graph_model=graph_model,
    likelihood_model=likelihood_model,
)

# Perform initial sampling
key, subk = random.split(key)
gs, thetas = dibs.sample(
    key=subk,
    n_particles=20,
    steps=2000,
    callback_every=100,
    callback=dibs.visualize_callback(),
)


# Function to perform an intervention and update posterior
def perform_intervention_and_update(dibs, data, key, intervention):
    # Apply intervention (modify data.x and possibly interv_mask)
    # Update data.x with new interventional data
    data.x = apply_intervention(data.x, intervention)

    # Re-initialize DiBS sampler with updated data
    dibs = JointDiBS(
        x=data.x,
        interv_mask=data.interv_mask,
        graph_model=graph_model,
        likelihood_model=likelihood_model,
    )

    # Perform sampling with updated data
    key, subk = random.split(key)
    gs, thetas = dibs.sample(
        key=subk,
        n_particles=20,
        steps=2000,
        callback_every=100,
        callback=dibs.visualize_callback(),
    )

    return dibs, gs, thetas


# Iteratively perform interventions and update posterior
for intervention in interventions:
    dibs, gs, thetas = perform_intervention_and_update(dibs, data, key, intervention)

# Analyze final results
dibs_empirical = dibs.get_empirical(gs, thetas)
dibs_mixture = dibs.get_mixture(gs, thetas)
