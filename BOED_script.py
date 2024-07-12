import os

# set the directory to the root
# os.chdir("/Users/jeandurand/Documents/Masters Thesis/causal_bayes_opt")
os.chdir("/vol/bitbucket/jd123/causal_bayes_opt")
from algorithms.BOED_algorithm import BOED
from algorithms.PARENT_algorithm import PARENT
from diffcbed.envs.graph_to_env import GraphStructureEnv
from diffcbed.models import DagBootstrap
from diffcbed.strategies import PolicyOptNMC
from graphs.data_setup import setup_observational_interventional
from graphs.graph_6_nodes import Graph6Nodes
from graphs.graph_chain import ChainGraph
from graphs.toy_graph import ToyGraph
from scripts.base_script import parse_args

args = parse_args()
# graph = ToyGraph()
graph = Graph6Nodes()
graph = ChainGraph(num_nodes=20)
args.num_nodes = len(graph.variables)
graph_env = GraphStructureEnv(graph, args)
graph_variables = graph_env.nodes
node_ranges = []
interventional_ranges = graph.get_interventional_range()
for node in graph.variables:
    if node in interventional_ranges.keys():
        node_ranges.append(tuple(interventional_ranges[node]))
    else:
        node_ranges.append((0, 0))
args.node_range = node_ranges
graph_env = GraphStructureEnv(graph, args)

strategy_name = "policyoptnmc"
model = DagBootstrap(graph_env, args)
strategy = PolicyOptNMC(model, graph_env, args)
parent = PARENT(graph, graph_env, model, strategy)
D_O, D_I, _ = setup_observational_interventional(None, graph=graph, n_int=0)
parent.set_values(D_O, D_I)
parent.run_algorithm()


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
