import os
import pickle
import sys

# set the directory to the root
# os.chdir("/Users/jeandurand/Documents/Masters Thesis/causal_bayes_opt")
os.chdir("..")

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import numpy as np

from algorithms.PARENT_algorithm import PARENT
from diffcbed.envs.graph_to_env import GraphStructureEnv
from diffcbed.models import DagBootstrap
from diffcbed.strategies import PolicyOptNMC
from graphs.data_setup import setup_observational_interventional
from graphs.graph import GraphStructure
from graphs.graph_5_nodes import Graph5Nodes
from graphs.graph_6_nodes import Graph6Nodes
from graphs.graph_chain import ChainGraph
from graphs.graph_erdos_renyi import ErdosRenyiGraph
from graphs.toy_graph import ToyGraph
from scripts.base_script import parse_args


def test_method_accuracy(
    graph: GraphStructure, n_int: int, n_obs: int, num_simulations: int = 40
):
    graph_env = GraphStructureEnv(graph, args)
    model = DagBootstrap(graph_env, args)
    strategy = PolicyOptNMC(model, graph_env, args)
    accuracies = []
    for i in range(num_simulations):
        parent = PARENT(graph, graph_env, model, strategy)
        D_O, D_I, _ = setup_observational_interventional(
            None, graph=graph, n_int=n_int, n_obs=n_obs, seed=i
        )
        parent.set_values(D_O, D_I)
        accuracy = parent.check_dr_parent_accuracy()
        print(f"THE ACCURACY IS {accuracy}")
        accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    return mean_accuracy, std_accuracy


n_obs = 200
n_ints = [0, 2, 4, 6, 8, 10, 12, 14]
# n_ints = [10, 20, 30, 40, 50]
args = parse_args()

results_dict = {}
for n_int in n_ints:
    graph = ToyGraph()
    mean_acc, std_acc = test_method_accuracy(graph, n_int, n_obs)
    results_dict[f"Toy_{n_int}"] = {"mean": mean_acc, "std": std_acc}

    graph = Graph6Nodes()
    mean_acc, std_acc = test_method_accuracy(graph, n_int, n_obs)
    results_dict[f"Health_{n_int}"] = {"mean": mean_acc, "std": std_acc}

    graph = ChainGraph(num_nodes=5)
    mean_acc, std_acc = test_method_accuracy(graph, n_int, n_obs)
    results_dict[f"Chain_5_{n_int}"] = {"mean": mean_acc, "std": std_acc}

    graph = ChainGraph(num_nodes=10)
    mean_acc, std_acc = test_method_accuracy(graph, n_int, n_obs)
    results_dict[f"Chain_10_{n_int}"] = {"mean": mean_acc, "std": std_acc}

    graph = ChainGraph(num_nodes=15)
    mean_acc, std_acc = test_method_accuracy(graph, n_int, n_obs)
    results_dict[f"Chain_15_{n_int}"] = {"mean": mean_acc, "std": std_acc}

    graph = ChainGraph(num_nodes=20)
    mean_acc, std_acc = test_method_accuracy(graph, n_int, n_obs)
    results_dict[f"Chain_20_{n_int}"] = {"mean": mean_acc, "std": std_acc}

with open("results/dr_sweep/results_dict_2.pickle", "wb") as file:
    pickle.dump(results_dict, file)
