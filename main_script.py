import os
import pickle

from algorithms.BO_algorithm import BO
from algorithms.CBO_algorithm import CBO
from algorithms.CEO_algorithm import CEO
from graphs.graph_6_nodes import Graph6Nodes
from graphs.toy_graph import ToyGraph
from utils.sem_sampling import sample_from_SEM_hat, sample_model

all_graph_edges = [
    [("Z", "X"), ("Z", "Y")],
    [("X", "Z"), ("X", "Y")],
    [("X", "Z"), ("X", "Y"), ("Z", "Y")],
    [("X", "Y"), ("Z", "Y")],
    [("Z", "X"), ("Z", "Y"), ("X", "Y")],
    [("Z", "X"), ("X", "Y")],
    [("X", "Z"), ("Z", "Y")],
]
directory = "/Users/jeandurand/Documents/Masters Thesis/CEO/"
print(os.listdir(directory))

with open("/Users/jeandurand/Documents/Masters Thesis/CEO/D_0.pickle", "rb") as handle:
    D_O = pickle.load(handle)

with open("/Users/jeandurand/Documents/Masters Thesis/CEO/D_I.pickle", "rb") as handle:
    D_I = pickle.load(handle)


# print(D_I)

exploration_set = [("Z",), ("X",), ("X", "Z")]
# # exploration_set = [("X", "Z")]
model = CEO(all_graph_edges=all_graph_edges)
# model.set_values(D_O, D_I, exploration_set)
model.run_algorithm()
