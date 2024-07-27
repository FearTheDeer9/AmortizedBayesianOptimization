import itertools

import numpy as np

from graphs.data_setup import setup_observational_interventional
from graphs.graph_chain import ChainGraph
from graphs.graph_erdos_renyi import ErdosRenyiGraph
from posterior_model.model import LinearSCMModel


# Function to standardize data
def standardize(data, mean, std):
    return (data - mean) / std


# Function to reverse the standardization
def reverse_standardize(data, mean, std):
    return (data * std) + mean


graph = ErdosRenyiGraph(num_nodes=6, nonlinear=False)
n_obs = 100
n_int = 5
manipulative_variables = graph.get_sets()[2]
manipulative_variables = [item[0] for item in manipulative_variables]

random_seed = np.random.randint(1, 10000)
D_O, D_I, exploration_set = setup_observational_interventional(
    graph_type=None, graph=graph, n_obs=n_obs, n_int=n_int
)

# print(D_I[("0",)])
# set everything up so that it works better with your calculated likelihood
input_keys = [key for key in D_O.keys() if key != graph.target]
means = {key: np.mean(D_O[key]) for key in input_keys}
std = {key: np.std(D_O[key]) for key in input_keys}

D_O_scaled = {}
for key in D_O:
    if key in input_keys:
        D_O_scaled[key] = standardize(D_O[key], means[key], std[key])
    else:
        D_O_scaled[key] = D_O[key]

interventions = D_I.keys()
D_I_scaled = {intervention: {} for intervention in interventions}
for intervention in interventions:
    for key in D_I[intervention]:
        if key in input_keys:
            D_I_scaled[intervention][key] = standardize(
                D_I[intervention][key], means[key], std[key]
            )
        else:
            D_I_scaled[intervention][key] = D_I[intervention][key]

# Generate all unique combinations
combinations = []
for r in range(1, len(manipulative_variables) + 1):
    combinations.extend(itertools.combinations(manipulative_variables, r))

# Print the combinations
# print(D_O_scaled)

prior_probabilities = {combo: 1 / len(combinations) for combo in combinations}
# print(prior_probabilities)
model = LinearSCMModel(prior_probabilities, graph)
model.set_data(D_O_scaled)

# for n in range(5):
#     x_dict = {key: D_O_scaled[key][n] for key in D_O}
#     y = D_O_scaled[graph.target][n]
#     model.update_all(x_dict, y)


intervention = ("3",)
for n in range(5):
    x_dict = {
        key: D_I_scaled[intervention][key][n] for key in D_O if key != graph.target
    }
    y = D_I_scaled[intervention][graph.target][n]
    model.update_all(x_dict, y)

print(graph.parents[graph.target])
# compare how it change for obs data vs int data
