# # Import necessary modules
# import argparse

# import numpy as np
# from avici.synthetic import Beta, RandInt, Uniform
# from avici.synthetic.gene import GRNSergio

# from diffcbed.envs.erdos_renyi import ErdosRenyi
# from graphs.graph_dream import Dream4Graph
# from utils.sem_sampling import change_obs_data_format_to_bo_sergio

# # Set up the random number generator
# rng = np.random.default_rng(seed=42)

# # Define distributions for sampling parameters
# b = Uniform(low=0.5, high=1.5)  # Basal reproduction rates
# k_param = Uniform(low=0.1, high=0.5)  # Interaction strengths (non-negative)
# k_sign_p = Beta(a=2.0, b=2.0)  # Probability of positive interaction signs
# cell_types = RandInt(low=1, high=2)  # Number of cell types

# dream_env_name = "Ecoli1"
# yml_name = f"InSilicoSize10-{dream_env_name}"
# graph = Dream4Graph(yml_name=yml_name)


# # Just normally simulating the data
# hill = 2.0
# decays = 0.8
# noise_params = 0.1
# args = argparse.Namespace(scm_bias=0.0, noise_bias=0.0, old_er_logic=True)

# # Initialize the GRNSergio simulator
# simulator = GRNSergio(
#     b=b,
#     k_param=k_param,
#     k_sign_p=k_sign_p,
#     hill=hill,
#     decays=decays,
#     noise_params=noise_params,
#     cell_types=cell_types,
#     noise_type="dpd",
#     sampling_state=10,
#     dt=0.01,
#     tech_noise_config=None,
#     add_outlier_effect=False,
#     add_lib_size_effect=False,
#     add_dropout_effect=False,
#     return_count_data=False,
#     n_ko_genes=9,  # No knockouts; observational data only
# )

# # Create a random adjacency matrix representing the GRN
# n_vars = 10  # Number of genes
# # Random adjacency matrix with binary values (0 or 1)
# g = graph.causal_env.adjacency_matrix
# print(g)

# # Define the number of observations
# n_observations_obs = 200  # Number of observational samples
# n_observations_int = 20  # Number of interventional samples (none in this case)

# # Run the simulator
# data = simulator(rng, g, n_observations_obs, n_observations_int)

# # # Access the simulated data
# x_obs = data.x_obs  # Observational data
# x_int = data.x_int  # Interventional data (empty in this case)

# # # Print shapes of the generated data
# print(f"x_obs shape: {x_obs.shape}")  # Should be (n_observations_obs, n_vars, 2)
# print(
#     f"x_int shape: {x_int.shape}"
# )  # Should be (0, n_vars, 2) since n_observations_int = 0

# # Extract gene expression data and intervention masks from x_obs
# gene_expression = x_obs[:, :, 0]
# intervention_mask = x_obs[:, :, 1]

# # Print first few samples


# # Now adjusting the expression levels of the genes

# D_O = change_obs_data_format_to_bo_sergio(x_obs, graph.variables)
# # print(bo_data)


# # print(x_int.shape)
# # Initialize D_I as a dictionary where each key is mapped to None
# D_I = {str(var): None for var in graph.variables}

# # Iterate over each element in x_int
# for x in x_int:
#     intervention = x[:, 1]  # Select the second column (as per your structure)
#     bo_temp = change_obs_data_format_to_bo_sergio(
#         np.expand_dims(x, axis=0), graph.variables
#     )  # Get formatted data
#     # print(bo_temp)
#     node = next(
#         (i for i, val in enumerate(intervention) if val == 1), None
#     )  # Find the node index where value is 1
#     node = str(node)  # Convert node to string (since D_I keys are strings)

#     # Check if D_I[node] is None (i.e., first time accessing this node)
#     if D_I[node] is not None:
#         # If D_I[node] exists, vertically stack bo_temp's values for the corresponding intervention_value
#         for intervention_value in graph.variables:
#             print(D_I[node][intervention_value])
#             print(bo_temp[intervention_value])
#             D_I[node][intervention_value] = np.hstack(
#                 [
#                     D_I[node][intervention_value].flatten(),
#                     bo_temp[intervention_value].flatten(),
#                 ]
#             )
#     else:
#         # If this is the first time accessing D_I[node], initialize it with bo_temp values
#         # Reshape bo_temp[intervention_value] if necessary (e.g., if it's a scalar)
#         D_I[node] = {
#             var: bo_temp[var] if bo_temp[var].ndim == 1 else bo_temp[var]
#             for var in graph.variables
#         }


# print(D_I)

import argparse

from diffcbed.envs.erdos_renyi import ErdosRenyi
from diffcbed.envs.graph_to_env import GraphStructureEnv
from graphs.data_setup import setup_observational_interventional
from graphs.graph_chain import ChainGraph
from graphs.graph_erdos_renyi import ErdosRenyiGraph
from graphs.toy_graph import ToyGraph
from utils.sem_sampling import sample_model

graph = ErdosRenyiGraph(num_nodes=200, nonlinear=False)
graph.set_target("2")
# graph.parents['1']
D_O, D_I, exploration_set = setup_observational_interventional(
    graph_type=None, graph=graph
)
