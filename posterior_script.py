# Import necessary modules
import argparse

import numpy as np
from avici.synthetic import Beta, RandInt, Uniform
from avici.synthetic.gene import GRNSergio

from diffcbed.envs.erdos_renyi import ErdosRenyi

# Set up the random number generator
rng = np.random.default_rng(seed=42)

# Define distributions for sampling parameters
b = Uniform(low=0.5, high=1.5)  # Basal reproduction rates
k_param = Uniform(low=0.1, high=0.5)  # Interaction strengths (non-negative)
k_sign_p = Beta(a=2.0, b=2.0)  # Probability of positive interaction signs
cell_types = RandInt(low=1, high=2)  # Number of cell types


# Just normally simulating the data
hill = 2.0
decays = 0.8
noise_params = 0.1
args = argparse.Namespace(scm_bias=0.0, noise_bias=0.0, old_er_logic=True)
causal_env = ErdosRenyi(args, num_nodes=10, binary_nodes=True)

# Initialize the GRNSergio simulator
simulator = GRNSergio(
    b=b,
    k_param=k_param,
    k_sign_p=k_sign_p,
    hill=hill,
    decays=decays,
    noise_params=noise_params,
    cell_types=cell_types,
    noise_type="dpd",
    sampling_state=10,
    dt=0.01,
    tech_noise_config=None,
    add_outlier_effect=False,
    add_lib_size_effect=False,
    add_dropout_effect=False,
    return_count_data=False,
    n_ko_genes=9,  # No knockouts; observational data only
)

# Create a random adjacency matrix representing the GRN
n_vars = 10  # Number of genes
# Random adjacency matrix with binary values (0 or 1)
g = causal_env.adjacency_matrix
print(g)
np.fill_diagonal(g, 0)  # Remove self-loops

# Define the number of observations
n_observations_obs = 100  # Number of observational samples
n_observations_int = 4  # Number of interventional samples (none in this case)

# Run the simulator
data = simulator(rng, g, n_observations_obs, n_observations_int)

# # Access the simulated data
x_obs = data.x_obs  # Observational data
x_int = data.x_int  # Interventional data (empty in this case)

# # Print shapes of the generated data
print(f"x_obs shape: {x_obs.shape}")  # Should be (n_observations_obs, n_vars, 2)
print(
    f"x_int shape: {x_int.shape}"
)  # Should be (0, n_vars, 2) since n_observations_int = 0

# Extract gene expression data and intervention masks from x_obs
gene_expression = x_obs[:, :, 0]
intervention_mask = x_obs[:, :, 1]

# Print first few samples
print("Gene expression data (first 5 samples):")
print(gene_expression[:5])
print("Intervention mask (should be zeros for observational data):")
# print(intervention_mask[:5])
print(intervention_mask)

# Now adjusting the expression levels of the genes
