import os
import sys

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.4"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Set this environment variable before importing jax
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Define the varying parameters
seeds_replicate = [71, 11, 89, 69, 7, 50, 100, 111, 619, 77]
run_nums = range(1, 11)
# graphs = ["Erdos5", "Erdos6", "Erdos7", "Erdos8", "Erdos9", "Erdos10"]
graphs = ["Erdos7"]
n_obs_samples = [50, 100, 200]

# Loop through the combinations of seeds and run numbers
for seed, run_num in zip(seeds_replicate, run_nums):
    for graph in graphs:
        for n_obs in n_obs_samples:
            # linear one
            command = (
                f"python3 erdos_renyi_posterior.py --seeds_replicate {seed} --n_obs {n_obs} "
                f"--run_num {run_num} --p 1 "
                f'--graph_type "{graph}"  --include_nint '
            )
            os.system(command)
            print(f"Executed: {command}")

            # non-linear one
            # command = (
            #     f"python3 erdos_renyi_posterior.py --seeds_replicate {seed} --n_obs {n_obs} "
            #     f"--run_num {run_num} --nonlinear --p 1 "
            #     f'--graph_type "{graph}"  --include_nint '
            # )
            # os.system(command)
            # print(f"Executed: {command}")
