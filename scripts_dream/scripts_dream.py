import os
import sys

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

n_trials = 50
n_anchor_points = 35
noiseless = "--noiseless"

# Define the varying parameters
seeds_replicate = [71, 11, 89, 69, 7, 50, 100, 111, 619, 77]
run_nums = range(1, 11)
# graph_type = "Ecoli1"
graph_type = "Ecoli2"

# Loop through the combinations of seeds and run numbers
for seed, run_num in zip(seeds_replicate, run_nums):
    # linear one
    command = (
        f"python3 base_script_dream.py --seeds_replicate {seed} --n_observational 200 "
        f"--n_trials {n_trials} --n_anchor_points {n_anchor_points} --run_num {run_num} {noiseless} "
        f'--graph_type "{graph_type}" --parent_method dr2'
    )
    os.system(command)
    print(f"Executed: {command}")
