import os
import sys

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Get parent_method from command line argument

if len(sys.argv) == 4:
    graph_type = sys.argv[1]
    parent_method = sys.argv[2]
    n_observational = int(sys.argv[3])
elif len(sys.argv) == 5:
    graph_type = sys.argv[1]
    parent_method = sys.argv[2]
    nonlinear = sys.argv[3]
    n_observational = int(sys.argv[4])
else:
    print("Usage: python3 run_graph_scripts.py <GraphType> <parent_method>")
    sys.exit(1)

# Define the fixed parameters
# n_observational = 500
n_trials = 50
n_anchor_points = 35
noiseless = "--noiseless"

# Define the varying parameters
seeds_replicate = [71, 11, 89, 69, 7, 50, 100, 111, 619, 77]
run_nums = range(1, 11)

# Loop through the combinations of seeds and run numbers
for seed, run_num in zip(seeds_replicate, run_nums):
    # linear one
    command = (
        f"python3 large_graph_script.py --seeds_replicate {seed} --n_observational {n_observational} "
        f"--n_trials {n_trials} --n_anchor_points {n_anchor_points} --run_num {run_num} {noiseless} "
        f'--graph_type "{graph_type}" --parent_method "{parent_method}"'
    )
    os.system(command)
    print(f"Executed: {command}")

    # non-linear one
    command = (
        f"python3 large_graph_script.py --seeds_replicate {seed} --n_observational {n_observational} "
        f"--n_trials {n_trials} --n_anchor_points {n_anchor_points} --run_num {run_num} {noiseless} "
        f'--graph_type "{graph_type}" --parent_method "{parent_method}" --nonlinear'
    )
    os.system(command)
    print(f"Executed: {command}")
