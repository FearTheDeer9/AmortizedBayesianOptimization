import argparse
import logging
import os
import sys

os.chdir("..")

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from scripts.base_script import run_script, run_script_uncertainty

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Set the format of log messages
    datefmt="%m/%d/%Y %I:%M:%S %p",  # Set the date format
)

# this is the arguments for running the script
parser = argparse.ArgumentParser()
parser.add_argument("--seeds_replicate", type=int)
parser.add_argument("--n_observational", type=int)
parser.add_argument("--n_trials", type=int)
parser.add_argument("--n_anchor_points", type=int)
parser.add_argument("--run_num", type=int)
parser.add_argument("--noiseless", action="store_true", help="Run without noise")

# using the arguments
args = parser.parse_args()
n_int = 2
seeds_int_data = args.seeds_replicate

n_anchor_points = args.n_anchor_points
n_trials = args.n_trials
n_obs = args.n_observational
run_num = args.run_num
noiseless = args.noiseless
noisy_string = "" if noiseless else "_noisy"
if run_num <= 5:
    safe_optimization = True
else:
    safe_optimization = False
# run the CEO method using all the edges and then the CBO for each of the e

all_graph_edges = [
    [
        ("A", "B"),
        ("A", "C"),
        ("A", "S"),
        ("A", "As"),
        ("A", "Y"),
        ("B", "As"),
        ("B", "C"),
        ("B", "S"),
        ("B", "Y"),
        ("As", "C"),
        ("As", "Y"),
        ("S", "C"),
        ("S", "Y"),
        ("C", "Y"),
    ],
    [
        # ("A", "B"),
        ("A", "C"),
        ("A", "S"),
        ("A", "As"),
        ("A", "Y"),
        ("B", "As"),
        ("B", "C"),
        ("B", "S"),
        ("B", "Y"),
        ("As", "C"),
        ("As", "Y"),
        ("S", "C"),
        ("S", "Y"),
        ("C", "Y"),
    ],
    [
        ("A", "B"),
        # ("A", "C"),
        ("A", "S"),
        ("A", "As"),
        ("A", "Y"),
        ("B", "As"),
        ("B", "C"),
        ("B", "S"),
        ("B", "Y"),
        ("As", "C"),
        ("As", "Y"),
        ("S", "C"),
        ("S", "Y"),
        ("C", "Y"),
    ],
    [
        ("A", "B"),
        ("A", "C"),
        ("A", "S"),
        ("A", "As"),
        ("A", "Y"),
        ("B", "As"),
        # ("B", "C"),
        ("B", "S"),
        ("B", "Y"),
        ("As", "C"),
        ("As", "Y"),
        ("S", "C"),
        ("S", "Y"),
        ("C", "Y"),
    ],
    [
        ("A", "B"),
        ("A", "C"),
        ("A", "S"),
        ("A", "As"),
        ("A", "Y"),
        ("B", "As"),
        # ("B", "C"),
        ("B", "S"),
        ("B", "Y"),
        ("As", "C"),
        ("As", "Y"),
        ("S", "C"),
        ("S", "Y"),
        ("C", "Y"),
    ],
    [
        ("A", "B"),
        ("A", "C"),
        ("A", "S"),
        ("A", "As"),
        # ("A", "Y"),
        ("B", "As"),
        ("B", "C"),
        ("B", "S"),
        ("B", "Y"),
        ("As", "C"),
        ("As", "Y"),
        ("S", "C"),
        ("S", "Y"),
        ("C", "Y"),
    ],
    [
        ("A", "B"),
        ("A", "C"),
        ("A", "S"),
        ("A", "As"),
        ("A", "Y"),
        ("B", "As"),
        ("B", "C"),
        ("B", "S"),
        # ("B", "Y"),
        ("As", "C"),
        ("As", "Y"),
        ("S", "C"),
        ("S", "Y"),
        ("C", "Y"),
    ],
]


run_script(
    graph_type="Graph6",
    run_num=run_num,
    all_graph_edges=all_graph_edges,
    noiseless=noiseless,
    noisy_string=noisy_string,
    seeds_int_data=seeds_int_data,
    n_obs=n_obs,
    n_int=n_int,
    n_anchor_points=n_anchor_points,
    n_trials=n_trials,
    filename="Graph6_v2",
)

run_script_uncertainty(
    graph_type="Graph6",
    run_num=run_num,
    all_graph_edges=all_graph_edges,
    noiseless=noiseless,
    noisy_string=noisy_string,
    seeds_int_data=seeds_int_data,
    n_obs=n_obs,
    n_int=n_int,
    n_anchor_points=n_anchor_points,
    n_trials=n_trials,
    filename="Graph6_v2",
)
