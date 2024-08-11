import os
import sys

os.chdir("..")
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from scripts.base_script import parse_args
from scripts.base_script_unknown import run_script_unknown

args = parse_args()
n_int = 2
seeds_int_data = args.seeds_replicate

n_anchor_points = args.n_anchor_points
n_trials = args.n_trials
n_obs = args.n_observational
run_num = args.run_num
noiseless = args.noiseless
doubly_robust = args.doubly_robust
noisy_string = "" if noiseless else "_noisy"

run_script_unknown(
    graph_type="Toy",
    run_num=run_num,
    noiseless=noiseless,
    noisy_string=noisy_string,
    seeds_int_data=seeds_int_data,
    n_obs=n_obs,
    n_int=n_int,
    n_trials=n_trials,
    doubly_robust=doubly_robust,
    filename="ToyGraphUnknown",
)
