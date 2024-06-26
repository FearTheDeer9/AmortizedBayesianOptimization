import argparse

from algorithms.BOED_algorithm import BOED
from diffcbed.envs import Chain, ErdosRenyi, OnlyDAGDream4Environment, ScaleFree
from diffcbed.envs.graph_to_env import GraphStructureEnv
from diffcbed.models import DagBootstrap
from diffcbed.replay_buffer import ReplayBuffer
from diffcbed.strategies import (
    ABCDStrategy,
    BALDStrategy,
    BatchBALDStrategy,
    FScoreBatchStrategy,
    GreedyABCDStrategy,
    GridOptPCE,
    PCEBatchStrategy,
    PolicyOptCovEig,
    PolicyOptNMC,
    PolicyOptNMCFixedValue,
    PolicyOptPCE,
    RandomAcquisitionStrategy,
    RandOptPCE_BO,
    RandOptPCE_GD,
    ReplayStrategy,
    SoftBALDStrategy,
    SoftFScoreStrategy,
    SoftPCE_BO,
    SoftPCE_GD,
    SSFinite,
)
from graphs.data_setup import setup_observational_interventional
from graphs.graph_6_nodes import Graph6Nodes
from graphs.toy_graph import ToyGraph
from graphs.graph_erdos_renyi import ErdosRenyiGraph
from graphs.graph_chain import ChainGraph

STRATEGIES = {
    "greedyabcd": GreedyABCDStrategy,
    "abcd": ABCDStrategy,
    "softbald": SoftBALDStrategy,
    "batchbald": BatchBALDStrategy,
    "bald": BALDStrategy,
    "random": RandomAcquisitionStrategy,
    "replay": ReplayStrategy,
    "f-score": FScoreBatchStrategy,
    "softf-score": SoftFScoreStrategy,
    "pce": PCEBatchStrategy,
    "softpce_bo": SoftPCE_BO,
    "randoptpce_bo": RandOptPCE_BO,
    "softpce_gd": SoftPCE_GD,
    "randoptpce_gd": RandOptPCE_GD,
    "policyoptpce": PolicyOptPCE,
    "ss_finite": SSFinite,
    "policyoptnmc": PolicyOptNMC,
    "policyoptnmc_fixed_value": PolicyOptNMCFixedValue,
    "gridpce": GridOptPCE,
    "policyoptcoveig": PolicyOptCovEig,
}

MODELS = {
    "dag_bootstrap": DagBootstrap,
}

ENVS = {
    "erdos": ErdosRenyi,
    "chain": Chain,
    "sf": ScaleFree,
    "semidream4": OnlyDAGDream4Environment,
    "graph": GraphStructureEnv,
}



strategy_name = "policyoptnmc"
model_name = "dag_bootstrap"
env = "graph"

# setting up each of these three

def remap_labels(D_O, D_I, nodes_to_int_map):
    D_O = {nodes_to_int_map[key]: value for key, value in D_O.items()}
    D_I_new = {}
    for intervention_keys in D_I.keys():
        intervention_data = D_I[intervention_keys]
        new_intervention_keys = tuple([nodes_to_int_map[key] for key in intervention_keys])
        D_I_new[new_intervention_keys] = {nodes_to_int_map[key]: value for key, value in intervention_data.items()}

    D_I = D_I_new.copy()
    return D_O, D_I

def parse_args():
    parser = argparse.ArgumentParser(
        description="Differentiable Multi-Target Causal Bayesian Experimental Design"
    )
    parser.add_argument(
        "--save_path", type=str, default="results/", help="Path to save result files"
    )
    parser.add_argument("--id", type=str, default=None, help="ID for the run")
    parser.add_argument(
        "--data_seed",
        type=int,
        default=20,
        help="random seed for generating data (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed (default: 42)",
    )
    parser.add_argument(
        "--num_nodes", type=int, default=5, help="Number of nodes in the causal model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dag_bootstrap",
        help="Posterior model to use {dag_bootstrap}",
    )
    parser.add_argument("--env", type=str, default="erdos", help="SCM to use")
    parser.add_argument(
        "--strategy",
        type=str,
        default="random",
        help="Acqusition strategy to use {abcd, random}",
    )
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument(
        "--sparsity_factor",
        type=float,
        default=0.0,
        help="Hyperparameter for sparsity regulariser",
    )
    parser.add_argument(
        "--exp_edges",
        type=float,
        default=0.1,
        help="probability of expected edges in random graphs",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Total number of samples in the synthetic data",
    )
    parser.add_argument(
        "--num_targets",
        type=int,
        default=1,
        help="Total number of targets.",
    )
    parser.add_argument(
        "--num_particles",
        type=int,
        default=30,
        help="Total number of posterior samples",
    )
    parser.add_argument(
        "--num_starting_samples",
        type=int,
        default=100,
        help="Total number of samples in the synthetic data to start with",
    )
    parser.add_argument(
        "--temperature_type",
        type=str,
        default="anneal",
        help="Whether to anneal the relaxed distribution temperature or keep it fixed",
    )
    parser.add_argument(
        "--exploration_steps",
        type=int,
        default=1,
        help="Total number of exploration steps in gp-ucb",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="isotropic-gaussian",
        help="Type of noise of causal model",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature of soft bald/ reparameterized sampling",
    )
    parser.add_argument(
        "--noise_sigma", type=float, default=0.1, help="Std of Noise Variables"
    )
    parser.add_argument(
        "--scm_bias",
        type=float,
        default=0.0,
        help="Bias term of the additive gaussian noise.",
    )
    parser.add_argument(
        "--theta_mu", type=float, default=2.0, help="Mean of Parameter Variables"
    )
    parser.add_argument(
        "--theta_sigma", type=float, default=1.0, help="Std of Parameter Variables"
    )
    parser.add_argument(
        "--gibbs_temp", type=float, default=1000.0, help="Temperature of Gibbs factor"
    )

    # TODO: improve names
    parser.add_argument(
        "--num_intervention_values",
        type=int,
        default=5,
        help="Number of interventional values to consider.",
    )
    parser.add_argument(
        "--intervention_values",
        type=float,
        nargs="+",
        help="Interventioanl values to set in `grid` value_strategy, else ignored.",
    )
    parser.add_argument(
        "--intervention_value",
        type=float,
        default=0.0,
        help="Interventional value to set in `fixed` value_strategy, else ingored.",
    )

    parser.add_argument(
        "--noise_bias",
        type=float,
        default=0.0,
        # help="Interventional value to set in `fixed` value_strategy, else ingored.",
    )


    parser.add_argument("--group_interventions", action="store_true")
    parser.add_argument("--plot_graphs", action="store_true")
    parser.add_argument("--save_models", action="store_true", default=False)
    parser.set_defaults(group_interventions=False)
    parser.add_argument("--nonlinear", action="store_true")
    parser.add_argument("--old_er_logic", action="store_true")
    parser.set_defaults(nonlinear=False)
    parser.add_argument("--weighted_posterior", action="store_true")
    parser.set_defaults(weighted_posterior=False)
    parser.add_argument("--reuse_posterior_samples", action="store_true")
    parser.set_defaults(reuse_posterior_samples=False)
    parser.add_argument("--include_gt_mec", action="store_true")
    parser.set_defaults(include_gt_mec=False)
    parser.add_argument(
        "--value_strategy",
        type=str,
        default="fixed",
        help="Possible strategies: gp-ucb, grid, fixed, sample-dist",
    )
    parser.add_argument(
        "--dream4_path",
        type=str,
        default="envs/dream4/configurations/",
        help="Path of DREAM4 files.",
    )
    parser.add_argument(
        "--dream4_name",
        type=str,
        default="insilico_size10_1",
        help="Name of DREAM4 experiment to load.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--save_graphs", action="store_true")

    # optimizers
    parser.add_argument(
        "--opt_lr",
        type=float,
        default=0.5,
        help="Learning rate of the gradient based optimizer.",
    )
    parser.add_argument(
        "--opt_epochs",
        type=int,
        default=100,
        help="Epochs for the gradient based optimizers",
    )

    parser.add_argument(
        "--node_range", default="-10:10", help="Node value ranges (constraints)"
    )

    parser.set_defaults(nonlinear=False)

    args = parser.parse_args()
    args.node_range = [float(item) for item in args.node_range.split(":")]

    if args.env == "sf":
        args.dibs_graph_prior = "sf"

    # if args.nonlinear == False:
    #     args.group_interventions = True

    if args.include_gt_mec:
        print("Setting weighted_posterior to True because include_gt_mec is set")
        args.weighted_posterior = True

    if args.reuse_posterior_samples:
        print(
            "Setting weighted_posterior to True because reuse_posterior_samples is set"
        )
        args.weighted_posterior = True

    return args


# graph = Graph6Nodes()
# num_nodes = len(graph.variables)
# args = parse_args()
# args.num_nodes = len(graph.variables)
# graph_env = GraphStructureEnv(graph, args)
# print(graph_env.sample_linear(50))
# graph_variables = graph_env.nodes
# D_O, D_I, _ = setup_observational_interventional("Graph6", n_obs=50, n_int=2, seed=42)

# model = MODELS[model_name](graph_env, args)
# strategy = STRATEGIES[strategy_name](model, graph_env, args)
# boed = BOED(graph_env, model, strategy, args, graph_variables)

# # remapping the keys in the datasets
# nodes_to_int_map = graph_env.node_map
# D_O, D_I = remap_labels(D_O, D_I, nodes_to_int_map)

# boed.set_values(D_O, D_I)
# boed.run_algorithm()

args = parse_args()
graph = ChainGraph(num_nodes=10)
args.num_nodes = len(graph.variables)
graph_env = GraphStructureEnv(graph, args)
print(graph_env.sample_linear(50))
graph_variables = graph_env.nodes
D_O, D_I, _ = setup_observational_interventional(None, n_obs=50, n_int=2, seed=42, graph=graph)

model = MODELS[model_name](graph_env, args)
strategy = STRATEGIES[strategy_name](model, graph_env, args)
boed = BOED(graph_env, model, strategy, args, graph_variables)

# remapping the keys in the datasets
nodes_to_int_map = graph_env.node_map
D_O, D_I = remap_labels(D_O, D_I, nodes_to_int_map)

boed.set_values(D_O, D_I)
boed.run_algorithm()

