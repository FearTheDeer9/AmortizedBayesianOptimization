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


graph = ToyGraph()
num_nodes = len(graph.variables)
args = argparse.Namespace(scm_bias=0.0, noise_bias=0.0, num_nodes=num_nodes, 
                          num_particles=10, group_interventions=False, seed=42, num_samples=100, batch_size=1, value_strategy="fixed", num_targets=1, node_range=[-1.0,1.0])
graph_env = GraphStructureEnv(graph, args)
D_O, D_I, _ = setup_observational_interventional("Toy", n_obs=100, n_int=2, seed=42)

model = MODELS[model_name](graph_env, args)
strategy = STRATEGIES[strategy_name](model, graph_env, args)
boed = BOED(graph_env, model, strategy, args)
boed.set_values(D_O, D_I)
boed.run_algorithm()
