import argparse
import logging
from collections import OrderedDict
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple

import numpy as np
from GPy.models.gp_regression import GPRegression

from config import NOISE_TYPE_INDEX, NOISE_TYPES
from diffcbed.envs.causal_environment import CausalEnvironment
from diffcbed.envs.erdos_renyi import ErdosRenyi
from diffcbed.envs.samplers import D
from graphs.graph import GraphStructure
from graphs.graph_chain import define_SEM_causalenv


class ErdosRenyiGraph(GraphStructure):

    def __init__(self, num_nodes: int, seed: int = 17, nonlinear: bool = False):
        args = argparse.Namespace(scm_bias=0.0, noise_bias=0.0, old_er_logic=True)
        self.num_nodes = num_nodes

        self.causal_env: CausalEnvironment = ErdosRenyi(
            args=args, num_nodes=num_nodes, binary_nodes=True, nonlinear=nonlinear
        )
        self._SEM = self.define_SEM()
        self._variables = [str(i) for i in range(num_nodes)]

        self._edges = [
            (str(edge[0]), str(edge[1])) for edge in self.causal_env.graph.edges
        ]

        self._nodes = sorted(set(self.variables))
        self._parents, self._children = self.build_relationships()
        self._target = f"{num_nodes - 1}"
        self._functions: Optional[Dict[str, Callable]] = None
        self._G = self.make_graphical_model()

        self.rng = np.random.default_rng(seed)
        self._standardised = False
        self.use_intervention_range_data = False

    def set_target(self, target: str):
        # choose the variable that is the best one to optimize for the ErdosRenyi graph
        self._target = target

    def define_SEM(self):
        sem_functions = define_SEM_causalenv(
            self.causal_env.graph, self.causal_env.weighted_adjacency_matrix
        )
        return sem_functions

    def get_error_distribution(self, noiseless=False):
        err_dist = {}

        noise_type = NOISE_TYPES[NOISE_TYPE_INDEX]
        if noise_type.endswith("gaussian"):
            # Identifiable
            if noise_type == "isotropic-gaussian":
                self._noise_std = [self.noise_sigma] * len(self.nodes)
            elif noise_type == "gaussian":
                self._noise_std = np.linspace(0.1, 1.0, len(self.nodes))
            for i in range(len(self.nodes)):
                err_dist[self.nodes[i]] = D(
                    self.rng.normal, loc=0.0, scale=self._noise_std[i]
                ).sample(1)
        elif noise_type == "exponential":
            noise_std = [self.noise_sigma] * len(self.nodes)
            for i in range(len(self.nodes)):
                err_dist[self.nodes[i]] = D(
                    self.rng.exponential, scale=noise_std[i]
                ).sample(1)

        return err_dist

    def get_sets(self):
        mis = []
        pomis = []
        manipulative_variables = [var for var in self.variables if var != self.target]
        return mis, pomis, manipulative_variables
