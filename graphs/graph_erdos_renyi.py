import argparse
import logging
from collections import OrderedDict
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple

import numpy as np
from GPy.models.gp_regression import GPRegression

from config import NOISE_TYPE_INDEX, NOISE_TYPES
from envs.erdos_renyi import CausalEnvironment, ErdosRenyi
from graphs.graph import GraphStructure
from graphs.graph_chain import define_SEM_causalenv


class ErdosRenyiGraph(GraphStructure):

    def __init__(self, num_nodes: int):
        args = argparse.Namespace(scm_bias=1.0, noise_bias=1.0, old_er_logic=True)
        self.num_nodes = num_nodes

        self.causal_env: CausalEnvironment = ErdosRenyi(
            args=args, num_nodes=num_nodes, binary_nodes=True
        )
        self._SEM = self.define_SEM()
        self._variables = [str(i) for i in range(num_nodes)]

        self._edges = [
            (str(edge[0]), str(edge[1])) for edge in self.causal_env.graph.edges
        ]

        self._nodes = sorted(set(chain(*self.edges)))
        self._parents, self._children = self.build_relationships()
        self._target = f"{num_nodes - 1}"
        self._functions: Optional[Dict[str, Callable]] = None

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
