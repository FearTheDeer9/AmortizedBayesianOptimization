import logging
from collections import Counter, namedtuple
from typing import List

import numpy as np
import pandas as pd
import tqdm

from graphs.graph import GraphStructure
from posterior_model.doubly_robust import DoublyRobustClassWrapper

Data = namedtuple("Data", ["samples", "nodes"])


class DoublyRobustModel:
    def __init__(
        self,
        graph: GraphStructure,
        topological_order: List,
        target: str,
        num_bootstraps: int = 30,
    ):
        self.graph = graph
        self.topological_order = topological_order
        self.target = target
        self.num_bootstraps = num_bootstraps
        # list of all dags for which we estimate a posterior distribution
        self.markov_dags = []
        # probability estimate
        self.prob_estimate = None

    def run_method(self, data: Data):
        data_samples = data.samples
        data_nodes = data.nodes
        intervened_samples = np.any(data_nodes, axis=1)
        n = len(data_samples)
        obs_indices = np.arange(n)[~intervened_samples]
        int_indices = np.arange(n)[intervened_samples]
        n_obs = len(obs_indices)
        parents = self.graph.parents[self.target]
        groundtruth = np.zeros(shape=len(self.graph.variables) - 1)
        for i, var in enumerate(self.topological_order):
            if var != self.graph.target and var in parents:
                groundtruth[i] = 1
        groundtruth = pd.Series(groundtruth.astype(bool))

        data_conf = {}
        index_counts = Counter()

        # Verification step
        assert (
            len(np.intersect1d(obs_indices, int_indices)) == 0
        ), "Observational and interventional indices are not mutually exclusive"

        for i in tqdm.tqdm(range(self.num_bootstraps)):
            # Randomly sample indices with replacement
            bootstrap_indices = np.random.choice(obs_indices, size=n_obs, replace=True)
            combined_indices = np.concatenate((bootstrap_indices, int_indices))

            # Verification step
            assert (
                len(np.intersect1d(obs_indices, int_indices)) == 0
            ), "Observational and interventional indices are not mutually exclusive"

            data_use = Data(
                samples=data_samples[combined_indices],
                nodes=data_nodes[combined_indices],
            )

            doubly_robust_method = DoublyRobustClassWrapper(
                data_use, groundtruth, data_conf, self.topological_order, self.target
            )
            estimate = doubly_robust_method.infer_causal_parents()
            parents_estimate = tuple(estimate[estimate == 1].index)
            self.markov_dags.append(parents_estimate)
            index_counts.update([parents_estimate])

        # Step 3: Calculate proportions
        index_proportions = {
            index: count / self.num_bootstraps for index, count in index_counts.items()
        }

        # Output the counts and proportions
        logging.info(f"Counts of each selected index {index_counts}")
        logging.info(f"Proportions of each selected index: {index_proportions}")
        self.prob_estimate = index_proportions
        self.prob_estimate = index_proportions

        # this is just for the sanity check of testing these methods
        return estimate
