from collections import Counter
from typing import List

import numpy as np
import pandas as pd
import tqdm

from graphs.graph import GraphStructure
from posterior_model.doubly_robust import DoublyRobustClassWrapper


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

    def run_bootstrap_obs(self, data):
        n_obs = len(data)
        indices = np.arange(n_obs)
        parents = self.graph.parents[self.target]
        groundtruth = np.zeros(shape=len(self.graph.variables) - 1)
        for i, var in enumerate(self.topological_order):
            if var != self.graph.target and var in parents:
                groundtruth[i] = 1
        groundtruth = pd.Series(groundtruth.astype(bool))

        data_conf = {}
        index_counts = Counter()

        for i in tqdm.tqdm(range(self.num_bootstraps)):
            # Randomly sample indices with replacement
            bootstrap_indices = np.random.choice(indices, size=n_obs, replace=True)
            data_use = data.loc[bootstrap_indices]

            doubly_robust_method = DoublyRobustClassWrapper(
                data_use, groundtruth, data_conf, self.topological_order, self.target
            )
            estimate = doubly_robust_method.infer_causal_parents()
            self.markov_dags.append((estimate[estimate == 1].index))
            index_counts.update((estimate[estimate == 1].index))

        # Step 3: Calculate proportions
        index_proportions = {
            index: count / self.num_bootstraps for index, count in index_counts.items()
        }

        # Output the counts and proportions
        print("Counts of each selected index:", index_counts)
        print("Proportions of each selected index:", index_proportions)
