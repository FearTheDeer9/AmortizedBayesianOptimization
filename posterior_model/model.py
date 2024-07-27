import logging
from collections import Counter, namedtuple
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tqdm

from graphs.graph import GraphStructure
from posterior_model.doubly_robust_temp import DoublyRobustClassWrapper

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


class LinearSCMModel:
    def __init__(
        self,
        prior_probabilities: Dict[Tuple, float],
        graph: GraphStructure,
        sigma_y: float = 1.0,
        sigma_theta: float = 5.0,
    ):
        self.prior_probabilities = prior_probabilities
        self.graph = graph
        self.sigma_y = sigma_y
        self.sigma_theta = sigma_theta

    def update_all(self, x_dict: Dict[str, float], y: float):
        # suppose we observed a new sample, update all probabilities
        updated_posterior = {}
        total_prob = 0
        print("-------------------------------------")
        print(self.prior_probabilities)
        for parents in self.prior_probabilities:
            x_vec = np.array([x_dict[parent] for parent in parents])
            # print("----------------------")
            # print(parents)
            N = self.X_dict[parents].shape[1]
            self.X_obs = self.X_dict[parents]
            # self.X_obs = (1 / np.sqrt(N)) * self.X_dict[parents]
            updated_posterior[parents] = self.unnormalized_posterior(
                y, x_vec, self.prior_probabilities[parents]
            )
            print(updated_posterior[parents])
            total_prob += updated_posterior[parents]

        # print("UNNORMALIZED")
        # print(updated_posterior)
        self.prior_probabilities = {
            parents: updated_posterior[parents] / total_prob
            for parents in self.prior_probabilities
        }

        # print("NORMALIZED")
        print(self.prior_probabilities)

    def set_data(self, D_O_scaled: Dict[str, np.ndarray]):
        X_dict = {}
        self.y_obs = D_O_scaled[self.graph.target].reshape(-1)
        for parents in self.prior_probabilities:
            X_dict[parents] = np.vstack(
                [D_O_scaled[key].reshape(1, -1) for key in parents]
            )

        self.X_dict = X_dict.copy()

    def add_data(self, D_I: Dict[str, np.ndarray]):
        X_dict = {}
        self.y_obs = np.stack(self.y_obs, D_I[self.graph.target].reshape(-1))
        for parents in self.prior_probabilities:
            X_dict[parents] = np.vstack(
                [X_dict[parents], [D_I[key] for key in parents]]
            )

        self.X_dict = X_dict.copy()

    def log_unnormalized_posterior(self, y: float, x: np.ndarray, pg: float):
        p = len(x)
        # added this N here as well
        N = self.X_obs.shape[1]
        A = self.X_obs @ self.X_obs.T / (N * self.sigma_y**2) + np.eye(p) / (
            N * self.sigma_y**2
        )
        b = y * x + 1 / N * self.X_obs @ self.y_obs
        Sigma_inv = np.outer(x, x) / self.sigma_y**2 + A
        Sigma = np.linalg.inv(Sigma_inv)

        # Log of part1
        log_part1 = (
            np.log(pg)
            - 0.5 * np.log(2 * np.pi * self.sigma_y**2)
            + 1 / 2 * np.linalg.slogdet(A)[1]
        )

        x_T = x.T
        term1 = np.outer(x, x) / (self.sigma_y**2)
        I = np.eye(p)
        term2 = I / (self.sigma_theta**2)
        matrix_sum = term1 + term2

        # Log of part2
        log_part2 = (
            -1 / (2 * self.sigma_y**2) * y**2
            - (1 / (2 * self.sigma_y**4 * N**2))
            * self.y_obs.T
            @ self.X_obs.T
            @ A
            @ self.X_obs
            @ self.y_obs
            + 1 / (2 * self.sigma_y**2) * b.T @ Sigma @ b
        )

        log_det = np.linalg.slogdet((np.outer(x, x) / self.sigma_y**2 + A))[1]
        log_part3 = -0.5 * log_det

        log_posterior = log_part1 + log_part2 + log_part3

        return log_posterior

    # def log_unnormalized_posterior_2(self, y: float, x: np.ndarray, pg: float):
    #     p = len(x)
    #     # added this N here as well
    #     N = self.X_obs.shape[1]
    #     A = self.X_obs @ self.X_obs.T / (self.sigma_y**2) + np.eye(p) / (
    #         self.sigma_y**2
    #     )
    #     b = y * x + self.X_obs @ self.y_obs
    #     Sigma_inv = np.outer(x, x) / self.sigma_y**2 + A
    #     Sigma = np.linalg.inv(Sigma_inv)

    #     # trying to understand what is wrong
    #     print("-----------------------------------")

    #     # Log of part1
    #     log_part1 = (
    #         np.log(pg)
    #         - 0.5 * np.log(2 * np.pi * self.sigma_y**2)
    #         + 1 / 2 * np.linalg.slogdet(A)[1]
    #     )

    #     # print(pg)
    #     # print(np.log(2 * np.pi * self.sigma_y**2))
    #     # print(np.linalg.slogdet(A)[1])
    #     # print(Sigma)
    #     # print(log_part1)
    #     # The term inside the exponent
    #     x_T = x.T
    #     # print(x)
    #     # print(np.outer(x, x))
    #     term1 = np.outer(x, x) / (self.sigma_y**2)
    #     I = np.eye(p)
    #     term2 = I / (self.sigma_theta**2)
    #     matrix_sum = term1 + term2
    #     quadratic = x_T @ matrix_sum @ x
    #     # print(quadratic)

    #     # Log of part2
    #     log_part2 = (
    #         -1 / (2 * self.sigma_y**2) * y**2
    #         - (1 / (2 * self.sigma_y**4))
    #         * self.y_obs.T
    #         @ self.X_obs.T
    #         @ A
    #         @ self.X_obs
    #         @ self.y_obs
    #         + 1 / (2 * self.sigma_y**2) * b.T @ Sigma @ b
    #     )

    #     # print(A)
    #     print(self.y_obs.T @ self.X_obs.T @ A @ self.X_obs @ self.y_obs)
    #     print(b.T @ Sigma @ b)
    #     print(log_part2)
    #     log_det = np.linalg.slogdet((np.outer(x, x) / self.sigma_y**2 + A))[1]
    #     log_part3 = -0.5 * log_det
    #     # print(log_part3)

    #     log_posterior = log_part1 + log_part2 + log_part3

    #     return log_posterior

    def unnormalized_posterior(self, y: float, x: np.ndarray, pg: float):
        log_posterior = self.log_unnormalized_posterior(y, x, pg)
        return np.exp(log_posterior)

    # def unnormalized_posterior_2(self, y: float, x: np.ndarray, pg: float):
    #     log_posterior = self.log_unnormalized_posterior_2(y, x, pg)
    #     return np.exp(log_posterior)
