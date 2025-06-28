import abc
import logging
from collections import Counter, namedtuple
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tqdm

from graphs.graph import GraphStructure
from posterior_model.doubly_robust import DoublyRobustClassWrapper
from posterior_model.doubly_robust_temp import (
    DoublyRobustClassWrapper as DoublyRobustClassWrapperIndividual,
)

Data = namedtuple("Data", ["samples", "nodes"])


class RandomFourierFeatures:
    def __init__(self, input_dim: int, D: int, sigma: float = 1.0):
        self.D = D
        self.sigma = sigma
        self.omega = np.random.normal(scale=1 / sigma, size=(input_dim, D))
        self.b = np.random.uniform(0, 2 * np.pi, size=D)

    def transform(self, X) -> np.ndarray:
        projection = X @ self.omega + self.b
        return np.sqrt(2 / self.D) * np.cos(projection)


class DoublyRobustModel:
    def __init__(
        self,
        graph: GraphStructure,
        topological_order: List,
        target: str,
        num_bootstraps: int = 30,
        indivdual: bool = False,
    ):
        self.graph = graph
        self.topological_order = topological_order
        self.target = target
        self.num_bootstraps = num_bootstraps
        # list of all dags for which we estimate a posterior distribution
        self.markov_dags = []
        # probability estimate
        self.prob_estimate = None
        self.individual = indivdual

    def run_method(self, data: Data):
        data_samples = data.samples
        data_nodes = data.nodes
        intervened_samples = np.any(data_nodes, axis=1)
        n = len(data_samples)
        obs_indices = np.arange(n)[~intervened_samples]
        int_indices = np.arange(n)[intervened_samples]
        n_obs = len(obs_indices)
        parents = self.graph.parents[self.target]
        non_target_vars = [var for var in self.topological_order if var != self.graph.target]
        groundtruth = np.zeros(shape=len(non_target_vars))
        for i, var in enumerate(non_target_vars):
            if var in parents:
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

            if not self.individual:
                doubly_robust_method = DoublyRobustClassWrapper(
                    data_use,
                    groundtruth,
                    data_conf,
                    self.topological_order,
                    self.target,
                )
            else:
                doubly_robust_method = DoublyRobustClassWrapperIndividual(
                    data_use,
                    groundtruth,
                    data_conf,
                    self.topological_order,
                    self.target,
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

        # this is just for the sanity check of testing these methods
        return estimate


MESSAGE = "Subclass should implement this"


class SCMModel:
    __metaclass__ = abc.ABCMeta

    @property
    def prior_probabilities(self):
        return self._prior_probabilities

    def get_graph(self) -> GraphStructure:
        return self.graph

    def update_all(self, x_dict: Dict[str, float], y: float):
        raise NotImplementedError(MESSAGE)

    def set_data(self, D_O_scaled: Dict[str, np.ndarray]):
        raise NotImplementedError(MESSAGE)

    def add_data(self, D_I: Dict[str, np.ndarray]):
        raise NotImplementedError(MESSAGE)

    def log_unnormalized_posterior(self, y: float, x: np.ndarray, pg: float):
        raise NotImplementedError(MESSAGE)

    def redefine_prior_probabilities(self):

        prior_probabilities = {}
        for parents in self._prior_probabilities:
            if self._prior_probabilities[parents] > 1e-5:
                prior_probabilities[parents] = self._prior_probabilities[parents]

        self._prior_probabilities = prior_probabilities

    def unnormalized_posterior(self, y: float, x: np.ndarray, pg: float):
        log_posterior = self.log_unnormalized_posterior(y, x, pg)
        return np.exp(log_posterior)

    def update_probabilities(self, non_null_parents):
        # Create a list of keys to be deleted
        keys_to_delete = [
            parents
            for parents in self._prior_probabilities
            if parents not in non_null_parents
        ]

        # Delete the keys
        for parents in keys_to_delete:
            del self._prior_probabilities[parents]

    def calculate_metrics(self):
        graph = self.get_graph()
        true_parents = set(graph.parents[graph.target])
        true_non_parents = set(
            [
                var
                for var in graph.variables
                if var not in true_parents and var != graph.target
            ]
        )
        predicted_parents = list(self.prior_probabilities.keys())

        accuracies = np.zeros(len(predicted_parents))
        precision = np.zeros(len(predicted_parents))
        recall = np.zeros(len(predicted_parents))
        f1_scores = np.zeros(len(predicted_parents))

        for i, parents in enumerate(predicted_parents):
            parent_set = set(parents)
            negative_prediction = [
                var
                for var in graph.variables
                if var != graph.target and var not in parent_set
            ]

            positive_correct_predictions = len(true_parents.intersection(parent_set))
            negative_correct_predictions = len(
                true_non_parents.intersection(negative_prediction)
            )
            num_correct = positive_correct_predictions + negative_correct_predictions

            # Calculate metrics
            accuracies[i] = num_correct / (len(graph.variables) - 1)
            precision[i] = (
                positive_correct_predictions / len(parent_set)
                if len(parent_set) > 0
                else 0
            )
            recall[i] = (
                positive_correct_predictions / len(true_parents)
                if len(true_parents) > 0
                else 0
            )
            f1_scores[i] = (
                (2 * precision[i] * recall[i]) / (precision[i] + recall[i])
                if (precision[i] + recall[i]) > 0
                else 0
            )

        # Convert prior probabilities to numpy array (ensure 1D scalar values)
        prob_values = []
        for prob_val in self.prior_probabilities.values():
            if isinstance(prob_val, (int, float, np.number)):
                prob_values.append(float(prob_val))
            else:
                # If it's an array, flatten it and take the mean or first value
                array_val = np.asarray(prob_val).flatten()
                if array_val.size == 1:
                    prob_values.append(float(array_val[0]))
                elif array_val.size > 1:
                    # If multiple values, take the mean
                    prob_values.append(float(np.mean(array_val)))
                else:
                    # Empty array, use 0
                    prob_values.append(0.0)
        probs = np.array(prob_values)

        # Compute weighted metrics (ensure dimensions match)
        if len(accuracies) == len(probs) and accuracies.size > 0 and probs.size > 0:
            self.accuracy.append(np.dot(accuracies, probs))
            self.precision.append(np.dot(precision, probs))
            self.recall.append(np.dot(recall, probs))
            self.f1_score.append(np.dot(f1_scores, probs))
        else:
            # Fallback: use simple average if dimensions don't match
            self.accuracy.append(np.mean(accuracies) if accuracies.size > 0 else 0.0)
            self.precision.append(np.mean(precision) if precision.size > 0 else 0.0)
            self.recall.append(np.mean(recall) if recall.size > 0 else 0.0)
            self.f1_score.append(np.mean(f1_scores) if f1_scores.size > 0 else 0.0)


class LinearSCMModel(SCMModel):
    def __init__(
        self,
        prior_probabilities: Dict[Tuple, float],
        graph: GraphStructure,
        sigma_y: float = 1.0,
        sigma_theta: float = 1.0,
    ):
        self._prior_probabilities = prior_probabilities
        self.graph = graph
        self.sigma_y = sigma_y
        self.sigma_theta = sigma_theta
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.f1_score = []

    def update_all(self, x_dict: Dict[str, float], y: float):
        # suppose we observed a new sample, update all probabilities
        updated_posterior = {}
        total_prob = 0
        for parents in self._prior_probabilities:
            # start with the prior Gaussian
            Sigma_prior_inv = np.linalg.inv(self.Sigma_dict[parents])
            mu_prior = self.mu_dict[parents]
            # getting the correct format for the data
            x_vec = np.array([x_dict[parent] for parent in parents])

            self.Sigma_post = Sigma_prior_inv
            self.mu_post = mu_prior

            # update the probabilities
            updated_posterior[parents] = self.unnormalized_posterior(
                y, x_vec, self._prior_probabilities[parents]
            )

            # update for the Bayesian Linear Regression
            self.Sigma_post = np.linalg.inv(
                Sigma_prior_inv + 1 / (self.sigma_y**2) * np.outer(x_vec, x_vec)
            )
            self.mu_post = self.Sigma_post @ (
                Sigma_prior_inv @ mu_prior + 1 / (self.sigma_y**2) * y * x_vec
            )

            self.Sigma_dict[parents] = self.Sigma_post
            self.mu_dict[parents] = self.mu_post
            total_prob += updated_posterior[parents]

        # Robust conversion to scalar probabilities
        normalized_probabilities = {}
        for parents in self._prior_probabilities:
            prob_value = updated_posterior[parents] / total_prob
            # Convert to numpy array first for consistent handling
            prob_array = np.asarray(prob_value)
            
            # Extract scalar value
            if prob_array.size == 1:
                # Single element - extract it
                normalized_probabilities[parents] = float(prob_array.flat[0])
            elif prob_array.size == 0:
                # Empty array - use 0
                normalized_probabilities[parents] = 0.0
            else:
                # Multi-element array - take mean
                normalized_probabilities[parents] = float(np.mean(prob_array))
        
        self._prior_probabilities = normalized_probabilities
        self.calculate_metrics()

    def set_data(self, D_O_scaled: Dict[str, np.ndarray]):
        X_dict = {}
        Sigma_dict = {}
        mu_dict = {}
        self.y_obs = D_O_scaled[self.graph.target].reshape(-1)
        self.N = len(self.y_obs)
        for parents in self._prior_probabilities:
            X_dict[parents] = np.vstack(
                [D_O_scaled[key].reshape(1, -1) for key in parents]
            )
            p = len(parents)
            A = (X_dict[parents] @ X_dict[parents].T) / (self.sigma_y**2) + np.eye(
                p
            ) / (self.sigma_theta**2)
            A_inv = np.linalg.inv(A)
            mu_dict[parents] = (
                1 / (self.sigma_y**2) * A_inv @ X_dict[parents] @ self.y_obs
            )
            Sigma_dict[parents] = A_inv

        self.X_dict = X_dict.copy()
        self.Sigma_dict = Sigma_dict
        self.mu_dict = mu_dict

    def add_data(self, D_I: Dict[str, np.ndarray]):
        X_dict = {}
        self.y_obs = np.hstack([self.y_obs, D_I[self.graph.target].reshape(-1)])
        for parents in self._prior_probabilities:
            if len(parents) == 0:
                # For empty parent sets, X_dict should remain empty
                X_dict[parents] = self.X_dict[parents]
            else:
                # Stack parent values as a column vector
                X_key = np.array([D_I[key] for key in parents]).reshape(-1, 1)
                X_temp = np.hstack([self.X_dict[parents], X_key])
                X_dict[parents] = X_temp

        self.X_dict = X_dict.copy()

    def log_unnormalized_posterior(self, y: float, x: np.ndarray, pg: float):
        p = len(x)
        N = self.N
        log_part1 = (
            np.log(pg)
            - 1 / 2 * np.log(2 * np.pi * self.sigma_y**2)
            - 1 / 2 * np.linalg.slogdet(self.Sigma_post)[1]
        )

        Sigma_post_inv = np.linalg.inv(self.Sigma_post)
        Sigma_use_inv = Sigma_post_inv + np.outer(x, x) / (self.sigma_y**2)
        Sigma_use = np.linalg.inv(Sigma_use_inv)
        mu_use = (
            (y * x) / (self.sigma_y**2) + self.mu_post @ Sigma_post_inv
        ) @ Sigma_use

        log_part2 = (
            1 / 2 * mu_use.T @ Sigma_use_inv @ mu_use
            - y**2 / (2 * self.sigma_y**2)
            - 1 / 2 * self.mu_post.T @ Sigma_post_inv @ self.mu_post
        )
        log_part3 = 1 / 2 * np.linalg.slogdet(Sigma_use)[1]
        return log_part1 + log_part2 + log_part3

    def unnormalized_posterior(self, y: float, x: np.ndarray, pg: float):
        log_posterior = self.log_unnormalized_posterior(y, x, pg)
        return np.exp(log_posterior)


class NonLinearSCMModel(SCMModel):
    # extending the class so that it works with Gaussian Processes
    def __init__(
        self,
        prior_probabilities: Dict[Tuple, float],
        graph: GraphStructure,
        sigma_y: float = 1.0,
        sigma_theta: float = 1.0,
        D: int = 100,  # this is the dimension we are projecting towards
    ):
        self._prior_probabilities = prior_probabilities
        self.graph = graph
        self.sigma_y = sigma_y
        self.sigma_theta = sigma_theta
        self.D = D
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.f1_score = []

        # setting up the fourier series
        self.fourier_series: Dict[Tuple, RandomFourierFeatures] = {}
        for parents in self._prior_probabilities:
            input_dim = len(parents)
            fourier_series = RandomFourierFeatures(input_dim, self.D)
            self.fourier_series[parents] = fourier_series

    def set_data(self, D_O_scaled: Dict[str, np.ndarray]):
        X_dict = {}
        Sigma_dict = {}
        mu_dict = {}
        self.y_obs = D_O_scaled[self.graph.target].reshape(-1)
        for parents in self._prior_probabilities:
            X_temp = np.vstack([D_O_scaled[key].reshape(1, -1) for key in parents]).T
            # Transform the data as if we are using a radial basis kernel
            X_dict[parents] = self.fourier_series[parents].transform(X_temp).T
            p = self.D
            A = (X_dict[parents] @ X_dict[parents].T) / (self.sigma_y**2) + np.eye(
                p
            ) / (self.sigma_theta**2)
            A_inv = np.linalg.inv(A)
            mu_dict[parents] = (
                1 / (self.sigma_y**2) * A_inv @ X_dict[parents] @ self.y_obs
            )
            Sigma_dict[parents] = A_inv

        self.X_dict_transform = X_dict
        self.Sigma_dict = Sigma_dict
        self.mu_dict = mu_dict

    def add_data(self, D_I: Dict[str, np.ndarray]):
        X_dict = {}
        self.y_obs = np.hstack([self.y_obs, D_I[self.graph.target].reshape(-1)])
        for parents in self._prior_probabilities:
            X_key = np.array([D_I[key] for key in parents]).T
            X_key_transform = self.fourier_series[parents].transform(X_key).T
            X_temp = np.hstack([self.X_dict_transform[parents], X_key_transform])
            X_dict[parents] = X_temp

        self.X_dict_transform = X_dict.copy()

    def update_all(self, x_dict: Dict[str, float], y: float):
        # suppose we observed a new sample, update all probabilities
        updated_posterior = {}
        total_prob = 0
        for parents in self._prior_probabilities:
            x_vec = np.array([x_dict[parent] for parent in parents])
            x_vec = self.fourier_series[parents].transform(x_vec)

            # start with the prior Gaussian
            Sigma_prior_inv = np.linalg.inv(self.Sigma_dict[parents])
            mu_prior = self.mu_dict[parents]
            # N = self.X_dict_transform[parents].shape[1]
            self.Sigma_post = np.linalg.inv(
                Sigma_prior_inv + 1 / (self.sigma_y**2) * np.outer(x_vec, x_vec)
            )
            self.mu_post = self.Sigma_post @ (
                Sigma_prior_inv @ mu_prior + 1 / (self.sigma_y**2) * y * x_vec
            )

            unnormalized_prob = self.unnormalized_posterior(
                y, x_vec, self._prior_probabilities[parents]
            )
            if not np.isnan(unnormalized_prob):
                updated_posterior[parents] = unnormalized_prob

                self.Sigma_dict[parents] = self.Sigma_post
                self.mu_dict[parents] = self.mu_post
                total_prob += updated_posterior[parents]

        # Robust conversion to scalar probabilities
        normalized_probabilities = {}
        for parents in self._prior_probabilities:
            prob_value = updated_posterior[parents] / total_prob
            # Convert to numpy array first for consistent handling
            prob_array = np.asarray(prob_value)
            
            # Extract scalar value
            if prob_array.size == 1:
                # Single element - extract it
                normalized_probabilities[parents] = float(prob_array.flat[0])
            elif prob_array.size == 0:
                # Empty array - use 0
                normalized_probabilities[parents] = 0.0
            else:
                # Multi-element array - take mean
                normalized_probabilities[parents] = float(np.mean(prob_array))
        
        self._prior_probabilities = normalized_probabilities
        self.calculate_metrics()

    def log_unnormalized_posterior(self, y: float, x: np.ndarray, pg: float):
        p = len(x)
        log_part1 = (
            np.log(pg)
            - 1 / 2 * np.log(2 * np.pi * self.sigma_y**2)
            - 1 / 2 * np.linalg.slogdet(self.Sigma_post)[1]
        )

        Sigma_post_inv = np.linalg.inv(self.Sigma_post)
        Sigma_use_inv = Sigma_post_inv + np.outer(x, x) / (self.sigma_y**2)
        Sigma_use = np.linalg.inv(Sigma_use_inv)
        mu_use = (
            (y * x) / (self.sigma_y**2) + self.mu_post @ Sigma_post_inv
        ) @ Sigma_use

        log_part2 = (
            1 / 2 * mu_use.T @ Sigma_use_inv @ mu_use
            - y**2 / (2 * self.sigma_y**2)
            - 1 / 2 * self.mu_post.T @ Sigma_post_inv @ self.mu_post
        )
        log_part3 = 1 / 2 * np.linalg.slogdet(Sigma_use)[1]
        return log_part1 + log_part2 + log_part3

    def unnormalized_posterior(self, y: float, x: np.ndarray, pg: float):
        log_posterior = self.log_unnormalized_posterior(y, x, pg)
        return np.exp(log_posterior)