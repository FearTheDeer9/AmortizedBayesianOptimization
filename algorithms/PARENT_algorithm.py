import logging
from collections import namedtuple
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import KFold

from algorithms.BASE_algorithm import BASE
from diffcbed.envs.causal_environment import CausalEnvironment
from diffcbed.models.posterior_model import PosteriorModel
from diffcbed.replay_buffer import ReplayBuffer
from diffcbed.strategies.acquisition_strategy import AcquisitionStrategy
from graphs.graph import GraphStructure
from utils.sem_sampling import (
    change_int_data_format_to_mi,
    change_obs_data_format_to_mi,
)

Data = namedtuple("Data", ["samples", "intervention_node"])


class PARENT(BASE):
    """
    This is the class of my developed methodology
    """

    def __init__(
        self,
        graph: GraphStructure,
        graph_env: CausalEnvironment,
        posterior_model: PosteriorModel,
        acquisition_strategy: AcquisitionStrategy,
    ):
        self.graph = graph
        self.num_nodes = len(self.graph.variables)
        self.variables = self.graph.variables
        self.target = self.graph.target
        self.acquisition_strategy = acquisition_strategy

        # setting up some more variables
        self.graph_env = graph_env
        self.posterior_model = posterior_model
        self.buffer = ReplayBuffer(binary=True)

    def set_values(self, D_O: Dict, D_I: Dict):
        self.D_O_bo_format = deepcopy(D_O)
        self.D_O = change_obs_data_format_to_mi(
            D_O,
            graph_variables=self.variables,
            intervention_node=np.zeros(shape=len(self.variables)),
        )
        df_D_O = pd.DataFrame(self.D_O.samples)
        self.posterior_model.covariance_matrix = np.cov(self.D_O.samples.T)
        self.D_I = change_int_data_format_to_mi(D_I, graph_variables=self.variables)
        # just using the observational data for now
        self.buffer.update(self.D_O)
        # for intervention in self.D_I:
        #     self.buffer.update(intervention)

        # self.posterior_model.update(self.buffer.data())

        # writing the dataframe to a csv file
        logging.info("Writing to the csv file")
        df_D_O.to_csv(
            "/vol/bitbucket/jd123/causal_bayes_opt/data/test.csv", index=False
        )

    def run_algorithm(self, T: int = 10):
        parents_Y = corth_features(
            self.D_O_bo_format, self.target, regression_technique="Random Forest"
        )

        # for i in range(T):
        #     logging.info(f"------------------EXPERIMENT {i}-------------------")
        #     # just keep it like this for now and don't split it into manipulative and non-manipulative
        #     valid_interventions = self.graph_env.get_valid_interventions()

        #     interventions, _ = self.acquisition_strategy.acquire(valid_interventions, i)

        #     # assuming a batch size of 1
        #     intervention_node = interventions["nodes"][0]
        #     intervention_value = interventions["values"][0]
        #     intervention_results = self.graph_env.intervene(
        #         i, 1000, interventions["nodes"][0], interventions["values"][0]
        #     )
        #     intervention_results = Data(
        #         samples=intervention_results.samples.mean(axis=0).reshape(1, -1),
        #         intervention_node=intervention_results.intervention_node,
        #     )
        #     self.buffer.update(intervention_results)
        print(parents_Y)
        # parents_X = corth_features(
        #     self.D_O,
        #     "X",
        # )
        # print(parents_X)
        # parents_Z = corth_features(
        #     self.D_O,
        #     "Z",
        # )
        # print(parents_Z)


def corth_features(
    D_O: Dict,
    target: str,
    regression_technique: str = "Lasso",
    alpha: float = 0.1,
    K: int = 2,
):
    """
    Orthogonal feature selection approach using cross-fitting with specified regression technique.
    """
    Y = D_O[target]
    X = np.hstack([D_O[var] for var in D_O.keys() if var != target])
    N, d = X.shape
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    theta_hat = np.zeros(d)
    khi_hat = np.zeros(d)
    sigma_squared_hat = np.zeros(d)

    for i in range(d):
        theta_k = []
        khi_k = []
        sigma_k = []
        for train_index, test_index in kf.split(X):
            Z_train = np.delete(X[train_index], i, axis=1)
            Z_test = np.delete(X[test_index], i, axis=1)
            D_train = X[train_index, i]
            D_test = X[test_index, i]
            Y_train = Y[train_index]
            Y_test = Y[test_index]

            # Choose regression technique based on parameter
            if regression_technique == "Lasso":
                model_m = LassoCV(cv=5).fit(Z_train, D_train.reshape(-1))
                model_g = LassoCV(cv=5).fit(Z_train, Y_train.reshape(-1))
            elif regression_technique == "Random Forest":
                model_m = RandomForestRegressor(max_leaf_nodes=10).fit(
                    Z_train, D_train.reshape(-1)
                )
                model_g = RandomForestRegressor(max_leaf_nodes=10).fit(
                    Z_train, Y_train.reshape(-1)
                )
            else:
                raise ValueError(
                    "Unsupported regression technique. Choose 'Lasso' or 'Random Forest'."
                )

            D_hat = model_m.predict(Z_test)
            Y_hat = model_g.predict(Z_test)

            v_k = D_test - D_hat
            u_k = Y_test - Y_hat

            theta = np.sum(v_k * u_k) / np.sum(v_k * D_test)
            khi = np.mean((Y_hat - Y_test) * (D_test - D_hat))
            sigma = np.mean(((Y_hat - Y_test) * (D_test - D_hat) - khi) ** 2)

            theta_k.append(theta)
            khi_k.append(khi)
            sigma_k.append(sigma)

        theta_hat[i] = np.mean(theta_k)
        khi_hat[i] = np.mean(khi_k)
        sigma_squared_hat[i] = np.mean(sigma_k)

    # Calculate p-values and apply Bonferroni correction
    decision_vector = np.zeros(d, dtype=bool)
    for i in range(d):
        test_statistic = np.abs(khi_hat[i]) / np.sqrt(sigma_squared_hat[i] / N)
        p_value = 2 * norm.sf(test_statistic)
        # Apply Bonferroni correction: Adjust alpha based on the number of tests
        bonferroni_alpha = alpha / d
        print(test_statistic, p_value, bonferroni_alpha)
        decision_vector[i] = p_value < bonferroni_alpha

    return decision_vector
