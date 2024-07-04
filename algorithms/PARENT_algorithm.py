import logging
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import KFold

from algorithms.BASE_algorithm import BASE
from graphs.graph import GraphStructure


class PARENT(BASE):
    """
    This is the class of my developed methodology
    """

    def __init__(self, graph: GraphStructure):
        self.graph = graph
        self.num_nodes = len(self.graph.variables)
        self.variables = self.graph.variables
        self.target = self.graph.target

    def set_values(self, D_O: Dict, D_I: Dict):
        self.D_O = D_O
        self.D_I = D_I

    def run_algorithm(self):
        parents_Y = corth_features(
            self.D_O, self.target, regression_technique="Random Forest"
        )
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
            print(
                Z_train.shape,
                Z_test.shape,
                D_train.shape,
                D_test.shape,
                Y_train.shape,
                Y_test.shape,
            )

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
