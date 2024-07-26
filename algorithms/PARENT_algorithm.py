import os
import uuid
from collections import namedtuple
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

from algorithms.BASE_algorithm import BASE
from diffcbed.envs.causal_environment import CausalEnvironment
from diffcbed.models.posterior_model import PosteriorModel
from diffcbed.replay_buffer import ReplayBuffer
from diffcbed.strategies.acquisition_strategy import AcquisitionStrategy
from graphs.graph import GraphStructure

# from posterior_model.doubly_robust import DoublyRobustClassWrapper
from posterior_model.model import DoublyRobustModel
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
        self.topological_order = list(self.D_O_bo_format.keys())
        self.D_O = change_obs_data_format_to_mi(
            D_O,
            graph_variables=self.variables,
            intervention_node=np.zeros(shape=len(self.variables)),
        )
        self.posterior_model.covariance_matrix = np.cov(self.D_O.samples.T)
        self.D_I = change_int_data_format_to_mi(D_I, graph_variables=self.variables)
        # just using the observational data for now
        self.buffer.update(self.D_O)
        for intervention in self.D_I:
            self.buffer.update(intervention)

        # self.posterior_model.update(self.buffer.data())

    def fit_parents_to_target(self):
        # this is the number of priors that the doubly robust method picked up
        num_priors = len(self.model.prob_estimate.keys())
        self.all_functions = {}
        for parents in self.model.prob_estimate.keys():
            if len(parents) > 0:
                X = np.hstack([self.D_O_bo_format[parent] for parent in parents])
                y = np.array(self.D_O_bo_format[self.target]).reshape(-1, 1)
                kernel = RBF(
                    input_dim=len(parents), variance=1.0, ARD=False, lengthscale=1.0
                )
                gp = GPRegression(X=X, Y=y, kernel=kernel)

                gp.optimize()
            else:
                # you can fit the KDE estimate here as well
                pass
            self.all_functions[parents] = gp

        print(self.all_functions)

    def run_algorithm(self, T: int = 10, python_code: bool = True):
        # parents_Y = corth_features(
        #     self.D_O_bo_format, self.target, regression_technique="Random Forest"
        # )

        target = self.graph.target
        # define the model as well
        self.model = DoublyRobustModel(
            self.graph, self.topological_order, target, num_bootstraps=30
        )

        parents = self.graph.parents[self.graph.target]
        groundtruth = np.zeros(shape=len(self.graph.variables) - 1)
        for i, var in enumerate(self.topological_order):
            if var != self.graph.target and var in parents:
                groundtruth[i] = 1
        groundtruth = pd.Series(groundtruth.astype(bool))

        # need to change this to incorporate the interventional data as well
        data: Data = self.buffer.data()
        # data = pd.DataFrame(self.D_O.samples, columns=self.topological_order)
        # data_conf = {}

        # # this part is for the python doubly robust estimator
        if python_code:
            self.model.run_method(data)
            self.fit_parents_to_target()
        else:
            run_doubly_robust(data, self.topological_order, target)

    def check_dr_parent_accuracy(self):
        target = self.graph.target
        # just running it with one bootstrap
        self.model = DoublyRobustModel(
            self.graph, self.topological_order, target, num_bootstraps=1
        )

        parents = self.graph.parents[self.graph.target]
        groundtruth = np.zeros(shape=len(self.graph.variables) - 1)
        for i, var in enumerate(self.topological_order):
            if var != self.graph.target and var in parents:
                groundtruth[i] = 1
        groundtruth = pd.Series(groundtruth.astype(bool))
        data: Data = self.buffer.data()
        parents = self.model.run_method(data)
        return np.mean(parents.to_numpy() == groundtruth.to_numpy())


def run_doubly_robust(data: pd.DataFrame, topological_order: List, target: str):

    os.chdir("/vol/bitbucket/jd123/causal_bayes_opt/")
    target_index = topological_order.index(target)
    X = data.iloc[:, [i for i in range(data.shape[1]) if i != target_index]].to_numpy()
    # T_ones = np.ones((len(X), 1))
    # X = np.hstack((T_ones, X))
    y = data.iloc[:, target_index].to_numpy()
    df_X = pd.DataFrame(X)
    df_y = pd.DataFrame(y)
    uid = str(uuid.uuid4())

    tmp_path = "tmp"
    os.makedirs(tmp_path, exist_ok=True)
    dags_path = os.path.join(tmp_path, "dags/")
    os.makedirs(dags_path, exist_ok=True)
    df_X.to_csv(f"{dags_path}/X.csv", index=False)
    df_y.to_csv(f"{dags_path}/y.csv", index=False)

    rfile = os.path.join("posterior_model", "corth_algorithm.R")
    r_command = f"Rscript {rfile}"
    os.system(r_command)

    # assert (
    #     data.nodes.dtype == bool
    # ), "Please input boolean mask for interventional samples"
    # interventions = np.array(data.nodes)

    # is_single_target = (interventions.sum((-1)) <= 1).sum() == interventions.shape[0]
    # idx = np.array(range(len(interventions)))
    # if is_single_target:
    #     if group_interventions:
    #         idx = np.argsort(interventions)
    # # order samples by interventions to group similar interventions together
    # interventions = interventions[idx]
    # for i in tqdm.tqdm(range(n_boot)):
    #     data_indices, unique_targets, target_indices = get_bootstrap_indices(
    #         interventions, is_single_target, maintain_int_dist=maintain_int_dist
    #     )
    #     np.savetxt(tmp_path / "samples.csv", data.samples[data_indices], delimiter=" ")
    #     with open(tmp_path / "unique_targets.csv", "w", newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerows(unique_targets)
    #     np.array(target_indices).tofile(
    #         tmp_path / "target_indices.csv", sep="\n", format="%d"
    #     )
    #     open(tmp_path / "target_indices.csv", "a").write("\n")
    #     if not os.path.exists(dags_path):
    #         os.mkdir(dags_path)
    #     rfile = os.path.join("diffcbed", "models", "dag_bootstrap_lib", "run_gies.r")
    #     r_command = "Rscript {} {} {} {} {} {}".format(
    #         rfile,
    #         str(tmp_path / "samples.csv"),
    #         str(tmp_path / "unique_targets.csv"),
    #         str(tmp_path / "target_indices.csv"),
    #         dags_path,
    #         i,
    #     )

    #     os.system(r_command)
    # return tmp_path


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
