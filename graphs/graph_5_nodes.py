import logging
from collections import OrderedDict
from itertools import chain
from typing import Dict, List, Optional, OrderedDict, Tuple

import numpy as np
from GPy.models.gp_regression import GPRegression
from scipy.special import expit

from graphs.graph import GraphStructure


class Graph5Nodes(GraphStructure):
    """
    This is the graph in the causal entropy optimization algorithm
    """

    def __init__(
        self,
        B: np.ndarray = None,
        T: np.ndarray = None,
        L: np.ndarray = None,
        R: np.ndarray = None,
        Y: np.ndarray = None,
    ):
        self.B = B
        self.T = T
        self.L = L
        self.R = R
        self.Y = Y
        self._SEM = self.define_SEM()
        self._edges = [
            ("B", "L"),
            ("B", "Y"),
            ("T", "L"),
            ("T", "R"),
            ("T", "Y"),
            ("L", "R"),
            ("R", "Y"),
        ]
        self._G = self.make_graphical_model()
        self._nodes = set(chain(*self.edges))
        self._parents, self._children = self.build_relationships()
        self._target = "Y"
        self._functions: Optional[Dict[str, GPRegression]] = None
        self._variables = ["B", "T", "L", "R", "Y"]

    def define_SEM(self):
        fb = lambda epsilon, sample: epsilon
        ft = lambda epsilon, sample: epsilon
        fl = lambda epsilon, sample: expit(0.5 * sample["T"] + sample["B"])
        fr = lambda epsilon, sample: 4 + sample["L"] * sample["T"]
        fy = (
            lambda epsilon, sample: 0.5
            + np.cos(4 * sample["T"])
            + np.sin(-sample["L"] + 2 * sample["R"])
            + sample["B"]
            + epsilon
        )

        graph = OrderedDict([("B", fb), ("T", ft), ("L", fl), ("R", fr), ("Y", fy)])
        return graph

    def fit_all_models(self):
        return super().fit_all_models()

    def refit_models(self, observational_samples):
        return super().refit_models(observational_samples)

    def get_interventional_range(self):
        min_intervention_t = 4
        max_intervention_t = 8

        # XXX This was not checked yet
        min_intervention_r = -5
        max_intervention_r = 5

        dict_ranges = OrderedDict(
            [
                ("T", [min_intervention_t, max_intervention_t]),
                ("R", [min_intervention_r, max_intervention_r]),
            ]
        )
        return dict_ranges

    def get_cost_structure(self, type_cost: int) -> OrderedDict:
        return super().get_cost_structure(type_cost)

    def get_sets(self):
        mis = [["T"], ["R"]]
        pomis = [["T", "R"]]
        manipulative_variables = ["T", "R"]
        return mis, pomis, manipulative_variables

    def get_fixed_equal_costs(self) -> OrderedDict:
        logging.info("Using the fixed equal cost structure")
        cost_fix_T_equal = lambda intervention_value: 1.0
        cost_fix_R_equal = lambda intervention_value: 1.0
        costs = OrderedDict([("T", cost_fix_T_equal), ("R", cost_fix_R_equal)])
        return costs

    def get_fixed_different_costs(self) -> OrderedDict:
        logging.info("Using the fixed different cost structure")
        cost_fix_T_different = lambda intervention_value: 1.0
        cost_fix_R_different = lambda intervention_value: 10.0
        costs = OrderedDict([("T", cost_fix_T_different), ("R", cost_fix_R_different)])
        return costs

    def get_variable_equal_costs(self) -> OrderedDict:
        logging.info("Using the variable equal cost structure")
        cost_variable_T_equal = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 1.0
        )
        cost_variable_R_equal = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 1.0
        )
        costs = OrderedDict(
            [("T", cost_variable_T_equal), ("R", cost_variable_R_equal)]
        )
        return costs

    def get_variable_different_costs(self) -> OrderedDict:
        logging.info("Using the variable different cost structure")
        cost_variable_T_different = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 1.0
        )
        cost_variable_R_different = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 10.0
        )
        costs = OrderedDict(
            [("T", cost_variable_T_different), ("R", cost_variable_R_different)]
        )
        return costs

    def get_interventional_domain(self):
        return super().get_interventional_domain()

    def get_set_BO(self):
        return super().get_set_BO()

    def get_error_distribution(self):
        err_dist = {}
        err_dist["B"] = np.random.uniform(-1, 1)
        err_dist["T"] = np.random.uniform(4, 8)
        err_dist["L"] = 0
        err_dist["R"] = 0
        err_dist["Y"] = np.random.normal()
        return err_dist

    def get_all_do(self):
        do_dict = {}
        do_dict["compute_do_T"] = self.compute_do_T
        do_dict["compute_do_R"] = self.compute_do_R
        do_dict["compute_do_TR"] = self.compute_do_TR
        return do_dict

    def compute_do_T(self, observational_samples, value):
        interventions_nodes = ["T"]
        mean_do, var_do = self.compute_do(
            observational_samples, value, interventions_nodes
        )

        return mean_do, var_do

    def compute_do_R(self, observational_samples, value):

        interventions_nodes = ["R"]
        mean_do, var_do = self.compute_do(
            observational_samples, value, interventions_nodes
        )

        return mean_do, var_do

    def compute_do_TR(self, observational_samples, value):

        interventions_nodes = ["T", "R"]
        mean_do, var_do = self.compute_do(
            observational_samples, value, interventions_nodes
        )

        return mean_do, var_do
