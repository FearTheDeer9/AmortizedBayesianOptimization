import logging
from collections import OrderedDict
from typing import Dict, List, Optional, OrderedDict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from emukit.core import ContinuousParameter, ParameterSpace
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression
from pgmpy.models import BayesianNetwork

from utils.graph_utils.graph import GraphStructure


class Graph6Nodes(GraphStructure):
    """
    This is the graph for the healthcare experiment
    """

    def __init__(
        self,
        A: np.ndarray = None,
        B: np.ndarray = None,
        C: np.ndarray = None,
        S: np.ndarray = None,
        As: np.ndarray = None,
        Y: np.ndarray = None,
    ):
        self.A = A
        self.B = B
        self.C = C
        self.S = S
        self.As = As
        self.Y = Y
        self._SEM = self.define_SEM()
        self.edges = [
            ("A", "C"),
            ("A", "S"),
            ("A", "B"),
            ("B", "C"),
            ("B", "As"),
            ("As", "C"),
            ("As", "Y"),
            ("S", "C"),
            ("S", "Y"),
            ("A", "Y"),
            ("A", "As"),
            ("C", "Y"),
        ]
        self.G = self.make_graphical_model()
        self._target = "Y"
        self.functions: Optional[Dict[str, GPRegression]] = None

    def define_SEM(self):
        fa = lambda epsilon, sample: epsilon
        fb = lambda epsilon, sample: 27 - 0.01 * sample["A"] + epsilon
        fas = lambda epsilon, sample: 1 / (
            1 + np.exp(-0.8 + 0.1 * sample["A"] + 0.03 * sample["B"])
        )
        fs = lambda epsilon, sample: 1 / (
            1 + np.exp(-13 + 0.1 * sample["A"] + 0.2 * sample["B"])
        )
        fc = lambda epsilon, sample: 1 / (
            1
            + np.exp(
                2.2
                - 0.05 * sample["A"]
                + 0.01 * sample["B"]
                - 0.04 * sample["S"]
                + 0.02 * sample["As"]
            )
        )
        fy = (
            lambda epsilon, sample: 6.8
            + 0.04 * sample["A"]
            - 0.15 * sample["B"]
            - 0.6 * sample["S"]
            + 0.55 * sample["As"]
            + sample["C"]
            + epsilon
        )
        graph = OrderedDict(
            [("A", fa), ("B", fb), ("As", fas), ("S", fs), ("C", fc), ("Y", fy)]
        )
        return graph

    def fit_all_models(self):
        return super().fit_all_models()

    def refit_models(self, observational_samples):
        return super().refit_models(observational_samples)

    def get_all_do(self):
        return super().get_all_do()

    def get_variables(self):
        return super().get_variables()

    def get_interventional_range(self):
        return super().get_interventional_range()

    def get_parameter_space(self, exploration_set):
        return super().get_parameter_space(exploration_set)

    def get_cost_structure(self, type_cost: int) -> OrderedDict:
        return super().get_cost_structure(type_cost)

    def get_sets(self):
        return super().get_sets()

    def get_variable_different_costs(self):
        return super().get_variable_different_costs()

    def get_fixed_different_costs(self):
        return super().get_fixed_different_costs()

    def get_fixed_equal_costs(self):
        return super().get_fixed_equal_costs()

    def get_variable_equal_costs(self):
        return super().get_variable_equal_costs()

    def get_interventional_domain(self):
        return super().get_interventional_domain()

    def get_set_BO(self):
        return super().get_set_BO()
