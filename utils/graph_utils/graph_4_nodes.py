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


class Graph4Nodes(GraphStructure):
    """
    This is my own synthetic toy graph
    """

    def __init__(
        self,
        X: np.ndarray = None,
        A: np.ndarray = None,
        Z: np.ndarray = None,
        Y: np.ndarray = None,
    ):
        self.B = B
        self.T = T
        self.L = L
        self.R = R
        self.Y = Y
        self._SEM = self.define_SEM()
        self.edges = []
        self._G = self.make_graphical_model()
        self._target = "Y"
        self.functions: Optional[Dict[str, GPRegression]] = None

    def define_SEM(self):
        pass

    def fit_all_models(self):
        return super().fit_all_models()

    def refit_models(self, observational_samples):
        return super().refit_models(observational_samples)

    def get_all_do(self):
        return super().get_all_do()

    def make_graphical_model(self):
        return super().make_graphical_model()

    def show_graphical_model(self):
        return super().show_graphical_model()

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
