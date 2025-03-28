import logging
from collections import OrderedDict
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple

import numpy as np

from graphs.graph import GraphStructure


class Graph10Nodes(GraphStructure):
    """
    A 10-node graph for testing the ACD algorithm.
    This graph has a more complex structure than the training graphs.
    """

    def __init__(
        self,
        X: np.ndarray = None,
        Z: np.ndarray = None,
        T: np.ndarray = None,
        A: np.ndarray = None,
        B: np.ndarray = None,
        C: np.ndarray = None,
        D: np.ndarray = None,
        E: np.ndarray = None,
        F: np.ndarray = None,
        Y: np.ndarray = None,
    ):
        self.X = X
        self.Z = Z
        self.T = T
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.F = F
        self.Y = Y
        self._variables = ["X", "Z", "T", "A", "B", "C", "D", "E", "F", "Y"]
        self._SEM = self.define_SEM()
        # Define a more complex graph structure
        self._edges = [
            ("X", "A"), ("X", "B"), ("X", "Z"),
            ("Z", "B"), ("Z", "C"), ("Z", "Y"),
            ("T", "A"), ("T", "C"), ("T", "D"),
            ("A", "E"), ("A", "F"),
            ("B", "D"), ("B", "E"),
            ("C", "F"), ("C", "Y"),
            ("D", "Y"),
            ("E", "F"),
            ("F", "Y"),
        ]
        self._target = "Y"
        self._functions: Optional[Dict[str, Callable]] = None
        self._nodes = set(chain(*self.edges))
        self._parents, self._children = self.build_relationships()
        self._G = self.make_graphical_model()
        self._standardised = False
        self.use_intervention_range_data = False

    def define_SEM(self):
        def fx(epsilon, sample): return epsilon
        def fz(epsilon, sample): return 1.5 * \
            np.tanh(sample.get("X", 0)) + epsilon

        def ft(epsilon, sample): return epsilon

        def fa(epsilon, sample):
            return 0.8 * sample.get("X", 0) + 0.6 * sample.get("T", 0) + epsilon

        def fb(epsilon, sample):
            return 0.7 * sample.get("X", 0) + 0.5 * sample.get("Z", 0) + epsilon

        def fc(epsilon, sample):
            return 0.9 * sample.get("Z", 0) + 0.4 * sample.get("T", 0) + epsilon

        def fd(epsilon, sample):
            return 0.6 * sample.get("T", 0) + 0.5 * sample.get("B", 0) + epsilon

        def fe(epsilon, sample):
            return 0.7 * sample.get("A", 0) + 0.8 * sample.get("B", 0) + epsilon

        def ff(epsilon, sample):
            return 0.6 * sample.get("A", 0) + 0.7 * sample.get("C", 0) + 0.4 * sample.get("E", 0) + epsilon

        def fy(epsilon, sample):
            return (
                0.8 * sample.get("Z", 0) +
                0.7 * sample.get("C", 0) +
                0.6 * sample.get("D", 0) +
                0.9 * sample.get("F", 0) +
                epsilon
            )

        graph = OrderedDict([
            ("X", fx), ("Z", fz), ("T", ft),
            ("A", fa), ("B", fb), ("C", fc),
            ("D", fd), ("E", fe), ("F", ff),
            ("Y", fy)
        ])
        return graph

    def get_all_do(self):
        return super().get_all_do()

    def get_interventional_range(self):
        # Define intervention ranges for all variables
        ranges = {}
        for var in self.variables:
            if var != self.target:
                ranges[var] = (-2.0, 2.0)
        return ranges

    def get_exploration_set(self) -> List[Tuple[str]]:
        return [(var,) for var in self.variables if var != self.target]

    def get_sets(self):
        mis = []
        pomis = []
        manipulative_variables = [
            var for var in self.variables if var != self.target]
        return mis, pomis, manipulative_variables
