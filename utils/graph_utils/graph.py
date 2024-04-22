import abc
import logging
from collections import OrderedDict, defaultdict
from typing import OrderedDict

import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.models import BayesianNetwork


class GraphStructure:
    """
    This is the generic graph structure that all the simulated examples will follow
    """

    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def SEM(self):
        """
        Specifies the type of graph (e.g., 'directed', 'undirected')
        """
        return self._SEM

    @property
    @abc.abstractmethod
    def target(self):
        return self._target

    @abc.abstractmethod
    def define_SEM():
        """
        This method defines the structural equation model (SEM) for
        each of the simulated graph structures
        """
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def fit_all_models(self):
        """
        Fit the models based on the initial data
        """
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def refit_models(self, observational_samples):
        """
        Refit the GP models based on the new observational samples
        """
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def get_all_do(self):
        """
        This assigns assigns the calculation of the interventional distribution
        to each of the variables in the SCM
        """
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def get_interventional_range(self):
        raise NotImplementedError("Subclass should implement this")

    @abc.abstractmethod
    def get_set_BO(self):
        raise NotImplementedError("Subclass should implement this")

    @abc.abstractmethod
    def get_sets(self):
        raise NotImplementedError("Subclass should implement this")

    @abc.abstractmethod
    def get_interventional_domain(self):
        raise NotImplementedError("Subclass should implement this")

    @abc.abstractmethod
    def get_cost_structure(self, type_cost: int) -> OrderedDict:
        if type_cost == 1:
            costs = self.get_fixed_equal_costs()
        elif type_cost == 2:
            costs = self.get_fixed_different_costs()
        elif type_cost == 3:
            costs = self.get_variable_equal_costs()
        elif type_cost == 4:
            costs = self.get_variable_different_costs()
        else:
            logging.warning("Undefined cost structure")

        assert isinstance(costs, OrderedDict)
        return costs

    @abc.abstractmethod
    def get_fixed_equal_costs(self):
        raise NotImplementedError("Subclass should implement this")

    @abc.abstractmethod
    def get_fixed_different_costs(self):
        raise NotImplementedError("Subclass should implement this")

    @abc.abstractmethod
    def get_variable_equal_costs(self):
        raise NotImplementedError("Subclass should implement this")

    @abc.abstractmethod
    def get_variable_different_costs(self):
        raise NotImplementedError("Subclass should implement this")

    @abc.abstractmethod
    def get_parameter_space(self, exploration_set):
        raise NotImplementedError("Subclass should implement this")

    @abc.abstractmethod
    def get_variables(self):
        raise NotImplementedError("Subclass should implement this")

    @abc.abstractmethod
    def make_graphical_model(self):
        model = BayesianNetwork(self.edges)

        # Manually creating a networkx graph from the BayesianModel
        G = nx.MultiDiGraph()
        G.add_edges_from(model.edges())
        return G

    @abc.abstractmethod
    def show_graphical_model(self):
        pos = nx.spring_layout(self.G)  # positions for all nodes
        nx.draw(
            self.G,
            pos,
            with_labels=True,
            node_size=700,
            node_color="lightblue",
            font_size=10,
            font_weight="bold",
        )
        plt.title("Bayesian Network")
        plt.show()


class DAGModel:
    """
    This just creates a graphical model to leverage the structure of the graph
    """

    def __init__(self, edges):
        self.edges = edges
        self.functions = {}

    def parse_edges(self):

        graph = defaultdict(list)
        nodes = set()
        for parent, child in self.edges:
            graph[child].append(parent)
            nodes.update([parent, child])
        return graph, nodes
