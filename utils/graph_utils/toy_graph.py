import logging
import sys
from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from emukit.core import ContinuousParameter, ParameterSpace
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression
from pgmpy.models import BayesianNetwork

from utils.graph_utils.graph import GraphStructure

sys.path.append("..")


class ToyGraph(GraphStructure):
    """
    This is the class for the TOY graph structure used in the CBO
    paper
    """

    def __init__(
        self, X: np.ndarray = None, Z: np.ndarray = None, Y: np.ndarray = None
    ):
        logging.info("Initializing the Toy Graph Structures")
        self.X = X
        self.Z = Z
        self.Y = Y
        self._SEM = self.define_SEM()
        self.edges = [("X", "Z"), ("Z", "Y")]
        self.G = self.make_graphical_model()
        self._target = "Y"
        self.functions = None

    def SEM(self) -> OrderedDict:
        return self._SEM

    def define_SEM(self) -> OrderedDict:
        logging.info("Setting up the structural equation model for the ToyGraph")
        fx = lambda epsilon, sample: epsilon
        fz = lambda epsilon, sample: np.exp(-sample["X"]) + epsilon
        fy = (
            lambda epsilon, sample: np.cos(sample["Z"])
            - np.exp(-sample["Z"] / 20)
            + epsilon
        )

        graph = OrderedDict([("X", fx), ("Z", fz), ("Y", fy)])
        return graph

    def set_data(self, X: np.ndarray, Z: np.ndarray, Y: np.ndarray):
        """
        setting the data for the first time
        """
        self.X = X
        self.Z = Z
        self.Y = Y

    def fit_all_models(self) -> None:
        """
        Fit the model based on the original data
        """
        logging.info("Fitting the Gaussian Processes based on the original data")
        assert self.X is not None

        # regress Y on Z
        num_features = self.Z.shape[1]
        kernel = RBF(num_features, ARD=False, lengthscale=1.0, variance=1.0)
        gp_Y = GPRegression(X=self.Z, Y=self.Y, kernel=kernel, noise_var=1.0)
        gp_Y.optimize()

        # regress Z on Y
        num_features = self.X.shape[1]
        kernel = RBF(num_features, ARD=False, lengthscale=1.0, variance=1.0)
        gp_Z = GPRegression(X=self.X, Y=self.Z, kernel=kernel)
        gp_Z.optimize()

        # there is no variable to regress X on
        self.functions = OrderedDict([("Y", gp_Y), ("Z", gp_Z), ("X", [])])

    def refit_models(self, observational_samples: dict) -> None:
        """
        Refit the gaussian processes based on the observed samples
        """
        logging.info("Fitting the Gaussian Processes based on the new data")
        X = np.asarray(observational_samples["X"])
        Z = np.asarray(observational_samples["Z"])
        Y = np.asarray(observational_samples["Y"])

        functions = {}

        num_features = Z.shape[1]
        kernel = RBF(num_features, ARD=False, lengthscale=1.0, variance=1.0)
        gp_Y = GPRegression(X=Z, Y=Y, kernel=kernel, noise_var=1.0)
        gp_Y.optimize()

        # should it not still be x on z
        num_features = X.shape[1]
        kernel = RBF(num_features, ARD=False, lengthscale=1.0, variance=1.0)
        gp_Z = GPRegression(X=X, Y=Z, kernel=kernel)
        gp_Z.optimize()

        self.functions = OrderedDict([("Y", gp_Y), ("Z", gp_Z), ("X", [])])

    def get_interventional_range(self) -> OrderedDict:
        logging.info("Getting the inverventional range for the ToyGraph")
        min_intervention_x = -5
        max_intervention_x = 5

        min_intervention_z = -5
        max_intervention_z = 20

        dict_ranges = OrderedDict(
            [
                ("X", [min_intervention_x, max_intervention_x]),
                ("Z", [min_intervention_z, max_intervention_z]),
            ]
        )
        return dict_ranges

    def get_parameter_space(self, exploration_set: List) -> ParameterSpace:
        interventional_range = self.get_interventional_range()
        space = {
            "X": ContinuousParameter(
                "X", interventional_range["X"][0], interventional_range["X"][1]
            ),
            "Z": ContinuousParameter(
                "Z", interventional_range["Z"][0], interventional_range["Z"][1]
            ),
        }

        es_space = []
        for var in exploration_set:
            es_space.append(space[var])
        return ParameterSpace(es_space)

    def get_cost_structure(self, type_cost: int) -> OrderedDict:
        return super().get_cost_structure(type_cost)

    def get_fixed_equal_costs(self) -> OrderedDict:
        logging.info("Using the fixed equal cost structure")
        cost_fix_X_equal = lambda intervention_value: 1.0
        cost_fix_Z_equal = lambda intervention_value: 1.0
        costs = OrderedDict([("X", cost_fix_X_equal), ("Z", cost_fix_Z_equal)])
        return costs

    def get_fixed_different_costs(self) -> OrderedDict:
        logging.info("Using the fixed different cost structure")
        cost_fix_X_different = lambda intervention_value: 1.0
        cost_fix_Z_different = lambda intervention_value: 10.0
        costs = OrderedDict([("X", cost_fix_X_different), ("Z", cost_fix_Z_different)])
        return costs

    def get_variable_equal_costs(self) -> OrderedDict:
        logging.info("Using the variable equal cost structure")
        cost_variable_X_equal = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 1.0
        )
        cost_variable_Z_equal = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 1.0
        )
        costs = OrderedDict(
            [("X", cost_variable_X_equal), ("Z", cost_variable_Z_equal)]
        )
        return costs

    def get_variable_different_costs(self) -> OrderedDict:
        logging.info("Using the variable different cost structure")
        cost_variable_X_different = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 1.0
        )
        cost_variable_Z_different = (
            lambda intervention_value: np.sum(np.abs(intervention_value)) + 10.0
        )
        costs = OrderedDict(
            [("X", cost_variable_X_different), ("Z", cost_variable_Z_different)]
        )
        return costs

    def get_set_BO(self) -> List:
        """
        This is just for the regular BayesOpt algorithm
        """
        logging.info("Getting the variables for the BO algorithm")
        manipulative_variables = ["X", "Z"]
        return manipulative_variables

    def get_sets(self) -> Tuple[List, List, List]:
        """
        This is for the CBO algorithm
        """
        logging.info("Getting the variables (mis and pomis) for the CBO algorithm")
        mis = [["X"], ["Z"]]
        pomis = [["Z"]]
        manipulative_variables = ["X", "Z"]
        return mis, pomis, manipulative_variables

    def get_all_do(self):
        logging.info("Getting the do-functions for the ToyGraph")
        do_dict = {}
        do_dict["compute_do_X"] = self.compute_do_X
        do_dict["compute_do_Z"] = self.compute_do_Z
        do_dict["compute_do_XZ"] = self.compute_do_XZ
        return do_dict

    def get_variables(self):
        return ["X", "Z", "Y"]

    def target(self):
        return self._target

    def make_graphical_model(self):
        # Create the Bayesian Model and add the edges
        model = BayesianNetwork(self.edges)

        # Manually creating a networkx graph from the BayesianModel
        G = nx.MultiDiGraph()
        G.add_edges_from(model.edges())
        return G

    def show_graphical_model(self):
        # Draw the graph
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

    def compute_do_X(self, observational_samples, value):

        # Compute Do effects as in the notebook
        Z = observational_samples["Z"]

        parents_Y = OrderedDict([("Z", Z)])

        functions = OrderedDict(
            [("Y", self.functions["Y"]), ("Z", self.functions["Z"]), ("X", [])]
        )

        children = OrderedDict([("X", OrderedDict([("Z", Z)]))])
        independent_nodes = OrderedDict([("X", OrderedDict([]))])
        parents_nodes = OrderedDict([("X", OrderedDict([]))])

        num_interventions = value.shape[0]
        mean_do = np.zeros((num_interventions, 1))
        var_do = np.zeros((num_interventions, 1))

        for i in range(num_interventions):

            mean_do[i], var_do[i] = causal_effect_DO(
                {"X": value[i]},
                functions=functions,
                parents_Y=parents_Y,
                children=children,
                parents=parents_nodes,
                independent_nodes=independent_nodes,
            )
        return mean_do, var_do

    def compute_do_Z(self, observational_samples, value):

        # Compute Do effects as in the notebook
        Z = observational_samples["Z"]

        parents_Y = OrderedDict([("Z", Z)])

        functions = OrderedDict(
            [("Y", self.functions["Y"]), ("Z", self.functions["Z"]), ("X", [])]
        )

        children = OrderedDict([("X", OrderedDict([("Z", Z)])), ("Z", OrderedDict([]))])
        independent_nodes = OrderedDict(
            [("X", OrderedDict([])), ("Z", OrderedDict([]))]
        )
        parents_nodes = OrderedDict(
            [("X", OrderedDict([])), ("Z", OrderedDict([("X", Z)]))]
        )

        num_interventions = value.shape[0]
        mean_do = np.zeros((num_interventions, 1))
        var_do = np.zeros((num_interventions, 1))

        for i in range(num_interventions):
            mean_do[i], var_do[i] = causal_effect_DO(
                {"Z": value[i]},
                functions=functions,
                parents_Y=parents_Y,
                children=children,
                parents=parents_nodes,
                independent_nodes=independent_nodes,
            )

        return mean_do, var_do

    def compute_do_XZ(self, observational_samples, value):

        # Compute Do effects as in the notebook
        Z = observational_samples["Z"]

        parents_Y = OrderedDict([("Z", Z)])

        functions = OrderedDict(
            [("Y", self.functions["Y"]), ("Z", self.functions["Z"]), ("X", [])]
        )

        children = OrderedDict([("X", OrderedDict([("Z", Z)])), ("Z", OrderedDict([]))])
        independent_nodes = OrderedDict(
            [("X", OrderedDict([])), ("Z", OrderedDict([]))]
        )
        parents_nodes = OrderedDict(
            [("X", OrderedDict([])), ("Z", OrderedDict([("X", Z)]))]
        )

        mean_do, var_do = causal_effect_DO(
            {"X": value[0], "Z": value[1]},
            functions=functions,
            parents_Y=parents_Y,
            children=children,
            parents=parents_nodes,
            independent_nodes=independent_nodes,
        )

        return mean_do, var_do


def causal_effect_DO(
    *interventions,
    functions: OrderedDict,
    parents_Y: OrderedDict,
    children: OrderedDict,
    parents: OrderedDict,
    independent_nodes: OrderedDict
):
    """
    This function was also taken from the original paper
    """
    ## This function can be used to compute the CE of variables that are confounded via Back door adjustment
    ## so that no adjustment is needed

    final_variables = OrderedDict()

    num_models = len(functions)
    num_interventions = len(children)
    num_observations = list(parents_Y.items())[0][1].shape[0]

    ## We should aggregate the tuple here
    for variable, value in interventions[0].items():

        num_children = len(children[variable])
        num_parents = len(parents[variable])
        num_independent_nodes = len(independent_nodes[variable])

        subset_children = children[variable]
        subset_parents = parents[variable]
        subset_independent_nodes = independent_nodes[variable]

        ## This is changing the intervention variable
        final_variables[variable] = value * np.ones((num_observations, 1))

        ## This is changing the children
        if num_children != 0:
            for i in range(num_children):
                ## This should update the values of the children - eg for X this is modifying Z
                functions_to_get = list(subset_children.items())[0][0][-1]
                ## If this function gets other variables that are not children of the intervention we need to add them here
                ## This is changing the X_2 - taking the mean value of GP2
                children_value = functions[functions_to_get].predict(
                    value * np.ones((num_observations, 1))
                )[0]
                final_variables[list(subset_children.keys())[0]] = children_value

        ## The independent nodes stay the same - If dont exist dont need to provide
        if num_independent_nodes != 0:
            for j in range(num_independent_nodes):
                final_variables[list(subset_independent_nodes.keys())[j]] = list(
                    subset_independent_nodes.items()
                )[j][1]

        ## The parents nodes stay the same - If dont exist dont need to provide
        if num_parents != 0:
            for j in range(num_parents):
                final_variables[list(subset_parents.keys())[j]] = list(
                    subset_parents.items()
                )[j][1]

    ## after having changed all the variables, we predict the Y
    num_parents_Y = len(parents_Y.keys())
    inputs_Y = np.zeros((num_observations, num_parents_Y))

    ## Getting the parents of Y to compute the CE on Y
    for i in range(len(parents_Y.keys())):
        var = list(parents_Y.keys())[i]
        inputs_Y[:, i] = final_variables[var][:, 0]

    gp_Y = functions["Y"]

    causal_effect_mean = np.mean(gp_Y.predict(inputs_Y)[0])
    causal_effect_var = np.mean(gp_Y.predict(inputs_Y)[1])

    return causal_effect_mean, causal_effect_var
