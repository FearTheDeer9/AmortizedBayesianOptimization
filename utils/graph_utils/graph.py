import abc
import logging
from collections import OrderedDict, defaultdict, deque
from typing import Callable, Dict, List, OrderedDict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from emukit.core import ContinuousParameter, ParameterSpace
from GPy.core.parameterization import priors
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression
from pgmpy.models import BayesianNetwork

MESSAGE = "Subclass should implement this."


# this is for calculating the causal effect in a more clear way
# these functions are just outside of the class as it is not needed in the class
def set_intervention_values(variables, interventions, num_observations):
    """
    Changing the variables that was intervened upon when computing
    the outcome of the new graph
    """
    for variable, value in interventions.items():
        variables[variable] = value * np.ones((num_observations, 1))


def update_children_values(
    functions: OrderedDict,
    variables: OrderedDict,
    children: OrderedDict,
    parents: OrderedDict,
    observational_samples,
):
    """
    Propogates the intervention effects down in the graph. There are many cases
    to consider, that is why the function is so messy
    """
    # trying to update the children correctly in the DAG
    for var, children_list in children.items():
        if var in variables:
            parent_values = {}
            # parent_values = variables[var]
            for child in children_list:
                if child in functions and child not in variables:
                    # this makes sure child was not previously intervened upon
                    parents_of_child = parents[child]
                    for current_parent in parents_of_child:
                        # makes sure that you are considering other variables that
                        # are not the current variable we are considering
                        if current_parent in variables:
                            parent_values[current_parent] = variables[current_parent]
                        else:
                            parent_values[current_parent] = observational_samples[
                                current_parent
                            ]

                    variables[child] = predict_child(
                        functions[child], parent_values, parents_of_child
                    )


def predict_child(
    functions: GPRegression, parent_values: Dict[str, np.ndarray], parents: List
):
    """
    Makes sure that the parent variables are in the correct order for the GP model
    """
    parent_values_cols = np.hstack([parent_values[val] for val in parents])
    return functions.predict(parent_values_cols)[0]


def maintain_independent_and_parents(variables, nodes):
    """
    Makes sure the independent nodes do not change, adds a check if the variable
    has not been changed in one of the previous interventions
    """
    for _, independent_nodes in nodes.items():
        for var, value in independent_nodes.items():
            if var not in variables:
                variables[var] = value


def predict_causal_effect(functions, variables, parents_Y, num_observations, target):
    """
    Predicts the causal effect after all the variables has been intervened upon
    Returns the mean and the variance of the observation
    """
    input_Y = np.hstack(
        [variables[parent].reshape(num_observations, -1) for parent in parents_Y]
    )
    gp_Y = functions[target]
    predictions = gp_Y.predict(input_Y)
    return np.mean(predictions[0]), np.mean(predictions[1])


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
    def target(self) -> str:
        return self._target

    @property
    @abc.abstractmethod
    def edges(self) -> List[Tuple[str, str]]:
        return self._edges

    @property
    @abc.abstractmethod
    def functions(self):
        return self._functions

    @property
    @abc.abstractmethod
    def parents(self):
        return self._parents

    @property
    @abc.abstractmethod
    def children(self):
        return self._children

    @property
    @abc.abstractmethod
    def nodes(self) -> List[str]:
        return self._nodes

    @property
    @abc.abstractmethod
    def target(self):
        return self._target

    @property
    @abc.abstractmethod
    def variables(self):
        return self._variables

    @abc.abstractmethod
    def define_SEM():
        """
        This method defines the structural equation model (SEM) for
        each of the simulated graph structures
        """
        raise NotImplementedError(MESSAGE)

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
        self.fit_samples_to_graph(observational_samples)

    @abc.abstractmethod
    def get_all_do(self):
        """
        This assigns assigns the calculation of the interventional distribution
        to each of the variables in the SCM
        """
        raise NotImplementedError(MESSAGE)

    @abc.abstractmethod
    def get_interventional_range(self):
        """
        Sets the range of the variables we can intervene upon
        """
        raise NotImplementedError(MESSAGE)

    @abc.abstractmethod
    def get_set_BO(self):
        raise NotImplementedError(MESSAGE)

    @abc.abstractmethod
    def get_sets(self) -> Tuple[List, List, List]:
        raise NotImplementedError(MESSAGE)

    @abc.abstractmethod
    def get_interventional_domain(self):
        raise NotImplementedError(MESSAGE)

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
        raise NotImplementedError(MESSAGE)

    @abc.abstractmethod
    def get_fixed_different_costs(self):
        raise NotImplementedError(MESSAGE)

    @abc.abstractmethod
    def get_variable_equal_costs(self):
        raise NotImplementedError(MESSAGE)

    @abc.abstractmethod
    def get_variable_different_costs(self):
        raise NotImplementedError(MESSAGE)

    @abc.abstractmethod
    def get_parameter_space(self, exploration_set) -> ParameterSpace:
        interventional_range = self.get_interventional_range()
        space = {}

        # Create ContinuousParameter objects dynamically for all variables in the interventional range
        for var in interventional_range:
            bounds = interventional_range[var]
            space[var] = ContinuousParameter(var, bounds[0], bounds[1])

        # Generate a list of parameters for only the variables in the exploration set
        es_space = [space[var] for var in exploration_set if var in space]

        return ParameterSpace(es_space)

    @abc.abstractmethod
    def make_graphical_model(self) -> nx.MultiDiGraph:
        model = BayesianNetwork(self.edges)

        # Manually creating a networkx graph from the BayesianModel
        G = nx.MultiDiGraph()
        G.add_edges_from(model.edges())
        return G

    @abc.abstractmethod
    def show_graphical_model(self) -> None:
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

    @abc.abstractmethod
    def get_error_distribution(self) -> Dict[str, Callable]:
        # raise NotImplementedError(MESSAGE)
        return {var: np.random.rand for var in self.variables}

    @abc.abstractmethod
    def fit_samples_to_graph(
        self,
        samples: OrderedDict,
        parameters: OrderedDict = None,
        set_priors: bool = False,
    ) -> None:

        # first step is get the set of all the child nodes
        logging.info("Fitting each sample to the graph based on the parents")
        children_parents = self.parents

        self._functions = OrderedDict()
        for child, parents in children_parents.items():
            if not parents:
                continue
            Y = samples[child]
            X = np.hstack([samples[parent] for parent in parents])
            kernel = RBF(
                input_dim=len(parents), variance=1.0, ARD=False, lengthscale=1.0
            )
            gp = GPRegression(X=X, Y=Y, kernel=kernel)

            # this can also be a flag - not sure why it is added
            if set_priors:
                prior_len = priors.InverseGamma.from_EV(1.0, 1.0)
                prior_sigma_f = priors.InverseGamma.from_EV(4.0, 0.5)
                prior_lik = priors.InverseGamma.from_EV(3, 1)

                gp.kern.variance.set_prior(prior_sigma_f)
                gp.kern.lengthscale.set_prior(prior_len)
                gp.likelihood.variance.set_prior(prior_lik)

            gp.optimize()
            self._functions[child] = gp

    @abc.abstractmethod
    def build_relationships(self) -> Tuple[dict, dict]:
        parents = defaultdict(set)
        children = defaultdict(set)
        for node in self.nodes:
            parents[node] = []
            children[node] = []

        for parent, child in self.edges:
            children[parent].append(child)
            parents[child].append(parent)

        return dict(parents), dict(children)

    @abc.abstractmethod
    def find_independent_nodes(self, target: str) -> List[str]:
        """
        Find nodes that are neither ancestors nor descendants of the target.
        """

        # Function to find all descendants of a given node
        def find_descendants(node):
            visited = set()
            queue = deque([node])
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                for child in self.children.get(current, []):
                    if child not in visited:
                        queue.append(child)
            return visited

        # Function to find all ancestors of a given node
        def find_ancestors(node):
            visited = set()
            queue = deque([node])
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                for parent in self.parents.get(current, []):
                    if parent not in visited:
                        queue.append(parent)
            return visited

        descendants = find_descendants(target)
        ancestors = find_ancestors(target)

        # Independent nodes are those not in descendants or ancestors of the target node
        return [
            node
            for node in self.nodes
            if node not in descendants and node not in ancestors
        ]

    @abc.abstractmethod
    def causal_effect_DO(
        self,
        *interventions,
        functions,
        parents_Y,
        children,
        parents,
        independent_nodes,
        observational_samples
    ):
        """
        Computes the causal effect based on the interventions in the system
        """
        final_variables = OrderedDict()
        num_observations = list(parents_Y.values())[0].shape[0]

        # Process interventions
        for intervention_dict in interventions:
            set_intervention_values(
                final_variables, intervention_dict, num_observations
            )

        # Update children based on the new values from interventions
        update_children_values(
            functions, final_variables, children, self.parents, observational_samples
        )

        # Independent nodes and parents should maintain their original values
        maintain_independent_and_parents(final_variables, independent_nodes)
        maintain_independent_and_parents(final_variables, parents)

        # Predict the causal effect on the target variable 'Y'
        mean_effect, variance_effect = predict_causal_effect(
            functions,
            final_variables,
            parents_Y,
            num_observations,
            self.target,
        )

        return mean_effect, variance_effect

    @abc.abstractmethod
    def get_parents_children_independent(
        self, observational_samples: dict, interventions_nodes: List[str]
    ):
        """
        Get the parents, children and independent nodes of a specific intervention
        node
        """
        # getting the children nodes for these interventions
        children = OrderedDict(
            [
                (
                    var,
                    OrderedDict(
                        [
                            (child, observational_samples[child])
                            for child in self.children[var]
                        ]
                    ),
                )
                for var in interventions_nodes
            ]
        )

        # getting the parent nodes for these interventions
        parents_nodes = OrderedDict(
            [
                (
                    var,
                    OrderedDict(
                        [
                            (child, observational_samples[child])
                            for child in self.parents[var]
                        ]
                    ),
                )
                for var in interventions_nodes
            ]
        )

        # getting the independent nodes for these interventions
        independent_nodes = OrderedDict(
            [
                (
                    var,
                    OrderedDict(
                        [
                            (node, observational_samples[node])
                            for node in self.find_independent_nodes(var)
                        ]
                    ),
                )
                for var in interventions_nodes
            ]
        )

        return children, parents_nodes, independent_nodes

    @abc.abstractmethod
    def compute_do(
        self,
        observational_samples: dict,
        value: np.ndarray,
        interventions_nodes: List[str],
    ):
        """
        Computes the interventional outcome on the system
        """
        # this may cause issues if multiple interventions are considered at once
        value = value.reshape(1, -1)
        parents_Y = OrderedDict(
            [(var, observational_samples[var]) for var in self.parents[self.target]]
        )

        functions = self.functions
        children, parents, independent = self.get_parents_children_independent(
            observational_samples, interventions_nodes
        )
        num_interventions = value.shape[0]
        mean_do = np.zeros((num_interventions, 1))
        var_do = np.zeros((num_interventions, 1))

        for i in range(num_interventions):
            interventions = {
                var: value[i, j] for j, var in enumerate(interventions_nodes)
            }
            mean_do[i], var_do[i] = self.causal_effect_DO(
                interventions,
                functions=functions,
                parents_Y=parents_Y,
                children=children,
                parents=parents,
                independent_nodes=independent,
                observational_samples=observational_samples,
            )

        return mean_do, var_do

    @abc.abstractmethod
    def break_dependency_structure(self):
        """
        Changing the edges for the BO method
        """
        target = self.target
        nodes = self.nodes
        self._edges = []
        for node in nodes:
            if node != target:
                self._edges.append((node, target))

    @abc.abstractmethod
    def mispecify_graph(self, edges):
        """
        Flip some of the edges of the graph for further experimentation
        """
        self._edges = edges
        self._parents, self._children = self.build_relationships()
