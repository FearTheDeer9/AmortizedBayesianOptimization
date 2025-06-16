#!/usr/bin/env python3
"""
GraphStructure Classes for PARENT_SCALE Integration

Contains the abstract GraphStructure base class from PARENT_SCALE and our
concrete ACBOGraphStructure implementation that bridges our SCM data structures
with PARENT_SCALE's expected interfaces.
"""

import abc
import logging
from typing import List, Dict, Any, Optional, Tuple, FrozenSet, Callable
from collections import OrderedDict, deque
from functools import partial

import numpy as onp
import pyrsistent as pyr
import networkx as nx
from emukit.core import ContinuousParameter, ParameterSpace
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression
from pgmpy.models import BayesianNetwork

# Import our data structures
from causal_bayes_opt.data_structures.scm import get_variables, get_edges, get_target, get_parents

# Import helper functions
from .helpers import (
    MyKDE, safe_optimization, set_intervention_values, predict_child,
    predict_causal_effect, propogate_effects
)

# Helper classes from PARENT_SCALE
MESSAGE = "Subclass should implement this."


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
    def iscm_paramters(self):
        return self.population_mean_variance

    @property
    @abc.abstractmethod
    def target(self) -> str:
        return self._target

    @property
    @abc.abstractmethod
    def standardised(self) -> bool:
        return self._standardised

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
    def variables(self):
        return self._variables

    @property
    def G(self):
        return self._G

    @abc.abstractmethod
    def get_sets(self) -> Tuple[List, List, List]:
        raise NotImplementedError(MESSAGE)

    @abc.abstractmethod
    def get_cost_structure(self, type_cost: int) -> OrderedDict:
        if type_cost == 1:
            costs = self.get_fixed_equal_costs()
        else:
            logging.warning("Undefined cost structure")

        assert isinstance(costs, OrderedDict)
        return costs

    @abc.abstractmethod
    def get_fixed_equal_costs(self) -> OrderedDict[str, Callable]:
        manipulative_variables = self.get_sets()[2]
        costs = OrderedDict()
        for var in manipulative_variables:
            func = lambda intervention_value: 1.0
            costs[var] = func

        return costs

    @abc.abstractmethod
    def get_parameter_space(self, exploration_set) -> ParameterSpace:
        if hasattr(self, 'use_intervention_range_data') and self.use_intervention_range_data:
            interventional_range = self.interventional_range_data
        else:
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

        # Create a new MultiDiGraph
        G = nx.MultiDiGraph()

        # Add all nodes from the BayesianNetwork
        G.add_nodes_from(self.variables)

        # Add all edges from the BayesianNetwork
        G.add_edges_from(model.edges())

        return G

    @abc.abstractmethod
    def set_interventional_range_data(self, D_O):
        interventional_range = OrderedDict()
        for var in self.variables:
            interventional_range[var] = [D_O[var].min(), D_O[var].max()]
        self.use_intervention_range_data = True
        self.interventional_range_data = interventional_range

    @abc.abstractmethod
    def get_interventional_range(self, D_O: Dict = None):
        """
        Sets the range of the variables we can intervene upon
        """
        interventional_range = OrderedDict()
        for var in self.variables:
            interventional_range[var] = [-5, 5]
        return interventional_range

    @abc.abstractmethod
    def fit_samples_to_graph(
        self,
        samples: OrderedDict,
        parameters: OrderedDict = None,
        set_priors: bool = False,
    ) -> None:

        # first step is get the set of all the child nodes
        children_parents = self.parents

        self._functions = OrderedDict()
        for child, parents in children_parents.items():
            if parents:
                logging.debug(
                    f"Fitting child: {child} to parents: {parents} for {self.edges}"
                )
                Y = samples[child]
                X = onp.hstack([samples[parent] for parent in parents])
                kernel = RBF(
                    input_dim=len(parents), variance=1.0, ARD=False, lengthscale=1.0
                )
                gp = GPRegression(X=X, Y=Y, kernel=kernel)

                gp.optimize()
                gp = safe_optimization(gp)
                self._functions[child] = gp
            else:
                logging.debug(f"Fitting marginal distribution for child {child}")
                Y = samples[child]
                self._functions[child] = MyKDE(kernel="gaussian").fit_and_update(Y)

    @abc.abstractmethod
    def build_relationships(self) -> Tuple[dict, dict]:
        parents = {}
        children = {}
        for node in self.nodes:
            parents[node] = []
            children[node] = []

        for parent, child in self.edges:
            children[parent].append(child)
            parents[child].append(parent)

        return dict(parents), dict(children)

    @abc.abstractmethod
    def mispecify_graph(self, edges):
        """
        Flip some of the edges of the graph for further experimentation
        """
        self._edges = edges
        self._parents, self._children = self.build_relationships()
        self._G = self.make_graphical_model()

    @abc.abstractmethod
    def causal_effect_DO(
        self,
        interventions,
        functions,
        children,
        parents,
        independent_nodes,
        observational_samples,
    ):
        """
        Computes the causal effect based on the interventions in the system
        """
        final_variables = OrderedDict()
        num_observations = observational_samples[self.target].shape[0]
        update_check_dict = {var: False for var in self.variables}

        topological_order = list(nx.topological_sort(self.G))
        # Process interventions
        for intervention in interventions:
            set_intervention_values(
                final_variables,
                intervention,
                interventions[intervention],
                num_observations,
            )
            # this one has been updated
            update_check_dict[intervention] = True

        for var in topological_order:
            var_parents = self.parents[var]
            # this condition checks if any of the parents of this
            if update_check_dict[var]:
                # this one has already been updated
                continue
            # variable has been updated through an intervention
            if var_parents:
                parents_updated = any(
                    [update_check_dict[parent] for parent in var_parents]
                )
            else:
                parents_updated = False

            # if none of them have been updated use the observational samples
            if not parents_updated:

                final_variables[var] = observational_samples[var]
                # continue to the next variable
                continue

            parents_dict = {var: final_variables[var] for var in var_parents}
            final_variables[var] = predict_child(
                functions[var], parents_dict, var_parents
            )
            update_check_dict[var] = True

        parents_Y = self.parents[self.target]
        mean_effect, variance_effect = predict_causal_effect(
            functions,
            final_variables,
            parents_Y,
            num_observations,
            self.target,
        )

        return mean_effect, variance_effect


# Concrete implementation for our SCM data structures
class ACBOGraphStructure(GraphStructure):
    """Concrete GraphStructure implementation for ACBO SCM integration."""
    
    def __init__(self, scm: pyr.PMap):
        """Initialize from our pyrsistent SCM format."""
        self._variables = list(get_variables(scm))
        self._edges = list(get_edges(scm))
        self._target = get_target(scm)
        self._nodes = self._variables
        self._standardised = False
        self.population_mean_variance = {}
        self.use_intervention_range_data = False
        
        # Standardization attributes (like original PARENT_SCALE)
        self.scale_data = True  # Enable by default
        self.means = {}
        self.stds = {}
        
        # Build parent-child relationships
        self._parents, self._children = self.build_relationships()
        
        # Create graph model
        self._G = self.make_graphical_model()
        
        # Initialize functions
        self._functions = OrderedDict()
        
        # Initialize SEM from SCM mechanisms using original PARENT_SCALE pattern
        # The SEM should be an OrderedDict of mechanism functions in topological order
        self._SEM = OrderedDict()
        
        # Get topological order for proper SEM evaluation
        from causal_bayes_opt.data_structures.scm import topological_sort
        topo_order = topological_sort(scm)
        
        # Create mechanism functions compatible with PARENT_SCALE
        # These should be functions that take (epsilon, sample) and return values
        mechanisms = scm.get('mechanisms', {})
        
        for var in topo_order:  # Important: use topological order!
            if var in mechanisms:
                # Create a proper SEM function compatible with PARENT_SCALE sampling
                mech = mechanisms[var]
                var_parents = list(get_parents(scm, var))
                
                if hasattr(mech, 'sample') and hasattr(mech, 'coefficients'):
                    # Convert ACBO LinearMechanism to proper SEM format
                    def make_sem_function(mechanism, parents, variable_name):
                        def sem_function(epsilon, sample_dict):
                            # Extract parent values from sample dict
                            result = getattr(mechanism, 'intercept', 0.0)
                            
                            # Add contributions from parents using actual coefficients
                            for parent in parents:
                                if parent in sample_dict:
                                    coeff = mechanism.coefficients.get(parent, 0.0)
                                    result += coeff * sample_dict[parent]
                            
                            # Add noise term (epsilon already scaled)
                            noise_scale = getattr(mechanism, 'noise_scale', 1.0)
                            result += noise_scale * epsilon
                            
                            return result
                        return sem_function
                    
                    self._SEM[var] = make_sem_function(mech, var_parents, var)
                elif hasattr(mech, 'mean') and len(var_parents) == 0:
                    # Root mechanism from create_root_mechanism
                    mean_val = getattr(mech, 'mean', 0.0)
                    noise_scale = getattr(mech, 'noise_scale', 1.0)
                    
                    def make_root_sem_function(mean, noise_scale):
                        def root_sem_function(epsilon, sample_dict):
                            return mean + noise_scale * epsilon
                        return root_sem_function
                    
                    self._SEM[var] = make_root_sem_function(mean_val, noise_scale)
                else:
                    # For other mechanisms, try to extract coefficients and intercept
                    if len(var_parents) == 0:
                        # Root variable - use noise only
                        def root_sem(epsilon, sample_dict):
                            return epsilon
                        self._SEM[var] = root_sem
                    else:
                        # Try to get coefficients from mechanism attributes
                        def make_linear_sem(mechanism, parents):
                            def linear_sem(epsilon, sample_dict):
                                result = getattr(mechanism, 'intercept', 0.0)
                                
                                # Add contributions from parents
                                for parent in parents:
                                    if parent in sample_dict:
                                        # Try to get coefficient from mechanism
                                        coeff = 1.0  # Default coefficient
                                        if hasattr(mechanism, 'coefficients') and parent in mechanism.coefficients:
                                            coeff = mechanism.coefficients[parent]
                                        result += coeff * sample_dict[parent]
                                
                                # Add noise
                                noise_scale = getattr(mechanism, 'noise_scale', 1.0)
                                result += noise_scale * epsilon
                                
                                return result
                            return linear_sem
                        
                        self._SEM[var] = make_linear_sem(mech, var_parents)
            else:
                # No mechanism defined, create appropriate default
                var_parents = list(get_parents(scm, var))
                if len(var_parents) == 0:
                    # Root variable
                    def root_sem(epsilon, sample_dict):
                        return epsilon
                    self._SEM[var] = root_sem
                else:
                    # Variable with parents - create simple linear SEM
                    def default_sem(epsilon, sample_dict):
                        result = 0.0
                        for parent in var_parents:
                            if parent in sample_dict:
                                result += sample_dict[parent]
                        return result + epsilon
                    self._SEM[var] = default_sem

    @property
    def SEM(self):
        return self._SEM

    @property
    def iscm_paramters(self):
        return self.population_mean_variance

    @property
    def target(self) -> str:
        return self._target

    @property
    def standardised(self) -> bool:
        return self._standardised

    @property
    def edges(self) -> List[Tuple[str, str]]:
        return self._edges

    @property
    def functions(self):
        return self._functions

    @property
    def parents(self):
        return self._parents

    @property
    def children(self):
        return self._children

    @property
    def nodes(self) -> List[str]:
        return self._nodes

    @property
    def variables(self):
        return self._variables
        
        # Standardization attributes (like original PARENT_SCALE)
        self.scale_data = True  # Enable by default
        self.means = {}
        self.stds = {}
        self._standardised = False

    def get_sets(self) -> Tuple[List, List, List]:
        """Return (mis, pomis, manipulative_variables)."""
        mis = []  # Multi-intervention sets (not used in our simple case)
        pomis = []  # Parent of multiple intervention sets (not used)
        manipulative_variables = [var for var in self.variables if var != self.target]
        return mis, pomis, manipulative_variables

    def get_cost_structure(self, type_cost: int = 1) -> OrderedDict:
        """Get cost structure for interventions."""
        if type_cost == 1:
            costs = self.get_fixed_equal_costs()
        else:
            logging.warning("Undefined cost structure")
        
        assert isinstance(costs, OrderedDict)
        return costs

    def get_fixed_equal_costs(self) -> OrderedDict[str, Callable]:
        """Get fixed equal costs for all manipulative variables."""
        manipulative_variables = self.get_sets()[2]
        costs = OrderedDict()
        for var in manipulative_variables:
            func = lambda intervention_value: 1.0
            costs[var] = func
        return costs

    def get_parameter_space(self, exploration_set) -> ParameterSpace:
        """Get parameter space for exploration set."""
        if hasattr(self, 'use_intervention_range_data') and self.use_intervention_range_data:
            interventional_range = self.interventional_range_data
        else:
            interventional_range = self.get_interventional_range()
        space = {}

        # Create ContinuousParameter objects dynamically for all variables in the interventional range
        for var in interventional_range:
            bounds = interventional_range[var]
            space[var] = ContinuousParameter(var, bounds[0], bounds[1])

        # Generate a list of parameters for only the variables in the exploration set
        es_space = [space[var] for var in exploration_set if var in space]

        return ParameterSpace(es_space)

    def make_graphical_model(self) -> nx.MultiDiGraph:
        """Create NetworkX graph model."""
        model = BayesianNetwork(self.edges)

        # Create a new MultiDiGraph
        G = nx.MultiDiGraph()

        # Add all nodes from the BayesianNetwork
        G.add_nodes_from(self.variables)

        # Add all edges from the BayesianNetwork
        G.add_edges_from(model.edges())

        return G

    def set_interventional_range_data(self, D_O):
        """Set intervention ranges based on observational data."""
        interventional_range = OrderedDict()
        for var in self.variables:
            interventional_range[var] = [D_O[var].min(), D_O[var].max()]
        self.use_intervention_range_data = True
        self.interventional_range_data = interventional_range

    def get_interventional_range(self, D_O: Dict = None):
        """Sets the range of the variables we can intervene upon."""
        interventional_range = OrderedDict()
        
        # If we have observational data, derive ranges from it (following original pattern)
        if D_O is not None:
            for var in self.variables:
                if var in D_O:
                    data_min = float(onp.min(D_O[var]))
                    data_max = float(onp.max(D_O[var]))
                    # Extend range slightly beyond observed data
                    range_extend = (data_max - data_min) * 0.2
                    interventional_range[var] = [data_min - range_extend, data_max + range_extend]
                else:
                    # Fallback to reasonable range
                    interventional_range[var] = [-3, 3]
        else:
            # Default ranges when no data available (following original graph patterns)
            for var in self.variables:
                if var == self.target:
                    # Don't intervene on target
                    continue
                else:
                    # Use reasonable range similar to original graphs
                    interventional_range[var] = [-3, 3]
        
        return interventional_range

    def fit_samples_to_graph(
        self,
        samples: OrderedDict,
        parameters: OrderedDict = None,
        set_priors: bool = False,
    ) -> None:
        """Fit GP models to samples for causal modeling."""
        # first step is get the set of all the child nodes
        children_parents = self.parents

        self._functions = OrderedDict()
        for child, parents in children_parents.items():
            if parents:
                logging.debug(
                    f"Fitting child: {child} to parents: {parents} for {self.edges}"
                )
                Y = samples[child].reshape(-1, 1)  # Ensure 2D column vector for GPy
                X = onp.hstack([samples[parent] for parent in parents])
                kernel = RBF(
                    input_dim=len(parents), variance=1.0, ARD=False, lengthscale=1.0
                )
                gp = GPRegression(X=X, Y=Y, kernel=kernel)

                gp.optimize()
                gp = safe_optimization(gp)
                self._functions[child] = gp
            else:
                logging.debug(f"Fitting marginal distribution for child {child}")
                Y = samples[child]
                self._functions[child] = MyKDE(kernel="gaussian").fit_and_update(Y)

    def build_relationships(self) -> Tuple[dict, dict]:
        """Build parent-child relationships from edges."""
        parents = {}
        children = {}
        for node in self.nodes:
            parents[node] = []
            children[node] = []

        for parent, child in self.edges:
            children[parent].append(child)
            parents[child].append(parent)

        return dict(parents), dict(children)

    def mispecify_graph(self, edges):
        """Modify the graph edges for testing different causal hypotheses."""
        self._edges = edges
        self._parents, self._children = self.build_relationships()
        self._G = self.make_graphical_model()

    def causal_effect_DO(
        self,
        interventions,
        functions,
        children,
        parents,
        independent_nodes,
        observational_samples,
    ):
        """Computes the causal effect based on the interventions in the system."""
        final_variables = OrderedDict()
        num_observations = observational_samples[self.target].shape[0]
        update_check_dict = {var: False for var in self.variables}

        topological_order = list(nx.topological_sort(self.G))
        # Process interventions
        for intervention in interventions:
            set_intervention_values(
                final_variables,
                intervention,
                interventions[intervention],
                num_observations,
            )
            # this one has been updated
            update_check_dict[intervention] = True

        for var in topological_order:
            var_parents = self.parents[var]
            # this condition checks if any of the parents of this
            if update_check_dict[var]:
                # this one has already been updated
                continue
            # variable has been updated through an intervention
            if var_parents:
                parents_updated = any(
                    [update_check_dict[parent] for parent in var_parents]
                )
            else:
                parents_updated = False

            # if none of them have been updated use the observational samples
            if not parents_updated:

                final_variables[var] = observational_samples[var]
                # continue to the next variable
                continue

            parents_dict = {var: final_variables[var] for var in var_parents}
            final_variables[var] = predict_child(
                functions[var], parents_dict, var_parents
            )
            update_check_dict[var] = True

        parents_Y = self.parents[self.target]
        mean_effect, variance_effect = predict_causal_effect(
            functions,
            final_variables,
            parents_Y,
            num_observations,
            self.target,
        )

        return mean_effect, variance_effect

    # Implement remaining abstract methods with minimal implementations for bridge use
    def define_SEM(self):
        """Define SEM (not used in bridge)."""
        pass

    def fit_all_models(self):
        """Fit all models (implemented via fit_samples_to_graph)."""
        pass

    def get_exploration_set(self):
        """Get exploration set matching ToyGraph pattern for fair comparison."""
        # For ToyGraph chain X -> Z -> Y with target Y:
        # Original exploration set is [("X",), ("Z",), ("X", "Z")]
        variables = [var for var in self.variables if var != self.target]
        
        if self.target == 'Y' and set(variables) == {'X', 'Z'}:
            # Match ToyGraph exactly
            return [("X",), ("Z",), ("X", "Z")]
        else:
            # Fallback: generate all combinations for other structures
            import itertools
            exploration_set = []
            for r in range(1, len(variables) + 1):
                exploration_set.extend(list(itertools.combinations(variables, r)))
            return exploration_set

    def refit_models(self, observational_samples):
        """Refit GP models based on new observational samples."""
        self.fit_samples_to_graph(observational_samples)

    def get_all_do(self):
        """Get all do-intervention functions."""
        do_dict = {}
        exploration_set = self.get_exploration_set()
        for es in exploration_set:
            key = "_".join(es)
            do_function_name = f"compute_do_{key}"
            do_dict[do_function_name] = partial(
                self.compute_do_generic, intervention_nodes=es
            )
        return do_dict

    def get_original_interventional_range(self):
        """Get original intervention ranges."""
        return self.get_interventional_range()

    def get_set_BO(self):
        """Get BO set (not implemented for bridge)."""
        raise NotImplementedError("Not implemented for bridge")

    def get_fixed_different_costs(self):
        """Get different fixed costs (not implemented for bridge)."""
        raise NotImplementedError("Not implemented for bridge")

    def get_variable_equal_costs(self):
        """Get variable equal costs (not implemented for bridge)."""
        raise NotImplementedError("Not implemented for bridge")

    def get_variable_different_costs(self):
        """Get variable different costs (not implemented for bridge)."""
        raise NotImplementedError("Not implemented for bridge")

    def show_graphical_model(self) -> None:
        """Show graphical model (not implemented for bridge)."""
        pass

    def get_error_distribution(self, noiseless: bool = False) -> Dict[str, float]:
        """Get error distribution."""
        if noiseless:
            # Return zero noise for noiseless mode
            err_dist = {var: 0.0 for var in self.variables}
        else:
            # Return random noise
            err_dist = {var: onp.random.normal(0, 0.1) for var in self.variables}
        return err_dist

    def find_independent_nodes(self, target: str) -> List[str]:
        """Find nodes that are neither ancestors nor descendants of the target."""
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

    def get_parents_children_independent(
        self, observational_samples: dict, interventions_nodes: List[str]
    ):
        """Get the parents, children and independent nodes of a specific intervention node."""
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

    def compute_do(
        self,
        observational_samples: dict,
        value: onp.ndarray,
        interventions_nodes: List[str],
    ):
        """Computes the interventional outcome on the system."""
        # this may cause issues if multiple interventions are considered at once
        value = value.reshape(1, -1)

        functions = self.functions
        children, parents, independent = self.get_parents_children_independent(
            observational_samples, interventions_nodes
        )
        num_interventions = value.shape[0]
        mean_do = onp.zeros((num_interventions, 1))
        var_do = onp.zeros((num_interventions, 1))

        for i in range(num_interventions):
            interventions = {
                var: value[i, j] for j, var in enumerate(interventions_nodes)
            }
            mean_do[i], var_do[i] = self.causal_effect_DO(
                interventions,
                functions=functions,
                children=children,
                parents=parents,
                independent_nodes=independent,
                observational_samples=observational_samples,
            )

        return mean_do, var_do

    def break_dependency_structure(self):
        """Changing the edges for the BO method."""
        target = self.target
        nodes = self.nodes
        self._edges = []
        for node in nodes:
            if node != target:
                self._edges.append((node, target))

        self._parents, self._children = self.build_relationships()
        self._G = self.make_graphical_model()

    def entropy_decomposition(self, observational_samples: Dict, num_points: int = 100):
        """Decompose entropy into components."""
        _, _, manipulative_variables = self.get_sets()
        total_entropy = 0
        for var in self.variables:
            parents = self.parents[var]
            function = self.functions[var]
            if parents:
                dataset = onp.hstack(
                    [observational_samples[parent] for parent in parents]
                )
                variance = function.predict(dataset)[1]
                entropy = 1 / 2 * onp.log(2 * onp.pi * onp.exp(1) * variance)
                total_entropy += onp.mean(entropy)
                print(f"{var} taking average {onp.mean(entropy)}")
            else:
                # since we assume gaussian noise and use gausian kernel use gausian again
                variance = function.predict()[1]
                entropy = 1 / 2 * onp.log(2 * onp.pi * onp.exp(1) * variance)
                total_entropy += entropy
                print(f"{var} taking marginal {onp.mean(entropy)}")

        entropy_dict = {"entropy": total_entropy}
        return entropy_dict

    def decompose_variance(
        self, variable: str, observational_samples: Dict, num_points: int = 100
    ):
        """Decompose variance into epistemic and aleatoric components."""
        parents = self.parents[variable]

        _, _, manipulative_variables = self.get_sets()
        n_obs = len(observational_samples[self.target])

        aleatoric_uncertainty = 0
        epistemic_uncertainty = 0

        count_m = 0
        count_nm = 0

        if parents:
            target_function: GPRegression = self.functions[variable]
            for i, var in enumerate(parents):
                dataset = onp.hstack(
                    [observational_samples[parent] for parent in parents]
                )
                min_var = onp.min(observational_samples[var])
                max_var = onp.max(observational_samples[var])
                vals = onp.linspace(start=min_var, stop=max_var, num=num_points)

                variance = onp.zeros(shape=num_points)
                for j, val in enumerate(vals):
                    dataset[:, i] = onp.repeat(val, n_obs)
                    variance[j] = onp.mean(target_function.predict(dataset)[1])

                if var in manipulative_variables:
                    # this means we are in the subset X
                    count_m += 1
                    epistemic_uncertainty += onp.mean(variance)
                else:
                    # this means we are in the subset C
                    count_nm += 1
                    aleatoric_uncertainty += onp.mean(variance)
        if count_m > 0:
            epistemic_uncertainty = epistemic_uncertainty

        if count_nm > 0:
            aleatoric_uncertainty = aleatoric_uncertainty
        uncertanties = {
            "epistemic": epistemic_uncertainty,
            "aleatoric": aleatoric_uncertainty,
        }
        return uncertanties

    def decompose_target_variance(
        self, observational_samples: Dict, num_points: int = 100
    ):
        """Decompose target variable variance."""
        return self.decompose_variance(
            variable=self.target,
            observational_samples=observational_samples,
            num_points=num_points,
        )

    def decompose_all_variance(
        self, observational_samples: Dict, num_points: int = 100
    ):
        """Decompose variance for all variables."""
        aleatoric_uncertainty = 0
        epistemic_uncertainty = 0
        for var in self.variables:
            uncertanties = self.decompose_variance(
                variable=var,
                observational_samples=observational_samples,
                num_points=num_points,
            )
            aleatoric_uncertainty += uncertanties["aleatoric"]
            epistemic_uncertainty += uncertanties["epistemic"]

        uncertanties = {
            "aleatoric": aleatoric_uncertainty,
            "epistemic": epistemic_uncertainty,
        }
        return uncertanties

    def compute_do_generic(self, intervention_nodes, observational_samples, value):
        """Generic do-intervention computation."""
        mean_do, var_do = self.compute_do(
            observational_samples, value, intervention_nodes
        )
        return mean_do, var_do

    def set_data_standardised_flag(
        self,
        standardised: bool = True,
        means: Dict[str, onp.ndarray] = None,
        stds: Dict[str, onp.ndarray] = None,
    ):
        """Set data standardization flag and parameters."""
        if standardised:
            self._standardised = standardised
            self.means = means
            self.stds = stds
        else:
            self._standardised = standardised

    def standardize_all_data(self, D_O: Dict, D_I: Dict):
        """
        Standardize all data exactly like original PARENT_SCALE.
        
        This standardizes input variables (not target) using observational data statistics.
        """
        if not self.scale_data:
            return D_O, D_I
            
        input_keys = [key for key in D_O.keys() if key != self.target]
        self.means = {key: onp.mean(D_O[key]) for key in input_keys}
        self.stds = {key: onp.std(D_O[key]) for key in input_keys}
        
        # Standardize observational data
        D_O_scaled = {}
        for key in D_O:
            if key in input_keys:
                D_O_scaled[key] = self.standardize(D_O[key], self.means[key], self.stds[key])
            else:
                D_O_scaled[key] = D_O[key]  # Target remains unstandardized
        
        # Standardize interventional data
        D_I_scaled = {}
        for intervention_key in D_I:
            D_I_scaled[intervention_key] = {}
            for key in D_I[intervention_key]:
                if key in input_keys:
                    D_I_scaled[intervention_key][key] = self.standardize(
                        D_I[intervention_key][key], self.means[key], self.stds[key]
                    )
                else:
                    D_I_scaled[intervention_key][key] = D_I[intervention_key][key]
        
        self._standardised = True
        return D_O_scaled, D_I_scaled
    
    def standardize(self, data, mean, std):
        """Standardize data: (data - mean) / std"""
        return (data - mean) / std
    
    def reverse_standardize(self, data, mean, std):
        """Reverse standardization: data * std + mean"""
        return (data * std) + mean
    
    def standardize_intervention_value(self, variable: str, value: float) -> float:
        """Standardize a single intervention value."""
        if not self._standardised or variable not in self.means:
            return value
        return self.standardize(value, self.means[variable], self.stds[variable])
    
    def destandardize_intervention_value(self, variable: str, standardized_value: float) -> float:
        """Destandardize a single intervention value."""
        if not self._standardised or variable not in self.means:
            return standardized_value
        return self.reverse_standardize(standardized_value, self.means[variable], self.stds[variable])
    
    def set_params_iscm(self, var: str, mean: float, std: float):
        """Set ISCM parameters for a variable."""
        self.population_mean_variance[var]["mean"] = mean
        self.population_mean_variance[var]["std"] = std