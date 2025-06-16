import argparse
import logging
from collections import OrderedDict
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple

import jax.numpy as jnp
import networkx as nx
import numpy as np
from GPy.models.gp_regression import GPRegression
from scipy.special import expit

# Note: We removed diffcbed dependencies and created a simplified implementation
from graphs.graph import GraphStructure


def define_SEM_causalenv_linear(
    graph: nx.MultiDiGraph, weighted_adjacency_matrix: np.ndarray
) -> OrderedDict[str, Callable]:

    sem_functions = OrderedDict()
    for node in nx.topological_sort(graph):
        parents = list(graph.predecessors(node))
        if not parents:
            sem_functions[str(node)] = lambda epsilon, sample, node=node: epsilon
        else:
            sem_functions[str(node)] = (
                lambda epsilon, sample, node=node, parents=parents: sum(
                    weighted_adjacency_matrix[parent, node] * sample[str(parent)]
                    for parent in parents
                )
                + epsilon
            )

    return sem_functions


class ChainGraph(GraphStructure):

    def __init__(
        self,
        num_nodes: int,
        noise_sigma: float = 1.0,
        seed: int = None,
        nonlinear: bool = False,
    ):
        self.num_nodes = num_nodes

        # Create a simple chain graph manually (simplified without diffcbed dependency)
        self._variables = [str(i) for i in range(num_nodes)]
        self._edges = [(str(i), str(i+1)) for i in range(num_nodes-1)]
        self.nonlinear = nonlinear
        self._nodes = sorted(set(chain(*self.edges)))
        self._parents, self._children = self.build_relationships()
        self._target = f"{num_nodes - 1}"
        self._functions: Optional[Dict[str, Callable]] = None
        self.noise_sigma = noise_sigma
        self.rng = np.random.default_rng(seed)
        self._G = self.make_graphical_model()
        self._standardised = False
        self.use_intervention_range_data = False
        
        # Create simple weighted adjacency matrix for linear chain
        self.weighted_adjacency_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes - 1):
            self.weighted_adjacency_matrix[i, i+1] = 1.0  # Simple linear chain with weight 1
        
        self._SEM = self.define_SEM()

    def define_SEM(self):
        if self.nonlinear:
            # For simplicity, we'll use linear for now
            pass
        
        # Create simple linear SEM for chain structure
        sem_functions = OrderedDict()
        
        # Build parents dict for proper dependencies
        parents_dict = {}
        for parent, child in self._edges:
            if child not in parents_dict:
                parents_dict[child] = []
            parents_dict[child].append(parent)
        
        # CRITICAL: Use topological order to ensure parents are defined before children
        import networkx as nx
        
        # Create a graph with the current variable names and edges
        temp_graph = nx.DiGraph()
        temp_graph.add_nodes_from(self._variables)
        temp_graph.add_edges_from(self._edges)
        topological_order = list(nx.topological_sort(temp_graph))
        
        for node in topological_order:
            if node not in parents_dict or not parents_dict[node]:
                # Root node - create function with proper closure
                def make_root_function():
                    return lambda epsilon, sample: epsilon
                sem_functions[node] = make_root_function()
            else:
                # Child node - depends on parents
                node_parents = parents_dict[node]
                if len(node_parents) == 1:
                    parent = node_parents[0]
                    # Create function with proper closure
                    def make_single_parent_function(parent_var):
                        return lambda epsilon, sample: sample[parent_var] + epsilon
                    sem_functions[node] = make_single_parent_function(parent)
                else:
                    # Multiple parents
                    def make_multi_parent_function(parent_vars):
                        return lambda epsilon, sample: sum(sample[p] for p in parent_vars) + epsilon
                    sem_functions[node] = make_multi_parent_function(node_parents)
        return sem_functions

    def get_error_distribution(self, noiseless=False):
        err_dist = {}
        for node in self.variables:  # Use self.variables instead of self.nodes
            if noiseless:
                err_dist[node] = 0.0
            else:
                err_dist[node] = self.rng.normal(0.0, self.noise_sigma)
        return err_dist

    def get_sets(self):
        mis = []
        pomis = []
        manipulative_variables = [var for var in self.variables if var != self.target]
        return mis, pomis, manipulative_variables

    # Abstract method implementations
    @property
    def SEM(self):
        return self._SEM
    
    @property
    def iscm_paramters(self):
        if not hasattr(self, 'population_mean_variance'):
            self.population_mean_variance = {
                var: {"mean": 0.0, "std": 1.0} for var in self._variables
            }
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
        return self._variables
    
    @property
    def variables(self):
        return self._variables

    def get_interventional_range(self, D_O: Dict = None):
        """Sets the range of the variables we can intervene upon"""
        interventional_range = OrderedDict()
        for var in self.variables:
            interventional_range[var] = [-3, 3]  # Default range for chain
        return interventional_range