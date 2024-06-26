from collections import namedtuple
from typing import Dict, List, OrderedDict

import graphical_models
import networkx as nx
import numpy as np

from config import NOISE_TYPES
from diffcbed.envs.causal_environment import CausalEnvironment
from diffcbed.envs.samplers import D
from graphs.graph import GraphStructure

Data = namedtuple("Data", ["samples", "intervention_node"])


class GraphStructureEnv(CausalEnvironment):
    """
    The aim of this class is to convert the GraphStructure environment
    to this CausalEnvironment so that all the mutual information strategies
    can now be tested.
    """

    def __init__(
        self,
        graph: GraphStructure,
        args: Dict,
        # exp_edges: int = 1,
        noise_type: str = "isotropic-gaussian",
        noise_sigma: float = 1.0,
        node_range: List[int] = [-10, 10],
        num_samples: int = 1000,
        mu_prior: float = 2.0,
        sigma_prior: float = 1.0,
        seed: int = 10,
        nonlinear: bool = False,
        binary_nodes: bool = True,
        logger=None,
    ):
        # this uses the networkx Graph as used in the graph datastructure
        self.graph: nx.DiGraph = nx.DiGraph(graph.G)
        self.SEM: OrderedDict = graph.SEM
        self.adjacency_matrix = nx.to_numpy_array(self.graph)

        self.args = args
        num_nodes = len(graph.nodes)
        num_edges = len(graph.edges)
        self.nonlinear = True
        self.noise_sigma = noise_sigma
        self.seed = seed
        self.allow_cycles = False
        self.node_range = node_range
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        assert (
            noise_type in NOISE_TYPES
        ), "Noise types must correspond to {} but got {}".format(
            NOISE_TYPES, noise_type
        )
        self.noise_type = noise_type
        self.num_samples = num_samples
        self.mu_prior = mu_prior
        self.sigma_prior = sigma_prior
        self.seed = seed
        self.nonlinear = nonlinear
        self.binary_nodes = binary_nodes
        self.logger = logger

        self.node_map = {node: i for i, node in enumerate(self.graph.nodes)}
        self.node_map_inv = {i: node for i, node in enumerate(self.graph.nodes)}
        nx.relabel_nodes(self.graph, self.node_map, copy=False)

        # super().__init__(
        #     args,
        #     num_nodes,
        #     len(self.graph.edges),
        #     noise_type,
        #     num_samples,
        #     node_range=node_range,
        #     mu_prior=mu_prior,
        #     sigma_prior=sigma_prior,
        #     seed=seed,
        #     nonlinear=nonlinear,
        #     binary_nodes=binary_nodes,
        #     logger=logger,
        # )

        self.reseed(self.seed)
        self.init_sampler()

        self.dag = graphical_models.DAG.from_nx(self.graph)

        print(f"Expected degree: {np.mean(list(dict(self.graph.in_degree).values()))}")

        self.nodes = self.dag.nodes
        self.arcs = self.dag.arcs

    def __getitem__(self, index):
        return self.samples[index]

    def init_sampler(self, graph=None):
        if graph is None:
            graph = self.graph

        print(graph)
        nodes = list(graph.nodes)
        if self.noise_type.endswith("gaussian"):
            # Identifiable
            if self.noise_type == "isotropic-gaussian":
                self._noise_std = [self.noise_sigma] * self.num_nodes
            elif self.noise_type == "gaussian":
                self._noise_std = np.linspace(0.1, 1.0, self.num_nodes)
            for i in range(self.num_nodes):
                graph.nodes[nodes[i]]["sampler"] = D(
                    self.rng.normal, loc=0.0, scale=self._noise_std[i]
                )

        elif self.noise_type == "exponential":
            noise_std = [self.noise_sigma] * self.num_nodes
            for i in range(self.num_nodes):
                graph.nodes[i]["sampler"] = D(self.rng.exponential, scale=noise_std[i])

        return graph

    def sample_linear(
        self,
        num_samples: int,
        graph: nx.DiGraph = None,
        intervention_node: str = None,
        values: float = None,
        onehot: bool = False,
    ):
        """
        Sample observations given a graph
        num_samples: Scalar
        graph: networkx DiGraph
        node: If intervention is performed, specify which node
        value: value set to node after intervention

        Outputs: Observations [num_samples x num_nodes]
        """

        if graph is None:
            graph = self.graph

        samples = np.zeros((num_samples, self.num_nodes))
        sample_dict = {}
        edge_pointer = 0
        for i, node in enumerate(nx.topological_sort(graph)):
            if onehot and intervention_node[i] == 1:
                noise = values[i]
            elif not onehot and node == intervention_node:
                noise = values
            else:
                noise = self.args.scm_bias + self.graph.nodes[node]["sampler"].sample(
                    num_samples
                )
            parents = list(graph.predecessors(node))

            if len(parents) == 0:
                samples[:, i] = noise
                sample_dict[node] = noise
            else:
                # Prepare parent values as input to the function
                parent_values = {self.node_map_inv[p]: sample_dict[p] for p in nx.ancestors(graph, node)}

                # Compute current node values using its function
                current_values = self.SEM[self.node_map_inv[node]](noise, parent_values)
                sample_dict[node] = current_values

                samples[:, i] = current_values

        return Data(samples=samples, intervention_node=-1)

    def intervene(self, iteration, num_samples, nodes, values, _log=False):
        """Perform intervention to obtain a mutilated graph"""

        mutated_graph = self.adjacency_matrix.copy()
        # nodes_int = [self.node_map[node] for node in nodes]
        # print(nodes)
        if self.binary_nodes:
            mutated_graph[:, nodes.astype(np.bool_)] = 0
        else:
            mutated_graph[:, nodes] = 0

        # Initialize a new DiGraph from the mutated adjacency matrix
        new_graph = nx.from_numpy_array(mutated_graph, create_using=nx.DiGraph)

        # label_mapping = {idx: node for idx, node in enumerate(self.node_map)}
        # nx.relabel_nodes(new_graph, label_mapping, copy=False)

        # Set node attributes from old graph (if necessary)
        # for node in new_graph.nodes:
        #     new_graph.nodes[node].update(self.graph.nodes[node])

        # Sample from the new graph
        samples = self.sample_linear(
            num_samples,
            new_graph,
            nodes,
            values,
            onehot=self.binary_nodes,
        ).samples

        if _log:
            self.logger.log_interventions(iteration, nodes, samples)

        return Data(samples=samples, intervention_node=nodes)
