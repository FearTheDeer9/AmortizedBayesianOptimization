from collections import namedtuple
from typing import Dict

import causaldag as cd
import cdt
import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import f1_score
import jax

# Replace the try-except block with direct import from our utility
from diffcbed.models.dibs.models.nonLinearGaussian import DenseNonlinearGaussian

from config import NOISE_TYPES, PRESETS, VARIABLE_TYPES
from diffcbed.envs.samplers import D
from .causal_dataset import CausalDataset

Data = namedtuple("Data", ["samples", "intervention_node"])


def logmeanexp(A, axis):
    return logsumexp(A, axis=axis) - np.log(A.shape[axis])


def mmd(x, y, kernel):
    # compare kernel MMD paper and code:
    # A. Gretton et al.: A kernel two-sample test, JMLR 13 (2012)
    # http://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm
    # x shape [n, d] y shape [m, d]
    # n_perm number of bootstrap permutations to get p-value, pass none to not get p-value

    n, d = x.shape
    m, d2 = y.shape
    assert d == d2, "x and y must have same dimensionality"
    k_x = kernel(x, x)
    k_y = kernel(y, y)
    k_xy = kernel(x, y)

    # The diagonals are always 1 (up to numerical error, this is (3) in Gretton et al.)
    # note that their code uses the biased (and differently scaled mmd)
    mmd = (
        k_x.sum() / (n * (n - 1)) + k_y.sum() / (m * (m - 1)) - 2 * k_xy.sum() / (n * m)
    )
    return mmd


class CausalEnvironment(torch.utils.data.Dataset):
    """Base class for generating different graphs and performing ancestral sampling.

    This class handles the broader experimental setup while delegating data management
    to the CausalDataset class.
    """

    def __init__(
        self,
        args,
        num_nodes,
        num_edges,
        noise_type,
        num_samples,
        node_range,
        mu_prior=None,
        sigma_prior=None,
        seed=None,
        nonlinear=False,
        logger=None,
    ):
        """Initialize the causal environment."""
        self.args = args
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.noise_type = noise_type
        self.num_samples = num_samples
        self.node_range = node_range
        self.mu_prior = mu_prior
        self.sigma_prior = sigma_prior
        self.seed = seed
        self.nonlinear = nonlinear
        self.logger = logger

        # Initialize random number generators
        self.rng = np.random.default_rng(seed)
        self.rng_jax = jax.random.PRNGKey(seed if seed is not None else 0)

        # Initialize noise standard deviations based on noise type
        if self.noise_type.endswith("gaussian"):
            if self.noise_type == "isotropic-gaussian":
                # Default isotropic noise
                self._noise_std = [1.0] * self.num_nodes
            elif self.noise_type == "gaussian":
                self._noise_std = np.linspace(0.1, 1.0, self.num_nodes)
        elif self.noise_type == "exponential":
            # Default exponential noise
            self._noise_std = [1.0] * self.num_nodes

        # Initialize nonlinear conditionals if needed
        if nonlinear:
            self.conditionals = DenseNonlinearGaussian(
                obs_noise=self._noise_std,
                sig_param=1.0,
                hidden_layers=[5,],
            )

        # Sample weights and build graph
        self.sample_weights()
        self.build_graph()

        # Initialize sampler
        self.init_sampler()

        # Get SCM before initializing dataset
        scm = self._get_scm()

        # Initialize CausalDataset for data management
        self.dataset = CausalDataset(
            self.graph,
            scm,
            n_obs=num_samples,
            n_int=2,  # Default number of interventional samples
            seed=seed
        )

        # Initialize held-out data after dataset is ready
        self.held_out_interventions = []
        self.held_out_nodes = []
        self.held_out_values = []

        # Create held-out interventions
        for node in range(self.num_nodes):
            for value in [-20, 20]:
                self.held_out_nodes.append(node)
                self.held_out_values.append(value)

                # Add intervention to dataset
                self.dataset.add_intervention(node, value, n_samples=200)
                intervention_samples = self.dataset.get_intervention_samples(
                    node, value)

                # Convert samples to array format
                sample_array = np.zeros((200, self.num_nodes))
                for i in range(self.num_nodes):
                    sample_array[:, i] = intervention_samples[str(i)].flatten()

                self.held_out_interventions.append({
                    "node": node,
                    "value": value,
                    "samples": sample_array
                })

        # Sample held-out observational data
        self.held_out_data = self.sample(1000).samples

    def _get_scm(self) -> Dict[str, callable]:
        """Get the structural causal model based on current configuration."""
        if self.nonlinear:
            return self._get_nonlinear_scm()
        else:
            return self._get_linear_scm()

    def _get_linear_scm(self) -> Dict[str, callable]:
        """Get linear SCM based on weighted adjacency matrix."""
        mechanisms = {}

        def create_mechanism(i, parents, weights):
            def mechanism(n_samples=1):
                # Create a dictionary to store intermediate values
                node_values = {}

                # Get topological ordering starting from parents
                topo_order = list(nx.topological_sort(self.graph))

                # Calculate values for all nodes in topological order
                for node in topo_order:
                    if str(node) not in node_values:
                        node_parents = list(self.graph.predecessors(node))
                        if not node_parents:
                            # Root node
                            node_values[str(node)] = self.graph.nodes[node]["sampler"].sample(
                                n_samples).reshape(-1, 1)
                        else:
                            # Non-root node
                            node_weights = [
                                self.weighted_adjacency_matrix[j, node] for j in node_parents]
                            parent_values = np.hstack(
                                [node_values[str(p)] for p in node_parents])
                            node_values[str(node)] = (parent_values @ np.array(node_weights).reshape(-1, 1) +
                                                      self.graph.nodes[node]["sampler"].sample(n_samples).reshape(-1, 1))

                # Return the value for the target node
                return node_values[str(i)].reshape(-1)

            return mechanism

        # Create mechanisms for all nodes
        for i in range(self.num_nodes):
            parents = list(self.graph.predecessors(i))
            weights = [self.weighted_adjacency_matrix[j, i]
                       for j in parents] if parents else []
            mechanisms[str(i)] = create_mechanism(i, parents, weights)

        return mechanisms

    def _get_nonlinear_scm(self) -> Dict[str, callable]:
        """Get nonlinear SCM based on current conditionals."""
        mechanisms = {}

        def create_mechanism(i, parents):
            def mechanism(n_samples=1):
                # Create a dictionary to store intermediate values
                node_values = {}

                # Get topological ordering starting from parents
                topo_order = list(nx.topological_sort(self.graph))

                # Calculate values for all nodes in topological order
                for node in topo_order:
                    if str(node) not in node_values:
                        node_parents = list(self.graph.predecessors(node))
                        if not node_parents:
                            # Root node
                            node_values[str(node)] = self.graph.nodes[node]["sampler"].sample(
                                n_samples).reshape(-1, 1)
                        else:
                            # Non-root node
                            parent_values = np.hstack(
                                [node_values[str(p)] for p in node_parents])
                            with torch.no_grad():
                                output = self.conditionals.forward(
                                    torch.FloatTensor(parent_values))
                            node_values[str(node)] = output.numpy(
                            ) + self.graph.nodes[node]["sampler"].sample(n_samples).reshape(-1, 1)

                # Return the value for the target node
                return node_values[str(i)].reshape(-1)

            return mechanism

        # Create mechanisms for all nodes
        for i in range(self.num_nodes):
            parents = list(self.graph.predecessors(i))
            mechanisms[str(i)] = create_mechanism(i, parents)

        return mechanisms

    def reseed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        if hasattr(self, 'dataset'):
            self.dataset.seed = seed

    def __getitem__(self, index):
        raise NotImplementedError

    def build_graph(self):
        """Initilises the adjacency matrix and the weighted adjacency matrix"""

        self.adjacency_matrix = nx.to_numpy_array(self.graph)

        if self.nonlinear:
            self.weighted_adjacency_matrix = None
        else:
            self.weighted_adjacency_matrix = self.adjacency_matrix.copy()
            edge_pointer = 0
            for i in nx.topological_sort(self.graph):
                parents = list(self.graph.predecessors(i))
                if len(parents) == 0:
                    continue
                else:
                    for j in parents:
                        self.weighted_adjacency_matrix[j, i] = self.weights[
                            edge_pointer
                        ]
                        edge_pointer += 1

        print("GT causal graph")
        print(self.adjacency_matrix.astype(np.uint8))

    def init_sampler(self, graph=None):
        if graph is None:
            graph = self.graph

        if self.noise_type.endswith("gaussian"):
            for i in range(self.num_nodes):
                graph.nodes[i]["sampler"] = D(
                    self.rng.normal, loc=0.0, scale=self._noise_std[i]
                )
        elif self.noise_type == "exponential":
            for i in range(self.num_nodes):
                graph.nodes[i]["sampler"] = D(
                    self.rng.exponential, scale=self._noise_std[i])

        return graph

    def sample_weights(self):
        """Sample the edge weights"""
        if self.nonlinear:
            self.weights = self.conditionals.sample_parameters(
                key=self.rng_jax, n_vars=self.num_nodes
            )
        else:
            if self.mu_prior is not None:
                # self.weights = torch.distributions.normal.Normal(self.mu_prior, self.sigma_prior).sample([self.num_edges])
                self.weights = D(
                    self.rng.normal, self.mu_prior, self.sigma_prior
                ).sample(size=self.num_edges)
            else:
                dist = D(self.rng.uniform, -5, 5)
                self.weights = torch.zeros(self.num_edges)
                for k in range(self.num_edges):
                    sample = 0.0
                    while sample > -0.5 and sample < 0.5:
                        sample = dist.sample(size=1)
                        self.weights[k] = sample

    def sample_linear(self, num_samples, graph=None, node=None, values=None, onehot=False):
        """Sample observations given a graph with optional intervention.

        Args:
            num_samples: Number of samples to generate
            graph: networkx DiGraph (uses self.graph if None)
            node: Node to intervene on (can be array-like for onehot=True)
            values: Value for intervention
            onehot: If True, node is a one-hot vector and values is an array

        Returns:
            Data object containing samples and intervention information
        """
        if graph is None:
            graph = self.graph

        samples = np.zeros((num_samples, self.num_nodes))

        # Sample in topological order
        for i in nx.topological_sort(graph):
            # Handle intervention
            if onehot and isinstance(node, (np.ndarray, list)) and node[i] == 1.0:
                # Binary node case - use value from values array
                samples[:, i] = values[i]
            elif not onehot and isinstance(node, (int, np.integer)) and i == node:
                # Integer node case - use single value
                samples[:, i] = values
            else:
                # No intervention - sample normally
                noise = self.args.scm_bias + \
                    graph.nodes[i]["sampler"].sample(num_samples)
                parents = list(graph.predecessors(i))

                if len(parents) == 0:
                    # Root node - just noise
                    samples[:, i] = noise
                else:
                    # Add weighted parent contributions
                    parent_contribution = sum(
                        self.weighted_adjacency_matrix[j, i] * samples[:, j]
                        for j in parents
                    )
                    samples[:, i] = parent_contribution + noise

        return Data(samples=samples, intervention_node=node)

    def sample_nonlinear(self, num_samples, graph=None, node=None, values=None):
        if graph is None:
            graph = self.graph
        mat = nx.to_numpy_array(graph)
        g = ig.Graph.Weighted_Adjacency(mat.tolist())
        samples = self.conditionals.sample_obs(
            key=self.rng_jax,
            n_samples=num_samples,
            g=g,
            theta=self.weights,
            node=node,
            # value_sampler=value_sampler,
            values=values,
        )
        return Data(samples=samples, intervention_node=-1)

    def intervene(self, nodes, values, num_samples=1000):
        """Intervene on one or more nodes with given values.

        Args:
            nodes: Node(s) to intervene on. Can be:
                - Single integer: Intervene on one node
                - List of integers: Intervene on multiple nodes simultaneously
                - np.ndarray: Vector-valued intervention (each element corresponds to a node)
            values: Value(s) to set the node(s) to. Can be:
                - Single float: Value for single node intervention
                - List of floats: Values for multiple node interventions
                - np.ndarray: Vector of values for vector-valued intervention
            num_samples: Number of samples to generate

        Returns:
            Data object containing samples and intervention node information
        """
        # Convert inputs to consistent format
        if isinstance(nodes, (int, np.integer)):
            # Single node intervention
            nodes = [int(nodes)]
            values = [float(values)]
        elif isinstance(nodes, list):
            # Multiple node intervention
            nodes = [int(n) for n in nodes]
            values = [float(v) for v in values]
        elif isinstance(nodes, np.ndarray):
            # Vector-valued intervention
            if nodes.ndim == 1:
                # Get indices of non-zero elements
                nodes = np.where(nodes != 0)[0].tolist()
                if isinstance(values, np.ndarray):
                    # Extract values at the non-zero indices
                    values = [float(values[i]) for i in nodes]
                else:
                    values = [float(values)] * len(nodes)
            else:
                raise ValueError("Node array must be 1-dimensional")
        else:
            raise TypeError(
                "Nodes must be integer, list of integers, or numpy array")

        # Create mutated graph by removing incoming edges to intervention nodes
        mutated_graph = self.graph.copy()
        for node in nodes:
            for edge in list(mutated_graph.in_edges(node)):
                mutated_graph.remove_edge(*edge)

        # Initialize sampler for mutated graph
        mutated_graph = self.init_sampler(mutated_graph)

        # Sample from appropriate mechanism
        if self.nonlinear:
            samples = self.sample_nonlinear(
                num_samples,
                mutated_graph,
                nodes,  # Pass list of nodes
                values,  # Pass list of values
            ).samples
        else:
            samples = self.sample_linear(
                num_samples,
                mutated_graph,
                nodes,  # Pass list of nodes
                values,  # Pass list of values
                onehot=False,
            ).samples

        # Update dataset with interventions
        for node, value in zip(nodes, values):
            self.dataset.add_intervention(node, value, n_samples=num_samples)

        # Get formatted samples
        sample_array = np.zeros((num_samples, self.num_nodes))
        for i in range(self.num_nodes):
            if i in nodes:
                # For intervened nodes, use the intervention value
                value_idx = nodes.index(i)
                sample_array[:, i] = np.full(num_samples, values[value_idx])
            else:
                # For non-intervened nodes, get samples from dataset
                intervention_samples = self.dataset.get_intervention_samples(
                    nodes[0], values[0])
                sample_array[:, i] = intervention_samples[str(i)].flatten()

        return Data(samples=sample_array, intervention_node=nodes)

    def sample(self, num_samples):
        """Sample from the observational distribution"""
        obs_data = self.dataset.get_obs_data()
        sample_array = np.zeros((num_samples, self.num_nodes))
        for i in range(self.num_nodes):
            sample_array[:, i] = obs_data[str(i)].flatten()
        return Data(samples=sample_array, intervention_node=-1)

    def interventional_likelihood_linear(self, data, intervention):
        graph = self.graph
        logprobs = np.zeros(data.shape[0])
        for i in nx.topological_sort(graph):
            if i == intervention:
                continue
            noise_std = self._noise_std[i]
            parents = list(graph.predecessors(i))
            # TODO: for now we're assuming a zero bias. make it non-zero as well
            bias = self.args.noise_bias
            if len(parents) == 0:
                logprobs += norm.logpdf(data[:, i], bias, noise_std)
            else:
                mean = 0.0
                for j in parents:
                    mean += self.weighted_adjacency_matrix[j, i] * data[:, j]
                # mean += bias
                logprobs += norm.logpdf(data[:, i], mean, noise_std)
        return logprobs

    def interventional_likelihood(self, data, interventions):
        return self.interventional_likelihood_linear(data, interventions)

    def __len__(self):
        return self.num_samples

    def plot_graph(
        self,
        path,
        A=None,
        scores=None,
        dashed_cpdag=True,
        ax=None,
        legend=True,
        save=True,
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        if A is None:
            A = self.adjacency_matrix

        try:
            cpdag = cd.DAG().from_amat(amat=A).cpdag().to_amat()[0]
        except:
            cpdag = None

        G = nx.DiGraph(A)
        g = nx.convert_matrix.from_numpy_array(A, create_using=nx.DiGraph)

        pos = {}
        labels = {}
        r = 1
        for i, n in enumerate(range(A.shape[0])):
            theta = np.deg2rad(i * 360 / A.shape[0])
            x, y = r * np.sin(theta), r * np.cos(theta)
            pos[n] = (x, y)
            labels[n] = f"{n+1}"

        edges = G.edges()
        CPDAG_A = np.zeros(A.shape)
        NON_CPDAG_A = np.zeros(A.shape)
        for i, j in edges:
            if cpdag is not None and cpdag[i, j] == cpdag[j, i]:
                CPDAG_A[i][j] = 1
            else:
                NON_CPDAG_A[i][j] = 1

        cmap = plt.cm.plasma

        nodes = nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=g.nodes(),
            node_color=scores,
            node_size=2000,
            edgecolors="black",
            linewidths=5,
            cmap="coolwarm",
        )
        nx.draw_networkx_labels(g, pos, labels, font_color="white")

        nx.draw_networkx_edges(
            nx.convert_matrix.from_numpy_array(
                NON_CPDAG_A, create_using=nx.DiGraph),
            pos,
            style="solid",
            node_size=1000,
            width=5,
            arrowsize=20,
            connectionstyle="arc3, rad = 0.08",
        )

        collection = nx.draw_networkx_edges(
            nx.convert_matrix.from_numpy_array(
                CPDAG_A, create_using=nx.DiGraph),
            pos,
            style="dashed",
            node_size=1000,
            width=5,
            arrowsize=20,
            connectionstyle="arc3, rad = 0.08",
        )

        if dashed_cpdag and collection is not None:
            for patch in collection:
                patch.set_linestyle("--")

        ax.set_axis_off()
        if scores is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0)
            cb = plt.colorbar(nodes, cax=cax)
            cb.outline.set_visible(False)

            cax.get_yaxis().labelpad = -60
            cax.set_ylabel("score", rotation=270)

        if legend:
            ax.legend(
                [
                    Line2D([0, 1], [0, 1], linewidth=3,
                           linestyle="-", color="black"),
                    Line2D([0, 1], [0, 1], linewidth=3,
                           linestyle="--", color="black"),
                ],
                [r"$\notin$ CPDAG", r"$\in$ CPDAG"],
                frameon=False,
            )

        if save:
            plt.savefig(path)

    def eshd(self, model, samples, double_for_anticausal=True, force_ensemble=False):
        shds = []

        if model.ensemble or force_ensemble:
            for i in range(len(model.all_graphs)):
                if getattr(model.all_graphs[0], "to_amat", False):
                    G = (model.all_graphs[i].to_amat() != 0).astype(np.uint8)
                else:
                    G = np.array(model.all_graphs[i])
                shds.append(
                    cdt.metrics.SHD(
                        self.adjacency_matrix.copy(),
                        np.array(G),
                        double_for_anticausal=double_for_anticausal,
                    )
                )
        else:
            Gs = model.sample(samples)
            for G in Gs:
                shds.append(
                    cdt.metrics.SHD(
                        self.adjacency_matrix.copy(),
                        np.array(G),
                        double_for_anticausal=double_for_anticausal,
                    )
                )

        return (np.array(shds) * np.exp(model.normalized_log_weights)).sum()

    def get_graphs(self, model, samples=1000, force_ensemble=False):
        if model.ensemble or force_ensemble:
            Gs = []
            for i in range(len(model.all_graphs)):
                if getattr(model.all_graphs[0], "to_amat", False):
                    G = (model.all_graphs[i].to_amat() != 0).astype(np.uint8)
                else:
                    G = np.array(model.all_graphs[i])
                Gs.append(G)
            Gs = np.array(Gs)
        else:
            Gs = np.array(model.sample(samples))

        return Gs

    def f1_score(self, model, samples=1000, force_ensemble=False):
        Gs = self.get_graphs(model, samples, force_ensemble)
        gtGs = self.adjacency_matrix.copy()

        _gtGs = np.broadcast_to(gtGs[None], (Gs.shape[0],) + gtGs.shape)
        f1_scores = []
        for i in range(Gs.shape[0]):
            f1_scores.append(f1_score(_gtGs[i].ravel(), Gs[i].ravel()))
        return (np.array(f1_scores) * np.exp(model.normalized_log_weights)).sum()

    def i_mmd(self, model):
        # get samples from models
        model_samples = model.batch_interventional_samples(
            np.array(self.held_out_nodes)[:, None],
            np.array(self.held_out_values)[:, None],
            200,
            onehot=True,
        )

        # Dags x T x B x N x D
        mmds = []
        for i, (node, value) in enumerate(
            zip(self.held_out_nodes, self.held_out_values)
        ):
            intervention = self.held_out_interventions[i]
            node = intervention["node"]
            value = intervention["value"]
            gt_samples = intervention["samples"]

            _mmds = []
            for dag in range(len(model.all_graphs)):
                _model_samples = model_samples[dag, 0, i]

                # via https://torchdrift.org/notebooks/note_on_mmd.html
                # Gretton et. al. recommend to set the parameter to the median
                # distance between points.

                dists = torch.cdist(
                    torch.Tensor(np.array(_model_samples)),
                    torch.Tensor(np.array(gt_samples)),
                )
                sigma = (dists.median() / 2).item()
                kernel = RBF(length_scale=sigma)
                _mmds.append(mmd(_model_samples, gt_samples, kernel))
            mmds.append(
                (np.array(_mmds) * np.exp(model.normalized_log_weights)).sum())
        return np.array(mmds).mean()

    def get_valid_interventions(self):
        return list(range(self.num_nodes))
