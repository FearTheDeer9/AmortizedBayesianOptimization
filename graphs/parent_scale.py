import numpy as np
import networkx as nx
import logging
import itertools
from copy import deepcopy
from typing import Dict, List, Tuple, Union, Optional, Any, OrderedDict
from collections import OrderedDict, namedtuple

from graphs.graph import GraphStructure
from graphs.causal_env_adapter import CausalEnvironmentAdapter
from graphs.causal_dataset import CausalDataset
from graphs.scm_generators import sample_observational, sample_interventional


def sample_model(scm, interventions, sample_count, graph=None):
    """Sample from an SCM with interventions.

    Args:
        scm: Structural Causal Model
        interventions: Dictionary mapping node names to intervention values
        sample_count: Number of samples to generate
        graph: Graph structure (optional, used for causal ordering)

    Returns:
        Dictionary of samples
    """
    # This is a wrapper around sample_interventional for multiple interventions
    # Get node ordering from graph if available
    if graph is not None and hasattr(graph, 'nodes'):
        nodes = list(graph.nodes())
    elif graph is not None and hasattr(graph, 'G'):
        nodes = list(graph.G.nodes())
    else:
        # Default to SCM node ordering
        nodes = list(scm.keys())

    # Get a single node to intervene on
    if len(interventions) == 1:
        node, value = next(iter(interventions.items()))
        return sample_interventional(scm, node, value, sample_count, seed=None, graph=graph)

    # For multiple interventions, we need to handle specially
    # This is a simplification - for actual implementation we would need to
    # respect the causal structure when doing multiple interventions

    # Start with observational samples
    samples = sample_observational(scm, sample_count, seed=None)

    # Overwrite intervention nodes
    for node, value in interventions.items():
        samples[str(node)] = np.full(sample_count, value)

    # Get children of intervention nodes
    all_children = set()
    for node in interventions:
        if graph is not None:
            if hasattr(graph, 'children'):
                # GraphStructure has children property
                children = graph.children.get(str(node), [])
            elif hasattr(graph, 'G'):
                # Using the NetworkX graph
                children = list(graph.G.successors(node))
            else:
                # Direct NetworkX graph
                children = list(graph.successors(node))
            all_children.update(children)

    # Recalculate values for children in topological order
    # This is a simplification
    for node in sorted(all_children, key=lambda x: int(x) if x.isdigit() else x):
        if node not in interventions and node in scm:
            # Get parent values
            parent_values = {}
            for parent, parent_fn in scm[node].items():
                if parent in samples:
                    parent_values[parent] = samples[parent]

            # Recalculate node value
            if parent_values:
                # If we have SCM function for this node, use it
                samples[node] = np.zeros(sample_count)
                for i in range(sample_count):
                    # Get individual parent values for this sample
                    sample_parent_vals = {p: v[i]
                                          for p, v in parent_values.items()}
                    # Apply SCM function
                    samples[node][i] = scm[node](sample_parent_vals)

    return samples


class ReplayBuffer:
    """Simple replay buffer to store data samples."""

    def __init__(self, binary=False):
        self._data = []
        self.binary = binary

    def update(self, data):
        self._data.append(data)

    def data(self):
        return self._data


class DoFunctions:
    """Class that synthesizes all the do functions into one class."""

    def __init__(
        self,
        do_effects_functions: Dict,
        observational_samples: Dict,
        intervention_variables: List,
    ) -> None:
        self.do_effects_functions = do_effects_functions
        self.observational_samples = observational_samples
        self.intervention_variables = intervention_variables
        self.set_do_effects_function()
        # Cache for previously computed values
        self.xi_dict_mean = {}
        self.xi_dict_var = {}

    def get_do_function_name(self) -> str:
        """Returns the name of the do function based on intervened variables."""
        string = ""
        for i in range(len(self.intervention_variables)):
            string += str(self.intervention_variables[i])
        do_function_name = "compute_do_" + string
        return do_function_name

    def set_do_effects_function(self) -> None:
        """Set the do function if the current list of interventions changes."""
        self.do_effects_function = self.do_effects_functions[self.get_do_function_name(
        )]

    def mean_function_do(self, x) -> np.float64:
        """Calculate the interventional mean based on the specific value."""
        num_interventions = x.shape[0]
        mean_do = np.zeros((num_interventions, 1))
        for i in range(num_interventions):
            xi_str = str(x[i])
            if xi_str in self.xi_dict_mean:
                mean_do[i] = self.xi_dict_mean[xi_str]
            else:
                mean_do[i], _ = self.do_effects_function(
                    observational_samples=self.observational_samples, value=x[i]
                )
                self.xi_dict_mean[xi_str] = mean_do[i]
        return np.float64(mean_do)

    def var_function_do(self, x) -> np.float64:
        """Calculate the interventional variance based on the specific x value."""
        num_interventions = x.shape[0]
        var_do = np.zeros((num_interventions, 1))
        for i in range(num_interventions):
            xi_str = str(x[i])
            if xi_str in self.xi_dict_var:
                var_do[i] = self.xi_dict_var[xi_str]
            else:
                _, var_do[i] = self.do_effects_function(
                    observational_samples=self.observational_samples, value=x[i]
                )
                self.xi_dict_var[xi_str] = var_do[i]
        return np.float64(var_do)


class SCMModel:
    """Base class for SCM models."""

    def __init__(self, prior_probabilities, graph=None):
        self.prior_probabilities = prior_probabilities.copy()
        self.graph = graph
        self.data = {}

    def set_data(self, data):
        self.data = data.copy()

    def add_data(self, new_data):
        for key, value in new_data.items():
            if key not in self.data:
                self.data[key] = value
            else:
                self.data[key] = np.concatenate([self.data[key], value])

    def update_all(self, x_dict, y):
        """Update probabilities based on new data."""
        # Calculate likelihood for each parent set
        likelihoods = {}
        total_likelihood = 0.0

        for parents in self.prior_probabilities:
            if parents not in likelihoods:
                # Calculate likelihood P(Y|X,pa(Y))
                likelihood = self.calculate_likelihood(x_dict, y, parents)
                likelihoods[parents] = likelihood
                total_likelihood += likelihood * \
                    self.prior_probabilities[parents]

        # Update posterior using Bayes' rule
        if total_likelihood > 0:
            for parents in self.prior_probabilities:
                self.prior_probabilities[parents] = (
                    likelihoods[parents] *
                    self.prior_probabilities[parents] / total_likelihood
                )

    def calculate_likelihood(self, x_dict, y, parents):
        """Calculate likelihood P(Y|X,pa(Y)) for a particular parent set."""
        # Default implementation - override in subclasses
        return 1.0

    def update_probabilities(self, parent_sets):
        """Update probabilities ensuring they sum to 1."""
        total_prob = sum(
            self.prior_probabilities[parent] for parent in parent_sets)
        if total_prob > 0:
            for parent in parent_sets:
                self.prior_probabilities[parent] /= total_prob


class LinearSCMModel(SCMModel):
    """Linear SCM model for posterior updates."""

    def calculate_likelihood(self, x_dict, y, parents):
        """Calculate likelihood using a linear model."""
        if len(parents) == 0:
            # For empty parent set, use variance of y in data
            if self.graph and self.graph.target in self.data:
                var = np.var(self.data[self.graph.target])
                if var == 0:
                    var = 1.0
                return 1.0 / np.sqrt(2 * np.pi * var)
            return 0.1

        # Get parent values
        x_values = np.array([x_dict.get(parent, 0)
                            for parent in parents]).reshape(1, -1)

        # Extract data for these parents
        if all(parent in self.data for parent in parents):
            # Use data to compute weights and variance
            X = np.hstack([self.data[parent].reshape(-1, 1)
                          for parent in parents])
            Y = self.data[self.graph.target].reshape(-1, 1)

            # Fit linear regression
            try:
                # Use pseudo-inverse for numerical stability
                weights = np.linalg.pinv(X.T @ X) @ X.T @ Y

                # Predict and compute residual variance
                y_pred = X @ weights
                residuals = Y - y_pred
                variance = np.var(residuals)
                if variance == 0:
                    variance = 0.1

                # Predict for new data point
                y_pred_new = x_values @ weights

                # Calculate likelihood using Gaussian density
                likelihood = np.exp(-0.5 * ((y - y_pred_new)**2) /
                                    variance) / np.sqrt(2 * np.pi * variance)
                return float(likelihood)
            except:
                return 0.1

        return 0.1


class NonLinearSCMModel(SCMModel):
    """Non-linear SCM model for posterior updates using kernel density estimation."""

    def calculate_likelihood(self, x_dict, y, parents):
        """Calculate likelihood using kernel density estimation."""
        if len(parents) == 0:
            # For empty parent set, use variance of y in data
            if self.graph and self.graph.target in self.data:
                var = np.var(self.data[self.graph.target])
                if var == 0:
                    var = 1.0
                return 1.0 / np.sqrt(2 * np.pi * var)
            return 0.1

        # Get parent values
        x_values = np.array([x_dict.get(parent, 0) for parent in parents])

        # Extract data for these parents
        if all(parent in self.data for parent in parents):
            try:
                # Use data to compute kernel density estimate
                X = np.hstack([self.data[parent].reshape(-1, 1)
                              for parent in parents])
                Y = self.data[self.graph.target].reshape(-1, 1)

                # Simple KDE implementation
                bandwidth = 0.1  # Could be optimized
                distances = np.sum((X - x_values.reshape(1, -1))**2, axis=1)
                weights = np.exp(-distances / (2 * bandwidth**2))
                weights /= weights.sum()

                # Weighted mean and variance
                mean = weights @ Y
                variance = weights @ ((Y - mean)**2)
                if variance < 1e-6:
                    variance = 0.1

                # Calculate likelihood using Gaussian density
                likelihood = np.exp(-0.5 * ((y - mean)**2) /
                                    variance) / np.sqrt(2 * np.pi * variance)
                return float(likelihood)
            except:
                return 0.1

        return 0.1


class PARENT_SCALE:
    """Implementation of the PARENT_SCALE algorithm using the new graphs framework."""

    def __init__(
        self,
        graph: Union[GraphStructure, nx.DiGraph],
        nonlinear: bool = True,
        causal_prior: bool = True,
        noiseless: bool = True,
        cost_num: int = 1,
        scale_data: bool = True,
        individual: bool = False,
        use_doubly_robust: bool = True,
        use_iscm: bool = False,
        num_samples: int = 1000,
        seed: Optional[int] = None,
    ):
        # Don't create adapter for now to avoid GraphStructure initialization issues
        self.graph_is_structure = isinstance(graph, GraphStructure)

        # Set basic properties
        if self.graph_is_structure:
            self.graph = graph
            self.graph_nx = graph.G if hasattr(graph, 'G') else None
            self.variables = graph.variables if hasattr(
                graph, 'variables') else None
            self.target = graph.target if hasattr(graph, 'target') else None
        else:
            self.graph = None  # Will set up after obtaining observational data
            self.graph_nx = graph
            # Get variables from nx.DiGraph nodes
            self.variables = [str(n) for n in graph.nodes()]
            # Set target as the last node (convention)
            self.target = self.variables[-1]

        self.num_nodes = len(
            self.variables) if self.variables else len(graph.nodes())
        self.nonlinear = nonlinear

        # Set up buffer for storage
        self.buffer = ReplayBuffer(binary=True)

        # Get manipulative variables (all except target)
        self.manipulative_variables = [
            v for v in self.variables if v != self.target]

        # Store other options
        self.causal_prior = causal_prior
        self.noiseless = noiseless
        self.cost_num = cost_num
        self.scale_data = scale_data
        self.individual = individual
        self.use_doubly_robust = use_doubly_robust
        self.use_iscm = use_iscm
        self.seed = seed

        # Initialize empty data structures
        self.D_O = None
        self.D_I = None
        self.exploration_set = None

    def set_values(self, D_O, D_I, exploration_set):
        """Set the observational/interventional data and exploration set."""
        self.D_O = deepcopy(D_O)
        self.D_I = deepcopy(D_I)
        if self.graph_is_structure and hasattr(self.graph, 'set_interventional_range_data'):
            self.graph.set_interventional_range_data(self.D_O)
        self.topological_order = list(self.D_O.keys())
        self.exploration_set = exploration_set
        self.es_to_n_mapping = {
            tuple(es): i for i, es in enumerate(self.exploration_set)
        }

    def determine_initial_probabilities(self) -> Dict[Tuple, float]:
        """Determine initial probabilities for parent sets."""
        # Consider all individual parent nodes
        variables = [var for var in self.variables if var != self.target]

        # Create all possible parent set combinations (up to a reasonable limit)
        combinations = []
        max_parents = min(4, len(variables))
        for r in range(1, max_parents + 1):
            combinations.extend(itertools.combinations(variables, r))

        # Add empty parent set
        combinations = list(combinations) + [()]

        # Uniform prior over all combinations
        probabilities = {combo: 1.0 / len(combinations)
                         for combo in combinations}
        return probabilities

    def standardize_all_data(self):
        """Standardize the dataset."""
        input_keys = [key for key in self.D_O.keys() if key != self.target]
        self.means = {key: np.mean(self.D_O[key]) for key in input_keys}
        self.stds = {key: np.std(self.D_O[key]) for key in input_keys}

        # Standardize observational data
        D_O_scaled = {}
        for key in self.D_O:
            if key in input_keys:
                D_O_scaled[key] = (
                    self.D_O[key] - self.means[key]) / self.stds[key]
            else:
                D_O_scaled[key] = self.D_O[key]

        # Standardize interventional data
        interventions = self.D_I.keys()
        D_I_scaled = {intervention: {} for intervention in interventions}
        for intervention in interventions:
            for key in self.D_I[intervention]:
                if key in input_keys:
                    D_I_scaled[intervention][key] = (
                        self.D_I[intervention][key] - self.means[key]) / self.stds[key]
                else:
                    D_I_scaled[intervention][key] = self.D_I[intervention][key]

        self.D_O_scaled = D_O_scaled
        self.D_I_scaled = D_I_scaled

        # Set up observational samples for use in algorithm
        self.observational_samples = {
            var: self.D_O[var] for var in self.variables}

    def define_all_possible_graphs(self, error_tol=1e-5):
        """Define all possible graphs based on prior probabilities."""
        self.graphs = {}
        self.posterior = []
        for parents in self.prior_probabilities:
            if self.prior_probabilities[parents] < error_tol:
                continue

            # Create graph representation
            parents_graph = {}
            parents_graph['target'] = self.target
            parents_graph['parents'] = parents
            parents_graph['probability'] = self.prior_probabilities[parents]

            self.graphs[parents] = parents_graph
            self.posterior.append(self.prior_probabilities[parents])

    def redefine_all_possible_graphs(self, error_tol=1e-4):
        """Redefine graphs based on updated prior probabilities."""
        self.posterior = []
        parents_to_remove = []
        for parents in self.prior_probabilities:
            if self.prior_probabilities[parents] < error_tol:
                parents_to_remove.append(parents)
                continue
            self.posterior.append(self.prior_probabilities[parents])

        for parents in parents_to_remove:
            if parents in self.graphs:
                del self.graphs[parents]

    def data_and_prior_setup(self):
        """Set up data and priors for the algorithm."""
        # Normalize datasets for posterior probability calculations
        self.standardize_all_data()

        # Set initial probabilities
        self.prior_probabilities = self.determine_initial_probabilities()

        # Create SCM model for posterior updates
        if self.nonlinear:
            self.posterior_model = NonLinearSCMModel(
                self.prior_probabilities, self.graph)
        else:
            self.posterior_model = LinearSCMModel(
                self.prior_probabilities, self.graph)

        # Set initial data
        self.posterior_model.set_data(self.D_O_scaled)

        # Create graphs based on priors
        self.define_all_possible_graphs()

        # Add interventional data to posterior
        for intervention in self.D_I_scaled:
            D_I_sample = self.D_I_scaled[intervention]
            num_samples = len(D_I_sample[self.target])
            for n in range(num_samples):
                x_dict = {
                    obs_key: D_I_sample[obs_key][n]
                    for obs_key in D_I_sample
                    if obs_key != self.target
                }
                y = D_I_sample[self.target][n]
                self.posterior_model.update_all(x_dict, y)

        # Update prior probabilities with posterior
        self.prior_probabilities = self.posterior_model.prior_probabilities.copy()

    def evaluate_causal_effect(self, intervention_vars, values):
        """Evaluate estimated causal effect of intervention."""
        # Calculate weighted average across parent sets
        effect = 0.0
        total_prob = 0.0

        # For each parent set, calculate effect and weight by probability
        for parents, graph_data in self.graphs.items():
            prob = self.prior_probabilities[parents]

            # If probability is too small, skip
            if prob < 1e-4:
                continue

            # Calculate effect for this parent set
            # For simplicity, here we just use the linear effect model
            if self.target in self.D_O and all(var in self.D_O for var in parents):
                # Get data for target and parents
                y = self.D_O[self.target]

                if not parents:  # Empty parent set
                    # Use mean as prediction
                    effect_i = np.mean(y)
                else:
                    # Fit simple linear model
                    X = np.hstack([self.D_O[parent].reshape(-1, 1)
                                  for parent in parents])
                    Y = y.reshape(-1, 1)

                    try:
                        # Calculate weights using least squares
                        weights = np.linalg.lstsq(X, Y, rcond=None)[0]

                        # Get intervention values for relevant parents
                        int_values = []
                        for parent in parents:
                            if parent in intervention_vars:
                                idx = intervention_vars.index(parent)
                                int_values.append(values[idx])
                            else:
                                # Use mean value for non-intervened parents
                                int_values.append(np.mean(self.D_O[parent]))

                        # Predict effect
                        int_values = np.array(int_values).reshape(-1, 1)
                        effect_i = float(int_values.T @ weights)
                    except:
                        effect_i = np.mean(y)
            else:
                effect_i = 0.0

            # Add weighted effect
            effect += prob * effect_i
            total_prob += prob

        # Normalize
        if total_prob > 0:
            effect /= total_prob

        return effect

    def propose_intervention(self):
        """Propose the next intervention based on current knowledge."""
        best_intervention = None
        best_value = None
        best_expected_effect = float('-inf')

        # Sample candidate interventions
        for es in self.exploration_set:
            # Get intervention ranges
            int_ranges = {}
            for var in es:
                if hasattr(self.graph, 'get_interventional_range'):
                    int_range = self.graph.get_interventional_range()[var]
                else:
                    # Use range of observed data
                    data = self.D_O[var]
                    int_range = (np.min(data), np.max(data))
                int_ranges[var] = int_range

            # Sample candidate values
            n_candidates = 10
            for _ in range(n_candidates):
                values = []
                for var in es:
                    low, high = int_ranges[var]
                    values.append(np.random.uniform(low, high))

                # Evaluate expected effect
                expected_effect = self.evaluate_causal_effect(es, values)

                # Update best intervention
                if expected_effect > best_expected_effect:
                    best_expected_effect = expected_effect
                    best_intervention = es
                    best_value = values

        return best_intervention, best_value, best_expected_effect

    def perform_intervention(self, intervention_vars, values):
        """Perform intervention and update model with results."""
        # Log the intervention
        if len(intervention_vars) == 1:
            logging.info(
                f"Performing intervention {intervention_vars[0]}={values[0]}")
        else:
            logging.info(
                f"Performing intervention on {intervention_vars} with values {values}")

        # Create intervention dictionary
        intervention = {var: val for var,
                        val in zip(intervention_vars, values)}

        # Sample from the SCM with intervention
        if hasattr(self.graph, 'SEM'):
            # Sample using the SEM
            samples = sample_model(
                self.graph.SEM,
                interventions=intervention,
                sample_count=10,
                graph=self.graph
            )

            # Get target value
            target_value = np.mean(samples[self.target])

            # Update data
            int_key = tuple(intervention_vars)
            int_value = tuple(values)

            if int_key not in self.D_I:
                self.D_I[int_key] = {}
            if int_value not in self.D_I[int_key]:
                self.D_I[int_key][int_value] = {}

            # Add samples to interventional data
            for var in samples:
                if var not in self.D_I[int_key][int_value]:
                    self.D_I[int_key][int_value][var] = samples[var]
                else:
                    self.D_I[int_key][int_value][var] = np.concatenate(
                        [self.D_I[int_key][int_value][var], samples[var]]
                    )

            # Update posterior with new data
            x_dict = {var: values[i]
                      for i, var in enumerate(intervention_vars)}
            self.posterior_model.update_all(x_dict, target_value)

            # Update prior probabilities
            self.prior_probabilities = self.posterior_model.prior_probabilities.copy()

            # Redefine graphs
            self.redefine_all_possible_graphs()

            return target_value
        else:
            # Simplified approach
            if self.target in self.D_O:
                # Use mean as estimate
                target_value = np.mean(self.D_O[self.target])

                # Update posterior slightly
                for parents in self.prior_probabilities:
                    if all(p in intervention_vars for p in parents):
                        # Increase probability of parent sets that include the intervention
                        self.prior_probabilities[parents] *= 1.1

                # Normalize
                total = sum(self.prior_probabilities.values())
                for parents in self.prior_probabilities:
                    self.prior_probabilities[parents] /= total

                return target_value
            return 0.0

    def find_best_parents(self):
        """Find the most likely parent set."""
        if not self.prior_probabilities:
            return ()

        return max(self.prior_probabilities.items(), key=lambda x: x[1])[0]

    def run_algorithm(self, T=30, show_graphics=False):
        """Run the PARENT_SCALE algorithm for T iterations."""
        logging.info("Starting PARENT_SCALE algorithm")

        # Initial setup
        self.data_and_prior_setup()

        # Results tracking
        current_y = []
        global_opt = []
        current_cost = []
        intervention_set = []
        intervention_values = []

        # Cost tracking
        total_cost = 0.0

        # Run the algorithm for T iterations
        for t in range(T):
            logging.info(f"Starting iteration {t+1}/{T}")

            # Propose intervention
            int_vars, int_values, expected_effect = self.propose_intervention()

            # Perform intervention
            target_value = self.perform_intervention(int_vars, int_values)

            # Track results
            current_y.append(target_value)
            intervention_set.append(int_vars)
            intervention_values.append(int_values)

            # Update global optimum
            if not global_opt:
                global_opt.append(target_value)
            else:
                global_opt.append(max(global_opt[-1], target_value))

            # Calculate cost
            if hasattr(self.graph, 'get_cost_structure'):
                cost_functions = self.graph.get_cost_structure(self.cost_num)
                step_cost = sum(cost_functions[var](val)
                                for var, val in zip(int_vars, int_values))
            else:
                step_cost = 1.0  # Default cost

            total_cost += step_cost
            current_cost.append(total_cost)

            # Log progress
            logging.info(
                f"Iteration {t+1} complete. Target value: {target_value}")
            logging.info(
                f"Current best parent set: {self.find_best_parents()}")

        logging.info("PARENT_SCALE algorithm completed")

        return global_opt, current_y, current_cost, intervention_set, intervention_values, self.prior_probabilities
