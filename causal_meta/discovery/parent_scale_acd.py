"""
Implementation of the PARENT_SCALE_ACD algorithm adapted for the causal_meta framework.

This module provides a causal discovery algorithm that uses GNN-based models
to infer causal structure and guide interventions for optimizing a target variable.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Any, Set, Union, Callable

from torch_geometric.data import Data, Batch
import networkx as nx

from causal_meta.graph import CausalGraph
from causal_meta.environments.scm import StructuralCausalModel
from causal_meta.environments.interventions import PerfectIntervention
from causal_meta.inference.models.encoder import BaseGNNEncoder
from causal_meta.inference.models.decoder import BaseGNNDecoder
# Assuming BaseGNNEncoder and BaseGNNDecoder are the correct base classes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PARENT_SCALE_ACD:
    """
    PARENT_SCALE algorithm adapted for Amortized Causal Discovery (ACD) within the causal_meta framework.

    This algorithm uses pre-trained GNN encoder and decoder models to infer causal structure
    and predict intervention outcomes to guide the selection of interventions for optimizing
    a target variable based on causal insights.
    """

    def __init__(
        self,
        target_node: str,
        encoder: BaseGNNEncoder,
        decoder: BaseGNNDecoder,
        variables: List[str],
        manipulative_variables: Optional[List[str]] = None,
        # Ground truth SCM for simulation
        scm: Optional[StructuralCausalModel] = None,
        task: str = "min",  # Optimization task: "min" or "max"
        cost_functions: Optional[Dict[str, Callable[[Any], float]]] = None,
        scale_data: bool = True,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Initialize the PARENT_SCALE_ACD algorithm.

        Args:
            target_node: The name of the target variable to optimize.
            encoder: A pre-trained GNN encoder instance (subclass of BaseGNNEncoder).
            decoder: A pre-trained GNN decoder instance (subclass of BaseGNNDecoder).
                     Assumed capable of predicting outcomes or node features representing outcomes.
            variables: List of all variable names in the system.
            manipulative_variables: List of variable names that can be intervened upon.
                                    If None, all variables except target are considered manipulative.
            scm: The ground truth StructuralCausalModel for simulating interventions. Required for run_algorithm.
            task: The optimization goal ("min" or "max").
            cost_functions: Dictionary mapping variable names to cost functions for intervention.
                            If None, assumes unit cost.
            scale_data: Whether to standardize observational data (recommended).
            device: The device ('cpu' or 'cuda') to run computations on.
        """
        if target_node not in variables:
            raise ValueError(
                f"Target node '{target_node}' not found in variable list.")
        if task not in ["min", "max"]:
            raise ValueError("Task must be either 'min' or 'max'.")

        self.target_node = target_node
        # Ensure model is on device and in eval mode
        self.encoder = encoder.to(device).eval()
        # Ensure model is on device and in eval mode
        self.decoder = decoder.to(device).eval()
        self.variables = list(variables)
        self.num_nodes = len(variables)
        self.variable_indices = {name: i for i, name in enumerate(variables)}
        self.scm = scm  # Store the SCM for simulation
        self.task = task
        self.cost_functions = cost_functions or {
            var: lambda x: 1.0 for var in variables}
        self.scale_data = scale_data
        self.device = torch.device(device)

        if manipulative_variables is None:
            self.manipulative_variables = [
                v for v in variables if v != target_node]
        else:
            if not all(v in variables for v in manipulative_variables):
                raise ValueError(
                    "All manipulative variables must be in the variable list.")
            self.manipulative_variables = list(manipulative_variables)

        # Attributes to be initialized by set_data
        self.observational_data: Optional[Data] = None
        self.observational_df: Optional[pd.DataFrame] = None
        # Store interventional data if needed later, though not used in current logic
        self.interventional_data: Optional[Dict[Tuple[str], Data]] = None
        self.node_embeddings: Optional[torch.Tensor] = None
        # Adjacency matrix format [N, N]
        self.edge_probabilities: Optional[torch.Tensor] = None
        # Probabilities of being a parent of target [N]
        self.parent_probabilities: Optional[np.ndarray] = None
        # Normalized parent set probabilities
        self.prior_probabilities: Optional[Dict[Tuple[str], float]] = None
        self.exploration_set: Optional[List[List[str]]] = None
        self.means: Optional[Dict[str, float]] = None
        self.stds: Optional[Dict[str, float]] = None

        logger.info(
            f"Initialized PARENT_SCALE_ACD for target '{target_node}' on device '{device}'.")
        logger.info(f"Manipulative variables: {self.manipulative_variables}")

    # --- Helper Function for Simple Edge Index Estimation (Internal) ---
    def _estimate_edge_index_simple(
        self,
        obs_df: pd.DataFrame,
        variables: List[str],
        corr_threshold: float = 0.1
    ) -> Optional[torch.Tensor]:
        """Estimates graph edges based on correlation matrix and returns edge_index.

        Internal helper method used within set_data if ground truth is unavailable.

        Args:
            obs_df: DataFrame containing observational data.
            variables: List of all variable names in the desired order.
            corr_threshold: Absolute correlation threshold to consider an edge.

        Returns:
            A torch.LongTensor of shape [2, num_edges] representing the
            estimated edge_index, or None if no edges meet the threshold.
        """
        logger.info(
            f"Estimating edge_index using correlation (threshold: {corr_threshold})...")
        if obs_df is None or obs_df.empty:
            logger.warning(
                "Observational DataFrame is missing or empty. Cannot estimate edge_index.")
            return None

        # Ensure columns match self.variables order
        if not all(v in obs_df.columns for v in variables):
            logger.warning(
                "DataFrame columns mismatch variables. Cannot estimate edge_index reliably.")
            return None  # Or try with available columns?

        corr_matrix = obs_df[variables].corr().abs()
        edges = []
        variable_indices = {name: i for i, name in enumerate(variables)}

        for i_idx, i_var in enumerate(variables):
            for j_idx, j_var in enumerate(variables):
                if i_idx == j_idx:
                    continue
                correlation = corr_matrix.iloc[i_idx, j_idx]
                if not np.isnan(correlation) and correlation > corr_threshold:
                    edges.append(
                        (variable_indices[i_var], variable_indices[j_var]))

        if not edges:
            logger.warning(
                f"No edges found above correlation threshold {corr_threshold}. GCN might fail.")
            return None

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        logger.info(
            f"Estimated {edge_index.shape[1]} edges based on correlation.")
        return edge_index.to(self.device)  # Move to the correct device
    # ------------------------------------------------------------------

    def set_data(self,
                 observational_data: Union[pd.DataFrame, Data],
                 interventional_data: Optional[Dict[Tuple[str], Union[pd.DataFrame, Data]]] = None) -> None:
        """
        Set the observational (and optionally interventional) data for the algorithm.
        Infers the causal structure based on the observational data.

        Args:
            observational_data: Observational data as a pandas DataFrame or PyG Data object.
                                If DataFrame, columns must match self.variables.
                                If Data object, data.x should contain node features [num_samples, num_nodes].
            interventional_data: Optional dictionary mapping intervention targets (tuples)
                                 to interventional data (DataFrame or Data). Not used in current logic.
        """
        logger.info("Setting data and inferring initial causal structure...")

        # --- Prepare DataFrame and Data object --- #
        if isinstance(observational_data, pd.DataFrame):
            if not all(v in observational_data.columns for v in self.variables):
                raise ValueError(
                    "Observational DataFrame columns must match algorithm variables.")
            self.observational_df = observational_data.copy()
            data_np = self.observational_df[self.variables].values
            self.observational_data = Data(
                x=torch.tensor(data_np, dtype=torch.float32)
            ).to(self.device)
        elif isinstance(observational_data, Data):
            self.observational_data = observational_data.to(self.device)
            if self.observational_data.x is not None:
                self.observational_df = pd.DataFrame(
                    self.observational_data.x.cpu().numpy(), columns=self.variables)
            else:
                logger.warning(
                    "Input Data object has no features (x). Cannot create DataFrame.")
                self.observational_df = None
        else:
            raise TypeError(
                "observational_data must be a pandas DataFrame or PyG Data object.")

        self.interventional_data = interventional_data  # Store if provided

        # --- Standardize Data (using the DataFrame) --- #
        if self.scale_data and self.observational_df is not None:
            self._standardize_data()
            # Update the PyG Data object's features (x) with scaled data
            scaled_data_np = self.observational_df_scaled[self.variables].values
            self.observational_data.x = torch.tensor(
                scaled_data_np, dtype=torch.float32).to(self.device)
        elif self.observational_data.x is not None:
            # Ensure x is on the correct device even if not scaling
            self.observational_data.x = self.observational_data.x.to(
                self.device)

        # --- Estimate or Set Edge Index --- #
        # Try to estimate edge_index using the simple correlation helper
        # Requires the DataFrame to be available
        estimated_edge_index = self._estimate_edge_index_simple(
            self.observational_df, self.variables
        )

        if estimated_edge_index is not None:
            self.observational_data.edge_index = estimated_edge_index.long()
            logger.info("Using estimated edge_index for GNN encoder.")
        elif hasattr(self.observational_data, 'edge_index') and self.observational_data.edge_index is not None:
            # Fallback: Use edge_index if it was already present in an input Data object
            self.observational_data.edge_index = self.observational_data.edge_index.long()
            logger.warning(
                "Using edge_index provided with input Data object (estimation failed/not possible). GCNEncoder might fail if inadequate.")
        else:
            # Critical fallback: Assume fully connected if no other option
            logger.warning(
                "Edge index estimation failed and none provided. Assuming fully connected graph for GCNEncoder.")
            num_nodes = len(self.variables)
            adj = torch.ones(num_nodes, num_nodes, dtype=torch.long)
            adj.fill_diagonal_(0)  # No self-loops
            self.observational_data.edge_index = adj.nonzero().t().contiguous().to(self.device)

        # Ensure edge_index is LongTensor if it exists
        if hasattr(self.observational_data, 'edge_index') and self.observational_data.edge_index is not None:
            self.observational_data.edge_index = self.observational_data.edge_index.long()

        # --- Infer Structure using GNN --- #
        self._infer_structure()
        # Initial determination based on inferred structure
        self.determine_exploration_set()
        logger.info("Data set and initial structure inference complete.")

    def _standardize_data(self) -> None:
        """Standardize observational data (mean=0, std=1)."""
        if self.observational_df is None:
            logger.warning(
                "Cannot standardize data without observational DataFrame.")
            return

        logger.debug("Standardizing observational data...")
        # Standardize only non-target variables? Original code did this.
        input_keys = [key for key in self.variables if key != self.target_node]
        self.means = {
            key: self.observational_df[key].mean() for key in input_keys}
        self.stds = {
            key: self.observational_df[key].std() for key in input_keys}

        self.observational_df_scaled = self.observational_df.copy()
        for key in input_keys:
            if self.stds[key] > 1e-8:  # Avoid division by zero
                self.observational_df_scaled[key] = (
                    self.observational_df[key] - self.means[key]) / self.stds[key]
            else:
                # Just center if std is zero
                self.observational_df_scaled[key] = self.observational_df[key] - \
                    self.means[key]

        # Note: Interventional data scaling is omitted as it's not directly used in this refactored version.
        logger.debug("Data standardization complete.")

    def _infer_structure(self) -> None:
        """
        Infer causal structure using the GNN encoder and decoder.
        Assumes encoder provides node embeddings and decoder predicts edges.
        """
        if self.observational_data is None or self.observational_data.x is None:
            raise ValueError(
                "Observational data (with features 'x') must be set before inferring structure.")

        logger.debug("Inferring structure from observational data...")
        with torch.no_grad():
            # *** Crucial Assumption: Encoder's forward pass returns node embeddings ***
            # The BaseGNNEncoder interface only guarantees returning the final latent representation.
            # We might need to modify the encoder or assume it has a method like `get_node_embeddings`.
            # Let's assume `encoder.forward` CAN return node embeddings before pooling if needed,
            # or that the decoder can work directly with the final latent representation.
            # The GCNDecoder expects a latent representation, so we'll pass the encoder's output.
            latent_representation = self.encoder(self.observational_data)

            # *** Another Assumption: Decoder predicts edges from latent representation ***
            # The GCNDecoder's `predict_edges` takes `node_embeddings`.
            # How do we get node embeddings from the `latent_representation`?
            # Option 1: Decoder's `predict_node_features` -> `predict_edges`
            # Option 2: Modify encoder to return node embeddings.
            # Let's try Option 1, using the decoder to get node features first.

            # Get node features (potentially refined based on latent context)
            # Need to handle potential batch dimension if encoder returns [batch_size, latent_dim]
            # even for single graph input. Let's assume it returns [1, latent_dim] or [latent_dim].
            if latent_representation.dim() == 1:
                latent_representation = latent_representation.unsqueeze(
                    0)  # Add batch dim

            # Assuming observational_data represents a single graph context
            num_obs_nodes = self.observational_data.num_nodes
            if num_obs_nodes != self.num_nodes:
                logger.warning(
                    f"Observational data has {num_obs_nodes} nodes, but model expects {self.num_nodes}. Structure inference might be inaccurate.")
                # Decide how to handle this: error out, or proceed with caution?
                # For now, proceed, but this indicates a potential issue.

            # Predict node features using the decoder
            # We need to tell the decoder how many nodes to generate features for.
            # Let's assume predict_node_features can handle this based on self.num_nodes
            # or an argument. GCNDecoder has num_nodes in __init__.
            node_features = self.decoder.predict_node_features(
                latent_representation)  # Shape: [batch, N, feat_dim]

            # Now predict edges using these node features
            # predict_edges expects [N, feat_dim] or [batch, N, feat_dim]
            edge_index, edge_attr = self.decoder.predict_edges(
                node_features)  # edge_attr are probs

            # Convert edge_index and edge_attr to an adjacency matrix of probabilities
            adj_matrix = torch.zeros(
                (self.num_nodes, self.num_nodes), device=self.device)
            # Handle potential batch dimension in edge_index/edge_attr if decoder returns batched output
            # Assuming edge_index is [2, num_edges] and edge_attr is [num_edges] for single graph inference
            src, dst = edge_index
            # Ensure indices are within bounds BEFORE assignment
            valid_mask = (src < self.num_nodes) & (dst < self.num_nodes)
            if not valid_mask.all():
                logger.warning(
                    f"Decoder predicted edge indices out of bounds (max={self.num_nodes-1}). Clamping.")
                src = torch.clamp(src, 0, self.num_nodes - 1)
                dst = torch.clamp(dst, 0, self.num_nodes - 1)
                # Only keep valid edges after clamping - or maybe filter based on original mask?
                # Let's filter based on original mask to be safe
                src = src[valid_mask]
                dst = dst[valid_mask]
                edge_attr = edge_attr[valid_mask]

            adj_matrix[src, dst] = edge_attr
            # Store the probability matrix [N, N]
            self.edge_probabilities = adj_matrix

            # Extract parent probabilities for the target variable
            target_idx = self.variable_indices[self.target_node]
            # Probabilities P(parent -> target) correspond to column `target_idx`
            self.parent_probabilities = self.edge_probabilities[:, target_idx].cpu(
            ).numpy()

            # Create dictionary of parent sets with their probabilities (prior for exploration)
            self.prior_probabilities = {}
            for i, var in enumerate(self.variables):
                # Consider only manipulative variables as potential parents to intervene on
                if var != self.target_node and var in self.manipulative_variables:
                    # Check probability P(var -> target)
                    prob = self.parent_probabilities[i]
                    if prob > 0.1:  # Threshold from original code
                        # Store P(var -> target)
                        self.prior_probabilities[(var,)] = prob

            # Normalize probabilities among potential parents
            total_prob = sum(self.prior_probabilities.values())
            if total_prob > 1e-6:
                for parent_set in self.prior_probabilities:
                    self.prior_probabilities[parent_set] /= total_prob
            else:
                # If no significant parents found, use uniform distribution over manipulative vars
                num_manipulative = len(self.manipulative_variables)
                if num_manipulative > 0:
                    uniform_prob = 1.0 / num_manipulative
                    for var in self.manipulative_variables:
                        if var != self.target_node:
                            self.prior_probabilities[(var,)] = uniform_prob
                logger.warning(
                    "No significant parents found via structure inference. Using uniform prior.")

            logger.info(
                f"Inferred parent probabilities (P(parent -> {self.target_node})): {self.prior_probabilities}")

    def determine_exploration_set(self, threshold: float = 0.1, max_interventions: int = 3) -> None:
        """
        Determine the set of single-node interventions to explore based on inferred parent probabilities.

        Args:
            threshold: Minimum probability P(parent -> target) to consider a variable for exploration.
            max_interventions: Maximum number of single-variable interventions to include.
        """
        if self.parent_probabilities is None:
            raise RuntimeError(
                "Must infer structure before determining exploration set.")

        parent_scores = []
        for i, var in enumerate(self.variables):
            # Only consider manipulative variables that are not the target
            if var != self.target_node and var in self.manipulative_variables:
                parent_scores.append((self.parent_probabilities[i], var))

        # Sort by probability P(parent -> target) in descending order
        parent_scores.sort(key=lambda x: x[0], reverse=True)

        # Select top individual interventions based on threshold and max_interventions
        self.exploration_set = []
        for prob, var in parent_scores:
            if prob >= threshold:
                # Store as list containing single var
                self.exploration_set.append([var])
                if len(self.exploration_set) >= max_interventions:
                    break

        # Fallback: if no variables meet threshold, explore all manipulative variables
        if not self.exploration_set:
            logger.warning(
                f"No potential parents met threshold {threshold}. Exploring all manipulative variables.")
            self.exploration_set = [
                [var] for var in self.manipulative_variables if var != self.target_node]
            # Limit by max_interventions even in fallback
            self.exploration_set = self.exploration_set[:max_interventions]

        # Ensure exploration set is not empty if there are manipulative variables
        if not self.exploration_set and any(v != self.target_node for v in self.manipulative_variables):
            logger.warning(
                "Exploration set is empty, adding the first manipulative variable as fallback.")
            first_manip = next(
                (v for v in self.manipulative_variables if v != self.target_node), None)
            if first_manip:
                self.exploration_set = [[first_manip]]

        logger.info(f"Determined exploration set: {self.exploration_set}")

    def predict_intervention_outcome(self,
                                     intervention_vars: List[str],
                                     intervention_values: np.ndarray) -> np.ndarray:
        """
        Predict the outcome of the target variable under intervention using SCM simulation.

        Args:
            intervention_vars: List of variables being intervened on.
            intervention_values: Numpy array of intervention values [num_samples, num_intervention_vars].

        Returns:
            Numpy array of predicted target variable outcomes [num_samples, 1].
        """
        if self.scm is None:
            raise ValueError(
                "SCM instance must be provided for simulating interventions.")
        if not all(var in self.variables for var in intervention_vars):
            raise ValueError(
                "Intervention variables must be in the model's variable list.")
        if intervention_values.shape[1] != len(intervention_vars):
            raise ValueError(
                "Shape mismatch between intervention_vars and intervention_values.")

        num_samples = intervention_values.shape[0]
        predicted_outcomes = np.zeros((num_samples, 1))

        # Simulate each intervention sample
        for i in range(num_samples):
            # Create intervention dictionary for this sample
            intervention_dict = {var: intervention_values[i, j]
                                 for j, var in enumerate(intervention_vars)}

            # Apply intervention(s) to a copy of the SCM
            intervened_scm = self.scm
            for var, val in intervention_dict.items():
                intervention = PerfectIntervention(target_node=var, value=val)
                # Apply sequentially to the copy
                intervened_scm = intervention.apply(intervened_scm)

            # Sample the outcome from the intervened SCM (only need 1 sample per intervention setting)
            # We assume the SCM is deterministic or we want the expected outcome.
            # SCM's sample_data might be stochastic. Let's sample once and take the target value.
            # TODO: Clarify if multiple samples are needed for expectation over noise. Assuming 1 sample is sufficient.
            try:
                outcome_sample_df = intervened_scm.sample_data(sample_size=1)
                predicted_outcomes[i,
                                   0] = outcome_sample_df[self.target_node].iloc[0]
            except Exception as e:
                logger.error(
                    f"Error during SCM sampling for intervention {intervention_dict}: {e}")
                # Handle error: e.g., return NaN or a default value
                predicted_outcomes[i, 0] = np.nan

        return predicted_outcomes

    def evaluate_acquisition(self,
                             intervention_vars: List[str],
                             candidate_values: np.ndarray,
                             current_best: float) -> Tuple[float, np.ndarray]:
        """
        Evaluate the acquisition function (Expected Improvement per unit Cost) for candidate intervention values.

        Args:
            intervention_vars: List of variables being intervened on.
            candidate_values: Numpy array of candidate intervention values [num_candidates, num_intervention_vars].
            current_best: The current best observed value of the target variable.

        Returns:
            Tuple containing:
                - best_acq (float): The highest acquisition value found.
                - best_value (np.ndarray): The intervention values [1, num_intervention_vars] corresponding to the best acquisition.
        """
        logger.debug(
            f"Evaluating acquisition for intervention on {intervention_vars}...")
        num_candidates = candidate_values.shape[0]

        # Predict outcomes for all candidates
        predicted_outcomes = self.predict_intervention_outcome(
            intervention_vars, candidate_values)  # Shape: [num_candidates, 1]

        # Calculate expected improvement (EI)
        if self.task == "min":
            improvement = current_best - predicted_outcomes
        else:  # task == "max"
            improvement = predicted_outcomes - current_best
        improvement[improvement < 0] = 0  # Clip negative improvement

        # Calculate intervention cost for each candidate
        costs = np.zeros(num_candidates)
        for i, var in enumerate(intervention_vars):
            cost_func = self.cost_functions.get(
                var, lambda x: 1.0)  # Default cost 1
            # Apply cost function element-wise if candidate_values has multiple rows
            var_costs = np.array([cost_func(val)
                                 for val in candidate_values[:, i]])
            costs += var_costs

        # Add epsilon to avoid division by zero
        costs = costs + 1e-8

        # Calculate acquisition value (EI per unit cost)
        # Ensure costs shape matches improvement shape for division
        costs = costs.reshape(-1, 1)  # Shape: [num_candidates, 1]
        acquisition_values = improvement / costs

        # Handle potential NaNs from prediction or division
        # Replace NaN with very low value
        acquisition_values = np.nan_to_num(acquisition_values, nan=-np.inf)

        # Find best candidate
        if acquisition_values.size == 0:
            logger.warning(
                f"No valid acquisition values calculated for {intervention_vars}. Returning 0.")
            best_idx = 0
            best_acq = 0.0
            # Return a default value matching the expected shape
            best_value = candidate_values[best_idx:best_idx +
                                          1] if num_candidates > 0 else np.array([[]])
        else:
            best_idx = np.argmax(acquisition_values.flatten())
            best_acq = acquisition_values[best_idx].item()
            # Keep shape [1, num_intervention_vars]
            best_value = candidate_values[best_idx:best_idx+1]

        logger.debug(
            f"Best acquisition for {intervention_vars}: {best_acq:.4f} at values {best_value}")
        return best_acq, best_value

    def run_algorithm(self,
                      T: int = 30,
                      num_candidates_per_var: int = 50) -> Tuple[List[float], List[float], List[float], List[Tuple[str]], List[Tuple[Any]]]:
        """
        Run the PARENT_SCALE_ACD optimization algorithm for T iterations.

        Args:
            T: Number of optimization iterations (interventions).
            num_candidates_per_var: Number of candidate values to evaluate per variable dimension.

        Returns:
            Tuple containing:
                - global_opt (List[float]): History of the best target value found so far.
                - current_y (List[float]): History of the observed target values after each intervention.
                - current_cost (List[float]): History of the cumulative intervention cost.
                - intervention_set (List[Tuple[str]]): History of the intervened variable tuples.
                - intervention_values (List[Tuple[Any]]): History of the intervention value tuples.
        """
        if self.observational_data is None or self.scm is None:
            raise ValueError(
                "Must call set_data and provide an SCM before running algorithm.")
        if self.exploration_set is None:
            logger.warning(
                "Exploration set not determined. Running determine_exploration_set().")
            self.determine_exploration_set()  # Ensure it's set
        if not self.exploration_set:
            logger.error(
                "Cannot run algorithm: Exploration set is empty and no manipulative variables available.")
            return [], [], [0.0], [], []

        logger.info(
            f"Starting PARENT_SCALE_ACD optimization for {T} iterations...")

        # --- Initialization ---
        global_opt_history = []
        observed_y_history = []
        cumulative_cost_history = [0.0]
        intervention_set_history = []
        intervention_values_history = []

        # Initial best value from observational data (use mean)
        if self.observational_df is not None:
            initial_best_y = self.observational_df[self.target_node].mean()
        else:
            logger.warning(
                "No observational DataFrame available, cannot initialize best_y from data. Using 0.")
            initial_best_y = 0.0  # Fallback

        current_best_y = initial_best_y
        global_opt_history.append(current_best_y)
        logger.info(
            f"Initial best target value (observational mean): {current_best_y:.4f}")

        # --- Main Optimization Loop ---
        for i in range(T):
            logger.info(f"--- Iteration {i+1}/{T} ---")

            # Generate candidate values for each intervention set in the current exploration_set
            intervention_candidates: Dict[Tuple[str], np.ndarray] = {}
            # Assuming SCM has this method or similar
            # intervention_domain = self.scm.get_variable_domains()
            # Access the protected attribute directly
            intervention_domain = self.scm._variable_domains

            for es_vars in self.exploration_set:  # es_vars is List[str]
                n_vars = len(es_vars)
                candidates = np.empty(
                    (num_candidates_per_var ** n_vars, n_vars))

                # Create grid of candidates
                linspaces = []
                for var in es_vars:
                    # Get domain/range for intervention - requires SCM or graph info
                    # Placeholder: assume range [0, 1] if not defined
                    domain = intervention_domain.get(var, None)
                    min_val, max_val = 0.0, 1.0  # Default range
                    if isinstance(domain, (tuple, list)) and len(domain) == 2:
                        min_val, max_val = domain
                    elif isinstance(domain, str) and domain == "binary":
                        min_val, max_val = 0, 1
                    elif isinstance(domain, str) and domain == "continuous":
                        # Need a sensible default range for continuous
                        logger.warning(
                            f"Using default range [0, 1] for continuous variable {var}")
                    linspaces.append(np.linspace(
                        min_val, max_val, num_candidates_per_var))

                # Create meshgrid
                mesh = np.meshgrid(*linspaces, indexing='ij')
                for var_idx in range(n_vars):
                    candidates[:, var_idx] = mesh[var_idx].flatten()

                intervention_candidates[tuple(es_vars)] = candidates

            # Evaluate acquisition function for each exploration set
            best_overall_acq = -np.inf
            best_overall_es = None
            best_overall_value = None

            for es_tuple, candidates in intervention_candidates.items():
                es_list = list(es_tuple)
                try:
                    acq, value = self.evaluate_acquisition(
                        es_list, candidates, current_best_y
                    )
                    if acq > best_overall_acq:
                        best_overall_acq = acq
                        best_overall_es = es_list
                        # Shape [1, num_intervention_vars]
                        best_overall_value = value
                except Exception as e:
                    logger.error(
                        f"Error evaluating acquisition for {es_list}: {e}", exc_info=True)

            # Check if a valid intervention was found
            if best_overall_es is None or best_overall_value is None:
                logger.error(
                    "Failed to find a valid intervention in this iteration. Stopping.")
                break  # Stop algorithm if no intervention can be selected

            logger.info(
                f"Selected intervention: {best_overall_es} with value {best_overall_value[0]} (Acq: {best_overall_acq:.4f})")

            # Perform the selected intervention using SCM simulation
            # predict_intervention_outcome already does the simulation
            # We need the outcome for the *single* best value chosen
            y_new_array = self.predict_intervention_outcome(
                best_overall_es, best_overall_value)
            y_new = y_new_array[0, 0]  # Extract scalar outcome

            if np.isnan(y_new):
                logger.error(
                    f"SCM simulation returned NaN for intervention {best_overall_es}={best_overall_value}. Stopping.")
                break

            logger.info(f"Observed outcome: {y_new:.4f}")

            # --- Update State ---
            observed_y_history.append(y_new)
            intervention_set_history.append(tuple(best_overall_es))
            # Store the chosen value tuple
            intervention_values_history.append(tuple(best_overall_value[0]))

            # Update best value found so far
            if self.task == "min":
                current_best_y = min(current_best_y, y_new)
            else:  # task == "max"
                current_best_y = max(current_best_y, y_new)
            global_opt_history.append(current_best_y)

            # Update cost
            cost = 0.0
            for j, var in enumerate(best_overall_es):
                cost_func = self.cost_functions.get(var, lambda x: 1.0)
                cost += cost_func(best_overall_value[0, j])
            cumulative_cost_history.append(cumulative_cost_history[-1] + cost)

            # Note: Model update step (`update_models`) is removed in this refactoring.
            # Structure inference is done once at the beginning.
            # If online structure refinement is needed, it requires further design.

            # Optionally: Re-determine exploration set if needed (e.g., based on uncertainty)
            # self.determine_exploration_set() # Keep fixed for now based on initial inference

        logger.info(
            f"PARENT_SCALE_ACD finished. Final best value: {current_best_y:.4f}")
        return (
            global_opt_history,
            observed_y_history,
            cumulative_cost_history,
            intervention_set_history,
            intervention_values_history
        )

    def get_inferred_structure(self) -> Optional[torch.Tensor]:
        """Returns the inferred edge probability matrix."""
        return self.edge_probabilities

    def get_parent_prior(self) -> Optional[Dict[Tuple[str], float]]:
        """Returns the normalized prior probability for potential parent sets."""
        return self.prior_probabilities
