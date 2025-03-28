import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

from algorithms.BASE_algorithm import BASE
from graphs.graph import GraphStructure
from utils.acd_models import GraphEncoder, DynamicsDecoder
from utils.sem_sampling import change_intervention_list_format, sample_model
from utils.cbo_classes import TargetClass


class PARENT_SCALE_ACD(BASE):
    """
    Adaptation of PARENT_SCALE algorithm using Amortized Causal Discovery.
    Uses neural networks to predict causal structure and intervention outcomes.
    """

    def __init__(
        self,
        graph: GraphStructure,
        encoder_path: Optional[str] = None,
        decoder_path: Optional[str] = None,
        noiseless: bool = True,
        cost_num: int = 1,
        scale_data: bool = True,
        device: str = "cpu",
    ):
        # Store graph information
        self.graph = graph
        self.num_nodes = len(self.graph.variables)
        self.variables = self.graph.variables
        self.target = self.graph.target
        self.manipulative_variables = self.graph.get_sets()[2]

        # Configuration parameters
        self.noiseless = noiseless
        self.cost_num = cost_num
        self.scale_data = scale_data
        self.device = torch.device(device)

        # Load neural network models
        self.encoder = self._load_encoder(encoder_path)
        self.decoder = self._load_decoder(decoder_path)

        # Set models to evaluation mode
        self.encoder.eval()
        self.decoder.eval()

        # Attributes to be initialized later
        self.D_O = None
        self.D_I = None
        self.edge_probabilities = None
        self.exploration_set = None
        self.es_to_n_mapping = None
        self.means = None
        self.stds = None

    def _load_encoder(self, path: Optional[str]) -> GraphEncoder:
        """Load the pre-trained encoder model or initialize a new one."""
        encoder = GraphEncoder(n_variables=self.num_nodes)
        if path is not None:
            try:
                encoder.load_state_dict(torch.load(
                    path, map_location=self.device))
                logging.info(f"Loaded encoder model from {path}")
            except FileNotFoundError:
                logging.warning(
                    f"Encoder model not found at {path}, using new model")
        encoder.to(self.device)
        return encoder

    def _load_decoder(self, path: Optional[str]) -> DynamicsDecoder:
        """Load the pre-trained decoder model or initialize a new one."""
        decoder = DynamicsDecoder(n_variables=self.num_nodes)
        if path is not None:
            try:
                decoder.load_state_dict(torch.load(
                    path, map_location=self.device))
                logging.info(f"Loaded decoder model from {path}")
            except FileNotFoundError:
                logging.warning(
                    f"Decoder model not found at {path}, using new model")
        decoder.to(self.device)
        return decoder

    def set_values(self, D_O, D_I, exploration_set=None):
        """Initialize with observational and interventional data."""
        self.D_O = deepcopy(D_O)
        self.D_I = deepcopy(D_I)
        self.graph.set_interventional_range_data(self.D_O)
        self.topological_order = list(self.D_O.keys())

        # Set exploration set first if provided
        if exploration_set is not None:
            self.exploration_set = exploration_set
        else:
            # Initialize with all manipulative variables
            self.exploration_set = [
                [var] for var in self.manipulative_variables if var != self.target]

        # Create mapping for exploration set
        self.es_to_n_mapping = {
            tuple(es): i for i, es in enumerate(self.exploration_set)
        }

        # Standardize data if needed
        if self.scale_data:
            self.standardize_all_data()

        # Infer causal structure using the encoder
        self.infer_causal_structure()

    def standardize_all_data(self):
        """Standardize input data for better model performance."""
        input_keys = [key for key in self.D_O.keys() if key !=
                      self.graph.target]
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
        self.D_O_scaled = D_O_scaled

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
        self.D_I_scaled = D_I_scaled

        # Prepare tensor for the encoder
        self.observational_samples = np.hstack(
            [self.D_O_scaled[var] for var in self.variables])
        self.observational_tensor = torch.tensor(
            self.observational_samples, dtype=torch.float32).to(self.device)

        # Convert interventional data for the algorithm
        self.interventional_samples = change_intervention_list_format(
            self.D_I_scaled, self.exploration_set, target=self.graph.target
        )

    def infer_causal_structure(self):
        """Infer causal structure using the neural encoder."""
        with torch.no_grad():
            self.edge_probabilities = self.encoder(self.observational_tensor)

        # Extract parent probabilities for the target variable
        target_idx = self.variables.index(self.target)
        self.parent_probabilities = self.edge_probabilities[:, target_idx].cpu(
        ).numpy()

        # Create dictionary of parent sets with their probabilities
        self.prior_probabilities = {}
        for i, var in enumerate(self.variables):
            if var != self.target and self.parent_probabilities[i] > 0.1:
                self.prior_probabilities[(var,)] = self.parent_probabilities[i]

        # Normalize probabilities
        total_prob = sum(self.prior_probabilities.values())
        if total_prob > 0:
            for parent_set in self.prior_probabilities:
                self.prior_probabilities[parent_set] /= total_prob
        else:
            # If no significant parents found, use uniform distribution
            for var in self.variables:
                if var != self.target:
                    self.prior_probabilities[(
                        var,)] = 1.0 / (len(self.variables) - 1)

        logging.info(
            f"Inferred parent probabilities: {self.prior_probabilities}")

    def determine_exploration_set(self, threshold=0.1, max_interventions=3):
        """Determine exploration set based on inferred causal structure."""
        target_idx = self.variables.index(self.target)

        # Get direct effects to target
        direct_effects = self.edge_probabilities[:, target_idx].cpu().numpy()

        # Filter to include only manipulative variables
        parent_scores = []
        for i, var in enumerate(self.variables):
            if var in self.manipulative_variables and var != self.target:
                parent_scores.append((direct_effects[i], var))

        # Sort by probability (direct effect strength)
        parent_scores.sort(reverse=True)

        # Select top individual interventions
        self.exploration_set = []
        for prob, var in parent_scores:
            if prob > threshold:
                self.exploration_set.append([var])
                if len(self.exploration_set) >= max_interventions:
                    break

        # If no variables selected, include all manipulative variables
        if not self.exploration_set:
            self.exploration_set = [[var]
                                    for var in self.manipulative_variables]

        logging.info(f"Determined exploration set: {self.exploration_set}")

    def predict_intervention_outcome(self, intervention_vars, intervention_values):
        """Predict intervention outcome using the decoder."""
        # Convert input to tensor format and ensure it has batch dimension
        if isinstance(intervention_values, np.ndarray):
            values_tensor = torch.tensor(
                intervention_values, dtype=torch.float32).to(self.device)
        else:
            values_tensor = intervention_values.to(self.device)

        # Ensure values_tensor has shape [batch_size, num_intervention_vars]
        if len(values_tensor.shape) == 1:
            values_tensor = values_tensor.unsqueeze(0)

        # Create a full input tensor with all variables
        batch_size = values_tensor.shape[0]
        full_input = torch.zeros(
            (batch_size, self.num_nodes), device=self.device)

        # Fill in the intervened variables
        for i, var in enumerate(intervention_vars):
            var_idx = self.variables.index(var)
            full_input[:, var_idx] = values_tensor[:, i]

        # Fill in non-intervened variables with their mean values
        for i, var in enumerate(self.variables):
            if var not in intervention_vars:
                mean_val = torch.tensor(
                    np.mean(self.D_O_scaled[var]), dtype=torch.float32).to(self.device)
                full_input[:, i] = mean_val

        # Create intervention mask
        intervention_mask = torch.zeros(
            self.num_nodes, dtype=torch.bool).to(self.device)
        for var in intervention_vars:
            var_idx = self.variables.index(var)
            intervention_mask[var_idx] = True

        # Make prediction
        with torch.no_grad():
            predicted = self.decoder(
                full_input, self.edge_probabilities, intervention_mask)

        # Return predicted outcome for target variable
        target_idx = self.variables.index(self.target)
        pred_target = predicted[:, target_idx]

        # Ensure output has correct shape
        if len(pred_target.shape) == 0:
            pred_target = pred_target.unsqueeze(0)
        if len(pred_target.shape) == 1:
            pred_target = pred_target.unsqueeze(1)

        return pred_target.cpu().numpy()

    def update_models(self, intervention_vars, intervention_values, observed_outcome, learning_rate=0.001, num_iterations=5):
        """Update encoder and decoder models with new interventional data."""
        # Convert to tensors and ensure correct shapes
        values_tensor = torch.tensor(
            intervention_values, dtype=torch.float32).to(self.device)
        if len(values_tensor.shape) == 1:
            values_tensor = values_tensor.unsqueeze(0)

        outcome_tensor = torch.tensor(
            observed_outcome, dtype=torch.float32).to(self.device)
        if len(outcome_tensor.shape) == 1:
            outcome_tensor = outcome_tensor.unsqueeze(1)

        # Create a full input tensor
        batch_size = values_tensor.shape[0]
        full_input = torch.zeros(
            (batch_size, self.num_nodes), device=self.device)

        # Fill in the intervened variables
        for i, var in enumerate(intervention_vars):
            var_idx = self.variables.index(var)
            full_input[:, var_idx] = values_tensor[:, i]

        # Fill in non-intervened variables with their means
        for i, var in enumerate(self.variables):
            if var not in intervention_vars:
                mean_val = torch.tensor(
                    np.mean(self.D_O_scaled[var]), dtype=torch.float32).to(self.device)
                full_input[:, i] = mean_val

        # Create intervention mask
        intervention_mask = torch.zeros(
            self.num_nodes, dtype=torch.bool).to(self.device)
        for var in intervention_vars:
            var_idx = self.variables.index(var)
            intervention_mask[var_idx] = True

        # Set models to training mode
        self.encoder.train()
        self.decoder.train()

        # Create optimizers
        encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = torch.optim.Adam(
            self.decoder.parameters(), lr=learning_rate)

        # Save initial edge probabilities for KL divergence
        initial_edge_probs = self.edge_probabilities.clone().detach()

        # Training loop
        for _ in range(num_iterations):
            # Forward pass through encoder
            new_edge_probs = self.encoder(self.observational_tensor)

            # Forward pass through decoder
            predicted = self.decoder(
                full_input, new_edge_probs, intervention_mask)

            # Calculate reconstruction loss
            target_idx = self.variables.index(self.target)
            pred_target = predicted[:, target_idx]

            # Ensure pred_target and outcome_tensor have the same shape
            if len(pred_target.shape) == 1:
                pred_target = pred_target.unsqueeze(1)

            reconstruction_loss = F.mse_loss(pred_target, outcome_tensor)

            # Calculate KL divergence to prevent drastic changes to the graph
            kl_divergence = F.kl_div(
                F.log_softmax(new_edge_probs.view(-1), dim=0),
                F.softmax(initial_edge_probs.view(-1), dim=0),
                reduction='batchmean'
            )

            # Total loss with regularization
            loss = reconstruction_loss + 0.1 * kl_divergence

            # Backpropagation
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

        # Set models back to evaluation mode
        self.encoder.eval()
        self.decoder.eval()

        # Update edge probabilities
        with torch.no_grad():
            self.edge_probabilities = self.encoder(self.observational_tensor)

        # Update parent probabilities
        target_idx = self.variables.index(self.target)
        self.parent_probabilities = self.edge_probabilities[:, target_idx].cpu(
        ).numpy()

        # Update prior probabilities
        self.prior_probabilities = {}
        for i, var in enumerate(self.variables):
            if var != self.target and self.parent_probabilities[i] > 0.1:
                self.prior_probabilities[(var,)] = self.parent_probabilities[i]

        # Normalize
        total_prob = sum(self.prior_probabilities.values())
        if total_prob > 0:
            for parent_set in self.prior_probabilities:
                self.prior_probabilities[parent_set] /= total_prob

    def evaluate_acquisition(self, intervention_vars, candidate_values, current_best):
        """Evaluate acquisition function for candidate intervention values."""
        # Predict outcomes
        predicted_values = self.predict_intervention_outcome(
            intervention_vars, candidate_values)

        # Ensure predicted_values has shape [batch_size, 1]
        if len(predicted_values.shape) == 1:
            predicted_values = predicted_values.reshape(-1, 1)

        # Convert current_best to numpy array with correct shape
        if isinstance(current_best, (int, float)):
            current_best = np.array([[current_best]])
        elif len(current_best.shape) == 1:
            current_best = current_best.reshape(-1, 1)

        # Calculate expected improvement
        if self.graph.task == "min":
            improvement = current_best - predicted_values
        else:
            improvement = predicted_values - current_best

        # Zero out negative improvements
        improvement[improvement < 0] = 0

        # Calculate intervention cost
        costs = np.zeros(len(candidate_values))
        cost_functions = self.graph.get_cost_structure(self.cost_num)
        for i, var in enumerate(intervention_vars):
            for j in range(len(candidate_values)):
                costs[j] += cost_functions[var](candidate_values[j, i])

        # Add small constant to avoid division by zero
        costs = costs + 1e-6

        # Calculate acquisition value (expected improvement per unit cost)
        # Ensure costs has same shape as improvement for division
        costs = costs.reshape(-1, 1)
        acquisition_values = improvement / costs

        # Find best candidate
        best_idx = np.argmax(acquisition_values.flatten())
        best_value = candidate_values[best_idx:best_idx+1]
        best_acq = acquisition_values[best_idx].item()

        return best_acq, best_value

    def run_algorithm(self, T=30, show_graphics=False, file=None):
        """Run optimization algorithm for T iterations."""
        # Prepare data
        if self.D_O is None or self.D_I is None:
            raise ValueError("Must call set_values before running algorithm")

        if self.exploration_set is None:
            self.determine_exploration_set()

        # Initialize tracking arrays
        global_opt = []
        current_y = []
        current_cost = [0.0]
        intervention_set = []
        intervention_values = []

        # Create target classes for actual interventions
        target_classes = [
            TargetClass(
                self.graph.SEM,
                es,
                self.variables,
                graph=self.graph,
                noiseless=self.noiseless
            )
            for es in self.exploration_set
        ]

        # Initialize with best value from observational data
        current_global_min = np.mean(self.D_O[self.target])
        global_opt.append(current_global_min)

        # Generate candidate intervention values for each exploration set
        intervention_domain = self.graph.get_interventional_range()
        candidate_values = []
        for es in self.exploration_set:
            n_vars = len(es)
            n_candidates = 50  # Number of candidates to evaluate
            candidates = np.empty((n_candidates, n_vars))
            for i, var in enumerate(es):
                min_val, max_val = intervention_domain[var]
                candidates[:, i] = np.linspace(min_val, max_val, n_candidates)
            candidate_values.append(candidates)

        # Main optimization loop
        for i in range(T):
            logging.info(f"---------------ITERATION {i+1}/{T}---------------")

            # Evaluate acquisition for each intervention set
            best_acqs = []
            best_values = []

            for j, es in enumerate(self.exploration_set):
                acq, value = self.evaluate_acquisition(
                    es, candidate_values[j], current_global_min
                )
                best_acqs.append(acq)
                best_values.append(value)

            # Select best intervention
            best_idx = np.argmax(best_acqs)
            best_es = self.exploration_set[best_idx]
            best_value = best_values[best_idx]

            logging.info(
                f"Selected intervention: {best_es} with value {best_value[0]}")

            # Perform intervention
            y_new = target_classes[best_idx].compute_target(
                best_value).reshape(-1)

            logging.info(f"Intervention outcome: {y_new[0]}")

            # Update tracking data
            current_y.append(y_new[0])
            intervention_set.append(tuple(best_es))
            intervention_values.append(tuple(best_value[0]))

            # Update best value
            if self.graph.task == "min":
                current_global_min = min(current_global_min, y_new[0])
            else:
                current_global_min = max(current_global_min, y_new[0])

            global_opt.append(current_global_min)

            # Update cost
            cost = 0
            cost_functions = self.graph.get_cost_structure(self.cost_num)
            for j, var in enumerate(best_es):
                cost += cost_functions[var](best_value[0, j])

            current_cost.append(current_cost[i] + cost)

            # Update models
            self.update_models(best_es, best_value, y_new.reshape(1, -1))

            # Update exploration set based on new edge probabilities
            self.determine_exploration_set()

            # Visualize current state if requested
            if show_graphics:
                self.visualize_edge_probabilities(
                    save_path=f"{file}_edges_iter_{i+1}.png" if file else None
                )

        return (
            global_opt,
            current_y,
            current_cost,
            intervention_set,
            intervention_values,
            None  # No average uncertainty measure
        )

    def visualize_edge_probabilities(self, save_path=None):
        """Visualize inferred causal structure as a heatmap."""
        import matplotlib.pyplot as plt

        # Convert to numpy for plotting
        edge_probs = self.edge_probabilities.cpu().numpy()

        plt.figure(figsize=(10, 8))
        plt.imshow(edge_probs, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(label='Edge Probability')
        plt.xticks(range(len(self.variables)), self.variables, rotation=90)
        plt.yticks(range(len(self.variables)), self.variables)
        plt.title('Inferred Causal Edge Probabilities')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def estimate_uncertainty(self, method="dropout", dropout_runs=10):
        """Estimate uncertainty in the causal structure prediction."""
        if method == "dropout":
            # Enable dropout during inference
            def enable_dropout(model):
                for module in model.modules():
                    if isinstance(module, nn.Dropout):
                        module.train()

            enable_dropout(self.encoder)

            # Run multiple forward passes
            edge_probs_samples = []
            with torch.no_grad():
                for _ in range(dropout_runs):
                    edge_probs = self.encoder(self.observational_tensor)
                    edge_probs_samples.append(edge_probs)

            # Calculate standard deviation as uncertainty measure
            edge_probs_tensor = torch.stack(edge_probs_samples)
            uncertainty = torch.std(edge_probs_tensor, dim=0)

            # Set model back to eval mode
            self.encoder.eval()

            return uncertainty.cpu().numpy()
        else:
            # Return basic uncertainty estimate based on edge probability itself
            # Uncertainty is highest at p=0.5 and lowest at p=0 or p=1
            with torch.no_grad():
                uncertainty = 0.25 - (self.edge_probabilities - 0.5).pow(2)
            return uncertainty.cpu().numpy()
