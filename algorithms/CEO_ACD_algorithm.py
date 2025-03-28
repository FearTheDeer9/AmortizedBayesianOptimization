import logging
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import matplotlib.pyplot as plt

from algorithms.BASE_algorithm import BASE
from config import SHOW_GRAPHICS
from graphs.graph import GraphStructure
from utils.acd_models import GraphEncoder, DynamicsDecoder
from utils.sem_sampling import change_intervention_list_format, sample_model
from utils.cbo_classes import TargetClass


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    filename="logfile.log",
    filemode="w",
)


class CEO_ACD(BASE):
    """
    Causal Entropy Optimization with Amortized Causal Discovery.
    Uses pre-trained encoder-decoder models to infer causal structure and predict intervention outcomes.
    """

    def __init__(
        self,
        graph_type: str = "Toy",
        graph: GraphStructure = None,
        encoder_path: str = "models/encoder.pt",
        decoder_path: str = "models/decoder.pt",
        cost_num: int = 1,
        task: str = "min",
        noiseless: bool = True,
        n_anchor_points: int = 35,
        allow_fine_tuning: bool = False,
        device: str = "cpu",
    ):
        self._graph_type = graph_type
        self.noiseless = noiseless
        self.task = task
        self.n_anchor_points = n_anchor_points
        self.device = torch.device(device)
        self.allow_fine_tuning = allow_fine_tuning

        # Load reference graph (for obtaining variable information and ranges)
        if graph is not None:
            self.reference_graph = graph
        else:
            assert graph_type in ["Toy", "Synthetic",
                                  "Graph6", "Graph5", "Graph4"]
            self.reference_graph = self.chosen_structure()

        # Get basic information from the reference graph
        self.target = self.reference_graph.target
        self.variables = self.reference_graph.variables
        self.manipulative_variables = self.reference_graph.get_sets()[2]
        self.cost_num = cost_num
        self.cost_functions = self.reference_graph.get_cost_structure(cost_num)

        # Load pre-trained encoder and decoder models
        self.encoder = self._load_encoder(encoder_path)
        self.decoder = self._load_decoder(decoder_path)

        # Set models to evaluation mode unless fine-tuning is allowed
        if not allow_fine_tuning:
            self.encoder.eval()
            self.decoder.eval()

        # Will be set after observational data is provided
        self.exploration_set = None
        self.edge_probabilities = None
        self.observational_features = None
        self.es_to_n_mapping = None

    def _load_encoder(self, path: str) -> GraphEncoder:
        """Load the pre-trained encoder model."""
        # If path is None, create a new model
        if path is None:
            encoder = GraphEncoder(n_variables=len(self.variables))
            encoder.to(self.device)
            return encoder

        try:
            encoder = GraphEncoder(n_variables=len(self.variables))
            encoder.load_state_dict(torch.load(path, map_location=self.device))
            encoder.to(self.device)
            return encoder
        except FileNotFoundError:
            logging.warning(
                f"Encoder model not found at {path}. Initializing new model.")
            encoder = GraphEncoder(n_variables=len(self.variables))
            encoder.to(self.device)
            return encoder

    def _load_decoder(self, path: str) -> DynamicsDecoder:
        """Load the pre-trained decoder model."""
        # If path is None, create a new model
        if path is None:
            decoder = DynamicsDecoder(n_variables=len(self.variables))
            decoder.to(self.device)
            return decoder

        try:
            decoder = DynamicsDecoder(n_variables=len(self.variables))
            decoder.load_state_dict(torch.load(path, map_location=self.device))
            decoder.to(self.device)
            return decoder
        except FileNotFoundError:
            logging.warning(
                f"Decoder model not found at {path}. Initializing new model.")
            decoder = DynamicsDecoder(n_variables=len(self.variables))
            decoder.to(self.device)
            return decoder

    def set_values(self, D_O: Dict, D_I: Dict, exploration_set: Optional[List[List[str]]] = None):
        """Set observational and interventional data and infer initial causal structure."""
        logging.info("Setting up data and inferring causal structure")
        self.D_O = D_O
        self.D_I = D_I

        # Convert observational data to model-compatible format
        self.observational_samples = np.hstack(
            [self.D_O[var] for var in self.variables])
        self.observational_features = torch.tensor(
            self.observational_samples, dtype=torch.float32
        ).to(self.device)

        # Infer causal structure using the encoder
        with torch.no_grad():
            self.edge_probabilities = self.encoder(self.observational_features)

        # Determine exploration set based on edge probabilities to target
        if exploration_set is None:
            self.determine_exploration_set()
        else:
            self.exploration_set = exploration_set

        # Create mapping for exploration set
        self.es_to_n_mapping = {
            tuple(es): i for i, es in enumerate(self.exploration_set)
        }

        # Format interventional data
        self.interventional_samples = change_intervention_list_format(
            self.D_I, self.exploration_set, target=self.target
        )

        # Initialize exploration distribution (uniform)
        self.arm_distribution = np.array(
            [1 / len(self.exploration_set)] * len(self.exploration_set)
        )

    def determine_exploration_set(self, n_interventions=3):
        """
        Determine exploration set based on edge probabilities to target.
        Selects interventions based only on direct causal effects (immediate parents).
        """
        target_idx = self.variables.index(self.target)

        # Get direct effects (immediate parents)
        direct_effects = self.edge_probabilities[:, target_idx].cpu()

        # Filter to only include manipulative variables
        manipulative_indices = [self.variables.index(
            var) for var in self.manipulative_variables]

        # Create list of (probability, variable_index) pairs
        parent_scores = [(direct_effects[i].item(), i)
                         for i in manipulative_indices]

        # Sort by probability (direct effect strength)
        parent_scores.sort(reverse=True)

        # Select top individual interventions
        exploration_set = []
        for prob, idx in parent_scores:
            if prob > 0.1:  # Only consider variables with meaningful effect probability
                exploration_set.append([self.variables[idx]])
                if len(exploration_set) >= n_interventions:
                    break

        # If no variables selected, include all manipulative variables
        if not exploration_set:
            exploration_set = [[var] for var in self.manipulative_variables]

        logging.info(f"Determined exploration set: {exploration_set}")
        self.exploration_set = exploration_set

    def predict_intervention_outcome(self, intervention_vars: List[str], intervention_values: np.ndarray) -> np.ndarray:
        """Predict the outcome of an intervention using the decoder."""
        # Convert to tensor
        values_tensor = torch.tensor(
            intervention_values, dtype=torch.float32).to(self.device)

        # Create a full input tensor with all variables
        full_input = torch.zeros(
            (values_tensor.shape[0], len(self.variables)), device=self.device)

        # Fill in the intervened variables
        for i, var in enumerate(intervention_vars):
            var_idx = self.variables.index(var)
            full_input[:, var_idx] = values_tensor[:, i]

        # Fill in non-intervened variables with their current values
        for var in self.variables:
            if var not in intervention_vars:
                var_idx = self.variables.index(var)
                full_input[:, var_idx] = self.observational_features[:,
                                                                     var_idx].mean()

        # Create intervention mask
        intervention_mask = torch.zeros(
            len(self.variables), dtype=torch.bool).to(self.device)
        for var in intervention_vars:
            var_idx = self.variables.index(var)
            intervention_mask[var_idx] = True

        # Make prediction
        with torch.no_grad():
            predicted_outcome = self.decoder(
                full_input,
                self.edge_probabilities,
                intervention_mask
            )

        # Return predicted target value
        target_idx = self.variables.index(self.target)
        return predicted_outcome[:, target_idx].cpu().numpy()

    def evaluate_acquisition(self, es: Tuple[str], candidate_values: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Evaluate acquisition function for a given intervention.
        Balances exploration (causal discovery) and exploitation (optimization).
        """
        # Predict outcomes for candidate values
        predicted_values = self.predict_intervention_outcome(
            es, candidate_values)

        # Get target variable index
        target_idx = self.variables.index(self.target)

        # Calculate exploitation component (predicted improvement)
        if self.task == "min":
            exploitation = -predicted_values
        else:
            exploitation = predicted_values

        # Calculate exploration component (uncertainty in causal structure)
        # Higher value = more uncertainty
        uncertainty = 0.5 - torch.abs(
            self.edge_probabilities[:, target_idx] - 0.5).mean().item()

        # Get cost of intervention
        cost = 0
        for i, var in enumerate(es):
            cost += self.cost_functions[var](candidate_values[0, i])

        # Combine components
        acquisition_values = exploitation / (cost + 1e-6) + 0.1 * uncertainty

        # Find best candidate
        best_idx = np.argmax(acquisition_values)
        best_value = candidate_values[best_idx:best_idx+1]
        best_acq = acquisition_values[best_idx]

        return best_acq, best_value

    def update_models(self, intervention_vars: List[str], intervention_values: np.ndarray, observed_outcome: np.ndarray):
        """Optional fine-tuning of models based on intervention outcomes."""
        if not self.allow_fine_tuning:
            return

        # Prepare data for fine-tuning
        values_tensor = torch.tensor(
            intervention_values, dtype=torch.float32).to(self.device)
        outcome_tensor = torch.tensor(
            observed_outcome, dtype=torch.float32).to(self.device)

        # Create a full input tensor with all variables
        full_input = torch.zeros(
            (values_tensor.shape[0], len(self.variables)), device=self.device)

        # Fill in the intervened variables
        for i, var in enumerate(intervention_vars):
            var_idx = self.variables.index(var)
            full_input[:, var_idx] = values_tensor[:, i]

        # Fill in non-intervened variables with their current values
        for var in self.variables:
            if var not in intervention_vars:
                var_idx = self.variables.index(var)
                full_input[:, var_idx] = self.observational_features[:,
                                                                     var_idx].mean()

        # Create intervention mask
        intervention_mask = torch.zeros(
            len(self.variables), dtype=torch.bool).to(self.device)
        for var in intervention_vars:
            var_idx = self.variables.index(var)
            intervention_mask[var_idx] = True

        # Set to training mode
        self.encoder.train()
        self.decoder.train()

        # Create optimizers
        encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=0.001)
        decoder_optimizer = torch.optim.Adam(
            self.decoder.parameters(), lr=0.001)

        # Fine-tuning iterations
        for _ in range(5):  # Limited fine-tuning iterations
            # Get edge probabilities
            edge_probs = self.encoder(self.observational_features)

            # Make prediction
            predicted = self.decoder(
                full_input, edge_probs, intervention_mask)

            # Calculate loss
            target_idx = self.variables.index(self.target)
            pred_target = predicted[:, target_idx]
            loss = torch.nn.functional.mse_loss(pred_target, outcome_tensor)

            # Update models
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

        # Set back to evaluation mode
        self.encoder.eval()
        self.decoder.eval()

        # Update edge probabilities
        with torch.no_grad():
            self.edge_probabilities = self.encoder(self.observational_features)

    def visualize_edge_probabilities(self, save_path: Optional[str] = None):
        """Visualize current edge probabilities as a heatmap."""
        if self.edge_probabilities is None:
            logging.warning("No edge probabilities available to visualize.")
            return

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

    def run_algorithm(self, T: int = 30, file: str = None):
        """Run the optimization algorithm for T iterations."""
        logging.info(f"Starting CEO_ACD optimization for {T} iterations")

        # Set up result tracking
        best_y_array = [np.mean(self.D_O[self.target])]
        current_y_array = []
        cost_array = np.zeros(shape=T + 1)
        intervention_set = []
        intervention_values = []

        # Create target classes for computing actual interventions
        target_classes = [
            TargetClass(
                self.reference_graph.SEM,
                es,
                self.variables,
                graph=self.reference_graph,
                noiseless=self.noiseless
            )
            for es in self.exploration_set
        ]

        # Initialize data structures
        data_x_list = [np.empty((0, len(es))) for es in self.exploration_set]
        data_y_list = [np.empty(0) for _ in self.exploration_set]

        # Add initial interventional data if available
        for i, es in enumerate(self.exploration_set):
            es_tuple = tuple(es)
            if es_tuple in self.interventional_samples:
                x_data = self.interventional_samples[es_tuple]['X']
                y_data = self.interventional_samples[es_tuple]['Y']
                data_x_list[i] = np.vstack((data_x_list[i], x_data))
                data_y_list[i] = np.concatenate(
                    (data_y_list[i], y_data.reshape(-1)))

        # Generate candidate intervention values
        intervention_domain = self.reference_graph.get_interventional_range()
        candidate_values = []
        for es in self.exploration_set:
            n_vars = len(es)
            candidates = np.empty((self.n_anchor_points, n_vars))
            for i, var in enumerate(es):
                min_val, max_val = intervention_domain[var]
                candidates[:, i] = np.linspace(
                    min_val, max_val, self.n_anchor_points)
            candidate_values.append(candidates)

        # Main optimization loop
        for i in range(T):
            logging.info(f"Iteration {i+1}/{T}")

            # Evaluate acquisition for each intervention in exploration set
            best_acqs = []
            best_values = []

            for j, es in enumerate(self.exploration_set):
                acq, value = self.evaluate_acquisition(
                    tuple(es), candidate_values[j])
                best_acqs.append(acq)
                best_values.append(value)

            # Select best intervention
            best_idx = np.argmax(best_acqs)
            best_es = self.exploration_set[best_idx]
            best_value = best_values[best_idx]

            logging.info(
                f"Selected intervention: {best_es} with value {best_value[0]}")

            # Perform intervention (or simulate it)
            y_new = target_classes[best_idx].compute_target(
                best_value).reshape(-1)

            # Update tracking data
            data_x_list[best_idx] = np.vstack(
                (data_x_list[best_idx], best_value))
            data_y_list[best_idx] = np.concatenate(
                (data_y_list[best_idx], y_new))

            current_y_array.append(y_new[0])
            intervention_set.append(tuple(best_es))
            intervention_values.append(tuple(best_value[0]))

            # Update best value found
            if self.task == "min":
                best_y = min(best_y_array[-1], y_new[0])
            else:
                best_y = max(best_y_array[-1], y_new[0])
            best_y_array.append(best_y)

            # Update cost
            current_cost = 0
            for j, var in enumerate(best_es):
                current_cost += self.cost_functions[var](best_value[0, j])
            cost_array[i+1] = cost_array[i] + current_cost

            # Update models (if fine-tuning is enabled)
            self.update_models(best_es, best_value, y_new.reshape(1, -1))

            # Visualize current state if requested
            if SHOW_GRAPHICS:
                self.visualize_edge_probabilities(
                    save_path=f"{file}_edges_iter_{i+1}.png" if file else None
                )

        return (
            best_y_array,
            current_y_array,
            cost_array,
            intervention_set,
            intervention_values,
            None  # No uncertainty measure in this implementation
        )
