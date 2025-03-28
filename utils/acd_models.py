import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphEncoder(nn.Module):
    """
    Encoder that predicts causal edge probabilities from observational data.
    Designed to handle graphs of any size through a more general architecture.
    """

    def __init__(self, n_variables=None, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_variables = n_variables

        # Node feature extraction (for any number of variables)
        self.node_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),  # Process each node value individually
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Edge prediction network
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights with small random values."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass to predict edge probabilities.

        Args:
            x: Tensor of shape [batch_size, n_variables] containing observational data

        Returns:
            edge_probs: Tensor of shape [n_variables, n_variables] containing edge probabilities
        """
        # Ensure input has batch dimension
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension if missing

        batch_size, n_variables = x.shape

        # Create empty edge probability matrix - this is always [n_variables, n_variables]
        edge_probs = torch.zeros((n_variables, n_variables), device=x.device)

        # Extract features for each node
        node_features = []
        for i in range(n_variables):
            # Get values for variable i across all samples
            var_i_values = x[:, i:i+1]  # Shape: [batch_size, 1]

            # Process with node encoder
            # Shape: [batch_size, hidden_dim]
            var_i_features = self.node_encoder(var_i_values)

            # Average across batch dimension
            var_i_embedding = var_i_features.mean(dim=0)  # Shape: [hidden_dim]

            node_features.append(var_i_embedding)

        # Predict edge probabilities
        for i in range(n_variables):
            for j in range(n_variables):
                if i == j:  # No self-loops
                    continue

                # Combine features from potential parent (i) and child (j)
                combined_features = torch.cat(
                    [node_features[i], node_features[j]])

                # Predict edge probability
                prob = self.edge_scorer(combined_features)
                edge_probs[i, j] = prob.item()

        return edge_probs


class DynamicsDecoder(nn.Module):
    """
    Decoder that predicts intervention outcomes based on current values and causal structure.
    Designed to handle graphs of any size through a more general architecture.
    """

    def __init__(self, n_variables=None, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_variables = n_variables

        # Encode individual variables
        self.var_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
        )

        # Process incoming messages
        self.message_processor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        # Predict value updates based on messages
        self.prediction_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights with small random values."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_probabilities, intervention_mask=None):
        """
        Predict the outcome of interventions.

        Args:
            x: Input values for all variables [batch_size, n_variables]
            edge_probabilities: Probability of causal edges [n_variables, n_variables]
            intervention_mask: Boolean mask indicating which variables are intervened upon

        Returns:
            Predicted values for all variables [batch_size, n_variables]
        """
        # Ensure input has batch dimension
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension if missing

        batch_size, n_variables = x.shape
        device = x.device

        # Initialize output with input values
        predictions = x.clone()

        # If no intervention mask provided, assume no interventions
        if intervention_mask is None:
            intervention_mask = torch.zeros(
                n_variables, dtype=torch.bool, device=device)

        # Encode each variable's value
        var_encodings = []
        for i in range(n_variables):
            # List of [batch_size, hidden_dim]
            var_encodings.append(self.var_encoder(x[:, i:i+1]))

        # For each variable that is not intervened upon
        for j in range(n_variables):
            if intervention_mask[j]:
                # If variable is intervened upon, keep its original value
                continue

            # Collect incoming messages from potential parent nodes
            messages = torch.zeros(
                (batch_size, self.hidden_dim), device=device)
            total_weight = 0.0

            # Look at all potential parent nodes
            for i in range(n_variables):
                if i != j:  # No self-loops
                    # Get edge probability (causal influence)
                    edge_prob = edge_probabilities[i, j].item()

                    if edge_prob > 0.01:  # Only consider meaningful edges
                        # Get parent encoding and current node encoding
                        # [batch_size, hidden_dim]
                        parent_encoding = var_encodings[i]
                        # [batch_size, hidden_dim]
                        target_encoding = var_encodings[j]

                        # Combine encodings
                        combined = torch.cat(
                            [parent_encoding, target_encoding], dim=1)

                        # Process message
                        message = self.message_processor(combined)

                        # Weight by edge probability and add to total
                        messages += edge_prob * message
                        total_weight += edge_prob

            # Normalize messages if we have any
            if total_weight > 0:
                messages = messages / total_weight

            # Combine messages with current value encoding to predict new value
            current_encoding = var_encodings[j]
            prediction_input = torch.cat([messages, current_encoding], dim=1)

            # Predict delta (change in value)
            delta = self.prediction_network(prediction_input)

            # Update the predicted value
            predictions[:, j] = x[:, j] + delta.squeeze()

        return predictions
