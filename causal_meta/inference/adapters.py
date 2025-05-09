"""
Adapter classes for various model architectures to implement the common interfaces.

This module provides adapter classes that wrap different model architectures
and make them implement the standard interfaces defined in causal_meta.inference.interfaces.
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
try:
    import torch
    import torch_geometric
except ImportError:
    torch = None
    torch_geometric = None

from causal_meta.inference.interfaces import (
    CausalStructureInferenceModel,
    InterventionOutcomeModel,
    Graph,
    Data,
    UncertaintyEstimate
)
from causal_meta.graph.causal_graph import CausalGraph
from causal_meta.inference.models.mlp_encoder import MLPGraphEncoder
from causal_meta.inference.models.transformer_encoder import TransformerGraphEncoder

class GraphEncoderAdapter(CausalStructureInferenceModel):
    """
    Adapter for GNN-based graph encoder models.
    
    This adapter allows graph encoder models to be used with the standard
    CausalStructureInferenceModel interface.
    
    This class wraps a GraphEncoder model from the GNN framework and makes
    it implement the CausalStructureInferenceModel interface.
    """
    
    def __init__(self, model, device=None):
        """
        Initialize the adapter with a graph encoder model.
        
        Args:
            model: The graph encoder model to adapt
            device: The device to run the model on (default: None, uses model's device)
        """
        super().__init__()
        self.model = model
        self.device = device
        self._last_graph = None
    
    def infer_structure(self, data: Data) -> Graph:
        """
        Infer a causal structure from data.
        
        Args:
            data: Dictionary containing observational data
            
        Returns:
            A Graph object representing the inferred causal structure
        """
        # Validate data
        self._validate_data(data)
        
        # Convert to tensor
        observations = self._prepare_observations(data.get("observations"))
        
        # Predict the graph
        self.model.eval()
        with torch.no_grad():
            edge_probs, graph = self.model.predict_graph_from_data(observations)
        
        # Store for uncertainty estimation
        self._last_graph = graph
        
        return graph
    
    def update_model(self, data: Data) -> None:
        """
        Update the model based on new data.
        
        Args:
            data: Dictionary containing observational and optionally interventional data
        """
        # For now, just validate the data
        self._validate_data(data)
        
        # In a real implementation, this would potentially update model parameters
        # based on new data, but this is optional and depends on whether online
        # learning is supported
        pass
    
    def estimate_uncertainty(self) -> UncertaintyEstimate:
        """
        Estimate the uncertainty of the last inferred structure.
        
        Returns:
            A dictionary containing uncertainty estimates
        """
        # Edge probabilities provide an uncertainty estimate
        if hasattr(self.model, "_last_edge_probs") and self.model._last_edge_probs is not None:
            edge_probs = self.model._last_edge_probs.cpu().numpy()
        else:
            # If no previous inference, return identity matrix
            n_nodes = 10  # default size
            if self._last_graph is not None:
                n_nodes = len(self._last_graph.get_nodes())
            edge_probs = np.eye(n_nodes)
        
        return {"edge_probabilities": edge_probs}
    
    def _validate_data(self, data: Data) -> None:
        """
        Validate the input data format.
        
        Args:
            data: Dictionary containing observational data
            
        Raises:
            ValueError: If data is invalid
            TypeError: If data has incorrect type
        """
        if not isinstance(data, dict):
            raise TypeError("Data must be a dictionary")
        
        if "observations" not in data:
            raise ValueError("Data must contain 'observations' key")
        
        observations = data["observations"]
        if not isinstance(observations, (np.ndarray, torch.Tensor)):
            raise TypeError("Observations must be a numpy array or torch tensor")
        
        # Check dimensions for time series data: [batch_size, seq_length, n_features]
        if len(observations.shape) != 3:
            raise ValueError("Observations must be 3-dimensional (batch_size, seq_length, n_features)")
    
    def _prepare_observations(self, observations: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Prepare observations for use with the model.
        
        Args:
            observations: Numpy array or torch tensor of observations
            
        Returns:
            Tensor of observations on the correct device
        """
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations).float()
        
        # Move to device if specified
        if self.device is not None:
            observations = observations.to(self.device)
        
        return observations


class MLPGraphEncoderAdapter(CausalStructureInferenceModel):
    """
    Adapter for MLP-based graph encoder models.
    
    This adapter allows MLP-based structure inference models to be used with
    the standard CausalStructureInferenceModel interface.
    """
    
    def __init__(self, model: MLPGraphEncoder, device=None):
        """
        Initialize the adapter with an MLP graph encoder model.
        
        Args:
            model: The MLP graph encoder model to adapt
            device: The device to run the model on (default: None, uses model's device)
        """
        super().__init__()
        self.model = model
        self.device = device
        self._last_graph = None
    
    def infer_structure(self, data: Data) -> Graph:
        """
        Infer a causal structure from data.
        
        Args:
            data: Dictionary containing observational data
            
        Returns:
            A Graph object representing the inferred causal structure
        """
        # Validate data
        self._validate_data(data)
        
        # Convert to tensor
        observations = self._prepare_observations(data.get("observations"))
        
        # Predict the graph
        self.model.eval()
        with torch.no_grad():
            edge_probs, graph = self.model.predict_graph_from_data(observations)
        
        # Store for uncertainty estimation
        self._last_graph = graph
        
        return graph
    
    def update_model(self, data: Data) -> None:
        """
        Update the model based on new data.
        
        Args:
            data: Dictionary containing observational and optionally interventional data
        """
        # For now, just validate the data
        self._validate_data(data)
        
        # In a real implementation, this would potentially update model parameters
        # based on new data, but this is optional and depends on whether online
        # learning is supported
        pass
    
    def estimate_uncertainty(self) -> UncertaintyEstimate:
        """
        Estimate the uncertainty of the last inferred structure.
        
        Returns:
            A dictionary containing uncertainty estimates
        """
        # Edge probabilities provide an uncertainty estimate
        if hasattr(self.model, "_last_edge_probs") and self.model._last_edge_probs is not None:
            edge_probs = self.model._last_edge_probs.cpu().numpy()
        else:
            # If no previous inference, return identity matrix
            n_nodes = 10  # default size
            if self._last_graph is not None:
                n_nodes = len(self._last_graph.get_nodes())
            edge_probs = np.eye(n_nodes)
        
        return {"edge_probabilities": edge_probs}
    
    def _validate_data(self, data: Data) -> None:
        """
        Validate the input data format.
        
        Args:
            data: Dictionary containing observational data
            
        Raises:
            ValueError: If data is invalid
            TypeError: If data has incorrect type
        """
        if not isinstance(data, dict):
            raise TypeError("Data must be a dictionary")
        
        if "observations" not in data:
            raise ValueError("Data must contain 'observations' key")
        
        observations = data["observations"]
        if not isinstance(observations, (np.ndarray, torch.Tensor)):
            raise TypeError("Observations must be a numpy array or torch tensor")
        
        # Check dimensions for time series data: [batch_size, seq_length, n_features]
        if len(observations.shape) != 3:
            raise ValueError("Observations must be 3-dimensional (batch_size, seq_length, n_features)")
    
    def _prepare_observations(self, observations: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Prepare observations for use with the model.
        
        Args:
            observations: Numpy array or torch tensor of observations
            
        Returns:
            Tensor of observations on the correct device
        """
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations).float()
        
        # Move to device if specified
        if self.device is not None:
            observations = observations.to(self.device)
        
        return observations


class TransformerGraphEncoderAdapter(CausalStructureInferenceModel):
    """
    Adapter for Transformer-based graph encoder models.
    
    This adapter allows Transformer-based structure inference models to be used with
    the standard CausalStructureInferenceModel interface.
    """
    
    def __init__(self, model: TransformerGraphEncoder, device=None):
        """
        Initialize the adapter with a Transformer graph encoder model.
        
        Args:
            model: The Transformer graph encoder model to adapt
            device: The device to run the model on (default: None, uses model's device)
        """
        super().__init__()
        self.model = model
        self.device = device
        self._last_graph = None
    
    def infer_structure(self, data: Data) -> Graph:
        """
        Infer a causal structure from data.
        
        Args:
            data: Dictionary containing observational data
            
        Returns:
            A Graph object representing the inferred causal structure
        """
        # Validate data
        self._validate_data(data)
        
        # Convert to tensor
        observations = self._prepare_observations(data.get("observations"))
        
        # Predict the graph
        self.model.eval()
        with torch.no_grad():
            edge_probs, graph = self.model.predict_graph_from_data(observations)
        
        # Store for uncertainty estimation
        self._last_graph = graph
        
        return graph
    
    def update_model(self, data: Data) -> None:
        """
        Update the model based on new data.
        
        Args:
            data: Dictionary containing observational and optionally interventional data
        """
        # For now, just validate the data
        self._validate_data(data)
        
        # In a real implementation, this would potentially update model parameters
        # based on new data, but this is optional and depends on whether online
        # learning is supported
        pass
    
    def estimate_uncertainty(self) -> UncertaintyEstimate:
        """
        Estimate the uncertainty of the last inferred structure.
        
        Returns:
            A dictionary containing uncertainty estimates
        """
        # Edge probabilities provide an uncertainty estimate
        if hasattr(self.model, "_last_edge_probs") and self.model._last_edge_probs is not None:
            edge_probs = self.model._last_edge_probs.cpu().numpy()
        else:
            # If no previous inference, return identity matrix
            n_nodes = 10  # default size
            if self._last_graph is not None:
                n_nodes = len(self._last_graph.get_nodes())
            edge_probs = np.eye(n_nodes)
        
        return {"edge_probabilities": edge_probs}
    
    def _validate_data(self, data: Data) -> None:
        """
        Validate the input data format.
        
        Args:
            data: Dictionary containing observational data
            
        Raises:
            ValueError: If data is invalid
            TypeError: If data has incorrect type
        """
        if not isinstance(data, dict):
            raise TypeError("Data must be a dictionary")
        
        if "observations" not in data:
            raise ValueError("Data must contain 'observations' key")
        
        observations = data["observations"]
        if not isinstance(observations, (np.ndarray, torch.Tensor)):
            raise TypeError("Observations must be a numpy array or torch tensor")
        
        # Check dimensions for time series data: [batch_size, seq_length, n_features]
        if len(observations.shape) != 3:
            raise ValueError("Observations must be 3-dimensional (batch_size, seq_length, n_features)")
    
    def _prepare_observations(self, observations: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Prepare observations for use with the model.
        
        Args:
            observations: Numpy array or torch tensor of observations
            
        Returns:
            Tensor of observations on the correct device
        """
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations).float()
        
        # Move to device if specified
        if self.device is not None:
            observations = observations.to(self.device)
        
        return observations


class DynamicsDecoderAdapter(InterventionOutcomeModel):
    """Adapter for DynamicsDecoder components to implement InterventionOutcomeModel interface.
    
    This adapter wraps existing DynamicsDecoder instances to make them compatible
    with the InterventionOutcomeModel interface, enabling them to be used
    with the new architecture without modifying their core implementation.
    
    Args:
        dynamics_decoder: An instance of a DynamicsDecoder class from meta_learning
        uncertainty_estimator: An optional UncertaintyEstimator to use for quantifying prediction uncertainty
        return_uncertainty: Whether to return uncertainty by default (default: False)
        device: Device to run the model on (default: 'cuda' if available else 'cpu')
    """
    
    def __init__(
        self, 
        dynamics_decoder, 
        uncertainty_estimator=None,
        return_uncertainty: bool = False, 
        device: Optional[str] = None
    ):
        """Initialize the adapter with a DynamicsDecoder instance."""
        self.dynamics_decoder = dynamics_decoder
        self.uncertainty_estimator = uncertainty_estimator
        self.return_uncertainty_flag = return_uncertainty
        
        # Determine device
        if device is None:
            self.device = 'cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu'
        else:
            self.device = device
            
        # Store the last predictions for uncertainty estimation
        self._last_predictions = None
        self._last_uncertainty = None
        self._last_data = None
    
    def predict_intervention_outcome(
        self, 
        graph: Graph, 
        intervention: Dict[str, Any], 
        data: Data,
        return_uncertainty: Optional[bool] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Predict intervention outcomes using the wrapped DynamicsDecoder.
        
        Args:
            graph: Causal graph structure (adjacency matrix)
            intervention: Dictionary specifying the intervention target and value
            data: Dictionary containing observations to condition on
            return_uncertainty: Whether to return uncertainty estimates
                (overrides the default set in the constructor)
                
        Returns:
            If return_uncertainty is False:
                Array of predicted intervention outcomes
            If return_uncertainty is True:
                Tuple of (predictions, uncertainty) where uncertainty is a dictionary
                
        Raises:
            ValueError: If intervention specification or data format is invalid.
            TypeError: If graph, intervention, or data types are incompatible.
        """
        if 'observations' not in data:
            raise ValueError("Data must contain 'observations' key with observations")
        
        if 'target_node' not in intervention or 'value' not in intervention:
            raise ValueError("Intervention must contain 'target_node' and 'value' keys")
        
        # Store the data for later use with uncertainty estimator
        self._last_data = data
        
        # Determine whether to return uncertainty
        should_return_uncertainty = self.return_uncertainty_flag
        if return_uncertainty is not None:
            should_return_uncertainty = return_uncertainty
        
        # Get observations and convert to tensor if needed
        obs = data['observations']
        if torch is not None and not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
            
        # Convert graph to tensor if needed
        if torch is not None and not isinstance(graph, torch.Tensor):
            graph = torch.tensor(graph, dtype=torch.float32)
            
        # Move to device
        if torch is not None:
            obs = obs.to(self.device)
            graph = graph.to(self.device)
            
        # Add batch dimension if not present
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(0)
            
        if len(graph.shape) == 2:
            graph = graph.unsqueeze(0)
        
        # Prepare data for DynamicsDecoder
        batch_size = graph.size(0)
        num_nodes = graph.size(1)
        num_samples = obs.size(1) if len(obs.shape) > 2 else obs.size(0)
        
        # Create edge index from the adjacency matrix
        edge_index = self._adjacency_to_edge_index(graph)
        
        # Convert node features to appropriate format
        node_features = self._prepare_node_features(obs, num_nodes)
        
        # Create batch assignments
        batch = torch.arange(batch_size, device=self.device).repeat_interleave(num_nodes)
        
        # Extract intervention information
        target_node = torch.tensor([intervention['target_node']], device=self.device)
        value = torch.tensor([intervention['value']], dtype=torch.float32, device=self.device)
        
        # Forward pass through dynamics decoder
        with torch.no_grad():
            # Use the model's built-in uncertainty if we don't have an estimator
            # or if we specifically want to return uncertainty from the forward pass
            if should_return_uncertainty and (self.uncertainty_estimator is None or hasattr(self.dynamics_decoder, 'uncertainty')):
                predictions, uncertainty = self.dynamics_decoder.predict_intervention_outcome(
                    x=node_features,
                    edge_index=edge_index,
                    batch=batch,
                    adj_matrices=graph,
                    intervention_targets=target_node,
                    intervention_values=value,
                    return_uncertainty=True
                )
                
                # Store predictions for later use
                self._last_predictions = predictions.reshape(batch_size, num_nodes, -1).cpu().numpy()
                
                # Convert uncertainty tensor to standardized format
                if isinstance(uncertainty, torch.Tensor):
                    std_dev = uncertainty.reshape(batch_size, num_nodes, -1).cpu().numpy()
                    self._last_uncertainty = self._standardize_uncertainty(std_dev)
                else:
                    # If uncertainty is already a dict, ensure it's in the standard format
                    self._last_uncertainty = self._standardize_uncertainty(uncertainty)
                
                if should_return_uncertainty:
                    return (
                        self._last_predictions.squeeze(0),
                        self._last_uncertainty
                    )
                else:
                    return self._last_predictions.squeeze(0)
            else:
                # Just get predictions without uncertainty
                predictions = self.dynamics_decoder.predict_intervention_outcome(
                    x=node_features,
                    edge_index=edge_index,
                    batch=batch,
                    adj_matrices=graph,
                    intervention_targets=target_node,
                    intervention_values=value,
                    return_uncertainty=False
                )
                
                # Store predictions for later use
                self._last_predictions = predictions.reshape(batch_size, num_nodes, -1).cpu().numpy()
                self._last_uncertainty = None  # Reset uncertainty since we didn't calculate it
                
                # If uncertainty is requested but we didn't get it from forward pass, estimate it now
                if should_return_uncertainty and self.uncertainty_estimator is not None:
                    uncertainty = self.estimate_uncertainty()
                    return self._last_predictions.squeeze(0), uncertainty
                else:
                    return self._last_predictions.squeeze(0)
    
    def update_model(self, data: Data) -> None:
        """
        Update the DynamicsDecoder with new data through additional training.
        
        This requires that the wrapped DynamicsDecoder has an update or train method.
        If not, this method raises NotImplementedError.
        
        Args:
            data: Dictionary containing observations and optional interventions.
            
        Raises:
            NotImplementedError: If the wrapped DynamicsDecoder doesn't support updating.
            ValueError: If data format is invalid.
        """
        # Check if update method exists
        if not hasattr(self.dynamics_decoder, 'update') and not hasattr(self.dynamics_decoder, 'train'):
            raise NotImplementedError(
                "The wrapped DynamicsDecoder doesn't support updating. "
                "Use a trainable DynamicsDecoder or implement the update method."
            )
            
        # Call appropriate update method
        if hasattr(self.dynamics_decoder, 'update'):
            self.dynamics_decoder.update(data)
        else:  # Use train method
            # This assumes the train method accepts data in the format we have
            # In practice, you might need a more complex adapter method here
            self.dynamics_decoder.train(data)
        
    def estimate_uncertainty(self) -> UncertaintyEstimate:
        """
        Estimate uncertainty for the most recently predicted outcomes.
        
        If an uncertainty estimator is provided, it will be used.
        Otherwise, falls back to built-in uncertainty estimation.
        
        Returns:
            Dictionary containing uncertainty estimates for the predictions.
            
        Raises:
            RuntimeError: If called before any predictions have been made.
        """
        if self._last_predictions is None:
            raise RuntimeError(
                "No predictions have been made yet. Call predict_intervention_outcome first."
            )
        
        # If we have an uncertainty estimator, use it
        if self.uncertainty_estimator is not None and self._last_data is not None:
            # Use the estimator to compute uncertainty
            uncertainty = self.uncertainty_estimator.estimate_uncertainty(self, self._last_data)
            
            # Store the uncertainty for future reference
            self._last_uncertainty = self._standardize_uncertainty(uncertainty)
            
            return self._last_uncertainty
        
        # If we already have uncertainty estimates, return them
        if self._last_uncertainty is not None:
            return self._last_uncertainty
        
        # Otherwise, create simple uncertainty estimates based on the predictions
        # This is a fallback in case the model doesn't provide uncertainty
        
        # Create a simple standard deviation estimate (10% of prediction magnitude)
        prediction_std = np.abs(self._last_predictions) * 0.1
        
        # Calculate confidence intervals
        confidence = 0.95
        z = 1.96  # 95% confidence interval
        
        lower = self._last_predictions - z * prediction_std
        upper = self._last_predictions + z * prediction_std
        
        # Create and store standardized uncertainty
        self._last_uncertainty = {
            'prediction_std': prediction_std,
            'confidence_intervals': {
                'lower': lower,
                'upper': upper,
                'confidence': confidence
            }
        }
        
        return self._last_uncertainty
    
    def calibrate_uncertainty(self, validation_data: Data) -> None:
        """
        Calibrate uncertainty estimates using validation data.
        
        This method is only effective if an uncertainty estimator
        that supports calibration is provided.
        
        Args:
            validation_data: Dictionary containing validation data.
        
        Raises:
            ValueError: If no uncertainty estimator is available.
        """
        if self.uncertainty_estimator is None:
            raise ValueError("Cannot calibrate without an uncertainty estimator")
        
        # Call the estimator's calibration method
        self.uncertainty_estimator.calibrate(self, validation_data)
    
    def _standardize_uncertainty(self, uncertainty) -> Dict[str, Any]:
        """
        Standardize uncertainty format to ensure consistent structure.
        
        Args:
            uncertainty: Uncertainty estimates, either as a tensor or dictionary.
            
        Returns:
            Dictionary with standardized uncertainty format.
        """
        # If uncertainty is already a dictionary, ensure it has the right structure
        if isinstance(uncertainty, dict):
            # Create a new dict with required keys, preserving any existing values
            standardized = {}
            
            # Ensure prediction_std is present
            if 'prediction_std' in uncertainty:
                standardized['prediction_std'] = uncertainty['prediction_std']
            else:
                # Try to derive from other fields or use a default
                if 'std' in uncertainty:
                    standardized['prediction_std'] = uncertainty['std']
                elif 'variance' in uncertainty:
                    standardized['prediction_std'] = np.sqrt(uncertainty['variance'])
                else:
                    # Default fallback
                    standardized['prediction_std'] = np.abs(self._last_predictions) * 0.1
            
            # Ensure confidence_intervals are present
            if 'confidence_intervals' in uncertainty:
                intervals = uncertainty['confidence_intervals']
                # Ensure the intervals dict has the required structure
                standardized_intervals = {}
                
                # Copy or create required fields
                if isinstance(intervals, dict):
                    # Lower bound
                    if 'lower' in intervals:
                        standardized_intervals['lower'] = intervals['lower']
                    else:
                        standardized_intervals['lower'] = self._last_predictions - 1.96 * standardized['prediction_std']
                    
                    # Upper bound
                    if 'upper' in intervals:
                        standardized_intervals['upper'] = intervals['upper']
                    else:
                        standardized_intervals['upper'] = self._last_predictions + 1.96 * standardized['prediction_std']
                    
                    # Confidence level
                    if 'confidence' in intervals:
                        standardized_intervals['confidence'] = intervals['confidence']
                    else:
                        standardized_intervals['confidence'] = 0.95
                else:
                    # If intervals is not a dict, create a default one
                    standardized_intervals = {
                        'lower': self._last_predictions - 1.96 * standardized['prediction_std'],
                        'upper': self._last_predictions + 1.96 * standardized['prediction_std'],
                        'confidence': 0.95
                    }
                
                standardized['confidence_intervals'] = standardized_intervals
            else:
                # Create default confidence intervals
                standardized['confidence_intervals'] = {
                    'lower': self._last_predictions - 1.96 * standardized['prediction_std'],
                    'upper': self._last_predictions + 1.96 * standardized['prediction_std'],
                    'confidence': 0.95
                }
            
            # Copy any additional fields
            for key, value in uncertainty.items():
                if key not in standardized and key != 'confidence_intervals':
                    standardized[key] = value
                    
            return standardized
            
        # If uncertainty is a tensor/array, convert to standardized dict
        elif isinstance(uncertainty, (np.ndarray, torch.Tensor)):
            # Convert to numpy if it's a tensor
            if isinstance(uncertainty, torch.Tensor):
                uncertainty = uncertainty.cpu().numpy()
                
            # Create standard format
            return {
                'prediction_std': uncertainty,
                'confidence_intervals': {
                    'lower': self._last_predictions - 1.96 * uncertainty,
                    'upper': self._last_predictions + 1.96 * uncertainty,
                    'confidence': 0.95
                }
            }
        else:
            # Fallback for unexpected types
            fallback_std = np.abs(self._last_predictions) * 0.1
            return {
                'prediction_std': fallback_std,
                'confidence_intervals': {
                    'lower': self._last_predictions - 1.96 * fallback_std,
                    'upper': self._last_predictions + 1.96 * fallback_std,
                    'confidence': 0.95
                }
            }
    
    def _adjacency_to_edge_index(self, adj_matrices: torch.Tensor) -> torch.Tensor:
        """
        Convert adjacency matrices to edge index format for PyTorch Geometric.
        
        Args:
            adj_matrices: Adjacency matrices [batch_size, num_nodes, num_nodes]
            
        Returns:
            Edge index tensor [2, num_edges]
        """
        batch_size = adj_matrices.size(0)
        num_nodes = adj_matrices.size(1)
        
        # Get indices of edges (non-zero elements in the adjacency matrix)
        edge_indices = []
        
        for b in range(batch_size):
            adj = adj_matrices[b]
            sources, targets = torch.nonzero(adj, as_tuple=True)
            
            # Adjust indices for batch
            batch_offset = b * num_nodes
            sources = sources + batch_offset
            targets = targets + batch_offset
            
            # Create edge index for this batch
            if sources.numel() > 0:
                batch_edges = torch.stack([sources, targets], dim=0)
                edge_indices.append(batch_edges)
        
        # Combine edge indices from all batches
        if edge_indices:
            edge_index = torch.cat(edge_indices, dim=1)
        else:
            # Create an empty edge index
            edge_index = torch.zeros((2, 0), device=self.device, dtype=torch.long)
            
        return edge_index
    
    def _prepare_node_features(
        self, 
        observations: torch.Tensor, 
        num_nodes: int
    ) -> torch.Tensor:
        """
        Prepare node features for use with the dynamics decoder.
        
        Args:
            observations: Observation data [batch_size, num_samples, num_features]
            num_nodes: Number of nodes in the graph
            
        Returns:
            Node features in the format expected by the dynamics decoder
        """
        batch_size = observations.size(0)
        
        # If observations are time series, use the mean as node features
        if len(observations.shape) > 2:
            # [batch_size, num_samples, num_features] -> [batch_size, num_features]
            node_features = observations.mean(dim=1)
        else:
            # Already in the right format
            node_features = observations
            
        # Expand to match the number of nodes if needed
        if node_features.size(1) != num_nodes:
            # Assume features are per-node
            # [batch_size, num_features] -> [batch_size * num_nodes, 1]
            node_features = node_features.repeat_interleave(num_nodes // node_features.size(1), dim=0)
        else:
            # [batch_size, num_nodes] -> [batch_size * num_nodes, 1]
            node_features = node_features.reshape(batch_size * num_nodes, 1)
            
        return node_features 