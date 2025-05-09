"""
Core interfaces for causal inference models.

This module provides abstract base classes and interfaces for causal 
structure inference and intervention outcome prediction. These interfaces 
define the contract that concrete implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple, TypeVar, Protocol
import numpy as np
try:
    import torch
except ImportError:
    torch = None  # Make torch optional

# Type definitions
Graph = TypeVar('Graph')  # Generic type for graph structures
Data = Dict[str, Union[np.ndarray, Dict[str, np.ndarray], 'torch.Tensor']]
UncertaintyEstimate = Dict[str, Any]


class CausalStructureInferenceModel(ABC):
    """Interface for models that infer causal structure from data.
    
    This interface defines the contract for models that infer causal structure
    from observational and/or interventional data. Implementations should be
    capable of processing both types of data and returning a graph structure
    that represents the inferred causal relationships.
    
    The interface also requires methods for updating the model with new data
    and estimating uncertainty in the inferred structure.
    
    Example:
        ```python
        # Initialize a concrete implementation
        model = MyCausalStructureInferenceModel(hidden_dim=64, layers=3)
        
        # Prepare data
        obs_data = {"observations": np.array(...)}
        
        # Infer causal structure
        graph = model.infer_structure(obs_data)
        
        # Get uncertainty estimates
        uncertainty = model.estimate_uncertainty()
        
        # Update model with new data
        new_data = {"observations": np.array(...), 
                   "interventions": {"node_1": np.array(...)}}
        model.update_model(new_data)
        ```
    """
    
    @abstractmethod
    def infer_structure(self, data: Data) -> Graph:
        """
        Infers causal structure from data.
        
        This method should analyze the provided data and return an inferred
        causal graph structure. The data can include both observational and
        interventional data.
        
        Args:
            data: Dictionary containing input data for structure inference.
                Must contain an 'observations' key with array-like data.
                May optionally contain an 'interventions' key with a dictionary
                mapping node names/indices to array-like intervention data.
                
        Returns:
            A causal graph structure, typically as an adjacency matrix or
            a CausalGraph object depending on the implementation.
            
        Raises:
            ValueError: If data format is invalid or missing required keys.
            TypeError: If data types are incompatible with the model.
        """
        pass
    
    @abstractmethod
    def update_model(self, data: Data) -> None:
        """
        Updates the model with new data.
        
        This method should update the model's internal state or parameters
        based on the provided data. This allows for incremental learning
        as new data becomes available.
        
        Args:
            data: Dictionary containing input data for model updating.
                Must contain an 'observations' key with array-like data.
                May optionally contain an 'interventions' key with a dictionary
                mapping node names/indices to array-like intervention data.
                
        Raises:
            ValueError: If data format is invalid or missing required keys.
            TypeError: If data types are incompatible with the model.
        """
        pass
    
    @abstractmethod
    def estimate_uncertainty(self) -> UncertaintyEstimate:
        """
        Provides uncertainty estimates for inferred structures.
        
        This method should return uncertainty estimates for the most recently
        inferred causal structure. The format of the uncertainty estimates
        may vary depending on the implementation.
        
        Returns:
            Dictionary containing uncertainty estimates for the inferred
            causal structure. Common keys include:
            - 'edge_probabilities': Matrix of edge existence probabilities
            - 'confidence_intervals': Dictionary with 'lower' and 'upper' matrices
            - 'entropy': Matrix of edge uncertainty entropy values
            
        Raises:
            RuntimeError: If called before any structure has been inferred.
        """
        pass


class InterventionOutcomeModel(ABC):
    """Interface for models that predict outcomes of interventions.
    
    This interface defines the contract for models that predict the outcomes of
    interventions in causal systems. Given a causal graph structure, intervention
    specification, and observational data, implementations should predict the
    outcome of applying the intervention.
    
    The interface also requires methods for updating the model with new data
    and estimating uncertainty in the predictions.
    
    Example:
        ```python
        # Initialize a concrete implementation
        model = MyInterventionOutcomeModel(hidden_dim=64, layers=3)
        
        # Prepare data
        graph = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # DAG adjacency matrix
        intervention = {'target_node': 0, 'value': 2.0}  # Set node 0 to value 2.0
        obs_data = {"observations": np.array(...)}
        
        # Predict intervention outcome
        predictions = model.predict_intervention_outcome(graph, intervention, obs_data)
        
        # Get uncertainty estimates
        uncertainty = model.estimate_uncertainty()
        
        # Update model with new data
        new_data = {"observations": np.array(...), 
                   "interventions": {"node_1": np.array(...)}}
        model.update_model(new_data)
        ```
    """
    
    @abstractmethod
    def predict_intervention_outcome(
        self, 
        graph: Graph, 
        intervention: Dict[str, Any], 
        data: Data
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Predicts outcomes of interventions.
        
        This method should predict the outcome of applying the specified
        intervention in the given causal graph, conditioned on the observational
        data. The intervention specification can vary depending on the implementation,
        but typically includes the target node and intervention value.
        
        Args:
            graph: Causal graph structure, typically as an adjacency matrix or
                CausalGraph object depending on the implementation.
            intervention: Dictionary specifying the intervention. Common keys include:
                - 'target_node': The node to intervene on (index or name)
                - 'value': The value to set the node to
                - 'type': The type of intervention (e.g., 'perfect', 'imperfect', 'soft')
                - 'params': Additional parameters for the intervention
            data: Dictionary containing observational data to condition on.
                Must contain an 'observations' key with array-like data.
                
        Returns:
            Predicted outcomes of the intervention, either as an array or
            a dictionary with additional information.
            
        Raises:
            ValueError: If intervention specification or data format is invalid.
            TypeError: If graph, intervention, or data types are incompatible.
        """
        pass
    
    @abstractmethod
    def update_model(self, data: Data) -> None:
        """
        Updates the model with new data.
        
        This method should update the model's internal state or parameters
        based on the provided data. This allows for incremental learning
        as new data becomes available.
        
        Args:
            data: Dictionary containing input data for model updating.
                Must contain an 'observations' key with array-like data.
                May optionally contain an 'interventions' key with a dictionary
                mapping node names/indices to array-like intervention data.
                
        Raises:
            ValueError: If data format is invalid or missing required keys.
            TypeError: If data types are incompatible with the model.
        """
        pass
    
    @abstractmethod
    def estimate_uncertainty(self) -> UncertaintyEstimate:
        """
        Provides uncertainty estimates for predicted outcomes.
        
        This method should return uncertainty estimates for the most recently
        predicted intervention outcomes. The format of the uncertainty estimates
        may vary depending on the implementation.
        
        Returns:
            Dictionary containing uncertainty estimates for the predicted
            outcomes. Common keys include:
            - 'prediction_std': Standard deviation of predictions
            - 'confidence_intervals': Dictionary with 'lower' and 'upper' bounds
            - 'entropy': Entropy of predicted distributions
            
        Raises:
            RuntimeError: If called before any predictions have been made.
        """
        pass


class Updatable(ABC):
    """Interface for models that can be updated with new data.
    
    This interface defines the contract for models that can be updated
    incrementally with new data. It provides methods for updating the
    model and resetting it to its initial state. Implementations should
    handle different update strategies such as incremental updates,
    experience replay, or full retraining.
    
    Example:
        ```python
        # Initialize a concrete implementation
        model = MyUpdatableModel(hidden_dim=64, layers=3)
        
        # Prepare data
        data = {"observations": np.array(...)}
        
        # Update the model with new data
        success = model.update(data)
        
        # Reset the model to its initial state
        model.reset()
        ```
    """
    
    @abstractmethod
    def update(self, data: Data) -> bool:
        """
        Updates the model with new data.
        
        This method should update the model's internal state or parameters
        based on the provided data. The update strategy depends on the
        implementation, but could include incremental updates, experience
        replay, or full retraining.
        
        Args:
            data: Dictionary containing input data for model updating.
                Must contain an 'observations' key with array-like data.
                May optionally contain additional data needed for updating.
                
        Returns:
            Boolean indicating whether the update was successful.
            
        Raises:
            ValueError: If data format is invalid or missing required keys.
            TypeError: If data types are incompatible with the model.
            RuntimeError: If the update fails for any other reason.
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Resets the model to its initial state.
        
        This method should reset the model's internal state or parameters
        to their initial values, effectively undoing all updates. This is
        useful for restarting learning from scratch or for implementing
        early stopping with model checkpoint restoration.
        
        Raises:
            RuntimeError: If the reset fails for any reason.
        """
        pass


# More interfaces will be added in future implementations
# AcquisitionStrategy 