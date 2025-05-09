"""
Core interfaces for causal optimization components.

This module provides abstract base classes and interfaces for acquisition
strategies in causal Bayesian optimization. These interfaces define the
contract that concrete implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, TypeVar
import numpy as np

# Import types from other interfaces
try:
    from causal_meta.inference.interfaces import (
        InterventionOutcomeModel, Graph, Data, UncertaintyEstimate
    )
except ImportError:
    # Define fallback types for documentation or standalone usage
    InterventionOutcomeModel = TypeVar('InterventionOutcomeModel')
    Graph = TypeVar('Graph')
    Data = Dict[str, Any]
    UncertaintyEstimate = Dict[str, Any]

# Type definitions
Intervention = Dict[str, Any]  # Dict specifying intervention target and value


class AcquisitionStrategy(ABC):
    """Interface for strategies that select interventions.
    
    This interface defines the contract for strategies that select interventions
    based on acquisition functions in causal Bayesian optimization. Implementations
    should compute acquisition values for possible interventions and select the
    best intervention(s) based on these values.
    
    The interface requires methods for computing acquisition values for possible
    interventions, selecting a single best intervention, and selecting a batch of
    diverse interventions.
    
    Example:
        ```python
        # Initialize a concrete implementation
        strategy = MyAcquisitionStrategy(exploration_weight=0.1)
        
        # Prepare data
        model = MyInterventionOutcomeModel(...)
        graph = CausalGraph(...)
        obs_data = {"observations": np.array(...)}
        budget = 1.0
        
        # Compute acquisition values for possible interventions
        acq_values = strategy.compute_acquisition(model, graph, obs_data)
        
        # Select the best intervention
        intervention = strategy.select_intervention(model, graph, obs_data, budget)
        
        # Select a batch of diverse interventions
        batch = strategy.select_batch(model, graph, obs_data, budget, batch_size=5)
        ```
    """
    
    @abstractmethod
    def compute_acquisition(
        self, 
        model: InterventionOutcomeModel, 
        graph: Graph, 
        data: Data
    ) -> Dict[str, float]:
        """
        Compute acquisition values for possible interventions.
        
        This method should compute acquisition values for possible interventions
        based on the provided model, graph, and data. The acquisition values
        represent the expected utility of performing each intervention.
        
        Args:
            model: Model used to predict intervention outcomes
            graph: Causal graph structure
            data: Dictionary containing observational data
                Must contain an 'observations' key with array-like data
                
        Returns:
            Dictionary mapping intervention identifiers to their acquisition values
            
        Raises:
            ValueError: If data format is invalid or missing required keys
            TypeError: If model, graph, or data types are incompatible
        """
        pass
    
    @abstractmethod
    def select_intervention(
        self, 
        model: InterventionOutcomeModel, 
        graph: Graph, 
        data: Data, 
        budget: float
    ) -> Intervention:
        """
        Select the best intervention based on acquisition values.
        
        This method should select the intervention with the highest acquisition
        value that satisfies the budget constraint.
        
        Args:
            model: Model used to predict intervention outcomes
            graph: Causal graph structure
            data: Dictionary containing observational data
                Must contain an 'observations' key with array-like data
            budget: Budget constraint for the intervention
                
        Returns:
            Dictionary specifying the selected intervention
            Common keys include:
            - 'target_node': The node to intervene on (index or name)
            - 'value': The value to set the node to
            
        Raises:
            ValueError: If data format is invalid or budget is non-positive
            TypeError: If model, graph, or data types are incompatible
        """
        pass
    
    @abstractmethod
    def select_batch(
        self, 
        model: InterventionOutcomeModel, 
        graph: Graph, 
        data: Data, 
        budget: float, 
        batch_size: int
    ) -> List[Intervention]:
        """
        Select a batch of diverse interventions.
        
        This method should select a batch of diverse interventions that satisfy
        the budget constraint. The batch should balance exploration and exploitation
        to maximize information gain.
        
        Args:
            model: Model used to predict intervention outcomes
            graph: Causal graph structure
            data: Dictionary containing observational data
                Must contain an 'observations' key with array-like data
            budget: Budget constraint for all interventions combined
            batch_size: Number of interventions to select
                
        Returns:
            List of dictionaries specifying the selected interventions
            Each dictionary should have common keys like:
            - 'target_node': The node to intervene on (index or name)
            - 'value': The value to set the node to
            
        Raises:
            ValueError: If data format is invalid, budget is non-positive,
                or batch_size is non-positive
            TypeError: If model, graph, or data types are incompatible
        """
        pass 