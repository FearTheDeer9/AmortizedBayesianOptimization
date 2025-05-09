"""
Acquisition strategy implementations for causal Bayesian optimization.

This module provides concrete implementations of the AcquisitionStrategy interface
for causal Bayesian optimization.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np

from causal_meta.optimization.interfaces import AcquisitionStrategy, Intervention
from causal_meta.inference.interfaces import InterventionOutcomeModel, Graph, Data


class ExpectedImprovement(AcquisitionStrategy):
    """
    Expected Improvement acquisition strategy for causal Bayesian optimization.
    
    This strategy selects interventions that maximize the expected improvement
    over the current best known value, taking into account both the predicted
    mean and uncertainty.
    
    Attributes:
        exploration_weight: Weight for the exploration term (uncertainty)
        maximize: Whether to maximize (True) or minimize (False) the objective
        _best_value: Current best observed value (set during optimization)
    """
    
    def __init__(
        self, 
        exploration_weight: float = 0.01, 
        maximize: bool = True,
        intervention_candidates: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize the ExpectedImprovement acquisition strategy.
        
        Args:
            exploration_weight: Weight for the exploration term (uncertainty)
            maximize: Whether to maximize (True) or minimize (False) the objective
            intervention_candidates: Optional list of candidate interventions to consider
                If None, candidates will be generated based on the graph structure
        """
        self.exploration_weight = exploration_weight
        self.maximize = maximize
        self._best_value = float('-inf') if maximize else float('inf')
        self.intervention_candidates = intervention_candidates
    
    def set_best_value(self, value: float) -> None:
        """
        Set the current best observed value.
        
        Args:
            value: Best observed value so far
        """
        self._best_value = value
    
    def _generate_candidates(
        self, 
        graph: Graph, 
        n_values_per_node: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate candidate interventions based on the graph structure.
        
        Args:
            graph: Causal graph structure
            n_values_per_node: Number of candidate values per node
            
        Returns:
            List of candidate interventions
        """
        candidates = []
        
        # Try to get nodes from graph
        try:
            if hasattr(graph, 'get_nodes'):
                nodes = graph.get_nodes()
            elif hasattr(graph, 'nodes'):
                nodes = graph.nodes()
            else:
                # Default to assuming it's an adjacency matrix
                adj_matrix = graph
                if hasattr(graph, 'get_adjacency_matrix'):
                    adj_matrix = graph.get_adjacency_matrix()
                nodes = [i for i in range(adj_matrix.shape[0])]
        except Exception:
            # If we can't determine nodes, use indices 0-9 as default
            nodes = list(range(10))
        
        # Generate candidate values for each node
        for node in nodes:
            for value in np.linspace(-2.0, 2.0, n_values_per_node):
                candidates.append({
                    'target_node': node,
                    'value': float(value)
                })
        
        return candidates
    
    def compute_acquisition(
        self, 
        model: InterventionOutcomeModel, 
        graph: Graph, 
        data: Data
    ) -> Dict[str, float]:
        """
        Compute acquisition values for possible interventions.
        
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
        if not hasattr(model, 'predict_intervention_outcome'):
            raise TypeError("Model must implement predict_intervention_outcome method")
        
        if 'observations' not in data:
            raise ValueError("Data must contain 'observations' key")
        
        # Get intervention candidates
        candidates = self.intervention_candidates
        if candidates is None:
            candidates = self._generate_candidates(graph)
        
        # Compute acquisition values for each candidate
        acquisition_values = {}
        
        for candidate in candidates:
            # Create a unique identifier for this intervention
            node = candidate['target_node']
            value = candidate['value']
            key = f"node_{node}_value_{value}"
            
            # Predict outcome of intervention
            try:
                # Try to get uncertainty if available
                prediction = model.predict_intervention_outcome(
                    graph, candidate, data, return_uncertainty=True
                )
                
                # Check if prediction includes uncertainty
                if isinstance(prediction, tuple) and len(prediction) == 2:
                    mean, uncertainty_dict = prediction
                    uncertainty = uncertainty_dict.get('prediction_std', np.ones_like(mean) * 0.1)
                else:
                    # No uncertainty provided, use mean only
                    mean = prediction
                    uncertainty = np.ones_like(mean) * 0.1
                    
                    # Try to get uncertainty separately
                    try:
                        uncertainty_dict = model.estimate_uncertainty()
                        uncertainty = uncertainty_dict.get('prediction_std', uncertainty)
                    except (AttributeError, NotImplementedError, RuntimeError):
                        pass
                
                # Calculate improvement over current best
                if self.maximize:
                    improvement = mean - self._best_value
                else:
                    improvement = self._best_value - mean
                
                # Calculate expected improvement
                # For simplicity, we'll use a weighted sum of improvement and uncertainty
                # In a full implementation, we would use the proper EI formula
                if np.any(improvement > 0):
                    ei = float(np.mean(improvement) + self.exploration_weight * np.mean(uncertainty))
                else:
                    ei = float(self.exploration_weight * np.mean(uncertainty))
                
                acquisition_values[key] = ei
                
            except Exception as e:
                # If prediction fails, assign a low acquisition value
                acquisition_values[key] = 0.0
        
        return acquisition_values
    
    def select_intervention(
        self, 
        model: InterventionOutcomeModel, 
        graph: Graph, 
        data: Data, 
        budget: float
    ) -> Intervention:
        """
        Select the best intervention based on acquisition values.
        
        Args:
            model: Model used to predict intervention outcomes
            graph: Causal graph structure
            data: Dictionary containing observational data
                Must contain an 'observations' key with array-like data
            budget: Budget constraint for the intervention
                
        Returns:
            Dictionary specifying the selected intervention
            
        Raises:
            ValueError: If data format is invalid or budget is non-positive
            TypeError: If model, graph, or data types are incompatible
        """
        if budget <= 0:
            raise ValueError("Budget must be positive")
        
        # Compute acquisition values
        acq_values = self.compute_acquisition(model, graph, data)
        
        # Select the intervention with the highest acquisition value
        if not acq_values:
            # If no valid interventions, return a default
            return {'target_node': 0, 'value': 0.0}
        
        best_key = max(acq_values, key=acq_values.get)
        
        # Parse the key (format: "node_{index}_value_{value}")
        parts = best_key.split('_')
        try:
            target_node = int(parts[1]) if parts[1].isdigit() else parts[1]
            value = float(parts[3])
        except (IndexError, ValueError):
            # If parsing fails, use default values
            target_node = 0
            value = 0.0
        
        return {
            'target_node': target_node,
            'value': value
        }
    
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
        
        Args:
            model: Model used to predict intervention outcomes
            graph: Causal graph structure
            data: Dictionary containing observational data
                Must contain an 'observations' key with array-like data
            budget: Budget constraint for all interventions combined
            batch_size: Number of interventions to select
                
        Returns:
            List of dictionaries specifying the selected interventions
            
        Raises:
            ValueError: If data format is invalid, budget is non-positive,
                or batch_size is non-positive
            TypeError: If model, graph, or data types are incompatible
        """
        if budget <= 0:
            raise ValueError("Budget must be positive")
        
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        # Compute acquisition values
        acq_values = self.compute_acquisition(model, graph, data)
        
        # If no valid interventions, return default interventions
        if not acq_values:
            return [{'target_node': 0, 'value': 0.0}] * batch_size
        
        # Sort interventions by acquisition value
        sorted_keys = sorted(acq_values.keys(), key=lambda k: acq_values[k], reverse=True)
        
        # Select top interventions
        result = []
        for key in sorted_keys[:batch_size]:
            # Parse the key (format: "node_{index}_value_{value}")
            parts = key.split('_')
            try:
                target_node = int(parts[1]) if parts[1].isdigit() else parts[1]
                value = float(parts[3])
            except (IndexError, ValueError):
                # If parsing fails, use default values
                target_node = 0
                value = 0.0
            
            result.append({
                'target_node': target_node,
                'value': value
            })
        
        # If we don't have enough interventions, pad with the last one
        while len(result) < batch_size:
            result.append(result[-1] if result else {'target_node': 0, 'value': 0.0})
        
        return result


class UpperConfidenceBound(AcquisitionStrategy):
    """
    Upper Confidence Bound acquisition strategy for causal Bayesian optimization.
    
    This strategy selects interventions that maximize the upper confidence bound,
    which balances exploitation (high mean) and exploration (high uncertainty).
    
    Attributes:
        beta: Exploration parameter (higher means more exploration)
        maximize: Whether to maximize (True) or minimize (False) the objective
        intervention_candidates: Optional list of candidate interventions to consider
    """
    
    def __init__(
        self, 
        beta: float = 2.0, 
        maximize: bool = True,
        intervention_candidates: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize the UpperConfidenceBound acquisition strategy.
        
        Args:
            beta: Exploration parameter (higher means more exploration)
            maximize: Whether to maximize (True) or minimize (False) the objective
            intervention_candidates: Optional list of candidate interventions to consider
                If None, candidates will be generated based on the graph structure
        """
        self.beta = beta
        self.maximize = maximize
        self.intervention_candidates = intervention_candidates
    
    # Implementation details similar to ExpectedImprovement
    # For brevity, only the acquisition-specific parts will be shown
    
    def compute_acquisition(
        self, 
        model: InterventionOutcomeModel, 
        graph: Graph, 
        data: Data
    ) -> Dict[str, float]:
        """
        Compute acquisition values using Upper Confidence Bound.
        
        Args:
            model: Model used to predict intervention outcomes
            graph: Causal graph structure
            data: Dictionary containing observational data
                
        Returns:
            Dictionary mapping intervention identifiers to their acquisition values
        """
        # This would be implemented similarly to ExpectedImprovement.compute_acquisition
        # with UCB-specific calculations
        
        # For now, return a placeholder implementation
        return {'node_0_value_1.0': 0.5}
    
    def select_intervention(
        self, 
        model: InterventionOutcomeModel, 
        graph: Graph, 
        data: Data, 
        budget: float
    ) -> Intervention:
        """Select the best intervention using UCB."""
        # This would follow the same pattern as ExpectedImprovement
        return {'target_node': 0, 'value': 1.0}
    
    def select_batch(
        self, 
        model: InterventionOutcomeModel, 
        graph: Graph, 
        data: Data, 
        budget: float, 
        batch_size: int
    ) -> List[Intervention]:
        """Select a batch of interventions using UCB."""
        # This would follow the same pattern as ExpectedImprovement
        return [{'target_node': 0, 'value': 1.0}] * batch_size 