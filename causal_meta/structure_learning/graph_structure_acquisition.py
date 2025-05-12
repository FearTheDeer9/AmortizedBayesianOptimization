"""
Acquisition strategies for causal graph structure learning.

This module provides acquisition strategies for selecting interventions
to improve causal graph structure learning.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

from causal_meta.optimization.interfaces import AcquisitionStrategy, Intervention
from causal_meta.structure_learning.simple_graph_learner import SimpleGraphLearner


class GraphStructureAcquisition:
    """
    Acquisition strategy for causal graph structure learning.
    
    This class implements strategies for selecting interventions to improve
    causal graph structure learning. The strategies are designed to select
    interventions that maximize information gain about the graph structure.
    
    Args:
        strategy_type: Type of acquisition strategy
            - 'uncertainty': Selects interventions with highest edge uncertainty
            - 'random': Selects random interventions
            - 'information_gain': Selects interventions to maximize information gain
        custom_strategy: Optional custom strategy function
        intervention_values: Optional predefined values for interventions
        node_weights: Optional weights for prioritizing nodes
    """
    
    def __init__(
        self,
        strategy_type: str = "uncertainty",
        custom_strategy: Optional[Callable] = None,
        intervention_values: Optional[List[float]] = None,
        node_weights: Optional[List[float]] = None
    ):
        """Initialize the acquisition strategy."""
        valid_strategies = ["uncertainty", "random", "information_gain"]
        if strategy_type not in valid_strategies and custom_strategy is None:
            raise ValueError(f"Strategy type must be one of {valid_strategies} or a custom strategy function must be provided")
        
        self.strategy_type = strategy_type
        self.custom_strategy = custom_strategy
        self.intervention_values = intervention_values or [1.0, -1.0, 2.0, -2.0, 0.5, -0.5]
        self.node_weights = node_weights
    
    def compute_edge_uncertainties(
        self,
        model: SimpleGraphLearner,
        data: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute edge probability uncertainties.
        
        This method computes the uncertainty in edge probabilities based on
        how close they are to 0.5 (maximum uncertainty).
        
        Args:
            model: Trained SimpleGraphLearner model
            data: Input data tensor
            
        Returns:
            Tensor of edge uncertainties
        """
        with torch.no_grad():
            # Get edge probabilities from model
            edge_probs = model(data)
            
            # Compute uncertainty as distance from 0.5 (maximum uncertainty)
            # Uncertainty is highest when probability is close to 0.5
            uncertainties = 0.5 - torch.abs(edge_probs - 0.5)
            
            return uncertainties
    
    def compute_node_uncertainty(
        self,
        edge_uncertainties: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute node uncertainty scores.
        
        This method computes uncertainty scores for each node based on
        the uncertainties of its incoming and outgoing edges.
        
        Args:
            edge_uncertainties: Tensor of edge uncertainties
            
        Returns:
            Tensor of node uncertainty scores
        """
        num_nodes = edge_uncertainties.shape[0]
        node_scores = torch.zeros(num_nodes)
        
        # Sum uncertainties for incoming and outgoing edges for each node
        for i in range(num_nodes):
            # Incoming edges (column i)
            incoming = edge_uncertainties[:, i].sum()
            # Outgoing edges (row i)
            outgoing = edge_uncertainties[i, :].sum()
            # Combined score
            node_scores[i] = incoming + outgoing
        
        # Apply node weights if available
        if self.node_weights is not None:
            if len(self.node_weights) == num_nodes:
                weight_tensor = torch.tensor(self.node_weights)
                node_scores = node_scores * weight_tensor
        
        return node_scores
    
    def select_intervention_uncertainty(
        self,
        model: SimpleGraphLearner,
        data: torch.Tensor,
        budget: float = 1.0
    ) -> Dict[str, Any]:
        """
        Select intervention using uncertainty-based strategy.
        
        This method selects the node with highest uncertainty score
        as the intervention target.
        
        Args:
            model: Trained SimpleGraphLearner model
            data: Input data tensor
            budget: Intervention budget constraint
            
        Returns:
            Dictionary specifying the selected intervention
        """
        # Compute edge uncertainties
        edge_uncertainties = self.compute_edge_uncertainties(model, data)
        
        # Compute node scores
        node_scores = self.compute_node_uncertainty(edge_uncertainties)
        
        # Select node with highest uncertainty score
        target_node = int(torch.argmax(node_scores).item())
        
        # Select intervention value
        # For simplicity, we'll use a fixed set of intervention values
        # and select one at random
        value_idx = np.random.randint(len(self.intervention_values))
        value = self.intervention_values[value_idx]
        
        return {
            'target_node': target_node,
            'value': value,
            'uncertainty_score': node_scores[target_node].item()
        }
    
    def select_intervention_random(
        self,
        model: SimpleGraphLearner,
        data: torch.Tensor,
        budget: float = 1.0
    ) -> Dict[str, Any]:
        """
        Select intervention using random strategy.
        
        This method selects a random node and intervention value.
        
        Args:
            model: Trained SimpleGraphLearner model
            data: Input data tensor
            budget: Intervention budget constraint
            
        Returns:
            Dictionary specifying the selected intervention
        """
        # Get number of nodes from data
        num_nodes = data.shape[1]
        
        # Select random node
        target_node = np.random.randint(num_nodes)
        
        # Select random intervention value
        value_idx = np.random.randint(len(self.intervention_values))
        value = self.intervention_values[value_idx]
        
        return {
            'target_node': target_node,
            'value': value
        }
    
    def select_intervention_information_gain(
        self,
        model: SimpleGraphLearner,
        data: torch.Tensor,
        budget: float = 1.0
    ) -> Dict[str, Any]:
        """
        Select intervention using information gain strategy.
        
        This method is a placeholder for more sophisticated information
        gain based strategy. Currently, it falls back to uncertainty.
        
        Args:
            model: Trained SimpleGraphLearner model
            data: Input data tensor
            budget: Intervention budget constraint
            
        Returns:
            Dictionary specifying the selected intervention
        """
        # Currently, this is a placeholder for a more sophisticated
        # information gain based strategy. For now, it falls back to
        # the uncertainty-based strategy.
        return self.select_intervention_uncertainty(model, data, budget)
    
    def select_intervention(
        self,
        model: SimpleGraphLearner,
        data: torch.Tensor,
        budget: float = 1.0
    ) -> Dict[str, Any]:
        """
        Select intervention based on the specified strategy.
        
        This method dispatches to the appropriate strategy method
        based on the strategy_type.
        
        Args:
            model: Trained SimpleGraphLearner model
            data: Input data tensor
            budget: Intervention budget constraint
            
        Returns:
            Dictionary specifying the selected intervention
        """
        if self.custom_strategy is not None:
            return self.custom_strategy(model, data, budget)
        
        if self.strategy_type == "uncertainty":
            return self.select_intervention_uncertainty(model, data, budget)
        elif self.strategy_type == "random":
            return self.select_intervention_random(model, data, budget)
        elif self.strategy_type == "information_gain":
            return self.select_intervention_information_gain(model, data, budget)
        else:
            # Should never reach here due to validation in __init__
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")
    
    def select_batch(
        self,
        model: SimpleGraphLearner,
        data: torch.Tensor,
        budget: float = 1.0,
        batch_size: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Select a batch of interventions.
        
        This method selects multiple interventions, ensuring diversity
        in the selected nodes.
        
        Args:
            model: Trained SimpleGraphLearner model
            data: Input data tensor
            budget: Intervention budget constraint
            batch_size: Number of interventions to select
            
        Returns:
            List of dictionaries specifying the selected interventions
        """
        # Ensure batch_size is positive
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        
        num_nodes = data.shape[1]
        
        # If batch_size is larger than the number of nodes,
        # limit it to the number of nodes
        batch_size = min(batch_size, num_nodes)
        
        # For uncertainty-based strategy, we want to select diverse nodes
        if self.strategy_type == "uncertainty":
            # Compute edge uncertainties
            edge_uncertainties = self.compute_edge_uncertainties(model, data)
            
            # Compute node scores
            node_scores = self.compute_node_uncertainty(edge_uncertainties)
            
            # Select nodes with highest uncertainty scores
            selected_nodes = torch.argsort(node_scores, descending=True)[:batch_size]
            
            # Create batch of interventions
            batch = []
            for node_idx in selected_nodes:
                # Select intervention value
                value_idx = np.random.randint(len(self.intervention_values))
                value = self.intervention_values[value_idx]
                
                batch.append({
                    'target_node': int(node_idx.item()),
                    'value': value,
                    'uncertainty_score': node_scores[node_idx].item()
                })
            
            return batch
        
        # For random strategy, we randomly select nodes without replacement
        elif self.strategy_type == "random":
            # Randomly select nodes without replacement
            selected_nodes = np.random.choice(num_nodes, batch_size, replace=False)
            
            # Create batch of interventions
            batch = []
            for node_idx in selected_nodes:
                # Select intervention value
                value_idx = np.random.randint(len(self.intervention_values))
                value = self.intervention_values[value_idx]
                
                batch.append({
                    'target_node': int(node_idx),
                    'value': value
                })
            
            return batch
        
        # For information gain strategy, we currently fall back to uncertainty
        elif self.strategy_type == "information_gain":
            return self.select_batch(model, data, budget, batch_size)
        
        # For custom strategy, we repeatedly call it
        elif self.custom_strategy is not None:
            batch = []
            for _ in range(batch_size):
                batch.append(self.custom_strategy(model, data, budget))
            return batch
        
        else:
            # Should never reach here due to validation in __init__
            raise ValueError(f"Unknown strategy type: {self.strategy_type}") 