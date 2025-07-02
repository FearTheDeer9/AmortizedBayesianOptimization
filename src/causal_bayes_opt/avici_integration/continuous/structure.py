"""
Differentiable Structure Learning.

This module implements full causal structure learning using continuous 
parent probabilities with differentiable acyclicity constraints.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional

from .model import ContinuousParentSetPredictionModel


class DifferentiableStructureLearning(hk.Module):
    """
    Learn full causal structure using continuous parent probabilities.
    
    This model learns parent probabilities for ALL variables simultaneously
    while enforcing acyclicity constraints through differentiable penalties.
    """
    
    def __init__(self,
                 n_vars: int,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 acyclicity_penalty_weight: float = 1.0,
                 name: str = "DifferentiableStructureLearning"):
        super().__init__(name=name)
        self.n_vars = n_vars
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.acyclicity_penalty_weight = acyclicity_penalty_weight
    
    def __call__(self, 
                 data: jnp.ndarray,        # [N, d, 3] intervention data
                 is_training: bool = True  # Training mode flag
                 ) -> jnp.ndarray:         # [d, d] parent probability matrix
        """
        Learn parent probabilities for all variables simultaneously.
        
        Args:
            data: Intervention data [N, d, 3]
            is_training: Whether model is in training mode
            
        Returns:
            Parent probability matrix [d, d] where entry (i, j) is the
            probability that variable j is a parent of variable i.
            Diagonal entries are always 0 (no self-loops).
        """
        N, d, channels = data.shape
        assert d == self.n_vars, f"Expected {self.n_vars} variables, got {d}"
        
        # Learn parent probabilities for each variable
        def single_variable_parents(target_var):
            model = ContinuousParentSetPredictionModel(
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads
            )
            return model(data, target_var, is_training)
        
        # Vectorize across all target variables
        all_parent_probs = jax.vmap(single_variable_parents)(jnp.arange(d))  # [d, d]
        
        # Ensure diagonal is zero (no self-loops)
        mask = 1.0 - jnp.eye(d)
        all_parent_probs = all_parent_probs * mask
        
        # Renormalize each row to sum to 1.0
        row_sums = jnp.sum(all_parent_probs, axis=1, keepdims=True)
        all_parent_probs = all_parent_probs / (row_sums + 1e-8)
        
        # Apply acyclicity constraint if training
        if is_training and self.acyclicity_penalty_weight > 0:
            acyclic_probs = self.enforce_acyclicity(all_parent_probs)
            return acyclic_probs
        
        return all_parent_probs
    
    def enforce_acyclicity(self, parent_probs: jnp.ndarray) -> jnp.ndarray:
        """
        Apply soft acyclicity constraint to parent probabilities.
        
        Args:
            parent_probs: Parent probability matrix [d, d]
            
        Returns:
            Adjusted parent probabilities with reduced cycles
        """
        # Compute acyclicity penalty
        penalty = self.compute_acyclicity_penalty(parent_probs)
        
        # Apply gradient-based adjustment (simple approximation)
        # In practice, this could be more sophisticated
        acyclicity_grad = jax.grad(self.compute_acyclicity_penalty)(parent_probs)
        
        # Soft constraint: reduce probabilities in direction of penalty gradient
        adjustment = self.acyclicity_penalty_weight * acyclicity_grad
        adjusted_probs = parent_probs - 0.01 * adjustment  # Small step size
        
        # Ensure probabilities remain valid
        adjusted_probs = jnp.maximum(adjusted_probs, 0.0)
        
        # Renormalize rows
        row_sums = jnp.sum(adjusted_probs, axis=1, keepdims=True)
        adjusted_probs = adjusted_probs / (row_sums + 1e-8)
        
        # Ensure diagonal remains zero
        mask = 1.0 - jnp.eye(self.n_vars)
        adjusted_probs = adjusted_probs * mask
        
        return adjusted_probs
    
    def compute_acyclicity_penalty(self, parent_probs: jnp.ndarray) -> float:
        """
        Compute differentiable acyclicity penalty.
        
        Uses the trace of matrix powers as a measure of cycles.
        For a DAG, the trace of A^n should be zero for n >= 1.
        
        Args:
            parent_probs: Parent probability matrix [d, d]
            
        Returns:
            Scalar penalty value (higher = more cycles)
        """
        d = parent_probs.shape[0]
        
        # Compute powers of the adjacency matrix
        penalty = 0.0
        current_power = parent_probs
        
        for power in range(1, d + 1):
            # Add trace of current power
            trace = jnp.trace(current_power)
            penalty += trace / power  # Weight by inverse power
            
            # Compute next power
            if power < d:
                current_power = jnp.matmul(current_power, parent_probs)
        
        return penalty
    
    def compute_structure_entropy(self, parent_probs: jnp.ndarray) -> float:
        """
        Compute entropy of the learned structure.
        
        Args:
            parent_probs: Parent probability matrix [d, d]
            
        Returns:
            Total entropy across all parent distributions
        """
        # Entropy for each variable's parent distribution
        entropies = -jnp.sum(parent_probs * jnp.log(parent_probs + 1e-8), axis=1)
        return jnp.sum(entropies)
    
    def get_adjacency_matrix(self, 
                           parent_probs: jnp.ndarray, 
                           threshold: float = 0.5) -> jnp.ndarray:
        """
        Convert parent probabilities to binary adjacency matrix.
        
        Args:
            parent_probs: Parent probability matrix [d, d]
            threshold: Probability threshold for edge inclusion
            
        Returns:
            Binary adjacency matrix [d, d]
        """
        return (parent_probs > threshold).astype(jnp.float32)
    
    def get_topological_order(self, parent_probs: jnp.ndarray) -> jnp.ndarray:
        """
        Compute approximate topological ordering from parent probabilities.
        
        Args:
            parent_probs: Parent probability matrix [d, d]
            
        Returns:
            Approximate topological ordering [d] (variable indices)
        """
        # Compute number of expected parents for each variable
        num_parents = jnp.sum(parent_probs, axis=1)
        
        # Sort by number of parents (ascending)
        # Variables with fewer parents come first in topological order
        return jnp.argsort(num_parents)


class StructureLearningLoss(hk.Module):
    """Loss functions for structure learning."""
    
    def __init__(self, 
                 acyclicity_weight: float = 1.0,
                 sparsity_weight: float = 0.1,
                 name: str = "StructureLearningLoss"):
        super().__init__(name=name)
        self.acyclicity_weight = acyclicity_weight
        self.sparsity_weight = sparsity_weight
    
    def __call__(self,
                 parent_probs: jnp.ndarray,     # [d, d] predicted probabilities
                 data: jnp.ndarray,             # [N, d, 3] intervention data
                 true_structure: Optional[jnp.ndarray] = None  # [d, d] true adjacency (if available)
                 ) -> dict:
        """
        Compute structure learning loss.
        
        Args:
            parent_probs: Predicted parent probabilities [d, d]
            data: Intervention data [N, d, 3]
            true_structure: True adjacency matrix [d, d] (optional)
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Data likelihood loss (simplified - could be more sophisticated)
        data_loss = self._compute_data_likelihood_loss(parent_probs, data)
        losses['data_likelihood'] = data_loss
        
        # Acyclicity penalty
        acyclicity_penalty = self._compute_acyclicity_loss(parent_probs)
        losses['acyclicity'] = self.acyclicity_weight * acyclicity_penalty
        
        # Sparsity penalty (encourage sparse structures)
        sparsity_penalty = self._compute_sparsity_loss(parent_probs)
        losses['sparsity'] = self.sparsity_weight * sparsity_penalty
        
        # Supervised loss (if true structure available)
        if true_structure is not None:
            supervised_loss = self._compute_supervised_loss(parent_probs, true_structure)
            losses['supervised'] = supervised_loss
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses
    
    def _compute_data_likelihood_loss(self, 
                                    parent_probs: jnp.ndarray, 
                                    data: jnp.ndarray) -> float:
        """Simplified data likelihood loss."""
        # This is a placeholder - in practice, this would involve
        # computing the likelihood of data given the structure
        N, d, _ = data.shape
        
        # Simple loss: penalize high uncertainty
        entropies = -jnp.sum(parent_probs * jnp.log(parent_probs + 1e-8), axis=1)
        return jnp.mean(entropies)
    
    def _compute_acyclicity_loss(self, parent_probs: jnp.ndarray) -> float:
        """Acyclicity constraint loss."""
        d = parent_probs.shape[0]
        penalty = 0.0
        current_power = parent_probs
        
        for power in range(1, d + 1):
            trace = jnp.trace(current_power)
            penalty += trace ** 2  # Quadratic penalty
            if power < d:
                current_power = jnp.matmul(current_power, parent_probs)
        
        return penalty
    
    def _compute_sparsity_loss(self, parent_probs: jnp.ndarray) -> float:
        """Sparsity penalty to encourage sparse structures."""
        # L1 penalty on probabilities
        return jnp.sum(parent_probs)
    
    def _compute_supervised_loss(self, 
                               parent_probs: jnp.ndarray, 
                               true_structure: jnp.ndarray) -> float:
        """Supervised loss when true structure is available."""
        # Binary cross-entropy between probabilities and true adjacency
        true_structure = true_structure.astype(jnp.float32)
        
        # Clip probabilities for numerical stability
        probs_clipped = jnp.clip(parent_probs, 1e-8, 1 - 1e-8)
        
        # Binary cross-entropy
        bce = -(true_structure * jnp.log(probs_clipped) + 
                (1 - true_structure) * jnp.log(1 - probs_clipped))
        
        return jnp.mean(bce)