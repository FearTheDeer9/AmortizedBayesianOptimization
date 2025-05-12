"""
Linear Structural Causal Model (SCM) generator for causal graph structure learning.
"""

import numpy as np
from typing import Optional, Dict, List, Union, Tuple, Any

from causal_meta.graph import CausalGraph
from causal_meta.environments.scm import StructuralCausalModel
from causal_meta.environments.mechanisms import LinearMechanism


class LinearSCMGenerator:
    """
    Generator for linear Structural Causal Models (SCMs).

    This class provides utilities for creating linear SCMs from adjacency matrices,
    leveraging the existing StructuralCausalModel implementation.
    """

    @staticmethod
    def generate_linear_scm(
        adj_matrix: np.ndarray,
        noise_scale: float = 0.1,
        min_weight: float = -2.0,
        max_weight: float = 2.0,
        seed: Optional[int] = None
    ) -> StructuralCausalModel:
        """
        Generate a linear Structural Causal Model (SCM) from an adjacency matrix.

        Args:
            adj_matrix (np.ndarray): Adjacency matrix representation of a DAG
            noise_scale (float): Scale of Gaussian noise for all variables
            min_weight (float): Minimum value for random edge weights
            max_weight (float): Maximum value for random edge weights
            seed (Optional[int]): Random seed for reproducibility

        Returns:
            StructuralCausalModel: A linear SCM with random weights
        """
        # Validate the adjacency matrix
        if not np.allclose(adj_matrix, np.triu(adj_matrix, k=1), rtol=1e-5, atol=1e-8):
            raise ValueError("Adjacency matrix must be upper triangular (representing a DAG)")

        # Set random seed
        np.random.seed(seed)
        random_state = np.random.RandomState(seed)

        # Create a CausalGraph from the adjacency matrix
        n = adj_matrix.shape[0]

        # Create empty SCM first
        scm = StructuralCausalModel(random_state=seed)
        
        # Add variables (valid Python identifiers: x0, x1, ...)
        node_names = [f"x{i}" for i in range(n)]
        for name in node_names:
            scm.add_variable(name)
        
        # Define causal relationships (build the graph structure)
        for i in range(n):
            for j in range(n):
                if adj_matrix[i, j] > 0:
                    scm.define_causal_relationship(node_names[j], [node_names[i]])  # i -> j means i is a parent of j

        # Generate random weights for each edge
        weights = LinearSCMGenerator.generate_linear_weights(
            adj_matrix, min_weight=min_weight, max_weight=max_weight, random_state=random_state
        )

        # Define the structural equations
        for i in range(n):
            node_id = node_names[i]
            parents = scm.get_parents(node_id)
            
            if not parents:  # No parents (exogenous/root node)
                # Define as just noise
                scm.define_probabilistic_equation(
                    variable=node_id,
                    equation_function=lambda noise: noise,
                    noise_distribution=lambda sample_size, random_state=None: (random_state or random_state or np.random).normal(0, noise_scale, sample_size)
                )
            else:
                # Get the weights for this node's parents
                node_weights = {p: weights[node_names.index(p), i] for p in parents}
                
                # Use the built-in method for linear Gaussian equations
                scm.define_linear_gaussian_equation(
                    variable=node_id,
                    coefficients=node_weights,
                    intercept=0.0,
                    noise_std=noise_scale
                )

        return scm

    @staticmethod
    def generate_linear_weights(
        adj_matrix: np.ndarray,
        min_weight: float = -2.0,
        max_weight: float = 2.0,
        random_state: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        """
        Generate random weights for a linear SCM based on an adjacency matrix.

        Args:
            adj_matrix (np.ndarray): Adjacency matrix representation of a DAG
            min_weight (float): Minimum value for random weights
            max_weight (float): Maximum value for random weights
            random_state (Optional[np.random.RandomState]): Random state for reproducibility

        Returns:
            np.ndarray: Matrix of edge weights with same shape as adj_matrix
        """
        if random_state is None:
            random_state = np.random.RandomState()

        n = adj_matrix.shape[0]
        weights = np.zeros_like(adj_matrix, dtype=float)
        
        # Only generate weights for existing edges
        for i in range(n):
            for j in range(n):
                if adj_matrix[i, j] > 0:
                    weights[i, j] = random_state.uniform(min_weight, max_weight)
                    
        return weights

    @staticmethod
    def add_noise_distributions(
        scm: StructuralCausalModel,
        noise_scale: float = 0.1,
        seed: Optional[int] = None
    ) -> StructuralCausalModel:
        """
        Add Gaussian noise distributions to all variables in the SCM.

        Args:
            scm (StructuralCausalModel): The SCM to add noise to
            noise_scale (float): Scale of Gaussian noise
            seed (Optional[int]): Random seed for reproducibility

        Returns:
            StructuralCausalModel: The SCM with noise distributions added
        """
        random_state = np.random.RandomState(seed)
        
        for var in scm.get_variable_names():
            # Skip if already has a noise distribution
            if var in scm._exogenous_functions:
                continue
                
            # Add noise distribution
            scm._exogenous_functions[var] = lambda n, rs=random_state: rs.normal(0, noise_scale, n)
            
        return scm 