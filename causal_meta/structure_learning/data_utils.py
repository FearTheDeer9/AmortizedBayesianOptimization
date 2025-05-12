"""
Data generation and processing utilities for causal structure learning.

This module provides functions for generating observational and interventional data
from structural causal models, creating intervention masks, and converting data
to formats suitable for neural network processing.
"""

import numpy as np
import pandas as pd
import torch
import random
from typing import List, Dict, Tuple, Any, Optional, Union

from causal_meta.environments.scm import StructuralCausalModel


def generate_observational_data(
    scm: StructuralCausalModel,
    n_samples: int = 100,
    as_tensor: bool = False
) -> Union[pd.DataFrame, torch.Tensor]:
    """
    Generate observational data from a structural causal model.
    
    Args:
        scm: Structural causal model to sample from
        n_samples: Number of samples to generate
        as_tensor: Whether to return a PyTorch tensor (True) or pandas DataFrame (False)
        
    Returns:
        Generated observational data as a DataFrame or tensor
    """
    # Sample data from the SCM
    data = scm.sample_data(sample_size=n_samples, as_array=False)
    
    # Convert to tensor if requested
    if as_tensor:
        return convert_to_tensor(data)
    
    return data


def generate_interventional_data(
    scm: StructuralCausalModel,
    node: str,
    value: float,
    n_samples: int = 100,
    as_tensor: bool = False
) -> Union[pd.DataFrame, torch.Tensor]:
    """
    Generate interventional data from a structural causal model.
    
    Args:
        scm: Structural causal model to sample from
        node: Node to intervene on
        value: Value to set the intervened node to
        n_samples: Number of samples to generate
        as_tensor: Whether to return a PyTorch tensor (True) or pandas DataFrame (False)
        
    Returns:
        Generated interventional data as a DataFrame or tensor
    """
    # Create a deep copy of the SCM to avoid modifying the original
    scm_copy = scm  # Not copying since do_intervention is currently in-place
    
    # Perform the intervention
    scm_copy.do_intervention(node, value)
    
    # Sample data from the intervened SCM
    data = scm_copy.sample_data(sample_size=n_samples, as_array=False)
    
    # Convert to tensor if requested
    if as_tensor:
        return convert_to_tensor(data)
    
    return data


def generate_random_intervention_data(
    scm: StructuralCausalModel,
    n_samples: int = 100,
    intervention_values: Optional[Dict[str, List[float]]] = None,
    as_tensor: bool = False
) -> Tuple[Union[pd.DataFrame, torch.Tensor], str, float]:
    """
    Generate data with a random intervention.
    
    Args:
        scm: Structural causal model to sample from
        n_samples: Number of samples to generate
        intervention_values: Dictionary mapping node names to lists of possible
                            intervention values. If None, random values will be used.
        as_tensor: Whether to return a PyTorch tensor (True) or pandas DataFrame (False)
        
    Returns:
        Tuple of (generated data, intervened node, intervention value)
    """
    # Get all variable names from the SCM
    variable_names = scm.get_variable_names()
    
    # Randomly select a node to intervene on
    intervened_node = random.choice(variable_names)
    
    # Determine the intervention value
    if intervention_values is not None and intervened_node in intervention_values:
        # Use a value from the provided options
        intervention_value = random.choice(intervention_values[intervened_node])
    else:
        # Use a random value between -2 and 2
        intervention_value = random.uniform(-2.0, 2.0)
    
    # Generate interventional data
    data = generate_interventional_data(
        scm=scm,
        node=intervened_node,
        value=intervention_value,
        n_samples=n_samples,
        as_tensor=as_tensor
    )
    
    return data, intervened_node, intervention_value


def create_intervention_mask(
    data: pd.DataFrame, 
    intervened_nodes: List[str]
) -> np.ndarray:
    """
    Create a binary mask indicating which nodes were intervened on.
    
    Args:
        data: The data for which to create the mask
        intervened_nodes: List of nodes that were intervened on
        
    Returns:
        Binary mask with shape (n_samples, n_variables)
    """
    # Initialize mask with zeros
    n_samples = len(data)
    n_variables = len(data.columns)
    mask = np.zeros((n_samples, n_variables))
    
    # Set intervened nodes to 1
    for node in intervened_nodes:
        if node in data.columns:
            node_idx = list(data.columns).index(node)
            mask[:, node_idx] = 1
    
    return mask


def convert_to_tensor(
    data: pd.DataFrame,
    intervention_mask: Optional[np.ndarray] = None
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Convert data and optional intervention mask to PyTorch tensors.
    
    Args:
        data: DataFrame to convert
        intervention_mask: Optional intervention mask to convert
        
    Returns:
        If mask is provided: Tuple of (data tensor, mask tensor)
        If no mask: Just the data tensor
    """
    # Convert data to tensor
    data_tensor = torch.tensor(data.values, dtype=torch.float32)
    
    # If mask is provided, convert it too and return both
    if intervention_mask is not None:
        mask_tensor = torch.tensor(intervention_mask, dtype=torch.float32)
        return data_tensor, mask_tensor
    
    # Otherwise just return the data tensor
    return data_tensor 