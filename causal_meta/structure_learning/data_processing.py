"""
Data processing utilities for neural network models in causal structure learning.

This module provides classes and functions for processing data for neural network
models, including dataset creation, normalization, and train-test splitting.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Union, Tuple, List, Dict, Any, Optional

from causal_meta.environments.scm import StructuralCausalModel
from causal_meta.structure_learning.data_utils import (
    generate_observational_data,
    convert_to_tensor
)


class CausalDataset(Dataset):
    """
    Dataset for causal data including observations, interventions, and graph structure.
    
    This dataset can handle:
    - Observational data only
    - Observational data with interventional data
    - Data with adjacency matrix
    - Data with intervention masks
    
    Args:
        data: DataFrame containing the data (observational or interventional)
        intervention_mask: Binary mask indicating which nodes were intervened on
        adjacency_matrix: Adjacency matrix of the causal graph (if available)
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        intervention_mask: Optional[np.ndarray] = None,
        adjacency_matrix: Optional[np.ndarray] = None
    ):
        """Initialize the dataset with data and optional masks/graph."""
        # Store data column names
        self.column_names = list(data.columns)
        
        # Store the data as a tensor
        self.data = torch.tensor(data.values, dtype=torch.float32)
        
        # Store the intervention mask if provided
        self.has_mask = intervention_mask is not None
        if self.has_mask:
            self.intervention_mask = torch.tensor(
                intervention_mask, dtype=torch.float32
            )
        else:
            self.intervention_mask = None
        
        # Store the adjacency matrix if provided
        self.has_adjacency = adjacency_matrix is not None
        if self.has_adjacency:
            self.adjacency_matrix = torch.tensor(
                adjacency_matrix, dtype=torch.float32
            )
        else:
            self.adjacency_matrix = None
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Get a sample from the dataset.
        
        Returns different combinations depending on what's available:
        - Data only: if no mask or adjacency
        - (Data, Mask): if mask is available but no adjacency
        - (Data, Adjacency): if adjacency is available but no mask
        - (Data, Mask, Adjacency): if both mask and adjacency are available
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Sample data and optional mask/adjacency
        """
        # Get the data for this sample
        data_sample = self.data[idx]
        
        # Return different combinations based on what's available
        if self.has_mask and self.has_adjacency:
            return data_sample, self.intervention_mask[idx], self.adjacency_matrix
        elif self.has_mask:
            return data_sample, self.intervention_mask[idx]
        elif self.has_adjacency:
            return data_sample, self.adjacency_matrix
        else:
            return data_sample


def create_dataloader(
    data: Optional[pd.DataFrame] = None,
    intervention_mask: Optional[np.ndarray] = None,
    adjacency_matrix: Optional[np.ndarray] = None,
    scm: Optional[StructuralCausalModel] = None,
    n_samples: int = 100,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a PyTorch DataLoader for causal data.
    
    This function can create a DataLoader from either:
    - Existing data (DataFrame + optional mask + optional adjacency)
    - An SCM (which will be used to generate data)
    
    Args:
        data: DataFrame containing the data (observational or interventional)
        intervention_mask: Binary mask indicating which nodes were intervened on
        adjacency_matrix: Adjacency matrix of the causal graph (if available)
        scm: Structural causal model to generate data from (if data not provided)
        n_samples: Number of samples to generate if using an SCM
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker threads for loading data
        
    Returns:
        PyTorch DataLoader for the data
    """
    # If data is not provided but SCM is, generate data from SCM
    if data is None and scm is not None:
        data = generate_observational_data(
            scm=scm,
            n_samples=n_samples,
            as_tensor=False
        )
    
    # Create the dataset
    dataset = CausalDataset(
        data=data,
        intervention_mask=intervention_mask,
        adjacency_matrix=adjacency_matrix
    )
    
    # Define collate function to handle different return types
    def collate_fn(batch):
        if isinstance(batch[0], tuple):
            # If batch contains tuples, separate the elements
            transposed = list(zip(*batch))
            
            # Stack the data tensors
            data_batch = torch.stack(transposed[0])
            
            # Handle the other elements
            if len(transposed) >= 2:
                if isinstance(transposed[1][0], torch.Tensor) and transposed[1][0].dim() == 1:
                    # It's an intervention mask, stack it
                    mask_batch = torch.stack(transposed[1])
                    if len(transposed) == 3:
                        # If we have adjacency matrix too, it's the same for all samples, so just use the first one
                        adj_batch = transposed[2][0]
                        return data_batch, mask_batch, adj_batch
                    return data_batch, mask_batch
                else:
                    # It's an adjacency matrix, which should be repeated for each item in the batch
                    adj_matrix = transposed[1][0]  # All items have the same adjacency matrix
                    batch_size = len(batch)
                    # Repeat the adjacency matrix for each item in the batch
                    adj_batch = adj_matrix.repeat(batch_size, 1, 1)
                    return data_batch, adj_batch
        else:
            # If batch contains tensors, stack them
            return torch.stack(batch)
    
    # Create and return the DataLoader
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


def normalize_data(
    data: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    preserve_interventions: bool = False,
    intervention_mask: Optional[np.ndarray] = None,
    node_names: Optional[List[str]] = None
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, StandardScaler]]:
    """
    Normalize data using StandardScaler.
    
    Args:
        data: Input data as DataFrame
        scaler: Pre-fitted scaler (optional)
        preserve_interventions: Whether to preserve intervention values (default: False)
        intervention_mask: Binary mask indicating interventions (required if preserve_interventions=True)
        node_names: List of node names (required if preserve_interventions=True)
        
    Returns:
        If scaler is None: Tuple of (normalized data, fitted scaler)
        If scaler is provided: Normalized data
    """
    # Print debug info at the start
    print(f"normalize_data: data columns = {data.columns}")
    
    # Check if data is a DataFrame or convert it
    if not isinstance(data, pd.DataFrame):
        # Convert numpy array to DataFrame if needed
        if node_names is None:
            # Use generic column names
            node_names = [f'x{i}' for i in range(data.shape[1])]
        data = pd.DataFrame(data, columns=node_names)
    
    # Make a copy to avoid modifying the original
    data_copy = data.copy()
    
    # If preserving interventions, extract their values
    intervention_values = {}
    if preserve_interventions and intervention_mask is not None:
        if node_names is None:
            node_names = list(data_copy.columns)
            
        for i, node in enumerate(node_names):
            if node not in data_copy.columns:
                continue
                
            # Find rows where this node is intervened
            mask = intervention_mask[:, i] == 1
            if mask.sum() > 0:
                # Store the intervention value (assuming all interventions on this node have the same value)
                intervention_values[node] = data_copy.loc[mask, node].iloc[0]
    
    # Create and fit scaler if not provided
    if scaler is None:
        # Use a simpler approach for handling interventions
        if preserve_interventions:
            # Create a copy without interventions for fitting the scaler
            temp_data = data_copy.copy()
            
            # Get non-intervention data by creating a mask of non-intervened values
            non_intervention_mask = np.ones((len(data_copy), len(data_copy.columns)), dtype=bool)
            
            # Update mask to exclude intervention points
            for i, node in enumerate(node_names):
                if node in data_copy.columns:
                    col_idx = data_copy.columns.get_loc(node)
                    for j in range(len(intervention_mask)):
                        if intervention_mask[j, i] == 1:  # If this is an intervention
                            non_intervention_mask[j, col_idx] = False
            
            # Create a version of the data with interventions set to NaN
            for i in range(len(temp_data)):
                for j in range(len(temp_data.columns)):
                    if not non_intervention_mask[i, j]:
                        temp_data.iloc[i, j] = np.nan
            
            # Now create a scaler that ignores NaNs by handling each column separately
            scaler = StandardScaler()
            scaler.fit(data_copy)  # Initialize with a dummy fit first
            
            # For each column, fit a separate scaler on non-NaN values
            feature_means = np.zeros(len(data_copy.columns))
            feature_scales = np.ones(len(data_copy.columns))
            
            for j, col in enumerate(temp_data.columns):
                col_values = temp_data[col].dropna().values
                if len(col_values) > 0:  # Only compute if we have non-NaN values
                    feature_means[j] = np.mean(col_values)
                    feature_scales[j] = np.std(col_values, ddof=0) or 1.0  # Default to 1.0 if std is 0
            
            # Manually override the scaler's parameters
            scaler.mean_ = feature_means
            scaler.scale_ = feature_scales
            scaler.var_ = feature_scales ** 2
        else:
            # Standard fit on all data
            scaler = StandardScaler()
            scaler.fit(data_copy)
        
        # Transform the data
        normalized_values = scaler.transform(data_copy)
        normalized_data = pd.DataFrame(
            normalized_values,
            columns=data_copy.columns,
            index=data_copy.index
        )
        
        # Set intervention values directly in normalized data
        if preserve_interventions:
            for i, node in enumerate(node_names):
                if node in intervention_values:
                    # Find rows where this node is intervened
                    mask = intervention_mask[:, i] == 1
                    if mask.sum() > 0:
                        # Calculate the normalized intervention value
                        col_idx = data_copy.columns.get_loc(node)
                        norm_value = (intervention_values[node] - scaler.mean_[col_idx]) / scaler.scale_[col_idx]
                        # Set all intervened values to this normalized value
                        normalized_data.loc[mask, node] = norm_value
        
        return normalized_data, scaler
    
    else:
        # Use provided scaler to transform the data
        normalized_values = scaler.transform(data_copy)
        normalized_data = pd.DataFrame(
            normalized_values,
            columns=data_copy.columns,
            index=data_copy.index
        )
        
        # Set intervention values directly in normalized data
        if preserve_interventions:
            for i, node in enumerate(node_names):
                if node in intervention_values:
                    # Find rows where this node is intervened
                    mask = intervention_mask[:, i] == 1
                    if mask.sum() > 0:
                        # Calculate the normalized intervention value
                        col_idx = data_copy.columns.get_loc(node)
                        norm_value = (intervention_values[node] - scaler.mean_[col_idx]) / scaler.scale_[col_idx]
                        # Set all intervened values to this normalized value
                        normalized_data.loc[mask, node] = norm_value
        
        return normalized_data


def create_train_test_split(
    data: pd.DataFrame,
    intervention_mask: Optional[np.ndarray] = None,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    stratify: Optional[np.ndarray] = None
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame],
    Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]
]:
    """
    Split data into training and test sets, with optional matching intervention mask split.
    
    Args:
        data: DataFrame to split
        intervention_mask: Binary mask indicating interventions
        test_size: Proportion of data to use for test set
        random_state: Random state for reproducibility
        stratify: Array-like for stratified splitting
        
    Returns:
        If intervention_mask is None:
            Tuple of (train_data, test_data)
        If intervention_mask is not None:
            Tuple of (train_data, test_data, train_mask, test_mask)
    """
    # Split the data
    if intervention_mask is None:
        # Simple split without mask
        train_data, test_data = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        return train_data, test_data
    else:
        # Get the indices for train and test sets
        indices = np.arange(len(data))
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        # Split the data using the indices
        train_data = data.iloc[train_indices]
        test_data = data.iloc[test_indices]
        
        # Split the intervention mask using the same indices
        train_mask = intervention_mask[train_indices]
        test_mask = intervention_mask[test_indices]
        
        return train_data, test_data, train_mask, test_mask 

# Add this utility function to help with scaler.inverse_transform
def inverse_transform_to_df(
    scaler: StandardScaler,
    data: Union[pd.DataFrame, np.ndarray],
    original_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Apply inverse_transform and convert the result to a DataFrame.
    
    Args:
        scaler: StandardScaler to use for inverse transformation
        data: Data to inverse transform
        original_columns: Column names to use for the result DataFrame
        
    Returns:
        DataFrame with inverse transformed data
    """
    # Get column names
    if original_columns is None and isinstance(data, pd.DataFrame):
        original_columns = data.columns
    
    # Convert to numpy if it's a DataFrame
    if isinstance(data, pd.DataFrame):
        values = data.values
        index = data.index
    else:
        values = data
        index = None
    
    # Apply inverse transform
    inverse_values = scaler.inverse_transform(values)
    
    # Convert back to DataFrame
    if original_columns is not None:
        if index is not None:
            return pd.DataFrame(inverse_values, columns=original_columns, index=index)
        else:
            return pd.DataFrame(inverse_values, columns=original_columns)
    else:
        if index is not None:
            return pd.DataFrame(inverse_values, index=index)
        else:
            return pd.DataFrame(inverse_values) 