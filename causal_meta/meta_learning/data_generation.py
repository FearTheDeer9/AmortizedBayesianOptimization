import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from torch.utils.data import Dataset, DataLoader

from causal_meta.environments.scm import StructuralCausalModel
from causal_meta.graph.causal_graph import CausalGraph


class SyntheticDataGenerator:
    """
    Generator for synthetic observational and interventional data from a given SCM.
    
    This class provides methods to generate various types of data for training and
    evaluating neural causal discovery methods.
    """
    
    def __init__(self, scm: StructuralCausalModel):
        """
        Initialize the data generator with a structural causal model.
        
        Args:
            scm: Structural causal model for data generation
        """
        self.scm = scm
        self.graph = scm.get_causal_graph()
        self.n_variables = len(self.graph.get_nodes())
        self.nodes = list(self.graph.get_nodes())
    
    def generate_observational_data(self, n_samples: int = 1000, add_noise: bool = True,
                                   noise_type: str = 'gaussian', noise_scale: float = 0.1) -> torch.Tensor:
        """
        Generate observational data from the SCM.
        
        Args:
            n_samples: Number of samples to generate
            add_noise: Whether to add observation noise
            noise_type: Type of noise ('gaussian' or 'uniform')
            noise_scale: Scale of the noise
            
        Returns:
            Tensor of shape [n_samples, n_variables] with observational data
        """
        # Sample from the SCM
        data_np = self.scm.sample_data(sample_size=n_samples)
        
        # Convert to tensor
        data = torch.tensor(data_np, dtype=torch.float32)
        
        # Add observation noise if requested
        if add_noise:
            data = self.add_noise(data, noise_type, noise_scale)
        
        return data
    
    def generate_interventional_data(self, target_node: Any, intervention_value: float,
                                    n_samples: int = 1000, add_noise: bool = True,
                                    noise_type: str = 'gaussian', noise_scale: float = 0.1) -> torch.Tensor:
        """
        Generate data under an intervention on a single node.
        
        Args:
            target_node: Node to intervene on
            intervention_value: Value to set the node to
            n_samples: Number of samples to generate
            add_noise: Whether to add observation noise
            noise_type: Type of noise ('gaussian' or 'uniform')
            noise_scale: Scale of the noise
            
        Returns:
            Tensor of shape [n_samples, n_variables] with interventional data
        """
        # Perform intervention in the SCM
        intervened_scm = self.scm.do_intervention(target_node, intervention_value)
        
        # Sample from intervened SCM
        data_np = intervened_scm.sample_data(sample_size=n_samples)
        
        # Convert to tensor
        data = torch.tensor(data_np, dtype=torch.float32)
        
        # Add observation noise if requested
        if add_noise:
            data = self.add_noise(data, noise_type, noise_scale)
        
        return data
    
    def generate_multiple_interventions(self, interventions: Dict[Any, float],
                                       n_samples: int = 1000, add_noise: bool = True,
                                       noise_type: str = 'gaussian', noise_scale: float = 0.1) -> torch.Tensor:
        """
        Generate data under multiple interventions.
        
        Args:
            interventions: Dictionary mapping nodes to intervention values
            n_samples: Number of samples to generate
            add_noise: Whether to add observation noise
            noise_type: Type of noise ('gaussian' or 'uniform')
            noise_scale: Scale of the noise
            
        Returns:
            Tensor of shape [n_samples, n_variables] with interventional data
        """
        # Create a copy of the SCM
        current_scm = self.scm
        
        # Apply each intervention
        for node, value in interventions.items():
            current_scm = current_scm.do_intervention(node, value)
        
        # Sample from the intervened SCM
        data_np = current_scm.sample_data(sample_size=n_samples)
        
        # Convert to tensor
        data = torch.tensor(data_np, dtype=torch.float32)
        
        # Add observation noise if requested
        if add_noise:
            data = self.add_noise(data, noise_type, noise_scale)
        
        return data
    
    def add_noise(self, data: torch.Tensor, noise_type: str = 'gaussian',
                 noise_scale: float = 0.1) -> torch.Tensor:
        """
        Add noise to the data.
        
        Args:
            data: Input data tensor
            noise_type: Type of noise ('gaussian' or 'uniform')
            noise_scale: Scale of the noise
            
        Returns:
            Data with added noise
        """
        if noise_type == 'gaussian':
            noise = torch.randn_like(data) * noise_scale
        elif noise_type == 'uniform':
            noise = (torch.rand_like(data) * 2 - 1) * noise_scale
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        return data + noise
    
    def generate_batch(self, batch_size: int = 32, seq_length: int = 10,
                      add_noise: bool = True, noise_type: str = 'gaussian',
                      noise_scale: float = 0.1, time_dependent: bool = False,
                      interventions: Optional[Dict[Any, float]] = None) -> torch.Tensor:
        """
        Generate a batch of time series data.
        
        Args:
            batch_size: Number of samples in the batch
            seq_length: Length of each time series
            add_noise: Whether to add observation noise
            noise_type: Type of noise ('gaussian' or 'uniform')
            noise_scale: Scale of the noise
            time_dependent: Whether to generate time-dependent sequences
            interventions: Optional interventions to apply
            
        Returns:
            Tensor of shape [batch_size, seq_length, n_variables]
        """
        # Choose which SCM to sample from
        if interventions is not None:
            # Apply interventions
            current_scm = self.scm
            for node, value in interventions.items():
                current_scm = current_scm.do_intervention(node, value)
        else:
            # Use original SCM
            current_scm = self.scm
        
        if time_dependent:
            # Generate time-dependent sequences
            # Start with initial values
            batch_data = []
            
            # Generate initial values
            initial_values = current_scm.sample_data(sample_size=batch_size)
            
            # Convert to tensor
            initial_tensor = torch.tensor(initial_values, dtype=torch.float32)
            
            # For each time step, generate next values based on previous ones
            for t in range(seq_length):
                if t == 0:
                    # Use initial values
                    batch_data.append(initial_tensor)
                else:
                    # Generate new values based on the previous ones
                    # This is a simplified approach - in reality, you'd need a more
                    # complex time-dependent SCM
                    prev_values = batch_data[-1].numpy()
                    new_values = current_scm.sample_data(sample_size=batch_size)
                    
                    # Add some continuity with previous values
                    alpha = 0.8  # Controls how much the new values depend on previous ones
                    time_values = alpha * prev_values + (1 - alpha) * new_values
                    
                    # Convert to tensor
                    time_tensor = torch.tensor(time_values, dtype=torch.float32)
                    batch_data.append(time_tensor)
            
            # Stack along time dimension
            batch = torch.stack(batch_data, dim=1)
        else:
            # Generate independent sequences
            batch_np = np.zeros((batch_size, seq_length, self.n_variables))
            
            # Generate independent samples for each time step
            for t in range(seq_length):
                batch_np[:, t, :] = current_scm.sample_data(sample_size=batch_size)
            
            # Convert to tensor
            batch = torch.tensor(batch_np, dtype=torch.float32)
        
        # Add observation noise if requested
        if add_noise:
            batch = self.add_noise(batch, noise_type, noise_scale)
        
        return batch
    
    def get_adjacency_matrix(self) -> torch.Tensor:
        """
        Get the adjacency matrix of the causal graph.
        
        Returns:
            Binary adjacency matrix tensor
        """
        # Get adjacency matrix from causal graph
        adj_matrix = self.graph.get_adjacency_matrix()
        
        # Convert to tensor
        adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
        
        return adj_tensor


class GraphDataset(Dataset):
    """
    Dataset for causal graph discovery from time series data.
    
    Provides samples of time series data along with ground truth adjacency matrices.
    """
    
    def __init__(self, scm: StructuralCausalModel, n_samples: int = 1000,
                seq_length: int = 10, add_noise: bool = True,
                noise_type: str = 'gaussian', noise_scale: float = 0.1,
                time_dependent: bool = False):
        """
        Initialize the dataset.
        
        Args:
            scm: Structural causal model
            n_samples: Number of samples in the dataset
            seq_length: Length of each time series
            add_noise: Whether to add observation noise
            noise_type: Type of noise ('gaussian' or 'uniform')
            noise_scale: Scale of the noise
            time_dependent: Whether to generate time-dependent sequences
        """
        self.generator = SyntheticDataGenerator(scm)
        self.n_samples = n_samples
        self.seq_length = seq_length
        self.add_noise = add_noise
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.time_dependent = time_dependent
        
        # Cache adjacency matrix
        self.adj_matrix = self.generator.get_adjacency_matrix()
        
        # Generate data
        self.data = self._generate_data()
    
    def _generate_data(self) -> List[torch.Tensor]:
        """Generate all samples for the dataset."""
        data = []
        
        for _ in range(self.n_samples):
            # Generate a single sample (time series)
            sample = self.generator.generate_batch(
                batch_size=1,
                seq_length=self.seq_length,
                add_noise=self.add_noise,
                noise_type=self.noise_type,
                noise_scale=self.noise_scale,
                time_dependent=self.time_dependent
            )
            
            # Remove batch dimension
            sample = sample.squeeze(0)
            
            data.append(sample)
        
        return data
    
    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (time_series, adjacency_matrix)
        """
        return self.data[idx], self.adj_matrix


class GraphDataLoader(DataLoader):
    """
    Data loader for causal graph discovery datasets.
    
    Provides batches of data with appropriate collation.
    """
    
    def __init__(self, dataset: GraphDataset, batch_size: int = 32,
                shuffle: bool = True, num_workers: int = 0):
        """
        Initialize the data loader.
        
        Args:
            dataset: GraphDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
        """
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate function for batching.
        
        Args:
            batch: List of (time_series, adjacency_matrix) tuples
            
        Returns:
            Tuple of batched tensors (time_series_batch, adjacency_matrix_batch)
        """
        # Separate time series and adjacency matrices
        time_series = [item[0] for item in batch]
        adj_matrices = [item[1] for item in batch]
        
        # Stack into batches
        time_series_batch = torch.stack(time_series, dim=0)
        adj_matrices_batch = torch.stack(adj_matrices, dim=0)
        
        return time_series_batch, adj_matrices_batch 