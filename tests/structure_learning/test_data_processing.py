"""Tests for neural network data processing utilities."""

import unittest
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from causal_meta.structure_learning.graph_generators import RandomDAGGenerator
from causal_meta.structure_learning.scm_generators import LinearSCMGenerator
from causal_meta.structure_learning.data_utils import (
    generate_observational_data,
    generate_interventional_data,
    create_intervention_mask
)

# Import the module to be tested (not implemented yet, will fail initially)
# from causal_meta.structure_learning.data_processing import (
#     CausalDataset,
#     create_dataloader,
#     normalize_data,
#     create_train_test_split
# )


class TestDataProcessingBase(unittest.TestCase):
    """Base class for data processing tests with common setup."""
    
    def setUp(self):
        """Create a simple SCM and generate data for testing."""
        # Fix random seed for reproducibility
        self.seed = 42
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # Create a simple DAG adjacency matrix
        self.adj_matrix = np.array([
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0]
        ])
        self.node_names = [f"x{i}" for i in range(3)]
        self.noise_scale = 0.1
        self.n_samples = 100
        
        # Generate an SCM
        self.scm = LinearSCMGenerator.generate_linear_scm(
            adj_matrix=self.adj_matrix,
            noise_scale=self.noise_scale,
            seed=self.seed
        )
        
        # Generate observational data
        self.obs_data = generate_observational_data(
            scm=self.scm,
            n_samples=self.n_samples,
            as_tensor=False
        )
        
        # Generate interventional data for each node
        self.int_data_list = []
        self.int_values = [1.0, -1.0, 2.0]
        self.int_masks_list = []
        
        for i, node in enumerate(self.node_names):
            int_data = generate_interventional_data(
                scm=self.scm,
                node=node,
                value=self.int_values[i],
                n_samples=self.n_samples // 3,  # Fewer samples per intervention
                as_tensor=False
            )
            self.int_data_list.append(int_data)
            
            # Create intervention mask
            int_mask = create_intervention_mask(
                data=int_data,
                intervened_nodes=[node]
            )
            self.int_masks_list.append(int_mask)
        
        # Combine interventional data
        self.int_data = pd.concat(self.int_data_list)
        self.int_masks = np.vstack(self.int_masks_list)


class TestCausalDataset(TestDataProcessingBase):
    """Test cases for the CausalDataset class."""
    
    def test_dataset_initialization(self):
        """Test that the dataset is initialized correctly."""
        from causal_meta.structure_learning.data_processing import CausalDataset
        
        # Create a dataset with observational data only
        obs_dataset = CausalDataset(self.obs_data)
        
        # Check length
        self.assertEqual(len(obs_dataset), len(self.obs_data))
        
        # Create a dataset with both observational and interventional data
        full_dataset = CausalDataset(
            data=pd.concat([self.obs_data, self.int_data]),
            intervention_mask=np.vstack([
                np.zeros((len(self.obs_data), 3)),  # No interventions for obs data
                self.int_masks
            ])
        )
        
        # Check length
        self.assertEqual(len(full_dataset), len(self.obs_data) + len(self.int_data))
    
    def test_dataset_getitem(self):
        """Test that __getitem__ returns the correct data."""
        from causal_meta.structure_learning.data_processing import CausalDataset
        
        # Create dataset with interventional data and masks
        dataset = CausalDataset(
            data=self.int_data,
            intervention_mask=self.int_masks
        )
        
        # Get first item
        item = dataset[0]
        
        # Check that it returns a tuple of (data, mask)
        self.assertIsInstance(item, tuple)
        self.assertEqual(len(item), 2)
        
        # Check data shape
        data, mask = item
        self.assertIsInstance(data, torch.Tensor)
        self.assertEqual(data.shape, (3,))  # 3 nodes
        
        # Check mask shape
        self.assertIsInstance(mask, torch.Tensor)
        self.assertEqual(mask.shape, (3,))  # 3 nodes
    
    def test_dataset_without_mask(self):
        """Test that the dataset works without intervention masks."""
        from causal_meta.structure_learning.data_processing import CausalDataset
        
        # Create dataset without mask
        dataset = CausalDataset(self.obs_data)
        
        # Get first item
        item = dataset[0]
        
        # If no mask is provided, should return just the data
        self.assertIsInstance(item, torch.Tensor)
        self.assertEqual(item.shape, (3,))  # 3 nodes
    
    def test_dataset_with_adjacency(self):
        """Test that the dataset works with adjacency matrix."""
        from causal_meta.structure_learning.data_processing import CausalDataset
        
        # Create dataset with adjacency matrix
        dataset = CausalDataset(
            data=self.obs_data,
            intervention_mask=None,
            adjacency_matrix=self.adj_matrix
        )
        
        # Get first item
        item = dataset[0]
        
        # Should return a tuple of (data, adjacency)
        self.assertIsInstance(item, tuple)
        self.assertEqual(len(item), 2)
        
        # Check data shape
        data, adj = item
        self.assertIsInstance(data, torch.Tensor)
        self.assertEqual(data.shape, (3,))  # 3 nodes
        
        # Check adjacency shape
        self.assertIsInstance(adj, torch.Tensor)
        self.assertEqual(adj.shape, (3, 3))  # 3x3 matrix
    
    def test_dataset_with_mask_and_adjacency(self):
        """Test dataset with both intervention mask and adjacency matrix."""
        from causal_meta.structure_learning.data_processing import CausalDataset
        
        # Create dataset with mask and adjacency
        dataset = CausalDataset(
            data=self.int_data,
            intervention_mask=self.int_masks,
            adjacency_matrix=self.adj_matrix
        )
        
        # Get first item
        item = dataset[0]
        
        # Should return a tuple of (data, mask, adjacency)
        self.assertIsInstance(item, tuple)
        self.assertEqual(len(item), 3)
        
        # Check all components
        data, mask, adj = item
        self.assertEqual(data.shape, (3,))  # 3 nodes
        self.assertEqual(mask.shape, (3,))  # 3 nodes
        self.assertEqual(adj.shape, (3, 3))  # 3x3 matrix


class TestCreateDataloader(TestDataProcessingBase):
    """Test cases for the create_dataloader function."""
    
    def test_create_dataloader(self):
        """Test creating a DataLoader from data."""
        from causal_meta.structure_learning.data_processing import create_dataloader
        
        # Create dataloader with observational data
        loader = create_dataloader(self.obs_data, batch_size=10)
        
        # Check it's a DataLoader
        self.assertIsInstance(loader, DataLoader)
        
        # Check batch size
        self.assertEqual(loader.batch_size, 10)
        
        # Check number of batches (approximately)
        self.assertGreaterEqual(len(loader), len(self.obs_data) // 10)
    
    def test_create_dataloader_with_mask(self):
        """Test creating a DataLoader with intervention mask."""
        from causal_meta.structure_learning.data_processing import create_dataloader
        
        # Create dataloader with interventional data and mask
        loader = create_dataloader(
            data=self.int_data,
            intervention_mask=self.int_masks,
            batch_size=8
        )
        
        # Check first batch
        batch = next(iter(loader))
        
        # Should be a tuple of (data, mask)
        self.assertIsInstance(batch, tuple)
        self.assertEqual(len(batch), 2)
        
        # Check shapes
        data_batch, mask_batch = batch
        self.assertEqual(data_batch.shape, (torch.Size([8, 3])))  # [batch_size, nodes]
        self.assertEqual(mask_batch.shape, (torch.Size([8, 3])))  # [batch_size, nodes]
    
    def test_create_dataloader_with_adjacency(self):
        """Test creating a DataLoader with adjacency matrix."""
        from causal_meta.structure_learning.data_processing import create_dataloader
        
        # Create dataloader with observational data and adjacency matrix
        loader = create_dataloader(
            data=self.obs_data,
            adjacency_matrix=self.adj_matrix,
            batch_size=16
        )
        
        # Check first batch
        batch = next(iter(loader))
        
        # Should be a tuple of (data, adjacency)
        self.assertIsInstance(batch, tuple)
        self.assertEqual(len(batch), 2)
        
        # Check shapes
        data_batch, adj_batch = batch
        self.assertEqual(data_batch.shape, (torch.Size([16, 3])))  # [batch_size, nodes]
        self.assertEqual(adj_batch.shape, (torch.Size([16, 3, 3])))  # [batch_size, nodes, nodes]
        
        # Adjacency should be the same for all samples in batch
        for i in range(16):
            self.assertTrue(torch.all(adj_batch[i] == adj_batch[0]))
    
    def test_create_dataloader_with_scm(self):
        """Test creating a DataLoader directly from an SCM."""
        from causal_meta.structure_learning.data_processing import create_dataloader
        
        # Create dataloader from SCM directly
        loader = create_dataloader(
            scm=self.scm,
            n_samples=50,
            batch_size=10
        )
        
        # Check it's a DataLoader
        self.assertIsInstance(loader, DataLoader)
        
        # Check first batch shape
        data_batch = next(iter(loader))
        self.assertEqual(data_batch.shape, (torch.Size([10, 3])))  # [batch_size, nodes]


class TestNormalizeData(TestDataProcessingBase):
    """Test cases for the normalize_data function."""
    
    def test_normalize_data(self):
        """Test that data is correctly normalized."""
        from causal_meta.structure_learning.data_processing import normalize_data
        
        # Normalize the observational data
        normalized_data, scaler = normalize_data(self.obs_data)
        
        # Check type
        self.assertIsInstance(normalized_data, pd.DataFrame)
        
        # Check that the mean is close to 0 and std close to 1 for each column
        for col in normalized_data.columns:
            self.assertAlmostEqual(normalized_data[col].mean(), 0, delta=0.1)
            self.assertAlmostEqual(normalized_data[col].std(), 1, delta=0.1)
        
        # Check that the scaler is returned
        self.assertIsNotNone(scaler)
    
    def test_apply_existing_scaler(self):
        """Test applying an existing scaler to new data."""
        from causal_meta.structure_learning.data_processing import normalize_data, inverse_transform_to_df
        
        # Normalize the observational data
        _, scaler = normalize_data(self.obs_data)
        
        # Normalize the interventional data using the same scaler
        normalized_int_data = normalize_data(
            self.int_data,
            scaler=scaler
        )
        
        # Check type
        self.assertIsInstance(normalized_int_data, pd.DataFrame)
        
        # The stats won't necessarily be mean=0, std=1 for the interventional data
        # but we can check that the transformation was applied correctly by inversing
        # Check reconstruction
        reconstructed_data = inverse_transform_to_df(
            scaler=scaler, 
            data=normalized_int_data, 
            original_columns=self.int_data.columns
        )
        
        # Check shape
        self.assertEqual(reconstructed_data.shape, self.int_data.shape)
        
        # Column names should match
        for col in self.int_data.columns:
            self.assertIn(col, reconstructed_data.columns)
    
    def test_normalize_with_intervention(self):
        """Test normalization while preserving intervention values."""
        from causal_meta.structure_learning.data_processing import normalize_data
        
        # Normalize data but preserve intervention values on 'x0'
        normalized_data, _ = normalize_data(
            self.int_data,
            preserve_interventions=True,
            intervention_mask=self.int_masks,
            node_names=self.node_names
        )
        
        # The first third of the data has intervention on x0
        # Find rows where x0 is intervened
        mask_x0 = self.int_masks[:, 0] == 1
        
        # For these rows, x0 should be a fixed value (the intervention value)
        intervened_values = normalized_data.loc[mask_x0, 'x0']
        
        # All values should be the same (or very close) 
        # for rows where x0 is intervened
        if len(intervened_values) > 0:
            self.assertLess(intervened_values.std(), 0.01)


class TestTrainTestSplit(TestDataProcessingBase):
    """Test cases for the create_train_test_split function."""
    
    def test_split_proportions(self):
        """Test that the split has correct proportions."""
        from causal_meta.structure_learning.data_processing import create_train_test_split
        
        # Create a train-test split
        train_data, test_data = create_train_test_split(
            self.obs_data,
            test_size=0.2,
            random_state=self.seed
        )
        
        # Check that the proportions are approximately correct
        expected_train_size = int(len(self.obs_data) * 0.8)
        expected_test_size = len(self.obs_data) - expected_train_size
        
        self.assertEqual(len(train_data), expected_train_size)
        self.assertEqual(len(test_data), expected_test_size)
    
    def test_split_with_masks(self):
        """Test splitting both data and mask together."""
        from causal_meta.structure_learning.data_processing import create_train_test_split
        
        # Create a train-test split with masks
        result = create_train_test_split(
            self.int_data,
            intervention_mask=self.int_masks,
            test_size=0.3,
            random_state=self.seed
        )
        
        # Should return a tuple of (train_data, test_data, train_mask, test_mask)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        
        train_data, test_data, train_mask, test_mask = result
        
        # Check proportions
        expected_train_size = int(len(self.int_data) * 0.7)
        expected_test_size = len(self.int_data) - expected_train_size
        
        self.assertEqual(len(train_data), expected_train_size)
        self.assertEqual(len(test_data), expected_test_size)
        
        # Check mask shapes
        self.assertEqual(train_mask.shape, (expected_train_size, 3))
        self.assertEqual(test_mask.shape, (expected_test_size, 3))
    
    def test_split_with_stratify(self):
        """Test that stratification maintains intervention distribution."""
        from causal_meta.structure_learning.data_processing import create_train_test_split
        
        # Create a categorical variable based on which node is intervened
        strata = np.zeros(len(self.int_masks))
        for i in range(3):
            strata[self.int_masks[:, i] == 1] = i + 1
        
        # Create a train-test split with stratification
        result = create_train_test_split(
            self.int_data,
            intervention_mask=self.int_masks,
            test_size=0.3,
            random_state=self.seed,
            stratify=strata
        )
        
        train_data, test_data, train_mask, test_mask = result
        
        # Count interventions in train set
        train_interventions = [train_mask[:, i].sum() for i in range(3)]
        
        # Count interventions in test set
        test_interventions = [test_mask[:, i].sum() for i in range(3)]
        
        # Calculate expected proportions
        expected_train_ratio = len(train_data) / len(self.int_data)
        
        # The proportion of each intervention type should be approximately the same
        # in both train and test sets
        for i in range(3):
            total_interventions = train_interventions[i] + test_interventions[i]
            if total_interventions > 0:  # Avoid division by zero
                train_ratio = train_interventions[i] / total_interventions
                self.assertAlmostEqual(train_ratio, expected_train_ratio, delta=0.1)


if __name__ == "__main__":
    unittest.main() 