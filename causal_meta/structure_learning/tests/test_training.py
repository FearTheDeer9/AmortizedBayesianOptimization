"""
Tests for the training module in structure learning.
"""

import os
import shutil
import pytest
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

from causal_meta.structure_learning.simple_graph_learner import SimpleGraphLearner
from causal_meta.structure_learning.training import (
    calculate_structural_hamming_distance,
    evaluate_graph,
    train_step,
    SimpleGraphLearnerTrainer,
    train_simple_graph_learner
)
from causal_meta.structure_learning.graph_generators import RandomDAGGenerator
from causal_meta.structure_learning.data_utils import (
    generate_observational_data,
    generate_interventional_data
)


class TestTrainingModule:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data and model for testing."""
        # Create a simple adjacency matrix
        n_nodes = 4
        adj_matrix = np.zeros((n_nodes, n_nodes))
        adj_matrix[0, 1] = 1
        adj_matrix[1, 2] = 1
        adj_matrix[2, 3] = 1
        
        # Create random data
        np.random.seed(42)
        obs_data = np.random.randn(50, n_nodes)
        
        # Create intervention mask
        int_mask = np.zeros((20, n_nodes))
        int_mask[:, 1] = 1
        
        # Create intervention data
        int_data = np.random.randn(20, n_nodes)
        
        # Convert to tensors
        obs_tensor = torch.tensor(obs_data, dtype=torch.float32)
        int_tensor = torch.tensor(int_data, dtype=torch.float32)
        int_mask_tensor = torch.tensor(int_mask, dtype=torch.float32)
        adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
        
        # Initialize model
        model = SimpleGraphLearner(
            input_dim=n_nodes,
            hidden_dim=32,
            num_layers=1,
            dropout=0.0,
            sparsity_weight=0.1,
            acyclicity_weight=1.0
        )
        
        # Make sure all parameters require gradients
        for param in model.parameters():
            param.requires_grad = True
        
        # Create dummy forward pass to initialize model
        with torch.no_grad():
            model(obs_tensor)
        
        return {
            'n_nodes': n_nodes,
            'adj_matrix': adj_matrix,
            'adj_tensor': adj_tensor,
            'obs_data': obs_data,
            'obs_tensor': obs_tensor,
            'int_data': int_data,
            'int_tensor': int_tensor,
            'int_mask': int_mask,
            'int_mask_tensor': int_mask_tensor,
            'model': model
        }
    
    def test_structural_hamming_distance(self, sample_data):
        """Test SHD calculation."""
        # Create predicted adjacency matrix with some mistakes
        pred_adj = sample_data['adj_matrix'].copy()
        pred_adj[0, 2] = 1  # Extra edge
        pred_adj[2, 3] = 0  # Missing edge
        
        # Calculate SHD
        shd = calculate_structural_hamming_distance(pred_adj, sample_data['adj_matrix'])
        
        # Expected SHD: 1 extra edge + 1 missing edge = 2
        assert shd == 2
    
    def test_evaluate_graph(self, sample_data):
        """Test graph evaluation metrics."""
        # Create predicted adjacency matrix with some mistakes
        pred_adj = sample_data['adj_matrix'].copy()
        pred_adj[0, 2] = 1  # Extra edge
        pred_adj[2, 3] = 0  # Missing edge
        
        # Calculate metrics
        metrics = evaluate_graph(pred_adj, sample_data['adj_matrix'])
        
        # Check if all expected metrics are present
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'shd' in metrics
        
        # Check specific values
        assert metrics['shd'] == 2
        assert metrics['accuracy'] == 14/16  # 14 correct out of 16 elements
    
    def test_train_step(self, sample_data):
        """Test training step."""
        # Get model and data
        model = sample_data['model']
        data = sample_data['obs_tensor']
        
        # Create adjacency tensor with requires_grad=True
        adj_tensor = sample_data['adj_tensor'].clone().detach().requires_grad_(True)
        
        # Create optimizer
        optimizer = Adam(model.parameters(), lr=0.01)
        
        # Perform training step
        loss_dict = train_step(
            model=model,
            data=data,
            intervention_mask=None,
            true_adj=adj_tensor,
            optimizer=optimizer
        )
        
        # Check if loss was calculated
        assert 'total_loss' in loss_dict
        assert isinstance(loss_dict['total_loss'], float)
        assert loss_dict['total_loss'] > 0
    
    def test_trainer_initialization(self, sample_data):
        """Test SimpleGraphLearnerTrainer initialization."""
        # Create trainer with existing model
        trainer1 = SimpleGraphLearnerTrainer(
            model=sample_data['model'],
            lr=0.01
        )
        
        # Check trainer properties
        assert trainer1.model is sample_data['model']
        assert isinstance(trainer1.optimizer, Adam)
        assert isinstance(trainer1.history, dict)
        
        # Create trainer without model
        trainer2 = SimpleGraphLearnerTrainer(
            input_dim=sample_data['n_nodes'],
            hidden_dim=32,
            lr=0.01
        )
        
        # Check trainer properties
        assert isinstance(trainer2.model, SimpleGraphLearner)
        assert trainer2.model.input_dim == sample_data['n_nodes']
    
    def test_train_epoch(self, sample_data):
        """Test training for one epoch."""
        # Create trainer
        trainer = SimpleGraphLearnerTrainer(
            model=sample_data['model'],
            lr=0.01
        )
        
        # Create adjacency tensor with requires_grad=True
        adj_tensor = sample_data['adj_tensor'].clone().detach().requires_grad_(True)
        
        # Train for one epoch
        losses = trainer.train_epoch(
            train_data=sample_data['obs_tensor'],
            true_adj=adj_tensor,
            batch_size=16
        )
        
        # Check losses
        assert 'total_loss' in losses
        assert isinstance(losses['total_loss'], float)
        assert losses['total_loss'] > 0
    
    def test_evaluate(self, sample_data):
        """Test model evaluation."""
        # Create trainer
        trainer = SimpleGraphLearnerTrainer(
            model=sample_data['model'],
            lr=0.01
        )
        
        # Evaluate model
        metrics = trainer.evaluate(
            data=sample_data['obs_tensor'],
            true_adj=sample_data['adj_tensor']
        )
        
        # Check metrics
        assert 'total_loss' in metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'shd' in metrics
    
    def test_simple_training(self, sample_data):
        """Test a simple training run."""
        # Create trainer
        trainer = SimpleGraphLearnerTrainer(
            model=sample_data['model'],
            lr=0.01
        )
        
        # Create adjacency tensor with requires_grad=True
        adj_tensor = sample_data['adj_tensor'].clone().detach().requires_grad_(True)
        
        # Train for a few epochs
        history = trainer.train(
            train_data=sample_data['obs_tensor'],
            true_adj=adj_tensor,
            batch_size=16,
            epochs=5,
            verbose=False
        )
        
        # Check history
        assert 'train_loss' in history
        assert len(history['train_loss']) == 5
    
    def test_training_with_validation(self, sample_data):
        """Test training with validation data."""
        # Create trainer
        trainer = SimpleGraphLearnerTrainer(
            model=sample_data['model'],
            lr=0.01
        )
        
        # Create adjacency tensor with requires_grad=True
        adj_tensor = sample_data['adj_tensor'].clone().detach().requires_grad_(True)
        
        # Split data for validation
        train_size = 40
        train_data = sample_data['obs_tensor'][:train_size]
        val_data = sample_data['obs_tensor'][train_size:]
        
        # Train with validation
        history = trainer.train(
            train_data=train_data,
            val_data=val_data,
            true_adj=adj_tensor,
            batch_size=16,
            epochs=5,
            verbose=False
        )
        
        # Check history
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 5
        assert len(history['val_loss']) == 5
    
    def test_save_load_model(self, sample_data, tmp_path):
        """Test saving and loading the model."""
        # Create sample model with specific architecture
        model = SimpleGraphLearner(
            input_dim=sample_data['n_nodes'],
            hidden_dim=32,  # Use the same architecture for saving and loading
            num_layers=1,
            dropout=0.0,
            sparsity_weight=0.1,
            acyclicity_weight=1.0
        )
        
        # Create trainer
        trainer = SimpleGraphLearnerTrainer(
            model=model,
            lr=0.01
        )
        
        # Create adjacency tensor with requires_grad=True
        adj_tensor = sample_data['adj_tensor'].clone().detach().requires_grad_(True)
        
        # Train for a few epochs
        trainer.train(
            train_data=sample_data['obs_tensor'],
            true_adj=adj_tensor,
            batch_size=16,
            epochs=2,
            verbose=False
        )
        
        # Save the model
        save_path = tmp_path / "model.pt"
        trainer.save(save_path)
        
        # Check if file exists
        assert os.path.exists(save_path)
        
        # Create a new trainer with exactly the same architecture
        new_trainer = SimpleGraphLearnerTrainer(
            input_dim=sample_data['n_nodes'],
            hidden_dim=32,  # Match the hidden dimension
            num_layers=1,  # Match the number of layers
            dropout=0.0,  # Match the dropout
            lr=0.01
        )
        
        # Load the model
        new_trainer.load(save_path)
        
        # Check if history was loaded
        assert len(new_trainer.history['train_loss']) == 2
    
    def test_train_simple_graph_learner(self, sample_data):
        """Test the high-level training function."""
        # Create adjacency tensor with requires_grad=True
        adj_tensor = sample_data['adj_tensor'].clone().detach().requires_grad_(True)
        
        # Train model
        model, history = train_simple_graph_learner(
            train_data=sample_data['obs_tensor'],
            true_adj=adj_tensor,
            hidden_dim=32,
            num_layers=1,
            batch_size=16,
            epochs=3,
            verbose=False
        )
        
        # Check results
        assert isinstance(model, SimpleGraphLearner)
        assert 'train_loss' in history
        assert len(history['train_loss']) == 3 