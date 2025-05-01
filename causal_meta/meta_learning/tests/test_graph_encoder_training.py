import torch
import pytest
import os
import tempfile
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from causal_meta.meta_learning.acd_models import GraphEncoder
from causal_meta.meta_learning.graph_encoder_training import (
    GraphEncoderTrainer, GraphStructureLoss, calculate_total_loss,
    load_model_checkpoint, save_model_checkpoint
)


class TestGraphEncoderTraining:
    @pytest.fixture
    def sample_data(self):
        """Create a small dataset for testing."""
        # Create a sample dataset
        batch_size = 8
        n_variables = 5
        seq_length = 10
        
        # Generate random time series data
        X = torch.randn(batch_size, seq_length, n_variables)
        
        # Create a sample adjacency matrix for a DAG
        adj_matrix = np.zeros((n_variables, n_variables))
        # Add edges 0->1, 0->2, 1->3, 2->3, 2->4
        adj_matrix[0, 1] = 1
        adj_matrix[0, 2] = 1
        adj_matrix[1, 3] = 1
        adj_matrix[2, 3] = 1
        adj_matrix[2, 4] = 1
        
        # Create target edge probabilities (ground truth + noise)
        edge_probs = torch.tensor(adj_matrix, dtype=torch.float32)
        
        # Create dataset
        dataset = TensorDataset(X, edge_probs)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        return {
            'X': X,
            'edge_probs': edge_probs,
            'dataloader': dataloader,
            'n_variables': n_variables
        }
    
    def test_graph_structure_loss(self, sample_data):
        """Test the graph structure loss function."""
        X = sample_data['X']
        true_edges = sample_data['edge_probs']
        
        # Create encoder
        encoder = GraphEncoder(hidden_dim=64, attention_heads=2)
        
        # Forward pass
        pred_edges = encoder(X)
        
        # Create loss function
        loss_fn = GraphStructureLoss(sparsity_weight=0.1, acyclicity_weight=1.0)
        
        # Compute loss
        loss, loss_components = loss_fn(pred_edges, true_edges)
        
        # Check loss is a scalar
        assert isinstance(loss.item(), float)
        
        # Check loss components
        assert 'supervised_loss' in loss_components
        assert 'sparsity_loss' in loss_components
        assert 'acyclicity_loss' in loss_components
        
        # Check supervised loss is positive
        assert loss_components['supervised_loss'] > 0
    
    def test_calculate_total_loss(self, sample_data):
        """Test the total loss calculation."""
        X = sample_data['X']
        true_edges = sample_data['edge_probs']
        
        # Create encoder
        encoder = GraphEncoder(hidden_dim=64, attention_heads=2)
        
        # Forward pass
        pred_edges = encoder(X)
        
        # Compute loss with different weights
        no_regularization = calculate_total_loss(
            pred_edges, true_edges, 
            sparsity_weight=0.0, 
            acyclicity_weight=0.0
        )
        
        with_sparsity = calculate_total_loss(
            pred_edges, true_edges, 
            sparsity_weight=1.0, 
            acyclicity_weight=0.0
        )
        
        with_acyclicity = calculate_total_loss(
            pred_edges, true_edges, 
            sparsity_weight=0.0, 
            acyclicity_weight=1.0
        )
        
        with_both = calculate_total_loss(
            pred_edges, true_edges, 
            sparsity_weight=0.5, 
            acyclicity_weight=0.5
        )
        
        # Check all losses are scalars
        assert isinstance(no_regularization.item(), float)
        assert isinstance(with_sparsity.item(), float)
        assert isinstance(with_acyclicity.item(), float)
        assert isinstance(with_both.item(), float)
        
        # Regularization should increase loss
        assert with_sparsity >= no_regularization
        assert with_acyclicity >= no_regularization
    
    def test_save_load_checkpoint(self):
        """Test saving and loading model checkpoints."""
        # Create a model
        encoder = GraphEncoder(hidden_dim=64, attention_heads=2)
        
        # Create a temporary directory for saving
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "model_checkpoint.pt")
            
            # Save checkpoint
            optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
            metrics = {'loss': 1.5, 'epoch': 5}
            
            save_model_checkpoint(
                encoder, optimizer, metrics, 
                filepath=checkpoint_path
            )
            
            # Check file exists
            assert os.path.exists(checkpoint_path)
            
            # Load checkpoint
            new_encoder = GraphEncoder(hidden_dim=64, attention_heads=2)
            new_optimizer = torch.optim.Adam(new_encoder.parameters(), lr=0.002)
            loaded_metrics = {}
            
            new_encoder, new_optimizer, loaded_metrics = load_model_checkpoint(
                new_encoder, new_optimizer, checkpoint_path
            )
            
            # Check metrics loaded correctly
            assert loaded_metrics['loss'] == 1.5
            assert loaded_metrics['epoch'] == 5
            
            # Check optimizer was updated
            assert new_optimizer.param_groups[0]['lr'] == 0.001
    
    def test_graph_encoder_trainer(self, sample_data):
        """Test the GraphEncoderTrainer class."""
        dataloader = sample_data['dataloader']
        
        # Create encoder
        encoder = GraphEncoder(hidden_dim=64, attention_heads=2)
        
        # Create trainer
        trainer = GraphEncoderTrainer(
            model=encoder,
            lr=0.001,
            sparsity_weight=0.1,
            acyclicity_weight=1.0
        )
        
        # Train for one epoch
        history = trainer.train(
            train_loader=dataloader,
            num_epochs=1,
            validate=False
        )
        
        # Check history contains expected metrics
        assert 'train_loss' in history
        assert len(history['train_loss']) == 1
        
        # Check model was updated (parameters changed)
        for param in encoder.parameters():
            # At least some parameters should have gradients
            if param.grad is not None:
                assert param.grad.norm().item() > 0
                break
    
    def test_curriculum_learning(self, sample_data):
        """Test curriculum learning functionality."""
        X = sample_data['X']
        edge_probs = sample_data['edge_probs']
        
        # Create multiple datasets of increasing complexity
        n_variables = sample_data['n_variables']
        datasets = []
        
        # Simple dataset (less variables/edges)
        X_simple = X[:, :, :3]
        edge_probs_simple = edge_probs[:3, :3]
        simple_dataset = TensorDataset(X_simple, edge_probs_simple)
        simple_loader = DataLoader(simple_dataset, batch_size=4, shuffle=True)
        
        # Medium dataset
        X_medium = X[:, :, :4]
        edge_probs_medium = edge_probs[:4, :4]
        medium_dataset = TensorDataset(X_medium, edge_probs_medium)
        medium_loader = DataLoader(medium_dataset, batch_size=4, shuffle=True)
        
        # Complex dataset (full)
        complex_dataset = TensorDataset(X, edge_probs)
        complex_loader = DataLoader(complex_dataset, batch_size=4, shuffle=True)
        
        # Create curriculum
        curriculum = [simple_loader, medium_loader, complex_loader]
        
        # Create encoder
        encoder = GraphEncoder(hidden_dim=64, attention_heads=2)
        
        # Create trainer
        trainer = GraphEncoderTrainer(
            model=encoder,
            lr=0.001,
            sparsity_weight=0.1,
            acyclicity_weight=1.0
        )
        
        # Train with curriculum
        history = trainer.train_with_curriculum(
            curriculum=curriculum,
            epochs_per_stage=[1, 1, 1],
            validate=False
        )
        
        # Check history contains expected metrics for all stages
        assert 'train_loss' in history
        assert len(history['train_loss']) == 3  # One entry per curriculum stage 