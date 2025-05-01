import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

from causal_meta.meta_learning.acd_models import GraphEncoder
from causal_meta.meta_learning.graph_inference_utils import (
    threshold_graph, compute_shd, compute_precision_recall
)


def calculate_total_loss(pred_edges: torch.Tensor, true_edges: torch.Tensor,
                         sparsity_weight: float = 0.1, acyclicity_weight: float = 1.0,
                         ) -> torch.Tensor:
    """
    Calculate the total loss for graph structure learning.
    
    Args:
        pred_edges: Predicted edge probabilities
        true_edges: Ground truth adjacency matrix
        sparsity_weight: Weight for sparsity regularization
        acyclicity_weight: Weight for acyclicity constraint
        
    Returns:
        Total loss (scalar tensor)
    """
    # Binary cross entropy loss for supervised learning
    # Add a small epsilon to avoid log(0)
    epsilon = 1e-8
    bce_loss = -true_edges * torch.log(pred_edges + epsilon) - (1 - true_edges) * torch.log(1 - pred_edges + epsilon)
    bce_loss = bce_loss.mean()
    
    # Sparsity regularization (L1 penalty)
    sparsity_loss = sparsity_weight * torch.sum(torch.abs(pred_edges))
    
    # Acyclicity constraint
    n = pred_edges.shape[0]
    W_squared = pred_edges * pred_edges
    
    # Compute matrix exponential (h(W) = tr(e^(W◦W)) - n)
    try:
        exp_W = torch.matrix_exp(W_squared)
        acyclicity_loss = torch.trace(exp_W) - n
    except:
        # Fallback if matrix_exp encounters numerical issues
        identity = torch.eye(n, device=pred_edges.device)
        W_power = identity
        exp_W_approx = identity.clone()
        
        # Use first few terms of the series
        for k in range(1, 10):
            W_power = W_power @ W_squared / k
            exp_W_approx = exp_W_approx + W_power
        
        acyclicity_loss = torch.trace(exp_W_approx) - n
    
    # Apply weight
    acyclicity_loss = acyclicity_weight * acyclicity_loss
    
    # Total loss
    total_loss = bce_loss + sparsity_loss + acyclicity_loss
    
    return total_loss


class GraphStructureLoss(nn.Module):
    """
    Loss function for graph structure learning.
    
    Combines supervised loss with sparsity and acyclicity regularization.
    """
    
    def __init__(self, sparsity_weight: float = 0.1, acyclicity_weight: float = 1.0):
        """
        Initialize the loss function.
        
        Args:
            sparsity_weight: Weight for sparsity regularization
            acyclicity_weight: Weight for acyclicity constraint
        """
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.acyclicity_weight = acyclicity_weight
    
    def forward(self, pred_edges: torch.Tensor, true_edges: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the loss.
        
        Args:
            pred_edges: Predicted edge probabilities
            true_edges: Ground truth adjacency matrix
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Binary cross entropy loss for supervised learning
        # Add a small epsilon to avoid log(0)
        epsilon = 1e-8
        bce_loss = -true_edges * torch.log(pred_edges + epsilon) - (1 - true_edges) * torch.log(1 - pred_edges + epsilon)
        supervised_loss = bce_loss.mean()
        
        # Sparsity regularization (L1 penalty)
        sparsity_loss = torch.sum(torch.abs(pred_edges)) * self.sparsity_weight
        
        # Acyclicity constraint
        n = pred_edges.shape[0]
        W_squared = pred_edges * pred_edges
        
        # Compute matrix exponential (h(W) = tr(e^(W◦W)) - n)
        try:
            exp_W = torch.matrix_exp(W_squared)
            acyclicity_loss = torch.trace(exp_W) - n
        except:
            # Fallback if matrix_exp encounters numerical issues
            identity = torch.eye(n, device=pred_edges.device)
            W_power = identity
            exp_W_approx = identity.clone()
            
            # Use first few terms of the series
            for k in range(1, 10):
                W_power = W_power @ W_squared / k
                exp_W_approx = exp_W_approx + W_power
            
            acyclicity_loss = torch.trace(exp_W_approx) - n
        
        # Apply weight
        acyclicity_loss = acyclicity_loss * self.acyclicity_weight
        
        # Total loss
        total_loss = supervised_loss + sparsity_loss + acyclicity_loss
        
        # Return loss components for tracking
        loss_components = {
            'supervised_loss': supervised_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'acyclicity_loss': acyclicity_loss.item()
        }
        
        return total_loss, loss_components


def save_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                         metrics: Dict[str, Any], filepath: str) -> None:
    """
    Save model checkpoint to file.
    
    Args:
        model: Neural network model
        optimizer: Optimizer
        metrics: Dictionary of metrics to save
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, filepath)
    
    print(f"Checkpoint saved to {filepath}")


def load_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                         filepath: str) -> Tuple[nn.Module, torch.optim.Optimizer, Dict[str, Any]]:
    """
    Load model checkpoint from file.
    
    Args:
        model: Neural network model
        optimizer: Optimizer
        filepath: Path to checkpoint file
        
    Returns:
        Tuple of (model, optimizer, metrics)
    """
    # Check if checkpoint exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    # Load checkpoint
    checkpoint = torch.load(filepath)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load metrics
    metrics = checkpoint['metrics']
    
    print(f"Checkpoint loaded from {filepath}")
    
    return model, optimizer, metrics


class EarlyStopping:
    """
    Early stopping implementation to prevent overfitting.
    
    Stops training if validation loss doesn't improve for a given number of epochs.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0, restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement after which training will stop
            min_delta: Minimum change in monitored value to qualify as improvement
            restore_best_weights: Whether to restore model weights from the best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.best_weights = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, model: nn.Module, current_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            model: Current model
            current_loss: Current validation loss
            
        Returns:
            True if training should stop, False otherwise
        """
        if current_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = current_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        
        return False


class GraphEncoderTrainer:
    """
    Trainer class for the GraphEncoder model.
    
    Handles training, validation, early stopping, and performance tracking.
    """
    
    def __init__(self, model: GraphEncoder, lr: float = 0.001, sparsity_weight: float = 0.1,
                acyclicity_weight: float = 1.0, device: Optional[torch.device] = None):
        """
        Initialize the trainer.
        
        Args:
            model: GraphEncoder model
            lr: Learning rate
            sparsity_weight: Weight for sparsity regularization
            acyclicity_weight: Weight for acyclicity constraint
            device: Device to use (CPU or GPU)
        """
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Loss function
        self.loss_fn = GraphStructureLoss(sparsity_weight, acyclicity_weight)
        
        # Training history
        self.history = {'train_loss': [], 'val_loss': [], 'metrics': {}}
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_loss_components = {'supervised_loss': 0.0, 'sparsity_loss': 0.0, 'acyclicity_loss': 0.0}
        num_batches = len(train_loader)
        
        for X, y in train_loader:
            X = X.to(self.device)
            y = y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_edges = self.model(X)
            
            # Compute loss
            loss, loss_components = self.loss_fn(pred_edges, y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            for k, v in loss_components.items():
                epoch_loss_components[k] += v
        
        # Average losses
        epoch_loss /= num_batches
        for k in epoch_loss_components:
            epoch_loss_components[k] /= num_batches
        
        # Add metrics
        metrics = {
            'loss': epoch_loss,
            **epoch_loss_components
        }
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on validation data.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        val_loss_components = {'supervised_loss': 0.0, 'sparsity_loss': 0.0, 'acyclicity_loss': 0.0}
        num_batches = len(val_loader)
        
        # Track recovery metrics
        shd_values = []
        precision_values = []
        recall_values = []
        
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                
                # Forward pass
                pred_edges = self.model(X)
                
                # Compute loss
                loss, loss_components = self.loss_fn(pred_edges, y)
                
                # Track loss
                val_loss += loss.item()
                for k, v in loss_components.items():
                    val_loss_components[k] += v
                
                # Compute recovery metrics
                pred_adj = threshold_graph(pred_edges, threshold=0.5)
                
                # Compute SHD
                shd = compute_shd(y, pred_adj)
                shd_values.append(shd)
                
                # Compute precision and recall
                precision, recall = compute_precision_recall(y, pred_adj)
                precision_values.append(precision)
                recall_values.append(recall)
        
        # Average losses
        val_loss /= num_batches
        for k in val_loss_components:
            val_loss_components[k] /= num_batches
        
        # Average recovery metrics
        avg_shd = np.mean(shd_values)
        avg_precision = np.mean(precision_values)
        avg_recall = np.mean(recall_values)
        
        # Compute F1 score
        if avg_precision + avg_recall > 0:
            f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        else:
            f1_score = 0.0
        
        # Add metrics
        metrics = {
            'loss': val_loss,
            **val_loss_components,
            'shd': avg_shd,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': f1_score
        }
        
        return metrics
    
    def train(self, train_loader: DataLoader, num_epochs: int, val_loader: Optional[DataLoader] = None,
             validate: bool = True, checkpoint_dir: Optional[str] = None,
             patience: int = 10, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            num_epochs: Number of epochs to train
            val_loader: DataLoader for validation data (optional)
            validate: Whether to perform validation if val_loader is provided
            checkpoint_dir: Directory to save checkpoints (optional)
            patience: Patience for early stopping
            verbose: Whether to print progress
            
        Returns:
            Dictionary of training history
        """
        # Set up early stopping if validation is enabled
        early_stopping = None
        if validate and val_loader is not None:
            early_stopping = EarlyStopping(patience=patience)
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'metrics': {}}
        
        # Training loop
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train one epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Add to history
            history['train_loss'].append(train_metrics['loss'])
            
            # Validate if requested
            if validate and val_loader is not None:
                val_metrics = self.validate(val_loader)
                history['val_loss'].append(val_metrics['loss'])
                
                # Update metrics history
                for key, value in val_metrics.items():
                    if key not in history['metrics']:
                        history['metrics'][key] = []
                    history['metrics'][key].append(value)
                
                # Check for early stopping
                if early_stopping is not None:
                    if early_stopping(self.model, val_metrics['loss']):
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Save checkpoint if requested
            if checkpoint_dir is not None:
                checkpoint_path = os.path.join(checkpoint_dir, f"graph_encoder_epoch_{epoch+1}.pt")
                save_model_checkpoint(
                    self.model, self.optimizer, 
                    {'epoch': epoch + 1, **train_metrics}, 
                    checkpoint_path
                )
            
            # Print progress
            if verbose:
                epoch_time = time.time() - start_time
                log_message = f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s - train_loss: {train_metrics['loss']:.4f}"
                
                if validate and val_loader is not None:
                    log_message += f" - val_loss: {val_metrics['loss']:.4f}"
                    if 'f1' in val_metrics:
                        log_message += f" - F1: {val_metrics['f1']:.4f}"
                
                print(log_message)
        
        return history
    
    def train_with_curriculum(self, curriculum: List[DataLoader], epochs_per_stage: List[int],
                             val_loader: Optional[DataLoader] = None, validate: bool = True,
                             checkpoint_dir: Optional[str] = None, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train with curriculum learning (gradual increase in complexity).
        
        Args:
            curriculum: List of DataLoaders for each curriculum stage (increasing complexity)
            epochs_per_stage: Number of epochs to train for each stage
            val_loader: DataLoader for validation data (optional)
            validate: Whether to perform validation
            checkpoint_dir: Directory to save checkpoints (optional)
            verbose: Whether to print progress
            
        Returns:
            Dictionary of training history
        """
        assert len(curriculum) == len(epochs_per_stage), "Number of curriculum stages must match epochs_per_stage"
        
        # Combined history
        history = {'train_loss': [], 'val_loss': [], 'metrics': {}}
        
        # Train through each curriculum stage
        for stage, (loader, epochs) in enumerate(zip(curriculum, epochs_per_stage)):
            if verbose:
                print(f"\nCurriculum Stage {stage+1}/{len(curriculum)}")
            
            # Train for this stage
            stage_history = self.train(
                train_loader=loader,
                num_epochs=epochs,
                val_loader=val_loader,
                validate=validate,
                checkpoint_dir=checkpoint_dir,
                verbose=verbose
            )
            
            # Add to combined history
            history['train_loss'].extend(stage_history['train_loss'])
            history['val_loss'].extend(stage_history['val_loss'])
            
            for key, values in stage_history['metrics'].items():
                if key not in history['metrics']:
                    history['metrics'][key] = []
                history['metrics'][key].extend(values)
        
        return history 