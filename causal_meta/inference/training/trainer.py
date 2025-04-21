"""
Training utilities for GNN encoder-decoder models.

This module provides functions and classes for training and evaluating
GNN-based encoder-decoder models for graph representation learning.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import time
from typing import Dict, List, Optional, Union, Tuple, Callable, Any
from pathlib import Path
import logging

import torch_geometric
from torch_geometric.data import Data, Batch

from causal_meta.inference.models.wrapper import EncoderDecoderWrapper, create_model
from causal_meta.inference.training.data_loader import create_data_loaders


class Trainer:
    """
    Trainer class for GNN encoder-decoder models.

    This class handles the training and evaluation of GNN-based 
    encoder-decoder models for graph representation learning.
    """

    def __init__(
        self,
        model: EncoderDecoderWrapper,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        log_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize the trainer.

        Args:
            model: EncoderDecoderWrapper model to train
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            test_loader: Optional DataLoader for test data
            optimizer: PyTorch optimizer (defaults to Adam if None)
            scheduler: Learning rate scheduler
            device: Device to use for training ('cuda', 'cpu')
            log_dir: Directory to save logs
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler
        self.device = device

        # Create directories if needed
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.checkpoint_dir = Path(
            checkpoint_dir) if checkpoint_dir else Path("checkpoints")

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Move model to device
        self.model.to(device)

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("GNN Trainer")

        # Initialize training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
            "learning_rates": []
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train the model for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = {}

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = batch.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            reconstructed_batch, latent_rep = self.model(batch)

            # Compute loss
            losses = self.model.compute_loss(
                batch, reconstructed_batch, latent_rep)
            loss = losses["total_loss"]

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Track loss and metrics
            batch_size = 1 if not hasattr(
                batch, 'num_graphs') else batch.num_graphs
            epoch_loss += loss.item() * batch_size

            # Log progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(self.train_loader):
                self.logger.info(f"Epoch {epoch} [{batch_idx+1}/{len(self.train_loader)}] "
                                 f"Loss: {loss.item():.4f}")

        # Calculate average loss
        epoch_loss /= len(self.train_loader.dataset)

        # Evaluate on training set for metrics
        train_metrics = self.evaluate(self.train_loader, "Training")
        train_metrics["loss"] = epoch_loss

        # Step scheduler if provided
        if self.scheduler is not None:
            self.scheduler.step()
            # Record learning rate
            self.history["learning_rates"].append(
                self.optimizer.param_groups[0]['lr']
            )

        return train_metrics

    def validate(self) -> Dict[str, float]:
        """
        Validate the model on the validation set.

        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}

        return self.evaluate(self.val_loader, "Validation")

    def evaluate(self, data_loader: DataLoader, split_name: str = "Test") -> Dict[str, float]:
        """
        Evaluate the model on the given data loader.

        Args:
            data_loader: DataLoader to evaluate on
            split_name: Name of the data split for logging

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        metrics_sum = {}

        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = batch.to(self.device)

                # Forward pass
                reconstructed_batch, latent_rep = self.model(batch)

                # Compute loss
                losses = self.model.compute_loss(
                    batch, reconstructed_batch, latent_rep)
                loss = losses["total_loss"]

                # Track loss
                batch_size = 1 if not hasattr(
                    batch, 'num_graphs') else batch.num_graphs
                total_loss += loss.item() * batch_size

                # Get metrics for this batch
                metrics = self.model.evaluate(batch)

                # Accumulate metrics
                for k, v in metrics.items():
                    if k in metrics_sum:
                        metrics_sum[k] += v * batch_size
                    else:
                        metrics_sum[k] = v * batch_size

        # Calculate averages
        avg_loss = total_loss / len(data_loader.dataset)
        avg_metrics = {k: v / len(data_loader.dataset)
                       for k, v in metrics_sum.items()}
        avg_metrics["loss"] = avg_loss

        # Log results
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k,
                                 v in avg_metrics.items()])
        self.logger.info(f"{split_name} metrics: {metrics_str}")

        return avg_metrics

    def train(self, num_epochs: int, early_stopping_patience: int = 10) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train for
            early_stopping_patience: Number of epochs to wait for validation improvement

        Returns:
            Training history dictionary
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")

        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0

        # Train for specified number of epochs
        for epoch in range(1, num_epochs + 1):
            # Training
            epoch_start_time = time.time()
            train_metrics = self.train_epoch(epoch)
            epoch_end_time = time.time()

            # Record metrics
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_metrics"].append(train_metrics)

            # Validation
            val_metrics = self.validate() if self.val_loader is not None else {}
            if val_metrics:
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_metrics"].append(val_metrics)

                # Check for improvement
                val_loss = val_metrics["loss"]
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0

                    # Save best model
                    self.save_checkpoint(epoch, best=True)
                else:
                    patience_counter += 1

            # Log epoch summary
            epoch_time = epoch_end_time - epoch_start_time
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s - "
                             f"Train loss: {train_metrics['loss']:.4f}" +
                             (f" - Val loss: {val_metrics['loss']:.4f}" if val_metrics else ""))

            # Save checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(epoch)

            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch} epochs. "
                                 f"Best epoch was {best_epoch}.")
                break

        # Save final model
        self.save_checkpoint(epoch, final=True)

        # Load best model for evaluation
        if os.path.exists(self.checkpoint_dir / "best_model.pt"):
            self.load_checkpoint(self.checkpoint_dir / "best_model.pt")

        # Final evaluation on test set
        if self.test_loader is not None:
            test_metrics = self.evaluate(self.test_loader, "Test")
            self.logger.info(f"Final test metrics: {test_metrics}")

        return self.history

    def save_checkpoint(self, epoch: int, best: bool = False, final: bool = False) -> None:
        """
        Save a checkpoint of the model.

        Args:
            epoch: Current epoch number
            best: Whether this is the best model so far
            final: Whether this is the final model
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "model_config": self.model.get_config()
        }

        # Add scheduler state if available
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save checkpoint
        if best:
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
        elif final:
            checkpoint_path = self.checkpoint_dir / "final_model.pt"
        else:
            checkpoint_path = self.checkpoint_dir / \
                f"checkpoint_epoch_{epoch}.pt"

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Load a checkpoint of the model.

        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state if available
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load history
        self.history = checkpoint["history"]

        self.logger.info(f"Loaded checkpoint from {checkpoint_path} "
                         f"(epoch {checkpoint['epoch']})")

    def get_model_summary(self) -> str:
        """
        Get a summary of the model architecture.

        Returns:
            String with model summary
        """
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)

        # Get model configuration
        model_config = self.model.get_config()

        # Build summary string
        summary = [
            "Model Summary:",
            f"  Encoder: {model_config['encoder_type']}",
            f"  Decoder: {model_config['decoder_type']}",
            f"  Total parameters: {total_params:,}",
            f"  Trainable parameters: {trainable_params:,}",
            "  Configuration:",
        ]

        # Add encoder configuration
        summary.append("    Encoder:")
        for k, v in model_config["encoder_config"].items():
            summary.append(f"      {k}: {v}")

        # Add decoder configuration
        summary.append("    Decoder:")
        for k, v in model_config["decoder_config"].items():
            summary.append(f"      {k}: {v}")

        # Add training configuration
        summary.append("    Training:")
        for k, v in model_config.items():
            if k not in ["encoder_type", "decoder_type", "encoder_config", "decoder_config"]:
                summary.append(f"      {k}: {v}")

        return "\n".join(summary)

    def save_training_history(self) -> None:
        """
        Save the training history to a JSON file.
        """
        # Convert numpy values to Python types
        history_copy = {}
        for key, value in self.history.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                history_copy[key] = []
                for d in value:
                    history_copy[key].append(
                        {k: float(v) for k, v in d.items()})
            else:
                history_copy[key] = [float(v) for v in value]

        # Save history
        history_path = self.log_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history_copy, f, indent=2)

        self.logger.info(f"Saved training history to {history_path}")


def train_model(
    model: Optional[EncoderDecoderWrapper] = None,
    architecture: str = "gcn",
    input_dim: int = 3,
    hidden_dim: int = 64,
    latent_dim: int = 32,
    output_dim: int = 3,
    num_nodes: int = 10,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    dataset: Optional[Union[List[Data], torch_geometric.data.Dataset]] = None,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    num_epochs: int = 100,
    early_stopping_patience: int = 10,
    loss_type: str = "bce",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    log_dir: str = "logs",
    checkpoint_dir: str = "checkpoints",
    **model_kwargs
) -> Tuple[EncoderDecoderWrapper, Dict[str, List[float]]]:
    """
    Convenience function to train a GNN encoder-decoder model from scratch.

    Args:
        model: Optional pre-initialized model (if None, one will be created)
        architecture: GNN architecture to use ('gcn', 'gat', 'gin')
        input_dim: Dimension of input node features
        hidden_dim: Dimension of hidden layers
        latent_dim: Dimension of latent space
        output_dim: Dimension of output node features
        num_nodes: Number of nodes in the graph
        train_loader: Optional pre-initialized training data loader
        val_loader: Optional pre-initialized validation data loader
        test_loader: Optional pre-initialized test data loader
        dataset: Optional dataset to use if loaders not provided
        batch_size: Batch size for data loaders
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        num_epochs: Number of epochs to train for
        early_stopping_patience: Number of epochs to wait for validation improvement
        loss_type: Type of loss function to use
        device: Device to use for training
        log_dir: Directory to save logs
        checkpoint_dir: Directory to save checkpoints
        **model_kwargs: Additional arguments for model creation

    Returns:
        Tuple of (trained_model, training_history)
    """
    # Create model if not provided
    if model is None:
        model = create_model(
            architecture=architecture,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            num_nodes=num_nodes,
            loss_type=loss_type,
            **model_kwargs
        )

    # Create data loaders if not provided
    if train_loader is None and dataset is not None:
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset=dataset,
            batch_size=batch_size,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
    elif train_loader is None:
        raise ValueError("Either train_loader or dataset must be provided")

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir
    )

    # Log model summary
    print(trainer.get_model_summary())

    # Train model
    history = trainer.train(
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience
    )

    # Save training history
    trainer.save_training_history()

    return model, history
