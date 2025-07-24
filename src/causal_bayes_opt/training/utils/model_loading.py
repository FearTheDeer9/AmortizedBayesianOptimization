#!/usr/bin/env python3
"""
Generic Model Loading Utilities for Behavioral Cloning

This module provides utilities for loading BC model checkpoints and wrapping them
for integration with the ACBO comparison framework.

Key Features:
1. Generic checkpoint loading for any BC model type
2. ACBO-compatible model wrappers
3. Model validation and compatibility checking
4. Error handling and metadata extraction

Design Principles (Rich Hickey Approved):
- Pure functions for loading and conversion
- Immutable data structures
- Clear separation of concerns
- Composable model interfaces
"""

import logging
import pickle
import gzip
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import time

import jax.numpy as jnp
import pyrsistent as pyr

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelMetadata:
    """Immutable metadata for loaded models."""
    model_type: str
    checkpoint_path: str
    file_size_kb: float
    creation_time: float
    training_epochs: Optional[int] = None
    final_loss: Optional[float] = None
    final_accuracy: Optional[float] = None


@dataclass(frozen=True)
class LoadedModel:
    """Immutable container for loaded model data."""
    model_type: str
    model_params: Any
    training_state: Any
    config: Any
    metadata: ModelMetadata
    success: bool = True
    error_message: Optional[str] = None


def load_checkpoint_model(checkpoint_path: Union[str, Path], model_type: str) -> LoadedModel:
    """
    Generic checkpoint loader for any BC model type.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_type: Type of model ('surrogate' or 'acquisition')
        
    Returns:
        LoadedModel containing model data and metadata
    """
    checkpoint_path = Path(checkpoint_path)
    
    try:
        if not checkpoint_path.exists():
            return LoadedModel(
                model_type=model_type,
                model_params=None,
                training_state=None,
                config=None,
                metadata=ModelMetadata(
                    model_type=model_type,
                    checkpoint_path=str(checkpoint_path),
                    file_size_kb=0.0,
                    creation_time=0.0
                ),
                success=False,
                error_message=f"Checkpoint not found: {checkpoint_path}"
            )
        
        # Load checkpoint data (handle both gzip and regular pickle files)
        try:
            # First try as gzip
            with gzip.open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
        except (gzip.BadGzipFile, OSError):
            # If not gzip, try regular pickle
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
        
        # Extract model parameters based on type
        if model_type == 'surrogate':
            model_params = checkpoint_data.get('model_params')
            training_state = checkpoint_data.get('training_state')
        elif model_type == 'acquisition':
            model_params = checkpoint_data.get('policy_params')
            training_state = checkpoint_data.get('training_state')
        else:
            return LoadedModel(
                model_type=model_type,
                model_params=None,
                training_state=None,
                config=None,
                metadata=ModelMetadata(
                    model_type=model_type,
                    checkpoint_path=str(checkpoint_path),
                    file_size_kb=0.0,
                    creation_time=0.0
                ),
                success=False,
                error_message=f"Unknown model type: {model_type}"
            )
        
        # Extract training metrics
        final_loss = None
        final_accuracy = None
        training_epochs = None
        
        if training_state:
            if hasattr(training_state, 'epoch'):
                training_epochs = training_state.epoch
            elif isinstance(training_state, dict):
                training_epochs = training_state.get('epoch')
                final_loss = training_state.get('loss')
                final_accuracy = training_state.get('accuracy')
        
        # Create metadata
        file_stats = checkpoint_path.stat()
        metadata = ModelMetadata(
            model_type=model_type,
            checkpoint_path=str(checkpoint_path),
            file_size_kb=file_stats.st_size / 1024,
            creation_time=file_stats.st_mtime,
            training_epochs=training_epochs,
            final_loss=final_loss,
            final_accuracy=final_accuracy
        )
        
        return LoadedModel(
            model_type=model_type,
            model_params=model_params,
            training_state=training_state,
            config=checkpoint_data.get('config'),
            metadata=metadata,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
        return LoadedModel(
            model_type=model_type,
            model_params=None,
            training_state=None,
            config=None,
            metadata=ModelMetadata(
                model_type=model_type,
                checkpoint_path=str(checkpoint_path),
                file_size_kb=0.0,
                creation_time=0.0
            ),
            success=False,
            error_message=str(e)
        )


# Note: SurrogateModelWrapper and AcquisitionPolicyWrapper have been removed
# Use bc_model_inference.py for real model inference instead of mock implementations


# wrap_for_acbo has been removed - use bc_model_inference.py functions instead


# create_acbo_optimization_runner has been removed - use bc_method_wrappers.py instead


def find_latest_checkpoint(checkpoint_dir: Union[str, Path], model_type: str) -> Optional[Path]:
    """
    Find the latest checkpoint file in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        model_type: Type of model to look for
        
    Returns:
        Path to latest checkpoint or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    # Look for checkpoint files
    patterns = [
        f"{model_type}_bc_*.pkl",
        f"{model_type}_*.pkl",
        "*.pkl"
    ]
    
    checkpoint_files = []
    for pattern in patterns:
        checkpoint_files.extend(checkpoint_dir.glob(pattern))
    
    if not checkpoint_files:
        return None
    
    # Return the most recently modified file
    return max(checkpoint_files, key=lambda p: p.stat().st_mtime)


# validate_model_compatibility has been removed - use bc_model_inference.py functions instead