"""
Unified checkpoint management for all ACBO models.

This module provides standardized checkpoint saving and loading functionality
for both policy and surrogate models, ensuring consistent format and easy
model reconstruction.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from datetime import datetime

import jax
import jax.numpy as jnp
import haiku as hk

logger = logging.getLogger(__name__)

# Checkpoint format version
CHECKPOINT_VERSION = "2.0"


def save_checkpoint(
    path: Path,
    params: Dict[str, Any],
    architecture: Dict[str, Any],
    model_type: str,
    model_subtype: str,
    training_config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model checkpoint in standardized format.
    
    Args:
        path: Path to save checkpoint (can be file or directory)
        params: Model parameters
        architecture: Model architecture configuration
        model_type: 'policy' or 'surrogate'
        model_subtype: Specific model type ('grpo', 'bc', 'continuous_parent_set')
        training_config: Optional training configuration
        metadata: Optional metadata about training
        metrics: Optional training metrics
    """
    # Validate inputs
    if model_type not in ['policy', 'surrogate']:
        raise ValueError(f"Invalid model_type: {model_type}")
    
    if model_subtype not in ['grpo', 'bc', 'continuous_parent_set']:
        raise ValueError(f"Invalid model_subtype: {model_subtype}")
    
    # Ensure path
    path = Path(path)
    if path.suffix != '.pkl':
        # If directory, create checkpoint.pkl inside
        path.mkdir(parents=True, exist_ok=True)
        checkpoint_file = path / 'checkpoint.pkl'
    else:
        # If file, ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_file = path
    
    # Build checkpoint
    checkpoint = {
        'version': CHECKPOINT_VERSION,
        'model_type': model_type,
        'model_subtype': model_subtype,
        'params': params,
        'architecture': architecture,
        'training_config': training_config or {},
        'metadata': metadata or {},
        'metrics': metrics or {},
        'timestamp': datetime.now().isoformat()
    }
    
    # Save
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    logger.info(f"Saved {model_type}/{model_subtype} checkpoint to {checkpoint_file}")
    logger.info(f"  Architecture: {architecture}")


def load_checkpoint(path: Path) -> Dict[str, Any]:
    """
    Load and validate checkpoint.
    
    Args:
        path: Path to checkpoint file or directory
        
    Returns:
        Checkpoint dictionary
        
    Raises:
        ValueError: If checkpoint is invalid or wrong version
    """
    path = Path(path)
    
    # Find checkpoint file
    if path.is_dir():
        checkpoint_file = path / 'checkpoint.pkl'
    else:
        checkpoint_file = path
    
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    
    # Load checkpoint
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Validate version
    version = checkpoint.get('version', '1.0')
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Checkpoint version mismatch: expected {CHECKPOINT_VERSION}, "
            f"got {version}. Please retrain the model."
        )
    
    # Validate required fields
    required_fields = ['model_type', 'model_subtype', 'params', 'architecture']
    for field in required_fields:
        if field not in checkpoint:
            raise ValueError(f"Invalid checkpoint: missing '{field}' field")
    
    logger.info(f"Loaded {checkpoint['model_type']}/{checkpoint['model_subtype']} checkpoint")
    logger.info(f"  Architecture: {checkpoint['architecture']}")
    
    return checkpoint


def create_model_from_checkpoint(checkpoint: Dict[str, Any]) -> Tuple[hk.Transformed, Dict[str, Any]]:
    """
    Recreate model network from checkpoint.
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        
    Returns:
        Tuple of (transformed_model, params)
    """
    model_type = checkpoint['model_type']
    model_subtype = checkpoint['model_subtype']
    architecture = checkpoint['architecture']
    params = checkpoint['params']
    
    # Policy models
    if model_type == 'policy':
        hidden_dim = architecture['hidden_dim']
        
        if model_subtype == 'bc':
            from ..policies.clean_bc_policy_factory import create_clean_bc_policy
            policy_fn = create_clean_bc_policy(hidden_dim=hidden_dim)
            net = hk.transform(policy_fn)
            
        elif model_subtype == 'grpo':
            from ..policies.clean_policy_factory import create_clean_grpo_policy
            policy_fn = create_clean_grpo_policy(hidden_dim=hidden_dim)
            net = hk.transform(policy_fn)
            
        else:
            raise ValueError(f"Unknown policy subtype: {model_subtype}")
    
    # Surrogate models
    elif model_type == 'surrogate':
        if model_subtype == 'continuous_parent_set':
            from ..avici_integration.continuous.model import ContinuousParentSetPredictionModel
            
            # Recreate model with exact architecture
            def surrogate_fn(data: jnp.ndarray, target_variable: int, is_training: bool = False):
                model = ContinuousParentSetPredictionModel(
                    hidden_dim=architecture['hidden_dim'],
                    num_layers=architecture['num_layers'],
                    num_heads=architecture['num_heads'],
                    key_size=architecture['key_size'],  # Explicit!
                    dropout=architecture.get('dropout', 0.1) if is_training else 0.0
                )
                return model(data, target_variable, is_training)
            
            net = hk.transform(surrogate_fn)
            
        else:
            raise ValueError(f"Unknown surrogate subtype: {model_subtype}")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return net, params


def load_policy_model(path: Path, seed: int = 42) -> Tuple[Any, Dict[str, Any]]:
    """
    Load policy model and create acquisition function.
    
    Args:
        path: Path to checkpoint
        seed: Random seed for stochastic policies
        
    Returns:
        Tuple of (acquisition_function, checkpoint_info)
    """
    checkpoint = load_checkpoint(path)
    net, params = create_model_from_checkpoint(checkpoint)
    
    # Return checkpoint info for metadata
    info = {
        'architecture': checkpoint['architecture'],
        'model_subtype': checkpoint['model_subtype'],
        'metrics': checkpoint.get('metrics', {})
    }
    
    return (net, params), info


def load_surrogate_model(path: Path) -> Tuple[Any, Dict[str, Any]]:
    """
    Load surrogate model for structure learning.
    
    Args:
        path: Path to checkpoint
        
    Returns:
        Tuple of (predict_function, checkpoint_info)
    """
    checkpoint = load_checkpoint(path)
    net, params = create_model_from_checkpoint(checkpoint)
    
    # Create predict function wrapper
    from ..training.continuous_surrogate_integration import create_surrogate_fn_wrapper
    predict_fn = create_surrogate_fn_wrapper(net, params)
    
    # Return checkpoint info
    info = {
        'architecture': checkpoint['architecture'],
        'metrics': checkpoint.get('metrics', {})
    }
    
    return predict_fn, info