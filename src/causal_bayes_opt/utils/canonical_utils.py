"""
Canonical utilities for ACBO workflow.

This module provides the canonical patterns for checkpoint management,
model loading, and configuration that are used throughout the ACBO pipeline.
These patterns have been extracted from the working train_acbo_methods.py
and evaluate_acbo_methods.py scripts.

Key principles:
1. Consistent checkpoint format with metadata
2. Simple pickle-based serialization
3. Clear separation between training and inference
4. Minimal dependencies
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
import jax
import jax.numpy as jnp
import haiku as hk

logger = logging.getLogger(__name__)


# ============================================================================
# Checkpoint Management
# ============================================================================

def save_checkpoint(
    checkpoint_path: Path,
    params: Any,
    config: Dict[str, Any],
    metadata: Dict[str, Any],
    checkpoint_type: str = "checkpoint.pkl"
) -> None:
    """
    Canonical checkpoint saving pattern.
    
    This is the standard way to save model checkpoints in ACBO.
    
    Args:
        checkpoint_path: Directory to save checkpoint
        params: Model parameters (JAX pytree)
        config: Configuration used for training
        metadata: Additional metadata (trainer type, metrics, etc.)
        checkpoint_type: Filename for checkpoint (default: checkpoint.pkl)
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'params': params,
        'config': config,
        'metadata': metadata
    }
    
    # For specific model types, add type-specific fields
    if metadata.get('model_type') == 'grpo':
        checkpoint['policy_params'] = params
        checkpoint['has_surrogate'] = metadata.get('has_surrogate', False)
        if checkpoint['has_surrogate']:
            checkpoint['surrogate_params'] = metadata.get('surrogate_params')
    
    with open(checkpoint_path / checkpoint_type, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Canonical checkpoint loading pattern.
    
    Args:
        checkpoint_path: Path to checkpoint file or directory
        
    Returns:
        Checkpoint dictionary with params, config, and metadata
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Handle both file and directory paths
    if checkpoint_path.is_file():
        checkpoint_file = checkpoint_path
    else:
        checkpoint_file = checkpoint_path / 'checkpoint.pkl'
    
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)
    
    return checkpoint


# ============================================================================
# Model Creation Patterns
# ============================================================================

def create_model_from_checkpoint(
    checkpoint_path: Path,
    model_factory: Callable,
    seed: int = 42
) -> Tuple[Callable, Dict[str, Any]]:
    """
    Canonical pattern for creating a model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model_factory: Function that creates the model (e.g., create_clean_grpo_policy)
        seed: Random seed
        
    Returns:
        Tuple of (model_fn, checkpoint_data)
    """
    checkpoint = load_checkpoint(checkpoint_path)
    params = checkpoint.get('params') or checkpoint.get('policy_params')
    config = checkpoint.get('config', {})
    
    # Create model function
    model_fn = model_factory(**config)
    
    # Transform with Haiku if needed
    if not hasattr(model_fn, 'apply'):
        model_fn = hk.transform(model_fn)
    
    # Create inference function
    def inference_fn(*args, **kwargs):
        key = jax.random.PRNGKey(seed)
        return model_fn.apply(params, key, *args, **kwargs)
    
    return inference_fn, checkpoint


# ============================================================================
# Configuration Patterns
# ============================================================================

def create_training_config(
    method: str,
    episodes: int,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    **kwargs
) -> Dict[str, Any]:
    """
    Canonical configuration creation pattern.
    
    Args:
        method: Training method ('grpo', 'bc')
        episodes: Number of training episodes
        batch_size: Batch size
        learning_rate: Learning rate
        **kwargs: Additional method-specific arguments
        
    Returns:
        Configuration dictionary
    """
    config = {
        'method': method,
        'episodes': episodes,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'seed': kwargs.get('seed', 42)
    }
    
    # Method-specific defaults
    if method == 'grpo':
        config.update({
            'entropy_coeff': kwargs.get('entropy_coeff', 0.1),
            'group_size': kwargs.get('group_size', batch_size),
            'use_surrogate': kwargs.get('use_surrogate', False),
            'convergence_window': kwargs.get('convergence_window', 50),
            'patience': kwargs.get('patience', 10)
        })
    elif method == 'bc':
        config.update({
            'model_type': 'policy',  # Note: train_acbo_methods.py only supports policy training for BC
            'hidden_dim': kwargs.get('hidden_dim', 256),
            'max_epochs': kwargs.get('max_epochs', episodes),
            'early_stopping_patience': kwargs.get('early_stopping_patience', 10),
            'demo_path': kwargs.get('demo_path'),
            'max_demos': kwargs.get('max_demos')
        })
    
    # Add any additional kwargs
    config.update(kwargs)
    
    return config


def create_evaluation_config(
    n_initial_obs: int = 100,
    max_interventions: int = 20,
    n_intervention_samples: int = 10,
    optimization_direction: str = 'MINIMIZE',
    **kwargs
) -> Dict[str, Any]:
    """
    Canonical evaluation configuration pattern.
    
    Args:
        n_initial_obs: Number of initial observations
        max_interventions: Maximum number of interventions
        n_intervention_samples: Samples per intervention
        optimization_direction: 'MINIMIZE' or 'MAXIMIZE'
        **kwargs: Additional arguments
        
    Returns:
        Evaluation configuration dictionary
    """
    return {
        'n_initial_obs': n_initial_obs,
        'max_interventions': max_interventions,
        'n_intervention_samples': n_intervention_samples,
        'optimization_direction': optimization_direction,
        'seed': kwargs.get('seed', 42),
        **kwargs
    }


# ============================================================================
# Training Result Patterns  
# ============================================================================

def format_training_results(
    params: Any,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    trainer_type: str,
    model_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Canonical pattern for formatting training results.
    
    Args:
        params: Trained model parameters
        config: Training configuration
        metrics: Training metrics
        trainer_type: Type of trainer used
        model_type: Optional model type (e.g., 'policy', 'surrogate')
        
    Returns:
        Formatted results dictionary
    """
    results = {
        "params": params,
        "config": config,
        "metrics": metrics,
        "metadata": {
            "trainer_type": trainer_type,
            "model_type": model_type or config.get('model_type', 'unknown')
        }
    }
    
    # Add any additional fields based on trainer type
    if trainer_type == "UnifiedGRPOTrainer":
        results["policy_params"] = params
        results["has_surrogate"] = config.get('use_surrogate', False)
    
    return results


# ============================================================================
# Common Patterns
# ============================================================================

def ensure_path(path: str | Path) -> Path:
    """Ensure path is a Path object."""
    return Path(path) if isinstance(path, str) else path


def setup_logging(level: str = "INFO") -> None:
    """Setup logging with canonical format."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# ============================================================================
# Model Interface Patterns
# ============================================================================

def create_acquisition_function(
    model_fn: Callable,
    params: Any,
    seed: int = 42
) -> Callable:
    """
    Create a simple acquisition function from model and params.
    
    This is the canonical pattern for converting trained models into
    acquisition functions for evaluation.
    
    Args:
        model_fn: Model function (Haiku transformed)
        params: Model parameters
        seed: Random seed
        
    Returns:
        Acquisition function: (tensor, posterior, target) -> intervention
    """
    def acquisition_fn(tensor, posterior, target):
        key = jax.random.PRNGKey(seed)
        
        # Get variable names from tensor
        n_vars = tensor.shape[1]
        if n_vars == 3 and target in ['X', 'Y', 'Z']:
            variables = ['X', 'Y', 'Z']
        else:
            variables = [f'X{i}' for i in range(n_vars)]
        
        # Get target index
        target_idx = variables.index(target) if target in variables else 0
        
        # Apply model
        outputs = model_fn.apply(params, key, tensor, target_idx)
        
        # Extract intervention
        if 'variable_logits' in outputs:
            # Policy model output
            var_idx = int(jnp.argmax(outputs['variable_logits']))
            value = float(outputs['value_params'][var_idx, 0])  # mean
        else:
            # Other model types - adapt as needed
            var_idx = 0
            value = 0.0
        
        intervention_var = variables[var_idx]
        
        # Create intervention
        from ..interventions.handlers import create_perfect_intervention
        return create_perfect_intervention([(intervention_var, value)])
    
    return acquisition_fn