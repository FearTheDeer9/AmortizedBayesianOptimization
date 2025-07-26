"""
BC Model Loading Utilities

This module provides utilities for loading BC model checkpoints and reconstructing
the Haiku-transformed functions needed for inference. Follows JAX/Haiku best practices
by storing model configuration and reconstructing architecture during loading.
"""

import logging
import pickle
import gzip
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk

from .model_registry import create_model_from_config

logger = logging.getLogger(__name__)


def load_bc_surrogate_model(
    checkpoint_path: str
) -> Tuple[Callable, Callable, Callable, Callable, Any]:
    """
    Load BC surrogate model from checkpoint.
    
    Reconstructs the Haiku-transformed model functions from saved configuration
    and applies the saved parameters.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Tuple of (init_fn, apply_fn, encoder_init, encoder_apply, params)
        as expected by BC evaluator
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    try:
        # Load checkpoint data
        logger.info(f"Loading BC surrogate checkpoint from {checkpoint_path}")
        
        # Handle both gzip and regular pickle files
        try:
            with gzip.open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
        except (gzip.BadGzipFile, OSError):
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
        
        # Extract model information
        model_type = checkpoint_data.get('model_type', 'continuous_surrogate')
        model_config = checkpoint_data.get('model_config', {})
        model_params = checkpoint_data.get('model_params')
        
        if model_params is None:
            raise ValueError("No model parameters found in checkpoint")
        
        logger.info(f"Found model type: {model_type} with config: {model_config}")
        
        # Reconstruct model from configuration
        haiku_model, _ = create_model_from_config(model_type, model_config)
        
        # For surrogate models, we need to provide the standard interface
        # with separate init/apply and encoder functions
        
        # The main model init/apply functions
        init_fn = haiku_model.init
        apply_fn = haiku_model.apply
        
        # For BC evaluator compatibility, we need encoder functions
        # These are typically the same as the main model for parent set models
        encoder_init = init_fn
        encoder_apply = apply_fn
        
        logger.info("✅ Successfully loaded BC surrogate model")
        
        return init_fn, apply_fn, encoder_init, encoder_apply, model_params
        
    except Exception as e:
        logger.error(f"Failed to load BC surrogate model: {e}")
        raise


def load_bc_acquisition_model(
    checkpoint_path: str
) -> Callable:
    """
    Load BC acquisition model from checkpoint.
    
    Reconstructs the Haiku-transformed policy function from saved configuration
    and creates a callable that applies the saved parameters.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Callable acquisition function as expected by BC evaluator
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    try:
        # Load checkpoint data
        logger.info(f"Loading BC acquisition checkpoint from {checkpoint_path}")
        
        # Handle both gzip and regular pickle files
        try:
            with gzip.open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
        except (gzip.BadGzipFile, OSError):
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
        
        # Extract model information
        model_type = checkpoint_data.get('model_type', 'enhanced_acquisition')
        model_config = checkpoint_data.get('model_config', {})
        policy_params = checkpoint_data.get('policy_params')
        
        if policy_params is None:
            raise ValueError("No policy parameters found in checkpoint")
        
        logger.info(f"Found model type: {model_type} with config: {model_config}")
        
        # Reconstruct model from configuration
        haiku_model, _ = create_model_from_config(model_type, model_config)
        
        # Create acquisition function that matches expected interface
        def acquisition_fn(state: Any, key: jax.Array) -> Dict[str, Any]:
            """
            Acquisition function for BC evaluator.
            
            Args:
                state: Acquisition state (can be dict or object)
                key: JAX random key
                
            Returns:
                Dictionary with intervention decision
            """
            # Convert state to dictionary format if needed
            if hasattr(state, '__dict__'):
                # Convert object to dict
                state_dict = {
                    'state_tensor': getattr(state, 'belief_state', jnp.zeros((5, 10))),
                    'target_variable_idx': getattr(state, 'target_idx', 0),
                    'history_tensor': getattr(state, 'history', None)
                }
            elif isinstance(state, dict):
                state_dict = state
            else:
                # Fallback - create minimal state
                state_dict = {
                    'state_tensor': jnp.zeros((5, 10)),
                    'target_variable_idx': 0,
                    'history_tensor': None
                }
            
            # Apply policy network
            policy_output = haiku_model.apply(
                policy_params,
                key,
                state_dict,
                False  # is_training=False for inference
            )
            
            # Extract intervention decision
            variable_logits = policy_output.get('variable_logits', jnp.zeros(5))
            value_params = policy_output.get('value_params', jnp.zeros((5, 2)))
            
            # Sample intervention variable
            var_key, val_key = random.split(key)
            intervention_var = random.categorical(var_key, variable_logits)
            
            # Sample intervention value from Gaussian
            mean, log_std = value_params[intervention_var]
            std = jnp.exp(log_std)
            intervention_val = mean + std * random.normal(val_key)
            
            # Return in expected format
            return {
                'intervention_variables': frozenset([int(intervention_var)]),
                'intervention_values': (float(intervention_val),),
                'confidence': float(jnp.max(jax.nn.softmax(variable_logits)))
            }
        
        logger.info("✅ Successfully loaded BC acquisition model")
        
        return acquisition_fn
        
    except Exception as e:
        logger.error(f"Failed to load BC acquisition model: {e}")
        raise


def load_bc_model(
    checkpoint_path: str,
    model_type: str
) -> Any:
    """
    Generic BC model loader that dispatches to appropriate loader.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_type: Type of model ('surrogate' or 'acquisition')
        
    Returns:
        Loaded model in appropriate format
    """
    if model_type == 'surrogate':
        return load_bc_surrogate_model(checkpoint_path)
    elif model_type == 'acquisition':
        return load_bc_acquisition_model(checkpoint_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def validate_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Validate checkpoint file and return metadata.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        return {
            'valid': False,
            'error': 'File not found'
        }
    
    try:
        # Try to load checkpoint
        try:
            with gzip.open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
        except (gzip.BadGzipFile, OSError):
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
        
        # Extract metadata
        metadata = {
            'valid': True,
            'model_type': checkpoint_data.get('model_type', 'unknown'),
            'model_config': checkpoint_data.get('model_config', {}),
            'epoch': checkpoint_data.get('epoch', 0),
            'has_params': 'model_params' in checkpoint_data or 'policy_params' in checkpoint_data,
            'checkpoint_keys': list(checkpoint_data.keys())
        }
        
        return metadata
        
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }


__all__ = [
    'load_bc_surrogate_model',
    'load_bc_acquisition_model',
    'load_bc_model',
    'validate_checkpoint'
]