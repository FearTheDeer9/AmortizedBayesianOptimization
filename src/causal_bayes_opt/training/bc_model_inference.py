#!/usr/bin/env python3
"""
BC Model Inference Utilities

Provides inference functions for BC-trained surrogate and acquisition models.
Handles checkpoint loading, model initialization, and proper data conversions.

Key Features:
1. Load BC checkpoints and extract trained parameters
2. Create inference functions using Haiku transformations
3. Handle data format conversions for CBO integration
4. Provide clean interfaces for both model types

Design Principles:
- Pure functions for inference
- Explicit data format handling
- Clear separation between training and inference
- JAX-compatible implementations
"""

import logging
from typing import Dict, Any, Callable, Tuple, List, Optional, Union, FrozenSet
from pathlib import Path
import pickle
import gzip

import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
import numpy as onp
import pyrsistent as pyr

from ..avici_integration.continuous.model import ContinuousParentSetPredictionModel
from ..avici_integration.parent_set.posterior import create_parent_set_posterior
from ..acquisition.enhanced_policy_network import EnhancedPolicyNetwork
from ..acquisition.state import AcquisitionState
from .acquisition_state_converter import convert_acquisition_state_to_tensors

logger = logging.getLogger(__name__)


def load_bc_checkpoint(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load BC checkpoint file and extract model data.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary containing checkpoint data
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint (handle both gzip and regular pickle)
    try:
        with gzip.open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
    except (gzip.BadGzipFile, OSError):
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
    
    return checkpoint_data


def create_bc_surrogate_inference_fn(
    checkpoint_path: str,
    threshold: float = 0.1
) -> Callable[[jnp.ndarray, List[str], str], Any]:
    """
    Create inference function for BC-trained surrogate model.
    
    Args:
        checkpoint_path: Path to BC surrogate checkpoint
        threshold: Probability threshold for including parent sets
        
    Returns:
        Function that takes AVICI data and returns parent set posterior
    """
    # Load checkpoint
    checkpoint_data = load_bc_checkpoint(checkpoint_path)
    model_params = checkpoint_data.get('model_params')
    
    if not model_params:
        raise ValueError(f"No model_params found in checkpoint: {checkpoint_path}")
    
    # Extract model config from checkpoint or use defaults
    config = checkpoint_data.get('config', {})
    
    # Handle BCTrainingConfig object
    if hasattr(config, 'surrogate_config'):
        # It's a BCTrainingConfig object
        surrogate_config = config.surrogate_config
        hidden_dim = getattr(surrogate_config, 'hidden_dim', 64)
        num_layers = getattr(surrogate_config, 'num_layers', 3)
        num_heads = getattr(surrogate_config, 'num_heads', 4)
        key_size = getattr(surrogate_config, 'key_size', 32)
    else:
        # Fallback to defaults
        hidden_dim = 64
        num_layers = 3
        num_heads = 4
        key_size = 32
    
    logger.info(f"Loaded BC surrogate with config: hidden_dim={hidden_dim}, "
                f"num_layers={num_layers}, num_heads={num_heads}")
    
    # Create Haiku-transformed model
    def model_fn(data: jnp.ndarray, target_idx: int) -> Dict[str, jnp.ndarray]:
        """Apply continuous parent set model."""
        model = ContinuousParentSetPredictionModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            key_size=key_size,
            dropout=0.0  # No dropout for inference
        )
        return model(data, target_idx, is_training=False)
    
    # Transform model for JAX
    model = hk.without_apply_rng(hk.transform(model_fn))
    
    def surrogate_inference_fn(
        avici_data: jnp.ndarray,  # [N, d, 3]
        variables: List[str],
        target: str,
        params: Optional[Any] = None  # Ignored, we use checkpoint params
    ) -> Any:
        """
        Run BC surrogate inference to predict parent set posterior.
        
        Args:
            avici_data: Intervention data in AVICI format [N, d, 3]
            variables: List of variable names
            target: Target variable name
            params: Ignored (for interface compatibility)
            
        Returns:
            Parent set posterior object
        """
        # Get target variable index
        try:
            target_idx = variables.index(target)
        except ValueError:
            raise ValueError(f"Target {target} not found in variables: {variables}")
        
        # Apply model (model_fn expects 2 args: data, target_idx)
        output = model.apply(model_params, avici_data, target_idx)
        
        # Extract parent probabilities
        parent_probs = output['parent_probabilities']  # [d]
        
        # Convert continuous probabilities to discrete parent sets
        # Strategy: Include all parent sets above threshold probability
        parent_sets = []
        probabilities = []
        
        # Add empty parent set
        empty_prob = jnp.prod(1.0 - parent_probs)  # Probability no variables are parents
        if empty_prob > threshold:
            parent_sets.append(frozenset())
            probabilities.append(float(empty_prob))
        
        # Add single parent sets
        for i, var in enumerate(variables):
            if i != target_idx and parent_probs[i] > threshold:
                parent_sets.append(frozenset([var]))
                probabilities.append(float(parent_probs[i]))
        
        # Normalize probabilities
        if probabilities:
            total_prob = sum(probabilities)
            probabilities = [p / total_prob for p in probabilities]
        else:
            # Fallback: uniform over empty set and first non-target variable
            non_target_vars = [v for v in variables if v != target]
            parent_sets = [frozenset(), frozenset([non_target_vars[0]]) if non_target_vars else frozenset()]
            probabilities = [0.5, 0.5] if len(parent_sets) == 2 else [1.0]
        
        # Create posterior object
        return create_parent_set_posterior(
            target_variable=target,
            parent_sets=parent_sets,
            probabilities=jnp.array(probabilities),
            metadata={
                'type': 'bc_surrogate',
                'checkpoint': str(checkpoint_path),
                'continuous_probs': parent_probs.tolist()
            }
        )
    
    return surrogate_inference_fn


def create_bc_acquisition_inference_fn(
    checkpoint_path: str,
    variables: List[str],
    target_variable: str
) -> Callable[[jax.Array], Dict[str, Any]]:
    """
    Create inference function for BC-trained acquisition policy.
    
    Args:
        checkpoint_path: Path to BC acquisition checkpoint
        variables: List of variable names in SCM
        target_variable: Name of target variable
        
    Returns:
        Function that takes random key and returns intervention decision
    """
    # Load checkpoint
    checkpoint_data = load_bc_checkpoint(checkpoint_path)
    policy_params = checkpoint_data.get('policy_params')
    
    if not policy_params:
        raise ValueError(f"No policy_params found in checkpoint: {checkpoint_path}")
    
    # Extract policy config from checkpoint
    training_state = checkpoint_data.get('training_state', {})
    model_config = checkpoint_data.get('model_config', {})
    n_vars = len(variables)
    target_idx = variables.index(target_variable)
    
    # Use config from checkpoint or defaults
    hidden_dim = model_config.get('hidden_dim', 128)
    num_layers = model_config.get('num_layers', 3)
    num_heads = model_config.get('num_heads', 4)
    key_size = model_config.get('key_size', 32)
    
    logger.info(f"Loaded BC acquisition policy for {n_vars} variables, target={target_variable}")
    logger.info(f"Using model config: hidden_dim={hidden_dim}, num_layers={num_layers}, num_heads={num_heads}")
    
    # Create Haiku-transformed policy
    def policy_fn(state_tensor: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Apply enhanced policy network."""
        # Use architecture from checkpoint config
        policy = EnhancedPolicyNetwork(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            key_size=key_size,
            num_variables=n_vars,
            intervention_dim=64,
            dropout=0.0,  # No dropout for inference
            name="EnhancedPolicyNetwork"
        )
        
        return policy(
            state_tensor=state_tensor,
            target_variable_idx=target_idx,
            history_tensor=None,  # No history for now
            is_training=False
        )
    
    # Transform policy for JAX (keep RNG for dropout/attention)
    policy = hk.transform(policy_fn)
    
    def acquisition_inference_fn(key: jax.Array) -> Dict[str, Any]:
        """
        Run BC acquisition inference to select intervention.
        
        Note: This is a simplified version that doesn't use the full
        acquisition state. In production, would need to:
        1. Get current acquisition state from CBO environment
        2. Convert to tensor representation
        3. Apply policy network
        
        Args:
            key: JAX random key
            
        Returns:
            Intervention decision dict
        """
        # Create enriched state tensor with 5 channels
        num_channels = 5
        state_tensor = jnp.zeros((n_vars, num_channels))
        
        # Channel 0: Variable values (simulated)
        noise_key, value_key = random.split(key)
        state_tensor = state_tensor.at[:, 0].set(random.normal(noise_key, (n_vars,)) * 0.5)
        
        # Channel 1: Intervention indicators (all zeros for initial state)
        state_tensor = state_tensor.at[:, 1].set(0.0)
        
        # Channel 2: Target indicators
        state_tensor = state_tensor.at[target_idx, 2].set(1.0)
        
        # Channel 3: Marginal parent probabilities (simulated)
        state_tensor = state_tensor.at[:, 3].set(0.3)
        
        # Channel 4: Intervention recency (all zeros for initial state)
        state_tensor = state_tensor.at[:, 4].set(0.0)
        
        # Apply policy with random key
        policy_output = policy.apply(policy_params, value_key, state_tensor)
        
        # Extract intervention decision
        variable_logits = policy_output['variable_logits']  # [n_vars]
        value_params = policy_output['value_params']  # [n_vars, 2] (mean, std)
        
        # Mask target variable
        masked_logits = variable_logits.at[target_idx].set(-jnp.inf)
        
        # Sample intervention variable
        var_key, val_key = random.split(value_key)
        selected_idx = random.categorical(var_key, masked_logits)
        selected_var = variables[int(selected_idx)]
        
        # Sample intervention value from predicted distribution
        mean = value_params[selected_idx, 0]
        std = jnp.abs(value_params[selected_idx, 1]) + 0.1  # Ensure positive std
        value = random.normal(val_key) * std + mean
        
        # Clip to reasonable range
        value = jnp.clip(value, -2.0, 2.0)
        
        # Return in format expected by full_acquisition_fn
        return {
            'intervention_variables': frozenset([selected_var]),
            'intervention_values': (float(value),)
        }
    
    return acquisition_inference_fn


def create_full_bc_acquisition_fn(
    checkpoint_path: str,
    variables: List[str],
    target_variable: str,
    value_range: Tuple[float, float] = (-2.0, 2.0)
) -> Callable[[AcquisitionState, jax.Array], Dict[str, Any]]:
    """
    Create full BC acquisition function that uses actual acquisition state.
    
    This is the production version that properly converts CBO state.
    
    Args:
        checkpoint_path: Path to BC acquisition checkpoint
        variables: List of variable names
        target_variable: Target variable name
        value_range: Range for intervention values
        
    Returns:
        Function that takes acquisition state and returns intervention
    """
    # Load checkpoint and create base inference function
    checkpoint_data = load_bc_checkpoint(checkpoint_path)
    policy_params = checkpoint_data.get('policy_params')
    
    if not policy_params:
        raise ValueError(f"No policy_params found in checkpoint: {checkpoint_path}")
    
    n_vars = len(variables)
    target_idx = variables.index(target_variable)
    
    # Extract actual config from checkpoint
    model_config = checkpoint_data.get('model_config', {})
    
    # Create policy network
    def policy_fn(state_dict: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """Apply policy to state dictionary."""
        # Extract state tensor from dict
        state_tensor = state_dict['state_tensor']  # [n_vars, num_channels]
        history_tensor = state_dict.get('history_tensor', None)  # [T, n_vars, num_channels]
        
        # Use the exact same configuration as training
        policy = EnhancedPolicyNetwork(
            hidden_dim=model_config.get('hidden_dim', 128),
            num_layers=model_config.get('num_layers', 4),
            num_heads=model_config.get('num_heads', 8),
            key_size=model_config.get('key_size', 32),
            num_variables=model_config.get('num_variables', n_vars),
            intervention_dim=64,
            dropout=0.0,  # No dropout for inference
            name="EnhancedPolicyNetwork"
        )
        
        # The EnhancedPolicyNetwork will concatenate state_tensor with history_tensor
        # so they must have the same number of channels
        # Since history_tensor has 5 channels, state_tensor must also have 5 channels
        
        # Use the enriched state directly since it already has 5 channels
        return policy(
            state_tensor=state_tensor,  # This has shape [n_vars, 5]
            target_variable_idx=target_idx,
            history_tensor=history_tensor,  # This has shape [T, n_vars, 5]
            is_training=False
        )
    
    # Transform for JAX - keep RNG for dropout
    policy = hk.transform(policy_fn)
    
    def full_acquisition_fn(
        acquisition_state: AcquisitionState,
        key: jax.Array
    ) -> Dict[str, Any]:
        """
        Select intervention based on acquisition state.
        
        Args:
            acquisition_state: Current CBO acquisition state
            key: JAX random key
            
        Returns:
            Intervention decision
        """
        # Use enriched history builder directly
        from ..acquisition.enriched.state_enrichment import EnrichedHistoryBuilder
        
        # Create enriched history builder
        history_builder = EnrichedHistoryBuilder(
            standardize_values=True,
            include_temporal_features=True,
            max_history_size=100,
            support_variable_scms=True,
            num_channels=5  # Use 5 enriched channels
        )
        
        # Build enriched history from acquisition state
        enriched_history, variable_mask = history_builder.build_enriched_history(acquisition_state)
        
        # Extract current state (last timestep)
        enriched_state = enriched_history[-1]  # [n_vars, 5]
        
        # Use last 3 timesteps for history (to match training)
        history_tensor = enriched_history[-3:]  # [3, n_vars, 5]
        
        state_dict = {
            'state_tensor': enriched_state,  # [n_vars, 5]
            'target_variable_idx': target_idx,
            'history_tensor': history_tensor  # [3, n_vars, 5]
        }
        
        # Apply policy with random key
        policy_output = policy.apply(policy_params, key, state_dict)
        
        # Extract decision
        variable_logits = policy_output['variable_logits']
        value_params = policy_output['value_params']
        
        # Mask target
        masked_logits = variable_logits.at[target_idx].set(-jnp.inf)
        
        # Sample intervention
        var_key, val_key = random.split(key)
        selected_idx = random.categorical(var_key, masked_logits)
        selected_var = variables[int(selected_idx)]
        
        # Sample value
        mean = value_params[selected_idx, 0]
        std = jnp.abs(value_params[selected_idx, 1]) + 0.1
        value = random.normal(val_key) * std + mean
        value = jnp.clip(value, value_range[0], value_range[1])
        
        return {
            'intervention_variables': frozenset([selected_var]),
            'intervention_values': (float(value),)
        }
    
    return full_acquisition_fn