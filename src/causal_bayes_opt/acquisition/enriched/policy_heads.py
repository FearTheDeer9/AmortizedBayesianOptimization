"""
Simplified policy heads for enriched context architecture.

This module implements clean, simple policy heads that use only the enriched
transformer output without additional feature engineering, demonstrating the
power of learning context through attention rather than hand-crafted features.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SimplifiedPolicyHeads(hk.Module):
    """
    Simplified policy heads using only enriched transformer output.
    
    These heads demonstrate that with proper enriched input to the transformer,
    we don't need complex post-processing feature engineering. The transformer
    learns all necessary patterns from the enriched temporal context.
    """
    
    def __init__(self,
                 hidden_dim: int = 128,
                 intermediate_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 name: str = "SimplifiedPolicyHeads"):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim or hidden_dim // 2
        self.dropout = dropout
        # Use smaller initialization to prevent early saturation and maintain gradient flow
        self.w_init = hk.initializers.VarianceScaling(0.5, "fan_in", "uniform")
    
    def __call__(self,
                 variable_embeddings: jnp.ndarray,  # [n_vars, hidden_dim] or [batch_size, n_vars, hidden_dim]
                 target_variable_idx: int,          # Target variable index for masking
                 is_training: bool = True           # Training mode flag
                 ) -> Dict[str, jnp.ndarray]:
        """
        Compute policy outputs from enriched transformer embeddings.
        
        Args:
            variable_embeddings: Embeddings from enriched transformer 
                                Single: [n_vars, hidden_dim] or Batched: [batch_size, n_vars, hidden_dim]
            target_variable_idx: Index of target variable (for masking)
            is_training: Training mode flag
            
        Returns:
            Dictionary containing:
            - 'variable_logits': [n_vars] or [batch_size, n_vars] - Which variable to intervene on
            - 'value_params': [n_vars, 2] or [batch_size, n_vars, 2] - (mean, log_std) for intervention values  
            - 'state_value': [] or [batch_size] - State value estimate
        """
        # Handle both single and batched inputs
        if len(variable_embeddings.shape) == 2:
            # Single state: [n_vars, hidden_dim]
            n_vars, hidden_dim = variable_embeddings.shape
            is_batched = False
        else:
            # Batched state: [batch_size, n_vars, hidden_dim]
            batch_size, n_vars, hidden_dim = variable_embeddings.shape
            is_batched = True
        
        dropout_rate = self.dropout if is_training else 0.0
        
        # Variable selection head
        variable_logits = self._variable_selection_head(
            variable_embeddings, target_variable_idx, dropout_rate
        )
        
        # Value selection head
        value_params = self._value_selection_head(
            variable_embeddings, dropout_rate
        )
        
        # State value head
        state_value = self._state_value_head(
            variable_embeddings, dropout_rate
        )
        
        return {
            'variable_logits': variable_logits,
            'value_params': value_params,
            'state_value': state_value
        }
    
    def _variable_selection_head(self,
                               variable_embeddings: jnp.ndarray,  # [n_vars, hidden_dim] or [batch_size, n_vars, hidden_dim]
                               target_variable_idx: int,
                               dropout_rate: float) -> jnp.ndarray:
        """
        Simple variable selection head using only transformer embeddings.
        
        Args:
            variable_embeddings: Variable embeddings from transformer
            target_variable_idx: Target variable index (for masking)
            dropout_rate: Dropout rate
            
        Returns:
            Variable selection logits [n_vars] or [batch_size, n_vars]
        """
        # Handle both single and batched inputs
        if len(variable_embeddings.shape) == 2:
            # Single state: [n_vars, hidden_dim]
            n_vars = variable_embeddings.shape[0]
            is_batched = False
        else:
            # Batched state: [batch_size, n_vars, hidden_dim]
            batch_size, n_vars = variable_embeddings.shape[:2]
            is_batched = True
        
        # Simple MLP for variable selection
        x = variable_embeddings
        
        # First layer with dropout
        x = hk.Linear(self.intermediate_dim, w_init=self.w_init, name="var_select_linear1")(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="var_select_norm1")(x)
        x = jax.nn.relu(x)
        x = hk.dropout(hk.next_rng_key(), dropout_rate, x)
        
        # Second layer
        x = hk.Linear(self.intermediate_dim, w_init=self.w_init, name="var_select_linear2")(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="var_select_norm2")(x)
        x = jax.nn.relu(x)
        x = hk.dropout(hk.next_rng_key(), dropout_rate, x)
        
        # Output layer - use standard initialization to avoid vanishing gradients
        # Using standard variance scaling (1.0) instead of very small (0.01) to maintain gradient flow
        variable_logits = hk.Linear(1, w_init=self.w_init, name="var_select_output")(x).squeeze(-1)
        
        # Mask target variable (cannot intervene on target)
        if is_batched:
            # Batched case: mask across all batch items
            mask = jnp.arange(n_vars) == target_variable_idx  # [n_vars]
            mask = mask[None, :]  # [1, n_vars] - broadcast over batch dimension
            masked_logits = jnp.where(mask, -1e9, variable_logits)
        else:
            # Single case: original logic
            masked_logits = jnp.where(
                jnp.arange(n_vars) == target_variable_idx,
                -1e9,  # Large negative value for target variable
                variable_logits
            )
        
        return masked_logits
    
    def _value_selection_head(self,
                            variable_embeddings: jnp.ndarray,  # [n_vars, hidden_dim] or [batch_size, n_vars, hidden_dim]
                            dropout_rate: float) -> jnp.ndarray:
        """
        Simple value selection head for intervention values.
        
        Args:
            variable_embeddings: Variable embeddings from transformer
            dropout_rate: Dropout rate
            
        Returns:
            Value parameters [n_vars, 2] or [batch_size, n_vars, 2] where [..., 0] = mean, [..., 1] = log_std
        """
        # Simple MLP for value parameters
        x = variable_embeddings
        
        # First layer
        x = hk.Linear(self.intermediate_dim, w_init=self.w_init, name="val_select_linear1")(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="val_select_norm1")(x)
        x = jax.nn.relu(x)
        x = hk.dropout(hk.next_rng_key(), dropout_rate, x)
        
        # Second layer
        x = hk.Linear(self.intermediate_dim, w_init=self.w_init, name="val_select_linear2")(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="val_select_norm2")(x)
        x = jax.nn.relu(x)
        
        # Output layer for mean and log_std - use standard initialization to avoid vanishing gradients
        # Using standard variance scaling (2.0) instead of very small (0.01) to maintain gradient flow
        value_params = hk.Linear(2, w_init=self.w_init, name="val_select_output")(x)
        
        # Split into means and log_stds and apply constraints
        if len(value_params.shape) == 2:
            # Single state: [n_vars, 2]
            means = value_params[:, 0]
            log_stds = jnp.clip(value_params[:, 1], -2.0, 2.0)  # Reasonable variance range
            return jnp.stack([means, log_stds], axis=1)
        else:
            # Batched state: [batch_size, n_vars, 2]
            means = value_params[:, :, 0]
            log_stds = jnp.clip(value_params[:, :, 1], -2.0, 2.0)  # Reasonable variance range
            return jnp.stack([means, log_stds], axis=2)
    
    def _state_value_head(self,
                        variable_embeddings: jnp.ndarray,  # [n_vars, hidden_dim]
                        dropout_rate: float) -> jnp.ndarray:
        """
        Simple state value head for baseline estimation.
        
        Args:
            variable_embeddings: Variable embeddings from transformer
            dropout_rate: Dropout rate
            
        Returns:
            State value estimate []
        """
        # Global pooling to get state-level representation
        # Handle both single and batched inputs properly
        if len(variable_embeddings.shape) == 2:
            # Single state: [n_vars, hidden_dim]
            state_embedding = jnp.mean(variable_embeddings, axis=0)  # [hidden_dim]
        else:
            # Batched state: [batch_size, n_vars, hidden_dim] 
            # Average over variable dimension (axis=1, not axis=0)
            state_embedding = jnp.mean(variable_embeddings, axis=1)  # [batch_size, hidden_dim]
        
        # Simple MLP for state value
        x = state_embedding
        
        # First layer
        x = hk.Linear(self.intermediate_dim, w_init=self.w_init, name="state_val_linear1")(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="state_val_norm1")(x)
        x = jax.nn.relu(x)
        x = hk.dropout(hk.next_rng_key(), dropout_rate, x)
        
        # Second layer
        x = hk.Linear(self.intermediate_dim // 2, w_init=self.w_init, name="state_val_linear2")(x)
        x = jax.nn.relu(x)
        
        # Output layer - use standard initialization to avoid vanishing gradients
        # Using standard variance scaling (2.0) instead of very small (0.01) to maintain gradient flow
        state_value = hk.Linear(1, w_init=self.w_init, name="state_val_output")(x).squeeze(-1)
        
        return state_value


class EnrichedAcquisitionPolicyNetwork(hk.Module):
    """
    Complete enriched acquisition policy network.
    
    Combines enriched transformer encoder with simplified policy heads
    to demonstrate the power of learning through enriched attention
    rather than hand-crafted feature engineering.
    """
    
    def __init__(self,
                 # Transformer parameters
                 num_layers: int = 4,
                 num_heads: int = 8,
                 hidden_dim: int = 128,
                 key_size: int = 32,
                 widening_factor: int = 4,
                 dropout: float = 0.1,
                 # Policy head parameters
                 policy_intermediate_dim: Optional[int] = None,
                 name: str = "EnrichedAcquisitionPolicyNetwork"):
        super().__init__(name=name)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.key_size = key_size
        self.widening_factor = widening_factor
        self.dropout = dropout
        self.policy_intermediate_dim = policy_intermediate_dim or hidden_dim // 2
    
    def __call__(self,
                 enriched_history: jnp.ndarray,  # [max_history_size, n_vars, num_channels]
                 target_variable_idx: int,       # Target variable index
                 is_training: bool = True        # Training mode flag
                 ) -> Dict[str, jnp.ndarray]:
        """
        Complete forward pass for enriched acquisition policy.
        
        Args:
            enriched_history: Multi-channel temporal input
            target_variable_idx: Target variable index
            is_training: Training mode flag
            
        Returns:
            Policy outputs dictionary
        """
        from .enriched_policy import EnrichedAttentionEncoder
        
        # Encode enriched history through transformer
        encoder = EnrichedAttentionEncoder(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            key_size=self.key_size,
            widening_factor=self.widening_factor,
            dropout=self.dropout
        )
        
        variable_embeddings = encoder(enriched_history, is_training)  # [n_vars, hidden_dim]
        
        # Apply simplified policy heads
        policy_heads = SimplifiedPolicyHeads(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.policy_intermediate_dim,
            dropout=self.dropout
        )
        
        policy_outputs = policy_heads(variable_embeddings, target_variable_idx, is_training)
        
        # Add embeddings to output for potential downstream use
        policy_outputs['variable_embeddings'] = variable_embeddings
        
        return policy_outputs


class PolicyOutputValidator:
    """Validator for policy network outputs."""
    
    @staticmethod
    def validate_policy_outputs(outputs: Dict[str, jnp.ndarray], 
                              n_vars: int) -> bool:
        """
        Validate policy network outputs.
        
        Args:
            outputs: Policy outputs dictionary
            n_vars: Number of variables
            
        Returns:
            True if outputs are valid, False otherwise
        """
        required_keys = ['variable_logits', 'value_params', 'state_value']
        
        # Check required keys
        for key in required_keys:
            if key not in outputs:
                logger.error(f"Missing required output key: {key}")
                return False
        
        # Check variable_logits
        var_logits = outputs['variable_logits']
        # Handle both single and batched outputs
        if len(var_logits.shape) == 1:
            # Single state case: [n_vars]
            if var_logits.shape != (n_vars,):
                logger.error(f"Invalid variable_logits shape: {var_logits.shape}, expected ({n_vars},)")
                return False
        else:
            # Batched case: [batch_size, n_vars]
            if len(var_logits.shape) != 2 or var_logits.shape[1] != n_vars:
                logger.error(f"Invalid variable_logits shape: {var_logits.shape}, expected (batch_size, {n_vars})")
                return False
        
        if not jnp.all(jnp.isfinite(var_logits)):
            logger.error("Non-finite values in variable_logits")
            return False
        
        # Check value_params
        val_params = outputs['value_params']
        # Handle both single and batched outputs
        if len(val_params.shape) == 2:
            # Single state case: [n_vars, 2]
            if val_params.shape != (n_vars, 2):
                logger.error(f"Invalid value_params shape: {val_params.shape}, expected ({n_vars}, 2)")
                return False
        else:
            # Batched case: [batch_size, n_vars, 2]
            if len(val_params.shape) != 3 or val_params.shape[1] != n_vars or val_params.shape[2] != 2:
                logger.error(f"Invalid value_params shape: {val_params.shape}, expected (batch_size, {n_vars}, 2)")
                return False
        
        if not jnp.all(jnp.isfinite(val_params)):
            logger.error("Non-finite values in value_params")
            return False
        
        # Check log_std constraints
        if len(val_params.shape) == 2:
            # Single state case: [n_vars, 2]
            log_stds = val_params[:, 1]
        else:
            # Batched case: [batch_size, n_vars, 2]
            log_stds = val_params[:, :, 1]
        
        if jnp.any(log_stds < -3.0) or jnp.any(log_stds > 3.0):
            logger.warning("log_std values outside reasonable range [-3, 3]")
        
        # Check state_value
        state_val = outputs['state_value']
        # Handle both single and batched state values
        if jnp.isscalar(state_val):
            # Single state case
            if not jnp.isfinite(state_val):
                logger.error("Non-finite state_value")
                return False
        else:
            # Batched case: should be 1D array with one value per batch item
            if len(state_val.shape) != 1:
                logger.error(f"state_value should be scalar or 1D array, got shape: {state_val.shape}")
                return False
            if not jnp.all(jnp.isfinite(state_val)):
                logger.error("Non-finite values in state_value")
                return False
        
        return True
    
    @staticmethod
    def validate_enriched_history(enriched_history: jnp.ndarray,
                                expected_channels: int = 10) -> bool:
        """
        Validate enriched history input.
        
        Args:
            enriched_history: Input tensor [T, n_vars, num_channels]
            expected_channels: Expected number of channels
            
        Returns:
            True if input is valid, False otherwise
        """
        if len(enriched_history.shape) != 3:
            logger.error(f"Enriched history should be 3D, got shape: {enriched_history.shape}")
            return False
        
        T, n_vars, num_channels = enriched_history.shape
        
        if num_channels != expected_channels:
            logger.error(f"Expected {expected_channels} channels, got {num_channels}")
            return False
        
        if n_vars < 1:
            logger.error(f"Need at least 1 variable, got {n_vars}")
            return False
        
        if T < 1:
            logger.error(f"Need at least 1 timestep, got {T}")
            return False
        
        if not jnp.all(jnp.isfinite(enriched_history)):
            logger.error("Non-finite values in enriched_history")
            return False
        
        return True


def create_enriched_policy_factory(config: Dict[str, any]) -> callable:
    """
    Factory function for creating enriched policy networks.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Haiku-compatible policy function
    """
    def policy_fn(enriched_history: jnp.ndarray,
                  target_variable_idx: int,
                  is_training: bool = True) -> Dict[str, jnp.ndarray]:
        """
        Enriched policy network function.
        
        Args:
            enriched_history: Multi-channel temporal input
            target_variable_idx: Target variable index
            is_training: Training mode flag
            
        Returns:
            Policy outputs dictionary
        """
        # Validate inputs (skip during JAX compilation)
        # Note: validation is skipped during JAX compilation to avoid boolean conversion errors
        # Input validation should be done before calling the compiled function
        
        # Create policy network
        policy = EnrichedAcquisitionPolicyNetwork(
            num_layers=config.get('num_layers', 4),
            num_heads=config.get('num_heads', 8),
            hidden_dim=config.get('hidden_dim', 128),
            key_size=config.get('key_size', 32),
            widening_factor=config.get('widening_factor', 4),
            dropout=config.get('dropout', 0.1),
            policy_intermediate_dim=config.get('policy_intermediate_dim', None)
        )
        
        outputs = policy(enriched_history, target_variable_idx, is_training)
        
        # Note: output validation is skipped during JAX compilation to avoid boolean conversion errors
        # Output validation should be done outside the compiled function if needed
        
        return outputs
    
    return policy_fn