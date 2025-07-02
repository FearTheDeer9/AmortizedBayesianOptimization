"""
Enhanced Policy Network Factory for GRPO Training.

This module provides factory functions for creating enhanced policy networks
that use enriched attention architectures for improved intervention selection.
These networks are designed to work with GRPO training and provide the
policy backbone for enhanced ACBO.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Dict, List, Optional, Callable, Any, Tuple
import logging
import pyrsistent as pyr

from .enriched.enriched_policy import EnrichedAttentionEncoder
from .enriched.policy_heads import SimplifiedPolicyHeads
from .enriched.state_enrichment import EnrichedHistoryBuilder
from ..data_structures import ExperienceBuffer

logger = logging.getLogger(__name__)


class EnhancedPolicyNetwork(hk.Module):
    """
    Enhanced policy network using enriched attention architecture.
    
    Integrates:
    - Enriched attention for processing multi-channel temporal context
    - Policy heads for intervention selection and value estimation
    - JAX-native implementation for GRPO training compatibility
    """
    
    def __init__(self,
                 # Architecture parameters
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 key_size: int = 32,
                 # Policy parameters
                 num_variables: int = 5,
                 intervention_dim: int = 64,
                 # Training parameters
                 dropout: float = 0.1,
                 name: str = "EnhancedPolicyNetwork"):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.key_size = key_size
        self.num_variables = num_variables
        self.intervention_dim = intervention_dim
        self.dropout = dropout
        
        # Initialize sub-components
        self.history_builder = EnrichedHistoryBuilder(
            standardize_values=True,
            include_temporal_features=True
        )
        
        self.attention_encoder = EnrichedAttentionEncoder(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            key_size=key_size,
            dropout=dropout,
            name="attention_encoder"
        )
        
        self.policy_head = SimplifiedPolicyHeads(
            hidden_dim=hidden_dim,
            intermediate_dim=intervention_dim,
            dropout=dropout,
            name="policy_head"
        )
    
    def __call__(self, 
                 state_tensor: jnp.ndarray,  # [n_vars, feature_dim]
                 target_variable_idx: int = 0,  # Index of target variable
                 history_tensor: Optional[jnp.ndarray] = None,  # [history_len, n_vars, feature_dim]
                 is_training: bool = False) -> Dict[str, jnp.ndarray]:
        """
        Forward pass through enhanced policy network.
        
        Args:
            state_tensor: Current state representation
            target_variable_idx: Index of target variable (for masking)
            history_tensor: Optional history for temporal context
            is_training: Whether in training mode
            
        Returns:
            Dictionary containing policy outputs:
            - 'variable_logits': Logits for variable selection
            - 'value_params': Parameters for intervention values
            - 'state_value': State value estimate
        """
        # Prepare input for attention encoder
        if history_tensor is not None:
            # Concatenate current state with history
            enriched_history = jnp.concatenate([history_tensor, state_tensor[None]], axis=0)
        else:
            enriched_history = state_tensor[None]  # Add sequence dimension
        
        # Process through attention encoder
        current_representation = self.attention_encoder(
            enriched_history=enriched_history,
            is_training=is_training
        )  # Returns [n_vars, hidden_dim] directly
        
        # Generate policy outputs
        policy_outputs = self.policy_head(
            variable_embeddings=current_representation,
            target_variable_idx=target_variable_idx,
            is_training=is_training
        )
        
        # Rename outputs to match expected interface
        return {
            'intervention_logits': policy_outputs['variable_logits'],
            'value_params': policy_outputs['value_params'],
            'value_estimate': policy_outputs['state_value']
        }


def create_enhanced_policy_for_grpo(
    variables: List[str],
    target_variable: str,
    architecture_level: str = "full",
    performance_mode: str = "balanced"
) -> Tuple[Callable, Dict[str, Any]]:
    """
    Factory function for creating enhanced policy networks for GRPO training.
    
    Args:
        variables: List of variable names in the SCM
        target_variable: Name of target variable
        architecture_level: Architecture complexity ("full", "simplified", "baseline")
        performance_mode: Performance optimization ("fast", "balanced", "quality")
        
    Returns:
        Tuple of (policy_function, config_dict)
    """
    num_variables = len(variables)
    
    # Configure architecture based on level
    if architecture_level == "full":
        config = {
            "hidden_dim": 256,
            "num_layers": 6,
            "num_heads": 16,
            "key_size": 64,
            "intervention_dim": 128,
            "dropout": 0.1
        }
    elif architecture_level == "simplified":
        config = {
            "hidden_dim": 128,
            "num_layers": 4,
            "num_heads": 8,
            "key_size": 32,
            "intervention_dim": 64,
            "dropout": 0.1
        }
    else:  # baseline
        config = {
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "key_size": 16,
            "intervention_dim": 32,
            "dropout": 0.1
        }
    
    # Adjust for performance mode
    if performance_mode == "fast":
        config["hidden_dim"] = int(config["hidden_dim"] * 0.75)
        config["num_layers"] = max(1, config["num_layers"] - 1)
    elif performance_mode == "quality":
        config["hidden_dim"] = int(config["hidden_dim"] * 1.25)
        config["dropout"] = 0.05
    
    def policy_function(state_tensor: jnp.ndarray, 
                       history_tensor: Optional[jnp.ndarray] = None,
                       is_training: bool = False) -> Dict[str, jnp.ndarray]:
        """Enhanced policy function for GRPO training."""
        # Only use network parameters, not metadata
        network_params = {
            "hidden_dim": config["hidden_dim"],
            "num_layers": config["num_layers"],
            "num_heads": config["num_heads"],
            "key_size": config["key_size"],
            "intervention_dim": config["intervention_dim"],
            "dropout": config["dropout"]
        }
        network = EnhancedPolicyNetwork(
            num_variables=num_variables,
            **network_params
        )
        return network(state_tensor, target_variable_idx=num_variables-1, history_tensor=history_tensor, is_training=is_training)
    
    # Add metadata to config for return
    full_config = config.copy()
    full_config.update({
        "num_variables": num_variables,
        "target_variable": target_variable,
        "architecture_level": architecture_level,
        "performance_mode": performance_mode,
        "policy_type": "enhanced_attention"
    })
    
    return policy_function, full_config


def validate_enhanced_policy_integration() -> bool:
    """
    Validate that enhanced policy components are properly integrated.
    
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Test basic component creation
        test_variables = ["A", "B", "C", "D"]
        target_variable = "D"
        
        policy_fn, config = create_enhanced_policy_for_grpo(
            variables=test_variables,
            target_variable=target_variable,
            architecture_level="baseline"
        )
        
        # Test forward pass
        key = jax.random.PRNGKey(42)
        dummy_state = jax.random.normal(key, (len(test_variables), 32))
        
        # Transform function for testing
        transformed_policy = hk.transform(lambda x: policy_fn(x, is_training=False))
        params = transformed_policy.init(key, dummy_state)
        outputs = transformed_policy.apply(params, key, dummy_state)
        
        # Validate outputs
        required_keys = ["intervention_logits", "value_estimate"]
        for key_name in required_keys:
            if key_name not in outputs:
                logger.error(f"Missing required output: {key_name}")
                return False
        
        # Validate shapes
        if outputs["intervention_logits"].shape[0] != len(test_variables):
            logger.error(f"Invalid intervention_logits shape: {outputs['intervention_logits'].shape}")
            return False
        
        logger.info("Enhanced policy integration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Enhanced policy validation failed: {e}")
        return False


# Export main functions
__all__ = [
    'EnhancedPolicyNetwork',
    'create_enhanced_policy_for_grpo',
    'validate_enhanced_policy_integration'
]