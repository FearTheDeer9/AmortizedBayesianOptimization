"""
Model Registry for JAX/Haiku Model Reconstruction

This module provides a registry system for model creation functions,
enabling reconstruction of models from saved configurations.
Follows functional programming principles with immutable registries.
"""

import logging
from typing import Dict, Any, Callable, Optional, Tuple
import pyrsistent as pyr
import haiku as hk

logger = logging.getLogger(__name__)


# Global model registry (immutable)
_MODEL_REGISTRY: pyr.PMap = pyr.pmap()


def register_model_creator(
    model_type: str,
    creator_fn: Callable[[Dict[str, Any]], Tuple[hk.Transformed, Dict[str, Any]]]
) -> None:
    """
    Register a model creation function.
    
    Args:
        model_type: Unique identifier for the model type
        creator_fn: Function that creates model from config
    """
    global _MODEL_REGISTRY
    _MODEL_REGISTRY = _MODEL_REGISTRY.set(model_type, creator_fn)
    logger.info(f"Registered model creator for type: {model_type}")


def get_model_creator(model_type: str) -> Optional[Callable]:
    """
    Get registered model creator function.
    
    Args:
        model_type: Model type identifier
        
    Returns:
        Model creator function or None if not found
    """
    return _MODEL_REGISTRY.get(model_type)


def create_model_from_config(
    model_type: str,
    config: Dict[str, Any]
) -> Tuple[hk.Transformed, Dict[str, Any]]:
    """
    Create model from configuration using registered creator.
    
    Args:
        model_type: Model type identifier
        config: Model configuration
        
    Returns:
        Tuple of (haiku_transformed_model, model_config)
        
    Raises:
        ValueError: If model type not registered
    """
    creator_fn = get_model_creator(model_type)
    if creator_fn is None:
        raise ValueError(f"No creator registered for model type: {model_type}")
    
    return creator_fn(config)


# Register default model creators
def _register_default_creators():
    """Register default model creators for BC models."""
    
    # Surrogate model creators
    def create_continuous_surrogate(config: Dict[str, Any]) -> Tuple[hk.Transformed, Dict[str, Any]]:
        """Create continuous parent set surrogate model."""
        from ..avici_integration.continuous.factory import create_continuous_parent_set_model
        from ..avici_integration.continuous.factory import create_continuous_parent_set_config
        
        # Create model configuration
        model_config = create_continuous_parent_set_config(
            variables=config.get("variables", []),
            target_variable=config.get("target_variable", ""),
            model_complexity=config.get("model_complexity", "medium"),
            use_attention=config.get("use_attention", True),
            temperature=config.get("temperature", 1.0)
        )
        
        # Update with any additional config parameters
        model_config.update(config.get("parameters", {}))
        
        # Create model
        return create_continuous_parent_set_model(model_config)
    
    def create_jax_unified_surrogate(config: Dict[str, Any]) -> Tuple[hk.Transformed, Dict[str, Any]]:
        """Create JAX unified parent set surrogate model."""
        from ..avici_integration.parent_set.unified.jax_model import create_jax_unified_parent_set_model
        from ..avici_integration.parent_set.unified.config import create_structure_only_config
        
        # Create base configuration
        base_config = create_structure_only_config()
        
        # Create model with variable names
        return create_jax_unified_parent_set_model(
            config=base_config,
            variable_names=config.get("variables", [])
        )
    
    # Acquisition model creators
    def create_enhanced_acquisition(config: Dict[str, Any]) -> Tuple[hk.Transformed, Dict[str, Any]]:
        """Create enhanced acquisition policy model."""
        from ..acquisition.enhanced_policy_network import EnhancedPolicyNetwork
        
        def policy_fn(state_dict, is_training):
            # Extract state components
            state_tensor = state_dict.get('state_tensor')
            target_idx = state_dict.get('target_variable_idx', 0)
            history = state_dict.get('history_tensor', None)
            
            # Create network with config
            network = EnhancedPolicyNetwork(
                hidden_dim=config.get("hidden_dim", 128),
                num_layers=config.get("num_layers", 3),
                num_heads=config.get("num_heads", 4),
                key_size=config.get("key_size", 32),
                dropout=config.get("dropout", 0.1),
                num_variables=None  # Dynamic
            )
            
            return network(state_tensor, target_idx, history, is_training)
        
        # Transform to Haiku function
        model = hk.transform(policy_fn)
        
        model_config = {
            "model_type": "enhanced_acquisition",
            "parameters": config
        }
        
        return model, model_config
    
    def create_standard_acquisition(config: Dict[str, Any]) -> Tuple[hk.Transformed, Dict[str, Any]]:
        """Create standard acquisition policy model."""
        from ..acquisition.policy_networks import AcquisitionPolicyNetwork
        
        def policy_fn(state, is_training):
            network = AcquisitionPolicyNetwork(
                hidden_dim=config.get("hidden_dim", 128),
                num_layers=config.get("num_layers", 3),
                num_heads=config.get("num_heads", 4),
                dropout=config.get("dropout", 0.1)
            )
            return network(state, is_training)
        
        # Transform to Haiku function
        model = hk.transform(policy_fn)
        
        model_config = {
            "model_type": "standard_acquisition",
            "parameters": config
        }
        
        return model, model_config
    
    # Register all creators
    register_model_creator("continuous_surrogate", create_continuous_surrogate)
    register_model_creator("jax_unified_surrogate", create_jax_unified_surrogate)
    register_model_creator("enhanced_acquisition", create_enhanced_acquisition)
    register_model_creator("standard_acquisition", create_standard_acquisition)
    
    logger.info("Registered default model creators")


# Initialize default creators on module import
_register_default_creators()


def list_registered_models() -> list[str]:
    """List all registered model types."""
    return list(_MODEL_REGISTRY.keys())


__all__ = [
    'register_model_creator',
    'get_model_creator',
    'create_model_from_config',
    'list_registered_models'
]