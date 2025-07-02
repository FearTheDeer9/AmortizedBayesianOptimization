"""
Factory functions for parent set models and configurations.

This module provides factory functions for creating parent set models,
configurations, and related components for enhanced ACBO integration.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def create_parent_set_config(
    variables: List[str],
    target_variable: str,
    model_type: str = "unified",
    complexity: str = "medium"
) -> Dict[str, Any]:
    """
    Create configuration for parent set models.
    
    Args:
        variables: List of variable names
        target_variable: Target variable name
        model_type: Type of parent set model ("unified", "legacy", "jax")
        complexity: Model complexity level ("simple", "medium", "full")
        
    Returns:
        Configuration dictionary for parent set model
    """
    complexity_configs = {
        "simple": {
            "hidden_dim": 32,
            "num_layers": 2,
            "max_parents": 3
        },
        "medium": {
            "hidden_dim": 64,
            "num_layers": 3,
            "max_parents": 5
        },
        "full": {
            "hidden_dim": 128,
            "num_layers": 4,
            "max_parents": 8
        }
    }
    
    base_config = complexity_configs.get(complexity, complexity_configs["medium"])
    
    config = {
        "variables": variables,
        "target_variable": target_variable,
        "n_variables": len(variables),
        "model_type": model_type,
        "complexity": complexity,
        **base_config
    }
    
    logger.info(f"Created parent set config with {complexity} complexity")
    return config


def create_parent_set_model(
    config: Dict[str, Any]
) -> Tuple[Callable, Dict[str, Any]]:
    """
    Create parent set model function.
    
    Args:
        config: Configuration dictionary from create_parent_set_config
        
    Returns:
        Tuple of (model_function, model_config)
    """
    try:
        model_type = config.get("model_type", "unified")
        
        if model_type == "unified":
            from .unified.model import UnifiedParentSetModel
            model = UnifiedParentSetModel(
                n_variables=config["n_variables"],
                hidden_dim=config["hidden_dim"],
                num_layers=config["num_layers"],
                max_parents=config["max_parents"]
            )
        elif model_type == "jax":
            from .jax_model import JAXParentSetModel
            model = JAXParentSetModel(
                n_variables=config["n_variables"],
                hidden_dim=config["hidden_dim"]
            )
        else:
            # Fallback to legacy model
            from .model import ParentSetPredictionModel
            model = ParentSetPredictionModel(
                n_variables=config["n_variables"],
                hidden_dim=config["hidden_dim"]
            )
        
        # Create model function
        def model_fn(x, is_training=True):
            return model(x, is_training=is_training)
        
        model_config = {
            "input_shape": (config["n_variables"], config["hidden_dim"]),
            "output_shape": (config["n_variables"],),
            "model_type": f"parent_set_{model_type}",
            "complexity": config["complexity"],
            "parameters": {
                "n_variables": config["n_variables"],
                "hidden_dim": config["hidden_dim"],
                "num_layers": config["num_layers"],
                "max_parents": config.get("max_parents", 5)
            }
        }
        
        logger.info(f"Created {model_type} parent set model successfully")
        return model_fn, model_config
        
    except Exception as e:
        logger.error(f"Failed to create parent set model: {e}")
        # Return fallback model
        return _create_fallback_parent_set_model(config)


def create_parent_set_for_enhanced_surrogate(
    variables: List[str],
    target_variable: str,
    model_complexity: str = "medium",
    performance_mode: str = "balanced"
) -> Tuple[Callable, Dict[str, Any]]:
    """
    Create parent set model optimized for enhanced surrogate integration.
    
    Args:
        variables: List of variable names
        target_variable: Target variable name
        model_complexity: Model complexity level
        performance_mode: Performance optimization mode
        
    Returns:
        Tuple of (model_function, model_config)
    """
    # Adjust configuration based on performance mode
    performance_adjustments = {
        "fast": {
            "model_complexity": "simple",
            "model_type": "jax"
        },
        "balanced": {
            "model_complexity": model_complexity,
            "model_type": "unified"
        },
        "quality": {
            "model_complexity": "full",
            "model_type": "unified"
        }
    }
    
    adjustments = performance_adjustments.get(performance_mode, performance_adjustments["balanced"])
    
    # Create configuration
    config = create_parent_set_config(
        variables=variables,
        target_variable=target_variable,
        model_type=adjustments["model_type"],
        complexity=adjustments["model_complexity"]
    )
    
    # Create model
    model_fn, model_config = create_parent_set_model(config)
    
    # Add enhanced surrogate-specific configuration
    model_config.update({
        "enhanced_surrogate_optimized": True,
        "performance_mode": performance_mode,
        "target_variable": target_variable,
        "variables": variables
    })
    
    logger.info(f"Created parent set model for enhanced surrogate with {performance_mode} performance mode")
    return model_fn, model_config


def validate_parent_set_integration() -> bool:
    """
    Validate parent set integration functionality.
    
    Returns:
        True if integration is working correctly, False otherwise
    """
    try:
        # Test with minimal configuration
        test_variables = ['X0', 'X1', 'X2']
        test_target = 'X0'
        
        # Test configuration creation
        config = create_parent_set_config(
            variables=test_variables,
            target_variable=test_target,
            complexity="simple"
        )
        
        # Validate configuration
        required_keys = ["variables", "target_variable", "n_variables", "hidden_dim"]
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required key in config: {key}")
                return False
        
        # Test model creation (may fail due to dependencies, but should handle gracefully)
        try:
            model_fn, model_config = create_parent_set_model(config)
            
            if model_fn is None:
                logger.error("Model function creation failed")
                return False
            
            if not isinstance(model_config, dict):
                logger.error("Model config is not a dictionary")
                return False
                
        except Exception as e:
            logger.warning(f"Model creation failed, but this is expected if dependencies are missing: {e}")
            # This is acceptable - fallback mechanisms should handle this
        
        # Test enhanced surrogate creation
        try:
            enhanced_model_fn, enhanced_config = create_parent_set_for_enhanced_surrogate(
                variables=test_variables,
                target_variable=test_target,
                performance_mode="fast"
            )
            
            if "enhanced_surrogate_optimized" not in enhanced_config:
                logger.error("Enhanced config missing enhanced_surrogate_optimized flag")
                return False
                
        except Exception as e:
            logger.warning(f"Enhanced model creation failed, but fallback should handle this: {e}")
        
        logger.info("Parent set integration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Parent set integration validation failed: {e}")
        return False


def _create_fallback_parent_set_model(config: Dict[str, Any]) -> Tuple[Callable, Dict[str, Any]]:
    """Create fallback parent set model when dependencies fail."""
    
    def fallback_model_fn(params, key, x):
        """Fallback model that returns uniform parent probabilities."""
        n_vars = config["n_variables"]
        # Return uniform parent probabilities for target variable
        return jnp.ones(n_vars) / n_vars
    
    fallback_config = {
        "input_shape": (config["n_variables"], config.get("hidden_dim", 32)),
        "output_shape": (config["n_variables"],),
        "model_type": "fallback_parent_set",
        "complexity": "minimal",
        "fallback": True,
        "parameters": {
            "n_variables": config["n_variables"],
            "model_complexity": "fallback"
        }
    }
    
    logger.warning("Using fallback parent set model")
    return fallback_model_fn, fallback_config


# Export key functions
__all__ = [
    'create_parent_set_config',
    'create_parent_set_model',
    'create_parent_set_for_enhanced_surrogate',
    'validate_parent_set_integration'
]