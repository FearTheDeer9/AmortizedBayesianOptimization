"""
Encoder Factory for BC Surrogate Model.

This module provides factory functions for creating different encoder architectures
and managing encoder selection based on configuration.

Available Encoders:
- node_feature: NodeFeatureEncoder (recommended) - prevents embedding collapse
- node: Original NodeEncoder - may suffer from uniformity issues
- simple: SimpleNodeFeatureEncoder - shared parameters version
- improved: ImprovedNodeEncoder - uses cross-sample attention
"""

import logging
from typing import Any, Dict, Optional, Callable, Tuple
import haiku as hk

logger = logging.getLogger(__name__)


def create_encoder(encoder_type: str, 
                  config: Optional[Dict[str, Any]] = None) -> hk.Module:
    """
    Create an encoder based on the specified type.
    
    Args:
        encoder_type: Type of encoder to create. Options:
            - "node_feature": NodeFeatureEncoder (recommended)
            - "node": Original NodeEncoder
            - "simple": SimpleNodeFeatureEncoder
            - "improved": ImprovedNodeEncoder
        config: Optional configuration dictionary with parameters like:
            - hidden_dim: Hidden dimension (default: 128)
            - num_layers: Number of layers (default: 2)
            - dropout: Dropout rate (default: 0.1)
            
    Returns:
        Encoder module instance
        
    Raises:
        ValueError: If encoder_type is not recognized
    """
    if config is None:
        config = {}
    
    # Extract common parameters
    hidden_dim = config.get('hidden_dim', 128)
    num_layers = config.get('num_layers', 2)
    dropout = config.get('dropout', 0.1)
    
    logger.debug(f"Creating encoder of type '{encoder_type}' with config: {config}")
    
    if encoder_type == "node_feature":
        from .node_feature_encoder import NodeFeatureEncoder
        return NodeFeatureEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout
        )
    
    elif encoder_type == "simple":
        from .node_feature_encoder import SimpleNodeFeatureEncoder
        return SimpleNodeFeatureEncoder(
            hidden_dim=hidden_dim
        )
    
    elif encoder_type == "node":
        # Original encoder from model.py
        from .model import NodeEncoder
        return NodeEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
    
    elif encoder_type == "improved":
        # Improved encoder with cross-sample attention
        from .improved_model import ImprovedNodeEncoder
        return ImprovedNodeEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_attention_heads=config.get('num_attention_heads', 4),
            key_size=config.get('key_size', 32)
        )
    
    else:
        raise ValueError(
            f"Unknown encoder type: '{encoder_type}'. "
            f"Available options: 'node_feature', 'node', 'simple', 'improved'"
        )


def create_attention_layer(attention_type: str,
                          config: Optional[Dict[str, Any]] = None) -> hk.Module:
    """
    Create an attention layer based on the specified type.
    
    Args:
        attention_type: Type of attention layer. Options:
            - "pairwise": ParentAttentionLayer with pairwise features
            - "simple": SimpleParentAttentionLayer
            - "original": Original ParentAttentionLayer from model.py
            
    Returns:
        Attention layer module instance
    """
    if config is None:
        config = {}
    
    hidden_dim = config.get('hidden_dim', 128)
    num_heads = config.get('num_heads', 8)
    key_size = config.get('key_size', 32)
    dropout = config.get('dropout', 0.1)
    
    if attention_type == "pairwise":
        from .parent_attention import ParentAttentionLayer
        return ParentAttentionLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            key_size=key_size
        )
    
    elif attention_type == "simple":
        from .parent_attention import SimpleParentAttentionLayer
        return SimpleParentAttentionLayer(
            hidden_dim=hidden_dim,
            key_size=key_size
        )
    
    elif attention_type == "original":
        from .model import ParentAttentionLayer
        return ParentAttentionLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            key_size=key_size
        )
    
    else:
        raise ValueError(
            f"Unknown attention type: '{attention_type}'. "
            f"Available options: 'pairwise', 'simple', 'original'"
        )


def get_encoder_config(encoder_type: str) -> Dict[str, Any]:
    """
    Get default configuration for a specific encoder type.
    
    Args:
        encoder_type: Type of encoder
        
    Returns:
        Default configuration dictionary
    """
    configs = {
        "node_feature": {
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.1,
            "description": "Per-variable feature encoder (prevents collapse)"
        },
        "node": {
            "hidden_dim": 128,
            "num_layers": 2,
            "description": "Original node encoder"
        },
        "simple": {
            "hidden_dim": 128,
            "description": "Simplified feature encoder"
        },
        "improved": {
            "hidden_dim": 128,
            "num_layers": 2,
            "num_attention_heads": 4,
            "key_size": 32,
            "description": "Encoder with cross-sample attention"
        }
    }
    
    if encoder_type not in configs:
        logger.warning(f"No default config for encoder type '{encoder_type}', using node_feature defaults")
        return configs["node_feature"]
    
    return configs[encoder_type]


def validate_encoder(encoder_type: str) -> bool:
    """
    Validate that an encoder type is available and can be created.
    
    Args:
        encoder_type: Type of encoder to validate
        
    Returns:
        True if encoder can be created, False otherwise
    """
    try:
        # Try to create encoder with default config
        config = get_encoder_config(encoder_type)
        encoder = create_encoder(encoder_type, config)
        return encoder is not None
    except Exception as e:
        logger.error(f"Failed to validate encoder '{encoder_type}': {e}")
        return False


def list_available_encoders() -> Dict[str, str]:
    """
    List all available encoder types with descriptions.
    
    Returns:
        Dictionary mapping encoder type to description
    """
    encoders = {
        "node_feature": "NodeFeatureEncoder - Per-variable features, prevents embedding collapse (recommended)",
        "node": "NodeEncoder - Original encoder, may suffer from uniformity issues",
        "simple": "SimpleNodeFeatureEncoder - Simplified version with shared parameters",
        "improved": "ImprovedNodeEncoder - Uses cross-sample attention for relationship awareness"
    }
    return encoders


# Compatibility functions for smooth integration
def create_encoder_and_attention(encoder_type: str,
                                attention_type: Optional[str] = None,
                                config: Optional[Dict[str, Any]] = None) -> Tuple[hk.Module, hk.Module]:
    """
    Create both encoder and attention layer with compatible configurations.
    
    Args:
        encoder_type: Type of encoder
        attention_type: Type of attention layer (auto-selected if None)
        config: Configuration for both modules
        
    Returns:
        Tuple of (encoder, attention_layer)
    """
    if config is None:
        config = {}
    
    # Auto-select compatible attention type
    if attention_type is None:
        if encoder_type == "node_feature":
            attention_type = "pairwise"  # Best pairing
        elif encoder_type == "improved":
            attention_type = "original"  # Already has cross-sample attention
        else:
            attention_type = "original"  # Default
    
    encoder = create_encoder(encoder_type, config)
    attention = create_attention_layer(attention_type, config)
    
    logger.info(f"Created encoder-attention pair: {encoder_type} + {attention_type}")
    
    return encoder, attention