"""
Bridge Layer for Legacy-to-JAX Conversion

This module provides conversion functions between the legacy dictionary-based
acquisition system and the new JAX-native tensor-based architecture.

Key components:
- Legacy-to-JAX state conversion
- Data format transformation
- Compatibility wrappers
- Validation functions

Design principles:
- One-way conversion (legacy → JAX)
- No JAX compilation in bridge layer
- Clean error handling and validation
- Performance monitoring for conversion overhead
"""

from .legacy_to_jax import (
    convert_legacy_to_jax,
    convert_legacy_buffer_to_jax,
    convert_legacy_mechanisms_to_jax,
    validate_conversion_result
)

from .compatibility import (
    JAXCompatibilityWrapper,
    create_compatibility_layer,
    legacy_api_adapter
)

__all__ = [
    # Conversion functions
    "convert_legacy_to_jax",
    "convert_legacy_buffer_to_jax", 
    "convert_legacy_mechanisms_to_jax",
    "validate_conversion_result",
    
    # Compatibility layer
    "JAXCompatibilityWrapper",
    "create_compatibility_layer",
    "legacy_api_adapter"
]