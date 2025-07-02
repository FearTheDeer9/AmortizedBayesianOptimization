"""
Integration module for architectural compatibility and A/B testing.

This module provides unified interfaces that can switch between different
architectural implementations while maintaining backward compatibility.
"""

from .unified_interfaces import (
    UnifiedParentSetModel,
    UnifiedAcquisitionPolicy,
    create_unified_acbo_pipeline
)

# Import validation framework when available
try:
    from .validation_framework import (
        ArchitectureValidator,
        SideBySideValidator,
        ValidationMetrics
    )
    _validation_available = True
except ImportError:
    _validation_available = False

if _validation_available:
    __all__ = [
        "UnifiedParentSetModel",
        "UnifiedAcquisitionPolicy", 
        "create_unified_acbo_pipeline",
        "ArchitectureValidator",
        "SideBySideValidator",
        "ValidationMetrics",
    ]
else:
    __all__ = [
        "UnifiedParentSetModel",
        "UnifiedAcquisitionPolicy", 
        "create_unified_acbo_pipeline",
    ]