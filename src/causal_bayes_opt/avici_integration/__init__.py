"""
AVICI Integration Module

This module provides the data format bridge between our functional SCM implementation
and AVICI's neural network architecture. It enables target-aware causal discovery
by extending AVICI's input format from [N, d, 2] to [N, d, 3] with target conditioning.

Key functionality:
- Convert Sample objects to AVICI-compatible tensor format
- Support target-aware learning with conditioning channel
- Preserve all information during conversion
- Provide validation and analysis utilities

Main functions:
- samples_to_avici_format: Core conversion function
- create_training_batch: AVICI-compatible batch creation
- validate_data_conversion: Information preservation validation
- analyze_avici_data: Data analysis and debugging utilities
"""

# Import core conversion functions (main public API)
from .conversion import (
    samples_to_avici_format,
    create_training_batch,
)

# Import key validation functions
from .validation import (
    validate_data_conversion,
)

# Import analysis and debugging utilities
from .analysis import (
    analyze_avici_data,
    reconstruct_samples_from_avici_data,
    get_variable_order_from_scm,
    compare_data_conversions,
    debug_sample_conversion,
)


# Import data utilities (Phase 1.3)
from .data_utils import (
    target_aware_standardize_default,
    target_aware_get_x,
    target_aware_get_train_x,
)

# Import type aliases for user convenience
from .conversion import (
    SampleList,
    VariableOrder,
    AVICIDataBatch,
)

# Module metadata
__version__ = "0.1.0"
__author__ = "Harel Lidar"
__description__ = "AVICI Integration for Target-Aware Causal Discovery"

# Export all public functions
__all__ = [
    # Core conversion functions (most important)
    "samples_to_avici_format",
    "create_training_batch",
    
    # Validation functions
    "validate_data_conversion",
    
    # Analysis and debugging utilities
    "analyze_avici_data",
    "reconstruct_samples_from_avici_data", 
    "get_variable_order_from_scm",
    "compare_data_conversions",
    "debug_sample_conversion",
    
    # Note: Target-aware model classes removed (deprecated in favor of modular architecture)
    
    # Note: Parent set classes moved to modular parent_set/ package
    
    # Note: Simple training utilities moved to tests/examples/
    
    # Data utilities (Phase 1.3)
    "target_aware_standardize_default",
    "target_aware_get_x",
    "target_aware_get_train_x",
    
    # Type aliases
    "SampleList",
    "VariableOrder",
    "AVICIDataBatch",
]

# Note: Internal modules (_helpers.py) are not exposed in __all__
# Users should only import from the public API defined above