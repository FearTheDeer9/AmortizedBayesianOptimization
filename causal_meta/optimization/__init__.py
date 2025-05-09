"""Optimization module for causal meta learning."""

# Import interfaces for easy access
from causal_meta.optimization.interfaces import AcquisitionStrategy, Intervention

# Import acquisition strategies
from causal_meta.optimization.acquisition import (
    ExpectedImprovement,
    UpperConfidenceBound
)
