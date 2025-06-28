"""
Analysis utilities for ACBO experiments.

This module provides functions for computing derived metrics from trajectory data
without modifying core data structures.
"""

from .trajectory_metrics import (
    compute_true_parent_likelihood,
    compute_trajectory_metrics,
    analyze_convergence_trajectory,
    extract_learning_curves
)

__all__ = [
    'compute_true_parent_likelihood',
    'compute_trajectory_metrics', 
    'analyze_convergence_trajectory',
    'extract_learning_curves'
]