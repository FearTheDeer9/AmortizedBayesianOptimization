"""
Evaluation framework for ACBO experiments.

This module provides utilities for loading trained models, collecting metrics,
and visualizing results across different training methods.
"""

from .core.model_loader import ModelLoader
try:
    from .core.metric_collector import MetricCollector
except ImportError:
    MetricCollector = None
from .core.plotting_utils import plot_evaluation_results

__all__ = ['ModelLoader', 'MetricCollector', 'plot_evaluation_results']