"""
ACBO Comparison Framework

This module provides a clean, modular framework for comparing different ACBO methods.
The framework separates concerns into distinct components for better maintainability
and extensibility.
"""

from .experiment_runner import ACBOExperimentRunner
from .method_registry import MethodRegistry, ExperimentMethod
from .scm_manager import SCMManager
from .metrics_collector import MetricsCollector
from .wandb_integration import WandBLogger
from .statistical_analysis import StatisticalAnalyzer
from .visualization import VisualizationManager

__all__ = [
    'ACBOExperimentRunner',
    'MethodRegistry', 
    'ExperimentMethod',
    'SCMManager',
    'MetricsCollector',
    'WandBLogger',
    'StatisticalAnalyzer',
    'VisualizationManager'
]

__version__ = '1.0.0'