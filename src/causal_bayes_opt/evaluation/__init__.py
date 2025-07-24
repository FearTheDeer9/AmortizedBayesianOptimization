"""
Comprehensive Evaluation Framework for ACBO.

This module provides robust evaluation metrics, baseline comparisons,
and statistical significance testing for the ACBO system.
"""

from .metrics import *
from .baselines import *
from .statistical_tests import *
from .bc_performance_tracker import (
    BCPerformanceTracker,
    PerformanceMetrics,
    PerformanceTrajectory,
    calculate_shd,
    calculate_f1_score,
    extract_ground_truth_graph
)
from .bc_visualization import (
    plot_performance_trajectories,
    plot_performance_comparison,
    create_performance_summary_table
)

__all__ = [
    # Metrics
    'CausalDiscoveryMetrics',
    'OptimizationMetrics', 
    'EfficiencyMetrics',
    'CompositeMetrics',
    'compute_causal_discovery_metrics',
    'compute_optimization_metrics',
    'compute_efficiency_metrics',
    'compute_composite_metrics',
    
    # Baselines
    'RandomBaseline',
    'ParentScaleBaseline', 
    'GreedyBaseline',
    'create_baseline_comparison',
    
    # Statistical tests
    'StatisticalTestResult',
    'paired_t_test',
    'wilcoxon_signed_rank_test',
    'bootstrap_confidence_interval',
    'compute_effect_size',
    
    # BC Performance Tracking
    'BCPerformanceTracker',
    'PerformanceMetrics',
    'PerformanceTrajectory',
    'calculate_shd',
    'calculate_f1_score',
    'extract_ground_truth_graph',
    
    # BC Visualization
    'plot_performance_trajectories',
    'plot_performance_comparison',
    'create_performance_summary_table'
]