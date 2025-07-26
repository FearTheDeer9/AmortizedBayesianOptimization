"""
Unified Evaluation Framework for Causal Bayesian Optimization

This package provides a consistent interface for evaluating different
causal discovery and optimization methods, including GRPO and BC approaches.

New unified framework provides:
- BaseEvaluator: Abstract base class for all evaluation methods
- Standardized result types for consistent data format
- UnifiedEvaluationRunner: Single runner for all methods

Legacy framework preserved for backward compatibility.
"""

# New unified framework imports
try:
    from .base_evaluator import BaseEvaluator
except ImportError:
    BaseEvaluator = None

try:
    from .result_types import (
        ExperimentResult,
        StepResult,
        ComparisonResults,
        MethodMetrics
    )
except ImportError:
    ExperimentResult = None
    StepResult = None
    ComparisonResults = None
    MethodMetrics = None

try:
    from .unified_runner import UnifiedEvaluationRunner
except ImportError:
    UnifiedEvaluationRunner = None

try:
    from .grpo_evaluator import GRPOEvaluator
except ImportError:
    GRPOEvaluator = None

try:
    from .bc_evaluator import BCEvaluator
except ImportError:
    BCEvaluator = None

try:
    from .baseline_evaluators import (
        RandomBaselineEvaluator,
        OracleBaselineEvaluator,
        LearningBaselineEvaluator
    )
except ImportError:
    RandomBaselineEvaluator = None
    OracleBaselineEvaluator = None
    LearningBaselineEvaluator = None

try:
    from .notebook_helpers import (
        setup_evaluation_runner,
        run_evaluation_comparison,
        results_to_dataframe,
        plot_learning_curves,
        create_summary_report,
        load_and_visualize_results
    )
except ImportError:
    setup_evaluation_runner = None
    run_evaluation_comparison = None
    results_to_dataframe = None
    plot_learning_curves = None
    create_summary_report = None
    load_and_visualize_results = None

try:
    from .run_evaluation import run_evaluation
except ImportError:
    run_evaluation = None

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

# Build __all__ dynamically to include new framework if available
__all__ = []

# New unified framework exports
if BaseEvaluator is not None:
    __all__.append('BaseEvaluator')

if ExperimentResult is not None:
    __all__.extend([
        'ExperimentResult',
        'StepResult', 
        'ComparisonResults',
        'MethodMetrics'
    ])

if UnifiedEvaluationRunner is not None:
    __all__.append('UnifiedEvaluationRunner')

if GRPOEvaluator is not None:
    __all__.append('GRPOEvaluator')

if BCEvaluator is not None:
    __all__.append('BCEvaluator')

if RandomBaselineEvaluator is not None:
    __all__.extend([
        'RandomBaselineEvaluator',
        'OracleBaselineEvaluator',
        'LearningBaselineEvaluator'
    ])

if setup_evaluation_runner is not None:
    __all__.extend([
        'setup_evaluation_runner',
        'run_evaluation_comparison',
        'results_to_dataframe',
        'plot_learning_curves',
        'create_summary_report',
        'load_and_visualize_results'
    ])

if run_evaluation is not None:
    __all__.append('run_evaluation')

# Legacy exports
__all__.extend([
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
])