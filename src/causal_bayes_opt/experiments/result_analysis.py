"""
Result Analysis and Visualization for ACBO Experiments

This module provides functions for analyzing experimental results,
computing statistics, and preparing data for visualization.
"""

import logging
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings

# Standard numerical libraries
import jax.numpy as jnp
import numpy as onp  # For I/O only, following CLAUDE.md
import pyrsistent as pyr

# Statistical functions (available via scipy if PARENT_SCALE installed)
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    warnings.warn("SciPy not available. Some statistical functions will be limited.")
    SCIPY_AVAILABLE = False

# Local imports
from .experiment_runner import ExperimentResult, ExperimentType
from .baseline_methods import BaselineResult, BaselineType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComparisonResult:
    """Results from comparing different methods."""
    method_a: str
    method_b: str
    metric_name: str
    
    # Performance comparison
    mean_a: float
    mean_b: float
    improvement_ratio: float
    improvement_absolute: float
    
    # Statistical significance
    p_value: Optional[float] = None
    is_significant: Optional[bool] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    
    # Sample information
    n_samples_a: int = 0
    n_samples_b: int = 0
    
    metadata: Dict[str, Any] = None


@dataclass(frozen=True)
class ExperimentSummary:
    """Summary statistics for a collection of experiments."""
    experiment_type: str
    dataset_name: str
    
    # Performance metrics
    mean_f1_score: float
    std_f1_score: float
    mean_target_improvement: float
    std_target_improvement: float
    mean_sample_efficiency: float
    std_sample_efficiency: float
    
    # Timing metrics
    mean_time_seconds: float
    std_time_seconds: float
    mean_time_per_intervention: float
    
    # Success metrics
    success_rate: float
    n_experiments: int
    n_successful: int
    
    # Ranges
    min_f1: float
    max_f1: float
    median_f1: float


def load_experiment_results(results_dir: str) -> List[ExperimentResult]:
    """
    Load experiment results from directory.
    
    Args:
        results_dir: Directory containing experiment result JSON files
        
    Returns:
        List of ExperimentResult objects
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.warning(f"Results directory does not exist: {results_dir}")
        return []
    
    results = []
    
    for json_file in results_path.glob("*.json"):
        if json_file.name == "validation_summary.json":
            continue  # Skip summary files
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Convert back to ExperimentResult (simplified reconstruction)
            result = ExperimentResult(
                experiment_config=data.get('experiment_config', {}),
                dataset_summary=data.get('dataset_summary', {}),
                final_f1_score=data.get('final_f1_score', 0.0),
                structure_recovery_accuracy=data.get('structure_recovery_accuracy', 0.0),
                target_optimization_improvement=data.get('target_optimization_improvement', 0.0),
                sample_efficiency=data.get('sample_efficiency', 0.0),
                convergence_steps=data.get('convergence_steps', 0),
                total_time_seconds=data.get('total_time_seconds', 0.0),
                time_per_intervention=data.get('time_per_intervention', 0.0),
                timestamp=data.get('timestamp', ''),
                success=data.get('success', True),
                error_message=data.get('error_message')
            )
            
            results.append(result)
            
        except Exception as e:
            logger.warning(f"Failed to load result from {json_file}: {e}")
    
    logger.info(f"Loaded {len(results)} experiment results from {results_dir}")
    return results


def compute_experiment_summary(results: List[ExperimentResult]) -> Dict[str, ExperimentSummary]:
    """
    Compute summary statistics for experiment results grouped by type and dataset.
    
    Args:
        results: List of experiment results
        
    Returns:
        Dictionary mapping (experiment_type, dataset) to summary statistics
    """
    # Group results by experiment type and dataset
    grouped_results = {}
    
    for result in results:
        if not result.success:
            continue  # Skip failed experiments
        
        exp_type = result.experiment_config.get('experiment_type', 'unknown')
        dataset = result.experiment_config.get('dataset_name', 'unknown')
        key = f"{exp_type}_{dataset}"
        
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(result)
    
    # Compute summaries
    summaries = {}
    
    for key, group_results in grouped_results.items():
        exp_type, dataset = key.split('_', 1)
        
        # Extract metrics
        f1_scores = [r.final_f1_score for r in group_results]
        target_improvements = [r.target_optimization_improvement for r in group_results]
        sample_efficiencies = [r.sample_efficiency for r in group_results]
        times = [r.total_time_seconds for r in group_results]
        time_per_intervention = [r.time_per_intervention for r in group_results]
        
        # Compute statistics
        summary = ExperimentSummary(
            experiment_type=exp_type,
            dataset_name=dataset,
            
            # Performance metrics
            mean_f1_score=float(onp.mean(f1_scores)),
            std_f1_score=float(onp.std(f1_scores)),
            mean_target_improvement=float(onp.mean(target_improvements)),
            std_target_improvement=float(onp.std(target_improvements)),
            mean_sample_efficiency=float(onp.mean(sample_efficiencies)),
            std_sample_efficiency=float(onp.std(sample_efficiencies)),
            
            # Timing metrics
            mean_time_seconds=float(onp.mean(times)),
            std_time_seconds=float(onp.std(times)),
            mean_time_per_intervention=float(onp.mean(time_per_intervention)),
            
            # Success metrics
            success_rate=1.0,  # We only included successful experiments
            n_experiments=len(group_results),
            n_successful=len(group_results),
            
            # Ranges
            min_f1=float(onp.min(f1_scores)),
            max_f1=float(onp.max(f1_scores)),
            median_f1=float(onp.median(f1_scores))
        )
        
        summaries[key] = summary
    
    logger.info(f"Computed summaries for {len(summaries)} experiment groups")
    return summaries


def compare_methods(
    results_a: List[ExperimentResult],
    results_b: List[ExperimentResult],
    method_name_a: str,
    method_name_b: str,
    metric: str = 'f1_score'
) -> ComparisonResult:
    """
    Compare two methods statistically.
    
    Args:
        results_a: Results from method A
        results_b: Results from method B
        method_name_a: Name of method A
        method_name_b: Name of method B
        metric: Metric to compare ('f1_score', 'target_improvement', 'sample_efficiency')
        
    Returns:
        ComparisonResult with statistical comparison
    """
    # Extract metric values
    values_a = _extract_metric_values(results_a, metric)
    values_b = _extract_metric_values(results_b, metric)
    
    if not values_a or not values_b:
        logger.warning(f"Insufficient data for comparison: {len(values_a)} vs {len(values_b)} samples")
        return ComparisonResult(
            method_a=method_name_a,
            method_b=method_name_b,
            metric_name=metric,
            mean_a=0.0,
            mean_b=0.0,
            improvement_ratio=1.0,
            improvement_absolute=0.0,
            n_samples_a=len(values_a),
            n_samples_b=len(values_b)
        )
    
    # Compute basic statistics
    mean_a = float(onp.mean(values_a))
    mean_b = float(onp.mean(values_b))
    improvement_absolute = mean_b - mean_a
    improvement_ratio = mean_b / mean_a if mean_a != 0 else float('inf')
    
    # Statistical significance testing
    p_value = None
    is_significant = None
    confidence_interval = None
    
    if SCIPY_AVAILABLE and len(values_a) >= 3 and len(values_b) >= 3:
        try:
            # Welch's t-test (unequal variances)
            statistic, p_value = stats.ttest_ind(values_b, values_a, equal_var=False)
            is_significant = p_value < 0.05
            
            # 95% confidence interval for the difference
            pooled_se = onp.sqrt(onp.var(values_a)/len(values_a) + onp.var(values_b)/len(values_b))
            df = len(values_a) + len(values_b) - 2
            t_critical = stats.t.ppf(0.975, df)
            margin_error = t_critical * pooled_se
            confidence_interval = (improvement_absolute - margin_error, improvement_absolute + margin_error)
            
        except Exception as e:
            logger.warning(f"Statistical test failed: {e}")
    
    return ComparisonResult(
        method_a=method_name_a,
        method_b=method_name_b,
        metric_name=metric,
        mean_a=mean_a,
        mean_b=mean_b,
        improvement_ratio=improvement_ratio,
        improvement_absolute=improvement_absolute,
        p_value=p_value,
        is_significant=is_significant,
        confidence_interval=confidence_interval,
        n_samples_a=len(values_a),
        n_samples_b=len(values_b)
    )


def _extract_metric_values(results: List[ExperimentResult], metric: str) -> List[float]:
    """
    Extract metric values from experiment results.
    
    Args:
        results: List of experiment results
        metric: Metric name to extract
        
    Returns:
        List of metric values
    """
    values = []
    
    for result in results:
        if not result.success:
            continue
        
        if metric == 'f1_score':
            values.append(result.final_f1_score)
        elif metric == 'target_improvement':
            values.append(result.target_optimization_improvement)
        elif metric == 'sample_efficiency':
            values.append(result.sample_efficiency)
        elif metric == 'time_seconds':
            values.append(result.total_time_seconds)
        elif metric == 'structure_accuracy':
            values.append(result.structure_recovery_accuracy)
        else:
            logger.warning(f"Unknown metric: {metric}")
    
    return values


def generate_performance_table(summaries: Dict[str, ExperimentSummary]) -> Dict[str, Any]:
    """
    Generate a performance comparison table.
    
    Args:
        summaries: Dictionary of experiment summaries
        
    Returns:
        Dictionary containing table data suitable for export
    """
    table_data = {
        'headers': ['Method', 'Dataset', 'F1 Score', 'Target Improvement', 'Sample Efficiency', 'Time (s)'],
        'rows': []
    }
    
    for key, summary in summaries.items():
        row = [
            summary.experiment_type,
            summary.dataset_name,
            f"{summary.mean_f1_score:.3f} ± {summary.std_f1_score:.3f}",
            f"{summary.mean_target_improvement:.3f} ± {summary.std_target_improvement:.3f}",
            f"{summary.mean_sample_efficiency:.4f} ± {summary.std_sample_efficiency:.4f}",
            f"{summary.mean_time_seconds:.1f} ± {summary.std_time_seconds:.1f}"
        ]
        table_data['rows'].append(row)
    
    # Sort by F1 score descending
    table_data['rows'].sort(key=lambda x: float(x[2].split(' ±')[0]), reverse=True)
    
    return table_data


def analyze_scaling_performance(results: List[ExperimentResult]) -> Dict[str, Any]:
    """
    Analyze how performance scales with graph size.
    
    Args:
        results: List of experiment results
        
    Returns:
        Dictionary with scaling analysis
    """
    # Group by dataset size (approximate from dataset name)
    size_groups = {}
    
    for result in results:
        if not result.success:
            continue
        
        dataset_name = result.experiment_config.get('dataset_name', '')
        dataset_summary = result.dataset_summary
        
        # Extract size information
        n_variables = dataset_summary.get('n_variables', 0)
        if n_variables == 0:
            continue
        
        # Group by size ranges
        if n_variables <= 10:
            size_category = 'small'
        elif n_variables <= 30:
            size_category = 'medium'
        else:
            size_category = 'large'
        
        if size_category not in size_groups:
            size_groups[size_category] = []
        size_groups[size_category].append(result)
    
    # Compute scaling statistics
    scaling_analysis = {}
    
    for size_category, group_results in size_groups.items():
        f1_scores = [r.final_f1_score for r in group_results]
        times = [r.total_time_seconds for r in group_results]
        
        scaling_analysis[size_category] = {
            'n_experiments': len(group_results),
            'mean_f1': float(onp.mean(f1_scores)),
            'std_f1': float(onp.std(f1_scores)),
            'mean_time': float(onp.mean(times)),
            'std_time': float(onp.std(times)),
            'avg_graph_size': float(onp.mean([r.dataset_summary.get('n_variables', 0) for r in group_results]))
        }
    
    return scaling_analysis


def create_learning_curve_data(results: List[ExperimentResult]) -> Dict[str, Any]:
    """
    Extract learning curve data for visualization.
    
    Args:
        results: List of experiment results with learning trajectories
        
    Returns:
        Dictionary with learning curve data
    """
    learning_curves = {}
    
    for result in results:
        if not result.success or not result.learning_trajectory:
            continue
        
        exp_type = result.experiment_config.get('experiment_type', 'unknown')
        dataset = result.experiment_config.get('dataset_name', 'unknown')
        key = f"{exp_type}_{dataset}"
        
        if key not in learning_curves:
            learning_curves[key] = []
        
        # Extract trajectory data
        trajectory = []
        for step_data in result.learning_trajectory:
            trajectory.append({
                'step': step_data.get('step', 0),
                'target_value': step_data.get('outcome_value', 0.0),
                'loss': step_data.get('loss', 0.0),
                'uncertainty': step_data.get('uncertainty', 0.0)
            })
        
        learning_curves[key].append(trajectory)
    
    return learning_curves


def export_results_for_plotting(
    results: List[ExperimentResult],
    output_dir: str = "analysis_output"
) -> None:
    """
    Export analysis results in formats suitable for external plotting tools.
    
    Args:
        results: List of experiment results
        output_dir: Directory to save analysis outputs
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Compute summaries
    summaries = compute_experiment_summary(results)
    
    # Export performance table
    performance_table = generate_performance_table(summaries)
    with open(output_path / "performance_table.json", 'w') as f:
        json.dump(performance_table, f, indent=2)
    
    # Export scaling analysis
    scaling_analysis = analyze_scaling_performance(results)
    with open(output_path / "scaling_analysis.json", 'w') as f:
        json.dump(scaling_analysis, f, indent=2)
    
    # Export learning curves
    learning_curves = create_learning_curve_data(results)
    with open(output_path / "learning_curves.json", 'w') as f:
        json.dump(learning_curves, f, indent=2)
    
    # Export raw summaries
    summaries_dict = {k: asdict(v) for k, v in summaries.items()}
    with open(output_path / "experiment_summaries.json", 'w') as f:
        json.dump(summaries_dict, f, indent=2)
    
    logger.info(f"Exported analysis results to {output_path}")


def validate_experimental_hypotheses(results: List[ExperimentResult]) -> Dict[str, Any]:
    """
    Validate key experimental hypotheses from the validation plan.
    
    Args:
        results: List of experiment results
        
    Returns:
        Dictionary with hypothesis validation results
    """
    # Group results by experiment type
    type_groups = {}
    for result in results:
        if not result.success:
            continue
        
        exp_type = result.experiment_config.get('experiment_type', 'unknown')
        if exp_type not in type_groups:
            type_groups[exp_type] = []
        type_groups[exp_type].append(result)
    
    validation_results = {
        'hypotheses_tested': [],
        'success_criteria': {},
        'overall_validation': True
    }
    
    # Hypothesis 1: Trained models outperform untrained
    if 'untrained_baseline' in type_groups and 'fully_trained' in type_groups:
        comparison = compare_methods(
            type_groups['untrained_baseline'],
            type_groups['fully_trained'],
            'Untrained',
            'Fully Trained',
            'f1_score'
        )
        
        improvement_threshold = 0.1  # 10% improvement required
        hypothesis_1_success = comparison.improvement_absolute > improvement_threshold
        
        validation_results['hypotheses_tested'].append('trained_vs_untrained')
        validation_results['success_criteria']['trained_vs_untrained'] = {
            'success': hypothesis_1_success,
            'improvement': comparison.improvement_absolute,
            'threshold': improvement_threshold,
            'p_value': comparison.p_value
        }
        
        if not hypothesis_1_success:
            validation_results['overall_validation'] = False
    
    # Hypothesis 2: F1 score > 0.7 on standard datasets
    f1_threshold = 0.7
    high_f1_results = [r for r in results if r.success and r.final_f1_score > f1_threshold]
    
    hypothesis_2_success = len(high_f1_results) > 0
    validation_results['hypotheses_tested'].append('f1_threshold')
    validation_results['success_criteria']['f1_threshold'] = {
        'success': hypothesis_2_success,
        'threshold': f1_threshold,
        'n_passing': len(high_f1_results),
        'n_total': len([r for r in results if r.success]),
        'best_f1': max([r.final_f1_score for r in results if r.success], default=0.0)
    }
    
    if not hypothesis_2_success:
        validation_results['overall_validation'] = False
    
    # Hypothesis 3: Sample efficiency improvement
    efficiency_threshold = 2.0  # 2x improvement required
    baseline_efficiency = 0.01  # Assumed baseline
    
    efficient_results = [r for r in results if r.success and r.sample_efficiency > baseline_efficiency * efficiency_threshold]
    
    hypothesis_3_success = len(efficient_results) > 0
    validation_results['hypotheses_tested'].append('sample_efficiency')
    validation_results['success_criteria']['sample_efficiency'] = {
        'success': hypothesis_3_success,
        'threshold_multiplier': efficiency_threshold,
        'n_passing': len(efficient_results),
        'best_efficiency': max([r.sample_efficiency for r in results if r.success], default=0.0)
    }
    
    if not hypothesis_3_success:
        validation_results['overall_validation'] = False
    
    return validation_results


def generate_validation_report(results: List[ExperimentResult], output_file: str = "validation_report.json") -> None:
    """
    Generate a comprehensive validation report.
    
    Args:
        results: List of experiment results
        output_file: Output file path
    """
    # Compute all analyses
    summaries = compute_experiment_summary(results)
    scaling_analysis = analyze_scaling_performance(results)
    hypothesis_validation = validate_experimental_hypotheses(results)
    
    # Generate report
    report = {
        'validation_summary': {
            'total_experiments': len(results),
            'successful_experiments': len([r for r in results if r.success]),
            'overall_validation_passed': hypothesis_validation['overall_validation'],
            'timestamp': onp.datetime64('now').astype(str)
        },
        'experiment_summaries': {k: asdict(v) for k, v in summaries.items()},
        'scaling_analysis': scaling_analysis,
        'hypothesis_validation': hypothesis_validation,
        'performance_table': generate_performance_table(summaries)
    }
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Generated validation report: {output_file}")
    
    # Print summary
    print(f"\n🔬 ACBO Experimental Validation Report")
    print(f"=" * 50)
    print(f"Total experiments: {report['validation_summary']['total_experiments']}")
    print(f"Successful experiments: {report['validation_summary']['successful_experiments']}")
    print(f"Overall validation: {'✅ PASSED' if report['validation_summary']['overall_validation_passed'] else '❌ FAILED'}")
    
    if hypothesis_validation['success_criteria']:
        print(f"\nHypothesis Validation:")
        for hypothesis, criteria in hypothesis_validation['success_criteria'].items():
            status = "✅" if criteria['success'] else "❌"
            print(f"  {status} {hypothesis}: {criteria.get('improvement', criteria.get('best_f1', 'N/A'))}")


# Export public interface
__all__ = [
    'ComparisonResult',
    'ExperimentSummary',
    'load_experiment_results',
    'compute_experiment_summary',
    'compare_methods',
    'generate_performance_table',
    'analyze_scaling_performance',
    'create_learning_curve_data',
    'export_results_for_plotting',
    'validate_experimental_hypotheses',
    'generate_validation_report'
]