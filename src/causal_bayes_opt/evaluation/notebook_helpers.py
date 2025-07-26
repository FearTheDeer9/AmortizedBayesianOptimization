"""
Notebook Helper Functions

Provides convenient functions for using the unified evaluation framework
in Jupyter notebooks. These helpers simplify common evaluation workflows.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np

from .unified_runner import UnifiedEvaluationRunner, MethodRegistry
from .grpo_evaluator import GRPOEvaluator
from .bc_evaluator import BCEvaluator
from .baseline_evaluators import (
    RandomBaselineEvaluator,
    OracleBaselineEvaluator,
    LearningBaselineEvaluator
)
from .result_types import ComparisonResults, ExperimentResult

logger = logging.getLogger(__name__)


def setup_evaluation_runner(methods: List[str], 
                          checkpoint_paths: Optional[Dict[str, Union[str, Path]]] = None,
                          parallel: bool = True,
                          output_dir: Optional[Union[str, Path]] = None) -> UnifiedEvaluationRunner:
    """
    Set up unified evaluation runner with specified methods.
    
    Args:
        methods: List of method names to include. Options:
            - "random": Random baseline
            - "oracle": Oracle baseline
            - "learning": Learning baseline
            - "grpo": GRPO method (requires checkpoint)
            - "bc_surrogate": BC with surrogate only
            - "bc_acquisition": BC with acquisition only
            - "bc_both": BC with both surrogate and acquisition
        checkpoint_paths: Dict mapping method names to checkpoint paths
        parallel: Whether to run evaluations in parallel
        output_dir: Optional output directory for results
        
    Returns:
        Configured UnifiedEvaluationRunner
    """
    # Use default output directory if not specified
    if output_dir is None:
        output_dir = Path("evaluation_results")
    
    runner = UnifiedEvaluationRunner(output_dir=output_dir, parallel=parallel)
    
    # Register requested methods
    for method in methods:
        if method == "random":
            evaluator = RandomBaselineEvaluator()
            runner.register_method(evaluator)
            
        elif method == "oracle":
            evaluator = OracleBaselineEvaluator()
            runner.register_method(evaluator)
            
        elif method == "learning":
            evaluator = LearningBaselineEvaluator()
            runner.register_method(evaluator)
            
        elif method == "grpo":
            if not checkpoint_paths or "grpo" not in checkpoint_paths:
                raise ValueError(f"GRPO method requires checkpoint path")
            evaluator = GRPOEvaluator(Path(checkpoint_paths["grpo"]))
            runner.register_method(evaluator)
            
        elif method == "bc_surrogate":
            if not checkpoint_paths or "bc_surrogate" not in checkpoint_paths:
                raise ValueError(f"BC surrogate method requires checkpoint path")
            evaluator = BCEvaluator(
                surrogate_checkpoint=Path(checkpoint_paths["bc_surrogate"]),
                name="BC_Surrogate_Random"
            )
            runner.register_method(evaluator)
            
        elif method == "bc_acquisition":
            if not checkpoint_paths or "bc_acquisition" not in checkpoint_paths:
                raise ValueError(f"BC acquisition method requires checkpoint path")
            evaluator = BCEvaluator(
                acquisition_checkpoint=Path(checkpoint_paths["bc_acquisition"]),
                name="BC_Acquisition_Learning"
            )
            runner.register_method(evaluator)
            
        elif method == "bc_both":
            if not checkpoint_paths:
                raise ValueError(f"BC both method requires checkpoint paths")
            if "bc_surrogate" not in checkpoint_paths or "bc_acquisition" not in checkpoint_paths:
                raise ValueError(f"BC both method requires both surrogate and acquisition checkpoints")
            evaluator = BCEvaluator(
                surrogate_checkpoint=Path(checkpoint_paths["bc_surrogate"]),
                acquisition_checkpoint=Path(checkpoint_paths["bc_acquisition"]),
                name="BC_Both"
            )
            runner.register_method(evaluator)
            
        else:
            raise ValueError(f"Unknown method: {method}")
    
    logger.info(f"Registered {len(methods)} evaluation methods")
    return runner


def run_evaluation_comparison(
    runner: UnifiedEvaluationRunner,
    test_scms: List[Any],
    config: Dict[str, Any],
    n_seeds: int = 5,
    output_dir: Optional[Path] = None
) -> ComparisonResults:
    """
    Run evaluation comparison across multiple SCMs and seeds.
    
    Args:
        runner: Configured evaluation runner
        test_scms: List of SCMs to evaluate on
        config: Evaluation configuration
        n_seeds: Number of random seeds per SCM
        output_dir: Optional directory to save results
        
    Returns:
        ComparisonResults object
    """
    logger.info(f"Running evaluation on {len(test_scms)} SCMs with {n_seeds} seeds each")
    
    results = runner.run_comparison(
        test_scms=test_scms,
        config=config,
        n_runs_per_scm=n_seeds,
        base_seed=42
    )
    
    logger.info("Evaluation complete")
    return results


def results_to_dataframe(results: ComparisonResults) -> pd.DataFrame:
    """
    Convert comparison results to pandas DataFrame for analysis.
    
    Args:
        results: ComparisonResults from evaluation
        
    Returns:
        DataFrame with one row per method containing aggregated metrics
    """
    rows = []
    
    for method_name, method_metrics in results.method_metrics.items():
        row = {
            'method': method_name,
            'mean_improvement': method_metrics.mean_improvement,
            'std_improvement': method_metrics.std_improvement,
            'mean_final_value': method_metrics.mean_final_value,
            'std_final_value': method_metrics.std_final_value,
            'success_rate': method_metrics.success_rate,
            'mean_time': method_metrics.mean_time,
            'n_successful': method_metrics.n_successful,
            'n_runs': method_metrics.n_runs
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('mean_improvement', ascending=False)
    return df


def plot_learning_curves(results: ComparisonResults, 
                        scm_idx: int = 0,
                        metric: str = 'outcome_value',
                        title: Optional[str] = None) -> None:
    """
    Plot learning curves for all methods on a specific SCM.
    
    Args:
        results: ComparisonResults from evaluation
        scm_idx: Index of SCM to plot (default: 0)
        metric: Metric to plot ('outcome_value', 'f1_score', 'shd')
        title: Optional plot title
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    
    if results.raw_results is None:
        logger.warning("No raw results available for plotting learning curves")
        return
    
    for method_name, method_results in results.raw_results.items():
        # Get results for specific SCM (aggregate over seeds)
        scm_results = [r for r in method_results if r.metadata.get('scm_idx', 0) == scm_idx]
        
        if not scm_results:
            continue
            
        # Extract learning curves
        all_curves = []
        for result in scm_results:
            if metric == 'outcome_value':
                curve = [step.outcome_value for step in result.learning_history]
            elif metric == 'f1_score':
                # Compute F1 scores from marginals
                true_parents = result.metadata['scm_info']['true_parents']
                curve = []
                for step in result.learning_history:
                    from ..analysis.trajectory_metrics import compute_f1_score_from_marginals
                    f1 = compute_f1_score_from_marginals(step.marginals, true_parents)
                    curve.append(f1)
            elif metric == 'shd':
                # Compute SHD from marginals
                true_parents = result.metadata['scm_info']['true_parents']
                curve = []
                for step in result.learning_history:
                    from ..analysis.trajectory_metrics import compute_shd_from_marginals
                    shd = compute_shd_from_marginals(step.marginals, true_parents)
                    curve.append(shd)
            else:
                raise ValueError(f"Unknown metric: {metric}")
                
            all_curves.append(curve)
        
        # Average over seeds
        if all_curves:
            mean_curve = np.mean(all_curves, axis=0)
            std_curve = np.std(all_curves, axis=0)
            steps = np.arange(len(mean_curve))
            
            plt.plot(steps, mean_curve, label=method_name, linewidth=2)
            plt.fill_between(steps, 
                           mean_curve - std_curve,
                           mean_curve + std_curve,
                           alpha=0.2)
    
    plt.xlabel('Step')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(title or f'{metric.replace("_", " ").title()} Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_summary_report(results: ComparisonResults, 
                         save_path: Optional[Path] = None) -> str:
    """
    Create a text summary report of evaluation results.
    
    Args:
        results: ComparisonResults from evaluation
        save_path: Optional path to save report
        
    Returns:
        Summary report as string
    """
    report = []
    report.append("=" * 80)
    report.append("EVALUATION SUMMARY REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Configuration summary
    report.append("Configuration:")
    report.append(f"  Number of SCMs: {results.config.get('n_scms', 'unknown')}")
    report.append(f"  Seeds per SCM: {results.config.get('n_seeds', 'unknown')}")
    report.append(f"  Max interventions: {results.config.get('experiment', {}).get('target', {}).get('max_interventions', 'unknown')}")
    report.append("")
    
    # Method performance summary
    report.append("Method Performance Summary:")
    report.append("-" * 80)
    
    # Sort methods by mean improvement
    sorted_methods = sorted(
        results.method_metrics.items(),
        key=lambda x: x[1].mean_improvement,
        reverse=True
    )
    
    for method_name, metrics in sorted_methods:
        report.append(f"\n{method_name}:")
        report.append(f"  Target Improvement: {metrics.mean_improvement:.3f} ± {metrics.std_improvement:.3f}")
        report.append(f"  Final Value: {metrics.mean_final_value:.3f} ± {metrics.std_final_value:.3f}")
        report.append(f"  Success Rate: {metrics.success_rate:.1%}")
        report.append(f"  Successful Runs: {metrics.n_successful}/{metrics.n_runs}")
        report.append(f"  Mean Time: {metrics.mean_time:.2f}s")
    
    # Statistical comparisons
    if results.statistical_tests is not None and results.statistical_tests:
        report.append("\n" + "=" * 80)
        report.append("Statistical Comparisons (vs best baseline):")
        report.append("-" * 80)
        
        for test_name, test_result in results.statistical_tests.items():
            report.append(f"\n{test_name}:")
            report.append(f"  Statistic: {test_result.get('statistic', 'N/A'):.3f}")
            report.append(f"  P-value: {test_result.get('p_value', 'N/A'):.4f}")
            report.append(f"  Significant: {'Yes' if test_result.get('p_value', 1.0) < 0.05 else 'No'}")
    
    report.append("\n" + "=" * 80)
    
    report_text = "\n".join(report)
    
    if save_path:
        save_path.write_text(report_text)
        logger.info(f"Report saved to {save_path}")
    
    return report_text


def load_and_visualize_results(results_path: Path) -> ComparisonResults:
    """
    Load saved results and create visualizations.
    
    Args:
        results_path: Path to saved results pickle file
        
    Returns:
        Loaded ComparisonResults
    """
    import pickle
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    # Create DataFrame summary
    df = results_to_dataframe(results)
    print("Method Performance Summary:")
    print(df.to_string(index=False))
    print()
    
    # Plot learning curves for first SCM
    plot_learning_curves(results, scm_idx=0, metric='outcome_value')
    
    # Print summary report
    report = create_summary_report(results)
    print(report)
    
    return results


def compute_f1_shd_trajectories(
    experiment_results: List[ExperimentResult],
    true_parents: List[str]
) -> Dict[str, List[float]]:
    """
    Compute F1 and SHD trajectories for experiment results.
    
    This helper function handles the computation of structure learning metrics
    for experiments that might not have them pre-computed.
    
    Args:
        experiment_results: List of experiment results to process
        true_parents: List of true parent variable names
        
    Returns:
        Dictionary with 'f1_scores', 'shd_values', 'steps' lists
    """
    from ..analysis.trajectory_metrics import (
        compute_f1_score_from_marginals,
        compute_shd_from_marginals
    )
    
    all_f1_scores = []
    all_shd_values = []
    all_steps = []
    
    for result in experiment_results:
        if not result.learning_history:
            continue
            
        f1_scores = []
        shd_values = []
        steps = []
        
        for step_result in result.learning_history:
            steps.append(step_result.step)
            
            # Check if this method predicts structure (has marginals)
            if step_result.marginals:
                f1 = compute_f1_score_from_marginals(
                    step_result.marginals, true_parents
                )
                shd = compute_shd_from_marginals(
                    step_result.marginals, true_parents
                )
                f1_scores.append(f1)
                shd_values.append(shd)
            else:
                # No structure prediction (e.g., Random baseline)
                f1_scores.append(0.0)
                shd_values.append(len(true_parents))  # Worst case
        
        all_f1_scores.append(f1_scores)
        all_shd_values.append(shd_values)
        all_steps.append(steps)
    
    # Average across runs if multiple
    if all_f1_scores:
        # Find max length
        max_len = max(len(scores) for scores in all_f1_scores)
        
        # Pad trajectories
        padded_f1 = []
        padded_shd = []
        for f1_traj, shd_traj in zip(all_f1_scores, all_shd_values):
            # Pad with last value
            if len(f1_traj) < max_len:
                f1_traj = f1_traj + [f1_traj[-1]] * (max_len - len(f1_traj))
                shd_traj = shd_traj + [shd_traj[-1]] * (max_len - len(shd_traj))
            padded_f1.append(f1_traj[:max_len])
            padded_shd.append(shd_traj[:max_len])
        
        # Compute mean trajectories
        mean_f1 = np.mean(padded_f1, axis=0).tolist()
        mean_shd = np.mean(padded_shd, axis=0).tolist()
        steps = list(range(max_len))
        
        return {
            'f1_scores': mean_f1,
            'shd_values': mean_shd,
            'steps': steps
        }
    else:
        return {
            'f1_scores': [],
            'shd_values': [],
            'steps': []
        }


def add_true_parents_to_results(
    results: ComparisonResults,
    scm_true_parents: Dict[int, List[str]]
) -> None:
    """
    Add true parent information to experiment results retroactively.
    
    This is useful when results were generated without true_parents in metadata.
    
    Args:
        results: ComparisonResults object to update
        scm_true_parents: Dict mapping SCM index to list of true parent names
    """
    if not results.raw_results:
        logger.warning("No raw results available to update")
        return
        
    for method_name, method_results in results.raw_results.items():
        for i, exp_result in enumerate(method_results):
            # Determine which SCM this result is from
            scm_idx = i // (len(method_results) // len(scm_true_parents))
            
            # Add true_parents to metadata
            if scm_idx in scm_true_parents:
                exp_result.metadata['true_parents'] = scm_true_parents[scm_idx]
    
    logger.info(f"Added true_parents to {len(results.raw_results)} methods' results")