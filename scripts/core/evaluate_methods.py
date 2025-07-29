#!/usr/bin/env python3
"""
Unified Evaluation Script

Evaluates trained models (GRPO, BC) against baselines on test SCMs.
Calculates comprehensive metrics including F1, SHD, and target trajectories.
"""

import argparse
import logging
import time
from pathlib import Path
import sys
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.evaluation import run_evaluation
from src.causal_bayes_opt.evaluation.result_types import ComparisonResults
from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
from scripts.core.utils.checkpoint_utils import CheckpointManager, extract_model_info
from scripts.core.utils.metric_utils import format_metrics_table

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_evaluation_config(
    max_interventions: int = 20,
    n_observational_samples: int = 100,
    n_test_scms: int = 10,
    n_runs_per_scm: int = 3,
    optimization_direction: str = "MINIMIZE"
) -> Dict[str, Any]:
    """Create evaluation configuration."""
    
    config = {
        "experiment": {
            "target": {
                "max_interventions": max_interventions,
                "n_observational_samples": n_observational_samples,
                "intervention_value_range": (-2.0, 2.0),
                "optimization_direction": optimization_direction,
                "learning_rate": 1e-3
            },
            "runs_per_method": n_runs_per_scm,
            "scm_generation": {
                "use_variable_factory": True,
                "variable_range": [3, 6],
                "structure_types": ["fork", "chain", "collider", "mixed"]
            }
        },
        "n_scms": n_test_scms,
        "n_seeds": n_runs_per_scm,
        "parallel": False,  # More stable for debugging
        "use_fixed_grpo_evaluator": True,  # Use fixed evaluator with surrogate
        "visualization": {
            "enabled": True,
            "plot_types": ["target_trajectory", "f1_trajectory", "shd_trajectory", "method_comparison"]
        }
    }
    
    return config


def generate_test_scms(n_scms: int, seed: int = 100) -> List[Any]:
    """Generate diverse test SCMs."""
    factory = VariableSCMFactory(
        noise_scale=0.5,
        coefficient_range=(-2.0, 2.0),
        seed=seed  # Different from training seed
    )
    
    scms = []
    structure_types = ["fork", "chain", "collider", "mixed"]
    variable_counts = [3, 4, 5, 6]
    
    # Generate balanced set
    scm_idx = 0
    while len(scms) < n_scms:
        for structure_type in structure_types:
            for n_vars in variable_counts:
                if len(scms) >= n_scms:
                    break
                    
                scm = factory.create_variable_scm(
                    num_variables=n_vars,
                    structure_type=structure_type,
                    target_variable=None
                )
                scms.append(scm)
                scm_idx += 1
    
    logger.info(f"Generated {len(scms)} test SCMs")
    return scms


def evaluate_methods(
    checkpoint_paths: Dict[str, Path],
    config: Dict[str, Any],
    output_dir: Path,
    test_scms: Optional[List[Any]] = None
) -> ComparisonResults:
    """
    Evaluate methods with proper checkpoint loading.
    
    Args:
        checkpoint_paths: Dictionary mapping method -> checkpoint path
        config: Evaluation configuration
        output_dir: Output directory
        test_scms: Optional pre-generated test SCMs
        
    Returns:
        ComparisonResults object
    """
    # Determine methods to evaluate
    methods = []
    
    # Always include baselines
    methods.extend(["random", "learning", "oracle"])
    
    # Add checkpoint-based methods
    if "grpo" in checkpoint_paths:
        methods.append("grpo")
    if "bc_surrogate" in checkpoint_paths:
        methods.append("bc_surrogate")
    if "bc_acquisition" in checkpoint_paths:
        methods.append("bc_acquisition")
    if "bc_surrogate" in checkpoint_paths and "bc_acquisition" in checkpoint_paths:
        methods.append("bc_both")
    
    logger.info(f"Methods to evaluate: {methods}")
    
    # Prepare checkpoint paths for evaluation
    eval_checkpoint_paths = {}
    
    if "grpo" in checkpoint_paths:
        eval_checkpoint_paths["grpo"] = checkpoint_paths["grpo"]
        
    if "bc_surrogate" in checkpoint_paths:
        eval_checkpoint_paths["bc_surrogate"] = checkpoint_paths["bc_surrogate"]
        config["bc_surrogate_checkpoint"] = str(checkpoint_paths["bc_surrogate"])
        
    if "bc_acquisition" in checkpoint_paths:
        eval_checkpoint_paths["bc_acquisition"] = checkpoint_paths["bc_acquisition"]
        config["bc_acquisition_checkpoint"] = str(checkpoint_paths["bc_acquisition"])
    
    # Run evaluation using unified framework
    logger.info("Running unified evaluation...")
    
    results = run_evaluation(
        checkpoint_path=eval_checkpoint_paths.get("grpo"),  # Primary checkpoint
        output_dir=output_dir,
        config=config,
        test_scms=test_scms,
        methods=methods
    )
    
    return results


def generate_visualizations(
    results: ComparisonResults,
    output_dir: Path
) -> Dict[str, Path]:
    """Generate comprehensive visualizations."""
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    plot_paths = {}
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Method comparison bar plot
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        method_names = []
        final_values = []
        std_values = []
        
        for method_name, metrics in results.method_metrics.items():
            method_names.append(method_name)
            final_values.append(metrics.mean_final_value)
            std_values.append(metrics.std_final_value)
        
        x_pos = np.arange(len(method_names))
        bars = ax.bar(x_pos, final_values, yerr=std_values, capsize=5, alpha=0.8)
        
        # Color code by method type
        colors = []
        for name in method_names:
            if "grpo" in name.lower():
                colors.append("blue")
            elif "bc" in name.lower():
                colors.append("green")
            elif "oracle" in name.lower():
                colors.append("gold")
            else:
                colors.append("gray")
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel("Method")
        ax.set_ylabel("Final Target Value")
        ax.set_title("Method Performance Comparison")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        path = plots_dir / "method_comparison.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths["method_comparison"] = path
        
    except Exception as e:
        logger.warning(f"Failed to create method comparison plot: {e}")
    
    # 2. Learning curves (if trajectory data available)
    try:
        curves_data = results.get_learning_curves()
        
        if curves_data:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Target value trajectories
            ax = axes[0]
            for method_name, data in curves_data.items():
                if 'target_mean' in data and data['target_mean']:
                    steps = data.get('steps', list(range(len(data['target_mean']))))
                    ax.plot(steps, data['target_mean'], label=method_name, marker='o', markersize=4)
                    if 'target_std' in data:
                        ax.fill_between(steps,
                                      np.array(data['target_mean']) - np.array(data['target_std']),
                                      np.array(data['target_mean']) + np.array(data['target_std']),
                                      alpha=0.2)
            
            ax.set_xlabel('Intervention Step')
            ax.set_ylabel('Target Value')
            ax.set_title('Target Value Optimization')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # F1 score trajectories
            ax = axes[1]
            has_f1 = False
            for method_name, data in curves_data.items():
                if 'f1_mean' in data and data['f1_mean']:
                    has_f1 = True
                    steps = data.get('steps', list(range(len(data['f1_mean']))))
                    ax.plot(steps, data['f1_mean'], label=method_name, marker='o', markersize=4)
                    if 'f1_std' in data:
                        ax.fill_between(steps,
                                      np.array(data['f1_mean']) - np.array(data['f1_std']),
                                      np.array(data['f1_mean']) + np.array(data['f1_std']),
                                      alpha=0.2)
            
            if has_f1:
                ax.set_xlabel('Intervention Step')
                ax.set_ylabel('F1 Score')
                ax.set_title('Structure Learning (F1)')
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-0.1, 1.1)
            else:
                ax.text(0.5, 0.5, 'F1 data not available', ha='center', va='center')
                ax.set_title('Structure Learning (F1)')
            
            # SHD trajectories
            ax = axes[2]
            has_shd = False
            for method_name, data in curves_data.items():
                if 'shd_mean' in data and data['shd_mean']:
                    has_shd = True
                    steps = data.get('steps', list(range(len(data['shd_mean']))))
                    ax.plot(steps, data['shd_mean'], label=method_name, marker='o', markersize=4)
                    if 'shd_std' in data:
                        ax.fill_between(steps,
                                      np.array(data['shd_mean']) - np.array(data['shd_std']),
                                      np.array(data['shd_mean']) + np.array(data['shd_std']),
                                      alpha=0.2)
            
            if has_shd:
                ax.set_xlabel('Intervention Step')
                ax.set_ylabel('SHD')
                ax.set_title('Structural Hamming Distance')
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'SHD data not available', ha='center', va='center')
                ax.set_title('Structural Hamming Distance')
            
            plt.tight_layout()
            path = plots_dir / "learning_curves.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths["learning_curves"] = path
            
    except Exception as e:
        logger.warning(f"Failed to create learning curves: {e}")
    
    # 3. Performance distribution
    try:
        # Extract raw performance data
        fig, ax = plt.subplots(figsize=(10, 6))
        
        all_data = []
        labels = []
        
        for method_name, metrics in results.method_metrics.items():
            # Try to get individual run data from raw results
            if hasattr(results, 'raw_results') and method_name in results.raw_results:
                method_runs = results.raw_results[method_name]
                run_values = [r.final_target_value for r in method_runs if hasattr(r, 'final_target_value')]
                if run_values:
                    all_data.append(run_values)
                    labels.append(method_name)
        
        if all_data:
            box_plot = ax.boxplot(all_data, labels=labels, patch_artist=True)
            
            # Color boxes
            colors = []
            for label in labels:
                if "grpo" in label.lower():
                    colors.append("lightblue")
                elif "bc" in label.lower():
                    colors.append("lightgreen")
                elif "oracle" in label.lower():
                    colors.append("gold")
                else:
                    colors.append("lightgray")
            
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel('Final Target Value')
            ax.set_title('Performance Distribution Across Runs')
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            path = plots_dir / "performance_distribution.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths["performance_distribution"] = path
            
    except Exception as e:
        logger.warning(f"Failed to create performance distribution: {e}")
    
    logger.info(f"Generated {len(plot_paths)} visualizations")
    return plot_paths


def create_summary_report(
    results: ComparisonResults,
    checkpoint_info: Dict[str, Any],
    output_dir: Path
) -> str:
    """Create comprehensive summary report."""
    
    lines = []
    lines.append("="*60)
    lines.append("ACBO EVALUATION SUMMARY")
    lines.append("="*60)
    lines.append("")
    
    # Configuration summary
    lines.append("Evaluation Configuration:")
    lines.append(f"  Test SCMs: {results.config.get('n_scms', 'N/A')}")
    lines.append(f"  Runs per SCM: {results.config.get('n_seeds', 'N/A')}")
    lines.append(f"  Max interventions: {results.config.get('experiment', {}).get('target', {}).get('max_interventions', 'N/A')}")
    lines.append(f"  Optimization: {results.config.get('experiment', {}).get('target', {}).get('optimization_direction', 'N/A')}")
    lines.append("")
    
    # Checkpoint information
    lines.append("Checkpoints Used:")
    for method, info in checkpoint_info.items():
        lines.append(f"  {method}: {info.get('path', 'N/A')}")
        if info.get('model_type'):
            lines.append(f"    Type: {info['model_type']}")
        if info.get('optimization_direction'):
            lines.append(f"    Optimization: {info['optimization_direction']}")
    lines.append("")
    
    # Performance summary
    lines.append("Performance Summary:")
    lines.append("-"*60)
    
    # Sort methods by performance
    sorted_methods = sorted(
        results.method_metrics.items(),
        key=lambda x: x[1].mean_final_value,
        reverse=(results.config.get('experiment', {}).get('target', {}).get('optimization_direction') == "MAXIMIZE")
    )
    
    for rank, (method_name, metrics) in enumerate(sorted_methods, 1):
        lines.append(f"{rank}. {method_name}")
        lines.append(f"   Final value: {metrics.mean_final_value:.4f} ± {metrics.std_final_value:.4f}")
        lines.append(f"   Improvement: {metrics.mean_improvement:.4f} ± {metrics.std_improvement:.4f}")
        lines.append(f"   Success rate: {metrics.n_successful}/{metrics.n_runs} ({metrics.n_successful/metrics.n_runs*100:.1f}%)")
        lines.append("")
    
    # Statistical significance (if available)
    if hasattr(results, 'statistical_tests') and results.statistical_tests:
        lines.append("Statistical Significance:")
        lines.append("-"*60)
        for test_name, test_result in results.statistical_tests.items():
            if isinstance(test_result, dict) and 'p_value' in test_result:
                p_value = test_result['p_value']
                significant = p_value < 0.05
                lines.append(f"  {test_name}: p={p_value:.4f} {'(significant)' if significant else '(not significant)'}")
        lines.append("")
    
    # Key findings
    lines.append("Key Findings:")
    lines.append("-"*60)
    
    # Find best method
    best_method = sorted_methods[0][0]
    best_metrics = sorted_methods[0][1]
    lines.append(f"• Best performing method: {best_method}")
    lines.append(f"  Average final value: {best_metrics.mean_final_value:.4f}")
    
    # Compare trained vs baselines
    trained_methods = [m for m, _ in sorted_methods if any(x in m.lower() for x in ['grpo', 'bc'])]
    baseline_methods = [m for m, _ in sorted_methods if any(x in m.lower() for x in ['random', 'oracle', 'learning'])]
    
    if trained_methods and baseline_methods:
        best_trained = next((m for m, _ in sorted_methods if m in trained_methods), None)
        best_baseline = next((m for m, _ in sorted_methods if m in baseline_methods), None)
        
        if best_trained and best_baseline:
            trained_value = results.method_metrics[best_trained].mean_final_value
            baseline_value = results.method_metrics[best_baseline].mean_final_value
            
            if results.config.get('experiment', {}).get('target', {}).get('optimization_direction') == "MINIMIZE":
                improvement_pct = (baseline_value - trained_value) / abs(baseline_value) * 100
            else:
                improvement_pct = (trained_value - baseline_value) / abs(baseline_value) * 100
                
            lines.append(f"• Best trained method ({best_trained}) vs best baseline ({best_baseline}):")
            lines.append(f"  Improvement: {improvement_pct:.1f}%")
    
    lines.append("")
    lines.append("="*60)
    
    report = "\n".join(lines)
    
    # Save report
    report_path = output_dir / "evaluation_summary.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    return report


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/evaluation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--grpo-checkpoint",
        type=str,
        default=None,
        help="Path to GRPO checkpoint directory"
    )
    parser.add_argument(
        "--bc-surrogate-checkpoint",
        type=str,
        default=None,
        help="Path to BC surrogate checkpoint directory"
    )
    parser.add_argument(
        "--bc-acquisition-checkpoint",
        type=str,
        default=None,
        help="Path to BC acquisition checkpoint directory"
    )
    parser.add_argument(
        "--max-interventions",
        type=int,
        default=20,
        help="Maximum interventions per run"
    )
    parser.add_argument(
        "--n-test-scms",
        type=int,
        default=10,
        help="Number of test SCMs"
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=3,
        help="Number of runs per SCM"
    )
    parser.add_argument(
        "--optimization",
        type=str,
        choices=["MINIMIZE", "MAXIMIZE"],
        default="MINIMIZE",
        help="Optimization direction"
    )
    
    args = parser.parse_args()
    
    # Validate checkpoints
    checkpoint_paths = {}
    checkpoint_info = {}
    
    if args.grpo_checkpoint:
        path = Path(args.grpo_checkpoint)
        if not path.exists():
            raise ValueError(f"GRPO checkpoint not found: {path}")
        checkpoint_paths["grpo"] = path
        checkpoint_info["grpo"] = extract_model_info(path)
        
    if args.bc_surrogate_checkpoint:
        path = Path(args.bc_surrogate_checkpoint)
        if not path.exists():
            raise ValueError(f"BC surrogate checkpoint not found: {path}")
        checkpoint_paths["bc_surrogate"] = path
        checkpoint_info["bc_surrogate"] = extract_model_info(path)
        
    if args.bc_acquisition_checkpoint:
        path = Path(args.bc_acquisition_checkpoint)
        if not path.exists():
            raise ValueError(f"BC acquisition checkpoint not found: {path}")
        checkpoint_paths["bc_acquisition"] = path
        checkpoint_info["bc_acquisition"] = extract_model_info(path)
    
    if not checkpoint_paths:
        logger.warning("No checkpoints provided - will only evaluate baselines")
    
    # Create configuration
    config = create_evaluation_config(
        max_interventions=args.max_interventions,
        n_test_scms=args.n_test_scms,
        n_runs_per_scm=args.n_runs,
        optimization_direction=args.optimization
    )
    
    # Generate test SCMs
    test_scms = generate_test_scms(args.n_test_scms)
    
    # Run evaluation
    output_dir = Path(args.output_dir)
    results = evaluate_methods(
        checkpoint_paths=checkpoint_paths,
        config=config,
        output_dir=output_dir,
        test_scms=test_scms
    )
    
    # Generate visualizations
    plot_paths = generate_visualizations(results, output_dir)
    
    # Create summary report
    report = create_summary_report(results, checkpoint_info, output_dir)
    
    # Print summary
    print(report)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Visualizations: {len(plot_paths)} plots generated")
    
    return results


if __name__ == "__main__":
    main()