#!/usr/bin/env python3
"""
Main script to run initial comparison experiments.

This script loads configuration, sets up methods, runs experiments,
and saves results.
"""

import argparse
import logging
import json
import yaml
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add paths
import sys
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from experiments.evaluation.initial_comparison.src.experiment_runner import (
    ExperimentRunner, ExperimentConfig, MethodConfig
)
from experiments.evaluation.core.plotting_utils import PlottingUtils

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_methods(config: dict, 
                  policy_checkpoint: Path = None,
                  surrogate_checkpoint: Path = None) -> list:
    """
    Setup method configurations.
    
    Args:
        config: Experiment configuration
        policy_checkpoint: Optional path to policy checkpoint
        surrogate_checkpoint: Optional path to surrogate checkpoint
        
    Returns:
        List of MethodConfig objects
    """
    methods = []
    
    # Baseline 1: Random policy
    methods.append(MethodConfig(
        name="Random",
        policy_type="random",
        surrogate_checkpoint=surrogate_checkpoint,
        use_surrogate=surrogate_checkpoint is not None
    ))
    
    # Baseline 2: Oracle policy
    methods.append(MethodConfig(
        name="Oracle",
        policy_type="oracle",
        surrogate_checkpoint=surrogate_checkpoint,
        use_surrogate=surrogate_checkpoint is not None
    ))
    
    # Evaluated method: Provided policy (or untrained)
    if policy_checkpoint and policy_checkpoint.exists():
        methods.append(MethodConfig(
            name="Trained Policy",
            policy_type="checkpoint",
            policy_checkpoint=policy_checkpoint,
            surrogate_checkpoint=surrogate_checkpoint,
            use_surrogate=surrogate_checkpoint is not None
        ))
    else:
        logger.warning("No policy checkpoint provided or not found, using untrained policy")
        methods.append(MethodConfig(
            name="Untrained Policy",
            policy_type="untrained",
            surrogate_checkpoint=surrogate_checkpoint,
            use_surrogate=surrogate_checkpoint is not None
        ))
    
    return methods


def create_plots(results_df: pd.DataFrame, 
                aggregated: dict,
                output_dir: Path) -> None:
    """
    Create visualization plots.
    
    Args:
        results_df: Raw results
        aggregated: Aggregated results
        output_dir: Directory to save plots
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    PlottingUtils.setup_style()
    
    # Get valid results
    valid_df = results_df[~results_df['final_target'].isna()].copy()
    
    # 1. Target value vs SCM size
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method in valid_df['method'].unique():
        method_data = valid_df[valid_df['method'] == method]
        means = method_data.groupby('scm_size')['final_target'].mean()
        stds = method_data.groupby('scm_size')['final_target'].std()
        
        ax.errorbar(means.index, means.values, yerr=stds.values,
                   label=method, marker='o', capsize=5, linewidth=2)
    
    ax.set_xlabel('SCM Size (number of variables)')
    ax.set_ylabel('Final Target Value')
    ax.set_title('Final Target Value vs SCM Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'target_vs_size.png', dpi=150)
    plt.close()
    
    # 2. F1 score vs SCM size (if available)
    if 'graph_f1' in valid_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for method in valid_df['method'].unique():
            method_data = valid_df[valid_df['method'] == method]
            if 'graph_f1' in method_data.columns:
                means = method_data.groupby('scm_size')['graph_f1'].mean()
                stds = method_data.groupby('scm_size')['graph_f1'].std()
                
                ax.errorbar(means.index, means.values, yerr=stds.values,
                           label=method, marker='o', capsize=5, linewidth=2)
        
        ax.set_xlabel('SCM Size (number of variables)')
        ax.set_ylabel('F1 Score')
        ax.set_title('Graph Discovery F1 Score vs SCM Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        plt.tight_layout()
        plt.savefig(plots_dir / 'f1_vs_size.png', dpi=150)
        plt.close()
    
    # 3. SHD vs SCM size (if available)
    if 'graph_shd' in valid_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for method in valid_df['method'].unique():
            method_data = valid_df[valid_df['method'] == method]
            if 'graph_shd' in method_data.columns:
                means = method_data.groupby('scm_size')['graph_shd'].mean()
                stds = method_data.groupby('scm_size')['graph_shd'].std()
                
                ax.errorbar(means.index, means.values, yerr=stds.values,
                           label=method, marker='o', capsize=5, linewidth=2)
        
        ax.set_xlabel('SCM Size (number of variables)')
        ax.set_ylabel('Structural Hamming Distance')
        ax.set_title('SHD vs SCM Size (lower is better)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'shd_vs_size.png', dpi=150)
        plt.close()
    
    logger.info(f"Saved plots to {plots_dir}")


def save_results(results_df: pd.DataFrame,
                aggregated: dict,
                output_dir: Path) -> None:
    """
    Save experiment results.
    
    Args:
        results_df: Raw results DataFrame
        aggregated: Aggregated results
        output_dir: Directory to save results
    """
    # Save raw results
    results_df.to_csv(output_dir / 'raw_results.csv', index=False)
    
    # Save aggregated results
    aggregated_dir = output_dir / 'aggregated'
    aggregated_dir.mkdir(exist_ok=True)
    
    for name, df in aggregated.items():
        df.to_csv(aggregated_dir / f'{name}.csv')
    
    # Save summary statistics as JSON
    summary = {}
    valid_df = results_df[~results_df['final_target'].isna()]
    
    for method in valid_df['method'].unique():
        method_data = valid_df[valid_df['method'] == method]
        summary[method] = {
            'n_experiments': len(method_data),
            'mean_final_target': float(method_data['final_target'].mean()),
            'std_final_target': float(method_data['final_target'].std()),
            'mean_best_target': float(method_data['best_target'].mean()),
            'std_best_target': float(method_data['best_target'].std())
        }
        
        # Add graph metrics if available
        for col in method_data.columns:
            if col.startswith('graph_') and col != 'graph_metrics':
                metric_name = col.replace('graph_', '')
                # Only compute mean/std for numeric columns
                if method_data[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    summary[method][f'mean_{metric_name}'] = float(method_data[col].mean())
                    summary[method][f'std_{metric_name}'] = float(method_data[col].std())
    
    with open(output_dir / 'summary_stats.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved results to {output_dir}")


def generate_report(results_df: pd.DataFrame,
                   aggregated: dict,
                   output_dir: Path) -> None:
    """
    Generate human-readable report.
    
    Args:
        results_df: Raw results DataFrame
        aggregated: Aggregated results
        output_dir: Directory to save report
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("INITIAL COMPARISON EXPERIMENT REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Summary statistics
    valid_df = results_df[~results_df['final_target'].isna()]
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("-" * 40)
    report_lines.append(f"Total experiments: {len(results_df)}")
    report_lines.append(f"Successful experiments: {len(valid_df)}")
    report_lines.append(f"Failed experiments: {len(results_df) - len(valid_df)}")
    report_lines.append("")
    
    # Results by method
    report_lines.append("RESULTS BY METHOD")
    report_lines.append("-" * 40)
    
    for method in valid_df['method'].unique():
        method_data = valid_df[valid_df['method'] == method]
        report_lines.append(f"\n{method}:")
        report_lines.append(f"  Final target: {method_data['final_target'].mean():.3f} ± {method_data['final_target'].std():.3f}")
        report_lines.append(f"  Best target: {method_data['best_target'].mean():.3f} ± {method_data['best_target'].std():.3f}")
        
        # Add graph metrics if available
        if 'graph_f1' in method_data.columns:
            report_lines.append(f"  F1 score: {method_data['graph_f1'].mean():.3f} ± {method_data['graph_f1'].std():.3f}")
        if 'graph_shd' in method_data.columns:
            report_lines.append(f"  SHD: {method_data['graph_shd'].mean():.1f} ± {method_data['graph_shd'].std():.1f}")
    
    report_lines.append("")
    
    # Results by SCM size
    report_lines.append("RESULTS BY SCM SIZE")
    report_lines.append("-" * 40)
    
    for size in sorted(valid_df['scm_size'].unique()):
        size_data = valid_df[valid_df['scm_size'] == size]
        report_lines.append(f"\nSize {size}:")
        report_lines.append(f"  Experiments: {len(size_data)}")
        report_lines.append(f"  Mean final target: {size_data['final_target'].mean():.3f}")
        
        # Best method for this size
        best_method = size_data.groupby('method')['final_target'].mean().idxmin()
        best_value = size_data.groupby('method')['final_target'].mean().min()
        report_lines.append(f"  Best method: {best_method} ({best_value:.3f})")
    
    # Write report
    report_path = output_dir / 'report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Generated report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Run initial comparison experiments')
    parser.add_argument('--config', type=Path, 
                       default=Path(__file__).parent.parent / 'configs' / 'quick_test_config.yaml',
                       help='Path to configuration file (defaults to quick_test_config.yaml for testing)')
    parser.add_argument('--policy-checkpoint', type=Path,
                       help='Path to policy checkpoint')
    parser.add_argument('--surrogate-checkpoint', type=Path,
                       help='Path to surrogate checkpoint')
    parser.add_argument('--output-dir', type=Path,
                       help='Output directory (default: results/run_[timestamp])')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(__file__).parent.parent / 'results' / f'run_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Save configuration
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Create experiment config
    exp_config = ExperimentConfig(
        scm_sizes=config['scm_generation']['sizes'],
        n_scms_per_size=config['scm_generation']['n_scms_per_size'],
        n_observational_samples=config['data_generation']['n_observational_samples'],
        n_interventions=config['data_generation']['n_interventions'],
        structure_types=config['scm_generation']['structure_types'],
        edge_density=config['scm_generation']['edge_density'],
        seed=config['experiment']['seed'],
        metrics_to_track=config['metrics']['track']
    )
    
    # Setup methods
    methods = setup_methods(config, args.policy_checkpoint, args.surrogate_checkpoint)
    
    logger.info(f"Running experiment with {len(methods)} methods")
    for method in methods:
        logger.info(f"  - {method.name}: {method.policy_type}")
    
    # Run experiments
    runner = ExperimentRunner(exp_config)
    results_df = runner.run_all_experiments(methods)
    
    # Aggregate results
    aggregated = runner.aggregate_results(results_df)
    
    # Save results
    save_results(results_df, aggregated, output_dir)
    
    # Create plots
    create_plots(results_df, aggregated, output_dir)
    
    # Generate report
    generate_report(results_df, aggregated, output_dir)
    
    logger.info("Experiment complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()