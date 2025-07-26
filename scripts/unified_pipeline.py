#!/usr/bin/env python3
"""
Unified Pipeline Script for End-to-End GRPO Evaluation

This script demonstrates the complete pipeline:
1. Load trained GRPO checkpoint
2. Generate test SCMs dynamically
3. Run ACBO comparison with trained model vs baselines
4. Save results with trajectory data
5. Generate the three-panel time-series plots

Usage:
    poetry run python scripts/unified_pipeline.py
    poetry run python scripts/unified_pipeline.py --checkpoint path/to/checkpoint
    poetry run python scripts/unified_pipeline.py --num-scms 5 --runs-per-method 3
"""

import argparse
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import new interfaces
from scripts.notebooks.pipeline_interfaces import (
    OptimizationConfig, CheckpointInterface, EvaluationResults
)
from scripts.notebooks.interface_adapters import (
    ResultsAdapter, ensure_standard_output_location
)
from scripts.notebooks.base_components import CheckpointManager


def extract_trajectories_from_raw_results(method_results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract trajectory data from raw method results."""
    plot_data = {}
    
    for method_name, results in method_results.items():
        if isinstance(results, list) and results:
            # Check if first result has detailed_results
            if isinstance(results[0], dict) and 'detailed_results' in results[0]:
                # Aggregate trajectories from all runs
                all_target_values = []
                all_f1_scores = []
                all_shd_values = []
                steps = None
                
                for result in results:
                    detailed = result.get('detailed_results', {})
                    if 'target_progress' in detailed:
                        all_target_values.append(detailed['target_progress'])
                    if 'f1_scores' in detailed:
                        all_f1_scores.append(detailed['f1_scores'])
                    if 'shd_values' in detailed:
                        all_shd_values.append(detailed['shd_values'])
                    if steps is None and 'steps' in detailed:
                        steps = detailed['steps']
                
                # Create aggregated data
                if all_target_values:
                    import numpy as np
                    plot_data[method_name] = {
                        'steps': steps or list(range(len(all_target_values[0]))),
                        'target_mean': np.mean(all_target_values, axis=0).tolist() if all_target_values else [],
                        'target_std': np.std(all_target_values, axis=0).tolist() if all_target_values else [],
                        'f1_mean': np.mean(all_f1_scores, axis=0).tolist() if all_f1_scores else [],
                        'f1_std': np.std(all_f1_scores, axis=0).tolist() if all_f1_scores else [],
                        'shd_mean': np.mean(all_shd_values, axis=0).tolist() if all_shd_values else [],
                        'shd_std': np.std(all_shd_values, axis=0).tolist() if all_shd_values else [],
                        'n_runs': len(results)
                    }
    
    return plot_data


def find_latest_checkpoint(checkpoint_dir: Path) -> CheckpointInterface:
    """Find the latest GRPO checkpoint using standardized interface."""
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    
    # Get all available checkpoints using new interface
    checkpoints = checkpoint_manager.find_checkpoint_interfaces()
    
    if not checkpoints:
        raise ValueError(f"No valid checkpoints found in {checkpoint_dir}")
    
    # Sort by timestamp (newest first)
    latest = max(checkpoints, key=lambda c: c.timestamp)
    
    # Validate the checkpoint
    is_valid, issues = latest.validate()
    if not is_valid:
        logger.warning(f"Latest checkpoint has issues: {issues}")
        # Try to find a valid one
        for checkpoint in sorted(checkpoints, key=lambda c: c.timestamp, reverse=True):
            is_valid, issues = checkpoint.validate()
            if is_valid:
                logger.info(f"Using valid checkpoint: {checkpoint.name}")
                return checkpoint
        
        raise ValueError(f"No valid checkpoints found. Issues with latest: {issues}")
    
    logger.info(f"Found latest checkpoint: {latest.name}")
    logger.info(f"  Optimization: {latest.optimization_config.direction}")
    logger.info(f"  Training mode: {latest.training_mode}")
    logger.info(f"  Success: {latest.success}")
    
    return latest


def run_acbo_comparison(
    checkpoint: CheckpointInterface,
    num_scms: int = 3,
    runs_per_method: int = 3,
    intervention_budget: int = 10,
    output_dir: Path = None
) -> EvaluationResults:
    """Run ACBO comparison with the given checkpoint."""
    
    logger.info("🚀 Running ACBO comparison...")
    
    # Validate inputs
    is_valid, issues = checkpoint.validate()
    if not is_valid:
        raise ValueError(f"Invalid checkpoint: {issues}")
    
    if num_scms < 1:
        raise ValueError(f'Invalid num_scms: {num_scms} (must be >= 1)')
    
    if runs_per_method < 1:
        raise ValueError(f'Invalid runs_per_method: {runs_per_method} (must be >= 1)')
    
    if intervention_budget < 1:
        raise ValueError(f'Invalid intervention_budget: {intervention_budget} (must be >= 1)')
    
    logger.info(f"Configuration validated:")
    logger.info(f"  - Checkpoint: {checkpoint.name}")
    logger.info(f"  - Optimization: {checkpoint.optimization_config.direction}")
    logger.info(f"  - SCMs: {num_scms}")
    logger.info(f"  - Runs per method: {runs_per_method}")
    logger.info(f"  - Intervention budget: {intervention_budget}")
    
    # Set up output directory
    if output_dir is None:
        output_dir = project_root / "results" / f"unified_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the unified evaluation system directly
    from src.causal_bayes_opt.evaluation import run_evaluation
    
    logger.info(f"Running evaluation using unified system")
    
    start_time = time.time()
    
    try:
        # Create evaluation config
        eval_config = {
            'n_scms': num_scms,
            'experiment': {
                'runs_per_method': runs_per_method,
                'target': {
                    'max_interventions': intervention_budget,
                    'n_observational_samples': 100,
                    'optimization_direction': checkpoint.optimization_config.direction
                },
                'methods': {
                    "Random + Untrained": "random_untrained",
                    "Random + Learning": "random_learning",
                    "Oracle + Learning": "oracle_learning",
                    "Trained Policy + Learning": "learned_enriched_policy"
                }
            },
            'policy_checkpoint_path': str(checkpoint.path),
            'visualization': {
                'enabled': True,
                'plot_types': ['target_trajectory', 'f1_trajectory', 'shd_trajectory', 'method_comparison']
            }
        }
        
        # Run evaluation
        comparison_results = run_evaluation(
            checkpoint_path=checkpoint.path,
            output_dir=output_dir,
            config=eval_config
        )
        
        duration = time.time() - start_time
        logger.info(f"✅ Comparison completed in {duration/60:.1f} minutes")
        
        # Convert to EvaluationResults format for compatibility
        evaluation_results = EvaluationResults(
            evaluation_timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            checkpoint_name=checkpoint.name,
            optimization_config=checkpoint.optimization_config,
            num_scms=num_scms,
            runs_per_method=runs_per_method,
            intervention_budget=intervention_budget,
            method_results={},
            summary_statistics={},
            pairwise_comparisons={},
            total_duration_minutes=duration / 60
        )
        
        # Convert method results
        for method_name, metrics in comparison_results.method_results.items():
            evaluation_results.method_results[method_name] = {
                'mean': metrics.mean_improvement,
                'std': metrics.std_improvement,
                'count': metrics.n_runs
            }
            evaluation_results.summary_statistics[method_name] = {
                'target_improvement_mean': metrics.mean_improvement,
                'target_improvement_std': metrics.std_improvement,
                'structure_accuracy_mean': metrics.mean_final_f1,
                'target_improvement_count': metrics.n_runs
            }
        
        evaluation_results.pairwise_comparisons = comparison_results.statistical_tests
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Failed to run evaluation: {e}")
        raise


def generate_trajectory_plots(
    evaluation_results: EvaluationResults,
    output_dir: Path
) -> bool:
    """Generate the three-panel trajectory plots from results."""
    
    logger.info("📊 Generating trajectory plots...")
    
    try:
        # First try to use aggregated trajectories if available
        plot_data = {}
        
        if hasattr(evaluation_results, 'aggregated_trajectories') and evaluation_results.aggregated_trajectories:
            logger.info("Using aggregated trajectory data")
            # Convert aggregated trajectories to plot format
            for method_name, trajectory_data in evaluation_results.aggregated_trajectories.items():
                plot_data[method_name] = trajectory_data
        else:
            # Fall back to extracting from method_results
            logger.info("Extracting plot data from method results")
            plot_data = ResultsAdapter.extract_plot_data(evaluation_results)
        
        if not plot_data:
            logger.warning("No plot data could be extracted from results")
            # Try to extract from raw method_results as last resort
            if hasattr(evaluation_results, 'method_results') and evaluation_results.method_results:
                logger.info("Attempting to extract trajectory data from raw method results")
                plot_data = extract_trajectories_from_raw_results(evaluation_results.method_results)
        
        if not plot_data:
            logger.error("No trajectory data available for plotting")
            return False
        
        # Create the three-panel plot
        plot_path = output_dir / "trajectory_comparison.png"
        
        try:
            from src.causal_bayes_opt.visualization.plots import plot_baseline_comparison
            
            fig = plot_baseline_comparison(
                plot_data,
                title=f"ACBO Methods - {evaluation_results.optimization_config.direction} Optimization",
                save_path=str(plot_path),
                figsize=(14, 10)
            )
            
            logger.info(f"✅ Saved trajectory plot to: {plot_path}")
        except ImportError:
            logger.warning("Could not import plot_baseline_comparison, creating simple plot")
            create_simple_trajectory_plot(plot_data, plot_path, evaluation_results.optimization_config)
        
        # Create summary plot
        summary_path = output_dir / "performance_summary.png"
        create_summary_plot(plot_data, summary_path, evaluation_results.optimization_config)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate plots: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_summary_plot(plot_data: Dict[str, Any], output_path: Path, 
                       optimization_config: OptimizationConfig):
    """Create a summary plot showing final performance."""
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract final values
    methods = []
    final_f1 = []
    final_shd = []
    final_target = []
    
    for method, data in plot_data.items():
        if data and 'steps' in data and len(data['steps']) > 0:
            methods.append(method)
            final_f1.append(data['f1_mean'][-1] if 'f1_mean' in data else 0.0)
            final_shd.append(data['shd_mean'][-1] if 'shd_mean' in data else 0.0)
            final_target.append(data['target_mean'][-1] if 'target_mean' in data else 0.0)
    
    if not methods:
        logger.warning("No data for summary plot")
        return
    
    # Create bar plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    x = np.arange(len(methods))
    
    # F1 Score
    ax1.bar(x, final_f1, color=['red' if 'Trained' in m else 'blue' for m in methods])
    ax1.set_ylabel('F1 Score')
    ax1.set_title('Final F1 Score (Higher is Better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace(' + ', '\n') for m in methods], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # SHD
    ax2.bar(x, final_shd, color=['red' if 'Trained' in m else 'blue' for m in methods])
    ax2.set_ylabel('SHD')
    ax2.set_title('Final SHD (Lower is Better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace(' + ', '\n') for m in methods], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Target Value - adapt title based on optimization direction
    ax3.bar(x, final_target, color=['red' if 'Trained' in m else 'blue' for m in methods])
    ax3.set_ylabel('Target Improvement')
    if optimization_config.is_minimizing:
        ax3.set_title('Final Target Improvement (Higher is Better)')
        ax3.invert_yaxis()  # Invert so lower values appear higher
    else:
        ax3.set_title('Final Target Improvement (Higher is Better)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.replace(' + ', '\n') for m in methods], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f'Final Performance Summary - {optimization_config.direction}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Saved summary plot to: {output_path}")


def create_simple_trajectory_plot(plot_data: Dict[str, Any], output_path: Path,
                                optimization_config: OptimizationConfig):
    """Create a simple trajectory plot when advanced plotting is not available."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for method, data in plot_data.items():
        if not data or 'steps' not in data:
            continue
            
        steps = data['steps']
        color = 'red' if 'Trained' in method else 'blue'
        
        # Target values
        if 'target_mean' in data:
            axes[0].plot(steps, data['target_mean'], label=method, color=color)
        
        # F1 scores
        if 'f1_mean' in data:
            axes[1].plot(steps, data['f1_mean'], label=method, color=color)
        
        # SHD values
        if 'shd_mean' in data:
            axes[2].plot(steps, data['shd_mean'], label=method, color=color)
    
    axes[0].set_xlabel('Intervention Step')
    axes[0].set_ylabel('Target Value')
    axes[0].set_title(f'Target Progress ({optimization_config.direction})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Intervention Step') 
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Structure Learning (F1)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Intervention Step')
    axes[2].set_ylabel('SHD')
    axes[2].set_title('Structure Learning (SHD)')  
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'ACBO Methods Comparison - {optimization_config.direction}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Saved simple trajectory plot to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified pipeline for GRPO evaluation with trajectory plots"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to GRPO checkpoint (default: find latest)"
    )
    parser.add_argument(
        "--num-scms",
        type=int,
        default=3,
        help="Number of test SCMs to generate"
    )
    parser.add_argument(
        "--runs-per-method",
        type=int,
        default=3,
        help="Number of runs per method"
    )
    parser.add_argument(
        "--intervention-budget",
        type=int,
        default=10,
        help="Number of interventions per run"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results and plots"
    )
    parser.add_argument(
        "--optimization-direction",
        type=str,
        choices=["MINIMIZE", "MAXIMIZE"],
        help="Optimization direction (MINIMIZE or MAXIMIZE)"
    )
    
    args = parser.parse_args()
    
    print("🎯 Unified GRPO Evaluation Pipeline")
    print("=" * 50)
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        checkpoint_manager = CheckpointManager(checkpoint_path.parent)
        try:
            checkpoint = checkpoint_manager.load_checkpoint_interface(checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return
    else:
        # Find latest checkpoint
        checkpoint_dir = project_root / "checkpoints" / "grpo_training"
        if not checkpoint_dir.exists():
            checkpoint_dir = project_root / "checkpoints"
        
        try:
            checkpoint = find_latest_checkpoint(checkpoint_dir)
        except ValueError as e:
            logger.error(str(e))
            return
    
    logger.info(f"📁 Using checkpoint: {checkpoint.name}")
    logger.info(f"📊 Optimization: {checkpoint.optimization_config.direction}")
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "results" / f"unified_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📂 Output directory: {output_dir}")
    
    # Step 1: Run ACBO comparison
    try:
        evaluation_results = run_acbo_comparison(
            checkpoint=checkpoint,
            num_scms=args.num_scms,
            runs_per_method=args.runs_per_method,
            intervention_budget=args.intervention_budget,
            output_dir=output_dir
        )
        
        # Save standardized results
        evaluation_results.save_to_file(output_dir)
        
    except Exception as e:
        logger.error(f"Failed to run comparison: {e}")
        return
    
    # Step 2: Generate plots
    plot_success = generate_trajectory_plots(evaluation_results, output_dir)
    
    if plot_success:
        print("\n✅ Pipeline completed successfully!")
        print(f"📂 Results saved to: {output_dir}")
        print(f"📊 Optimization: {evaluation_results.optimization_config.direction}")
        print(f"⏱️ Duration: {evaluation_results.total_duration_minutes:.1f} minutes")
        print("\nGenerated files:")
        for file in sorted(output_dir.iterdir()):
            if file.is_file():
                print(f"  - {file.name}")
    else:
        print("\n⚠️ Pipeline completed with warnings")
        print("Comparison ran but plots could not be generated")


if __name__ == "__main__":
    main()