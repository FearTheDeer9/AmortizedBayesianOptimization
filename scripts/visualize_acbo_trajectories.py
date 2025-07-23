#!/usr/bin/env python3
"""
ACBO Trajectory Visualization Script

This script demonstrates how to visualize trajectories from ACBO comparison experiments,
including integration with the full ACBO comparison framework.

Usage:
    # Visualize from demo experiments
    poetry run python scripts/visualize_acbo_trajectories.py --source demo
    
    # Visualize from ACBO comparison results
    poetry run python scripts/visualize_acbo_trajectories.py --source acbo --results-file results.json
    
    # Create custom visualizations
    poetry run python scripts/visualize_acbo_trajectories.py --source custom --plot-type dashboard
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.causal_bayes_opt.visualization.plots import (
    plot_convergence, plot_target_optimization, plot_structure_learning_dashboard,
    plot_baseline_comparison, plot_method_comparison, plot_calibration_curves,
    plot_precision_recall_curves, save_all_plots
)
from scripts.core.acbo_comparison.metrics_collector import MetricsCollector
from examples.demo_evaluation import extract_trajectory_metrics_from_demo
from examples.complete_workflow_demo import run_progressive_learning_demo, DemoConfig


def extract_trajectories_from_acbo_results(acbo_results: Dict[str, Any]) -> Dict[str, Dict[str, List[float]]]:
    """
    Extract trajectory data from ACBO comparison framework results.
    
    Args:
        acbo_results: Results from ACBOExperimentRunner
        
    Returns:
        Dictionary mapping method names to trajectory data
    """
    method_trajectories = {}
    
    # Check for aggregated trajectories first
    if 'aggregated_trajectories' in acbo_results:
        for method_name, trajectories in acbo_results['aggregated_trajectories'].items():
            if any(key.endswith('_mean') for key in trajectories):
                # Extract mean trajectories
                method_data = {
                    'steps': list(range(1, len(trajectories.get('target_values_mean', [])) + 1)),
                    'target_mean': trajectories.get('target_values_mean', []),
                    'f1_mean': trajectories.get('f1_scores_mean', []),
                    'shd_mean': trajectories.get('shd_values_mean', []),
                    'parent_prob_mean': trajectories.get('true_parent_likelihood_mean', []),
                    'n_runs': len(acbo_results.get('method_results', {}).get(method_name, []))
                }
                method_trajectories[method_name] = method_data
    
    # Fallback: Extract from method results
    if not method_trajectories and 'method_results' in acbo_results:
        for method_name, results_list in acbo_results['method_results'].items():
            # Collect trajectories from all runs
            all_target_trajectories = []
            all_f1_trajectories = []
            all_shd_trajectories = []
            
            for result in results_list:
                if result.get('success', True):
                    # Check detailed_results first
                    detailed = result.get('detailed_results', {})
                    
                    # Extract trajectories
                    if 'target_progress' in detailed:
                        all_target_trajectories.append(detailed['target_progress'])
                    elif 'target_values' in detailed:
                        all_target_trajectories.append(detailed['target_values'])
                    
                    if 'f1_scores' in detailed:
                        all_f1_trajectories.append(detailed['f1_scores'])
                    
                    if 'shd_values' in detailed:
                        all_shd_trajectories.append(detailed['shd_values'])
            
            if all_target_trajectories:
                # Compute means
                max_len = max(len(t) for t in all_target_trajectories)
                
                # Pad trajectories
                padded_targets = []
                padded_f1s = []
                padded_shds = []
                
                for traj in all_target_trajectories:
                    padded = list(traj) + [traj[-1]] * (max_len - len(traj))
                    padded_targets.append(padded)
                
                for traj in all_f1_trajectories:
                    padded = list(traj) + [traj[-1]] * (max_len - len(traj))
                    padded_f1s.append(padded)
                    
                for traj in all_shd_trajectories:
                    padded = list(traj) + [traj[-1]] * (max_len - len(traj))
                    padded_shds.append(padded)
                
                method_trajectories[method_name] = {
                    'steps': list(range(1, max_len + 1)),
                    'target_mean': np.mean(padded_targets, axis=0).tolist() if padded_targets else [],
                    'f1_mean': np.mean(padded_f1s, axis=0).tolist() if padded_f1s else [],
                    'shd_mean': np.mean(padded_shds, axis=0).tolist() if padded_shds else [],
                    'n_runs': len(all_target_trajectories)
                }
    
    return method_trajectories


def visualize_demo_trajectories(output_dir: Path):
    """Run demo experiments and create visualizations."""
    print("\n🚀 Running demo experiments for visualization...")
    
    # Run a demo experiment
    config = DemoConfig(
        n_observational_samples=20,
        n_intervention_steps=15,
        learning_rate=1e-3,
        random_seed=42
    )
    
    results = run_progressive_learning_demo(config)
    
    # Extract trajectory metrics
    trajectory_metrics = extract_trajectory_metrics_from_demo(results)
    
    # Save all plots
    print("\n📊 Generating visualization plots...")
    saved_files = save_all_plots(
        {'trajectory_metrics': trajectory_metrics},
        output_dir=str(output_dir),
        prefix="demo"
    )
    
    print(f"\n✅ Saved {len(saved_files)} plots:")
    for file in saved_files:
        print(f"   - {file}")
    
    return trajectory_metrics


def visualize_acbo_results(results_file: Path, output_dir: Path):
    """Visualize results from ACBO comparison framework."""
    print(f"\n📂 Loading ACBO results from: {results_file}")
    
    with open(results_file, 'r') as f:
        acbo_results = json.load(f)
    
    # Extract method trajectories
    method_trajectories = extract_trajectories_from_acbo_results(acbo_results)
    
    if not method_trajectories:
        print("❌ No trajectory data found in results file!")
        return
    
    print(f"\n📊 Found trajectories for {len(method_trajectories)} methods")
    
    # Create comparison plots
    output_dir.mkdir(exist_ok=True)
    
    # 1. Baseline comparison plot
    comparison_path = output_dir / "acbo_method_comparison.png"
    plot_baseline_comparison(
        method_trajectories,
        title="ACBO Methods Comparison",
        save_path=str(comparison_path)
    )
    print(f"✅ Saved comparison plot: {comparison_path}")
    
    # 2. Method comparison with all metrics
    detailed_path = output_dir / "acbo_detailed_comparison.png"
    plot_method_comparison(
        method_trajectories,
        title="Detailed Method Comparison",
        save_path=str(detailed_path),
        metrics=['shd', 'f1', 'target']
    )
    print(f"✅ Saved detailed comparison: {detailed_path}")
    
    # 3. Individual method dashboards
    for method_name, trajectory_data in method_trajectories.items():
        # Convert to format expected by plotting functions
        metrics = {
            'steps': trajectory_data['steps'],
            'target_values': trajectory_data.get('target_mean', []),
            'f1_scores': trajectory_data.get('f1_mean', []),
            'shd_values': trajectory_data.get('shd_mean', []),
            'true_parent_likelihood': trajectory_data.get('parent_prob_mean', [])
        }
        
        dashboard_path = output_dir / f"{method_name.replace(' ', '_')}_dashboard.png"
        plot_structure_learning_dashboard(
            metrics,
            title=f"{method_name} - Structure Learning Dashboard",
            save_path=str(dashboard_path)
        )
        print(f"✅ Saved {method_name} dashboard: {dashboard_path}")


def create_custom_visualization(plot_type: str, output_dir: Path):
    """Create custom visualizations with synthetic data."""
    print(f"\n🎨 Creating custom {plot_type} visualization...")
    
    # Generate synthetic trajectory data
    n_steps = 20
    steps = list(range(1, n_steps + 1))
    
    # Simulate learning curves
    def sigmoid(x, k=0.5, x0=10):
        return 1 / (1 + np.exp(-k * (x - x0)))
    
    # Create synthetic data for multiple methods
    methods_data = {
        "Random Baseline": {
            'steps': steps,
            'f1_mean': [sigmoid(i, k=0.3, x0=12) * 0.7 for i in steps],
            'shd_mean': [5 - sigmoid(i, k=0.3, x0=12) * 3.5 for i in steps],
            'target_mean': [i * 0.05 + np.random.normal(0, 0.02) for i in steps],
            'n_runs': 5
        },
        "Learned Policy": {
            'steps': steps,
            'f1_mean': [sigmoid(i, k=0.5, x0=8) * 0.9 for i in steps],
            'shd_mean': [5 - sigmoid(i, k=0.5, x0=8) * 4.5 for i in steps],
            'target_mean': [i * 0.08 + np.random.normal(0, 0.01) for i in steps],
            'n_runs': 5
        },
        "Oracle Policy": {
            'steps': steps,
            'f1_mean': [sigmoid(i, k=0.8, x0=5) * 0.95 for i in steps],
            'shd_mean': [5 - sigmoid(i, k=0.8, x0=5) * 4.8 for i in steps],
            'target_mean': [i * 0.1 + np.random.normal(0, 0.005) for i in steps],
            'n_runs': 5
        }
    }
    
    output_dir.mkdir(exist_ok=True)
    
    if plot_type == "dashboard":
        # Create individual dashboards
        for method_name, data in methods_data.items():
            metrics = {
                'steps': data['steps'],
                'f1_scores': data['f1_mean'],
                'shd_values': data['shd_mean'],
                'target_values': data['target_mean'],
                'true_parent_likelihood': data['f1_mean'],  # Use F1 as proxy
                'uncertainty_bits': [5 - f1 * 4 for f1 in data['f1_mean']]  # Inverse of F1
            }
            
            dashboard_path = output_dir / f"custom_{method_name.replace(' ', '_')}_dashboard.png"
            plot_structure_learning_dashboard(
                metrics,
                title=f"{method_name} - Custom Visualization",
                save_path=str(dashboard_path)
            )
            print(f"✅ Created dashboard: {dashboard_path}")
    
    elif plot_type == "comparison":
        # Create comparison plot
        comparison_path = output_dir / "custom_comparison.png"
        plot_baseline_comparison(
            methods_data,
            title="Custom Method Comparison",
            save_path=str(comparison_path)
        )
        print(f"✅ Created comparison: {comparison_path}")
    
    elif plot_type == "convergence":
        # Create convergence plots for each method
        for method_name, data in methods_data.items():
            metrics = {
                'steps': data['steps'],
                'f1_scores': data['f1_mean'],
                'true_parent_likelihood': data['f1_mean'],
                'uncertainty_bits': [5 - f1 * 4 for f1 in data['f1_mean']]
            }
            
            convergence_path = output_dir / f"custom_{method_name.replace(' ', '_')}_convergence.png"
            plot_convergence(
                metrics,
                title=f"{method_name} - Convergence Analysis",
                save_path=str(convergence_path)
            )
            print(f"✅ Created convergence plot: {convergence_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize ACBO experiment trajectories"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["demo", "acbo", "custom"],
        default="demo",
        help="Source of trajectory data"
    )
    parser.add_argument(
        "--results-file",
        type=str,
        help="Path to ACBO results JSON file (for source=acbo)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="acbo_visualizations",
        help="Directory to save plots"
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=["dashboard", "comparison", "convergence"],
        default="dashboard",
        help="Type of plot for custom visualization"
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    print("📊 ACBO Trajectory Visualization Tool")
    print("=" * 50)
    
    if args.source == "demo":
        visualize_demo_trajectories(output_dir)
    elif args.source == "acbo":
        if not args.results_file:
            print("❌ Error: --results-file required for source=acbo")
            return
        results_file = Path(args.results_file)
        if not results_file.exists():
            print(f"❌ Error: Results file not found: {results_file}")
            return
        visualize_acbo_results(results_file, output_dir)
    elif args.source == "custom":
        create_custom_visualization(args.plot_type, output_dir)
    
    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()