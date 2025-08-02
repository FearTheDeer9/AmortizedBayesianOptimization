#!/usr/bin/env python3
"""
Visualization Script for ACBO Demo Trajectories

This script demonstrates how to:
1. Run ACBO demo experiments
2. Extract trajectory metrics
3. Generate publication-ready plots

Usage:
    poetry run python scripts/visualize_demo_trajectory.py
    poetry run python scripts/visualize_demo_trajectory.py --experiment oracle
    poetry run python scripts/visualize_demo_trajectory.py --output-dir my_plots
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from examples.demo_scms import create_easy_scm, create_medium_scm, create_hard_scm
from examples.demo_learning import DemoConfig
from examples.demo_evaluation import extract_trajectory_metrics_from_demo
from examples.complete_workflow_demo import (
    run_progressive_learning_demo_with_scm,
    run_progressive_learning_demo_with_oracle_interventions,
    run_difficulty_comparative_study
)
from src.causal_bayes_opt.visualization.plots import (
    plot_convergence, plot_target_optimization, plot_structure_learning_dashboard,
    plot_baseline_comparison, save_all_plots
)


def visualize_single_experiment(experiment_type: str = "random", output_dir: str = "demo_plots"):
    """
    Run a single experiment and generate visualizations.
    
    Args:
        experiment_type: Type of experiment ("random", "oracle", "fixed")
        output_dir: Directory to save plots
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Configuration
    config = DemoConfig(
        n_observational_samples=20,
        n_intervention_steps=15,
        learning_rate=1e-3,
        random_seed=42
    )
    
    # Select SCM and run experiment
    print(f"\n🚀 Running {experiment_type} intervention experiment...")
    scm = create_easy_scm()
    
    if experiment_type == "oracle":
        results = run_progressive_learning_demo_with_oracle_interventions(scm, config)
    else:
        results = run_progressive_learning_demo_with_scm(scm, config)
    
    # Extract trajectory metrics
    print("\n📊 Extracting trajectory metrics...")
    trajectory_metrics = extract_trajectory_metrics_from_demo(results)
    
    # Generate plots
    print("\n📈 Generating visualizations...")
    
    # 1. Convergence plot
    convergence_path = output_path / f"{experiment_type}_convergence.png"
    plot_convergence(
        trajectory_metrics,
        title=f"Convergence to True Parent Set ({experiment_type.capitalize()} Interventions)",
        save_path=str(convergence_path),
        show_f1=True,
        show_uncertainty=True
    )
    print(f"   ✅ Saved convergence plot: {convergence_path}")
    
    # 2. Target optimization plot
    target_path = output_path / f"{experiment_type}_target_optimization.png"
    plot_target_optimization(
        trajectory_metrics,
        title=f"Target Variable Optimization ({experiment_type.capitalize()} Interventions)",
        save_path=str(target_path)
    )
    print(f"   ✅ Saved target optimization plot: {target_path}")
    
    # 3. Structure learning dashboard
    dashboard_path = output_path / f"{experiment_type}_structure_dashboard.png"
    plot_structure_learning_dashboard(
        trajectory_metrics,
        title=f"Structure Learning Dashboard ({experiment_type.capitalize()} Interventions)",
        save_path=str(dashboard_path)
    )
    print(f"   ✅ Saved structure learning dashboard: {dashboard_path}")
    
    # Print summary
    print("\n📝 Experiment Summary:")
    print(f"   Target variable: {trajectory_metrics['target_variable']}")
    print(f"   True parents: {trajectory_metrics['true_parents']}")
    print(f"   Converged: {trajectory_metrics['converged']}")
    print(f"   Final F1 score: {trajectory_metrics['f1_scores'][-1]:.3f}")
    print(f"   Final parent likelihood: {trajectory_metrics['true_parent_likelihood'][-1]:.3f}")
    print(f"   Target improvement: {trajectory_metrics['improvement']:.3f}")
    
    return trajectory_metrics


def visualize_method_comparison(output_dir: str = "demo_plots"):
    """
    Compare different intervention methods and generate comparison plots.
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Configuration
    config = DemoConfig(
        n_observational_samples=20,
        n_intervention_steps=15,
        learning_rate=1e-3,
        random_seed=42
    )
    
    # Run experiments for different methods
    print("\n🔬 Running method comparison experiments...")
    scm = create_easy_scm()
    
    # Random interventions (baseline)
    print("\n1. Random interventions...")
    random_results = run_progressive_learning_demo_with_scm(scm, config)
    random_metrics = extract_trajectory_metrics_from_demo(random_results)
    
    # Oracle interventions
    print("\n2. Oracle interventions...")
    oracle_results = run_progressive_learning_demo_with_oracle_interventions(scm, config)
    oracle_metrics = extract_trajectory_metrics_from_demo(oracle_results)
    
    # Prepare data for comparison plot
    results_by_method = {
        "Random Interventions": {
            'steps': random_metrics['steps'],
            'shd_mean': random_metrics['shd_values'],
            'f1_mean': random_metrics['f1_scores'],
            'target_mean': random_metrics['target_values'],
            'n_runs': 1
        },
        "Oracle Interventions": {
            'steps': oracle_metrics['steps'],
            'shd_mean': oracle_metrics['shd_values'],
            'f1_mean': oracle_metrics['f1_scores'],
            'target_mean': oracle_metrics['target_values'],
            'n_runs': 1
        }
    }
    
    # Generate comparison plot
    print("\n📊 Generating comparison plot...")
    comparison_path = output_path / "method_comparison.png"
    plot_baseline_comparison(
        results_by_method,
        title="Random vs Oracle Intervention Comparison",
        save_path=str(comparison_path)
    )
    print(f"   ✅ Saved comparison plot: {comparison_path}")
    
    # Print comparison summary
    print("\n📝 Method Comparison Summary:")
    print(f"{'Method':<20} {'Final F1':<10} {'Final SHD':<10} {'Converged':<10}")
    print("-" * 50)
    print(f"{'Random':<20} {random_metrics['f1_scores'][-1]:<10.3f} {random_metrics['shd_values'][-1]:<10} {'Yes' if random_metrics['converged'] else 'No':<10}")
    print(f"{'Oracle':<20} {oracle_metrics['f1_scores'][-1]:<10.3f} {oracle_metrics['shd_values'][-1]:<10} {'Yes' if oracle_metrics['converged'] else 'No':<10}")


def visualize_difficulty_study(output_dir: str = "demo_plots"):
    """
    Run difficulty comparative study and generate visualizations.
    """
    # Create output directory
    output_path = Path(output_dir) / "difficulty_study"
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Configuration
    config = DemoConfig(
        n_observational_samples=20,
        n_intervention_steps=20,
        learning_rate=1e-3,
        random_seed=42
    )
    
    print("\n🔬 Running difficulty comparative study...")
    study_results = run_difficulty_comparative_study(config)
    
    # Extract and visualize each difficulty level
    for difficulty in ['easy', 'medium', 'hard']:
        print(f"\n📊 Visualizing {difficulty} difficulty...")
        
        demo_results = study_results['results'][difficulty]
        trajectory_metrics = extract_trajectory_metrics_from_demo(demo_results)
        
        # Generate dashboard for each difficulty
        dashboard_path = output_path / f"{difficulty}_dashboard.png"
        plot_structure_learning_dashboard(
            trajectory_metrics,
            title=f"Structure Learning - {difficulty.capitalize()} Difficulty",
            save_path=str(dashboard_path)
        )
        print(f"   ✅ Saved {difficulty} dashboard: {dashboard_path}")
    
    # Create combined comparison
    print("\n📊 Creating difficulty comparison plot...")
    
    # Prepare data for comparison
    results_by_method = {}
    for difficulty in ['easy', 'medium', 'hard']:
        demo_results = study_results['results'][difficulty]
        metrics = extract_trajectory_metrics_from_demo(demo_results)
        
        results_by_method[f"{difficulty.capitalize()} SCM"] = {
            'steps': metrics['steps'],
            'shd_mean': metrics['shd_values'],
            'f1_mean': metrics['f1_scores'],
            'target_mean': metrics['target_values'],
            'n_runs': 1
        }
    
    comparison_path = output_path / "difficulty_comparison.png"
    plot_baseline_comparison(
        results_by_method,
        title="Difficulty Level Comparison",
        save_path=str(comparison_path),
        figsize=(16, 12)
    )
    print(f"   ✅ Saved difficulty comparison: {comparison_path}")
    
    # Print summary
    print("\n📝 Difficulty Study Summary:")
    print(f"{'Difficulty':<10} {'Converged':<10} {'Final F1':<10} {'Final SHD':<10} {'Uncertainty':<12}")
    print("-" * 60)
    
    for difficulty in ['easy', 'medium', 'hard']:
        results = study_results['results'][difficulty]
        converged = results['converged_to_truth']['converged']
        final_f1 = results['converged_to_truth'].get('final_accuracy', 0.0)
        metrics = extract_trajectory_metrics_from_demo(results)
        final_shd = metrics['shd_values'][-1]
        final_uncertainty = results['final_uncertainty']
        
        print(f"{difficulty.capitalize():<10} {'Yes' if converged else 'No':<10} {final_f1:<10.3f} {final_shd:<10} {final_uncertainty:<12.2f}")


def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(
        description="Visualize ACBO demo experiment trajectories"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["single", "comparison", "difficulty", "all"],
        default="single",
        help="Type of experiment to run"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["random", "oracle"],
        default="random",
        help="Intervention method for single experiment"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="demo_plots",
        help="Directory to save plots"
    )
    
    args = parser.parse_args()
    
    print("🎨 ACBO Trajectory Visualization Tool")
    print("=" * 50)
    
    if args.experiment == "single":
        visualize_single_experiment(args.method, args.output_dir)
    elif args.experiment == "comparison":
        visualize_method_comparison(args.output_dir)
    elif args.experiment == "difficulty":
        visualize_difficulty_study(args.output_dir)
    elif args.experiment == "all":
        # Run all visualizations
        print("\n📊 Running all visualization experiments...")
        visualize_single_experiment("random", args.output_dir)
        visualize_single_experiment("oracle", args.output_dir)
        visualize_method_comparison(args.output_dir)
        visualize_difficulty_study(args.output_dir)
    
    print("\n✅ Visualization complete! Check the output directory for plots.")


if __name__ == "__main__":
    main()