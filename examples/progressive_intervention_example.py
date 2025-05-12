"""
Example of using progressive interventions for causal graph structure learning.

This script demonstrates how to:
1. Generate a random DAG and linear SCM
2. Create initial observational data
3. Configure and run a progressive intervention experiment
4. Compare strategic interventions with random interventions
5. Analyze and visualize the results
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd

from causal_meta.structure_learning import (
    RandomDAGGenerator,
    LinearSCMGenerator,
    generate_observational_data
)
from causal_meta.structure_learning.progressive_intervention import (
    ProgressiveInterventionConfig,
    ProgressiveInterventionLoop
)


def create_experiment_data(config: ProgressiveInterventionConfig) -> Dict[str, Any]:
    """
    Create experimental data for the progressive intervention experiment.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dictionary containing generated data and models
    """
    # Set random seed for reproducibility
    if config.random_seed is not None:
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
    
    # Generate a random DAG
    print(f"Generating random DAG with {config.num_nodes} nodes...")
    adj_matrix = RandomDAGGenerator.generate_random_dag(
        num_nodes=config.num_nodes,
        edge_probability=config.edge_probability,
        as_adjacency_matrix=True,
        seed=config.random_seed
    )
    print("Adjacency matrix:")
    print(adj_matrix)
    
    # Create a linear SCM
    print("\nCreating linear SCM...")
    scm = LinearSCMGenerator.generate_linear_scm(
        adj_matrix=adj_matrix,
        noise_scale=config.noise_scale,
        seed=config.random_seed
    )
    
    # Generate observational data
    print(f"\nGenerating {config.num_obs_samples} observational samples...")
    obs_data = generate_observational_data(
        scm=scm,
        n_samples=config.num_obs_samples,
        as_tensor=False
    )
    print(f"Observational data shape: {obs_data.shape}")
    
    return {
        'adj_matrix': adj_matrix,
        'scm': scm,
        'obs_data': obs_data
    }


def run_strategic_vs_random_comparison(
    config: ProgressiveInterventionConfig,
    data: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run a comparison between strategic and random intervention strategies.
    
    Args:
        config: Experiment configuration
        data: Dictionary with experimental data
        
    Returns:
        Dictionary with results from both strategies
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output_dir, f"comparison_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run strategic interventions
    print("\n" + "="*50)
    print("Running strategic intervention experiment")
    print("="*50)
    
    strategic_config = config.copy()
    strategic_config.experiment_name = "strategic_interventions"
    strategic_config.output_dir = output_dir
    
    strategic_loop = ProgressiveInterventionLoop(
        config=strategic_config,
        scm=data['scm'],
        obs_data=data['obs_data'],
        true_adj_matrix=data['adj_matrix']
    )
    
    strategic_results = strategic_loop.run_experiment()
    
    # Run random interventions
    print("\n" + "="*50)
    print("Running random intervention experiment")
    print("="*50)
    
    random_config = config.copy()
    random_config.experiment_name = "random_interventions"
    random_config.acquisition_strategy = "random"
    random_config.output_dir = output_dir
    
    random_loop = ProgressiveInterventionLoop(
        config=random_config,
        scm=data['scm'],
        obs_data=data['obs_data'],
        true_adj_matrix=data['adj_matrix']
    )
    
    random_results = random_loop.run_experiment()
    
    # Compare results
    compare_strategies(strategic_results, random_results, output_dir)
    
    return {
        'strategic': strategic_results,
        'random': random_results
    }


def compare_strategies(
    strategic_results: List[Dict[str, Any]],
    random_results: List[Dict[str, Any]],
    output_dir: str
) -> None:
    """
    Compare strategic and random intervention strategies.
    
    Args:
        strategic_results: Results from strategic intervention
        random_results: Results from random intervention
        output_dir: Directory to save comparison results
    """
    print("\n==================================================")
    print("Comparing intervention strategies")
    print("==================================================\n")
    
    # Extract final metrics
    strategic_final = strategic_results[-1]['metrics']
    random_final = random_results[-1]['metrics']
    
    # Prepare metrics for comparison
    metrics_to_compare = ['accuracy', 'shd']
    
    # Create a table for comparison
    print("Final Results:")
    print(f"{'Metric':<10} {'Strategic':<10} {'Random':<10} {'Improvement':<10}")
    print("-" * 40)
    
    # Results DataFrame for saving
    results_data = []
    
    for metric in metrics_to_compare:
        strategic_value = strategic_final.get(metric, 0)
        random_value = random_final.get(metric, 0)
        
        # Calculate improvement percentage, avoiding division by zero
        if random_value != 0:
            improvement = (strategic_value - random_value) / random_value * 100
        else:
            # Handle zero division case
            if strategic_value > random_value:
                improvement = 100.0  # Positive improvement
            elif strategic_value < random_value:
                improvement = -100.0  # Negative improvement
            else:
                improvement = 0.0  # No improvement (both are zero)
        
        # Print the comparison
        print(f"{metric:<10} {strategic_value:<10.4f} {random_value:<10.4f} {improvement:<10.2f}%")
        
        # Add to results data
        results_data.append({
            'metric': metric,
            'strategic': strategic_value,
            'random': random_value,
            'improvement': improvement
        })
    
    # Save results to CSV
    results_df = pd.DataFrame(results_data)
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, "comparison_results.csv"), index=False)
    
    # Create comparison plots
    create_comparison_plots(strategic_results, random_results, output_dir)


def create_comparison_plots(
    strategic_results: List[Dict[str, Any]],
    random_results: List[Dict[str, Any]],
    output_dir: str
) -> None:
    """
    Create comparison plots between strategic and random intervention strategies.
    
    Args:
        strategic_results: Results from strategic interventions
        random_results: Results from random interventions
        output_dir: Directory to save comparison plots
    """
    # Extract iterations and metrics
    iterations = [r['iteration'] for r in strategic_results]
    
    # Common metrics to compare
    metrics_to_compare = ['accuracy', 'shd', 'f1', 'precision', 'recall']
    
    # Create plots for each metric
    for metric in metrics_to_compare:
        if all(metric in r['metrics'] for r in strategic_results) and \
           all(metric in r['metrics'] for r in random_results):
            
            strategic_values = [r['metrics'][metric] for r in strategic_results]
            random_values = [r['metrics'][metric] for r in random_results]
            
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, strategic_values, 'b-o', label='Strategic Interventions')
            plt.plot(iterations, random_values, 'r-o', label='Random Interventions')
            plt.xlabel('Iteration')
            plt.ylabel(metric.capitalize())
            plt.title(f'{metric.capitalize()} Comparison')
            plt.grid(True)
            plt.legend()
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"comparison_{metric}.png"))
            plt.close()


def main():
    """Run the progressive intervention example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Progressive Intervention Example")
    parser.add_argument("--num_nodes", type=int, default=5, help="Number of nodes in the graph")
    parser.add_argument("--edge_prob", type=float, default=0.3, help="Edge probability")
    parser.add_argument("--noise_scale", type=float, default=0.1, help="Noise scale")
    parser.add_argument("--num_obs_samples", type=int, default=200, help="Number of observational samples")
    parser.add_argument("--num_int_samples", type=int, default=50, help="Number of interventional samples per iteration")
    parser.add_argument("--num_iterations", type=int, default=5, help="Number of intervention iterations")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size")
    parser.add_argument("--acquisition", type=str, default="uncertainty", 
                       choices=["uncertainty", "random", "information_gain"],
                       help="Acquisition strategy")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Create configuration
    config = ProgressiveInterventionConfig(
        num_nodes=args.num_nodes,
        edge_probability=args.edge_prob,
        noise_scale=args.noise_scale,
        hidden_dim=args.hidden_dim,
        num_iterations=args.num_iterations,
        num_obs_samples=args.num_obs_samples,
        num_int_samples=args.num_int_samples,
        acquisition_strategy=args.acquisition,
        random_seed=args.seed,
        output_dir=args.output_dir,
        experiment_name="progressive_intervention_example",
        # Add required parameters for training
        epochs=50,  # Reduce epochs for faster execution
        batch_size=8,  # Small batch size for small datasets
        learning_rate=0.005,  # Slightly higher learning rate
        sparsity_weight=0.1,
        acyclicity_weight=1.0
    )
    
    # Create experimental data
    data = create_experiment_data(config)
    
    # Run comparison experiment
    results = run_strategic_vs_random_comparison(config, data)
    
    print("\nExperiment completed! Results saved to:", config.output_dir)


if __name__ == "__main__":
    main() 