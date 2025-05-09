#!/usr/bin/env python
"""
Evaluation script for structure recovery models

This script conducts experiments to compare different configurations of
the progressive structure recovery algorithm on various graph sizes.
"""

import os
import numpy as np
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import logging
import time
import json
from datetime import datetime

from demos.progressive_structure_recovery_demo import (
    EnhancedMAMLForCausalDiscovery,
    SimplifiedCausalDiscovery,
    progressive_structure_recovery,
    select_intervention_node,
    generate_random_linear_scm,
    get_assets_dir
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def run_experiment(
    experiment_name,
    graph_sizes=[3, 4, 5, 6, 8, 10],
    num_trials=5,
    max_interventions=30,
    intervention_strategies=['uncertainty', 'random', 'random_cyclic', 'parent_count', 'combined'],
    inner_lr=0.01,
    num_inner_steps=5,
    l2_reg_weight=0.01,
    sparsity_reg_weight=0.1,
    acyclicity_reg_weight=0.5,
    anneal_regularization=True,
    sample_size=1000,
    edge_probability=0.3,
    noise_scale=0.1,
    results_dir=None
):
    """
    Run a comprehensive experiment to evaluate structure recovery performance.
    
    Args:
        experiment_name: Name of the experiment
        graph_sizes: List of graph sizes to test
        num_trials: Number of trials per configuration
        max_interventions: Maximum interventions per trial
        intervention_strategies: List of intervention strategies to test
        inner_lr: Inner learning rate for MAML
        num_inner_steps: Number of inner adaptation steps
        l2_reg_weight: Weight for L2 regularization
        sparsity_reg_weight: Weight for sparsity regularization
        acyclicity_reg_weight: Weight for acyclicity regularization
        anneal_regularization: Whether to anneal regularization weights
        sample_size: Number of data samples
        edge_probability: Probability of edge creation
        noise_scale: Scale of the noise in the SCM
        results_dir: Directory to save results
        
    Returns:
        DataFrame with experiment results
    """
    # Set up results directory
    if results_dir is None:
        results_dir = os.path.join(get_assets_dir(), 'experiment_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Create timestamp for the experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_path = os.path.join(results_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_path, exist_ok=True)
    
    # Set up logging to file
    log_file = os.path.join(experiment_path, "experiment.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)
    
    # Log experiment parameters
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Graph sizes: {graph_sizes}")
    logger.info(f"Trials per configuration: {num_trials}")
    logger.info(f"Max interventions: {max_interventions}")
    logger.info(f"Intervention strategies: {intervention_strategies}")
    logger.info(f"MAML parameters: lr={inner_lr}, steps={num_inner_steps}")
    logger.info(f"Regularization: L2={l2_reg_weight}, Sparsity={sparsity_reg_weight}, Acyclicity={acyclicity_reg_weight}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create empty DataFrame for results
    results = []
    
    # Create progress bar for the entire experiment
    total_configs = len(graph_sizes) * len(intervention_strategies) * num_trials
    pbar = tqdm(total=total_configs, desc=f"Running {experiment_name}")
    
    # Run experiments for each configuration
    for graph_size in graph_sizes:
        for strategy in intervention_strategies:
            for trial in range(num_trials):
                # Set random seed for reproducibility
                seed = 42 + trial
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                
                # Generate synthetic data
                scm = generate_random_linear_scm(
                    num_nodes=graph_size,
                    edge_probability=edge_probability,
                    noise_scale=noise_scale
                )
                true_graph = scm.get_causal_graph()
                obs_data = scm.sample_data(sample_size=sample_size)
                
                # Create model with attention for larger graphs
                model = SimplifiedCausalDiscovery(
                    input_dim=graph_size,
                    hidden_dim=64,
                    num_layers=2,
                    dropout=0.1,
                    sparsity_weight=sparsity_reg_weight,
                    acyclicity_weight=acyclicity_reg_weight,
                    use_attention=(graph_size > 3),
                    num_heads=min(4, graph_size)
                ).to(device)
                
                # Create MAML wrapper
                maml_model = EnhancedMAMLForCausalDiscovery(
                    model=model,
                    inner_lr=inner_lr,
                    num_inner_steps=num_inner_steps,
                    l2_reg_weight=l2_reg_weight,
                    sparsity_reg_weight=sparsity_reg_weight,
                    acyclicity_reg_weight=acyclicity_reg_weight,
                    anneal_regularization=anneal_regularization,
                    device=device
                )
                
                # Start timer
                start_time = time.time()
                
                # Run progressive recovery
                trial_results = progressive_structure_recovery(
                    model=model,
                    scm=scm,
                    obs_data=obs_data,
                    true_graph=true_graph,
                    max_interventions=max_interventions,
                    device=device,
                    inner_lr=inner_lr,
                    num_inner_steps=num_inner_steps,
                    l2_reg_weight=l2_reg_weight,
                    sparsity_reg_weight=sparsity_reg_weight,
                    acyclicity_reg_weight=acyclicity_reg_weight,
                    intervention_strategy=strategy,
                    visualize=False
                )
                
                # End timer
                runtime = time.time() - start_time
                
                # Compute metrics
                initial_shd = trial_results['shd_history'][0]
                final_shd = trial_results['final_shd']
                best_shd = trial_results['best_shd']
                relative_improvement = 1.0 - (final_shd / initial_shd) if initial_shd > 0 else 1.0
                convergence_interventions = trial_results['interventions_to_converge'] if trial_results['converged'] else max_interventions
                
                # Save results for this trial
                result_entry = {
                    'graph_size': graph_size,
                    'strategy': strategy,
                    'trial': trial,
                    'seed': seed,
                    'initial_shd': initial_shd,
                    'final_shd': final_shd,
                    'best_shd': best_shd,
                    'converged': trial_results['converged'],
                    'interventions_to_converge': convergence_interventions,
                    'relative_improvement': relative_improvement,
                    'runtime': runtime,
                    'inner_lr': inner_lr,
                    'num_inner_steps': num_inner_steps,
                    'l2_reg_weight': l2_reg_weight,
                    'sparsity_reg_weight': sparsity_reg_weight,
                    'acyclicity_reg_weight': acyclicity_reg_weight,
                    'anneal_regularization': anneal_regularization
                }
                results.append(result_entry)
                
                # Save additional data for detailed analysis
                trial_path = os.path.join(experiment_path, f"size{graph_size}_strategy{strategy}_trial{trial}")
                os.makedirs(trial_path, exist_ok=True)
                
                # Save intervention history
                if 'intervention_nodes' in trial_results:
                    with open(os.path.join(trial_path, "interventions.json"), "w") as f:
                        json.dump({
                            'nodes': trial_results['intervention_nodes'],
                            'shd_history': trial_results['shd_history']
                        }, f)
                
                # Log results of this trial
                logger.info(f"Trial {trial}: Graph size {graph_size}, Strategy {strategy}, Converged: {trial_results['converged']}, Final SHD: {final_shd}")
                
                # Update progress bar
                pbar.update(1)
    
    # Close progress bar
    pbar.close()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save full results to CSV
    results_csv = os.path.join(experiment_path, "results.csv")
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Saved full results to {results_csv}")
    
    # Create and save summary plots
    create_summary_plots(results_df, experiment_path)
    
    # Log summary statistics
    log_summary_statistics(results_df)
    
    return results_df

def create_summary_plots(results_df, output_dir):
    """
    Create summary plots from experiment results.
    
    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save plots
    """
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create averaged results by graph size and strategy
    summary = results_df.groupby(['graph_size', 'strategy']).agg({
        'initial_shd': 'mean',
        'final_shd': 'mean',
        'best_shd': 'mean',
        'converged': 'mean',
        'interventions_to_converge': 'mean',
        'relative_improvement': 'mean',
        'runtime': 'mean'
    }).reset_index()
    
    # Plot 1: Convergence rate by graph size and strategy
    plt.figure(figsize=(12, 8))
    graph_sizes = sorted(results_df['graph_size'].unique())
    strategies = sorted(results_df['strategy'].unique())
    
    for strategy in strategies:
        strategy_data = summary[summary['strategy'] == strategy]
        plt.plot(strategy_data['graph_size'], strategy_data['converged'], 'o-', label=strategy)
    
    plt.xlabel('Graph Size (Nodes)')
    plt.ylabel('Convergence Rate')
    plt.title('Convergence Rate by Graph Size and Strategy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "convergence_rate.png"), dpi=150)
    
    # Plot 2: Interventions to convergence by graph size and strategy
    plt.figure(figsize=(12, 8))
    
    for strategy in strategies:
        strategy_data = summary[summary['strategy'] == strategy]
        plt.plot(strategy_data['graph_size'], strategy_data['interventions_to_converge'], 'o-', label=strategy)
    
    plt.xlabel('Graph Size (Nodes)')
    plt.ylabel('Interventions to Convergence')
    plt.title('Interventions Required for Convergence by Graph Size and Strategy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "interventions_to_convergence.png"), dpi=150)
    
    # Plot 3: Relative improvement by graph size and strategy
    plt.figure(figsize=(12, 8))
    
    for strategy in strategies:
        strategy_data = summary[summary['strategy'] == strategy]
        plt.plot(strategy_data['graph_size'], strategy_data['relative_improvement'] * 100, 'o-', label=strategy)
    
    plt.xlabel('Graph Size (Nodes)')
    plt.ylabel('Relative Improvement (%)')
    plt.title('Relative SHD Improvement by Graph Size and Strategy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "relative_improvement.png"), dpi=150)
    
    # Plot 4: Runtime by graph size and strategy
    plt.figure(figsize=(12, 8))
    
    for strategy in strategies:
        strategy_data = summary[summary['strategy'] == strategy]
        plt.plot(strategy_data['graph_size'], strategy_data['runtime'], 'o-', label=strategy)
    
    plt.xlabel('Graph Size (Nodes)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime by Graph Size and Strategy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "runtime.png"), dpi=150)
    
    # Plot 5: Initial vs Final SHD by graph size and strategy
    plt.figure(figsize=(15, 10))
    
    for i, strategy in enumerate(strategies):
        plt.subplot(1, len(strategies), i+1)
        strategy_data = summary[summary['strategy'] == strategy]
        
        x = strategy_data['graph_size']
        initial_shd = strategy_data['initial_shd']
        final_shd = strategy_data['final_shd']
        
        plt.plot(x, initial_shd, 'r-o', label='Initial SHD')
        plt.plot(x, final_shd, 'g-o', label='Final SHD')
        plt.xlabel('Graph Size (Nodes)')
        plt.ylabel('SHD')
        plt.title(f'Strategy: {strategy}')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "initial_vs_final_shd.png"), dpi=150)
    
    # Save summary as CSV
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)

def log_summary_statistics(results_df):
    """
    Log summary statistics from experiment results.
    
    Args:
        results_df: DataFrame with experiment results
    """
    # Overall statistics
    logger.info("\n=== Overall Summary Statistics ===")
    logger.info(f"Total trials: {len(results_df)}")
    logger.info(f"Average convergence rate: {results_df['converged'].mean():.2f}")
    logger.info(f"Average relative improvement: {results_df['relative_improvement'].mean() * 100:.2f}%")
    
    # Statistics by graph size
    logger.info("\n=== Summary by Graph Size ===")
    size_summary = results_df.groupby('graph_size').agg({
        'converged': 'mean',
        'interventions_to_converge': 'mean',
        'relative_improvement': 'mean'
    })
    
    for size, row in size_summary.iterrows():
        logger.info(f"Graph size {size}: {row['converged'] * 100:.1f}% convergence, " +
                    f"{row['interventions_to_converge']:.1f} interventions, " +
                    f"{row['relative_improvement'] * 100:.1f}% improvement")
    
    # Statistics by strategy
    logger.info("\n=== Summary by Strategy ===")
    strategy_summary = results_df.groupby('strategy').agg({
        'converged': 'mean',
        'interventions_to_converge': 'mean',
        'relative_improvement': 'mean'
    })
    
    for strategy, row in strategy_summary.iterrows():
        logger.info(f"Strategy {strategy}: {row['converged'] * 100:.1f}% convergence, " +
                    f"{row['interventions_to_converge']:.1f} interventions, " +
                    f"{row['relative_improvement'] * 100:.1f}% improvement")

def main():
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate structure recovery algorithms")
    
    # Experiment configuration
    parser.add_argument('--experiment-name', type=str, default="structure_recovery_evaluation",
                      help="Name of the experiment")
    parser.add_argument('--graph-sizes', type=int, nargs='+', default=[3, 4, 5, 6, 8],
                      help="Graph sizes to test")
    parser.add_argument('--num-trials', type=int, default=5,
                      help="Number of trials per configuration")
    parser.add_argument('--max-interventions', type=int, default=30,
                      help="Maximum number of interventions per trial")
    parser.add_argument('--intervention-strategies', type=str, nargs='+', 
                      default=['uncertainty', 'random', 'random_cyclic', 'parent_count', 'combined'],
                      help="Intervention strategies to test")
    
    # Model parameters
    parser.add_argument('--inner-lr', type=float, default=0.01,
                      help="Inner learning rate for MAML")
    parser.add_argument('--num-inner-steps', type=int, default=5,
                      help="Number of inner adaptation steps")
    parser.add_argument('--l2-reg-weight', type=float, default=0.01,
                      help="Weight for L2 regularization")
    parser.add_argument('--sparsity-reg-weight', type=float, default=0.1,
                      help="Weight for sparsity regularization")
    parser.add_argument('--acyclicity-reg-weight', type=float, default=0.5,
                      help="Weight for acyclicity regularization")
    parser.add_argument('--anneal-regularization', action='store_true',
                      help="Whether to anneal regularization weights")
    
    # Data generation parameters
    parser.add_argument('--sample-size', type=int, default=1000,
                      help="Number of samples to generate")
    parser.add_argument('--edge-probability', type=float, default=0.3,
                      help="Probability of edge creation in random graphs")
    parser.add_argument('--noise-scale', type=float, default=0.1,
                      help="Scale of the noise in the SCM")
    
    # Output parameters
    parser.add_argument('--results-dir', type=str, default=None,
                      help="Directory to save results")
    
    args = parser.parse_args()
    
    # Run the experiment
    results_df = run_experiment(
        experiment_name=args.experiment_name,
        graph_sizes=args.graph_sizes,
        num_trials=args.num_trials,
        max_interventions=args.max_interventions,
        intervention_strategies=args.intervention_strategies,
        inner_lr=args.inner_lr,
        num_inner_steps=args.num_inner_steps,
        l2_reg_weight=args.l2_reg_weight,
        sparsity_reg_weight=args.sparsity_reg_weight,
        acyclicity_reg_weight=args.acyclicity_reg_weight,
        anneal_regularization=args.anneal_regularization,
        sample_size=args.sample_size,
        edge_probability=args.edge_probability,
        noise_scale=args.noise_scale,
        results_dir=args.results_dir
    )
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main() 