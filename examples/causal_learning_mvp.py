"""
Causal Graph Structure Learning MVP

This script demonstrates learning a small graph's causal relationships through
strategic interventions, addressing the "no edge prediction bias" issue.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from causal_meta.structure_learning import (
    RandomDAGGenerator,
    LinearSCMGenerator,
    generate_observational_data,
    SimpleGraphLearner,
    ProgressiveInterventionConfig,
    ProgressiveInterventionLoop
)
from causal_meta.utils.advanced_visualization import (
    plot_edge_probabilities,
    plot_edge_probability_histogram,
    plot_edge_probability_distribution,
    compare_intervention_strategies
)

# Configuration for the experiment
def create_config(model_type="enhanced", strategy="uncertainty", seed=42):
    """Create configuration with specified model type and intervention strategy."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/{model_type}_model_{strategy}_{timestamp}"
    
    # Create base configuration
    config = ProgressiveInterventionConfig(
        num_nodes=4,                      # Small graph for MVP
        edge_probability=0.3,             # Moderate sparsity
        num_obs_samples=100,              # Start with 100 observational samples
        num_int_samples=20,               # 20 samples per intervention
        num_iterations=5,                 # 5 intervention iterations
        hidden_dim=64,                    # Hidden dimension for neural networks
        acquisition_strategy=strategy,    # "uncertainty" or "random"
        output_dir=output_dir,
        experiment_name=f"causal_mvp_{model_type}_{strategy}"
    )
    
    # Set model-specific parameters
    if model_type == "original":
        # Original model settings (with bias issue)
        config.sparsity_weight = 0.1
        config.acyclicity_weight = 1.0
        config.pos_weight = 1.0          # No class balancing
        config.consistency_weight = 0.0   # No consistency regularization
        config.edge_prob_bias = 0.0       # No bias adjustment
        config.expected_density = None    # No density regularization
        config.density_weight = 0.0
    else:
        # Enhanced model settings (addressing bias)
        config.sparsity_weight = 0.01     # Reduced sparsity weight
        config.acyclicity_weight = 1.0
        config.pos_weight = 5.0           # Weight positive examples 5x more
        config.consistency_weight = 0.1    # Add consistency regularization
        config.edge_prob_bias = 0.3       # Bias toward some edge predictions
        config.expected_density = 0.3     # Expected density around 30%
        config.density_weight = 0.5   
        config.threshold = 0.3    # Add density regularization
    
    return config

def compare_strategies(seed=42):
    """Compare strategic vs random intervention strategies."""
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create random DAG and SCM 
    print("Generating random DAG and SCM...")
    adj_matrix = RandomDAGGenerator.generate_random_dag(
        num_nodes=4,
        edge_probability=0.3,
        seed=seed
    )
    
    # Print the adjacency matrix
    print("True adjacency matrix:")
    print(adj_matrix)
    
    # Create SCM
    scm = LinearSCMGenerator.generate_linear_scm(
        adj_matrix=adj_matrix,
        noise_scale=0.1,
        seed=seed
    )
    
    # Generate observational data
    print("Generating observational data...")
    obs_data = generate_observational_data(
        scm=scm,
        n_samples=100
    )
    
    # Run experiment with strategic interventions (uncertainty-based)
    print("\n=== Running experiment with strategic interventions ===")
    strategic_config = create_config("enhanced", "uncertainty", seed)
    strategic_loop = ProgressiveInterventionLoop(
        config=strategic_config,
        scm=scm,
        obs_data=obs_data,
        true_adj_matrix=adj_matrix
    )
    strategic_results = strategic_loop.run_experiment()
    
    # Run experiment with random interventions
    print("\n=== Running experiment with random interventions ===")
    random_config = create_config("enhanced", "random", seed)
    random_loop = ProgressiveInterventionLoop(
        config=random_config,
        scm=scm,
        obs_data=obs_data,
        true_adj_matrix=adj_matrix
    )
    random_results = random_loop.run_experiment()
    
    # Create additional diagnostic visualizations
    print("\n=== Creating diagnostic visualizations ===")
    strategic_loop.plot_graph_comparison()
    random_loop.plot_graph_comparison()
    
    # Create comparison plots directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = f"results/strategy_comparison_{timestamp}"
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Compare learning trajectories and save to comparison directory
    strategic_metrics = {}
    random_metrics = {}
    
    # Extract metrics for visualization
    iterations = list(range(len(strategic_results)))
    metrics_to_track = ['accuracy', 'precision', 'recall', 'f1', 'shd']
    
    for metric in metrics_to_track:
        strategic_metrics[metric] = [r['metrics'].get(metric, 0) for r in strategic_results]
        random_metrics[metric] = [r['metrics'].get(metric, 0) for r in random_results]
    
    # Compare strategies with visualization
    compare_intervention_strategies(
        iterations=iterations,
        strategic_metrics=strategic_metrics,
        random_metrics=random_metrics,
        metrics_to_plot=metrics_to_track,
        save_path=os.path.join(comparison_dir, "strategy_comparison.png")
    )
    
    # Final performance comparison
    print("\n=== Strategy Performance Comparison ===")
    print("Strategic Intervention Final Metrics:")
    for metric, value in strategic_results[-1]['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nRandom Intervention Final Metrics:")
    for metric, value in random_results[-1]['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Extract final edge probabilities from both models
    with torch.no_grad():
        strategic_probs = strategic_loop.model(strategic_loop.all_tensor).detach().cpu().numpy()
        random_probs = random_loop.model(random_loop.all_tensor).detach().cpu().numpy()
    
    # Compare edge probability distributions
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(strategic_probs.flatten(), bins=20, alpha=0.7, label='Strategic')
    plt.title("Strategic Intervention Edge Probabilities")
    plt.xlabel("Edge Probability")
    plt.ylabel("Count")
    
    plt.subplot(1, 2, 2)
    plt.hist(random_probs.flatten(), bins=20, alpha=0.7, label='Random')
    plt.title("Random Intervention Edge Probabilities")
    plt.xlabel("Edge Probability")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "probability_distribution_comparison.png"))
    
    print(f"\nComparison visualizations saved to: {comparison_dir}")
    
    return {
        'strategic': {'results': strategic_results, 'config': strategic_config},
        'random': {'results': random_results, 'config': random_config},
        'comparison_dir': comparison_dir
    }

def main():
    """Run the MVP experiment."""
    print("Running strategy comparison experiment...")
    results = compare_strategies(seed=42)
    
    print("\nExperiment complete!")

if __name__ == "__main__":
    main()