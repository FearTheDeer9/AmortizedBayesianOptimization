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
        config.sparsity_weight = 0.05      # More balanced sparsity penalty
        config.acyclicity_weight = 1.0
        config.pos_weight = 2.0           # More moderate positive class weight
        config.consistency_weight = 0.1    # Add consistency regularization
        config.edge_prob_bias = 0.25      # Slightly higher bias for proof of concept
        config.expected_density = 0.2     # Lower expected density
        config.density_weight = 0.5   
        config.threshold = 0.25    # Lower, fixed threshold for edge detection
        config.epochs = 400        # Longer training for proof of concept
        config.num_int_samples = 100 # More samples per intervention for stronger signal
        config.temperature = 0.5    # Temperature for sigmoid scaling
        config.edge_temperature = 0.3  # Sharper edge probability temperature (lower = sharper)
    
    return config

def compare_strategies(seed=42):
    """Run only the random intervention strategy for debugging."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Generating random DAG and SCM...")
    adj_matrix = RandomDAGGenerator.generate_random_dag(
        num_nodes=4,
        edge_probability=0.3,
        seed=seed
    )
    print("True adjacency matrix:")
    print(adj_matrix)
    scm = LinearSCMGenerator.generate_linear_scm(
        adj_matrix=adj_matrix,
        noise_scale=0.1,
        seed=seed
    )
    print("Generating observational data...")
    obs_data = generate_observational_data(
        scm=scm,
        n_samples=100
    )
    # Only run random interventions for debugging
    print("\n=== Running experiment with random interventions ===")
    random_config = create_config("enhanced", "random", seed)
    random_loop = ProgressiveInterventionLoop(
        config=random_config,
        scm=scm,
        obs_data=obs_data,
        true_adj_matrix=adj_matrix
    )
    random_results = random_loop.run_experiment()
    # Diagnostic plots
    print("\n=== Creating diagnostic visualizations ===")
    random_loop.plot_graph_comparison()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = f"results/strategy_comparison_{timestamp}"
    os.makedirs(comparison_dir, exist_ok=True)
    # Edge probability distribution plot
    with torch.no_grad():
        random_probs = random_loop.model(random_loop.all_tensor).detach().cpu().numpy()
    plt.figure(figsize=(6, 5))
    plt.hist(random_probs.flatten(), bins=20, alpha=0.7, label='Random')
    plt.title("Random Intervention Edge Probabilities")
    plt.xlabel("Edge Probability")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "probability_distribution_random.png"))
    print(f"\nComparison visualizations saved to: {comparison_dir}")
    # Final metrics summary
    print("\n=== Final Metrics (Random Interventions) ===")
    final_metrics = random_results[-1]['metrics']
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'shd']:
        if metric in final_metrics:
            print(f"  {metric}: {final_metrics[metric]:.4f}")
    # Print raw edge probability matrix, edge differentiation, and true adjacency for final iteration
    print("\nFinal raw edge probability matrix:")
    with torch.no_grad():
        edge_probs = random_loop.model(random_loop.all_tensor).cpu().detach().numpy()
        np.set_printoptions(precision=3, suppress=True)
        print(edge_probs)
    print("\nTrue adjacency matrix:")
    print(adj_matrix)
    # Edge differentiation
    true_edge_mask = adj_matrix > 0
    false_edge_mask = (adj_matrix == 0) & (~np.eye(adj_matrix.shape[0], dtype=bool))
    true_edge_probs = edge_probs[true_edge_mask]
    false_edge_probs = edge_probs[false_edge_mask]
    mean_true_prob = true_edge_probs.mean() if len(true_edge_probs) > 0 else 0
    mean_false_prob = false_edge_probs.mean() if len(false_edge_probs) > 0 else 0
    separation = mean_true_prob - mean_false_prob
    print(f"\nEdge Differentiation:")
    print(f"  Mean probability for true edges: {mean_true_prob:.4f}")
    print(f"  Mean probability for false edges: {mean_false_prob:.4f}")
    print(f"  Separation: {separation:.4f}")
    # Print success/failure
    final_SHD = final_metrics.get('shd', None)
    if final_SHD is not None:
        print("\nSUCCESS" if final_SHD == 0 else "NOT PERFECT RECOVERY")
    return {
        'random': {'results': random_results, 'config': random_config},
        'comparison_dir': comparison_dir
    }

def main():
    print("Running random intervention experiment...")
    results = compare_strategies(seed=42)
    print("\nExperiment complete!")

if __name__ == "__main__":
    main()