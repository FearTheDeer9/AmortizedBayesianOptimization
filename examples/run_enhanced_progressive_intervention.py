"""
Script to run progressive intervention with both original and enhanced models.

This script compares the performance of the original SimpleGraphLearner model
(which suffers from "no edge" prediction bias) with our enhanced model that
includes class-balanced loss, bias initialization, consistency regularization,
and density regularization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from causal_meta.structure_learning.config import ProgressiveInterventionConfig
from causal_meta.structure_learning.progressive_intervention import ProgressiveInterventionLoop
from causal_meta.structure_learning.scm_generators import LinearSCMGenerator
from causal_meta.graph.generators.random_graphs import RandomGraphGenerator
from causal_meta.structure_learning.data_utils import generate_observational_data
from causal_meta.utils.advanced_visualization import (
    plot_edge_probabilities,
    plot_edge_probability_histogram,
    plot_edge_probability_distribution,
    plot_threshold_sensitivity,
    compare_intervention_strategies
)

def create_config(model_type="enhanced", seed=42):
    """
    Create configuration for the experiment.
    
    Args:
        model_type: Type of model to use ('enhanced' or 'original')
        seed: Random seed for reproducibility
        
    Returns:
        ProgressiveInterventionConfig instance
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/{model_type}_model_{timestamp}"
    
    # Base configuration shared between both models
    config = ProgressiveInterventionConfig(
        num_nodes=5,
        edge_probability=0.3,
        num_obs_samples=1000,
        num_int_samples=100,
        num_iterations=5,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        learning_rate=0.001,
        weight_decay=0.0,
        batch_size=32,
        epochs=100,
        early_stopping_patience=10,
        acquisition_strategy="uncertainty",  # Use strategic intervention selection
        random_seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
        threshold=0.5,
        output_dir=output_dir,
        experiment_name=f"progressive_intervention_{model_type}"
    )
    
    # Set model-specific parameters
    if model_type == "original":
        # Original model settings
        config.sparsity_weight = 0.1
        config.acyclicity_weight = 1.0
        config.pos_weight = 1.0  # No class balancing
        config.consistency_weight = 0.0  # No consistency regularization
        config.edge_prob_bias = 0.0  # No bias adjustment
        config.expected_density = None  # No density regularization
        config.density_weight = 0.0
    else:
        # Enhanced model settings
        config.sparsity_weight = 0.05  # Reduced sparsity weight
        config.acyclicity_weight = 1.0
        config.pos_weight = 5.0  # Weight positive examples 5x more
        config.consistency_weight = 0.1  # Add consistency regularization
        config.edge_prob_bias = 0.1  # Bias toward some edge predictions
        config.expected_density = 0.3  # Expected density around 30%
        config.density_weight = 0.1  # Add density regularization
    
    return config

def run_experiment(model_type="enhanced", seed=42):
    """
    Run progressive intervention experiment with specified model type.
    
    Args:
        model_type: Type of model to use ('enhanced' or 'original')
        seed: Random seed for reproducibility
        
    Returns:
        Results of the experiment
    """
    # Set random seeds
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    # Create configuration
    config = create_config(model_type, seed)
    
    # Generate random DAG
    causal_graph = RandomGraphGenerator.random_dag(
        num_nodes=config.num_nodes,
        edge_probability=config.edge_probability,
        seed=seed
    )
    
    # Convert to adjacency matrix
    adjacency_matrix = np.zeros((config.num_nodes, config.num_nodes))
    for i in range(config.num_nodes):
        for j in causal_graph.get_children(i):
            adjacency_matrix[i, j] = 1
    
    # Create SCM
    scm = LinearSCMGenerator.generate_linear_scm(
        adj_matrix=adjacency_matrix,
        noise_scale=0.1,
        seed=seed
    )
    
    # Generate observational data
    obs_data = generate_observational_data(
        scm=scm,
        n_samples=config.num_obs_samples
    )
    
    # Convert data to numpy array
    data_numpy = np.column_stack([obs_data[f'x{i}'] for i in range(config.num_nodes)])
    
    # Create progressive intervention loop
    intervention_loop = ProgressiveInterventionLoop(
        config=config,
        scm=scm,
        obs_data=data_numpy,
        true_adj_matrix=adjacency_matrix
    )
    
    # Run experiment
    results = intervention_loop.run_experiment()
    
    # Plot and save results
    intervention_loop.plot_graph_comparison()
    intervention_loop.plot_metrics()
    intervention_loop.analyze_threshold_sensitivity()
    intervention_loop.save_results()
    
    # Print final metrics
    final_metrics = results[-1]['metrics']
    print(f"\n{model_type.capitalize()} Model Final Metrics:")
    for metric, value in final_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return results, config.output_dir

def compare_models(seed=42):
    """
    Run both models and compare their performance.
    
    Args:
        seed: Random seed for reproducibility
    """
    print("Running experiment with original model...")
    original_results, original_dir = run_experiment("original", seed)
    
    print("\nRunning experiment with enhanced model...")
    enhanced_results, enhanced_dir = run_experiment("enhanced", seed)
    
    # Extract metrics for comparison
    iterations = list(range(len(original_results)))
    
    original_metrics = {
        'accuracy': [r['metrics']['accuracy'] for r in original_results],
        'precision': [r['metrics']['precision'] for r in original_results],
        'recall': [r['metrics']['recall'] for r in original_results],
        'f1': [r['metrics']['f1'] for r in original_results],
        'shd': [r['metrics']['shd'] for r in original_results]
    }
    
    enhanced_metrics = {
        'accuracy': [r['metrics']['accuracy'] for r in enhanced_results],
        'precision': [r['metrics']['precision'] for r in enhanced_results],
        'recall': [r['metrics']['recall'] for r in enhanced_results],
        'f1': [r['metrics']['f1'] for r in enhanced_results],
        'shd': [r['metrics']['shd'] for r in enhanced_results]
    }
    
    # Create comparison plots
    comparison_dir = f"results/model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Plot comparison of metrics
    compare_intervention_strategies(
        iterations=iterations,
        random_metrics=original_metrics,  # Using "random" slot for original model
        strategic_metrics=enhanced_metrics,  # Using "strategic" slot for enhanced model
        metrics_to_plot=['accuracy', 'precision', 'recall', 'f1', 'shd'],
        save_path=os.path.join(comparison_dir, "model_comparison.png")
    )
    
    # Save comparison metrics
    with open(os.path.join(comparison_dir, "comparison_summary.txt"), "w") as f:
        f.write("=== Model Comparison Summary ===\n\n")
        f.write("Original vs Enhanced Model\n\n")
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'shd']:
            original_final = original_metrics[metric][-1]
            enhanced_final = enhanced_metrics[metric][-1]
            improvement = enhanced_final - original_final
            improvement_pct = (improvement / original_final) * 100 if original_final != 0 else float('inf')
            
            f.write(f"{metric.upper()}:\n")
            f.write(f"  Original: {original_final:.4f}\n")
            f.write(f"  Enhanced: {enhanced_final:.4f}\n")
            f.write(f"  Improvement: {improvement:.4f} ({improvement_pct:.2f}%)\n\n")
    
    print(f"\nComparison results saved to {comparison_dir}")
    
    return {
        'original': {'results': original_results, 'output_dir': original_dir},
        'enhanced': {'results': enhanced_results, 'output_dir': enhanced_dir},
        'comparison_dir': comparison_dir
    }

if __name__ == "__main__":
    import torch
    compare_models(seed=42) 