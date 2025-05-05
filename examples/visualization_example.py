#!/usr/bin/env python3
"""
Example script demonstrating the use of visualization components for neural causal discovery and optimization.

This script shows how to use the visualization utilities to:
1. Visualize causal graph inference results
2. Compare intervention outcomes
3. Track optimization progress
4. Compare performance across different methods
5. Visualize predictions with uncertainty

To run this example:
    python examples/visualization_example.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error
import seaborn as sns

from causal_meta.graph import CausalGraph
from causal_meta.graph.generators.factory import GraphFactory
from causal_meta.environments.scm import StructuralCausalModel
from causal_meta.meta_learning import (
    plot_graph_inference_results,
    plot_intervention_outcomes,
    plot_optimization_progress,
    plot_performance_comparison,
    plot_uncertainty
)


def example_graph_inference():
    """Example showing how to visualize graph inference results."""
    print("\n=== Graph Inference Visualization Example ===")
    
    # Create a true graph
    print("Creating ground truth graph...")
    true_graph = GraphFactory.create_random_dag(num_nodes=5, edge_probability=0.3)
    
    # Create a "predicted" graph (with some differences)
    print("Creating predicted graph...")
    pred_graph = true_graph.copy()
    
    # Add an extra edge that's not in the true graph
    all_nodes = pred_graph.get_nodes()
    for i in range(len(all_nodes)):
        for j in range(i+1, len(all_nodes)):
            if not true_graph.has_edge(all_nodes[i], all_nodes[j]) and np.random.random() < 0.2:
                pred_graph.add_edge(all_nodes[i], all_nodes[j])
                break
    
    # Remove an edge that is in the true graph
    true_edges = true_graph.get_edges()
    if true_edges:
        edge_to_remove = true_edges[np.random.randint(0, len(true_edges))]
        if np.random.random() < 0.5 and len(true_edges) > 1:
            pred_graph.remove_edge(edge_to_remove[0], edge_to_remove[1])
    
    # Create edge probabilities (for the case where we have probabilities instead of a discrete graph)
    adj_matrix = true_graph.get_adjacency_matrix()
    
    # Add some noise to create "predicted" probabilities
    edge_probs = adj_matrix.astype(float)
    noise = np.random.normal(0, 0.2, size=edge_probs.shape)
    edge_probs = np.clip(edge_probs + noise, 0, 1)
    
    # Visualize results (graph comparison)
    print("Visualizing graph inference results...")
    plt.figure(figsize=(15, 5))
    ax1 = plot_graph_inference_results(
        true_graph, 
        pred_graph,
        metrics=True, 
        confusion_matrix=True,
        title="Graph Structure Inference Results"
    )
    plt.tight_layout()
    plt.savefig("graph_inference_discrete.png")
    print("Saved graph_inference_discrete.png")
    
    # Visualize results (edge probability comparison)
    plt.figure(figsize=(15, 5))
    ax2 = plot_graph_inference_results(
        true_graph, 
        edge_probs,
        threshold=0.5,
        metrics=True, 
        confusion_matrix=True,
        title="Edge Probability Inference Results"
    )
    plt.tight_layout()
    plt.savefig("graph_inference_probabilities.png")
    print("Saved graph_inference_probabilities.png")


def example_intervention_outcomes():
    """Example showing how to visualize intervention outcomes."""
    print("\n=== Intervention Outcomes Visualization Example ===")
    
    # Create synthetic data instead of using SCM for the example
    print("Creating synthetic data...")
    np.random.seed(42)
    
    # Create observational data
    obs_data = pd.DataFrame({
        "X": np.random.normal(0, 1, 500),
        "Y": np.random.normal(0, 1, 500),
        "Z": np.random.normal(0, 1, 500)
    })
    
    # Add causal relationships
    obs_data["Y"] = 0.8 * obs_data["X"] + np.random.normal(0, 0.5, 500)
    obs_data["Z"] = 0.7 * obs_data["Y"] + np.random.normal(0, 0.3, 500)
    
    # Create intervention data (Y set to 2.0)
    int_data = obs_data.copy()
    int_data["Y"] = 2.0
    int_data["Z"] = 0.7 * int_data["Y"] + np.random.normal(0, 0.3, 500)
    
    # Create "predicted" outcomes with slight errors
    print("Creating prediction data...")
    pred_data = int_data.copy()
    # Add some prediction error
    pred_data["Z"] = pred_data["Z"] + np.random.normal(0.2, 0.3, size=len(pred_data))
    
    # Visualize intervention outcomes with distributions
    print("Visualizing intervention outcomes with distributions...")
    plt.figure(figsize=(15, 10))
    ax1 = plot_intervention_outcomes(
        obs_data, int_data, pred_data,
        intervention_nodes=["Y"],
        show_distributions=True,
        show_errors=True,
        title="Intervention Outcomes (Y=2.0)"
    )
    plt.tight_layout()
    plt.savefig("intervention_outcomes_distributions.png")
    print("Saved intervention_outcomes_distributions.png")
    
    # Visualize intervention outcomes as bar chart
    print("Visualizing intervention outcomes as bar chart...")
    plt.figure(figsize=(10, 6))
    ax2 = plot_intervention_outcomes(
        obs_data, int_data, pred_data,
        intervention_nodes=["Y"],
        show_distributions=False,
        show_errors=True,
        title="Intervention Outcomes (Y=2.0) - Mean Values"
    )
    plt.tight_layout()
    plt.savefig("intervention_outcomes_means.png")
    print("Saved intervention_outcomes_means.png")


def example_optimization_progress():
    """Example showing how to visualize optimization progress."""
    print("\n=== Optimization Progress Visualization Example ===")
    
    # Create synthetic optimization data
    print("Creating synthetic optimization data...")
    np.random.seed(42)
    iterations = 15
    
    # Method 1: Our method with some fluctuations but overall improvement
    method1_data = []
    current_best = 0.5
    for i in range(iterations):
        # Value increases with some randomness
        value = current_best + np.random.normal(0.1, 0.15)
        current_best = max(current_best, value)
        
        # Add intervention info for some iterations
        intervention = None
        if i % 3 == 0 and i > 0:
            intervention = {"X": 1.0 + i * 0.1, "Y": -0.5 - i * 0.05}
        
        method1_data.append({
            "iteration": i,
            "value": value,
            "intervention": intervention
        })
    
    # Method 2: Baseline with slower improvement
    method2_data = []
    current_best = 0.4
    for i in range(iterations):
        # Value increases slower with some randomness
        value = current_best + np.random.normal(0.07, 0.1)
        current_best = max(current_best, value)
        
        method2_data.append({
            "iteration": i,
            "value": value
        })
    
    # Method 3: Another baseline with different pattern
    method3_data = []
    current_best = 0.3
    for i in range(iterations):
        # Different improvement pattern
        value = current_best + (0.2 if i > iterations/2 else 0.05) + np.random.normal(0, 0.05)
        current_best = max(current_best, value)
        
        method3_data.append({
            "iteration": i,
            "value": value
        })
    
    # Visualize optimization progress for our method
    print("Visualizing optimization progress...")
    plt.figure(figsize=(10, 6))
    ax1 = plot_optimization_progress(
        method1_data,
        target_variable="Target Function",
        objective="maximize",
        show_interventions=True,
        show_baseline=True,
        title="Optimization Progress"
    )
    plt.tight_layout()
    plt.savefig("optimization_progress.png")
    print("Saved optimization_progress.png")
    
    # Visualize comparative optimization progress
    print("Visualizing comparative optimization progress...")
    plt.figure(figsize=(10, 6))
    ax2 = plot_optimization_progress(
        method1_data,
        target_variable="Target Function",
        objective="maximize",
        comparison_methods={
            "Baseline 1": method2_data,
            "Baseline 2": method3_data
        },
        show_interventions=True,
        show_baseline=True,
        title="Comparative Optimization Progress"
    )
    plt.tight_layout()
    plt.savefig("comparative_optimization_progress.png")
    print("Saved comparative_optimization_progress.png")


def example_performance_comparison():
    """Example showing how to visualize performance comparison between methods."""
    print("\n=== Performance Comparison Visualization Example ===")
    
    # Create benchmark results for multiple methods
    print("Creating benchmark results...")
    benchmark_results = {
        "Our Method": {
            "accuracy": 0.88,
            "f1": 0.86,
            "recall": 0.84,
            "precision": 0.88,
            "shd": 2.5,
            "runtime": 12.3
        },
        "Baseline 1": {
            "accuracy": 0.82,
            "f1": 0.80,
            "recall": 0.79,
            "precision": 0.81,
            "shd": 3.7,
            "runtime": 8.2
        },
        "Baseline 2": {
            "accuracy": 0.85,
            "f1": 0.83,
            "recall": 0.82,
            "precision": 0.84,
            "shd": 3.0,
            "runtime": 15.1
        }
    }
    
    # Basic bar chart comparison
    print("Creating bar chart comparison...")
    plt.figure(figsize=(10, 6))
    ax1 = plot_performance_comparison(
        benchmark_results,
        plot_type="bar",
        metrics=["accuracy", "f1", "precision", "recall"],
        title="Performance Comparison (Bar Chart)"
    )
    plt.tight_layout()
    plt.savefig("performance_comparison_bar.png")
    print("Saved performance_comparison_bar.png")
    
    # Radar chart comparison
    print("Creating radar chart comparison...")
    plt.figure(figsize=(10, 8))
    ax2 = plot_performance_comparison(
        benchmark_results,
        plot_type="radar",
        metrics=["accuracy", "f1", "precision", "recall", "shd"],
        title="Performance Comparison (Radar Chart)"
    )
    plt.tight_layout()
    plt.savefig("performance_comparison_radar.png")
    print("Saved performance_comparison_radar.png")
    
    # Create boxplot data with multiple runs (simulated)
    print("Creating box plot comparison (with simulated multiple runs)...")
    np.random.seed(42)
    
    # Create simplified multiple run data instead of the complex nested structure
    box_plot_data = pd.DataFrame()
    
    # Add data for each method
    for method, values in benchmark_results.items():
        # For each metric, create 5 samples with slight variations
        for metric in ["accuracy", "f1", "recall", "precision"]:
            base_value = values[metric]
            # Create 5 samples with normal variation around the base value
            samples = [max(0, min(1, base_value + np.random.normal(0, 0.02))) for _ in range(5)]
            
            # Add each sample to the DataFrame
            for sample in samples:
                box_plot_data = pd.concat([box_plot_data, 
                                          pd.DataFrame({
                                              'Method': [method],
                                              'Metric': [metric],
                                              'Value': [sample]
                                          })], ignore_index=True)
    
    # Plot the boxplot directly using seaborn
    plt.figure(figsize=(12, 6))
    ax3 = sns.boxplot(data=box_plot_data, x='Metric', y='Value', hue='Method')
    plt.title("Performance Comparison Across Multiple Runs")
    plt.tight_layout()
    plt.savefig("performance_comparison_boxplot.png")
    print("Saved performance_comparison_boxplot.png")


def example_uncertainty_visualization():
    """Example showing how to visualize predictions with uncertainty."""
    print("\n=== Uncertainty Visualization Example ===")
    
    # Create synthetic data
    print("Creating synthetic data with uncertainty...")
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    
    # True function: sine wave with linear component
    true_function = lambda x: np.sin(x) + 0.1 * x
    y_true = true_function(x)
    
    # Add noise to create predictions
    y_pred = y_true + np.random.normal(0, 0.2, size=len(x))
    
    # Generate uncertainty estimates (higher in the middle)
    uncertainty = 0.1 + 0.2 * np.sin(x/5)
    
    # Visualize predictions with uncertainty
    print("Visualizing predictions with uncertainty...")
    plt.figure(figsize=(12, 6))
    ax = plot_uncertainty(
        x, 
        y_pred, 
        uncertainty, 
        true_values=y_true,
        confidence_level=0.95,
        title="Predictions with 95% Confidence Intervals",
        x_label="X Value",
        y_label="Prediction"
    )
    plt.tight_layout()
    plt.savefig("uncertainty_visualization.png")
    print("Saved uncertainty_visualization.png")
    
    # Create a more complex example with intervention effects
    print("Creating uncertainty visualization for interventions...")
    interventions = np.linspace(0, 2, 20)  # Different intervention values
    outcomes = []
    uncertainties = []
    true_outcomes = []
    
    # Generate outcomes for each intervention with uncertainty
    for intervention in interventions:
        # True effect is quadratic
        true_effect = 0.5 * intervention**2 + 0.2 * intervention
        true_outcomes.append(true_effect)
        
        # Predicted effect has some error
        pred_effect = true_effect + np.random.normal(0, 0.05 + 0.1 * intervention)
        outcomes.append(pred_effect)
        
        # Uncertainty increases with intervention magnitude
        uncertainties.append(0.05 + 0.1 * intervention)
    
    # Visualize intervention outcomes with uncertainty
    plt.figure(figsize=(12, 6))
    ax = plot_uncertainty(
        interventions,
        np.array(outcomes),
        np.array(uncertainties),
        true_values=np.array(true_outcomes),
        confidence_level=0.9,
        title="Intervention Effects with 90% Confidence Intervals",
        x_label="Intervention Value",
        y_label="Outcome"
    )
    plt.tight_layout()
    plt.savefig("intervention_uncertainty.png")
    print("Saved intervention_uncertainty.png")


def main():
    """Run all visualization examples."""
    print("Running visualization examples...")
    
    # Create output directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    # Change working directory to outputs
    os.chdir("outputs")
    
    # Run examples
    example_graph_inference()
    example_intervention_outcomes()
    example_optimization_progress()
    example_performance_comparison()
    example_uncertainty_visualization()
    
    print("\nAll examples completed. Output saved to the 'outputs' directory.")


if __name__ == "__main__":
    main()
