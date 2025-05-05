import pytest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from unittest.mock import patch, MagicMock

from causal_meta.graph import CausalGraph
from causal_meta.meta_learning.visualization import (
    plot_graph_inference_results,
    plot_intervention_outcomes,
    plot_optimization_progress,
    plot_performance_comparison,
    plot_uncertainty
)


class TestVisualization:
    
    @pytest.fixture
    def setup_graphs(self):
        # Create a simple true graph
        true_graph = CausalGraph()
        true_graph.add_node("A")
        true_graph.add_node("B")
        true_graph.add_node("C")
        true_graph.add_edge("A", "B")
        true_graph.add_edge("B", "C")
        
        # Create a different predicted graph
        pred_graph = CausalGraph()
        pred_graph.add_node("A")
        pred_graph.add_node("B")
        pred_graph.add_node("C")
        pred_graph.add_edge("A", "B")
        pred_graph.add_edge("A", "C")  # Different edge
        
        # Create edge probabilities
        edge_probs = np.zeros((3, 3))
        edge_probs[0, 1] = 0.9  # A->B: high probability
        edge_probs[0, 2] = 0.7  # A->C: medium probability
        edge_probs[1, 2] = 0.3  # B->C: low probability
        
        return true_graph, pred_graph, edge_probs
    
    @pytest.fixture
    def setup_data(self):
        # Create simple observational data
        obs_data = pd.DataFrame({
            "A": np.random.normal(0, 1, 100),
            "B": np.random.normal(0, 1, 100),
            "C": np.random.normal(0, 1, 100)
        })
        
        # Create intervention data (B has been intervened on)
        int_data = pd.DataFrame({
            "A": np.random.normal(0, 1, 100),
            "B": np.random.normal(2, 0.5, 100),  # Intervened
            "C": np.random.normal(1, 1, 100)     # Affected by intervention
        })
        
        # Create predicted outcomes (slightly off from true)
        pred_data = pd.DataFrame({
            "A": np.random.normal(0, 1, 100),
            "B": np.random.normal(2, 0.5, 100),
            "C": np.random.normal(1.2, 1.1, 100)  # Prediction slightly off
        })
        
        return obs_data, int_data, pred_data
    
    @pytest.fixture
    def setup_optimization_data(self):
        # Create optimization history data
        iterations = 10
        data = []
        
        for i in range(iterations):
            # Simulate increasing performance with some noise
            value = 0.5 + i * 0.2 + np.random.normal(0, 0.1)
            
            # Add intervention info for even iterations
            intervention = None
            if i % 2 == 0 and i > 0:
                intervention = {"B": 1.0 + i * 0.1}
            
            data.append({
                "iteration": i,
                "value": value,
                "intervention": intervention
            })
        
        # Create comparison method data (different performance curve)
        comparison_data = []
        for i in range(iterations):
            # Different performance profile
            value = 0.4 + i * 0.15 + np.random.normal(0, 0.1)
            comparison_data.append({
                "iteration": i,
                "value": value
            })
        
        return data, {"Baseline Method": comparison_data}
    
    @pytest.fixture
    def setup_benchmark_data(self):
        # Create benchmark results for multiple methods
        return {
            "Method1": {
                "accuracy": 0.85,
                "f1": 0.82,
                "recall": 0.80,
                "precision": 0.84,
                "runtime": 10.5
            },
            "Method2": {
                "accuracy": 0.82,
                "f1": 0.79,
                "recall": 0.75,
                "precision": 0.83,
                "runtime": 8.2
            },
            "Method3": {
                "accuracy": 0.90,
                "f1": 0.88,
                "recall": 0.87,
                "precision": 0.89,
                "runtime": 15.3
            }
        }
    
    @pytest.fixture
    def setup_uncertainty_data(self):
        # Create data for uncertainty visualization
        x = np.linspace(0, 10, 50)
        true_function = lambda x: np.sin(x) + 0.1 * x
        y_true = true_function(x)
        
        # Add noise to create predictions
        y_pred = y_true + np.random.normal(0, 0.2, size=len(x))
        
        # Generate uncertainty estimates (higher in the middle)
        uncertainty = 0.1 + 0.2 * np.sin(x/5)
        
        return x, y_pred, uncertainty, y_true
    
    @patch("matplotlib.pyplot.show")
    def test_plot_graph_inference_results(self, mock_show, setup_graphs):
        true_graph, pred_graph, edge_probs = setup_graphs
        
        # Test with CausalGraph as prediction
        ax1 = plot_graph_inference_results(true_graph, pred_graph, metrics=True, confusion_matrix=True)
        assert len(ax1) == 3  # Should have 3 subplots (true, pred, confusion)
        
        # Test with numpy array as prediction
        ax2 = plot_graph_inference_results(true_graph, edge_probs, threshold=0.5, metrics=True, confusion_matrix=False)
        assert len(ax2) == 2  # Should have 2 subplots (true, pred)
        
        # Test with torch tensor as prediction
        edge_probs_tensor = torch.tensor(edge_probs)
        ax3 = plot_graph_inference_results(true_graph, edge_probs_tensor, threshold=0.5)
        assert len(ax3) == 3  # Default includes confusion matrix
    
    @patch("matplotlib.pyplot.show")
    def test_plot_intervention_outcomes(self, mock_show, setup_data):
        obs_data, int_data, pred_data = setup_data
        
        # Test with pandas DataFrames
        ax1 = plot_intervention_outcomes(
            obs_data, int_data, pred_data, 
            intervention_nodes=["B"], 
            show_distributions=True
        )
        assert len(ax1) > 1  # Should have multiple subplots for distributions
        
        # Test with numpy arrays
        ax2 = plot_intervention_outcomes(
            obs_data.values, int_data.values, pred_data.values,
            node_names=obs_data.columns.tolist(),
            show_distributions=False
        )
        assert isinstance(ax2, plt.Axes)  # Should be a single axis
        
        # Test with torch tensors
        obs_tensor = torch.tensor(obs_data.values)
        int_tensor = torch.tensor(int_data.values)
        pred_tensor = torch.tensor(pred_data.values)
        
        ax3 = plot_intervention_outcomes(
            obs_tensor, int_tensor, pred_tensor,
            node_names=obs_data.columns.tolist(),
            show_errors=True
        )
        assert len(ax3) > 1  # Should have multiple subplots
    
    @patch("matplotlib.pyplot.show")
    def test_plot_optimization_progress(self, mock_show, setup_optimization_data):
        opt_data, comparison = setup_optimization_data
        
        # Basic test
        ax1 = plot_optimization_progress(
            opt_data, target_variable="Target", 
            objective="maximize"
        )
        assert isinstance(ax1, plt.Axes)
        
        # Test with comparison methods
        ax2 = plot_optimization_progress(
            opt_data, target_variable="Target",
            objective="maximize",
            comparison_methods=comparison,
            show_interventions=True,
            show_baseline=True
        )
        assert isinstance(ax2, plt.Axes)
        
        # Test with minimize objective
        ax3 = plot_optimization_progress(
            opt_data, target_variable="Target",
            objective="minimize"
        )
        assert isinstance(ax3, plt.Axes)
    
    @patch("matplotlib.pyplot.show")
    def test_plot_performance_comparison(self, mock_show, setup_benchmark_data):
        benchmark_results = setup_benchmark_data
        metrics = ["accuracy", "f1", "recall", "precision"]
        
        # Test bar plot
        ax1 = plot_performance_comparison(
            benchmark_results, metrics, 
            plot_type="bar"
        )
        assert isinstance(ax1, plt.Axes)
        
        # Test radar plot
        ax2 = plot_performance_comparison(
            benchmark_results, metrics, 
            plot_type="radar"
        )
        assert isinstance(ax2, plt.Axes)
        
        # Test box plot
        ax3 = plot_performance_comparison(
            benchmark_results, metrics, 
            plot_type="box"
        )
        assert isinstance(ax3, plt.Axes)
        
        # Test with invalid plot type
        with pytest.raises(ValueError):
            plot_performance_comparison(
                benchmark_results, metrics, 
                plot_type="invalid"
            )
    
    @patch("matplotlib.pyplot.show")
    def test_plot_uncertainty(self, mock_show, setup_uncertainty_data):
        x, y_pred, uncertainty, y_true = setup_uncertainty_data
        
        # Test with numpy arrays
        ax1 = plot_uncertainty(
            x, y_pred, uncertainty, 
            true_values=y_true,
            confidence_level=0.95
        )
        assert isinstance(ax1, plt.Axes)
        
        # Test with torch tensors
        x_tensor = torch.tensor(x)
        y_pred_tensor = torch.tensor(y_pred)
        uncertainty_tensor = torch.tensor(uncertainty)
        y_true_tensor = torch.tensor(y_true)
        
        ax2 = plot_uncertainty(
            x_tensor, y_pred_tensor, uncertainty_tensor,
            true_values=y_true_tensor
        )
        assert isinstance(ax2, plt.Axes)
        
        # Test without true values
        ax3 = plot_uncertainty(
            x, y_pred, uncertainty,
            x_label="X Label",
            y_label="Y Label"
        )
        assert isinstance(ax3, plt.Axes) 