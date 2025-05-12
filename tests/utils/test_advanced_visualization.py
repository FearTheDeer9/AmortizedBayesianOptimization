import pytest
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys
from unittest.mock import patch, MagicMock

# Ensure the path includes the project root for imports
MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if MODULE_PATH not in sys.path:
    sys.path.insert(0, MODULE_PATH)

# Import the modules we want to test
# We'll be creating these functions in causal_meta/utils/advanced_visualization.py
from causal_meta.utils.advanced_visualization import (
    plot_edge_probabilities,
    plot_edge_probability_histogram,
    plot_threshold_sensitivity,
    compare_intervention_strategies,
    plot_edge_probability_distribution
)

# --- Test Fixtures ---

@pytest.fixture
def sample_edge_probabilities():
    """Generate sample edge probabilities."""
    # Create a 4x4 matrix of edge probabilities with a bias toward zeros
    np.random.seed(42)  # For reproducibility
    probs = np.random.beta(0.5, 2.0, (4, 4))  # Beta distribution skewed toward 0
    # Set diagonal to zero (no self-loops)
    np.fill_diagonal(probs, 0)
    return probs

@pytest.fixture
def sample_true_adjacency():
    """Generate sample true adjacency matrix."""
    # Create a 4x4 binary adjacency matrix
    adj = np.zeros((4, 4))
    # Add some edges
    adj[0, 1] = 1
    adj[1, 2] = 1
    adj[2, 3] = 1
    adj[0, 3] = 1
    return adj

@pytest.fixture
def sample_thresholded_adjacency(sample_edge_probabilities):
    """Generate sample thresholded adjacency matrix."""
    return (sample_edge_probabilities > 0.5).astype(float)

@pytest.fixture
def sample_metrics():
    """Generate sample metrics for strategic vs random interventions."""
    np.random.seed(42)
    # Create metrics for 5 iterations
    iterations = list(range(5))
    random_metrics = {
        'accuracy': np.linspace(0.7, 0.9, 5) + np.random.normal(0, 0.02, 5),
        'precision': np.linspace(0.4, 0.7, 5) + np.random.normal(0, 0.05, 5),
        'recall': np.linspace(0.3, 0.6, 5) + np.random.normal(0, 0.05, 5),
        'f1': np.linspace(0.35, 0.65, 5) + np.random.normal(0, 0.03, 5),
        'shd': np.linspace(3, 1, 5).astype(int)
    }
    strategic_metrics = {
        'accuracy': np.linspace(0.65, 0.85, 5) + np.random.normal(0, 0.02, 5),
        'precision': np.linspace(0.35, 0.6, 5) + np.random.normal(0, 0.05, 5),
        'recall': np.linspace(0.25, 0.5, 5) + np.random.normal(0, 0.05, 5),
        'f1': np.linspace(0.3, 0.55, 5) + np.random.normal(0, 0.03, 5),
        'shd': np.linspace(4, 2, 5).astype(int)
    }
    return iterations, random_metrics, strategic_metrics

# --- Test Cases ---

class TestAdvancedVisualization:
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_edge_probabilities(self, mock_close, mock_savefig, sample_true_adjacency, 
                                     sample_edge_probabilities, sample_thresholded_adjacency):
        """Test that plot_edge_probabilities correctly visualizes all three matrices."""
        # Call the function
        fig = plot_edge_probabilities(
            true_adj=sample_true_adjacency,
            thresholded_adj=sample_thresholded_adjacency,
            edge_probs=sample_edge_probabilities,
            save_path="test_output.png"
        )
        
        # Check that a figure was created
        assert isinstance(fig, plt.Figure)
        
        # Check that savefig was called with the correct path
        mock_savefig.assert_called_once_with("test_output.png")
        
        # Check that close was called
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_edge_probability_histogram(self, mock_close, mock_savefig, sample_edge_probabilities):
        """Test that plot_edge_probability_histogram correctly visualizes the distribution."""
        # Call the function
        fig = plot_edge_probability_histogram(
            edge_probs=sample_edge_probabilities,
            threshold=0.5,
            save_path="test_histogram.png"
        )
        
        # Check that a figure was created
        assert isinstance(fig, plt.Figure)
        
        # Check that savefig was called with the correct path
        mock_savefig.assert_called_once_with("test_histogram.png")
        
        # Check that close was called
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_threshold_sensitivity(self, mock_close, mock_savefig, sample_true_adjacency, 
                                       sample_edge_probabilities):
        """Test that plot_threshold_sensitivity correctly analyzes different thresholds."""
        # Call the function
        fig = plot_threshold_sensitivity(
            true_adj=sample_true_adjacency,
            edge_probs=sample_edge_probabilities,
            thresholds=np.linspace(0.1, 0.9, 9),
            save_path="test_threshold.png"
        )
        
        # Check that a figure was created
        assert isinstance(fig, plt.Figure)
        
        # Check that savefig was called with the correct path
        mock_savefig.assert_called_once_with("test_threshold.png")
        
        # Check that close was called
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_compare_intervention_strategies(self, mock_close, mock_savefig, sample_metrics):
        """Test that compare_intervention_strategies correctly compares random vs strategic."""
        # Unpack sample metrics
        iterations, random_metrics, strategic_metrics = sample_metrics
        
        # Call the function
        fig = compare_intervention_strategies(
            iterations=iterations,
            random_metrics=random_metrics,
            strategic_metrics=strategic_metrics,
            metrics_to_plot=['accuracy', 'precision', 'recall', 'f1'],
            save_path="test_comparison.png"
        )
        
        # Check that a figure was created
        assert isinstance(fig, plt.Figure)
        
        # Check that savefig was called with the correct path
        mock_savefig.assert_called_once_with("test_comparison.png")
        
        # Check that close was called
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_edge_probability_distribution(self, mock_close, mock_savefig, sample_true_adjacency, 
                                              sample_edge_probabilities):
        """Test that plot_edge_probability_distribution correctly shows probability distribution."""
        # Call the function
        fig = plot_edge_probability_distribution(
            true_adj=sample_true_adjacency,
            edge_probs=sample_edge_probabilities,
            save_path="test_distribution.png"
        )
        
        # Check that a figure was created
        assert isinstance(fig, plt.Figure)
        
        # Check that savefig was called with the correct path
        mock_savefig.assert_called_once_with("test_distribution.png")
        
        # Check that close was called
        mock_close.assert_called_once() 