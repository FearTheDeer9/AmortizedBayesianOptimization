"""
Tests for plots.py

Tests for visualization functions ensuring proper plot generation,
styling, and error handling.
"""

import pytest
import numpy as onp
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.causal_bayes_opt.visualization.plots import (
    plot_convergence,
    plot_target_optimization,
    plot_method_comparison,
    plot_intervention_efficiency,
    plot_calibration_curves,
    plot_precision_recall_curves,
    create_experiment_dashboard,
    save_all_plots
)


@pytest.fixture
def sample_trajectory_metrics():
    """Sample trajectory metrics for testing."""
    return {
        'steps': [1, 2, 3, 4, 5],
        'true_parent_likelihood': [0.1, 0.3, 0.5, 0.7, 0.9],
        'f1_scores': [0.0, 0.2, 0.4, 0.6, 0.8],
        'target_values': [1.0, 1.2, 1.5, 1.8, 2.0],
        'uncertainties': [3.0, 2.5, 2.0, 1.5, 1.0],
        'rewards': [0, 1, 0, 1, 1]
    }


@pytest.fixture
def sample_learning_curves():
    """Sample learning curves for method comparison."""
    return {
        'static_surrogate': {
            'mean_f1': [0.0, 0.1, 0.2, 0.3, 0.4],
            'std_f1': [0.05, 0.05, 0.05, 0.05, 0.05],
            'mean_likelihood': [0.1, 0.2, 0.3, 0.4, 0.5],
            'std_likelihood': [0.1, 0.1, 0.1, 0.1, 0.1],
            'steps': [1, 2, 3, 4, 5]
        },
        'learning_surrogate': {
            'mean_f1': [0.0, 0.3, 0.5, 0.7, 0.8],
            'std_f1': [0.05, 0.1, 0.1, 0.1, 0.05],
            'mean_likelihood': [0.1, 0.4, 0.6, 0.8, 0.9],
            'std_likelihood': [0.1, 0.15, 0.15, 0.1, 0.05],
            'steps': [1, 2, 3, 4, 5]
        }
    }


@pytest.fixture
def temp_dir():
    """Temporary directory for saving plots."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestPlotConvergence:
    """Test convergence plotting functionality."""
    
    def test_plot_convergence_basic(self, sample_trajectory_metrics):
        """Test basic convergence plot generation."""
        fig = plot_convergence(sample_trajectory_metrics)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1, "Should have at least one subplot"
        plt.close(fig)
    
    def test_plot_convergence_with_save(self, sample_trajectory_metrics, temp_dir):
        """Test convergence plot with file saving."""
        save_path = os.path.join(temp_dir, "convergence_test.png")
        
        fig = plot_convergence(
            sample_trajectory_metrics,
            title="Test Convergence",
            save_path=save_path
        )
        
        assert os.path.exists(save_path), "Plot file should be saved"
        assert os.path.getsize(save_path) > 0, "Plot file should not be empty"
        plt.close(fig)
    
    def test_plot_convergence_options(self, sample_trajectory_metrics):
        """Test convergence plot with different options."""
        fig = plot_convergence(
            sample_trajectory_metrics,
            show_f1=False,
            show_uncertainty=False
        )
        
        # Should still generate a plot even with options disabled
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_convergence_empty_data(self):
        """Test convergence plot with empty data."""
        empty_metrics = {
            'steps': [],
            'true_parent_likelihood': [],
            'f1_scores': [],
            'target_values': [],
            'uncertainties': []
        }
        
        fig = plot_convergence(empty_metrics)
        
        # Should handle empty data gracefully
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_convergence_missing_fields(self):
        """Test convergence plot with missing optional fields."""
        minimal_metrics = {
            'steps': [1, 2, 3],
            'true_parent_likelihood': [0.1, 0.5, 0.8]
        }
        
        fig = plot_convergence(minimal_metrics)
        
        # Should work with minimal required fields
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotTargetOptimization:
    """Test target optimization plotting functionality."""
    
    def test_plot_target_optimization_basic(self, sample_trajectory_metrics):
        """Test basic target optimization plot."""
        fig = plot_target_optimization(sample_trajectory_metrics)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1
        plt.close(fig)
    
    def test_plot_target_optimization_with_rewards(self, sample_trajectory_metrics):
        """Test target optimization plot with reward signals."""
        fig = plot_target_optimization(
            sample_trajectory_metrics,
            show_rewards=True
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_target_optimization_save(self, sample_trajectory_metrics, temp_dir):
        """Test target optimization plot saving."""
        save_path = os.path.join(temp_dir, "target_opt_test.png")
        
        fig = plot_target_optimization(
            sample_trajectory_metrics,
            save_path=save_path
        )
        
        assert os.path.exists(save_path)
        plt.close(fig)
    
    def test_plot_target_optimization_missing_target_values(self):
        """Test target optimization plot when target values are missing."""
        metrics_no_target = {
            'steps': [1, 2, 3],
            'true_parent_likelihood': [0.1, 0.5, 0.8]
        }
        
        fig = plot_target_optimization(metrics_no_target)
        
        # Should handle missing target values
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotMethodComparison:
    """Test method comparison plotting functionality."""
    
    def test_plot_method_comparison_basic(self, sample_learning_curves):
        """Test basic method comparison plot."""
        fig = plot_method_comparison(sample_learning_curves)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1
        plt.close(fig)
    
    def test_plot_method_comparison_save(self, sample_learning_curves, temp_dir):
        """Test method comparison plot saving."""
        save_path = os.path.join(temp_dir, "method_comparison_test.png")
        
        fig = plot_method_comparison(
            sample_learning_curves,
            title="Test Method Comparison",
            save_path=save_path
        )
        
        assert os.path.exists(save_path)
        plt.close(fig)
    
    def test_plot_method_comparison_single_method(self):
        """Test method comparison with single method."""
        single_method = {
            'only_method': {
                'mean_f1': [0.0, 0.2, 0.4],
                'std_f1': [0.05, 0.05, 0.05],
                'steps': [1, 2, 3]
            }
        }
        
        fig = plot_method_comparison(single_method)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_method_comparison_empty_curves(self):
        """Test method comparison with empty learning curves."""
        fig = plot_method_comparison({})
        
        # Should handle empty input gracefully
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_method_comparison_missing_std(self):
        """Test method comparison when standard deviation is missing."""
        curves_no_std = {
            'method1': {
                'mean_f1': [0.0, 0.2, 0.4],
                'steps': [1, 2, 3]
            }
        }
        
        fig = plot_method_comparison(curves_no_std)
        
        # Should work without confidence intervals
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotInterventionEfficiency:
    """Test intervention efficiency plotting functionality."""
    
    def test_plot_intervention_efficiency_basic(self, sample_learning_curves):
        """Test basic intervention efficiency plot."""
        fig = plot_intervention_efficiency(sample_learning_curves)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_intervention_efficiency_save(self, sample_learning_curves, temp_dir):
        """Test intervention efficiency plot saving."""
        save_path = os.path.join(temp_dir, "efficiency_test.png")
        
        fig = plot_intervention_efficiency(
            sample_learning_curves,
            save_path=save_path
        )
        
        assert os.path.exists(save_path)
        plt.close(fig)


class TestCreateExperimentDashboard:
    """Test experiment dashboard creation."""
    
    def test_create_dashboard_basic(self, sample_trajectory_metrics, sample_learning_curves):
        """Test basic dashboard creation."""
        analysis_results = {
            'trajectory_results': {
                'method1': [{'trajectory_metrics': sample_trajectory_metrics}]
            },
            'learning_curves': sample_learning_curves
        }
        
        fig = create_experiment_dashboard(analysis_results)
        
        assert isinstance(fig, plt.Figure)
        # Dashboard should have multiple subplots
        assert len(fig.axes) >= 2
        plt.close(fig)
    
    def test_create_dashboard_save(self, sample_trajectory_metrics, sample_learning_curves, temp_dir):
        """Test dashboard saving."""
        analysis_results = {
            'trajectory_results': {
                'method1': [{'trajectory_metrics': sample_trajectory_metrics}]
            },
            'learning_curves': sample_learning_curves
        }
        save_path = os.path.join(temp_dir, "dashboard_test.png")
        
        fig = create_experiment_dashboard(
            analysis_results,
            save_path=save_path
        )
        
        assert os.path.exists(save_path)
        plt.close(fig)


class TestSaveAllPlots:
    """Test batch plot saving functionality."""
    
    def test_save_all_plots_basic(self, sample_trajectory_metrics, sample_learning_curves, temp_dir):
        """Test saving all plots in batch."""
        analysis_results = {
            'trajectory_results': {
                'static_surrogate': [{'trajectory_metrics': sample_trajectory_metrics}],
                'learning_surrogate': [{'trajectory_metrics': sample_trajectory_metrics}]
            },
            'learning_curves': sample_learning_curves
        }
        
        saved_paths = save_all_plots(analysis_results, temp_dir)
        
        assert isinstance(saved_paths, list)
        assert len(saved_paths) > 0, "Should save at least some plots"
        
        # Check that files actually exist
        for path in saved_paths:
            assert os.path.exists(path), f"Plot file {path} should exist"
    
    def test_save_all_plots_empty_analysis(self, temp_dir):
        """Test saving plots with empty analysis results."""
        saved_paths = save_all_plots({}, temp_dir)
        
        # Should handle empty analysis gracefully
        assert isinstance(saved_paths, list)


class TestErrorHandling:
    """Test error handling in plot functions."""
    
    def test_plot_convergence_invalid_data(self):
        """Test convergence plot with invalid data types."""
        invalid_metrics = {
            'steps': "not a list",
            'true_parent_likelihood': [0.1, 0.5, 0.8]
        }
        
        # Should handle invalid data gracefully (may warn but not crash)
        try:
            fig = plot_convergence(invalid_metrics)
            plt.close(fig)
        except Exception as e:
            # If it does throw an exception, it should be a reasonable one
            assert isinstance(e, (ValueError, TypeError))
    
    def test_plot_save_invalid_path(self, sample_trajectory_metrics):
        """Test plot saving with invalid file path."""
        invalid_path = "/nonexistent/directory/plot.png"
        
        # Should handle invalid save path gracefully
        try:
            fig = plot_convergence(sample_trajectory_metrics, save_path=invalid_path)
            plt.close(fig)
        except Exception as e:
            # Should be a reasonable file-related exception
            assert isinstance(e, (OSError, FileNotFoundError, PermissionError))
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_save_failure_handling(self, mock_savefig, sample_trajectory_metrics):
        """Test handling of save failures."""
        mock_savefig.side_effect = OSError("Disk full")
        
        # Should not crash the entire plotting function
        try:
            fig = plot_convergence(sample_trajectory_metrics, save_path="test.png")
            plt.close(fig)
        except OSError:
            pass  # Expected behavior


class TestPlotStyling:
    """Test plot styling and appearance."""
    
    def test_plot_has_labels(self, sample_trajectory_metrics):
        """Test that plots have proper labels."""
        fig = plot_convergence(sample_trajectory_metrics)
        
        ax = fig.axes[0]
        assert ax.get_xlabel(), "Plot should have x-axis label"
        assert ax.get_ylabel(), "Plot should have y-axis label"
        assert ax.get_title(), "Plot should have title"
        
        plt.close(fig)
    
    def test_plot_has_legend(self, sample_learning_curves):
        """Test that method comparison plot has legend."""
        fig = plot_method_comparison(sample_learning_curves)
        
        ax = fig.axes[0]
        legend = ax.get_legend()
        if len(sample_learning_curves) > 1:
            assert legend is not None, "Multi-method plot should have legend"
        
        plt.close(fig)
    
    def test_plot_color_consistency(self, sample_learning_curves):
        """Test that plots use consistent colors for methods."""
        fig1 = plot_method_comparison(sample_learning_curves)
        fig2 = plot_method_comparison(sample_learning_curves)
        
        # Colors should be consistent between plots
        # (This is a basic test - full color consistency would require more detailed checking)
        assert isinstance(fig1, plt.Figure)
        assert isinstance(fig2, plt.Figure)
        
        plt.close(fig1)
        plt.close(fig2)


class TestIntegrationWithAnalysis:
    """Test integration between plotting and analysis modules."""
    
    def test_plot_with_analysis_output(self):
        """Test plotting with realistic analysis module output."""
        # Simulate output from analysis module
        analysis_output = {
            'overall_stats': {'mean_f1_score': 0.65},
            'trajectory_results': {
                'static_surrogate': [{
                    'trajectory_metrics': {
                        'steps': [1, 2, 3, 4],
                        'true_parent_likelihood': [0.1, 0.4, 0.7, 0.8],
                        'f1_scores': [0.0, 0.3, 0.6, 0.8],
                        'target_values': [1.0, 1.3, 1.6, 1.8]
                    }
                }]
            },
            'learning_curves': {
                'static_surrogate': {
                    'mean_f1': [0.0, 0.3, 0.6, 0.8],
                    'std_f1': [0.1, 0.1, 0.1, 0.1],
                    'steps': [1, 2, 3, 4]
                }
            }
        }
        
        # Should work with realistic analysis output
        trajectory_metrics = analysis_output['trajectory_results']['static_surrogate'][0]['trajectory_metrics']
        learning_curves = analysis_output['learning_curves']
        
        fig1 = plot_convergence(trajectory_metrics)
        fig2 = plot_method_comparison(learning_curves)
        
        assert isinstance(fig1, plt.Figure)
        assert isinstance(fig2, plt.Figure)
        
        plt.close(fig1)
        plt.close(fig2)


class TestCalibrationPlots:
    """Test calibration curve plotting functions."""
    
    def test_plot_calibration_curves_basic(self):
        """Test basic calibration plot creation."""
        marginals_over_time = [
            {'X1': 0.3, 'X2': 0.7, 'X3': 0.2},
            {'X1': 0.5, 'X2': 0.5, 'X3': 0.4},
            {'X1': 0.8, 'X2': 0.3, 'X3': 0.7},
            {'X1': 0.9, 'X2': 0.1, 'X3': 0.85}
        ]
        true_parents = ['X1', 'X3']
        
        fig = plot_calibration_curves(marginals_over_time, true_parents)
        
        assert isinstance(fig, plt.Figure)
        # Should have 2x2 subplot layout
        assert len(fig.axes) == 4
        plt.close(fig)
    
    def test_plot_calibration_curves_save(self, temp_dir):
        """Test saving calibration plot."""
        marginals_over_time = [
            {'X1': 0.9, 'X2': 0.1, 'X3': 0.8}
        ]
        true_parents = ['X1', 'X3']
        save_path = os.path.join(temp_dir, "calibration_test.png")
        
        fig = plot_calibration_curves(
            marginals_over_time, 
            true_parents,
            save_path=save_path
        )
        
        assert os.path.exists(save_path)
        plt.close(fig)
    
    def test_plot_calibration_empty_marginals(self):
        """Test calibration plot with empty marginals."""
        marginals_over_time = []
        true_parents = ['X1']
        
        fig = plot_calibration_curves(marginals_over_time, true_parents)
        
        # Should handle empty data gracefully
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPrecisionRecallPlots:
    """Test precision-recall curve plotting functions."""
    
    def test_plot_precision_recall_curves_basic(self):
        """Test basic precision-recall plot creation."""
        marginals_over_time = [
            {'X1': 0.3, 'X2': 0.7, 'X3': 0.2, 'X4': 0.5},
            {'X1': 0.6, 'X2': 0.4, 'X3': 0.5, 'X4': 0.3},
            {'X1': 0.9, 'X2': 0.1, 'X3': 0.8, 'X4': 0.2}
        ]
        true_parents = ['X1', 'X3']
        
        fig = plot_precision_recall_curves(marginals_over_time, true_parents)
        
        assert isinstance(fig, plt.Figure)
        # Should have 1x2 subplot layout
        assert len(fig.axes) == 2
        plt.close(fig)
    
    def test_plot_precision_recall_curves_save(self, temp_dir):
        """Test saving precision-recall plot."""
        marginals_over_time = [
            {'X1': 0.9, 'X2': 0.1, 'X3': 0.8}
        ]
        true_parents = ['X1', 'X3']
        save_path = os.path.join(temp_dir, "pr_curve_test.png")
        
        fig = plot_precision_recall_curves(
            marginals_over_time,
            true_parents,
            save_path=save_path
        )
        
        assert os.path.exists(save_path)
        plt.close(fig)
    
    def test_plot_precision_recall_progression(self):
        """Test PR curves showing learning progression."""
        # Simulate improving predictions over time
        marginals_over_time = [
            {'X1': 0.4, 'X2': 0.6, 'X3': 0.3, 'X4': 0.7},  # Poor initial
            {'X1': 0.6, 'X2': 0.4, 'X3': 0.5, 'X4': 0.5},  # Improving
            {'X1': 0.9, 'X2': 0.1, 'X3': 0.85, 'X4': 0.15} # Good final
        ]
        true_parents = ['X1', 'X3']
        
        fig = plot_precision_recall_curves(marginals_over_time, true_parents)
        
        # Should show multiple curves (early, middle, late)
        ax1 = fig.axes[0]
        lines = ax1.get_lines()
        # At least 3 curves (one for each time point)
        assert len(lines) >= 3
        plt.close(fig)


class TestUpdatedSaveAllPlots:
    """Test updated save_all_plots with new visualization functions."""
    
    def test_save_all_plots_with_calibration(self, temp_dir):
        """Test saving all plots including calibration plots."""
        experiment_results = {
            'trajectory_metrics': {
                'steps': [1, 2, 3],
                'true_parent_likelihood': [0.3, 0.6, 0.9],
                'f1_scores': [0.2, 0.5, 0.8]
            },
            'true_parents': ['X1', 'X3'],
            'detailed_results': {
                'learning_history': [
                    {'step': 1, 'marginals': {'X1': 0.4, 'X2': 0.6, 'X3': 0.3}},
                    {'step': 2, 'marginals': {'X1': 0.7, 'X2': 0.3, 'X3': 0.6}},
                    {'step': 3, 'marginals': {'X1': 0.9, 'X2': 0.1, 'X3': 0.85}}
                ]
            }
        }
        
        saved_paths = save_all_plots(experiment_results, temp_dir, prefix="test")
        
        # Should save more plots now (including calibration and PR curves)
        assert len(saved_paths) >= 5  # convergence, target, calibration, PR, dashboard
        
        # Check specific plot files
        expected_files = [
            "test_convergence.png",
            "test_target_optimization.png",
            "test_calibration.png",
            "test_precision_recall.png",
            "test_dashboard.png"
        ]
        
        for filename in expected_files:
            full_path = os.path.join(temp_dir, filename)
            assert any(full_path in path for path in saved_paths), f"Expected {filename} to be saved"


class TestPerformance:
    """Test performance aspects of plotting functions."""
    
    def test_large_trajectory_plotting(self):
        """Test plotting with large trajectory data."""
        large_metrics = {
            'steps': list(range(1, 1001)),  # 1000 steps
            'true_parent_likelihood': list(onp.random.random(1000)),
            'f1_scores': list(onp.random.random(1000)),
            'target_values': list(onp.random.random(1000) * 10),
            'uncertainties': list(onp.random.random(1000) * 5)
        }
        
        # Should handle large data efficiently
        fig = plot_convergence(large_metrics)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_many_methods_comparison(self):
        """Test method comparison with many methods."""
        many_methods = {}
        for i in range(10):  # 10 methods
            many_methods[f'method_{i}'] = {
                'mean_f1': list(onp.random.random(20)),
                'std_f1': list(onp.random.random(20) * 0.1),
                'steps': list(range(1, 21))
            }
        
        # Should handle many methods
        fig = plot_method_comparison(many_methods)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_large_calibration_data(self):
        """Test calibration plots with many time steps."""
        # Generate 100 time steps
        marginals_over_time = []
        for _ in range(100):
            marginals = {}
            for var in ['X1', 'X2', 'X3', 'X4', 'X5']:
                marginals[var] = onp.random.random()
            marginals_over_time.append(marginals)
        
        true_parents = ['X1', 'X3']
        
        # Should handle large data efficiently
        fig = plot_calibration_curves(marginals_over_time, true_parents)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])