"""
Comprehensive tests for evaluation metrics computation.

This module tests causal discovery metrics, optimization metrics, efficiency metrics,
and composite scoring using property-based testing with Hypothesis.
"""

import pytest
import jax.numpy as jnp
import jax.random as random
import numpy as onp
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, patch

from causal_bayes_opt.evaluation.metrics import (
    CausalDiscoveryMetrics, OptimizationMetrics, EfficiencyMetrics, CompositeMetrics,
    compute_causal_discovery_metrics, compute_optimization_metrics,
    compute_efficiency_metrics, compute_composite_metrics,
    compute_intervention_quality_metrics, compute_learning_curve_metrics
)


class TestCausalDiscoveryMetrics:
    """Test causal discovery metrics computation."""
    
    def test_causal_discovery_metrics_immutable(self):
        """Test that CausalDiscoveryMetrics is immutable."""
        metrics = CausalDiscoveryMetrics(
            precision=0.8,
            recall=0.7,
            f1_score=0.75,
            structural_hamming_distance=5,
            edge_accuracy=0.85,
            orientation_accuracy=0.9,
            true_positives=8,
            false_positives=2,
            false_negatives=3,
            true_negatives=12
        )
        
        with pytest.raises(AttributeError):
            metrics.precision = 0.9
    
    def test_perfect_causal_discovery(self):
        """Test metrics for perfect causal discovery."""
        # Perfect match - identical graphs
        true_graph = jnp.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        predicted_graph = true_graph.copy()
        
        metrics = compute_causal_discovery_metrics(true_graph, predicted_graph)
        
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.structural_hamming_distance == 0
        assert metrics.edge_accuracy == 1.0
        assert metrics.true_positives == 2  # Two edges in the graph
        assert metrics.false_positives == 0
        assert metrics.false_negatives == 0
    
    def test_completely_wrong_causal_discovery(self):
        """Test metrics for completely wrong causal discovery."""
        # True graph has edges, predicted has none
        true_graph = jnp.array([
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0]
        ])
        predicted_graph = jnp.zeros((3, 3))
        
        metrics = compute_causal_discovery_metrics(true_graph, predicted_graph)
        
        assert metrics.precision == 0.0  # No predictions, so precision is 0
        assert metrics.recall == 0.0     # Missed all true edges
        assert metrics.f1_score == 0.0   # No true positives
        assert metrics.true_positives == 0
        assert metrics.false_negatives == 3  # Three true edges missed
        assert metrics.false_positives == 0
    
    @given(
        graph_size=st.integers(min_value=2, max_value=10),
        edge_prob=st.floats(min_value=0.1, max_value=0.8),
        noise_level=st.floats(min_value=0.0, max_value=0.3)
    )
    @settings(max_examples=20)
    def test_causal_discovery_metrics_properties(self, graph_size, edge_prob, noise_level):
        """Property-based test for causal discovery metrics."""
        key = random.PRNGKey(42)
        
        # Generate random true graph
        key, subkey = random.split(key)
        true_graph = random.bernoulli(subkey, edge_prob, (graph_size, graph_size)).astype(jnp.float32)
        # Make it upper triangular (DAG)
        true_graph = jnp.triu(true_graph, k=1)
        
        # Generate noisy predicted graph
        key, subkey = random.split(key)
        noise = random.bernoulli(subkey, noise_level, (graph_size, graph_size)).astype(jnp.float32)
        predicted_graph = jnp.where(noise, 1 - true_graph, true_graph)
        predicted_graph = jnp.triu(predicted_graph, k=1)
        
        metrics = compute_causal_discovery_metrics(true_graph, predicted_graph)
        
        # Basic sanity checks
        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.recall <= 1.0
        assert 0.0 <= metrics.f1_score <= 1.0
        assert 0.0 <= metrics.edge_accuracy <= 1.0
        assert metrics.structural_hamming_distance >= 0
        
        # Confusion matrix should be consistent
        total_elements = graph_size * graph_size
        assert (metrics.true_positives + metrics.false_positives + 
                metrics.true_negatives + metrics.false_negatives) == total_elements
        
        # F1 score should be harmonic mean of precision and recall
        if metrics.precision + metrics.recall > 0:
            expected_f1 = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
            assert abs(metrics.f1_score - expected_f1) < 1e-6
    
    def test_orientation_accuracy_computation(self):
        """Test orientation accuracy computation for directed graphs."""
        # True graph: 0 -> 1 -> 2
        true_graph = jnp.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        # Predicted graph: 0 -> 1, 2 -> 1 (wrong direction for second edge)
        predicted_graph = jnp.array([
            [0, 1, 0],
            [0, 0, 0],
            [0, 1, 0]
        ])
        
        metrics = compute_causal_discovery_metrics(true_graph, predicted_graph, compute_orientations=True)
        
        # Should detect the orientation error
        assert metrics.orientation_accuracy < 1.0
        assert metrics.orientation_accuracy >= 0.0
    
    def test_binary_threshold_handling(self):
        """Test handling of continuous-valued graphs with binary thresholding."""
        # Continuous-valued graphs
        true_graph = jnp.array([
            [0.0, 0.8, 0.0],
            [0.0, 0.0, 0.9],
            [0.0, 0.0, 0.0]
        ])
        
        predicted_graph = jnp.array([
            [0.0, 0.7, 0.1],  # Small value that should be thresholded to 0
            [0.0, 0.0, 0.85],
            [0.0, 0.0, 0.0]
        ])
        
        metrics = compute_causal_discovery_metrics(true_graph, predicted_graph)
        
        # Should handle thresholding correctly
        assert metrics.precision > 0.0
        assert metrics.recall > 0.0


class TestOptimizationMetrics:
    """Test optimization performance metrics computation."""
    
    def test_optimization_metrics_immutable(self):
        """Test that OptimizationMetrics is immutable."""
        metrics = OptimizationMetrics(
            final_objective_value=0.8,
            optimization_improvement=0.3,
            convergence_steps=50,
            intervention_efficiency=0.6,
            sample_efficiency=0.006,
            regret=0.2,
            cumulative_regret=5.0
        )
        
        with pytest.raises(AttributeError):
            metrics.final_objective_value = 0.9
    
    def test_optimization_metrics_improving_trajectory(self):
        """Test optimization metrics for improving trajectory."""
        # Improving objective values
        objective_values = jnp.array([0.1, 0.3, 0.5, 0.7, 0.8])
        true_optimum = 0.9
        
        metrics = compute_optimization_metrics(objective_values, true_optimum)
        
        assert metrics.final_objective_value == 0.8
        assert metrics.optimization_improvement == 0.7  # 0.8 - 0.1
        assert metrics.sample_efficiency > 0  # Positive improvement per step
        assert metrics.regret == 0.1  # |0.9 - 0.8|
        assert metrics.cumulative_regret > 0
    
    def test_optimization_metrics_converged_trajectory(self):
        """Test optimization metrics for converged trajectory."""
        # Trajectory that converges to optimum
        true_optimum = 1.0
        objective_values = jnp.array([0.5, 0.8, 0.95, 0.99, 1.0, 1.0])
        
        metrics = compute_optimization_metrics(objective_values, true_optimum)
        
        assert metrics.final_objective_value == 1.0
        assert metrics.convergence_steps <= len(objective_values)
        assert metrics.regret < 1e-6  # Should be very close to optimum
    
    @given(
        trajectory_length=st.integers(min_value=2, max_value=100),
        initial_value=st.floats(min_value=-10.0, max_value=10.0),
        final_value=st.floats(min_value=-10.0, max_value=10.0),
        true_optimum=st.floats(min_value=-10.0, max_value=10.0)
    )
    @settings(max_examples=30)
    def test_optimization_metrics_properties(self, trajectory_length, initial_value, final_value, true_optimum):
        """Property-based test for optimization metrics."""
        # Create monotonic trajectory
        objective_values = jnp.linspace(initial_value, final_value, trajectory_length)
        
        metrics = compute_optimization_metrics(objective_values, true_optimum)
        
        # Basic properties
        assert metrics.final_objective_value == float(objective_values[-1])
        assert metrics.optimization_improvement == float(objective_values[-1] - objective_values[0])
        assert 0 <= metrics.convergence_steps <= trajectory_length
        assert metrics.regret >= 0
        assert metrics.cumulative_regret >= 0
        
        # Sample efficiency should be improvement per step
        expected_sample_efficiency = metrics.optimization_improvement / trajectory_length
        assert abs(metrics.sample_efficiency - expected_sample_efficiency) < 1e-6
    
    def test_optimization_metrics_with_intervention_costs(self):
        """Test optimization metrics with intervention costs."""
        objective_values = jnp.array([0.2, 0.5, 0.8])
        true_optimum = 1.0
        intervention_costs = jnp.array([1.0, 2.0, 1.5])
        
        metrics = compute_optimization_metrics(objective_values, true_optimum, intervention_costs)
        
        # Intervention efficiency should account for costs
        total_cost = jnp.sum(intervention_costs)
        expected_efficiency = metrics.optimization_improvement / total_cost
        assert abs(metrics.intervention_efficiency - expected_efficiency) < 1e-6
    
    def test_empty_objective_values_error(self):
        """Test error handling for empty objective values."""
        with pytest.raises(ValueError, match="objective_values cannot be empty"):
            compute_optimization_metrics(jnp.array([]), 1.0)


class TestEfficiencyMetrics:
    """Test efficiency and resource usage metrics computation."""
    
    def test_efficiency_metrics_immutable(self):
        """Test that EfficiencyMetrics is immutable."""
        metrics = EfficiencyMetrics(
            total_interventions=100,
            computational_time_seconds=120.5,
            memory_usage_gb=4.2,
            interventions_per_second=0.83,
            time_to_convergence=100.0,
            inference_time_per_step=1.2
        )
        
        with pytest.raises(AttributeError):
            metrics.total_interventions = 200
    
    def test_efficiency_metrics_basic_computation(self):
        """Test basic efficiency metrics computation."""
        total_interventions = 100
        computational_time = 50.0
        memory_usage = 8.0
        inference_times = jnp.array([0.5, 0.6, 0.4, 0.5, 0.5])
        convergence_time = 30.0
        
        metrics = compute_efficiency_metrics(
            total_interventions=total_interventions,
            computational_time=computational_time,
            memory_usage=memory_usage,
            inference_times=inference_times,
            convergence_time=convergence_time
        )
        
        assert metrics.total_interventions == 100
        assert metrics.computational_time_seconds == 50.0
        assert metrics.memory_usage_gb == 8.0
        assert metrics.interventions_per_second == 2.0  # 100/50
        assert metrics.time_to_convergence == 30.0
        assert abs(metrics.inference_time_per_step - 0.5) < 1e-6  # Mean of inference times
    
    @given(
        total_interventions=st.integers(min_value=1, max_value=1000),
        computational_time=st.floats(min_value=0.1, max_value=1000.0),
        memory_usage=st.floats(min_value=0.1, max_value=64.0)
    )
    @settings(max_examples=20)
    def test_efficiency_metrics_properties(self, total_interventions, computational_time, memory_usage):
        """Property-based test for efficiency metrics."""
        metrics = compute_efficiency_metrics(
            total_interventions=total_interventions,
            computational_time=computational_time,
            memory_usage=memory_usage
        )
        
        # Basic properties
        assert metrics.total_interventions == total_interventions
        assert metrics.computational_time_seconds == computational_time
        assert metrics.memory_usage_gb == memory_usage
        assert metrics.interventions_per_second > 0
        assert metrics.inference_time_per_step > 0
        
        # Interventions per second should be consistent
        expected_rate = total_interventions / computational_time
        assert abs(metrics.interventions_per_second - expected_rate) < 1e-6
    
    def test_efficiency_metrics_with_inference_times(self):
        """Test efficiency metrics with detailed inference times."""
        inference_times = jnp.array([1.0, 1.2, 0.8, 1.1, 0.9])
        
        metrics = compute_efficiency_metrics(
            total_interventions=5,
            computational_time=5.0,
            memory_usage=2.0,
            inference_times=inference_times
        )
        
        expected_avg_inference = jnp.mean(inference_times)
        assert abs(metrics.inference_time_per_step - expected_avg_inference) < 1e-6


class TestCompositeMetrics:
    """Test composite metrics computation and weighting."""
    
    def test_composite_metrics_immutable(self):
        """Test that CompositeMetrics is immutable."""
        causal_metrics = CausalDiscoveryMetrics(0.8, 0.7, 0.75, 5, 0.85, 0.9, 8, 2, 3, 12)
        optimization_metrics = OptimizationMetrics(0.8, 0.3, 50, 0.6, 0.006, 0.2, 5.0)
        efficiency_metrics = EfficiencyMetrics(100, 120.5, 4.2, 0.83, 100.0, 1.2)
        
        composite = CompositeMetrics(
            causal_discovery=causal_metrics,
            optimization=optimization_metrics,
            efficiency=efficiency_metrics,
            overall_score=0.75,
            weighted_score={'causal': 0.3, 'opt': 0.3, 'eff': 0.15}
        )
        
        with pytest.raises(AttributeError):
            composite.overall_score = 0.8
    
    def test_composite_metrics_default_weights(self):
        """Test composite metrics with default weights."""
        causal_metrics = CausalDiscoveryMetrics(0.8, 0.7, 0.75, 5, 0.85, 0.9, 8, 2, 3, 12)
        optimization_metrics = OptimizationMetrics(0.8, 0.3, 50, 0.6, 0.006, 0.2, 5.0)
        efficiency_metrics = EfficiencyMetrics(100, 120.5, 4.2, 0.83, 100.0, 1.2)
        
        composite = compute_composite_metrics(causal_metrics, optimization_metrics, efficiency_metrics)
        
        # Should use default weights
        assert 'causal_discovery' in composite.weighted_score
        assert 'optimization' in composite.weighted_score  
        assert 'efficiency' in composite.weighted_score
        
        # Overall score should be sum of weighted scores
        expected_overall = sum(composite.weighted_score.values())
        assert abs(composite.overall_score - expected_overall) < 1e-6
    
    def test_composite_metrics_custom_weights(self):
        """Test composite metrics with custom weights."""
        causal_metrics = CausalDiscoveryMetrics(0.9, 0.8, 0.85, 2, 0.95, 0.9, 10, 1, 2, 15)
        optimization_metrics = OptimizationMetrics(0.9, 0.4, 30, 0.8, 0.01, 0.1, 3.0)
        efficiency_metrics = EfficiencyMetrics(50, 60.0, 2.0, 0.83, 50.0, 1.2)
        
        custom_weights = {
            'causal_discovery': 0.5,
            'optimization': 0.3,
            'efficiency': 0.2
        }
        
        composite = compute_composite_metrics(
            causal_metrics, optimization_metrics, efficiency_metrics, custom_weights
        )
        
        # Should use custom weights
        assert composite.weighted_score['causal_discovery'] == pytest.approx(0.9 * 0.5, abs=1e-2)
        
        # Weights should sum to 1.0
        total_weight = sum(custom_weights.values())
        assert abs(total_weight - 1.0) < 1e-6
    
    @given(
        causal_f1=st.floats(min_value=0.0, max_value=1.0),
        opt_efficiency=st.floats(min_value=0.0, max_value=10.0),
        interventions_per_sec=st.floats(min_value=0.1, max_value=1000.0)
    )
    @settings(max_examples=20)
    def test_composite_metrics_properties(self, causal_f1, opt_efficiency, interventions_per_sec):
        """Property-based test for composite metrics."""
        causal_metrics = CausalDiscoveryMetrics(0.8, 0.7, causal_f1, 5, 0.85, 0.9, 8, 2, 3, 12)
        optimization_metrics = OptimizationMetrics(0.8, 0.3, 50, 0.6, opt_efficiency, 0.2, 5.0)
        efficiency_metrics = EfficiencyMetrics(100, 120.5, 4.2, interventions_per_sec, 100.0, 1.2)
        
        composite = compute_composite_metrics(causal_metrics, optimization_metrics, efficiency_metrics)
        
        # Overall score should be between 0 and 1 (approximately)
        assert composite.overall_score >= 0.0
        assert composite.overall_score <= 3.0  # Could exceed 1 due to normalization
        
        # Individual component scores should be reasonable
        for score in composite.weighted_score.values():
            assert score >= 0.0


class TestInterventionQualityMetrics:
    """Test intervention quality and diversity metrics."""
    
    @given(
        num_interventions=st.integers(min_value=2, max_value=20),
        num_variables=st.integers(min_value=2, max_value=10)
    )
    @settings(max_examples=15)
    def test_intervention_quality_metrics_properties(self, num_interventions, num_variables):
        """Property-based test for intervention quality metrics."""
        key = random.PRNGKey(42)
        
        # Generate random interventions
        key, subkey = random.split(key)
        interventions = random.normal(subkey, (num_interventions, num_variables, num_variables))
        
        # Generate random targets (one-hot encoded)
        key, subkey = random.split(key)
        target_indices = random.choice(subkey, num_variables, (num_interventions,))
        targets = jnp.eye(num_variables)[target_indices]
        
        metrics = compute_intervention_quality_metrics(interventions, targets)
        
        # Basic properties
        assert 0.0 <= metrics['diversity_ratio'] <= 1.0
        assert 0.0 <= metrics['target_coverage'] <= 1.0
        assert metrics['avg_intervention_magnitude'] >= 0.0
        assert metrics['std_intervention_magnitude'] >= 0.0
        assert 0 <= metrics['num_unique_interventions'] <= num_interventions
    
    def test_perfectly_diverse_interventions(self):
        """Test metrics for perfectly diverse interventions."""
        # All different interventions
        interventions = jnp.array([
            [[1.0, 0.0], [0.0, 0.0]],  # Intervention 1
            [[0.0, 1.0], [0.0, 0.0]],  # Intervention 2
            [[0.0, 0.0], [1.0, 0.0]]   # Intervention 3
        ])
        targets = jnp.array([
            [1.0, 0.0],  # Target variable 0
            [0.0, 1.0],  # Target variable 1
            [1.0, 0.0]   # Target variable 0 again
        ])
        
        metrics = compute_intervention_quality_metrics(interventions, targets)
        
        assert metrics['diversity_ratio'] == 1.0  # All interventions are unique
        assert metrics['target_coverage'] == 1.0  # Both variables targeted
    
    def test_identical_interventions(self):
        """Test metrics for identical interventions."""
        # All same interventions
        intervention = jnp.array([[1.0, 0.0], [0.0, 0.0]])
        interventions = jnp.stack([intervention] * 5)
        targets = jnp.array([[1.0, 0.0]] * 5)  # Same target
        
        metrics = compute_intervention_quality_metrics(interventions, targets)
        
        assert metrics['diversity_ratio'] == 0.2  # Only 1 unique out of 5
        assert metrics['target_coverage'] == 0.5  # Only 1 out of 2 variables
        assert metrics['num_unique_interventions'] == 1


class TestLearningCurveMetrics:
    """Test learning curve analysis metrics."""
    
    def test_learning_curve_metrics_monotonic_improvement(self):
        """Test learning curve metrics for monotonically improving trajectory."""
        # Steadily improving objective values
        objective_values = jnp.array([0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9])
        
        metrics = compute_learning_curve_metrics(objective_values, window_size=3)
        
        # Should have positive learning rates (mostly)
        learning_rates = metrics['learning_rates']
        assert jnp.all(learning_rates >= 0)  # All improvements
        
        # Plateau fraction should be low for improving trajectory
        assert metrics['plateau_fraction'] <= 0.5
    
    def test_learning_curve_metrics_plateau(self):
        """Test learning curve metrics for trajectory with plateau."""
        # Trajectory that plateaus
        objective_values = jnp.array([0.1, 0.5, 0.7, 0.75, 0.75, 0.75, 0.75])
        
        metrics = compute_learning_curve_metrics(objective_values, window_size=3)
        
        # Should detect plateau (low learning rates at end)
        learning_rates = metrics['learning_rates']
        assert jnp.abs(learning_rates[-3:]).max() < 0.1  # Small improvements at end
        
        # High plateau fraction
        assert metrics['plateau_fraction'] > 0.3
    
    @given(
        trajectory_length=st.integers(min_value=5, max_value=50),
        window_size=st.integers(min_value=2, max_value=10)
    )
    @settings(max_examples=15)
    def test_learning_curve_metrics_properties(self, trajectory_length, window_size):
        """Property-based test for learning curve metrics."""
        assume(window_size < trajectory_length)
        
        # Generate random trajectory
        key = random.PRNGKey(42)
        objective_values = jnp.cumsum(random.normal(key, (trajectory_length,)) * 0.1)
        
        metrics = compute_learning_curve_metrics(objective_values, window_size)
        
        # Check output shapes and ranges
        assert len(metrics['learning_rates']) == trajectory_length - 1
        assert len(metrics['smoothed_learning_rates']) >= 1
        assert 0.0 <= metrics['plateau_fraction'] <= 1.0
        
        # Smoothed values should be shorter than original
        assert len(metrics['smoothed_objectives']) <= len(objective_values)


class TestMetricsIntegration:
    """Integration tests for metrics computation."""
    
    def test_complete_evaluation_pipeline(self):
        """Test complete evaluation pipeline with all metric types."""
        # Set up test data
        true_graph = jnp.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        predicted_graph = jnp.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])  # Some errors
        
        objective_values = jnp.array([0.2, 0.5, 0.7, 0.8])
        true_optimum = 0.9
        
        # Compute all metrics
        causal_metrics = compute_causal_discovery_metrics(true_graph, predicted_graph)
        optimization_metrics = compute_optimization_metrics(objective_values, true_optimum)
        efficiency_metrics = compute_efficiency_metrics(
            total_interventions=len(objective_values),
            computational_time=10.0,
            memory_usage=4.0
        )
        composite_metrics = compute_composite_metrics(causal_metrics, optimization_metrics, efficiency_metrics)
        
        # All should be valid
        assert isinstance(causal_metrics, CausalDiscoveryMetrics)
        assert isinstance(optimization_metrics, OptimizationMetrics)
        assert isinstance(efficiency_metrics, EfficiencyMetrics)
        assert isinstance(composite_metrics, CompositeMetrics)
        
        # Composite should reference the individual metrics
        assert composite_metrics.causal_discovery == causal_metrics
        assert composite_metrics.optimization == optimization_metrics
        assert composite_metrics.efficiency == efficiency_metrics
    
    def test_metrics_with_edge_cases(self):
        """Test metrics computation with edge cases."""
        # Empty graph
        empty_graph = jnp.zeros((2, 2))
        causal_metrics = compute_causal_discovery_metrics(empty_graph, empty_graph)
        
        # Should handle gracefully
        assert causal_metrics.precision == 0.0 or jnp.isnan(causal_metrics.precision)
        assert causal_metrics.structural_hamming_distance == 0
        
        # Single-step optimization
        single_objective = jnp.array([0.5])
        optimization_metrics = compute_optimization_metrics(single_objective, 1.0)
        
        assert optimization_metrics.optimization_improvement == 0.0
        assert optimization_metrics.convergence_steps == 1


class TestMetricsEdgeCases:
    """Test edge cases and error conditions in metrics computation."""
    
    def test_mismatched_graph_sizes(self):
        """Test error handling for mismatched graph sizes."""
        true_graph = jnp.zeros((3, 3))
        predicted_graph = jnp.zeros((4, 4))
        
        # Should handle gracefully or raise informative error
        with pytest.raises((ValueError, AssertionError)):
            compute_causal_discovery_metrics(true_graph, predicted_graph)
    
    def test_negative_optimization_values(self):
        """Test optimization metrics with negative objective values."""
        objective_values = jnp.array([-1.0, -0.5, 0.0, 0.5])
        true_optimum = 1.0
        
        metrics = compute_optimization_metrics(objective_values, true_optimum)
        
        # Should handle negative values correctly
        assert metrics.optimization_improvement == 1.5  # 0.5 - (-1.0)
        assert metrics.regret == 0.5  # |1.0 - 0.5|
    
    def test_zero_computational_time(self):
        """Test efficiency metrics with zero computational time."""
        # Should handle gracefully (add small epsilon)
        metrics = compute_efficiency_metrics(
            total_interventions=10,
            computational_time=0.0,
            memory_usage=1.0
        )
        
        # Should not crash and return reasonable values
        assert metrics.interventions_per_second > 0  # Should add epsilon
    
    def test_very_large_numbers(self):
        """Test metrics with very large input values."""
        large_objective = jnp.array([1e6, 1e6 + 1000, 1e6 + 2000])
        true_optimum = 1e6 + 3000
        
        metrics = compute_optimization_metrics(large_objective, true_optimum)
        
        # Should handle large numbers without numerical issues
        assert jnp.isfinite(metrics.optimization_improvement)
        assert jnp.isfinite(metrics.regret)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])