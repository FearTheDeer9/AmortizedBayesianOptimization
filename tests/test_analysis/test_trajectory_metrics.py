"""
Tests for trajectory_metrics.py

Comprehensive tests for all metric computation functions following
functional programming principles and property-based testing.
"""

import pytest
import numpy as onp
from hypothesis import given, strategies as st
from typing import Dict, List

from src.causal_bayes_opt.analysis.trajectory_metrics import (
    compute_true_parent_likelihood,
    compute_f1_score_from_marginals,
    compute_f1_with_multiple_thresholds,
    compute_precision_recall_curve,
    find_optimal_f1_threshold,
    find_youden_j_threshold,
    compute_expected_calibration_error,
    compute_brier_score,
    compute_comprehensive_metrics,
    compute_trajectory_metrics,
    extract_metrics_from_experiment_result,
    analyze_convergence_trajectory,
    compute_intervention_efficiency
)


class TestComputeTrueParentLikelihood:
    """Test the core true parent likelihood computation."""
    
    def test_perfect_discovery(self):
        """Test perfect parent discovery case."""
        marginals = {'X1': 1.0, 'X2': 0.0, 'X3': 1.0}
        true_parents = ['X1', 'X3']
        
        likelihood = compute_true_parent_likelihood(marginals, true_parents)
        
        assert likelihood == 1.0, "Perfect discovery should give likelihood 1.0"
    
    def test_no_discovery(self):
        """Test complete failure to discover parents."""
        marginals = {'X1': 0.0, 'X2': 1.0, 'X3': 0.0}
        true_parents = ['X1', 'X3']
        
        likelihood = compute_true_parent_likelihood(marginals, true_parents)
        
        assert likelihood == 0.0, "Complete failure should give likelihood 0.0"
    
    def test_partial_discovery(self):
        """Test partial parent discovery."""
        marginals = {'X1': 0.8, 'X2': 0.2, 'X3': 0.7}
        true_parents = ['X1', 'X3']
        
        expected = 0.8 * (1 - 0.2) * 0.7  # 0.8 * 0.8 * 0.7 = 0.448
        likelihood = compute_true_parent_likelihood(marginals, true_parents)
        
        assert abs(likelihood - expected) < 1e-10, f"Expected {expected}, got {likelihood}"
    
    def test_empty_true_parents(self):
        """Test case where target has no true parents."""
        marginals = {'X1': 0.3, 'X2': 0.1, 'X3': 0.2}
        true_parents = []
        
        expected = (1 - 0.3) * (1 - 0.1) * (1 - 0.2)  # 0.7 * 0.9 * 0.8 = 0.504
        likelihood = compute_true_parent_likelihood(marginals, true_parents)
        
        assert abs(likelihood - expected) < 1e-10, f"Expected {expected}, got {likelihood}"
    
    def test_frozen_set_parents(self):
        """Test that function works with frozenset as well as list."""
        marginals = {'X1': 0.9, 'X2': 0.1}
        true_parents_list = ['X1']
        true_parents_frozenset = frozenset(['X1'])
        
        likelihood_list = compute_true_parent_likelihood(marginals, true_parents_list)
        likelihood_frozenset = compute_true_parent_likelihood(marginals, true_parents_frozenset)
        
        assert likelihood_list == likelihood_frozenset, "Should work with both list and frozenset"
    
    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=3, alphabet="X123"),
            st.floats(min_value=0.0, max_value=1.0),
            min_size=1, max_size=5
        )
    )
    def test_probability_bounds(self, marginals: Dict[str, float]):
        """Property-based test: likelihood should always be in [0, 1]."""
        variables = list(marginals.keys())
        # Create some subset as true parents
        n_parents = len(variables) // 2
        true_parents = variables[:n_parents]
        
        likelihood = compute_true_parent_likelihood(marginals, true_parents)
        
        assert 0.0 <= likelihood <= 1.0, f"Likelihood {likelihood} not in [0, 1]"
    
    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=3, alphabet="X123"),
            st.just(0.5),  # All marginals = 0.5
            min_size=2, max_size=4
        )
    )
    def test_symmetric_case(self, marginals: Dict[str, float]):
        """Property: When all marginals = 0.5, likelihood = 0.5^n."""
        variables = list(marginals.keys())
        true_parents = variables[:len(variables)//2]
        
        likelihood = compute_true_parent_likelihood(marginals, true_parents)
        expected = 0.5 ** len(variables)
        
        assert abs(likelihood - expected) < 1e-10, f"Expected {expected}, got {likelihood}"


class TestComputeF1ScoreFromMarginals:
    """Test F1 score computation from marginal probabilities."""
    
    def test_perfect_f1(self):
        """Test perfect F1 score case."""
        marginals = {'X1': 1.0, 'X2': 0.0, 'X3': 1.0}
        true_parents = ['X1', 'X3']
        
        f1 = compute_f1_score_from_marginals(marginals, true_parents, threshold=0.5)
        
        assert f1 == 1.0, "Perfect prediction should give F1 = 1.0"
    
    def test_zero_f1(self):
        """Test zero F1 score case."""
        marginals = {'X1': 0.0, 'X2': 1.0, 'X3': 0.0}
        true_parents = ['X1', 'X3']
        
        f1 = compute_f1_score_from_marginals(marginals, true_parents, threshold=0.5)
        
        assert f1 == 0.0, "Complete wrong prediction should give F1 = 0.0"
    
    def test_threshold_sensitivity(self):
        """Test that F1 changes appropriately with threshold."""
        marginals = {'X1': 0.7, 'X2': 0.3, 'X3': 0.6}
        true_parents = ['X1', 'X3']
        
        f1_low = compute_f1_score_from_marginals(marginals, true_parents, threshold=0.4)
        f1_high = compute_f1_score_from_marginals(marginals, true_parents, threshold=0.8)
        
        # With threshold 0.4: X1, X3 predicted as parents (correct)
        # With threshold 0.8: only X1 predicted as parent (recall drops)
        assert f1_low >= f1_high, "Lower threshold should generally give higher or equal F1"
    
    def test_empty_true_parents(self):
        """Test F1 when there are no true parents."""
        marginals = {'X1': 0.2, 'X2': 0.1, 'X3': 0.3}
        true_parents = []
        
        # All predictions should be negative, so F1 should be 1.0
        f1 = compute_f1_score_from_marginals(marginals, true_parents, threshold=0.5)
        
        assert f1 == 1.0, "When no true parents and all predicted negative, F1 should be 1.0"


class TestComputeTrajectoryMetrics:
    """Test trajectory metrics extraction."""
    
    def test_empty_trajectory(self):
        """Test handling of empty trajectory."""
        trajectory = []
        true_parents = ['X1']
        target = 'X2'
        
        metrics = compute_trajectory_metrics(trajectory, true_parents, target)
        
        assert metrics['steps'] == []
        assert metrics['true_parent_likelihood'] == []
        assert metrics['f1_scores'] == []
        assert metrics['target_values'] == []
    
    def test_single_step_trajectory(self):
        """Test trajectory with single step."""
        # Mock trajectory step
        trajectory = [{
            'step': 1,
            'marginal_probs': {'X1': 0.8, 'X2': 0.2},
            'target_value': 2.5,
            'uncertainty': 1.0
        }]
        true_parents = ['X1']
        target = 'X2'
        
        metrics = compute_trajectory_metrics(trajectory, true_parents, target)
        
        assert len(metrics['steps']) == 1
        assert metrics['steps'][0] == 1
        assert len(metrics['true_parent_likelihood']) == 1
        assert len(metrics['f1_scores']) == 1
        assert metrics['target_values'][0] == 2.5
        assert metrics['uncertainties'][0] == 1.0


class TestExtractMetricsFromExperimentResult:
    """Test extraction of metrics from experiment results."""
    
    def test_valid_experiment_result(self):
        """Test extraction from valid experiment result."""
        experiment_result = {
            'detailed_results': {
                'learning_history': [
                    {
                        'step': 1,
                        'marginal_probs': {'X1': 0.5, 'X2': 0.3},
                        'target_value': 2.0,
                        'uncertainty': 2.0
                    },
                    {
                        'step': 2,
                        'marginal_probs': {'X1': 0.7, 'X2': 0.2},
                        'target_value': 2.3,
                        'uncertainty': 1.5
                    }
                ]
            }
        }
        true_parents = ['X1']
        
        metrics = extract_metrics_from_experiment_result(experiment_result, true_parents)
        
        assert len(metrics['steps']) == 2
        assert metrics['steps'] == [1, 2]
        assert len(metrics['true_parent_likelihood']) == 2
        assert len(metrics['target_values']) == 2
        assert metrics['target_values'] == [2.0, 2.3]
    
    def test_missing_learning_history(self):
        """Test handling when learning history is missing."""
        experiment_result = {
            'detailed_results': {}
        }
        true_parents = ['X1']
        
        metrics = extract_metrics_from_experiment_result(experiment_result, true_parents)
        
        # Should return empty metrics
        assert metrics['steps'] == []
        assert metrics['true_parent_likelihood'] == []
    
    def test_malformed_experiment_result(self):
        """Test handling of malformed experiment result."""
        experiment_result = {'some_other_key': 'value'}
        true_parents = ['X1']
        
        metrics = extract_metrics_from_experiment_result(experiment_result, true_parents)
        
        # Should return empty metrics without crashing
        assert metrics['steps'] == []


class TestAnalyzeConvergenceTrajectory:
    """Test convergence analysis functionality."""
    
    def test_converged_trajectory(self):
        """Test trajectory that converges."""
        trajectory_metrics = {
            'steps': [1, 2, 3, 4, 5],
            'true_parent_likelihood': [0.1, 0.4, 0.7, 0.8, 0.85],
            'f1_scores': [0.0, 0.3, 0.6, 0.8, 0.9]
        }
        
        analysis = analyze_convergence_trajectory(trajectory_metrics, threshold=0.8)
        
        assert analysis['converged'], "Should detect convergence"
        assert analysis['convergence_step'] == 4, "Should converge at step 4"
        assert analysis['final_likelihood'] == 0.85
    
    def test_non_converged_trajectory(self):
        """Test trajectory that doesn't converge."""
        trajectory_metrics = {
            'steps': [1, 2, 3, 4, 5],
            'true_parent_likelihood': [0.1, 0.2, 0.3, 0.4, 0.5],
            'f1_scores': [0.0, 0.1, 0.2, 0.3, 0.4]
        }
        
        analysis = analyze_convergence_trajectory(trajectory_metrics, threshold=0.8)
        
        assert not analysis['converged'], "Should not detect convergence"
        assert analysis['convergence_step'] is None
        assert analysis['final_likelihood'] == 0.5
    
    def test_empty_trajectory_analysis(self):
        """Test analysis of empty trajectory."""
        trajectory_metrics = {
            'steps': [],
            'true_parent_likelihood': [],
            'f1_scores': []
        }
        
        analysis = analyze_convergence_trajectory(trajectory_metrics)
        
        assert not analysis['converged']
        assert analysis['convergence_step'] is None
        assert analysis['final_likelihood'] == 0.0


class TestComputeInterventionEfficiency:
    """Test intervention efficiency computation."""
    
    def test_efficiency_calculation(self):
        """Test basic efficiency calculation."""
        trajectory_metrics = {
            'steps': [1, 2, 3, 4, 5],
            'f1_scores': [0.0, 0.2, 0.4, 0.6, 0.8],
            'true_parent_likelihood': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
        
        efficiency = compute_intervention_efficiency(trajectory_metrics)
        
        expected_f1_efficiency = (0.8 - 0.0) / 5  # 0.16
        expected_likelihood_efficiency = (0.9 - 0.1) / 5  # 0.16
        
        assert abs(efficiency['f1_per_intervention'] - expected_f1_efficiency) < 1e-10
        assert abs(efficiency['likelihood_per_intervention'] - expected_likelihood_efficiency) < 1e-10
    
    def test_zero_improvement_efficiency(self):
        """Test efficiency when no improvement occurs."""
        trajectory_metrics = {
            'steps': [1, 2, 3],
            'f1_scores': [0.5, 0.5, 0.5],
            'true_parent_likelihood': [0.3, 0.3, 0.3]
        }
        
        efficiency = compute_intervention_efficiency(trajectory_metrics)
        
        assert efficiency['f1_per_intervention'] == 0.0
        assert efficiency['likelihood_per_intervention'] == 0.0
    
    def test_empty_trajectory_efficiency(self):
        """Test efficiency computation for empty trajectory."""
        trajectory_metrics = {
            'steps': [],
            'f1_scores': [],
            'true_parent_likelihood': []
        }
        
        efficiency = compute_intervention_efficiency(trajectory_metrics)
        
        assert efficiency['f1_per_intervention'] == 0.0
        assert efficiency['likelihood_per_intervention'] == 0.0


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_full_pipeline(self):
        """Test complete analysis pipeline."""
        # Simulate realistic experiment result
        experiment_result = {
            'method': 'learning_surrogate',
            'detailed_results': {
                'learning_history': [
                    {'step': 1, 'marginal_probs': {'X1': 0.3, 'X2': 0.7}, 'target_value': 1.0, 'uncertainty': 3.0},
                    {'step': 2, 'marginal_probs': {'X1': 0.6, 'X2': 0.4}, 'target_value': 1.5, 'uncertainty': 2.0},
                    {'step': 3, 'marginal_probs': {'X1': 0.9, 'X2': 0.1}, 'target_value': 2.0, 'uncertainty': 1.0}
                ]
            }
        }
        true_parents = ['X1']
        
        # Extract metrics
        metrics = extract_metrics_from_experiment_result(experiment_result, true_parents)
        
        # Analyze convergence
        convergence = analyze_convergence_trajectory(metrics, threshold=0.8)
        
        # Compute efficiency
        efficiency = compute_intervention_efficiency(metrics)
        
        # Verify full pipeline works
        assert len(metrics['steps']) == 3
        assert convergence['converged'], "Should converge with this trajectory"
        assert efficiency['f1_per_intervention'] > 0, "Should show positive efficiency"
    
    def test_robustness_to_missing_data(self):
        """Test that functions handle missing/incomplete data gracefully."""
        # Experiment with missing fields
        experiment_result = {
            'method': 'static_surrogate',
            'detailed_results': {
                'learning_history': [
                    {'step': 1, 'marginal_probs': {'X1': 0.5}},  # Missing target_value, uncertainty
                    {'step': 2, 'target_value': 1.5},  # Missing marginal_probs
                ]
            }
        }
        true_parents = ['X1']
        
        # Should not crash
        metrics = extract_metrics_from_experiment_result(experiment_result, true_parents)
        convergence = analyze_convergence_trajectory(metrics)
        efficiency = compute_intervention_efficiency(metrics)
        
        # Should return reasonable defaults
        assert isinstance(metrics, dict)
        assert isinstance(convergence, dict)
        assert isinstance(efficiency, dict)


# Property-based tests using Hypothesis
class TestPropertyBased:
    """Property-based tests to verify mathematical properties."""
    
    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=3, alphabet="X123"),
            st.floats(min_value=0.0, max_value=1.0),
            min_size=1, max_size=5
        )
    )
    def test_likelihood_monotonicity(self, marginals: Dict[str, float]):
        """Property: Increasing marginal probabilities for true parents should increase likelihood."""
        if len(marginals) < 2:
            return  # Skip if not enough variables
        
        variables = list(marginals.keys())
        true_parents = [variables[0]]  # Single true parent
        
        # Original likelihood
        likelihood1 = compute_true_parent_likelihood(marginals, true_parents)
        
        # Increase probability of true parent
        modified_marginals = marginals.copy()
        old_prob = modified_marginals[variables[0]]
        new_prob = min(1.0, old_prob + 0.1)
        modified_marginals[variables[0]] = new_prob
        
        likelihood2 = compute_true_parent_likelihood(modified_marginals, true_parents)
        
        if new_prob > old_prob:
            assert likelihood2 >= likelihood1, "Increasing true parent probability should not decrease likelihood"
    
    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=3, alphabet="X123"),
            st.floats(min_value=0.0, max_value=1.0),
            min_size=2, max_size=4
        ),
        st.floats(min_value=0.1, max_value=0.9)
    )
    def test_f1_threshold_bounds(self, marginals: Dict[str, float], threshold: float):
        """Property: F1 score should be in [0, 1] for any threshold."""
        variables = list(marginals.keys())
        true_parents = variables[:len(variables)//2]
        
        f1 = compute_f1_score_from_marginals(marginals, true_parents, threshold)
        
        assert 0.0 <= f1 <= 1.0, f"F1 score {f1} not in [0, 1]"


class TestNewMetricFunctions:
    """Tests for newly added metric functions."""
    
    def test_compute_f1_with_multiple_thresholds(self):
        """Test F1 computation across multiple thresholds."""
        marginals = {'X1': 0.8, 'X2': 0.3, 'X3': 0.9}
        true_parents = ['X1', 'X3']
        
        f1_scores = compute_f1_with_multiple_thresholds(marginals, true_parents)
        
        # Should return dict of threshold -> F1 score
        assert isinstance(f1_scores, dict)
        assert len(f1_scores) > 0
        
        # All F1 scores should be in [0, 1]
        for threshold, f1 in f1_scores.items():
            assert 0.0 <= f1 <= 1.0
            assert 0.0 <= threshold <= 1.0
    
    def test_find_optimal_f1_threshold(self):
        """Test finding optimal F1 threshold."""
        marginals = {'X1': 0.9, 'X2': 0.1, 'X3': 0.8}
        true_parents = ['X1', 'X3']
        
        optimal_threshold, max_f1 = find_optimal_f1_threshold(marginals, true_parents)
        
        assert 0.0 <= optimal_threshold <= 1.0
        assert 0.0 <= max_f1 <= 1.0
        
        # Verify it's actually optimal
        f1_at_optimal = compute_f1_score_from_marginals(marginals, true_parents, optimal_threshold)
        assert abs(f1_at_optimal - max_f1) < 1e-10
    
    def test_find_youden_j_threshold(self):
        """Test Youden's J statistic threshold selection."""
        marginals = {'X1': 0.9, 'X2': 0.1, 'X3': 0.8, 'X4': 0.2}
        true_parents = ['X1', 'X3']
        
        youden_threshold, max_j = find_youden_j_threshold(marginals, true_parents)
        
        assert 0.0 <= youden_threshold <= 1.0
        assert -1.0 <= max_j <= 1.0  # J statistic ranges from -1 to 1
    
    def test_compute_expected_calibration_error(self):
        """Test ECE computation."""
        # Perfect calibration case
        marginals = {'X1': 1.0, 'X2': 0.0, 'X3': 1.0}
        true_parents = ['X1', 'X3']
        
        ece = compute_expected_calibration_error(marginals, true_parents)
        assert ece == 0.0, "Perfect calibration should have ECE = 0"
        
        # Poor calibration case
        marginals = {'X1': 0.5, 'X2': 0.5, 'X3': 0.5}
        true_parents = ['X1', 'X3']
        
        ece = compute_expected_calibration_error(marginals, true_parents)
        assert ece > 0.0, "Poor calibration should have ECE > 0"
    
    def test_compute_brier_score(self):
        """Test Brier score computation."""
        # Perfect predictions
        marginals = {'X1': 1.0, 'X2': 0.0, 'X3': 1.0}
        true_parents = ['X1', 'X3']
        
        brier = compute_brier_score(marginals, true_parents)
        assert brier == 0.0, "Perfect predictions should have Brier score = 0"
        
        # Worst case predictions
        marginals = {'X1': 0.0, 'X2': 1.0, 'X3': 0.0}
        true_parents = ['X1', 'X3']
        
        brier = compute_brier_score(marginals, true_parents)
        assert brier == 1.0, "Worst predictions should have Brier score = 1"
    
    def test_compute_comprehensive_metrics(self):
        """Test comprehensive metrics computation."""
        marginals = {'X1': 0.8, 'X2': 0.3, 'X3': 0.7, 'X4': 0.2}
        true_parents = ['X1', 'X3']
        
        metrics = compute_comprehensive_metrics(marginals, true_parents)
        
        # Should return all expected metrics
        assert 'true_parent_likelihood' in metrics
        assert 'f1_score' in metrics
        assert 'f1_score_optimal' in metrics
        assert 'optimal_f1_threshold' in metrics
        assert 'youden_j_threshold' in metrics
        assert 'max_youden_j' in metrics
        assert 'expected_calibration_error' in metrics
        assert 'brier_score' in metrics
        assert 'marginal_probs' in metrics
        
        # Verify metric ranges
        assert 0.0 <= metrics['true_parent_likelihood'] <= 1.0
        assert 0.0 <= metrics['f1_score'] <= 1.0
        assert 0.0 <= metrics['f1_score_optimal'] <= 1.0
        assert 0.0 <= metrics['expected_calibration_error'] <= 1.0
        assert 0.0 <= metrics['brier_score'] <= 1.0
    
    def test_compute_precision_recall_curve(self):
        """Test precision-recall curve computation."""
        marginals = {'X1': 0.9, 'X2': 0.3, 'X3': 0.7, 'X4': 0.1}
        true_parents = ['X1', 'X3']
        
        pr_data = compute_precision_recall_curve(marginals, true_parents)
        
        # Should return expected fields
        assert 'thresholds' in pr_data
        assert 'precision' in pr_data
        assert 'recall' in pr_data
        assert 'f1_scores' in pr_data
        
        # Same length for all arrays
        n_points = len(pr_data['thresholds'])
        assert len(pr_data['precision']) == n_points
        assert len(pr_data['recall']) == n_points
        assert len(pr_data['f1_scores']) == n_points
        
        # All values in valid range
        for i in range(n_points):
            assert 0.0 <= pr_data['precision'][i] <= 1.0
            assert 0.0 <= pr_data['recall'][i] <= 1.0
            assert 0.0 <= pr_data['f1_scores'][i] <= 1.0


class TestUpdatedExtractMetrics:
    """Test the updated extract_metrics_from_experiment_result function."""
    
    def test_extract_from_learning_history(self):
        """Test extraction from new learning_history format."""
        experiment_result = {
            'method': 'learning_surrogate',
            'true_parents': ['X0', 'X1'],
            'detailed_results': {
                'learning_history': [
                    {
                        'step': 1,
                        'outcome_value': 1.5,
                        'uncertainty': 2.0,
                        'marginals': {'X0': 0.3, 'X1': 0.2, 'X2': 0.8}
                    },
                    {
                        'step': 2,
                        'outcome_value': 2.0,
                        'uncertainty': 1.5,
                        'marginals': {'X0': 0.7, 'X1': 0.6, 'X2': 0.4}
                    },
                    {
                        'step': 3,
                        'outcome_value': 2.5,
                        'uncertainty': 1.0,
                        'marginals': {'X0': 0.9, 'X1': 0.8, 'X2': 0.1}
                    }
                ]
            }
        }
        true_parents = ['X0', 'X1']
        
        metrics = extract_metrics_from_experiment_result(experiment_result, true_parents)
        
        # Should extract correct number of steps
        assert len(metrics['steps']) == 3
        assert metrics['steps'] == [1, 2, 3]
        
        # Should extract target values and uncertainty
        assert metrics['target_values'] == [1.5, 2.0, 2.5]
        assert metrics['uncertainty_bits'] == [2.0, 1.5, 1.0]
        
        # Should compute F1 scores
        assert len(metrics['f1_scores']) == 3
        assert all(0.0 <= f1 <= 1.0 for f1 in metrics['f1_scores'])
        
        # Should compute likelihoods
        assert len(metrics['true_parent_likelihood']) == 3
        assert all(0.0 <= lik <= 1.0 for lik in metrics['true_parent_likelihood'])
        
        # Verify progression (should improve over time for this example)
        assert metrics['f1_scores'][-1] > metrics['f1_scores'][0]
        assert metrics['true_parent_likelihood'][-1] > metrics['true_parent_likelihood'][0]
    
    def test_configurable_threshold(self):
        """Test that F1 threshold parameter works correctly."""
        experiment_result = {
            'detailed_results': {
                'learning_history': [
                    {
                        'step': 1,
                        'marginals': {'X0': 0.6, 'X1': 0.4, 'X2': 0.3}
                    }
                ]
            }
        }
        true_parents = ['X0']
        
        # Test with different thresholds
        metrics_low = extract_metrics_from_experiment_result(
            experiment_result, true_parents, f1_threshold=0.3
        )
        metrics_high = extract_metrics_from_experiment_result(
            experiment_result, true_parents, f1_threshold=0.7
        )
        
        # Different thresholds should potentially give different F1 scores
        # With threshold=0.3: X0 and X1 predicted as parents (X0 correct, X1 false positive)
        # With threshold=0.7: No parents predicted (all below threshold)
        assert metrics_low['f1_scores'][0] != metrics_high['f1_scores'][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])