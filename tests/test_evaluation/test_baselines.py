"""
Comprehensive tests for evaluation baseline methods.

This module tests baseline intervention selection methods and comparison
functionality using property-based testing with Hypothesis.
"""

import pytest
import jax.numpy as jnp
import jax.random as random
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, patch

from causal_bayes_opt.evaluation.baselines import (
    BaselineResults, RandomBaseline, GreedyBaseline, ParentScaleBaseline,
    BaselineComparison, create_baseline_comparison
)


class TestBaselineResults:
    """Test BaselineResults data structure."""
    
    def test_baseline_results_immutable(self):
        """Test that BaselineResults is immutable."""
        results = BaselineResults(
            method_name="test_baseline",
            objective_values=(0.1, 0.3, 0.5),
            interventions=(jnp.array([1.0]), jnp.array([2.0])),
            targets=(0, 1),
            computational_time=10.0,
            total_cost=5.0
        )
        
        with pytest.raises(AttributeError):
            results.method_name = "modified"
    
    @given(
        method_name=st.text(min_size=1, max_size=50),
        num_steps=st.integers(min_value=1, max_value=20),
        computational_time=st.floats(min_value=0.1, max_value=100.0),
        total_cost=st.floats(min_value=0.1, max_value=50.0)
    )
    @settings(max_examples=10)
    def test_baseline_results_properties(self, method_name, num_steps, computational_time, total_cost):
        """Property-based test for BaselineResults creation."""
        objective_values = tuple(float(i) * 0.1 for i in range(num_steps))
        interventions = tuple(jnp.array([float(i)]) for i in range(num_steps))
        targets = tuple(i % 3 for i in range(num_steps))
        
        results = BaselineResults(
            method_name=method_name,
            objective_values=objective_values,
            interventions=interventions,
            targets=targets,
            computational_time=computational_time,
            total_cost=total_cost
        )
        
        assert results.method_name == method_name
        assert len(results.objective_values) == num_steps
        assert len(results.interventions) == num_steps
        assert len(results.targets) == num_steps
        assert results.computational_time == computational_time
        assert results.total_cost == total_cost


class TestRandomBaseline:
    """Test RandomBaseline intervention selection."""
    
    def test_random_baseline_initialization(self):
        """Test RandomBaseline initialization."""
        baseline = RandomBaseline()
        assert baseline.intervention_range == (-2.0, 2.0)
        
        custom_baseline = RandomBaseline(intervention_range=(-1.0, 1.0))
        assert custom_baseline.intervention_range == (-1.0, 1.0)
    
    @given(
        num_targets=st.integers(min_value=2, max_value=10),
        intervention_min=st.floats(min_value=-5.0, max_value=-0.1),
        intervention_max=st.floats(min_value=0.1, max_value=5.0)
    )
    @settings(max_examples=20)
    def test_random_baseline_selection_properties(self, num_targets, intervention_min, intervention_max):
        """Property-based test for random baseline selection."""
        baseline = RandomBaseline(intervention_range=(intervention_min, intervention_max))
        available_targets = jnp.arange(num_targets)
        key = random.PRNGKey(42)
        
        target_idx, intervention_value = baseline.select_intervention(
            state=None,
            available_targets=available_targets,
            key=key
        )
        
        # Target should be valid
        assert 0 <= target_idx < num_targets
        assert isinstance(target_idx, int)
        
        # Intervention value should be in range
        assert intervention_min <= intervention_value <= intervention_max
        assert isinstance(intervention_value, (float, jnp.ndarray))
    
    def test_random_baseline_deterministic_with_key(self):
        """Test that random baseline is deterministic given the same key."""
        baseline = RandomBaseline()
        available_targets = jnp.arange(5)
        key = random.PRNGKey(123)
        
        # Multiple calls with same key should give same result
        target1, intervention1 = baseline.select_intervention(None, available_targets, key)
        target2, intervention2 = baseline.select_intervention(None, available_targets, key)
        
        assert target1 == target2
        assert jnp.allclose(intervention1, intervention2)
    
    def test_random_baseline_different_keys_different_results(self):
        """Test that different keys produce different results."""
        baseline = RandomBaseline()
        available_targets = jnp.arange(5)
        
        key1 = random.PRNGKey(42)
        key2 = random.PRNGKey(123)
        
        target1, intervention1 = baseline.select_intervention(None, available_targets, key1)
        target2, intervention2 = baseline.select_intervention(None, available_targets, key2)
        
        # Should be different (with very high probability)
        assert target1 != target2 or not jnp.allclose(intervention1, intervention2)


class TestGreedyBaseline:
    """Test GreedyBaseline intervention selection."""
    
    def test_greedy_baseline_initialization(self):
        """Test GreedyBaseline initialization."""
        baseline = GreedyBaseline()
        assert baseline.step_size == 0.1
        assert baseline.exploration_prob == 0.1
        assert baseline.history == []
        
        custom_baseline = GreedyBaseline(step_size=0.2, exploration_prob=0.2)
        assert custom_baseline.step_size == 0.2
        assert custom_baseline.exploration_prob == 0.2
    
    def test_greedy_baseline_no_history(self):
        """Test greedy baseline selection with no history."""
        baseline = GreedyBaseline()
        available_targets = jnp.arange(3)
        key = random.PRNGKey(42)
        
        target_idx, intervention_value = baseline.select_intervention(
            state=None,
            available_targets=available_targets,
            key=key
        )
        
        # Should make valid selection even without history
        assert 0 <= target_idx < len(available_targets)
        assert isinstance(intervention_value, (float, jnp.ndarray))
    
    def test_greedy_baseline_with_history(self):
        """Test greedy baseline selection with history."""
        baseline = GreedyBaseline(exploration_prob=0.0)  # No exploration for deterministic test
        baseline.available_targets = jnp.arange(3)  # Set for analysis
        
        # Add some history
        baseline.update_history(target=0, intervention=0.1, objective=0.2)
        baseline.update_history(target=1, intervention=0.2, objective=0.5)
        baseline.update_history(target=0, intervention=0.15, objective=0.3)
        
        available_targets = jnp.arange(3)
        key = random.PRNGKey(42)
        
        # Should use history for selection
        target_idx, intervention_value = baseline.select_intervention(
            state=None,
            available_targets=available_targets,
            key=key
        )
        
        assert 0 <= target_idx < len(available_targets)
    
    def test_greedy_baseline_exploration_vs_exploitation(self):
        """Test exploration vs exploitation in greedy baseline."""
        # High exploration probability
        exploratory_baseline = GreedyBaseline(exploration_prob=0.9)
        
        # Low exploration probability  
        exploitative_baseline = GreedyBaseline(exploration_prob=0.1)
        
        available_targets = jnp.arange(5)
        
        # Test multiple selections (statistical test)
        exploration_selections = []
        exploitation_selections = []
        
        for i in range(20):
            key = random.PRNGKey(i)
            
            exp_target, _ = exploratory_baseline.select_intervention(None, available_targets, key)
            expl_target, _ = exploitative_baseline.select_intervention(None, available_targets, key)
            
            exploration_selections.append(exp_target)
            exploitation_selections.append(expl_target)
        
        # Higher exploration should lead to more diverse selections
        exploration_diversity = len(set(exploration_selections))
        exploitation_diversity = len(set(exploitation_selections))
        
        # This is a statistical test, so we allow some tolerance
        assert exploration_diversity >= exploitation_diversity - 1
    
    @given(
        step_size=st.floats(min_value=0.01, max_value=1.0),
        exploration_prob=st.floats(min_value=0.0, max_value=1.0),
        num_targets=st.integers(min_value=2, max_value=8)
    )
    @settings(max_examples=15)
    def test_greedy_baseline_properties(self, step_size, exploration_prob, num_targets):
        """Property-based test for greedy baseline."""
        baseline = GreedyBaseline(step_size=step_size, exploration_prob=exploration_prob)
        available_targets = jnp.arange(num_targets)
        key = random.PRNGKey(42)
        
        target_idx, intervention_value = baseline.select_intervention(
            state=None,
            available_targets=available_targets,
            key=key
        )
        
        # Basic validity checks
        assert 0 <= target_idx < num_targets
        assert isinstance(target_idx, int)
        assert isinstance(intervention_value, (float, jnp.ndarray))
        
        # Intervention value should be reasonable
        assert abs(intervention_value) <= 10 * step_size  # Reasonable bound


class TestParentScaleBaseline:
    """Test ParentScaleBaseline intervention selection."""
    
    def test_parent_scale_baseline_initialization(self):
        """Test ParentScaleBaseline initialization."""
        baseline = ParentScaleBaseline()
        assert baseline.n_particles == 100
        assert baseline.exploration_factor == 1.0
        assert baseline.particle_beliefs is None
        
        custom_baseline = ParentScaleBaseline(n_particles=50, exploration_factor=2.0)
        assert custom_baseline.n_particles == 50
        assert custom_baseline.exploration_factor == 2.0
    
    def test_parent_scale_baseline_particle_initialization(self):
        """Test particle initialization in ParentScaleBaseline."""
        baseline = ParentScaleBaseline(n_particles=10)
        available_targets = jnp.arange(4)
        key = random.PRNGKey(42)
        
        # First call should initialize particles
        target_idx, intervention_value = baseline.select_intervention(
            state=None,
            available_targets=available_targets,
            key=key
        )
        
        # Particles should be initialized
        assert baseline.particle_beliefs is not None
        assert baseline.particle_beliefs.shape == (10, 4, 4)  # (n_particles, n_vars, n_vars)
        
        # Should return valid selection
        assert 0 <= target_idx < len(available_targets)
        assert isinstance(intervention_value, (float, jnp.ndarray))
    
    @given(
        n_particles=st.integers(min_value=10, max_value=100),
        exploration_factor=st.floats(min_value=0.1, max_value=5.0),
        num_variables=st.integers(min_value=2, max_value=6)
    )
    @settings(max_examples=10)
    def test_parent_scale_baseline_properties(self, n_particles, exploration_factor, num_variables):
        """Property-based test for ParentScaleBaseline."""
        baseline = ParentScaleBaseline(n_particles=n_particles, exploration_factor=exploration_factor)
        available_targets = jnp.arange(num_variables)
        key = random.PRNGKey(42)
        
        target_idx, intervention_value = baseline.select_intervention(
            state=None,
            available_targets=available_targets,
            key=key
        )
        
        # Valid selection
        assert 0 <= target_idx < num_variables
        assert isinstance(target_idx, int)
        assert isinstance(intervention_value, (float, jnp.ndarray))
        
        # Particles should be initialized correctly
        assert baseline.particle_beliefs.shape == (n_particles, num_variables, num_variables)
    
    def test_parent_scale_baseline_consistent_with_beliefs(self):
        """Test that ParentScaleBaseline selections are consistent with particle beliefs."""
        baseline = ParentScaleBaseline(n_particles=20)
        available_targets = jnp.arange(3)
        key = random.PRNGKey(42)
        
        # Make multiple selections
        selections = []
        for i in range(5):
            subkey = random.fold_in(key, i)
            target, intervention = baseline.select_intervention(None, available_targets, subkey)
            selections.append((target, intervention))
        
        # Should make reasonable selections (this is a basic sanity check)
        targets = [s[0] for s in selections]
        assert all(0 <= t < len(available_targets) for t in targets)


class TestBaselineComparison:
    """Test baseline comparison functionality."""
    
    def test_baseline_comparison_immutable(self):
        """Test that BaselineComparison is immutable."""
        acbo_results = BaselineResults("acbo", (0.8,), (jnp.array([1.0]),), (0,), 10.0, 5.0)
        baseline_results = {"random": BaselineResults("random", (0.5,), (jnp.array([0.5]),), (1,), 8.0, 3.0)}
        
        comparison = BaselineComparison(
            acbo_results=acbo_results,
            baseline_results=baseline_results,
            relative_performance={"random": 0.3},
            statistical_significance={"random": True}
        )
        
        with pytest.raises(AttributeError):
            comparison.relative_performance = {}
    
    def test_create_baseline_comparison_basic(self):
        """Test basic baseline comparison creation."""
        acbo_performance = {
            'objective_values': [0.2, 0.5, 0.8],
            'interventions': [jnp.array([1.0]), jnp.array([2.0]), jnp.array([3.0])],
            'targets': [0, 1, 2],
            'computational_time': 15.0,
            'total_cost': 10.0
        }
        
        with patch('causal_bayes_opt.evaluation.baselines._evaluate_baseline') as mock_eval:
            # Mock baseline evaluation
            mock_eval.return_value = BaselineResults(
                method_name="random",
                objective_values=(0.1, 0.3, 0.6),
                interventions=(jnp.array([0.5]), jnp.array([1.0]), jnp.array([1.5])),
                targets=(0, 1, 0),
                computational_time=10.0,
                total_cost=8.0
            )
            
            comparison = create_baseline_comparison(
                acbo_performance=acbo_performance,
                baselines=['random'],
                n_trials=3,
                random_seed=42
            )
        
        assert isinstance(comparison, BaselineComparison)
        assert comparison.acbo_results.method_name == 'acbo'
        assert 'random' in comparison.baseline_results
        assert 'random' in comparison.relative_performance
        assert 'random' in comparison.statistical_significance
    
    def test_create_baseline_comparison_multiple_baselines(self):
        """Test baseline comparison with multiple baseline methods."""
        acbo_performance = {
            'objective_values': [0.3, 0.6, 0.9],
            'computational_time': 20.0,
            'total_cost': 12.0
        }
        
        def mock_evaluate_baseline(baseline_name, *args, **kwargs):
            if baseline_name == 'random':
                return BaselineResults("random", (0.2, 0.4, 0.7), (), (), 15.0, 8.0)
            elif baseline_name == 'greedy':
                return BaselineResults("greedy", (0.25, 0.5, 0.8), (), (), 18.0, 10.0)
            else:
                return BaselineResults(baseline_name, (0.1, 0.3, 0.6), (), (), 12.0, 6.0)
        
        with patch('causal_bayes_opt.evaluation.baselines._evaluate_baseline', side_effect=mock_evaluate_baseline):
            comparison = create_baseline_comparison(
                acbo_performance=acbo_performance,
                baselines=['random', 'greedy', 'parent_scale'],
                n_trials=5
            )
        
        # Should compare against all baselines
        assert len(comparison.baseline_results) == 3
        assert all(baseline in comparison.baseline_results for baseline in ['random', 'greedy', 'parent_scale'])
        assert len(comparison.relative_performance) == 3
        assert len(comparison.statistical_significance) == 3
    
    @given(
        acbo_final_value=st.floats(min_value=0.0, max_value=1.0),
        baseline_final_value=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=20)
    def test_relative_performance_calculation(self, acbo_final_value, baseline_final_value):
        """Property-based test for relative performance calculation."""
        acbo_performance = {
            'objective_values': [0.1, acbo_final_value],
            'computational_time': 10.0,
            'total_cost': 5.0
        }
        
        def mock_evaluate_baseline(baseline_name, *args, **kwargs):
            return BaselineResults(
                baseline_name, 
                (0.1, baseline_final_value), 
                (), (), 10.0, 5.0
            )
        
        with patch('causal_bayes_opt.evaluation.baselines._evaluate_baseline', side_effect=mock_evaluate_baseline):
            comparison = create_baseline_comparison(
                acbo_performance=acbo_performance,
                baselines=['test_baseline'],
                n_trials=2
            )
        
        expected_relative = acbo_final_value - baseline_final_value
        actual_relative = comparison.relative_performance['test_baseline']
        
        assert abs(actual_relative - expected_relative) < 1e-6
    
    def test_create_baseline_comparison_default_baselines(self):
        """Test baseline comparison with default baseline methods."""
        acbo_performance = {
            'objective_values': [0.2, 0.7],
            'computational_time': 10.0,
            'total_cost': 5.0
        }
        
        with patch('causal_bayes_opt.evaluation.baselines._evaluate_baseline') as mock_eval:
            mock_eval.return_value = BaselineResults("test", (0.1, 0.5), (), (), 8.0, 4.0)
            
            comparison = create_baseline_comparison(
                acbo_performance=acbo_performance,
                baselines=None,  # Use defaults
                n_trials=2
            )
        
        # Should use default baselines
        default_baselines = ['random', 'greedy', 'parent_scale']
        assert all(baseline in comparison.baseline_results for baseline in default_baselines)


class TestBaselineEvaluation:
    """Test individual baseline evaluation functionality."""
    
    def test_evaluate_random_baseline(self):
        """Test evaluation of random baseline."""
        from causal_bayes_opt.evaluation.baselines import _evaluate_baseline
        
        reference_setup = {
            'n_variables': 4,
            'n_trials': 5
        }
        
        with patch('jax.random.normal') as mock_normal:
            mock_normal.return_value = 0.1  # Mock objective improvement
            
            results = _evaluate_baseline('random', reference_setup, 5, random.PRNGKey(42))
        
        assert isinstance(results, BaselineResults)
        assert results.method_name == 'random'
        assert len(results.objective_values) == 5
        assert len(results.interventions) == 5
        assert len(results.targets) == 5
        assert results.computational_time > 0
        assert results.total_cost > 0
    
    def test_evaluate_greedy_baseline(self):
        """Test evaluation of greedy baseline."""
        from causal_bayes_opt.evaluation.baselines import _evaluate_baseline
        
        reference_setup = {'n_variables': 3}
        
        results = _evaluate_baseline('greedy', reference_setup, 3, random.PRNGKey(123))
        
        assert results.method_name == 'greedy'
        assert len(results.objective_values) == 3
    
    def test_evaluate_parent_scale_baseline(self):
        """Test evaluation of PARENT_SCALE baseline."""
        from causal_bayes_opt.evaluation.baselines import _evaluate_baseline
        
        reference_setup = {'n_variables': 4}
        
        results = _evaluate_baseline('parent_scale', reference_setup, 4, random.PRNGKey(456))
        
        assert results.method_name == 'parent_scale'
        assert len(results.objective_values) == 4
    
    def test_evaluate_unknown_baseline_error(self):
        """Test error handling for unknown baseline method."""
        from causal_bayes_opt.evaluation.baselines import _evaluate_baseline
        
        with pytest.raises(ValueError, match="Unknown baseline method"):
            _evaluate_baseline('unknown_method', {}, 5, random.PRNGKey(42))


class TestBaselineEdgeCases:
    """Test edge cases and error conditions in baseline methods."""
    
    def test_random_baseline_empty_targets(self):
        """Test random baseline with empty target list."""
        baseline = RandomBaseline()
        empty_targets = jnp.array([])
        key = random.PRNGKey(42)
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, IndexError)):
            baseline.select_intervention(None, empty_targets, key)
    
    def test_greedy_baseline_update_history(self):
        """Test greedy baseline history update functionality."""
        baseline = GreedyBaseline()
        
        # Update history
        baseline.update_history(target=0, intervention=0.1, objective=0.5)
        baseline.update_history(target=1, intervention=0.2, objective=0.7)
        
        assert len(baseline.history) == 2
        assert baseline.history[0]['target'] == 0
        assert baseline.history[0]['intervention'] == 0.1
        assert baseline.history[0]['objective'] == 0.5
    
    def test_parent_scale_baseline_repeated_calls(self):
        """Test ParentScaleBaseline with repeated calls (particles should persist)."""
        baseline = ParentScaleBaseline(n_particles=10)
        available_targets = jnp.arange(3)
        key = random.PRNGKey(42)
        
        # First call initializes particles
        baseline.select_intervention(None, available_targets, key)
        initial_particles = baseline.particle_beliefs.copy()
        
        # Second call should use existing particles
        baseline.select_intervention(None, available_targets, random.PRNGKey(123))
        
        # Particles should be the same object (not re-initialized)
        assert jnp.array_equal(baseline.particle_beliefs, initial_particles)
    
    def test_baseline_comparison_empty_acbo_performance(self):
        """Test baseline comparison with minimal ACBO performance data."""
        minimal_acbo = {
            'objective_values': [0.5],
            'computational_time': 1.0
        }
        
        with patch('causal_bayes_opt.evaluation.baselines._evaluate_baseline') as mock_eval:
            mock_eval.return_value = BaselineResults("test", (0.3,), (), (), 1.0, 1.0)
            
            comparison = create_baseline_comparison(
                acbo_performance=minimal_acbo,
                baselines=['random'],
                n_trials=1
            )
        
        # Should handle minimal data gracefully
        assert isinstance(comparison, BaselineComparison)
        assert comparison.acbo_results.method_name == 'acbo'


class TestBaselinePerformance:
    """Test performance characteristics of baseline methods."""
    
    def test_baseline_selection_speed(self):
        """Test that baseline selection is reasonably fast."""
        import time
        
        baselines = [
            RandomBaseline(),
            GreedyBaseline(),
            ParentScaleBaseline(n_particles=10)  # Small number for speed
        ]
        
        available_targets = jnp.arange(5)
        key = random.PRNGKey(42)
        
        for baseline in baselines:
            start_time = time.time()
            
            # Make multiple selections
            for i in range(10):
                subkey = random.fold_in(key, i)
                baseline.select_intervention(None, available_targets, subkey)
            
            elapsed = time.time() - start_time
            
            # Should be fast (less than 1 second for 10 selections)
            assert elapsed < 1.0
    
    @given(
        num_selections=st.integers(min_value=5, max_value=50),
        num_targets=st.integers(min_value=2, max_value=10)
    )
    @settings(max_examples=5)
    def test_baseline_selection_scalability(self, num_selections, num_targets):
        """Test that baseline selection scales reasonably with problem size."""
        baseline = RandomBaseline()  # Use random as it's simplest
        available_targets = jnp.arange(num_targets)
        
        # Should handle larger problem sizes without issues
        for i in range(num_selections):
            key = random.PRNGKey(i)
            target, intervention = baseline.select_intervention(None, available_targets, key)
            
            assert 0 <= target < num_targets
            assert isinstance(intervention, (float, jnp.ndarray))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])