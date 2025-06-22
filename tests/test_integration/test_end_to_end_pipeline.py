"""
End-to-End Pipeline Integration Tests

Tests that validate the complete JAX-native pipeline works correctly from 
configuration creation through policy feature extraction. Follows CLAUDE.md 
principles with pure functions and immutable data structures.

Test scenarios:
1. Complete pipeline: config -> state -> samples -> features
2. Multiple variable configurations
3. Circular buffer edge cases (empty, full, wraparound)  
4. Target variable masking consistency
5. Error handling and edge cases
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
from typing import Dict, List, Tuple
import time

from causal_bayes_opt.jax_native import (
    JAXConfig, JAXSampleBuffer, JAXAcquisitionState,
    create_jax_config, create_jax_state, create_empty_jax_buffer
)
from causal_bayes_opt.jax_native.state import add_sample_to_state_jax
from causal_bayes_opt.jax_native.sample_buffer import add_sample_jax
from causal_bayes_opt.jax_native.operations import (
    compute_mechanism_confidence_jax,
    compute_policy_features_jax,
    compute_optimization_progress_jax,
    compute_exploration_coverage_jax
)
from causal_bayes_opt.jax_native.sample_buffer import compute_buffer_statistics_jax


class TestBasicPipelineIntegration:
    """Test basic end-to-end pipeline functionality."""
    
    def test_complete_pipeline_small_case(self):
        """Test complete pipeline with 3 variables."""
        # 1. Create configuration
        config = create_jax_config(['X', 'Y', 'Z'], 'Y', max_samples=10)
        assert config.n_vars == 3
        assert config.target_idx == 1
        
        # 2. Create initial state
        state = create_jax_state(config)
        assert state.is_buffer_empty()
        assert state.best_value == 0.0
        assert state.current_step == 0
        
        # 3. Add samples to build up history
        key = random.PRNGKey(42)
        
        samples_added = 0
        for i in range(5):
            key, subkey = random.split(key)
            
            # Generate sample data
            values = random.normal(subkey, (3,))
            interventions = jnp.zeros(3, dtype=bool).at[i % 2].set(True)  # Alternate X/Z interventions
            target_value = float(values[1])  # Y is target
            
            # Add sample using buffer operations (JAX-compatible)
            new_buffer = add_sample_jax(state.sample_buffer, values, interventions, target_value)
            new_best_value = max(state.best_value, target_value)
            state = JAXAcquisitionState(
                sample_buffer=new_buffer,
                mechanism_features=state.mechanism_features,
                marginal_probs=state.marginal_probs,
                confidence_scores=state.confidence_scores,
                best_value=new_best_value,
                current_step=state.current_step + 1,
                uncertainty_bits=state.uncertainty_bits,
                config=state.config
            )
            samples_added += 1
        
        # 4. Verify state consistency
        assert state.sample_buffer.n_samples == samples_added
        assert state.current_step == samples_added
        assert not state.is_buffer_empty()
        
        # 5. Compute mechanism confidence
        confidence = compute_mechanism_confidence_jax(state)
        assert confidence.shape == (3,)
        assert confidence[1] == 0.0  # Target variable masked
        assert jnp.all(confidence >= 0.0)
        assert jnp.all(confidence <= 1.0)
        
        # 6. Extract policy features
        policy_features = compute_policy_features_jax(state)
        assert policy_features.shape[0] == 3  # n_vars
        assert policy_features.shape[1] > 3   # Multiple feature dimensions
        assert jnp.all(jnp.isfinite(policy_features))
        
        # 7. Compute progress metrics
        progress = compute_optimization_progress_jax(state)
        assert isinstance(progress, dict)
        assert 'improvement_from_start' in progress
        assert 'optimization_rate' in progress
        
        # 8. Compute exploration coverage
        coverage = compute_exploration_coverage_jax(state)
        assert isinstance(coverage, dict)
        assert 'target_coverage_rate' in coverage
        assert 'intervention_diversity' in coverage
        
        print(f"✓ Complete pipeline test passed with {samples_added} samples")
    
    def test_pipeline_with_different_variable_counts(self):
        """Test pipeline works with different numbers of variables."""
        test_cases = [
            (['A', 'B'], 'B', 2),
            (['X', 'Y', 'Z', 'W'], 'W', 4),
            (['V1', 'V2', 'V3', 'V4', 'V5'], 'V3', 5)
        ]
        
        for variables, target, expected_n_vars in test_cases:
            # Create configuration
            config = create_jax_config(variables, target, max_samples=20)
            assert config.n_vars == expected_n_vars
            assert config.get_target_name() == target
            
            # Create and populate state
            state = create_jax_state(config)
            key = random.PRNGKey(123)
            
            # Add samples with interventions on different variables
            for i in range(min(10, expected_n_vars * 2)):
                key, subkey = random.split(key)
                values = random.normal(subkey, (expected_n_vars,))
                
                # Intervention on non-target variable
                intervention_var = i % expected_n_vars
                if intervention_var == config.target_idx:
                    intervention_var = (intervention_var + 1) % expected_n_vars
                
                interventions = jnp.zeros(expected_n_vars, dtype=bool).at[intervention_var].set(True)
                target_value = float(values[config.target_idx])
                
                new_buffer = add_sample_jax(state.sample_buffer, values, interventions, target_value)
                state = JAXAcquisitionState(
                    sample_buffer=new_buffer,
                    mechanism_features=state.mechanism_features,
                    marginal_probs=state.marginal_probs,
                    confidence_scores=state.confidence_scores,
                    best_value=max(state.best_value, target_value),
                    current_step=state.current_step + 1,
                    uncertainty_bits=state.uncertainty_bits,
                    config=state.config
                )
            
            # Verify pipeline works
            confidence = compute_mechanism_confidence_jax(state)
            assert confidence.shape == (expected_n_vars,)
            assert confidence[config.target_idx] == 0.0
            
            features = compute_policy_features_jax(state)
            assert features.shape[0] == expected_n_vars
            
            print(f"✓ Pipeline works with {expected_n_vars} variables")
    
    def test_circular_buffer_wraparound(self):
        """Test pipeline behavior when circular buffer wraps around."""
        # Small buffer to force wraparound
        config = create_jax_config(['X', 'Y', 'Z'], 'Y', max_samples=3)
        state = create_jax_state(config)
        
        key = random.PRNGKey(456)
        
        # Add more samples than buffer capacity
        samples_to_add = 8
        for i in range(samples_to_add):
            key, subkey = random.split(key)
            values = random.normal(subkey, (3,))
            interventions = jnp.zeros(3, dtype=bool).at[i % 2].set(True)
            target_value = float(values[1])
            
            new_buffer = add_sample_jax(state.sample_buffer, values, interventions, target_value)
            state = JAXAcquisitionState(
                sample_buffer=new_buffer,
                mechanism_features=state.mechanism_features,
                marginal_probs=state.marginal_probs,
                confidence_scores=state.confidence_scores,
                best_value=max(state.best_value, target_value),
                current_step=state.current_step + 1,
                uncertainty_bits=state.uncertainty_bits,
                config=state.config
            )
        
        # Buffer should be full but not exceed capacity
        assert state.sample_buffer.n_samples == config.max_samples
        assert state.sample_buffer.is_full()
        
        # Pipeline should still work correctly
        confidence = compute_mechanism_confidence_jax(state)
        assert confidence.shape == (3,)
        assert confidence[1] == 0.0
        
        features = compute_policy_features_jax(state)
        assert features.shape[0] == 3
        
        # Buffer statistics should be consistent
        stats = compute_buffer_statistics_jax(state.sample_buffer)
        assert stats['total_samples'] == config.max_samples
        assert stats['buffer_utilization'] == 1.0
        
        print("✓ Circular buffer wraparound handled correctly")


class TestPipelineEdgeCases:
    """Test edge cases and error conditions in the pipeline."""
    
    def test_empty_buffer_operations(self):
        """Test pipeline operations with empty buffer."""
        config = create_jax_config(['X', 'Y'], 'Y', max_samples=10)
        state = create_jax_state(config)
        
        # All operations should work with empty buffer
        confidence = compute_mechanism_confidence_jax(state)
        assert confidence.shape == (2,)
        assert confidence[1] == 0.0  # Target masked
        
        features = compute_policy_features_jax(state)
        assert features.shape[0] == 2
        assert jnp.all(jnp.isfinite(features))
        
        progress = compute_optimization_progress_jax(state)
        assert progress['improvement_from_start'] == 0.0
        
        coverage = compute_exploration_coverage_jax(state)
        assert coverage['target_coverage_rate'] >= 0.0
        
        print("✓ Empty buffer operations work correctly")
    
    def test_single_variable_plus_target(self):
        """Test minimum case: one variable plus target."""
        config = create_jax_config(['X', 'Y'], 'Y', max_samples=5)
        state = create_jax_state(config)
        
        # Add a few samples
        key = random.PRNGKey(789)
        for i in range(3):
            key, subkey = random.split(key)
            values = random.normal(subkey, (2,))
            interventions = jnp.array([True, False])  # Only intervene on X
            target_value = float(values[1])
            
            new_buffer = add_sample_jax(state.sample_buffer, values, interventions, target_value)
            state = JAXAcquisitionState(
                sample_buffer=new_buffer,
                mechanism_features=state.mechanism_features,
                marginal_probs=state.marginal_probs,
                confidence_scores=state.confidence_scores,
                best_value=max(state.best_value, target_value),
                current_step=state.current_step + 1,
                uncertainty_bits=state.uncertainty_bits,
                config=state.config
            )
        
        # Pipeline should work
        confidence = compute_mechanism_confidence_jax(state)
        assert confidence.shape == (2,)
        assert confidence[1] == 0.0  # Y is target
        
        features = compute_policy_features_jax(state)
        assert features.shape[0] == 2
        
        print("✓ Minimum configuration (2 variables) works")
    
    def test_target_variable_consistency(self):
        """Test that target variable is consistently masked across operations."""
        config = create_jax_config(['A', 'B', 'C', 'D'], 'B', max_samples=10)
        state = create_jax_state(config)
        
        # Add samples
        key = random.PRNGKey(101112)
        for i in range(6):
            key, subkey = random.split(key)
            values = random.normal(subkey, (4,))
            
            # Never intervene on target variable
            intervention_vars = [0, 2, 3]  # A, C, D (not B)
            intervention_idx = intervention_vars[i % len(intervention_vars)]
            interventions = jnp.zeros(4, dtype=bool).at[intervention_idx].set(True)
            target_value = float(values[1])  # B is target
            
            new_buffer = add_sample_jax(state.sample_buffer, values, interventions, target_value)
            state = JAXAcquisitionState(
                sample_buffer=new_buffer,
                mechanism_features=state.mechanism_features,
                marginal_probs=state.marginal_probs,
                confidence_scores=state.confidence_scores,
                best_value=max(state.best_value, target_value),
                current_step=state.current_step + 1,
                uncertainty_bits=state.uncertainty_bits,
                config=state.config
            )
        
        # Target should be masked in confidence
        confidence = compute_mechanism_confidence_jax(state)
        assert confidence[config.target_idx] == 0.0
        
        # Target should be handled appropriately in features
        features = compute_policy_features_jax(state)
        # Target row should have specific characteristics (likely zeros or constant values)
        target_features = features[config.target_idx]
        assert jnp.all(jnp.isfinite(target_features))
        
        print(f"✓ Target variable (index {config.target_idx}) consistently masked")
    
    def test_extreme_values_handling(self):
        """Test pipeline behavior with extreme values."""
        config = create_jax_config(['X', 'Y', 'Z'], 'Y', max_samples=5)
        state = create_jax_state(config)
        
        # Add samples with extreme values
        extreme_samples = [
            jnp.array([1000.0, -1000.0, 500.0]),  # Large positive/negative
            jnp.array([0.0001, 0.0, -0.0001]),    # Very small values
            jnp.array([jnp.pi, jnp.e, jnp.sqrt(2.0)]),  # Irrational numbers
        ]
        
        for i, values in enumerate(extreme_samples):
            interventions = jnp.zeros(3, dtype=bool).at[i % 2].set(True)
            target_value = float(values[1])
            
            new_buffer = add_sample_jax(state.sample_buffer, values, interventions, target_value)
            state = JAXAcquisitionState(
                sample_buffer=new_buffer,
                mechanism_features=state.mechanism_features,
                marginal_probs=state.marginal_probs,
                confidence_scores=state.confidence_scores,
                best_value=max(state.best_value, target_value),
                current_step=state.current_step + 1,
                uncertainty_bits=state.uncertainty_bits,
                config=state.config
            )
        
        # Pipeline should handle extreme values gracefully
        confidence = compute_mechanism_confidence_jax(state)
        assert jnp.all(jnp.isfinite(confidence))
        assert jnp.all(confidence >= 0.0)
        assert jnp.all(confidence <= 1.0)
        
        features = compute_policy_features_jax(state)
        assert jnp.all(jnp.isfinite(features))
        
        print("✓ Extreme values handled gracefully")


class TestPipelinePerformance:
    """Test pipeline performance characteristics."""
    
    def test_pipeline_scales_with_problem_size(self):
        """Test that pipeline performance scales reasonably with problem size."""
        problem_sizes = [3, 5, 10, 20]
        timing_results = {}
        
        for n_vars in problem_sizes:
            # Create configuration
            variables = [f"X{i}" for i in range(n_vars)]
            target = variables[-1]  # Last variable as target
            config = create_jax_config(variables, target, max_samples=50)
            state = create_jax_state(config)
            
            # Add samples
            key = random.PRNGKey(n_vars)  # Different seed per size
            
            start_time = time.perf_counter()
            
            for i in range(20):  # Fixed number of samples
                key, subkey = random.split(key)
                values = random.normal(subkey, (n_vars,))
                intervention_var = i % (n_vars - 1)  # Avoid target
                interventions = jnp.zeros(n_vars, dtype=bool).at[intervention_var].set(True)
                target_value = float(values[-1])
                
                new_buffer = add_sample_jax(state.sample_buffer, values, interventions, target_value)
                state = JAXAcquisitionState(
                    sample_buffer=new_buffer,
                    mechanism_features=state.mechanism_features,
                    marginal_probs=state.marginal_probs,
                    confidence_scores=state.confidence_scores,
                    best_value=max(state.best_value, target_value),
                    current_step=state.current_step + 1,
                    uncertainty_bits=state.uncertainty_bits,
                    config=state.config
                )
            
            # Time complete pipeline operations
            pipeline_start = time.perf_counter()
            
            confidence = compute_mechanism_confidence_jax(state)
            features = compute_policy_features_jax(state)
            progress = compute_optimization_progress_jax(state)
            coverage = compute_exploration_coverage_jax(state)
            
            pipeline_time = time.perf_counter() - pipeline_start
            total_time = time.perf_counter() - start_time
            
            timing_results[n_vars] = {
                'total_time': total_time * 1000,  # ms
                'pipeline_time': pipeline_time * 1000,  # ms
                'samples': 20
            }
            
            # Verify correctness
            assert confidence.shape == (n_vars,)
            assert features.shape[0] == n_vars
            assert isinstance(progress, dict)
            assert isinstance(coverage, dict)
            
            print(f"✓ {n_vars} variables: {total_time*1000:.1f}ms total, {pipeline_time*1000:.2f}ms pipeline")
        
        # Verify scaling is reasonable (not exponential)
        for n_vars in problem_sizes[1:]:
            prev_vars = n_vars // 2 if n_vars > 3 else 3
            if prev_vars in timing_results:
                time_ratio = timing_results[n_vars]['pipeline_time'] / timing_results[prev_vars]['pipeline_time']
                vars_ratio = n_vars / prev_vars
                
                # Time should scale roughly linearly, not exponentially
                assert time_ratio < vars_ratio * 3, f"Poor scaling: {time_ratio:.1f}x time for {vars_ratio:.1f}x variables"
        
        print("✓ Pipeline scales reasonably with problem size")
    
    def test_compilation_amortization(self):
        """Test that JAX compilation cost is amortized over multiple runs."""
        config = create_jax_config(['X', 'Y', 'Z'], 'Y', max_samples=10)
        state = create_jax_state(config)
        
        # Add some initial samples
        key = random.PRNGKey(246)
        for i in range(5):
            key, subkey = random.split(key)
            values = random.normal(subkey, (3,))
            interventions = jnp.zeros(3, dtype=bool).at[i % 2].set(True)
            target_value = float(values[1])
            
            new_buffer = add_sample_jax(state.sample_buffer, values, interventions, target_value)
            state = JAXAcquisitionState(
                sample_buffer=new_buffer,
                mechanism_features=state.mechanism_features,
                marginal_probs=state.marginal_probs,
                confidence_scores=state.confidence_scores,
                best_value=max(state.best_value, target_value),
                current_step=state.current_step + 1,
                uncertainty_bits=state.uncertainty_bits,
                config=state.config
            )
        
        # Time first run (includes compilation)
        start_time = time.perf_counter()
        confidence1 = compute_mechanism_confidence_jax(state)
        features1 = compute_policy_features_jax(state)
        first_run_time = time.perf_counter() - start_time
        
        # Time subsequent runs (compiled)
        run_times = []
        for _ in range(5):
            start_time = time.perf_counter()
            confidence2 = compute_mechanism_confidence_jax(state)
            features2 = compute_policy_features_jax(state)
            run_times.append(time.perf_counter() - start_time)
            
            # Results should be identical
            assert jnp.allclose(confidence1, confidence2)
            assert jnp.allclose(features1, features2)
        
        avg_compiled_time = sum(run_times) / len(run_times)
        speedup = first_run_time / avg_compiled_time
        
        print(f"First run (with compilation): {first_run_time*1000:.1f}ms")
        print(f"Average compiled run: {avg_compiled_time*1000:.3f}ms")
        print(f"Compilation amortization: {speedup:.1f}x speedup")
        
        # Compiled runs should be much faster
        assert speedup > 5.0, f"Insufficient speedup from compilation: {speedup:.1f}x"
        
        print("✓ JAX compilation cost properly amortized")


class TestPipelineStateConsistency:
    """Test state consistency throughout pipeline operations."""
    
    def test_immutable_state_properties(self):
        """Test that state remains properly immutable during operations."""
        config = create_jax_config(['X', 'Y', 'Z'], 'Y', max_samples=10)
        original_state = create_jax_state(config)
        
        # Store original state values
        original_buffer = original_state.sample_buffer
        original_features = original_state.mechanism_features
        original_best = original_state.best_value
        original_step = original_state.current_step
        
        # Perform operations that should not modify original state
        confidence = compute_mechanism_confidence_jax(original_state)
        features = compute_policy_features_jax(original_state)
        progress = compute_optimization_progress_jax(original_state)
        
        # Original state should be unchanged
        assert original_state.sample_buffer is original_buffer
        assert jnp.array_equal(original_state.mechanism_features, original_features)
        assert original_state.best_value == original_best
        assert original_state.current_step == original_step
        
        # Add sample to create new state
        values = jnp.array([1.0, 2.0, 1.5])
        interventions = jnp.array([True, False, False])
        new_buffer = add_sample_jax(original_state.sample_buffer, values, interventions, 2.0)
        
        # Original buffer should be unchanged
        assert original_state.sample_buffer.n_samples == 0
        assert new_buffer.n_samples == 1
        
        print("✓ State immutability properly maintained")
    
    def test_configuration_consistency(self):
        """Test that configuration remains consistent throughout operations."""
        config = create_jax_config(['A', 'B', 'C', 'D'], 'C', max_samples=15, max_history=25)
        state = create_jax_state(config)
        
        # Configuration should be preserved in all components
        assert state.config is config
        assert state.sample_buffer.config is config
        
        # Add samples and verify config consistency
        key = random.PRNGKey(135)
        for i in range(8):
            key, subkey = random.split(key)
            values = random.normal(subkey, (4,))
            interventions = jnp.zeros(4, dtype=bool).at[i % 3].set(True)  # Avoid target (index 2)
            target_value = float(values[2])
            
            new_buffer = add_sample_jax(state.sample_buffer, values, interventions, target_value)
            state = JAXAcquisitionState(
                sample_buffer=new_buffer,
                mechanism_features=state.mechanism_features,
                marginal_probs=state.marginal_probs,
                confidence_scores=state.confidence_scores,
                best_value=max(state.best_value, target_value),
                current_step=state.current_step + 1,
                uncertainty_bits=state.uncertainty_bits,
                config=state.config
            )
            
            # Configuration should remain consistent
            assert state.config is config
            assert state.sample_buffer.config is config
            assert state.config.target_idx == 2  # C is index 2
            assert state.config.n_vars == 4
            assert state.config.max_samples == 15
        
        print("✓ Configuration consistency maintained")


def test_comprehensive_integration_suite():
    """Run comprehensive integration test suite."""
    print("\n=== End-to-End Pipeline Integration Suite ===")
    
    # Test basic functionality
    config = create_jax_config(['X', 'Y', 'Z'], 'Y')
    state = create_jax_state(config)
    
    # Add sample
    values = jnp.array([1.0, 2.0, 1.5])
    interventions = jnp.array([True, False, False])
    new_buffer = add_sample_jax(state.sample_buffer, values, interventions, 2.0)
    state = JAXAcquisitionState(
        sample_buffer=new_buffer,
        mechanism_features=state.mechanism_features,
        marginal_probs=state.marginal_probs,
        confidence_scores=state.confidence_scores,
        best_value=max(state.best_value, 2.0),
        current_step=state.current_step + 1,
        uncertainty_bits=state.uncertainty_bits,
        config=state.config
    )
    
    # Test all pipeline operations
    confidence = compute_mechanism_confidence_jax(state)
    features = compute_policy_features_jax(state)
    progress = compute_optimization_progress_jax(state)
    coverage = compute_exploration_coverage_jax(state)
    
    print(f"✓ Confidence computed: shape {confidence.shape}")
    print(f"✓ Features computed: shape {features.shape}")
    print(f"✓ Progress computed: {len(progress)} metrics")
    print(f"✓ Coverage computed: {len(coverage)} metrics")
    print("✓ End-to-end pipeline integration complete")


if __name__ == "__main__":
    # Run comprehensive test when called directly
    test_comprehensive_integration_suite()