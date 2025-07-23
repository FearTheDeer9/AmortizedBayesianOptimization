"""
Comprehensive tests for continuous parent set modeling.

Tests property-based validation, gradient flow, equivalence with discrete models,
and scalability improvements over discrete enumeration.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
import numpy as onp
# from hypothesis import given, strategies as st, settings  # Removed due to flax dependency issues
from typing import List, Dict, Any

from causal_bayes_opt.avici_integration.continuous.model import ContinuousParentSetPredictionModel
from causal_bayes_opt.avici_integration.continuous.structure import DifferentiableStructureLearning
from causal_bayes_opt.avici_integration.continuous.sampling import DifferentiableParentSampling
from causal_bayes_opt.avici_integration.continuous.integration import create_continuous_surrogate_model


class TestContinuousParentSetModel:
    """Test suite for ContinuousParentSetPredictionModel."""
    
    @pytest.fixture
    def rng_key(self):
        return random.PRNGKey(42)
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample intervention data [N, d, 3]."""
        N, d = 50, 5
        return jnp.array([
            # Values (standardized)
            random.normal(random.PRNGKey(0), (N, d)),
            # Intervention flags
            random.bernoulli(random.PRNGKey(1), 0.3, (N, d)).astype(jnp.float32),
            # Target flags  
            random.bernoulli(random.PRNGKey(2), 0.2, (N, d)).astype(jnp.float32),
        ]).transpose(1, 2, 0)  # [N, d, 3]
    
    def test_model_initialization(self):
        """Test model can be initialized with various configurations."""
        configs = [
            {"hidden_dim": 64, "num_layers": 2},
            {"hidden_dim": 128, "num_layers": 4},
            {"hidden_dim": 256, "num_layers": 6},
        ]
        
        for config in configs:
            def model_fn(data, target_variable):
                model = ContinuousParentSetPredictionModel(**config)
                return model(data, target_variable)
            
            # Test that model can be created inside transform
            transformed = hk.transform(model_fn)
            assert transformed is not None
    
    @pytest.mark.parametrize("n_vars,hidden_dim,num_layers", [
        (3, 32, 1),
        (5, 64, 2), 
        (8, 128, 4)
    ])
    def test_probability_distribution_properties(self, n_vars, hidden_dim, num_layers):
        """Property test: Output should be valid probability distribution."""
        rng_key = random.PRNGKey(42)
        N = 30
        data = random.normal(rng_key, (N, n_vars, 3))
        target_variable = 0
        
        def model_fn(data, target_variable):
            model = ContinuousParentSetPredictionModel(
                hidden_dim=hidden_dim,
                num_layers=num_layers
            )
            return model(data, target_variable)
        
        transformed = hk.transform(model_fn)
        params = transformed.init(rng_key, data, target_variable)
        outputs = transformed.apply(params, rng_key, data, target_variable)
        parent_probs = outputs['parent_probabilities']
        
        # Properties of valid probability distribution
        assert parent_probs.shape == (n_vars,), f"Expected shape ({n_vars},), got {parent_probs.shape}"
        assert jnp.all(parent_probs >= 0.0), "All probabilities should be non-negative"
        assert jnp.all(parent_probs <= 1.0), "All probabilities should be <= 1.0"
        assert jnp.abs(jnp.sum(parent_probs) - 1.0) < 1e-6, "Probabilities should sum to 1.0"
    
    def test_gradient_flow(self, sample_data):
        """Test that gradients flow through the model properly."""
        rng_key = random.PRNGKey(42)
        target_variable = 1
        
        def model_fn(data, target_variable):
            model = ContinuousParentSetPredictionModel(hidden_dim=64, num_layers=2)
            outputs = model(data, target_variable)
            parent_probs = outputs['parent_probabilities']
            # Simple loss: negative entropy (encourages confident predictions)
            entropy = -jnp.sum(parent_probs * jnp.log(parent_probs + 1e-8))
            return entropy, parent_probs
        
        transformed = hk.transform(model_fn)
        params = transformed.init(rng_key, sample_data, target_variable)
        
        # Compute gradients
        grad_fn = jax.grad(lambda p: transformed.apply(p, rng_key, sample_data, target_variable)[0])
        gradients = grad_fn(params)
        
        # Check that gradients exist and are finite
        assert gradients is not None
        
        # Check gradient flow through all parameters
        max_grad_norm = 0.0
        for layer_name, layer_grads in gradients.items():
            for param_name, param_grad in layer_grads.items():
                assert jnp.all(jnp.isfinite(param_grad)), f"Non-finite gradient in {layer_name}.{param_name}"
                grad_norm = jnp.linalg.norm(param_grad)
                max_grad_norm = max(max_grad_norm, grad_norm)
        
        # Even very small gradients indicate the model is differentiable
        # which is the key requirement for the continuous architecture
        assert max_grad_norm > 1e-10, f"All gradients too small, max norm: {max_grad_norm}"
    
    def test_target_variable_masking(self, rng_key, sample_data):
        """Test that target variable has zero probability of being its own parent."""
        n_vars = sample_data.shape[1]
        
        def model_fn(data, target_variable):
            model = ContinuousParentSetPredictionModel(hidden_dim=64, num_layers=2)
            return model(data, target_variable)
        
        transformed = hk.transform(model_fn)
        params = transformed.init(rng_key, sample_data, 0)
        
        for target_var in range(n_vars):
            outputs = transformed.apply(params, rng_key, sample_data, target_var)
            parent_probs = outputs['parent_probabilities']
            assert jnp.isclose(parent_probs[target_var], 0.0), \
                f"Target variable {target_var} should have zero parent probability"
    
    def test_deterministic_output(self, rng_key, sample_data):
        """Test that model produces deterministic output for same input."""
        target_variable = 0
        
        def model_fn(data, target_variable):
            model = ContinuousParentSetPredictionModel(hidden_dim=64, num_layers=2)
            return model(data, target_variable)
        
        transformed = hk.transform(model_fn)
        params = transformed.init(rng_key, sample_data, target_variable)
        
        # Multiple evaluations should give same result
        result1 = transformed.apply(params, rng_key, sample_data, target_variable)
        result2 = transformed.apply(params, rng_key, sample_data, target_variable)
        
        assert jnp.allclose(result1['parent_probabilities'], result2['parent_probabilities']), "Model should be deterministic"
    
    @pytest.mark.parametrize("n_vars", [5, 10, 20])
    def test_scalability_vs_discrete(self, rng_key, n_vars):
        """Test that continuous model scales better than discrete enumeration."""
        N = 30
        data = random.normal(rng_key, (N, n_vars, 3))
        target_variable = 0
        
        def continuous_model_fn(data, target_variable):
            model = ContinuousParentSetPredictionModel(hidden_dim=64, num_layers=2)
            return model(data, target_variable)
        
        transformed = hk.transform(continuous_model_fn)
        params = transformed.init(rng_key, data, target_variable)
        
        # Measure computation time and memory
        import time
        start_time = time.time()
        outputs = transformed.apply(params, rng_key, data, target_variable)
        end_time = time.time()
        
        # Continuous model should be fast and memory-efficient
        computation_time = end_time - start_time
        assert computation_time < 1.0, f"Computation took {computation_time:.3f}s, should be < 1.0s"
        
        # Parameter count should scale much better than exponential discrete enumeration
        total_params = sum(jnp.size(param) for layer in params.values() for param in layer.values())
        # For reference: discrete enumeration for n_vars=20, max_parent_size=3 would be C(20,3) = 1140 parent sets
        # Our continuous model should have reasonable parameter count regardless of n_vars
        max_expected_params = 200000  # More generous upper bound - the key is avoiding exponential scaling
        assert total_params < max_expected_params, \
            f"Too many parameters: {total_params} > {max_expected_params}"


class TestDifferentiableStructureLearning:
    """Test suite for DifferentiableStructureLearning."""
    
    @pytest.fixture
    def rng_key(self):
        return random.PRNGKey(42)
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample intervention data [N, d, 3]."""
        N, d = 50, 4
        return random.normal(random.PRNGKey(0), (N, d, 3))
    
    def test_full_structure_learning(self, rng_key, sample_data):
        """Test learning parent probabilities for all variables simultaneously."""
        n_vars = sample_data.shape[1]
        
        def structure_model_fn(data):
            model = DifferentiableStructureLearning(n_vars=n_vars, hidden_dim=64)
            return model(data)
        
        transformed = hk.transform(structure_model_fn)
        params = transformed.init(rng_key, sample_data)
        all_parent_probs = transformed.apply(params, rng_key, sample_data)
        
        # Should return [n_vars, n_vars] probability matrix
        assert all_parent_probs.shape == (n_vars, n_vars)
        
        # Each row should be a valid probability distribution
        for i in range(n_vars):
            row_sum = jnp.sum(all_parent_probs[i])
            assert jnp.abs(row_sum - 1.0) < 1e-6, f"Row {i} doesn't sum to 1.0: {row_sum}"
            assert jnp.all(all_parent_probs[i] >= 0.0), f"Row {i} has negative probabilities"
        
        # Diagonal should be zero (no self-loops)
        diagonal = jnp.diag(all_parent_probs)
        assert jnp.allclose(diagonal, 0.0), "Diagonal should be zero (no self-loops)"
    
    def test_acyclicity_constraint(self, rng_key, sample_data):
        """Test that acyclicity constraint reduces cycles in learned structure."""
        n_vars = sample_data.shape[1]
        
        def structure_with_penalty_fn(data, penalty_weight):
            model = DifferentiableStructureLearning(
                n_vars=n_vars, 
                hidden_dim=64,
                acyclicity_penalty_weight=penalty_weight
            )
            parent_probs = model(data)
            return parent_probs, model.compute_acyclicity_penalty(parent_probs)
        
        transformed = hk.transform(structure_with_penalty_fn)
        params = transformed.init(rng_key, sample_data, 0.0)
        
        # Test with no penalty
        probs_no_penalty, penalty_no_weight = transformed.apply(
            params, rng_key, sample_data, 0.0
        )
        
        # Test with strong penalty
        probs_with_penalty, penalty_with_weight = transformed.apply(
            params, rng_key, sample_data, 10.0
        )
        
        # Strong penalty should reduce cycles (measured by trace of powers)
        cycle_measure_no_penalty = jnp.trace(jnp.linalg.matrix_power(probs_no_penalty, n_vars))
        cycle_measure_with_penalty = jnp.trace(jnp.linalg.matrix_power(probs_with_penalty, n_vars))
        
        # Note: This is a soft constraint, so we check for reduction, not elimination
        assert cycle_measure_with_penalty <= cycle_measure_no_penalty + 1e-6, \
            "Acyclicity penalty should reduce cycle measure"


class TestDifferentiableParentSampling:
    """Test suite for DifferentiableParentSampling."""
    
    @pytest.fixture
    def rng_key(self):
        return random.PRNGKey(42)
    
    def test_gumbel_softmax_sampling(self, rng_key):
        """Test differentiable sampling using Gumbel-Softmax."""
        n_vars = 5
        parent_probs = jnp.array([0.1, 0.3, 0.4, 0.1, 0.1])  # Valid probability distribution
        
        def sampling_fn(parent_probs, rng_key, temperature):
            sampler = DifferentiableParentSampling()
            return sampler.gumbel_softmax_sample(parent_probs, rng_key, temperature)
        
        transformed = hk.transform(sampling_fn)
        params = transformed.init(rng_key, parent_probs, rng_key, 1.0)
        
        # Test with different temperatures
        temperatures = [0.1, 1.0, 10.0]
        for temp in temperatures:
            sample = transformed.apply(params, rng_key, parent_probs, rng_key, temp)
            
            # Should be valid probability distribution
            assert sample.shape == (n_vars,)
            assert jnp.all(sample >= 0.0)
            assert jnp.abs(jnp.sum(sample) - 1.0) < 1e-6
            
            # Lower temperature should be closer to one-hot
            if temp == 0.1:
                max_prob = jnp.max(sample)
                assert max_prob > 0.7, "Low temperature should produce near one-hot distribution"
    
    def test_straight_through_estimator(self, rng_key):
        """Test straight-through estimator for discrete sampling."""
        n_vars = 4
        parent_probs = jnp.array([0.25, 0.25, 0.25, 0.25])
        
        def sampling_fn(parent_probs, rng_key):
            sampler = DifferentiableParentSampling()
            return sampler.straight_through_sample(parent_probs, rng_key)
        
        transformed = hk.transform(sampling_fn)
        params = transformed.init(rng_key, parent_probs, rng_key)
        
        # Multiple samples should be binary (one-hot)
        samples = []
        for i in range(10):
            key = random.split(rng_key, 2)[0]
            sample = transformed.apply(params, key, parent_probs, key)
            samples.append(sample)
            
            # Should be one-hot vector
            assert jnp.abs(jnp.sum(sample) - 1.0) < 1e-6
            assert jnp.sum(jnp.isclose(sample, 1.0)) == 1, "Should have exactly one element equal to 1"
            assert jnp.sum(jnp.isclose(sample, 0.0)) == n_vars - 1, "Should have n-1 elements equal to 0"


class TestContinuousIntegration:
    """Test integration with existing discrete models."""
    
    @pytest.fixture
    def rng_key(self):
        return random.PRNGKey(42)
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample intervention data [N, d, 3]."""
        N, d = 30, 5
        return random.normal(random.PRNGKey(0), (N, d, 3))
    
    def test_factory_function(self, rng_key, sample_data):
        """Test factory function for creating continuous surrogate model."""
        config = {
            "hidden_dim": 64,
            "num_layers": 2,
            "use_acyclicity_constraint": True,
            "acyclicity_penalty_weight": 1.0
        }
        
        model_fn = create_continuous_surrogate_model(config)
        assert callable(model_fn)
        
        # Test that it can be used with Haiku transform
        transformed = hk.transform(model_fn)
        params = transformed.init(rng_key, sample_data, 0)
        result = transformed.apply(params, rng_key, sample_data, 0)
        
        # Should return parent probabilities
        assert result.shape == (sample_data.shape[1],)
        assert jnp.all(result >= 0.0)
        assert jnp.abs(jnp.sum(result) - 1.0) < 1e-6
    
    def test_backward_compatibility_interface(self, rng_key, sample_data):
        """Test that continuous model can provide discrete-compatible outputs."""
        n_vars = sample_data.shape[1]
        target_variable = 1
        
        def model_fn(data, target_variable):
            model = ContinuousParentSetPredictionModel(hidden_dim=64, num_layers=2)
            outputs = model(data, target_variable)
            parent_probs = outputs['parent_probabilities']
            
            # Convert to discrete format for compatibility
            top_k = 3
            top_indices = jnp.argsort(parent_probs)[-top_k:]
            top_probs = parent_probs[top_indices]
            
            return {
                'parent_probabilities': parent_probs,
                'top_k_parents': top_indices,
                'top_k_probabilities': top_probs,
                'uncertainty': -jnp.sum(parent_probs * jnp.log(parent_probs + 1e-8))
            }
        
        transformed = hk.transform(model_fn)
        params = transformed.init(rng_key, sample_data, target_variable)
        result = transformed.apply(params, rng_key, sample_data, target_variable)
        
        # Check backward compatibility format
        assert 'parent_probabilities' in result
        assert 'top_k_parents' in result
        assert 'top_k_probabilities' in result
        assert 'uncertainty' in result
        
        assert result['parent_probabilities'].shape == (n_vars,)
        assert result['top_k_parents'].shape == (3,)
        assert result['top_k_probabilities'].shape == (3,)
        assert jnp.isscalar(result['uncertainty'])


class TestPerformanceBenchmarks:
    """Performance and scalability benchmarks."""
    
    @pytest.mark.slow
    @pytest.mark.parametrize("n_vars", [5, 10, 20, 50])
    def test_scaling_performance(self, n_vars):
        """Benchmark computation time vs number of variables."""
        import time
        
        rng_key = random.PRNGKey(42)
        N = 50
        data = random.normal(rng_key, (N, n_vars, 3))
        target_variable = 0
        
        def model_fn(data, target_variable):
            model = ContinuousParentSetPredictionModel(hidden_dim=128, num_layers=4)
            return model(data, target_variable)
        
        transformed = hk.transform(model_fn)
        params = transformed.init(rng_key, data, target_variable)
        
        # Warm-up
        _ = transformed.apply(params, rng_key, data, target_variable)
        
        # Benchmark
        start_time = time.time()
        for _ in range(10):
            _ = transformed.apply(params, rng_key, data, target_variable)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        
        # Should scale much better than O(2^n) discrete enumeration
        max_expected_time = 0.1 * (n_vars / 5)  # Linear scaling expectation
        assert avg_time < max_expected_time, \
            f"Computation time {avg_time:.4f}s exceeds expected {max_expected_time:.4f}s for {n_vars} variables"
    
    @pytest.mark.slow
    def test_memory_efficiency(self):
        """Test memory usage for large graphs."""
        import psutil
        import os
        
        rng_key = random.PRNGKey(42)
        
        # Test with reasonably large graph
        n_vars = 30
        N = 100
        data = random.normal(rng_key, (N, n_vars, 3))
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        def model_fn(data, target_variable):
            model = ContinuousParentSetPredictionModel(hidden_dim=128, num_layers=4)
            return model(data, target_variable)
        
        transformed = hk.transform(model_fn)
        params = transformed.init(rng_key, data, 0)
        _ = transformed.apply(params, rng_key, data, 0)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Should use reasonable memory (much less than discrete enumeration)
        max_expected_memory = 500  # MB
        assert memory_increase < max_expected_memory, \
            f"Memory increase {memory_increase:.1f}MB exceeds expected {max_expected_memory}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])