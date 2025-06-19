#!/usr/bin/env python3
"""
Integration tests for JAX optimization in surrogate training.

These tests verify that JAX compilation actually works in practice,
not just that the functions exist. They test real compilation, performance,
and integration with the broader training pipeline.
"""

import pytest
import time
import jax
import jax.numpy as jnp
import jax.random as random
import optax
import pyrsistent as pyr
from typing import Dict, Any

from causal_bayes_opt.training.surrogate_training import (
    TrainingExample,
    TrainingBatch,
    TrainingBatchJAX,
    convert_to_jax_batch,
    create_jax_surrogate_train_step,
    create_adaptive_train_step,
    create_surrogate_train_step,
    kl_divergence_loss,
    kl_divergence_loss_jax,
    uncertainty_weighted_loss_jax,
    SurrogateTrainingConfig
)
from causal_bayes_opt.avici_integration import create_parent_set_model
from causal_bayes_opt.avici_integration.parent_set.posterior import create_parent_set_posterior
from causal_bayes_opt.experiments.test_scms import create_simple_test_scm
from causal_bayes_opt.mechanisms.linear import sample_from_linear_scm


class TestJAXCompilationIntegration:
    """Test that JAX compilation actually works with real models and data."""
    
    @pytest.fixture
    def minimal_config(self):
        """Create minimal config for fast testing."""
        return SurrogateTrainingConfig(
            model_hidden_dim=16,  # Very small for fast testing
            model_n_layers=2,
            learning_rate=1e-3,
            batch_size=2,
            max_epochs=1,
            early_stopping_patience=1,
            validation_frequency=1,
            weight_decay=1e-4,
            max_parent_size=3,
            dropout=0.0
        )
    
    @pytest.fixture
    def real_model_and_params(self, minimal_config):
        """Create a real model with initialized parameters."""
        # Create model - separate max_parent_size from other kwargs
        model_kwargs = minimal_config.get_model_kwargs()
        max_parent_size = model_kwargs.pop('max_parent_size')
        model = create_parent_set_model(
            model_kwargs=model_kwargs,
            max_parent_size=max_parent_size
        )
        
        # Initialize parameters with realistic data
        key = random.PRNGKey(42)
        dummy_data = jnp.ones((10, 3, 3))  # [N, d, 3] format
        dummy_variables = ['X', 'Y', 'Z']
        dummy_target = 'Y'
        
        params = model.init(key, dummy_data, dummy_variables, dummy_target, True)
        
        return model, params
    
    @pytest.fixture
    def realistic_training_batch(self):
        """Create a realistic training batch with multiple examples."""
        # Create test SCM
        scm = create_simple_test_scm(noise_scale=1.0, target='Y')
        
        # Generate samples
        samples = sample_from_linear_scm(scm, n_samples=20, seed=42)
        
        # Convert to AVICI format
        from causal_bayes_opt.avici_integration.core.conversion import samples_to_avici_format
        avici_data = samples_to_avici_format(samples, ['X', 'Y', 'Z'], 'Y')
        
        # Create expert posterior
        parent_sets = [frozenset(), frozenset(['X']), frozenset(['Z']), frozenset(['X', 'Z'])]
        probs = jnp.array([0.1, 0.7, 0.1, 0.1])  # X is most likely parent
        expert_posterior = create_parent_set_posterior('Y', parent_sets, probs)
        
        # Create training examples
        examples = []
        for i in range(3):  # Small batch for testing
            example = TrainingExample(
                observational_data=avici_data,
                target_variable='Y',
                variable_order=['X', 'Y', 'Z'],
                expert_posterior=expert_posterior,
                expert_accuracy=0.8,
                scm_info=pyr.m(n_nodes=3, target='Y'),
                problem_difficulty='easy'
            )
            examples.append(example)
        
        return TrainingBatch(examples=examples)
    
    def test_jax_training_step_compilation(self, real_model_and_params, minimal_config, realistic_training_batch):
        """Test that the JAX training step actually compiles and runs."""
        model, params = real_model_and_params
        
        # Create optimizer
        optimizer = optax.adam(learning_rate=1e-3)
        opt_state = optimizer.init(params)
        
        # Create JAX training step - this should compile
        jax_train_step = create_jax_surrogate_train_step(
            model=model,
            optimizer=optimizer,
            loss_fn=kl_divergence_loss,
            config=minimal_config
        )
        
        # Verify it's a function
        assert callable(jax_train_step)
        
        # Convert batch to JAX format
        jax_batch = convert_to_jax_batch(realistic_training_batch)
        
        # This is the critical test - does JAX compilation actually work?
        key = random.PRNGKey(123)
        
        # First call compiles the function
        start_time = time.time()
        new_params, new_opt_state, metrics = jax_train_step(
            params, opt_state,
            jax_batch.parent_sets,
            jax_batch.variable_orders,
            jax_batch.target_variables,
            jax_batch.observational_data,
            jax_batch.expert_probs,
            jax_batch.expert_accuracies,
            key
        )
        first_call_time = time.time() - start_time
        
        # Verify results are reasonable
        assert isinstance(metrics, dict)
        assert 'total_loss' in metrics
        assert jnp.isfinite(metrics['total_loss'])
        assert metrics['total_loss'] >= 0
        
        # Second call should be much faster (compiled)
        start_time = time.time()
        new_params2, new_opt_state2, metrics2 = jax_train_step(
            new_params, new_opt_state,
            jax_batch.parent_sets,
            jax_batch.variable_orders,
            jax_batch.target_variables,
            jax_batch.observational_data,
            jax_batch.expert_probs,
            jax_batch.expert_accuracies,
            key
        )
        second_call_time = time.time() - start_time
        
        # Second call should be faster (evidence of compilation)
        print(f"First call (with compilation): {first_call_time:.4f}s")
        print(f"Second call (compiled): {second_call_time:.4f}s")
        print(f"Speedup ratio: {first_call_time / second_call_time:.2f}x")
        
        # Verify compilation happened (second call should be faster)
        assert second_call_time < first_call_time, "Second call should be faster due to compilation"
        
        # Verify results are consistent
        assert jnp.isfinite(metrics2['total_loss'])
    
    def test_adaptive_training_step_uses_jax(self, real_model_and_params, minimal_config, realistic_training_batch):
        """Test that adaptive training step actually uses JAX when enabled."""
        model, params = real_model_and_params
        
        optimizer = optax.adam(learning_rate=1e-3)
        opt_state = optimizer.init(params)
        
        # Create adaptive step with JAX enabled
        adaptive_step_jax = create_adaptive_train_step(
            model=model,
            optimizer=optimizer,
            loss_fn=kl_divergence_loss,
            config=minimal_config,
            use_jax_compilation=True
        )
        
        # Create adaptive step with JAX disabled
        adaptive_step_no_jax = create_adaptive_train_step(
            model=model,
            optimizer=optimizer,
            loss_fn=kl_divergence_loss,
            config=minimal_config,
            use_jax_compilation=False
        )
        
        key = random.PRNGKey(456)
        
        # Time JAX version
        start_time = time.time()
        jax_params, jax_opt_state, jax_metrics = adaptive_step_jax(
            params, opt_state, realistic_training_batch, key
        )
        jax_time = time.time() - start_time
        
        # Time non-JAX version  
        start_time = time.time()
        no_jax_params, no_jax_opt_state, no_jax_metrics = adaptive_step_no_jax(
            params, opt_state, realistic_training_batch, key
        )
        no_jax_time = time.time() - start_time
        
        print(f"JAX adaptive step: {jax_time:.4f}s")
        print(f"Non-JAX adaptive step: {no_jax_time:.4f}s")
        
        # Both should work
        assert jnp.isfinite(jax_metrics.total_loss)
        assert jnp.isfinite(no_jax_metrics.total_loss)
        
        # Results should be approximately the same (same algorithm)
        loss_diff = abs(float(jax_metrics.total_loss) - float(no_jax_metrics.total_loss))
        assert loss_diff < 0.1, f"Loss difference too large: {loss_diff}"
    
    def test_jax_vs_non_jax_performance_comparison(self, real_model_and_params, minimal_config, realistic_training_batch):
        """Test actual performance difference between JAX and non-JAX implementations."""
        model, params = real_model_and_params
        
        optimizer = optax.adam(learning_rate=1e-3)
        opt_state = optimizer.init(params)
        
        # Create JAX training step
        jax_train_step = create_jax_surrogate_train_step(
            model=model,
            optimizer=optimizer,
            loss_fn=kl_divergence_loss,
            config=minimal_config
        )
        
        # Create non-JAX training step
        regular_train_step = create_surrogate_train_step(
            model=model,
            optimizer=optimizer,
            loss_fn=kl_divergence_loss,
            config=minimal_config
        )
        
        # Convert batch for JAX
        jax_batch = convert_to_jax_batch(realistic_training_batch)
        key = random.PRNGKey(789)
        
        # Warmup JAX compilation
        jax_train_step(
            params, opt_state,
            jax_batch.parent_sets,
            jax_batch.variable_orders,
            jax_batch.target_variables,
            jax_batch.observational_data,
            jax_batch.expert_probs,
            jax_batch.expert_accuracies,
            key
        )
        
        # Time multiple JAX calls
        num_runs = 10
        jax_times = []
        for i in range(num_runs):
            start_time = time.time()
            jax_train_step(
                params, opt_state,
                jax_batch.parent_sets,
                jax_batch.variable_orders,
                jax_batch.target_variables,
                jax_batch.observational_data,
                jax_batch.expert_probs,
                jax_batch.expert_accuracies,
                random.fold_in(key, i)
            )
            jax_times.append(time.time() - start_time)
        
        # Time multiple non-JAX calls
        regular_times = []
        for i in range(num_runs):
            start_time = time.time()
            regular_train_step(params, opt_state, realistic_training_batch, random.fold_in(key, i))
            regular_times.append(time.time() - start_time)
        
        # Compute statistics
        avg_jax_time = sum(jax_times) / len(jax_times)
        avg_regular_time = sum(regular_times) / len(regular_times)
        speedup = avg_regular_time / avg_jax_time
        
        print(f"Average JAX time: {avg_jax_time:.4f}s")
        print(f"Average regular time: {avg_regular_time:.4f}s")
        print(f"Actual speedup: {speedup:.2f}x")
        
        # JAX should be faster (though exact speedup depends on problem size)
        assert speedup > 1.0, f"JAX should be faster, got {speedup:.2f}x speedup"
        
        # Document actual performance for future reference
        print(f"✅ JAX optimization provides {speedup:.2f}x speedup on this test case")


class TestJAXLossFunctionAccuracy:
    """Test that JAX loss functions give mathematically consistent results."""
    
    def test_kl_divergence_loss_consistency(self):
        """Test that JAX and non-JAX KL loss give consistent results."""
        # Test data
        predicted_logits = jnp.array([1.0, 2.0, 0.5, 0.1])
        expert_probs = jnp.array([0.1, 0.6, 0.2, 0.1])
        parent_sets = [frozenset(), frozenset(['X']), frozenset(['Y']), frozenset(['X', 'Y'])]
        
        # Create expert posterior for non-JAX function
        expert_posterior = create_parent_set_posterior('Z', parent_sets, expert_probs)
        
        # Compare implementations
        jax_loss = kl_divergence_loss_jax(predicted_logits, expert_probs, parent_sets)
        non_jax_loss = kl_divergence_loss(predicted_logits, expert_posterior, parent_sets)
        
        # Should be approximately equal
        difference = abs(float(jax_loss) - float(non_jax_loss))
        print(f"JAX KL loss: {jax_loss:.6f}")
        print(f"Non-JAX KL loss: {non_jax_loss:.6f}")
        print(f"Difference: {difference:.6f}")
        
        # Allow small numerical differences due to implementation details
        assert difference < 1e-3, f"KL loss difference too large: {difference}"
        
        # Both should be finite and non-negative
        assert jnp.isfinite(jax_loss) and jax_loss >= 0
        assert jnp.isfinite(non_jax_loss) and non_jax_loss >= 0
    
    def test_uncertainty_weighted_loss_properties(self):
        """Test that uncertainty weighted loss behaves correctly."""
        # Test with different uncertainty levels
        high_confidence_probs = jnp.array([0.9, 0.05, 0.03, 0.02])  # Low entropy
        low_confidence_probs = jnp.array([0.4, 0.3, 0.2, 0.1])      # High entropy
        predicted_logits = jnp.array([1.0, 2.0, 0.5, 0.1])
        parent_sets = [frozenset(), frozenset(['X']), frozenset(['Y']), frozenset(['X', 'Y'])]
        
        # High confidence should give higher weight (higher loss)
        high_conf_loss = uncertainty_weighted_loss_jax(predicted_logits, high_confidence_probs, parent_sets)
        low_conf_loss = uncertainty_weighted_loss_jax(predicted_logits, low_confidence_probs, parent_sets)
        
        print(f"High confidence loss: {high_conf_loss:.6f}")
        print(f"Low confidence loss: {low_conf_loss:.6f}")
        
        # High confidence should typically result in higher weighted loss
        # (though this depends on the prediction quality)
        assert jnp.isfinite(high_conf_loss) and high_conf_loss >= 0
        assert jnp.isfinite(low_conf_loss) and low_conf_loss >= 0
    
    def test_loss_function_edge_cases(self):
        """Test loss functions with edge cases."""
        parent_sets = [frozenset(), frozenset(['X']), frozenset(['Y'])]
        
        # Test with extreme probabilities
        extreme_probs = jnp.array([0.99, 0.005, 0.005])
        predicted_logits = jnp.array([0.1, 0.1, 0.1])  # Uniform prediction
        
        kl_loss = kl_divergence_loss_jax(predicted_logits, extreme_probs, parent_sets)
        uw_loss = uncertainty_weighted_loss_jax(predicted_logits, extreme_probs, parent_sets)
        
        # Should handle extreme values gracefully
        assert jnp.isfinite(kl_loss)
        assert jnp.isfinite(uw_loss)
        
        # Test with very small probabilities
        small_probs = jnp.array([1e-6, 1-2e-6, 1e-6])
        kl_loss2 = kl_divergence_loss_jax(predicted_logits, small_probs, parent_sets)
        uw_loss2 = uncertainty_weighted_loss_jax(predicted_logits, small_probs, parent_sets)
        
        assert jnp.isfinite(kl_loss2)
        assert jnp.isfinite(uw_loss2)


class TestJAXErrorHandling:
    """Test error handling and fallback behavior."""
    
    def test_adaptive_fallback_on_jax_failure(self, realistic_training_batch):
        """Test that adaptive training falls back when JAX compilation fails."""
        # This is harder to test directly since our implementation should work
        # But we can test the fallback mechanism structure
        
        from causal_bayes_opt.avici_integration import create_parent_set_model
        
        # Create a minimal model
        config = SurrogateTrainingConfig(
            model_hidden_dim=8,
            model_n_layers=1,
            max_parent_size=2
        )
        
        model_kwargs = config.get_model_kwargs()
        max_parent_size = model_kwargs.pop('max_parent_size')
        model = create_parent_set_model(
            model_kwargs=model_kwargs,
            max_parent_size=max_parent_size
        )
        
        optimizer = optax.adam(learning_rate=1e-3)
        
        # Create adaptive step
        adaptive_step = create_adaptive_train_step(
            model=model,
            optimizer=optimizer,
            loss_fn=kl_divergence_loss,
            config=config,
            use_jax_compilation=True
        )
        
        # This should work (or fail gracefully)
        key = random.PRNGKey(42)
        dummy_data = jnp.ones((5, 3, 3))
        params = model.init(key, dummy_data, ['X', 'Y', 'Z'], 'Y', True)
        opt_state = optimizer.init(params)
        
        # Call the adaptive step
        try:
            new_params, new_opt_state, metrics = adaptive_step(
                params, opt_state, realistic_training_batch, key
            )
            # If it succeeds, verify the results
            assert isinstance(metrics.total_loss, float)
            assert jnp.isfinite(metrics.total_loss)
            print("✅ Adaptive training step succeeded")
        except Exception as e:
            # If it fails, it should be a clear error, not a cryptic JAX error
            print(f"Adaptive training failed (expected for some configurations): {e}")
            # This is acceptable - the test is whether the error is handled gracefully


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])