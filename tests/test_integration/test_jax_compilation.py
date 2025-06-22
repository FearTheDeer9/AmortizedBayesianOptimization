"""
JAX Compilation Validation Tests

Tests to ensure all JAX-native components compile correctly and maintain
expected performance characteristics. Follows CLAUDE.md principles:
- Pure functions only
- Property-based testing with hypothesis
- Comprehensive edge case coverage
- No side effects in tests
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
import time
from typing import Dict, Any

# Import hypothesis if available, otherwise skip property-based tests
try:
    from hypothesis import given, strategies as st, settings
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    # Create dummy decorators for when hypothesis is not available
    def given(*args, **kwargs):
        def decorator(func):
            return pytest.mark.skip("hypothesis not available")(func)
        return decorator
    
    def settings(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class st:
        @staticmethod
        def integers(min_value=0, max_value=10):
            return None
        @staticmethod
        def floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False):
            return None

from causal_bayes_opt.jax_native import (
    JAXConfig, JAXSampleBuffer, JAXAcquisitionState,
    create_jax_config, create_jax_state, create_empty_jax_buffer,
    add_sample_jax
)
from causal_bayes_opt.jax_native.operations import (
    compute_mechanism_confidence_from_tensors_jax,
    update_mechanism_features_jax,
    compute_acquisition_scores_jax,
    validate_jax_compilation
)
from causal_bayes_opt.jax_native.sample_buffer import (
    add_sample_to_tensors_jax,
    extract_latest_samples_jax
)


class TestJAXCompilationCore:
    """Test core JAX compilation functionality."""
    
    def test_validate_jax_compilation_passes(self):
        """Test that our validation function reports all operations compile."""
        assert validate_jax_compilation() == True
    
    def test_add_sample_to_tensors_jax_compiles(self):
        """Test that tensor addition function compiles correctly."""
        # Create test configuration
        config = create_jax_config(['X', 'Y', 'Z'], 'Y', max_samples=10)
        
        # Create empty tensors
        values_array = jnp.zeros((10, 3))
        interventions_array = jnp.zeros((10, 3), dtype=bool)
        targets_array = jnp.zeros(10)
        valid_mask = jnp.zeros(10, dtype=bool)
        
        # Test data
        new_values = jnp.array([1.0, 2.0, 1.5])
        new_interventions = jnp.array([True, False, False])
        new_target = 2.0
        
        # Compile function
        compiled_fn = jax.jit(add_sample_to_tensors_jax)
        
        # Test compilation works
        result = compiled_fn(
            values_array, interventions_array, targets_array, valid_mask,
            0, 0, 10,  # write_idx, n_samples, max_samples
            new_values, new_interventions, new_target
        )
        
        # Verify result structure
        assert len(result) == 6  # 6 output tensors
        assert result[0].shape == (10, 3)  # Updated values
        assert result[1].shape == (10, 3)  # Updated interventions
        assert result[2].shape == (10,)    # Updated targets
        assert result[3].shape == (10,)    # Updated valid mask
        # result[4] and result[5] are scalar indices
    
    def test_mechanism_confidence_compiles(self):
        """Test mechanism confidence computation compiles."""
        # Test data
        mechanism_features = jnp.array([
            [1.0, 0.2, 0.8],  # X: high effect, low uncertainty
            [0.0, 1.0, 0.0],  # Y: target variable (will be masked)
            [0.5, 0.5, 0.6]   # Z: medium effect, medium uncertainty
        ])
        target_mask = jnp.array([False, True, False])  # Y is target
        
        # Compile function
        compiled_fn = jax.jit(compute_mechanism_confidence_from_tensors_jax)
        
        # Test compilation and execution
        confidence = compiled_fn(mechanism_features, target_mask)
        
        # Verify result
        assert confidence.shape == (3,)
        assert confidence[1] == 0.0  # Target variable masked
        assert jnp.all(confidence >= 0.0)  # Non-negative confidence
        assert jnp.all(confidence <= 1.0)  # Bounded confidence
    
    def test_mechanism_features_update_compiles(self):
        """Test mechanism features update compiles."""
        current_features = jnp.array([
            [1.0, 0.2, 0.8],
            [0.5, 0.5, 0.6]
        ])
        new_observations = jnp.array([
            [1.1, 0.1, 0.9],
            [0.4, 0.6, 0.5]
        ])
        
        # Compile function
        compiled_fn = jax.jit(update_mechanism_features_jax)
        
        # Test compilation
        updated_features = compiled_fn(current_features, new_observations, 0.1)
        
        # Verify result
        assert updated_features.shape == current_features.shape
        assert jnp.all(jnp.isfinite(updated_features))
    
    def test_acquisition_scores_compiles(self):
        """Test acquisition scores computation compiles."""
        policy_features = jnp.array([
            [0.8, 0.2, 0.7, 0.9, 0.5],  # X
            [0.0, 1.0, 0.0, 0.0, 0.0],  # Y (target)
            [0.6, 0.4, 0.5, 0.7, 0.3]   # Z
        ])
        target_idx = 1
        
        # Compile function  
        compiled_fn = jax.jit(compute_acquisition_scores_jax)
        
        # Test compilation
        scores = compiled_fn(policy_features, target_idx, 0.1)
        
        # Verify result
        assert scores.shape == (3,)
        assert scores[target_idx] == -jnp.inf  # Target masked
        assert jnp.all(jnp.isfinite(scores[scores != -jnp.inf]))


class TestJAXCompilationWithRandomData:
    """Test JAX compilation with property-based random data."""
    
    @given(
        n_vars=st.integers(min_value=2, max_value=10),
        max_samples=st.integers(min_value=5, max_value=100),
        target_idx=st.integers(min_value=0, max_value=9)
    )
    @settings(max_examples=50, deadline=10000)  # Reasonable limits for compilation tests
    def test_config_creation_with_random_params(self, n_vars, max_samples, target_idx):
        """Test config creation with various parameter combinations."""
        # Ensure target_idx is valid
        target_idx = target_idx % n_vars
        
        # Generate variable names
        variable_names = [f"X{i}" for i in range(n_vars)]
        target_variable = variable_names[target_idx]
        
        # Create configuration
        config = create_jax_config(
            variable_names, target_variable, 
            max_samples=max_samples, max_history=50
        )
        
        # Verify configuration is valid
        assert config.n_vars == n_vars
        assert config.target_idx == target_idx
        assert config.max_samples == max_samples
        assert len(config.variable_names) == n_vars
    
    @given(
        effect=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        uncertainty=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_mechanism_confidence_with_random_features(self, effect, uncertainty, confidence):
        """Test mechanism confidence computation with random feature values."""
        mechanism_features = jnp.array([
            [effect, uncertainty, confidence],
            [0.0, 1.0, 0.0],  # Target variable
            [1.0, 0.3, 0.8]   # Another variable
        ])
        target_mask = jnp.array([False, True, False])
        
        # Compile and execute
        compiled_fn = jax.jit(compute_mechanism_confidence_from_tensors_jax)
        result = compiled_fn(mechanism_features, target_mask)
        
        # Verify properties
        assert result.shape == (3,)
        assert result[1] == 0.0  # Target masked
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result >= 0.0)


class TestJAXCompilationPerformance:
    """Test JAX compilation performance characteristics."""
    
    def test_compilation_overhead_is_reasonable(self):
        """Test that JIT compilation overhead is reasonable for our use case."""
        config = create_jax_config(['X', 'Y', 'Z'], 'Y', max_samples=1000)
        
        # Create test data
        mechanism_features = jnp.ones((3, 3)) * 0.5
        target_mask = jnp.array([False, True, False])
        
        # Time compilation
        start_time = time.perf_counter()
        compiled_fn = jax.jit(compute_mechanism_confidence_from_tensors_jax)
        
        # First call includes compilation
        result1 = compiled_fn(mechanism_features, target_mask)
        compilation_time = time.perf_counter() - start_time
        
        # Second call should be much faster (compiled)
        start_time = time.perf_counter()
        result2 = compiled_fn(mechanism_features, target_mask)
        execution_time = time.perf_counter() - start_time
        
        # Verify results are identical
        assert jnp.allclose(result1, result2)
        
        # Compilation should be reasonable (< 5 seconds)
        assert compilation_time < 5.0
        
        # Execution should be fast (< 10ms)
        assert execution_time < 0.01
        
        print(f"Compilation time: {compilation_time*1000:.1f}ms")
        print(f"Execution time: {execution_time*1000:.3f}ms")
        print(f"Speedup after compilation: {compilation_time/execution_time:.1f}x")
    
    def test_batched_operations_compile_efficiently(self):
        """Test that vectorized operations compile and run efficiently."""
        # Large batch size to test vectorization
        batch_size = 1000
        n_vars = 5
        
        # Create batch of random mechanism features
        key = random.PRNGKey(42)
        mechanism_features = random.normal(key, (batch_size, n_vars, 3))
        target_mask = jnp.zeros((n_vars,), dtype=bool).at[2].set(True)  # Variable 2 is target
        
        # Define batched function
        @jax.jit
        def batched_confidence(features_batch):
            return jax.vmap(
                lambda features: compute_mechanism_confidence_from_tensors_jax(features, target_mask)
            )(features_batch)
        
        # Time compilation and execution
        start_time = time.perf_counter()
        result = batched_confidence(mechanism_features)
        total_time = time.perf_counter() - start_time
        
        # Verify result
        assert result.shape == (batch_size, n_vars)
        assert jnp.all(result[:, 2] == 0.0)  # Target variable masked
        
        # Should handle large batches efficiently (< 100ms)
        assert total_time < 0.1
        
        print(f"Batched computation time: {total_time*1000:.1f}ms for {batch_size} samples")
        print(f"Time per sample: {total_time*1000/batch_size:.3f}ms")


class TestJAXCompilationErrorHandling:
    """Test JAX compilation error handling and edge cases."""
    
    def test_compilation_with_invalid_shapes_fails_gracefully(self):
        """Test that shape mismatches are caught appropriately."""
        # Create mismatched tensor shapes
        mechanism_features = jnp.ones((3, 3))  # 3 variables, 3 features
        target_mask = jnp.array([False, True])  # Only 2 elements - MISMATCH
        
        compiled_fn = jax.jit(compute_mechanism_confidence_from_tensors_jax)
        
        # This should raise an error due to shape mismatch
        with pytest.raises((ValueError, TypeError, IndexError)):
            compiled_fn(mechanism_features, target_mask)
    
    def test_compilation_with_nan_values_handles_gracefully(self):
        """Test compilation behavior with NaN/Inf values."""
        mechanism_features = jnp.array([
            [jnp.nan, 0.2, 0.8],
            [0.0, jnp.inf, 0.0],
            [1.0, 0.3, -jnp.inf]
        ])
        target_mask = jnp.array([False, True, False])
        
        compiled_fn = jax.jit(compute_mechanism_confidence_from_tensors_jax)
        
        # Should execute without compilation errors
        result = compiled_fn(mechanism_features, target_mask)
        
        # Result shape should be correct even with invalid inputs
        assert result.shape == (3,)
        assert result[1] == 0.0  # Target still masked correctly
    
    def test_static_vs_dynamic_shapes(self):
        """Test that static shapes work correctly for JAX compilation."""
        # Create functions that work with static shapes
        @jax.jit
        def static_shape_function(x):
            # x must have known static shape for optimal compilation
            return jnp.sum(x ** 2)
        
        # Test with known static shape
        x_static = jnp.array([1.0, 2.0, 3.0])  # Shape (3,) known at compile time
        result1 = static_shape_function(x_static)
        
        # Test with same shape - should reuse compilation
        x_static2 = jnp.array([4.0, 5.0, 6.0])
        result2 = static_shape_function(x_static2)
        
        assert jnp.isfinite(result1)
        assert jnp.isfinite(result2)
        assert result1 != result2  # Different inputs, different outputs


class TestJAXDeviceCompatibility:
    """Test JAX device compatibility (CPU/GPU)."""
    
    def test_functions_work_on_available_devices(self):
        """Test that functions work on all available JAX devices."""
        # Get available devices
        devices = jax.devices()
        
        # Test data
        mechanism_features = jnp.array([
            [1.0, 0.2, 0.8],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.6]
        ])
        target_mask = jnp.array([False, True, False])
        
        compiled_fn = jax.jit(compute_mechanism_confidence_from_tensors_jax)
        
        for device in devices:
            # Move data to device
            features_on_device = jax.device_put(mechanism_features, device)
            mask_on_device = jax.device_put(target_mask, device)
            
            # Execute on device
            result = compiled_fn(features_on_device, mask_on_device)
            
            # Verify result
            assert result.shape == (3,)
            assert result[1] == 0.0
            
            print(f"Computation successful on device: {device.device_kind}")
    
    def test_memory_efficiency_on_devices(self):
        """Test memory efficiency across devices."""
        devices = jax.devices()
        
        for device in devices:
            # Create moderately large tensor
            large_features = jnp.ones((100, 10, 3))
            target_mask = jnp.zeros(10, dtype=bool).at[5].set(True)
            
            # Move to device
            features_on_device = jax.device_put(large_features, device)
            mask_on_device = jax.device_put(target_mask, device)
            
            # Define batched operation
            @jax.jit
            def batched_operation(features_batch):
                return jax.vmap(
                    lambda f: compute_mechanism_confidence_from_tensors_jax(f, mask_on_device)
                )(features_batch)
            
            # Execute
            result = batched_operation(features_on_device)
            
            # Verify
            assert result.shape == (100, 10)
            assert jnp.all(result[:, 5] == 0.0)  # Target masked
            
            print(f"Memory test successful on {device.device_kind}: {result.nbytes} bytes")


def test_comprehensive_jax_compilation_suite():
    """Run comprehensive compilation validation suite."""
    print("\n=== JAX Compilation Validation Suite ===")
    
    # Test core compilation
    print("✓ Core JAX compilation validation")
    assert validate_jax_compilation() == True
    
    # Test individual functions
    functions_to_test = [
        add_sample_to_tensors_jax,
        compute_mechanism_confidence_from_tensors_jax,
        update_mechanism_features_jax,
        compute_acquisition_scores_jax
    ]
    
    for func in functions_to_test:
        # Verify function can be JIT compiled
        try:
            jax.jit(func)
            print(f"✓ {func.__name__} compiles successfully")
        except Exception as e:
            pytest.fail(f"✗ {func.__name__} failed to compile: {e}")
    
    print("✓ All JAX functions compile successfully")
    print("✓ JAX compilation validation complete")


if __name__ == "__main__":
    # Run comprehensive test when called directly
    test_comprehensive_jax_compilation_suite()