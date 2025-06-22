"""
Tests for JAX-Native Sample Buffer

Validates circular buffer operations, JAX compilation, and tensor consistency.
"""

import pytest
import jax
import jax.numpy as jnp

from causal_bayes_opt.jax_native.config import create_test_config
from causal_bayes_opt.jax_native.sample_buffer import (
    JAXSampleBuffer, create_empty_jax_buffer, add_sample_jax,
    get_latest_samples_jax, create_test_buffer
)


class TestJAXSampleBuffer:
    """Test JAX sample buffer creation and validation."""
    
    def test_empty_buffer_creation(self):
        """Test creating an empty buffer."""
        config = create_test_config()
        buffer = create_empty_jax_buffer(config)
        
        assert buffer.values.shape == (config.max_samples, config.n_vars)
        assert buffer.interventions.shape == (config.max_samples, config.n_vars)
        assert buffer.targets.shape == (config.max_samples,)
        assert buffer.valid_mask.shape == (config.max_samples,)
        assert buffer.write_idx == 0
        assert buffer.n_samples == 0
        assert buffer.config == config
        assert buffer.is_empty()
        assert not buffer.is_full()
    
    def test_buffer_validation_shape_mismatch(self):
        """Test buffer validation catches shape mismatches."""
        config = create_test_config()
        
        with pytest.raises(ValueError, match="values shape"):
            JAXSampleBuffer(
                values=jnp.zeros((50, config.n_vars)),  # Wrong max_samples
                interventions=jnp.zeros((config.max_samples, config.n_vars)),
                targets=jnp.zeros((config.max_samples,)),
                valid_mask=jnp.zeros((config.max_samples,), dtype=bool),
                write_idx=0,
                n_samples=0,
                config=config
            )
    
    def test_buffer_validation_invalid_indices(self):
        """Test buffer validation catches invalid indices."""
        config = create_test_config()
        
        with pytest.raises(ValueError, match="write_idx .* out of range"):
            JAXSampleBuffer(
                values=jnp.zeros((config.max_samples, config.n_vars)),
                interventions=jnp.zeros((config.max_samples, config.n_vars)),
                targets=jnp.zeros((config.max_samples,)),
                valid_mask=jnp.zeros((config.max_samples,), dtype=bool),
                write_idx=config.max_samples,  # Out of range
                n_samples=0,
                config=config
            )
    
    def test_get_latest_samples_empty(self):
        """Test getting latest samples from empty buffer."""
        config = create_test_config()
        buffer = create_empty_jax_buffer(config)
        
        values, interventions, targets = buffer.get_latest_samples(5)
        
        assert values.shape == (0, config.n_vars)
        assert interventions.shape == (0, config.n_vars)
        assert targets.shape == (0,)


class TestAddSampleJAX:
    """Test adding samples to buffer with JAX operations."""
    
    def test_add_single_sample(self):
        """Test adding a single sample."""
        config = create_test_config()
        buffer = create_empty_jax_buffer(config)
        
        # Add sample
        values = jnp.array([1.0, 2.0, 3.0])
        interventions = jnp.array([True, False, False])
        target_val = 2.0
        
        new_buffer = add_sample_jax(buffer, values, interventions, target_val)
        
        assert new_buffer.n_samples == 1
        assert new_buffer.write_idx == 1
        assert not new_buffer.is_empty()
        assert jnp.array_equal(new_buffer.values[0], values)
        assert jnp.array_equal(new_buffer.interventions[0], interventions.astype(jnp.float32))
        assert new_buffer.targets[0] == target_val
        assert new_buffer.valid_mask[0] == True
    
    def test_add_multiple_samples(self):
        """Test adding multiple samples."""
        config = create_test_config()
        buffer = create_empty_jax_buffer(config)
        
        # Add multiple samples
        for i in range(5):
            values = jnp.array([float(i), float(i+1), float(i+2)])
            interventions = jnp.zeros(config.n_vars, dtype=bool).at[i % 2].set(True)
            target_val = float(i + 1)
            
            buffer = add_sample_jax(buffer, values, interventions, target_val)
        
        assert buffer.n_samples == 5
        assert buffer.write_idx == 5
        assert not buffer.is_empty()
        
        # Check that all samples were stored correctly
        for i in range(5):
            assert buffer.valid_mask[i] == True
            assert buffer.targets[i] == float(i + 1)
    
    def test_circular_buffer_overflow(self):
        """Test circular buffer behavior when exceeding max_samples."""
        # Create small buffer for testing
        config = create_test_config()
        config = config.__class__(
            n_vars=config.n_vars,
            target_idx=config.target_idx,
            max_samples=3,  # Small buffer
            max_history=config.max_history,
            variable_names=config.variable_names,
            mechanism_types=config.mechanism_types
        )
        
        buffer = create_empty_jax_buffer(config)
        
        # Add more samples than buffer capacity
        for i in range(5):
            values = jnp.array([float(i), float(i+1), float(i+2)])
            interventions = jnp.zeros(config.n_vars, dtype=bool)
            target_val = float(i)
            
            buffer = add_sample_jax(buffer, values, interventions, target_val)
        
        # Should have 3 samples (max capacity)
        assert buffer.n_samples == 3
        assert buffer.write_idx == 2  # (5 % 3)
        assert buffer.is_full()
        
        # Should contain the last 3 samples (indices 2, 3, 4 from our loop)
        # But stored in circular positions
        assert jnp.all(buffer.valid_mask == True)
    
    def test_jax_compilation(self):
        """Test that the underlying tensor operations can be JIT compiled."""
        from causal_bayes_opt.jax_native.sample_buffer import add_sample_to_tensors_jax
        
        config = create_test_config()
        buffer = create_empty_jax_buffer(config)
        
        # The tensor-level function should be JAX-compiled
        values = jnp.array([1.0, 2.0, 3.0])
        interventions = jnp.array([True, False, False])
        target_val = 2.0
        
        # This should work since it operates on pure tensors
        new_values, new_interventions, new_targets, new_valid_mask, new_write_idx, new_n_samples = (
            add_sample_to_tensors_jax(
                buffer.values, buffer.interventions, buffer.targets, buffer.valid_mask,
                buffer.write_idx, buffer.n_samples, buffer.config.max_samples,
                values, interventions, target_val
            )
        )
        
        assert int(new_n_samples) == 1
        assert int(new_write_idx) == 1
        assert jnp.array_equal(new_values[0], values)


class TestGetLatestSamplesJAX:
    """Test extracting latest samples with JAX operations."""
    
    def test_get_latest_from_populated_buffer(self):
        """Test getting latest samples from populated buffer."""
        buffer = create_test_buffer()
        
        # Get latest 3 samples
        values, interventions, targets = get_latest_samples_jax(
            buffer.values, buffer.interventions, buffer.targets,
            buffer.valid_mask, buffer.write_idx, buffer.n_samples, 3
        )
        
        assert values.shape == (3, buffer.config.n_vars)
        assert interventions.shape == (3, buffer.config.n_vars)
        assert targets.shape == (3,)
    
    def test_get_more_than_available(self):
        """Test requesting more samples than available."""
        config = create_test_config()
        buffer = create_empty_jax_buffer(config)
        
        # Add 2 samples
        for i in range(2):
            values = jnp.array([float(i), float(i+1), float(i+2)])
            interventions = jnp.zeros(config.n_vars, dtype=bool)
            target_val = float(i)
            buffer = add_sample_jax(buffer, values, interventions, target_val)
        
        # Request 5 samples (more than available)
        values, interventions, targets = get_latest_samples_jax(
            buffer.values, buffer.interventions, buffer.targets,
            buffer.valid_mask, buffer.write_idx, buffer.n_samples, 5
        )
        
        # Should return only available samples
        assert values.shape == (2, config.n_vars)
        assert interventions.shape == (2, config.n_vars)
        assert targets.shape == (2,)
    
    def test_get_zero_samples(self):
        """Test requesting zero samples."""
        buffer = create_test_buffer()
        
        values, interventions, targets = get_latest_samples_jax(
            buffer.values, buffer.interventions, buffer.targets,
            buffer.valid_mask, buffer.write_idx, buffer.n_samples, 0
        )
        
        assert values.shape == (0, buffer.config.n_vars)
        assert interventions.shape == (0, buffer.config.n_vars)
        assert targets.shape == (0,)
    
    def test_latest_samples_jax_compilation(self):
        """Test that the tensor extraction logic works with fixed sizes."""
        buffer = create_test_buffer()
        
        # Test the actual extraction function with a fixed size
        values, interventions, targets = get_latest_samples_jax(
            buffer.values, buffer.interventions, buffer.targets,
            buffer.valid_mask, buffer.write_idx, buffer.n_samples, 3
        )
        
        assert values.shape == (3, buffer.config.n_vars)
        assert interventions.shape == (3, buffer.config.n_vars)
        assert targets.shape == (3,)


class TestBufferStatistics:
    """Test buffer statistics computation."""
    
    def test_buffer_statistics_empty(self):
        """Test statistics for empty buffer."""
        from causal_bayes_opt.jax_native.sample_buffer import compute_buffer_statistics_jax
        
        config = create_test_config()
        buffer = create_empty_jax_buffer(config)
        
        stats = compute_buffer_statistics_jax(buffer)
        
        assert stats['total_samples'] == 0
        assert stats['total_interventions'] == 0.0
        assert stats['coverage_rate'] == 0.0
        assert stats['buffer_utilization'] == 0.0
        assert stats['intervention_counts'].shape == (config.n_vars,)
    
    def test_buffer_statistics_populated(self):
        """Test statistics for populated buffer."""
        from causal_bayes_opt.jax_native.sample_buffer import compute_buffer_statistics_jax
        
        buffer = create_test_buffer()
        stats = compute_buffer_statistics_jax(buffer)
        
        assert stats['total_samples'] == buffer.n_samples
        assert stats['total_interventions'] >= 0.0
        assert 0.0 <= stats['coverage_rate'] <= 1.0
        assert 0.0 <= stats['buffer_utilization'] <= 1.0
        assert stats['intervention_counts'].shape == (buffer.config.n_vars,)
    
    def test_statistics_jax_compilation(self):
        """Test that statistics computation can be JIT compiled."""
        from causal_bayes_opt.jax_native.sample_buffer import compute_buffer_statistics_jax
        
        buffer = create_test_buffer()
        
        # This should compile without errors
        jitted_stats = jax.jit(compute_buffer_statistics_jax)
        stats = jitted_stats(buffer)
        
        assert 'total_samples' in stats
        assert 'total_interventions' in stats


class TestBufferIntegration:
    """Test buffer integration with other components."""
    
    def test_buffer_immutability(self):
        """Test that buffer operations are truly immutable."""
        config = create_test_config()
        original_buffer = create_empty_jax_buffer(config)
        
        # Add sample should return new buffer, not modify original
        values = jnp.array([1.0, 2.0, 3.0])
        interventions = jnp.array([True, False, False])
        target_val = 2.0
        
        new_buffer = add_sample_jax(original_buffer, values, interventions, target_val)
        
        # Original should be unchanged
        assert original_buffer.n_samples == 0
        assert original_buffer.write_idx == 0
        assert original_buffer.is_empty()
        
        # New buffer should have the sample
        assert new_buffer.n_samples == 1
        assert new_buffer.write_idx == 1
        assert not new_buffer.is_empty()
    
    def test_buffer_with_different_configs(self):
        """Test buffer behavior with different configurations."""
        # Small buffer
        small_config = create_test_config()
        small_config = small_config.__class__(
            n_vars=2,
            target_idx=1,
            max_samples=5,
            max_history=3,
            variable_names=('A', 'B'),
            mechanism_types=(0, 1)
        )
        
        buffer = create_empty_jax_buffer(small_config)
        
        # Fill buffer
        for i in range(7):  # More than capacity
            values = jnp.array([float(i), float(i+1)])
            interventions = jnp.array([True, False])
            target_val = float(i+1)
            
            buffer = add_sample_jax(buffer, values, interventions, target_val)
        
        assert buffer.n_samples == 5  # Max capacity
        assert buffer.is_full()
        
        # Latest samples should work correctly
        values, interventions, targets = buffer.get_latest_samples(3)
        assert values.shape == (3, 2)
        assert interventions.shape == (3, 2)
        assert targets.shape == (3,)