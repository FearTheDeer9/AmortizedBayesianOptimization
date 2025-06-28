"""
Comprehensive tests for distributed training infrastructure.

This module tests JAX device detection, memory management, gradient checkpointing,
and multi-GPU parallelism using property-based testing with Hypothesis.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as onp
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path

from causal_bayes_opt.training.distributed import (
    DeviceInfo, DistributedConfig, MemoryEstimate, CheckpointConfig,
    detect_available_devices, select_optimal_devices, estimate_memory_usage,
    calculate_optimal_batch_size, create_distributed_config,
    create_checkpointed_layer, optimize_checkpointing_strategy,
    scale_batch_size_for_memory, ParallelismConfig,
    create_parallel_training_setup, optimize_parallelism_strategy
)


class TestDeviceDetection:
    """Test device detection and selection functionality."""
    
    def test_detect_available_devices_returns_tuple(self):
        """Test that device detection returns a tuple of DeviceInfo objects."""
        devices = detect_available_devices()
        
        assert isinstance(devices, tuple)
        assert len(devices) > 0  # Should have at least CPU
        
        for device in devices:
            assert isinstance(device, DeviceInfo)
            assert device.device_id >= 0
            assert device.device_type in ['cpu', 'gpu', 'tpu']
            assert device.memory_gb > 0
            assert isinstance(device.is_available, bool)
    
    @given(
        min_memory=st.floats(min_value=0.1, max_value=32.0),
        prefer_gpu=st.booleans()
    )
    @settings(max_examples=20)
    def test_select_optimal_devices_properties(self, min_memory, prefer_gpu):
        """Property-based test for device selection."""
        # Create mock devices with various configurations
        mock_devices = (
            DeviceInfo(0, 'cpu', 16.0, None, True),
            DeviceInfo(1, 'gpu', 8.0, '7.5', True),
            DeviceInfo(2, 'gpu', 24.0, '8.0', True),
            DeviceInfo(3, 'tpu', 32.0, None, False),  # Unavailable
        )
        
        selected = select_optimal_devices(mock_devices, min_memory, prefer_gpu)
        
        # Properties that should hold
        assert isinstance(selected, tuple)
        assert len(selected) > 0  # Should select at least one device
        
        # All selected devices should meet memory requirement
        for device in selected:
            assert device.memory_gb >= min_memory or device.is_available
        
        # If prefer_gpu=True and GPUs are available, GPUs should come first
        if prefer_gpu:
            gpu_devices = [d for d in selected if d.device_type == 'gpu']
            cpu_devices = [d for d in selected if d.device_type == 'cpu']
            if gpu_devices and cpu_devices:
                # First GPU should come before first CPU
                gpu_idx = next(i for i, d in enumerate(selected) if d.device_type == 'gpu')
                cpu_idx = next(i for i, d in enumerate(selected) if d.device_type == 'cpu')
                assert gpu_idx < cpu_idx


class TestMemoryEstimation:
    """Test memory usage estimation and optimization."""
    
    @given(
        model_params=st.integers(min_value=1000, max_value=10_000_000),
        batch_size=st.integers(min_value=1, max_value=512),
        sequence_length=st.integers(min_value=10, max_value=2048),
        dtype_bytes=st.sampled_from([2, 4, 8])  # half, float32, float64
    )
    @settings(max_examples=50)
    def test_memory_estimation_properties(self, model_params, batch_size, sequence_length, dtype_bytes):
        """Property-based test for memory estimation."""
        estimate = estimate_memory_usage(model_params, batch_size, sequence_length, dtype_bytes)
        
        # Basic sanity checks
        assert isinstance(estimate, MemoryEstimate)
        assert estimate.model_params_gb > 0
        assert estimate.optimizer_state_gb > 0
        assert estimate.activations_gb >= 0
        assert estimate.batch_data_gb >= 0
        assert estimate.total_gb > 0
        
        # Memory components should sum to total (approximately)
        computed_total = (
            estimate.model_params_gb + 
            estimate.optimizer_state_gb + 
            estimate.activations_gb + 
            estimate.batch_data_gb
        )
        assert abs(computed_total - estimate.total_gb) < 1e-6
        
        # Optimizer state should be roughly 2x model params (Adam)
        assert estimate.optimizer_state_gb >= estimate.model_params_gb
        assert estimate.optimizer_state_gb <= 3 * estimate.model_params_gb
        
        # Memory should scale with parameters and batch size
        larger_params_estimate = estimate_memory_usage(
            model_params * 2, batch_size, sequence_length, dtype_bytes
        )
        assert larger_params_estimate.total_gb > estimate.total_gb
        
        larger_batch_estimate = estimate_memory_usage(
            model_params, batch_size * 2, sequence_length, dtype_bytes
        )
        assert larger_batch_estimate.total_gb > estimate.total_gb
    
    def test_calculate_optimal_batch_size_scaling(self):
        """Test batch size calculation with different memory constraints."""
        device = DeviceInfo(0, 'gpu', 16.0, '8.0', True)
        base_estimate = MemoryEstimate(
            model_params_gb=2.0,
            optimizer_state_gb=4.0,
            activations_gb=1.0,
            batch_data_gb=1.0,
            total_gb=8.0
        )
        base_batch_size = 16
        
        # Should fit with current memory
        optimal_size = calculate_optimal_batch_size(device, base_estimate, base_batch_size)
        assert optimal_size >= base_batch_size  # Should allow larger batch
        
        # Test with tight memory constraint
        tight_device = DeviceInfo(0, 'gpu', 4.0, '8.0', True)
        tight_size = calculate_optimal_batch_size(tight_device, base_estimate, base_batch_size)
        assert tight_size <= base_batch_size  # Should require smaller batch
        assert tight_size >= 1  # Should be at least 1
    
    @given(
        available_memory=st.floats(min_value=1.0, max_value=64.0),
        total_memory_usage=st.floats(min_value=0.5, max_value=32.0),
        base_batch_size=st.integers(min_value=1, max_value=128)
    )
    @settings(max_examples=30)
    def test_batch_size_calculation_properties(self, available_memory, total_memory_usage, base_batch_size):
        """Property-based test for batch size calculation."""
        device = DeviceInfo(0, 'gpu', available_memory, '8.0', True)
        estimate = MemoryEstimate(
            model_params_gb=total_memory_usage * 0.3,
            optimizer_state_gb=total_memory_usage * 0.4,
            activations_gb=total_memory_usage * 0.2,
            batch_data_gb=total_memory_usage * 0.1,
            total_gb=total_memory_usage
        )
        
        optimal_size = calculate_optimal_batch_size(device, estimate, base_batch_size)
        
        # Should always return a positive integer
        assert isinstance(optimal_size, int)
        assert optimal_size >= 1
        
        # Should be power of 2 for optimal performance (when scaling up)
        if optimal_size > base_batch_size:
            # Check if it's a power of 2
            assert optimal_size & (optimal_size - 1) == 0


class TestGradientCheckpointing:
    """Test gradient checkpointing functionality."""
    
    def test_checkpoint_config_immutable(self):
        """Test that CheckpointConfig is immutable."""
        config = CheckpointConfig(enable_checkpointing=True, checkpoint_every_n_layers=3)
        
        # Should not be able to modify
        with pytest.raises(AttributeError):
            config.enable_checkpointing = False
    
    def test_create_checkpointed_layer_disabled(self):
        """Test checkpointed layer creation when disabled."""
        config = CheckpointConfig(enable_checkpointing=False)
        
        def dummy_layer(x):
            return x * 2
        
        checkpointed_fn = create_checkpointed_layer(dummy_layer, config)
        
        # Should return original function when disabled
        assert checkpointed_fn is dummy_layer
    
    def test_create_checkpointed_layer_enabled(self):
        """Test checkpointed layer creation when enabled."""
        config = CheckpointConfig(enable_checkpointing=True)
        
        def dummy_layer(x):
            return x * 2
        
        checkpointed_fn = create_checkpointed_layer(dummy_layer, config)
        
        # Should return a different function when enabled
        assert checkpointed_fn is not dummy_layer
        assert callable(checkpointed_fn)
        
        # Should still work correctly
        test_input = jnp.array([1.0, 2.0, 3.0])
        result = checkpointed_fn(test_input)
        expected = dummy_layer(test_input)
        assert jnp.allclose(result, expected)
    
    @given(
        device_memory=st.floats(min_value=4.0, max_value=64.0),
        model_layers=st.integers(min_value=1, max_value=100),
        base_memory=st.floats(min_value=1.0, max_value=32.0)
    )
    @settings(max_examples=30)
    def test_optimize_checkpointing_strategy_properties(self, device_memory, model_layers, base_memory):
        """Property-based test for checkpointing strategy optimization."""
        device = DeviceInfo(0, 'gpu', device_memory, '8.0', True)
        estimate = MemoryEstimate(
            model_params_gb=base_memory * 0.3,
            optimizer_state_gb=base_memory * 0.4,
            activations_gb=base_memory * 0.2,
            batch_data_gb=base_memory * 0.1,
            total_gb=base_memory
        )
        
        config = optimize_checkpointing_strategy(device, model_layers, estimate)
        
        assert isinstance(config, CheckpointConfig)
        
        # If memory is very constrained, should enable checkpointing
        available_memory = device_memory * 0.8
        if base_memory > available_memory:
            assert config.enable_checkpointing is True
            assert config.checkpoint_every_n_layers <= 2
        
        # If memory is abundant, might disable checkpointing
        if base_memory <= available_memory * 0.7:
            assert config.enable_checkpointing is False


class TestDistributedConfig:
    """Test distributed training configuration."""
    
    def test_create_distributed_config_defaults(self):
        """Test creation of distributed config with defaults."""
        # Mock devices
        mock_devices = (
            DeviceInfo(0, 'cpu', 16.0, None, True),
            DeviceInfo(1, 'gpu', 8.0, '8.0', True),
        )
        
        with patch('causal_bayes_opt.training.distributed.detect_available_devices', 
                   return_value=mock_devices):
            config = create_distributed_config()
        
        assert isinstance(config, DistributedConfig)
        assert len(config.devices) > 0
        assert config.strategy == "data_parallel"
        assert config.batch_size_per_device > 0
        assert config.gradient_accumulation_steps >= 1
        assert isinstance(config.use_gradient_checkpointing, bool)
    
    @given(
        strategy=st.sampled_from(["data_parallel", "model_parallel"]),
        base_batch_size=st.integers(min_value=1, max_value=64),
        model_params=st.integers(min_value=1000, max_value=50_000_000)
    )
    @settings(max_examples=20)
    def test_create_distributed_config_properties(self, strategy, base_batch_size, model_params):
        """Property-based test for distributed config creation."""
        mock_devices = (
            DeviceInfo(0, 'gpu', 16.0, '8.0', True),
            DeviceInfo(1, 'gpu', 8.0, '7.5', True),
        )
        
        config = create_distributed_config(
            devices=mock_devices,
            strategy=strategy,
            base_batch_size=base_batch_size,
            model_params=model_params
        )
        
        assert config.strategy == strategy
        assert config.batch_size_per_device >= 1
        assert config.devices == mock_devices
        
        # Large models should enable checkpointing
        if model_params > 10_000_000:
            assert config.use_gradient_checkpointing is True


class TestParallelism:
    """Test multi-GPU parallelism functionality."""
    
    def test_parallelism_config_immutable(self):
        """Test that ParallelismConfig is immutable."""
        config = ParallelismConfig(data_parallel=True, tensor_parallel_size=2)
        
        with pytest.raises(AttributeError):
            config.data_parallel = False
    
    @given(
        num_devices=st.integers(min_value=1, max_value=8),
        model_size=st.integers(min_value=1000, max_value=10_000_000_000),
        batch_size=st.integers(min_value=1, max_value=128)
    )
    @settings(max_examples=30)
    def test_optimize_parallelism_strategy_properties(self, num_devices, model_size, batch_size):
        """Property-based test for parallelism strategy optimization."""
        devices = tuple(
            DeviceInfo(i, 'gpu', 16.0, '8.0', True) 
            for i in range(num_devices)
        )
        
        config = optimize_parallelism_strategy(devices, model_size, batch_size)
        
        assert isinstance(config, ParallelismConfig)
        
        # Small models should use data parallelism only
        if model_size < 100_000_000 or num_devices <= 2:
            assert config.data_parallel is True
            assert config.model_parallel is False
            assert config.tensor_parallel_size == 1
        
        # Very large models should consider model parallelism
        if model_size > 1_000_000_000:
            assert config.tensor_parallel_size >= 1
            assert config.tensor_parallel_size <= min(num_devices, 4)
    
    @patch('jax.devices')
    def test_create_parallel_training_setup_data_parallel(self, mock_jax_devices):
        """Test creation of data parallel training setup."""
        # Mock JAX devices
        mock_devices = [Mock(), Mock()]
        mock_jax_devices.return_value = mock_devices
        
        devices = (
            DeviceInfo(0, 'gpu', 16.0, '8.0', True),
            DeviceInfo(1, 'gpu', 16.0, '8.0', True),
        )
        config = DistributedConfig(
            devices=devices,
            strategy="data_parallel",
            batch_size_per_device=8,
            gradient_accumulation_steps=1,
            use_gradient_checkpointing=False
        )
        parallelism_config = ParallelismConfig(data_parallel=True)
        
        with patch('jax.sharding.Mesh'):
            setup = create_parallel_training_setup(config, parallelism_config)
        
        assert setup['strategy'] == 'data_parallel'
        assert 'create_pmapped_train_step' in setup
        assert 'replicate_params' in setup
        assert 'shard_batch' in setup


class TestMemoryScaling:
    """Test memory-based batch size scaling."""
    
    def test_scale_batch_size_for_memory_increase(self):
        """Test batch size scaling when memory allows increase."""
        devices = (DeviceInfo(0, 'gpu', 32.0, '8.0', True),)  # Large memory
        base_config = DistributedConfig(
            devices=devices,
            strategy="data_parallel", 
            batch_size_per_device=8,
            gradient_accumulation_steps=1,
            use_gradient_checkpointing=False
        )
        
        scaled_config = scale_batch_size_for_memory(base_config, target_memory_usage=0.6)
        
        # Should enable checkpointing when scaling
        assert scaled_config.use_gradient_checkpointing is True
        assert scaled_config.devices == base_config.devices
        assert scaled_config.strategy == base_config.strategy
    
    def test_scale_batch_size_for_memory_decrease(self):
        """Test batch size scaling when memory requires decrease."""
        devices = (DeviceInfo(0, 'gpu', 4.0, '8.0', True),)  # Small memory
        base_config = DistributedConfig(
            devices=devices,
            strategy="data_parallel",
            batch_size_per_device=32,  # Large batch size
            gradient_accumulation_steps=1,
            use_gradient_checkpointing=False
        )
        
        scaled_config = scale_batch_size_for_memory(base_config, target_memory_usage=0.8)
        
        # Batch size should be reduced
        assert scaled_config.batch_size_per_device <= base_config.batch_size_per_device
        assert scaled_config.batch_size_per_device >= 1
        assert scaled_config.use_gradient_checkpointing is True


# Integration tests
class TestDistributedIntegration:
    """Integration tests for distributed training components."""
    
    def test_end_to_end_device_selection_and_config(self):
        """Test end-to-end device selection and configuration creation."""
        # This test uses actual JAX devices (will be CPU in CI)
        devices = detect_available_devices()
        assert len(devices) > 0
        
        selected_devices = select_optimal_devices(devices, min_memory_gb=1.0)
        assert len(selected_devices) > 0
        
        config = create_distributed_config(
            devices=selected_devices,
            strategy="data_parallel",
            base_batch_size=4,
            model_params=100_000
        )
        
        assert isinstance(config, DistributedConfig)
        assert config.devices == selected_devices
        assert config.batch_size_per_device >= 1
    
    def test_memory_estimation_and_optimization_pipeline(self):
        """Test the complete memory estimation and optimization pipeline."""
        # Estimate memory for a typical model
        estimate = estimate_memory_usage(
            model_params=1_000_000,
            batch_size=16,
            sequence_length=512
        )
        
        # Create a device with limited memory
        device = DeviceInfo(0, 'gpu', 8.0, '8.0', True)
        
        # Optimize batch size
        optimal_batch_size = calculate_optimal_batch_size(device, estimate, 16)
        
        # Optimize checkpointing
        checkpoint_config = optimize_checkpointing_strategy(device, 12, estimate)
        
        # All should be consistent
        assert optimal_batch_size >= 1
        assert isinstance(checkpoint_config, CheckpointConfig)
        
        # If memory is tight, should enable checkpointing
        if estimate.total_gb > device.memory_gb * 0.8:
            assert checkpoint_config.enable_checkpointing is True


# Performance and edge case tests
class TestDistributedEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_device_list_handling(self):
        """Test handling of empty device list."""
        with pytest.raises(RuntimeError, match="No available devices found"):
            select_optimal_devices((), min_memory_gb=1.0)
    
    def test_zero_memory_estimation(self):
        """Test edge case with zero parameters."""
        with pytest.raises(ValueError):
            estimate_memory_usage(0, 1, 1)
    
    def test_zero_batch_size_handling(self):
        """Test edge case with zero batch size."""
        with pytest.raises(ValueError):
            estimate_memory_usage(1000, 0, 1)
    
    def test_negative_memory_device(self):
        """Test device with negative memory (invalid)."""
        device = DeviceInfo(0, 'gpu', -1.0, '8.0', True)
        estimate = MemoryEstimate(1.0, 2.0, 0.5, 0.5, 4.0)
        
        # Should handle gracefully
        result = calculate_optimal_batch_size(device, estimate, 8)
        assert result >= 1  # Should still return a valid batch size
    
    def test_very_large_model_handling(self):
        """Test handling of very large models."""
        # Test with a model that would require > 1TB memory
        estimate = estimate_memory_usage(
            model_params=100_000_000_000,  # 100B parameters
            batch_size=1,
            sequence_length=1
        )
        
        # Should still return valid estimates
        assert estimate.total_gb > 0
        assert estimate.model_params_gb > 100  # Should be very large
    
    @given(
        checkpoint_every_n=st.integers(min_value=0, max_value=100)
    )
    def test_checkpoint_config_validation(self, checkpoint_every_n):
        """Test checkpoint configuration validation."""
        # Should handle any reasonable checkpoint frequency
        config = CheckpointConfig(
            enable_checkpointing=True,
            checkpoint_every_n_layers=checkpoint_every_n
        )
        
        assert config.checkpoint_every_n_layers == checkpoint_every_n


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])