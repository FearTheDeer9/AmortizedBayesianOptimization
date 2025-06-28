"""
GPU/Distributed Training Support for ACBO.

This module provides pure functions for JAX device detection, memory management,
and distributed training utilities following functional programming principles.
"""

import jax
import jax.numpy as jnp
import jax.random as random
from jax import checkpoint
from typing import Dict, List, Tuple, Optional, NamedTuple, Union, Callable, Any
import pyrsistent as pyr
from dataclasses import dataclass
import logging
import functools

logger = logging.getLogger(__name__)


# Immutable data structures for device information
@dataclass(frozen=True)
class DeviceInfo:
    """Immutable device information."""
    device_id: int
    device_type: str  # 'cpu', 'gpu', 'tpu'
    memory_gb: float
    compute_capability: Optional[str] = None
    is_available: bool = True


@dataclass(frozen=True) 
class DistributedConfig:
    """Immutable configuration for distributed training."""
    devices: Tuple[DeviceInfo, ...]
    strategy: str  # 'data_parallel', 'model_parallel', 'pipeline_parallel'
    batch_size_per_device: int
    gradient_accumulation_steps: int = 1
    use_gradient_checkpointing: bool = False


@dataclass(frozen=True)
class MemoryEstimate:
    """Immutable memory usage estimate."""
    model_params_gb: float
    optimizer_state_gb: float
    activations_gb: float
    batch_data_gb: float
    total_gb: float
    safety_margin: float = 0.2  # Reserve 20% for JAX overhead


# Pure device detection functions
def detect_available_devices() -> Tuple[DeviceInfo, ...]:
    """
    Detect all available JAX devices.
    
    Returns:
        Tuple of DeviceInfo objects for all detected devices.
    """
    devices = []
    jax_devices = jax.devices()
    
    for i, device in enumerate(jax_devices):
        device_type = device.platform.lower()
        
        # Estimate memory based on device type
        memory_gb = _estimate_device_memory(device)
        
        # Get compute capability for GPUs
        compute_capability = None
        if device_type == 'gpu':
            try:
                compute_capability = _get_gpu_compute_capability(device)
            except Exception:
                compute_capability = "unknown"
        
        device_info = DeviceInfo(
            device_id=i,
            device_type=device_type,
            memory_gb=memory_gb,
            compute_capability=compute_capability,
            is_available=True
        )
        devices.append(device_info)
    
    return tuple(devices)


def _estimate_device_memory(device) -> float:
    """Estimate device memory in GB."""
    try:
        # Try to get actual memory info from JAX
        if hasattr(device, 'memory_stats'):
            stats = device.memory_stats()
            return stats.get('bytes_limit', 0) / (1024**3)  # Convert to GB
        
        # Fallback estimates based on device type
        platform = device.platform.lower()
        if platform == 'gpu':
            return 8.0  # Conservative estimate for most GPUs
        elif platform == 'tpu':
            return 16.0  # TPU v3/v4 estimate
        else:  # CPU
            return 32.0  # Reasonable CPU RAM estimate
            
    except Exception:
        # Very conservative fallback
        return 4.0


def _get_gpu_compute_capability(device) -> str:
    """Get GPU compute capability string."""
    try:
        # This would require CUDA/platform-specific code
        # For now, return a placeholder
        return "unknown"
    except Exception:
        return "unknown"


def select_optimal_devices(
    devices: Tuple[DeviceInfo, ...],
    min_memory_gb: float = 4.0,
    prefer_gpu: bool = True
) -> Tuple[DeviceInfo, ...]:
    """
    Select optimal devices for training based on memory and preferences.
    
    Args:
        devices: Available devices
        min_memory_gb: Minimum memory requirement per device
        prefer_gpu: Whether to prefer GPUs over other device types
        
    Returns:
        Tuple of selected DeviceInfo objects
    """
    # Filter devices by memory requirement
    suitable_devices = tuple(
        device for device in devices 
        if device.memory_gb >= min_memory_gb and device.is_available
    )
    
    if not suitable_devices:
        # Fallback to any available device
        suitable_devices = tuple(device for device in devices if device.is_available)
    
    if not suitable_devices:
        raise RuntimeError("No available devices found")
    
    # Sort by preference: GPU > TPU > CPU, then by memory
    def device_priority(device: DeviceInfo) -> Tuple[int, float]:
        type_priority = {
            'gpu': 0 if prefer_gpu else 1,
            'tpu': 1 if prefer_gpu else 0, 
            'cpu': 2
        }
        return (type_priority.get(device.device_type, 3), -device.memory_gb)
    
    sorted_devices = tuple(sorted(suitable_devices, key=device_priority))
    return sorted_devices


def estimate_memory_usage(
    model_params: int,
    batch_size: int,
    sequence_length: int = 1000,
    dtype_bytes: int = 4  # float32
) -> MemoryEstimate:
    """
    Estimate memory usage for training.
    
    Args:
        model_params: Number of model parameters
        batch_size: Training batch size
        sequence_length: Sequence length for models
        dtype_bytes: Bytes per parameter (4 for float32)
        
    Returns:
        MemoryEstimate with breakdown of memory usage
    """
    # Model parameters (weights + gradients + optimizer state)
    model_params_gb = (model_params * dtype_bytes) / (1024**3)
    
    # Optimizer state (Adam requires 2x params for momentum + variance)
    optimizer_state_gb = model_params_gb * 2
    
    # Activations (rough estimate based on batch size and sequence length)
    activations_gb = (batch_size * sequence_length * model_params * dtype_bytes) / (1024**3) * 0.1
    
    # Batch data
    batch_data_gb = (batch_size * sequence_length * dtype_bytes) / (1024**3)
    
    total_gb = model_params_gb + optimizer_state_gb + activations_gb + batch_data_gb
    
    return MemoryEstimate(
        model_params_gb=model_params_gb,
        optimizer_state_gb=optimizer_state_gb,
        activations_gb=activations_gb,
        batch_data_gb=batch_data_gb,
        total_gb=total_gb
    )


def calculate_optimal_batch_size(
    device: DeviceInfo,
    memory_estimate: MemoryEstimate,
    base_batch_size: int = 8
) -> int:
    """
    Calculate optimal batch size based on available device memory.
    
    Args:
        device: Target device information
        memory_estimate: Memory usage estimate for base batch size
        base_batch_size: Base batch size used for memory estimate
        
    Returns:
        Optimal batch size that fits in device memory
    """
    available_memory = device.memory_gb * (1 - memory_estimate.safety_margin)
    
    if memory_estimate.total_gb <= available_memory:
        # Can fit larger batch sizes
        scaling_factor = available_memory / memory_estimate.total_gb
        optimal_batch_size = int(base_batch_size * scaling_factor)
        
        # Ensure power of 2 for optimal performance
        optimal_batch_size = 2 ** int(jnp.log2(optimal_batch_size))
        
        return max(optimal_batch_size, base_batch_size)
    else:
        # Need smaller batch size
        scaling_factor = available_memory / memory_estimate.total_gb
        optimal_batch_size = int(base_batch_size * scaling_factor)
        
        return max(optimal_batch_size, 1)  # At least batch size 1


def create_distributed_config(
    devices: Optional[Tuple[DeviceInfo, ...]] = None,
    strategy: str = "data_parallel",
    base_batch_size: int = 8,
    model_params: int = 1_000_000
) -> DistributedConfig:
    """
    Create optimal distributed training configuration.
    
    Args:
        devices: Available devices (auto-detect if None)
        strategy: Distributed training strategy
        base_batch_size: Base batch size per device
        model_params: Number of model parameters
        
    Returns:
        DistributedConfig with optimal settings
    """
    if devices is None:
        all_devices = detect_available_devices()
        devices = select_optimal_devices(all_devices)
    
    # Calculate memory requirements
    memory_est = estimate_memory_usage(model_params, base_batch_size)
    
    # Optimize batch size for the device with least memory
    min_memory_device = min(devices, key=lambda d: d.memory_gb)
    optimal_batch_size = calculate_optimal_batch_size(
        min_memory_device, memory_est, base_batch_size
    )
    
    # Enable gradient checkpointing for large models or small memory
    use_checkpointing = (
        model_params > 10_000_000 or  # Large model
        min_memory_device.memory_gb < 8.0  # Small memory
    )
    
    return DistributedConfig(
        devices=devices,
        strategy=strategy,
        batch_size_per_device=optimal_batch_size,
        gradient_accumulation_steps=1,
        use_gradient_checkpointing=use_checkpointing
    )


# JAX device mesh utilities for multi-device training
def create_device_mesh(devices: Tuple[DeviceInfo, ...]) -> jax.sharding.Mesh:
    """Create JAX device mesh for parallel computation."""
    jax_devices = jax.devices()
    selected_jax_devices = [jax_devices[d.device_id] for d in devices]
    
    # Create 1D mesh for data parallelism
    mesh_shape = (len(selected_jax_devices),)
    mesh = jax.sharding.Mesh(selected_jax_devices, axis_names=('batch',))
    
    return mesh


def setup_distributed_training(config: DistributedConfig) -> Dict[str, any]:
    """
    Setup distributed training environment.
    
    Args:
        config: Distributed training configuration
        
    Returns:
        Dictionary with mesh, sharding specs, and other setup info
    """
    mesh = create_device_mesh(config.devices)
    
    # Create sharding specifications for different tensor types
    batch_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('batch',))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    
    setup_info = {
        'mesh': mesh,
        'batch_sharding': batch_sharding,  # For batch data
        'param_sharding': replicated_sharding,  # For model parameters
        'devices': config.devices,
        'strategy': config.strategy
    }
    
    logger.info(f"Distributed training setup complete:")
    logger.info(f"  Devices: {len(config.devices)} ({[d.device_type for d in config.devices]})")
    logger.info(f"  Strategy: {config.strategy}")
    logger.info(f"  Batch size per device: {config.batch_size_per_device}")
    logger.info(f"  Gradient checkpointing: {config.use_gradient_checkpointing}")
    
    return setup_info


# Memory-efficient training utilities with gradient checkpointing
@dataclass(frozen=True)
class CheckpointConfig:
    """Configuration for gradient checkpointing."""
    enable_checkpointing: bool = True
    checkpoint_every_n_layers: int = 2
    save_residuals: bool = True
    memory_efficient_attention: bool = True


def create_checkpointed_layer(
    layer_fn: Callable,
    config: CheckpointConfig
) -> Callable:
    """
    Wrap a layer function with gradient checkpointing.
    
    Args:
        layer_fn: The layer function to checkpoint
        config: Checkpointing configuration
        
    Returns:
        Checkpointed layer function
    """
    if not config.enable_checkpointing:
        return layer_fn
    
    @functools.wraps(layer_fn)
    def checkpointed_layer(*args, **kwargs):
        return checkpoint(layer_fn)(*args, **kwargs)
    
    return checkpointed_layer


def create_gradient_checkpointed_forward(
    forward_fn: Callable,
    config: CheckpointConfig,
    n_layers: int
) -> Callable:
    """
    Create memory-efficient forward pass with gradient checkpointing.
    
    Args:
        forward_fn: Original forward function
        config: Checkpointing configuration  
        n_layers: Number of layers in the model
        
    Returns:
        Memory-efficient forward function
    """
    if not config.enable_checkpointing:
        return forward_fn
    
    # Determine which layers to checkpoint
    checkpoint_layers = set(
        range(0, n_layers, config.checkpoint_every_n_layers)
    )
    
    @functools.wraps(forward_fn)
    def memory_efficient_forward(params, inputs, *args, **kwargs):
        """Memory-efficient forward pass with selective checkpointing."""
        
        def single_layer_forward(layer_idx, layer_input):
            """Forward pass for a single layer."""
            return forward_fn(params, layer_input, layer_idx, *args, **kwargs)
        
        def checkpointed_layer_forward(layer_idx, layer_input):
            """Checkpointed forward pass."""
            return checkpoint(single_layer_forward)(layer_idx, layer_input)
        
        # Apply checkpointing selectively
        current_input = inputs
        for layer_idx in range(n_layers):
            if layer_idx in checkpoint_layers:
                current_input = checkpointed_layer_forward(layer_idx, current_input)
            else:
                current_input = single_layer_forward(layer_idx, current_input)
        
        return current_input
    
    return memory_efficient_forward


def estimate_memory_savings(
    model_layers: int,
    checkpoint_config: CheckpointConfig,
    base_memory_gb: float
) -> Tuple[float, float]:
    """
    Estimate memory savings from gradient checkpointing.
    
    Args:
        model_layers: Number of model layers
        checkpoint_config: Checkpointing configuration
        base_memory_gb: Base memory usage without checkpointing
        
    Returns:
        Tuple of (memory_with_checkpointing_gb, savings_ratio)
    """
    if not checkpoint_config.enable_checkpointing:
        return base_memory_gb, 0.0
    
    # Estimate savings based on checkpointing frequency
    checkpoint_ratio = 1.0 / max(checkpoint_config.checkpoint_every_n_layers, 1)
    
    # Gradient checkpointing trades compute for memory
    # Rough estimate: save ~50-70% of activation memory
    activation_memory_ratio = 0.3  # Activations are ~30% of total memory
    savings_from_activations = 0.6  # Save ~60% of activation memory
    
    memory_savings = base_memory_gb * activation_memory_ratio * savings_from_activations
    memory_with_checkpointing = base_memory_gb - memory_savings
    savings_ratio = memory_savings / base_memory_gb
    
    return memory_with_checkpointing, savings_ratio


def optimize_checkpointing_strategy(
    device: DeviceInfo,
    model_layers: int,
    base_memory_estimate: MemoryEstimate
) -> CheckpointConfig:
    """
    Optimize gradient checkpointing strategy based on available memory.
    
    Args:
        device: Target device information
        model_layers: Number of model layers
        base_memory_estimate: Base memory estimate without checkpointing
        
    Returns:
        Optimized CheckpointConfig
    """
    available_memory = device.memory_gb * 0.8  # Leave 20% margin
    
    # If we have plenty of memory, disable checkpointing
    if base_memory_estimate.total_gb <= available_memory * 0.7:
        return CheckpointConfig(enable_checkpointing=False)
    
    # If memory is tight, use aggressive checkpointing
    if base_memory_estimate.total_gb > available_memory:
        return CheckpointConfig(
            enable_checkpointing=True,
            checkpoint_every_n_layers=1,  # Checkpoint every layer
            save_residuals=False,  # Don't save residuals to save memory
            memory_efficient_attention=True
        )
    
    # Moderate checkpointing for intermediate cases
    return CheckpointConfig(
        enable_checkpointing=True,
        checkpoint_every_n_layers=2,
        save_residuals=True,
        memory_efficient_attention=True
    )


# Multi-GPU model parallelism utilities
@dataclass(frozen=True)
class ParallelismConfig:
    """Configuration for model parallelism."""
    data_parallel: bool = True
    model_parallel: bool = False
    pipeline_parallel: bool = False
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    

def create_parallel_training_setup(
    config: DistributedConfig,
    parallelism_config: ParallelismConfig
) -> Dict[str, Any]:
    """
    Create parallel training setup with pmap/pjit for multi-device training.
    
    Args:
        config: Distributed training configuration
        parallelism_config: Parallelism strategy configuration
        
    Returns:
        Dictionary with parallel training functions and utilities
    """
    mesh = create_device_mesh(config.devices)
    
    # Create different parallel training functions based on strategy
    if config.strategy == "data_parallel":
        return _create_data_parallel_setup(mesh, config)
    elif config.strategy == "model_parallel":
        return _create_model_parallel_setup(mesh, config, parallelism_config)
    else:
        raise ValueError(f"Unsupported parallelism strategy: {config.strategy}")


def _create_data_parallel_setup(
    mesh: jax.sharding.Mesh,
    config: DistributedConfig
) -> Dict[str, Any]:
    """Create data parallel training setup using pmap."""
    
    # Create pmapped training step
    def create_pmapped_train_step(train_step_fn: Callable) -> Callable:
        """Create data-parallel training step using pmap."""
        
        @jax.pmap
        def pmapped_train_step(state, batch):
            """Data-parallel training step."""
            return train_step_fn(state, batch)
        
        return pmapped_train_step
    
    # Create pmapped evaluation step
    def create_pmapped_eval_step(eval_step_fn: Callable) -> Callable:
        """Create data-parallel evaluation step using pmap."""
        
        @jax.pmap
        def pmapped_eval_step(state, batch):
            """Data-parallel evaluation step."""
            return eval_step_fn(state, batch)
        
        return pmapped_eval_step
    
    # Utility functions for data parallelism
    def replicate_params(params):
        """Replicate parameters across devices."""
        return jax.tree_map(
            lambda x: jnp.broadcast_to(x, (len(config.devices),) + x.shape),
            params
        )
    
    def unreplicate_params(replicated_params):
        """Get parameters from first device."""
        return jax.tree_map(lambda x: x[0], replicated_params)
    
    def shard_batch(batch, device_count: int):
        """Shard batch across devices."""
        def shard_array(arr):
            batch_size = arr.shape[0]
            per_device_batch_size = batch_size // device_count
            return arr.reshape((device_count, per_device_batch_size) + arr.shape[1:])
        
        return jax.tree_map(shard_array, batch)
    
    return {
        'mesh': mesh,
        'create_pmapped_train_step': create_pmapped_train_step,
        'create_pmapped_eval_step': create_pmapped_eval_step,
        'replicate_params': replicate_params,
        'unreplicate_params': unreplicate_params,
        'shard_batch': shard_batch,
        'strategy': 'data_parallel'
    }


def _create_model_parallel_setup(
    mesh: jax.sharding.Mesh,
    config: DistributedConfig,
    parallelism_config: ParallelismConfig
) -> Dict[str, Any]:
    """Create model parallel training setup using pjit."""
    
    # Create sharding specifications for model parallelism
    def create_model_sharding_specs(model_structure: Dict[str, Any]):
        """Create sharding specifications for model components."""
        
        # Example sharding for transformer-like models
        sharding_specs = {
            'embeddings': jax.sharding.PartitionSpec(None, 'model'),  # Shard embedding dimension
            'attention_weights': jax.sharding.PartitionSpec('model', None),  # Shard attention heads
            'feed_forward': jax.sharding.PartitionSpec(None, 'model'),  # Shard FF dimension
            'layer_norm': jax.sharding.PartitionSpec(),  # Replicate layer norms
            'output_projection': jax.sharding.PartitionSpec('model', None)  # Shard output
        }
        
        return sharding_specs
    
    # Create pjit training step
    def create_pjit_train_step(
        train_step_fn: Callable,
        in_shardings: Dict[str, jax.sharding.PartitionSpec],
        out_shardings: Dict[str, jax.sharding.PartitionSpec]
    ) -> Callable:
        """Create model-parallel training step using pjit."""
        
        @functools.partial(
            jax.jit,
            in_shardings=in_shardings,
            out_shardings=out_shardings
        )
        def pjit_train_step(state, batch):
            """Model-parallel training step."""
            return train_step_fn(state, batch)
        
        return pjit_train_step
    
    return {
        'mesh': mesh,
        'create_model_sharding_specs': create_model_sharding_specs,
        'create_pjit_train_step': create_pjit_train_step,
        'strategy': 'model_parallel',
        'tensor_parallel_size': parallelism_config.tensor_parallel_size
    }


def optimize_parallelism_strategy(
    devices: Tuple[DeviceInfo, ...],
    model_size: int,
    batch_size: int
) -> ParallelismConfig:
    """
    Optimize parallelism strategy based on hardware and model characteristics.
    
    Args:
        devices: Available devices
        model_size: Number of model parameters
        batch_size: Training batch size
        
    Returns:
        Optimized ParallelismConfig
    """
    num_devices = len(devices)
    total_memory = sum(d.memory_gb for d in devices)
    
    # For small models or few devices, use data parallelism
    if model_size < 100_000_000 or num_devices <= 2:
        return ParallelismConfig(
            data_parallel=True,
            model_parallel=False,
            tensor_parallel_size=1
        )
    
    # For large models, consider model parallelism
    if model_size > 1_000_000_000:
        # Large model - use tensor parallelism
        tensor_parallel_size = min(num_devices, 4)  # Max 4-way tensor parallel
        return ParallelismConfig(
            data_parallel=True,
            model_parallel=True,
            tensor_parallel_size=tensor_parallel_size
        )
    
    # Default to data parallelism for medium models
    return ParallelismConfig(
        data_parallel=True,
        model_parallel=False,
        tensor_parallel_size=1
    )


# Batch size scaling utilities
def scale_batch_size_for_memory(
    base_config: DistributedConfig,
    target_memory_usage: float = 0.8
) -> DistributedConfig:
    """
    Scale batch size to target memory usage.
    
    Args:
        base_config: Base distributed configuration
        target_memory_usage: Target memory usage ratio (0.0-1.0)
        
    Returns:
        Updated DistributedConfig with scaled batch sizes
    """
    # Find device with minimum memory
    min_memory_device = min(base_config.devices, key=lambda d: d.memory_gb)
    available_memory = min_memory_device.memory_gb * target_memory_usage
    
    # Estimate current memory usage
    memory_est = estimate_memory_usage(
        model_params=1_000_000,  # Default estimate
        batch_size=base_config.batch_size_per_device
    )
    
    if memory_est.total_gb <= available_memory:
        # Can increase batch size
        scaling_factor = available_memory / memory_est.total_gb
        new_batch_size = int(base_config.batch_size_per_device * scaling_factor)
        
        # Keep it as power of 2
        new_batch_size = 2 ** int(jnp.log2(new_batch_size))
    else:
        # Need to decrease batch size
        scaling_factor = available_memory / memory_est.total_gb
        new_batch_size = max(1, int(base_config.batch_size_per_device * scaling_factor))
    
    return DistributedConfig(
        devices=base_config.devices,
        strategy=base_config.strategy,
        batch_size_per_device=new_batch_size,
        gradient_accumulation_steps=base_config.gradient_accumulation_steps,
        use_gradient_checkpointing=True  # Enable checkpointing when scaling
    )


__all__ = [
    'DeviceInfo',
    'DistributedConfig', 
    'MemoryEstimate',
    'CheckpointConfig',
    'ParallelismConfig',
    'detect_available_devices',
    'select_optimal_devices',
    'estimate_memory_usage',
    'calculate_optimal_batch_size',
    'create_distributed_config',
    'create_device_mesh',
    'setup_distributed_training',
    'create_checkpointed_layer',
    'create_gradient_checkpointed_forward',
    'estimate_memory_savings',
    'optimize_checkpointing_strategy',
    'create_parallel_training_setup',
    'optimize_parallelism_strategy',
    'scale_batch_size_for_memory'
]