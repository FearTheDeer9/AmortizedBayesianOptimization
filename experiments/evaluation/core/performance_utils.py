"""
Performance profiling utilities for evaluation experiments.

This module provides tools for measuring memory usage, timing,
and computational complexity of different methods.
"""

import logging
import time
import psutil
import gc
from typing import Dict, Any, Optional, Callable, ContextManager
from dataclasses import dataclass, field
from contextlib import contextmanager
import numpy as np

try:
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance measurements."""
    cpu_time: float = 0.0
    wall_time: float = 0.0
    peak_memory_mb: float = 0.0
    memory_delta_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    operation_count: int = 0
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


class PerformanceProfiler:
    """Profile performance metrics during evaluation."""
    
    def __init__(self, track_gpu: bool = True):
        """
        Initialize performance profiler.
        
        Args:
            track_gpu: Whether to track GPU memory (requires JAX)
        """
        self.track_gpu = track_gpu and JAX_AVAILABLE
        self.process = psutil.Process()
        self.baseline_memory = self._get_memory_usage()
        self.measurements = {}
        
        if self.track_gpu and JAX_AVAILABLE:
            try:
                # Test if we can access GPU memory
                jax.device_count()
                logger.info("GPU memory tracking enabled")
            except Exception:
                self.track_gpu = False
                logger.warning("GPU memory tracking disabled (no GPU access)")
        
        logger.info(f"PerformanceProfiler initialized (GPU tracking: {self.track_gpu})")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0
    
    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage in MB."""
        if not self.track_gpu:
            return 0.0
        
        try:
            # Try to get JAX device memory info
            devices = jax.devices()
            if devices:
                # This is a simplified approach - real GPU memory tracking
                # would require more sophisticated tools
                return 0.0  # Placeholder
        except Exception:
            return 0.0
    
    @contextmanager
    def profile_operation(self, operation_name: str) -> ContextManager[PerformanceMetrics]:
        """
        Context manager for profiling a single operation.
        
        Args:
            operation_name: Name of the operation being profiled
            
        Yields:
            PerformanceMetrics object that gets populated during execution
        """
        metrics = PerformanceMetrics()
        
        # Record start state
        start_time = time.time()
        start_cpu_time = time.process_time()
        start_memory = self._get_memory_usage()
        start_gpu_memory = self._get_gpu_memory()
        
        # Force garbage collection for cleaner measurements
        gc.collect()
        
        logger.debug(f"Starting profiling for: {operation_name}")
        
        try:
            yield metrics
        finally:
            # Record end state
            end_time = time.time()
            end_cpu_time = time.process_time()
            end_memory = self._get_memory_usage()
            end_gpu_memory = self._get_gpu_memory()
            
            # Compute metrics
            metrics.wall_time = end_time - start_time
            metrics.cpu_time = end_cpu_time - start_cpu_time
            metrics.memory_delta_mb = end_memory - start_memory
            metrics.peak_memory_mb = max(start_memory, end_memory)
            metrics.gpu_memory_mb = max(start_gpu_memory, end_gpu_memory)
            
            # Store measurement
            self.measurements[operation_name] = metrics
            
            logger.debug(f"Profiling complete for {operation_name}: "
                        f"{metrics.wall_time:.3f}s, {metrics.memory_delta_mb:.1f}MB")
    
    def profile_function(self, 
                        func: Callable,
                        operation_name: str,
                        *args, **kwargs) -> tuple:
        """
        Profile a function call and return both result and metrics.
        
        Args:
            func: Function to profile
            operation_name: Name for the operation
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Tuple of (function_result, performance_metrics)
        """
        with self.profile_operation(operation_name) as metrics:
            result = func(*args, **kwargs)
            metrics.operation_count = 1
        
        return result, metrics
    
    def profile_scaling(self,
                       func: Callable,
                       operation_name: str,
                       input_sizes: list,
                       input_generator: Callable,
                       **kwargs) -> Dict[int, PerformanceMetrics]:
        """
        Profile function across different input sizes for scaling analysis.
        
        Args:
            func: Function to profile
            operation_name: Base name for operations
            input_sizes: List of input sizes to test
            input_generator: Function that takes size and returns input args
            **kwargs: Additional arguments to pass to function
            
        Returns:
            Dictionary mapping input sizes to performance metrics
        """
        scaling_results = {}
        
        logger.info(f"Running scaling analysis for {operation_name}")
        logger.info(f"Testing sizes: {input_sizes}")
        
        for size in input_sizes:
            try:
                # Generate input for this size
                args, size_kwargs = input_generator(size)
                all_kwargs = {**kwargs, **size_kwargs}
                
                # Profile the operation
                with self.profile_operation(f"{operation_name}_size_{size}") as metrics:
                    result = func(*args, **all_kwargs)
                    metrics.operation_count = 1
                    metrics.additional_metrics['input_size'] = size
                    metrics.additional_metrics['result_type'] = type(result).__name__
                
                scaling_results[size] = metrics
                
                logger.debug(f"Size {size}: {metrics.wall_time:.3f}s, {metrics.memory_delta_mb:.1f}MB")
                
                # Force cleanup between runs
                del result
                gc.collect()
                
            except Exception as e:
                logger.error(f"Failed profiling at size {size}: {e}")
                # Create error metrics
                error_metrics = PerformanceMetrics()
                error_metrics.additional_metrics['error'] = str(e)
                scaling_results[size] = error_metrics
        
        return scaling_results
    
    def get_measurement(self, operation_name: str) -> Optional[PerformanceMetrics]:
        """Get performance measurement for a specific operation."""
        return self.measurements.get(operation_name)
    
    def get_all_measurements(self) -> Dict[str, PerformanceMetrics]:
        """Get all performance measurements."""
        return self.measurements.copy()
    
    def compare_methods(self, 
                       method_measurements: Dict[str, PerformanceMetrics]) -> Dict[str, Any]:
        """
        Compare performance metrics across methods.
        
        Args:
            method_measurements: Dictionary mapping method names to metrics
            
        Returns:
            Dictionary with comparative analysis
        """
        if not method_measurements:
            return {}
        
        # Extract metrics for comparison
        metrics_by_type = {
            'wall_time': {},
            'cpu_time': {},
            'memory_delta': {},
            'peak_memory': {}
        }
        
        for method_name, metrics in method_measurements.items():
            metrics_by_type['wall_time'][method_name] = metrics.wall_time
            metrics_by_type['cpu_time'][method_name] = metrics.cpu_time
            metrics_by_type['memory_delta'][method_name] = metrics.memory_delta_mb
            metrics_by_type['peak_memory'][method_name] = metrics.peak_memory_mb
        
        # Compute relative performance
        comparison = {}
        
        for metric_type, method_values in metrics_by_type.items():
            if not method_values:
                continue
                
            baseline_value = min(method_values.values())
            if baseline_value <= 0:
                baseline_value = 1.0  # Avoid division by zero
            
            comparison[metric_type] = {
                'absolute': method_values,
                'relative_to_best': {
                    method: value / baseline_value 
                    for method, value in method_values.items()
                },
                'fastest_method': min(method_values, key=method_values.get),
                'slowest_method': max(method_values, key=method_values.get),
                'speedup_factor': max(method_values.values()) / baseline_value
            }
        
        return comparison
    
    def export_scaling_analysis(self, 
                              scaling_results: Dict[int, PerformanceMetrics],
                              operation_name: str) -> Dict[str, Any]:
        """
        Export scaling analysis results.
        
        Args:
            scaling_results: Results from profile_scaling
            operation_name: Name of the operation
            
        Returns:
            Dictionary with scaling analysis
        """
        if not scaling_results:
            return {}
        
        sizes = sorted([s for s in scaling_results.keys() if not isinstance(s, str)])
        
        # Extract time and memory scaling
        times = []
        memories = []
        successful_sizes = []
        
        for size in sizes:
            metrics = scaling_results[size]
            if 'error' not in metrics.additional_metrics:
                successful_sizes.append(size)
                times.append(metrics.wall_time)
                memories.append(metrics.peak_memory_mb)
        
        if len(successful_sizes) < 2:
            return {'error': 'Insufficient successful measurements for scaling analysis'}
        
        # Compute scaling factors (rough approximation)
        time_scaling = self._estimate_scaling_factor(successful_sizes, times)
        memory_scaling = self._estimate_scaling_factor(successful_sizes, memories)
        
        return {
            'operation': operation_name,
            'successful_sizes': successful_sizes,
            'failed_sizes': [s for s in sizes if s not in successful_sizes],
            'time_scaling_factor': time_scaling,
            'memory_scaling_factor': memory_scaling,
            'max_feasible_size': max(successful_sizes),
            'time_by_size': dict(zip(successful_sizes, times)),
            'memory_by_size': dict(zip(successful_sizes, memories))
        }
    
    def _estimate_scaling_factor(self, sizes: list, values: list) -> float:
        """
        Estimate scaling factor (rough approximation).
        
        Args:
            sizes: List of input sizes
            values: Corresponding performance values
            
        Returns:
            Estimated scaling exponent (1.0 = linear, 2.0 = quadratic, etc.)
        """
        if len(sizes) < 2:
            return 1.0
        
        # Use log-log regression to estimate scaling
        log_sizes = np.log(sizes)
        log_values = np.log(np.maximum(values, 1e-6))  # Avoid log(0)
        
        try:
            # Simple linear regression in log space
            slope, _ = np.polyfit(log_sizes, log_values, 1)
            return max(0.1, min(10.0, slope))  # Clamp to reasonable range
        except Exception:
            return 1.0
    
    def reset(self) -> None:
        """Reset all measurements."""
        self.measurements.clear()
        self.baseline_memory = self._get_memory_usage()
        logger.info("Performance profiler reset")
    
    def summary(self) -> str:
        """Generate summary of all measurements."""
        if not self.measurements:
            return "No performance measurements recorded"
        
        lines = ["=" * 60]
        lines.append("PERFORMANCE PROFILER SUMMARY")
        lines.append("=" * 60)
        
        for operation, metrics in self.measurements.items():
            lines.append(f"\n{operation}:")
            lines.append(f"  Wall time: {metrics.wall_time:.3f}s")
            lines.append(f"  CPU time: {metrics.cpu_time:.3f}s")
            lines.append(f"  Memory delta: {metrics.memory_delta_mb:.1f}MB")
            lines.append(f"  Peak memory: {metrics.peak_memory_mb:.1f}MB")
            if self.track_gpu and metrics.gpu_memory_mb > 0:
                lines.append(f"  GPU memory: {metrics.gpu_memory_mb:.1f}MB")
            if metrics.additional_metrics:
                for key, value in metrics.additional_metrics.items():
                    lines.append(f"  {key}: {value}")
        
        lines.append(f"\nTotal operations: {len(self.measurements)}")
        lines.append("=" * 60)
        
        return "\n".join(lines)


@contextmanager
def quick_profile(operation_name: str) -> ContextManager[PerformanceMetrics]:
    """
    Simplified context manager for quick profiling.
    
    Args:
        operation_name: Name of the operation
        
    Yields:
        PerformanceMetrics object
    """
    profiler = PerformanceProfiler(track_gpu=False)
    with profiler.profile_operation(operation_name) as metrics:
        yield metrics


def benchmark_inference_scaling(inference_fn: Callable,
                               input_sizes: list,
                               input_generator: Callable,
                               n_repeats: int = 3) -> Dict[int, Dict[str, float]]:
    """
    Benchmark inference scaling across different input sizes.
    
    Args:
        inference_fn: Function to benchmark
        input_sizes: List of input sizes to test
        input_generator: Function that generates inputs for each size
        n_repeats: Number of repetitions for averaging
        
    Returns:
        Dictionary mapping sizes to averaged performance metrics
    """
    profiler = PerformanceProfiler()
    results = {}
    
    logger.info(f"Benchmarking inference scaling with {n_repeats} repeats")
    
    for size in input_sizes:
        size_results = []
        
        for repeat in range(n_repeats):
            try:
                # Generate input
                args, kwargs = input_generator(size)
                
                # Profile the inference
                with profiler.profile_operation(f"inference_size_{size}_rep_{repeat}") as metrics:
                    result = inference_fn(*args, **kwargs)
                    metrics.additional_metrics['input_size'] = size
                    metrics.additional_metrics['repeat'] = repeat
                
                size_results.append(metrics)
                
                # Cleanup
                del result
                gc.collect()
                
            except Exception as e:
                logger.error(f"Failed inference at size {size}, repeat {repeat}: {e}")
        
        # Average results for this size
        if size_results:
            avg_metrics = {
                'wall_time': np.mean([m.wall_time for m in size_results]),
                'cpu_time': np.mean([m.cpu_time for m in size_results]),
                'memory_delta': np.mean([m.memory_delta_mb for m in size_results]),
                'peak_memory': np.mean([m.peak_memory_mb for m in size_results]),
                'std_wall_time': np.std([m.wall_time for m in size_results]),
                'std_memory': np.std([m.memory_delta_mb for m in size_results]),
                'n_successful': len(size_results),
                'n_failed': n_repeats - len(size_results)
            }
            results[size] = avg_metrics
            
            logger.info(f"Size {size}: {avg_metrics['wall_time']:.3f}±{avg_metrics['std_wall_time']:.3f}s, "
                       f"{avg_metrics['memory_delta']:.1f}±{avg_metrics['std_memory']:.1f}MB")
    
    return results


def estimate_complexity_class(sizes: list, times: list) -> str:
    """
    Estimate computational complexity class from scaling data.
    
    Args:
        sizes: List of input sizes
        times: Corresponding execution times
        
    Returns:
        String describing estimated complexity (e.g., "O(n)", "O(n²)", "O(2^n)")
    """
    if len(sizes) < 3:
        return "Insufficient data"
    
    # Test different complexity models
    log_sizes = np.log(sizes)
    log_times = np.log(np.maximum(times, 1e-9))
    
    try:
        # Linear regression in log space: log(t) = a + b*log(n)
        slope, intercept = np.polyfit(log_sizes, log_times, 1)
        
        # Classify based on slope
        if slope < 0.5:
            return "O(1) - Constant"
        elif slope < 1.2:
            return "O(n) - Linear"
        elif slope < 1.8:
            return "O(n log n) - Linearithmic"  
        elif slope < 2.5:
            return "O(n²) - Quadratic"
        elif slope < 3.5:
            return "O(n³) - Cubic"
        else:
            # Check for exponential
            # Test if exponential model fits better
            try:
                exp_fit = np.polyfit(sizes, np.log(times), 1)
                exp_r2 = np.corrcoef(sizes, np.log(times))[0, 1] ** 2
                poly_r2 = np.corrcoef(log_sizes, log_times)[0, 1] ** 2
                
                if exp_r2 > poly_r2 and exp_r2 > 0.9:
                    return "O(2^n) - Exponential"
            except Exception:
                pass
            
            return f"O(n^{slope:.1f}) - Polynomial"
    
    except Exception:
        return "Unknown complexity"


def create_performance_comparison_table(results: Dict[str, PerformanceMetrics]) -> str:
    """
    Create formatted table comparing performance across methods.
    
    Args:
        results: Dictionary mapping method names to performance metrics
        
    Returns:
        Formatted table string
    """
    if not results:
        return "No performance data available"
    
    # Table header
    lines = []
    lines.append("Performance Comparison")
    lines.append("=" * 80)
    lines.append(f"{'Method':<20} {'Wall Time (s)':<15} {'Memory (MB)':<15} {'CPU Time (s)':<15}")
    lines.append("-" * 80)
    
    # Sort by wall time
    sorted_methods = sorted(results.items(), key=lambda x: x[1].wall_time)
    
    for method_name, metrics in sorted_methods:
        lines.append(f"{method_name:<20} {metrics.wall_time:<15.3f} "
                    f"{metrics.memory_delta_mb:<15.1f} {metrics.cpu_time:<15.3f}")
    
    lines.append("-" * 80)
    
    # Add relative performance
    fastest_time = min(m.wall_time for m in results.values())
    lines.append("\nRelative Performance (vs fastest):")
    lines.append("-" * 40)
    
    for method_name, metrics in sorted_methods:
        speedup = fastest_time / metrics.wall_time if metrics.wall_time > 0 else 1.0
        lines.append(f"{method_name:<20} {speedup:<15.2f}x")
    
    return "\n".join(lines)