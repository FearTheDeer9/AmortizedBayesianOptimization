"""
Phase 4.3: Computational Efficiency Benchmarks

This module implements comprehensive performance benchmarks to validate that our
119x improvement surrogate integration system maintains excellent computational efficiency:
- Inference time scaling with variable count
- Memory usage optimization
- JAX compilation and caching efficiency
- Batch processing performance
- Production deployment readiness

Tests both absolute performance and scaling characteristics.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
import pyrsistent as pyr
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import time
import psutil
import gc
import numpy as onp

# Core imports
from causal_bayes_opt.experiments.test_scms import create_chain_test_scm
from causal_bayes_opt.data_structures.scm import get_variables, get_target
from causal_bayes_opt.acquisition.grpo import _extract_policy_input_from_tensor_state
from causal_bayes_opt.surrogate.bootstrap import create_bootstrap_surrogate_features
from causal_bayes_opt.surrogate.phase_manager import PhaseConfig, BootstrapConfig


@dataclass
class PerformanceBenchmarkResult:
    """Results from a computational performance benchmark."""
    benchmark_name: str
    variable_count: int
    avg_inference_time_ms: float
    p95_inference_time_ms: float
    memory_usage_mb: float
    peak_memory_mb: float
    throughput_samples_per_sec: float
    jit_compilation_time_ms: float
    scaling_efficiency: float  # How well it scales compared to O(n)
    meets_production_requirements: bool
    error_message: str = ""


class ComputationalEfficiencyBenchmarker:
    """Comprehensive benchmarker for computational efficiency testing."""
    
    def __init__(self):
        """Initialize benchmarker."""
        self.results: List[PerformanceBenchmarkResult] = []
        self.phase_config = PhaseConfig()
        self.bootstrap_config = BootstrapConfig()
        
        # Production requirements
        self.max_inference_time_ms = 50.0  # 50ms max per inference
        self.max_memory_mb = 500.0  # 500MB max memory usage
        self.min_throughput = 20.0  # 20 samples/sec minimum
        
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def benchmark_inference_time(self, variable_counts: List[int], n_trials: int = 100) -> Dict[int, PerformanceBenchmarkResult]:
        """
        Benchmark inference time scaling with variable count.
        
        Args:
            variable_counts: List of variable counts to test
            n_trials: Number of trials per variable count
            
        Returns:
            Results for each variable count
        """
        results = {}
        
        print(f"🔬 Benchmarking Inference Time (n_trials={n_trials})")
        print("-" * 60)
        
        for n_vars in variable_counts:
            try:
                # Create test SCM
                scm = create_chain_test_scm(chain_length=n_vars, coefficient=1.2)
                variables = list(get_variables(scm))
                
                # Generate bootstrap features
                bootstrap_features = create_bootstrap_surrogate_features(
                    scm=scm,
                    step=10,
                    config=self.phase_config,
                    bootstrap_config=self.bootstrap_config,
                    rng_key=random.PRNGKey(42)
                )
                
                # Create mock state
                @dataclass
                class MockConfig:
                    n_vars: int
                    max_history: int = 50
                    
                @dataclass
                class MockSampleBuffer:
                    n_samples: int = 10
                    
                @dataclass
                class MockTensorBackedState:
                    config: Any
                    mechanism_features: jnp.ndarray
                    marginal_probs: jnp.ndarray
                    confidence_scores: jnp.ndarray
                    current_step: int
                    sample_buffer: Any
                    training_progress: float
                
                config = MockConfig(n_vars=n_vars)
                sample_buffer = MockSampleBuffer()
                
                state = MockTensorBackedState(
                    config=config,
                    mechanism_features=bootstrap_features.node_embeddings,
                    marginal_probs=bootstrap_features.parent_probabilities,
                    confidence_scores=1.0 - bootstrap_features.uncertainties,
                    current_step=10,
                    sample_buffer=sample_buffer,
                    training_progress=0.1
                )
                
                # JIT compile the function first
                jit_start = time.time()
                
                @jax.jit
                def extract_features_jit(state):
                    return _extract_policy_input_from_tensor_state(state)
                
                # Initial compilation
                _ = extract_features_jit(state)
                jit_compilation_time = (time.time() - jit_start) * 1000
                
                # Measure memory before trials
                gc.collect()  # Clean up before measurement
                initial_memory = self.get_memory_usage_mb()
                
                # Benchmark inference time
                inference_times = []
                peak_memory = initial_memory
                
                for trial in range(n_trials):
                    # Measure memory periodically
                    if trial % 10 == 0:
                        current_memory = self.get_memory_usage_mb()
                        peak_memory = max(peak_memory, current_memory)
                    
                    # Time the inference
                    start_time = time.time()
                    output = extract_features_jit(state)
                    end_time = time.time()
                    
                    # Ensure output is actually computed (prevent lazy evaluation)
                    _ = output.block_until_ready()
                    
                    inference_time_ms = (end_time - start_time) * 1000
                    inference_times.append(inference_time_ms)
                
                # Calculate statistics
                avg_inference_time = float(onp.mean(inference_times))
                p95_inference_time = float(onp.percentile(inference_times, 95))
                final_memory = self.get_memory_usage_mb()
                memory_usage = final_memory - initial_memory
                throughput = 1000.0 / avg_inference_time  # samples per second
                
                # Calculate scaling efficiency (compared to O(n))
                baseline_time = 1.0  # 1ms baseline for n=3
                expected_linear_time = baseline_time * (n_vars / 3.0)
                scaling_efficiency = expected_linear_time / avg_inference_time if avg_inference_time > 0 else 0
                
                # Check production requirements
                meets_requirements = (
                    avg_inference_time <= self.max_inference_time_ms and
                    memory_usage <= self.max_memory_mb and
                    throughput >= self.min_throughput
                )
                
                result = PerformanceBenchmarkResult(
                    benchmark_name=f"Inference_Time_{n_vars}vars",
                    variable_count=n_vars,
                    avg_inference_time_ms=avg_inference_time,
                    p95_inference_time_ms=p95_inference_time,
                    memory_usage_mb=memory_usage,
                    peak_memory_mb=peak_memory,
                    throughput_samples_per_sec=throughput,
                    jit_compilation_time_ms=jit_compilation_time,
                    scaling_efficiency=scaling_efficiency,
                    meets_production_requirements=meets_requirements
                )
                
                results[n_vars] = result
                
                print(f"  {n_vars:2d} vars: {avg_inference_time:.2f}ms avg, {p95_inference_time:.2f}ms p95, "
                      f"{throughput:.1f} samples/sec, {memory_usage:.1f}MB")
                
            except Exception as e:
                result = PerformanceBenchmarkResult(
                    benchmark_name=f"Inference_Time_{n_vars}vars",
                    variable_count=n_vars,
                    avg_inference_time_ms=0.0,
                    p95_inference_time_ms=0.0,
                    memory_usage_mb=0.0,
                    peak_memory_mb=0.0,
                    throughput_samples_per_sec=0.0,
                    jit_compilation_time_ms=0.0,
                    scaling_efficiency=0.0,
                    meets_production_requirements=False,
                    error_message=str(e)
                )
                results[n_vars] = result
                print(f"  {n_vars:2d} vars: ❌ ERROR - {str(e)}")
        
        return results
    
    def benchmark_batch_processing(self, batch_sizes: List[int], n_vars: int = 5) -> Dict[int, PerformanceBenchmarkResult]:
        """
        Benchmark batch processing efficiency.
        
        Args:
            batch_sizes: List of batch sizes to test
            n_vars: Number of variables to use for testing
            
        Returns:
            Results for each batch size
        """
        results = {}
        
        print(f"🔬 Benchmarking Batch Processing (n_vars={n_vars})")
        print("-" * 60)
        
        # Create test SCM
        scm = create_chain_test_scm(chain_length=n_vars, coefficient=1.2)
        
        # Generate bootstrap features
        bootstrap_features = create_bootstrap_surrogate_features(
            scm=scm,
            step=10,
            config=self.phase_config,
            bootstrap_config=self.bootstrap_config,
            rng_key=random.PRNGKey(42)
        )
        
        for batch_size in batch_sizes:
            try:
                # Create batch of states
                @dataclass
                class MockConfig:
                    n_vars: int
                    max_history: int = 50
                    
                @dataclass
                class MockSampleBuffer:
                    n_samples: int = 10
                    
                @dataclass
                class MockTensorBackedState:
                    config: Any
                    mechanism_features: jnp.ndarray
                    marginal_probs: jnp.ndarray
                    confidence_scores: jnp.ndarray
                    current_step: int
                    sample_buffer: Any
                    training_progress: float
                
                config = MockConfig(n_vars=n_vars)
                sample_buffer = MockSampleBuffer()
                
                # Create batched inputs
                batched_embeddings = jnp.broadcast_to(
                    bootstrap_features.node_embeddings[None, :, :], 
                    (batch_size, n_vars, 128)
                )
                batched_probs = jnp.broadcast_to(
                    bootstrap_features.parent_probabilities[None, :], 
                    (batch_size, n_vars)
                )
                batched_confidence = jnp.broadcast_to(
                    (1.0 - bootstrap_features.uncertainties)[None, :], 
                    (batch_size, n_vars)
                )
                
                # Define batched processing function
                @jax.jit
                def process_batch(embeddings, probs, confidence):
                    """Process a batch of states efficiently."""
                    batch_outputs = []
                    
                    for i in range(batch_size):
                        # Create individual state
                        state = MockTensorBackedState(
                            config=config,
                            mechanism_features=embeddings[i],
                            marginal_probs=probs[i],
                            confidence_scores=confidence[i],
                            current_step=10,
                            sample_buffer=sample_buffer,
                            training_progress=0.1
                        )
                        
                        # Extract features
                        features = _extract_policy_input_from_tensor_state(state)
                        batch_outputs.append(features)
                    
                    return jnp.stack(batch_outputs)
                
                # Warmup JIT
                jit_start = time.time()
                _ = process_batch(batched_embeddings, batched_probs, batched_confidence)
                jit_time = (time.time() - jit_start) * 1000
                
                # Benchmark batch processing
                n_trials = 50
                processing_times = []
                
                gc.collect()
                initial_memory = self.get_memory_usage_mb()
                
                for trial in range(n_trials):
                    start_time = time.time()
                    output = process_batch(batched_embeddings, batched_probs, batched_confidence)
                    _ = output.block_until_ready()
                    end_time = time.time()
                    
                    processing_time_ms = (end_time - start_time) * 1000
                    processing_times.append(processing_time_ms)
                
                final_memory = self.get_memory_usage_mb()
                
                # Calculate statistics
                avg_processing_time = float(onp.mean(processing_times))
                p95_processing_time = float(onp.percentile(processing_times, 95))
                memory_usage = final_memory - initial_memory
                throughput = (batch_size * 1000.0) / avg_processing_time
                
                # Batch efficiency: time per sample vs single sample
                single_sample_time = 1.0  # Estimated 1ms for single sample
                batch_efficiency = (single_sample_time * batch_size) / avg_processing_time
                
                meets_requirements = (
                    avg_processing_time <= self.max_inference_time_ms * 2 and  # Allow 2x for batching
                    memory_usage <= self.max_memory_mb and
                    throughput >= self.min_throughput
                )
                
                result = PerformanceBenchmarkResult(
                    benchmark_name=f"Batch_Processing_{batch_size}",
                    variable_count=n_vars,
                    avg_inference_time_ms=avg_processing_time,
                    p95_inference_time_ms=p95_processing_time,
                    memory_usage_mb=memory_usage,
                    peak_memory_mb=final_memory,
                    throughput_samples_per_sec=throughput,
                    jit_compilation_time_ms=jit_time,
                    scaling_efficiency=batch_efficiency,
                    meets_production_requirements=meets_requirements
                )
                
                results[batch_size] = result
                
                print(f"  Batch {batch_size:3d}: {avg_processing_time:.2f}ms total, "
                      f"{avg_processing_time/batch_size:.2f}ms/sample, {throughput:.1f} samples/sec")
                
            except Exception as e:
                result = PerformanceBenchmarkResult(
                    benchmark_name=f"Batch_Processing_{batch_size}",
                    variable_count=n_vars,
                    avg_inference_time_ms=0.0,
                    p95_inference_time_ms=0.0,
                    memory_usage_mb=0.0,
                    peak_memory_mb=0.0,
                    throughput_samples_per_sec=0.0,
                    jit_compilation_time_ms=0.0,
                    scaling_efficiency=0.0,
                    meets_production_requirements=False,
                    error_message=str(e)
                )
                results[batch_size] = result
                print(f"  Batch {batch_size:3d}: ❌ ERROR - {str(e)}")
        
        return results
    
    def benchmark_memory_efficiency(self, n_vars: int = 6) -> PerformanceBenchmarkResult:
        """
        Benchmark memory usage patterns and efficiency.
        
        Args:
            n_vars: Number of variables to use for testing
            
        Returns:
            Memory efficiency benchmark result
        """
        print(f"🔬 Benchmarking Memory Efficiency (n_vars={n_vars})")
        print("-" * 60)
        
        try:
            # Create test SCM
            scm = create_chain_test_scm(chain_length=n_vars, coefficient=1.2)
            
            # Measure baseline memory
            gc.collect()
            baseline_memory = self.get_memory_usage_mb()
            
            # Generate bootstrap features
            bootstrap_features = create_bootstrap_surrogate_features(
                scm=scm,
                step=10,
                config=self.phase_config,
                bootstrap_config=self.bootstrap_config,
                rng_key=random.PRNGKey(42)
            )
            
            features_memory = self.get_memory_usage_mb()
            
            # Create multiple states to test memory accumulation
            n_states = 100
            states = []
            
            for i in range(n_states):
                @dataclass
                class MockConfig:
                    n_vars: int
                    max_history: int = 50
                    
                @dataclass
                class MockSampleBuffer:
                    n_samples: int = 10
                    
                @dataclass
                class MockTensorBackedState:
                    config: Any
                    mechanism_features: jnp.ndarray
                    marginal_probs: jnp.ndarray
                    confidence_scores: jnp.ndarray
                    current_step: int
                    sample_buffer: Any
                    training_progress: float
                
                config = MockConfig(n_vars=n_vars)
                sample_buffer = MockSampleBuffer()
                
                state = MockTensorBackedState(
                    config=config,
                    mechanism_features=bootstrap_features.node_embeddings,
                    marginal_probs=bootstrap_features.parent_probabilities,
                    confidence_scores=1.0 - bootstrap_features.uncertainties,
                    current_step=10 + i,
                    sample_buffer=sample_buffer,
                    training_progress=0.1
                )
                
                states.append(state)
                
                # Check memory every 20 states
                if i % 20 == 19:
                    current_memory = self.get_memory_usage_mb()
                    print(f"    After {i+1:3d} states: {current_memory - baseline_memory:.1f} MB")
            
            peak_memory = self.get_memory_usage_mb()
            
            # Process all states and measure memory during processing
            processing_start = time.time()
            outputs = []
            
            for state in states:
                output = _extract_policy_input_from_tensor_state(state)
                outputs.append(output)
            
            processing_time = (time.time() - processing_start) * 1000
            
            final_memory = self.get_memory_usage_mb()
            
            # Clean up and measure final memory
            del states, outputs
            gc.collect()
            cleanup_memory = self.get_memory_usage_mb()
            
            # Calculate metrics
            memory_per_state = (peak_memory - baseline_memory) / n_states
            memory_efficiency = memory_per_state < 1.0  # Less than 1MB per state
            avg_processing_time = processing_time / n_states
            throughput = n_states * 1000.0 / processing_time
            
            meets_requirements = (
                memory_per_state <= 2.0 and  # Max 2MB per state
                avg_processing_time <= self.max_inference_time_ms and
                throughput >= self.min_throughput
            )
            
            print(f"    Memory per state: {memory_per_state:.2f} MB")
            print(f"    Memory efficiency: {'✅' if memory_efficiency else '⚠️'}")
            print(f"    Peak memory: {peak_memory - baseline_memory:.1f} MB")
            print(f"    Cleanup efficiency: {(final_memory - cleanup_memory):.1f} MB freed")
            
            return PerformanceBenchmarkResult(
                benchmark_name=f"Memory_Efficiency_{n_vars}vars",
                variable_count=n_vars,
                avg_inference_time_ms=avg_processing_time,
                p95_inference_time_ms=avg_processing_time * 1.2,  # Estimate
                memory_usage_mb=memory_per_state,
                peak_memory_mb=peak_memory - baseline_memory,
                throughput_samples_per_sec=throughput,
                jit_compilation_time_ms=0.0,
                scaling_efficiency=1.0 if memory_efficiency else 0.5,
                meets_production_requirements=meets_requirements
            )
            
        except Exception as e:
            return PerformanceBenchmarkResult(
                benchmark_name=f"Memory_Efficiency_{n_vars}vars",
                variable_count=n_vars,
                avg_inference_time_ms=0.0,
                p95_inference_time_ms=0.0,
                memory_usage_mb=0.0,
                peak_memory_mb=0.0,
                throughput_samples_per_sec=0.0,
                jit_compilation_time_ms=0.0,
                scaling_efficiency=0.0,
                meets_production_requirements=False,
                error_message=str(e)
            )
    
    def run_comprehensive_efficiency_benchmarks(self) -> Dict[str, Any]:
        """
        Run comprehensive computational efficiency benchmarks.
        
        Returns:
            Comprehensive benchmark results
        """
        print("🚀 Starting Comprehensive Efficiency Benchmarks")
        print("=" * 70)
        
        start_time = time.time()
        
        # Run inference time benchmarks
        inference_results = self.benchmark_inference_time([3, 4, 5, 6, 7, 8], n_trials=50)
        
        # Run batch processing benchmarks
        batch_results = self.benchmark_batch_processing([1, 4, 8, 16, 32], n_vars=5)
        
        # Run memory efficiency benchmark
        memory_result = self.benchmark_memory_efficiency(n_vars=6)
        
        # Combine all results
        all_results = {
            'inference_scaling': inference_results,
            'batch_processing': batch_results,
            'memory_efficiency': {'single_test': memory_result}
        }
        
        # Calculate summary statistics
        all_individual_results = []
        for category_results in all_results.values():
            if isinstance(category_results, dict):
                for result in category_results.values():
                    if isinstance(result, PerformanceBenchmarkResult):
                        all_individual_results.append(result)
                    elif hasattr(result, '__iter__'):
                        all_individual_results.extend(result)
        
        total_benchmarks = len(all_individual_results)
        production_ready = sum(1 for r in all_individual_results if r.meets_production_requirements)
        production_ready_rate = production_ready / total_benchmarks if total_benchmarks > 0 else 0
        
        # Performance statistics
        avg_inference_time = onp.mean([r.avg_inference_time_ms for r in all_individual_results])
        max_inference_time = onp.max([r.avg_inference_time_ms for r in all_individual_results])
        avg_throughput = onp.mean([r.throughput_samples_per_sec for r in all_individual_results if r.throughput_samples_per_sec > 0])
        avg_memory_usage = onp.mean([r.memory_usage_mb for r in all_individual_results if r.memory_usage_mb > 0])
        
        validation_time = time.time() - start_time
        
        summary = {
            'total_benchmarks': total_benchmarks,
            'production_ready_count': production_ready,
            'production_ready_rate': production_ready_rate,
            'avg_inference_time_ms': float(avg_inference_time),
            'max_inference_time_ms': float(max_inference_time),
            'avg_throughput_samples_per_sec': float(avg_throughput),
            'avg_memory_usage_mb': float(avg_memory_usage),
            'benchmark_time_seconds': validation_time,
            'overall_efficiency': production_ready_rate >= 0.8  # 80% production ready
        }
        
        return {
            'results': all_results,
            'summary': summary,
            'individual_results': all_individual_results
        }


# Test functions for pytest integration
def test_inference_time_meets_requirements():
    """Test that inference time meets production requirements."""
    benchmarker = ComputationalEfficiencyBenchmarker()
    results = benchmarker.benchmark_inference_time([5], n_trials=20)  # Quick test
    
    result = results[5]
    assert result.avg_inference_time_ms <= 50.0, f"Inference too slow: {result.avg_inference_time_ms}ms"
    assert result.throughput_samples_per_sec >= 20.0, f"Throughput too low: {result.throughput_samples_per_sec}"


def test_memory_usage_reasonable():
    """Test that memory usage is reasonable for production."""
    benchmarker = ComputationalEfficiencyBenchmarker()
    result = benchmarker.benchmark_memory_efficiency(n_vars=5)
    
    assert result.memory_usage_mb <= 5.0, f"Memory usage too high: {result.memory_usage_mb}MB per state"
    assert result.meets_production_requirements, f"Memory requirements not met: {result.error_message}"


def test_scaling_efficiency():
    """Test that performance scales reasonably with variable count."""
    benchmarker = ComputationalEfficiencyBenchmarker()
    results = benchmarker.benchmark_inference_time([3, 5, 7], n_trials=10)
    
    # Check that scaling is not worse than quadratic
    time_3 = results[3].avg_inference_time_ms
    time_7 = results[7].avg_inference_time_ms
    
    # 7/3 = 2.33, so quadratic scaling would be 2.33^2 = 5.4x
    max_acceptable_scaling = 6.0  # Allow some overhead
    actual_scaling = time_7 / max(time_3, 0.001)
    
    assert actual_scaling <= max_acceptable_scaling, f"Poor scaling: {actual_scaling}x (expected <{max_acceptable_scaling}x)"


def test_batch_processing_efficiency():
    """Test that batch processing provides efficiency gains."""
    benchmarker = ComputationalEfficiencyBenchmarker()
    results = benchmarker.benchmark_batch_processing([1, 8], n_vars=4)
    
    single_result = results[1]
    batch_result = results[8]
    
    # Batch processing should be more efficient per sample
    single_time_per_sample = single_result.avg_inference_time_ms
    batch_time_per_sample = batch_result.avg_inference_time_ms / 8
    
    efficiency_gain = single_time_per_sample / batch_time_per_sample
    assert efficiency_gain >= 1.5, f"Insufficient batch efficiency: {efficiency_gain}x"


if __name__ == "__main__":
    """Run comprehensive efficiency benchmarks when executed directly."""
    benchmarker = ComputationalEfficiencyBenchmarker()
    results = benchmarker.run_comprehensive_efficiency_benchmarks()
    
    print("\n📊 COMPREHENSIVE EFFICIENCY BENCHMARK RESULTS")
    print("=" * 70)
    
    summary = results['summary']
    print(f"Total Benchmarks: {summary['total_benchmarks']}")
    print(f"Production Ready: {summary['production_ready_count']}/{summary['total_benchmarks']} ({summary['production_ready_rate']:.2%})")
    print(f"Average Inference Time: {summary['avg_inference_time_ms']:.2f}ms")
    print(f"Maximum Inference Time: {summary['max_inference_time_ms']:.2f}ms")
    print(f"Average Throughput: {summary['avg_throughput_samples_per_sec']:.1f} samples/sec")
    print(f"Average Memory Usage: {summary['avg_memory_usage_mb']:.2f}MB")
    print(f"Benchmark Time: {summary['benchmark_time_seconds']:.1f} seconds")
    
    print(f"\n🎯 OVERALL EFFICIENCY: {'🎉 EXCELLENT' if summary['overall_efficiency'] else '⚠️ NEEDS OPTIMIZATION'}")
    
    # Detailed results by category
    inference_results = results['results']['inference_scaling']
    print(f"\n📋 INFERENCE TIME SCALING:")
    for n_vars, result in inference_results.items():
        status = "✅" if result.meets_production_requirements else "⚠️"
        print(f"  {status} {n_vars} vars: {result.avg_inference_time_ms:.2f}ms, {result.throughput_samples_per_sec:.1f} samples/sec")
    
    batch_results = results['results']['batch_processing']
    print(f"\n📋 BATCH PROCESSING:")
    for batch_size, result in batch_results.items():
        status = "✅" if result.meets_production_requirements else "⚠️"
        time_per_sample = result.avg_inference_time_ms / batch_size
        print(f"  {status} Batch {batch_size}: {time_per_sample:.2f}ms/sample, {result.throughput_samples_per_sec:.1f} samples/sec")
    
    memory_result = results['results']['memory_efficiency']['single_test']
    print(f"\n📋 MEMORY EFFICIENCY:")
    status = "✅" if memory_result.meets_production_requirements else "⚠️"
    print(f"  {status} Memory per state: {memory_result.memory_usage_mb:.2f}MB")
    print(f"  Peak memory: {memory_result.peak_memory_mb:.1f}MB")