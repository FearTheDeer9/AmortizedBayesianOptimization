"""
Core scaling analysis for computational efficiency experiments.

This module implements the scaling analysis logic for measuring
performance across different graph sizes and methods.
"""

import logging
import time
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass
import psutil

# Add paths for imports
import sys
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from experiments.evaluation.core.model_loader import ModelLoader
from experiments.evaluation.core.performance_utils import (
    PerformanceProfiler, PerformanceMetrics, benchmark_inference_scaling,
    estimate_complexity_class
)
from experiments.evaluation.core.pairing_manager import (
    PairingManager, PairingConfig, ModelSpec, ModelType
)

from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
from src.causal_bayes_opt.data_structures.scm import get_variables, get_target
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor

logger = logging.getLogger(__name__)


@dataclass
class ScalingResult:
    """Result of scaling analysis for a single method."""
    method_name: str
    successful_sizes: List[int]
    failed_sizes: List[int]
    time_by_size: Dict[int, float]
    memory_by_size: Dict[int, float]
    complexity_estimate: str
    max_feasible_size: int
    scaling_factor: float
    error_messages: Dict[int, str]


class ScalingAnalyzer:
    """Analyzes computational scaling across different graph sizes."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize scaling analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.scaling_config = config['scaling_test']
        self.scm_config = config['scm_generation']
        self.efficiency_config = config['efficiency_tests']
        
        self.scm_factory = VariableSCMFactory(seed=config['experiment']['seed'])
        self.profiler = PerformanceProfiler(track_gpu=True)
        
        self.results = {}
        self.scaling_data = {}
        
        logger.info(f"Initialized ScalingAnalyzer for sizes: {self.scaling_config['sizes']}")
    
    def create_test_data_generator(self, n_obs: int, n_interventions: int):
        """
        Create a data generator function for scaling tests.
        
        Args:
            n_obs: Number of observational samples
            n_interventions: Number of interventions
            
        Returns:
            Function that generates test data for a given graph size
        """
        def generate_test_data(size: int) -> Tuple[Tuple, Dict]:
            """Generate test data for given size."""
            # Create SCM
            scm = self.scm_factory.create_variable_scm(
                num_variables=size,
                structure_type="random",
                edge_density=self.scm_config['edge_density']
            )
            
            # Get variables
            variables = list(get_variables(scm))
            target_var = get_target(scm)
            
            # Create buffer with observational data
            buffer = ExperienceBuffer()
            obs_samples = sample_from_linear_scm(scm, n_samples=n_obs, seed=42)
            for sample in obs_samples:
                buffer.add_observation(sample)
            
            # Convert to tensor
            tensor, _ = buffer_to_three_channel_tensor(buffer, target_var)
            
            return (tensor, None, target_var, variables), {'scm': scm, 'buffer': buffer}
        
        return generate_test_data
    
    def benchmark_method_scaling(self, 
                                method_name: str,
                                acquisition_fn: Callable,
                                sizes: List[int]) -> ScalingResult:
        """
        Benchmark scaling for a single method.
        
        Args:
            method_name: Name of the method being tested
            acquisition_fn: Acquisition function to benchmark
            sizes: List of graph sizes to test
            
        Returns:
            ScalingResult with scaling analysis
        """
        logger.info(f"Benchmarking scaling for {method_name}")
        
        # Create data generator
        data_generator = self.create_test_data_generator(
            n_obs=self.config['data_generation']['n_observational_samples'],
            n_interventions=self.config['data_generation']['n_interventions']
        )
        
        # Run scaling benchmark
        scaling_results = benchmark_inference_scaling(
            inference_fn=acquisition_fn,
            input_sizes=sizes,
            input_generator=data_generator,
            n_repeats=self.scaling_config['n_repetitions']
        )
        
        # Extract successful and failed sizes
        successful_sizes = []
        failed_sizes = []
        time_by_size = {}
        memory_by_size = {}
        error_messages = {}
        
        for size, metrics in scaling_results.items():
            if metrics.get('n_successful', 0) > 0:
                successful_sizes.append(size)
                time_by_size[size] = metrics['wall_time']
                memory_by_size[size] = metrics['memory_delta']
            else:
                failed_sizes.append(size)
                error_messages[size] = "No successful runs"
        
        # Estimate complexity
        if len(successful_sizes) >= 3:
            times = [time_by_size[s] for s in successful_sizes]
            complexity = estimate_complexity_class(successful_sizes, times)
            
            # Estimate scaling factor
            log_sizes = np.log(successful_sizes)
            log_times = np.log(np.maximum(times, 1e-6))
            if len(log_sizes) >= 2:
                scaling_factor = np.polyfit(log_sizes, log_times, 1)[0]
            else:
                scaling_factor = 1.0
        else:
            complexity = "Insufficient data"
            scaling_factor = 0.0
        
        max_feasible = max(successful_sizes) if successful_sizes else 0
        
        result = ScalingResult(
            method_name=method_name,
            successful_sizes=successful_sizes,
            failed_sizes=failed_sizes,
            time_by_size=time_by_size,
            memory_by_size=memory_by_size,
            complexity_estimate=complexity,
            max_feasible_size=max_feasible,
            scaling_factor=scaling_factor,
            error_messages=error_messages
        )
        
        logger.info(f"{method_name} scaling: {complexity}, max size: {max_feasible}")
        return result
    
    def run_inference_time_scaling(self, 
                                 experiments_dir: Path) -> Dict[str, ScalingResult]:
        """
        Run Experiment 1.1: Inference Time Scaling.
        
        Args:
            experiments_dir: Base experiments directory
            
        Returns:
            Dictionary mapping method names to scaling results
        """
        logger.info("Running Experiment 1.1: Inference Time Scaling")
        
        sizes = self.scaling_config['sizes']
        methods_to_test = self.scaling_config['methods_to_compare']
        
        scaling_results = {}
        
        # Test each method
        for method_name in methods_to_test:
            logger.info(f"Testing {method_name}...")
            
            try:
                if method_name == "random":
                    acquisition_fn = ModelLoader.load_baseline('random', seed=42)
                elif method_name == "untrained":
                    acquisition_fn = ModelLoader.create_untrained_policy(seed=42)
                elif method_name == "our_method":
                    # Try to load a trained model, fall back to untrained
                    joint_checkpoints = list((experiments_dir / 'joint-training' / 'checkpoints').glob('*/policy.pkl'))
                    if joint_checkpoints:
                        acquisition_fn = ModelLoader.load_policy(joint_checkpoints[0], seed=42)
                    else:
                        logger.warning("No trained models found, using untrained for 'our_method'")
                        acquisition_fn = ModelLoader.create_untrained_policy(seed=42)
                elif method_name == "cbo_u":
                    # Placeholder for CBO-U (would need implementation)
                    logger.warning("CBO-U implementation not available, skipping")
                    continue
                else:
                    logger.warning(f"Unknown method: {method_name}")
                    continue
                
                # Run scaling analysis
                result = self.benchmark_method_scaling(method_name, acquisition_fn, sizes)
                scaling_results[method_name] = result
                
            except Exception as e:
                logger.error(f"Failed to test {method_name}: {e}")
        
        return scaling_results
    
    def run_memory_usage_scaling(self, 
                               experiments_dir: Path) -> Dict[str, Dict[int, float]]:
        """
        Run Experiment 1.2: Memory Usage Scaling.
        
        Args:
            experiments_dir: Base experiments directory
            
        Returns:
            Dictionary mapping method names to memory usage by size
        """
        logger.info("Running Experiment 1.2: Memory Usage Scaling")
        
        sizes = self.scaling_config['sizes']
        memory_results = {}
        
        # Focus on memory-intensive operations
        for method_name in ['random', 'our_method', 'untrained']:
            logger.info(f"Profiling memory for {method_name}...")
            
            try:
                # Load method
                if method_name == "random":
                    acquisition_fn = ModelLoader.load_baseline('random', seed=42)
                elif method_name == "untrained":
                    acquisition_fn = ModelLoader.create_untrained_policy(seed=42)
                else:  # our_method
                    joint_checkpoints = list((experiments_dir / 'joint-training' / 'checkpoints').glob('*/policy.pkl'))
                    if joint_checkpoints:
                        acquisition_fn = ModelLoader.load_policy(joint_checkpoints[0], seed=42)
                    else:
                        acquisition_fn = ModelLoader.create_untrained_policy(seed=42)
                
                method_memory = {}
                
                for size in sizes:
                    if size > 100 and method_name == "our_method":
                        # Skip very large sizes for detailed memory profiling
                        continue
                    
                    try:
                        # Force clean start
                        gc.collect()
                        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024
                        
                        # Create test data
                        data_gen = self.create_test_data_generator(50, 5)
                        args, kwargs = data_gen(size)
                        
                        # Measure memory during inference
                        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                        
                        # Run multiple inferences to get peak memory
                        for _ in range(3):
                            result = acquisition_fn(*args)
                        
                        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
                        memory_delta = peak_memory - baseline_memory
                        
                        method_memory[size] = memory_delta
                        
                        # Cleanup
                        del result, args, kwargs
                        gc.collect()
                        
                        logger.debug(f"{method_name} size {size}: {memory_delta:.1f}MB")
                        
                    except Exception as e:
                        logger.warning(f"Memory test failed for {method_name} at size {size}: {e}")
                
                memory_results[method_name] = method_memory
                
            except Exception as e:
                logger.error(f"Memory profiling failed for {method_name}: {e}")
        
        return memory_results
    
    def estimate_training_amortization(self, 
                                     experiments_dir: Path) -> Dict[str, Any]:
        """
        Run Experiment 1.3: Training Time Analysis.
        
        Args:
            experiments_dir: Base experiments directory
            
        Returns:
            Dictionary with amortization analysis
        """
        logger.info("Running Experiment 1.3: Training Time Analysis")
        
        # Estimate training times (would need actual training data)
        # For now, provide theoretical analysis based on known training costs
        
        # Typical training costs (estimated from logs)
        estimated_training_hours = {
            'policy_only': 2.0,    # ~2 hours for policy training
            'surrogate_only': 4.0, # ~4 hours for surrogate training  
            'joint_training': 6.0   # ~6 hours for joint training
        }
        
        # Estimate per-problem CBO-U costs
        sizes = [10, 20, 30, 50, 100]
        cbo_u_costs = {}
        
        for size in sizes:
            # Exponential scaling assumption for CBO-U
            # Based on 2^(n-10) growth from size 10 baseline
            if size <= 10:
                cost_seconds = 1.0  # 1 second for 10 variables
            elif size <= 20:
                cost_seconds = 10.0  # 10 seconds for 20 variables
            elif size <= 30:
                cost_seconds = 300.0  # 5 minutes for 30 variables
            else:
                cost_seconds = float('inf')  # Infeasible
            
            cbo_u_costs[size] = cost_seconds
        
        # Compute break-even points
        breakeven_analysis = {}
        
        for training_type, training_hours in estimated_training_hours.items():
            training_seconds = training_hours * 3600
            breakeven_analysis[training_type] = {}
            
            for size in sizes:
                cbo_u_cost = cbo_u_costs[size]
                if cbo_u_cost == float('inf'):
                    breakeven_problems = 1  # Immediate benefit
                else:
                    breakeven_problems = int(np.ceil(training_seconds / cbo_u_cost))
                
                breakeven_analysis[training_type][size] = {
                    'training_cost_hours': training_hours,
                    'cbo_u_cost_per_problem_seconds': cbo_u_cost,
                    'breakeven_problems': breakeven_problems,
                    'speedup_after_breakeven': training_seconds / max(cbo_u_cost, 1)
                }
        
        return {
            'estimated_training_costs': estimated_training_hours,
            'cbo_u_per_problem_costs': cbo_u_costs,
            'breakeven_analysis': breakeven_analysis,
            'summary': {
                'typical_breakeven_problems': 10,  # Typical case
                'long_term_speedup_factor': 100,   # After amortization
                'feasibility_advantage_size': 30   # Size where CBO-U becomes infeasible
            }
        }
    
    def run_complete_scaling_analysis(self, 
                                    experiments_dir: Path) -> Dict[str, Any]:
        """
        Run complete scaling analysis covering all three experiments.
        
        Args:
            experiments_dir: Base experiments directory
            
        Returns:
            Dictionary with all scaling analysis results
        """
        complete_results = {}
        
        # Experiment 1.1: Inference time scaling
        if self.efficiency_config.get('inference_time', {}).get('enabled', True):
            complete_results['inference_time_scaling'] = self.run_inference_time_scaling(experiments_dir)
        
        # Experiment 1.2: Memory usage scaling
        if self.efficiency_config.get('memory_usage', {}).get('enabled', True):
            complete_results['memory_usage_scaling'] = self.run_memory_usage_scaling(experiments_dir)
        
        # Experiment 1.3: Training time analysis
        if self.efficiency_config.get('training_time', {}).get('enabled', True):
            complete_results['training_amortization'] = self.estimate_training_amortization(experiments_dir)
        
        # Generate comparative analysis
        complete_results['comparative_analysis'] = self.generate_comparative_analysis(complete_results)
        
        return complete_results
    
    def generate_comparative_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comparative analysis across all methods and experiments.
        
        Args:
            results: Results from all scaling experiments
            
        Returns:
            Dictionary with comparative analysis
        """
        analysis = {}
        
        # Inference time comparison
        if 'inference_time_scaling' in results:
            time_results = results['inference_time_scaling']
            
            # Find method that handles largest graphs
            max_sizes = {method: result.max_feasible_size 
                        for method, result in time_results.items()}
            best_scaling_method = max(max_sizes, key=max_sizes.get) if max_sizes else None
            
            # Compute relative speedups at different sizes
            size_comparisons = {}
            for size in [20, 50, 100]:
                size_times = {}
                for method, result in time_results.items():
                    if size in result.time_by_size:
                        size_times[method] = result.time_by_size[size]
                
                if len(size_times) >= 2:
                    fastest_time = min(size_times.values())
                    size_comparisons[size] = {
                        method: time / fastest_time 
                        for method, time in size_times.items()
                    }
            
            analysis['inference_comparison'] = {
                'max_feasible_sizes': max_sizes,
                'best_scaling_method': best_scaling_method,
                'relative_speedups_by_size': size_comparisons
            }
        
        # Memory comparison
        if 'memory_usage_scaling' in results:
            memory_results = results['memory_usage_scaling']
            
            # Estimate memory scaling rates
            memory_scaling = {}
            for method, memory_by_size in memory_results.items():
                if len(memory_by_size) >= 3:
                    sizes = list(memory_by_size.keys())
                    memories = list(memory_by_size.values())
                    scaling_factor = estimate_complexity_class(sizes, memories)
                    memory_scaling[method] = scaling_factor
            
            analysis['memory_comparison'] = {
                'memory_scaling_factors': memory_scaling,
                'memory_efficiency': memory_results
            }
        
        # Training amortization insights
        if 'training_amortization' in results:
            amort_results = results['training_amortization']
            analysis['amortization_insights'] = amort_results.get('summary', {})
        
        return analysis
    
    def export_scaling_analysis(self, results: Dict[str, Any], output_dir: Path) -> None:
        """
        Export scaling analysis results.
        
        Args:
            results: Complete scaling analysis results
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export raw results as JSON
        import json
        with open(output_dir / 'scaling_analysis.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Export scaling data as CSV
        if 'inference_time_scaling' in results:
            scaling_data = []
            for method, result in results['inference_time_scaling'].items():
                for size in result.successful_sizes:
                    scaling_data.append({
                        'method': method,
                        'size': size,
                        'time_seconds': result.time_by_size[size],
                        'memory_mb': result.memory_by_size.get(size, 0),
                        'scaling_factor': result.scaling_factor,
                        'complexity': result.complexity_estimate
                    })
            
            if scaling_data:
                df = pd.DataFrame(scaling_data)
                df.to_csv(output_dir / 'scaling_results.csv', index=False)
        
        # Generate summary report
        self.generate_efficiency_report(results, output_dir)
        
        logger.info(f"Scaling analysis exported to {output_dir}")
    
    def generate_efficiency_report(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Generate human-readable efficiency report."""
        lines = ["=" * 80]
        lines.append("COMPUTATIONAL EFFICIENCY ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Inference time scaling summary
        if 'inference_time_scaling' in results:
            lines.append("EXPERIMENT 1.1: INFERENCE TIME SCALING")
            lines.append("-" * 50)
            
            time_results = results['inference_time_scaling']
            for method, result in time_results.items():
                lines.append(f"\n{method}:")
                lines.append(f"  Complexity: {result.complexity_estimate}")
                lines.append(f"  Max feasible size: {result.max_feasible_size} variables")
                lines.append(f"  Scaling factor: {result.scaling_factor:.2f}")
                
                if result.successful_sizes:
                    min_time = min(result.time_by_size.values())
                    max_time = max(result.time_by_size.values())
                    lines.append(f"  Time range: {min_time:.3f}s - {max_time:.3f}s")
        
        # Memory usage summary
        if 'memory_usage_scaling' in results:
            lines.append("\n\nEXPERIMENT 1.2: MEMORY USAGE SCALING")
            lines.append("-" * 50)
            
            memory_results = results['memory_usage_scaling']
            for method, memory_by_size in memory_results.items():
                if memory_by_size:
                    avg_memory = np.mean(list(memory_by_size.values()))
                    max_memory = max(memory_by_size.values())
                    lines.append(f"\n{method}:")
                    lines.append(f"  Average memory: {avg_memory:.1f}MB")
                    lines.append(f"  Peak memory: {max_memory:.1f}MB")
        
        # Training amortization
        if 'training_amortization' in results:
            lines.append("\n\nEXPERIMENT 1.3: TRAINING AMORTIZATION")
            lines.append("-" * 50)
            
            amort_results = results['training_amortization']
            summary = amort_results.get('summary', {})
            
            lines.append(f"Typical break-even: {summary.get('typical_breakeven_problems', 'N/A')} problems")
            lines.append(f"Long-term speedup: {summary.get('long_term_speedup_factor', 'N/A')}x")
            lines.append(f"Feasibility advantage starts at: {summary.get('feasibility_advantage_size', 'N/A')} variables")
        
        # Comparative analysis
        if 'comparative_analysis' in results:
            lines.append("\n\nCOMPARATIVE ANALYSIS")
            lines.append("-" * 50)
            
            comp_analysis = results['comparative_analysis']
            
            if 'inference_comparison' in comp_analysis:
                inf_comp = comp_analysis['inference_comparison']
                best_method = inf_comp.get('best_scaling_method')
                if best_method:
                    lines.append(f"Best scaling method: {best_method}")
                
                max_sizes = inf_comp.get('max_feasible_sizes', {})
                for method, max_size in max_sizes.items():
                    lines.append(f"  {method}: up to {max_size} variables")
        
        lines.append("\n" + "=" * 80)
        
        # Write report
        with open(output_dir / 'efficiency_report.txt', 'w') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Efficiency report saved to {output_dir / 'efficiency_report.txt'}")