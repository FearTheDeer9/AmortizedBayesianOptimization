"""
Benchmark Runner for Causal Discovery and Intervention Optimization.

This module provides a utility class for running multiple benchmarks and aggregating results,
making it easy to compare different methods across various tasks and metrics.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import torch

from causal_meta.meta_learning.benchmark import Benchmark, CausalDiscoveryBenchmark, CBOBenchmark

# Configure logging
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Utility class for running multiple benchmarks and aggregating results.
    
    This class makes it easy to:
    1. Register multiple benchmarks with different configurations
    2. Register multiple methods to be evaluated
    3. Run all benchmarks and collect results
    4. Generate summary reports and visualizations
    """
    
    def __init__(
        self,
        name: str = "benchmark_run",
        output_dir: str = "benchmark_results",
        seed: Optional[int] = None
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            name: Name of this benchmark run
            output_dir: Directory to save results
            seed: Random seed for reproducibility
        """
        self.name = name
        self.output_dir = output_dir
        self.seed = seed
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create run-specific output directory
        self.run_dir = os.path.join(output_dir, f"{name}_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize containers
        self.benchmarks = {}
        self.models = {}
        self.baselines = {}
        self.results = {}
    
    def add_benchmark(self, benchmark: Benchmark) -> None:
        """
        Add a benchmark to be run.
        
        Args:
            benchmark: Benchmark instance to run
        """
        self.benchmarks[benchmark.name] = benchmark
    
    def add_model(self, name: str, model: Any) -> None:
        """
        Add a model to be evaluated in all benchmarks.
        
        Args:
            name: Unique identifier for the model
            model: Model instance to evaluate
        """
        self.models[name] = model
    
    def register_model(self, name: str, model: Any) -> None:
        """
        Register a model to be evaluated in all benchmarks.
        
        This is an alias for add_model for backward compatibility.
        
        Args:
            name: Unique identifier for the model
            model: Model instance to evaluate
        """
        return self.add_model(name, model)
    
    def add_baseline(self, name: str, baseline: Any) -> None:
        """
        Add a baseline method for comparison in all benchmarks.
        
        Args:
            name: Unique identifier for the baseline
            baseline: Baseline method (function or class instance)
        """
        self.baselines[name] = baseline
    
    def run_all(self) -> Dict[str, Any]:
        """
        Run all registered benchmarks with all registered methods.
        
        Returns:
            Dictionary containing results from all benchmarks
        """
        logger.info(f"Starting benchmark run: {self.name}")
        
        # Initialize results storage
        self.results = {
            "run_info": {
                "name": self.name,
                "timestamp": self.timestamp,
                "seed": self.seed
            },
            "benchmarks": {}
        }
        
        # Run each benchmark
        for benchmark_name, benchmark in self.benchmarks.items():
            logger.info(f"Running benchmark: {benchmark_name}")
            
            # Add models and baselines to the benchmark
            for model_name, model in self.models.items():
                benchmark.add_model(model_name, model)
            
            for baseline_name, baseline in self.baselines.items():
                benchmark.add_baseline(baseline_name, baseline)
            
            # Set up the benchmark
            benchmark.setup()
            
            # Run the benchmark
            benchmark_results = benchmark.run()
            
            # Store results
            self.results["benchmarks"][benchmark_name] = benchmark_results
        
        # Save combined results
        self._save_combined_results()
        
        logger.info(f"Benchmark run complete: {self.name}")
        
        return self.results
    
    def _save_combined_results(self) -> str:
        """
        Save combined results from all benchmarks to a file.
        
        Returns:
            Path to the saved file
        """
        # Clean results for saving
        cleaned_results = self._clean_for_json(self.results)
        
        # Save to file
        path = os.path.join(self.run_dir, "combined_results.json")
        with open(path, "w") as f:
            json.dump(cleaned_results, f, indent=2)
        
        return path
    
    def _clean_for_json(self, obj: Any) -> Any:
        """
        Clean an object for JSON serialization.
        
        Args:
            obj: Object to clean
            
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, (np.ndarray, np.number)):
            return obj.tolist()
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # Try to convert to string
            try:
                return str(obj)
            except:
                return f"<Object of type {type(obj).__name__}>"
    
    def generate_summary_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a summary report of benchmark results.
        
        Args:
            output_path: Path to save the report (defaults to run directory)
            
        Returns:
            Path to the saved report
        """
        if not hasattr(self, "results") or not self.results:
            logger.warning("No results available. Run the benchmarks first.")
            return None
        
        # Default output path
        if output_path is None:
            output_path = os.path.join(self.run_dir, "summary_report.md")
        
        # Create report content
        report = [
            f"# Benchmark Summary Report: {self.name}",
            f"**Run Date:** {self.timestamp}",
            f"**Seed:** {self.seed}",
            "\n## Overview",
            f"- Number of benchmarks: {len(self.results.get('benchmarks', {}))}",
            f"- Models evaluated: {', '.join(self.models.keys())}",
            f"- Baselines evaluated: {', '.join(self.baselines.keys())}",
            "\n## Benchmark Results"
        ]
        
        # Add results for each benchmark
        for benchmark_name, benchmark_results in self.results.get("benchmarks", {}).items():
            report.append(f"\n### {benchmark_name}")
            
            # Add benchmark configuration
            if "benchmark_config" in benchmark_results:
                report.append("\n**Configuration:**")
                for config_key, config_value in benchmark_results["benchmark_config"].items():
                    report.append(f"- {config_key}: {config_value}")
            
            # Add aggregated results
            if "aggregated" in benchmark_results:
                # Models
                if "models" in benchmark_results["aggregated"]:
                    report.append("\n**Model Results:**")
                    report.append("\n| Model | " + " | ".join(benchmark_results["aggregated"]["models"].get(list(benchmark_results["aggregated"]["models"].keys())[0], {}).keys()) + " |")
                    report.append("| --- | " + " | ".join(["---"] * len(benchmark_results["aggregated"]["models"].get(list(benchmark_results["aggregated"]["models"].keys())[0], {}))) + " |")
                    
                    for model_name, model_results in benchmark_results["aggregated"]["models"].items():
                        row = [model_name]
                        for metric_name, metric_value in model_results.items():
                            if isinstance(metric_value, dict) and "mean" in metric_value:
                                row.append(f"{metric_value['mean']:.4f} ± {metric_value['std']:.4f}")
                            else:
                                row.append(str(metric_value))
                        report.append("| " + " | ".join(row) + " |")
                
                # Baselines
                if "baselines" in benchmark_results["aggregated"] and benchmark_results["aggregated"]["baselines"]:
                    report.append("\n**Baseline Results:**")
                    report.append("\n| Baseline | " + " | ".join(benchmark_results["aggregated"]["baselines"].get(list(benchmark_results["aggregated"]["baselines"].keys())[0], {}).keys()) + " |")
                    report.append("| --- | " + " | ".join(["---"] * len(benchmark_results["aggregated"]["baselines"].get(list(benchmark_results["aggregated"]["baselines"].keys())[0], {}))) + " |")
                    
                    for baseline_name, baseline_results in benchmark_results["aggregated"]["baselines"].items():
                        row = [baseline_name]
                        for metric_name, metric_value in baseline_results.items():
                            if isinstance(metric_value, dict) and "mean" in metric_value:
                                row.append(f"{metric_value['mean']:.4f} ± {metric_value['std']:.4f}")
                            else:
                                row.append(str(metric_value))
                        report.append("| " + " | ".join(row) + " |")
        
        # Write report to file
        with open(output_path, "w") as f:
            f.write("\n".join(report))
        
        logger.info(f"Summary report saved to: {output_path}")
        
        return output_path
    
    def get_best_models(self) -> Dict[str, Dict[str, str]]:
        """
        Get the best performing models for each benchmark and metric.
        
        Returns:
            Dictionary mapping benchmark names to dictionaries mapping
            metric names to best model names.
        """
        if not hasattr(self, "results") or not self.results:
            logger.warning("No results available. Run the benchmarks first.")
            return {}
        
        best_models = {}
        
        for benchmark_name, benchmark_results in self.results.get("benchmarks", {}).items():
            # Skip if no aggregated results
            if "aggregated" not in benchmark_results:
                continue
            
            benchmark_best = {}
            aggregated = benchmark_results["aggregated"]
            
            # Combine models and baselines for comparison
            all_methods = {}
            if "models" in aggregated:
                all_methods.update(aggregated["models"])
            if "baselines" in aggregated:
                all_methods.update(aggregated["baselines"])
            
            # Find best method for each metric
            metrics_found = set()
            for method_name, method_results in all_methods.items():
                for metric_name, metric_value in method_results.items():
                    metrics_found.add(metric_name)
            
            for metric_name in metrics_found:
                best_method = None
                best_value = None
                
                for method_name, method_results in all_methods.items():
                    if metric_name not in method_results:
                        continue
                    
                    metric_value = method_results[metric_name]
                    
                    # Extract mean if value is a dictionary
                    if isinstance(metric_value, dict) and "mean" in metric_value:
                        metric_value = metric_value["mean"]
                    
                    # Skip if value is not numeric or None
                    if metric_value is None or not isinstance(metric_value, (int, float)):
                        continue
                    
                    # Update best if this is better
                    # For metrics like SHD, lower is better; for others like F1, higher is better
                    is_better = False
                    if best_value is None:
                        is_better = True
                    elif metric_name.lower() in ["shd", "runtime", "min_time", "max_time", "avg_time"]:
                        # For these metrics, lower is better
                        is_better = metric_value < best_value
                    else:
                        # For others (f1, precision, recall, etc.), higher is better
                        is_better = metric_value > best_value
                    
                    if is_better:
                        best_value = metric_value
                        best_method = method_name
                
                # Add to results if a best method was found
                if best_method is not None:
                    benchmark_best[metric_name] = best_method
            
            # Add benchmark results
            if benchmark_best:
                best_models[benchmark_name] = benchmark_best
        
        return best_models
    
    def generate_comparison_plots(self, output_dir: Optional[str] = None) -> List[str]:
        """
        Generate comparison plots for each benchmark.
        
        Args:
            output_dir: Directory to save plots (defaults to run directory)
            
        Returns:
            List of paths to saved plots
        """
        if not hasattr(self, "results") or not self.results:
            logger.warning("No results available. Run the benchmarks first.")
            return []
        
        # Default output directory
        if output_dir is None:
            output_dir = os.path.join(self.run_dir, "plots")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # List to track saved plot paths
        saved_plots = []
        
        # Generate plots for each benchmark
        for benchmark_name, benchmark in self.benchmarks.items():
            # Skip if this benchmark doesn't have results
            if benchmark_name not in self.results.get("benchmarks", {}):
                continue
            
            # Generate plot
            try:
                fig = benchmark.plot_results(
                    title=f"{benchmark_name} Results",
                    save_path=None  # Don't save yet
                )
                
                if fig:
                    # Save plot
                    plot_path = os.path.join(output_dir, f"{benchmark_name}_results.png")
                    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    
                    saved_plots.append(plot_path)
                    logger.info(f"Plot saved to: {plot_path}")
            except Exception as e:
                logger.error(f"Error generating plot for {benchmark_name}: {str(e)}")
        
        return saved_plots
    
    @classmethod
    def create_standard_suite(
        cls,
        name: str = "standard_benchmark_suite",
        output_dir: str = "benchmark_results",
        seed: Optional[int] = None,
        device: str = "cpu",
        graph_sizes: List[int] = [5, 10, 20],
        num_graphs: int = 10,
        num_samples: int = 1000
    ) -> "BenchmarkRunner":
        """
        Create a standard benchmark suite with typical configurations.
        
        Args:
            name: Name of the benchmark run
            output_dir: Directory to save results
            seed: Random seed for reproducibility
            device: Device to run on
            graph_sizes: List of graph sizes to benchmark
            num_graphs: Number of graphs per size
            num_samples: Number of samples per graph
            
        Returns:
            Configured BenchmarkRunner instance
        """
        # Create runner
        runner = cls(name, output_dir, seed)
        
        # Add causal discovery benchmarks for different graph sizes
        for size in graph_sizes:
            # ER graphs
            er_benchmark = CausalDiscoveryBenchmark(
                name=f"causal_discovery_er_{size}",
                output_dir=os.path.join(runner.run_dir, "benchmarks"),
                seed=seed,
                device=device,
                num_nodes=size,
                num_graphs=num_graphs,
                num_samples=num_samples,
                graph_type="random",
                edge_prob=3.0 / size  # Keep average degree around 3
            )
            runner.add_benchmark(er_benchmark)
            
            # Scale-free graphs
            sf_benchmark = CausalDiscoveryBenchmark(
                name=f"causal_discovery_sf_{size}",
                output_dir=os.path.join(runner.run_dir, "benchmarks"),
                seed=seed,
                device=device,
                num_nodes=size,
                num_graphs=num_graphs,
                num_samples=num_samples,
                graph_type="scale_free"
            )
            runner.add_benchmark(sf_benchmark)
        
        # Add CBO benchmarks for different graph sizes
        for size in graph_sizes:
            # ER graphs for CBO
            er_cbo_benchmark = CBOBenchmark(
                name=f"cbo_er_{size}",
                output_dir=os.path.join(runner.run_dir, "benchmarks"),
                seed=seed,
                device=device,
                num_nodes=size,
                num_graphs=num_graphs,
                num_samples=num_samples,
                graph_type="random",
                edge_prob=3.0 / size,  # Keep average degree around 3
                num_interventions=max(2, size // 5),
                intervention_budget=10
            )
            runner.add_benchmark(er_cbo_benchmark)
        
        return runner
    
    def create_scalability_suite(
        self,
        min_nodes: int = 5,
        max_nodes: int = 50,
        step_size: int = 5,
        num_graphs_per_size: int = 3,
        num_samples: int = 1000,
        graph_type: str = "erdos_renyi",
        seed: Optional[int] = None,
        measure_mode: str = "both",
        device: Union[str, torch.device] = "cpu",
        models: Optional[Dict[str, Any]] = None,
        baselines: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Create a benchmark suite for evaluating scalability with respect to graph size.
        
        This creates a single ScalabilityBenchmark that tests methods on graphs 
        of increasing size and measures performance metrics.
        
        Args:
            min_nodes: Minimum number of nodes in test graphs
            max_nodes: Maximum number of nodes in test graphs
            step_size: Increment between graph sizes
            num_graphs_per_size: Number of test graphs to generate for each size
            num_samples: Number of samples for each test graph
            graph_type: Type of graphs to generate ("erdos_renyi", "scale_free", etc.)
            seed: Random seed for reproducibility
            measure_mode: What to measure ("discovery", "cbo", or "both")
            device: Device to run benchmarks on
            models: Dictionary of models to evaluate
            baselines: Dictionary of baseline methods to evaluate
            
        Returns:
            List of benchmark IDs (only one in this case)
        """
        # Create timestamp for this benchmark suite
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create benchmark name
        benchmark_name = f"scalability_{graph_type}_{min_nodes}_to_{max_nodes}"
        
        # If models or baselines are provided, register them
        if models:
            for name, model in models.items():
                self.register_model(name, model)
                
        if baselines:
            for name, baseline in baselines.items():
                self.register_baseline(name, baseline)
        
        # Create the scalability benchmark
        from causal_meta.meta_learning.benchmark import ScalabilityBenchmark
        
        benchmark = ScalabilityBenchmark(
            name=benchmark_name,
            output_dir=self.output_dir,
            seed=seed,
            device=device,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            step_size=step_size,
            num_graphs_per_size=num_graphs_per_size,
            num_samples=num_samples,
            graph_type=graph_type,
            measure_mode=measure_mode
        )
        
        # Register the benchmark
        benchmark_id = f"{benchmark_name}_{timestamp}"
        self.benchmarks[benchmark_id] = benchmark
        
        # Set up the benchmark
        benchmark.setup()
        
        # Register models and baselines with the benchmark
        for name, model in self.models.items():
            benchmark.add_model(name, model)
        
        for name, baseline in self.baselines.items():
            benchmark.add_baseline(name, baseline)
        
        return [benchmark_id]
    
    def run_scalability_analysis(
        self,
        benchmark_id: str,
        plot_metrics: bool = True,
        generate_report: bool = True,
        save_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Run a scalability benchmark and analyze the results.
        
        Args:
            benchmark_id: ID of the scalability benchmark to run
            plot_metrics: Whether to generate and display scaling plots
            generate_report: Whether to generate a comprehensive report
            save_plots: Whether to save plots to files
            
        Returns:
            Dictionary with benchmark results including scaling analysis
        """
        # Ensure benchmark exists
        if benchmark_id not in self.benchmarks:
            raise ValueError(f"Benchmark {benchmark_id} not found")
        
        # Get the benchmark
        benchmark = self.benchmarks[benchmark_id]
        
        # Ensure it's a ScalabilityBenchmark
        from causal_meta.meta_learning.benchmark import ScalabilityBenchmark
        if not isinstance(benchmark, ScalabilityBenchmark):
            raise TypeError(f"Benchmark {benchmark_id} is not a ScalabilityBenchmark")
        
        # Run the benchmark
        logger.info(f"Running scalability benchmark: {benchmark_id}")
        results = benchmark.run()
        
        # Analyze scaling behavior
        logger.info(f"Analyzing scaling behavior for benchmark: {benchmark_id}")
        scaling_analysis = benchmark.analyze_scaling()
        
        # Generate plots if requested
        if plot_metrics:
            metrics = ["runtime", "cpu_memory", "accuracy"]
            for metric in metrics:
                logger.info(f"Generating plot for {metric} scaling")
                if save_plots:
                    plot_path = os.path.join(benchmark.benchmark_dir, f"{metric}_scaling.png")
                    benchmark.plot_scaling_curves(metric=metric, save_path=plot_path)
                else:
                    benchmark.plot_scaling_curves(metric=metric)
        
        # Generate report if requested
        if generate_report:
            logger.info(f"Generating scalability report for benchmark: {benchmark_id}")
            report_path = os.path.join(benchmark.benchmark_dir, "scalability_report.json")
            report = benchmark.generate_scaling_report(output_path=report_path)
            results["scaling_report"] = report
        
        return results 