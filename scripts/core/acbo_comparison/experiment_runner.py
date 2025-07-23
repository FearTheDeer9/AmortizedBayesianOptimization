"""
Main Experiment Runner for ACBO Comparison Framework

This module provides the main orchestrator for running ACBO comparison experiments.
It coordinates all components and provides a clean, simple interface for experiments.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from omegaconf import DictConfig, OmegaConf

from .method_registry import MethodRegistry, MethodResult
from .scm_manager import SCMManager
from .metrics_collector import MetricsCollector
from .wandb_integration import create_wandb_logger
from .statistical_analysis import StatisticalAnalyzer
from .visualization import VisualizationManager

logger = logging.getLogger(__name__)


class ACBOExperimentRunner:
    """Main orchestrator for ACBO comparison experiments."""
    
    def __init__(self, config: DictConfig):
        """
        Initialize experiment runner.
        
        Args:
            config: Complete experiment configuration
        """
        self.config = config
        self.start_time = time.time()
        
        # Initialize all components
        self.method_registry = MethodRegistry()
        self.scm_manager = SCMManager(config)
        self.metrics_collector = MetricsCollector()
        self.wandb_logger = create_wandb_logger(
            config.get('logging', {}).get('wandb', {}),
            config.experiment.name
        )
        self.statistical_analyzer = StatisticalAnalyzer(
            significance_level=config.get('analysis', {}).get('significance_level', 0.05)
        )
        self.visualization_manager = VisualizationManager()
        
        # Checkpoint settings
        self.checkpoint_enabled = config.get('checkpoint', {}).get('enabled', True)
        self.checkpoint_dir = Path(config.get('checkpoint', {}).get('dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / f"acbo_checkpoint_{int(time.time())}.json"
        
        logger.info(f"Initialized ACBO experiment runner: {config.experiment.name}")
    
    def run_experiment(self) -> Dict[str, Any]:
        """
        Run complete ACBO comparison experiment.
        
        Returns:
            Comprehensive experiment results
        """
        logger.info("🚀 Starting ACBO comparison experiment")
        
        try:
            # Initialize WandB logging
            wandb_initialized = self.wandb_logger.initialize_run(
                OmegaConf.to_container(self.config, resolve=True)
            )
            
            # Generate test SCMs
            test_scms = self._generate_test_scms()
            
            # Run experiments for all methods
            method_results = self._run_all_methods(test_scms)
            
            # Collect and analyze results
            analysis_results = self._analyze_results(method_results)
            
            # Generate visualizations
            plots = self._create_visualizations(method_results, analysis_results)
            
            # Log everything to WandB
            if wandb_initialized:
                self._log_to_wandb(method_results, analysis_results, plots)
            
            # Compile final results
            final_results = self._compile_final_results(
                method_results, analysis_results, plots, test_scms
            )
            
            total_time = time.time() - self.start_time
            logger.info(f"✅ Experiment completed successfully in {total_time:.1f}s")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Experiment failed: {e}")
            if self.wandb_logger.enabled and self.wandb_logger.run is not None:
                self.wandb_logger.run.log({"experiment_status": "failed", "error": str(e)})
            raise
        finally:
            self.wandb_logger.finish_run()
    
    def _generate_test_scms(self) -> List[tuple]:
        """Generate test SCMs based on configuration."""
        logger.info("Generating test SCMs...")
        
        # Try to load from cache first
        cached_scms = self.scm_manager.load_cached_scms()
        if cached_scms:
            test_scms = cached_scms
        else:
            # Generate new SCMs
            test_scms = self.scm_manager.generate_test_scms()
            # Cache for future use
            self.scm_manager.cache_scms(test_scms)
        
        # Log SCM metadata to WandB
        for scm_name, scm in test_scms:
            scm_info = self.scm_manager.get_scm_metadata(scm_name)
            self.wandb_logger.log_scm_metadata(scm_name, scm_info)
        
        logger.info(f"Generated {len(test_scms)} test SCMs")
        return test_scms
    
    def _run_all_methods(self, test_scms: List[tuple]) -> Dict[str, List[MethodResult]]:
        """Run all configured methods on all SCMs."""
        experiment_config = self.config.experiment
        methods_to_run = experiment_config.methods
        n_runs = experiment_config.get('runs_per_method', 3)
        
        logger.info(f"Running {len(methods_to_run)} methods × {n_runs} runs × {len(test_scms)} SCMs")
        
        all_results = {}
        
        for method_display_name, method_type in methods_to_run.items():
            logger.info(f"\n=== Running {method_display_name} ===")
            
            method_results = []
            
            for run_idx in range(n_runs):
                for scm_idx, (scm_name, scm) in enumerate(test_scms):
                    logger.info(f"  Run {run_idx + 1}/{n_runs}, SCM '{scm_name}'")
                    
                    # Run single method
                    result = self.method_registry.run_method(
                        method_type, scm, self.config, run_idx, scm_idx
                    )
                    
                    # Add SCM context
                    if result.success:
                        result.metadata['scm_name'] = scm_name
                        result.metadata['scm_type'] = self.scm_manager.get_scm_metadata(scm_name).get('structure_type', 'unknown')
                    
                    method_results.append(result)
                    
                    # Collect metrics for this run
                    if result.success:
                        # Pass the full result including detailed_results
                        result_dict = result.__dict__.copy()
                        # Ensure detailed_results are included at top level for backward compatibility
                        if 'detailed_results' in result_dict and isinstance(result_dict['detailed_results'], dict):
                            # Merge detailed_results into the main result dict for metrics collection
                            result_with_details = {**result_dict, **result_dict['detailed_results']}
                        else:
                            result_with_details = result_dict
                        
                        run_metrics = self.metrics_collector.collect_run_metrics(
                            result_with_details, scm, method_display_name, run_idx, scm_idx
                        )
                        
                        # Log to WandB
                        self.wandb_logger.log_run_metrics(
                            run_metrics, method_display_name, run_idx, scm_name
                        )
            
            all_results[method_display_name] = method_results
            
            # Save checkpoint after each method completes
            if self.checkpoint_enabled:
                self._save_checkpoint(all_results)
            
            # Log method summary
            # Convert results to dicts, preserving detailed_results
            successful_results = []
            for r in method_results:
                if r.success:
                    result_dict = r.__dict__.copy()
                    # Ensure detailed_results are preserved
                    if 'detailed_results' in result_dict and isinstance(result_dict['detailed_results'], dict):
                        # Also merge into top-level for backward compatibility
                        result_with_trajectory = {**result_dict, **result_dict['detailed_results']}
                        successful_results.append(result_with_trajectory)
                    else:
                        successful_results.append(result_dict)
            
            aggregated_metrics = self.metrics_collector.aggregate_method_metrics(
                method_display_name, successful_results
            )
            
            # Store aggregated metrics in collector for later retrieval
            self.metrics_collector.collected_metrics[method_display_name] = aggregated_metrics
            
            self.wandb_logger.log_method_summary(method_display_name, aggregated_metrics)
            
            success_rate = len(successful_results) / len(method_results) * 100
            logger.info(f"=== {method_display_name} completed: {success_rate:.1f}% success rate ===")
        
        return all_results
    
    def _analyze_results(self, method_results: Dict[str, List[MethodResult]]) -> Dict[str, Any]:
        """Perform statistical analysis on results."""
        logger.info("Performing statistical analysis...")
        
        # Convert to format expected by analyzer
        analysis_input = {}
        for method_name, results in method_results.items():
            analysis_input[method_name] = [r.__dict__ for r in results if r.success]
        
        # Perform analyses
        analysis_results = {
            'summary_statistics': self.statistical_analyzer.compute_summary_statistics(analysis_input),
            'pairwise_comparisons': self.statistical_analyzer.compare_all_methods(analysis_input),
            'anova_results': self.statistical_analyzer.perform_anova(analysis_input),
            'effect_sizes': self.statistical_analyzer.compute_effect_sizes(analysis_input)
        }
        
        # Log comparison results to WandB
        self.wandb_logger.log_comparison_results(analysis_results['pairwise_comparisons'])
        
        logger.info("Statistical analysis completed")
        return analysis_results
    
    def _create_visualizations(self, method_results: Dict[str, List[MethodResult]], 
                             analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate visualization plots."""
        logger.info("Generating visualizations...")
        
        # Prepare data for visualization
        viz_data = {}
        for method_name, results in method_results.items():
            successful_results = [r for r in results if r.success]
            if successful_results:
                viz_data[method_name] = [r.__dict__ for r in successful_results]
        
        # Generate plots
        plots = self.visualization_manager.create_all_plots(
            viz_data, 
            analysis_results,
            self.config.experiment.name
        )
        
        logger.info(f"Generated {len(plots)} visualization plots")
        return plots
    
    def _log_to_wandb(self, method_results: Dict[str, List[MethodResult]], 
                     analysis_results: Dict[str, Any], plots: Dict[str, str]) -> None:
        """Log all results to WandB."""
        if not self.wandb_logger.enabled:
            return
        
        logger.info("Logging results to WandB...")
        
        # Log experiment summary
        experiment_summary = {
            'method_results': {name: [r.__dict__ for r in results] for name, results in method_results.items()},
            'scms_tested': len(self.scm_manager.scms),
            'runs_per_method': self.config.experiment.get('runs_per_method', 3)
        }
        self.wandb_logger.log_experiment_summary(experiment_summary)
        
        # Log plots as artifacts
        self.wandb_logger.log_plots_as_artifacts(plots)
    
    def _compile_final_results(self, method_results: Dict[str, List[MethodResult]], 
                              analysis_results: Dict[str, Any], plots: Dict[str, str], 
                              test_scms: List[tuple]) -> Dict[str, Any]:
        """Compile comprehensive final results."""
        
        # Convert method results preserving detailed_results with trajectory data
        method_results_dict = {}
        for name, results in method_results.items():
            method_results_dict[name] = []
            for r in results:
                result_dict = r.__dict__.copy()
                # Ensure detailed_results with trajectory data are preserved
                if 'detailed_results' in result_dict and isinstance(result_dict['detailed_results'], dict):
                    # Keep trajectory data in detailed_results
                    pass
                method_results_dict[name].append(result_dict)
        
        # Extract aggregated trajectory data from metrics collector
        aggregated_trajectories = {}
        for method_name in method_results.keys():
            if method_name in self.metrics_collector.collected_metrics:
                method_metrics = self.metrics_collector.collected_metrics[method_name]
                if isinstance(method_metrics, dict) and any(key.endswith('_mean') for key in method_metrics):
                    aggregated_trajectories[method_name] = {
                        k: v for k, v in method_metrics.items() 
                        if k.endswith('_mean') or k.endswith('_trajectory')
                    }
        
        return {
            'experiment_config': OmegaConf.to_container(self.config),
            'method_results': method_results_dict,
            'statistical_analysis': analysis_results,
            'visualizations': plots,
            'scm_summary': self.scm_manager.get_scm_characteristics_summary(),
            'execution_metadata': {
                'total_time': time.time() - self.start_time,
                'scms_tested': len(test_scms),
                'methods_tested': len(method_results),
                'runs_per_method': self.config.experiment.get('runs_per_method', 3),
                'total_experiments': sum(len(results) for results in method_results.values())
            },
            'aggregated_trajectories': aggregated_trajectories,
            'trajectory_data': self._serialize_trajectory_data()  # Raw trajectory data
        }
    
    def _serialize_trajectory_data(self) -> Dict[str, Any]:
        """Serialize trajectory data to ensure JSON compatibility."""
        serialized = {}
        
        for key, value in self.metrics_collector.trajectory_data.items():
            # Ensure key is a string
            str_key = str(key) if not isinstance(key, str) else key
            
            # Recursively serialize the value
            serialized[str_key] = self._serialize_value(value)
        
        return serialized
    
    def _serialize_value(self, value: Any) -> Any:
        """Recursively serialize values for JSON compatibility."""
        import jax.numpy as jnp
        import numpy as onp
        
        if isinstance(value, dict):
            return {str(k): self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        elif isinstance(value, (jnp.ndarray, onp.ndarray)):
            return value.tolist()
        elif isinstance(value, (int, float, str, bool, type(None))):
            return value
        else:
            # Convert any other type to string
            return str(value)
    
    def _save_checkpoint(self, partial_results: Dict[str, Any]) -> None:
        """Save intermediate results as checkpoint."""
        try:
            import json
            import tempfile
            import shutil
            from dataclasses import is_dataclass, asdict
            
            # Prepare checkpoint data
            checkpoint_data = {
                'timestamp': time.time(),
                'elapsed_time': time.time() - self.start_time,
                'partial_results': self._serialize_for_checkpoint(partial_results),
                'completed_methods': list(partial_results.keys())
            }
            
            # Atomic write
            with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json',
                                           dir=self.checkpoint_dir) as temp_file:
                json.dump(checkpoint_data, temp_file, indent=2, default=str)
                temp_file.flush()
                temp_path = temp_file.name
            
            shutil.move(temp_path, self.checkpoint_file)
            logger.debug(f"Checkpoint saved: {self.checkpoint_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def _serialize_for_checkpoint(self, obj: Any) -> Any:
        """Serialize object for checkpoint saving."""
        from dataclasses import is_dataclass, asdict
        import jax.numpy as jnp
        import numpy as onp
        
        if isinstance(obj, dict):
            return {str(k): self._serialize_for_checkpoint(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_for_checkpoint(item) for item in obj]
        elif isinstance(obj, (jnp.ndarray, onp.ndarray)):
            return obj.tolist()
        elif is_dataclass(obj):
            return self._serialize_for_checkpoint(asdict(obj))
        elif hasattr(obj, '__dict__'):
            return self._serialize_for_checkpoint(obj.__dict__)
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)