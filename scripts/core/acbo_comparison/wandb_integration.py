"""
WandB Integration for ACBO Comparison Framework

This module handles all WandB logging, metric tracking, and artifact management
for ACBO experiments. It provides a clean interface for experiment tracking.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import time

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class WandBLogger:
    """Handler for WandB experiment logging and tracking."""
    
    def __init__(self, config: Dict[str, Any], experiment_name: str = "acbo_comparison"):
        """
        Initialize WandB logger.
        
        Args:
            config: WandB configuration
            experiment_name: Name for the experiment
        """
        self.config = config
        self.experiment_name = experiment_name
        self.run = None
        self.enabled = config.get('enabled', False) and WANDB_AVAILABLE
        
        if not WANDB_AVAILABLE and config.get('enabled', False):
            logger.warning("WandB not available but logging enabled in config")
            self.enabled = False
    
    def initialize_run(self, full_config: Dict[str, Any]) -> bool:
        """
        Initialize WandB run.
        
        Args:
            full_config: Full experiment configuration
            
        Returns:
            True if initialization successful
        """
        if not self.enabled:
            logger.info("WandB logging disabled")
            return False
        
        try:
            # Create run name with timestamp
            run_name = f"{self.experiment_name}_{int(time.time())}"
            
            # Initialize run
            self.run = wandb.init(
                project=self.config.get('project', 'causal_bayes_opt'),
                entity=self.config.get('entity'),
                config=full_config,
                tags=self.config.get('tags', []) + ["acbo_comparison", "causal_discovery"],
                group=self.config.get('group', 'acbo_experiments'),
                name=run_name
            )
            
            # Define custom metrics for proper visualization
            self._define_custom_metrics()
            
            logger.info(f"WandB initialized: {wandb.run.url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")
            self.enabled = False
            return False
    
    def log_scm_metadata(self, scm_name: str, scm_info: Dict[str, Any]) -> None:
        """Log SCM characteristics to WandB."""
        if not self.enabled or not self.run:
            return
        
        try:
            scm_metrics = {
                f"scm/{scm_name}/num_variables": scm_info.get('num_variables', 0),
                f"scm/{scm_name}/num_edges": scm_info.get('num_edges', 0),
                f"scm/{scm_name}/edge_density": scm_info.get('edge_density', 0.0),
                f"scm/{scm_name}/target_variable": scm_info.get('target_variable', 'unknown'),
                f"scm/{scm_name}/structure_type": scm_info.get('structure_type', 'unknown'),
            }
            
            # Log coefficients as formatted string
            coefficients = scm_info.get('coefficients', {})
            if coefficients:
                coeff_str = ', '.join([f"{k}:{v:.2f}" for k, v in coefficients.items()])
                scm_metrics[f"scm/{scm_name}/coefficients"] = coeff_str
            
            self.run.log(scm_metrics)
            logger.debug(f"Logged metadata for SCM '{scm_name}'")
            
        except Exception as e:
            logger.warning(f"Failed to log SCM metadata for '{scm_name}': {e}")
    
    def log_run_metrics(self, metrics: Dict[str, Any], method_name: str, 
                       run_idx: int, scm_name: str) -> None:
        """
        Log metrics from a single experimental run.
        
        Args:
            metrics: Collected metrics from the run
            method_name: Name of the experimental method
            run_idx: Run index
            scm_name: Name of the SCM used
        """
        if not self.enabled or not self.run:
            return
        
        try:
            # Log trajectory data with custom x-axis (intervention steps)
            trajectory_metrics = self._extract_trajectory_metrics(metrics)
            
            for step_idx, step_metrics in enumerate(trajectory_metrics):
                step_data = {
                    # Main metrics for easy dashboard creation
                    f"{method_name}/target_value": step_metrics.get('target_value', 0.0),
                    f"{method_name}/true_parent_likelihood": step_metrics.get('true_parent_likelihood', 0.0),
                    f"{method_name}/f1_score": step_metrics.get('f1_score', 0.0),
                    f"{method_name}/shd": step_metrics.get('shd', 0.0),
                    f"{method_name}/uncertainty": step_metrics.get('uncertainty', 0.0),
                    
                    # P(Parents|Data) - clearer naming
                    f"{method_name}/parent_probability": step_metrics.get('true_parent_likelihood', 0.0),
                    
                    # Structure recovery metrics
                    f"{method_name}/structure_f1": step_metrics.get('f1_score', 0.0),
                    f"{method_name}/structure_distance": step_metrics.get('shd', 0.0),
                    
                    # SCM-specific metrics for detailed analysis
                    f"{method_name}/target_value_{scm_name}": step_metrics.get('target_value', 0.0),
                    f"{method_name}/f1_score_{scm_name}": step_metrics.get('f1_score', 0.0),
                    f"{method_name}/parent_prob_{scm_name}": step_metrics.get('true_parent_likelihood', 0.0),
                    
                    # Context information
                    "intervention_step": step_idx,
                    "run_idx": run_idx,
                    "scm_name": scm_name,
                    "method": method_name
                }
                
                self.run.log(step_data)
            
            # Log summary metrics for this run
            self._log_run_summary(metrics, method_name, run_idx)
            
        except Exception as e:
            logger.warning(f"Failed to log run metrics for {method_name}: {e}")
    
    def log_method_summary(self, method_name: str, aggregated_metrics: Dict[str, Any]) -> None:
        """Log aggregated method summary metrics."""
        if not self.enabled or not self.run:
            return
        
        try:
            summary_metrics = {}
            
            # Extract key summary statistics
            for metric_base in ['target_improvement', 'structure_accuracy', 'sample_efficiency', 'convergence_steps']:
                for stat in ['mean', 'std', 'min', 'max']:
                    key = f"{metric_base}_{stat}"
                    if key in aggregated_metrics:
                        summary_metrics[f"summary/{method_name}_{key}"] = aggregated_metrics[key]
            
            # Add run statistics
            summary_metrics.update({
                f"summary/{method_name}_total_runs": aggregated_metrics.get('total_runs', 0),
                f"summary/{method_name}_valid_runs": aggregated_metrics.get('valid_runs', 0),
                f"summary/{method_name}_success_rate": aggregated_metrics.get('success_rate', 0.0)
            })
            
            self.run.log(summary_metrics)
            logger.debug(f"Logged summary for {method_name}")
            
        except Exception as e:
            logger.warning(f"Failed to log method summary for {method_name}: {e}")
    
    def log_comparison_results(self, comparison_stats: Dict[str, Any]) -> None:
        """Log statistical comparison results."""
        if not self.enabled or not self.run:
            return
        
        try:
            comparison_metrics = {}
            
            for comparison_name, stats in comparison_stats.items():
                if isinstance(stats, dict):
                    comparison_metrics.update({
                        f"comparison/{comparison_name}_p_value": stats.get('p_value', 1.0),
                        f"comparison/{comparison_name}_significant": stats.get('significant', False),
                        f"comparison/{comparison_name}_effect_size": stats.get('effect_size', 0.0),
                        f"comparison/{comparison_name}_mean_diff": (
                            stats.get('method1_mean', 0.0) - stats.get('method2_mean', 0.0)
                        )
                    })
            
            if comparison_metrics:
                self.run.log(comparison_metrics)
            
        except Exception as e:
            logger.warning(f"Failed to log comparison results: {e}")
    
    def log_experiment_summary(self, experiment_results: Dict[str, Any]) -> None:
        """Log final experiment summary."""
        if not self.enabled or not self.run:
            return
        
        try:
            summary = {
                "experiment/total_methods": len(experiment_results.get('method_results', {})),
                "experiment/scms_tested": experiment_results.get('scms_tested', 0),
                "experiment/runs_per_method": experiment_results.get('runs_per_method', 0),
                "experiment/total_experiments": sum(
                    len(method_results) for method_results in 
                    experiment_results.get('method_results', {}).values()
                ),
                "experiment/status": "completed"
            }
            
            self.run.log(summary)
            
        except Exception as e:
            logger.warning(f"Failed to log experiment summary: {e}")
    
    def log_plots_as_artifacts(self, plots: Dict[str, str]) -> None:
        """Log generated plots as WandB artifacts."""
        if not self.enabled or not self.run:
            return
        
        for plot_name, plot_path in plots.items():
            try:
                plot_file = Path(plot_path)
                if plot_file.exists():
                    # Log as image
                    self.run.log({f"plots/{plot_name}": wandb.Image(str(plot_file))})
                    
                    # Save as artifact for archival
                    artifact = wandb.Artifact(f"{plot_name}_plot", type="plot")
                    artifact.add_file(str(plot_file))
                    self.run.log_artifact(artifact)
                    
                    logger.debug(f"Logged plot: {plot_name}")
                    
            except Exception as e:
                logger.warning(f"Failed to log plot {plot_name}: {e}")
    
    def finish_run(self) -> None:
        """Finish WandB run."""
        if self.enabled and self.run:
            try:
                wandb.finish()
                logger.info("WandB run finished")
            except Exception as e:
                logger.warning(f"Error finishing WandB run: {e}")
    
    def _define_custom_metrics(self) -> None:
        """Define custom metrics for proper time-series visualization."""
        if not self.run:
            return
        
        try:
            # Define metrics that use intervention_step as x-axis
            self.run.define_metric("intervention_step")
            
            # Common method types (will be expanded dynamically)
            method_types = [
                "Random Policy + Untrained Model",
                "Random Policy + Learning Model", 
                "Oracle Policy + Learning Model",
                "Learned Enriched Policy + Learning Model"
            ]
            
            for method in method_types:
                # Core optimization metrics
                self.run.define_metric(f"{method}/target_value", step_metric="intervention_step")
                
                # Structure learning metrics (main focus)
                self.run.define_metric(f"{method}/f1_score", step_metric="intervention_step")
                self.run.define_metric(f"{method}/structure_f1", step_metric="intervention_step")
                self.run.define_metric(f"{method}/true_parent_likelihood", step_metric="intervention_step")
                self.run.define_metric(f"{method}/parent_probability", step_metric="intervention_step")
                
                # Distance metrics
                self.run.define_metric(f"{method}/shd", step_metric="intervention_step")
                self.run.define_metric(f"{method}/structure_distance", step_metric="intervention_step")
                
                # Uncertainty metrics
                self.run.define_metric(f"{method}/uncertainty", step_metric="intervention_step")
            
            # Create custom charts for better visualization
            self._create_custom_charts()
            
        except Exception as e:
            logger.warning(f"Failed to define custom metrics: {e}")
    
    def _create_custom_charts(self) -> None:
        """Create custom chart configurations for better visualization."""
        if not self.run:
            return
        
        try:
            # Create F1 Score comparison chart
            f1_chart = {
                "title": "F1 Score by Step (Structure Recovery)",
                "x_axis": "intervention_step",
                "y_axis": "f1_score",
                "chart_type": "line",
                "series": [
                    "Random Policy + Untrained Model/f1_score",
                    "Random Policy + Learning Model/f1_score",
                    "Oracle Policy + Learning Model/f1_score",
                    "Learned Enriched Policy + Learning Model/f1_score"
                ]
            }
            
            # Create P(Parents|Data) comparison chart
            parent_prob_chart = {
                "title": "P(Parents|Data) by Step",
                "x_axis": "intervention_step",
                "y_axis": "parent_probability",
                "chart_type": "line",
                "series": [
                    "Random Policy + Untrained Model/parent_probability",
                    "Random Policy + Learning Model/parent_probability",
                    "Oracle Policy + Learning Model/parent_probability",
                    "Learned Enriched Policy + Learning Model/parent_probability"
                ]
            }
            
            # Log custom charts config
            self.run.log({
                "custom_charts/f1_score_comparison": f1_chart,
                "custom_charts/parent_probability_comparison": parent_prob_chart
            })
            
        except Exception as e:
            logger.warning(f"Failed to create custom charts: {e}")
    
    def _extract_trajectory_metrics(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract trajectory metrics for step-by-step logging."""
        trajectory_data = []
        
        # First check for detailed_results (from improved metrics collector)
        detailed_results = metrics.get('detailed_results', {})
        
        # Extract trajectory lists with multiple possible key formats
        # Check detailed_results first, then main metrics
        steps = (detailed_results.get('steps') or 
                metrics.get('steps_trajectory') or 
                metrics.get('steps', []))
        
        target_values = (detailed_results.get('target_progress') or
                        detailed_results.get('target_values') or
                        metrics.get('target_values_trajectory') or 
                        metrics.get('target_values') or
                        metrics.get('target_progress', []))
        
        f1_scores = (detailed_results.get('f1_scores') or
                    metrics.get('f1_scores_trajectory') or 
                    metrics.get('f1_scores', []))
        
        shd_values = (detailed_results.get('shd_values') or
                     metrics.get('shd_values_trajectory') or 
                     metrics.get('shd_values', []))
        
        true_parent_likelihoods = (detailed_results.get('true_parent_likelihood') or
                                  detailed_results.get('parent_probability') or
                                  metrics.get('true_parent_likelihood_trajectory') or
                                  metrics.get('true_parent_likelihood', []))
        
        uncertainties = (detailed_results.get('uncertainty_progress') or
                        detailed_results.get('uncertainty_bits') or
                        metrics.get('uncertainty_bits_trajectory') or
                        metrics.get('uncertainty_bits', []))
        
        # Also check for learning_history format
        learning_history = detailed_results.get('learning_history') or metrics.get('learning_history', [])
        if learning_history and not target_values:
            target_values = [step.get('outcome_value', 0.0) for step in learning_history]
            f1_scores = [step.get('f1_score', 0.0) for step in learning_history]
            shd_values = [step.get('shd', 0.0) for step in learning_history]
            true_parent_likelihoods = [step.get('true_parent_likelihood', 0.0) for step in learning_history]
            uncertainties = [step.get('uncertainty', 0.0) for step in learning_history]
        
        # Ensure all trajectories have the same length
        trajectory_lists = [steps, target_values, f1_scores, shd_values, true_parent_likelihoods, uncertainties]
        max_length = max(len(lst) for lst in trajectory_lists if lst)
        
        if max_length == 0:
            return []
        
        # Create step-by-step metrics
        for i in range(max_length):
            step_metrics = {
                'target_value': target_values[i] if i < len(target_values) else 0.0,
                'f1_score': f1_scores[i] if i < len(f1_scores) else 0.0,
                'shd': shd_values[i] if i < len(shd_values) else 0.0,
                'true_parent_likelihood': true_parent_likelihoods[i] if i < len(true_parent_likelihoods) else 0.0,
                'uncertainty': uncertainties[i] if i < len(uncertainties) else 0.0
            }
            trajectory_data.append(step_metrics)
        
        return trajectory_data
    
    def _log_run_summary(self, metrics: Dict[str, Any], method_name: str, run_idx: int) -> None:
        """Log summary metrics for a single run."""
        summary_metrics = {
            f"{method_name}/final_target_improvement": metrics.get('target_improvement', 0.0),
            f"{method_name}/final_structure_accuracy": metrics.get('structure_accuracy', 0.0),
            f"{method_name}/sample_efficiency": metrics.get('sample_efficiency', 0.0),
            f"{method_name}/convergence_steps": metrics.get('convergence_steps', 0),
            "run_idx": run_idx
        }
        
        self.run.log(summary_metrics)


class MockWandBLogger:
    """Mock WandB logger for when WandB is not available."""
    
    def __init__(self, *args, **kwargs):
        self.enabled = False
    
    def initialize_run(self, *args, **kwargs) -> bool:
        return False
    
    def log_scm_metadata(self, *args, **kwargs) -> None:
        pass
    
    def log_run_metrics(self, *args, **kwargs) -> None:
        pass
    
    def log_method_summary(self, *args, **kwargs) -> None:
        pass
    
    def log_comparison_results(self, *args, **kwargs) -> None:
        pass
    
    def log_experiment_summary(self, *args, **kwargs) -> None:
        pass
    
    def log_plots_as_artifacts(self, *args, **kwargs) -> None:
        pass
    
    def finish_run(self) -> None:
        pass


def create_wandb_logger(config: Dict[str, Any], experiment_name: str = "acbo_comparison") -> WandBLogger:
    """Factory function for creating WandB logger."""
    if WANDB_AVAILABLE:
        return WandBLogger(config, experiment_name)
    else:
        logger.warning("WandB not available, using mock logger")
        return MockWandBLogger(config, experiment_name)