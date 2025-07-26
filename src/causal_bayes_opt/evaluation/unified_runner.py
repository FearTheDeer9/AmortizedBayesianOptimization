"""
Unified Evaluation Runner

Central runner that manages evaluation of multiple methods on test SCMs,
handles parallel execution, and produces standardized comparison results.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import json

from .base_evaluator import BaseEvaluator
from .result_types import ExperimentResult, MethodMetrics, ComparisonResults

logger = logging.getLogger(__name__)


class MethodRegistry:
    """Registry for managing evaluation methods."""
    
    def __init__(self):
        self._methods: Dict[str, BaseEvaluator] = {}
        
    def register(self, evaluator: BaseEvaluator) -> None:
        """Register an evaluation method."""
        name = evaluator.get_method_name()
        if name in self._methods:
            logger.warning(f"Overwriting existing method: {name}")
        self._methods[name] = evaluator
        logger.info(f"Registered method: {name}")
        
    def get(self, name: str) -> Optional[BaseEvaluator]:
        """Get a registered method by name."""
        return self._methods.get(name)
    
    def get_all(self) -> List[BaseEvaluator]:
        """Get all registered methods."""
        return list(self._methods.values())
    
    def list_methods(self) -> List[str]:
        """List all registered method names."""
        return list(self._methods.keys())
    
    def clear(self) -> None:
        """Clear all registered methods."""
        self._methods.clear()


class UnifiedEvaluationRunner:
    """
    Central runner for evaluating multiple methods on test SCMs.
    
    This class handles:
    - Method registration
    - Parallel execution of evaluations
    - Result aggregation and comparison
    - Saving results to disk
    """
    
    def __init__(self, output_dir: Union[str, Path], parallel: bool = True):
        """
        Initialize the evaluation runner.
        
        Args:
            output_dir: Directory to save results
            parallel: Whether to run evaluations in parallel
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parallel = parallel
        self.registry = MethodRegistry()
        
    def register_method(self, evaluator: BaseEvaluator) -> None:
        """Register an evaluation method."""
        self.registry.register(evaluator)
        
    def register_baseline_methods(self) -> None:
        """Register standard baseline methods."""
        # Import here to avoid circular dependencies
        try:
            from .baseline_evaluators import (
                RandomBaselineEvaluator,
                OracleBaselineEvaluator,
                LearningBaselineEvaluator
            )
            
            self.register_method(RandomBaselineEvaluator())
            self.register_method(OracleBaselineEvaluator())
            self.register_method(LearningBaselineEvaluator())
            logger.info("Registered baseline methods")
        except ImportError:
            logger.warning("Baseline evaluators not available")
    
    def run_single_scm_comparison(
        self,
        scm: Any,
        config: Dict[str, Any],
        scm_idx: int,
        n_runs: int = 3,
        base_seed: int = 42
    ) -> Dict[str, List[ExperimentResult]]:
        """
        Run all registered methods on a single SCM.
        
        Args:
            scm: Structural Causal Model to evaluate
            config: Evaluation configuration
            scm_idx: Index of this SCM (for logging)
            n_runs: Number of runs per method
            base_seed: Base random seed
            
        Returns:
            Dict mapping method names to lists of results
        """
        results = {}
        
        for method in self.registry.get_all():
            method_name = method.get_method_name()
            logger.info(f"Evaluating {method_name} on SCM {scm_idx}")
            
            try:
                method_results = method.evaluate_multiple_runs(
                    scm, config, n_runs, base_seed + scm_idx * 1000
                )
                results[method_name] = method_results
            except Exception as e:
                logger.error(f"Failed to evaluate {method_name}: {e}")
                # Create failed results
                results[method_name] = [
                    ExperimentResult(
                        learning_history=[],
                        final_metrics={'error': str(e)},
                        metadata={'scm_idx': scm_idx},
                        success=False,
                        error_message=str(e)
                    )
                    for _ in range(n_runs)
                ]
        
        return results
    
    def run_comparison(
        self,
        test_scms: List[Any],
        config: Dict[str, Any],
        n_runs_per_scm: int = 3,
        base_seed: int = 42,
        scm_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> ComparisonResults:
        """
        Run comparison of all registered methods on test SCMs.
        
        Args:
            test_scms: List of SCMs to test on
            config: Evaluation configuration
            n_runs_per_scm: Number of runs per SCM per method
            base_seed: Base random seed
            scm_metadata: Optional metadata about SCMs
            
        Returns:
            ComparisonResults with aggregated metrics
        """
        if not self.registry.get_all():
            raise ValueError("No methods registered for evaluation")
        
        logger.info(f"Starting comparison with {len(self.registry.get_all())} methods "
                   f"on {len(test_scms)} SCMs")
        
        start_time = time.time()
        
        # Collect all results by method
        all_results_by_method: Dict[str, List[ExperimentResult]] = {
            method.get_method_name(): [] for method in self.registry.get_all()
        }
        
        if self.parallel and len(test_scms) > 1:
            # Parallel execution
            logger.info(f"Running parallel evaluation (parallel={self.parallel})")
            
            # Test if SCMs are serializable before attempting parallel execution
            try:
                import pickle
                for idx, scm in enumerate(test_scms):
                    pickle.dumps(scm)
            except Exception as e:
                logger.warning(f"SCMs are not serializable with pickle: {e}")
                logger.info("Falling back to sequential execution")
                self.parallel = False
            
            if self.parallel:
                try:
                    with ProcessPoolExecutor() as executor:
                        # Submit jobs
                        future_to_scm = {
                            executor.submit(
                                self.run_single_scm_comparison,
                                scm, config, idx, n_runs_per_scm, base_seed
                            ): idx
                            for idx, scm in enumerate(test_scms)
                        }
                        
                        # Collect results
                        for future in as_completed(future_to_scm):
                            scm_idx = future_to_scm[future]
                            try:
                                scm_results = future.result()
                                # Aggregate results
                                for method_name, results in scm_results.items():
                                    all_results_by_method[method_name].extend(results)
                            except Exception as e:
                                logger.error(f"SCM {scm_idx} evaluation failed: {e}")
                                logger.debug(f"Full error: {type(e).__name__}: {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"Parallel execution failed: {e}")
                    logger.info("Falling back to sequential execution")
                    self.parallel = False
        
        # If not parallel or parallel failed, run sequentially
        if not self.parallel:
            logger.info(f"Running sequential evaluation (parallel={self.parallel})")
            for idx, scm in enumerate(test_scms):
                scm_results = self.run_single_scm_comparison(
                    scm, config, idx, n_runs_per_scm, base_seed
                )
                # Aggregate results
                for method_name, results in scm_results.items():
                    all_results_by_method[method_name].extend(results)
        
        # Compute aggregated metrics
        method_metrics = {}
        for method_name, results in all_results_by_method.items():
            metrics = MethodMetrics.from_results(method_name, results)
            method_metrics[method_name] = metrics
            logger.info(f"{method_name}: {metrics.n_successful}/{metrics.n_runs} successful, "
                       f"mean final value: {metrics.mean_final_value:.3f}")
        
        # Create comparison results
        comparison = ComparisonResults(
            method_metrics=method_metrics,
            scm_metadata={
                'n_scms': len(test_scms),
                'n_runs_per_scm': n_runs_per_scm,
                'scm_details': scm_metadata or []
            },
            config=config,
            raw_results=all_results_by_method,
            statistical_tests={}  # Can be populated later if needed
        )
        
        total_time = time.time() - start_time
        logger.info(f"Comparison completed in {total_time:.1f} seconds")
        
        # Save results
        self.save_results(comparison, all_results_by_method)
        
        return comparison
    
    def save_results(
        self, 
        comparison: ComparisonResults,
        raw_results: Optional[Dict[str, List[ExperimentResult]]] = None
    ) -> None:
        """Save comparison results to disk."""
        # Save summary
        summary_path = self.output_dir / "comparison_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(comparison.to_dict(), f, indent=2)
        logger.info(f"Saved comparison summary to {summary_path}")
        
        # Save detailed results if provided
        if raw_results:
            details_path = self.output_dir / "detailed_results.json"
            detailed_data = {}
            for method_name, results in raw_results.items():
                detailed_data[method_name] = [r.to_dict() for r in results]
            
            with open(details_path, 'w') as f:
                json.dump(detailed_data, f, indent=2)
            logger.info(f"Saved detailed results to {details_path}")
        
        # Save summary DataFrame if pandas available
        try:
            df = comparison.summary_dataframe()
            if hasattr(df, 'to_csv'):  # Check if it's a DataFrame
                csv_path = self.output_dir / "comparison_summary.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved summary CSV to {csv_path}")
        except Exception as e:
            logger.debug(f"Could not save CSV summary: {e}")
    
    def load_results(self, results_dir: Union[str, Path]) -> ComparisonResults:
        """Load previously saved results."""
        results_dir = Path(results_dir)
        summary_path = results_dir / "comparison_summary.json"
        
        if not summary_path.exists():
            raise FileNotFoundError(f"No results found at {summary_path}")
        
        with open(summary_path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct ComparisonResults
        # Note: This creates a simplified version without full trajectory data
        # For full reconstruction, would need to load detailed_results.json
        method_metrics = {}
        for name, metrics_data in data['method_metrics'].items():
            # Create simplified MethodMetrics
            method_metrics[name] = MethodMetrics(
                method_name=name,
                n_runs=metrics_data['n_runs'],
                n_successful=metrics_data['n_successful'],
                mean_final_value=metrics_data['mean_final_value'],
                std_final_value=metrics_data['std_final_value'],
                mean_improvement=metrics_data['mean_improvement'],
                std_improvement=metrics_data['std_improvement'],
                mean_steps=metrics_data['mean_steps'],
                mean_time=metrics_data['mean_time'],
                outcome_trajectory_mean=[],  # Not saved in summary
                outcome_trajectory_std=[],
                uncertainty_trajectory_mean=[],
                uncertainty_trajectory_std=[]
            )
        
        return ComparisonResults(
            method_metrics=method_metrics,
            scm_metadata=data['scm_metadata'],
            config=data['config'],
            timestamp=data['timestamp']
        )