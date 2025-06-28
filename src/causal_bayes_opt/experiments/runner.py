#!/usr/bin/env python3
"""
Simple Experiment Runner

Easy-to-use functions for running ACBO experiments and analyzing results.
No complex configuration - just direct function calls.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# JAX and numerical libraries
import jax.numpy as jnp
import jax.random as random
import numpy as onp

# Local imports
from .benchmark_graphs import create_erdos_renyi_scm
from .benchmark_datasets import load_benchmark_dataset
from ..analysis.trajectory_metrics import (
    compute_trajectory_metrics, extract_metrics_from_experiment_result,
    analyze_convergence_trajectory, compute_intervention_efficiency,
    extract_learning_curves
)
from ..visualization.plots import (
    plot_convergence, plot_target_optimization, plot_method_comparison,
    save_all_plots
)
from ..storage.results import (
    save_experiment_result, load_experiment_results, create_results_summary
)
from scripts.erdos_renyi_scaling_experiment import create_static_surrogate_baseline
from examples.complete_workflow_demo import run_progressive_learning_demo_with_scm
from examples.demo_learning import DemoConfig
from ..data_structures.scm import get_variables, get_target
from .benchmark_graphs import get_benchmark_graph_summary
        

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_experiments(
    scms: List[Any],
    methods: List[str] = None,
    n_runs: int = 3,
    n_interventions: int = 20,
    n_observational: int = 50,
    output_dir: str = "results",
    save_plots: bool = True
) -> Dict[str, Any]:
    """
    Run experiments on multiple SCMs with different methods.
    
    Args:
        scms: List of SCM objects to test
        methods: List of method names ['static_surrogate', 'learning_surrogate']
        n_runs: Number of runs per SCM-method combination
        n_interventions: Number of intervention steps per run
        n_observational: Number of initial observational samples
        output_dir: Directory to save results
        save_plots: Whether to save plots for each experiment
        
    Returns:
        Dictionary with experiment results and analysis
    """
    if methods is None:
        methods = ['static_surrogate', 'learning_surrogate']
    
    logger.info(f"Running experiments: {len(scms)} SCMs × {len(methods)} methods × {n_runs} runs")
    
    all_results = []
    experiment_count = 0
    total_experiments = len(scms) * len(methods) * n_runs
    
    for scm_idx, scm in enumerate(scms):
        for method in methods:
            for run_id in range(n_runs):
                experiment_count += 1
                logger.info(f"Experiment {experiment_count}/{total_experiments}: SCM {scm_idx}, {method}, run {run_id}")
                
                try:
                    # Run single experiment
                    result = run_single_experiment(
                        scm=scm,
                        method=method,
                        n_interventions=n_interventions,
                        n_observational=n_observational,
                        random_seed=42 + experiment_count
                    )
                    
                    # Add metadata
                    result.update({
                        'scm_index': scm_idx,
                        'method': method,
                        'run_id': run_id,
                        'experiment_id': experiment_count
                    })
                    
                    # Save result
                    timestamp = f"scm{scm_idx}_{method}_run{run_id}"
                    save_path = save_experiment_result(result, output_dir, timestamp=timestamp)
                    result['saved_to'] = save_path
                    
                    all_results.append(result)
                    
                    logger.info(f"✓ Completed: F1={result.get('final_f1_score', 0):.3f}")
                    
                except Exception as e:
                    logger.error(f"✗ Failed: {e}")
                    # Save error result
                    error_result = {
                        'scm_index': scm_idx,
                        'method': method,
                        'run_id': run_id,
                        'experiment_id': experiment_count,
                        'error': str(e),
                        'success': False
                    }
                    all_results.append(error_result)
    
    # Analyze results
    logger.info("Analyzing experiment results...")
    analysis = analyze_experiment_results(all_results)
    
    # Save summary
    summary_path = Path(output_dir) / "experiment_summary.json"
    create_results_summary(all_results, str(summary_path))
    
    # Generate plots if requested
    if save_plots:
        logger.info("Generating plots...")
        plot_dir = Path(output_dir) / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        try:
            generate_summary_plots(analysis, str(plot_dir))
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")
    
    logger.info(f"Experiments complete! Results saved to {output_dir}")
    
    return {
        'results': all_results,
        'analysis': analysis,
        'summary': {
            'total_experiments': total_experiments,
            'successful_experiments': len([r for r in all_results if r.get('success', True)]),
            'methods_tested': methods,
            'scms_tested': len(scms),
            'runs_per_config': n_runs
        }
    }


def run_single_experiment(
    scm: Any,
    method: str,
    n_interventions: int = 20,
    n_observational: int = 50,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Run a single experiment with the specified method using the provided SCM.
    
    This function uses the SAME SCM for both methods to ensure fair comparison.
    
    Args:
        scm: SCM object to use (SAME for both methods)
        method: Method name ('static_surrogate' or 'learning_surrogate')
        n_interventions: Number of intervention steps
        n_observational: Number of initial observational samples  
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with experiment results
    """
    start_time = time.time()
    
    try:
        
        # Get SCM properties
        variables = list(get_variables(scm))
        target = get_target(scm)
        graph_size = len(variables)
        
        # Create demo configuration
        demo_config = DemoConfig(
            n_observational_samples=n_observational,
            n_intervention_steps=n_interventions,
            learning_rate=1e-3,
            random_seed=random_seed
        )
        
        # Get graph properties for metadata
        try:
            graph_summary = get_benchmark_graph_summary(scm)
            actual_edges = graph_summary.get('n_edges', 0)
            edge_density = graph_summary.get('edge_density', 0.0)
        except Exception:
            # Fallback if graph summary fails
            actual_edges = 0
            edge_density = 0.0
            logger.warning("Failed to get graph summary, using defaults")
        
        # Run appropriate method using the PROVIDED SCM
        if method == 'static_surrogate':
            results = create_static_surrogate_baseline(scm, demo_config)
        elif method == 'learning_surrogate':
            results = run_progressive_learning_demo_with_scm(scm, demo_config)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Extract key metrics
        final_f1_score = results.get('final_f1_score', 0.0)
        target_improvement = results.get('target_improvement', 0.0)
        
        # Get true parents for the target variable
        true_parents = []
        if 'true_parents' in results:
            true_parents = results['true_parents']
        elif hasattr(scm, 'get') and 'adjacency_matrix' in scm:
            # Extract true parents from adjacency matrix
            adj_matrix = scm['adjacency_matrix']
            target_idx = variables.index(target) if target in variables else -1
            if target_idx >= 0 and hasattr(adj_matrix, 'shape'):
                true_parents = [variables[i] for i in range(len(variables)) 
                              if adj_matrix[i, target_idx] != 0]
        
        # Convert to our standardized format
        experiment_result = {
            'method': method,
            'graph_size': graph_size,
            'target_variable': target,
            'variables': variables,
            'true_parents': true_parents,
            'final_f1_score': final_f1_score,
            'target_improvement': target_improvement,
            'runtime_seconds': time.time() - start_time,
            'success': True,
            'config': {
                'n_interventions': n_interventions,
                'n_observational': n_observational,
                'random_seed': random_seed
            },
            'graph_properties': {
                'actual_edges': actual_edges,
                'edge_density': edge_density
            },
            'detailed_results': results
        }
        
        return experiment_result
        
    except Exception as e:
        logger.error(f"Experiment failed for method {method}: {e}")
        return {
            'method': method,
            'error': str(e),
            'success': False,
            'runtime_seconds': time.time() - start_time
        }


def create_erdos_renyi_scms(
    sizes: List[int] = None,
    edge_probs: List[float] = None,
    n_scms_per_config: int = 2
) -> List[Any]:
    """
    Create a collection of Erdos-Renyi SCMs for testing.
    
    Args:
        sizes: List of graph sizes (number of nodes)
        edge_probs: List of edge probabilities
        n_scms_per_config: Number of SCMs per size-probability combination
        
    Returns:
        List of SCM objects
    """
    if sizes is None:
        sizes = [5, 8, 10]
    if edge_probs is None:
        edge_probs = [0.3, 0.5]
    
    scms = []
    scm_id = 0
    
    for size in sizes:
        for edge_prob in edge_probs:
            for i in range(n_scms_per_config):
                try:
                    scm = create_erdos_renyi_scm(
                        n_nodes=size,
                        edge_prob=edge_prob,
                        seed=42 + scm_id
                    )
                    scms.append(scm)
                    scm_id += 1
                except Exception as e:
                    logger.warning(f"Failed to create SCM {scm_id}: {e}")
                    continue
    
    logger.info(f"Created {len(scms)} Erdos-Renyi SCMs")
    return scms


def load_and_analyze_results(
    results_dir: str = "results",
    pattern: str = "*.json"
) -> Dict[str, Any]:
    """
    Load saved results and compute analysis.
    
    Args:
        results_dir: Directory containing saved results
        pattern: Pattern for matching result files
        
    Returns:
        Dictionary with loaded results and analysis
    """
    logger.info(f"Loading results from {results_dir}")
    
    # Load results
    results = load_experiment_results(results_dir, pattern)
    
    if not results:
        logger.warning("No results found")
        return {'results': [], 'analysis': {}}
    
    # Analyze results
    analysis = analyze_experiment_results(results)
    
    logger.info(f"Loaded and analyzed {len(results)} results")
    
    return {
        'results': results,
        'analysis': analysis
    }


def analyze_experiment_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze experiment results and compute metrics.
    
    Args:
        results: List of experiment result dictionaries
        
    Returns:
        Dictionary with analysis results
    """
    if not results:
        return {}
    
    successful_results = [r for r in results if r.get('success', True)]
    
    if not successful_results:
        return {'error': 'No successful results to analyze'}
    
    # Group by method
    by_method = {}
    for result in successful_results:
        method = result.get('method', 'unknown')
        if method not in by_method:
            by_method[method] = []
        by_method[method].append(result)
    
    # Compute trajectory metrics for each result
    trajectory_results = {}
    for method, method_results in by_method.items():
        trajectory_results[method] = []
        
        for result in method_results:
            # Extract trajectory metrics using our utility function
            true_parents = result.get('true_parents', [])
            trajectory_metrics = extract_metrics_from_experiment_result(result, true_parents)
            
            # Compute convergence analysis
            convergence_analysis = analyze_convergence_trajectory(trajectory_metrics)
            
            # Compute efficiency metrics
            efficiency_metrics = compute_intervention_efficiency(trajectory_metrics)
            
            trajectory_results[method].append({
                'trajectory_metrics': trajectory_metrics,
                'convergence_analysis': convergence_analysis,
                'efficiency_metrics': efficiency_metrics,
                'original_result': result
            })
    
    # Extract learning curves for comparison
    learning_curves = {}
    for method, method_data in trajectory_results.items():
        metrics_list = [data['trajectory_metrics'] for data in method_data]
        learning_curves[method] = extract_learning_curves({method: metrics_list})[method]
    
    # Overall statistics
    f1_scores = [r.get('final_f1_score', 0) for r in successful_results]
    improvements = [r.get('target_improvement', 0) for r in successful_results]
    runtimes = [r.get('runtime_seconds', 0) for r in successful_results]
    
    overall_stats = {
        'total_results': len(results),
        'successful_results': len(successful_results),
        'success_rate': len(successful_results) / len(results),
        'mean_f1_score': float(onp.mean(f1_scores)),
        'std_f1_score': float(onp.std(f1_scores)),
        'mean_improvement': float(onp.mean(improvements)),
        'mean_runtime': float(onp.mean(runtimes)),
        'methods_tested': list(by_method.keys())
    }
    
    return {
        'overall_stats': overall_stats,
        'by_method': by_method,
        'trajectory_results': trajectory_results,
        'learning_curves': learning_curves
    }


def generate_summary_plots(analysis: Dict[str, Any], output_dir: str) -> List[str]:
    """
    Generate summary plots from analysis results.
    
    Args:
        analysis: Analysis results from analyze_experiment_results
        output_dir: Directory to save plots
        
    Returns:
        List of saved plot paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    saved_plots = []
    
    try:
        # Method comparison plot
        learning_curves = analysis.get('learning_curves', {})
        if learning_curves:
            comparison_path = output_path / "method_comparison.png"
            plot_method_comparison(
                learning_curves,
                title="Method Comparison - Learning Curves",
                save_path=str(comparison_path)
            )
            saved_plots.append(str(comparison_path))
        
        # Individual method plots
        trajectory_results = analysis.get('trajectory_results', {})
        for method, method_data in trajectory_results.items():
            if method_data:
                # Take first result as example
                example_data = method_data[0]
                trajectory_metrics = example_data['trajectory_metrics']
                
                # Convergence plot
                conv_path = output_path / f"{method}_convergence.png"
                plot_convergence(
                    trajectory_metrics,
                    title=f"Convergence - {method}",
                    save_path=str(conv_path)
                )
                saved_plots.append(str(conv_path))
                
                # Target optimization plot
                target_path = output_path / f"{method}_target_optimization.png"
                plot_target_optimization(
                    trajectory_metrics,
                    title=f"Target Optimization - {method}",
                    save_path=str(target_path)
                )
                saved_plots.append(str(target_path))
        
        logger.info(f"Generated {len(saved_plots)} summary plots")
        
    except Exception as e:
        logger.error(f"Failed to generate some plots: {e}")
    
    return saved_plots


def quick_test():
    """Run a quick test of the experiment pipeline."""
    logger.info("Running quick test of experiment pipeline")
    
    # Create small test SCMs
    test_scms = create_erdos_renyi_scms(sizes=[5], edge_probs=[0.3], n_scms_per_config=1)
    
    if not test_scms:
        logger.error("Failed to create test SCMs")
        return
    
    # Run experiments
    results = run_experiments(
        scms=test_scms[:1],  # Just one SCM for testing
        methods=['static_surrogate', 'learning_surrogate'],
        n_runs=1,
        n_interventions=5,  # Very short for testing
        output_dir="test_results",
        save_plots=True
    )
    
    logger.info("Quick test completed!")
    logger.info(f"Results: {results['summary']}")
    
    return results


if __name__ == "__main__":
    # Run quick test
    quick_test()