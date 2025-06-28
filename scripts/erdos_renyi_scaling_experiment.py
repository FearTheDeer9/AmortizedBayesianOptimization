#!/usr/bin/env python3
"""
Erdos-Renyi Scaling Experiment

Tests whether self-supervised surrogate learning improves performance over static 
surrogate models across different graph sizes (5-20 nodes).

Both approaches use random intervention policies. The key difference:
- Baseline: Random interventions + static/untrained surrogate model  
- Active Learning: Random interventions + self-supervised surrogate training

This tests our core hypothesis about amortized structure learning.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# JAX and numerical libraries
import jax.numpy as jnp
import jax.random as random
import numpy as onp

# Local imports
from src.causal_bayes_opt.experiments.benchmark_graphs import create_erdos_renyi_scm, get_benchmark_graph_summary
from src.causal_bayes_opt.data_structures.scm import get_variables, get_target
from examples.complete_workflow_demo import run_progressive_learning_demo_with_scm
from examples.demo_learning import DemoConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScalingExperimentConfig:
    """Configuration for the scaling experiment."""
    min_nodes: int = 5
    max_nodes: int = 20
    edge_probability: float = 0.3
    n_intervention_steps: int = 20
    n_observational_samples: int = 100
    learning_rate: float = 1e-3
    n_runs_per_config: int = 3
    base_random_seed: int = 42


@dataclass(frozen=True)
class SingleRunResult:
    """Results from a single experimental run."""
    graph_size: int
    method: str  # 'static_surrogate' or 'learning_surrogate'
    run_id: int
    random_seed: int
    
    # Performance metrics
    final_f1_score: float
    structure_accuracy: float
    target_improvement: float
    sample_efficiency: float
    convergence_steps: int
    
    # Graph properties
    actual_edges: int
    edge_density: float
    target_variable: str
    
    # Runtime
    runtime_seconds: float
    
    # Detailed results (optional)
    detailed_results: Dict[str, Any] = None


def create_static_surrogate_baseline(
    scm,
    config: DemoConfig
) -> Dict[str, Any]:
    """
    Run baseline with static (non-learning) surrogate.
    
    This modifies the progressive learning demo to freeze surrogate parameters,
    creating a true baseline where only intervention outcomes are recorded.
    """
    logger.info("Running static surrogate baseline")
    
    # Import the required components
    from examples.demo_learning import create_learnable_surrogate_model, create_acquisition_state_from_buffer
    from src.causal_bayes_opt.data_structures.scm import get_variables, get_target
    from src.causal_bayes_opt.data_structures.buffer import create_empty_buffer
    from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
    from src.causal_bayes_opt.environments.sampling import sample_with_intervention
    from src.causal_bayes_opt.data_structures.sample import get_values
    from examples.demo_evaluation import analyze_convergence, compute_data_likelihood_from_posterior
    
    # Setup
    key = random.PRNGKey(config.random_seed)
    variables = sorted(get_variables(scm))
    target = get_target(scm)
    
    # Create surrogate model but DON'T update its parameters
    key, surrogate_key = random.split(key)
    surrogate_fn, _net, initial_params, opt_state, update_fn = create_learnable_surrogate_model(
        variables, surrogate_key, config.learning_rate, config.scoring_method
    )
    
    # CRITICAL: Keep parameters frozen throughout
    frozen_params = initial_params
    
    # Create random intervention policy  
    from examples.demo_learning import create_random_intervention_policy
    intervention_fn = create_random_intervention_policy(variables, target, config.intervention_value_range)
    
    # Generate initial data
    key, data_key = random.split(key)
    if config.n_observational_samples > 0:
        initial_samples = sample_from_linear_scm(scm, config.n_observational_samples, seed=int(data_key[0]))
        buffer = create_empty_buffer()
        for sample in initial_samples:
            buffer.add_observation(sample)
    else:
        initial_samples = []
        buffer = create_empty_buffer()
    
    # Track progress
    learning_history = []
    target_progress = []
    uncertainty_progress = []
    marginal_prob_progress = []
    
    # Initial progress tracking
    if initial_samples:
        initial_values = [get_values(s)[target] for s in initial_samples]
        best_so_far = max(initial_values)
    else:
        best_so_far = float('-inf')
    target_progress.append(best_so_far)
    
    # Get initial posterior (with frozen parameters)
    if initial_samples:
        current_state = create_acquisition_state_from_buffer(buffer, surrogate_fn, variables, target, 0, frozen_params)
        uncertainty_progress.append(current_state.uncertainty_bits)
        marginal_prob_progress.append(dict(current_state.marginal_parent_probs))
    else:
        uncertainty_progress.append(float('inf'))
        marginal_prob_progress.append({v: 0.0 for v in variables if v != target})
    
    # Run intervention loop with FROZEN parameters
    keys = random.split(key, config.n_intervention_steps)
    
    for step in range(config.n_intervention_steps):
        # Execute random intervention
        intervention = intervention_fn(key=keys[step])
        _, outcome_key = random.split(keys[step])
        
        if intervention and intervention.get('values'):
            # Apply intervention
            outcome = sample_with_intervention(scm, intervention, n_samples=1, seed=int(outcome_key[0]))[0]
            buffer.add_intervention(intervention, outcome)
        else:
            # Observational fallback
            outcome = sample_from_linear_scm(scm, n_samples=1, seed=int(outcome_key[0]))[0]
            buffer.add_observation(outcome)
        
        # Track progress (NO parameter updates)
        outcome_value = get_values(outcome)[target]
        best_so_far = max(best_so_far, outcome_value) if best_so_far != float('-inf') else outcome_value
        target_progress.append(best_so_far)
        
        # Get state with FROZEN parameters
        all_samples = buffer.get_all_samples()
        if len(all_samples) >= 5:
            updated_state = create_acquisition_state_from_buffer(buffer, surrogate_fn, variables, target, step+1, frozen_params)
            uncertainty_progress.append(updated_state.uncertainty_bits)
            marginal_prob_progress.append(dict(updated_state.marginal_parent_probs))
        else:
            uncertainty_progress.append(float('inf'))
            marginal_prob_progress.append({v: 0.0 for v in variables if v != target})
        
        # Store step info
        learning_history.append({
            'step': step + 1,
            'intervention': intervention,
            'outcome_value': outcome_value,
            'loss': 0.0,  # No learning happening
            'param_norm': 0.0,  # Parameters frozen
            'grad_norm': 0.0,   # No gradients computed
            'update_norm': 0.0,  # No updates applied
            'uncertainty': uncertainty_progress[-1],
            'marginals': marginal_prob_progress[-1]
        })
    
    # Final analysis with frozen parameters
    final_state = create_acquisition_state_from_buffer(buffer, surrogate_fn, variables, target, config.n_intervention_steps, frozen_params)
    
    # Get true parents for comparison
    from examples.demo_learning import _get_true_parents_for_scm
    true_parents = _get_true_parents_for_scm(scm, target)
    
    return {
        'target_variable': target,
        'true_parents': true_parents,
        'config': config,
        'initial_best': target_progress[0],
        'final_best': target_progress[-1],
        'improvement': target_progress[-1] - target_progress[0],
        'total_samples': buffer.size(),
        
        # Learning progress metrics
        'learning_history': learning_history,
        'target_progress': target_progress,
        'uncertainty_progress': uncertainty_progress,
        'marginal_prob_progress': marginal_prob_progress,
        
        # Final state
        'final_uncertainty': final_state.uncertainty_bits,
        'final_marginal_probs': final_state.marginal_parent_probs,
        'converged_to_truth': analyze_convergence(marginal_prob_progress, true_parents),
        
        # Method identifier
        'method': 'static_surrogate'
    }


def run_single_experiment(
    graph_size: int,
    method: str,
    run_id: int,
    config: ScalingExperimentConfig
) -> SingleRunResult:
    """
    Run a single experiment (either static or learning surrogate).
    
    Args:
        graph_size: Number of nodes in the graph
        method: 'static_surrogate' or 'learning_surrogate'
        run_id: Run identifier for this configuration
        config: Experiment configuration
        
    Returns:
        SingleRunResult with performance metrics
    """
    start_time = time.time()
    
    # Generate unique seed for this run
    run_seed = config.base_random_seed + graph_size * 100 + run_id * 10 + (1 if method == 'learning_surrogate' else 0)
    
    logger.info(f"Running {method} on {graph_size}-node graph (run {run_id}, seed {run_seed})")
    
    try:
        # Generate Erdos-Renyi SCM
        scm = create_erdos_renyi_scm(
            n_nodes=graph_size,
            edge_prob=config.edge_probability,
            target_variable=None,  # Will choose last variable
            seed=run_seed
        )
        
        # Get graph properties
        graph_summary = get_benchmark_graph_summary(scm)
        
        # Create demo configuration
        demo_config = DemoConfig(
            n_observational_samples=config.n_observational_samples,
            n_intervention_steps=config.n_intervention_steps,
            learning_rate=config.learning_rate,
            random_seed=run_seed
        )
        
        # Run appropriate method
        if method == 'static_surrogate':
            results = create_static_surrogate_baseline(scm, demo_config)
        elif method == 'learning_surrogate':
            results = run_progressive_learning_demo_with_scm(scm, demo_config)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Extract metrics
        convergence_info = results.get('converged_to_truth', {})
        final_f1_score = convergence_info.get('final_accuracy', 0.0)
        
        # Compute structure accuracy
        structure_accuracy = final_f1_score  # Simplified for now
        
        runtime = time.time() - start_time
        
        # Create result
        result = SingleRunResult(
            graph_size=graph_size,
            method=method,
            run_id=run_id,
            random_seed=run_seed,
            final_f1_score=final_f1_score,
            structure_accuracy=structure_accuracy,
            target_improvement=results.get('improvement', 0.0),
            sample_efficiency=results.get('improvement', 0.0) / config.n_intervention_steps,
            convergence_steps=config.n_intervention_steps,  # Simplified
            actual_edges=graph_summary['n_edges'],
            edge_density=graph_summary['edge_density'],
            target_variable=graph_summary['target_variable'],
            runtime_seconds=runtime,
            detailed_results=results
        )
        
        logger.info(f"Completed: F1={final_f1_score:.3f}, improvement={results.get('improvement', 0.0):.3f}, time={runtime:.1f}s")
        return result
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        # Return error result
        return SingleRunResult(
            graph_size=graph_size,
            method=method,
            run_id=run_id,
            random_seed=run_seed,
            final_f1_score=0.0,
            structure_accuracy=0.0,
            target_improvement=0.0,
            sample_efficiency=0.0,
            convergence_steps=0,
            actual_edges=0,
            edge_density=0.0,
            target_variable="",
            runtime_seconds=time.time() - start_time,
            detailed_results={'error': str(e)}
        )


def run_scaling_experiment(config: ScalingExperimentConfig = None) -> List[SingleRunResult]:
    """
    Run the complete scaling experiment.
    
    Args:
        config: Experiment configuration
        
    Returns:
        List of all experimental results
    """
    if config is None:
        config = ScalingExperimentConfig()
    
    logger.info(f"Starting Erdos-Renyi scaling experiment")
    logger.info(f"Graph sizes: {config.min_nodes}-{config.max_nodes} nodes")
    logger.info(f"Methods: static_surrogate vs learning_surrogate")
    logger.info(f"Runs per config: {config.n_runs_per_config}")
    
    all_results = []
    total_experiments = (config.max_nodes - config.min_nodes + 1) * 2 * config.n_runs_per_config
    
    logger.info(f"Total experiments: {total_experiments}")
    
    experiment_count = 0
    
    # Iterate over graph sizes
    for graph_size in range(config.min_nodes, config.max_nodes + 1):
        # Test both methods
        for method in ['static_surrogate', 'learning_surrogate']:
            # Multiple runs for statistical validity
            for run_id in range(config.n_runs_per_config):
                experiment_count += 1
                
                logger.info(f"Experiment {experiment_count}/{total_experiments}: "
                           f"{graph_size} nodes, {method}, run {run_id}")
                
                result = run_single_experiment(graph_size, method, run_id, config)
                all_results.append(result)
                
                # Brief pause to prevent overheating
                time.sleep(0.1)
    
    logger.info(f"Scaling experiment complete: {len(all_results)} results")
    return all_results


def analyze_scaling_results(results: List[SingleRunResult]) -> Dict[str, Any]:
    """
    Analyze the scaling experiment results.
    
    Args:
        results: List of experimental results
        
    Returns:
        Dictionary with analysis results
    """
    logger.info("Analyzing scaling experiment results")
    
    # Group results by method and graph size
    grouped_results = {}
    for result in results:
        key = (result.method, result.graph_size)
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(result)
    
    # Compute summary statistics
    summary_stats = {}
    for (method, graph_size), group in grouped_results.items():
        f1_scores = [r.final_f1_score for r in group]
        improvements = [r.target_improvement for r in group]
        runtimes = [r.runtime_seconds for r in group]
        
        summary_stats[(method, graph_size)] = {
            'n_runs': len(group),
            'mean_f1': float(onp.mean(f1_scores)),
            'std_f1': float(onp.std(f1_scores)),
            'mean_improvement': float(onp.mean(improvements)),
            'std_improvement': float(onp.std(improvements)),
            'mean_runtime': float(onp.mean(runtimes)),
            'mean_edges': float(onp.mean([r.actual_edges for r in group]))
        }
    
    # Compare methods at each graph size
    comparisons = {}
    for graph_size in range(5, 21):  # 5-20 nodes
        static_key = ('static_surrogate', graph_size)
        learning_key = ('learning_surrogate', graph_size)
        
        if static_key in summary_stats and learning_key in summary_stats:
            static_stats = summary_stats[static_key]
            learning_stats = summary_stats[learning_key]
            
            f1_improvement = learning_stats['mean_f1'] - static_stats['mean_f1']
            f1_improvement_ratio = learning_stats['mean_f1'] / static_stats['mean_f1'] if static_stats['mean_f1'] > 0 else float('inf')
            
            target_improvement_diff = learning_stats['mean_improvement'] - static_stats['mean_improvement']
            
            comparisons[graph_size] = {
                'static_f1': static_stats['mean_f1'],
                'learning_f1': learning_stats['mean_f1'],
                'f1_improvement_absolute': f1_improvement,
                'f1_improvement_ratio': f1_improvement_ratio,
                'static_target_improvement': static_stats['mean_improvement'],
                'learning_target_improvement': learning_stats['mean_improvement'],
                'target_improvement_diff': target_improvement_diff,
                'learning_wins_f1': f1_improvement > 0.01,  # At least 1% improvement
                'mean_edges': static_stats['mean_edges']
            }
    
    # Overall analysis
    learning_wins = sum(1 for comp in comparisons.values() if comp['learning_wins_f1'])
    total_comparisons = len(comparisons)
    
    analysis = {
        'summary_statistics': summary_stats,
        'pairwise_comparisons': comparisons,
        'overall_analysis': {
            'total_graph_sizes': total_comparisons,
            'learning_wins_count': learning_wins,
            'learning_win_rate': learning_wins / total_comparisons if total_comparisons > 0 else 0.0,
            'mean_f1_improvement': float(onp.mean([comp['f1_improvement_absolute'] for comp in comparisons.values()])),
            'mean_target_improvement_diff': float(onp.mean([comp['target_improvement_diff'] for comp in comparisons.values()]))
        }
    }
    
    return analysis


def save_results(results: List[SingleRunResult], analysis: Dict[str, Any], output_dir: str = "scaling_experiment_results"):
    """
    Save experiment results and analysis.
    
    Args:
        results: List of experimental results
        analysis: Analysis results
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save raw results
    results_data = [asdict(r) for r in results]
    with open(output_path / "raw_results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Save analysis
    with open(output_path / "analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Create summary report
    overall = analysis['overall_analysis']
    comparisons = analysis['pairwise_comparisons']
    
    report = f"""
# Erdos-Renyi Scaling Experiment Results

## Overall Results
- **Graph sizes tested**: 5-20 nodes
- **Learning surrogate wins**: {overall['learning_wins_count']}/{overall['total_graph_sizes']} ({overall['learning_win_rate']:.1%})
- **Mean F1 improvement**: {overall['mean_f1_improvement']:.3f}
- **Mean target improvement difference**: {overall['mean_target_improvement_diff']:.3f}

## Results by Graph Size
| Nodes | Edges | Static F1 | Learning F1 | F1 Improvement | Target Improvement Diff | Winner |
|-------|-------|-----------|-------------|----------------|------------------------|--------|
"""
    
    for size in sorted(comparisons.keys()):
        comp = comparisons[size]
        winner = "🏆 Learning" if comp['learning_wins_f1'] else "📊 Static"
        report += f"| {size:5d} | {comp['mean_edges']:5.1f} | {comp['static_f1']:9.3f} | {comp['learning_f1']:11.3f} | {comp['f1_improvement_absolute']:14.3f} | {comp['target_improvement_diff']:22.3f} | {winner} |\n"
    
    report += f"\n## Conclusion\n"
    if overall['learning_win_rate'] > 0.6:
        report += "✅ **VALIDATION SUCCESSFUL**: Learning surrogate consistently outperforms static surrogate.\n"
    elif overall['learning_win_rate'] > 0.4:
        report += "⚠️ **MIXED RESULTS**: Learning surrogate shows some benefit but not consistently.\n"
    else:
        report += "❌ **VALIDATION FAILED**: Learning surrogate does not provide clear benefits.\n"
    
    with open(output_path / "summary_report.md", 'w') as f:
        f.write(report)
    
    logger.info(f"Results saved to {output_path}")
    print(report)


def main():
    """Run the complete scaling experiment."""
    logger.info("Starting Erdos-Renyi scaling experiment")
    
    # Create configuration
    config = ScalingExperimentConfig(
        min_nodes=5,
        max_nodes=20,
        edge_probability=0.3,
        n_intervention_steps=20,
        n_observational_samples=100,
        learning_rate=1e-3,
        n_runs_per_config=3,  # 3 runs for statistical validity
        base_random_seed=42
    )
    
    # Run experiment
    start_time = time.time()
    results = run_scaling_experiment(config)
    total_time = time.time() - start_time
    
    # Analyze results
    analysis = analyze_scaling_results(results)
    
    # Save results
    save_results(results, analysis)
    
    logger.info(f"Experiment completed in {total_time:.1f} seconds")
    
    # Print quick summary
    overall = analysis['overall_analysis']
    print(f"\n🎯 QUICK SUMMARY:")
    print(f"Learning surrogate wins: {overall['learning_wins_count']}/{overall['total_graph_sizes']} graph sizes")
    print(f"Mean F1 improvement: {overall['mean_f1_improvement']:.3f}")
    
    if overall['learning_win_rate'] > 0.6:
        print("✅ Validation SUCCESSFUL - Learning approach works!")
    else:
        print("❌ Validation INCONCLUSIVE - Need to investigate further")


if __name__ == "__main__":
    main()