#!/usr/bin/env python3
"""
Active Learning vs Surrogate Approach Comparison with WandB Logging

This script runs a comprehensive comparison between active learning and surrogate-based
approaches for causal discovery, with full WandB experiment tracking and visualization.

Usage:
    # Run complete comparison
    python scripts/acbo_wandb_experiment.py
    
    # Custom configuration
    python scripts/acbo_wandb_experiment.py --multirun \
        experiment.n_variables=5,10 \
        experiment.n_runs=3,5 \
        logging.wandb.tags=["comparison","linear_gaussian"]

Features:
- Compares active learning vs multiple surrogate baselines
- Tracks causal discovery metrics (SHD, precision, recall, intervention efficiency)
- Full WandB experiment logging with visualizations
- Statistical significance testing
- Automatic plot generation and artifact tracking
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time
import json

import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as onp
import pyrsistent as pyr
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Import existing infrastructure
from causal_bayes_opt.experiments.baseline_methods import (
    BaselineType, BaselineConfig, BaselineResult, compare_baselines
)

# Import ACBO methods
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ACBO method identifiers
class ACBOMethodType:
    RANDOM_UNTRAINED = "random_untrained"
    RANDOM_LEARNING = "random_learning"
    ORACLE_LEARNING = "oracle_learning"
from causal_bayes_opt.experiments.benchmark_graphs import create_erdos_renyi_scm
from causal_bayes_opt.analysis.trajectory_metrics import (
    compute_trajectory_metrics, analyze_convergence_trajectory,
    extract_learning_curves, compute_intervention_efficiency
)
from causal_bayes_opt.visualization.plots import (
    plot_method_comparison, plot_intervention_efficiency,
    create_experiment_dashboard, save_all_plots
)
from causal_bayes_opt.data_structures.scm import get_variables, get_target, get_parents

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../config", config_name="acbo_comparison_config")
def run_acbo_comparison_experiment(cfg: DictConfig) -> None:
    """Main experiment comparing active learning vs surrogate approaches."""
    
    logger.info("Starting ACBO Active Learning vs Surrogate Comparison")
    logger.info(f"Configuration:\\n{OmegaConf.to_yaml(cfg)}")
    
    # Initialize WandB if enabled
    wandb_run = None
    if cfg.logging.wandb.enabled and WANDB_AVAILABLE:
        wandb_run = wandb.init(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.logging.wandb.tags + ["acbo_comparison", "causal_discovery"],
            group="active_vs_surrogate",
            name=f"acbo_comparison_{cfg.experiment.problem.difficulty}_{int(time.time())}"
        )
        
        # Define custom metrics for proper time-series visualization with intervention steps
        if wandb_run:
            # Define metrics that use intervention_step as x-axis instead of the global step
            methods = ["Random Policy + Untrained Model", "Random Policy + Learning Model", "Oracle Policy + Learning Model"]
            scm_names = ["fork_3var", "chain_3var", "collider_3var"]
            scm_types = ["fork", "chain", "collider"]
            
            for method in methods:
                wandb_run.define_metric(f"{method}/target_value", step_metric="intervention_step")
                wandb_run.define_metric(f"{method}/true_parent_likelihood", step_metric="intervention_step") 
                wandb_run.define_metric(f"{method}/f1_score", step_metric="intervention_step")
                wandb_run.define_metric(f"{method}/shd", step_metric="intervention_step")
                wandb_run.define_metric(f"{method}/uncertainty", step_metric="intervention_step")
                
                # SCM-specific metrics
                for scm_name in scm_names:
                    wandb_run.define_metric(f"{method}/target_value_{scm_name}", step_metric="intervention_step")
                    wandb_run.define_metric(f"{method}/f1_score_{scm_name}", step_metric="intervention_step")
                    wandb_run.define_metric(f"{method}/shd_{scm_name}", step_metric="intervention_step")
            
            # SCM type aggregated metrics  
            for scm_type in scm_types:
                wandb_run.define_metric(f"{scm_type}/target_value", step_metric="intervention_step")
                wandb_run.define_metric(f"{scm_type}/f1_score", step_metric="intervention_step")
                wandb_run.define_metric(f"{scm_type}/shd", step_metric="intervention_step")
        
        logger.info(f"WandB initialized: {wandb.run.url}")
    
    # Run experiments
    try:
        results = run_comprehensive_comparison(cfg, wandb_run)
        
        # Log final comparison metrics to WandB
        if wandb_run:
            log_final_comparison_metrics(results, wandb_run)
            
            # Upload plots as artifacts
            upload_plots_to_wandb(results, wandb_run)
        
        # Save results locally
        save_experiment_results(results, cfg)
        
        logger.info("Experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        if wandb_run:
            wandb.log({"experiment_status": "failed", "error": str(e)})
        raise
    finally:
        if wandb_run:
            wandb.finish()

def run_comprehensive_comparison(cfg: DictConfig, wandb_run: Optional[Any]) -> Dict[str, Any]:
    """Run comprehensive comparison between active learning and surrogate methods."""
    
    # Extract experiment parameters
    n_variables = cfg.experiment.environment.num_variables
    n_runs = cfg.get('n_runs', 3)
    intervention_budget = cfg.experiment.target.max_interventions
    
    logger.info(f"Running {n_runs} experiments with {n_variables} variables, {intervention_budget} interventions")
    
    # Generate test SCMs
    scm_data = generate_test_scms(cfg)
    logger.info(f"Generated {len(scm_data)} test SCMs")
    
    # Log SCM metadata to WandB if enabled
    if wandb_run:
        for scm_name, scm in scm_data:
            log_scm_metadata_to_wandb(scm_name, scm, wandb_run)
    
    # Define methods to compare (3 clean methods without cheats)
    methods_to_compare = {
        "Random Policy + Untrained Model": ACBOMethodType.RANDOM_UNTRAINED,
        "Random Policy + Learning Model": ACBOMethodType.RANDOM_LEARNING,
        "Oracle Policy + Learning Model": ACBOMethodType.ORACLE_LEARNING,
    }
    
    # Run experiments for each method
    all_results = {}
    method_metrics = {}
    global_step_counter = 0  # Track global step counter for WandB
    
    for method_idx, (method_name, method_type) in enumerate(methods_to_compare.items()):
        logger.info(f"\\n=== Running {method_name} (Type: {method_type}) ===")
        
        method_results = []
        method_trajectories = []
        
        for run_idx in range(n_runs):
            for scm_idx, (scm_name, scm) in enumerate(scm_data):
                logger.info(f"  Run {run_idx + 1}/{n_runs}, SCM '{scm_name}' ({scm_idx + 1}/{len(scm_data)})")
                
                # Run single experiment
                result = run_acbo_method(method_type, scm, cfg, run_idx, scm_idx)
                
                # Add SCM name to result for tracking
                if result:
                    result['scm_name'] = scm_name
                    result['scm_type'] = scm.get('metadata', {}).get('structure_type', 'unknown')
                
                # Debug logging for method results
                if result:
                    logger.info(f"    {method_name} - Run {run_idx+1} SCM {scm_idx+1}: SUCCESS")
                    logger.debug(f"    Result keys: {list(result.keys())}")
                    logger.debug(f"    Method name in result: {result.get('method_name', 'MISSING')}")
                else:
                    logger.warning(f"    {method_name} - Run {run_idx+1} SCM {scm_idx+1}: FAILED - Empty result")
                
                method_results.append(result)
                
                # Extract trajectory metrics
                if result and 'detailed_results' in result:
                    trajectory_metrics = extract_trajectory_from_result(result, scm)
                    method_trajectories.append(trajectory_metrics)
                    
                    # Log per-run metrics to WandB
                    if wandb_run:
                        # Remove step offset - each method should show steps 0-15
                        global_step_counter = log_run_metrics_to_wandb(
                            result, trajectory_metrics, method_name, 
                            run_idx, scm_idx, wandb_run, 0, global_step_counter,
                            scm_name=scm_name, scm_type=result.get('scm_type', 'unknown')
                        )
        
        # Aggregate results for this method
        valid_results = [r for r in method_results if r]
        all_results[method_name] = valid_results
        
        # Debug logging for method completion
        logger.info(f"=== {method_name} COMPLETED ===")
        logger.info(f"    Total runs attempted: {len(method_results)}")
        logger.info(f"    Valid results: {len(valid_results)}")
        logger.info(f"    Success rate: {len(valid_results)/len(method_results)*100:.1f}%")
        
        if method_trajectories:
            # Compute learning curves
            learning_curves = extract_learning_curves({method_name: method_trajectories})
            extracted_metrics = learning_curves.get(method_name, {})
            method_metrics[method_name] = extracted_metrics
            logger.info(f"    Extracted metrics for {method_name}: {list(extracted_metrics.keys())}")
            
            # Log method summary to WandB
            if wandb_run:
                log_method_summary_to_wandb(method_name, method_results, learning_curves, wandb_run, global_step_counter)
                global_step_counter += 1  # Increment for next method
        else:
            logger.warning(f"    No trajectories found for {method_name}")
            # Still log summary even without trajectories
            if wandb_run:
                log_method_summary_to_wandb(method_name, method_results, {}, wandb_run, global_step_counter)
                global_step_counter += 1
    
    # Statistical comparison
    comparison_stats = compute_statistical_comparison(all_results)
    
    # Debug logging before plotting
    logger.info(f"\\n=== FINAL RESULTS SUMMARY ===")
    expected_methods = list(methods_to_compare.keys())
    for method_name, results in all_results.items():
        logger.info(f"{method_name}: {len(results)} valid results")
    logger.info(f"Method metrics keys: {list(method_metrics.keys())}")
    
    # Validation check
    missing_from_results = set(expected_methods) - set(all_results.keys())
    missing_from_metrics = set(expected_methods) - set(method_metrics.keys())
    
    if missing_from_results:
        logger.error(f"Missing methods from all_results: {missing_from_results}")
    if missing_from_metrics:
        logger.error(f"Missing methods from method_metrics: {missing_from_metrics}")
    
    # Generate visualizations
    plots = generate_comparison_plots(method_metrics, all_results, cfg)
    
    return {
        'method_results': all_results,
        'method_metrics': method_metrics,
        'comparison_stats': comparison_stats,
        'plots': plots,
        'experiment_config': OmegaConf.to_container(cfg),
        'scms_tested': len(scm_data),
        'runs_per_method': n_runs
    }

def log_scm_metadata_to_wandb(scm_name: str, scm: pyr.PMap, wandb_run: Any) -> None:
    """Log SCM characteristics to WandB for analysis."""
    from causal_bayes_opt.experiments.benchmark_scms import get_scm_characteristics
    
    try:
        characteristics = get_scm_characteristics(scm)
        
        scm_info = {
            f"scm/{scm_name}/num_variables": characteristics['num_variables'],
            f"scm/{scm_name}/num_edges": characteristics['num_edges'],
            f"scm/{scm_name}/edge_density": characteristics['edge_density'],
            f"scm/{scm_name}/target_variable": characteristics['target_variable'],
            f"scm/{scm_name}/num_target_parents": characteristics['num_target_parents'],
            f"scm/{scm_name}/structure_type": characteristics['structure_type'],
            f"scm/{scm_name}/complexity": characteristics['complexity'],
            f"scm/{scm_name}/description": characteristics['description'],
        }
        
        # Log coefficients as a formatted string for readability
        if characteristics['coefficients']:
            coeff_str = ', '.join([f"{k}:{v:.2f}" for k, v in characteristics['coefficients'].items()])
            scm_info[f"scm/{scm_name}/coefficients"] = coeff_str
        
        wandb_run.log(scm_info)
        logger.info(f"Logged metadata for SCM '{scm_name}' to WandB")
        
    except Exception as e:
        logger.warning(f"Failed to log SCM metadata for '{scm_name}': {e}")


def generate_test_scms(cfg: DictConfig) -> List[Tuple[str, pyr.PMap]]:
    """Generate test SCMs based on configuration.
    
    Returns:
        List of (scm_name, scm) tuples for tracking and logging
    """
    
    # Check if predefined SCM suite is enabled
    if cfg.experiment.get('scm_suite', {}).get('enabled', False):
        # Use predefined SCM suite
        from causal_bayes_opt.experiments.benchmark_scms import create_scm_suite
        
        scm_suite = create_scm_suite()
        selected_scms = cfg.experiment.scm_suite.get('scm_names', list(scm_suite.keys())[:3])
        
        logger.info(f"Using predefined SCM suite with SCMs: {selected_scms}")
        return [(name, scm_suite[name]) for name in selected_scms if name in scm_suite]
    
    else:
        # Use random generation (current behavior)
        n_variables = cfg.experiment.environment.num_variables
        edge_density = cfg.experiment.problem.edge_density
        n_scms = cfg.get('n_scms', 2)  # Number of different SCMs to test
        
        scms = []
        for i in range(n_scms):
            scm = create_erdos_renyi_scm(
                n_nodes=n_variables,
                edge_prob=edge_density,
                noise_scale=cfg.experiment.environment.noise_scale,
                seed=cfg.seed + i
            )
            scm_name = f"erdos_renyi_{i}"
            scms.append((scm_name, scm))
        
        logger.info(f"Using random SCM generation with {n_scms} SCMs")
        return scms


def run_acbo_method(
    method_type: str,
    scm: pyr.PMap,
    cfg: DictConfig,
    run_idx: int,
    scm_idx: int
) -> Dict[str, Any]:
    """Run ACBO methods (static or learning surrogate)."""
    
    try:
        # Import ACBO components with proper path
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        
        from examples.demo_learning import DemoConfig
        from examples.complete_workflow_demo import (
            run_progressive_learning_demo_with_scm,
            run_progressive_learning_demo_with_oracle_interventions
        )
        
        # Create ACBO config
        acbo_config = DemoConfig(
            n_observational_samples=10,
            n_intervention_steps=cfg.experiment.target.max_interventions,
            learning_rate=1e-3,
            scoring_method='bic',  # Use 'bic' instead of 'marginal_likelihood'
            intervention_value_range=(-2.0, 2.0),
            random_seed=cfg.seed + run_idx * 100 + scm_idx
        )
        
        if method_type == ACBOMethodType.RANDOM_UNTRAINED:
            # Run random interventions with untrained surrogate (no learning)
            result = run_random_untrained_demo(scm, acbo_config)
            
        elif method_type == ACBOMethodType.RANDOM_LEARNING:
            # Run random interventions with learning surrogate
            result = run_progressive_learning_demo_with_scm(scm, acbo_config)
            
        elif method_type == ACBOMethodType.ORACLE_LEARNING:
            # Run oracle interventions with learning surrogate
            result = run_progressive_learning_demo_with_oracle_interventions(scm, acbo_config)
            
        else:
            logger.error(f"Unknown ACBO method: {method_type}")
            return {}
        
        if not result:
            logger.error(f"ACBO method {method_type} returned empty result")
            return {}
        
        # Convert to standard format
        standardized_result = {
            'method_name': method_type,
            'final_target_value': result.get('final_best', 0.0),
            'target_improvement': result.get('improvement', 0.0),
            'structure_accuracy': compute_structure_accuracy_from_result(result, scm),
            'sample_efficiency': result.get('improvement', 0.0) / result.get('total_samples', 1),
            'intervention_count': len(result.get('learning_history', [])),
            'convergence_steps': len(result.get('target_progress', [])),
            'detailed_results': {
                'learning_history': result.get('learning_history', [])
            },
            'metadata': {
                'method': result.get('method', method_type),
                'total_samples': result.get('total_samples', 0),
                'final_uncertainty': result.get('final_uncertainty', 0.0),
                'converged_to_truth': result.get('converged_to_truth', False)
            },
            'run_idx': run_idx,
            'scm_idx': scm_idx
        }
        
        # Add final marginal probabilities for structure accuracy calculation
        final_marginals = result.get('final_marginal_probs', {})
        if final_marginals:
            standardized_result['final_marginal_probs'] = final_marginals
        else:
            # Fallback: empty marginals if not available
            logger.debug(f"No final_marginal_probs found for ACBO method {method_type}")
            standardized_result['final_marginal_probs'] = {}
        
        return standardized_result
        
    except Exception as e:
        logger.error(f"Failed to run ACBO method {method_type}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def run_random_untrained_demo(scm: pyr.PMap, acbo_config) -> Dict[str, Any]:
    """Run random interventions with untrained surrogate model (no learning)."""
    
    try:
        from examples.demo_learning import DemoConfig, create_random_intervention_policy
        from examples.complete_workflow_demo import generate_initial_data
        from causal_bayes_opt.data_structures.scm import get_variables, get_target
        from causal_bayes_opt.environments.sampling import sample_with_intervention
        
        variables = sorted(get_variables(scm))
        target = get_target(scm)
        
        # Generate initial observational data
        key = jax.random.PRNGKey(acbo_config.random_seed)
        initial_samples, buffer = generate_initial_data(scm, acbo_config, key)
        
        # Create random intervention policy
        intervention_fn = create_random_intervention_policy(
            variables, target, acbo_config.intervention_value_range
        )
        
        # Run interventions WITHOUT learning (untrained model)
        learning_history = []
        target_progress = []
        
        # Handle initial target value
        if initial_samples:
            target_values = [sample.get(target, 0.0) for sample in initial_samples]
            initial_target = max(target_values)  # Best from initial data
        else:
            initial_target = 0.0
        
        best_value = initial_target
        
        for step in range(acbo_config.n_intervention_steps):
            # Generate random intervention
            key, subkey = jax.random.split(key)
            intervention = intervention_fn(key=subkey)
            
            # Apply intervention and get outcome
            outcome_samples = sample_with_intervention(scm, intervention, n_samples=1, seed=int(subkey[0]))
            outcome_sample = outcome_samples[0]  # Extract the single sample from the list
            outcome_value = outcome_sample['values'].get(target, 0.0)
            
            # Track progress
            if outcome_value > best_value:
                best_value = outcome_value
            
            target_improvement = best_value - initial_target
            
            step_info = {
                'step': step,
                'intervention': intervention,
                'outcome_value': outcome_value,
                'target_improvement': target_improvement,
                'uncertainty': 0.0,  # No learning = no uncertainty tracking
                'marginals': {}  # No learning = no structure learning
            }
            
            learning_history.append(step_info)
            target_progress.append(outcome_value)
        
        # Return result in standard format
        return {
            'method': 'random_untrained',
            'learning_history': learning_history,
            'target_progress': target_progress,
            'final_best': best_value,
            'improvement': best_value - initial_target,
            'total_samples': acbo_config.n_observational_samples + acbo_config.n_intervention_steps,
            'final_uncertainty': 0.0,
            'converged_to_truth': False,
            'final_marginal_probs': {}  # No structure learning
        }
        
    except Exception as e:
        logger.error(f"Random untrained demo failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def compute_structure_accuracy_from_result(result: Dict[str, Any], scm: pyr.PMap) -> float:
    """Compute structure learning accuracy from ACBO result."""
    
    try:
        target = get_target(scm)
        if not target:
            return 0.0
        
        true_parents = set(get_parents(scm, target))
        final_marginals = result.get('final_marginal_probs', {})
        
        # Debug logging to understand the issue
        logger.debug(f"Structure accuracy - Target: {target}, True parents: {true_parents}")
        logger.debug(f"Final marginals type: {type(final_marginals)}, content: {final_marginals}")
        
        if not final_marginals:
            logger.warning(f"No final_marginal_probs found in result. Available keys: {list(result.keys())}")
            return 0.0
        
        # Compute accuracy based on thresholded predictions
        correct = 0
        total = 0
        
        for var, prob in final_marginals.items():
            if var != target:
                predicted_parent = prob > 0.5
                is_true_parent = var in true_parents
                
                if predicted_parent == is_true_parent:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
        
    except Exception as e:
        logger.warning(f"Failed to compute structure accuracy: {e}")
        return 0.0

def extract_trajectory_from_result(result: Dict[str, Any], scm: pyr.PMap) -> Dict[str, List[float]]:
    """Extract trajectory metrics from experiment result."""
    
    target = get_target(scm)
    true_parents = list(get_parents(scm, target)) if target else []
    
    # Use existing analysis infrastructure
    try:
        from causal_bayes_opt.analysis.trajectory_metrics import extract_metrics_from_experiment_result
        return extract_metrics_from_experiment_result(result, true_parents)
    except Exception as e:
        logger.warning(f"Failed to extract trajectory metrics: {e}")
        
        # Fallback: create minimal trajectory from learning history
        learning_history = result.get('detailed_results', {}).get('learning_history', [])
        
        if not learning_history:
            return {
                'steps': [0],
                'true_parent_likelihood': [0.0],
                'f1_scores': [0.0],
                'shd_values': [len(true_parents)],  # Worst case SHD
                'target_values': [result.get('final_target_value', 0.0)],
                'uncertainty_bits': [0.0],
                'intervention_counts': [result.get('intervention_count', 0)]
            }
        
        steps = [step_data['step'] for step_data in learning_history]
        target_values = [step_data['outcome_value'] for step_data in learning_history]
        improvements = [step_data['target_improvement'] for step_data in learning_history]
        
        return {
            'steps': steps,
            'true_parent_likelihood': [0.0] * len(steps),  # Baselines don't track this
            'f1_scores': [0.0] * len(steps),  # Baselines don't track this
            'shd_values': [len(true_parents)] * len(steps),  # Worst case SHD for baselines
            'target_values': target_values,
            'uncertainty_bits': [0.0] * len(steps),
            'intervention_counts': steps  # Use step as proxy for intervention count
        }

def log_run_metrics_to_wandb(
    result: Dict[str, Any],
    trajectory_metrics: Dict[str, List[float]],
    method_name: str,
    run_idx: int,
    scm_idx: int,
    wandb_run: Any,
    step_offset: int = 0,
    global_step_counter: int = 0,
    scm_name: str = "unknown",
    scm_type: str = "unknown"
) -> int:
    """Log metrics from a single run to WandB.
    
    Returns:
        Updated global step counter
    """
    
    if not trajectory_metrics or not trajectory_metrics.get('steps'):
        return global_step_counter
    
    steps = trajectory_metrics['steps']
    
    # Log trajectory data with custom x-axis
    for i, step in enumerate(steps):
        # Base metrics
        target_value = trajectory_metrics['target_values'][i] if i < len(trajectory_metrics['target_values']) else 0
        f1_score = trajectory_metrics['f1_scores'][i] if i < len(trajectory_metrics['f1_scores']) else 0
        shd_value = trajectory_metrics['shd_values'][i] if i < len(trajectory_metrics['shd_values']) else 0
        true_parent_likelihood = trajectory_metrics['true_parent_likelihood'][i] if i < len(trajectory_metrics['true_parent_likelihood']) else 0
        uncertainty = trajectory_metrics['uncertainty_bits'][i] if i < len(trajectory_metrics['uncertainty_bits']) else 0
        
        metrics = {
            # General metrics (existing)
            f"{method_name}/target_value": target_value,
            f"{method_name}/true_parent_likelihood": true_parent_likelihood,
            f"{method_name}/f1_score": f1_score,
            f"{method_name}/shd": shd_value,
            f"{method_name}/uncertainty": uncertainty,
            
            # SCM-specific metrics (NEW)
            f"{method_name}/target_value_{scm_name}": target_value,
            f"{method_name}/f1_score_{scm_name}": f1_score,
            f"{method_name}/shd_{scm_name}": shd_value,
            f"{scm_type}/target_value": target_value,
            f"{scm_type}/f1_score": f1_score,
            f"{scm_type}/shd": shd_value,
            
            # Context information
            "intervention_step": step,  # Custom x-axis for proper visualization
            "run_idx": run_idx,
            "scm_idx": scm_idx,
            "scm_name": scm_name,
            "scm_type": scm_type,
            "method": method_name
        }
        
        # Log metrics with custom intervention_step field for proper time-series visualization
        # WandB will use the intervention_step field for our custom metrics
        wandb_run.log(metrics)
    
    # Log final summary metrics for this run
    summary_metrics = {
        f"{method_name}/final_target_improvement": result.get('target_improvement', 0),
        f"{method_name}/final_structure_accuracy": result.get('structure_accuracy', 0),
        f"{method_name}/sample_efficiency": result.get('sample_efficiency', 0),
        f"{method_name}/convergence_steps": result.get('convergence_steps', 0),
        "run_idx": run_idx,
        "scm_idx": scm_idx
    }
    
    # Log summary metrics without step parameter to avoid conflicts with trajectory visualization
    wandb_run.log(summary_metrics)
    
    return global_step_counter

def log_method_summary_to_wandb(
    method_name: str,
    method_results: List[Dict[str, Any]],
    learning_curves: Dict[str, Dict[str, Any]],
    wandb_run: Any,
    step: int = None
) -> None:
    """Log aggregated method summary to WandB."""
    
    if not method_results:
        return
    
    # Compute aggregate statistics
    target_improvements = [r.get('target_improvement', 0) for r in method_results if r]
    structure_accuracies = [r.get('structure_accuracy', 0) for r in method_results if r]
    sample_efficiencies = [r.get('sample_efficiency', 0) for r in method_results if r]
    convergence_steps = [r.get('convergence_steps', 0) for r in method_results if r]
    
    summary = {
        f"summary/{method_name}_target_improvement_mean": onp.mean(target_improvements) if target_improvements else 0,
        f"summary/{method_name}_target_improvement_std": onp.std(target_improvements) if target_improvements else 0,
        f"summary/{method_name}_structure_accuracy_mean": onp.mean(structure_accuracies) if structure_accuracies else 0,
        f"summary/{method_name}_structure_accuracy_std": onp.std(structure_accuracies) if structure_accuracies else 0,
        f"summary/{method_name}_sample_efficiency_mean": onp.mean(sample_efficiencies) if sample_efficiencies else 0,
        f"summary/{method_name}_sample_efficiency_std": onp.std(sample_efficiencies) if sample_efficiencies else 0,
        f"summary/{method_name}_convergence_steps_mean": onp.mean(convergence_steps) if convergence_steps else 0,
        f"summary/{method_name}_convergence_steps_std": onp.std(convergence_steps) if convergence_steps else 0,
        f"summary/{method_name}_n_runs": len(method_results)
    }
    
    if step is not None:
        wandb_run.log(summary, step=step)
    else:
        wandb_run.log(summary)
    
    # Note: Learning curves are already logged in log_run_metrics_to_wandb with proper step tracking

def compute_statistical_comparison(all_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Compute statistical comparison between methods."""
    
    if not SCIPY_AVAILABLE:
        logger.warning("scipy not available - skipping statistical tests")
        return {}
    
    comparison_stats = {}
    method_names = list(all_results.keys())
    
    # Compare each pair of methods
    for i, method1 in enumerate(method_names):
        for j, method2 in enumerate(method_names[i+1:], i+1):
            
            results1 = all_results[method1]
            results2 = all_results[method2]
            
            # Extract target improvements
            improvements1 = [r.get('target_improvement', 0) for r in results1 if r]
            improvements2 = [r.get('target_improvement', 0) for r in results2 if r]
            
            if len(improvements1) > 1 and len(improvements2) > 1:
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(improvements1, improvements2)
                
                comparison_stats[f"{method1}_vs_{method2}"] = {
                    'method1_mean': onp.mean(improvements1),
                    'method1_std': onp.std(improvements1),
                    'method2_mean': onp.mean(improvements2),
                    'method2_std': onp.std(improvements2),
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'effect_size': (onp.mean(improvements1) - onp.mean(improvements2)) / onp.sqrt((onp.var(improvements1) + onp.var(improvements2)) / 2)
                }
    
    return comparison_stats

def generate_comparison_plots(
    method_metrics: Dict[str, Dict[str, Any]],
    all_results: Dict[str, List[Dict[str, Any]]],
    cfg: DictConfig
) -> Dict[str, str]:
    """Generate comparison plots and return paths."""
    
    plots_dir = Path("experiment_plots")
    plots_dir.mkdir(exist_ok=True)
    
    plot_paths = {}
    
    try:
        # Method comparison plot
        if method_metrics:
            logger.info(f"Generating method comparison plot with methods: {list(method_metrics.keys())}")
            comparison_path = plots_dir / "method_comparison.png"
            plot_method_comparison(
                method_metrics,
                title="Active Learning vs Surrogate Methods Comparison",
                save_path=str(comparison_path),
                metrics=['shd', 'f1', 'target']
            )
            plot_paths['method_comparison'] = str(comparison_path)
        else:
            logger.warning("No method_metrics available for plotting")
        
        # Intervention efficiency plot
        efficiency_results = {}
        for method_name, results in all_results.items():
            efficiency_data = []
            for result in results:
                if result:
                    efficiency_data.append({
                        'steps_to_threshold': result.get('convergence_steps'),
                        'efficiency_ratio': result.get('sample_efficiency', 0),
                        'interventions_to_threshold': result.get('intervention_count')
                    })
            efficiency_results[method_name] = efficiency_data
        
        if efficiency_results:
            efficiency_path = plots_dir / "intervention_efficiency.png"
            plot_intervention_efficiency(
                efficiency_results,
                title="Intervention Efficiency Comparison",
                save_path=str(efficiency_path)
            )
            plot_paths['intervention_efficiency'] = str(efficiency_path)
        
        logger.info(f"Generated {len(plot_paths)} comparison plots")
        
    except Exception as e:
        logger.error(f"Failed to generate plots: {e}")
    
    return plot_paths

def log_final_comparison_metrics(results: Dict[str, Any], wandb_run: Any) -> None:
    """Log final comparison summary to WandB."""
    
    comparison_stats = results.get('comparison_stats', {})
    
    # Log statistical comparison results
    for comparison_name, stats in comparison_stats.items():
        comparison_metrics = {
            f"comparison/{comparison_name}_p_value": stats['p_value'],
            f"comparison/{comparison_name}_significant": stats['significant'],
            f"comparison/{comparison_name}_effect_size": stats['effect_size'],
            f"comparison/{comparison_name}_mean_diff": stats['method1_mean'] - stats['method2_mean']
        }
        wandb_run.log(comparison_metrics)
    
    # Log experiment summary
    experiment_summary = {
        "experiment/total_methods": len(results.get('method_results', {})),
        "experiment/scms_tested": results.get('scms_tested', 0),
        "experiment/runs_per_method": results.get('runs_per_method', 0),
        "experiment/total_experiments": sum(len(method_results) for method_results in results.get('method_results', {}).values()),
        "experiment/status": "completed"
    }
    wandb_run.log(experiment_summary)

def upload_plots_to_wandb(results: Dict[str, Any], wandb_run: Any) -> None:
    """Upload generated plots as WandB artifacts."""
    
    plots = results.get('plots', {})
    
    for plot_name, plot_path in plots.items():
        if Path(plot_path).exists():
            # Log as image
            wandb_run.log({f"plots/{plot_name}": wandb.Image(plot_path)})
            
            # Also save as artifact for archival
            artifact = wandb.Artifact(f"{plot_name}_plot", type="plot")
            artifact.add_file(plot_path)
            wandb_run.log_artifact(artifact)

def save_experiment_results(results: Dict[str, Any], cfg: DictConfig) -> None:
    """Save experiment results to local files."""
    
    timestamp = int(time.time())
    results_dir = Path("experiment_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save main results
    results_file = results_dir / f"acbo_comparison_{timestamp}.json"
    
    # Convert to JSON-serializable format
    json_results = {}
    for key, value in results.items():
        if key != 'plots':  # Skip plot paths
            try:
                json.dumps(value)  # Test if serializable
                json_results[key] = value
            except (TypeError, ValueError):
                json_results[key] = str(value)  # Convert to string if not serializable
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Saved experiment results to {results_file}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    run_acbo_comparison_experiment()
