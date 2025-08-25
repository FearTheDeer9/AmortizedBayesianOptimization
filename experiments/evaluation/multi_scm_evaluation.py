#!/usr/bin/env python3
"""
Multi-SCM evaluation script for comprehensive comparison of ACBO methods.

Evaluates multiple policy+surrogate combinations across randomly sampled SCMs
and generates plots showing average performance trajectories.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
from src.causal_bayes_opt.data_structures.scm import get_variables, get_target, get_parents
from src.causal_bayes_opt.evaluation.universal_evaluator import create_universal_evaluator
from src.causal_bayes_opt.evaluation.model_interfaces import (
    create_random_acquisition,
    create_optimal_oracle_acquisition
)
from src.causal_bayes_opt.analysis.trajectory_metrics import (
    compute_f1_score_from_marginals,
    compute_shd_from_marginals
)
from experiments.evaluation.core.model_loader import ModelLoader
from experiments.evaluation.simple_evaluation import load_surrogate_model

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(message)s')
logger = logging.getLogger(__name__)


def create_random_scms(n_scms: int = 10, seed: int = 42) -> List[Any]:
    """Create a diverse set of random SCMs for evaluation."""
    np.random.seed(seed)
    scm_factory = VariableSCMFactory(seed=seed, noise_scale=0.1)
    
    scms = []
    structure_types = ['fork', 'chain', 'collider', 'mixed', 'random']
    num_variables_options = [3, 4, 5, 6, 8]
    
    for i in range(n_scms):
        # Randomly select structure and size
        structure_type = np.random.choice(structure_types)
        num_vars = np.random.choice(num_variables_options)
        
        # Create SCM
        scm = scm_factory.create_variable_scm(
            num_variables=num_vars,
            structure_type=structure_type,
            target_variable=None,  # Auto-select target
            edge_density=0.5 if structure_type == 'random' else 0.5
        )
        
        # Add metadata about the SCM
        logger.info(f"SCM {i+1}: {num_vars} vars, {structure_type} structure, target={get_target(scm)}")
        scms.append(scm)
    
    return scms


def compute_metrics(buffer, scm, surrogate_predictions=None) -> Dict[str, float]:
    """Compute evaluation metrics for current state."""
    from src.causal_bayes_opt.data_structures.sample import get_values
    
    target = get_target(scm)
    true_parents = get_parents(scm, target)
    variables = get_variables(scm)
    
    metrics = {}
    
    # Get target values from buffer
    target_values = []
    for sample in buffer:
        values = get_values(sample)
        if target in values:
            target_values.append(values[target])
    
    if target_values:
        metrics['mean_target'] = float(np.mean(target_values))
        metrics['best_target'] = float(np.min(target_values))
        metrics['worst_target'] = float(np.max(target_values))
    
    # Compute structure learning metrics if surrogate available
    if surrogate_predictions and 'marginal_parent_probs' in surrogate_predictions:
        parent_probs = surrogate_predictions['marginal_parent_probs']
        
        # Convert probabilities to binary predictions (threshold = 0.5)
        predicted_parents = set()
        for var, prob in parent_probs.items():
            if prob > 0.5 and var != target:
                predicted_parents.add(var)
        
        # Compute F1, precision, recall
        tp = len(true_parents & predicted_parents)
        fp = len(predicted_parents - true_parents)
        fn = len(true_parents - predicted_parents)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics['f1_score'] = f1
        metrics['precision'] = precision
        metrics['recall'] = recall
        
        # Compute SHD (Structural Hamming Distance)
        # For simplicity, count edge differences
        shd = fp + fn  # Missing edges + extra edges
        metrics['shd'] = float(shd)
    
    return metrics


def normalize_target_value(value: float, scm: Any) -> float:
    """Normalize target value based on SCM's theoretical range."""
    # Get variable ranges from metadata if available
    metadata = scm.get('metadata', {})
    if 'variable_ranges' in metadata:
        target = get_target(scm)
        if target in metadata['variable_ranges']:
            min_val, max_val = metadata['variable_ranges'][target]
            # Normalize to [0, 1] where 0 is best (minimum)
            if max_val > min_val:
                return (value - min_val) / (max_val - min_val)
    
    # Fallback: use standard normalization
    return value


def evaluate_method_on_scm(
    method_name: str,
    acquisition_fn: callable,
    surrogate_fn: Optional[callable],
    scm: Any,
    config: Dict[str, Any]
) -> Dict[str, List[float]]:
    """Evaluate a single method on a single SCM."""
    
    evaluator = create_universal_evaluator()
    
    # Run evaluation
    results = evaluator.evaluate(
        acquisition_fn=acquisition_fn,
        scm=scm,
        config=config,
        surrogate_fn=surrogate_fn,
        seed=42
    )
    
    # Extract trajectory metrics
    trajectories = {
        'target_values': [],
        'normalized_targets': [],
        'f1_scores': [],
        'shd_values': [],
        'precision_values': [],
        'recall_values': []
    }
    
    # Extract from result object
    if hasattr(results, 'history'):  # Changed from 'trajectory'
        # Get true parents for structure metrics
        target = get_target(scm)
        true_parents = set(get_parents(scm, target))  # Convert to regular set
        
        # Extract target values from history
        for step_metric in results.history:
            trajectories['target_values'].append(step_metric.outcome_value)
            trajectories['normalized_targets'].append(
                normalize_target_value(step_metric.outcome_value, scm)
            )
            
            # Compute F1 and SHD from marginals if available
            if step_metric.marginals:
                f1 = compute_f1_score_from_marginals(
                    step_metric.marginals, true_parents
                )
                shd = compute_shd_from_marginals(
                    step_metric.marginals, true_parents
                )
                
                trajectories['f1_scores'].append(f1)
                trajectories['shd_values'].append(shd)
                
                # Compute precision and recall
                pred_parents = {var for var, prob in step_metric.marginals.items() 
                              if prob > 0.5}
                tp = len(true_parents & pred_parents)
                fp = len(pred_parents - true_parents)
                fn = len(true_parents - pred_parents)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                trajectories['precision_values'].append(precision)
                trajectories['recall_values'].append(recall)
            else:
                # No surrogate - fill with NaN
                trajectories['f1_scores'].append(np.nan)
                trajectories['shd_values'].append(np.nan)
                trajectories['precision_values'].append(np.nan)
                trajectories['recall_values'].append(np.nan)
    
    # Add debug logging
    logger.debug(f"  Extracted {len(trajectories['target_values'])} steps for {method_name}")
    if trajectories['target_values']:
        logger.debug(f"    Target values (first 3): {trajectories['target_values'][:3]}")
        logger.debug(f"    F1 scores (first 3): {trajectories['f1_scores'][:3]}")
    
    return trajectories


def run_multi_scm_evaluation(
    policy_path: Optional[Path] = None,
    surrogate_path: Optional[Path] = None,
    n_scms: int = 10,
    seed: int = 42
) -> Dict[str, Any]:
    """Run evaluation across multiple SCMs."""
    
    # Create random SCMs
    logger.info(f"Creating {n_scms} random SCMs...")
    scms = create_random_scms(n_scms, seed)
    
    # Evaluation config
    config = {
        'n_observational': 50,
        'max_interventions': 20,  # Full evaluation
        'n_intervention_samples': 1,
        'optimization_direction': 'MINIMIZE'
    }
    
    # Results storage
    all_results = {
        'Random': [],
        'Oracle': [],
        'Trained Policy': [],
        'Trained Policy + Surrogate': []
    }
    
    # Load models once
    logger.info("Loading models...")
    
    # Load surrogate if available
    surrogate_fn = None
    if surrogate_path and surrogate_path.exists():
        surrogate_fn = load_surrogate_model(surrogate_path)
        logger.info("Loaded surrogate model")
    
    # Evaluate each SCM
    for i, scm in enumerate(scms):
        logger.info(f"\nEvaluating SCM {i+1}/{n_scms}...")
        
        target = get_target(scm)
        num_vars = len(get_variables(scm))
        logger.info(f"  Variables: {num_vars}, Target: {target}")
        
        # Extract SCM metadata for quantile policies
        scm_metadata = scm.get('metadata', {}) if hasattr(scm, 'get') else {}
        
        # Method 1: Random baseline
        logger.info("  Running Random baseline...")
        random_fn = create_random_acquisition(seed=seed+i)
        random_results = evaluate_method_on_scm(
            "Random", random_fn, None, scm, config
        )
        all_results['Random'].append(random_results)
        
        # Method 2: Oracle baseline
        logger.info("  Running Oracle baseline...")
        oracle_fn = create_optimal_oracle_acquisition(scm)
        oracle_results = evaluate_method_on_scm(
            "Oracle", oracle_fn, None, scm, config
        )
        all_results['Oracle'].append(oracle_results)
        
        # Method 3: Trained Policy (no surrogate)
        if policy_path and policy_path.exists():
            logger.info("  Running Trained Policy...")
            policy_fn = ModelLoader.load_policy(
                policy_path, seed=seed+i, scm_metadata=scm_metadata
            )
            policy_results = evaluate_method_on_scm(
                "Trained Policy", policy_fn, None, scm, config
            )
            all_results['Trained Policy'].append(policy_results)
            
            # Method 4: Trained Policy + Surrogate
            if surrogate_fn:
                logger.info("  Running Trained Policy + Surrogate...")
                policy_surrogate_results = evaluate_method_on_scm(
                    "Trained Policy + Surrogate", policy_fn, surrogate_fn, scm, config
                )
                all_results['Trained Policy + Surrogate'].append(policy_surrogate_results)
    
    return all_results, scms


def compute_average_trajectories(results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Compute mean and std trajectories across SCMs."""
    averaged = {}
    
    for method_name, method_results in results.items():
        if not method_results:
            continue
            
        # Stack trajectories from all SCMs
        all_targets = []
        all_normalized = []
        all_f1 = []
        all_shd = []
        
        for scm_result in method_results:
            all_normalized.append(scm_result['normalized_targets'])
            all_f1.append(scm_result['f1_scores'])
            all_shd.append(scm_result['shd_values'])
        
        # Convert to numpy arrays and compute statistics
        all_normalized = np.array(all_normalized)
        all_f1 = np.array(all_f1)
        all_shd = np.array(all_shd)
        
        averaged[method_name] = {
            'normalized_mean': np.nanmean(all_normalized, axis=0),
            'normalized_std': np.nanstd(all_normalized, axis=0),
            'f1_mean': np.nanmean(all_f1, axis=0),
            'f1_std': np.nanstd(all_f1, axis=0),
            'shd_mean': np.nanmean(all_shd, axis=0),
            'shd_std': np.nanstd(all_shd, axis=0)
        }
    
    return averaged


def save_results(results: Dict, scms: List, output_dir: Path):
    """Save evaluation results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw results as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert results to serializable format
    serializable_results = {}
    for method, method_results in results.items():
        serializable_results[method] = []
        for scm_result in method_results:
            serializable_scm = {}
            for key, values in scm_result.items():
                serializable_scm[key] = [float(v) if not np.isnan(v) else None for v in values]
            serializable_results[method].append(serializable_scm)
    
    with open(output_dir / f"results_{timestamp}.json", 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Save summary statistics as CSV
    averaged = compute_average_trajectories(results)
    summary_data = []
    
    for method_name, metrics in averaged.items():
        # Final performance (last intervention)
        summary_data.append({
            'Method': method_name,
            'Final_Normalized_Target': metrics['normalized_mean'][-1] if len(metrics['normalized_mean']) > 0 else np.nan,
            'Final_F1': metrics['f1_mean'][-1] if len(metrics['f1_mean']) > 0 else np.nan,
            'Final_SHD': metrics['shd_mean'][-1] if len(metrics['shd_mean']) > 0 else np.nan,
            'Min_Normalized_Target': np.min(metrics['normalized_mean']) if len(metrics['normalized_mean']) > 0 else np.nan,
            'Max_F1': np.nanmax(metrics['f1_mean']) if len(metrics['f1_mean']) > 0 else np.nan
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / f"summary_{timestamp}.csv", index=False)
    
    logger.info(f"Results saved to {output_dir}")
    return timestamp


if __name__ == "__main__":
    # Define checkpoint paths
    project_root = Path(__file__).parent.parent.parent
    policy_path = project_root / "checkpoints/grpo_runs/grpo_multi_scm_20250825_140559/final_policy.pkl"
    surrogate_path = project_root / "experiments/surrogate-only-training/scripts/checkpoints/avici_runs/avici_style_20250822_213115/checkpoint_step_200.pkl"
    
    # Run evaluation
    logger.info("Starting multi-SCM evaluation...")
    results, scms = run_multi_scm_evaluation(
        policy_path=policy_path,
        surrogate_path=surrogate_path,
        n_scms=10,  # Full evaluation
        seed=42
    )
    
    # Save results
    output_dir = Path(__file__).parent / "results"  # Fixed path
    timestamp = save_results(results, scms, output_dir)
    
    # Generate plots (will be implemented in plotting_utils.py)
    logger.info("Generating plots...")
    from experiments.evaluation.core.plotting_utils import plot_evaluation_results
    plot_evaluation_results(results, output_dir, timestamp)
    
    logger.info("Evaluation complete!")