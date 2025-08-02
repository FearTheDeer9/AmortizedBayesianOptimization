#!/usr/bin/env python3
"""
Simplified evaluation script that runs ACBO comparison without Hydra complications.
This script is designed to work directly with the evaluation notebook.
"""

import argparse
import sys
import json
import time
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random

# Import experiment modules
from causal_bayes_opt.experiments.test_scms import get_fork_scm, get_chain_scm, get_collider_scm
from causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
from causal_bayes_opt.core.acbo_comparison import compare_acbo_methods


def generate_test_scms(num_scms: int, seed: int = 42) -> List[Any]:
    """Generate test SCMs for evaluation."""
    logger.info(f"Generating {num_scms} test SCMs...")
    
    scm_factory = VariableSCMFactory(seed=seed)
    test_scms = []
    
    # Generate balanced set
    structures = ['fork', 'chain', 'collider', 'mixed']
    variable_counts = [3, 4, 5, 6]
    
    scm_idx = 0
    while len(test_scms) < num_scms:
        structure = structures[scm_idx % len(structures)]
        n_vars = variable_counts[scm_idx % len(variable_counts)]
        
        scm = scm_factory.create_scm(
            num_variables=n_vars,
            structure_type=structure,
            noise_scale=1.0,
            intervention_targets=None
        )
        test_scms.append(scm)
        scm_idx += 1
    
    logger.info(f"Generated {len(test_scms)} SCMs")
    return test_scms


def run_simple_comparison(
    checkpoint_path: Path,
    num_scms: int = 3,
    runs_per_method: int = 3,
    intervention_budget: int = 10,
    output_dir: Path = None
) -> Dict[str, Any]:
    """Run a simple ACBO comparison without Hydra."""
    
    logger.info("🚀 Running Simple ACBO Comparison")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"SCMs: {num_scms}, Runs: {runs_per_method}, Budget: {intervention_budget}")
    
    # Load checkpoint
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load metadata
    metadata_path = checkpoint_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        optimization_direction = metadata.get('optimization_config', {}).get('direction', 'MINIMIZE')
    else:
        optimization_direction = 'MINIMIZE'
    
    logger.info(f"Optimization direction: {optimization_direction}")
    
    # Generate test SCMs
    test_scms = generate_test_scms(num_scms)
    
    # Set up methods to compare
    methods = {
        "Random + Untrained": "random_untrained",
        "Random + Learning": "random_learning",
        "Oracle + Learning": "oracle_learning"
    }
    
    # Try to add trained policy if checkpoint has model parameters
    checkpoint_pkl = checkpoint_path / "checkpoint.pkl"
    if checkpoint_pkl.exists():
        methods["Trained Policy + Learning"] = "learned_enriched_policy"
        logger.info("✅ Added trained policy to comparison")
    else:
        logger.info("⚠️ No checkpoint.pkl found, skipping trained policy")
    
    # Run comparison
    start_time = time.time()
    
    try:
        # Create config for comparison
        config = {
            'experiment': {
                'runs_per_method': runs_per_method,
                'intervention_budget': intervention_budget,
                'target': {
                    'max_interventions': intervention_budget
                }
            },
            'seed': 42,
            'policy_checkpoint_path': str(checkpoint_path) if checkpoint_pkl.exists() else None,
            'logging': {
                'level': 'INFO'
            }
        }
        
        # Convert to proper config object
        from omegaconf import DictConfig
        config = DictConfig(config)
        
        # Run comparison
        results = compare_acbo_methods(
            scms=test_scms,
            methods=methods,
            runs_per_method=runs_per_method,
            intervention_budget=intervention_budget,
            seed=42
        )
        
        duration = time.time() - start_time
        
        # Process results
        processed_results = {
            'execution_metadata': {
                'total_time': duration,
                'methods_tested': len(methods),
                'scms_tested': num_scms,
                'total_experiments': len(methods) * num_scms * runs_per_method,
                'optimization_direction': optimization_direction
            },
            'method_results': {},
            'statistical_analysis': {
                'summary_statistics': {},
                'pairwise_comparisons': {}
            }
        }
        
        # Extract method results
        for method_name, method_results in results.items():
            processed_results['method_results'][method_name] = []
            
            for result in method_results:
                if result['success']:
                    processed_results['method_results'][method_name].append({
                        'scm_idx': result.get('scm_idx', 0),
                        'run_idx': result.get('run_idx', 0),
                        'success': True,
                        'target_improvement': result.get('target_improvement', 0.0),
                        'structure_accuracy': result.get('structure_accuracy', 0.0),
                        'convergence_steps': result.get('convergence_steps', intervention_budget),
                        'detailed_results': {
                            'target_progress': result.get('target_values', []),
                            'f1_scores': result.get('f1_scores', []),
                            'shd_values': result.get('shd_values', []),
                            'steps': list(range(1, len(result.get('target_values', [])) + 1))
                        }
                    })
        
        # Calculate summary statistics
        for method_name, method_runs in processed_results['method_results'].items():
            improvements = [r['target_improvement'] for r in method_runs if r['success']]
            accuracies = [r['structure_accuracy'] for r in method_runs if r['success']]
            
            processed_results['statistical_analysis']['summary_statistics'][method_name] = {
                'target_improvement_mean': np.mean(improvements) if improvements else 0.0,
                'target_improvement_std': np.std(improvements) if improvements else 0.0,
                'target_improvement_count': len(improvements),
                'structure_accuracy_mean': np.mean(accuracies) if accuracies else 0.0,
                'structure_accuracy_std': np.std(accuracies) if accuracies else 0.0
            }
        
        logger.info(f"✅ Comparison completed in {duration:.1f} seconds")
        
        # Save results
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            results_file = output_dir / "comparison_results.json"
            
            with open(results_file, 'w') as f:
                json.dump(processed_results, f, indent=2, default=str)
            
            logger.info(f"✅ Results saved to: {results_file}")
        
        return processed_results
        
    except Exception as e:
        logger.error(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Simple ACBO evaluation without Hydra complications"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to GRPO checkpoint"
    )
    parser.add_argument(
        "--num-scms",
        type=int,
        default=3,
        help="Number of test SCMs"
    )
    parser.add_argument(
        "--runs-per-method",
        type=int,
        default=3,
        help="Number of runs per method"
    )
    parser.add_argument(
        "--intervention-budget",
        type=int,
        default=10,
        help="Number of interventions per run"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Run comparison
    results = run_simple_comparison(
        checkpoint_path=Path(args.checkpoint),
        num_scms=args.num_scms,
        runs_per_method=args.runs_per_method,
        intervention_budget=args.intervention_budget,
        output_dir=Path(args.output_dir)
    )
    
    # Print summary
    print("\n📊 Results Summary:")
    for method_name, stats in results['statistical_analysis']['summary_statistics'].items():
        print(f"  {method_name}:")
        print(f"    Target improvement: {stats['target_improvement_mean']:.4f} ± {stats['target_improvement_std']:.4f}")
        print(f"    Structure accuracy: {stats['structure_accuracy_mean']:.4f}")


if __name__ == "__main__":
    main()