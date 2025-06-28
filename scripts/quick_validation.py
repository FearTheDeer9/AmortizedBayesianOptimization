#!/usr/bin/env python3
"""
Quick validation experiment with multiple runs for statistical significance.
Tests the core hypothesis: learning surrogate > static surrogate.
"""

import sys
from pathlib import Path

# Add project root to path  
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import time
import json
from scripts.erdos_renyi_scaling_experiment import run_single_experiment, ScalingExperimentConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_quick_validation():
    """Run quick validation with multiple seeds."""
    logger.info("Starting quick validation experiment")
    
    # Test configuration - reduced for speed
    config = ScalingExperimentConfig(
        min_nodes=5,
        max_nodes=5,  # Just 5 nodes
        edge_probability=0.3,
        n_intervention_steps=10,  # Reduced steps for speed
        n_observational_samples=50,  # Reduced samples
        learning_rate=1e-3,
        n_runs_per_config=5,  # 5 runs for statistics
        base_random_seed=42
    )
    
    all_results = []
    
    # Run multiple experiments
    for run_id in range(config.n_runs_per_config):
        for method in ['static_surrogate', 'learning_surrogate']:
            logger.info(f"Run {run_id + 1}/{config.n_runs_per_config}: {method}")
            
            try:
                result = run_single_experiment(5, method, run_id, config)
                all_results.append(result)
                logger.info(f"  Result: F1={result.final_f1_score:.3f}, improvement={result.target_improvement:.3f}")
            except Exception as e:
                logger.error(f"  Failed: {e}")
    
    # Analyze results
    static_results = [r for r in all_results if r.method == 'static_surrogate' and r.final_f1_score > 0]
    learning_results = [r for r in all_results if r.method == 'learning_surrogate' and r.final_f1_score > 0]
    
    logger.info(f"\n🔬 VALIDATION RESULTS:")
    logger.info(f"Static surrogate runs: {len(static_results)}")
    logger.info(f"Learning surrogate runs: {len(learning_results)}")
    
    if static_results:
        static_f1_mean = sum(r.final_f1_score for r in static_results) / len(static_results)
        static_improvement_mean = sum(r.target_improvement for r in static_results) / len(static_results)
        logger.info(f"Static surrogate - Mean F1: {static_f1_mean:.3f}, Mean improvement: {static_improvement_mean:.3f}")
    
    if learning_results:
        learning_f1_mean = sum(r.final_f1_score for r in learning_results) / len(learning_results)
        learning_improvement_mean = sum(r.target_improvement for r in learning_results) / len(learning_results)
        logger.info(f"Learning surrogate - Mean F1: {learning_f1_mean:.3f}, Mean improvement: {learning_improvement_mean:.3f}")
    
    # Conclusion
    if learning_results and static_results:
        f1_improvement = learning_f1_mean - static_f1_mean
        logger.info(f"\n📊 F1 Score Improvement: {f1_improvement:.3f}")
        if f1_improvement > 0.1:
            logger.info("✅ VALIDATION SUCCESSFUL: Learning surrogate significantly outperforms static surrogate!")
        else:
            logger.info("⚠️ VALIDATION INCONCLUSIVE: Improvement is small.")
    elif learning_results and not static_results:
        logger.info("✅ VALIDATION SUCCESSFUL: Learning surrogate works, static surrogate failed!")
    else:
        logger.info("❌ VALIDATION FAILED: Need to investigate further.")
    
    # Save results
    output_file = "quick_validation_results.json"
    results_data = [
        {
            'method': r.method,
            'f1_score': r.final_f1_score,
            'target_improvement': r.target_improvement,
            'runtime_seconds': r.runtime_seconds
        }
        for r in all_results
    ]
    
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    return all_results

if __name__ == "__main__":
    run_quick_validation()