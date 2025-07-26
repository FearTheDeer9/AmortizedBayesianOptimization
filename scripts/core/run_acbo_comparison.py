#!/usr/bin/env python3
"""
Simple ACBO Comparison Experiment Runner

This script provides a clean, simple interface for running ACBO comparison experiments
using the modular framework. It replaces the monolithic acbo_wandb_experiment.py.

Usage:
    # Run 3-method comparison
    poetry run python scripts/core/run_acbo_comparison.py --config-name=acbo_3method_comparison
    
    # Run 4-method comparison with trained policy
    poetry run python scripts/core/run_acbo_comparison.py --config-name=acbo_4method_comparison
    
    # Custom overrides
    poetry run python scripts/core/run_acbo_comparison.py --config-name=acbo_4method_comparison \
        experiment.runs_per_method=3 logging.wandb.enabled=false
"""

import logging
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add the directory containing this script to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import hydra
from omegaconf import DictConfig, OmegaConf

# Import our unified evaluation system
from src.causal_bayes_opt.evaluation import run_evaluation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def make_json_serializable(obj):
    """Recursively convert non-JSON-serializable objects."""
    import jax.numpy as jnp
    import numpy as onp
    from dataclasses import is_dataclass, asdict
    
    if isinstance(obj, dict):
        # Convert dict with potentially non-string keys
        new_dict = {}
        for key, value in obj.items():
            # Convert tuple keys to strings
            if isinstance(key, tuple):
                new_key = str(key)
            else:
                new_key = str(key) if not isinstance(key, (str, int, float, bool, type(None))) else key
            new_dict[new_key] = make_json_serializable(value)
        return new_dict
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (jnp.ndarray, onp.ndarray)):
        return obj.tolist()
    elif is_dataclass(obj):
        # Handle dataclass objects explicitly
        return make_json_serializable(asdict(obj))
    elif hasattr(obj, '__dict__'):
        # Convert objects with __dict__ to dict
        return make_json_serializable(obj.__dict__)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # Convert any other type to string
        return str(obj)


@hydra.main(version_base=None, config_path="../../config/experiment", config_name="acbo_3method_comparison")
def main(cfg: DictConfig) -> None:
    """
    Main function for running ACBO comparison experiments.
    
    Args:
        cfg: Hydra configuration
    """
    
    logger.info("🚀 Starting ACBO Comparison Experiment")
    logger.info(f"Experiment: {cfg.experiment.name}")
    logger.info(f"Methods: {list(cfg.experiment.methods.keys())}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    try:
        # Get output directory from config or use default
        output_dir = Path(cfg.get('output_dir', './results/acbo_comparison'))
        
        # Extract checkpoint path
        checkpoint_path = None
        if 'policy_checkpoint_path' in cfg:
            checkpoint_path = Path(cfg.policy_checkpoint_path)
        
        # Run evaluation using unified system
        logger.info(f"Running evaluation with output directory: {output_dir}")
        comparison_results = run_evaluation(
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            config=cfg
        )
        
        # Convert to legacy format for compatibility
        results = {
            'method_results': {},
            'execution_metadata': {
                'methods_tested': len(comparison_results.method_results),
                'scms_tested': comparison_results.config.get('n_scms', 0),
                'runs_per_method': comparison_results.config.get('n_seeds', 0),
                'total_experiments': len(comparison_results.method_results) * 
                                   comparison_results.config.get('n_scms', 0) * 
                                   comparison_results.config.get('n_seeds', 0),
                'total_time': sum(m.mean_time for m in comparison_results.method_results.values())
            },
            'statistical_analysis': {
                'summary_statistics': {},
                'pairwise_comparisons': comparison_results.statistical_tests
            },
            'visualizations': {}
        }
        
        # Convert method results to legacy format
        for method_name, metrics in comparison_results.method_results.items():
            results['statistical_analysis']['summary_statistics'][method_name] = {
                'target_improvement_mean': metrics.mean_improvement,
                'target_improvement_std': metrics.std_improvement,
                'structure_accuracy_mean': metrics.mean_final_f1,
                'target_improvement_count': metrics.n_runs
            }
        
        # Log summary
        execution_metadata = results['execution_metadata']
        logger.info("✅ Experiment completed successfully!")
        logger.info(f"📊 Total time: {execution_metadata['total_time']:.1f}s")
        logger.info(f"🔬 Methods tested: {execution_metadata['methods_tested']}")
        logger.info(f"🧪 SCMs tested: {execution_metadata['scms_tested']}")
        logger.info(f"🎯 Total experiments: {execution_metadata['total_experiments']}")
        
        # Print method performance summary
        logger.info("\n📈 Method Performance Summary:")
        summary_stats = results['statistical_analysis']['summary_statistics']
        for method_name, stats in summary_stats.items():
            target_mean = stats.get('target_reduction_mean', stats.get('target_improvement_mean', 0.0))
            target_std = stats.get('target_reduction_std', stats.get('target_improvement_std', 0.0))
            structure_mean = stats.get('structure_accuracy_mean', 0.0)
            valid_runs = stats.get('target_reduction_count', stats.get('target_improvement_count', 0))
            
            logger.info(f"  {method_name}:")
            logger.info(f"    Target reduction: {target_mean:.4f} ± {target_std:.4f} (positive = better)")
            logger.info(f"    Structure accuracy: {structure_mean:.4f}")
            logger.info(f"    Valid runs: {valid_runs}")
        
        # Print statistical significance results
        comparisons = results['statistical_analysis']['pairwise_comparisons']
        if comparisons:
            logger.info("\n🔍 Statistical Significance:")
            for comp_name, comp_result in comparisons.items():
                significance = "✅ Significant" if comp_result.significant else "❌ Not significant"
                logger.info(f"  {comp_name}: p={comp_result.p_value:.4f} ({significance})")
        
        # Print visualization info
        plots = results['visualizations']
        if plots:
            logger.info(f"\n📊 Generated {len(plots)} visualization plots:")
            for plot_name, plot_path in plots.items():
                logger.info(f"  {plot_name}: {plot_path}")
        
        logger.info("\n🎉 Experiment analysis complete!")
        
        # Save results to JSON file with atomic write
        import json
        from pathlib import Path
        import tempfile
        import shutil
        import hydra
        
        # Results are already saved by run_evaluation
        results_file = output_dir / "comparison_results.json"
        
        logger.info(f"\n✅ Results saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()