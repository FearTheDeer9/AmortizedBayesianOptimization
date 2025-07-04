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

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf

from acbo_comparison.experiment_runner import ACBOExperimentRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        # Create and run experiment
        runner = ACBOExperimentRunner(cfg)
        results = runner.run_experiment()
        
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
            target_mean = stats.get('target_improvement_mean', 0.0)
            target_std = stats.get('target_improvement_std', 0.0)
            structure_mean = stats.get('structure_accuracy_mean', 0.0)
            valid_runs = stats.get('target_improvement_count', 0)
            
            logger.info(f"  {method_name}:")
            logger.info(f"    Target improvement: {target_mean:.4f} ± {target_std:.4f}")
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
        
    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()