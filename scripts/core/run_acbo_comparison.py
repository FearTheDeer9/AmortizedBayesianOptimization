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

# Import from current directory
from acbo_comparison.experiment_runner import ACBOExperimentRunner

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
        
        # Use Hydra's working directory if available, otherwise fall back to current directory
        try:
            # Get Hydra's output directory (respects hydra.run.dir override)
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            results_dir = Path(hydra_cfg.runtime.output_dir)
            logger.info(f"Using Hydra output directory: {results_dir}")
        except:
            # Fallback to relative results directory
            results_dir = Path("results")
            logger.info(f"Hydra not configured, using fallback directory: {results_dir}")
        
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Use consistent filename expected by unified pipeline
        results_file = results_dir / "comparison_results.json"
        
        # Ensure results are JSON serializable
        logger.info("\n🔄 Serializing results...")
        serializable_results = make_json_serializable(results)
        
        # Atomic write: write to temp file first, then move
        try:
            with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json', 
                                           dir=results_dir) as temp_file:
                logger.info(f"Writing to temporary file: {temp_file.name}")
                json.dump(serializable_results, temp_file, indent=2, default=str)
                temp_file.flush()
                os.fsync(temp_file.fileno())  # Force write to disk
                temp_path = temp_file.name
            
            # Validate the JSON file
            logger.info("Validating JSON file...")
            with open(temp_path, 'r') as f:
                json.load(f)  # This will raise if JSON is invalid
            
            # Move temp file to final location (atomic on most systems)
            shutil.move(temp_path, results_file)
            logger.info(f"\n✅ Results successfully saved to: {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            # Clean up temp file if it exists
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
        
    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()