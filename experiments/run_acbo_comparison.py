#!/usr/bin/env python3
"""
Fixed ACBO Comparison Runner

This script properly loads the ACBO configuration and runs the comparison
without Hydra interpolation errors and with proper WandB handling.
"""

import sys
import os
from pathlib import Path
import yaml
import logging
from omegaconf import DictConfig, OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.core.acbo_comparison.experiment_runner import ACBOExperimentRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_fix_config(config_path: str, checkpoint_path: str) -> DictConfig:
    """Load ACBO config and fix issues for direct usage."""
    
    # Load yaml config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Fix policy checkpoint path
    config['policy_checkpoint_path'] = checkpoint_path
    
    # Disable WandB if it's causing issues
    if 'logging' not in config:
        config['logging'] = {}
    if 'wandb' not in config['logging']:
        config['logging']['wandb'] = {}
    config['logging']['wandb']['enabled'] = False  # Disable for now
    
    # Create OmegaConf object
    return OmegaConf.create(config)


def run_comparison(config_path: str, checkpoint_path: str, training_mode: str = "QUICK"):
    """Run ACBO comparison with fixed configuration."""
    
    print("🏁 Running Fixed ACBO Comparison")
    print("=" * 60)
    
    # Load and fix configuration
    config = load_and_fix_config(config_path, checkpoint_path)
    
    # Adjust for training mode
    if training_mode == "QUICK":
        config.experiment.runs_per_method = 3
        config.experiment.intervention_budget = 10
        print("🏃 Using QUICK mode: 3 runs, 10 interventions")
    
    print(f"💾 Policy checkpoint: {checkpoint_path}")
    print(f"📊 Methods: {list(config.experiment.methods.keys())}")
    print(f"🔄 Runs per method: {config.experiment.runs_per_method}")
    
    try:
        # Initialize and run experiment
        runner = ACBOExperimentRunner(config)
        results = runner.run_experiment()
        
        print("\n✅ Comparison completed successfully!")
        
        # Extract key results
        if 'method_results' in results:
            print("\n📊 Results Summary:")
            for method, method_results in results['method_results'].items():
                if method_results:
                    avg_improvement = sum(r.get('target_improvement', 0) for r in method_results) / len(method_results)
                    print(f"  {method}: {avg_improvement:.3f} avg improvement")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Get arguments from command line or use defaults
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(project_root / "config/experiment/acbo_4method_comparison.yaml")
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else str(project_root / "checkpoints/grpo_testing/enriched_grpo_final")
    training_mode = sys.argv[3] if len(sys.argv) > 3 else "QUICK"
    
    # Run comparison
    results = run_comparison(config_path, checkpoint_path, training_mode)
    
    if results:
        print("\n🎉 ACBO comparison completed successfully!")
    else:
        print("\n❌ ACBO comparison failed")