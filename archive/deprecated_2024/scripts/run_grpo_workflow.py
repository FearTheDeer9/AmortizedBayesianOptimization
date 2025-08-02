#!/usr/bin/env python3
"""
GRPO Training Workflow Runner

Demonstrates the complete workflow for training and comparing GRPO models.
This script shows how to:
1. Train multiple GRPO models with different configurations
2. Compare their performance systematically
3. Generate comprehensive reports

Usage:
    # Run full workflow
    poetry run python scripts/run_grpo_workflow.py
    
    # Run with WandB logging
    poetry run python scripts/run_grpo_workflow.py --enable-wandb
    
    # Custom number of training runs
    poetry run python scripts/run_grpo_workflow.py --n-runs 5

This script demonstrates the methodology similar to acbo_wandb_experiment.py
for systematic model creation and comparison.
"""

import argparse
import logging
import subprocess
import time
from pathlib import Path
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_training_experiment(config_overrides: str = "", run_name: str = "") -> bool:
    """Run a single training experiment."""
    cmd = [
        "poetry", "run", "python", "scripts/train_full_scale_grpo.py"
    ]
    
    if config_overrides:
        cmd.extend(config_overrides.split())
    
    logger.info(f"Starting training run: {run_name}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"✅ Training run completed: {run_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Training run failed: {run_name}")
        logger.error(f"Error: {e.stderr}")
        return False


def run_model_comparison(enable_wandb: bool = False) -> bool:
    """Run model comparison analysis."""
    cmd = ["poetry", "run", "python", "scripts/compare_grpo_models.py"]
    
    if enable_wandb:
        cmd.append("logging.wandb.enabled=true")
    
    logger.info("Starting model comparison")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("✅ Model comparison completed")
        logger.info("Results:")
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("❌ Model comparison failed")
        logger.error(f"Error: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run GRPO training workflow")
    parser.add_argument("--n-runs", type=int, default=3, 
                       help="Number of training runs with different configurations")
    parser.add_argument("--enable-wandb", action="store_true",
                       help="Enable WandB logging")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training and only run comparison")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting GRPO Training Workflow")
    logger.info(f"Configuration: {args}")
    
    # Check that we're in the right directory
    if not Path("scripts/train_full_scale_grpo.py").exists():
        logger.error("Please run this script from the project root directory")
        sys.exit(1)
    
    workflow_start_time = time.time()
    successful_runs = 0
    total_runs = args.n_runs
    
    if not args.skip_training:
        logger.info(f"📊 Phase 1: Training {total_runs} models with different configurations")
        
        # Define different training configurations to test
        training_configs = [
            {
                "name": "baseline",
                "overrides": ""
            },
            {
                "name": "high_lr",
                "overrides": "training.learning_rate=0.001"
            },
            {
                "name": "large_network",
                "overrides": "training.hidden_size=128 training.num_layers=3"
            },
            {
                "name": "long_episodes",
                "overrides": "training.episode_length=30 training.n_episodes=200"
            },
            {
                "name": "reward_weighted",
                "overrides": "training.reward_weights.optimization=1.5 training.reward_weights.structure=0.3"
            }
        ]
        
        # Run requested number of configurations
        for i in range(min(args.n_runs, len(training_configs))):
            config = training_configs[i]
            
            # Add WandB if requested
            overrides = config["overrides"]
            if args.enable_wandb:
                overrides += " logging.wandb.enabled=true"
            
            success = run_training_experiment(overrides, config["name"])
            if success:
                successful_runs += 1
            
            # Brief pause between runs
            if i < args.n_runs - 1:
                time.sleep(5)
        
        logger.info(f"📈 Phase 1 Complete: {successful_runs}/{total_runs} training runs successful")
    else:
        logger.info("⏭️  Skipping training phase")
        successful_runs = "unknown"
    
    # Phase 2: Model Comparison
    logger.info("🔍 Phase 2: Model Comparison and Analysis")
    
    # Check if we have models to compare
    checkpoints_dir = Path("checkpoints")
    if checkpoints_dir.exists():
        model_dirs = [d for d in checkpoints_dir.iterdir() 
                     if d.is_dir() and (d / "model.pkl").exists()]
        logger.info(f"Found {len(model_dirs)} models to compare")
        
        if len(model_dirs) >= 2:
            comparison_success = run_model_comparison(args.enable_wandb)
            
            if comparison_success:
                logger.info("📊 Phase 2 Complete: Model comparison successful")
            else:
                logger.error("❌ Phase 2 Failed: Model comparison unsuccessful")
        else:
            logger.warning("⚠️  Need at least 2 models for comparison")
            comparison_success = False
    else:
        logger.warning("⚠️  No checkpoints directory found")
        comparison_success = False
    
    # Final Summary
    workflow_time = time.time() - workflow_start_time
    logger.info("\n" + "="*60)
    logger.info("🎯 GRPO Training Workflow Summary")
    logger.info("="*60)
    logger.info(f"⏱️  Total time: {workflow_time:.1f} seconds")
    
    if not args.skip_training:
        logger.info(f"🏋️  Training runs: {successful_runs}/{total_runs} successful")
    
    if 'comparison_success' in locals():
        logger.info(f"🔬 Comparison: {'✅ Success' if comparison_success else '❌ Failed'}")
    
    logger.info("\n📁 Results:")
    if checkpoints_dir.exists():
        logger.info(f"   Models saved in: {checkpoints_dir.absolute()}")
    
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        logger.info(f"   Logs and outputs in: {outputs_dir.absolute()}")
    
    logger.info("\n🚀 Next Steps:")
    logger.info("   1. Review model checkpoints in checkpoints/")
    logger.info("   2. Examine comparison results in outputs/")
    logger.info("   3. Use best-performing model for downstream tasks")
    
    if args.enable_wandb:
        logger.info("   4. Check WandB dashboard for detailed metrics")
    
    logger.info("\n✨ Workflow completed!")


if __name__ == "__main__":
    main()