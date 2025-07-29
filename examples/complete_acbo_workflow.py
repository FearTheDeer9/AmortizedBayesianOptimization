#!/usr/bin/env python3
"""
Complete ACBO Workflow Demo

This script demonstrates the full workflow:
1. Train GRPO model
2. Train BC model  
3. Evaluate and compare all methods

Usage:
    python examples/complete_acbo_workflow.py
"""

import logging
import subprocess
import sys
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd: list, description: str):
    """Run a command and check for errors."""
    logger.info(f"\n{'='*60}")
    logger.info(f"{description}")
    logger.info(f"{'='*60}")
    logger.info(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Command failed with code {result.returncode}")
        logger.error(f"Error: {result.stderr}")
        sys.exit(1)
    
    # Print key outputs
    for line in result.stdout.split('\n'):
        if any(keyword in line.lower() for keyword in ['completed', 'final', 'saved', 'mean']):
            logger.info(f"  > {line.strip()}")
    
    return result


def main():
    """Run complete ACBO workflow."""
    logger.info("COMPLETE ACBO WORKFLOW DEMO")
    logger.info("="*80)
    logger.info("This demo will:")
    logger.info("1. Train a GRPO model (100 episodes, ~30s)")
    logger.info("2. Train a BC model with oracle expert (100 episodes, ~20s)")
    logger.info("3. Evaluate all methods on test SCMs")
    logger.info("4. Show comparison results")
    
    # Quick settings for demo (reduced from defaults)
    episodes = 100
    demo_episodes = 50
    
    # Step 1: Train GRPO
    logger.info("\nStep 1: Training GRPO Model")
    grpo_cmd = [
        sys.executable, "scripts/train_acbo_methods.py",
        "--method", "grpo",
        "--episodes", str(episodes),
        "--use_surrogate",  # Enable structure learning
        "--checkpoint_dir", "demo_checkpoints"
    ]
    run_command(grpo_cmd, "Training GRPO with surrogate learning")
    
    # Step 2: Train BC
    logger.info("\nStep 2: Training BC Model")
    bc_cmd = [
        sys.executable, "scripts/train_acbo_methods.py",
        "--method", "bc",
        "--episodes", str(episodes),
        "--demo_episodes", str(demo_episodes),
        "--expert", "oracle",
        "--use_surrogate",
        "--checkpoint_dir", "demo_checkpoints"
    ]
    run_command(bc_cmd, "Training BC with oracle expert")
    
    # Step 3: Evaluate
    logger.info("\nStep 3: Evaluating All Methods")
    eval_cmd = [
        sys.executable, "scripts/evaluate_acbo_methods.py",
        "--grpo", "demo_checkpoints/clean_grpo_final",
        "--bc", "demo_checkpoints/clean_bc_final",
        "--n_scms", "5",  # Fewer SCMs for demo
        "--n_interventions", "10",  # Fewer interventions
        "--output_dir", "demo_results",
        "--plot"
    ]
    result = run_command(eval_cmd, "Comparing all methods")
    
    # Step 4: Show results
    logger.info("\n" + "="*80)
    logger.info("WORKFLOW COMPLETED!")
    logger.info("="*80)
    
    # Load and display results
    results_file = Path("demo_results/evaluation_results.json")
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        
        logger.info("\nMethod Performance Summary:")
        logger.info("-"*40)
        
        # Sort methods by performance
        methods = sorted(
            results.items(),
            key=lambda x: x[1]['aggregate_metrics']['mean_improvement'],
            reverse=True
        )
        
        for rank, (method, data) in enumerate(methods, 1):
            metrics = data['aggregate_metrics']
            logger.info(f"{rank}. {method:10s}: improvement = {metrics['mean_improvement']:+.3f}")
            if 'mean_f1_score' in metrics and metrics['mean_f1_score'] > 0:
                logger.info(f"   {'':10s}  structure F1 = {metrics['mean_f1_score']:.3f}")
    
    logger.info("\nOutputs:")
    logger.info(f"  - Checkpoints: demo_checkpoints/")
    logger.info(f"  - Results: demo_results/evaluation_results.json")
    logger.info(f"  - Plots: demo_results/*.png")
    
    logger.info("\nKey Insights:")
    logger.info("  - Oracle (knows true structure) provides upper bound")
    logger.info("  - GRPO learns intervention policy from rewards")
    logger.info("  - BC mimics oracle demonstrations")
    logger.info("  - Random provides lower bound baseline")
    
    logger.info("\nNext Steps:")
    logger.info("  - Train longer for better performance (--episodes 1000)")
    logger.info("  - Test on more SCMs (--n_scms 20)")
    logger.info("  - Try BC with random expert (--expert random)")
    logger.info("  - Disable surrogate learning (remove --use_surrogate)")


if __name__ == "__main__":
    main()