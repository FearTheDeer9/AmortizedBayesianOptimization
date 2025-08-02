#!/usr/bin/env python3
"""
Training-to-Evaluation Pipeline Orchestrator

This script coordinates the complete pipeline from training an enriched GRPO policy
to evaluating it against baselines in the 4-method comparison experiment.

Usage:
    # Run full pipeline
    poetry run python scripts/pipeline_train_and_evaluate.py
    
    # Skip training if checkpoint exists
    poetry run python scripts/pipeline_train_and_evaluate.py --skip-training
    
    # Custom training episodes
    poetry run python scripts/pipeline_train_and_evaluate.py --training-episodes 200

Features:
- Orchestrates existing proven components (no code duplication)
- Validates checkpoint compatibility
- Provides clear progress tracking
- Follows CLAUDE.md principles (functional, single responsibility)
"""

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_checkpoint_format(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Validate that checkpoint matches expected format for 4-method comparison.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        Dict with validation results
    """
    logger.info(f"🔍 Validating checkpoint format: {checkpoint_path}")
    
    result = {'valid': False, 'details': {}}
    
    try:
        checkpoint_file = checkpoint_path / "checkpoint.pkl"
        if not checkpoint_file.exists():
            result['details']['error'] = f"Checkpoint file not found: {checkpoint_file}"
            return result
        
        # Load and validate checkpoint structure
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        required_keys = ['policy_params', 'policy_config', 'enriched_architecture']
        missing_keys = [key for key in required_keys if key not in checkpoint_data]
        
        if missing_keys:
            result['details']['error'] = f"Missing required keys: {missing_keys}"
            return result
        
        # Validate enriched architecture flag
        if not checkpoint_data.get('enriched_architecture', False):
            result['details']['error'] = "Checkpoint not marked as enriched architecture"
            return result
        
        # Validate policy config structure
        policy_config = checkpoint_data['policy_config']
        if not policy_config.get('variable_agnostic', False):
            result['details']['warning'] = "Policy may not be variable-agnostic"
        
        result['valid'] = True
        result['details'] = {
            'checkpoint_size_mb': checkpoint_file.stat().st_size / (1024 * 1024),
            'enriched_architecture': checkpoint_data['enriched_architecture'],
            'variable_agnostic': policy_config.get('variable_agnostic', False),
            'num_variables': policy_config.get('num_variables', 'unknown'),
            'episode': checkpoint_data.get('episode', 'unknown')
        }
        
        logger.info(f"✅ Checkpoint validation passed: {result['details']}")
        return result
        
    except Exception as e:
        result['details']['error'] = f"Validation failed: {e}"
        logger.error(f"❌ Checkpoint validation failed: {e}")
        return result


def run_training(training_episodes: int = 100) -> Dict[str, Any]:
    """
    Run enriched GRPO training using existing modular trainer.
    
    Args:
        training_episodes: Number of training episodes
        
    Returns:
        Dict with training results
    """
    logger.info(f"🏋️ Starting GRPO training ({training_episodes} episodes)")
    
    cmd = [
        "poetry", "run", "python", "scripts/train_enriched_acbo_modular.py",
        f"training.n_episodes={training_episodes}"
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=Path.cwd())
        
        logger.info("✅ Training completed successfully")
        logger.info("Training output:")
        for line in result.stdout.split('\n')[-10:]:  # Last 10 lines
            if line.strip():
                logger.info(f"  {line}")
        
        return {
            'success': True,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.CalledProcessError as e:
        logger.error("❌ Training failed")
        logger.error(f"Error output: {e.stderr}")
        return {
            'success': False,
            'error': str(e),
            'stderr': e.stderr
        }


def run_4method_comparison() -> Dict[str, Any]:
    """
    Run 4-method comparison experiment using existing comparison script.
    
    Returns:
        Dict with comparison results
    """
    logger.info("🔬 Starting 4-method comparison experiment")
    
    cmd = [
        "poetry", "run", "python", "scripts/core/run_acbo_comparison.py",
        "--config-name=acbo_4method_comparison"
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=Path.cwd())
        
        logger.info("✅ 4-method comparison completed successfully")
        
        # Extract key results from output
        output_lines = result.stdout.split('\n')
        summary_lines = [line for line in output_lines if any(keyword in line.lower() 
                        for keyword in ['target improvement', 'structure accuracy', 'significant'])]
        
        logger.info("📊 Comparison Results Summary:")
        for line in summary_lines:
            logger.info(f"  {line}")
        
        return {
            'success': True,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'summary_lines': summary_lines
        }
        
    except subprocess.CalledProcessError as e:
        logger.error("❌ 4-method comparison failed")
        logger.error(f"Error output: {e.stderr}")
        return {
            'success': False,
            'error': str(e),
            'stderr': e.stderr
        }


def main():
    """Main pipeline orchestrator function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run training-to-evaluation pipeline")
    parser.add_argument("--skip-training", action="store_true", 
                       help="Skip training if checkpoint exists")
    parser.add_argument("--training-episodes", type=int, default=100,
                       help="Number of training episodes")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Training-to-Evaluation Pipeline")
    logger.info(f"Configuration: {args}")
    
    pipeline_start_time = time.time()
    
    # Expected checkpoint location
    expected_checkpoint = Path("checkpoints/enriched_grpo/enriched_grpo_final")
    
    # Phase 1: Training
    if args.skip_training and expected_checkpoint.exists():
        logger.info("⏭️ Skipping training (checkpoint exists)")
        training_result = {'success': True, 'skipped': True}
    else:
        training_result = run_training(args.training_episodes)
        if not training_result['success']:
            logger.error("❌ Pipeline failed: Training unsuccessful")
            return False
    
    # Phase 2: Checkpoint Validation
    if expected_checkpoint.exists():
        validation_result = validate_checkpoint_format(expected_checkpoint)
        if not validation_result['valid']:
            logger.error(f"❌ Pipeline failed: {validation_result['details']['error']}")
            return False
    else:
        logger.error(f"❌ Pipeline failed: Expected checkpoint not found at {expected_checkpoint}")
        return False
    
    # Phase 3: 4-Method Comparison
    comparison_result = run_4method_comparison()
    if not comparison_result['success']:
        logger.error("❌ Pipeline failed: 4-method comparison unsuccessful")
        return False
    
    # Pipeline Summary
    pipeline_time = time.time() - pipeline_start_time
    
    logger.info("\n" + "="*60)
    logger.info("🎯 TRAINING-TO-EVALUATION PIPELINE SUMMARY")
    logger.info("="*60)
    logger.info(f"⏱️ Total time: {pipeline_time:.1f} seconds")
    logger.info(f"🏋️ Training: {'✅ Success' if training_result['success'] else '❌ Failed'}")
    logger.info(f"🔍 Validation: ✅ Success")
    logger.info(f"🔬 Comparison: ✅ Success")
    
    logger.info(f"\n📁 Results:")
    logger.info(f"   Trained model: {expected_checkpoint}")
    
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        logger.info(f"   Experiment results: {outputs_dir.absolute()}")
    
    logger.info(f"\n🎉 Pipeline completed successfully!")
    logger.info(f"✅ Enriched GRPO policy trained and evaluated against baselines")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)