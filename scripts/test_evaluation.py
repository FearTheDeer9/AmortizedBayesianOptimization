#!/usr/bin/env python3
"""
Test script to verify evaluation pipeline functionality.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_checkpoint_loading():
    """Test loading a checkpoint."""
    from scripts.notebooks.base_components import CheckpointManager
    
    checkpoint_dir = project_root / "checkpoints" / "grpo_training"
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    
    # Find checkpoints
    checkpoints = checkpoint_manager.list_checkpoints()
    logger.info(f"Found {len(checkpoints)} checkpoints")
    
    # Try to load the specific checkpoint
    checkpoint_path = checkpoint_dir / "grpo_quick_minimize_20250723_101252_fixed"
    if checkpoint_path.exists():
        logger.info(f"Checkpoint exists: {checkpoint_path}")
        
        # Check for required files
        has_pkl = (checkpoint_path / "checkpoint.pkl").exists()
        has_meta = (checkpoint_path / "metadata.json").exists()
        
        logger.info(f"  checkpoint.pkl: {has_pkl}")
        logger.info(f"  metadata.json: {has_meta}")
        
        if has_pkl and has_meta:
            # Try to load metadata
            import json
            with open(checkpoint_path / "metadata.json", 'r') as f:
                metadata = json.load(f)
            logger.info(f"  Optimization: {metadata.get('optimization_config', {}).get('direction', 'unknown')}")
            return True
    
    return False

def test_subprocess_command():
    """Test the subprocess command directly."""
    import subprocess
    
    checkpoint_path = project_root / "checkpoints" / "grpo_training" / "grpo_quick_minimize_20250723_101252_fixed"
    output_dir = project_root / "test_output"
    output_dir.mkdir(exist_ok=True)
    
    # Test if unified_pipeline.py exists and is runnable
    unified_pipeline = project_root / "scripts" / "unified_pipeline.py"
    if not unified_pipeline.exists():
        logger.error(f"unified_pipeline.py not found at {unified_pipeline}")
        return False
    
    # Try a minimal run
    cmd = [
        "poetry", "run", "python",
        str(unified_pipeline),
        f"--checkpoint={checkpoint_path}",
        "--num-scms=1",
        "--runs-per-method=1",
        "--intervention-budget=3",
        f"--output-dir={output_dir}"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.info("✅ Command succeeded!")
            
            # Check for output files
            results_file = output_dir / "comparison_results.json"
            if results_file.exists():
                logger.info(f"✅ Results file created: {results_file}")
                return True
            else:
                logger.warning("⚠️ No results file found")
                # Check for Hydra output
                hydra_outputs = list(output_dir.glob("**/comparison_results.json"))
                if hydra_outputs:
                    logger.info(f"Found results in Hydra directory: {hydra_outputs[0]}")
        else:
            logger.error(f"❌ Command failed with code {result.returncode}")
            logger.error(f"STDOUT:\n{result.stdout}")
            logger.error(f"STDERR:\n{result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Command timed out after 60 seconds")
    except Exception as e:
        logger.error(f"❌ Command failed with exception: {e}")
    
    return False

def test_direct_import():
    """Test importing evaluation modules directly."""
    try:
        from src.causal_bayes_opt.evaluation.unified_runner import UnifiedEvaluationRunner
        logger.info("✅ Successfully imported UnifiedEvaluationRunner")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import UnifiedEvaluationRunner: {e}")
        
    try:
        from src.causal_bayes_opt.evaluation.grpo_evaluator import GRPOEvaluator
        logger.info("✅ Successfully imported GRPOEvaluator")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import GRPOEvaluator: {e}")
    
    return False

def main():
    """Run all tests."""
    logger.info("🧪 Testing Evaluation Pipeline Components")
    logger.info("=" * 60)
    
    # Test 1: Checkpoint loading
    logger.info("\n1. Testing checkpoint loading...")
    checkpoint_ok = test_checkpoint_loading()
    
    # Test 2: Direct imports
    logger.info("\n2. Testing direct module imports...")
    import_ok = test_direct_import()
    
    # Test 3: Subprocess command
    logger.info("\n3. Testing subprocess command...")
    subprocess_ok = test_subprocess_command()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary:")
    logger.info(f"  Checkpoint loading: {'✅' if checkpoint_ok else '❌'}")
    logger.info(f"  Module imports: {'✅' if import_ok else '❌'}")
    logger.info(f"  Subprocess command: {'✅' if subprocess_ok else '❌'}")
    
    if checkpoint_ok and (import_ok or subprocess_ok):
        logger.info("\n✅ Evaluation pipeline appears functional!")
        logger.info("The issue might be with Hydra output directory configuration.")
    else:
        logger.info("\n❌ Evaluation pipeline has issues that need fixing.")

if __name__ == "__main__":
    main()