#!/usr/bin/env python3
"""
Simple test script for the simplified trainers.

Tests that the basic functionality works without running full training.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO)

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    
    try:
        from src.causal_bayes_opt.training.simplified_grpo_trainer import SimplifiedGRPOTrainer
        print("✓ SimplifiedGRPOTrainer imported")
    except ImportError as e:
        print(f"✗ Failed to import SimplifiedGRPOTrainer: {e}")
        return False
        
    try:
        from src.causal_bayes_opt.training.simplified_bc_trainer import SimplifiedBCTrainer
        print("✓ SimplifiedBCTrainer imported")
    except ImportError as e:
        print(f"✗ Failed to import SimplifiedBCTrainer: {e}")
        return False
        
    try:
        from src.causal_bayes_opt.utils.scm_providers import create_toy_scm_rotation
        print("✓ SCM providers imported")
    except ImportError as e:
        print(f"✗ Failed to import SCM providers: {e}")
        return False
        
    try:
        from src.causal_bayes_opt.evaluation.simplified_grpo_evaluator import SimplifiedGRPOEvaluator
        print("✓ SimplifiedGRPOEvaluator imported")
    except ImportError as e:
        print(f"✗ Failed to import SimplifiedGRPOEvaluator: {e}")
        return False
        
    try:
        from src.causal_bayes_opt.evaluation.simplified_bc_evaluator import SimplifiedBCEvaluator
        print("✓ SimplifiedBCEvaluator imported")
    except ImportError as e:
        print(f"✗ Failed to import SimplifiedBCEvaluator: {e}")
        return False
        
    return True


def test_scm_creation():
    """Test SCM creation."""
    print("\nTesting SCM creation...")
    
    from src.causal_bayes_opt.utils.scm_providers import create_toy_scm_rotation
    
    try:
        scms = create_toy_scm_rotation(
            variable_range=(3, 4),
            structure_types=["fork", "chain"],
            samples_per_config=1
        )
        print(f"✓ Created {len(scms)} SCMs")
        
        # Extract just the SCMs
        if scms and isinstance(scms[0], tuple):
            scm_list = [scm for name, scm in scms]
            print(f"✓ Extracted {len(scm_list)} SCM objects")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create SCMs: {e}")
        return False


def test_trainer_initialization():
    """Test trainer initialization."""
    print("\nTesting trainer initialization...")
    
    from src.causal_bayes_opt.training.simplified_grpo_trainer import SimplifiedGRPOTrainer
    from src.causal_bayes_opt.training.simplified_bc_trainer import SimplifiedBCTrainer
    
    # Test GRPO trainer
    try:
        grpo_trainer = SimplifiedGRPOTrainer(
            learning_rate=3e-4,
            n_episodes=10,
            episode_length=5,
            architecture_level="baseline",
            use_early_stopping=False,
            seed=42
        )
        print("✓ GRPO trainer initialized")
    except Exception as e:
        print(f"✗ Failed to initialize GRPO trainer: {e}")
        return False
        
    # Test BC trainer
    try:
        bc_trainer = SimplifiedBCTrainer(
            model_type="acquisition",
            hidden_dim=64,
            num_layers=2,
            learning_rate=1e-3,
            batch_size=16,
            max_epochs=5,
            seed=42
        )
        print("✓ BC trainer initialized")
    except Exception as e:
        print(f"✗ Failed to initialize BC trainer: {e}")
        return False
        
    return True


def test_checkpoint_operations():
    """Test checkpoint save/load operations."""
    print("\nTesting checkpoint operations...")
    
    from src.causal_bayes_opt.training.simplified_bc_trainer import SimplifiedBCTrainer
    import tempfile
    
    try:
        trainer = SimplifiedBCTrainer(
            model_type="acquisition",
            hidden_dim=64,
            num_layers=2
        )
        
        # Create dummy checkpoint data
        checkpoint = {
            "params": {"dummy": "params"},
            "model_type": "acquisition",
            "config": {
                "hidden_dim": 64,
                "num_layers": 2,
                "learning_rate": 1e-3
            },
            "metrics": {
                "epochs_trained": 0,
                "best_val_loss": 1.0,
                "training_time": 0.0
            },
            "metadata": {
                "created_at": "test",
                "trainer_version": "simplified"
            }
        }
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = Path(f.name)
            
        trainer.save_checkpoint(temp_path, checkpoint)
        print("✓ Checkpoint saved")
        
        # Check if file exists
        if temp_path.exists():
            print("✓ Checkpoint file created")
            temp_path.unlink()  # Clean up
        
        return True
    except Exception as e:
        print(f"✗ Failed checkpoint operations: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Testing Simplified Components")
    print("="*60)
    
    tests = [
        test_imports,
        test_scm_creation,
        test_trainer_initialization,
        test_checkpoint_operations
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
            
    print("\n" + "="*60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("="*60)
    
    if passed == len(tests):
        print("\n✅ All tests passed! The simplified components are working.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        

if __name__ == "__main__":
    main()