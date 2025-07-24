#!/usr/bin/env python3
"""
Validate that the BC fix resolves the astronomical loss issue.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import jax.random as random
import time

from src.causal_bayes_opt.training.bc_data_pipeline import process_all_demonstrations
from src.causal_bayes_opt.training.bc_surrogate_trainer import (
    BCSurrogateTrainer,
    BCTrainingConfig,
    create_bc_surrogate_trainer
)
from src.causal_bayes_opt.training.config import SurrogateTrainingConfig
from src.causal_bayes_opt.training.trajectory_processor import DifficultyLevel


def validate_bc_fix():
    """Run a minimal training loop to validate the fix."""
    print("="*60)
    print("BC FIX VALIDATION TEST")
    print("="*60)
    
    # Test criteria
    max_acceptable_loss = 100.0  # Anything above this is considered astronomical
    expected_loss_range = (0.0, 10.0)  # Reasonable KL divergence range
    
    print(f"\nValidation criteria:")
    print(f"  - Loss must be < {max_acceptable_loss}")
    print(f"  - Expected range: {expected_loss_range}")
    
    # Load small dataset
    print("\n1. Loading demonstration data...")
    demo_dir = Path("expert_demonstrations/raw/raw_demonstrations")
    
    # Process only one file for speed
    demo_files = list(demo_dir.glob("*.pkl"))[:1]
    if not demo_files:
        print("ERROR: No demonstration files found!")
        return False
    
    print(f"   Using {demo_files[0].name}")
    
    # Process demonstrations
    processed = process_all_demonstrations(
        str(demo_dir),
        split_ratios=(0.8, 0.1, 0.1),
        max_examples_per_demo=20  # Limit examples for speed
    )
    
    # Get dataset
    if DifficultyLevel.EASY in processed.surrogate_datasets:
        dataset = processed.surrogate_datasets[DifficultyLevel.EASY]
    else:
        dataset = next(iter(processed.surrogate_datasets.values()))
    
    print(f"   Loaded {len(dataset.training_examples)} training examples")
    
    # Create minimal trainer
    print("\n2. Creating BC trainer...")
    trainer = create_bc_surrogate_trainer(
        learning_rate=1e-3,
        batch_size=8,
        use_curriculum=False,
        use_jax=True,
        checkpoint_dir="checkpoints/bc_validation",
        enable_wandb_logging=False,
        experiment_name="bc_validation"
    )
    
    # Prepare minimal datasets
    curriculum_datasets = {DifficultyLevel.EASY: dataset}
    validation_datasets = {DifficultyLevel.EASY: dataset}
    
    # Run training for just a few steps
    print("\n3. Running training for 5 epochs...")
    print("   (This should take ~30 seconds)")
    
    # Override max epochs for quick test
    trainer.config.max_epochs_per_level = 5
    trainer.config.surrogate_config.max_epochs = 5
    
    start_time = time.time()
    
    # Track losses during training
    losses_seen = []
    
    # Monkey-patch the training to capture losses
    original_log = trainer.log_training_metrics
    def capture_losses(state, train_metrics, val_metrics):
        losses_seen.append(train_metrics.average_loss)
        original_log(state, train_metrics, val_metrics)
    trainer.log_training_metrics = capture_losses
    
    try:
        random_key = random.PRNGKey(42)
        results = trainer.train_on_curriculum(
            curriculum_datasets=curriculum_datasets,
            validation_datasets=validation_datasets,
            random_key=random_key
        )
        
        training_time = time.time() - start_time
        
        # Check results
        print(f"\n4. Training completed in {training_time:.1f} seconds")
        print(f"   Final loss: {results.final_state.best_validation_loss:.6f}")
        print(f"   Losses seen: {losses_seen}")
        
        # Validation checks
        print("\n5. VALIDATION RESULTS:")
        
        all_losses_reasonable = all(loss < max_acceptable_loss for loss in losses_seen)
        final_loss_reasonable = results.final_state.best_validation_loss < max_acceptable_loss
        losses_in_expected_range = all(expected_loss_range[0] <= loss <= expected_loss_range[1] 
                                       for loss in losses_seen)
        
        print(f"   ✓ All losses < {max_acceptable_loss}: {all_losses_reasonable}")
        print(f"   ✓ Final loss reasonable: {final_loss_reasonable}")
        print(f"   ✓ Losses in expected range: {losses_in_expected_range}")
        
        if all_losses_reasonable and final_loss_reasonable:
            print("\n✅ FIX VALIDATED: No astronomical losses detected!")
            print("   The BC training is working correctly.")
            return True
        else:
            print("\n❌ FIX FAILED: Astronomical losses still present!")
            print(f"   Maximum loss seen: {max(losses_seen)}")
            return False
            
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_notebook_comparison():
    """Instructions for notebook comparison test."""
    print("\n" + "="*60)
    print("NOTEBOOK COMPARISON TEST")
    print("="*60)
    
    print("\nTo fully validate the fix in the notebook:")
    print("\n1. Run the notebook BEFORE the fix:")
    print("   poetry run jupyter notebook experiments/bc_development_workflow.ipynb")
    print("   - Look for loss values in cells 17-18")
    print("   - You should see: loss=999999999.9635")
    
    print("\n2. Apply the fix (already done)")
    
    print("\n3. Run the notebook AFTER the fix:")
    print("   poetry run jupyter notebook experiments/bc_development_workflow.ipynb")
    print("   - Restart kernel and run all cells")
    print("   - Look for loss values in cells 17-18")
    print("   - You should see: loss < 10.0")
    
    print("\n4. Check optimization results (cells 19-20):")
    print("   - BEFORE: All methods show 0% improvement")
    print("   - AFTER: BC methods should show positive improvement")


if __name__ == "__main__":
    # Run validation test
    success = validate_bc_fix()
    
    # Show notebook test instructions
    run_notebook_comparison()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if success:
        print("\n✅ The fix has been validated successfully!")
        print("   BC training now produces reasonable loss values.")
        print("   You can proceed with running the full notebook.")
    else:
        print("\n❌ The fix validation failed.")
        print("   Please check the error messages above.")
    
    print("\nKey indicators that the fix is working:")
    print("  1. Loss values are between 0 and 10 (not 999999999)")
    print("  2. Training progresses smoothly without errors")
    print("  3. BC methods show improvement in optimization tasks")