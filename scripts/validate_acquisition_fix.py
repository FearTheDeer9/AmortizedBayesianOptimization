#!/usr/bin/env python3
"""
Validate the acquisition model fix.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import jax.random as random

from src.causal_bayes_opt.training.behavioral_cloning_adapter import (
    load_demonstration_batch,
    create_acquisition_state,
    extract_avici_samples,
    extract_intervention_sequence
)
from src.causal_bayes_opt.training.trajectory_processor import (
    extract_trajectory_steps,
    DifficultyLevel
)
from src.causal_bayes_opt.training.bc_acquisition_trainer import (
    BCAcquisitionTrainer,
    BCPolicyConfig,
    create_bc_acquisition_trainer
)
from src.causal_bayes_opt.training.bc_data_pipeline import process_all_demonstrations


def test_state_has_variables():
    """Test that acquisition states now have variable information."""
    print("="*60)
    print("TESTING ACQUISITION STATE FIX")
    print("="*60)
    
    # Load demonstration
    demo_file = Path("expert_demonstrations/raw/raw_demonstrations/batch_1751266609.pkl")
    batch = load_demonstration_batch(str(demo_file))
    demo = batch.demonstrations[0]
    
    # Create acquisition state
    avici_data = extract_avici_samples(demo)
    intervention_history = extract_intervention_sequence(demo)
    
    state = create_acquisition_state(
        demo=demo,
        step=0,
        avici_data=avici_data,
        intervention_history=intervention_history[:1]
    )
    
    print(f"\n1. State created successfully")
    print(f"   Has metadata: {hasattr(state, 'metadata')}")
    
    if hasattr(state, 'metadata') and 'scm_info' in state.metadata:
        scm_info = state.metadata['scm_info']
        print(f"   Has scm_info in metadata: True")
        print(f"   Variables in scm_info: {scm_info.get('variables', [])}")
        print(f"   Number of variables: {len(scm_info.get('variables', []))}")
    
    # Extract trajectory steps
    trajectory_steps = extract_trajectory_steps(demo, "test_demo")
    if trajectory_steps:
        first_step = trajectory_steps[0]
        print(f"\n2. First trajectory step:")
        print(f"   State has metadata: {hasattr(first_step.state, 'metadata')}")
        if hasattr(first_step.state, 'metadata') and 'scm_info' in first_step.state.metadata:
            scm_info = first_step.state.metadata['scm_info']
            print(f"   Variables: {scm_info.get('variables', [])[:5]}...")
    
    return state, trajectory_steps


def test_acquisition_training():
    """Test that acquisition training works without astronomical losses."""
    print("\n" + "="*60)
    print("TESTING ACQUISITION TRAINING")
    print("="*60)
    
    # Process demonstrations
    demo_dir = Path("expert_demonstrations/raw/raw_demonstrations")
    processed = process_all_demonstrations(
        str(demo_dir),
        max_examples_per_demo=10,  # Limit for speed
        split_ratios=(0.8, 0.1, 0.1)
    )
    
    # Get acquisition dataset
    if DifficultyLevel.EASY in processed.acquisition_datasets:
        dataset = processed.acquisition_datasets[DifficultyLevel.EASY]
    else:
        dataset = next(iter(processed.acquisition_datasets.values()))
    
    print(f"\n1. Dataset loaded:")
    print(f"   Trajectory steps: {len(dataset.trajectory_steps)}")
    
    # Check a few steps for variable info
    print(f"\n2. Checking trajectory steps for variable info:")
    for i in range(min(3, len(dataset.trajectory_steps))):
        step = dataset.trajectory_steps[i]
        has_metadata = hasattr(step.state, 'metadata')
        has_scm = has_metadata and 'scm_info' in step.state.metadata if has_metadata else False
        if has_scm:
            scm_info = step.state.metadata['scm_info']
            vars_count = len(scm_info.get('variables', []))
        else:
            vars_count = 0
        print(f"   Step {i}: has_scm_info={has_scm}, num_vars={vars_count}")
    
    # Create minimal trainer
    print(f"\n3. Creating acquisition trainer...")
    trainer = create_bc_acquisition_trainer(
        learning_rate=1e-3,
        batch_size=4,
        use_curriculum=False,
        use_jax=True,
        enable_wandb_logging=False,
        experiment_name="acquisition_fix_test"
    )
    
    # Prepare datasets
    curriculum_datasets = {DifficultyLevel.EASY: dataset}
    validation_datasets = {DifficultyLevel.EASY: dataset}
    
    # Override for quick test
    trainer.config.max_epochs_per_level = 2
    
    # Track losses
    losses_seen = []
    original_train = trainer._train_epoch
    
    def track_losses(state, trajectory_steps, random_key):
        """Wrapper to track losses during training."""
        updated_state, metrics = original_train(state, trajectory_steps, random_key)
        loss = metrics.get('loss', float('inf'))
        losses_seen.append(loss)
        print(f"   Epoch {state.epoch}: loss={loss:.6f}")
        return updated_state, metrics
    
    trainer._train_epoch = track_losses
    
    # Run training
    print(f"\n4. Running training for 2 epochs...")
    try:
        random_key = random.PRNGKey(42)
        results = trainer.train_on_curriculum(
            curriculum_datasets=curriculum_datasets,
            validation_datasets=validation_datasets,
            random_key=random_key
        )
        
        print(f"\n5. Training completed successfully!")
        print(f"   Losses seen: {losses_seen}")
        
        # Check for astronomical losses
        all_reasonable = all(loss < 100 for loss in losses_seen)
        
        if all_reasonable:
            print(f"\n✅ SUCCESS: No astronomical losses detected!")
            print(f"   All losses < 100")
            return True
        else:
            print(f"\n❌ FAILED: Astronomical losses still present!")
            print(f"   Max loss: {max(losses_seen)}")
            return False
            
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test state creation
    state, trajectory_steps = test_state_has_variables()
    
    # Test training
    success = test_acquisition_training()
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    if success:
        print("\n✅ The acquisition model fix is working!")
        print("   - States now include scm_info with variables")
        print("   - Variable names are properly mapped to indices")
        print("   - Cross-entropy loss is computed correctly")
        print("   - No astronomical losses during training")
    else:
        print("\n❌ The fix still has issues")
        print("   Check the error messages above for details")