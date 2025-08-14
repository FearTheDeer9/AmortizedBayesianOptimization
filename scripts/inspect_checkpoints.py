#!/usr/bin/env python3
"""
Script to inspect checkpoint contents and verify they can be loaded correctly.
"""

import pickle
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

def inspect_checkpoint(checkpoint_path: Path):
    """Inspect contents of a checkpoint file."""
    print(f"\n{'='*70}")
    print(f"Inspecting: {checkpoint_path}")
    print('='*70)
    
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
    
    # Check if it's a directory or file
    if checkpoint_path.is_dir():
        # Look for checkpoint.pkl inside
        checkpoint_file = checkpoint_path / 'checkpoint.pkl'
        if checkpoint_file.exists():
            checkpoint_path = checkpoint_file
        else:
            print(f"ERROR: No checkpoint.pkl found in {checkpoint_path}")
            return
    
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        print(f"✓ Successfully loaded checkpoint")
        print(f"\nTop-level keys: {list(checkpoint.keys())}")
        
        # Check model type
        if 'model_type' in checkpoint:
            print(f"Model type: {checkpoint['model_type']}")
        else:
            print("WARNING: No 'model_type' key found")
        
        # Check metadata
        if 'metadata' in checkpoint:
            print(f"\nMetadata keys: {list(checkpoint['metadata'].keys())}")
            if 'model_type' in checkpoint['metadata']:
                print(f"  Model type (from metadata): {checkpoint['metadata']['model_type']}")
            if 'architecture_type' in checkpoint['metadata']:
                print(f"  Architecture type: {checkpoint['metadata']['architecture_type']}")
        
        # Check architecture
        if 'architecture' in checkpoint:
            print(f"\nArchitecture: {checkpoint['architecture']}")
        else:
            print("WARNING: No 'architecture' key found")
        
        # Check params structure
        if 'params' in checkpoint:
            print(f"\nParams structure:")
            if isinstance(checkpoint['params'], dict):
                # Show first few param keys
                param_keys = list(checkpoint['params'].keys())[:10]
                print(f"  First few param keys: {param_keys}")
                print(f"  Total param keys: {len(checkpoint['params'])}")
            else:
                print(f"  Params type: {type(checkpoint['params'])}")
        
        # Check for specific trainer info
        if 'trainer_type' in checkpoint:
            print(f"\nTrainer type: {checkpoint['trainer_type']}")
        
        # Check training metrics
        if 'training_metrics' in checkpoint:
            metrics = checkpoint['training_metrics']
            if isinstance(metrics, dict):
                print(f"\nTraining metrics:")
                if 'epochs_trained' in metrics:
                    print(f"  Epochs trained: {metrics['epochs_trained']}")
                if 'best_val_loss' in metrics:
                    print(f"  Best val loss: {metrics['best_val_loss']:.4f}")
                if 'final_train_loss' in metrics:
                    print(f"  Final train loss: {metrics['final_train_loss']:.4f}")
        
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Inspect all relevant checkpoints."""
    checkpoints_to_inspect = [
        Path("checkpoints/bc_final"),
        Path("checkpoints/bc_surrogate_final"),
        Path("checkpoints/unified_grpo_final"),
    ]
    
    print("Inspecting ACBO checkpoints...")
    
    for checkpoint_path in checkpoints_to_inspect:
        inspect_checkpoint(checkpoint_path)
    
    print(f"\n{'='*70}")
    print("Inspection complete")
    print('='*70)

if __name__ == "__main__":
    main()