#!/usr/bin/env python3
"""
Simple debug to check what the BC surrogate is actually outputting.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import logging
logging.basicConfig(level=logging.DEBUG)

from src.causal_bayes_opt.evaluation.model_interfaces import create_bc_surrogate
import jax.numpy as jnp

# Load the BC surrogate
checkpoint_path = Path('checkpoints/test_v2/bc_surrogate_final')
print(f"Loading BC surrogate from: {checkpoint_path}")

try:
    predict_fn, update_fn = create_bc_surrogate(checkpoint_path, allow_updates=False)
    print("BC surrogate loaded successfully!")
    
    # Test with zero input
    test_tensor = jnp.zeros((10, 3, 3))
    variables = ['X', 'Y', 'Z']
    target = 'Y'
    
    print(f"\nTesting with zero tensor...")
    result = predict_fn(test_tensor, target, variables)
    
    print(f"\nResult type: {type(result)}")
    print(f"Result: {result}")
    
    if hasattr(result, 'metadata'):
        print(f"\nMetadata: {result.metadata}")
        if 'marginal_parent_probs' in result.metadata:
            probs = result.metadata['marginal_parent_probs']
            print(f"Marginal parent probabilities: {probs}")
            
            # Check if all are 0.5
            all_half = all(abs(p - 0.5) < 1e-6 for p in probs.values())
            print(f"\nAll probabilities are 0.5? {all_half}")
            
            if all_half:
                print("\nThis indicates the model is outputting uniform probabilities.")
                print("Possible reasons:")
                print("1. Model wasn't trained long enough (only 5 epochs)")
                print("2. Model architecture is producing constant outputs")
                print("3. Model parameters are initialized to produce 0.5")
                
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Also check if we can look at the raw model output
print("\n" + "="*60)
print("Checking raw model internals...")

try:
    from src.causal_bayes_opt.utils.checkpoint_utils import load_checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    
    print(f"\nCheckpoint metadata:")
    print(f"  Model type: {checkpoint.get('model_type')}")
    print(f"  Epochs trained: {checkpoint.get('metadata', {}).get('epochs_trained', 'unknown')}")
    print(f"  Final train loss: {checkpoint.get('metrics', {}).get('final_train_loss', 'unknown')}")
    print(f"  Best val loss: {checkpoint.get('metrics', {}).get('best_val_loss', 'unknown')}")
    
    # Check training samples
    metadata = checkpoint.get('metadata', {})
    print(f"\nTraining info:")
    print(f"  Samples: {metadata.get('n_train_samples', 'unknown')}")
    print(f"  Demonstrations: {metadata.get('n_demonstrations', 'unknown')}")
    
except Exception as e:
    print(f"Error loading checkpoint directly: {e}")