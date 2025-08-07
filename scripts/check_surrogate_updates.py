#!/usr/bin/env python3
"""
Check if BC surrogate supports active learning updates.
"""

import sys
from pathlib import Path

# Add src to path  
sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.evaluation.surrogate_registry import SurrogateRegistry

# Load BC surrogate
registry = SurrogateRegistry()
bc_checkpoint = "checkpoints/comprehensive_20250804_190724/bc_surrogate_final/checkpoint.pkl"

try:
    registry.register('bc', bc_checkpoint)
    print("✓ Loaded BC surrogate")
    
    # Get surrogate
    surrogate = registry.get('bc')
    
    # Check available methods
    print("\nSurrogate methods:")
    for attr in dir(surrogate):
        if not attr.startswith('_'):
            print(f"  - {attr}")
    
    # Check if it supports updates
    if hasattr(surrogate, 'update_from_buffer'):
        print("\n✓ Surrogate HAS update_from_buffer method")
    else:
        print("\n✗ Surrogate DOES NOT have update_from_buffer method")
        print("  This explains why active learning doesn't work!")
        
    # Check surrogate type
    print(f"\nSurrogate type: {type(surrogate)}")
    print(f"Surrogate class: {surrogate.__class__.__name__}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()