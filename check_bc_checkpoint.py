#!/usr/bin/env python3
"""Check BC checkpoint configuration."""

import pickle
from pathlib import Path

checkpoint_path = Path("checkpoints/behavioral_cloning/dev/acquisition/bc_demo_acquisition_epoch_6_level_3_1753693092.pkl")

import gzip

# Try loading as gzip first
try:
    with gzip.open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
except:
    # Fallback to regular pickle
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)

print("Checkpoint keys:", list(data.keys()))
print("\nModel config:", data.get('model_config', {}))
print("\nConfig:", data.get('config', {}))

# Check parameter structure
if 'policy_params' in data:
    params = data['policy_params']
    
    def print_param_structure(params, prefix=""):
        if isinstance(params, dict):
            for key, value in params.items():
                if isinstance(value, dict):
                    print(f"{prefix}{key}/")
                    print_param_structure(value, prefix + "  ")
                else:
                    print(f"{prefix}{key}: shape={getattr(value, 'shape', '?')}")
    
    print("\nParameter structure:")
    print_param_structure(params)