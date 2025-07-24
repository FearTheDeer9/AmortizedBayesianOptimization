#!/usr/bin/env python3
"""
Inspect BC checkpoint structure to understand format.
"""

import pickle
import gzip
import sys
from pathlib import Path
from pprint import pprint

# Set up paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def inspect_checkpoint(checkpoint_path):
    """Load and inspect checkpoint structure."""
    print(f"\nInspecting: {checkpoint_path.name}")
    print("=" * 60)
    
    try:
        # Check if gzipped by looking at magic bytes
        with open(checkpoint_path, 'rb') as f:
            magic = f.read(2)
        
        if magic == b'\x1f\x8b':  # gzip magic number
            with gzip.open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
        
        print("Checkpoint keys:")
        for key in data.keys():
            print(f"  - {key}: {type(data[key])}")
            
        # Check specific fields
        if 'config' in data:
            print("\nConfig type:", type(data['config']))
            if hasattr(data['config'], '__dict__'):
                print("Config attributes:")
                for attr in dir(data['config']):
                    if not attr.startswith('_'):
                        print(f"    {attr}")
        
        if 'model_params' in data:
            print("\nModel params structure:")
            if isinstance(data['model_params'], dict):
                for key in data['model_params'].keys():
                    print(f"  - {key}")
                    
        if 'policy_params' in data:
            print("\nPolicy params structure:")
            if isinstance(data['policy_params'], dict):
                for key in data['policy_params'].keys():
                    print(f"  - {key}")
                    
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

# Inspect surrogate checkpoint
surrogate_path = project_root / "checkpoints/behavioral_cloning/dev/surrogate/surrogate_bc_development_epoch_22_level_3_1753298905.pkl"
if surrogate_path.exists():
    inspect_checkpoint(surrogate_path)

# Inspect acquisition checkpoint  
acquisition_path = project_root / "checkpoints/behavioral_cloning/dev/acquisition/bc_demo_acquisition_epoch_6_level_3_1753299449.pkl"
if acquisition_path.exists():
    inspect_checkpoint(acquisition_path)