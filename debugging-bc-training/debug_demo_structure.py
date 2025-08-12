#!/usr/bin/env python3
"""Debug the structure of demonstrations to understand why permuted version gets 0 examples."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.training.data_preprocessing import load_demonstrations_from_path

# Load demonstrations
demos_path = Path("../expert_demonstrations/raw/raw_demonstrations")
if not demos_path.exists():
    demos_path = Path("expert_demonstrations/raw/raw_demonstrations")

raw_demos = load_demonstrations_from_path(str(demos_path), max_files=1)

# Flatten
flat_demos = []
for item in raw_demos:
    if hasattr(item, 'demonstrations'):
        flat_demos.extend(item.demonstrations[:1])
    else:
        flat_demos.append(item)

print("Demo structure:")
print("-"*40)

if flat_demos:
    demo = flat_demos[0]
    print(f"Type: {type(demo)}")
    print(f"Attributes: {dir(demo)}")
    
    if hasattr(demo, 'trajectory'):
        print(f"\nTrajectory length: {len(demo.trajectory)}")
        if demo.trajectory:
            print(f"First trajectory item: {demo.trajectory[0]}")
    
    if hasattr(demo, 'scm'):
        print(f"\nSCM type: {type(demo.scm)}")
        if hasattr(demo.scm, 'variables'):
            print(f"SCM variables: {demo.scm.variables}")
        if hasattr(demo.scm, 'get_variables'):
            print(f"SCM get_variables: {demo.scm.get_variables()}")
            
    if hasattr(demo, 'target'):
        print(f"\nTarget: {demo.target}")