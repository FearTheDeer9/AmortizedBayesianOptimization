#!/usr/bin/env python3
"""
Analyze target variable distribution across ALL training data.
"""

import sys
from pathlib import Path
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.training.data_preprocessing import load_demonstrations_from_path
from src.causal_bayes_opt.training.demonstration_to_tensor import create_bc_training_dataset

print("Creating full training dataset to analyze targets...")

# Load ALL demonstrations
demos_path = 'expert_demonstrations/raw/raw_demonstrations'
raw_demos = load_demonstrations_from_path(demos_path)

# Flatten
flat_demos = []
for item in raw_demos:
    if hasattr(item, 'demonstrations'):
        flat_demos.extend(item.demonstrations)
    else:
        flat_demos.append(item)

print(f"Processing {len(flat_demos)} demonstrations...")

# Create full training dataset
all_inputs, all_labels, metadata = create_bc_training_dataset(
    flat_demos, max_trajectory_length=100
)

print(f"\nCreated {len(all_inputs)} training examples")

# Count all targets
target_counter = Counter()
for label in all_labels:
    if 'targets' in label and label['targets']:
        for target in label['targets']:
            target_counter[target] += 1

print("\n" + "="*60)
print("COMPLETE TARGET DISTRIBUTION")
print("="*60)

total = len(all_labels)
for target, count in sorted(target_counter.items()):
    percentage = (count / total) * 100
    bar = '█' * int(percentage / 2)
    print(f"{target:5s}: {count:5d} ({percentage:5.1f}%) {bar}")

# Check for X4 specifically
print("\n" + "="*60)
if 'X4' in target_counter:
    print(f"X4 found: {target_counter['X4']} times")
else:
    print("X4 NEVER appears as a target in ANY training example!")
    print("This explains the 0% accuracy - the model has never seen X4 as a target.")
    
# Check which variables exist
all_vars = set()
for label in all_labels:
    if 'variables' in label:
        all_vars.update(label['variables'])
        
print(f"\nAll variables that appear in data: {sorted(all_vars)}")
print(f"Is X4 in the variable list? {'X4' in all_vars}")