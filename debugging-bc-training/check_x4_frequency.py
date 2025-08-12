#!/usr/bin/env python3
"""
Check how often X4 appears as a target in the demonstrations.
"""

import sys
from pathlib import Path
import pickle
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.training.data_preprocessing import load_demonstrations_from_path
from src.causal_bayes_opt.training.demonstration_to_tensor import create_bc_training_dataset

print("Analyzing X4 frequency in demonstrations...")

# Load ALL demonstrations
demos_path = 'expert_demonstrations/raw/raw_demonstrations'
print(f"Loading demonstrations from {demos_path}")
raw_demos = load_demonstrations_from_path(demos_path)

# Flatten demonstrations
flat_demos = []
for item in raw_demos:
    if hasattr(item, 'demonstrations'):
        flat_demos.extend(item.demonstrations)
    else:
        flat_demos.append(item)

print(f"Total demonstrations: {len(flat_demos)}")

# Count target variables
target_counter = Counter()
scm_type_counter = Counter()
x4_examples = []

for i, demo in enumerate(flat_demos):
    # Get SCM type
    if hasattr(demo, 'scm_type'):
        scm_type = demo.scm_type
    elif hasattr(demo, 'metadata') and 'scm_type' in demo.metadata:
        scm_type = demo.metadata['scm_type']
    else:
        scm_type = 'unknown'
    
    scm_type_counter[scm_type] += 1
    
    # Count interventions in trajectory
    if hasattr(demo, 'trajectory'):
        trajectory = demo.trajectory
    elif hasattr(demo, 'steps'):
        trajectory = demo.steps
    else:
        continue
    
    for step in trajectory:
        if hasattr(step, 'intervention'):
            intervention = step.intervention
            if intervention and hasattr(intervention, 'targets'):
                targets = intervention.targets
                if targets:
                    for target in targets:
                        target_counter[target] += 1
                        if target == 'X4':
                            x4_examples.append((i, scm_type))

print("\n" + "="*60)
print("TARGET VARIABLE FREQUENCY")
print("="*60)

# Sort by frequency
sorted_targets = sorted(target_counter.items(), key=lambda x: x[1], reverse=True)
total_interventions = sum(target_counter.values())

print(f"Total interventions: {total_interventions}")
print("\nTarget frequencies:")
for target, count in sorted_targets[:20]:  # Top 20
    percentage = (count / total_interventions) * 100
    print(f"  {target:10s}: {count:5d} ({percentage:5.1f}%)")

# Check X4 specifically
x4_count = target_counter.get('X4', 0)
x4_percentage = (x4_count / total_interventions * 100) if total_interventions > 0 else 0

print("\n" + "="*60)
print(f"X4 ANALYSIS")
print("="*60)
print(f"X4 as target: {x4_count} times ({x4_percentage:.2f}%)")
x4_rank = [i+1 for i, (t, _) in enumerate(sorted_targets) if t == 'X4']
print(f"X4 rank: {x4_rank[0] if x4_rank else 'Not found'}")

if x4_examples:
    print(f"\nX4 appears in demonstrations: {x4_examples[:10]}...")  # First 10
    print(f"\nSCM types with X4:")
    x4_scm_types = Counter([scm for _, scm in x4_examples])
    for scm_type, count in x4_scm_types.items():
        print(f"  {scm_type}: {count} times")

print("\n" + "="*60)
print("SCM TYPE DISTRIBUTION")
print("="*60)
for scm_type, count in scm_type_counter.most_common():
    print(f"  {scm_type}: {count} demonstrations")

# Now create training dataset and check labels
print("\n" + "="*60)
print("TRAINING DATASET ANALYSIS")
print("="*60)

# Use a subset for speed
subset_demos = flat_demos[:100]
all_inputs, all_labels, metadata = create_bc_training_dataset(subset_demos, max_trajectory_length=100)

label_target_counter = Counter()
for label in all_labels:
    if 'targets' in label and label['targets']:
        for target in label['targets']:
            label_target_counter[target] += 1

print(f"Created {len(all_inputs)} training examples from {len(subset_demos)} demos")
print("\nTarget distribution in training labels:")
for target, count in label_target_counter.most_common():
    percentage = (count / len(all_labels)) * 100
    print(f"  {target:10s}: {count:5d} ({percentage:5.1f}%)")

x4_in_labels = label_target_counter.get('X4', 0)
print(f"\nX4 in training labels: {x4_in_labels} ({x4_in_labels/len(all_labels)*100:.1f}%)")