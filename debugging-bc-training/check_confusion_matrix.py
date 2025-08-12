#!/usr/bin/env python3
"""
Check the confusion matrix to see if X4 appears as a target.
"""

import pickle
import numpy as np
from pathlib import Path

# Load the metrics from the full training run
metrics_file = Path('debugging-bc-training/results_fixed/metrics_history.pkl')

if not metrics_file.exists():
    print(f"Metrics file not found: {metrics_file}")
    # Try the other location
    metrics_file = Path('debugging-bc-training/results/metrics_history.pkl')

print(f"Loading metrics from: {metrics_file}")

with open(metrics_file, 'rb') as f:
    data = pickle.load(f)

print("\n" + "="*60)
print("CONFUSION MATRIX ANALYSIS")
print("="*60)

# Check confusion matrices
if 'confusion_matrices' in data and data['confusion_matrices']:
    for epoch, cm_data in data['confusion_matrices'].items():
        cm = cm_data['matrix']
        labels = cm_data.get('labels', [])
        
        print(f"\nEpoch {epoch}:")
        print(f"Labels: {labels}")
        print(f"Confusion matrix shape: {cm.shape}")
        
        # Check if X4 is in labels
        if 'X4' in labels:
            x4_idx = labels.index('X4')
            print(f"\nX4 found at index {x4_idx}")
            
            # Get X4 row (when X4 is the true target)
            x4_as_target = cm[x4_idx, :]
            total_x4_targets = x4_as_target.sum()
            
            print(f"X4 as target: {total_x4_targets} times")
            print(f"X4 row in confusion matrix: {x4_as_target}")
            
            if total_x4_targets > 0:
                print("\nWhen X4 is the target, model predicts:")
                for i, count in enumerate(x4_as_target):
                    if count > 0:
                        pred_label = labels[i] if i < len(labels) else f"idx_{i}"
                        percentage = (count / total_x4_targets) * 100
                        print(f"  {pred_label}: {count} times ({percentage:.1f}%)")
                
                # Which variable is most often predicted instead of X4?
                most_predicted_idx = np.argmax(x4_as_target)
                most_predicted = labels[most_predicted_idx] if most_predicted_idx < len(labels) else f"idx_{most_predicted_idx}"
                print(f"\nMost common prediction when target is X4: {most_predicted}")
        else:
            print("X4 NOT in labels!")

# Check per-variable stats
print("\n" + "="*60)
print("PER-VARIABLE STATISTICS")
print("="*60)

if 'per_variable_stats' in data and data['per_variable_stats']:
    for var_name, stats in sorted(data['per_variable_stats'].items()):
        if 'X4' in var_name or var_name == 'X4':
            print(f"\n{var_name}:")
            print(f"  Attempts: {stats['attempts']}")
            print(f"  Correct: {stats['correct']}")
            if stats['attempts'] > 0:
                acc = stats['correct'] / stats['attempts']
                print(f"  Accuracy: {acc:.3f}")
    
    # Also show all variables for context
    print("\nAll variables with attempts:")
    for var_name, stats in sorted(data['per_variable_stats'].items()):
        if stats['attempts'] > 0:
            acc = stats['correct'] / stats['attempts'] if stats['attempts'] > 0 else 0
            print(f"  {var_name}: {stats['attempts']} attempts, {acc:.3f} accuracy")