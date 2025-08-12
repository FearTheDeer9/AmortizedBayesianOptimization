#!/usr/bin/env python3
"""
Check the final results and X4 performance from the optimal training run.
"""

import pickle
from pathlib import Path
import numpy as np

# Load the metrics
metrics_file = Path('debugging-bc-training/results_optimal/metrics_history.pkl')

if not metrics_file.exists():
    print(f"Metrics file not found: {metrics_file}")
    exit(1)

print("Loading metrics from optimal training run...")

with open(metrics_file, 'rb') as f:
    data = pickle.load(f)

print("\n" + "="*60)
print("TRAINING RESULTS SUMMARY")
print("="*60)

# Check epoch metrics
if 'epoch_metrics' in data and data['epoch_metrics']:
    epochs = len(data['epoch_metrics'])
    best_acc = max(m.get('val_accuracy', 0) for m in data['epoch_metrics'])
    final_acc = data['epoch_metrics'][-1].get('val_accuracy', 0)
    
    print(f"\nTraining Statistics:")
    print(f"  Total epochs: {epochs}")
    print(f"  Best validation accuracy: {best_acc:.3f}")
    print(f"  Final validation accuracy: {final_acc:.3f}")

# Check per-variable stats
print("\n" + "="*60)
print("PER-VARIABLE PERFORMANCE")
print("="*60)

if 'per_variable_stats' in data and data['per_variable_stats']:
    print("\nVariable | Attempts | Correct | Accuracy")
    print("-" * 45)
    
    for var_name in sorted(data['per_variable_stats'].keys()):
        stats = data['per_variable_stats'][var_name]
        attempts = stats['attempts']
        correct = stats['correct']
        if attempts > 0:
            acc = correct / attempts
            print(f"{var_name:8s} | {attempts:8d} | {correct:7d} | {acc:.3f}")
        else:
            print(f"{var_name:8s} | {attempts:8d} | {correct:7d} | N/A")
    
    # Check X4 specifically
    if 'X4' in data['per_variable_stats']:
        x4_stats = data['per_variable_stats']['X4']
        if x4_stats['attempts'] > 0:
            x4_acc = x4_stats['correct'] / x4_stats['attempts']
            print(f"\n🎯 X4 ACCURACY: {x4_acc:.3f} ({x4_stats['correct']}/{x4_stats['attempts']})")
            
            if x4_acc > 0:
                print("✅ SUCCESS! X4 is being predicted correctly!")
            else:
                print("❌ X4 still has 0% accuracy despite permutation")
        else:
            print("\n⚠️ X4 never appeared as a target in validation")

# Check confusion matrix
print("\n" + "="*60)
print("CONFUSION MATRIX ANALYSIS")
print("="*60)

if 'confusion_matrices' in data and data['confusion_matrices']:
    # Get the last confusion matrix
    last_epoch = max(data['confusion_matrices'].keys())
    cm_data = data['confusion_matrices'][last_epoch]
    cm = cm_data['matrix']
    labels = cm_data.get('labels', ['X0', 'X1', 'X2', 'X3', 'X4'])
    
    print(f"\nConfusion Matrix (Epoch {last_epoch}):")
    print("True \\ Pred", end="")
    for label in labels:
        print(f"\t{label}", end="")
    print()
    
    for i, true_label in enumerate(labels):
        print(f"{true_label}", end="")
        for j in range(len(labels)):
            print(f"\t{cm[i, j]}", end="")
        print()
    
    # Check if X4 appears
    if 'X4' in labels:
        x4_idx = labels.index('X4')
        x4_as_target = cm[x4_idx, :].sum()
        if x4_as_target > 0:
            x4_correct = cm[x4_idx, x4_idx]
            print(f"\nX4 appeared as target {x4_as_target} times")
            print(f"X4 predicted correctly {x4_correct} times")
            print(f"X4 accuracy from confusion matrix: {x4_correct/x4_as_target:.3f}")
        else:
            print("\nX4 never appeared as target in confusion matrix")

# Analyze target distribution
print("\n" + "="*60)
print("TARGET VARIABLE DISTRIBUTION")
print("="*60)

if 'per_variable_stats' in data:
    total_attempts = sum(stats['attempts'] for stats in data['per_variable_stats'].values())
    
    print("\nVariable | Frequency | Percentage")
    print("-" * 35)
    
    for var_name in sorted(data['per_variable_stats'].keys()):
        attempts = data['per_variable_stats'][var_name]['attempts']
        if total_attempts > 0:
            pct = attempts / total_attempts * 100
            bar = '█' * int(pct / 2)
            print(f"{var_name:8s} | {attempts:9d} | {pct:5.1f}% {bar}")

print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)

# Check if permutation helped
if 'X4' in data['per_variable_stats']:
    x4_stats = data['per_variable_stats']['X4']
    if x4_stats['attempts'] > 0:
        x4_acc = x4_stats['correct'] / x4_stats['attempts']
        
        if x4_acc > 0.1:
            print("\n✅ Variable permutation is working! X4 is being learned.")
            print("   Continue training with these settings:")
            print("   - Learning rate: 3e-3")
            print("   - Hidden dim: 128")
            print("   - Variable permutation: ON")
        else:
            print("\n⚠️ X4 still has very low accuracy despite permutation.")
            print("   Recommendations:")
            print("   1. Try permuting every batch (not just every epoch)")
            print("   2. Increase learning rate to 1e-2")
            print("   3. Add focal loss for rare classes")
            print("   4. Use curriculum learning (start with X4-heavy batches)")
    else:
        print("\n⚠️ X4 didn't appear in validation set.")
        print("   This suggests the train/val split might be problematic.")
        print("   Try stratified splitting to ensure X4 appears in validation.")
else:
    print("\nNo X4 statistics available.")