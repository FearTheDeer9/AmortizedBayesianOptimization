#!/usr/bin/env python3
"""
Analyze the suspicious accuracy pattern where some variables have perfect accuracy
while others (especially X2) have terrible accuracy.
"""

import sys
from pathlib import Path
import numpy as np
from collections import defaultdict
import pickle

sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.training.data_preprocessing import load_demonstrations_from_path
from demonstration_to_tensor_fixed import create_bc_training_dataset
from variable_mapping_fixed import VariableMapper

def analyze_target_distribution(labels):
    """Analyze how often each variable appears as a target."""
    target_counts = defaultdict(int)
    total_by_scm_size = defaultdict(int)
    targets_by_scm_size = defaultdict(lambda: defaultdict(int))
    
    for label in labels:
        variables = label.get('variables', [])
        targets = list(label.get('targets', []))
        
        if not targets:
            continue
            
        target_var = targets[0]
        n_vars = len(variables)
        
        target_counts[target_var] += 1
        total_by_scm_size[n_vars] += 1
        targets_by_scm_size[n_vars][target_var] += 1
    
    return target_counts, total_by_scm_size, targets_by_scm_size

def analyze_variable_existence(labels):
    """Check which variables actually exist in the data."""
    variable_existence = defaultdict(int)
    scm_sizes = defaultdict(int)
    
    for label in labels:
        variables = label.get('variables', [])
        scm_sizes[len(variables)] += 1
        
        for var in variables:
            variable_existence[var] += 1
    
    return variable_existence, scm_sizes

def analyze_confusion_patterns(predictions_log):
    """Analyze what the model predicts when it should predict each variable."""
    confusion = defaultdict(lambda: defaultdict(int))
    
    # This would need actual model predictions
    # For now, we'll analyze the structure
    return confusion

def main():
    print("="*80)
    print("ANALYZING SUSPICIOUS ACCURACY PATTERNS")
    print("="*80)
    
    # Load demonstrations
    demos_path = Path("../expert_demonstrations/raw/raw_demonstrations")
    if not demos_path.exists():
        demos_path = Path("expert_demonstrations/raw/raw_demonstrations")
    
    raw_demos = load_demonstrations_from_path(str(demos_path), max_files=100)
    
    # Flatten
    flat_demos = []
    for item in raw_demos:
        if hasattr(item, 'demonstrations'):
            flat_demos.extend(item.demonstrations)
        else:
            flat_demos.append(item)
    
    # Create dataset
    all_inputs, all_labels, metadata = create_bc_training_dataset(
        flat_demos, max_trajectory_length=100
    )
    
    print(f"\nAnalyzing {len(all_labels)} training examples")
    
    # 1. Analyze target distribution
    print("\n" + "="*60)
    print("TARGET VARIABLE DISTRIBUTION")
    print("="*60)
    
    target_counts, scm_totals, targets_by_scm = analyze_target_distribution(all_labels)
    
    # Sort by count
    sorted_targets = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("\nOverall target frequencies:")
    print(f"{'Variable':<10} | {'Count':<10} | {'Percentage':<10}")
    print("-"*40)
    total_targets = sum(target_counts.values())
    for var, count in sorted_targets:
        pct = (count / total_targets) * 100
        print(f"{var:<10} | {count:<10} | {pct:.1f}%")
    
    # Suspicious variables
    suspicious_perfect = ['X3', 'X6', 'X8', 'X10']
    suspicious_bad = ['X2']
    
    print("\n⚠️ SUSPICIOUS VARIABLES:")
    print("\nPerfect accuracy (100%):")
    for var in suspicious_perfect:
        count = target_counts.get(var, 0)
        pct = (count / total_targets * 100) if total_targets > 0 else 0
        print(f"  {var}: {count} occurrences ({pct:.1f}%)")
    
    print("\nTerrible accuracy (34.5%):")
    for var in suspicious_bad:
        count = target_counts.get(var, 0)
        pct = (count / total_targets * 100) if total_targets > 0 else 0
        print(f"  {var}: {count} occurrences ({pct:.1f}%)")
    
    # 2. Analyze by SCM size
    print("\n" + "="*60)
    print("TARGET DISTRIBUTION BY SCM SIZE")
    print("="*60)
    
    for scm_size in sorted(scm_totals.keys()):
        print(f"\nSCM with {scm_size} variables ({scm_totals[scm_size]} examples):")
        targets = targets_by_scm[scm_size]
        for var in sorted(targets.keys()):
            count = targets[var]
            pct = (count / scm_totals[scm_size]) * 100
            print(f"  {var}: {count} ({pct:.1f}%)")
    
    # 3. Analyze variable existence
    print("\n" + "="*60)
    print("VARIABLE EXISTENCE IN DATASET")
    print("="*60)
    
    var_existence, scm_size_counts = analyze_variable_existence(all_labels)
    
    print("\nSCM size distribution:")
    for size, count in sorted(scm_size_counts.items()):
        print(f"  {size} variables: {count} examples")
    
    print("\nVariable appearance counts:")
    for var in sorted(var_existence.keys(), key=lambda x: (int(x[1:]) if x[1:].isdigit() else 999)):
        count = var_existence[var]
        pct = (count / len(all_labels)) * 100
        print(f"  {var}: appears in {count}/{len(all_labels)} examples ({pct:.1f}%)")
    
    # 4. Key insights
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    
    # Check if suspicious variables only appear in certain SCM sizes
    print("\n1. WHERE DO SUSPICIOUS VARIABLES APPEAR?")
    
    for var in suspicious_perfect + suspicious_bad:
        print(f"\n{var} appears as target in SCMs of size:")
        appears_in = []
        for scm_size, targets in targets_by_scm.items():
            if var in targets:
                appears_in.append(f"{scm_size} ({targets[var]} times)")
        if appears_in:
            print(f"  {', '.join(appears_in)}")
        else:
            print(f"  Never appears as target!")
    
    # Check if there's a pattern
    print("\n2. HYPOTHESIS CHECK:")
    
    # Check if perfect accuracy vars are rare
    perfect_total = sum(target_counts.get(var, 0) for var in suspicious_perfect)
    print(f"\nPerfect accuracy variables (X3,X6,X8,X10) total: {perfect_total} targets")
    
    # Check if X2 is common
    x2_count = target_counts.get('X2', 0)
    print(f"X2 appears as target: {x2_count} times")
    
    # Check if perfect vars only appear in large SCMs
    print("\n3. SCM SIZE PATTERN:")
    small_scm_vars = set()
    large_scm_vars = set()
    
    for scm_size, targets in targets_by_scm.items():
        if scm_size <= 5:
            small_scm_vars.update(targets.keys())
        else:
            large_scm_vars.update(targets.keys())
    
    only_large = large_scm_vars - small_scm_vars
    only_small = small_scm_vars - large_scm_vars
    both = small_scm_vars & large_scm_vars
    
    print(f"\nVariables only in SCMs with ≤5 vars: {sorted(only_small)}")
    print(f"Variables only in SCMs with >5 vars: {sorted(only_large)}")
    print(f"Variables in both: {sorted(both)}")
    
    # The smoking gun
    print("\n" + "="*60)
    print("🔍 THE SMOKING GUN")
    print("="*60)
    
    if not any(target_counts.get(var, 0) > 0 for var in suspicious_perfect):
        print("\n⚠️ CRITICAL: Variables with 'perfect' accuracy never appear in training data!")
        print("This suggests the model is getting credit for never predicting them.")
    
    if x2_count > 100:
        print(f"\n⚠️ X2 appears {x2_count} times but has terrible accuracy.")
        print("This suggests a systematic prediction error for X2.")

if __name__ == "__main__":
    main()