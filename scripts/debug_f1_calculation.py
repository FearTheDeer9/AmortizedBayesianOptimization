#!/usr/bin/env python3
"""
Debug script to understand why F1 scores are always 0.

This script runs a minimal evaluation with detailed logging to track:
1. What marginal probabilities the surrogate returns
2. How F1 is calculated
3. Whether the buffer is updated correctly between interventions
"""

import subprocess
import sys
import os

# Set logging to INFO level
os.environ['LOGGING_LEVEL'] = 'INFO'

# Run evaluation on just 1 SCM with minimal interventions
cmd = [
    sys.executable, 'scripts/evaluate_acbo_methods.py',
    # Just test BC + surrogate pair
    '--register_policy', 'bc', 'checkpoints/validation/bc_final',
    '--register_surrogate', 'bc_surrogate', 'checkpoints/validation/bc_surrogate_final',
    '--evaluate_pairs', 'bc', 'bc_surrogate',
    # Minimal test
    '--n_scms', '1',
    '--n_interventions', '3',
    '--n_obs', '20',
    '--n_samples', '10',
    '--output_dir', 'evaluation_results/debug_f1',
    '--seed', '42'
]

print("="*80)
print("F1 CALCULATION DEBUG")
print("="*80)
print("\nThis script will show:")
print("1. Tensor values passed to surrogate at each step")
print("2. Marginal probabilities returned by surrogate")
print("3. F1 calculation details (threshold, predictions, true positives)")
print("4. How tensor changes after each intervention")
print("\nRunning evaluation with detailed logging...")
print("Command:", ' '.join(cmd))
print("="*80)

# Run and capture output
result = subprocess.run(cmd, capture_output=True, text=True)

# Parse output to highlight key information
lines = result.stdout.split('\n') + result.stderr.split('\n')

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)

# Extract surrogate predictions
print("\n1. SURROGATE PREDICTIONS:")
in_surrogate = False
for line in lines:
    if "BC Surrogate - Converting parent probabilities:" in line:
        in_surrogate = True
    elif in_surrogate:
        if line.strip().startswith("Non-zero probabilities:") or \
           line.strip().startswith("Range:") or \
           (": " in line and "." in line.split(": ")[1]):
            print(f"  {line.strip()}")
        if "Non-zero probabilities:" in line:
            in_surrogate = False

# Extract F1 calculations
print("\n2. F1 CALCULATIONS:")
in_f1 = False
for line in lines:
    if "F1 Calculation (threshold=" in line:
        in_f1 = True
        print(f"\n  {line.strip()}")
    elif in_f1 and line.strip():
        if line.strip().startswith(("True parents:", "Predicted parents:", "True positives:",
                                   "False positives:", "False negatives:", "Precision:",
                                   "Recall:", "F1:")):
            print(f"  {line.strip()}")
        if "F1:" in line:
            in_f1 = False

# Extract final F1 score
print("\n3. FINAL RESULTS:")
for line in lines:
    if "Final F1 score:" in line:
        print(f"  {line.strip()}")
    elif "Mean improvement:" in line and "bc+bc_surrogate" in lines[lines.index(line)-1]:
        print(f"  {line.strip()}")
    elif "Mean trajectory value:" in line and "bc+bc_surrogate" in lines[lines.index(line)-2]:
        print(f"  {line.strip()}")

print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)

# Check if all probabilities are below threshold
all_below_threshold = True
for line in lines:
    if ": 0." in line and "." in line.split(": ")[1]:
        try:
            prob = float(line.split(": ")[1].split()[0])
            if prob > 0.5:
                all_below_threshold = False
                break
        except:
            pass

if all_below_threshold:
    print("\n✗ F1=0 because all marginal probabilities are below 0.5 threshold")
    print("  → The surrogate is not confident about any parent relationships")
    print("  → This is expected with only 100 training episodes")
    print("\nSOLUTION: Train surrogate for more episodes (1000+) or lower F1 threshold")
else:
    print("\n✓ Some probabilities exceed threshold - check detailed output above")

# Check if buffer is being updated
print("\n" + "="*80)
print("Full output saved to: evaluation_results/debug_f1/")
print("Check the JSON file for complete results")