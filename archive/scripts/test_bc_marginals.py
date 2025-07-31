#!/usr/bin/env python3
"""Quick test to verify BC surrogate marginals are extracted correctly."""

import subprocess
import sys

# Run evaluation on 1 SCM with just 1 intervention
cmd = [
    sys.executable, 'scripts/evaluate_acbo_methods.py',
    '--register_policy', 'bc', 'checkpoints/validation/bc_final',
    '--register_surrogate', 'bc_surrogate', 'checkpoints/validation/bc_surrogate_final',
    '--evaluate_pairs', 'bc', 'bc_surrogate',
    '--n_scms', '1',
    '--n_interventions', '1',  # Just 1 intervention
    '--n_obs', '20',
    '--n_samples', '10',
    '--output_dir', 'evaluation_results/test_marginals',
    '--seed', '42'
]

print("Running minimal test...")
result = subprocess.run(cmd, capture_output=True, text=True)

# Look for the key lines
lines = result.stdout.split('\n') + result.stderr.split('\n')

print("\n=== BC SURROGATE OUTPUT ===")
for i, line in enumerate(lines):
    if "BC Surrogate - Converting parent probabilities:" in line:
        # Print next 10 lines
        for j in range(i, min(i+10, len(lines))):
            print(lines[j])
        break

print("\n=== F1 CALCULATION ===")        
for i, line in enumerate(lines):
    if "F1 Calculation (threshold" in line:
        # Print next 15 lines
        for j in range(i, min(i+15, len(lines))):
            print(lines[j])
        break

print("\n=== FINAL F1 SCORE ===")
for line in lines:
    if "Final F1 score:" in line:
        print(line)
        break