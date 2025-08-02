#!/usr/bin/env python3
"""
Test evaluation with verbose logging using the existing framework.
"""

import subprocess
import sys
import os

# Enable debug logging
os.environ['LOGGING_LEVEL'] = 'DEBUG'

# Run evaluation with model registry approach
cmd = [
    sys.executable, 'scripts/evaluate_acbo_methods.py',
    # Register models
    '--register_policy', 'grpo', 'checkpoints/validation/unified_grpo_final',
    '--register_policy', 'bc', 'checkpoints/validation/bc_final',
    '--register_surrogate', 'bc_surrogate', 'checkpoints/validation/bc_surrogate_final',
    '--register_surrogate', 'none', 'dummy',
    # Evaluate specific pairs
    '--evaluate_pairs', 'grpo', 'none',
    '--evaluate_pairs', 'grpo', 'bc_surrogate',
    '--evaluate_pairs', 'bc', 'none', 
    '--evaluate_pairs', 'bc', 'bc_surrogate',
    # Small test
    '--n_scms', '2',
    '--n_interventions', '5',
    '--plot',
    '--output_dir', 'evaluation_results/debug_verbose'
]

print("Running evaluation with verbose logging...")
print("Command:", ' '.join(cmd))
print("="*80)

# Run with output
result = subprocess.run(cmd, text=True)

print("\n" + "="*80)
print("Evaluation completed!")
print("Check evaluation_results/debug_verbose/ for results and plots")