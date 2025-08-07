#!/usr/bin/env python3
"""
Minimal test for GRPO training to debug checkpoint saving.
"""

import sys
from pathlib import Path
import subprocess

# Just run train_acbo_methods.py with minimal config
cmd = [
    sys.executable,
    "scripts/train_acbo_methods.py",
    "--method", "grpo",
    "--episodes", "10",  # Very minimal
    "--batch_size", "8",
    "--hidden_dim", "64",
    "--scm_type", "fork",
    "--min_vars", "3",
    "--max_vars", "3",
    "--checkpoint_dir", "checkpoints/test_minimal_grpo",
    "--seed", "999"
]

print("Running minimal GRPO training...")
print("Command:", " ".join(cmd))
print("=" * 80)

# Run and capture output
result = subprocess.run(cmd, capture_output=False, text=True)

print("\n" + "=" * 80)
print("CHECKING RESULTS")
print("=" * 80)

# Check if checkpoint was created
checkpoint_path = Path("checkpoints/test_minimal_grpo/grpo_no_surrogate_final")
if checkpoint_path.exists():
    print(f"✓ Checkpoint created: {checkpoint_path}")
    print(f"  Contents: {list(checkpoint_path.iterdir())}")
else:
    print(f"✗ No checkpoint at: {checkpoint_path}")
    
    # Check for unified_grpo_final
    alt_path = Path("checkpoints/test_minimal_grpo/unified_grpo_final")
    if alt_path.exists():
        print(f"  Found at alternate location: {alt_path}")
    else:
        print(f"  Also not at: {alt_path}")
        
    # List all files
    parent_dir = Path("checkpoints/test_minimal_grpo")
    if parent_dir.exists():
        print(f"\nFiles in {parent_dir}:")
        for f in parent_dir.iterdir():
            print(f"  - {f}")

print("\nReturn code:", result.returncode)