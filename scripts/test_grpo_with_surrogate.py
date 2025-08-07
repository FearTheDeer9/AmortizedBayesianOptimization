#!/usr/bin/env python3
"""
Test GRPO with surrogate training.
"""

import sys
from pathlib import Path
import subprocess

# Run train_acbo_methods.py with grpo_with_surrogate
cmd = [
    sys.executable,
    "scripts/train_acbo_methods.py",
    "--method", "grpo_with_surrogate",
    "--episodes", "10",  # Very minimal
    "--batch_size", "8",
    "--hidden_dim", "64",
    "--scm_type", "fork",
    "--min_vars", "3",
    "--max_vars", "3",
    "--checkpoint_dir", "checkpoints/test_grpo_with_surrogate",
    "--seed", "998"
]

print("Running GRPO with surrogate training...")
print("Command:", " ".join(cmd))
print("=" * 80)

# Run and capture output
result = subprocess.run(cmd, capture_output=False, text=True)

print("\n" + "=" * 80)
print("CHECKING RESULTS")
print("=" * 80)

# Check if checkpoint was created
checkpoint_path = Path("checkpoints/test_grpo_with_surrogate/unified_grpo_final")
if checkpoint_path.exists():
    print(f"✓ Checkpoint created: {checkpoint_path}")
    print(f"  Contents: {list(checkpoint_path.iterdir())}")
else:
    print(f"✗ No checkpoint at: {checkpoint_path}")
        
    # List all files
    parent_dir = Path("checkpoints/test_grpo_with_surrogate")
    if parent_dir.exists():
        print(f"\nFiles in {parent_dir}:")
        for f in parent_dir.rglob("*"):
            if f.is_file():
                print(f"  - {f.relative_to(parent_dir)}")

print("\nReturn code:", result.returncode)