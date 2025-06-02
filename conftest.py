"""
Pytest configuration for causal_bayes_opt tests.

This file automatically sets up the Python path so that tests can import
from the causal_bayes_opt package without manual sys.path manipulation.
"""

import sys
import os
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent
src_dir = project_root / "src"

# Add src directory to Python path if not already present
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Verify the package can be imported
try:
    import causal_bayes_opt
    print(f"✓ causal_bayes_opt package found at: {causal_bayes_opt.__file__}")
except ImportError as e:
    print(f"✗ Failed to import causal_bayes_opt: {e}")
    print(f"  - Project root: {project_root}")
    print(f"  - Src directory: {src_dir}")
    print(f"  - Src exists: {src_dir.exists()}")
    raise
