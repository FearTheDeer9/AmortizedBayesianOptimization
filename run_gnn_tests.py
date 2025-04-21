#!/usr/bin/env python
"""
Specific test runner for GNN comprehensive tests.
"""

import os
import sys
import subprocess

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Run the tests
if __name__ == "__main__":
    # Run pytest on the GNN comprehensive tests
    result = subprocess.call([
        "python",
        "-m",
        "pytest",
        "tests/test_gnn_comprehensive.py",
        "-v"
    ])

    # Exit with appropriate status code
    sys.exit(result)
