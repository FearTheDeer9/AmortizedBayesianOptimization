#!/usr/bin/env python
"""
Test runner that ensures the proper Python path is set up.
"""

import os
import sys
import subprocess

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Discover and run tests
if __name__ == "__main__":
    # Run pytest on all test files
    result = subprocess.call([
        "python",
        "-m",
        "pytest",
        "tests/",
        "-v"
    ])

    # Exit with appropriate status code
    sys.exit(result)
