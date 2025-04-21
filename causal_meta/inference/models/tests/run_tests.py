#!/usr/bin/env python
"""
Script to run all tests for the GNN models.
"""

import unittest
import os
import sys

# Add the project root to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../../../')))


if __name__ == "__main__":
    # Discover and run all tests in the current directory
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(
        os.path.dirname(__file__), pattern="test_*.py")

    # Run the tests
    unittest.TextTestRunner(verbosity=2).run(test_suite)
