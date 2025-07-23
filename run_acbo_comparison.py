#!/usr/bin/env python3
"""
Wrapper script to run ACBO comparison experiments.
This avoids import path issues by running from the project root.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Change to the scripts/core directory so Hydra can find the config
os.chdir(project_root / "scripts" / "core")

# Now import and run the main function
from scripts.core.run_acbo_comparison import main

if __name__ == "__main__":
    main()