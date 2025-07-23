#!/usr/bin/env python3
"""
Fix plot labels to correctly reflect optimization direction.

This script updates plot labels to be clear about what's being shown
and whether higher or lower values are better.
"""

import sys
from pathlib import Path

# Files to fix
FILES_TO_FIX = [
    "src/causal_bayes_opt/visualization/plots.py",
    "scripts/core/acbo_comparison/visualization.py",
    "scripts/unified_pipeline.py"
]

# Replacements to make
REPLACEMENTS = [
    # Fix the main confusion in plot_baseline_comparison
    {
        "file": "src/causal_bayes_opt/visualization/plots.py",
        "old": "ax3.set_title('Target Value (Actual per Step - Higher is Better)')",
        "new": "ax3.set_title('Target Value Progress (Lower is Better for Minimization)')"
    },
    {
        "file": "src/causal_bayes_opt/visualization/plots.py", 
        "old": "ax3.set_ylabel('Target Value')",
        "new": "ax3.set_ylabel('Target Value (Minimization Goal)')"
    },
    # Fix structure learning plot
    {
        "file": "src/causal_bayes_opt/visualization/plots.py",
        "old": "ax3.set_title('Structure Distance (Lower is Better)')",
        "new": "ax3.set_title('Structural Hamming Distance (Lower is Better)')"
    },
    # Fix unified pipeline
    {
        "file": "scripts/unified_pipeline.py",
        "old": "ax3.set_ylabel('Target Value')",
        "new": "ax3.set_ylabel('Target Improvement')"
    },
    {
        "file": "scripts/unified_pipeline.py",
        "old": "ax3.set_title('Final Target Value (Lower is Better)')",
        "new": "ax3.set_title('Final Target Improvement (Higher is Better)')"
    },
    {
        "file": "scripts/unified_pipeline.py",
        "old": "ax3.set_title('Final Target Value (Higher is Better)')",
        "new": "ax3.set_title('Final Target Improvement (Higher is Better)')"
    }
]

def main():
    """Apply fixes to plot labels."""
    project_root = Path(__file__).parent.parent.parent
    
    print("Fixing plot labels for clarity...")
    
    for fix in REPLACEMENTS:
        file_path = project_root / fix["file"]
        
        if not file_path.exists():
            print(f"⚠️ File not found: {file_path}")
            continue
            
        # Read file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if old string exists
        if fix["old"] not in content:
            print(f"⚠️ String not found in {fix['file']}: {fix['old'][:50]}...")
            continue
        
        # Replace
        new_content = content.replace(fix["old"], fix["new"])
        
        # Write back
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        print(f"✅ Fixed: {fix['file']}")
    
    print("\n✅ Plot label fixes complete!")
    print("\nKey changes:")
    print("1. Target value plots now clearly indicate optimization direction")
    print("2. Removed confusing 'Higher is Better' for minimization targets")
    print("3. Made labels more descriptive about what's being shown")

if __name__ == "__main__":
    main()