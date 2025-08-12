#!/usr/bin/env python3
"""
Quick check of variable naming patterns in demonstrations.
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.training.data_preprocessing import load_demonstrations_from_path
from src.causal_bayes_opt.data_structures.scm import get_variables

def check_variable_naming():
    """Check how variables are named in different SCMs."""
    
    print("="*60)
    print("VARIABLE NAMING CHECK")
    print("="*60)
    
    # Load just a few demonstrations
    demos_path = 'expert_demonstrations/raw/raw_demonstrations'
    raw_demos = load_demonstrations_from_path(demos_path, max_files=10)
    
    # Flatten
    flat_demos = []
    for item in raw_demos:
        if hasattr(item, 'demonstrations'):
            flat_demos.extend(item.demonstrations[:5])  # Just first 5 from each
        else:
            flat_demos.append(item)
    
    print(f"\nProcessing {len(flat_demos)} demonstrations")
    
    # Track variable patterns
    scm_variable_patterns = defaultdict(list)
    
    for i, demo in enumerate(flat_demos[:20]):  # Just first 20
        if hasattr(demo, 'scm'):
            scm = demo.scm
            variables = sorted(list(get_variables(scm)))
            n_vars = len(variables)
            
            # Key insight: Group by number of variables
            scm_variable_patterns[n_vars].append(variables)
            
            # Check if target_variable exists
            if hasattr(demo, 'target_variable'):
                target = demo.target_variable
                print(f"\nDemo {i}: {n_vars} vars, target={target}")
                print(f"  Variables: {variables}")
                
                # Check trajectory
                if hasattr(demo, 'parent_posterior'):
                    trajectory = demo.parent_posterior.get('trajectory', {})
                    intervention_sequence = trajectory.get('intervention_sequence', [])
                    if intervention_sequence:
                        first_intervention = intervention_sequence[0]
                        if isinstance(first_intervention, tuple):
                            first_intervention = first_intervention[0]
                        print(f"  First intervention: {first_intervention}")
    
    # Analyze patterns
    print("\n" + "="*60)
    print("VARIABLE PATTERNS BY SCM SIZE")
    print("="*60)
    
    for n_vars in sorted(scm_variable_patterns.keys()):
        examples = scm_variable_patterns[n_vars]
        print(f"\nSCMs with {n_vars} variables:")
        # Show first unique pattern
        unique = list(set(tuple(ex) for ex in examples))
        for pattern in unique[:3]:
            print(f"  Pattern: {list(pattern)}")

if __name__ == "__main__":
    check_variable_naming()