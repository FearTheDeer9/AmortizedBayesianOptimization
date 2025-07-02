#!/usr/bin/env python3
"""
Test script to verify SCM suite functionality end-to-end.
This validates that the SCM suite can be used with the experiment infrastructure.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from omegaconf import DictConfig, OmegaConf
from scripts.core.acbo_wandb_experiment import generate_test_scms
from src.causal_bayes_opt.experiments.benchmark_scms import get_scm_characteristics
from examples.demo_learning import _extract_target_coefficients_from_scm


def test_scm_suite_integration():
    """Test complete SCM suite integration."""
    
    print("=== Testing SCM Suite Integration ===\n")
    
    # Test configuration for SCM suite
    cfg_dict = {
        'experiment': {
            'scm_suite': {
                'enabled': True,
                'scm_names': ['fork_3var', 'diamond_4var', 'butterfly_5var']
            },
            'environment': {'num_variables': 5, 'noise_scale': 1.0},
            'problem': {'edge_density': 0.3}
        },
        'n_scms': 2,
        'seed': 42
    }
    
    cfg = OmegaConf.create(cfg_dict)
    scm_data = generate_test_scms(cfg)
    
    print(f"Generated {len(scm_data)} SCMs from suite:")
    
    for scm_name, scm in scm_data:
        print(f"\n--- {scm_name} ---")
        
        # Get characteristics
        chars = get_scm_characteristics(scm)
        print(f"Structure: {chars['description']}")
        print(f"Variables: {chars['num_variables']} | Edges: {chars['num_edges']} | Target: {chars['target_variable']}")
        print(f"Target parents: {chars['target_parents']}")
        print(f"Coefficients: {chars['coefficients']}")
        
        # Test oracle coefficient extraction
        target = chars['target_variable']
        extracted_coeffs = _extract_target_coefficients_from_scm(scm, target)
        print(f"Oracle coefficients: {extracted_coeffs}")
        
        # Verify coefficients match
        expected_coeffs = {parent: chars['coefficients'].get((parent, target), 0.0) 
                          for parent in chars['target_parents']}
        
        if extracted_coeffs == expected_coeffs:
            print("✓ Oracle coefficient extraction: PASS")
        else:
            print(f"✗ Oracle coefficient extraction: FAIL")
            print(f"  Expected: {expected_coeffs}")
            print(f"  Got: {extracted_coeffs}")
    
    print("\n=== Test completed successfully! ===")


if __name__ == "__main__":
    test_scm_suite_integration()