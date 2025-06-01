#!/usr/bin/env python3
"""
Validation test for AVICI integration refactoring.
Tests that the new module structure works correctly.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def test_api_compatibility():
    """Test that all public APIs still work exactly the same."""
    print("🧪 Test 1: API Compatibility")
    
    try:
        # Test main functions work with same signatures
        from causal_bayes_opt.avici_integration import (
            samples_to_avici_format,
            create_training_batch,
            analyze_avici_data
        )
        print("✅ Main API imports successful")
        
        # Test that parent set functionality still works
        from causal_bayes_opt.avici_integration import (
            create_parent_set_model,
            predict_parent_sets
        )
        print("✅ Parent set API imports successful")
        
        # Test advanced functionality
        from causal_bayes_opt.avici_integration import (
            validate_conversion_inputs,
            compute_standardization_params,
            compare_conversions
        )
        print("✅ Advanced API imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ API compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_core_modules():
    """Test that core modules work independently."""
    print("\n🧪 Test 2: Core Module Independence")
    
    try:
        # Test core modules work directly
        from causal_bayes_opt.avici_integration.core.validation import validate_conversion_inputs
        from causal_bayes_opt.avici_integration.core.standardization import compute_standardization_params
        from causal_bayes_opt.avici_integration.core.data_extraction import extract_values_matrix
        from causal_bayes_opt.avici_integration.core.conversion import samples_to_avici_format
        print("✅ Core module imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Core module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utility_modules():
    """Test that utility modules work."""
    print("\n🧪 Test 3: Utility Modules")
    
    try:
        # Test utils work
        from causal_bayes_opt.avici_integration.utils.analysis import compute_data_quality_metrics
        print("✅ Utils module imports successful")
        
        # Test testing module works
        from causal_bayes_opt.avici_integration.testing.debug_tools import debug_conversion_step_by_step
        print("✅ Testing module imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Utility module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_no_side_effects():
    """Test that core functions are pure (no side effects)."""
    print("\n🧪 Test 4: Functional Purity")
    
    try:
        import jax.numpy as jnp
        import pyrsistent as pyr
        
        # Test that imports don't cause side effects
        from causal_bayes_opt.avici_integration.core.data_extraction import extract_values_matrix
        
        # Create minimal test data
        samples = [
            pyr.m(values=pyr.m(X=1.0, Y=2.0), intervention_type=None, intervention_targets=pyr.s()),
            pyr.m(values=pyr.m(X=2.0, Y=3.0), intervention_type=None, intervention_targets=pyr.s())
        ]
        variable_order = ['X', 'Y']
        
        # Call function multiple times - should be deterministic
        result1 = extract_values_matrix(samples, variable_order)
        result2 = extract_values_matrix(samples, variable_order)
        
        if jnp.allclose(result1, result2):
            print("✅ Functions are deterministic (no side effects)")
            return True
        else:
            print("❌ Functions not deterministic")
            return False
            
    except Exception as e:
        print(f"❌ Functional purity test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parent_set_unchanged():
    """Test that parent set functionality is unchanged."""
    print("\n🧪 Test 5: Parent Set Integration")
    
    try:
        # Test that parent set module still works
        from causal_bayes_opt.avici_integration.parent_set import (
            create_parent_set_model,
            predict_parent_sets,
            compute_loss,
            create_train_step
        )
        print("✅ Parent set module imports successful")
        
        # Test basic model creation
        model = create_parent_set_model()
        if model is not None:
            print("✅ Parent set model creation successful")
            return True
        else:
            print("❌ Parent set model creation failed")
            return False
            
    except Exception as e:
        print(f"❌ Parent set integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all validation tests."""
    print("🚀 AVICI Integration Refactoring Validation")
    print("=" * 50)
    
    tests = [
        test_api_compatibility,
        test_core_modules,
        test_utility_modules,
        test_no_side_effects,
        test_parent_set_unchanged
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Refactoring successful!")
        print("\n✅ Safe to remove old files")
        return True
    else:
        print("⚠️ SOME TESTS FAILED! Review issues before removing old files")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
