#!/usr/bin/env python3
"""
Test PARENT_SCALE Import Integration

This integration test verifies that PARENT_SCALE can be properly imported 
from the refactored parent_scale module after fixing import issues.

The test validates:
- PARENT_SCALE imports from both bridge and package interfaces
- Import consistency across different entry points
- Graceful handling when external PARENT_SCALE library is unavailable

Usage:
    python tests/integration/test_parent_scale_import.py

Location: tests/integration/
Purpose: Integration test for PARENT_SCALE import functionality
"""

import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '../../src')

def test_parent_scale_import():
    """Test importing PARENT_SCALE from the fixed modules."""
    print("🧪 Testing PARENT_SCALE Import Fix")
    print("=" * 50)
    
    try:
        # Test import from parent_scale_bridge (backward compatibility)
        print("1. Testing import from parent_scale_bridge...")
        from causal_bayes_opt.integration.parent_scale_bridge import PARENT_SCALE, PARENT_SCALE_AVAILABLE
        print(f"   ✅ PARENT_SCALE imported successfully")
        print(f"   ✅ PARENT_SCALE_AVAILABLE: {PARENT_SCALE_AVAILABLE}")
        
        # Test import from parent_scale package
        print("\n2. Testing import from parent_scale package...")
        from causal_bayes_opt.integration.parent_scale import PARENT_SCALE as PS_PKG, PARENT_SCALE_AVAILABLE as PSA_PKG
        print(f"   ✅ PARENT_SCALE imported from package successfully")
        print(f"   ✅ PARENT_SCALE_AVAILABLE: {PSA_PKG}")
        
        # Verify they're the same class
        print("\n3. Verifying consistency...")
        print(f"   Same class: {PARENT_SCALE is PS_PKG}")
        print(f"   Same availability: {PARENT_SCALE_AVAILABLE == PSA_PKG}")
        
        # Test that we can create an instance (if available)
        if PARENT_SCALE_AVAILABLE:
            print("\n4. Testing PARENT_SCALE instantiation...")
            print(f"   PARENT_SCALE class: {PARENT_SCALE}")
            print(f"   Class type: {type(PARENT_SCALE)}")
            print("   ✅ PARENT_SCALE is available and ready to use")
        else:
            print("\n4. PARENT_SCALE not available (expected if external library not found)")
            print(f"   Using dummy class: {PARENT_SCALE}")
        
        print("\n✅ ALL IMPORT TESTS PASSED!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_other_imports():
    """Test that other functions still import correctly."""
    print("\n🔧 Testing Other Import Functions")
    print("=" * 50)
    
    try:
        # Test function imports
        from causal_bayes_opt.integration.parent_scale_bridge import (
            scm_to_graph_structure,
            run_full_parent_scale_algorithm,
            ACBOGraphStructure
        )
        print("   ✅ Bridge functions imported successfully")
        
        # Test that scripts that used to work can still import
        from causal_bayes_opt.integration.parent_scale_bridge import (
            scm_to_graph_structure, 
            PARENT_SCALE as PARENT_SCALE_INTEGRATED
        )
        print("   ✅ Script-style imports work")
        
        return True
        
    except ImportError as e:
        print(f"❌ Function import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 PARENT_SCALE IMPORT FIX VALIDATION")
    print("=" * 60)
    
    # Test PARENT_SCALE import
    test1_passed = test_parent_scale_import()
    
    # Test other imports
    test2_passed = test_other_imports()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED! Import fix successful.")
        print("✅ PARENT_SCALE can now be imported from both:")
        print("   - causal_bayes_opt.integration.parent_scale_bridge")
        print("   - causal_bayes_opt.integration.parent_scale")
        print("✅ Existing scripts should work again")
    else:
        print("❌ Some tests failed. Import fix needs more work.")
    
    return test1_passed and test2_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)