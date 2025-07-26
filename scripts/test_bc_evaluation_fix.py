#!/usr/bin/env python3
"""
Test script to verify BC evaluation works after fixing imports.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import pyrsistent as pyr

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_bc_model_loading():
    """Test that BC models can be loaded properly."""
    from src.causal_bayes_opt.training.bc_model_loader import load_bc_model, validate_checkpoint
    
    # Create dummy checkpoints for testing
    test_dir = Path("test_bc_checkpoints")
    test_dir.mkdir(exist_ok=True)
    
    # Use the test checkpoint creation from our previous test
    from scripts.test_bc_model_loading import (
        create_test_surrogate_checkpoint,
        create_test_acquisition_checkpoint
    )
    
    surrogate_checkpoint = test_dir / "test_surrogate.pkl"
    acquisition_checkpoint = test_dir / "test_acquisition.pkl"
    
    print("Creating test checkpoints...")
    create_test_surrogate_checkpoint(surrogate_checkpoint)
    create_test_acquisition_checkpoint(acquisition_checkpoint)
    
    # Test loading through the fixed imports
    print("\n1️⃣ Testing direct BC model loading...")
    
    # Test surrogate
    validation = validate_checkpoint(str(surrogate_checkpoint))
    if validation['valid']:
        print("✅ Surrogate checkpoint validation passed")
        surrogate = load_bc_model(str(surrogate_checkpoint), 'surrogate')
        print(f"✅ Surrogate loaded, type: {type(surrogate)}")
    else:
        print(f"❌ Surrogate validation failed: {validation}")
        
    # Test acquisition
    validation = validate_checkpoint(str(acquisition_checkpoint))
    if validation['valid']:
        print("✅ Acquisition checkpoint validation passed")
        acquisition = load_bc_model(str(acquisition_checkpoint), 'acquisition')
        print(f"✅ Acquisition loaded, type: {type(acquisition)}")
    else:
        print(f"❌ Acquisition validation failed: {validation}")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    
    return True


def test_bc_evaluation_setup():
    """Test that BC evaluation can be set up through unified framework."""
    from src.causal_bayes_opt.evaluation import BCEvaluator
    
    print("\n2️⃣ Testing BC evaluator initialization...")
    
    # Test without checkpoints
    try:
        bc_eval = BCEvaluator(name="Test_Empty")
        print("✅ BC evaluator initialized without checkpoints")
    except Exception as e:
        print(f"❌ BC evaluator initialization failed: {e}")
        return False
        
    # Test with checkpoint paths (they don't need to exist for initialization)
    try:
        bc_eval = BCEvaluator(
            surrogate_checkpoint=Path("dummy_surrogate.pkl"),
            acquisition_checkpoint=Path("dummy_acquisition.pkl"),
            name="Test_Both"
        )
        print("✅ BC evaluator initialized with checkpoint paths")
        print(f"   Name: {bc_eval.name}")
        print(f"   Use surrogate: {bc_eval.use_surrogate}")
        print(f"   Use acquisition: {bc_eval.use_acquisition}")
    except Exception as e:
        print(f"❌ BC evaluator with checkpoints failed: {e}")
        return False
        
    return True


def test_method_registry_imports():
    """Test that method registry can import BC models."""
    print("\n3️⃣ Testing method registry imports...")
    
    try:
        # This will test if the imports work
        from scripts.core.acbo_comparison.method_registry import MethodRegistry
        registry = MethodRegistry()
        print("✅ Method registry imported successfully")
        
        # Test that BC methods are registered
        methods = registry.list_available_methods()
        bc_methods = [m for m in methods if 'bc' in m.lower()]
        print(f"✅ Found {len(bc_methods)} BC methods: {bc_methods}")
        
        return True
    except Exception as e:
        print(f"❌ Method registry import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scm_creation():
    """Test SCM creation for evaluation."""
    print("\n4️⃣ Testing SCM creation...")
    
    try:
        from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
        
        factory = VariableSCMFactory(seed=42)
        scm = factory.create_variable_scm(
            num_variables=3,
            structure_type='chain',
            target_variable=None
        )
        
        # Set optimization direction
        current_metadata = scm.get('metadata', pyr.pmap())
        updated_metadata = current_metadata.set('optimization_direction', 'MINIMIZE')
        scm = scm.set('metadata', updated_metadata)
        
        print("✅ SCM created successfully")
        print(f"   Variables: {len(list(scm.get('variables', [])))} ")
        print(f"   Optimization: {updated_metadata.get('optimization_direction')}")
        
        return True
    except Exception as e:
        print(f"❌ SCM creation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("BC EVALUATION FIX TEST")
    print("=" * 80)
    
    tests = [
        ("BC Model Loading", test_bc_model_loading),
        ("BC Evaluation Setup", test_bc_evaluation_setup),
        ("Method Registry Imports", test_method_registry_imports),
        ("SCM Creation", test_scm_creation)
    ]
    
    results = []
    for test_name, test_fn in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            passed = test_fn()
            results.append((test_name, passed))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\nThe BC evaluation should now work correctly!")
        print("Next steps:")
        print("1. Restart the Jupyter notebook kernel")
        print("2. Re-run all cells from the beginning")
        print("3. BC models should evaluate successfully")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the errors above")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())