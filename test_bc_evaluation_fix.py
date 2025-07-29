#!/usr/bin/env python3
"""
Test BC evaluation fix

Tests that BC evaluation now works correctly with the fixed acquisition interface.
"""

import sys
from pathlib import Path
import jax
import pyrsistent as pyr

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.evaluation.bc_evaluator import BCEvaluator
from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory


def test_bc_evaluation():
    """Test BC evaluation with a single SCM."""
    print("🧪 Testing BC evaluation fix...")
    
    # Create a simple test SCM
    factory = VariableSCMFactory(
        noise_scale=0.5,
        coefficient_range=(-2.0, 2.0),
        seed=42
    )
    
    test_scm = factory.create_variable_scm(
        num_variables=3,
        structure_type='fork',
        target_variable=None  # Last variable as target
    )
    
    # Set optimization direction
    current_metadata = test_scm.get('metadata', pyr.pmap())
    updated_metadata = current_metadata.set('optimization_direction', 'MINIMIZE')
    test_scm = test_scm.set('metadata', updated_metadata)
    
    # Find BC checkpoints
    checkpoint_dir = project_root / "checkpoints" / "behavioral_cloning" / "dev"
    surrogate_checkpoint = checkpoint_dir / "surrogate" / "surrogate_bc_development_epoch_22_level_3_1753451919.pkl"
    acquisition_checkpoint = checkpoint_dir / "acquisition" / "bc_demo_acquisition_epoch_6_level_3_1753452791.pkl"
    
    if not surrogate_checkpoint.exists():
        print(f"❌ Surrogate checkpoint not found: {surrogate_checkpoint}")
        return False
        
    if not acquisition_checkpoint.exists():
        print(f"❌ Acquisition checkpoint not found: {acquisition_checkpoint}")
        return False
    
    print(f"✅ Found checkpoints:")
    print(f"   Surrogate: {surrogate_checkpoint.name}")
    print(f"   Acquisition: {acquisition_checkpoint.name}")
    
    # Test BC with acquisition only
    print("\n📊 Testing BC_Acquisition_Learning...")
    try:
        evaluator = BCEvaluator(
            acquisition_checkpoint=acquisition_checkpoint,
            name="BC_Acquisition_Test"
        )
        
        config = {
            'experiment': {
                'target': {
                    'n_observational_samples': 100,
                    'max_interventions': 5,  # Small number for quick test
                    'intervention_value_range': (-2.0, 2.0),
                    'optimization_direction': 'MINIMIZE'
                }
            }
        }
        
        result = evaluator.evaluate_single_run(
            scm=test_scm,
            config=config,
            seed=42
        )
        
        if result.success:
            print(f"✅ BC evaluation succeeded!")
            print(f"   Initial value: {result.final_metrics.get('initial_value', 'N/A')}")
            print(f"   Final value: {result.final_metrics.get('final_value', 'N/A')}")
            print(f"   Improvement: {result.final_metrics.get('improvement', 'N/A')}")
            print(f"   Steps: {len(result.learning_history)}")
            return True
        else:
            print(f"❌ BC evaluation failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bc_both():
    """Test BC with both surrogate and acquisition."""
    print("\n📊 Testing BC_Both (surrogate + acquisition)...")
    
    # Create test SCM
    factory = VariableSCMFactory(
        noise_scale=0.5,
        coefficient_range=(-2.0, 2.0),
        seed=43
    )
    
    test_scm = factory.create_variable_scm(
        num_variables=3,
        structure_type='chain',
        target_variable=None
    )
    
    # Set optimization direction
    current_metadata = test_scm.get('metadata', pyr.pmap())
    updated_metadata = current_metadata.set('optimization_direction', 'MINIMIZE')
    test_scm = test_scm.set('metadata', updated_metadata)
    
    checkpoint_dir = project_root / "checkpoints" / "behavioral_cloning" / "dev"
    surrogate_checkpoint = checkpoint_dir / "surrogate" / "surrogate_bc_development_epoch_22_level_3_1753451919.pkl"
    acquisition_checkpoint = checkpoint_dir / "acquisition" / "bc_demo_acquisition_epoch_6_level_3_1753452791.pkl"
    
    try:
        evaluator = BCEvaluator(
            surrogate_checkpoint=surrogate_checkpoint,
            acquisition_checkpoint=acquisition_checkpoint,
            name="BC_Both_Test"
        )
        
        config = {
            'experiment': {
                'target': {
                    'n_observational_samples': 100,
                    'max_interventions': 5,
                    'intervention_value_range': (-2.0, 2.0),
                    'optimization_direction': 'MINIMIZE'
                }
            }
        }
        
        result = evaluator.evaluate_single_run(
            scm=test_scm,
            config=config,
            seed=42
        )
        
        if result.success:
            print(f"✅ BC_Both evaluation succeeded!")
            print(f"   Initial value: {result.final_metrics.get('initial_value', 'N/A')}")
            print(f"   Final value: {result.final_metrics.get('final_value', 'N/A')}")
            print(f"   Improvement: {result.final_metrics.get('improvement', 'N/A')}")
            return True
        else:
            print(f"❌ BC_Both evaluation failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during BC_Both evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Testing BC evaluation fixes...\n")
    
    # Test acquisition only
    test1_passed = test_bc_evaluation()
    
    # Test both surrogate and acquisition
    test2_passed = test_bc_both()
    
    print("\n📊 Test Summary:")
    print(f"   BC_Acquisition_Learning: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   BC_Both: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n✅ All tests passed! BC evaluation is working correctly.")
        print("   You can now re-run the full evaluation notebook.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")