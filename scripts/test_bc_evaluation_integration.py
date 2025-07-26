#!/usr/bin/env python3
"""
Test BC Evaluation Integration

This script tests that BC evaluation works correctly with the updated bc_runner.py
that properly uses BC inference functions.
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pyrsistent as pyr
from src.causal_bayes_opt.evaluation import BCEvaluator
from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_bc_evaluation():
    """Test BC evaluation with real checkpoints."""
    
    print("=" * 80)
    print("BC EVALUATION INTEGRATION TEST")
    print("=" * 80)
    
    # Check for checkpoints
    checkpoint_dir = project_root / "checkpoints" / "behavioral_cloning" / "dev"
    
    # Find latest checkpoints
    surrogate_checkpoints = list((checkpoint_dir / "surrogate").glob("*.pkl"))
    acquisition_checkpoints = list((checkpoint_dir / "acquisition").glob("*.pkl"))
    
    if not surrogate_checkpoints or not acquisition_checkpoints:
        print("❌ No checkpoints found. Please run BC training notebook first.")
        return False
        
    # Use latest checkpoints
    surrogate_checkpoint = sorted(surrogate_checkpoints)[-1]
    acquisition_checkpoint = sorted(acquisition_checkpoints)[-1]
    
    print(f"\n📁 Using checkpoints:")
    print(f"  Surrogate: {surrogate_checkpoint.name}")
    print(f"  Acquisition: {acquisition_checkpoint.name}")
    
    # Create test SCM
    print("\n🔬 Creating test SCM...")
    scm_factory = VariableSCMFactory(seed=42)
    test_scm = scm_factory.create_variable_scm(
        num_variables=4,  # Test with 4 variables
        structure_type='fork',
        target_variable=None
    )
    
    # Set optimization direction
    metadata = test_scm.get('metadata', pyr.pmap())
    metadata = metadata.set('optimization_direction', 'MINIMIZE')
    test_scm = test_scm.set('metadata', metadata)
    
    # Test configuration
    test_config = {
        'experiment': {
            'target': {
                'n_observational_samples': 50,
                'max_interventions': 10,
                'intervention_value_range': (-2.0, 2.0)
            }
        }
    }
    
    # Test 1: BC with both models
    print("\n1️⃣ Testing BC with both surrogate and acquisition...")
    try:
        bc_both = BCEvaluator(
            surrogate_checkpoint=surrogate_checkpoint,
            acquisition_checkpoint=acquisition_checkpoint,
            name="BC_Both_Test"
        )
        bc_both.initialize()
        
        result = bc_both.evaluate_single_run(
            scm=test_scm,
            config=test_config,
            seed=42,
            run_idx=0
        )
        
        if result.success:
            print("✅ BC Both evaluation succeeded!")
            print(f"   Initial value: {result.initial_value:.3f}")
            print(f"   Final value: {result.final_value:.3f}")
            print(f"   Improvement: {result.improvement:.3f}")
            print(f"   Steps: {result.n_steps}")
        else:
            print(f"❌ BC Both evaluation failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ BC Both test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: BC with surrogate only
    print("\n2️⃣ Testing BC with surrogate only...")
    try:
        bc_surrogate = BCEvaluator(
            surrogate_checkpoint=surrogate_checkpoint,
            name="BC_Surrogate_Test"
        )
        bc_surrogate.initialize()
        
        result = bc_surrogate.evaluate_single_run(
            scm=test_scm,
            config=test_config,
            seed=43,
            run_idx=0
        )
        
        if result.success:
            print("✅ BC Surrogate evaluation succeeded!")
            print(f"   Improvement: {result.improvement:.3f}")
        else:
            print(f"❌ BC Surrogate evaluation failed: {result.error_message}")
            
    except Exception as e:
        print(f"❌ BC Surrogate test failed: {e}")
        
    # Test 3: BC with acquisition only
    print("\n3️⃣ Testing BC with acquisition only...")
    try:
        bc_acquisition = BCEvaluator(
            acquisition_checkpoint=acquisition_checkpoint,
            name="BC_Acquisition_Test"
        )
        bc_acquisition.initialize()
        
        result = bc_acquisition.evaluate_single_run(
            scm=test_scm,
            config=test_config,
            seed=44,
            run_idx=0
        )
        
        if result.success:
            print("✅ BC Acquisition evaluation succeeded!")
            print(f"   Improvement: {result.improvement:.3f}")
        else:
            print(f"❌ BC Acquisition evaluation failed: {result.error_message}")
            
    except Exception as e:
        print(f"❌ BC Acquisition test failed: {e}")
    
    print("\n" + "=" * 80)
    print("✅ BC EVALUATION INTEGRATION TEST COMPLETE")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = test_bc_evaluation()
    sys.exit(0 if success else 1)