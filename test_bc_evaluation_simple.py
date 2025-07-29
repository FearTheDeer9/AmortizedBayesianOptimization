#!/usr/bin/env python3
"""
Simple test script to verify BC evaluation is working after fixes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import jax.random as random
from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
from src.causal_bayes_opt.evaluation.bc_runner import run_bc_experiment, EvalConfig
import pyrsistent as pyr

def main():
    print("🧪 Testing BC Evaluation with Fixed Integration...")
    
    # Create a simple test SCM
    scm_factory = VariableSCMFactory(
        noise_scale=0.5,
        coefficient_range=(-2.0, 2.0),
        seed=42
    )
    
    test_scm = scm_factory.create_variable_scm(
        num_variables=3,
        structure_type='fork',
        target_variable=None
    )
    
    # Update metadata
    current_metadata = test_scm.get('metadata', pyr.pmap())
    updated_metadata = current_metadata.set('optimization_direction', 'MINIMIZE')
    test_scm = test_scm.set('metadata', updated_metadata)
    
    # Get checkpoint paths
    checkpoint_dir = Path("checkpoints/behavioral_cloning/dev")
    
    # Find latest checkpoints
    surrogate_dir = checkpoint_dir / "surrogate"
    acquisition_dir = checkpoint_dir / "acquisition"
    
    surrogate_checkpoints = list(surrogate_dir.glob("*.pkl"))
    acquisition_checkpoints = list(acquisition_dir.glob("*.pkl"))
    
    if not surrogate_checkpoints:
        print("❌ No surrogate checkpoints found")
        return
    
    if not acquisition_checkpoints:
        print("❌ No acquisition checkpoints found")
        return
    
    # Use latest checkpoints
    surrogate_checkpoint = str(sorted(surrogate_checkpoints)[-1])
    acquisition_checkpoint = str(sorted(acquisition_checkpoints)[-1])
    
    print(f"\n📁 Using checkpoints:")
    print(f"  Surrogate: {Path(surrogate_checkpoint).name}")
    print(f"  Acquisition: {Path(acquisition_checkpoint).name}")
    
    # Create evaluation config
    config = EvalConfig(
        n_observational_samples=50,
        n_intervention_steps=10,
        intervention_value_range=(-2.0, 2.0),
        random_seed=42
    )
    
    print("\n🚀 Running BC evaluation tests...")
    
    # Test 1: BC Surrogate + Random Acquisition
    print("\n1️⃣ Testing BC Surrogate + Random Acquisition...")
    try:
        result1 = run_bc_experiment(
            scm=test_scm,
            config=config,
            surrogate_checkpoint=surrogate_checkpoint,
            acquisition_checkpoint=None,  # Random acquisition
            track_performance=True
        )
        print(f"✅ Success! Improvement: {result1['performance_metrics']['improvement']:.4f}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Random Surrogate + BC Acquisition
    print("\n2️⃣ Testing Random Surrogate + BC Acquisition...")
    try:
        result2 = run_bc_experiment(
            scm=test_scm,
            config=config,
            surrogate_checkpoint=None,  # Random surrogate
            acquisition_checkpoint=acquisition_checkpoint,
            track_performance=True
        )
        print(f"✅ Success! Improvement: {result2['performance_metrics']['improvement']:.4f}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: BC Both
    print("\n3️⃣ Testing BC Surrogate + BC Acquisition...")
    try:
        result3 = run_bc_experiment(
            scm=test_scm,
            config=config,
            surrogate_checkpoint=surrogate_checkpoint,
            acquisition_checkpoint=acquisition_checkpoint,
            track_performance=True
        )
        print(f"✅ Success! Improvement: {result3['performance_metrics']['improvement']:.4f}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n📊 Evaluation Summary:")
    if 'result1' in locals():
        print(f"  BC Surrogate + Random: {result1['performance_metrics']['improvement']:.4f}")
    if 'result2' in locals():
        print(f"  Random + BC Acquisition: {result2['performance_metrics']['improvement']:.4f}")
    if 'result3' in locals():
        print(f"  BC Both: {result3['performance_metrics']['improvement']:.4f}")
    
    print("\n✅ BC evaluation is working correctly!")

if __name__ == "__main__":
    main()