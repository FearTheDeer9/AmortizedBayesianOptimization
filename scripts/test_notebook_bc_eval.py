#!/usr/bin/env python3
"""
Test script to verify BC evaluation works in notebook context.
This simulates what happens in the notebook when running the unified evaluation.
"""

import sys
from pathlib import Path

# Add project root to path (as notebook does)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add scripts to path (as notebook might)
sys.path.insert(0, str(project_root / "scripts" / "core"))

import pyrsistent as pyr
from src.causal_bayes_opt.evaluation import setup_evaluation_runner
from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory


def test_notebook_bc_evaluation():
    """Test BC evaluation as it would run in the notebook."""
    print("Testing BC evaluation in notebook context...")
    
    # Create evaluation runner with BC methods (similar to notebook cell 20)
    checkpoint_paths = {
        'bc_surrogate': Path('dummy_surrogate.pkl'),  # Dummy path for testing
        'bc_acquisition': Path('dummy_acquisition.pkl')
    }
    
    output_dir = Path('test_eval_output')
    
    try:
        # Setup runner
        runner = setup_evaluation_runner(
            methods=['random', 'bc_surrogate', 'bc_acquisition', 'bc_both'],
            checkpoint_paths=checkpoint_paths,
            parallel=False,
            output_dir=output_dir
        )
        print("✅ Successfully created evaluation runner with BC methods")
        
        # Create test SCMs (similar to notebook cell 21)
        scm_factory = VariableSCMFactory(seed=42)
        test_scms = []
        
        # Just one SCM for quick test
        scm = scm_factory.create_variable_scm(
            num_variables=3,
            structure_type='chain',
            target_variable=None
        )
        # Set optimization direction
        current_metadata = scm.get('metadata', pyr.pmap())
        updated_metadata = current_metadata.set('optimization_direction', 'MINIMIZE')
        scm = scm.set('metadata', updated_metadata)
        test_scms.append(scm)
        
        print("✅ Created test SCM")
        
        # Create evaluation config
        eval_config = {
            'experiment': {
                'target': {
                    'n_observational_samples': 50,
                    'max_interventions': 3,
                    'intervention_value_range': (-2.0, 2.0),
                    'optimization_direction': 'MINIMIZE'
                }
            }
        }
        
        print("✅ Created evaluation config")
        
        # Run comparison (this would fail if BC evaluator has issues)
        print("\nRunning evaluation comparison...")
        results = runner.run_comparison(
            test_scms=test_scms,
            config=eval_config,
            n_runs_per_scm=1,
            base_seed=42
        )
        
        print("✅ Evaluation completed successfully!")
        
        # Check results
        print(f"\nMethods evaluated: {list(results.method_metrics.keys())}")
        for method_name, metrics in results.method_metrics.items():
            print(f"  {method_name}: {metrics.n_successful}/{metrics.n_runs} successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Notebook BC evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup test output
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)


def main():
    """Run notebook integration test."""
    print("=" * 80)
    print("NOTEBOOK BC EVALUATION TEST")
    print("=" * 80)
    print()
    
    success = test_notebook_bc_evaluation()
    
    print("\n" + "=" * 80)
    if success:
        print("✅ NOTEBOOK INTEGRATION TEST PASSED")
        print("The BC evaluation framework should work correctly in the notebook!")
    else:
        print("❌ NOTEBOOK INTEGRATION TEST FAILED")
        print("There may still be issues when running in the notebook")
    print("=" * 80)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)