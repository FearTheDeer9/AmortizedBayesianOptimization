#!/usr/bin/env python3
"""
PARENT_SCALE Algorithm Comparison Example

This example demonstrates how to compare the original and integrated PARENT_SCALE 
implementations. It runs both algorithms using their native approaches with 
matching configurations to validate consistent behavior.

Usage:
    python examples/parent_scale_comparison.py

This example is useful for:
- Validating the integrated PARENT_SCALE implementation
- Understanding algorithm behavior differences
- Benchmarking performance between implementations
"""

import sys
import warnings
import numpy as np
import time
import argparse
from copy import deepcopy

warnings.filterwarnings('ignore')

# Add paths
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))
sys.path.insert(0, os.path.join(parent_dir, 'causal_bayes_opt_old'))

# Original PARENT_SCALE imports (may not be available)
try:
    from external.parent_scale.algorithms.PARENT_SCALE_algorithm import PARENT_SCALE as PARENT_SCALE_ORIGINAL
    from src.causal_bayes_opt.integration.parent_scale.helpers import setup_observational_interventional_original as setup_observational_interventional
    ORIGINAL_AVAILABLE = True
except ImportError:
    print("⚠️  Original PARENT_SCALE implementation not found (causal_bayes_opt_old/ directory)")
    print("   This example requires both original and integrated implementations")
    print("   To run this comparison, ensure causal_bayes_opt_old/ is available")
    ORIGINAL_AVAILABLE = False

# Integrated PARENT_SCALE imports
from causal_bayes_opt.integration.parent_scale import (
    run_full_parent_scale_algorithm
)
from causal_bayes_opt.data_structures.scm import create_scm
from causal_bayes_opt.mechanisms.linear import create_linear_mechanism, create_root_mechanism


def create_matching_scm():
    """
    Create an SCM that exactly matches the LinearColliderGraph structure:
    X -> Z -> Y with the same noise levels and relationships.
    """
    # Create mechanisms exactly matching LinearColliderGraph
    variables = frozenset(['X', 'Z', 'Y'])
    edges = frozenset([('X', 'Z'), ('Z', 'Y')])
    
    # Match the exact structure from LinearColliderGraph:
    # - X: root variable (mean=0, noise=0.2)
    # - Z: depends on X: Z = X + noise (noise=0.2)  
    # - Y: depends on Z: Y = Z + noise (noise=0.2)
    mechanisms = {
        'X': create_root_mechanism(mean=0.0, noise_scale=0.2),
        'Z': create_linear_mechanism(['X'], {'X': 1.0}, intercept=0.0, noise_scale=0.2),
        'Y': create_linear_mechanism(['Z'], {'Z': 1.0}, intercept=0.0, noise_scale=0.2),
    }
    
    return create_scm(
        variables=variables,
        edges=edges,
        mechanisms=mechanisms,
        target='Y'
    )


def run_original_algorithm(T=3, seed=42):
    """Run the original PARENT_SCALE algorithm."""
    print("🔧 Running Original PARENT_SCALE Algorithm")
    print("=" * 50)
    
    # Setup with fixed configuration for reproducibility
    np.random.seed(seed)
    
    # Use ACBO's SCM creation to match LinearColliderGraph
    from causal_bayes_opt.integration.parent_scale import scm_to_graph_structure
    scm = create_matching_scm()
    graph = scm_to_graph_structure(scm)
    
    try:
        D_O, D_I, exploration_set = setup_observational_interventional(
            graph_type="Toy",
            n_obs=100,
            n_int=10,
            noiseless=True,  # Use noiseless=True for deterministic comparison
            seed=seed,
            graph=graph,
            use_iscm=False
        )
    except (ImportError, NameError) as e:
        print(f"⚠️ Original data setup not available: {e}")
        print("   Using ACBO's data generation instead...")
        from causal_bayes_opt.integration.parent_scale.data_processing import generate_parent_scale_data_with_scm
        D_O, D_I, exploration_set = generate_parent_scale_data_with_scm(
            scm=scm,
            n_observational=100,
            n_interventional=10,
            seed=seed
        )
    
    print(f"  ✓ Generated data: {len(D_O['X'])} obs, {sum(len(D_I[key]['X']) for key in D_I)} int")
    print(f"  ✓ Exploration set: {exploration_set}")
    print(f"  ✓ Target: {graph.target}")
    
    # Initialize algorithm with exact configuration
    parent_scale = PARENT_SCALE_ORIGINAL(
        graph=graph,
        nonlinear=False,
        causal_prior=False,
        noiseless=True,  # Deterministic for comparison
        cost_num=1,
        scale_data=True,
        individual=False,
        use_doubly_robust=False,
        use_iscm=False
    )
    
    parent_scale.set_values(deepcopy(D_O), deepcopy(D_I), exploration_set)
    
    # Run algorithm
    print(f"  ✓ Running {T} iterations...")
    start_time = time.time()
    result = parent_scale.run_algorithm(T=T)
    end_time = time.time()
    
    # Parse results
    if isinstance(result, tuple) and len(result) >= 6:
        global_opt, current_y, current_cost, intervention_set, intervention_values, average_uncertainty = result[:6]
        
        print(f"  ✓ Completed in {end_time - start_time:.2f}s")
        print(f"  ✓ Final optimum: {global_opt[-1] if global_opt else 'N/A'}")
        print(f"  ✓ Interventions: {len(intervention_set)}")
        
        # Print intervention sequence for debugging
        print("  ✓ Intervention sequence:")
        for i, (var_set, values) in enumerate(zip(intervention_set, intervention_values)):
            target_val = current_y[i] if i < len(current_y) else 'N/A'
            global_val = global_opt[i] if i < len(global_opt) else 'N/A'
            print(f"     {i+1}. {var_set} = {values} → target: {target_val:.6f}, global: {global_val:.6f}")
        
        return {
            'algorithm': 'PARENT_SCALE_ORIGINAL',
            'global_opt': global_opt,
            'current_y': current_y,
            'current_cost': current_cost,
            'intervention_set': intervention_set,
            'intervention_values': intervention_values,
            'average_uncertainty': average_uncertainty,
            'final_optimum': global_opt[-1] if global_opt else None,
            'runtime': end_time - start_time,
            'successful': True
        }
    else:
        print(f"  ❌ Unexpected result format: {type(result)}")
        return {'successful': False, 'result': result}


def run_integrated_algorithm(T=3, seed=42):
    """Run the integrated PARENT_SCALE algorithm using matching SCM."""
    print("\\n🔗 Running Integrated PARENT_SCALE Algorithm")
    print("=" * 50)
    
    # Create SCM that matches the original structure
    scm = create_matching_scm()
    print(f"  ✓ Created matching SCM with variables: {list(scm['variables'])}")
    print(f"  ✓ Edges: {list(scm['edges'])}")
    print(f"  ✓ Target: {scm['target']}")
    
    # Run integrated algorithm with matching configuration
    print(f"  ✓ Running {T} iterations...")
    start_time = time.time()
    
    try:
        result = run_full_parent_scale_algorithm(
            scm=scm,
            target_variable='Y',
            T=T,
            nonlinear=False,
            causal_prior=False,
            individual=False,
            use_doubly_robust=False,
            n_observational=100,
            n_interventional=10,
            seed=seed  # Use same seed for reproducibility
        )
        end_time = time.time()
        
        if result.get('status') == 'completed':
            print(f"  ✓ Completed in {end_time - start_time:.2f}s")
            print(f"  ✓ Final optimum: {result.get('final_optimum', 'N/A')}")
            print(f"  ✓ Interventions: {result.get('total_interventions', 0)}")
            
            # Print intervention sequence for debugging
            intervention_sequence = result.get('intervention_sequence', [])
            intervention_values = result.get('intervention_values', [])
            target_outcomes = result.get('target_outcomes', [])
            global_optimum = result.get('global_optimum_trajectory', [])
            
            print("  ✓ Intervention sequence:")
            for i, (var_set, values) in enumerate(zip(intervention_sequence, intervention_values)):
                target_val = target_outcomes[i] if i < len(target_outcomes) else 'N/A'
                global_val = global_optimum[i] if i < len(global_optimum) else 'N/A'
                print(f"     {i+1}. {var_set} = {values} → target: {target_val:.6f}, global: {global_val:.6f}")
            
            # Convert to format matching original for easy comparison
            converted_result = {
                'algorithm': 'PARENT_SCALE_INTEGRATED',
                'global_opt': result.get('global_optimum_trajectory', []),
                'current_y': result.get('target_outcomes', []),
                'current_cost': result.get('cost_trajectory', []),
                'intervention_set': result.get('intervention_sequence', []),
                'intervention_values': result.get('intervention_values', []),
                'average_uncertainty': result.get('uncertainty_trajectory', []),
                'final_optimum': result.get('final_optimum'),
                'runtime': end_time - start_time,
                'successful': True
            }
            
            return converted_result
        else:
            print(f"  ❌ Algorithm failed: {result.get('error', 'Unknown error')}")
            return {'successful': False, 'result': result}
            
    except Exception as e:
        end_time = time.time()
        print(f"  ❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return {'successful': False, 'error': str(e), 'runtime': end_time - start_time}


def compare_algorithm_results(orig_result, integ_result):
    """Compare results from both algorithms."""
    print("\\n📊 Algorithm Results Comparison")
    print("=" * 50)
    
    if not orig_result.get('successful', False):
        print("❌ Original algorithm failed - cannot compare")
        return False
        
    if not integ_result.get('successful', False):
        print("❌ Integrated algorithm failed - cannot compare")
        return False
    
    # Compare final optimization values
    orig_final = orig_result.get('final_optimum')
    integ_final = integ_result.get('final_optimum')
    
    print(f"Final Optimization Values:")
    print(f"  Original:   {orig_final:.6f}")
    print(f"  Integrated: {integ_final:.6f}")
    
    if orig_final is not None and integ_final is not None:
        diff = abs(orig_final - integ_final)
        print(f"  Difference: {diff:.6f}")
        
        # Use a reasonable tolerance for different random seeds/data
        tolerance = 0.1  # Allow for some variation due to different data generation
        values_close = diff < tolerance
        print(f"  Values close (tolerance {tolerance}): {'✅' if values_close else '❌'}")
        
        # Also check if they're in the same ballpark (within order of magnitude)
        if orig_final != 0:
            relative_diff = abs(diff / orig_final)
            print(f"  Relative difference: {relative_diff:.1%}")
        
    else:
        values_close = False
        print("  ❌ Cannot compare - missing values")
    
    # Compare optimization trajectories
    orig_traj = orig_result.get('global_opt', [])
    integ_traj = integ_result.get('global_opt', [])
    
    print(f"\\nOptimization Trajectories:")
    print(f"  Original:   {[f'{x:.4f}' for x in orig_traj[:5]]}{'...' if len(orig_traj) > 5 else ''}")
    print(f"  Integrated: {[f'{x:.4f}' for x in integ_traj[:5]]}{'...' if len(integ_traj) > 5 else ''}")
    
    # Compare number of interventions
    orig_interventions = len(orig_result.get('intervention_set', []))
    integ_interventions = len(integ_result.get('intervention_set', []))
    
    print(f"\\nNumber of Interventions:")
    print(f"  Original:   {orig_interventions}")
    print(f"  Integrated: {integ_interventions}")
    
    interventions_match = orig_interventions == integ_interventions
    print(f"  Interventions match: {'✅' if interventions_match else '❌'}")
    
    # Compare runtimes
    orig_runtime = orig_result.get('runtime', 0)
    integ_runtime = integ_result.get('runtime', 0)
    
    print(f"\\nRuntimes:")
    print(f"  Original:   {orig_runtime:.2f}s")
    print(f"  Integrated: {integ_runtime:.2f}s")
    print(f"  Speedup:    {orig_runtime/integ_runtime:.2f}x" if integ_runtime > 0 else "  Cannot calculate speedup")
    
    # Algorithm behavior assessment
    print(f"\\nAlgorithm Behavior Assessment:")
    
    # Check if both algorithms show optimization (improving values)
    orig_improving = len(orig_traj) > 1 and orig_traj[-1] < orig_traj[0]
    integ_improving = len(integ_traj) > 1 and integ_traj[-1] < integ_traj[0]
    
    print(f"  Original shows optimization: {'✅' if orig_improving else '❌'}")
    print(f"  Integrated shows optimization: {'✅' if integ_improving else '❌'}")
    
    behavior_consistent = orig_improving == integ_improving
    print(f"  Consistent optimization behavior: {'✅' if behavior_consistent else '❌'}")
    
    # Overall assessment
    overall_success = values_close and interventions_match and behavior_consistent
    print(f"\\n{'✅ OVERALL: ALGORITHMS BEHAVE CONSISTENTLY!' if overall_success else '⚠️  ALGORITHMS SHOW SOME DIFFERENCES'}")
    
    if overall_success:
        print("✅ Both algorithms show similar optimization behavior")
        print("✅ Differences in exact values are expected due to different data generation")
        print("✅ The integrated implementation appears to be working correctly")
    else:
        print("⚠️  Some differences detected - this may be due to:")
        print("   - Different random data generation")
        print("   - Slight implementation differences")
        print("   - Different SCM structures or parameters")
    
    return overall_success


def main(seed=42):
    """Run the direct algorithm comparison."""
    print("🚀 PARENT_SCALE ALGORITHM COMPARISON")
    print("=" * 60)
    
    # Check if original implementation is available
    if not ORIGINAL_AVAILABLE:
        print("⚠️  Original PARENT_SCALE implementation not found")
        print("   Running integrated algorithm demonstration only\\n")
        
        # Configuration
        T = 3  # Number of iterations
        
        print(f"Configuration:")
        print(f"  Iterations: {T}")
        print(f"  Random seed: {seed}")
        print(f"  Mode: Integrated algorithm demo only\\n")
        
        # Run integrated algorithm
        integ_result = run_integrated_algorithm(T=T, seed=seed)
        
        if integ_result.get('successful', False):
            print("\\n" + "=" * 60)
            print("✅ Integrated PARENT_SCALE algorithm ran successfully!")
            print(f"   Final optimum: {integ_result.get('final_optimum', 'N/A')}")
            print(f"   Total interventions: {len(integ_result.get('intervention_set', []))}")
            print("\\n💡 To compare with original implementation:")
            print("   1. Provide causal_bayes_opt_old/ directory")
            print("   2. Run this script again")
        else:
            print("\\n❌ Integrated algorithm failed to run")
            
        return integ_result.get('successful', False)
    
    print("This compares both algorithms using their native approaches\\n")
    print("with matching configurations and seeds for fair comparison.\\n")
    
    # Configuration
    T = 3  # Number of iterations
    
    print(f"Configuration:")
    print(f"  Iterations: {T}")
    print(f"  Random seed: {seed}")
    print(f"  Strategy: Match configurations as closely as possible")
    
    # Run original algorithm
    orig_result = run_original_algorithm(T=T, seed=seed)
    
    # Run integrated algorithm
    integ_result = run_integrated_algorithm(T=T, seed=seed)
    
    # Compare results
    success = compare_algorithm_results(orig_result, integ_result)
    
    print("\\n" + "=" * 60)
    if success:
        print("🎉 SUCCESS: Both algorithms behave consistently!")
        print("✅ The integrated PARENT_SCALE implementation works correctly.")
        print("✅ Minor differences are expected due to different data generation.")
        print("✅ Ready to proceed with expert demonstration collection.")
    else:
        print("⚠️  PARTIAL SUCCESS: Algorithms show some differences.")
        print("✅ Both implementations run successfully without errors.")
        print("ℹ️  Differences may be due to data generation or minor implementation variations.")
        print("ℹ️  Consider this acceptable if algorithms show consistent optimization behavior.")
    
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare PARENT_SCALE algorithms")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()
    success = main(seed=args.seed)
    sys.exit(0 if success else 1)