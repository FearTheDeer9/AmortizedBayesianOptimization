#!/usr/bin/env python3
"""
PARENT_SCALE Integration Demo

This example demonstrates how to use the integrated PARENT_SCALE algorithm
through the causal_bayes_opt framework.

Usage:
    python examples/parent_scale_demo.py

This example shows:
- Creating a simple SCM structure
- Running the integrated PARENT_SCALE algorithm
- Interpreting the optimization results
"""

import sys
import os
import warnings
import numpy as np
from typing import Dict, Any

warnings.filterwarnings('ignore')

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

# Import ACBO components
from causal_bayes_opt.integration.parent_scale_bridge import run_full_parent_scale_algorithm
from causal_bayes_opt.data_structures.scm import create_scm
from causal_bayes_opt.mechanisms.linear import create_linear_mechanism, create_root_mechanism


def create_demo_scm():
    """
    Create a simple demonstration SCM: X -> Z <- Y
    This is a collider structure where Z depends on both X and Y.
    """
    variables = frozenset(['X', 'Y', 'Z'])
    edges = frozenset([('X', 'Z'), ('Y', 'Z')])
    
    # Simple linear mechanisms
    mechanisms = {
        'X': create_root_mechanism(mean=0.0, noise_scale=0.5),
        'Y': create_root_mechanism(mean=0.0, noise_scale=0.5),
        'Z': create_linear_mechanism(['X', 'Y'], {'X': 1.0, 'Y': -0.5}, intercept=0.0, noise_scale=0.2),
    }
    
    return create_scm(
        variables=variables,
        edges=edges,
        mechanisms=mechanisms,
        target='Z'  # We want to optimize Z
    )


def run_demo(iterations: int = 5, seed: int = 42) -> Dict[str, Any]:
    """Run the PARENT_SCALE algorithm demonstration."""
    print("🚀 PARENT_SCALE Integration Demo")
    print("=" * 50)
    
    # Create the SCM
    scm = create_demo_scm()
    print(f"Created SCM:")
    print(f"  Variables: {sorted(scm['variables'])}")
    print(f"  Edges: {sorted(scm['edges'])}")
    print(f"  Target: {scm['target']}")
    
    # Run the algorithm
    print(f"\nRunning PARENT_SCALE algorithm:")
    print(f"  Iterations: {iterations}")
    print(f"  Observational samples: 50")
    print(f"  Interventional samples: 10")
    print(f"  Random seed: {seed}\n")
    
    result = run_full_parent_scale_algorithm(
        scm=scm,
        target_variable='Z',
        T=iterations,
        nonlinear=False,
        causal_prior=False,
        individual=False,
        use_doubly_robust=False,
        n_observational=50,
        n_interventional=10,
        seed=seed
    )
    
    return result


def display_results(result: Dict[str, Any]):
    """Display the optimization results in a user-friendly format."""
    if result.get('status') != 'completed':
        print(f"❌ Algorithm failed: {result.get('error', 'Unknown error')}")
        return
    
    print("✅ Algorithm completed successfully!\n")
    
    print("📊 Optimization Results:")
    print(f"  Final optimum value: {result['final_optimum']:.4f}")
    print(f"  Total interventions: {result['total_interventions']}")
    if 'runtime' in result:
        print(f"  Runtime: {result['runtime']:.2f} seconds")
    
    print("\n🎯 Intervention Sequence:")
    for i, (var_set, values, outcome) in enumerate(zip(
        result['intervention_sequence'],
        result['intervention_values'], 
        result['target_outcomes']
    )):
        print(f"  {i+1}. Intervened on {var_set} = {values[0]:.4f} → Z = {outcome:.4f}")
    
    print("\n📈 Optimization Trajectory:")
    trajectory = result['global_optimum_trajectory']
    for i, value in enumerate(trajectory):
        symbol = "📍" if i == len(trajectory) - 1 else "•"
        print(f"  {symbol} Step {i}: {value:.4f}")
    
    # Show improvement
    if len(trajectory) > 1:
        improvement = trajectory[-1] - trajectory[0]
        print(f"\n✨ Total improvement: {improvement:.4f}")
        if improvement < 0:
            print(f"   (Lower is better for minimization)")


def main():
    """Run the demonstration."""
    # Run the demo
    result = run_demo(iterations=5, seed=42)
    
    # Display results
    print("\n" + "=" * 50)
    display_results(result)
    
    print("\n" + "=" * 50)
    print("💡 This demo shows how to use the integrated PARENT_SCALE algorithm")
    print("   for causal Bayesian optimization within the ACBO framework.")
    print("\n🔍 Try modifying:")
    print("   - The SCM structure (add more variables/edges)")
    print("   - The number of iterations")
    print("   - The sample sizes")
    print("   - The target variable")


if __name__ == "__main__":
    main()