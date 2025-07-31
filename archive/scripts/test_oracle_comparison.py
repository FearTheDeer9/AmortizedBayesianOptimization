#!/usr/bin/env python3
"""
Test script to compare old oracle vs new optimal oracle.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
from src.causal_bayes_opt.evaluation.model_interfaces import (
    create_optimal_oracle_acquisition
)
from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm
from src.causal_bayes_opt.data_structures.scm import get_edges, get_mechanisms, get_target
from src.causal_bayes_opt.mechanisms.serializable_mechanisms import LinearMechanism

# Create a simple fork SCM for testing
scm = create_fork_scm(noise_scale=0.1)  # Low noise for clearer comparison

# Extract edges for old oracle
edges = get_edges(scm)
scm_edges = {}
for parent, child in edges:
    if child not in scm_edges:
        scm_edges[child] = []
    scm_edges[child].append(parent)

# Print SCM structure
print("SCM Structure:")
print(f"Edges: {edges}")
print(f"Target: {get_target(scm)}")

# Print coefficients
mechanisms = get_mechanisms(scm)
print("\nCoefficients:")
for var, mech in mechanisms.items():
    if isinstance(mech, LinearMechanism):
        print(f"{var}: {mech.coefficients}, intercept={mech.intercept}")

# Note: Old oracle has been removed - only testing new optimal oracle

# Test 1: Same range for all variables
new_oracle_same_range = create_optimal_oracle_acquisition(
    scm,
    optimization_direction='MINIMIZE',
    intervention_range=(-2.0, 2.0),
    seed=42
)

# Test 2: Different ranges for different variables
new_oracle_diff_range = create_optimal_oracle_acquisition(
    scm,
    optimization_direction='MINIMIZE',
    intervention_range={'X': (-2.0, 2.0), 'Y': (-2.0, 2.0), 'Z': (-5.0, 5.0)},
    seed=42
)

# Test with some dummy data
# Create a tensor with some observations
n_vars = 3  # Fork has X, Y, Z
tensor = np.zeros((5, n_vars, 3))  # 5 timesteps, 3 variables, 3 channels

# Add some observational data (channel 0)
# Let's say current values are X=1.0, Y=0.5, Z=-0.5
tensor[-1, :, 0] = [1.0, 0.5, -0.5]

variables = ['X', 'Y', 'Z']
target = get_target(scm)  # Use the actual target from SCM

print("\n" + "="*50)
print("Testing with current values: X=1.0, Y=0.5, Z=-0.5")
print(f"Target: {target} (MINIMIZE)")
print("="*50)

# Test new oracle with same range
print("\nNEW OPTIMAL ORACLE (same range [-2, 2] for all):")
intervention = new_oracle_same_range(tensor, None, target, variables)
selected_var = list(intervention['targets'])[0]
intervention_val = list(intervention['values'].values())[0]
print(f"Intervene on {selected_var} = {intervention_val:.2f}")

target_mech = mechanisms[target]
if isinstance(target_mech, LinearMechanism):
    print(f"  Coefficients: {target_mech.coefficients}")
    if selected_var in target_mech.coefficients:
        coeff = target_mech.coefficients[selected_var]
        effect = coeff * intervention_val
        print(f"  Direct effect: {coeff:.2f} * {intervention_val:.2f} = {effect:.2f}")
        print(f"  Max possible |effect|: |{coeff:.2f} * 2.0| = {abs(coeff * 2.0):.2f}")

# Test new oracle with different ranges
print("\nNEW OPTIMAL ORACLE (different ranges: X,Y:[-2,2], Z:[-5,5]):")
intervention = new_oracle_diff_range(tensor, None, target, variables)
selected_var = list(intervention['targets'])[0]
intervention_val = list(intervention['values'].values())[0]
print(f"Intervene on {selected_var} = {intervention_val:.2f}")

if isinstance(target_mech, LinearMechanism):
    print(f"  Coefficients: {target_mech.coefficients}")
    if selected_var in target_mech.coefficients:
        coeff = target_mech.coefficients[selected_var]
        effect = coeff * intervention_val
        print(f"  Direct effect: {coeff:.2f} * {intervention_val:.2f} = {effect:.2f}")
        
        # Show max effects for all parents
        print("\n  Max possible effects:")
        for parent in ['X', 'Z']:  # Y's parents
            if parent in target_mech.coefficients:
                parent_coeff = target_mech.coefficients[parent]
                if parent == 'Z':
                    max_val = 5.0
                else:
                    max_val = 2.0
                max_effect = abs(parent_coeff * max_val)
                print(f"    {parent}: |{parent_coeff:.2f} * {max_val:.1f}| = {max_effect:.2f}")

print("\n" + "="*50)
print("Analysis:")
print("- With same ranges: Oracle picks parent with largest |coefficient|")
print("- With different ranges: Oracle picks parent with largest |coefficient * range_extremum|")
print("- Z has smaller |coefficient| but larger range, so it might be selected")