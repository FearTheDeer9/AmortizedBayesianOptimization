#!/usr/bin/env python3
"""
Debug test script to identify the root cause of parent set prediction issues.
"""

import sys
from pathlib import Path
import logging

# Add src to path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Third-party imports
import jax
import jax.numpy as jnp
import jax.random as random

# Local imports
from causal_bayes_opt.experiments.test_scms import create_simple_test_scm
from causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from causal_bayes_opt.avici_integration.conversion import create_training_batch
from causal_bayes_opt.data_structures.scm import get_variables, get_edges, get_parents

# Import debug functions (you'll need to add these to your inference.py)
from causal_bayes_opt.avici_integration.parent_set import create_parent_set_model
# from causal_bayes_opt.avici_integration.parent_set.inference import (
#     predict_parent_sets_debug,
#     compute_loss_debug, 
#     debug_target_conditioning,
#     verify_parent_set_enumeration
# )

def debug_everything():
    """
    Comprehensive debug of the parent set prediction system.
    """
    print("🔍 COMPREHENSIVE PARENT SET DEBUG")
    print("=" * 50)
    
    # Step 1: Test parent set enumeration directly
    print("\n" + "="*30)
    print("STEP 1: PARENT SET ENUMERATION")
    print("="*30)
    
    from causal_bayes_opt.avici_integration.parent_set.enumeration import enumerate_possible_parent_sets
    
    variables = ['X', 'Y', 'Z']
    
    for target in variables:
        parent_sets = enumerate_possible_parent_sets(variables, target, max_parent_size=3)
        print(f"\nTarget {target}:")
        print(f"  Potential parents: {[v for v in variables if v != target]}")
        print(f"  Enumerated parent sets:")
        for i, ps in enumerate(parent_sets):
            ps_str = set(ps) if ps else "{}"
            print(f"    {i}: {ps_str}")
        
        # Check if true parent sets are included
        if target == 'Y':
            true_parents = frozenset(['X', 'Z'])
            in_list = true_parents in parent_sets
            print(f"  ✅ True parent set {set(true_parents)} in list: {in_list}")
        else:
            true_parents = frozenset()
            in_list = true_parents in parent_sets
            print(f"  ✅ True parent set {{}} in list: {in_list}")
    
    # Step 2: Test model initialization and data flow
    print("\n" + "="*30)
    print("STEP 2: MODEL INITIALIZATION")  
    print("="*30)
    
    # Create SCM and data
    scm = create_simple_test_scm(noise_scale=1.0, target="Y")
    variables = sorted(get_variables(scm))
    samples = sample_from_linear_scm(scm, n_samples=10, seed=42)
    
    print(f"SCM variables: {variables}")
    print(f"SCM edges: {sorted(get_edges(scm))}")
    
    # Test data conversion for each target
    for target in ['Y', 'X', 'Z']:
        batch = create_training_batch(scm, samples, target_variable=target)
        x = batch['x']
        print(f"\nTarget {target}:")
        print(f"  Data shape: {x.shape}")
        print(f"  Target channel (should be 1 for {target} only):")
        for i, var in enumerate(variables):
            target_sum = x[:, i, 2].sum()  # Channel 2 is target channel
            print(f"    {var}: {target_sum} (should be {len(samples) if var == target else 0})")
    
    # Step 3: Test model forward pass
    print("\n" + "="*30)
    print("STEP 3: MODEL FORWARD PASS")
    print("="*30)
    
    config = {
        'layers': 2,
        'dim': 32, 
        'key_size': 8,
        'num_heads': 2,
        'dropout': 0.0,
    }
    
    net = create_parent_set_model(
        model_kwargs=config,
        max_parent_size=3
    )
    
    # Initialize with first target
    target_variable = "Y"
    batch = create_training_batch(scm, samples, target_variable)
    x = batch['x']
    params = net.init(random.PRNGKey(42), x, variables, target_variable, True)
    
    print(f"Model initialized with {sum(p.size for p in jax.tree.leaves(params)):,} parameters")
    
    # Test forward pass for each target
    for target in ['Y', 'X', 'Z']:
        print(f"\nTarget {target}:")
        batch = create_training_batch(scm, samples, target_variable=target)
        x = batch['x']
        
        output = net.apply(params, random.PRNGKey(0), x, variables, target, False)
        
        parent_sets = output['parent_sets']
        logits = output['parent_set_logits']
        probs = jax.nn.softmax(logits)
        
        print(f"  Parent sets: {[set(ps) if ps else '{}' for ps in parent_sets]}")
        print(f"  Raw logits: {[f'{l:.3f}' for l in logits]}")
        print(f"  Probabilities: {[f'{p:.3f}' for p in probs]}")
        
        # Check if same parent sets for all targets (bad!)
        if target == 'Y':
            first_target_sets = parent_sets
        else:
            sets_same = all(ps1 == ps2 for ps1, ps2 in zip(parent_sets, first_target_sets))
            if sets_same:
                print(f"  ⚠️ SAME parent sets as Y - target conditioning not working!")
            else:
                print(f"  ✅ Different parent sets from Y - target conditioning working")
    
    # Step 4: Test loss computation
    print("\n" + "="*30)
    print("STEP 4: LOSS COMPUTATION")
    print("="*30)
    
    from causal_bayes_opt.avici_integration.parent_set.inference import compute_loss
    
    true_parent_sets = {
        'Y': frozenset(['X', 'Z']),
        'X': frozenset(),
        'Z': frozenset()
    }
    
    for target in ['Y', 'X', 'Z']:
        batch = create_training_batch(scm, samples, target_variable=target)
        x = batch['x']
        true_ps = true_parent_sets[target]
        
        loss = compute_loss(net, params, x, variables, target, true_ps, is_training=False)
        
        print(f"\nTarget {target}:")
        print(f"  True parent set: {set(true_ps)}")
        print(f"  Loss: {loss:.4f}")
        
        # Manual check: get probabilities and compute expected loss
        output = net.apply(params, random.PRNGKey(0), x, variables, target, False)
        parent_sets = output['parent_sets']
        probs = jax.nn.softmax(output['parent_set_logits'])
        
        if true_ps in parent_sets:
            true_idx = list(parent_sets).index(true_ps)
            true_prob = probs[true_idx]
            expected_loss = -jnp.log(true_prob)
            print(f"  True parent set at index {true_idx} with prob {true_prob:.6f}")
            print(f"  Expected loss: {expected_loss:.4f}")
            print(f"  Loss matches: {abs(loss - expected_loss) < 1e-4}")
        else:
            print(f"  ⚠️ True parent set NOT in top-k predictions!")


if __name__ == "__main__":
    debug_everything()