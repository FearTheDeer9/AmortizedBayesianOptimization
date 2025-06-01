"""
Debugging utilities for AVICI integration.
"""

import jax
import jax.numpy as jnp
import jax.random as random
from typing import List, FrozenSet


def debug_parent_set_enumeration(variables: List[str], target_variable: str, max_parent_size: int = 3):
    """Debug parent set enumeration to verify empty sets are included."""
    from causal_bayes_opt.avici_integration.parent_set.enumeration import enumerate_possible_parent_sets
    
    parent_sets = enumerate_possible_parent_sets(variables, target_variable, max_parent_size)
    
    print(f"\n🔍 Parent Set Enumeration for target '{target_variable}'")
    print(f"Variables: {variables}")
    print(f"Total parent sets: {len(parent_sets)}")
    
    empty_sets = [ps for ps in parent_sets if len(ps) == 0]
    print(f"Empty sets found: {len(empty_sets)}")
    
    print("All parent sets:")
    for i, ps in enumerate(parent_sets):
        ps_str = set(ps) if ps else "{}"
        print(f"  {i}: {ps_str}")
    
    return parent_sets


def debug_training_step(net, params, x, variable_order, target_variable, true_parent_set):
    """Debug a single training step."""
    print(f"\n🔍 Training Step Debug - Target: {target_variable}")
    print(f"True parent set: {set(true_parent_set) if true_parent_set else '{}'}")
    
    # Check parent set enumeration
    debug_parent_set_enumeration(variable_order, target_variable)
    
    # Get model output
    output = net.apply(params, random.PRNGKey(0), x, variable_order, target_variable, False)
    logits = output['parent_set_logits']
    parent_sets = output['parent_sets']
    
    print(f"\nModel predictions:")
    for i, (ps, logit) in enumerate(zip(parent_sets, logits)):
        ps_str = set(ps) if ps else "{}"
        is_true = "✅" if ps == true_parent_set else "  "
        print(f"{is_true} {i}: {ps_str} -> logit: {logit:.4f}")
    
    # Check if true parent set is included
    true_idx = None
    for i, ps in enumerate(parent_sets):
        if ps == true_parent_set:
            true_idx = i
            break
    
    if true_idx is None:
        print(f"⚠️ TRUE PARENT SET NOT IN PREDICTIONS!")
    else:
        print(f"✅ True parent set found at position {true_idx}")
    
    return output


def debug_logits_and_probabilities(logits, parent_sets, target_variable):
    """Debug logit ranges and probability distribution."""
    print(f"\n🔍 Logits Analysis for {target_variable}")
    print(f"Logit range: [{jnp.min(logits):.3f}, {jnp.max(logits):.3f}]")
    print(f"Logit std: {jnp.std(logits):.3f}")
    
    if jnp.max(logits) - jnp.min(logits) > 10:
        print("⚠️ WARNING: Large logit range may cause saturation!")
    
    probabilities = jax.nn.softmax(logits)
    print(f"Probability range: [{jnp.min(probabilities):.6f}, {jnp.max(probabilities):.6f}]")
    
    # Check for empty set specifically
    for i, ps in enumerate(parent_sets):
        if len(ps) == 0:
            print(f"Empty set: logit={logits[i]:.3f}, prob={probabilities[i]:.6f}")
            break
    
    return probabilities
