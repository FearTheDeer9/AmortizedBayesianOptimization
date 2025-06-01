#!/usr/bin/env python3
"""
Quick validation test for AVICI integration fixes.

Run this to verify the critical fixes are working:
- Empty set encoding fix
- Lower learning rate  
- Gradient clipping
- Debug utilities

This tests on a simple 2-variable case: X → Y
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

import jax
import jax.numpy as jnp
import jax.random as random
import optax

from causal_bayes_opt.experiments.test_scms import create_simple_test_scm
from causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
# Updated imports for new module structure
from causal_bayes_opt.avici_integration import (
    create_training_batch,
    create_parent_set_model,
    predict_parent_sets,
    compute_loss,
    create_train_step
)
from causal_bayes_opt.avici_integration.testing.debug_tools import (
    debug_parent_set_enumeration,
    debug_training_step,
    debug_logits_and_probabilities
)


def create_simple_config():
    """Simple config for quick testing."""
    return {
        'model_kwargs': {
            'layers': 2,  # Smaller for quick testing
            'dim': 32,    # Smaller for quick testing
            'key_size': 8,
            'num_heads': 2,
            'dropout': 0.0,  # No dropout for testing
        },
        'learning_rate': 1e-3,  # Fixed learning rate
        'batch_size': 16,
        'gradient_clip_norm': 1.0,
        'max_parent_size': 3,  # Allow up to 3 parents for full SCM
    }


def create_improved_optimizer(config):
    """Create optimizer with gradient clipping."""
    return optax.chain(
        optax.clip_by_global_norm(config['gradient_clip_norm']),
        optax.adam(learning_rate=config['learning_rate'])
    )


def test_simple_two_variable_case():
    """Test on simplest case: X → Y (only 1 edge)."""
    print("🧪 QUICK VALIDATION TEST")
    print("=" * 50)
    print("Testing SCM: X → Y ← Z")
    print("Expected: X should have {} parents, Y should have {X,Z} parents, Z should have {} parents")
    print("=" * 50)
    
    # Create simple SCM with X → Y ← Z
    config = create_simple_config()
    
    # Use the full SCM - don't try to subset variables
    scm = create_simple_test_scm(noise_scale=1.0, target="Y")
    from causal_bayes_opt.data_structures.scm import get_variables
    variables = sorted(get_variables(scm))  # Use ALL variables from SCM
    
    print(f"Variables in SCM: {variables}")
    
    # Test cases
    test_cases = [
        ('X', frozenset()),        # X is root -> empty parent set
        ('Z', frozenset()),        # Z is root -> empty parent set  
        ('Y', frozenset(['X', 'Z']))    # Y has X,Z as parents
    ]
    
    # Create model
    net = create_parent_set_model(
        model_kwargs=config['model_kwargs'],
        max_parent_size=config['max_parent_size']
    )
    
    for target_var, expected_parents in test_cases:
        print(f"\n" + "="*60)
        print(f"🎯 TESTING TARGET: {target_var}")
        print(f"Expected parent set: {set(expected_parents) if expected_parents else '{}'}")
        print("="*60)
        
        # Generate data
        samples = sample_from_linear_scm(scm, n_samples=config['batch_size'], seed=42)
        batch = create_training_batch(scm, samples, target_var)
        
        # Initialize model
        params = net.init(random.PRNGKey(42), batch['x'], variables, target_var, True)
        
        # DEBUG: Check parent set enumeration
        print("\n📋 PARENT SET ENUMERATION:")
        debug_parent_set_enumeration(variables, target_var, config['max_parent_size'])
        
        # DEBUG: Check initial model output (before training)
        print("\n🔍 INITIAL MODEL OUTPUT (before training):")
        initial_output = debug_training_step(net, params, batch['x'], variables, target_var, expected_parents)
        
        # Quick training test (just 5 steps to see if it improves)
        print("\n🏃 QUICK TRAINING (5 steps):")
        optimizer = create_improved_optimizer(config)
        opt_state = optimizer.init(params)
        train_step_fn = create_train_step(net, optimizer)
        
        for step in range(5):
            # Generate fresh samples for each step
            step_samples = sample_from_linear_scm(scm, n_samples=config['batch_size'], seed=42+step)
            step_batch = create_training_batch(scm, step_samples, target_var)
            
            params, opt_state, loss = train_step_fn(
                params, opt_state, step_batch['x'], variables, target_var, expected_parents
            )
            print(f"  Step {step}: loss = {loss:.4f}")
        
        # DEBUG: Check final model output (after training)
        print("\n🎯 FINAL MODEL OUTPUT (after training):")
        final_predictions = predict_parent_sets(net, params, batch['x'], variables, target_var)
        
        print("Final predictions:")
        for i, (ps, prob) in enumerate(zip(final_predictions['parent_sets'], final_predictions['probabilities'])):
            ps_str = set(ps) if ps else "{}"
            is_correct = "✅" if ps == expected_parents else "  "
            print(f"{is_correct} {i+1}. {ps_str}: {prob:.3f}")
        
        # Check for improvement
        top_prediction = final_predictions['parent_sets'][0]
        if top_prediction == expected_parents:
            print(f"✅ SUCCESS: Model correctly predicts {set(expected_parents) if expected_parents else '{}'}!")
        else:
            print(f"❌ NEEDS WORK: Expected {set(expected_parents) if expected_parents else '{}'}, got {set(top_prediction) if top_prediction else '{}'}")
        
        # Check for key improvements
        logits_output = net.apply(params, random.PRNGKey(0), batch['x'], variables, target_var, False)
        logits = logits_output['parent_set_logits']
        
        print(f"\n📊 MODEL HEALTH CHECK:")
        print(f"  Logit range: [{jnp.min(logits):.3f}, {jnp.max(logits):.3f}]")
        if jnp.max(logits) - jnp.min(logits) > 20:
            print("  ⚠️ WARNING: Still large logit range - may need more fixes")
        else:
            print("  ✅ GOOD: Reasonable logit range")
        
        # Check empty set probability specifically for root nodes
        if target_var in ['X', 'Z']:  # Root nodes
            parent_sets = logits_output['parent_sets']
            probabilities = jax.nn.softmax(logits)
            for i, ps in enumerate(parent_sets):
                if len(ps) == 0:  # Empty set
                    empty_prob = probabilities[i]
                    print(f"  Empty set probability: {empty_prob:.3f}")
                    if empty_prob > 0.1:  # Should be reasonably high for root
                        print(f"  ✅ GOOD: Empty set getting reasonable probability")
                    else:
                        print(f"  ⚠️ ISSUE: Empty set still getting low probability")
                    break
    
    print(f"\n" + "="*60)
    print("🎉 VALIDATION TEST COMPLETE!")
    print("\nWhat to look for:")
    print("  ✅ Logit ranges should be reasonable (not 50+)")
    print("  ✅ Empty sets should get >0.1 probability for root nodes")
    print("  ✅ Training losses should decrease (not explode)")
    print("  ✅ Model should eventually predict correct parent sets")
    print("="*60)


if __name__ == "__main__":
    test_simple_two_variable_case()
