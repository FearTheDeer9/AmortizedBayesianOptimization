#!/usr/bin/env python3
"""
Test the fixed input extraction in the actual training context.

This reproduces the training setup to see if the input extraction fix
resolves the zero intervention issue.
"""

import sys
import os
from pathlib import Path
sys.path.append('/Users/harellidar/Documents/Imperial/Individual_Project/causal_bayes_opt')

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as onp
import pyrsistent as pyr

# Import training components
from src.causal_bayes_opt.training.enriched_trainer import EnrichedGRPOTrainer
from src.causal_bayes_opt.data_structures.scm import create_scm
from src.causal_bayes_opt.acquisition.grpo import _extract_policy_input_from_tensor_state


def test_single_training_step():
    """Test a single training step with the fixed input extraction."""
    print("🧪 TESTING FIXED INPUT EXTRACTION IN TRAINING CONTEXT")
    print("=" * 60)
    
    # Create a simple test SCM
    scm = create_scm(
        variables=frozenset(['X0', 'X1', 'X2']),
        edges=frozenset([('X0', 'X1'), ('X1', 'X2')]),
        mechanisms={
            'X0': lambda: 0.5,
            'X1': lambda: 0.3,
            'X2': lambda: 0.8
        },
        target='X2'
    )
    
    print(f"Test SCM: {list(scm.get('variables', []))}")
    print(f"Target: {scm.get('target', 'unknown')}")
    
    # Test directly without full trainer setup
    try:
        # Test the state creation directly
        print("\n--- TESTING STATE CREATION ---")
        
        from src.causal_bayes_opt.jax_native.state import create_tensor_backed_state_from_scm
        
        state = create_tensor_backed_state_from_scm(
            scm=scm,
            step=5,
            best_value=1.2
        )
        
        print(f"✅ Created state: step={state.current_step}, best_value={state.best_value}")
        print(f"  Variables: {state.variable_names}")
        print(f"  Target: {state.current_target}")
        
        # Test our fixed input extraction directly
        print("\n--- TESTING FIXED INPUT EXTRACTION ---")
        
        policy_input = _extract_policy_input_from_tensor_state(state)
        
        print(f"Policy input shape: {policy_input.shape}")
        print(f"Policy input range: [{jnp.min(policy_input):.6f}, {jnp.max(policy_input):.6f}]")
        print(f"Policy input mean: {jnp.mean(policy_input):.6f}")
        print(f"Policy input std: {jnp.std(policy_input):.6f}")
        
        # Check for variation across channels (this was the main fix)
        print(f"\n--- CHECKING CHANNEL VARIATION ---")
        max_history, n_vars, num_channels = policy_input.shape
        
        varied_channels = 0
        for ch in range(num_channels):
            channel_data = policy_input[:, :, ch]
            unique_values = len(jnp.unique(channel_data))
            ch_std = jnp.std(channel_data)
            
            print(f"  Channel {ch}: {unique_values} unique values, std={ch_std:.6f}")
            
            if unique_values > 1 and ch_std > 1e-6:
                varied_channels += 1
        
        print(f"\nVariation Analysis:")
        print(f"  Channels with variation: {varied_channels}/{num_channels}")
        print(f"  Constant channels: {num_channels - varied_channels}/{num_channels}")
        
        if varied_channels >= num_channels * 0.7:  # At least 70% should have variation
            print("✅ SUCCESS: Good channel variation detected!")
        else:
            print("⚠️ WARNING: Many channels still lack variation")
        
        # Test policy network with the improved inputs
        print("\n--- TESTING POLICY NETWORK WITH IMPROVED INPUTS ---")
        
        # Create a minimal policy network for testing
        from src.causal_bayes_opt.acquisition.enriched.policy_heads import EnrichedAcquisitionPolicyNetwork
        import haiku as hk
        
        def policy_fn(enriched_input, target_idx, is_training):
            network = EnrichedAcquisitionPolicyNetwork(
                num_layers=4, num_heads=8, hidden_dim=128,
                key_size=32, widening_factor=4, dropout=0.0
            )
            return network(enriched_input, target_idx, is_training)
        
        init_fn = hk.transform(policy_fn)
        apply_fn = init_fn.apply
        
        key = random.PRNGKey(42)
        target_idx = state.variable_names.index(state.current_target)
        
        # Initialize and test
        params = init_fn.init(key, policy_input, target_idx, False)
        
        # Test with the improved policy input
        policy_output = apply_fn(params, key, policy_input, target_idx, False)
        
        var_logits = policy_output['variable_logits']
        print(f"Variable logits: {var_logits}")
        print(f"Variable logits range: [{jnp.min(var_logits):.6f}, {jnp.max(var_logits):.6f}]")
        
        # Check softmax probabilities
        probs = jax.nn.softmax(var_logits)
        print(f"Softmax probabilities: {probs}")
        print(f"Max non-target probability: {jnp.max(probs[jnp.arange(len(probs)) != target_idx]):.6f}")
        
        # Count how many variables have reasonable probabilities
        reasonable_probs = jnp.sum(probs[jnp.arange(len(probs)) != target_idx] > 0.01)
        print(f"Variables with >1% probability: {reasonable_probs}/{len(probs)-1}")
        
        if reasonable_probs > 0:
            print("✅ SUCCESS: Policy network produces reasonable intervention probabilities!")
            print("🎉 Input extraction fix appears to be working!")
        else:
            print("❌ ISSUE: Policy still produces very low probabilities for all variables")
            
    except Exception as e:
        print(f"❌ Error in test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_single_training_step()