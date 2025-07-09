#!/usr/bin/env python3
"""
Debug the input extraction pipeline.

This test examines what inputs are actually being fed to the policy network
from TensorBackedAcquisitionState, since the masking logic works correctly
in isolation but fails in the full pipeline.
"""

import jax
import jax.numpy as jnp
import pyrsistent as pyr
import sys
import os
sys.path.append('/Users/harellidar/Documents/Imperial/Individual_Project/causal_bayes_opt')

from src.causal_bayes_opt.jax_native.state import create_tensor_backed_state_from_scm
from src.causal_bayes_opt.acquisition.grpo import _extract_policy_input_from_tensor_state
from src.causal_bayes_opt.data_structures.scm import create_scm


def create_test_scm():
    """Create a simple test SCM."""
    return create_scm(
        variables=frozenset(['X0', 'X1', 'X2', 'X3']),
        edges=frozenset([('X0', 'X1'), ('X2', 'X1')]),  # X0 and X2 cause X1
        mechanisms={
            'X0': lambda: 0.5,
            'X1': lambda: 0.3, 
            'X2': lambda: 0.8,
            'X3': lambda: 0.2
        },
        target='X1'
    )


def test_input_extraction():
    """Test what inputs are extracted from TensorBackedAcquisitionState."""
    print("🔍 DEBUGGING INPUT EXTRACTION PIPELINE")
    print("=" * 60)
    
    # Create test SCM and state
    scm = create_test_scm()
    print(f"Test SCM variables: {list(scm.get('variables', []))}")
    print(f"Test SCM target: {scm.get('target', 'unknown')}")
    
    # Create TensorBackedAcquisitionState
    state = create_tensor_backed_state_from_scm(
        scm=scm,
        step=10,
        best_value=1.5,
        uncertainty_bits=2.0
    )
    
    print(f"\nTensorBackedAcquisitionState analysis:")
    print(f"  n_vars: {state.config.n_vars}")
    print(f"  max_history: {state.config.max_history}")
    print(f"  current_step: {state.current_step}")
    print(f"  best_value: {state.best_value}")
    print(f"  uncertainty_bits: {state.uncertainty_bits}")
    
    # Examine state tensor fields
    print(f"\nState tensor fields:")
    print(f"  mechanism_features shape: {state.mechanism_features.shape}")
    print(f"  mechanism_features range: [{jnp.min(state.mechanism_features):.6f}, {jnp.max(state.mechanism_features):.6f}]")
    print(f"  mechanism_features mean: {jnp.mean(state.mechanism_features):.6f}")
    
    print(f"  marginal_probs shape: {state.marginal_probs.shape}")
    print(f"  marginal_probs range: [{jnp.min(state.marginal_probs):.6f}, {jnp.max(state.marginal_probs):.6f}]")
    print(f"  marginal_probs: {state.marginal_probs}")
    
    print(f"  confidence_scores shape: {state.confidence_scores.shape}")
    print(f"  confidence_scores range: [{jnp.min(state.confidence_scores):.6f}, {jnp.max(state.confidence_scores):.6f}]")
    print(f"  confidence_scores: {state.confidence_scores}")
    
    # Extract policy input
    print(f"\n--- EXTRACTING POLICY INPUT ---")
    policy_input = _extract_policy_input_from_tensor_state(state)
    
    print(f"Policy input shape: {policy_input.shape}")
    print(f"Policy input range: [{jnp.min(policy_input):.6f}, {jnp.max(policy_input):.6f}]")
    print(f"Policy input mean: {jnp.mean(policy_input):.6f}")
    print(f"Policy input std: {jnp.std(policy_input):.6f}")
    
    # Check for problematic values
    nan_count = jnp.sum(jnp.isnan(policy_input))
    inf_count = jnp.sum(jnp.isinf(policy_input))
    extreme_count = jnp.sum(jnp.abs(policy_input) > 100)
    
    print(f"\nPolicy input quality check:")
    print(f"  NaN values: {nan_count}")
    print(f"  Infinite values: {inf_count}")  
    print(f"  Extreme values (>100): {extreme_count}")
    
    # Analyze each channel
    print(f"\n--- CHANNEL-BY-CHANNEL ANALYSIS ---")
    max_history, n_vars, num_channels = policy_input.shape
    
    for channel in range(num_channels):
        channel_data = policy_input[:, :, channel]
        print(f"Channel {channel}:")
        print(f"  Range: [{jnp.min(channel_data):.6f}, {jnp.max(channel_data):.6f}]")
        print(f"  Mean: {jnp.mean(channel_data):.6f}")
        print(f"  Std: {jnp.std(channel_data):.6f}")
        
        # Check for constant values (which might indicate problems)
        unique_values = len(jnp.unique(channel_data))
        print(f"  Unique values: {unique_values}")
        
        if unique_values == 1:
            print(f"  ⚠️  WARNING: Channel {channel} has constant value {jnp.unique(channel_data)[0]:.6f}")
        
        if jnp.max(jnp.abs(channel_data)) > 10:
            print(f"  ⚠️  WARNING: Channel {channel} has large values")
    
    return policy_input, state


def test_realistic_vs_problematic_inputs():
    """Compare the extracted inputs vs manually created realistic inputs."""
    print(f"\n" + "=" * 60)
    print("🔧 COMPARING EXTRACTED VS REALISTIC INPUTS")
    print("=" * 60)
    
    # Get extracted input
    extracted_input, state = test_input_extraction()
    
    # Create realistic input (similar to our working debug test)
    max_history, n_vars, num_channels = extracted_input.shape
    
    print(f"\n--- CREATING REALISTIC INPUT FOR COMPARISON ---")
    realistic_input = jnp.zeros((max_history, n_vars, num_channels))
    
    # Fill with reasonable values
    for var_idx in range(n_vars):
        # Channels 0-2: mechanism features (vary by variable)
        realistic_input = realistic_input.at[:, var_idx, 0].set(0.5 + var_idx * 0.2)  # Effect size
        realistic_input = realistic_input.at[:, var_idx, 1].set(0.3 + var_idx * 0.1)  # Uncertainty
        realistic_input = realistic_input.at[:, var_idx, 2].set(0.8 - var_idx * 0.1)  # Confidence
        
        # Channel 3: Parent probabilities  
        if var_idx != 1:  # Assume variable 1 is target
            realistic_input = realistic_input.at[:, var_idx, 3].set(0.4 + var_idx * 0.15)
        else:
            realistic_input = realistic_input.at[:, var_idx, 3].set(0.0)
        
        # Channel 4: Confidence scores
        realistic_input = realistic_input.at[:, var_idx, 4].set(0.6 + (var_idx % 2) * 0.2)
        
        # Channels 5-8: Global context (time-varying)
        for t in range(max_history):
            time_factor = t / max_history
            realistic_input = realistic_input.at[t, :, 5].set(1.5 + time_factor)      # Best value
            realistic_input = realistic_input.at[t, :, 6].set(2.0 - time_factor * 0.5) # Uncertainty
            realistic_input = realistic_input.at[t, :, 7].set(time_factor * 100)       # Step count  
            realistic_input = realistic_input.at[t, :, 8].set(time_factor * 50)        # Sample count
        
        # Channel 9: Step noise
        realistic_input = realistic_input.at[:, var_idx, 9].set(0.1 + var_idx * 0.05)
    
    print(f"Realistic input stats:")
    print(f"  Range: [{jnp.min(realistic_input):.6f}, {jnp.max(realistic_input):.6f}]")
    print(f"  Mean: {jnp.mean(realistic_input):.6f}")
    print(f"  Std: {jnp.std(realistic_input):.6f}")
    
    # Compare key differences
    print(f"\n--- COMPARISON ---")
    print(f"Extracted vs Realistic:")
    print(f"  Range difference: extracted=[{jnp.min(extracted_input):.6f}, {jnp.max(extracted_input):.6f}] vs realistic=[{jnp.min(realistic_input):.6f}, {jnp.max(realistic_input):.6f}]")
    print(f"  Mean difference: {jnp.mean(extracted_input) - jnp.mean(realistic_input):.6f}")
    print(f"  Std difference: {jnp.std(extracted_input) - jnp.std(realistic_input):.6f}")
    
    # Test both inputs with a simple policy network
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
    
    key = jax.random.PRNGKey(42)
    target_idx = 1
    
    # Initialize with realistic input
    params = init_fn.init(key, realistic_input, target_idx, False)
    
    print(f"\n--- POLICY NETWORK OUTPUTS ---")
    
    # Test extracted input
    try:
        extracted_output = apply_fn(params, key, extracted_input, target_idx, False)
        extracted_logits = extracted_output['variable_logits']
        print(f"Extracted input -> variable_logits: {extracted_logits}")
        print(f"  Range: [{jnp.min(extracted_logits):.6f}, {jnp.max(extracted_logits):.6f}]")
        
        extreme_count = jnp.sum(jnp.abs(extracted_logits) > 1e6)
        print(f"  Extreme values: {extreme_count}/{len(extracted_logits)}")
        
        if extreme_count > 1:  # More than just the target
            print("  ❌ PROBLEM: Extracted input produces extreme values for multiple variables!")
        else:
            print("  ✅ Extracted input produces reasonable outputs")
            
    except Exception as e:
        print(f"  ❌ ERROR with extracted input: {e}")
    
    # Test realistic input
    try:
        realistic_output = apply_fn(params, key, realistic_input, target_idx, False)
        realistic_logits = realistic_output['variable_logits']
        print(f"Realistic input -> variable_logits: {realistic_logits}")
        print(f"  Range: [{jnp.min(realistic_logits):.6f}, {jnp.max(realistic_logits):.6f}]")
        
        extreme_count = jnp.sum(jnp.abs(realistic_logits) > 1e6)
        print(f"  Extreme values: {extreme_count}/{len(realistic_logits)}")
        
        if extreme_count > 1:
            print("  ❌ PROBLEM: Even realistic input produces extreme values!")
        else:
            print("  ✅ Realistic input produces reasonable outputs")
            
    except Exception as e:
        print(f"  ❌ ERROR with realistic input: {e}")


if __name__ == "__main__":
    test_realistic_vs_problematic_inputs()