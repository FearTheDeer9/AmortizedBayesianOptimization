#!/usr/bin/env python3
"""
Phase 1: Policy Network Isolation Testing

Comprehensive tests to verify if the enriched policy network can produce 
non-zero outputs when tested in isolation from the training loop.

This will help determine if the zero-output issue is in:
1. Network architecture/initialization 
2. Input preprocessing
3. GRPO integration
4. Training loop dynamics
"""

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as onp
from typing import Dict, Any

# Add project root to path
import sys
import os
sys.path.append('/Users/harellidar/Documents/Imperial/Individual_Project/causal_bayes_opt')

from src.causal_bayes_opt.acquisition.enriched.policy_heads import (
    EnrichedAcquisitionPolicyNetwork, 
    SimplifiedPolicyHeads,
    PolicyOutputValidator
)


def create_realistic_enriched_input(n_vars: int = 4, max_history: int = 50, num_channels: int = 10) -> jnp.ndarray:
    """Create realistic enriched input tensor for testing."""
    
    # Create varied, non-zero input data that should lead to different policy outputs
    key = jax.random.PRNGKey(42)
    
    # Base features with realistic variance
    base_features = jax.random.normal(key, (max_history, n_vars, num_channels)) * 0.5
    
    # Add structured patterns that policy should learn from
    
    # Channel 0-2: Mechanism features (vary by variable)
    for var_idx in range(n_vars):
        base_features = base_features.at[:, var_idx, 0].set(0.5 + var_idx * 0.2)  # Effect size varies
        base_features = base_features.at[:, var_idx, 1].set(0.3 + var_idx * 0.1)  # Uncertainty varies  
        base_features = base_features.at[:, var_idx, 2].set(0.8 - var_idx * 0.1)  # Confidence varies
    
    # Channel 3: Parent probabilities (some variables more likely parents)
    for var_idx in range(n_vars):
        if var_idx != 1:  # Assume variable 1 is target
            base_features = base_features.at[:, var_idx, 3].set(0.4 + var_idx * 0.15)
        else:
            base_features = base_features.at[:, var_idx, 3].set(0.0)  # Target has no parent prob
    
    # Channel 4: Confidence scores
    for var_idx in range(n_vars):
        base_features = base_features.at[:, var_idx, 4].set(0.6 + (var_idx % 2) * 0.2)
    
    # Channels 5-8: Global context (time-varying)
    for t in range(max_history):
        time_factor = t / max_history
        base_features = base_features.at[t, :, 5].set(1.5 + time_factor)      # Best value increases
        base_features = base_features.at[t, :, 6].set(2.0 - time_factor * 0.5) # Uncertainty decreases  
        base_features = base_features.at[t, :, 7].set(time_factor * 100)       # Step count
        base_features = base_features.at[t, :, 8].set(time_factor * 50)        # Sample count
    
    # Channel 9: Step noise
    step_noise = jax.random.uniform(key, (max_history, n_vars, 1)) * 0.2
    base_features = base_features.at[:, :, 9:10].set(step_noise)
    
    return base_features


def test_policy_initialization():
    """Test 1: Verify policy network initializes with reasonable parameters."""
    print("=== Test 1: Policy Network Initialization ===")
    
    config = {
        'num_layers': 4,
        'num_heads': 8, 
        'hidden_dim': 128,
        'key_size': 32,
        'widening_factor': 4,
        'dropout': 0.1
    }
    
    def policy_fn(enriched_input, target_idx, is_training):
        network = EnrichedAcquisitionPolicyNetwork(**config)
        return network(enriched_input, target_idx, is_training)
    
    # Initialize network
    key = jax.random.PRNGKey(42)
    enriched_input = create_realistic_enriched_input(n_vars=4)
    target_idx = 1
    
    init_fn = hk.transform(policy_fn)
    params = init_fn.init(key, enriched_input, target_idx, True)
    
    # Check parameter statistics
    param_sizes = []
    param_means = []
    param_stds = []
    zero_params = 0
    total_params = 0
    
    def analyze_params(path, param):
        nonlocal zero_params, total_params
        if isinstance(param, jnp.ndarray):
            param_sizes.append(param.shape)
            param_means.append(float(jnp.mean(param)))
            param_stds.append(float(jnp.std(param)))
            
            zero_count = int(jnp.sum(param == 0.0))
            zero_params += zero_count
            total_params += param.size
            
            print(f"  {path}: shape={param.shape}, mean={jnp.mean(param):.6f}, std={jnp.std(param):.6f}, zeros={zero_count}/{param.size}")
    
    # Analyze all parameters - fix Haiku traverse syntax
    jax.tree_util.tree_map_with_path(analyze_params, params)
    
    print(f"\nParameter Summary:")
    print(f"  Total parameters: {total_params}")
    print(f"  Zero parameters: {zero_params} ({100*zero_params/total_params:.1f}%)")
    print(f"  Mean of means: {onp.mean(param_means):.6f}")
    print(f"  Mean of stds: {onp.mean(param_stds):.6f}")
    
    # Check if initialization looks reasonable
    if zero_params / total_params > 0.5:
        print("❌ WARNING: More than 50% of parameters are zero!")
    else:
        print("✅ Parameter initialization looks reasonable")
    
    return params, init_fn


def test_forward_pass(params, apply_fn):
    """Test 2: Verify network produces non-zero, finite outputs."""
    print("\n=== Test 2: Forward Pass Output Validation ===")
    
    key = jax.random.PRNGKey(123)
    enriched_input = create_realistic_enriched_input(n_vars=4)
    target_idx = 1
    
    # Forward pass
    outputs = apply_fn(params, key, enriched_input, target_idx, False)
    
    print("Network outputs:")
    for key_name, value in outputs.items():
        if isinstance(value, jnp.ndarray):
            print(f"  {key_name}: shape={value.shape}, mean={jnp.mean(value):.6f}, std={jnp.std(value):.6f}")
            print(f"    range=[{jnp.min(value):.6f}, {jnp.max(value):.6f}], finite={jnp.all(jnp.isfinite(value))}")
            
            # Check for zero outputs
            zero_count = int(jnp.sum(value == 0.0))
            total_count = value.size
            print(f"    zeros: {zero_count}/{total_count} ({100*zero_count/total_count:.1f}%)")
    
    # Validate using built-in validator
    is_valid = PolicyOutputValidator.validate_policy_outputs(outputs, n_vars=4)
    print(f"\nPolicy output validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
    
    # Check for specific zero-output issues
    variable_logits = outputs['variable_logits']
    value_params = outputs['value_params'] 
    state_value = outputs['state_value']
    
    issues = []
    
    # Check variable logits
    if jnp.allclose(variable_logits, 0.0, atol=1e-6):
        issues.append("Variable logits are all near zero")
    
    # Check value parameters
    if jnp.allclose(value_params, 0.0, atol=1e-6):
        issues.append("Value parameters are all near zero")
        
    # Check state value
    if jnp.allclose(state_value, 0.0, atol=1e-6):
        issues.append("State value is near zero")
    
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ All outputs have reasonable non-zero values")
    
    return outputs, issues


def test_gradient_flow(params, apply_fn):
    """Test 3: Verify gradients flow properly through network."""
    print("\n=== Test 3: Gradient Flow Analysis ===")
    
    def loss_fn(params):
        key = jax.random.PRNGKey(456)
        enriched_input = create_realistic_enriched_input(n_vars=4)
        target_idx = 1
        
        outputs = apply_fn(params, key, enriched_input, target_idx, True)
        
        # Simple loss that should produce gradients
        variable_logits = outputs['variable_logits'] 
        value_params = outputs['value_params']
        state_value = outputs['state_value']
        
        # Sum all outputs (should definitely produce gradients)
        total_loss = jnp.sum(variable_logits) + jnp.sum(value_params) + jnp.sum(state_value)
        return total_loss
    
    # Compute gradients
    loss_value, grads = jax.value_and_grad(loss_fn)(params)
    
    print(f"Loss value: {loss_value:.6f}")
    
    # Analyze gradients
    grad_norms = []
    zero_grads = 0
    total_grad_params = 0
    
    def analyze_grads(path, grad):
        nonlocal zero_grads, total_grad_params
        if isinstance(grad, jnp.ndarray):
            grad_norm = float(jnp.linalg.norm(grad))
            grad_norms.append(grad_norm)
            
            zero_count = int(jnp.sum(jnp.abs(grad) < 1e-8))
            zero_grads += zero_count
            total_grad_params += grad.size
            
            print(f"  {path}: norm={grad_norm:.8f}, mean={jnp.mean(grad):.8f}, zeros={zero_count}/{grad.size}")
    
    jax.tree_util.tree_map_with_path(analyze_grads, grads)
    
    print(f"\nGradient Summary:")
    print(f"  Total gradient parameters: {total_grad_params}")
    print(f"  Zero gradients: {zero_grads} ({100*zero_grads/total_grad_params:.1f}%)")
    print(f"  Mean gradient norm: {onp.mean(grad_norms):.8f}")
    print(f"  Max gradient norm: {onp.max(grad_norms):.8f}")
    
    # Check gradient flow
    if zero_grads / total_grad_params > 0.9:
        print("❌ WARNING: More than 90% of gradients are zero - poor gradient flow!")
    elif onp.mean(grad_norms) < 1e-6:
        print("❌ WARNING: Gradient norms are very small - weak gradient signal!")
    else:
        print("✅ Gradient flow looks reasonable")
    
    return grads, grad_norms


def test_different_inputs():
    """Test 4: Verify network responds differently to different inputs."""
    print("\n=== Test 4: Input Sensitivity Analysis ===")
    
    config = {
        'num_layers': 4,
        'num_heads': 8,
        'hidden_dim': 128,
        'key_size': 32,
        'widening_factor': 4,
        'dropout': 0.0  # No dropout for deterministic testing
    }
    
    def policy_fn(enriched_input, target_idx, is_training):
        network = EnrichedAcquisitionPolicyNetwork(**config)
        return network(enriched_input, target_idx, is_training)
    
    init_fn = hk.transform(policy_fn)
    apply_fn = init_fn.apply
    
    # Initialize once
    key = jax.random.PRNGKey(789)
    dummy_input = create_realistic_enriched_input(n_vars=4)
    params = init_fn.init(key, dummy_input, 1, False)
    
    # Test with different inputs
    inputs = [
        create_realistic_enriched_input(n_vars=4),  # Input 1
        create_realistic_enriched_input(n_vars=4) * 2.0,  # Input 2 (scaled)
        jnp.ones((50, 4, 10)) * 0.5,  # Input 3 (constant)
        jnp.zeros((50, 4, 10)),  # Input 4 (all zeros)
    ]
    
    input_names = ["Realistic", "Scaled", "Constant", "Zeros"]
    outputs_list = []
    
    for i, enriched_input in enumerate(inputs):
        output = apply_fn(params, key, enriched_input, 1, False)
        outputs_list.append(output)
        
        var_logits = output['variable_logits']
        print(f"  {input_names[i]}: variable_logits range=[{jnp.min(var_logits):.4f}, {jnp.max(var_logits):.4f}]")
    
    # Compare outputs between different inputs
    different_outputs = True
    for i in range(1, len(outputs_list)):
        var_logits_0 = outputs_list[0]['variable_logits']
        var_logits_i = outputs_list[i]['variable_logits']
        
        # Check if outputs are significantly different
        diff = jnp.mean(jnp.abs(var_logits_0 - var_logits_i))
        print(f"  Difference between Realistic and {input_names[i]}: {diff:.6f}")
        
        if diff < 1e-4:
            different_outputs = False
    
    if different_outputs:
        print("✅ Network produces different outputs for different inputs")
    else:
        print("❌ WARNING: Network produces very similar outputs for different inputs!")
    
    return outputs_list


def test_policy_heads_individually():
    """Test 5: Test individual policy heads in isolation."""
    print("\n=== Test 5: Individual Policy Head Testing ===")
    
    def test_head(head_name, head_fn):
        print(f"\n--- Testing {head_name} ---")
        
        key = jax.random.PRNGKey(999)
        
        # Create dummy variable embeddings (output of transformer)
        n_vars = 4
        hidden_dim = 128
        variable_embeddings = jax.random.normal(key, (n_vars, hidden_dim)) * 0.1
        
        try:
            output = head_fn(variable_embeddings, 1, 0.0)  # target_idx=1, dropout=0
            
            print(f"  Output shape: {output.shape}")
            print(f"  Output range: [{jnp.min(output):.6f}, {jnp.max(output):.6f}]")
            print(f"  Output mean: {jnp.mean(output):.6f}")
            print(f"  All finite: {jnp.all(jnp.isfinite(output))}")
            
            # Check for zero outputs
            zero_count = int(jnp.sum(jnp.abs(output) < 1e-6))
            print(f"  Near-zero values: {zero_count}/{output.size}")
            
            if zero_count == output.size:
                print(f"  ❌ WARNING: All outputs from {head_name} are near zero!")
            else:
                print(f"  ✅ {head_name} produces non-zero outputs")
                
        except Exception as e:
            print(f"  ❌ ERROR in {head_name}: {e}")
    
    # Test each head individually
    def policy_heads_fn(variable_embeddings, target_idx, dropout_rate):
        heads = SimplifiedPolicyHeads(hidden_dim=128, dropout=0.0)
        
        # Test individual methods
        var_logits = heads._variable_selection_head(variable_embeddings, target_idx, dropout_rate)
        val_params = heads._value_selection_head(variable_embeddings, dropout_rate)  
        state_val = heads._state_value_head(variable_embeddings, dropout_rate)
        
        return {
            'variable_logits': var_logits,
            'value_params': val_params,
            'state_value': state_val
        }
    
    init_fn = hk.transform(policy_heads_fn)
    apply_fn = init_fn.apply
    
    key = jax.random.PRNGKey(111)
    dummy_embeddings = jax.random.normal(key, (4, 128)) * 0.1
    params = init_fn.init(key, dummy_embeddings, 1, 0.0)
    
    outputs = apply_fn(params, key, dummy_embeddings, 1, 0.0)
    
    for head_name, output in outputs.items():
        print(f"\n{head_name}:")
        print(f"  Shape: {output.shape}")
        print(f"  Range: [{jnp.min(output):.6f}, {jnp.max(output):.6f}]")
        print(f"  Mean: {jnp.mean(output):.6f}")
        zero_frac = jnp.sum(jnp.abs(output) < 1e-6) / output.size
        print(f"  Zero fraction: {zero_frac:.2%}")


def main():
    """Run all isolation tests."""
    print("🔍 PHASE 1: POLICY NETWORK ISOLATION TESTING")
    print("=" * 60)
    
    # Test 1: Initialization
    params, apply_fn = test_policy_initialization()
    
    # Test 2: Forward pass
    outputs, issues = test_forward_pass(params, apply_fn.apply)
    
    # Test 3: Gradient flow
    grads, grad_norms = test_gradient_flow(params, apply_fn.apply)
    
    # Test 4: Input sensitivity
    different_outputs = test_different_inputs()
    
    # Test 5: Individual heads
    test_policy_heads_individually()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 PHASE 1 SUMMARY:")
    
    major_issues = []
    
    if len(issues) > 0:
        major_issues.extend(issues)
    
    if onp.mean(grad_norms) < 1e-6:
        major_issues.append("Very weak gradient signals")
        
    if major_issues:
        print("❌ ISSUES FOUND:")
        for issue in major_issues:
            print(f"  - {issue}")
        print("\n🔧 RECOMMENDATION: Policy network has fundamental issues that need fixing")
    else:
        print("✅ Policy network isolation tests PASSED")
        print("🔧 RECOMMENDATION: Issue likely in GRPO integration or training loop (proceed to Phase 2)")
    
    return len(major_issues) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)