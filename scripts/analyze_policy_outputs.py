#!/usr/bin/env python3
"""
Analyze GRPO policy outputs to understand why it always selects X.
"""

import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.policies.clean_policy_factory import create_clean_grpo_policy


def create_test_tensor(n_vars=3, T=100, scenario="diverse"):
    """Create test tensor with different intervention patterns."""
    # Create 5-channel tensor [T, n_vars, 5]
    # Channels: [value, intervened, is_target, graph_belief_1, graph_belief_2]
    tensor = np.zeros((T, n_vars, 5))
    
    # Set values
    tensor[:, :, 0] = np.random.randn(T, n_vars)
    
    # Mark target (last variable)
    tensor[:, -1, 2] = 1.0
    
    # Create intervention patterns based on scenario
    if scenario == "all_X":
        # All past interventions on X
        for t in range(T//2):
            tensor[t, 0, 1] = 1.0  # intervened on X
            tensor[t, 0, 0] = np.random.uniform(-2, 2)  # intervention value
    elif scenario == "all_Y":
        # All past interventions on Y
        for t in range(T//2):
            tensor[t, 1, 1] = 1.0  # intervened on Y
            tensor[t, 1, 0] = np.random.uniform(-2, 2)
    elif scenario == "diverse":
        # Mixed interventions
        for t in range(T//2):
            var_idx = t % (n_vars - 1)  # Don't intervene on target
            tensor[t, var_idx, 1] = 1.0
            tensor[t, var_idx, 0] = np.random.uniform(-2, 2)
    
    # Add some graph belief info (random for now)
    tensor[:, :, 3:5] = np.random.rand(T, n_vars, 2)
    
    return jnp.array(tensor)


def analyze_policy_behavior():
    """Analyze policy outputs for different scenarios."""
    print("="*80)
    print("ANALYZING GRPO POLICY OUTPUTS")
    print("="*80)
    
    # Create policy
    policy_fn = create_clean_grpo_policy(hidden_dim=256)
    policy = hk.transform(policy_fn)
    
    # Initialize with random key
    key = jax.random.PRNGKey(42)
    
    # Test scenarios
    scenarios = [
        ("3 variables, diverse history", 3, "diverse"),
        ("3 variables, all X history", 3, "all_X"),
        ("3 variables, all Y history", 3, "all_Y"),
        ("5 variables, diverse history", 5, "diverse"),
    ]
    
    for desc, n_vars, pattern in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {desc}")
        print(f"Variables: {n_vars}, Target: var_{n_vars-1}")
        print("="*60)
        
        # Create test tensor
        tensor = create_test_tensor(n_vars, T=100, scenario=pattern)
        target_idx = n_vars - 1
        
        # Initialize policy
        init_key, apply_key = jax.random.split(key)
        params = policy.init(init_key, tensor, target_idx)
        
        # Run policy multiple times to check consistency
        print("\nRunning policy 5 times:")
        outputs = []
        for i in range(5):
            run_key = jax.random.PRNGKey(i)
            output = policy.apply(params, run_key, tensor, target_idx)
            outputs.append(output)
            
            var_logits = output['variable_logits']
            value_params = output['value_params']
            
            print(f"\nRun {i+1}:")
            print(f"  Variable logits: {var_logits}")
            
            # Check which are valid (not -inf)
            valid_mask = jnp.isfinite(var_logits)
            valid_indices = jnp.where(valid_mask)[0]
            
            if len(valid_indices) > 0:
                # Compute softmax probabilities
                finite_logits = jnp.where(valid_mask, var_logits, -1e10)
                probs = jax.nn.softmax(finite_logits)
                
                print(f"  Variable probabilities: {probs}")
                print(f"  Valid variables: {valid_indices}")
                
                # Which has highest probability?
                max_idx = jnp.argmax(jnp.where(valid_mask, probs, 0))
                print(f"  Most likely choice: var_{max_idx} (prob={probs[max_idx]:.3f})")
                
                # Check entropy
                valid_probs = probs[valid_mask]
                entropy = -jnp.sum(valid_probs * jnp.log(valid_probs + 1e-8))
                print(f"  Entropy: {entropy:.3f}")
            
            print(f"  Value params shape: {value_params.shape}")
            print(f"  Value means: {value_params[:, 0]}")
            print(f"  Value log_stds: {value_params[:, 1]}")
        
        # Check if all runs give same output
        all_same_logits = all(jnp.allclose(outputs[0]['variable_logits'], o['variable_logits']) 
                             for o in outputs[1:])
        print(f"\nDeterministic variable selection? {all_same_logits}")
        
    # Analyze parameter structure
    print("\n" + "="*60)
    print("PARAMETER STRUCTURE ANALYSIS")
    print("="*60)
    
    dummy_tensor = create_test_tensor(3, T=10)
    params = policy.init(jax.random.PRNGKey(0), dummy_tensor, 2)
    
    def print_param_tree(params, prefix=""):
        """Print parameter tree structure."""
        for key, value in params.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}/")
                print_param_tree(value, prefix + "  ")
            else:
                print(f"{prefix}{key}: shape={value.shape}, mean={float(jnp.mean(value)):.3f}, std={float(jnp.std(value)):.3f}")
    
    print_param_tree(params)
    
    # Check initialization statistics
    print("\n" + "="*60)
    print("INITIALIZATION STATISTICS")
    print("="*60)
    
    # Get variable head parameters
    var_head_params = params['~']['variable_head']
    print(f"\nVariable head weights:")
    print(f"  Shape: {var_head_params['w'].shape}")
    print(f"  Mean: {float(jnp.mean(var_head_params['w'])):.3f}")
    print(f"  Std: {float(jnp.std(var_head_params['w'])):.3f}")
    print(f"  Min: {float(jnp.min(var_head_params['w'])):.3f}")
    print(f"  Max: {float(jnp.max(var_head_params['w'])):.3f}")
    
    # Check if bias exists
    if 'b' in var_head_params:
        print(f"\nVariable head bias: {var_head_params['b']}")


def test_gradient_flow():
    """Test if gradients flow properly through the policy."""
    print("\n" + "="*80)
    print("TESTING GRADIENT FLOW")
    print("="*80)
    
    policy_fn = create_clean_grpo_policy(hidden_dim=256)
    policy = hk.transform(policy_fn)
    
    # Create test data
    key = jax.random.PRNGKey(42)
    tensor = create_test_tensor(3, T=10)
    target_idx = 2
    
    # Initialize
    params = policy.init(key, tensor, target_idx)
    
    # Define loss function
    def loss_fn(params):
        output = policy.apply(params, key, tensor, target_idx)
        var_logits = output['variable_logits']
        
        # Encourage diversity - penalize if one variable dominates
        valid_mask = jnp.isfinite(var_logits)
        finite_logits = jnp.where(valid_mask, var_logits, -1e10)
        probs = jax.nn.softmax(finite_logits)
        
        # Entropy loss (negative because we want to maximize entropy)
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-8))
        
        return -entropy  # Minimize negative entropy = maximize entropy
    
    # Compute gradients
    loss_value, grads = jax.value_and_grad(loss_fn)(params)
    
    print(f"Loss value: {loss_value:.3f}")
    print("\nGradient norms:")
    
    def print_grad_norms(grads, prefix=""):
        for key, value in grads.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}/")
                print_grad_norms(value, prefix + "  ")
            else:
                grad_norm = float(jnp.linalg.norm(value))
                print(f"{prefix}{key}: norm={grad_norm:.6f}")
    
    print_grad_norms(grads)


if __name__ == "__main__":
    analyze_policy_behavior()
    test_gradient_flow()