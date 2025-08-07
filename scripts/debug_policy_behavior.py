#!/usr/bin/env python3
"""
Debug why GRPO policy always predicts variable X.
Investigate policy input/output shapes and distributions.
"""

import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.models.unified_model import UnifiedAcquisitionPolicy
from src.causal_bayes_opt.acquisition.policy import PolicyConfig
from src.causal_bayes_opt.acquisition.grpo import convert_to_tensor_native


def create_test_state(n_vars=3, n_samples=100):
    """Create a test state with intervention history."""
    # Create dummy data
    values = np.random.randn(n_samples, n_vars)
    intervened = np.zeros((n_samples, n_vars))
    is_target = np.zeros((n_samples, n_vars))
    
    # Mark some interventions
    for i in range(n_samples):
        if i % 3 == 0:  # Every 3rd sample
            var_idx = i % n_vars  # Rotate through variables
            intervened[i, var_idx] = 1
            values[i, var_idx] = np.random.uniform(-2, 2)
    
    # Mark target (last variable)
    is_target[:, -1] = 1
    
    return {
        'values': values,
        'intervened': intervened,
        'is_target': is_target
    }


def analyze_policy_behavior():
    """Analyze why policy always selects X."""
    print("="*80)
    print("DEBUGGING POLICY BEHAVIOR")
    print("="*80)
    
    # Initialize
    key = jax.random.PRNGKey(42)
    policy_config = PolicyConfig(
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        exploration_noise=0.1,
        variable_selection_temp=1.0
    )
    
    # Create test scenarios
    scenarios = [
        {"name": "3 vars (X,Y,Z)", "n_vars": 3, "target_idx": 2},
        {"name": "4 vars", "n_vars": 4, "target_idx": 3},
        {"name": "5 vars", "n_vars": 5, "target_idx": 4}
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"Variables: {scenario['n_vars']}, Target index: {scenario['target_idx']}")
        print("="*60)
        
        # Create state
        state_data = create_test_state(scenario['n_vars'])
        
        # Convert to tensor
        tensor = convert_to_tensor_native(
            state_data['values'],
            state_data['intervened'],
            state_data['is_target']
        )
        
        print(f"\nTensor shape: {tensor.shape}")
        print(f"Expected: (100, 3, {scenario['n_vars']})")
        
        # Create model
        def policy_fn(tensor_input):
            model = UnifiedAcquisitionPolicy(
                hidden_dim=policy_config.hidden_dim,
                num_layers=policy_config.num_layers,
                num_heads=policy_config.num_heads,
                dropout=policy_config.dropout
            )
            return model(tensor_input, is_training=False)
        
        # Initialize
        init_key, apply_key = jax.random.split(key)
        params = hk.transform(policy_fn).init(init_key, tensor)
        apply_fn = hk.transform(policy_fn).apply
        
        # Get policy output
        output = apply_fn(params, apply_key, tensor)
        
        print(f"\nPolicy output keys: {list(output.keys())}")
        
        # Analyze variable logits
        if 'variable_logits' in output:
            var_logits = output['variable_logits']
            print(f"\nVariable logits shape: {var_logits.shape}")
            print(f"Variable logits: {var_logits}")
            
            # Check if target is masked
            if jnp.isneginf(var_logits[scenario['target_idx']]):
                print("✓ Target variable correctly masked")
            else:
                print("✗ WARNING: Target variable NOT masked!")
            
            # Get probabilities (excluding masked)
            finite_mask = jnp.isfinite(var_logits)
            if jnp.sum(finite_mask) > 0:
                finite_logits = jnp.where(finite_mask, var_logits, -1e10)
                probs = jax.nn.softmax(finite_logits)
                print(f"\nVariable probabilities: {probs}")
                
                # Which variable has highest probability?
                valid_probs = jnp.where(finite_mask, probs, 0)
                max_idx = jnp.argmax(valid_probs)
                print(f"Highest probability variable: index {max_idx} (prob={valid_probs[max_idx]:.3f})")
                
                # Check distribution entropy
                entropy = -jnp.sum(valid_probs * jnp.log(valid_probs + 1e-8))
                print(f"Distribution entropy: {entropy:.3f}")
                if entropy < 0.1:
                    print("⚠️  WARNING: Very low entropy - policy is deterministic!")
        
        # Analyze value parameters
        if 'value_params' in output:
            value_params = output['value_params']
            print(f"\nValue params shape: {value_params.shape}")
            print("Value means and stds:")
            for i in range(scenario['n_vars']):
                mean, log_std = value_params[i]
                std = jnp.exp(log_std)
                print(f"  Var {i}: mean={mean:.3f}, std={std:.3f}")
        
        # Test multiple forward passes
        print("\nTesting consistency across forward passes:")
        logits_list = []
        for i in range(5):
            test_key = jax.random.PRNGKey(i)
            output = apply_fn(params, test_key, tensor)
            logits_list.append(output['variable_logits'])
        
        # Check if outputs are identical
        all_same = all(jnp.allclose(logits_list[0], logits) for logits in logits_list[1:])
        if all_same:
            print("✗ WARNING: Outputs are IDENTICAL across different random keys!")
            print("  This suggests the model is not using randomness properly")
        else:
            print("✓ Outputs vary with different random keys")
    
    # Test with different intervention histories
    print("\n" + "="*60)
    print("TESTING DIFFERENT INTERVENTION HISTORIES")
    print("="*60)
    
    # Scenario 1: All interventions on X
    state_x = create_test_state(3)
    state_x['intervened'][:50, 0] = 1  # First 50 on X
    
    # Scenario 2: All interventions on Y
    state_y = create_test_state(3)
    state_y['intervened'][:50, 1] = 1  # First 50 on Y
    
    # Scenario 3: Mixed interventions
    state_mixed = create_test_state(3)
    for i in range(50):
        state_mixed['intervened'][i, i % 2] = 1  # Alternate X,Y
    
    for state_data, name in [(state_x, "All X"), (state_y, "All Y"), (state_mixed, "Mixed")]:
        print(f"\nHistory: {name}")
        
        tensor = convert_to_tensor_native(
            state_data['values'],
            state_data['intervened'],
            state_data['is_target']
        )
        
        output = apply_fn(params, apply_key, tensor)
        var_logits = output['variable_logits']
        
        finite_mask = jnp.isfinite(var_logits)
        if jnp.sum(finite_mask) > 0:
            finite_logits = jnp.where(finite_mask, var_logits, -1e10)
            probs = jax.nn.softmax(finite_logits)
            valid_probs = jnp.where(finite_mask, probs, 0)
            
            print(f"Variable probabilities: {valid_probs}")
            max_idx = jnp.argmax(valid_probs)
            print(f"Prefers: Variable {max_idx}")


def check_diversity_reward():
    """Check how diversity reward is calculated."""
    print("\n" + "="*80)
    print("CHECKING DIVERSITY REWARD CALCULATION")
    print("="*80)
    
    from src.causal_bayes_opt.acquisition.clean_rewards import compute_reward_components
    
    # Simulate intervention history
    intervention_history = []
    
    # Test 1: Repeated interventions on same variable
    print("\nTest 1: Repeated interventions on X")
    for i in range(5):
        intervention_history.append({'targets': ['X'], 'values': [0.5]})
        
    reward_components = compute_reward_components(
        outcome_value=1.0,
        intervention={'targets': ['X'], 'values': [0.5]},
        intervention_history=intervention_history,
        posterior_before=None,
        posterior_after=None,
        use_surrogate=False,
        reward_weights={'target': 0.8, 'diversity': 0.2, 'exploration': 0.1}
    )
    
    print(f"Diversity reward after 5 X interventions: {reward_components.get('diversity_reward', 0):.3f}")
    
    # Test 2: Alternating interventions
    print("\nTest 2: Alternating interventions X,Y,X,Y")
    intervention_history = []
    for i in range(4):
        var = 'X' if i % 2 == 0 else 'Y'
        intervention_history.append({'targets': [var], 'values': [0.5]})
    
    reward_components = compute_reward_components(
        outcome_value=1.0,
        intervention={'targets': ['Z'], 'values': [0.5]},
        intervention_history=intervention_history,
        posterior_before=None,
        posterior_after=None,
        use_surrogate=False,
        reward_weights={'target': 0.8, 'diversity': 0.2, 'exploration': 0.1}
    )
    
    print(f"Diversity reward for intervening on new var Z: {reward_components.get('diversity_reward', 0):.3f}")


if __name__ == "__main__":
    analyze_policy_behavior()
    check_diversity_reward()