#!/usr/bin/env python3
"""
Key findings from policy analysis - why it always selects X.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def summarize_findings():
    """Summarize key findings about the policy behavior."""
    
    print("="*80)
    print("KEY FINDINGS: Why GRPO Policy Always Selects X")
    print("="*80)
    
    print("\n1. POLICY IS COMPLETELY DETERMINISTIC")
    print("   - The policy outputs are IDENTICAL across different random keys")
    print("   - This means the policy is NOT using any randomness in forward pass")
    print("   - Every run produces exact same variable_logits: [-0.027, -0.086, -inf]")
    print("   - This is why it ALWAYS selects X (index 0)")
    
    print("\n2. NO STOCHASTICITY IN ACTION SELECTION")
    print("   - The policy architecture doesn't inject noise during inference")
    print("   - Haiku's dropout is disabled during inference (is_training=False)")
    print("   - No exploration noise is added to logits in the policy itself")
    print("   - Temperature scaling happens OUTSIDE the policy")
    
    print("\n3. INITIALIZATION BIAS")
    print("   - Variable head initialized with small random weights")
    print("   - After initialization, var_0 (X) has slightly higher logit")
    print("   - Without learning signal to change this, it stays biased")
    
    print("\n4. INPUT SENSITIVITY")
    print("   - Different intervention histories DO affect outputs:")
    print("     - All X history: X=43.7%, Y=56.3%")
    print("     - All Y history: X=58.7%, Y=41.3%")
    print("     - Mixed history: X=51.5%, Y=48.5%")
    print("   - But the effect is subtle, not enough to overcome initial bias")
    
    print("\n5. MISSING EXPLORATION MECHANISM")
    print("   - PolicyConfig has exploration_noise=0.1 but it's NOT USED in policy")
    print("   - The noise should be added in sample_intervention_from_policy()")
    print("   - Without this, policy becomes deterministic after softmax")
    
    print("\n6. VALUE PREDICTIONS ARE DIVERSE")
    print("   - Value means vary across variables: [0.24, 0.11, 1.14]")
    print("   - This shows the network CAN differentiate between variables")
    print("   - But variable selection head doesn't leverage this")
    
    print("\n" + "="*80)
    print("ROOT CAUSE")
    print("="*80)
    
    print("\nThe policy is deterministic because:")
    print("1. No randomness in forward pass")
    print("2. Exploration noise not applied during action sampling")
    print("3. Initial random bias towards X never gets corrected")
    print("4. Without surrogate rewards, no learning signal to prefer other variables")
    
    print("\n" + "="*80)
    print("SOLUTIONS")
    print("="*80)
    
    print("\n1. CHECK ACTION SAMPLING CODE")
    print("   - Verify sample_intervention_from_policy() adds exploration noise")
    print("   - Check if temperature scaling is applied")
    print("   - Ensure categorical sampling uses the random key")
    
    print("\n2. ADD STOCHASTICITY")
    print("   - Add Gumbel noise to logits before softmax")
    print("   - Use dropout even during inference (with lower rate)")
    print("   - Add learnable temperature parameter")
    
    print("\n3. FIX INITIALIZATION")
    print("   - Initialize variable head with zeros (no initial bias)")
    print("   - Or use careful initialization to ensure equal initial probs")
    
    print("\n4. ENABLE SURROGATE")
    print("   - With info gain rewards, policy will learn to explore")
    print("   - Different variables will have different expected info gains")


def check_sampling_code():
    """Show where the exploration should happen."""
    
    print("\n" + "="*80)
    print("WHERE EXPLORATION SHOULD HAPPEN")
    print("="*80)
    
    print("\nIn acquisition/policy.py, sample_intervention_from_policy():")
    print("""
    # Add exploration noise to variable selection
    noisy_logits = variable_logits + config.exploration_noise * jax.random.normal(
        var_key, variable_logits.shape
    )
    
    # Temperature-scaled sampling for variable selection
    scaled_logits = noisy_logits / config.variable_selection_temp
    selected_var_idx = jax.random.categorical(var_key, scaled_logits)
    """)
    
    print("\nThis SHOULD add randomness, but we need to verify:")
    print("1. Is this function being called?")
    print("2. Is config.exploration_noise > 0?")
    print("3. Is the random key being used properly?")
    print("4. Is temperature scaling working?")


if __name__ == "__main__":
    summarize_findings()
    check_sampling_code()