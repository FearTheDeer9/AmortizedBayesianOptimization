#!/usr/bin/env python3
"""
Analyze why GRPO policy weights aren't updating to change variable selection.
"""

import jax
import jax.numpy as jnp
import numpy as np


def analyze_grpo_learning_problem():
    """Analyze the GRPO learning dynamics."""
    
    print("="*80)
    print("WHY GRPO WEIGHTS AREN'T UPDATING VARIABLE SELECTION")
    print("="*80)
    
    print("\n1. THE DETERMINISTIC SAMPLING PROBLEM")
    print("-" * 40)
    print("Current sampling (line 574):")
    print("  selected_var_idx = random.categorical(var_key, var_logits)")
    print("\nWith logits [-0.027, -0.086, -inf], this gives:")
    print("  Probabilities: [0.515, 0.485, 0.0]")
    print("  Selected: ALWAYS variable 0 (X)")
    print("\nBecause categorical() is deterministic when one prob > 0.5!")
    
    print("\n2. NO GRADIENT SIGNAL FOR OTHER VARIABLES")
    print("-" * 40)
    print("In the loss computation (line 815):")
    print("  selected_var = actions['variables'][i]  # Always 0 (X)")
    print("  log_prob = jnp.log(var_probs[selected_var] + 1e-8)")
    print("\nThe gradient only flows through var_probs[0]!")
    print("Variables Y and Z get NO gradient signal because they're never selected.")
    
    print("\n3. THE ADVANTAGE PROBLEM")
    print("-" * 40)
    print("All interventions are on X, so:")
    print("- All rewards are similar (intervening on X with different values)")
    print("- Advantages = rewards - mean(rewards) ≈ small noise")
    print("- No clear signal that X is bad or good")
    print("- Policy loss ≈ 0, so no weight updates")
    
    print("\n4. ENTROPY LOSS CAN'T HELP")
    print("-" * 40)
    print("Entropy = -sum(p * log(p)) = -[0.515*log(0.515) + 0.485*log(0.485)]")
    print("         ≈ 0.693 (already near maximum for 2 variables)")
    print("\nEntropy gradient wants equal probabilities [0.5, 0.5]")
    print("But with deterministic sampling, this still picks X every time!")
    
    print("\n5. THE VICIOUS CYCLE")
    print("-" * 40)
    print("1. Always sample X (deterministic)")
    print("2. Only X gets gradient updates")
    print("3. Y never tried, so no signal it might be better")
    print("4. Weights converge to make X even more likely")
    print("5. Repeat forever")
    
    # Simulate what happens
    print("\n" + "="*80)
    print("SIMULATION: What Happens During Training")
    print("="*80)
    
    # Initial state
    var_logits = jnp.array([-0.027, -0.086, -jnp.inf])
    learning_rate = 3e-4
    
    print("\nInitial state:")
    print(f"  Logits: {var_logits}")
    print(f"  Probs: {jax.nn.softmax(var_logits)}")
    
    # Simulate 10 updates where only X is selected
    for step in range(5):
        # Compute probs
        probs = jax.nn.softmax(var_logits)
        
        # Always selects X (index 0)
        selected = 0
        
        # Fake advantage (small positive because no diversity)
        advantage = 0.1  # Small positive
        
        # Gradient only flows through selected variable
        # Policy wants to increase log_prob of X if advantage > 0
        grad_logits = jnp.zeros_like(var_logits)
        grad_logits = grad_logits.at[selected].set(-advantage * probs[selected] * (1 - probs[selected]))
        
        # Update (simplified)
        var_logits = var_logits - learning_rate * grad_logits * 1000  # Scale for visibility
        
        print(f"\nStep {step + 1}:")
        print(f"  Selected: {selected} (always X)")
        print(f"  Advantage: {advantage}")
        print(f"  New logits: {var_logits}")
        print(f"  New probs: {jax.nn.softmax(var_logits)}")
    
    print("\n" + "="*80)
    print("THE CORE ISSUE")
    print("="*80)
    
    print("\nWithout exploration noise:")
    print("1. Sampling is deterministic (always picks highest prob)")
    print("2. Only the selected variable gets gradients")
    print("3. Unselected variables can't improve")
    print("4. Initial bias becomes permanent")
    
    print("\nEven if Y would give 10x better rewards:")
    print("- It will never be tried")
    print("- So the policy never learns this")
    print("- Stuck on X forever")
    
    print("\n" + "="*80)
    print("SOLUTIONS")
    print("="*80)
    
    print("\n1. ADD EXPLORATION NOISE (Critical!):")
    print("   noisy_logits = var_logits + 0.3 * random.normal(...)")
    print("   This allows trying Y sometimes")
    
    print("\n2. USE GUMBEL-SOFTMAX TRICK:")
    print("   Add Gumbel noise for exploration while maintaining differentiability")
    
    print("\n3. EPSILON-GREEDY:")
    print("   With prob ε, pick random variable instead of argmax")
    
    print("\n4. INITIALIZE WITH EQUAL LOGITS:")
    print("   Start with [0, 0, -inf] so no initial bias")


if __name__ == "__main__":
    analyze_grpo_learning_problem()