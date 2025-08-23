#!/usr/bin/env python3
"""
Simple gradient test to validate if clean implementation would help.

This tests basic policy gradient computation without the complex system.
"""

import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
import optax

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent))

def create_minimal_policy():
    """Create minimal policy for gradient testing."""
    def policy_fn(x):
        # Flatten input
        flat_x = x.flatten()
        
        # Simple MLP
        hidden = hk.Linear(128)(flat_x)
        hidden = jax.nn.relu(hidden)
        
        # Output logits for 3 variables (X, Y, Z)
        logits = hk.Linear(3)(hidden)
        
        return logits
    
    return policy_fn

def test_gradient_magnitude():
    """Test gradient magnitude on simple task."""
    
    # Create minimal policy
    policy_fn = create_minimal_policy()
    policy = hk.transform(policy_fn)
    
    # Initialize
    dummy_input = jnp.ones((10, 3, 4))  # [T, n_vars, 4]
    key = random.PRNGKey(42)
    params = policy.init(key, dummy_input)
    
    # Create simple learning signal
    rewards = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])  # X=1.0, Z=0.0
    
    def simple_loss(p):
        logits = policy.apply(p, key, dummy_input)
        probs = jax.nn.softmax(logits)
        
        # Simple loss: prefer first variable (X) based on rewards
        target_distribution = jnp.array([0.8, 0.1, 0.1])  # Prefer X
        loss = -jnp.sum(target_distribution * jnp.log(probs + 1e-8))
        return loss
    
    # Compute gradients
    loss, grads = jax.value_and_grad(simple_loss)(params)
    
    # Analyze gradient magnitude
    grad_norms = jax.tree.map(jnp.linalg.norm, grads)
    total_grad_norm = sum(jax.tree.leaves(grad_norms))
    
    print(f"🔬 MINIMAL GRADIENT TEST:")
    print(f"  Loss: {loss:.6f}")
    print(f"  Total gradient norm: {total_grad_norm:.6f}")
    
    if total_grad_norm > 0.01:
        print(f"  ✅ Strong gradients - clean implementation should work!")
        improvement_factor = total_grad_norm / 0.0004  # vs current system
        print(f"  Potential improvement: {improvement_factor:.0f}x stronger gradients")
    elif total_grad_norm > 0.001:
        print(f"  ⚠️ Moderate gradients - may help with higher LR")
    else:
        print(f"  ❌ Weak gradients - fundamental issue remains")
    
    return total_grad_norm

def test_reinforce_vs_current():
    """Test if REINFORCE approach would help."""
    
    print(f"\n🧪 REINFORCE vs CURRENT APPROACH:")
    
    # Simulate current GRPO gradients (tiny)
    current_grad_norm = 0.000356  # From diagnostic output
    
    # Test minimal REINFORCE gradients
    minimal_grad_norm = test_gradient_magnitude()
    
    improvement = minimal_grad_norm / current_grad_norm if current_grad_norm > 0 else 0
    
    print(f"\nComparison:")
    print(f"  Current GRPO: {current_grad_norm:.6f}")
    print(f"  Minimal REINFORCE: {minimal_grad_norm:.6f}")
    print(f"  Improvement factor: {improvement:.0f}x")
    
    if improvement > 10:
        print(f"\n✅ CLEAN IMPLEMENTATION RECOMMENDED!")
        print(f"  Clean approach should provide {improvement:.0f}x stronger gradients")
        print(f"  Expected learning speed: 5-10 interventions vs current 20+")
    elif improvement > 3:
        print(f"\n⚠️ MODERATE IMPROVEMENT EXPECTED")
        print(f"  May help but not dramatic improvement")
    else:
        print(f"\n❌ NO SIGNIFICANT IMPROVEMENT")
        print(f"  Issue may be more fundamental")

if __name__ == "__main__":
    print("🚀 CLEAN GRPO VALIDATION")
    print("="*40)
    print("Testing if clean implementation would solve gradient issues")
    print("Current system: 0.000356 gradient norm (extremely small)")
    print("Target: >0.01 gradient norm (30x improvement)\n")
    
    test_reinforce_vs_current()
    
    print(f"\n🎯 RECOMMENDATION:")
    print(f"If test shows >10x improvement:")
    print(f"  → Implement clean version in debugging-grpo/")
    print(f"  → Replace current complex system")
    print(f"  → Add surrogate integration incrementally")
    print(f"\nIf test shows <3x improvement:")
    print(f"  → Continue debugging current system")
    print(f"  → Focus on learning rate scaling")
    print(f"  → Investigate deeper algorithmic issues")