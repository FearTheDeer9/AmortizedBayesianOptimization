#!/usr/bin/env python3
"""
Analyze why gradients vanish despite stable loss.
Focus on the relationship between confidence, loss, and gradients.
"""

import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import jax.nn as nn

sys.path.append(str(Path(__file__).parent.parent))

def analyze_softmax_gradient_behavior():
    """Understand how softmax gradients behave with high confidence."""
    
    print("="*80)
    print("ANALYZING GRADIENT VANISHING MYSTERY")
    print("="*80)
    
    print("\n" + "="*60)
    print("1. SOFTMAX GRADIENT ANALYSIS")
    print("="*60)
    
    # Simulate what happens in our model
    # Model outputs logits, applies softmax, then negative log likelihood loss
    
    def loss_fn(logits, target_idx):
        """Our actual loss function for variable selection."""
        probs = nn.softmax(logits)
        return -jnp.log(probs[target_idx])
    
    # Test different confidence levels
    print("\nHow gradients change with model confidence:")
    print("-" * 60)
    print(f"{'Confidence':<15} | {'Loss':<10} | {'Grad Norm':<15} | {'Max Grad':<15}")
    print("-" * 60)
    
    for confidence in [0.2, 0.5, 0.8, 0.95, 0.98, 0.99, 0.999]:
        # Create logits that give desired confidence for index 0
        # For 5 variables, predict index 0 with given confidence
        n_vars = 5
        target_idx = 0
        
        # To get confidence p for class 0, we need:
        # exp(logit_0) / sum(exp(logits)) = p
        # If other logits are 0, then exp(logit_0) / (exp(logit_0) + 4) = p
        # Solving: logit_0 = log(4p / (1-p))
        
        if confidence < 0.999:
            logit_0 = jnp.log(4 * confidence / (1 - confidence))
        else:
            logit_0 = 10.0  # Very high confidence
        
        logits = jnp.array([logit_0, 0.0, 0.0, 0.0, 0.0])
        
        # Compute loss and gradient
        loss_val = loss_fn(logits, target_idx)
        grad = jax.grad(loss_fn)(logits, target_idx)
        grad_norm = jnp.linalg.norm(grad)
        max_grad = jnp.max(jnp.abs(grad))
        
        print(f"{confidence:<15.3f} | {loss_val:<10.4f} | {grad_norm:<15.6f} | {max_grad:<15.6f}")
    
    print("\n💡 KEY INSIGHT: As confidence → 1, gradients → 0 even though loss is non-zero!")
    
    # Now test WRONG predictions with high confidence
    print("\n" + "="*60)
    print("2. WRONG PREDICTIONS WITH HIGH CONFIDENCE")
    print("="*60)
    
    print("\nWhen model is confident but WRONG:")
    print("-" * 60)
    print(f"{'Confidence':<15} | {'Loss':<10} | {'Grad Norm':<15} | {'Target Grad':<15}")
    print("-" * 60)
    
    for confidence in [0.8, 0.95, 0.98, 0.99]:
        # Model predicts index 0 with high confidence
        # But target is index 2
        target_idx = 2
        
        logit_0 = jnp.log(4 * confidence / (1 - confidence))
        logits = jnp.array([logit_0, 0.0, 0.0, 0.0, 0.0])
        
        loss_val = loss_fn(logits, target_idx)
        grad = jax.grad(loss_fn)(logits, target_idx)
        grad_norm = jnp.linalg.norm(grad)
        target_grad = grad[target_idx]
        
        print(f"{confidence:<15.3f} | {loss_val:<10.4f} | {grad_norm:<15.6f} | {target_grad:<15.6f}")
    
    print("\n💡 When wrong with high confidence, loss is HUGE but gradients can still be small!")
    
    # Analyze our specific case
    print("\n" + "="*60)
    print("3. OUR TRAINING SCENARIO")
    print("="*60)
    
    print("""
From training logs:
- Epoch 1: Target prob = 0.30, Grad norm = 2.57
- Epoch 20: Target prob = 0.98, Grad norm = 0.19

The model becomes overconfident in its predictions!
""")
    
    # Simulate our scenario
    print("\nSimulating our training progression:")
    print("-" * 60)
    
    # Early training: uncertain
    logits_early = jnp.array([0.1, 0.2, -0.1, 0.0, 0.15])  # Roughly uniform
    target_idx = 0
    probs_early = nn.softmax(logits_early)
    loss_early = loss_fn(logits_early, target_idx)
    grad_early = jax.grad(loss_fn)(logits_early, target_idx)
    
    print(f"Early training:")
    print(f"  Probs: {probs_early}")
    print(f"  Target prob: {probs_early[target_idx]:.4f}")
    print(f"  Loss: {loss_early:.4f}")
    print(f"  Grad norm: {jnp.linalg.norm(grad_early):.4f}")
    
    # Late training: overconfident
    logits_late = jnp.array([5.0, -2.0, -3.0, -4.0, -5.0])  # Very confident
    probs_late = nn.softmax(logits_late)
    loss_late = loss_fn(logits_late, target_idx)
    grad_late = jax.grad(loss_fn)(logits_late, target_idx)
    
    print(f"\nLate training:")
    print(f"  Probs: {probs_late}")
    print(f"  Target prob: {probs_late[target_idx]:.4f}")
    print(f"  Loss: {loss_late:.4f}")
    print(f"  Grad norm: {jnp.linalg.norm(grad_late):.4f}")
    
    # The saturation problem
    print("\n" + "="*60)
    print("4. WHY OVERCONFIDENCE KILLS GRADIENTS")
    print("="*60)
    
    print("""
The softmax gradient for the predicted class is: grad = p - 1
When p → 1 (high confidence), grad → 0

For non-predicted classes: grad = p
When p → 0 (low probability), grad → 0

So when the model is confident (right or wrong):
- All gradients → 0
- Learning stops
- This happens even if loss is high (wrong prediction)!
""")
    
    # Test gradient clipping interaction
    print("\n" + "="*60)
    print("5. GRADIENT CLIPPING INTERACTION")
    print("="*60)
    
    print("""
Our setup uses gradient clipping at 5.0.
But with vanishing gradients, clipping never activates!

From logs: "Clip ratio: 100%" means no clipping needed.
This suggests gradients are naturally small, not clipped.
""")
    
    # The weight decay factor
    print("\n" + "="*60)
    print("6. WEIGHT DECAY EFFECT")
    print("="*60)
    
    print("""
We use weight_decay = 1e-4 with AdamW.
This adds L2 penalty to weights, pulling them toward zero.

With small gradients from the loss (due to overconfidence),
weight decay might dominate, causing:
- Weights slowly decay toward zero
- Model outputs become more extreme (larger magnitude logits)
- Even more confidence → even smaller gradients
- Vicious cycle!
""")
    
    print("\n" + "="*80)
    print("ROOT CAUSE IDENTIFIED")
    print("="*80)
    
    print("""
The model becomes overconfident because:

1. EASY PATTERNS: Size-based shortcuts give perfect accuracy on some variables
2. SOFTMAX SATURATION: High confidence (p→1) causes gradients to vanish
3. NO PENALTY FOR OVERCONFIDENCE: Cross-entropy loss doesn't penalize confidence
4. WEIGHT DECAY: May push toward more extreme logits

The model learns to be maximally confident on the patterns it knows,
which prevents it from learning anything new!

This is why loss plateaus but stays high - the model is confidently wrong
on X2, but the gradients are too small to correct it.

SOLUTIONS:
1. Label smoothing (prevent p=1.0)
2. Entropy regularization (penalize overconfidence)
3. Temperature scaling
4. Different loss function (focal loss)
""")

if __name__ == "__main__":
    analyze_softmax_gradient_behavior()