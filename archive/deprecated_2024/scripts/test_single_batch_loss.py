#!/usr/bin/env python3
"""
Quick test to diagnose loss values in a single batch.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as onp

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_cross_entropy_loss():
    """Test cross-entropy loss computation with label smoothing."""
    
    # Simulate a batch of predictions
    n_vars = 5
    batch_size = 4
    
    logger.info("=== Testing Cross-Entropy Loss Computation ===")
    
    # Case 1: Perfect prediction (overconfident)
    logits = jnp.array([-10., -10., 20., -10., -10.])  # Very confident on index 2
    expert_idx = 2
    
    # Apply softmax
    probs = jax.nn.softmax(logits)
    logger.info(f"Case 1 - Overconfident prediction:")
    logger.info(f"  Logits: {logits}")
    logger.info(f"  Probabilities: {probs}")
    logger.info(f"  Max prob: {jnp.max(probs):.6f}")
    
    # Standard cross-entropy
    ce_loss = -jnp.log(probs[expert_idx])
    logger.info(f"  Standard CE loss: {ce_loss:.6f}")
    
    # With label smoothing
    label_smoothing = 0.1
    smooth_labels = jnp.ones(n_vars) * (label_smoothing / (n_vars - 1))
    smooth_labels = smooth_labels.at[expert_idx].set(1.0 - label_smoothing)
    log_probs = jax.nn.log_softmax(logits)
    smoothed_ce_loss = -jnp.sum(smooth_labels * log_probs)
    logger.info(f"  Smoothed CE loss: {smoothed_ce_loss:.6f}")
    
    # Case 2: Wrong prediction with high confidence
    logits = jnp.array([20., -10., -10., -10., -10.])  # Confident on index 0
    expert_idx = 2  # But expert chose index 2
    
    probs = jax.nn.softmax(logits)
    logger.info(f"\nCase 2 - Wrong confident prediction:")
    logger.info(f"  Logits: {logits}")
    logger.info(f"  Probabilities: {probs}")
    logger.info(f"  Expert idx: {expert_idx}, Predicted idx: {jnp.argmax(logits)}")
    
    ce_loss = -jnp.log(probs[expert_idx] + 1e-8)  # Add epsilon to avoid log(0)
    logger.info(f"  Standard CE loss: {ce_loss:.6f}")
    
    smooth_labels = jnp.ones(n_vars) * (label_smoothing / (n_vars - 1))
    smooth_labels = smooth_labels.at[expert_idx].set(1.0 - label_smoothing)
    smoothed_ce_loss = -jnp.sum(smooth_labels * log_probs)
    logger.info(f"  Smoothed CE loss: {smoothed_ce_loss:.6f}")
    
    # Case 3: Uniform prediction
    logits = jnp.zeros(n_vars)
    expert_idx = 2
    
    probs = jax.nn.softmax(logits)
    logger.info(f"\nCase 3 - Uniform prediction:")
    logger.info(f"  Logits: {logits}")
    logger.info(f"  Probabilities: {probs}")
    
    ce_loss = -jnp.log(probs[expert_idx])
    logger.info(f"  Standard CE loss: {ce_loss:.6f}")
    
    # Test combined loss
    logger.info("\n=== Testing Combined Loss ===")
    var_weight = 1.0
    value_weight = 0.5
    
    # High cross-entropy + high MSE
    var_loss = 20.0  # From wrong confident prediction
    value_loss = 4.0  # Large MSE
    combined_loss = var_weight * var_loss + value_weight * value_loss
    logger.info(f"High losses: var={var_loss}, value={value_loss}, combined={combined_loss}")
    
    # Apply tanh clipping
    clipped_loss = 10.0 * jnp.tanh(combined_loss / 10.0)
    logger.info(f"After tanh clipping: {clipped_loss:.6f}")
    
    # Check gradient
    x = combined_loss / 10.0
    tanh_x = jnp.tanh(x)
    grad_tanh = 1 - tanh_x**2
    logger.info(f"Gradient of tanh at x={x:.2f}: {grad_tanh:.6f}")

if __name__ == "__main__":
    test_cross_entropy_loss()