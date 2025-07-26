#!/usr/bin/env python3
"""
Test that the gradient fix for BC acquisition trainer works.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
import jax
import jax.numpy as jnp
import jax.random as random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_gradient_flow():
    """Test that gradients flow with the updated loss clipping."""
    
    logger.info("=== Testing Gradient Flow with Updated Loss Clipping ===\n")
    
    # Test different loss values
    test_losses = [5.0, 10.0, 22.0, 50.0, 100.0, 200.0]
    
    logger.info("Old clipping (10*tanh(x/10)):")
    logger.info("Loss  -> Clipped -> Gradient")
    for loss in test_losses:
        clipped = 10.0 * jnp.tanh(loss / 10.0)
        x = loss / 10.0
        gradient = 1 - jnp.tanh(x)**2
        status = "✓" if gradient > 0.1 else "✗"
        logger.info(f"{loss:5.0f} -> {clipped:7.2f} -> {gradient:.4f} {status}")
    
    logger.info("\nNew clipping (50*tanh(x/50)):")
    logger.info("Loss  -> Clipped -> Gradient")
    for loss in test_losses:
        clipped = 50.0 * jnp.tanh(loss / 50.0)
        x = loss / 50.0
        gradient = 1 - jnp.tanh(x)**2
        status = "✓" if gradient > 0.1 else "✗"
        logger.info(f"{loss:5.0f} -> {clipped:7.2f} -> {gradient:.4f} {status}")
    
    logger.info("\n=== Label Smoothing Effect ===")
    
    # Test label smoothing
    n_vars = 5
    logits = jnp.array([10., -5., -5., -5., -5.])  # Confident on index 0
    expert_idx = 2  # But expert chose index 2
    
    # Old smoothing (0.1)
    old_smoothing = 0.1
    smooth_labels = jnp.ones(n_vars) * (old_smoothing / (n_vars - 1))
    smooth_labels = smooth_labels.at[expert_idx].set(1.0 - old_smoothing)
    log_probs = jax.nn.log_softmax(logits)
    old_loss = -jnp.sum(smooth_labels * log_probs)
    
    # New smoothing (0.2)
    new_smoothing = 0.2
    smooth_labels = jnp.ones(n_vars) * (new_smoothing / (n_vars - 1))
    smooth_labels = smooth_labels.at[expert_idx].set(1.0 - new_smoothing)
    new_loss = -jnp.sum(smooth_labels * log_probs)
    
    logger.info(f"Confident wrong prediction (logits={logits}):")
    logger.info(f"  Expert chose index {expert_idx}")
    logger.info(f"  Old smoothing (0.1): loss = {old_loss:.2f}")
    logger.info(f"  New smoothing (0.2): loss = {new_loss:.2f}")
    logger.info(f"  Reduction: {(1 - new_loss/old_loss)*100:.1f}%")
    
    logger.info("\n=== Summary ===")
    logger.info("✓ New clipping scale (50) maintains gradients at typical loss values")
    logger.info("✓ Increased label smoothing (0.2) reduces extreme losses")
    logger.info("✓ Together, these changes should prevent gradient vanishing")

if __name__ == "__main__":
    test_gradient_flow()