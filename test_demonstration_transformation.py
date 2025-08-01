#!/usr/bin/env poetry run python
"""
Test that demonstration_to_five_channel_tensor correctly transforms data.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import logging
import pickle
import numpy as np
import jax.numpy as jnp

from src.causal_bayes_opt.training.demonstration_to_tensor import (
    demonstration_to_five_channel_tensor,
    _compute_marginal_parent_probabilities
)
from src.causal_bayes_opt.data_structures.scm import get_variables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_transformation():
    """Test the transformation from demonstration to 5-channel tensor."""
    
    # Load a real demonstration
    demo_path = Path("expert_demonstrations/raw/raw_demonstrations")
    if not demo_path.exists():
        logger.error(f"Demo path not found: {demo_path}")
        return
    
    demo_files = list(demo_path.glob("*.pkl"))
    if not demo_files:
        logger.error("No demonstration files found")
        return
    
    # Load first demonstration
    with open(demo_files[0], 'rb') as f:
        demo_batch = pickle.load(f)
    
    if hasattr(demo_batch, 'demonstrations'):
        demo = demo_batch.demonstrations[0]
    else:
        demo = demo_batch
    
    logger.info("=== Testing Demonstration Transformation ===\n")
    
    # Test transformation
    input_tensors, intervention_labels, metadata = demonstration_to_five_channel_tensor(
        demo, max_trajectory_length=10
    )
    
    logger.info(f"Created {len(input_tensors)} training examples")
    logger.info(f"Metadata: {metadata}")
    
    if input_tensors:
        # Check first tensor
        first_tensor = input_tensors[0]
        logger.info(f"\nFirst tensor shape: {first_tensor.shape}")
        logger.info(f"Expected shape: [T, n_vars, 5]")
        
        # Verify 5 channels
        T, n_vars, n_channels = first_tensor.shape
        assert n_channels == 5, f"Expected 5 channels, got {n_channels}"
        
        # Check each channel
        logger.info("\n=== Channel Analysis ===")
        
        # Channel 0: Values (should be standardized)
        values = first_tensor[:, :, 0]
        logger.info(f"Channel 0 (values): mean={jnp.mean(values):.3f}, std={jnp.std(values):.3f}")
        assert jnp.abs(jnp.mean(values)) < 0.5, "Values should be approximately standardized"
        
        # Channel 1: Target indicator
        target_channel = first_tensor[:, :, 1]
        target_idx = metadata['target_idx']
        logger.info(f"Channel 1 (target): sum per timestep = {jnp.sum(target_channel, axis=1)}")
        assert jnp.all(target_channel[:, target_idx] == 1.0), "Target indicator incorrect"
        
        # Channel 2: Intervention indicator
        intervention_channel = first_tensor[:, :, 2]
        logger.info(f"Channel 2 (interventions): total interventions = {jnp.sum(intervention_channel)}")
        
        # Channel 3: Marginal parent probabilities
        marginal_channel = first_tensor[:, :, 3]
        logger.info(f"Channel 3 (marginals): shape = {marginal_channel.shape}")
        
        # Should be constant over time
        assert jnp.allclose(marginal_channel[0], marginal_channel[-1]), "Marginals should be constant over time"
        
        # Check marginal values
        marginals_at_t0 = marginal_channel[0]
        logger.info(f"Marginal probabilities:")
        for i, var in enumerate(metadata['variables']):
            logger.info(f"  {var}: {marginals_at_t0[i]:.3f}")
        
        # Channel 4: Recency
        recency_channel = first_tensor[:, :, 4]
        logger.info(f"Channel 4 (recency): range = [{jnp.min(recency_channel):.3f}, {jnp.max(recency_channel):.3f}]")
        assert jnp.all(recency_channel >= 0) and jnp.all(recency_channel <= 1), "Recency should be in [0,1]"
        
        # Check intervention labels
        logger.info(f"\n=== Intervention Labels ===")
        first_label = intervention_labels[0]
        logger.info(f"First intervention: {first_label}")
        assert 'targets' in first_label
        assert 'values' in first_label
        assert isinstance(first_label['targets'], frozenset)
        
        # Verify marginals match metadata
        logger.info(f"\n=== Marginal Verification ===")
        true_marginals = metadata['marginal_probs']
        for i, var in enumerate(metadata['variables']):
            if var != metadata['target_variable']:
                extracted = float(marginals_at_t0[i])
                # true_marginals is an array, not a dict
                expected = float(true_marginals[i])
                logger.info(f"{var}: extracted={extracted:.3f}, expected={expected:.3f}")
                assert abs(extracted - expected) < 1e-5, f"Marginal mismatch for {var}"
        
        logger.info("\n✅ All transformation tests passed!")
        
    else:
        logger.error("No tensors created - check demonstration format")


def test_marginal_computation():
    """Test marginal probability computation."""
    logger.info("\n=== Testing Marginal Computation ===")
    
    # Test case 1: Simple posterior
    posterior_dist = {
        frozenset(['X', 'Z']): 0.7,
        frozenset(['X']): 0.2,
        frozenset(['Z']): 0.1
    }
    variables = ['X', 'Y', 'Z']
    target = 'Y'
    
    marginals = _compute_marginal_parent_probabilities(posterior_dist, variables, target)
    
    logger.info(f"Posterior: {posterior_dist}")
    logger.info(f"Computed marginals: {marginals}")
    
    # X appears in sets with total prob 0.7 + 0.2 = 0.9
    assert abs(float(marginals[0]) - 0.9) < 1e-6, f"X marginal should be 0.9, got {marginals[0]}"
    # Y is target, should be 0
    assert float(marginals[1]) == 0.0, f"Target marginal should be 0, got {marginals[1]}"
    # Z appears in sets with total prob 0.7 + 0.1 = 0.8
    assert abs(float(marginals[2]) - 0.8) < 1e-6, f"Z marginal should be 0.8, got {marginals[2]}"
    
    logger.info("✅ Marginal computation test passed!")


if __name__ == "__main__":
    test_marginal_computation()
    test_transformation()