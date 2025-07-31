#!/usr/bin/env python3
"""Debug BC surrogate predictions directly."""

import jax.numpy as jnp
from pathlib import Path
import logging

from src.causal_bayes_opt.evaluation.model_interfaces import create_bc_surrogate
from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm
from src.causal_bayes_opt.data_structures.scm import get_target
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_surrogate_directly():
    """Test BC surrogate predictions directly."""
    
    # Load BC surrogate
    surrogate_path = Path('checkpoints/validation/bc_surrogate_final')
    if not surrogate_path.exists():
        logger.error(f"Surrogate checkpoint not found at {surrogate_path}")
        return
    
    logger.info(f"Loading BC surrogate from {surrogate_path}")
    bc_predict_fn, _ = create_bc_surrogate(surrogate_path, allow_updates=False)
    
    # Create test SCM
    scm = create_fork_scm(noise_scale=1.0)
    target = get_target(scm)
    
    # Create buffer with some data
    buffer = ExperienceBuffer()
    samples = sample_from_linear_scm(scm, 100, seed=42)
    for sample in samples:
        buffer.add_observation(sample)
    
    # Convert to tensor
    tensor, var_order = buffer_to_three_channel_tensor(buffer, target, standardize=True)
    
    logger.info(f"\nTensor shape: {tensor.shape}")
    logger.info(f"Variable order: {var_order}")
    logger.info(f"Target: {target}")
    
    # Test 1: Call with 2 arguments (old signature)
    try:
        logger.info("\nTest 1: Calling with 2 arguments (tensor, target)")
        result = bc_predict_fn(tensor, target)
        logger.info(f"Success! Result type: {type(result)}")
        if hasattr(result, 'metadata'):
            logger.info(f"Has metadata: {result.metadata.keys() if hasattr(result.metadata, 'keys') else result.metadata}")
    except Exception as e:
        logger.error(f"Failed with 2 args: {e}")
    
    # Test 2: Call with 3 arguments (new signature)
    try:
        logger.info("\nTest 2: Calling with 3 arguments (tensor, target, variables)")
        result = bc_predict_fn(tensor, target, var_order)
        logger.info(f"Success! Result type: {type(result)}")
        if hasattr(result, 'metadata'):
            logger.info(f"Has metadata: {result.metadata.keys() if hasattr(result.metadata, 'keys') else result.metadata}")
            if 'marginal_parent_probs' in result.metadata:
                marginals = result.metadata['marginal_parent_probs']
                logger.info("\nMarginal probabilities:")
                for var, prob in marginals.items():
                    logger.info(f"  {var}: {prob:.6f}")
    except Exception as e:
        logger.error(f"Failed with 3 args: {e}")
    
    # Test 3: Check what the universal evaluator sees
    logger.info("\nTest 3: Mimicking universal evaluator call")
    try:
        # This is how universal evaluator calls it
        posterior = bc_predict_fn(tensor, target, var_order)
        
        # Extract marginals like universal evaluator does
        if isinstance(posterior, dict):
            if 'marginal_parent_probs' in posterior:
                marginals = dict(posterior['marginal_parent_probs'])
                logger.info("Dictionary format marginals:")
                for var, prob in marginals.items():
                    logger.info(f"  {var}: {prob:.6f}")
        else:
            # ParentSetPosterior object
            if hasattr(posterior, 'metadata') and 'marginal_parent_probs' in posterior.metadata:
                raw_marginals = dict(posterior.metadata['marginal_parent_probs'])
                logger.info("ParentSetPosterior marginals:")
                for var, prob in raw_marginals.items():
                    logger.info(f"  {var}: {prob:.6f}")
                    
    except Exception as e:
        logger.error(f"Universal evaluator style call failed: {e}")

if __name__ == "__main__":
    test_surrogate_directly()