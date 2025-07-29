#!/usr/bin/env python3
"""
Test that Haiku parameter compatibility is maintained across training and loading.

This test verifies that the shared policy factory prevents the dreaded
"Unable to retrieve parameter" errors that waste training time.
"""

import tempfile
from pathlib import Path
import pickle
import logging

import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.policies.clean_policy_factory import (
    create_clean_grpo_policy, verify_parameter_compatibility
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_parameter_consistency():
    """Test that parameters saved in one context can be loaded in another."""
    logger.info("=== Testing Haiku Parameter Consistency ===")
    
    # Create policy using shared factory
    policy_fn = create_clean_grpo_policy(hidden_dim=64)
    policy = hk.transform(policy_fn)
    
    # Initialize in "training" context
    logger.info("\n1. Initializing policy in training context...")
    dummy_input = jnp.zeros((10, 5, 3))
    rng = random.PRNGKey(42)
    
    params_from_training = policy.init(rng, dummy_input, target_idx=0)
    logger.info(f"   Created parameters with keys: {list(params_from_training.keys())}")
    
    # Save parameters
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        pickle.dump(params_from_training, f)
        param_file = f.name
    
    logger.info(f"\n2. Saved parameters to {param_file}")
    
    # Load in "inference" context (different function instance)
    logger.info("\n3. Loading parameters in inference context...")
    
    # Create new policy instance (simulating loading in different file)
    new_policy_fn = create_clean_grpo_policy(hidden_dim=64)
    new_policy = hk.transform(new_policy_fn)
    
    # Load saved parameters
    with open(param_file, 'rb') as f:
        loaded_params = pickle.load(f)
    
    # Verify compatibility
    logger.info("\n4. Verifying parameter compatibility...")
    is_compatible = verify_parameter_compatibility(
        loaded_params, new_policy, dummy_input
    )
    
    if not is_compatible:
        logger.error("   FAILED: Parameters are not compatible!")
        return False
    
    # Try to use loaded parameters
    logger.info("\n5. Testing inference with loaded parameters...")
    try:
        output = new_policy.apply(loaded_params, rng, dummy_input, 0)
        logger.info(f"   SUCCESS: Got output with keys: {list(output.keys())}")
        logger.info(f"   Variable logits shape: {output['variable_logits'].shape}")
        logger.info(f"   Value params shape: {output['value_params'].shape}")
        return True
    except Exception as e:
        logger.error(f"   FAILED: {e}")
        return False


def test_module_path_consistency():
    """Test that module paths are consistent across contexts."""
    logger.info("\n=== Testing Module Path Consistency ===")
    
    # Create two "different" policies (but using same factory)
    policy1 = hk.transform(create_clean_grpo_policy(hidden_dim=32))
    policy2 = hk.transform(create_clean_grpo_policy(hidden_dim=32))
    
    # Initialize both
    dummy_input = jnp.zeros((5, 3, 3))
    rng = random.PRNGKey(0)
    
    params1 = policy1.init(rng, dummy_input, 0)
    params2 = policy2.init(rng, dummy_input, 0)
    
    # Extract parameter paths
    def get_param_paths(params, prefix=""):
        paths = []
        for k, v in params.items():
            path = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict):
                paths.extend(get_param_paths(v, path))
            else:
                paths.append(path)
        return sorted(paths)
    
    paths1 = get_param_paths(params1)
    paths2 = get_param_paths(params2)
    
    logger.info(f"\nPolicy 1 parameter paths: {paths1}")
    logger.info(f"\nPolicy 2 parameter paths: {paths2}")
    
    if paths1 == paths2:
        logger.info("\n✓ Module paths are consistent!")
        
        # Test cross-loading
        try:
            # Use params from policy1 in policy2
            output = policy2.apply(params1, rng, dummy_input, 0)
            logger.info("✓ Successfully used params from policy1 in policy2!")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to cross-load parameters: {e}")
            return False
    else:
        logger.error("\n✗ Module paths differ!")
        return False


def test_different_hidden_dims():
    """Test that different hidden dimensions are properly detected."""
    logger.info("\n=== Testing Different Hidden Dimensions ===")
    
    # Create policies with different hidden dims
    policy_64 = hk.transform(create_clean_grpo_policy(hidden_dim=64))
    policy_128 = hk.transform(create_clean_grpo_policy(hidden_dim=128))
    
    # Initialize
    dummy_input = jnp.zeros((10, 5, 3))
    rng = random.PRNGKey(0)
    
    params_64 = policy_64.init(rng, dummy_input, 0)
    params_128 = policy_128.init(rng, dummy_input, 0)
    
    # Try to verify compatibility (should fail)
    logger.info("\nVerifying incompatible dimensions are detected...")
    is_compatible = verify_parameter_compatibility(
        params_64, policy_128, dummy_input
    )
    
    if not is_compatible:
        logger.info("✓ Correctly detected incompatible parameters!")
        return True
    else:
        logger.error("✗ Failed to detect incompatible parameters!")
        return False


def main():
    """Run all tests."""
    logger.info("Testing Haiku Parameter Compatibility")
    logger.info("=" * 50)
    
    tests = [
        ("Parameter Consistency", test_parameter_consistency),
        ("Module Path Consistency", test_module_path_consistency),
        ("Different Dimensions Detection", test_different_hidden_dims)
    ]
    
    results = []
    for name, test_fn in tests:
        logger.info(f"\nRunning: {name}")
        logger.info("-" * 40)
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            logger.error(f"Test crashed: {e}")
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("Test Summary:")
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        logger.info(f"  {name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        logger.info("\n✓ All tests passed! The shared policy factory works correctly.")
    else:
        logger.error("\n✗ Some tests failed. Check the implementation.")
    
    return all_passed


if __name__ == "__main__":
    main()