#!/usr/bin/env python3
"""
Test permutation equivariance of the encoder implementations.

This ensures that permuting input variables results in correspondingly
permuted outputs, maintaining the key property of treating all variables equally.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from src.causal_bayes_opt.avici_integration.continuous.configurable_model import ConfigurableContinuousParentSetPredictionModel

def test_encoder_permutation_equivariance(encoder_type: str = "node_feature"):
    """Test that encoder maintains permutation equivariance."""
    print(f"\nTesting permutation equivariance for {encoder_type} encoder")
    print("="*60)
    
    # Create test data
    key = jax.random.PRNGKey(42)
    N = 20  # number of samples
    d = 5   # number of variables
    
    # Create random data [N, d, 3]
    data_key, perm_key = jax.random.split(key)
    data = jax.random.normal(data_key, (N, d, 3))
    
    # Create a random permutation
    perm = jax.random.permutation(perm_key, d)
    print(f"Permutation: {perm}")
    
    # Permute the data
    data_permuted = data[:, perm, :]
    
    # Create model function
    def model_fn(data, target_idx, is_training=False):
        model = ConfigurableContinuousParentSetPredictionModel(
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            key_size=32,
            dropout=0.0,  # No dropout for testing
            encoder_type=encoder_type,
            attention_type='pairwise' if encoder_type == 'node_feature' else 'original'
        )
        return model(data, target_idx, is_training)
    
    # Transform to Haiku
    net = hk.transform(model_fn)
    
    # Initialize parameters
    params = net.init(key, data, 0, False)
    
    # Test for each target variable
    all_tests_passed = True
    
    for target_idx in range(d):
        # Find where this target moved to after permutation
        target_idx_permuted = int(np.where(perm == target_idx)[0][0])
        
        print(f"\nTesting target variable {target_idx} -> {target_idx_permuted}")
        
        # Get outputs for original data
        output1 = net.apply(params, key, data, target_idx, False)
        probs1 = output1['parent_probabilities']
        embeddings1 = output1['node_embeddings']
        
        # Get outputs for permuted data
        output2 = net.apply(params, key, data_permuted, target_idx_permuted, False)
        probs2 = output2['parent_probabilities']
        embeddings2 = output2['node_embeddings']
        
        # Check embeddings are permuted correctly
        embeddings1_permuted = embeddings1[perm]
        embedding_diff = jnp.max(jnp.abs(embeddings1_permuted - embeddings2))
        print(f"  Max embedding difference: {embedding_diff:.6f}")
        
        # Check probabilities are permuted correctly
        probs1_permuted = probs1[perm]
        prob_diff = jnp.max(jnp.abs(probs1_permuted - probs2))
        print(f"  Max probability difference: {prob_diff:.6f}")
        
        # Test passes if differences are very small (numerical precision)
        test_passed = embedding_diff < 1e-5 and prob_diff < 1e-5
        print(f"  Test {'PASSED' if test_passed else 'FAILED'}")
        
        if not test_passed:
            all_tests_passed = False
            print(f"  Original probs: {probs1}")
            print(f"  Permuted probs: {probs2}")
            print(f"  Expected probs: {probs1_permuted}")
    
    print(f"\n{'='*60}")
    print(f"Overall result: {'ALL TESTS PASSED ✓' if all_tests_passed else 'SOME TESTS FAILED ✗'}")
    print(f"{'='*60}")
    
    return all_tests_passed


def test_simple_permutation():
    """Test with a simple 2-variable permutation."""
    print("\nSimple 2-variable permutation test")
    print("-"*40)
    
    # Create simple test data
    key = jax.random.PRNGKey(123)
    N = 10
    d = 2
    
    # Create distinct data for each variable
    data = jnp.zeros((N, d, 3))
    data = data.at[:, 0, 0].set(jnp.linspace(0, 1, N))  # Variable 0: linear
    data = data.at[:, 1, 0].set(jnp.sin(jnp.linspace(0, 2*jnp.pi, N)))  # Variable 1: sine
    data = data.at[:, :, 2].set(1.0)  # All observed
    
    # Swap variables
    data_swapped = data[:, [1, 0], :]
    
    # Create encoder
    from src.causal_bayes_opt.avici_integration.continuous.node_feature_encoder import NodeFeatureEncoder
    
    def encode_fn(data):
        encoder = NodeFeatureEncoder(hidden_dim=64, num_layers=2)
        return encoder(data, is_training=False)
    
    net = hk.transform(encode_fn)
    params = net.init(key, data)
    
    # Get embeddings
    embeddings1 = net.apply(params, key, data)
    embeddings2 = net.apply(params, key, data_swapped)
    
    # Check if swapped correctly
    diff = jnp.max(jnp.abs(embeddings1[0] - embeddings2[1])) + jnp.max(jnp.abs(embeddings1[1] - embeddings2[0]))
    
    print(f"Embedding shape: {embeddings1.shape}")
    print(f"Max difference after swap: {diff:.6f}")
    print(f"Test {'PASSED' if diff < 1e-5 else 'FAILED'}")
    
    return diff < 1e-5


def main():
    """Run all permutation equivariance tests."""
    print("Testing Permutation Equivariance of Encoder Architectures")
    print("="*60)
    
    # Test simple case first
    simple_passed = test_simple_permutation()
    
    # Test each encoder type
    encoder_types = ['node_feature', 'node', 'simple']
    results = {}
    
    for encoder_type in encoder_types:
        try:
            results[encoder_type] = test_encoder_permutation_equivariance(encoder_type)
        except Exception as e:
            print(f"\nError testing {encoder_type}: {e}")
            results[encoder_type] = False
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Simple test: {'PASSED' if simple_passed else 'FAILED'}")
    for encoder_type, passed in results.items():
        print(f"{encoder_type} encoder: {'PASSED' if passed else 'FAILED'}")
    
    all_passed = simple_passed and all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    
    return all_passed


if __name__ == "__main__":
    main()