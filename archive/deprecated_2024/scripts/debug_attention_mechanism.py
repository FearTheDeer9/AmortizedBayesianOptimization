#!/usr/bin/env python3
"""
Debug why the attention mechanism produces uniform outputs.
"""

import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
from pathlib import Path

# Set up paths
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))


def test_attention_mechanism():
    """Test the parent attention mechanism in isolation."""
    print("Testing Parent Attention Mechanism")
    print("=" * 60)
    
    # Create simple test case
    hidden_dim = 4
    n_vars = 3
    
    def attention_test(target_emb, all_embs):
        """Simplified attention mechanism."""
        # Multi-head attention
        attn = hk.MultiHeadAttention(
            num_heads=1,
            key_size=4,
            w_init_scale=2.0,
            model_size=hidden_dim,
        )
        
        # Expand target to match all variables
        query = jnp.tile(target_emb[None, :], (n_vars, 1))
        
        # Compute attention
        attended = attn(query, all_embs, all_embs)
        
        # Project to scores
        scores = hk.Linear(1)(attended).squeeze(-1)
        
        return scores, attended
    
    # Transform with Haiku
    model = hk.transform(attention_test)
    
    # Initialize
    key = random.PRNGKey(42)
    
    # Test Case 1: Distinct embeddings
    print("\nTest 1: Distinct embeddings")
    target_emb = jnp.array([1.0, 0.0, 0.0, 0.0])
    all_embs = jnp.array([
        [1.0, 0.0, 0.0, 0.0],  # Same as target
        [0.0, 1.0, 0.0, 0.0],  # Different
        [0.0, 0.0, 1.0, 0.0],  # Different
    ])
    
    params = model.init(key, target_emb, all_embs)
    scores, attended = model.apply(params, key, target_emb, all_embs)
    
    print(f"Target embedding: {target_emb}")
    print(f"All embeddings:\n{all_embs}")
    print(f"Attention scores: {scores}")
    print(f"Softmax probs: {jax.nn.softmax(scores)}")
    
    # Test Case 2: All same embeddings
    print("\n\nTest 2: All same embeddings")
    same_emb = jnp.array([0.5, 0.5, 0.5, 0.5])
    all_same = jnp.tile(same_emb[None, :], (n_vars, 1))
    
    scores2, attended2 = model.apply(params, key, same_emb, all_same)
    
    print(f"All embeddings (same):\n{all_same}")
    print(f"Attention scores: {scores2}")
    print(f"Softmax probs: {jax.nn.softmax(scores2)}")
    
    # Test Case 3: Check what happens with node encoder output
    print("\n\nTest 3: Simulating node encoder aggregation issue")
    
    # Simulate the mean aggregation effect
    data_samples = random.normal(key, (10, n_vars, hidden_dim))
    
    # Original aggregation (loses information)
    mean_aggregated = jnp.mean(data_samples, axis=0)  # [n_vars, hidden_dim]
    print(f"\nMean aggregated embeddings:\n{mean_aggregated}")
    print(f"Standard deviation across variables: {jnp.std(mean_aggregated, axis=0)}")
    
    # Check if all embeddings become similar
    pairwise_distances = jnp.sum((mean_aggregated[:, None, :] - mean_aggregated[None, :, :])**2, axis=2)
    print(f"\nPairwise distances between variable embeddings:\n{pairwise_distances}")
    
    # Apply attention with mean-aggregated embeddings
    target_idx = 2
    target_emb_agg = mean_aggregated[target_idx]
    scores_agg, _ = model.apply(params, key, target_emb_agg, mean_aggregated)
    probs_agg = jax.nn.softmax(scores_agg)
    
    print(f"\nWith mean aggregation:")
    print(f"Attention scores: {scores_agg}")
    print(f"Softmax probs: {probs_agg}")
    
    # The issue: When embeddings are similar (due to averaging), 
    # small differences in scores lead to uniform softmax probabilities
    print("\n\nAnalysis:")
    print("1. Mean aggregation makes all node embeddings very similar")
    print("2. Similar embeddings → similar attention scores")
    print("3. Similar scores → nearly uniform softmax probabilities")
    print("4. This explains why we always get [0.333, 0.333, 0.333, 0]")


def test_why_cross_sample_attention_fails():
    """Test why even cross-sample attention doesn't help."""
    print("\n\nWhy Cross-Sample Attention Doesn't Help")
    print("=" * 60)
    
    # The fundamental issue
    print("\nThe Problem:")
    print("1. We process each (sample, variable) pair independently")
    print("2. Then aggregate across samples for each variable")
    print("3. This loses all cross-variable relationships!")
    
    print("\nExample:")
    print("Sample 1: X1=1, X2=1  (correlated)")
    print("Sample 2: X1=0, X2=0  (correlated)")
    print("Sample 3: X1=1, X2=0  (anti-correlated)")
    
    print("\nAfter independent processing and averaging:")
    print("X1 embedding: average of embeddings for [1, 0, 1]")
    print("X2 embedding: average of embeddings for [1, 0, 0]")
    print("The correlation information is completely lost!")
    
    print("\nThe Fix Needs To:")
    print("1. Process relationships BETWEEN variables")
    print("2. Not just patterns WITHIN each variable")
    print("3. Preserve correlation structure through aggregation")


if __name__ == "__main__":
    test_attention_mechanism()
    test_why_cross_sample_attention_fails()