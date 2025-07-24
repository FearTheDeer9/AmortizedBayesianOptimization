#!/usr/bin/env python3
"""
Debug why the architecture produces uniform outputs even with random initialization.
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

from src.causal_bayes_opt.avici_integration.continuous.model import (
    ContinuousParentSetPredictionModel,
    NodeEncoder,
    ParentAttentionLayer
)


def debug_node_encoder():
    """Debug the NodeEncoder component."""
    print("\n" + "="*60)
    print("DEBUGGING NODE ENCODER")
    print("="*60)
    
    # Create simple test data
    key = random.PRNGKey(42)
    n_samples = 20
    n_vars = 4
    
    # Create data with clear differences between variables
    data = jnp.zeros((n_samples, n_vars, 3))
    # X0: constant 0
    data = data.at[:, 0, 0].set(0.0)
    # X1: constant 1
    data = data.at[:, 1, 0].set(1.0)
    # X2: constant -1
    data = data.at[:, 2, 0].set(-1.0)
    # X3: varying values
    data = data.at[:, 3, 0].set(jnp.linspace(-2, 2, n_samples))
    # All observed
    data = data.at[:, :, 2].set(1.0)
    
    print("Input data characteristics:")
    for i in range(n_vars):
        values = data[:, i, 0]
        print(f"X{i}: mean={jnp.mean(values):.3f}, std={jnp.std(values):.3f}, "
              f"min={jnp.min(values):.3f}, max={jnp.max(values):.3f}")
    
    # Create encoder
    def encoder_fn(data):
        encoder = NodeEncoder(hidden_dim=64, num_layers=2)
        return encoder(data)
    
    # Transform and initialize
    encoder = hk.without_apply_rng(hk.transform(encoder_fn))
    params = encoder.init(key, data)
    
    # Get embeddings
    embeddings = encoder.apply(params, data)
    print(f"\nOutput embeddings shape: {embeddings.shape}")
    
    # Check if embeddings are different for different variables
    print("\nEmbedding differences:")
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            diff = jnp.linalg.norm(embeddings[i] - embeddings[j])
            print(f"||embed(X{i}) - embed(X{j})|| = {diff:.6f}")
    
    # Check embedding statistics
    print("\nEmbedding statistics:")
    print(f"Mean norm: {jnp.mean(jnp.linalg.norm(embeddings, axis=1)):.6f}")
    print(f"Std of norms: {jnp.std(jnp.linalg.norm(embeddings, axis=1)):.6f}")
    
    # Check if all embeddings are similar
    all_similar = True
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            if jnp.linalg.norm(embeddings[i] - embeddings[j]) > 0.1:
                all_similar = False
                break
    
    if all_similar:
        print("\n❌ PROBLEM: All node embeddings are nearly identical!")
        print("   This explains why parent probabilities are uniform.")
    else:
        print("\n✅ Node embeddings are distinct.")
    
    return embeddings, all_similar


def debug_attention_computation():
    """Debug the attention mechanism."""
    print("\n" + "="*60)
    print("DEBUGGING ATTENTION MECHANISM")
    print("="*60)
    
    key = random.PRNGKey(42)
    hidden_dim = 64
    n_vars = 4
    
    # Create distinct embeddings manually
    embeddings = jnp.array([
        jnp.ones(hidden_dim) * 0.0,    # X0
        jnp.ones(hidden_dim) * 1.0,    # X1
        jnp.ones(hidden_dim) * -1.0,   # X2
        jnp.ones(hidden_dim) * 2.0,    # X3
    ])
    
    print("Test embeddings (all elements same within variable):")
    for i in range(n_vars):
        print(f"X{i}: all elements = {embeddings[i, 0]:.1f}")
    
    # Test attention for different targets
    def attention_fn(query, key_value):
        attention = ParentAttentionLayer(
            hidden_dim=hidden_dim,
            num_heads=4,
            key_size=32
        )
        return attention(query, key_value)
    
    # Transform and initialize
    attention = hk.without_apply_rng(hk.transform(attention_fn))
    params = attention.init(key, embeddings[0], embeddings)
    
    print("\nAttention scores for each target:")
    for target_idx in range(n_vars):
        query = embeddings[target_idx]
        logits = attention.apply(params, query, embeddings)
        
        # Mask target
        masked_logits = jnp.where(
            jnp.arange(n_vars) == target_idx,
            -1e9,
            logits
        )
        probs = jax.nn.softmax(masked_logits)
        
        print(f"\nTarget X{target_idx}:")
        print(f"  Logits: {logits}")
        print(f"  Probs:  {probs}")
        
        # Check if probabilities are uniform (excluding target)
        non_target_probs = [probs[i] for i in range(n_vars) if i != target_idx]
        if len(set(f"{p:.6f}" for p in non_target_probs)) == 1:
            print("  ❌ Non-target probabilities are uniform!")


def debug_full_model_step_by_step():
    """Debug the full model step by step."""
    print("\n" + "="*60)
    print("DEBUGGING FULL MODEL STEP BY STEP")
    print("="*60)
    
    # Create model with debugging
    class DebugModel(ContinuousParentSetPredictionModel):
        def __call__(self, data, target_variable, is_training=True):
            N, d, channels = data.shape
            
            print(f"\n1. Input shape: {data.shape}")
            print(f"   Target variable: {target_variable}")
            
            # Encode nodes
            node_encoder = NodeEncoder(
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers
            )
            node_embeddings = node_encoder(data)
            
            print(f"\n2. Node embeddings shape: {node_embeddings.shape}")
            # Check embedding variance
            embed_var = jnp.var(node_embeddings)
            print(f"   Embedding variance: {embed_var:.6f}")
            
            # Check if embeddings are different
            embed_diffs = []
            for i in range(d):
                for j in range(i+1, d):
                    diff = jnp.linalg.norm(node_embeddings[i] - node_embeddings[j])
                    embed_diffs.append(diff)
            print(f"   Mean pairwise embedding difference: {jnp.mean(jnp.array(embed_diffs)):.6f}")
            
            # Get target embedding
            target_embedding = node_embeddings[target_variable]
            print(f"\n3. Target embedding norm: {jnp.linalg.norm(target_embedding):.6f}")
            
            # Compute attention
            parent_attention = ParentAttentionLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                key_size=self.key_size
            )
            parent_logits = parent_attention(target_embedding, node_embeddings)
            
            print(f"\n4. Parent logits: {parent_logits}")
            print(f"   Logit variance: {jnp.var(parent_logits):.6f}")
            
            # Mask and softmax
            masked_logits = jnp.where(
                jnp.arange(d) == target_variable,
                -1e9,
                parent_logits
            )
            parent_probs = jax.nn.softmax(masked_logits)
            
            print(f"\n5. Masked logits: {masked_logits}")
            print(f"   Parent probs: {parent_probs}")
            
            return {
                'node_embeddings': node_embeddings,
                'target_embedding': target_embedding,
                'attention_logits': parent_logits,
                'parent_probabilities': parent_probs
            }
    
    # Create test data with strong correlation
    key = random.PRNGKey(42)
    n_samples = 20
    n_vars = 4
    
    data = jnp.zeros((n_samples, n_vars, 3))
    # X1 and X3 are perfectly correlated
    x1_vals = jnp.linspace(-2, 2, n_samples)
    data = data.at[:, 1, 0].set(x1_vals)
    data = data.at[:, 3, 0].set(x1_vals)  # Perfect correlation
    # X0 and X2 are random
    data = data.at[:, [0, 2], 0].set(random.normal(key, (n_samples, 2)))
    data = data.at[:, :, 2].set(1.0)  # All observed
    
    # Test model
    def model_fn(data, target_idx):
        model = DebugModel(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            key_size=32,
            dropout=0.0
        )
        return model(data, target_idx, is_training=False)
    
    model = hk.without_apply_rng(hk.transform(model_fn))
    params = model.init(key, data, 0)
    
    # Run with target = X3 (should identify X1 as parent)
    output = model.apply(params, data, 3)


def main():
    """Run all debugging steps."""
    print("="*60)
    print("ARCHITECTURAL DEBUGGING: Why Uniform Outputs?")
    print("="*60)
    
    # Debug components
    embeddings, encoder_issue = debug_node_encoder()
    debug_attention_computation()
    debug_full_model_step_by_step()
    
    # Summary
    print("\n" + "="*60)
    print("DEBUGGING SUMMARY")
    print("="*60)
    
    if encoder_issue:
        print("\n❌ ROOT CAUSE FOUND: NodeEncoder produces nearly identical embeddings")
        print("   regardless of input differences. This causes uniform attention scores.")
        print("\nThe issue is in the aggregation at line 113 of model.py:")
        print("   node_embeddings = jnp.mean(x, axis=0)")
        print("\nThis averages across ALL samples, destroying variable-specific information!")
    else:
        print("\n🤔 Need to investigate further...")


if __name__ == "__main__":
    main()