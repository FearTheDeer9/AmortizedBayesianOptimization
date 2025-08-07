#!/usr/bin/env python3
"""Quick test of encoder diversity after improvements."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import haiku as hk
from src.causal_bayes_opt.avici_integration.continuous.configurable_model import ConfigurableContinuousParentSetPredictionModel

# Create test data
key = jax.random.PRNGKey(42)
N = 100  # samples
d = 5    # variables

# Create diverse data
data = jnp.zeros((N, d, 3))
# Different patterns for each variable
data = data.at[:, 0, 0].set(jnp.linspace(0, 10, N))  # Linear
data = data.at[:, 1, 0].set(jnp.sin(jnp.linspace(0, 4*jnp.pi, N)) * 5)  # Sine
data = data.at[:, 2, 0].set(jax.random.normal(key, (N,)) * 2)  # Normal
data = data.at[:, 3, 0].set(jnp.exp(jnp.linspace(0, 2, N)))  # Exponential
data = data.at[:, 4, 0].set(jnp.ones(N) * 3.14)  # Constant

# All observed
data = data.at[:, :, 2].set(1.0)

# Test model
def model_fn(data, target_idx, is_training=False):
    model = ConfigurableContinuousParentSetPredictionModel(
        hidden_dim=128,
        num_layers=4,
        num_heads=8,
        key_size=32,
        dropout=0.1,
        encoder_type="node_feature",
        attention_type="pairwise"
    )
    return model(data, target_idx, is_training)

net = hk.transform(model_fn)
params = net.init(key, data, 0, False)

# Test predictions for different targets
print("Testing encoder diversity with improved architecture")
print("="*60)

all_predictions = []
for target_idx in range(d):
    output = net.apply(params, key, data, target_idx, False)
    probs = output['parent_probabilities']
    all_predictions.append(probs)
    
    print(f"\nTarget {target_idx}:")
    print(f"  Probabilities: {probs}")
    print(f"  Max prob: {jnp.max(probs):.4f}")
    print(f"  Std dev: {jnp.std(probs):.4f}")
    print(f"  Entropy: {-jnp.sum(probs * jnp.log(probs + 1e-8)):.4f}")

# Overall metrics
predictions_array = jnp.stack(all_predictions)
print(f"\nOverall metrics:")
print(f"  Mean prediction std: {jnp.mean(jnp.std(predictions_array, axis=1)):.4f}")
print(f"  Mean max probability: {jnp.mean(jnp.max(predictions_array, axis=1)):.4f}")
print(f"  Mean entropy: {jnp.mean(-jnp.sum(predictions_array * jnp.log(predictions_array + 1e-8), axis=1)):.4f}")

# Check embeddings
embeddings = output['node_embeddings']
norms = jnp.linalg.norm(embeddings, axis=1, keepdims=True)
normalized = embeddings / (norms + 1e-8)
similarity_matrix = jnp.dot(normalized, normalized.T)
n = similarity_matrix.shape[0]
upper_indices = jnp.triu_indices(n, k=1)
similarities = similarity_matrix[upper_indices]
print(f"\nEmbedding similarities:")
print(f"  Mean: {jnp.mean(similarities):.4f}")
print(f"  Max: {jnp.max(similarities):.4f}")
print(f"  Min: {jnp.min(similarities):.4f}")