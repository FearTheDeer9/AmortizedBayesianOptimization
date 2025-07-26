#!/usr/bin/env python3
"""
Find why gradients are zero in the policy network.
"""

import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import haiku as hk

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Finding Gradient Issue")
print("=" * 80)

# Test with a minimal example
print("\n1. Testing with minimal policy network:")
print("-" * 40)

def minimal_policy():
    """Minimal policy to test gradient flow."""
    def policy_fn(inputs):
        # Simple linear layer
        logits = hk.Linear(3)(inputs)
        
        # Mask one output (like target masking)
        masked_logits = logits.at[2].set(-1e9)
        
        # Compute softmax and entropy
        probs = jax.nn.softmax(masked_logits)
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-8))
        
        return entropy, logits
    
    return hk.transform(policy_fn)

# Test minimal network
transform = minimal_policy()
key = jax.random.PRNGKey(42)
dummy_input = jnp.ones(10)  # Simple input

params = transform.init(key, dummy_input)
print(f"Parameters: {jax.tree.map(lambda x: x.shape, params)}")

# Define loss (negative entropy)
def loss_fn(params, inputs):
    entropy, logits = transform.apply(params, key, inputs)
    return -entropy, (entropy, logits)

# Compute gradients
grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
(loss_val, (entropy, logits)), grads = grad_fn(params, dummy_input)

print(f"Loss: {loss_val:.6f}")
print(f"Entropy: {entropy:.6f}")
print(f"Logits: {logits}")
print(f"Gradient norm: {jax.tree.map(lambda x: jnp.linalg.norm(x), grads)}")

# 2. Test with actual encoder output
print("\n\n2. Testing with actual network components:")
print("-" * 40)

from src.causal_bayes_opt.acquisition.enriched.state_enrichment import EnrichedHistoryBuilder
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.data_structures.sample import create_sample
import pyrsistent as pyr

# Create realistic state
buffer = ExperienceBuffer()
buffer.add_observation(create_sample(values={'X': 1.0, 'Y': -0.5, 'Z': 0.0}))
intervention = pyr.m(type='perfect', targets=frozenset({'X'}), values={'X': 2.0})
outcome = create_sample(
    values={'X': 2.0, 'Y': 1.0, 'Z': 0.5},
    intervention_type='perfect',
    intervention_targets=frozenset({'X'})
)
buffer.add_intervention(intervention, outcome)

state = type('MockState', (), {
    'buffer': buffer,
    'current_target': 'Z',
    'marginal_parent_probs': {'X': 0.8, 'Y': 0.3, 'Z': 0.1}
})()

builder = EnrichedHistoryBuilder()
enriched_history, _ = builder.build_enriched_history(state)

print(f"Enriched history shape: {enriched_history.shape}")

# Test if the input has any variation
print(f"Input variance per channel:")
for c in range(5):
    channel_var = jnp.var(enriched_history[:, :, c])
    print(f"  Channel {c}: variance = {channel_var:.6f}")

# 3. Test encoder gradients
print("\n\n3. Testing encoder gradient flow:")
print("-" * 40)

from src.causal_bayes_opt.acquisition.enriched.enriched_policy import EnrichedAttentionEncoder

def test_encoder_gradients():
    def encoder_fn(enriched_history):
        encoder = EnrichedAttentionEncoder(
            num_layers=2,
            num_heads=4,
            hidden_dim=128,
            key_size=32
        )
        embeddings = encoder(enriched_history, is_training=False)
        
        # Simple loss: minimize L2 norm of embeddings
        loss = jnp.sum(embeddings**2)
        return loss, embeddings
    
    return hk.transform(encoder_fn)

encoder_transform = test_encoder_gradients()
encoder_params = encoder_transform.init(key, enriched_history)

def encoder_loss_fn(params, inputs):
    return encoder_transform.apply(params, key, inputs)

encoder_grad_fn = jax.value_and_grad(encoder_loss_fn, has_aux=True)
(encoder_loss, embeddings), encoder_grads = encoder_grad_fn(encoder_params, enriched_history)

# Check if encoder has gradients
encoder_grad_norm = jax.tree_util.tree_reduce(
    lambda x, y: x + jnp.sum(y**2),
    encoder_grads,
    0.0
)
encoder_grad_norm = jnp.sqrt(encoder_grad_norm)

print(f"Encoder loss: {encoder_loss:.6f}")
print(f"Embeddings shape: {embeddings.shape}")
print(f"Encoder gradient norm: {encoder_grad_norm:.6f}")

if encoder_grad_norm < 1e-6:
    print("❌ Encoder has vanishing gradients!")
else:
    print("✅ Encoder gradients are flowing")

# 4. Check for common issues
print("\n\n4. Diagnosing the issue:")
print("-" * 40)

# Check if inputs are all zeros
if jnp.allclose(enriched_history, 0.0):
    print("❌ Input is all zeros!")
elif jnp.var(enriched_history) < 1e-6:
    print("❌ Input has no variation!")
else:
    print("✅ Input has variation")

# Check if the issue is with initialization
all_zero_params = jax.tree_util.tree_all(
    jax.tree.map(lambda x: jnp.allclose(x, 0.0), encoder_params)
)
if all_zero_params:
    print("❌ All parameters initialized to zero!")
else:
    print("✅ Parameters properly initialized")

print("\n" + "=" * 80)
print("CONCLUSION:")
if encoder_grad_norm < 1e-6:
    print("The encoder is not producing gradients. This could be due to:")
    print("1. Attention mechanism saturating (all attention weights identical)")
    print("2. Layer normalization causing gradient vanishing")
    print("3. Initialization issues")
else:
    print("The encoder works fine. The issue is likely in the policy heads or GRPO training.")