#!/usr/bin/env python3
"""
Debug BC surrogate to see what's happening with the 0.5 probabilities.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import jax
import jax.numpy as jnp
import haiku as hk
from src.causal_bayes_opt.utils.checkpoint_utils import load_checkpoint
from src.causal_bayes_opt.models.continuous_parent_set_prediction import ContinuousParentSetPredictionNet

# Load checkpoint
checkpoint_path = Path('checkpoints/test_v2/bc_surrogate_final')
checkpoint = load_checkpoint(checkpoint_path)

print(f"Checkpoint keys: {checkpoint.keys()}")
print(f"Model type: {checkpoint['model_type']}")
print(f"Architecture: {checkpoint['architecture']}")

# Get params
params = checkpoint['params']

# Create model
def model_fn(x):
    net = ContinuousParentSetPredictionNet(
        hidden_dim=checkpoint['architecture']['hidden_dim'],
        num_layers=checkpoint['architecture']['num_layers'],
        num_heads=checkpoint['architecture']['num_heads'],
        key_size=checkpoint['architecture']['key_size'],
        dropout=checkpoint['architecture']['dropout']
    )
    return net(x, target_variable_idx=1, is_training=False)

# Transform
net = hk.without_apply_rng(hk.transform(model_fn))

# Create test input
test_input = jnp.zeros((10, 3, 3))  # [seq_len, n_vars, 3]

# Forward pass
output = net.apply(params, test_input)

print(f"\nModel output keys: {output.keys()}")
print(f"Parent probabilities shape: {output['parent_probabilities'].shape}")
print(f"Parent probabilities values: {output['parent_probabilities']}")
print(f"All values equal 0.5? {jnp.allclose(output['parent_probabilities'], 0.5)}")

# Check if it's just initialization
print(f"\nParameter statistics:")
for key, value in jax.tree_util.tree_flatten_with_keys(params)[0]:
    if 'w' in str(key).lower() or 'kernel' in str(key).lower():
        print(f"  {key}: mean={float(jnp.mean(value)):.4f}, std={float(jnp.std(value)):.4f}")

# Test with different inputs
print("\n" + "="*60)
print("Testing with different input patterns:")

# Pattern 1: Strong correlation
test_input2 = jnp.zeros((20, 3, 3))
# Set strong X->Y correlation in values
test_input2 = test_input2.at[:, 0, 0].set(jnp.linspace(-2, 2, 20))  # X values
test_input2 = test_input2.at[:, 1, 0].set(2 * jnp.linspace(-2, 2, 20))  # Y = 2*X
test_input2 = test_input2.at[:, 2, 0].set(jnp.random.normal(jax.random.PRNGKey(42), (20,)))  # Z random

output2 = net.apply(params, test_input2)
print(f"\nPattern 1 - Strong X->Y correlation:")
print(f"  Parent probabilities: {output2['parent_probabilities']}")

# Pattern 2: All ones (to test if model responds to input at all)
test_input3 = jnp.ones((10, 3, 3))
output3 = net.apply(params, test_input3)
print(f"\nPattern 2 - All ones:")
print(f"  Parent probabilities: {output3['parent_probabilities']}")

# Pattern 3: Random noise
key = jax.random.PRNGKey(123)
test_input4 = jax.random.normal(key, (10, 3, 3))
output4 = net.apply(params, test_input4)
print(f"\nPattern 3 - Random noise:")
print(f"  Parent probabilities: {output4['parent_probabilities']}")

print("\nConclusion:")
if jnp.allclose(output['parent_probabilities'], output2['parent_probabilities']) and \
   jnp.allclose(output['parent_probabilities'], output3['parent_probabilities']) and \
   jnp.allclose(output['parent_probabilities'], output4['parent_probabilities']):
    print("Model outputs are CONSTANT regardless of input - likely not trained properly!")
else:
    print("Model outputs vary with input - model is responding to data")