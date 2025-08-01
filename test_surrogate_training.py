#!/usr/bin/env python3
"""
Test BC surrogate training with more episodes to check if it learns non-uniform probabilities.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import logging
logging.basicConfig(level=logging.INFO)

from src.causal_bayes_opt.training.surrogate_bc_trainer import SurrogateBCTrainer

# Train surrogate with more episodes
trainer = SurrogateBCTrainer(
    hidden_dim=128,
    num_layers=4,
    num_heads=8,
    learning_rate=1e-3,
    batch_size=32,
    max_epochs=50,  # More epochs
    early_stopping_patience=10,
    validation_split=0.2,
    seed=42
)

# Train on demonstrations
results = trainer.train(
    demonstrations_path='expert_demonstrations/raw/raw_demonstrations',
    max_demos=20  # Use more demonstrations
)

print(f"\nTraining completed!")
print(f"Final train loss: {results['metrics']['final_train_loss']:.4f}")
print(f"Best validation loss: {results['metrics']['best_val_loss']:.4f}")

# Save checkpoint
checkpoint_path = Path('checkpoints/test_surrogate_detailed')
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
trainer.save_checkpoint(checkpoint_path, results)
print(f"\nCheckpoint saved to {checkpoint_path}")

# Now test the trained model
from src.causal_bayes_opt.evaluation.model_interfaces import create_bc_surrogate
import jax.numpy as jnp

# Load the model
predict_fn, _ = create_bc_surrogate(checkpoint_path, allow_updates=False)

# Create test tensors with different patterns
print("\n" + "="*60)
print("Testing surrogate predictions on different patterns:")
print("="*60)

# Test 1: Zero tensor (should give uniform or near-uniform)
test_tensor1 = jnp.zeros((10, 3, 3))
variables = ['X', 'Y', 'Z']
target_var = 'Y'

posterior1 = predict_fn(test_tensor1, target_var, variables)
print(f"\nTest 1 - Zero tensor:")
print(f"  Marginal probs: {posterior1.metadata['marginal_parent_probs']}")

# Test 2: Strong correlation pattern
import jax.random as random
key = random.PRNGKey(42)

# Create a tensor with strong X->Y correlation
test_tensor2 = jnp.zeros((20, 3, 3))
for i in range(20):
    x_val = random.normal(key, ())
    y_val = 2 * x_val + 0.1 * random.normal(key, ())  # Y strongly depends on X
    z_val = random.normal(key, ())
    
    test_tensor2 = test_tensor2.at[i, 0, 0].set(x_val)  # X value
    test_tensor2 = test_tensor2.at[i, 1, 0].set(y_val)  # Y value
    test_tensor2 = test_tensor2.at[i, 2, 0].set(z_val)  # Z value
    
    key = random.split(key)[0]

posterior2 = predict_fn(test_tensor2, target_var, variables)
print(f"\nTest 2 - Strong X->Y correlation:")
print(f"  Marginal probs: {posterior2.metadata['marginal_parent_probs']}")

# Test 3: Real demonstration data
from src.causal_bayes_opt.training.data_preprocessing import load_demonstrations_from_path
from src.causal_bayes_opt.training.surrogate_bc_trainer import convert_demo_to_training_examples

demos = load_demonstrations_from_path('expert_demonstrations/raw/raw_demonstrations', max_files=1)
if demos:
    examples = convert_demo_to_training_examples(demos[0])
    if examples:
        test_tensor3, target_vars, parent_sets = examples[0]
        test_tensor3 = jnp.array(test_tensor3)
        
        posterior3 = predict_fn(test_tensor3, target_vars[0], ['X0', 'X1', 'X2'])
        print(f"\nTest 3 - Real demonstration data:")
        print(f"  Target: {target_vars[0]}")
        print(f"  True parents: {parent_sets[0]}")
        print(f"  Marginal probs: {posterior3.metadata['marginal_parent_probs']}")