#!/usr/bin/env python3
"""
Quick test: Train a BC surrogate on synthetic data and visualize predictions.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import matplotlib.pyplot as plt
from typing import Dict, List
import numpy as np

from src.causal_bayes_opt.experiments.test_scms import (
    create_simple_linear_scm, create_chain_test_scm, 
    create_fork_test_scm, create_collider_test_scm
)
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.data_structures.scm import get_variables, get_parents
from src.causal_bayes_opt.avici_integration.continuous.configurable_model import ConfigurableContinuousParentSetPredictionModel


def create_synthetic_training_data(n_examples: int = 100):
    """Create synthetic training data for quick testing."""
    print(f"Creating {n_examples} synthetic training examples...")
    
    # Create a simple chain SCM for training
    scm = create_chain_test_scm(chain_length=3, coefficient=0.8, noise_scale=0.3)
    variables = list(get_variables(scm))
    n_vars = len(variables)
    
    training_data = []
    key = jax.random.PRNGKey(42)
    
    for i in range(n_examples):
        # Generate samples
        samples = sample_from_linear_scm(scm, n_samples=50, seed=i)
        
        # Convert to tensor
        data = jnp.zeros((50, n_vars, 3))
        for j, sample in enumerate(samples):
            values = jnp.array([sample['values'][var] for var in variables])
            data = data.at[j, :, 0].set(values)
        
        # Choose random target
        key, target_key = jax.random.split(key)
        target_idx = int(jax.random.choice(target_key, n_vars))
        target_var = variables[target_idx]
        
        # Create true labels (one-hot for true parents)
        true_parents = get_parents(scm, target_var)
        labels = jnp.zeros(n_vars)
        for parent in true_parents:
            parent_idx = variables.index(parent)
            labels = labels.at[parent_idx].set(1.0 / len(true_parents) if true_parents else 0.0)
        
        training_data.append({
            'data': data,
            'target_idx': target_idx,
            'labels': labels,
            'variables': variables
        })
    
    print(f"Created {len(training_data)} training examples")
    return training_data


def train_simple_model(training_data: List[Dict], 
                      encoder_type: str = "node_feature",
                      n_epochs: int = 20) -> Dict:
    """Train model with simple supervised learning."""
    print(f"\nTraining {encoder_type} encoder for {n_epochs} epochs...")
    
    # Create model
    def model_fn(data, target_idx, is_training=False):
        model = ConfigurableContinuousParentSetPredictionModel(
            hidden_dim=64,  # Smaller for faster training
            num_layers=2,
            num_heads=4,
            key_size=16,
            dropout=0.1,
            encoder_type=encoder_type,
            attention_type="pairwise" if encoder_type == "node_feature" else "original"
        )
        return model(data, target_idx, is_training)
    
    net = hk.transform(model_fn)
    
    # Initialize
    example = training_data[0]
    key = jax.random.PRNGKey(42)
    params = net.init(key, example['data'], example['target_idx'], False)
    
    # Optimizer
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    
    # Loss function
    def loss_fn(params, batch, key):
        output = net.apply(params, key, batch['data'], batch['target_idx'], True)
        probs = output['parent_probabilities']
        loss = -jnp.sum(batch['labels'] * jnp.log(probs + 1e-8))
        return loss
    
    # Training loop
    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        
        for example in training_data:
            key, loss_key = jax.random.split(key)
            loss_value, grads = jax.value_and_grad(loss_fn)(params, example, loss_key)
            
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            epoch_loss += loss_value
        
        avg_loss = epoch_loss / len(training_data)
        losses.append(avg_loss)
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    print(f"Final loss: {losses[-1]:.4f}")
    
    return {'params': params, 'net': net, 'losses': losses}


def test_on_different_scms(trained_model: Dict, encoder_type: str):
    """Test trained model on different SCM structures."""
    params = trained_model['params']
    net = trained_model['net']
    
    print("\nTesting on different SCM structures...")
    
    # Test SCMs
    test_scms = [
        ('Chain-3', create_chain_test_scm(3, 0.8, 0.3)),
        ('Fork', create_fork_test_scm(0.3)),
        ('Collider', create_collider_test_scm(0.3))
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for idx, (name, scm) in enumerate(test_scms):
        variables = list(get_variables(scm))
        n_vars = len(variables)
        
        # Generate test data
        samples = sample_from_linear_scm(scm, n_samples=50, seed=123)
        data = jnp.zeros((50, n_vars, 3))
        for i, sample in enumerate(samples):
            values = jnp.array([sample['values'][var] for var in variables])
            data = data.at[i, :, 0].set(values)
        
        # Get predictions for all targets
        pred_matrix = []
        for target_idx in range(n_vars):
            data_with_target = data.copy()
            data_with_target = data_with_target.at[:, :, 2].set(0.0)
            data_with_target = data_with_target.at[:, target_idx, 2].set(1.0)
            
            output = net.apply(params, jax.random.PRNGKey(0), data_with_target, target_idx, False)
            pred_matrix.append(output['parent_probabilities'])
        
        pred_matrix = jnp.stack(pred_matrix)
        
        # Plot
        ax = axes[idx]
        im = ax.imshow(pred_matrix, cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_title(f'{name} SCM')
        ax.set_xlabel('Parent')
        ax.set_ylabel('Child')
        ax.set_xticks(range(n_vars))
        ax.set_xticklabels(variables)
        ax.set_yticks(range(n_vars))
        ax.set_yticklabels(variables)
        
        # Add values
        for i in range(n_vars):
            for j in range(n_vars):
                val = pred_matrix[i, j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       color=color, fontsize=8)
        
        # Print accuracy
        print(f"\n{name} predictions:")
        correct = 0
        total = 0
        for child_idx, child in enumerate(variables):
            true_parents = list(get_parents(scm, child))
            pred_parent_idx = jnp.argmax(pred_matrix[child_idx])
            pred_parent = variables[pred_parent_idx] if pred_matrix[child_idx, pred_parent_idx] > 0.3 else None
            
            is_correct = (pred_parent in true_parents) if true_parents else (pred_parent is None)
            if is_correct:
                correct += 1
            total += 1
            
            print(f"  {child} ← {pred_parent} (true: {true_parents})")
        
        print(f"  Accuracy: {correct}/{total} = {correct/total:.1%}")
    
    plt.tight_layout()
    plt.savefig(f'quick_trained_{encoder_type}_test.png', dpi=150)
    print(f"\nSaved plot as quick_trained_{encoder_type}_test.png")
    plt.show()


def main():
    """Quick training and testing."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', default='node_feature', choices=['node_feature', 'node', 'simple'])
    parser.add_argument('--n-examples', type=int, default=50)
    parser.add_argument('--n-epochs', type=int, default=20)
    args = parser.parse_args()
    
    # Create synthetic data
    training_data = create_synthetic_training_data(args.n_examples)
    
    # Train model
    trained_model = train_simple_model(training_data, args.encoder, args.n_epochs)
    
    # Test on different SCMs
    test_on_different_scms(trained_model, args.encoder)


if __name__ == "__main__":
    main()