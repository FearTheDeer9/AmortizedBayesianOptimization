#!/usr/bin/env python3
"""
Test script that trains a BC surrogate model on a few demonstrations
and then visualizes its predictions on different SCM structures.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pickle

from src.causal_bayes_opt.experiments.test_scms import (
    create_simple_linear_scm, create_chain_test_scm, 
    create_fork_test_scm, create_collider_test_scm
)
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.data_structures.scm import get_variables, get_parents
from src.causal_bayes_opt.avici_integration.continuous.configurable_model import ConfigurableContinuousParentSetPredictionModel
from src.causal_bayes_opt.training.data_preprocessing import (
    load_demonstrations_from_path, preprocess_demonstration_batch
)


def create_simple_training_data(n_demos: int = 10) -> List[Dict]:
    """Create simple training data from expert demonstrations."""
    print(f"Loading {n_demos} demonstrations for training...")
    
    # Load demonstrations
    demo_path = 'expert_demonstrations/raw/raw_demonstrations'
    raw_demos = load_demonstrations_from_path(demo_path, max_files=n_demos)
    
    if len(raw_demos) < n_demos:
        print(f"Warning: Only found {len(raw_demos)} demonstrations")
    
    # Preprocess demonstrations
    training_data = []
    for i, demo in enumerate(raw_demos):
        try:
            preprocessed = preprocess_demonstration_batch([demo])
            if preprocessed['surrogate_data']:
                # Just take first few examples from each demo to speed up
                examples_to_add = preprocessed['surrogate_data'][:5]
                training_data.extend(examples_to_add)
                print(f"  Demo {i+1}: Added {len(examples_to_add)} examples")
        except Exception as e:
            print(f"  Demo {i+1}: Failed - {e}")
    
    print(f"Total training examples: {len(training_data)}")
    return training_data


def train_surrogate_model(training_data: List, 
                         encoder_type: str = "node_feature",
                         n_epochs: int = 50,
                         batch_size: int = 32,
                         learning_rate: float = 1e-3) -> Dict:
    """Train a surrogate model on the training data."""
    print(f"\nTraining {encoder_type} encoder model for {n_epochs} epochs...")
    
    # Create model
    def model_fn(data, target_idx, is_training=False):
        model = ConfigurableContinuousParentSetPredictionModel(
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            key_size=32,
            dropout=0.1,
            encoder_type=encoder_type,
            attention_type="pairwise" if encoder_type == "node_feature" else "original"
        )
        return model(data, target_idx, is_training)
    
    net = hk.transform(model_fn)
    
    # Initialize with first example
    example = training_data[0]
    key = jax.random.PRNGKey(42)
    params = net.init(key, example.state_tensor, example.target_idx, False)
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Loss function (cross-entropy with marginal probabilities)
    def loss_fn(params, example, key):
        output = net.apply(params, key, example.state_tensor, example.target_idx, True)
        probs = output['parent_probabilities']
        
        # Get true marginal parent probabilities
        target = jnp.zeros(len(example.variables))
        for i, var in enumerate(example.variables):
            if var in example.marginal_parent_probs:
                target = target.at[i].set(example.marginal_parent_probs[var])
        
        # Normalize if needed (should already sum to 1 or less)
        target_sum = jnp.sum(target)
        if target_sum > 0:
            target = target / target_sum
        
        # Cross-entropy loss
        loss = -jnp.sum(target * jnp.log(probs + 1e-8))
        return loss
    
    # Training loop
    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        
        # Shuffle data
        key, shuffle_key = jax.random.split(key)
        indices = jax.random.permutation(shuffle_key, len(training_data))
        
        # Mini-batch training
        for i in range(0, len(training_data), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_loss = 0.0
            
            for idx in batch_indices:
                example = training_data[idx]
                key, loss_key = jax.random.split(key)
                
                # Compute loss and gradients
                loss_value, grads = jax.value_and_grad(loss_fn)(params, example, loss_key)
                batch_loss += loss_value
                
                # Update parameters
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
            
            epoch_loss += batch_loss
        
        avg_loss = epoch_loss / len(training_data)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    print(f"Final loss: {losses[-1]:.4f}")
    
    return {
        'params': params,
        'net': net,
        'losses': losses,
        'encoder_type': encoder_type
    }


def visualize_trained_predictions(trained_model: Dict):
    """Visualize predictions from the trained model."""
    params = trained_model['params']
    net = trained_model['net']
    encoder_type = trained_model['encoder_type']
    
    print(f"\nVisualizing trained {encoder_type} encoder predictions...")
    
    # Create test SCMs (same as visualization script)
    test_scms = []
    
    # Chain
    chain_scm = create_chain_test_scm(chain_length=4, coefficient=0.8, noise_scale=0.4)
    test_scms.append({'name': 'Chain (V0→V1→V2→V3)', 'scm': chain_scm})
    
    # Fork
    fork_scm = create_fork_test_scm(noise_scale=0.4)
    test_scms.append({'name': 'Fork (X←Z→Y)', 'scm': fork_scm})
    
    # Collider
    collider_scm = create_collider_test_scm(noise_scale=0.4)
    test_scms.append({'name': 'Collider (X→Z←Y)', 'scm': collider_scm})
    
    # Complex
    complex_scm = create_simple_linear_scm(
        variables=['A', 'B', 'C', 'D'],
        edges=[('A', 'C'), ('B', 'C'), ('B', 'D'), ('C', 'D')],
        coefficients={('A', 'C'): 0.6, ('B', 'C'): 0.7, ('B', 'D'): 0.5, ('C', 'D'): 0.8},
        noise_scales={'A': 0.4, 'B': 0.4, 'C': 0.4, 'D': 0.4},
        target='D'
    )
    test_scms.append({'name': 'Complex (A→C, B→C, B→D, C→D)', 'scm': complex_scm})
    
    # Create figure
    n_scms = len(test_scms)
    fig = plt.figure(figsize=(16, 4 * n_scms))
    
    key = jax.random.PRNGKey(123)
    
    for scm_idx, scm_info in enumerate(test_scms):
        scm = scm_info['scm']
        scm_name = scm_info['name']
        variables = list(get_variables(scm))
        n_vars = len(variables)
        
        # Build parent sets
        parent_sets = {}
        for var in variables:
            parent_sets[var] = list(get_parents(scm, var))
        
        print(f"\n{'='*60}")
        print(f"SCM: {scm_name}")
        print(f"Variables: {variables}")
        print(f"True parents: {parent_sets}")
        
        # Generate test data
        samples = sample_from_linear_scm(scm, n_samples=100, seed=42)
        
        # Convert to tensor format
        data = jnp.zeros((100, n_vars, 3))
        for i, sample in enumerate(samples):
            values = jnp.array([sample['values'][var] for var in variables])
            data = data.at[i, :, 0].set(values)
            # Add some random interventions
            key, int_key = jax.random.split(key)
            intervene_mask = jax.random.uniform(int_key, (n_vars,)) < 0.1
            data = data.at[i, :, 1].set(intervene_mask.astype(float))
        
        # Plot data statistics
        ax1 = plt.subplot(n_scms, 4, scm_idx * 4 + 1)
        values = data[:, :, 0]
        intervention_rates = jnp.mean(data[:, :, 1], axis=0)
        
        ax1.boxplot([values[:, i] for i in range(n_vars)], labels=variables)
        ax1.set_title(f'{scm_name}\nValue Distributions')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)
        
        # Intervention rates
        ax2 = plt.subplot(n_scms, 4, scm_idx * 4 + 2)
        ax2.bar(variables, intervention_rates)
        ax2.set_title('Intervention Rates')
        ax2.set_ylabel('Rate')
        ax2.set_ylim(0, 0.3)
        ax2.grid(True, alpha=0.3)
        
        # Get predictions from trained model
        predictions = {}
        for target_idx, target_var in enumerate(variables):
            # Set target indicators
            data_with_target = data.copy()
            data_with_target = data_with_target.at[:, :, 2].set(0.0)
            data_with_target = data_with_target.at[:, target_idx, 2].set(1.0)
            
            output = net.apply(params, key, data_with_target, target_idx, False)
            predictions[target_var] = output['parent_probabilities']
        
        # Plot posterior matrix
        ax3 = plt.subplot(n_scms, 4, scm_idx * 4 + 3)
        posterior_matrix = jnp.stack([predictions[var] for var in variables])
        im = ax3.imshow(posterior_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        ax3.set_xticks(range(n_vars))
        ax3.set_xticklabels(variables)
        ax3.set_yticks(range(n_vars))
        ax3.set_yticklabels([f'{var}←' for var in variables])
        ax3.set_title('Predicted Parent Posteriors\n(row=target, col=parent)')
        
        # Add values
        for i in range(n_vars):
            for j in range(n_vars):
                value = posterior_matrix[i, j]
                color = 'white' if value > 0.5 else 'black'
                ax3.text(j, i, f'{value:.2f}', ha='center', va='center', color=color, fontsize=8)
        
        # True adjacency matrix
        ax4 = plt.subplot(n_scms, 4, scm_idx * 4 + 4)
        true_adj = jnp.zeros((n_vars, n_vars))
        for child, parents in parent_sets.items():
            child_idx = variables.index(child)
            for parent in parents:
                parent_idx = variables.index(parent)
                true_adj = true_adj.at[child_idx, parent_idx].set(1)
        
        ax4.imshow(true_adj, cmap='Greys', vmin=0, vmax=1, aspect='auto')
        ax4.set_xticks(range(n_vars))
        ax4.set_xticklabels(variables)
        ax4.set_yticks(range(n_vars))
        ax4.set_yticklabels([f'{var}←' for var in variables])
        ax4.set_title('True Parent Structure')
        
        # Add checkmarks
        for i in range(n_vars):
            for j in range(n_vars):
                value = true_adj[i, j]
                color = 'white' if value > 0.5 else 'black'
                ax4.text(j, i, '✓' if value == 1 else '', ha='center', va='center', 
                        color=color, fontsize=12, fontweight='bold')
        
        # Print predictions
        print("\nPredictions:")
        accuracy = 0
        total = 0
        for target_var in variables:
            probs = predictions[target_var]
            top_parent_idx = jnp.argmax(probs)
            top_parent = variables[top_parent_idx] if probs[top_parent_idx] > 0.1 else "None"
            true_parents = parent_sets[target_var]
            
            # Check accuracy
            if true_parents:
                if top_parent in true_parents:
                    accuracy += 1
            else:
                if top_parent == "None":
                    accuracy += 1
            total += 1
            
            print(f"  {target_var} ← {top_parent} (p={probs[top_parent_idx]:.3f})", end="")
            if true_parents:
                true_parent_probs = [f"{p}:{probs[variables.index(p)]:.3f}" for p in true_parents]
                print(f" | True: {true_parents} [{', '.join(true_parent_probs)}]")
            else:
                print(f" | True: [] (no parents)")
        
        print(f"\nAccuracy: {accuracy}/{total} = {accuracy/total:.1%}")
    
    plt.tight_layout()
    plt.suptitle(f'Trained {encoder_type} Encoder Predictions', fontsize=16, y=1.02)
    
    # Add colorbar
    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
    plt.colorbar(im, cax=cbar_ax, label='Parent Probability')
    
    plt.savefig(f'trained_encoder_predictions_{encoder_type}.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as trained_encoder_predictions_{encoder_type}.png")
    plt.show()


def main():
    """Train and test the surrogate model."""
    import argparse
    parser = argparse.ArgumentParser(description='Train and test BC surrogate model')
    parser.add_argument('--encoder', type=str, default='node_feature',
                       choices=['node_feature', 'node', 'simple'],
                       help='Encoder type to use')
    parser.add_argument('--n-demos', type=int, default=10,
                       help='Number of demonstrations to train on')
    parser.add_argument('--n-epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--save-model', action='store_true',
                       help='Save trained model to file')
    parser.add_argument('--load-model', type=str, default=None,
                       help='Load model from file instead of training')
    args = parser.parse_args()
    
    if args.load_model:
        # Load existing model
        print(f"Loading model from {args.load_model}")
        with open(args.load_model, 'rb') as f:
            trained_model = pickle.load(f)
    else:
        # Load training data
        training_data = create_simple_training_data(args.n_demos)
        
        if not training_data:
            print("No training data found!")
            return
        
        # Train model
        trained_model = train_surrogate_model(
            training_data, 
            encoder_type=args.encoder,
            n_epochs=args.n_epochs
        )
        
        # Save model if requested
        if args.save_model:
            filename = f'trained_surrogate_{args.encoder}_{args.n_demos}demos.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(trained_model, f)
            print(f"\nModel saved to {filename}")
    
    # Visualize predictions
    visualize_trained_predictions(trained_model)


if __name__ == "__main__":
    main()