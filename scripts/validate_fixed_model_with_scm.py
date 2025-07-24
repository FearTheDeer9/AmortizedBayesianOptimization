#!/usr/bin/env python3
"""
Validate the fixed continuous parent set prediction model using synthetic SCMs.

This script:
1. Creates known SCM structures using native factories
2. Generates observational and interventional data
3. Trains both original and fixed models
4. Compares their ability to recover true causal structures
"""

import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
import optax
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np

# Set up paths
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

# Import native SCM factories
from src.causal_bayes_opt.experiments.benchmark_scms import (
    create_fork_scm,
    create_chain_scm,
    create_collider_scm
)

# Import native sampling and intervention functions
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.environments.sampling import sample_with_intervention
from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention

# Import native data conversion
from src.causal_bayes_opt.avici_integration.core.conversion import samples_to_avici_format
from src.causal_bayes_opt.data_structures.scm import get_variables, get_parents

# Import both models
from src.causal_bayes_opt.avici_integration.continuous.model import ContinuousParentSetPredictionModel
from src.causal_bayes_opt.avici_integration.continuous.fixed_model import FixedContinuousParentSetPredictionModel


def generate_data_from_scm(scm, target_var: str, n_obs: int = 50, n_int_per_var: int = 20, seed: int = 42):
    """
    Generate observational and interventional data from an SCM.
    
    Args:
        scm: The structural causal model
        target_var: Target variable name
        n_obs: Number of observational samples
        n_int_per_var: Number of interventional samples per variable
        seed: Random seed
        
    Returns:
        avici_data: Data in AVICI format [N, d, 3]
        true_parents: Set of true parent variables
        variable_order: Ordered list of variable names
    """
    variables = list(get_variables(scm))
    variable_order = sorted(variables)
    
    # Get true parents of target
    true_parents = get_parents(scm, target_var)
    
    # 1. Generate observational samples
    obs_samples = sample_from_linear_scm(scm, n_samples=n_obs, seed=seed)
    
    # 2. Generate interventional samples
    all_samples = list(obs_samples)
    
    key = random.PRNGKey(seed + 1)
    for var in variables:
        if var != target_var:
            # Intervene on this variable at different values
            for i, value in enumerate([-2.0, -1.0, 0.0, 1.0, 2.0]):
                key, subkey = random.split(key)
                intervention = create_perfect_intervention(frozenset([var]), {var: value})
                int_samples = sample_with_intervention(
                    scm, intervention, 
                    n_samples=n_int_per_var // 5,
                    seed=int(subkey[0])
                )
                all_samples.extend(int_samples)
    
    # 3. Convert to AVICI format
    avici_data = samples_to_avici_format(
        all_samples, 
        variable_order, 
        target_var,
        standardization_params=None  # Keep raw values for interpretability
    )
    
    print(f"Generated {len(all_samples)} samples ({n_obs} obs, {len(all_samples)-n_obs} int)")
    print(f"Data shape: {avici_data.shape}")
    print(f"True parents of {target_var}: {sorted(true_parents)}")
    
    return avici_data, true_parents, variable_order


def create_true_parent_probs(true_parents, variable_order, target_var):
    """Create ground truth parent probability distribution."""
    n_vars = len(variable_order)
    target_idx = variable_order.index(target_var)
    
    # Initialize with small uniform probability
    probs = jnp.ones(n_vars) * 0.01
    
    # Set high probability for true parents
    parent_prob = 0.9 / len(true_parents) if true_parents else 0.0
    
    for parent in true_parents:
        parent_idx = variable_order.index(parent)
        probs = probs.at[parent_idx].set(parent_prob)
    
    # Zero out target variable itself
    probs = probs.at[target_idx].set(0.0)
    
    # Normalize
    probs = probs / jnp.sum(probs)
    
    return probs


def train_model(model_class, data, target_idx, true_parent_probs, 
                n_steps=100, learning_rate=1e-3, model_name="Model"):
    """Train a model to predict parent probabilities."""
    
    # Create model function
    def model_fn(data, target_idx):
        model = model_class(
            hidden_dim=64,
            num_layers=3,
            num_heads=4,
            key_size=32,
            dropout=0.1
        )
        return model(data, target_idx, is_training=True)
    
    # Transform with Haiku
    model = hk.transform(model_fn)
    
    # Initialize
    key = random.PRNGKey(42)
    params = model.init(key, data, target_idx)
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Loss function
    def loss_fn(params, key, data, target_idx, true_probs):
        output = model.apply(params, key, data, target_idx)
        pred_probs = output['parent_probabilities']
        
        # Cross-entropy loss
        # Add small epsilon to avoid log(0)
        loss = -jnp.sum(true_probs * jnp.log(pred_probs + 1e-8))
        
        return loss, pred_probs
    
    # Training loop
    losses = []
    accuracies = []
    
    for step in range(n_steps):
        key, subkey = random.split(key)
        
        # Compute loss and gradients
        (loss, pred_probs), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, subkey, data, target_idx, true_parent_probs
        )
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        # Track metrics
        losses.append(float(loss))
        
        # Accuracy: does highest prob match a true parent?
        pred_parent_idx = jnp.argmax(pred_probs)
        true_parent_indices = jnp.where(true_parent_probs > 0.1)[0]
        is_correct = jnp.any(true_parent_indices == pred_parent_idx)
        accuracies.append(float(is_correct))
        
        if step % 20 == 0:
            print(f"{model_name} Step {step}: Loss = {loss:.4f}, Accuracy = {is_correct}")
    
    # Final evaluation
    final_output = model.apply(params, key, data, target_idx)
    final_probs = final_output['parent_probabilities']
    
    return losses, accuracies, final_probs, params


def evaluate_models_on_scm(scm_name: str, scm, target_var: str):
    """Evaluate both models on a specific SCM structure."""
    print(f"\n{'='*80}")
    print(f"Evaluating on {scm_name}")
    print('='*80)
    
    # Generate data
    data, true_parents, variable_order = generate_data_from_scm(
        scm, target_var, n_obs=100, n_int_per_var=20
    )
    
    target_idx = variable_order.index(target_var)
    true_parent_probs = create_true_parent_probs(true_parents, variable_order, target_var)
    
    print(f"\nTrue parent probabilities:")
    for i, (var, prob) in enumerate(zip(variable_order, true_parent_probs)):
        print(f"  {var}: {prob:.3f}")
    
    # Train original model
    print(f"\nTraining Original Model...")
    orig_losses, orig_accs, orig_final_probs, _ = train_model(
        ContinuousParentSetPredictionModel, data, target_idx, 
        true_parent_probs, n_steps=100, model_name="Original"
    )
    
    # Train fixed model
    print(f"\nTraining Fixed Model...")
    fixed_losses, fixed_accs, fixed_final_probs, _ = train_model(
        FixedContinuousParentSetPredictionModel, data, target_idx,
        true_parent_probs, n_steps=100, model_name="Fixed"
    )
    
    # Compare final predictions
    print(f"\nFinal Predictions:")
    print(f"{'Variable':<10} {'True Prob':<10} {'Original':<10} {'Fixed':<10}")
    print("-" * 40)
    for i, var in enumerate(variable_order):
        print(f"{var:<10} {true_parent_probs[i]:<10.3f} "
              f"{orig_final_probs[i]:<10.3f} {fixed_final_probs[i]:<10.3f}")
    
    # Identify predicted parents
    orig_pred_parent = variable_order[jnp.argmax(orig_final_probs)]
    fixed_pred_parent = variable_order[jnp.argmax(fixed_final_probs)]
    
    print(f"\nPredicted top parent:")
    print(f"  Original: {orig_pred_parent}")
    print(f"  Fixed: {fixed_pred_parent}")
    print(f"  True parents: {sorted(true_parents)}")
    
    return {
        'scm_name': scm_name,
        'orig_losses': orig_losses,
        'orig_accs': orig_accs,
        'fixed_losses': fixed_losses,
        'fixed_accs': fixed_accs,
        'orig_final_probs': orig_final_probs,
        'fixed_final_probs': fixed_final_probs,
        'true_parent_probs': true_parent_probs
    }


def plot_results(results):
    """Plot training curves for all SCMs."""
    n_scms = len(results)
    fig, axes = plt.subplots(n_scms, 2, figsize=(12, 4*n_scms))
    
    if n_scms == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        # Plot losses
        ax = axes[i, 0]
        ax.plot(result['orig_losses'], label='Original', alpha=0.7)
        ax.plot(result['fixed_losses'], label='Fixed', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title(f"{result['scm_name']} - Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot accuracies
        ax = axes[i, 1]
        ax.plot(result['orig_accs'], label='Original', alpha=0.7)
        ax.plot(result['fixed_accs'], label='Fixed', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy')
        ax.set_title(f"{result['scm_name']} - Parent Identification Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig('model_comparison_results.png', dpi=150)
    print("\nSaved results plot to model_comparison_results.png")


def main():
    """Run validation on multiple SCM structures."""
    print("="*80)
    print("VALIDATING FIXED MODEL WITH SYNTHETIC SCMS")
    print("="*80)
    
    # Test on different SCM structures
    test_cases = [
        ("Fork Structure", create_fork_scm(noise_scale=0.1, target='Y'), 'Y'),
        ("Chain Structure", create_chain_scm(chain_length=4, noise_scale=0.1), 'X3'),
        ("Collider Structure", create_collider_scm(noise_scale=0.1), 'Z'),
    ]
    
    results = []
    for scm_name, scm, target_var in test_cases:
        result = evaluate_models_on_scm(scm_name, scm, target_var)
        results.append(result)
    
    # Plot results
    plot_results(results)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for result in results:
        orig_final_acc = result['orig_accs'][-1] if result['orig_accs'] else 0
        fixed_final_acc = result['fixed_accs'][-1] if result['fixed_accs'] else 0
        
        print(f"\n{result['scm_name']}:")
        print(f"  Original Model - Final Accuracy: {orig_final_acc:.2%}")
        print(f"  Fixed Model - Final Accuracy: {fixed_final_acc:.2%}")
    
    # Overall assessment
    orig_avg_acc = np.mean([r['orig_accs'][-1] for r in results if r['orig_accs']])
    fixed_avg_acc = np.mean([r['fixed_accs'][-1] for r in results if r['fixed_accs']])
    
    print(f"\nOverall Average Accuracy:")
    print(f"  Original Model: {orig_avg_acc:.2%}")
    print(f"  Fixed Model: {fixed_avg_acc:.2%}")
    
    if fixed_avg_acc > orig_avg_acc + 0.2:
        print("\n✅ SUCCESS: Fixed model significantly outperforms original!")
        print("   The fixed architecture successfully learns parent relationships.")
    else:
        print("\n⚠️  Fixed model does not show significant improvement.")
        print("   Further investigation needed.")


if __name__ == "__main__":
    main()