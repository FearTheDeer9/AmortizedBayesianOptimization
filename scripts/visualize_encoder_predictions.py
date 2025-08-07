#!/usr/bin/env python3
"""
Visualize encoder input data and output posteriors on different SCMs.

This script shows:
1. The input data (values, interventions)
2. The true parent relationships
3. The predicted parent posteriors
4. Comparison across different SCM structures

Note: The data format is [N, d, 3] where:
- Channel 0: variable values
- Channel 1: intervention indicators (1 if intervened)
- Channel 2: target indicators (1 for target variable)
  
The model identifies the target via the target_idx parameter, not channel 2,
but we set channel 2 for consistency with the training data format.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from src.causal_bayes_opt.experiments.test_scms import (
    create_simple_linear_scm, create_chain_test_scm, 
    create_fork_test_scm, create_collider_test_scm
)
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.avici_integration.continuous.configurable_model import ConfigurableContinuousParentSetPredictionModel


def create_test_scms() -> List[Dict]:
    """Create different SCM structures for testing."""
    scms = []
    
    # 1. Chain structure: V0 -> V1 -> V2 -> V3 (default names)
    chain_scm = create_chain_test_scm(
        chain_length=4,
        coefficient=0.8,
        noise_scale=0.4
    )
    scms.append({
        'name': 'Chain (V0→V1→V2→V3)',
        'scm': chain_scm
    })
    
    # 2. Fork structure: X <- Z -> Y (default names)
    fork_scm = create_fork_test_scm(
        noise_scale=0.4
    )
    scms.append({
        'name': 'Fork (X←Z→Y)',
        'scm': fork_scm
    })
    
    # 3. Collider structure: X -> Z <- Y (default names)
    collider_scm = create_collider_test_scm(
        noise_scale=0.4
    )
    scms.append({
        'name': 'Collider (X→Z←Y)',
        'scm': collider_scm
    })
    
    # 4. Complex structure - use simple linear SCM with custom structure
    complex_scm = create_simple_linear_scm(
        variables=['A', 'B', 'C', 'D'],
        edges=[('A', 'C'), ('B', 'C'), ('B', 'D'), ('C', 'D')],
        coefficients={
            ('A', 'C'): 0.6,
            ('B', 'C'): 0.7,
            ('B', 'D'): 0.5,
            ('C', 'D'): 0.8
        },
        noise_scales={'A': 0.4, 'B': 0.4, 'C': 0.4, 'D': 0.4},
        target='D'
    )
    scms.append({
        'name': 'Complex (A→C, B→C, B→D, C→D)',
        'scm': complex_scm
    })
    
    return scms


def generate_data_with_interventions(scm, n_samples: int = 100, intervention_prob: float = 0.2) -> jnp.ndarray:
    """Generate data with some interventions."""
    from src.causal_bayes_opt.data_structures.scm import get_variables
    
    # Generate observational samples first
    samples = sample_from_linear_scm(scm, n_samples=n_samples, seed=42)
    
    variables = list(get_variables(scm))
    n_vars = len(variables)
    
    # Initialize data array [N, d, 3]
    data = jnp.zeros((n_samples, n_vars, 3))
    
    # Fill with observational data
    for i, sample in enumerate(samples):
        values = jnp.array([sample['values'][var] for var in variables])
        data = data.at[i, :, 0].set(values)
        # Channel 2 (target indicators) will be set when making predictions
    
    # Add some interventions (simulated as random values)
    key = jax.random.PRNGKey(42)
    for i in range(n_samples):
        key, int_key = jax.random.split(key)
        if jax.random.uniform(int_key) < 0.5:  # 50% chance of having interventions
            key, mask_key = jax.random.split(key)
            intervene_mask = jax.random.uniform(mask_key, (n_vars,)) < intervention_prob
            
            if jnp.any(intervene_mask):
                # Replace intervened values with random values
                key, val_key = jax.random.split(key)
                intervention_values = jax.random.normal(val_key, (n_vars,)) * 2.0
                new_values = jnp.where(intervene_mask, intervention_values, data[i, :, 0])
                
                data = data.at[i, :, 0].set(new_values)
                data = data.at[i, :, 1].set(intervene_mask.astype(float))
    
    return data


def visualize_scm_predictions(encoder_type: str = "node_feature"):
    """Visualize predictions for different SCM structures."""
    # Create test SCMs
    test_scms = create_test_scms()
    
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
    key = jax.random.PRNGKey(42)
    
    # Create figure
    n_scms = len(test_scms)
    fig = plt.figure(figsize=(16, 4 * n_scms))
    
    for scm_idx, scm_info in enumerate(test_scms):
        from src.causal_bayes_opt.data_structures.scm import get_variables, get_parents
        
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
        
        # Generate data
        base_data = generate_data_with_interventions(scm, n_samples=100)
        
        # Initialize model parameters
        if scm_idx == 0:
            # Note: Model uses target_idx parameter, not channel 2, but we set it for consistency
            data = base_data.copy()
            data = data.at[:, :, 2].set(0.0)  # Clear all target indicators
            data = data.at[:, 0, 2].set(1.0)  # Set target for variable 0 (for consistency with training)
            params = net.init(key, data, 0, False)
        
        # Plot data statistics
        ax1 = plt.subplot(n_scms, 4, scm_idx * 4 + 1)
        values = base_data[:, :, 0]
        intervention_rates = jnp.mean(base_data[:, :, 1], axis=0)
        
        # Box plot of values
        ax1.boxplot([values[:, i] for i in range(n_vars)], labels=variables)
        ax1.set_title(f'{scm_name}\nValue Distributions')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)
        
        # Intervention rates
        ax2 = plt.subplot(n_scms, 4, scm_idx * 4 + 2)
        ax2.bar(variables, intervention_rates)
        ax2.set_title('Intervention Rates')
        ax2.set_ylabel('Rate')
        ax2.set_ylim(0, 0.5)
        ax2.grid(True, alpha=0.3)
        
        # Predicted parent posteriors
        predictions = {}
        for target_idx, target_var in enumerate(variables):
            # Set target indicators in channel 2 (for consistency with training format)
            # Note: Model actually uses target_idx parameter, not channel 2
            data = base_data.copy()
            data = data.at[:, :, 2].set(0.0)  # Clear all target indicators
            data = data.at[:, target_idx, 2].set(1.0)  # Set target for this variable
            
            output = net.apply(params, key, data, target_idx, False)
            predictions[target_var] = output['parent_probabilities']
            
            # Debug: Check if self-prediction is masked
            if scm_idx == 0 and target_idx == 0:  # First SCM, first variable
                print(f"\nDEBUG - Target: {target_var} (index {target_idx})")
                print(f"Parent probabilities: {output['parent_probabilities']}")
                print(f"Self probability (should be ~0): {output['parent_probabilities'][target_idx]}")
        
        # Plot posterior matrix
        ax3 = plt.subplot(n_scms, 4, scm_idx * 4 + 3)
        posterior_matrix = jnp.stack([predictions[var] for var in variables])
        im = ax3.imshow(posterior_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        ax3.set_xticks(range(n_vars))
        ax3.set_xticklabels(variables)
        ax3.set_yticks(range(n_vars))
        ax3.set_yticklabels([f'{var}←' for var in variables])
        ax3.set_title('Predicted Parent Posteriors\n(row=target, col=parent)')
        
        # Add values to cells
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
        
        # Add values to cells
        for i in range(n_vars):
            for j in range(n_vars):
                value = true_adj[i, j]
                color = 'white' if value > 0.5 else 'black'
                ax4.text(j, i, '✓' if value == 1 else '', ha='center', va='center', 
                        color=color, fontsize=12, fontweight='bold')
        
        # Print predictions
        print("\nPredictions:")
        for target_var in variables:
            probs = predictions[target_var]
            top_parent_idx = jnp.argmax(probs)
            top_parent = variables[top_parent_idx] if probs[top_parent_idx] > 0 else "None"
            true_parents = parent_sets[target_var]
            
            print(f"  {target_var} ← {top_parent} (p={probs[top_parent_idx]:.3f})", end="")
            if true_parents:
                true_parent_probs = [f"{p}:{probs[variables.index(p)]:.3f}" for p in true_parents]
                print(f" | True: {true_parents} [{', '.join(true_parent_probs)}]")
            else:
                print(f" | True: [] (no parents)")
    
    plt.tight_layout()
    plt.suptitle(f'Encoder Predictions on Different SCM Structures ({encoder_type} encoder)', 
                 fontsize=16, y=1.02)
    
    # Add colorbar
    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
    plt.colorbar(im, cax=cbar_ax, label='Parent Probability')
    
    plt.savefig(f'encoder_predictions_{encoder_type}.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as encoder_predictions_{encoder_type}.png")
    plt.show()


def main():
    """Run visualization for different encoder types."""
    import argparse
    parser = argparse.ArgumentParser(description='Visualize encoder predictions on different SCMs')
    parser.add_argument('--encoder', type=str, default='node_feature',
                       choices=['node_feature', 'node', 'simple'],
                       help='Encoder type to use')
    args = parser.parse_args()
    
    print(f"Visualizing predictions with {args.encoder} encoder")
    visualize_scm_predictions(args.encoder)


if __name__ == "__main__":
    main()