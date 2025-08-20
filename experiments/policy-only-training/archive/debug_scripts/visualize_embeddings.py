#!/usr/bin/env python3
"""
Visualize embeddings to see if the model can distinguish between X0, X1, and X2.

This script:
1. Creates a simple 3-node chain SCM
2. Loads or initializes a policy
3. Extracts internal embeddings for each variable
4. Visualizes them to see if they're different or all the same
5. Checks if gradients flow properly
"""

import sys
import os
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import matplotlib.pyplot as plt
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.experiments.benchmark_scms import create_chain_scm
from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
from src.causal_bayes_opt.data_structures import ExperienceBuffer
from src.causal_bayes_opt.policies.permutation_invariant_alternating_policy import (
    create_permutation_invariant_alternating_policy
)


def extract_embeddings(policy_fn, params, tensor, target_idx):
    """Extract internal embeddings for each variable."""
    
    def extract_fn(tensor_input, target_idx):
        """Modified policy to return intermediate embeddings."""
        T, n_vars, n_channels = tensor_input.shape
        
        # Handle both 3 and 5 channel inputs
        if n_channels == 3:
            padded = jnp.zeros((T, n_vars, 5))
            padded = padded.at[:, :, :3].set(tensor_input)
            tensor_input = padded
            n_channels = 5
        
        # Extract channel statistics (from the policy)
        channel_mean = jnp.mean(tensor_input, axis=0)  # [n_vars, 5]
        channel_std = jnp.std(tensor_input, axis=0) + 1e-8  # [n_vars, 5]
        channel_max = jnp.max(tensor_input, axis=0)  # [n_vars, 5]
        channel_min = jnp.min(tensor_input, axis=0)  # [n_vars, 5]
        
        # Initial features before any transformation
        raw_features = {
            'mean': channel_mean,
            'std': channel_std,
            'max': channel_max,
            'min': channel_min,
            'raw_input': tensor_input[-1]  # Last timestep
        }
        
        # Combine into stats features
        stats_features = jnp.stack([channel_mean, channel_std, channel_max, channel_min], axis=0)  
        stats_features = stats_features.reshape(4, n_vars * 5).T  
        stats_features = stats_features.reshape(n_vars, 20)  # [n_vars, 20]
        
        # Initial projection
        hidden_dim = 256
        x_flat = tensor_input.reshape(-1, 5)
        x_proj = hk.Linear(hidden_dim, name="input_projection")(x_flat)
        x = x_proj.reshape(T, n_vars, hidden_dim)
        
        # Stats projection
        stats_proj = hk.Linear(hidden_dim // 4, name="stats_projection")(stats_features)
        stats_proj_expanded = stats_proj[None, :, :].repeat(T, axis=0)
        
        # Gate combination
        gate = hk.Linear(hidden_dim, name="stats_gate")(
            jnp.concatenate([x[:, :, :hidden_dim//4], stats_proj_expanded], axis=-1)
        )
        gate = jax.nn.sigmoid(gate)
        x_gated = x * gate
        
        # Get embeddings after first layer
        first_layer_embeddings = x_gated[-1]  # Last timestep, [n_vars, hidden_dim]
        
        # Continue through alternating attention layers (simplified)
        # We'll just get the final embeddings before the output heads
        
        # Temporal pooling (simplified - just take mean)
        x_pooled = jnp.mean(x_gated, axis=0)  # [n_vars, hidden_dim]
        
        # Combined features
        combined = jnp.concatenate([x_pooled, stats_proj], axis=-1)
        
        # Final embeddings before output heads
        x_final = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                               name="output_norm")(combined)
        
        # Also compute the actual outputs
        # Variable selection head
        var_hidden = hk.Sequential([
            hk.Linear(hidden_dim // 2, name="var_mlp_1"),
            jax.nn.gelu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="var_norm"),
            hk.Linear(hidden_dim // 4, name="var_mlp_2"),
            jax.nn.gelu,
            hk.Linear(1, name="var_mlp_output")
        ])(x_final)
        
        variable_logits = var_hidden.squeeze(-1)
        
        # Mask target
        variable_logits = jnp.where(
            jnp.arange(n_vars) == target_idx,
            -jnp.inf,
            variable_logits
        )
        
        # Value prediction head
        val_hidden = hk.Sequential([
            hk.Linear(hidden_dim // 2, name="val_mlp_1"),
            jax.nn.gelu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="val_norm"),
            hk.Linear(hidden_dim // 4, name="val_mlp_2"),
            jax.nn.gelu,
            hk.Linear(2, name="val_mlp_output")
        ])(x_final)
        
        value_params = val_hidden
        
        return {
            'raw_features': raw_features,
            'stats_features': stats_features,
            'first_layer_embeddings': first_layer_embeddings,
            'final_embeddings': x_final,
            'variable_logits': variable_logits,
            'value_params': value_params,
            'variable_probs': jax.nn.softmax(variable_logits)
        }
    
    # Transform and apply
    extract_transformed = hk.without_apply_rng(hk.transform(extract_fn))
    
    # Get a random key for initialization
    rng_key = jax.random.PRNGKey(42)
    
    # Apply the function
    embeddings = extract_transformed.apply(params, tensor, target_idx)
    
    return embeddings


def visualize_embeddings_analysis():
    """Main analysis function."""
    
    print("\n" + "="*80)
    print("EMBEDDING VISUALIZATION & ANALYSIS")
    print("="*80)
    
    # Create SCM
    scm = create_chain_scm(chain_length=3)
    variables = ['X0', 'X1', 'X2']
    target_var = 'X2'
    
    # Create buffer with some observations
    buffer = ExperienceBuffer()
    
    # Add observations
    from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
    obs_samples = sample_from_linear_scm(scm, 20, seed=42)
    for sample in obs_samples:
        buffer.add_observation(sample)
    
    # Add a few interventions to make it more interesting
    from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
    from src.causal_bayes_opt.environments.sampling import sample_with_intervention
    
    for i in range(5):
        var_to_intervene = variables[i % 2]  # Alternate X0 and X1
        value = np.random.randn()
        
        # Create proper intervention object
        intervention = create_perfect_intervention(
            targets=frozenset([var_to_intervene]),
            values={var_to_intervene: value}
        )
        
        # Sample outcome using the correct function
        outcomes = sample_with_intervention(scm, intervention, n_samples=1, seed=42+i)
        if outcomes:
            outcome = outcomes[0]
            
            buffer.add_intervention(intervention, outcome)
    
    # Convert to tensor
    tensor, mapper = buffer_to_three_channel_tensor(
        buffer, target_var, max_history_size=100, standardize=True
    )
    
    print(f"\n📊 Buffer Contents:")
    print(f"  Observations: {len(buffer.get_observations())}")
    print(f"  Interventions: {len(buffer.get_interventions())}")
    print(f"  Tensor shape: {tensor.shape}")
    print(f"  Variable order: {[mapper.get_name(i) for i in range(3)]}")
    print(f"  Target index: {mapper.target_idx}")
    
    # Initialize policy
    policy_fn_raw = create_permutation_invariant_alternating_policy(hidden_dim=256)
    policy_fn = hk.without_apply_rng(hk.transform(policy_fn_raw))
    
    # Initialize parameters
    rng_key = jax.random.PRNGKey(42)
    init_key, extract_key = jax.random.split(rng_key)
    policy_params = policy_fn.init(init_key, tensor, mapper.target_idx)
    
    # Extract embeddings
    print("\n🧠 Extracting Embeddings...")
    embeddings = extract_embeddings(policy_fn, policy_params, tensor, mapper.target_idx)
    
    # Analyze raw features
    print("\n📈 Raw Feature Statistics:")
    raw_features = embeddings['raw_features']
    for var_idx in range(3):
        var_name = mapper.get_name(var_idx)
        print(f"\n  {var_name}:")
        print(f"    Mean: {raw_features['mean'][var_idx][:3]}...")  # First 3 values
        print(f"    Std:  {raw_features['std'][var_idx][:3]}...")
        print(f"    Max:  {raw_features['max'][var_idx][:3]}...")
        print(f"    Min:  {raw_features['min'][var_idx][:3]}...")
    
    # Check if features are identical
    print("\n🔍 Feature Similarity Check:")
    for key in ['mean', 'std', 'max', 'min']:
        features = raw_features[key]
        # Compare X0 and X1
        diff_01 = jnp.linalg.norm(features[0] - features[1])
        diff_02 = jnp.linalg.norm(features[0] - features[2])
        diff_12 = jnp.linalg.norm(features[1] - features[2])
        print(f"  {key} differences:")
        print(f"    X0-X1: {diff_01:.6f}")
        print(f"    X0-X2: {diff_02:.6f}")
        print(f"    X1-X2: {diff_12:.6f}")
    
    # Analyze final embeddings
    print("\n🎯 Final Embedding Analysis:")
    final_emb = embeddings['final_embeddings']
    print(f"  Shape: {final_emb.shape}")
    
    # Compute cosine similarity between variable embeddings
    def cosine_similarity(a, b):
        return jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-8)
    
    print("\n  Cosine Similarities:")
    for i in range(3):
        for j in range(i+1, 3):
            sim = cosine_similarity(final_emb[i], final_emb[j])
            print(f"    {mapper.get_name(i)}-{mapper.get_name(j)}: {sim:.4f}")
    
    # Check output logits
    print("\n📊 Output Analysis:")
    logits = embeddings['variable_logits']
    probs = embeddings['variable_probs']
    
    print("  Variable Logits:", logits)
    print("  Variable Probs:", probs)
    print(f"  Most likely: {mapper.get_name(int(jnp.argmax(probs)))}")
    
    # Visualize embeddings using PCA
    print("\n📊 Creating Visualization...")
    
    # Reduce dimensionality for visualization
    from sklearn.decomposition import PCA
    
    # Stack embeddings
    first_layer = np.array(embeddings['first_layer_embeddings'])
    final_layer = np.array(embeddings['final_embeddings'])
    
    # Apply PCA
    pca_first = PCA(n_components=2)
    pca_final = PCA(n_components=2)
    
    first_2d = pca_first.fit_transform(first_layer)
    final_2d = pca_final.fit_transform(final_layer)
    
    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: First layer embeddings
    ax = axes[0]
    colors = ['red', 'blue', 'green']
    for i in range(3):
        var_name = mapper.get_name(i)
        ax.scatter(first_2d[i, 0], first_2d[i, 1], 
                  c=colors[i], s=200, label=var_name, alpha=0.7)
        ax.annotate(var_name, (first_2d[i, 0], first_2d[i, 1]),
                   fontsize=12, ha='center')
    ax.set_title('First Layer Embeddings (PCA)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Final embeddings
    ax = axes[1]
    for i in range(3):
        var_name = mapper.get_name(i)
        ax.scatter(final_2d[i, 0], final_2d[i, 1], 
                  c=colors[i], s=200, label=var_name, alpha=0.7)
        ax.annotate(var_name, (final_2d[i, 0], final_2d[i, 1]),
                   fontsize=12, ha='center')
    ax.set_title('Final Embeddings (PCA)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Variable probabilities
    ax = axes[2]
    var_names = [mapper.get_name(i) for i in range(3)]
    ax.bar(var_names, probs, color=colors, alpha=0.7)
    ax.set_title('Variable Selection Probabilities')
    ax.set_ylabel('Probability')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('experiments/joint-grpo-target-training/embedding_analysis.png', dpi=150)
    print(f"  Saved visualization to embedding_analysis.png")
    
    # Test gradient flow
    print("\n🔬 Testing Gradient Flow:")
    
    def loss_fn(params):
        """Simple loss to test gradients."""
        output = policy_fn.apply(params, tensor, mapper.target_idx)
        # Loss that should prefer X1 (index 1)
        target_probs = jnp.array([0.1, 0.9, 0.0])  # Want high prob for X1
        if mapper.target_idx == 2:  # X2 is masked
            target_probs = target_probs.at[2].set(0.0)
            target_probs = target_probs / target_probs.sum()
        
        loss = -jnp.sum(target_probs * jnp.log(output['variable_probs'] + 1e-8))
        return loss
    
    loss_val, grads = jax.value_and_grad(loss_fn)(policy_params)
    
    print(f"  Loss value: {loss_val:.4f}")
    
    # Check if gradients are non-zero
    grad_norms = jax.tree_map(lambda x: jnp.linalg.norm(x), grads)
    
    # Flatten and check
    all_norms = jax.tree_leaves(grad_norms)
    total_norm = sum(all_norms)
    zero_grads = sum(1 for norm in all_norms if norm < 1e-8)
    
    print(f"  Total gradient norm: {total_norm:.6f}")
    print(f"  Zero gradient layers: {zero_grads}/{len(all_norms)}")
    
    if total_norm < 1e-6:
        print("  ⚠️ WARNING: Gradients are nearly zero! Learning will not happen.")
    else:
        print("  ✅ Gradients are flowing properly.")
    
    # Check key parameter gradients
    print("\n  Key Parameter Gradients:")
    for key in ['input_projection', 'var_mlp_output', 'val_mlp_output']:
        for param_key in grads.keys():
            if key in param_key:
                param_grad = grads[param_key]
                for sub_key in ['w', 'b']:
                    if sub_key in param_grad:
                        norm = jnp.linalg.norm(param_grad[sub_key])
                        print(f"    {param_key}/{sub_key}: {norm:.6f}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    print("\n🔑 Key Findings:")
    print("1. Check if X0 and X1 have different embeddings")
    print("2. Check if gradients are flowing")
    print("3. Check if the model assigns different probabilities to variables")
    print("4. Look at the visualization to see clustering")
    
    plt.show()
    
    return embeddings, policy_params


if __name__ == "__main__":
    embeddings, params = visualize_embeddings_analysis()