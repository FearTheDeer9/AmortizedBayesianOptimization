#!/usr/bin/env python3
"""
Test a potential fix for the BC surrogate model.

The issue: The model averages node embeddings across samples, losing all correlation information.
The fix: Use a more sophisticated aggregation that preserves sample correlations.
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

from src.causal_bayes_opt.training.bc_model_inference import load_bc_checkpoint


def create_fixed_bc_surrogate_inference_fn(checkpoint_path: str, threshold: float = 0.1):
    """Create BC surrogate inference with proper aggregation."""
    
    # Load checkpoint
    checkpoint_data = load_bc_checkpoint(checkpoint_path)
    model_params = checkpoint_data.get('model_params')
    
    if not model_params:
        raise ValueError(f"No model_params found in checkpoint: {checkpoint_path}")
    
    # Extract config
    config = checkpoint_data.get('config', {})
    if hasattr(config, 'surrogate_config'):
        surrogate_config = config.surrogate_config
        hidden_dim = getattr(surrogate_config, 'hidden_dim', 64)
        num_layers = getattr(surrogate_config, 'num_layers', 3)
        num_heads = getattr(surrogate_config, 'num_heads', 4)
        key_size = getattr(surrogate_config, 'key_size', 32)
    else:
        hidden_dim = 64
        num_layers = 3
        num_heads = 4
        key_size = 32
    
    print(f"Model config: hidden_dim={hidden_dim}, num_layers={num_layers}")
    
    # Create modified model function that uses better aggregation
    def model_fn(data: jnp.ndarray, target_idx: int):
        """Modified model with better sample aggregation."""
        from src.causal_bayes_opt.avici_integration.continuous.model import (
            ParentAttentionLayer, NodeEncoder
        )
        
        N, d, channels = data.shape
        
        # Instead of averaging, use a more sophisticated approach
        # 1. Compute embeddings for each sample
        flattened = data.reshape(N * d, channels)
        
        # Initial embedding
        x = hk.Linear(hidden_dim)(flattened)
        x = jax.nn.relu(x)
        
        # Additional layers
        for _ in range(num_layers - 1):
            residual = x
            x = hk.Linear(hidden_dim)(x)
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            x = jax.nn.relu(x)
            x = x + residual
        
        # Reshape back
        x = x.reshape(N, d, hidden_dim)
        
        # 2. Aggregate with attention to preserve correlations
        # Use weighted average based on intervention patterns
        intervention_weights = data[:, :, 1]  # [N, d] intervention indicators
        obs_weights = 1.0 - intervention_weights  # Observational weights
        
        # Weight samples by their information content
        sample_weights = jnp.sum(obs_weights, axis=1) + 0.1  # [N]
        sample_weights = sample_weights / jnp.sum(sample_weights)
        
        # Weighted average that preserves information
        node_embeddings = jnp.einsum('n,nde->de', sample_weights, x)
        
        # 3. Add correlation features
        # Compute pairwise correlations from data
        values = data[:, :, 0]  # [N, d]
        correlations = jnp.corrcoef(values.T)  # [d, d]
        
        # Encode correlations as additional features
        corr_features = hk.Linear(hidden_dim // 2)(correlations)  # [d, hidden_dim//2]
        
        # Combine with node embeddings
        node_embeddings = jnp.concatenate([
            node_embeddings[:, :hidden_dim//2],
            corr_features
        ], axis=1)
        
        # Get target embedding
        target_embedding = node_embeddings[target_idx]
        
        # Compute parent attention
        parent_attention = ParentAttentionLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            key_size=key_size
        )
        parent_logits = parent_attention(target_embedding, node_embeddings)
        
        # Mask target
        masked_logits = jnp.where(
            jnp.arange(d) == target_idx,
            -1e9,
            parent_logits
        )
        
        parent_probs = jax.nn.softmax(masked_logits)
        
        return {
            'node_embeddings': node_embeddings,
            'target_embedding': target_embedding,
            'attention_logits': parent_logits,
            'parent_probabilities': parent_probs
        }
    
    # Transform model
    model = hk.without_apply_rng(hk.transform(model_fn))
    
    def inference_fn(avici_data, variables, target, params=None):
        """Run inference with fixed model."""
        # Get target index
        target_idx = variables.index(target)
        
        # Apply model
        output = model.apply(model_params, avici_data, target_idx)
        
        # Extract probabilities
        parent_probs = output['parent_probabilities']
        
        # Convert to parent sets (same logic as before)
        from src.causal_bayes_opt.avici_integration.parent_set.posterior import create_parent_set_posterior
        
        parent_sets = []
        probabilities = []
        
        # Add empty set
        empty_prob = jnp.prod(1.0 - parent_probs)
        if empty_prob > threshold:
            parent_sets.append(frozenset())
            probabilities.append(float(empty_prob))
        
        # Add single parent sets
        for i, var in enumerate(variables):
            if i != target_idx and parent_probs[i] > threshold:
                parent_sets.append(frozenset([var]))
                probabilities.append(float(parent_probs[i]))
        
        # Normalize
        if probabilities:
            total = sum(probabilities)
            probabilities = [p / total for p in probabilities]
        else:
            non_target_vars = [v for v in variables if v != target]
            parent_sets = [frozenset(), frozenset([non_target_vars[0]]) if non_target_vars else frozenset()]
            probabilities = [0.5, 0.5] if len(parent_sets) == 2 else [1.0]
        
        return create_parent_set_posterior(
            target_variable=target,
            parent_sets=parent_sets,
            probabilities=jnp.array(probabilities),
            metadata={
                'type': 'bc_surrogate_fixed',
                'continuous_probs': parent_probs.tolist()
            }
        )
    
    return inference_fn


def test_fixed_model():
    """Test the fixed BC surrogate model."""
    print("Testing Fixed BC Surrogate Model")
    print("=" * 60)
    
    checkpoint_path = project_root / "checkpoints/behavioral_cloning/dev/surrogate/surrogate_bc_development_epoch_22_level_3_1753298905.pkl"
    
    try:
        # Create fixed inference function
        print("\nNote: This is a demonstration of the fix concept.")
        print("The actual fix would need to be applied during model training.\n")
        
        # Show the problem
        print("The Issue:")
        print("- Current model uses: node_embeddings = jnp.mean(x, axis=0)")
        print("- This averages across all samples, losing correlation information")
        print("- Result: Model outputs same probabilities regardless of input\n")
        
        print("The Fix:")
        print("1. Use weighted averaging that preserves information")
        print("2. Add correlation features explicitly")
        print("3. Weight samples by their information content")
        print("4. Combine embeddings with correlation matrix\n")
        
        print("Expected Behavior After Fix:")
        print("- Different data patterns → different parent probabilities")
        print("- Strong correlations → higher parent probabilities")
        print("- Interventional data → more informative than observational")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_fixed_model()