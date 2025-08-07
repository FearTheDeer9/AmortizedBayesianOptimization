#!/usr/bin/env python3
"""
Compare the original vs improved encoder implementations to verify benefits are maintained.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from src.causal_bayes_opt.avici_integration.continuous.model import ContinuousParentSetPredictionModel
from src.causal_bayes_opt.avici_integration.continuous.configurable_model import ConfigurableContinuousParentSetPredictionModel
from src.causal_bayes_opt.training.data_preprocessing import load_demonstrations_from_path, preprocess_demonstration_batch

print("Comparing Original vs Improved Encoder Implementations")
print("="*80)

# Load some real data for testing
demo_path = 'expert_demonstrations/raw/raw_demonstrations'
raw_demos = load_demonstrations_from_path(demo_path, max_files=3)
test_data = []
for demo in raw_demos:
    preprocessed = preprocess_demonstration_batch([demo])
    if preprocessed['surrogate_data']:
        test_data.extend(preprocessed['surrogate_data'])

print(f"Loaded {len(test_data)} test examples")

# Test both implementations
results = {}

for implementation in ['original', 'improved']:
    print(f"\n{'='*60}")
    print(f"Testing {implementation.upper()} implementation")
    print(f"{'='*60}")
    
    # Create model function
    def model_fn(data, target_idx, is_training=False):
        if implementation == 'original':
            # Original model with default NodeEncoder
            model = ContinuousParentSetPredictionModel(
                hidden_dim=128,
                num_layers=4,
                num_heads=8,
                key_size=32,
                dropout=0.1
            )
        else:
            # Improved model with enhanced NodeFeatureEncoder
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
    
    # Transform and initialize
    net = hk.transform(model_fn)
    key = jax.random.PRNGKey(42)
    
    # Test on multiple examples
    all_predictions = []
    all_embeddings = []
    prediction_stds = []
    max_probs = []
    entropies = []
    
    for i, example in enumerate(test_data[:10]):  # Test first 10 examples
        try:
            # Initialize with this example
            if i == 0:
                params = net.init(key, example.state_tensor, example.target_idx, False)
            
            # Get predictions
            output = net.apply(params, key, example.state_tensor, example.target_idx, False)
            
            parent_probs = output['parent_probabilities']
            all_predictions.append(parent_probs)
            
            # Calculate metrics
            pred_std = float(jnp.std(parent_probs))
            max_prob = float(jnp.max(parent_probs))
            entropy = float(-jnp.sum(parent_probs * jnp.log(parent_probs + 1e-8)))
            
            prediction_stds.append(pred_std)
            max_probs.append(max_prob)
            entropies.append(entropy)
            
            # Store embeddings
            if 'node_embeddings' in output:
                all_embeddings.append(output['node_embeddings'])
            
            # Print first example details
            if i == 0:
                print(f"\nExample prediction for target variable {example.target_idx}:")
                print(f"  Probabilities: {parent_probs}")
                print(f"  Max prob: {max_prob:.4f}")
                print(f"  Std dev: {pred_std:.4f}")
                print(f"  Entropy: {entropy:.4f}")
                
        except Exception as e:
            print(f"Error processing example {i}: {e}")
    
    # Calculate overall metrics
    if prediction_stds:
        mean_std = np.mean(prediction_stds)
        mean_max_prob = np.mean(max_probs)
        mean_entropy = np.mean(entropies)
        
        print(f"\nOverall metrics across {len(prediction_stds)} examples:")
        print(f"  Mean prediction std: {mean_std:.4f}")
        print(f"  Mean max probability: {mean_max_prob:.4f}")
        print(f"  Mean entropy: {mean_entropy:.4f}")
        
        # Calculate embedding similarities
        if all_embeddings:
            similarities = []
            for embeddings in all_embeddings[:5]:  # Check first 5
                norms = jnp.linalg.norm(embeddings, axis=1, keepdims=True)
                normalized = embeddings / (norms + 1e-8)
                similarity_matrix = jnp.dot(normalized, normalized.T)
                n = similarity_matrix.shape[0]
                if n > 1:
                    upper_indices = jnp.triu_indices(n, k=1)
                    sim_values = similarity_matrix[upper_indices]
                    similarities.extend(sim_values)
            
            if similarities:
                print(f"  Mean embedding similarity: {np.mean(similarities):.4f}")
                print(f"  Max embedding similarity: {np.max(similarities):.4f}")
                print(f"  Min embedding similarity: {np.min(similarities):.4f}")
        
        # Store results
        results[implementation] = {
            'mean_std': mean_std,
            'mean_max_prob': mean_max_prob,
            'mean_entropy': mean_entropy,
            'mean_similarity': np.mean(similarities) if similarities else None
        }

# Compare results
print(f"\n{'='*80}")
print("COMPARISON SUMMARY")
print("="*80)

if len(results) == 2:
    print("\nMetric                  Original    Improved    Change")
    print("-" * 55)
    
    metrics = [
        ('Prediction Std Dev', 'mean_std', True),  # Higher is better
        ('Max Probability', 'mean_max_prob', False),  # Context dependent
        ('Entropy', 'mean_entropy', True),  # Higher means more diverse
        ('Embedding Similarity', 'mean_similarity', False)  # Lower is better
    ]
    
    for metric_name, key, higher_better in metrics:
        orig_val = results['original'].get(key, 0)
        impr_val = results['improved'].get(key, 0)
        if orig_val and impr_val:
            change = ((impr_val - orig_val) / orig_val) * 100
            better = (change > 0 and higher_better) or (change < 0 and not higher_better)
            symbol = "↑" if change > 0 else "↓"
            color = "✓" if better else "✗"
            print(f"{metric_name:<20} {orig_val:>8.4f}    {impr_val:>8.4f}    {change:>+6.1f}% {symbol} {color}")
    
    # Overall assessment
    print("\n" + "="*55)
    orig_std = results['original']['mean_std']
    impr_std = results['improved']['mean_std']
    improvement = ((impr_std - orig_std) / orig_std) * 100
    
    if impr_std > orig_std:
        print(f"✓ DIVERSITY IMPROVED: {improvement:.1f}% increase in prediction std dev")
        print(f"  Original: {orig_std:.4f} → Improved: {impr_std:.4f}")
    else:
        print(f"✗ DIVERSITY DECREASED: {improvement:.1f}% change in prediction std dev")
        print(f"  Original: {orig_std:.4f} → Improved: {impr_std:.4f}")
else:
    print("Could not compare - missing results")

print("\nConclusion:")
if 'improved' in results and results['improved']['mean_std'] > 0.3:
    print("✓ The improved encoder maintains high diversity (std > 0.3)")
    print("✓ Benefits from the architecture improvements are preserved")
else:
    print("⚠ Further investigation needed")