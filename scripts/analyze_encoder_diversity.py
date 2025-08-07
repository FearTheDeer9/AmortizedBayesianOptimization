#!/usr/bin/env python3
"""
Analyze diversity metrics from trained encoder models.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from src.causal_bayes_opt.utils.checkpoint_utils import load_checkpoint
from src.causal_bayes_opt.training.data_preprocessing import load_demonstrations_from_path, preprocess_demonstration_batch
from src.causal_bayes_opt.avici_integration.continuous.configurable_model import ConfigurableContinuousParentSetPredictionModel
from src.causal_bayes_opt.avici_integration.continuous.model import ContinuousParentSetPredictionModel
import haiku as hk

# Load some test data
demo_path = 'expert_demonstrations/raw/raw_demonstrations'
raw_demos = load_demonstrations_from_path(demo_path, max_files=2)
test_data = []
for demo in raw_demos:
    preprocessed = preprocess_demonstration_batch([demo])
    if preprocessed['surrogate_data']:
        test_data.extend(preprocessed['surrogate_data'])

print(f"Loaded {len(test_data)} test examples")

# Analyze each encoder
checkpoint_dir = Path('encoder_comparison_results/checkpoints')
encoders = ['node_feature', 'node', 'simple']

for encoder_type in encoders:
    checkpoint_path = checkpoint_dir / f'surrogate_{encoder_type}_checkpoint'
    if not checkpoint_path.exists():
        print(f"\nSkipping {encoder_type} - checkpoint not found")
        continue
        
    print(f"\n{'='*60}")
    print(f"Analyzing {encoder_type.upper()} encoder")
    print(f"{'='*60}")
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    params = checkpoint['params']
    
    # Create model function
    def model_fn(data, target_idx, is_training=False):
        if encoder_type == 'node':
            model = ContinuousParentSetPredictionModel(
                hidden_dim=128,
                num_layers=4,
                num_heads=8,
                key_size=32,
                dropout=0.1
            )
        else:
            model = ConfigurableContinuousParentSetPredictionModel(
                hidden_dim=128,
                num_layers=4,
                num_heads=8,
                key_size=32,
                dropout=0.1,
                encoder_type=encoder_type,
                attention_type='pairwise' if encoder_type == 'node_feature' else 'original'
            )
        return model(data, target_idx, is_training)
    
    # Transform to Haiku
    net = hk.transform(model_fn)
    
    # Analyze predictions
    all_predictions = []
    all_embeddings = []
    
    for i, example in enumerate(test_data[:5]):  # Analyze first 5 examples
        output = net.apply(
            params,
            jax.random.PRNGKey(0),
            example.state_tensor,
            example.target_idx,
            False
        )
        
        parent_probs = output['parent_probabilities']
        all_predictions.append(parent_probs)
        
        if 'node_embeddings' in output:
            embeddings = output['node_embeddings']
            all_embeddings.append(embeddings)
        
        # Print example details
        if i == 0:
            print(f"\nExample prediction for target variable {example.target_idx}:")
            print(f"  Probabilities: {parent_probs}")
            print(f"  Max prob: {jnp.max(parent_probs):.4f}")
            print(f"  Std dev: {jnp.std(parent_probs):.4f}")
            print(f"  Entropy: {-jnp.sum(parent_probs * jnp.log(parent_probs + 1e-8)):.4f}")
    
    # Compute overall metrics
    if all_predictions:
        prediction_stds = [float(jnp.std(p)) for p in all_predictions]
        max_probs = [float(jnp.max(p)) for p in all_predictions]
        entropies = [float(-jnp.sum(p * jnp.log(p + 1e-8))) for p in all_predictions]
        
        print(f"\nOverall metrics:")
        print(f"  Mean prediction std: {jnp.mean(jnp.array(prediction_stds)):.4f}")
        print(f"  Mean max probability: {jnp.mean(jnp.array(max_probs)):.4f}")
        print(f"  Mean entropy: {jnp.mean(jnp.array(entropies)):.4f}")
    
    # Compute embedding similarity if available
    if all_embeddings:
        embedding_similarities = []
        for embeddings in all_embeddings:
            # Compute pairwise cosine similarities
            norms = jnp.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / (norms + 1e-8)
            similarity_matrix = jnp.dot(normalized, normalized.T)
            # Get upper triangle (excluding diagonal)
            n = similarity_matrix.shape[0]
            upper_indices = jnp.triu_indices(n, k=1)
            similarities = similarity_matrix[upper_indices]
            embedding_similarities.extend(similarities)
        
        print(f"  Mean embedding similarity: {jnp.mean(jnp.array(embedding_similarities)):.4f}")
        print(f"  Max embedding similarity: {jnp.max(jnp.array(embedding_similarities)):.4f}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("\nThe node_feature encoder should show:")
print("- Higher prediction std (>0.3 vs ~0.05)")
print("- Lower embedding similarity (<0.5 vs >0.95)")
print("- More diverse predictions overall")