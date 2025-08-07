#!/usr/bin/env python3
"""
Compare different encoder architectures for BC surrogate model.

This script trains surrogate models with different encoder types and compares:
1. Prediction diversity (std deviation)
2. Embedding similarity
3. True parent ranking accuracy
4. Training dynamics

Usage:
    python scripts/compare_encoder_architectures.py --episodes 100
    python scripts/compare_encoder_architectures.py --episodes 500 --plot
"""

import argparse
import logging
import time
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.training.surrogate_bc_trainer import SurrogateBCTrainer
from src.causal_bayes_opt.utils.checkpoint_utils import load_checkpoint
import jax
import jax.numpy as jnp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_prediction_diversity(trainer: SurrogateBCTrainer, 
                               test_data: List,
                               checkpoint_path: Path) -> Dict[str, float]:
    """
    Compute prediction diversity metrics.
    
    Returns:
        Dictionary with diversity metrics
    """
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    params = checkpoint['params']
    
    all_predictions = []
    embedding_similarities = []
    
    for example in test_data[:10]:  # Sample a few examples
        # Get predictions
        output = trainer.net.apply(
            params, 
            jax.random.PRNGKey(0),
            example.state_tensor,
            example.target_idx,
            False  # not training
        )
        
        parent_probs = output['parent_probabilities']
        all_predictions.append(parent_probs)
        
        # Compute embedding similarity if available
        if 'node_embeddings' in output:
            embeddings = output['node_embeddings']
            # Compute pairwise cosine similarities
            norms = jnp.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / (norms + 1e-8)
            similarity_matrix = jnp.dot(normalized, normalized.T)
            # Get upper triangle (excluding diagonal)
            n = similarity_matrix.shape[0]
            upper_indices = jnp.triu_indices(n, k=1)
            similarities = similarity_matrix[upper_indices]
            embedding_similarities.extend(similarities)
    
    # Compute diversity metrics
    prediction_stds = [float(jnp.std(p)) for p in all_predictions]
    prediction_entropies = [float(-jnp.sum(p * jnp.log(p + 1e-8))) for p in all_predictions]
    max_probs = [float(jnp.max(p)) for p in all_predictions]
    
    prediction_std = np.mean(prediction_stds)
    prediction_entropy = np.mean(prediction_entropies)
    max_prob = np.mean(max_probs)
    
    metrics = {
        'prediction_std': float(prediction_std),
        'prediction_entropy': float(prediction_entropy),
        'max_probability': float(max_prob),
        'mean_embedding_similarity': float(np.mean(embedding_similarities)) if embedding_similarities else 0.0
    }
    
    return metrics


def train_and_evaluate_encoder(encoder_type: str, 
                             config: Dict,
                             demo_path: str,
                             checkpoint_dir: Path) -> Dict[str, any]:
    """
    Train a surrogate model with specified encoder and evaluate it.
    
    Returns:
        Dictionary with training results and metrics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training with encoder type: {encoder_type}")
    logger.info(f"{'='*60}")
    
    # Create trainer
    trainer = SurrogateBCTrainer(
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 4),
        num_heads=config.get('num_heads', 8),
        key_size=config.get('key_size', 32),
        learning_rate=config.get('learning_rate', 1e-3),
        batch_size=config.get('batch_size', 32),
        max_epochs=config.get('max_epochs', 100),
        early_stopping_patience=10,
        validation_split=0.2,
        gradient_clip=1.0,
        dropout=0.1,
        weight_decay=1e-4,
        seed=config.get('seed', 42),
        encoder_type=encoder_type,
        attention_type='pairwise' if encoder_type == 'node_feature' else 'original'
    )
    
    # Train
    start_time = time.time()
    results = trainer.train(demonstrations_path=demo_path, max_demos=config.get('max_demos'))
    training_time = time.time() - start_time
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / f'surrogate_{encoder_type}_checkpoint'
    trainer.save_checkpoint(checkpoint_path, results)
    
    # Compute diversity metrics
    from src.causal_bayes_opt.training.data_preprocessing import (
        load_demonstrations_from_path,
        preprocess_demonstration_batch
    )
    
    # Load test data
    raw_demos = load_demonstrations_from_path(demo_path, max_files=5)
    test_data = []
    for demo in raw_demos:
        preprocessed = preprocess_demonstration_batch([demo])
        if preprocessed['surrogate_data']:
            test_data.extend(preprocessed['surrogate_data'])
    
    diversity_metrics = compute_prediction_diversity(trainer, test_data, checkpoint_path)
    
    # Combine results
    combined_results = {
        'encoder_type': encoder_type,
        'training_time': training_time,
        'final_train_loss': results['metrics']['final_train_loss'],
        'best_val_loss': results['metrics']['best_val_loss'],
        'epochs_trained': results['metrics']['epochs_trained'],
        'train_history': results['metrics']['train_history'],
        'val_history': results['metrics']['val_history'],
        **diversity_metrics
    }
    
    logger.info(f"\nResults for {encoder_type}:")
    logger.info(f"  Training time: {training_time:.1f}s")
    logger.info(f"  Final train loss: {combined_results['final_train_loss']:.4f}")
    logger.info(f"  Best val loss: {combined_results['best_val_loss']:.4f}")
    logger.info(f"  Prediction std: {combined_results['prediction_std']:.4f}")
    logger.info(f"  Prediction entropy: {combined_results['prediction_entropy']:.4f}")
    logger.info(f"  Max probability: {combined_results['max_probability']:.4f}")
    logger.info(f"  Embedding similarity: {combined_results['mean_embedding_similarity']:.4f}")
    
    return combined_results


def plot_comparison(results: Dict[str, Dict], output_dir: Path):
    """Create comparison plots for different encoders."""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Encoder Architecture Comparison', fontsize=16)
    
    # Extract encoder types and metrics
    encoder_types = list(results.keys())
    colors = ['blue', 'red', 'green', 'orange']
    
    # 1. Training curves
    ax = axes[0, 0]
    for i, (encoder, data) in enumerate(results.items()):
        train_history = data['train_history']
        epochs = range(1, len(train_history) + 1)
        ax.plot(epochs, train_history, label=f'{encoder} (train)', 
                color=colors[i], linestyle='-', alpha=0.7)
        if data['val_history']:
            ax.plot(epochs, data['val_history'], label=f'{encoder} (val)', 
                    color=colors[i], linestyle='--', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Prediction diversity
    ax = axes[0, 1]
    stds = [data['prediction_std'] for data in results.values()]
    x_pos = np.arange(len(encoder_types))
    bars = ax.bar(x_pos, stds, color=colors[:len(encoder_types)])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(encoder_types, rotation=45)
    ax.set_ylabel('Prediction Std Dev')
    ax.set_title('Prediction Diversity')
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, stds)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    # 3. Prediction entropy
    ax = axes[0, 2]
    entropies = [data['prediction_entropy'] for data in results.values()]
    bars = ax.bar(x_pos, entropies, color=colors[:len(encoder_types)])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(encoder_types, rotation=45)
    ax.set_ylabel('Entropy')
    ax.set_title('Prediction Entropy')
    ax.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, entropies)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    # 4. Max probability
    ax = axes[1, 0]
    max_probs = [data['max_probability'] for data in results.values()]
    bars = ax.bar(x_pos, max_probs, color=colors[:len(encoder_types)])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(encoder_types, rotation=45)
    ax.set_ylabel('Max Probability')
    ax.set_title('Maximum Parent Probability')
    ax.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, max_probs)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    # 5. Embedding similarity
    ax = axes[1, 1]
    similarities = [data['mean_embedding_similarity'] for data in results.values()]
    bars = ax.bar(x_pos, similarities, color=colors[:len(encoder_types)])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(encoder_types, rotation=45)
    ax.set_ylabel('Mean Cosine Similarity')
    ax.set_title('Embedding Similarity')
    ax.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, similarities)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    # 6. Training efficiency
    ax = axes[1, 2]
    times = [data['training_time'] for data in results.values()]
    val_losses = [data['best_val_loss'] for data in results.values()]
    scatter = ax.scatter(times, val_losses, s=100, c=colors[:len(encoder_types)])
    for i, encoder in enumerate(encoder_types):
        ax.annotate(encoder, (times[i], val_losses[i]), 
                   xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Training Time (s)')
    ax.set_ylabel('Best Validation Loss')
    ax.set_title('Training Efficiency')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'encoder_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved comparison plot to {output_path}")
    plt.close()
    
    # Create summary table
    summary_path = output_dir / 'encoder_comparison_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Encoder Architecture Comparison Summary\n")
        f.write("="*60 + "\n\n")
        
        for encoder, data in results.items():
            f.write(f"{encoder.upper()} Encoder:\n")
            f.write(f"  Training time: {data['training_time']:.1f}s\n")
            f.write(f"  Final train loss: {data['final_train_loss']:.4f}\n")
            f.write(f"  Best val loss: {data['best_val_loss']:.4f}\n")
            f.write(f"  Prediction std: {data['prediction_std']:.4f}\n")
            f.write(f"  Prediction entropy: {data['prediction_entropy']:.4f}\n")
            f.write(f"  Max probability: {data['max_probability']:.4f}\n")
            f.write(f"  Embedding similarity: {data['mean_embedding_similarity']:.4f}\n")
            f.write("\n")
    
    logger.info(f"Saved summary to {summary_path}")


def main():
    """Main comparison script."""
    parser = argparse.ArgumentParser(description='Compare encoder architectures')
    parser.add_argument('--episodes', type=int, default=100, help='Training episodes')
    parser.add_argument('--demo_path', type=str, 
                       default='expert_demonstrations/raw/raw_demonstrations',
                       help='Path to demonstrations')
    parser.add_argument('--max_demos', type=int, default=None,
                       help='Max number of demonstrations to use')
    parser.add_argument('--plot', action='store_true', help='Create comparison plots')
    parser.add_argument('--encoders', nargs='+', 
                       default=['node_feature', 'node', 'simple'],
                       choices=['node_feature', 'node', 'simple', 'improved'],
                       help='Encoder types to compare')
    parser.add_argument('--output_dir', type=str, default='encoder_comparison_results',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Configuration
    config = {
        'max_epochs': args.episodes,
        'max_demos': args.max_demos,
        'seed': args.seed,
        'hidden_dim': 128,
        'num_layers': 4,
        'num_heads': 8,
        'key_size': 32,
        'learning_rate': 1e-3,
        'batch_size': 32
    }
    
    logger.info("Starting encoder architecture comparison")
    logger.info(f"Encoders to test: {args.encoders}")
    logger.info(f"Training episodes: {args.episodes}")
    logger.info(f"Output directory: {output_dir}")
    
    # Train and evaluate each encoder
    results = {}
    for encoder_type in args.encoders:
        try:
            results[encoder_type] = train_and_evaluate_encoder(
                encoder_type, config, args.demo_path, checkpoint_dir
            )
        except Exception as e:
            logger.error(f"Failed to train {encoder_type}: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            continue
    
    # Check if we have any results
    if not results:
        logger.error("No encoders were successfully trained!")
        return {}
    
    # Print comparison summary
    logger.info("\n" + "="*60)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*60)
    
    # Find best encoder for each metric
    metrics_to_compare = [
        ('prediction_std', 'highest'),  # Higher is better for diversity
        ('prediction_entropy', 'highest'),  # Higher is better
        ('best_val_loss', 'lowest'),  # Lower is better
        ('mean_embedding_similarity', 'lowest'),  # Lower is better for diversity
        ('training_time', 'lowest')  # Lower is better
    ]
    
    for metric, direction in metrics_to_compare:
        values = [(encoder, results[encoder].get(metric, float('inf'))) 
                  for encoder in results.keys()]
        
        if direction == 'highest':
            best_encoder, best_value = max(values, key=lambda x: x[1])
        else:
            best_encoder, best_value = min(values, key=lambda x: x[1])
        
        logger.info(f"Best {metric}: {best_encoder} ({best_value:.4f})")
    
    # Create plots if requested
    if args.plot and len(results) > 1:
        plot_comparison(results, output_dir)
    
    logger.info(f"\nComparison complete! Results saved to {output_dir}")
    
    # Return results for potential further analysis
    return results


if __name__ == "__main__":
    main()