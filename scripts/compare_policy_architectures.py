#!/usr/bin/env python3
"""
Compare simple vs attention policy architectures on multi-SCM training.

This script tests both architectures side by side to see which one
learns better causal relationships.
"""

import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import logging
import time

# Set up logging with DEBUG for more details
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Reduce JAX/XLA noise
logging.getLogger('jax').setLevel(logging.WARNING)
logging.getLogger('jax._src').setLevel(logging.WARNING)

from src.causal_bayes_opt.training.unified_grpo_trainer import UnifiedGRPOTrainer
from src.causal_bayes_opt.experiments.benchmark_scms import (
    create_fork_scm, create_chain_scm, create_collider_scm
)


def create_test_scms():
    """Create a small set of SCMs for testing."""
    return {
        'fork': create_fork_scm(),      # X -> Y <- Z
        'chain': create_chain_scm(),    # X -> Y -> Z
        'collider': create_collider_scm()  # X -> Z <- Y
    }


def run_architecture_comparison(architecture: str, scms: Dict) -> Dict:
    """Run training with specified architecture."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training with {architecture.upper()} architecture")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    
    # Track embedding statistics
    embedding_stats = []
    
    # Create trainer with specified architecture
    trainer = UnifiedGRPOTrainer(
        learning_rate=3e-4,
        n_episodes=200,  # Full run to match earlier success
        episode_length=20,
        batch_size=32,
        architecture_level=architecture,  # "simple" or "attention"
        use_early_stopping=False,
        convergence_config={
            'enabled': False,  # Disable convergence detection
            'patience': 1000,  # High patience to avoid early stopping
            'min_episodes': 200,  # Ensure minimum episodes
            'max_episodes_per_scm': 100  # Allow many episodes per SCM
        },
        reward_weights={
            'optimization': 0.8,
            'discovery': 0.2,
            'efficiency': 0.0,
            'info_gain': 0.0
        },
        optimization_direction="MINIMIZE",
        use_surrogate=False,
        seed=42
    )
    
    # Train with embedding monitoring
    result = trainer.train(scms)
    
    # Log final embedding statistics if available
    if 'embedding_stats' in result:
        embedding_stats = result['embedding_stats']
        logger.info(f"\nEmbedding Statistics for {architecture}:")
        for stat in embedding_stats[-5:]:  # Last 5 episodes
            logger.info(f"  Episode {stat.get('episode', '?')}: ")
            logger.info(f"    Mean std: {stat.get('mean_std', 0):.6f}")
            logger.info(f"    Max std: {stat.get('max_std', 0):.6f}")
            logger.info(f"    Min std: {stat.get('min_std', 0):.6f}")
    
    training_time = time.time() - start_time
    
    # Extract metrics
    all_metrics = result.get('all_metrics', [])
    
    # Organize by SCM
    scm_rewards = {name: [] for name in scms.keys()}
    
    # Track SCM for each metric
    for i, m in enumerate(all_metrics):
        # Determine which SCM this metric belongs to
        scm_name = m.get('scm_type', None)
        
        if scm_name and scm_name in scm_rewards and 'mean_reward' in m:
            scm_rewards[scm_name].append(m['mean_reward'])
    
    # Log how many episodes each SCM got
    for scm_name, rewards in scm_rewards.items():
        logger.info(f"{architecture} - {scm_name}: {len(rewards)} episodes")
    
    # Calculate improvements
    improvements = {}
    for scm_name, rewards in scm_rewards.items():
        if len(rewards) >= 2:
            early = np.mean(rewards[:5]) if len(rewards) >= 5 else rewards[0]
            late = np.mean(rewards[-5:]) if len(rewards) >= 5 else rewards[-1]
            improvements[scm_name] = (late - early) / early * 100
        else:
            improvements[scm_name] = 0.0
    
    return {
        'architecture': architecture,
        'training_time': training_time,
        'scm_rewards': scm_rewards,
        'improvements': improvements,
        'all_metrics': all_metrics,
        'embedding_stats': embedding_stats
    }


def plot_comparison(results_simple: Dict, results_attention: Dict):
    """Plot comparison of both architectures."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    scm_names = ['fork', 'chain', 'collider']
    
    # Plot learning curves
    for idx, scm_name in enumerate(scm_names):
        ax = axes[0, idx]
        
        # Simple architecture
        rewards_simple = results_simple['scm_rewards'].get(scm_name, [])
        if rewards_simple:
            ax.plot(rewards_simple, 'b-', label='Simple', linewidth=2, alpha=0.7)
        
        # Attention architecture
        rewards_attention = results_attention['scm_rewards'].get(scm_name, [])
        if rewards_attention:
            ax.plot(rewards_attention, 'r-', label='Attention', linewidth=2, alpha=0.7)
        
        ax.set_title(f'{scm_name.capitalize()} SCM')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Mean Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot improvements
    ax_improve = axes[1, 0]
    improvements_simple = list(results_simple['improvements'].values())
    improvements_attention = list(results_attention['improvements'].values())
    
    x = np.arange(len(scm_names))
    width = 0.35
    
    ax_improve.bar(x - width/2, improvements_simple, width, label='Simple', alpha=0.7)
    ax_improve.bar(x + width/2, improvements_attention, width, label='Attention', alpha=0.7)
    ax_improve.set_xlabel('SCM Type')
    ax_improve.set_ylabel('Improvement (%)')
    ax_improve.set_title('Learning Improvement by Architecture')
    ax_improve.set_xticks(x)
    ax_improve.set_xticklabels(scm_names)
    ax_improve.legend()
    ax_improve.grid(True, alpha=0.3, axis='y')
    
    # Training time comparison
    ax_time = axes[1, 1]
    times = [results_simple['training_time'], results_attention['training_time']]
    ax_time.bar(['Simple', 'Attention'], times, alpha=0.7)
    ax_time.set_ylabel('Training Time (s)')
    ax_time.set_title('Training Time Comparison')
    ax_time.grid(True, alpha=0.3, axis='y')
    
    # Average improvement
    ax_avg = axes[1, 2]
    avg_simple = np.mean(improvements_simple)
    avg_attention = np.mean(improvements_attention)
    ax_avg.bar(['Simple', 'Attention'], [avg_simple, avg_attention], alpha=0.7)
    ax_avg.set_ylabel('Average Improvement (%)')
    ax_avg.set_title('Average Performance Gain')
    ax_avg.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate([avg_simple, avg_attention]):
        ax_avg.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    # Plot embedding statistics if available
    if results_simple.get('embedding_stats') or results_attention.get('embedding_stats'):
        # Plot embedding std over time
        ax_embed = axes[2, 0]
        
        if results_simple.get('embedding_stats'):
            episodes = [s.get('episode', i) for i, s in enumerate(results_simple['embedding_stats'])]
            mean_stds = [s.get('mean_std', 0) for s in results_simple['embedding_stats']]
            ax_embed.plot(episodes, mean_stds, 'b-', label='Simple', linewidth=2)
        
        if results_attention.get('embedding_stats'):
            episodes = [s.get('episode', i) for i, s in enumerate(results_attention['embedding_stats'])]
            mean_stds = [s.get('mean_std', 0) for s in results_attention['embedding_stats']]
            ax_embed.plot(episodes, mean_stds, 'r-', label='Attention', linewidth=2)
        
        ax_embed.set_xlabel('Episode')
        ax_embed.set_ylabel('Mean Embedding Std')
        ax_embed.set_title('Embedding Variance Over Time')
        ax_embed.legend()
        ax_embed.grid(True, alpha=0.3)
        ax_embed.set_yscale('log')
    
    # Hide unused subplots
    axes[2, 1].axis('off')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('architecture_comparison.png', dpi=150, bbox_inches='tight')
    logger.info("Comparison plot saved to architecture_comparison.png")


def main():
    logger.info("=" * 80)
    logger.info("POLICY ARCHITECTURE COMPARISON")
    logger.info("=" * 80)
    
    # Create test SCMs
    scms = create_test_scms()
    
    # Run with simple architecture
    results_simple = run_architecture_comparison("simple", scms)
    
    # Run with attention architecture
    results_attention = run_architecture_comparison("attention", scms)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 80)
    
    # Compare improvements
    for scm_name in scms.keys():
        imp_simple = results_simple['improvements'].get(scm_name, 0)
        imp_attention = results_attention['improvements'].get(scm_name, 0)
        
        logger.info(f"\n{scm_name}:")
        logger.info(f"  Simple:    {imp_simple:+.1f}%")
        logger.info(f"  Attention: {imp_attention:+.1f}%")
        logger.info(f"  Winner:    {'Attention' if imp_attention > imp_simple else 'Simple'}")
    
    # Overall comparison
    avg_imp_simple = np.mean(list(results_simple['improvements'].values()))
    avg_imp_attention = np.mean(list(results_attention['improvements'].values()))
    
    logger.info(f"\nOverall Average Improvement:")
    logger.info(f"  Simple:    {avg_imp_simple:+.1f}%")
    logger.info(f"  Attention: {avg_imp_attention:+.1f}%")
    logger.info(f"  Winner:    {'Attention' if avg_imp_attention > avg_imp_simple else 'Simple'}")
    
    logger.info(f"\nTraining Time:")
    logger.info(f"  Simple:    {results_simple['training_time']:.1f}s")
    logger.info(f"  Attention: {results_attention['training_time']:.1f}s")
    logger.info(f"  Overhead:  {(results_attention['training_time'] / results_simple['training_time'] - 1) * 100:.1f}%")
    
    # Plot comparison
    plot_comparison(results_simple, results_attention)
    
    # Save detailed results
    with open('architecture_comparison_results.txt', 'w') as f:
        f.write("ARCHITECTURE COMPARISON RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        for arch_results in [results_simple, results_attention]:
            f.write(f"{arch_results['architecture'].upper()} ARCHITECTURE\n")
            f.write("-" * 30 + "\n")
            f.write(f"Training time: {arch_results['training_time']:.2f}s\n")
            f.write("\nImprovements by SCM:\n")
            for scm, imp in arch_results['improvements'].items():
                f.write(f"  {scm}: {imp:+.1f}%\n")
            f.write(f"\nAverage improvement: {np.mean(list(arch_results['improvements'].values())):+.1f}%\n")
            f.write("\n")
    
    logger.info("\nDetailed results saved to architecture_comparison_results.txt")


if __name__ == "__main__":
    main()