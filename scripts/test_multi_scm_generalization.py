#!/usr/bin/env python3
"""
Test GRPO training on multiple SCMs to verify generalization.

This tests if the policy can learn to identify causal parents from data
rather than memorizing specific interventions for each SCM type.
"""

import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.causal_bayes_opt.training.unified_grpo_trainer import UnifiedGRPOTrainer
from src.causal_bayes_opt.experiments.benchmark_scms import (
    create_fork_scm, create_chain_scm, create_collider_scm,
    create_diamond_scm, create_mixed_coeff_scm, create_dense_scm
)

def create_diverse_scm_dataset():
    """Create a diverse set of SCMs with different structures."""
    scms = {}
    
    # Basic 3-variable structures with Y as target
    scms['fork'] = create_fork_scm()  # X -> Y <- Z
    scms['chain'] = create_chain_scm()  # X -> Y -> Z  
    scms['collider'] = create_collider_scm()  # X -> Z <- Y
    
    # More complex structures
    scms['diamond'] = create_diamond_scm()  # Diamond structure
    scms['mixed_coeff'] = create_mixed_coeff_scm()  # Different coefficient patterns
    
    # For now, skip dense SCM since it may have different variable names
    # We'll focus on the well-defined structures above
    
    return scms

def analyze_intervention_patterns(metrics: List[Dict], scm_names: List[str]):
    """Analyze which variables the policy intervenes on for each SCM."""
    intervention_counts = {name: {'X': 0, 'Y': 0, 'Z': 0} for name in scm_names}
    
    # This would require parsing logs or tracking interventions
    # For now, we'll focus on reward trends
    
    return intervention_counts

def plot_learning_curves(all_metrics: Dict[str, List[Dict]], save_path: str = 'multi_scm_learning.png'):
    """Plot learning curves for all SCMs."""
    plt.figure(figsize=(15, 10))
    
    n_scms = len(all_metrics)
    cols = 3
    rows = (n_scms + cols - 1) // cols
    
    for idx, (scm_name, metrics) in enumerate(all_metrics.items()):
        plt.subplot(rows, cols, idx + 1)
        
        # Extract rewards - handle both list of dicts and list of floats
        episodes = []
        rewards = []
        
        if metrics and isinstance(metrics[0], dict):
            # Metrics are dictionaries
            for i, m in enumerate(metrics):
                if 'mean_reward' in m:
                    episodes.append(m.get('episode', i))
                    rewards.append(m['mean_reward'])
        elif metrics and isinstance(metrics[0], (float, int)):
            # Metrics are just reward values
            episodes = list(range(len(metrics)))
            rewards = metrics
        
        if rewards:
            plt.plot(episodes, rewards, 'b-', linewidth=2, alpha=0.7)
            
            # Add trend line
            if len(rewards) > 5:
                z = np.polyfit(episodes, rewards, 1)
                p = np.poly1d(z)
                plt.plot(episodes, p(episodes), 'r--', alpha=0.8, 
                        label=f'Trend: {z[0]:.4f}')
            
            # Calculate improvement
            if len(rewards) >= 2:
                improvement = (rewards[-1] - rewards[0]) / rewards[0] * 100
                plt.title(f'{scm_name}\nImprovement: {improvement:.1f}%')
            else:
                plt.title(scm_name)
            
            plt.xlabel('Episode')
            plt.ylabel('Mean Reward')
            plt.grid(True, alpha=0.3)
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No data', ha='center', va='center', 
                    transform=plt.gca().transAxes)
            plt.title(scm_name)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Learning curves saved to {save_path}")

def main():
    logger.info("=" * 80)
    logger.info("MULTI-SCM GENERALIZATION TEST")
    logger.info("=" * 80)
    
    # Create diverse SCM dataset
    scms = create_diverse_scm_dataset()
    logger.info(f"Created {len(scms)} SCMs for training")
    
    # Log SCM details
    for name, scm in scms.items():
        target = scm.get('target', 'Unknown')
        logger.info(f"\n{name}:")
        logger.info(f"  Variables: {list(scm.get('variables', []))}")
        logger.info(f"  Target: {target}")
        logger.info(f"  Edges: {scm.get('edges', [])}")
        if target in ['Y', 'X2', 'Z', 'W', 'D']:  # Known targets
            parents = [e[0] for e in scm.get('edges', []) if e[1] == target]
            logger.info(f"  True parents of {target}: {parents}")
    
    # Create trainer with settings optimized for multi-SCM learning
    trainer = UnifiedGRPOTrainer(
        learning_rate=3e-4,
        n_episodes=50,  # More episodes for multiple SCMs
        episode_length=20,
        batch_size=32,
        architecture_level="baseline",
        use_early_stopping=False,  # Run full episodes to see learning
        reward_weights={
            'optimization': 0.8,  # Slightly less focus on optimization
            'discovery': 0.2,     # More emphasis on exploration
            'efficiency': 0.0,
            'info_gain': 0.0      # Would be 0.3 with surrogate
        },
        optimization_direction="MINIMIZE",
        use_surrogate=False,  # Test without surrogate first
        seed=42
    )
    
    logger.info("\nTraining on multiple SCMs simultaneously...")
    logger.info("This tests if the policy can:")
    logger.info("  1. Learn to identify causal parents from tensor data")
    logger.info("  2. Generalize across different causal structures")
    logger.info("  3. Adapt to different edge weights and relationships")
    
    # Train on all SCMs
    logger.info("\nStarting training loop...")
    result = trainer.train(scms)
    
    # Extract metrics from result
    all_metrics = result.get('all_metrics', [])
    
    # Log raw metrics for debugging
    logger.info(f"\nReceived {len(all_metrics)} total metrics")
    if all_metrics:
        logger.info(f"First metric keys: {list(all_metrics[0].keys())}")
        logger.info(f"Sample metric: {all_metrics[0]}")
    
    # Analyze results
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS ANALYSIS")
    logger.info("=" * 80)
    
    # Group metrics by SCM
    scm_metrics = {name: [] for name in scms.keys()}
    current_scm = None
    episode_counter = 0
    
    # Track rewards per SCM per episode
    scm_episode_rewards = {name: [] for name in scms.keys()}
    
    for m in all_metrics:
        # Try to extract SCM name from metrics
        if 'current_scm' in m:
            current_scm = m['current_scm']
        elif 'scm_name' in m:
            current_scm = m['scm_name']
        elif 'scm_type' in m:
            current_scm = m['scm_type']
        else:
            # If no SCM identifier, try to infer from episode number
            if len(scms) > 0:
                scm_idx = episode_counter % len(scms)
                current_scm = list(scms.keys())[scm_idx]
        
        # Extract reward information
        if 'mean_reward' in m:
            reward = m['mean_reward']
            episode = m.get('episode', episode_counter)
            
            # Log every reward for debugging
            intervened_var = "Unknown"
            if 'structure_metrics' in m and 'intervened_variables' in m['structure_metrics']:
                intervened_var = m['structure_metrics']['intervened_variables']
            elif 'intervened_var' in m:
                intervened_var = m['intervened_var']
            
            logger.info(f"Episode {episode}, SCM: {current_scm}, Reward: {reward:.4f}, Intervened: {intervened_var}")
            
            if current_scm:
                scm_metrics[current_scm].append({
                    'episode': episode,
                    'mean_reward': reward,
                    **m  # Include all other metrics
                })
                scm_episode_rewards[current_scm].append(reward)
            
            episode_counter += 1
    
    # Log summary of collected metrics
    logger.info("\nMetrics summary:")
    for scm_name, rewards in scm_episode_rewards.items():
        logger.info(f"  {scm_name}: {len(rewards)} reward entries")
        if rewards:
            logger.info(f"    First 5 rewards: {rewards[:5]}")
            logger.info(f"    Last 5 rewards: {rewards[-5:]}")
    
    # Analyze each SCM's performance
    overall_improvements = []
    for scm_name, metrics in scm_metrics.items():
        logger.info(f"\nAnalyzing {scm_name}: {len(metrics)} metrics")
        if len(metrics) < 2:
            logger.info(f"  Skipping {scm_name} - not enough data")
            continue
            
        rewards = [m['mean_reward'] for m in metrics if 'mean_reward' in m]
        if len(rewards) >= 2:
            early_avg = np.mean(rewards[:5]) if len(rewards) >= 5 else rewards[0]
            late_avg = np.mean(rewards[-5:]) if len(rewards) >= 5 else rewards[-1]
            improvement = (late_avg - early_avg) / early_avg * 100
            overall_improvements.append(improvement)
            
            logger.info(f"\n{scm_name}:")
            logger.info(f"  Episodes: {len(rewards)}")
            logger.info(f"  Early reward: {early_avg:.3f}")
            logger.info(f"  Late reward: {late_avg:.3f}")
            logger.info(f"  Improvement: {improvement:+.1f}%")
            
            # Check trend
            if len(rewards) > 5:
                x = np.arange(len(rewards))
                slope = np.polyfit(x, rewards, 1)[0]
                logger.info(f"  Trend slope: {slope:.6f}")
    
    # Overall assessment
    if overall_improvements:
        avg_improvement = np.mean(overall_improvements)
        logger.info(f"\nOVERALL PERFORMANCE:")
        logger.info(f"  Average improvement: {avg_improvement:+.1f}%")
        logger.info(f"  SCMs with positive learning: {sum(1 for i in overall_improvements if i > 0)}/{len(overall_improvements)}")
        
        if avg_improvement > 5:
            logger.info("  ✅ Policy shows generalization ability!")
        else:
            logger.info("  ⚠️  Limited generalization observed")
    
    # Plot learning curves
    plot_learning_curves(scm_metrics)
    
    # Also save a text summary
    with open('multi_scm_summary.txt', 'w') as f:
        f.write("MULTI-SCM GENERALIZATION TEST RESULTS\n")
        f.write("=" * 50 + "\n\n")
        for scm_name, rewards in scm_episode_rewards.items():
            f.write(f"{scm_name}:\n")
            f.write(f"  Episodes: {len(rewards)}\n")
            if rewards:
                f.write(f"  First 5 rewards: {rewards[:5]}\n")
                f.write(f"  Last 5 rewards: {rewards[-5:]}\n")
                f.write(f"  Mean reward: {np.mean(rewards):.4f}\n")
                f.write("\n")
    logger.info("Summary saved to multi_scm_summary.txt")
    
    logger.info("\n" + "=" * 80)
    logger.info("KEY INSIGHTS")
    logger.info("=" * 80)
    logger.info("1. Multi-SCM training tests true learning vs memorization")
    logger.info("2. Policy must use tensor data to identify causal relationships")
    logger.info("3. Positive trends across diverse SCMs indicate generalization")
    logger.info("4. With surrogate, info gain would further improve structure discovery")
    
    # Log final stats
    logger.info(f"\nFinal training results:")
    logger.info(f"  Total episodes: {len(all_metrics)}")
    logger.info(f"  Training time: {result.get('training_time', 0):.2f}s")
    logger.info(f"  Episodes per SCM: {result.get('episodes_per_scm', {})}")

if __name__ == "__main__":
    main()