#!/usr/bin/env python3
"""
Debug why chain and collider SCMs show 0% improvement.
"""

import sys
sys.path.append('.')

import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.causal_bayes_opt.training.unified_grpo_trainer import UnifiedGRPOTrainer
from src.causal_bayes_opt.experiments.benchmark_scms import (
    create_fork_scm, create_chain_scm, create_collider_scm
)


def debug_scm_training():
    """Debug SCM rotation and metric collection."""
    
    # Create all three SCMs
    scms = {
        'fork': create_fork_scm(),
        'chain': create_chain_scm(), 
        'collider': create_collider_scm()
    }
    
    logger.info(f"Created {len(scms)} SCMs: {list(scms.keys())}")
    
    # Create trainer with short config
    trainer = UnifiedGRPOTrainer(
        learning_rate=3e-4,
        n_episodes=30,  # 10 episodes per SCM
        episode_length=10,
        batch_size=16,
        use_early_stopping=False,
        reward_weights={
            'optimization': 1.0,
            'discovery': 0.0,
            'efficiency': 0.0,
            'info_gain': 0.0
        },
        optimization_direction="MINIMIZE",
        use_surrogate=False,
        seed=42,
        convergence_config={
            'enabled': False,
            'patience': 1000,
            'min_episodes': 30,
            'max_episodes_per_scm': 10  # Force rotation every 10 episodes
        }
    )
    
    # Train
    result = trainer.train(scms)
    
    # Analyze metrics
    all_metrics = result.get('all_metrics', [])
    logger.info(f"\nTotal metrics collected: {len(all_metrics)}")
    
    # Count metrics by SCM
    scm_counts = defaultdict(int)
    scm_episodes = defaultdict(list)
    
    for i, m in enumerate(all_metrics):
        scm_type = m.get('scm_type', 'unknown')
        episode = m.get('episode', i)
        scm_counts[scm_type] += 1
        scm_episodes[scm_type].append(episode)
        
        # Log first few to see the pattern
        if i < 10:
            logger.info(f"Metric {i}: episode={episode}, scm_type={scm_type}, reward={m.get('mean_reward', 'N/A')}")
    
    logger.info(f"\nMetrics per SCM:")
    for scm, count in scm_counts.items():
        logger.info(f"  {scm}: {count} metrics (episodes: {scm_episodes[scm][:5]}...)")
    
    # Check episode rotation
    logger.info(f"\nEpisodes per SCM from result: {result.get('episodes_per_scm', {})}")
    
    # Look for the rotation pattern
    logger.info(f"\nSCM rotation pattern (first 30 metrics):")
    current_scm = None
    scm_switches = []
    
    for i, m in enumerate(all_metrics[:30]):
        scm = m.get('scm_type', 'unknown')
        if scm != current_scm:
            scm_switches.append((i, scm))
            current_scm = scm
    
    for episode, scm in scm_switches:
        logger.info(f"  Episode {episode}: switched to {scm}")
    
    # Calculate actual improvements
    logger.info(f"\nCalculating improvements:")
    scm_rewards = {name: [] for name in scms.keys()}
    
    for m in all_metrics:
        scm_name = m.get('scm_type')
        if scm_name and scm_name in scm_rewards and 'mean_reward' in m:
            scm_rewards[scm_name].append(m['mean_reward'])
    
    for scm_name, rewards in scm_rewards.items():
        logger.info(f"\n{scm_name}:")
        logger.info(f"  Rewards collected: {len(rewards)}")
        if rewards:
            logger.info(f"  First 5 rewards: {rewards[:5]}")
            logger.info(f"  Last 5 rewards: {rewards[-5:]}")
            
            if len(rewards) >= 2:
                early = rewards[0]
                late = rewards[-1]
                improvement = (late - early) / abs(early) * 100 if early != 0 else 0
                logger.info(f"  Improvement: {improvement:+.1f}%")


if __name__ == "__main__":
    debug_scm_training()