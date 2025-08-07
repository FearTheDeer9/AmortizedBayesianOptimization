#!/usr/bin/env python3
"""
Test GRPO with proper SCM rotation using max_episodes_per_scm.
"""

import sys
sys.path.append('.')

import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.causal_bayes_opt.training.unified_grpo_trainer import UnifiedGRPOTrainer
from src.causal_bayes_opt.experiments.benchmark_scms import (
    create_fork_scm, create_chain_scm, create_collider_scm
)


def test_with_proper_config():
    """Test GRPO with proper rotation config."""
    
    scms = {
        'fork': create_fork_scm(),
        'chain': create_chain_scm(), 
        'collider': create_collider_scm()
    }
    
    # Total episodes = 60, so 20 per SCM
    episodes_per_scm = 20
    total_episodes = episodes_per_scm * len(scms)
    
    logger.info(f"Testing with {total_episodes} total episodes ({episodes_per_scm} per SCM)")
    
    # The key insight: enable convergence but with high patience so it uses max_episodes_per_scm
    trainer = UnifiedGRPOTrainer(
        learning_rate=3e-3,
        n_episodes=total_episodes,
        episode_length=20,
        batch_size=32,
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
            'enabled': True,  # Must be True for rotation to work!
            'patience': 1000,  # High patience so it doesn't converge early
            'min_episodes': episodes_per_scm,  # Don't check until we've done enough
            'max_episodes_per_scm': episodes_per_scm  # Force rotation after this many
        }
    )
    
    result = trainer.train(scms)
    
    # Analyze results
    all_metrics = result.get('all_metrics', [])
    scm_rewards = {name: [] for name in scms.keys()}
    
    for m in all_metrics:
        scm_name = m.get('scm_type')
        if scm_name and scm_name in scm_rewards and 'mean_reward' in m:
            scm_rewards[scm_name].append(m['mean_reward'])
    
    # Report results
    logger.info("\n" + "="*60)
    logger.info("RESULTS WITH PROPER ROTATION")
    logger.info("="*60)
    
    for scm_name, rewards in scm_rewards.items():
        logger.info(f"\n{scm_name}:")
        logger.info(f"  Episodes trained: {len(rewards)}")
        
        if len(rewards) >= 2:
            early = np.mean(rewards[:5]) if len(rewards) >= 5 else rewards[0]
            late = np.mean(rewards[-5:]) if len(rewards) >= 5 else rewards[-1]
            improvement = (late - early) / abs(early) * 100 if early != 0 else 0
            
            logger.info(f"  Early reward (avg first 5): {early:.3f}")
            logger.info(f"  Late reward (avg last 5): {late:.3f}")
            logger.info(f"  Improvement: {improvement:+.1f}%")
            
            # Also show the trend
            if len(rewards) >= 10:
                # Fit a line to see if there's a consistent trend
                x = np.arange(len(rewards))
                slope, intercept = np.polyfit(x, rewards, 1)
                logger.info(f"  Trend slope: {slope:.6f} (positive = improving)")
                
                # R-squared to see how consistent the trend is
                y_pred = slope * x + intercept
                ss_res = np.sum((rewards - y_pred) ** 2)
                ss_tot = np.sum((rewards - np.mean(rewards)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                logger.info(f"  Trend consistency (R²): {r_squared:.3f}")
        else:
            logger.info(f"  Not enough data for analysis!")
    
    # Show rotation pattern
    logger.info("\n" + "="*60)
    logger.info("SCM ROTATION PATTERN")
    logger.info("="*60)
    
    current_scm = None
    switches = []
    
    for i, m in enumerate(all_metrics):
        scm = m.get('scm_type')
        if scm != current_scm:
            switches.append((i, scm))
            current_scm = scm
    
    for episode, scm in switches:
        logger.info(f"Episode {episode}: switched to {scm}")


if __name__ == "__main__":
    test_with_proper_config()