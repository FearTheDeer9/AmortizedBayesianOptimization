#!/usr/bin/env python3
"""
Test GRPO with a simple fix to ensure SCM rotation happens.
"""

import sys
sys.path.append('.')

import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.causal_bayes_opt.training.unified_grpo_trainer import UnifiedGRPOTrainer
from src.causal_bayes_opt.experiments.benchmark_scms import (
    create_fork_scm, create_chain_scm, create_collider_scm
)


def test_with_forced_rotation():
    """Test GRPO with forced SCM rotation."""
    
    # Create all three SCMs
    scms = {
        'fork': create_fork_scm(),
        'chain': create_chain_scm(), 
        'collider': create_collider_scm()
    }
    
    logger.info("Testing two approaches:")
    logger.info("1. With convergence detection (forces rotation)")
    logger.info("2. Without convergence (stuck on first SCM)")
    
    results = {}
    
    # Test 1: Enable convergence with low threshold to force rotation
    logger.info("\n" + "="*60)
    logger.info("Test 1: Convergence enabled (forces rotation)")
    logger.info("="*60)
    
    trainer1 = UnifiedGRPOTrainer(
        learning_rate=3e-3,
        n_episodes=30,
        episode_length=10,
        batch_size=16,
        use_early_stopping=False,
        reward_weights={'optimization': 1.0, 'discovery': 0.0, 'efficiency': 0.0, 'info_gain': 0.0},
        optimization_direction="MINIMIZE",
        use_surrogate=False,
        seed=42,
        convergence_config={
            'enabled': True,  # Enable convergence
            'patience': 5,    # Low patience to force rotation
            'min_episodes': 5, # Min episodes before checking
            'max_episodes_per_scm': 10  # Force rotation after 10 episodes
        }
    )
    
    result1 = trainer1.train(scms)
    results['with_convergence'] = analyze_results(result1, scms)
    
    # Test 2: Disable convergence (will get stuck)
    logger.info("\n" + "="*60)
    logger.info("Test 2: Convergence disabled (stuck on fork)")
    logger.info("="*60)
    
    trainer2 = UnifiedGRPOTrainer(
        learning_rate=3e-3,
        n_episodes=30,
        episode_length=10,
        batch_size=16,
        use_early_stopping=False,
        reward_weights={'optimization': 1.0, 'discovery': 0.0, 'efficiency': 0.0, 'info_gain': 0.0},
        optimization_direction="MINIMIZE",
        use_surrogate=False,
        seed=42,
        convergence_config={
            'enabled': False  # Disable convergence
        }
    )
    
    result2 = trainer2.train(scms)
    results['without_convergence'] = analyze_results(result2, scms)
    
    # Compare results
    logger.info("\n" + "="*60)
    logger.info("COMPARISON")
    logger.info("="*60)
    
    for test_name, test_results in results.items():
        logger.info(f"\n{test_name}:")
        for scm_name, data in test_results.items():
            logger.info(f"  {scm_name}: {data['count']} episodes, improvement: {data['improvement']:+.1f}%")


def analyze_results(result, scms):
    """Analyze training results."""
    all_metrics = result.get('all_metrics', [])
    
    # Count metrics and calculate improvements per SCM
    scm_data = {name: {'rewards': [], 'count': 0} for name in scms.keys()}
    
    for m in all_metrics:
        scm_name = m.get('scm_type')
        if scm_name and scm_name in scm_data and 'mean_reward' in m:
            scm_data[scm_name]['rewards'].append(m['mean_reward'])
            scm_data[scm_name]['count'] += 1
    
    # Calculate improvements
    for scm_name, data in scm_data.items():
        rewards = data['rewards']
        if len(rewards) >= 2:
            early = rewards[0]
            late = rewards[-1]
            data['improvement'] = (late - early) / abs(early) * 100 if early != 0 else 0
        else:
            data['improvement'] = 0.0
    
    return scm_data


if __name__ == "__main__":
    test_with_forced_rotation()