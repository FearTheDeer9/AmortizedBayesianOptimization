#!/usr/bin/env python3
"""
Debug script to verify reward calculation direction.
Tracks Y values before/after intervention and corresponding rewards.
"""

import sys
sys.path.append('.')

import numpy as np
import jax.numpy as jnp
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.causal_bayes_opt.training.unified_grpo_trainer import UnifiedGRPOTrainer
from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm
from src.causal_bayes_opt.acquisition.better_rewards import (
    adaptive_sigmoid_reward, RunningStats, compute_better_clean_reward
)
from src.causal_bayes_opt.data_structures.sample import get_values


class RewardDebugTrainer(UnifiedGRPOTrainer):
    """Modified trainer that logs detailed reward information."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_history = []
        self.y_value_history = []
        
    def _collect_grpo_data(self, episode: int, scm, scm_name: str, rng_key) -> dict:
        """Override to add detailed logging."""
        # Store Y value before any interventions
        baseline_sample = scm.sample(n_samples=1)
        # Get target variable from the SCM metadata
        target_var = getattr(scm, 'target', 'Y')  # Default to 'Y' for fork SCM
        y_before = float(get_values(baseline_sample)[target_var])
        
        # Call parent method
        grpo_data = super()._collect_grpo_data(episode, scm, scm_name, rng_key)
        
        # Log reward details for each transition
        for i, reward in enumerate(grpo_data.get('rewards', [])):
            self.reward_history.append({
                'episode': episode,
                'step': i,
                'reward': float(reward),
                'y_before': y_before,
                'scm': scm_name
            })
        
        return grpo_data


def test_reward_direction():
    """Test reward calculation with detailed logging."""
    
    # Test 1: Direct reward function test
    logger.info("="*60)
    logger.info("TEST 1: Direct Reward Function Test")
    logger.info("="*60)
    
    stats = RunningStats()
    test_values = [-10, -5, 0, 5, 10]
    
    for val in test_values:
        stats.update(val)
    
    logger.info(f"Stats: mean={stats.mean:.2f}, std={stats.std:.2f}")
    logger.info("\nFor MINIMIZE direction (lower is better):")
    
    for val in test_values:
        reward = adaptive_sigmoid_reward(val, stats, 'MINIMIZE', temperature_factor=2.0)
        logger.info(f"  Y={val:+6.1f} → reward={reward:.3f} {'✓ GOOD' if val < 0 else '✗ BAD'}")
    
    # Test 2: Full reward computation test
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Full Reward Computation Test")
    logger.info("="*60)
    
    scm = create_fork_scm()
    buffer = scm.sample(n_samples=10)
    
    # Test interventions that decrease Y (should get high reward)
    test_cases = [
        {'X': -2.0},  # Should decrease Y
        {'X': 0.0},   # Neutral
        {'X': 2.0},   # Should increase Y
        {'Z': -2.0},  # Should decrease Y  
        {'Z': 2.0},   # Should increase Y
    ]
    
    stats = RunningStats()
    
    for intervention_values in test_cases:
        var_name = list(intervention_values.keys())[0]
        value = intervention_values[var_name]
        
        # Get baseline Y
        baseline = scm.sample(n_samples=1)
        y_before = float(get_values(baseline)['Y'])
        
        # Apply intervention
        outcome = scm.do(intervention_values).sample(n_samples=1)
        y_after = float(get_values(outcome)['Y'])
        
        # Compute reward
        reward_info = compute_better_clean_reward(
            buffer_before=buffer,
            intervention={
                'targets': frozenset([var_name]),
                'values': intervention_values
            },
            outcome=outcome,
            target_variable='Y',
            config={
                'optimization_direction': 'MINIMIZE',
                'reward_type': 'adaptive_sigmoid',
                'temperature_factor': 2.0,
                'weights': {'target': 1.0}
            },
            stats=stats
        )
        
        reward = reward_info['target']
        delta_y = y_after - y_before
        
        logger.info(f"\nIntervention: {var_name}={value:+.1f}")
        logger.info(f"  Y: {y_before:.3f} → {y_after:.3f} (Δ={delta_y:+.3f})")
        logger.info(f"  Reward: {reward:.3f}")
        logger.info(f"  {'✓ CORRECT' if (delta_y < 0 and reward > 0.5) or (delta_y > 0 and reward < 0.5) else '✗ WRONG'}")
    
    # Test 3: Training with detailed logging
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Training Reward Tracking")
    logger.info("="*60)
    
    trainer = RewardDebugTrainer(
        learning_rate=1e-2,
        n_episodes=10,
        episode_length=5,
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
            'patience': 1000,
            'min_episodes': 100,
            'max_episodes_per_scm': 100
        }
    )
    
    scms = {'fork': create_fork_scm()}
    result = trainer.train(scms)
    
    # Analyze reward history
    rewards_by_episode = defaultdict(list)
    for entry in trainer.reward_history:
        rewards_by_episode[entry['episode']].append(entry['reward'])
    
    logger.info("\nReward progression by episode:")
    for ep in sorted(rewards_by_episode.keys()):
        rewards = rewards_by_episode[ep]
        mean_reward = np.mean(rewards)
        logger.info(f"  Episode {ep}: mean_reward={mean_reward:.3f}, "
                   f"min={min(rewards):.3f}, max={max(rewards):.3f}")
    
    # Check if rewards are increasing (they should for MINIMIZE)
    episode_means = [np.mean(rewards_by_episode[ep]) for ep in sorted(rewards_by_episode.keys())]
    if len(episode_means) >= 2:
        trend = episode_means[-1] - episode_means[0]
        logger.info(f"\nOverall trend: {trend:+.3f} ({'✓ IMPROVING' if trend > 0 else '✗ DEGRADING'})")
    
    # Check reward calculation consistency
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    
    # Get some metrics from training
    all_metrics = result.get('all_metrics', [])
    if all_metrics:
        first_metric = all_metrics[0]
        last_metric = all_metrics[-1]
        
        logger.info(f"Training metrics:")
        logger.info(f"  First episode mean_reward: {first_metric.get('mean_reward', 0):.3f}")
        logger.info(f"  Last episode mean_reward: {last_metric.get('mean_reward', 0):.3f}")
        logger.info(f"  Change: {last_metric.get('mean_reward', 0) - first_metric.get('mean_reward', 0):+.3f}")
    
    logger.info("\nKey insights:")
    logger.info("1. Check if lower Y values get higher rewards (for MINIMIZE)")
    logger.info("2. Check if mean_reward increases over time")
    logger.info("3. Check if the policy learns to choose interventions that decrease Y")


if __name__ == "__main__":
    test_reward_direction()