#!/usr/bin/env python3
"""
Test the clean reward system in isolation.

This script demonstrates how the clean reward components work
without running full training.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import jax.numpy as jnp
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.data_structures.sample import create_sample
from src.causal_bayes_opt.acquisition.clean_rewards import (
    compute_clean_reward,
    compute_target_reward,
    compute_diversity_reward,
    compute_exploration_reward
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_buffer():
    """Create a test buffer with some samples."""
    buffer = ExperienceBuffer()
    
    # Add observational samples
    logger.info("Adding observational samples...")
    for i in range(20):
        values = {
            'X': float(i % 5),
            'Y': float(2 * (i % 5) + 1),  # Y = 2X + 1
            'Z': float((i % 5) ** 2)
        }
        sample = create_sample(values=values)
        buffer.add_observation(sample)
    
    # Add some interventions
    logger.info("Adding intervention samples...")
    
    # Intervention 1: X=3
    from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
    
    int1 = create_perfect_intervention(targets=frozenset(['X']), values={'X': 3.0})
    for i in range(5):
        values = {'X': 3.0, 'Y': 7.0 + i*0.1, 'Z': 9.0}
        sample = create_sample(
            values=values,
            intervention_type='perfect',
            intervention_targets=frozenset(['X'])
        )
        buffer.add_intervention(int1, sample)
    
    # Intervention 2: X=1
    int2 = create_perfect_intervention(targets=frozenset(['X']), values={'X': 1.0})
    for i in range(3):
        values = {'X': 1.0, 'Y': 3.0 + i*0.1, 'Z': 1.0}
        sample = create_sample(
            values=values,
            intervention_type='perfect',
            intervention_targets=frozenset(['X'])
        )
        buffer.add_intervention(int2, sample)
    
    return buffer


def test_individual_rewards():
    """Test individual reward components."""
    logger.info("\n=== Testing Individual Reward Components ===")
    
    buffer = create_test_buffer()
    target_var = 'Y'
    
    # Test 1: Target reward for good intervention
    logger.info("\n1. Testing target reward (good intervention)...")
    good_outcome = create_sample(values={'X': 0.0, 'Y': 1.0, 'Z': 0.0})
    target_reward = compute_target_reward(
        buffer, target_var, good_outcome, 'MINIMIZE'
    )
    logger.info(f"   Target reward (Y=1.0): {target_reward:.3f}")
    
    # Test 2: Target reward for bad intervention
    logger.info("\n2. Testing target reward (bad intervention)...")
    bad_outcome = create_sample(values={'X': 5.0, 'Y': 11.0, 'Z': 25.0})
    target_reward = compute_target_reward(
        buffer, target_var, bad_outcome, 'MINIMIZE'
    )
    logger.info(f"   Target reward (Y=11.0): {target_reward:.3f}")
    
    # Test 3: Diversity reward for new variable
    logger.info("\n3. Testing diversity reward...")
    new_var_intervention = {'targets': frozenset(['Z']), 'values': {'Z': 5.0}}
    diversity_reward = compute_diversity_reward(buffer, new_var_intervention)
    logger.info(f"   Diversity reward (intervening on Z): {diversity_reward:.3f}")
    
    # Test 4: Diversity reward for already-intervened variable
    old_var_intervention = {'targets': frozenset(['X']), 'values': {'X': 2.0}}
    diversity_reward = compute_diversity_reward(buffer, old_var_intervention)
    logger.info(f"   Diversity reward (intervening on X again): {diversity_reward:.3f}")
    
    # Test 5: Exploration reward for new value
    logger.info("\n4. Testing exploration reward...")
    new_value_intervention = {'targets': frozenset(['X']), 'values': {'X': 10.0}}
    exploration_reward = compute_exploration_reward(buffer, new_value_intervention)
    logger.info(f"   Exploration reward (X=10.0, new value): {exploration_reward:.3f}")
    
    # Test 6: Exploration reward for existing value
    old_value_intervention = {'targets': frozenset(['X']), 'values': {'X': 3.0}}
    exploration_reward = compute_exploration_reward(buffer, old_value_intervention)
    logger.info(f"   Exploration reward (X=3.0, existing value): {exploration_reward:.3f}")


def test_combined_reward():
    """Test the combined reward function."""
    logger.info("\n=== Testing Combined Reward Function ===")
    
    buffer = create_test_buffer()
    target_var = 'Y'
    
    # Scenario 1: Good target value, new variable
    logger.info("\n1. Good target + new variable intervention...")
    intervention1 = {'targets': frozenset(['Z']), 'values': {'Z': 0.0}}
    outcome1 = create_sample(values={'X': 0.5, 'Y': 2.0, 'Z': 0.0})
    
    reward1 = compute_clean_reward(
        buffer, intervention1, outcome1, target_var,
        config={'optimization_direction': 'MINIMIZE'}
    )
    logger.info(f"   Rewards: {reward1}")
    
    # Scenario 2: Bad target value, old variable
    logger.info("\n2. Bad target + repeated variable intervention...")
    intervention2 = {'targets': frozenset(['X']), 'values': {'X': 3.0}}
    outcome2 = create_sample(values={'X': 3.0, 'Y': 7.0, 'Z': 9.0})
    
    reward2 = compute_clean_reward(
        buffer, intervention2, outcome2, target_var,
        config={'optimization_direction': 'MINIMIZE'}
    )
    logger.info(f"   Rewards: {reward2}")
    
    # Scenario 3: Medium target, exploration bonus
    logger.info("\n3. Medium target + exploration bonus...")
    intervention3 = {'targets': frozenset(['X']), 'values': {'X': 15.0}}
    outcome3 = create_sample(values={'X': 15.0, 'Y': 5.0, 'Z': 225.0})
    
    reward3 = compute_clean_reward(
        buffer, intervention3, outcome3, target_var,
        config={'optimization_direction': 'MINIMIZE'}
    )
    logger.info(f"   Rewards: {reward3}")


def main():
    """Run all reward tests."""
    logger.info("Clean Reward System Test")
    logger.info("=" * 50)
    
    test_individual_rewards()
    test_combined_reward()
    
    logger.info("\n" + "=" * 50)
    logger.info("Test complete!")
    logger.info("\nKey insights:")
    logger.info("- Target reward drives optimization (primary objective)")
    logger.info("- Diversity reward encourages trying different variables")
    logger.info("- Exploration reward encourages trying new values")
    logger.info("- Combined reward balances all objectives")


if __name__ == "__main__":
    main()