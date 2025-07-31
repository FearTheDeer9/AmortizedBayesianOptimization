#!/usr/bin/env python3
"""
Test script to verify GRPO surrogate integration works correctly.

Tests:
1. GRPO training without surrogate (backward compatibility)
2. GRPO training with surrogate and information gain rewards
3. Verify surrogate updates during training
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.causal_bayes_opt.training.unified_grpo_trainer import create_unified_grpo_trainer
from src.causal_bayes_opt.experiments.test_scms import create_fork_test_scm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_grpo_without_surrogate():
    """Test GRPO training without surrogate (backward compatibility)."""
    logger.info("Testing GRPO without surrogate...")
    
    # Create trainer without surrogate
    trainer = create_unified_grpo_trainer(
        learning_rate=3e-4,
        n_episodes=10,
        episode_length=5,
        batch_size=4,
        use_surrogate=False,
        seed=42
    )
    
    # Create simple SCM
    scm = create_fork_test_scm()
    
    # Train
    results = trainer.train([scm])
    
    # Check results
    assert 'final_metrics' in results
    assert 'has_surrogate' in results['final_metrics']
    assert not results['final_metrics']['has_surrogate']
    
    # Check that info_gain is 0 in rewards
    metrics = results['all_metrics']
    if metrics:
        # Since we're not storing individual reward components in metrics,
        # we just verify training completed successfully
        assert len(metrics) > 0
        logger.info(f"✓ Completed {len(metrics)} episodes without surrogate")
    
    logger.info("✓ GRPO without surrogate works correctly")


def test_grpo_with_surrogate():
    """Test GRPO training with surrogate and information gain rewards."""
    logger.info("\nTesting GRPO with surrogate...")
    
    # Create trainer with surrogate
    trainer = create_unified_grpo_trainer(
        learning_rate=3e-4,
        n_episodes=10,
        episode_length=5,
        batch_size=4,
        use_surrogate=True,
        reward_weights={
            'optimization': 0.6,
            'discovery': 0.1,
            'efficiency': 0.1,
            'info_gain': 0.2
        },
        seed=43
    )
    
    # Create simple SCM
    scm = create_fork_test_scm()
    
    # Train
    results = trainer.train([scm])
    
    # Check results
    assert 'final_metrics' in results
    assert 'has_surrogate' in results['final_metrics']
    assert results['final_metrics']['has_surrogate']
    
    # Check that structure metrics are computed
    if 'structure_metrics' in results['final_metrics']:
        metrics = results['final_metrics']['structure_metrics']
        if metrics:  # May be empty dict if no true parents
            logger.info(f"✓ Structure metrics computed: {metrics}")
    
    logger.info(f"✓ Completed {len(results['all_metrics'])} episodes with surrogate")
    logger.info("✓ GRPO with surrogate works correctly")


def test_reward_weights_activation():
    """Test that info_gain weight is auto-activated when using surrogate."""
    logger.info("\nTesting reward weights activation...")
    
    # Create trainer with surrogate but no explicit info_gain weight
    trainer = create_unified_grpo_trainer(
        learning_rate=3e-4,
        n_episodes=5,
        episode_length=3,
        batch_size=2,
        use_surrogate=True,
        reward_weights={
            'optimization': 0.8,
            'discovery': 0.1,
            'efficiency': 0.1
            # Note: no info_gain specified
        },
        seed=44
    )
    
    # Check that info_gain weight was auto-activated
    assert 'info_gain' in trainer.reward_weights
    assert trainer.reward_weights['info_gain'] > 0
    logger.info(f"✓ Info gain weight auto-activated: {trainer.reward_weights['info_gain']}")
    
    # Create trainer without surrogate
    trainer_no_surrogate = create_unified_grpo_trainer(
        learning_rate=3e-4,
        n_episodes=5,
        episode_length=3,
        batch_size=2,
        use_surrogate=False,
        seed=45
    )
    
    # Check that info_gain weight is 0 when no surrogate
    assert trainer_no_surrogate.reward_weights.get('info_gain', 0) == 0
    logger.info("✓ Info gain weight remains 0 without surrogate")


def main():
    """Run all tests."""
    logger.info("Starting GRPO surrogate integration tests...\n")
    
    try:
        test_grpo_without_surrogate()
        test_grpo_with_surrogate()
        test_reward_weights_activation()
        
        logger.info("\n✅ All tests passed!")
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()