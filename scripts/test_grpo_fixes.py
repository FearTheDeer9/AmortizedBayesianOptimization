#!/usr/bin/env python3
"""
Test the fixes for GRPO policy getting stuck on variable X.
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.training.unified_grpo_trainer import UnifiedGRPOTrainer

# Configure logging to see our debug messages
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG to see baseline logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_grpo_with_fixes():
    """Test GRPO training with exploration noise and non-intervention baseline."""
    
    print("="*80)
    print("TESTING GRPO WITH FIXES")
    print("="*80)
    print("\nFixes implemented:")
    print("1. Exploration noise (0.3) added to variable selection")
    print("2. Non-intervention baseline for advantages")
    print("3. Logging to track exploration behavior")
    print("\n" + "="*80)
    
    # Create trainer with small config for quick testing
    trainer = UnifiedGRPOTrainer(
        learning_rate=3e-4,
        n_episodes=3,  # Just 3 episodes to see baseline logs
        episode_length=10,
        batch_size=8,  # Small batch
        architecture_level="baseline",
        optimization_direction="MINIMIZE",
        seed=123,
        use_surrogate=False,  # Test without surrogate first
        checkpoint_dir="checkpoints/test_grpo_fixes",
        reward_weights={
            'optimization': 0.8,
            'discovery': 0.2,  # Some diversity reward
            'efficiency': 0.1,
            'info_gain': 0.0
        }
    )
    
    print("\nStarting training...")
    print("Watch for [EXPLORATION] logs to see variable selection diversity")
    print("Watch for [BASELINE] logs to see non-intervention baseline")
    print("\n")
    
    # Create simple SCMs for testing
    from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm
    
    # Use a simple fork SCM (X -> Y <- Z)
    scms = {
        "fork": create_fork_scm()
    }
    
    # Train
    metrics = trainer.train(scms)
    
    # Analyze results
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    if 'history' in metrics:
        history = metrics['history']
        
        # Check if we're exploring different variables
        print("\nChecking exploration diversity...")
        
        # Look at episode rewards
        episode_rewards = [ep['mean_reward'] for ep in history if 'mean_reward' in ep]
        if episode_rewards:
            print(f"\nEpisode rewards over time:")
            for i, reward in enumerate(episode_rewards[:10]):  # First 10
                print(f"  Episode {i}: {reward:.3f}")
            
            # Check if rewards are improving
            if len(episode_rewards) > 5:
                early_avg = sum(episode_rewards[:5]) / 5
                late_avg = sum(episode_rewards[-5:]) / 5
                improvement = ((late_avg - early_avg) / abs(early_avg)) * 100 if early_avg != 0 else 0
                
                print(f"\nLearning progress:")
                print(f"  Early episodes (0-4): {early_avg:.3f}")
                print(f"  Late episodes (last 5): {late_avg:.3f}")
                print(f"  Improvement: {improvement:+.1f}%")
                
                if improvement > 5:
                    print("  ✓ Policy is learning!")
                else:
                    print("  ⚠️  Limited improvement - may need more episodes")
    
    # Check final checkpoint
    final_checkpoint = Path("checkpoints/test_grpo_fixes/unified_grpo_final")
    if final_checkpoint.exists():
        print(f"\n✓ Final checkpoint saved at: {final_checkpoint}")
    else:
        print("\n⚠️  No final checkpoint found")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Check logs above for [EXPLORATION] entries showing variable diversity")
    print("2. Check [BASELINE] entries to confirm non-intervention baseline is used")
    print("3. If still stuck on X, increase exploration_noise (currently 0.3)")
    print("4. Try with use_surrogate=True for info gain rewards")


if __name__ == "__main__":
    test_grpo_with_fixes()