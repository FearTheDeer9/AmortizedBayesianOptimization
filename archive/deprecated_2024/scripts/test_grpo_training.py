#!/usr/bin/env python3
"""
Quick GRPO training test to verify posterior collapse fix.

This script runs a short training session and monitors:
1. Channel statistics (logged every 100 episodes)
2. Policy output differentiation
3. Learning metrics (F1, SHD, rewards)
4. Parameter updates
"""

import logging
import sys
from pathlib import Path
import time
from omegaconf import OmegaConf
import jax
import jax.numpy as jnp
import optax

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.training.enriched_trainer import EnrichedGRPOTrainer

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('grpo_training_test.log')
    ]
)

logger = logging.getLogger(__name__)


def run_training_test(n_episodes=200, log_interval=10):
    """Run a short training session with monitoring."""
    
    logger.info("\n" + "="*80)
    logger.info("GRPO TRAINING TEST - POSTERIOR COLLAPSE FIX VERIFICATION")
    logger.info("="*80)
    
    # Load config
    config_path = project_root / "config" / "grpo_enriched_acbo_training.yaml"
    config = OmegaConf.load(config_path)
    
    # Modify for testing
    config.training.n_episodes = n_episodes
    config.training.episode_length = 20  # Shorter episodes for faster testing
    config.training.learning_rate = 0.0005  # Good learning rate
    config.logging.level = "INFO"
    
    # Enable structure learning metrics
    config.training.track_structure_metrics = True
    
    # Fix reward weights config
    if 'discovery' not in config.training.reward_weights:
        config.training.reward_weights.discovery = config.training.reward_weights.get('structure', 0.5)
    if 'efficiency' not in config.training.reward_weights:
        config.training.reward_weights.efficiency = config.training.reward_weights.get('exploration', 0.1)
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Episodes: {config.training.n_episodes}")
    logger.info(f"  Episode length: {config.training.episode_length}")
    logger.info(f"  Learning rate: {config.training.learning_rate}")
    logger.info(f"  Channels: {config.training.state_config.num_channels}")
    
    # Create trainer
    logger.info("\nInitializing trainer...")
    trainer = EnrichedGRPOTrainer(config)
    
    # Track metrics
    episode_metrics = []
    initial_param_norm = float(optax.global_norm(trainer.policy_params))
    
    logger.info(f"\nInitial parameter norm: {initial_param_norm:.8f}")
    logger.info("\nStarting training...\n")
    
    start_time = time.time()
    
    # Run training manually to monitor progress
    for episode in range(n_episodes):
        episode_key = jax.random.PRNGKey(episode)
        
        try:
            # Run episode
            metrics = trainer._run_episode(episode, episode_key)
            episode_metrics.append(metrics)
            
            # Log progress
            if episode % log_interval == 0:
                current_param_norm = float(optax.global_norm(trainer.policy_params))
                param_change = abs(current_param_norm - initial_param_norm)
                
                log_msg = f"Episode {episode:3d}: "
                log_msg += f"reward={metrics.mean_reward:6.3f}, "
                log_msg += f"param_Δ={param_change:.6f}, "
                
                if metrics.f1_score is not None:
                    log_msg += f"F1={metrics.f1_score:.3f}, "
                if metrics.shd is not None:
                    log_msg += f"SHD={metrics.shd}, "
                if metrics.true_parent_likelihood is not None:
                    log_msg += f"P(parents)={metrics.true_parent_likelihood:.3f}"
                
                logger.info(log_msg)
                
                # Check for learning progress
                if episode > 50 and param_change < 1e-6:
                    logger.warning("⚠️ Parameters not changing - possible learning issue!")
                
        except Exception as e:
            logger.error(f"Episode {episode} failed: {e}")
            raise
    
    # Training complete
    elapsed_time = time.time() - start_time
    final_param_norm = float(optax.global_norm(trainer.policy_params))
    total_param_change = abs(final_param_norm - initial_param_norm)
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    
    logger.info(f"\nTraining time: {elapsed_time:.1f}s ({elapsed_time/n_episodes:.2f}s per episode)")
    logger.info(f"Total parameter change: {total_param_change:.8f}")
    
    # Analyze metrics
    if episode_metrics:
        rewards = [m.mean_reward for m in episode_metrics]
        f1_scores = [m.f1_score for m in episode_metrics if m.f1_score is not None]
        shd_values = [m.shd for m in episode_metrics if m.shd is not None]
        
        logger.info(f"\nReward progression:")
        logger.info(f"  Initial: {rewards[0]:.3f}")
        logger.info(f"  Final: {rewards[-1]:.3f}")
        logger.info(f"  Mean: {jnp.mean(jnp.array(rewards)):.3f}")
        logger.info(f"  Max: {jnp.max(jnp.array(rewards)):.3f}")
        
        if f1_scores:
            logger.info(f"\nF1 Score progression:")
            logger.info(f"  Initial: {f1_scores[0]:.3f}")
            logger.info(f"  Final: {f1_scores[-1]:.3f}")
            logger.info(f"  Max: {jnp.max(jnp.array(f1_scores)):.3f}")
            
        if shd_values:
            logger.info(f"\nSHD progression:")
            logger.info(f"  Initial: {shd_values[0]}")
            logger.info(f"  Final: {shd_values[-1]}")
            logger.info(f"  Min: {jnp.min(jnp.array(shd_values))}")
    
    # Check for key issues
    logger.info("\n" + "="*80)
    logger.info("DIAGNOSTICS")
    logger.info("="*80)
    
    issues = []
    
    if total_param_change < 1e-6:
        issues.append("❌ Parameters did not change - no learning occurred!")
    else:
        logger.info(f"✅ Parameters updated successfully (change: {total_param_change:.8f})")
    
    if rewards and all(r == rewards[0] for r in rewards):
        issues.append("❌ Rewards did not change - policy not exploring!")
    else:
        logger.info("✅ Rewards varied - policy is exploring")
    
    if f1_scores and max(f1_scores) < 0.1:
        issues.append("❌ F1 scores very low - structure learning not working!")
    elif f1_scores:
        logger.info(f"✅ Structure learning showing progress (max F1: {max(f1_scores):.3f})")
    
    if issues:
        logger.error("\n⚠️ ISSUES DETECTED:")
        for issue in issues:
            logger.error(f"  {issue}")
    else:
        logger.info("\n🎉 All checks passed! The fix appears to be working.")
    
    # Save final checkpoint
    logger.info("\nSaving checkpoint...")
    checkpoint_path = trainer.checkpoint_manager.save_checkpoint(
        trainer.policy_params,
        trainer.policy_config,
        n_episodes,
        episode_metrics[-1] if episode_metrics else None,
        is_final=True
    )
    logger.info(f"Checkpoint saved to: {checkpoint_path}")
    
    return len(issues) == 0


def main():
    """Run the training test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test GRPO training with posterior collapse fix")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes to run")
    parser.add_argument("--log-interval", type=int, default=10, help="Episode interval for logging")
    args = parser.parse_args()
    
    success = run_training_test(n_episodes=args.episodes, log_interval=args.log_interval)
    
    if success:
        logger.info("\n✅ Training test completed successfully!")
        return 0
    else:
        logger.error("\n❌ Training test revealed issues!")
        return 1


if __name__ == "__main__":
    sys.exit(main())