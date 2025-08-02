#!/usr/bin/env python3
"""
3-Minute Enhanced ACBO with GRPO Validation Script.

This script runs a 3-minute validation session to verify:
1. Enhanced ACBO with GRPO learning works properly
2. Rewards are being computed correctly
3. Learning curves show improvement over time
4. WandB logging is functioning

Run with: poetry run python scripts/run_3min_validation.py
"""

import logging
import time
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_wandb():
    """Setup WandB logging for validation."""
    try:
        import wandb
        
        wandb.init(
            project="enhanced_acbo_3min_validation",
            name=f"validation_run_{int(time.time())}",
            config={
                "validation_duration_minutes": 3,
                "architecture_level": "full",
                "training_steps": 30,
                "episodes_per_step": 1,
                "interventions_per_episode": 5,
                "learning_rate": 0.0003
            },
            tags=["enhanced", "acbo", "grpo", "3min_validation", "full_architecture"],
            group="validation_runs"
        )
        
        logger.info("✅ WandB initialized successfully")
        return True
        
    except ImportError:
        logger.warning("⚠️ WandB not available - install with: pip install wandb")
        return False
    except Exception as e:
        logger.error(f"❌ WandB initialization failed: {e}")
        return False

def create_validation_scm():
    """Create a validation SCM for testing."""
    from causal_bayes_opt.experiments.benchmark_graphs import create_erdos_renyi_scm
    
    # Create a moderate-sized problem for validation
    scm = create_erdos_renyi_scm(
        n_nodes=4,  # Small but not trivial
        edge_prob=0.4,  # Moderate connectivity
        noise_scale=0.1,
        seed=42  # Reproducible
    )
    
    logger.info(f"Created validation SCM with {len(scm['variables'])} variables")
    return scm

def create_validation_config():
    """Create validation configuration."""
    import sys
    from pathlib import Path
    
    # Add examples directory to path
    project_root = Path(__file__).parent.parent
    examples_path = project_root / "examples"
    if str(examples_path) not in sys.path:
        sys.path.insert(0, str(examples_path))
    
    from demo_learning import DemoConfig
    
    config = DemoConfig(
        n_observational_samples=10,  # Quick startup
        n_intervention_steps=5,      # Moderate episode length
        learning_rate=0.0003,        # Standard learning rate
        random_seed=42               # Reproducible
    )
    
    return config

def run_validation():
    """Run the 3-minute validation session."""
    logger.info("=" * 60)
    logger.info("🚀 ENHANCED ACBO WITH GRPO - 3 MINUTE VALIDATION")
    logger.info("=" * 60)
    
    # Setup WandB
    wandb_enabled = setup_wandb()
    
    try:
        # Create validation environment
        logger.info("📊 Creating validation environment...")
        scm = create_validation_scm()
        config = create_validation_config()
        
        # Import and run Enhanced ACBO with GRPO
        from causal_bayes_opt.experiments.enhanced_acbo_with_grpo import run_enhanced_acbo_with_grpo_learning
        
        logger.info("🧠 Starting Enhanced ACBO with GRPO learning...")
        logger.info("⏱️ Target duration: ~3 minutes (30 training steps)")
        
        start_time = time.time()
        
        # Run Enhanced ACBO with GRPO
        results = run_enhanced_acbo_with_grpo_learning(
            scm=scm,
            config=config,
            architecture_level="full",        # Use full architecture
            n_training_steps=30,             # Should take ~3 minutes at ~6s/step
            enable_wandb=wandb_enabled,
            learning_rate=0.0003
        )
        
        total_time = time.time() - start_time
        
        # Validate results
        logger.info("✅ Training completed! Validating results...")
        
        # Check basic result structure
        required_keys = ['method', 'training_history', 'final_results', 'total_time']
        for key in required_keys:
            if key not in results:
                raise ValueError(f"Missing required result key: {key}")
        
        # Analyze training history
        history = results['training_history']
        if len(history) < 5:
            logger.warning(f"⚠️ Short training history: {len(history)} steps")
        
        # Extract reward progression
        rewards = [step['avg_reward'] for step in history if 'avg_reward' in step]
        policy_losses = [step['policy_loss'] for step in history if 'policy_loss' in step]
        
        if rewards:
            initial_reward = rewards[0]
            final_reward = rewards[-1]
            max_reward = max(rewards)
            reward_improvement = final_reward - initial_reward
            
            logger.info(f"📈 REWARD ANALYSIS:")
            logger.info(f"   Initial Reward: {initial_reward:.4f}")
            logger.info(f"   Final Reward: {final_reward:.4f}")
            logger.info(f"   Max Reward: {max_reward:.4f}")
            logger.info(f"   Improvement: {reward_improvement:.4f}")
            
            # Log final metrics to WandB
            if wandb_enabled:
                import wandb
                wandb.log({
                    'validation/total_time_minutes': total_time / 60,
                    'validation/final_reward': final_reward,
                    'validation/max_reward': max_reward,
                    'validation/reward_improvement': reward_improvement,
                    'validation/training_steps_completed': len(history),
                    'validation/architecture_level': "full",
                    'validation/enhanced_networks_used': True
                })
        
        if policy_losses:
            initial_loss = policy_losses[0]
            final_loss = policy_losses[-1]
            
            logger.info(f"📉 POLICY LOSS ANALYSIS:")
            logger.info(f"   Initial Loss: {initial_loss:.4f}")
            logger.info(f"   Final Loss: {final_loss:.4f}")
            logger.info(f"   Loss Change: {final_loss - initial_loss:.4f}")
        
        # Performance metrics
        final_results = results['final_results']
        logger.info(f"🎯 FINAL PERFORMANCE:")
        logger.info(f"   Final Avg Reward: {final_results.get('final_avg_reward', 'N/A'):.4f}")
        logger.info(f"   Target Improvement: {final_results.get('final_target_improvement', 'N/A'):.4f}")
        logger.info(f"   Enhanced Architecture: {final_results.get('enhanced_architecture_validated', False)}")
        
        # Timing analysis
        logger.info(f"⏱️ TIMING ANALYSIS:")
        logger.info(f"   Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        logger.info(f"   Target Time: ~180s (3 minutes)")
        logger.info(f"   Training Steps: {len(history)}")
        logger.info(f"   Avg Time/Step: {total_time/len(history):.1f}s")
        
        # Enhanced features validation
        enhanced_features = results.get('enhanced_features', {})
        logger.info(f"🔧 ENHANCED FEATURES:")
        logger.info(f"   Architecture Level: {enhanced_features.get('architecture_level', 'unknown')}")
        logger.info(f"   GRPO Learning: {enhanced_features.get('grpo_learning_enabled', False)}")
        logger.info(f"   Real Networks: {enhanced_features.get('real_enhanced_networks', False)}")
        logger.info(f"   Policy Updates: {enhanced_features.get('policy_updates', 0)}")
        
        # Success criteria
        success_criteria = []
        
        # 1. Training completed without errors
        success_criteria.append(("Training Completed", True))
        
        # 2. Reasonable timing (within 2-4 minutes)
        timing_ok = 120 <= total_time <= 240  # 2-4 minutes tolerance
        success_criteria.append(("Timing Reasonable", timing_ok))
        
        # 3. Learning evidence (some reward improvement or loss decrease)
        learning_evidence = False
        if rewards and len(rewards) > 5:
            # Check if there's any upward trend in rewards
            mid_point = len(rewards) // 2
            early_avg = sum(rewards[:mid_point]) / mid_point
            late_avg = sum(rewards[mid_point:]) / (len(rewards) - mid_point)
            learning_evidence = late_avg > early_avg
        
        success_criteria.append(("Learning Evidence", learning_evidence))
        
        # 4. Enhanced networks used
        enhanced_used = enhanced_features.get('real_enhanced_networks', False)
        success_criteria.append(("Enhanced Networks", enhanced_used))
        
        # 5. GRPO learning enabled
        grpo_enabled = enhanced_features.get('grpo_learning_enabled', False)
        success_criteria.append(("GRPO Learning", grpo_enabled))
        
        # Summary
        logger.info("=" * 60)
        logger.info("📋 VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        passed = 0
        total = len(success_criteria)
        
        for criterion, passed_check in success_criteria:
            status = "✅ PASS" if passed_check else "❌ FAIL"
            logger.info(f"{status} {criterion}")
            if passed_check:
                passed += 1
        
        success_rate = passed / total
        logger.info(f"\n📊 Overall Success Rate: {passed}/{total} ({success_rate*100:.1f}%)")
        
        if success_rate >= 0.8:  # 80% success threshold
            logger.info("🎉 VALIDATION SUCCESSFUL!")
            logger.info("✅ Enhanced ACBO with GRPO is ready for longer training runs")
            return True
        else:
            logger.warning("⚠️ VALIDATION CONCERNS")
            logger.warning("Some criteria failed - review before long training runs")
            return False
            
    except Exception as e:
        logger.error(f"❌ Validation failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    finally:
        # Cleanup WandB
        if wandb_enabled:
            try:
                import wandb
                wandb.finish()
                logger.info("✅ WandB session finished")
            except:
                pass

def main():
    """Main entry point."""
    logger.info("Starting 3-minute Enhanced ACBO with GRPO validation...")
    
    success = run_validation()
    
    if success:
        logger.info("\n🎯 NEXT STEPS:")
        logger.info("1. ✅ 3-minute validation passed")
        logger.info("2. 🚀 Ready for 3-hour production training")
        logger.info("3. 📊 Use scripts/run_3hour_production.py for full training")
        exit(0)
    else:
        logger.error("\n⚠️ VALIDATION ISSUES:")
        logger.error("1. ❌ 3-minute validation had concerns")
        logger.error("2. 🔍 Review logs and fix issues before production run")
        logger.error("3. 🔧 Consider debugging with test_enhanced_fixes.py")
        exit(1)

if __name__ == "__main__":
    main()