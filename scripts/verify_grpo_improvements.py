#!/usr/bin/env python3
"""
Comprehensive test to verify GRPO improvements.
This script demonstrates that the fixes enable proper learning.
"""

import sys
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.training.unified_grpo_trainer import UnifiedGRPOTrainer
from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm, create_chain_scm

# Configure logging to capture intervention patterns
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('grpo_improvement_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def test_grpo_improvements():
    """Test GRPO with exploration and non-intervention baseline."""
    
    print("="*80)
    print("VERIFYING GRPO IMPROVEMENTS")
    print("="*80)
    print("\nThis test will demonstrate:")
    print("1. Policy explores different variables (not stuck on X)")
    print("2. Non-intervention baseline provides learning signal")
    print("3. Policy improves over episodes")
    print("4. Eventually focuses on causal parents")
    print("="*80)
    
    # Test on fork SCM: X -> Y <- Z
    scm = create_fork_scm()
    
    print("\nTesting on Fork SCM:")
    print("- Structure: X -> Y <- Z")
    print("- Target: Y")
    print("- True causal parents: X, Z")
    print("- Expected behavior: Policy should learn to intervene on X and Z")
    
    # Configure trainer
    trainer = UnifiedGRPOTrainer(
        learning_rate=3e-4,
        n_episodes=50,
        episode_length=10,
        batch_size=16,
        architecture_level="baseline",
        optimization_direction="MINIMIZE",
        seed=42,
        use_surrogate=False,  # Test core fixes first
        checkpoint_dir="checkpoints/verify_improvements",
        reward_weights={
            'optimization': 0.8,
            'discovery': 0.2,
            'efficiency': 0.0,
            'info_gain': 0.0
        }
    )
    
    print("\nTraining GRPO for 50 episodes...")
    print("-" * 60)
    
    # Train and collect metrics
    metrics = trainer.train({"fork": scm})
    
    # Analyze results
    print("\n" + "="*80)
    print("RESULTS ANALYSIS")
    print("="*80)
    
    # Parse intervention patterns from log
    intervention_counts = analyze_log_interventions('grpo_improvement_test.log')
    
    # 1. Check exploration diversity
    print("\n1. EXPLORATION DIVERSITY:")
    if intervention_counts:
        total = sum(intervention_counts.values())
        for var, count in sorted(intervention_counts.items()):
            pct = (count / total) * 100
            print(f"   Variable {var}: {count} times ({pct:.1f}%)")
        
        # Calculate diversity score
        probs = [count/total for count in intervention_counts.values()]
        entropy = -sum(p * np.log(p) for p in probs if p > 0)
        max_entropy = np.log(len(intervention_counts))
        diversity = entropy / max_entropy if max_entropy > 0 else 0
        
        print(f"\n   Diversity score: {diversity:.3f}")
        if diversity > 0.5:
            print("   ✓ Good exploration! Policy tries different variables")
        else:
            print("   ✗ Poor exploration - still stuck on few variables")
    
    # 2. Check learning progress
    if 'history' in metrics:
        rewards = [ep['mean_reward'] for ep in metrics['history'] if 'mean_reward' in ep]
        
        if len(rewards) > 10:
            early_avg = np.mean(rewards[:10])
            late_avg = np.mean(rewards[-10:])
            improvement = ((late_avg - early_avg) / abs(early_avg)) * 100 if early_avg != 0 else 0
            
            print("\n2. LEARNING PROGRESS:")
            print(f"   Early episodes (0-9): {early_avg:.3f}")
            print(f"   Late episodes (40-49): {late_avg:.3f}")
            print(f"   Improvement: {improvement:+.1f}%")
            
            if improvement > 10:
                print("   ✓ Significant learning! Policy is improving")
            elif improvement > 0:
                print("   ~ Some learning observed")
            else:
                print("   ✗ No improvement - policy not learning")
            
            # Plot learning curve
            plt.figure(figsize=(10, 6))
            plt.plot(rewards, 'b-', alpha=0.5, label='Episode rewards')
            
            # Add smoothed curve
            window = 10
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(rewards)), smoothed, 'r-', linewidth=2, label='Smoothed')
            
            plt.xlabel('Episode')
            plt.ylabel('Mean Reward')
            plt.title('GRPO Learning Progress')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig('grpo_learning_progress.png', dpi=150)
            print("\n   Learning curve saved to: grpo_learning_progress.png")
    
    # 3. Check baseline usage
    print("\n3. BASELINE ANALYSIS:")
    baseline_info = analyze_baseline_usage('grpo_improvement_test.log')
    if baseline_info['non_intervention_count'] > 0:
        print(f"   ✓ Non-intervention baseline used: {baseline_info['non_intervention_count']} times")
        if baseline_info['baseline_values']:
            print(f"   Average baseline value: {np.mean(baseline_info['baseline_values']):.3f}")
    else:
        print("   ✗ Non-intervention baseline not being used properly")
    
    # 4. Check advantages
    print("\n4. ADVANTAGE SIGNALS:")
    advantages = analyze_advantages('grpo_improvement_test.log')
    if advantages:
        print(f"   Mean advantage: {np.mean(advantages):.3f}")
        print(f"   Std advantage: {np.std(advantages):.3f}")
        print(f"   Max advantage: {max(advantages):.3f}")
        print(f"   Min advantage: {min(advantages):.3f}")
        
        if np.std(advantages) > 0.1:
            print("   ✓ Meaningful advantage signals for learning")
        else:
            print("   ✗ Advantages too small - no learning signal")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Overall assessment
    checks_passed = 0
    total_checks = 4
    
    if intervention_counts and len(intervention_counts) > 1:
        checks_passed += 1
        print("✓ Exploration: Policy explores multiple variables")
    else:
        print("✗ Exploration: Still stuck on one variable")
    
    if baseline_info['non_intervention_count'] > 0:
        checks_passed += 1
        print("✓ Baseline: Using non-intervention baseline")
    else:
        print("✗ Baseline: Not using proper baseline")
    
    if advantages and np.std(advantages) > 0.1:
        checks_passed += 1
        print("✓ Advantages: Meaningful learning signals")
    else:
        print("✗ Advantages: No learning signal")
    
    if 'history' in metrics and improvement > 5:
        checks_passed += 1
        print("✓ Learning: Policy improves over time")
    else:
        print("✗ Learning: No improvement observed")
    
    print(f"\nOverall: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed == total_checks:
        print("\n🎉 SUCCESS! All improvements are working correctly!")
    elif checks_passed >= 3:
        print("\n✅ Good! Most improvements are working.")
    else:
        print("\n⚠️  Issues remain. Check the detailed output above.")
    
    return metrics


def analyze_log_interventions(log_file):
    """Extract intervention patterns from log."""
    intervention_counts = Counter()
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Look for exploration patterns
                if "[EXPLORATION]" in line and "Selected:" in line:
                    # Extract selected variable index
                    parts = line.split("Selected:")
                    if len(parts) > 1:
                        selected = parts[1].split(",")[0].strip()
                        try:
                            var_idx = int(selected)
                            intervention_counts[var_idx] += 1
                        except ValueError:
                            pass
    except FileNotFoundError:
        pass
    
    return intervention_counts


def analyze_baseline_usage(log_file):
    """Analyze baseline usage from log."""
    baseline_info = {
        'non_intervention_count': 0,
        'fallback_count': 0,
        'baseline_values': []
    }
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if "[BASELINE] Using non-intervention baseline:" in line:
                    baseline_info['non_intervention_count'] += 1
                    # Extract baseline value
                    parts = line.split("baseline:")
                    if len(parts) > 1:
                        value_part = parts[1].split("(")[0].strip()
                        try:
                            value = float(value_part)
                            baseline_info['baseline_values'].append(value)
                        except ValueError:
                            pass
                elif "[BASELINE] No obs target values" in line:
                    baseline_info['fallback_count'] += 1
    except FileNotFoundError:
        pass
    
    return baseline_info


def analyze_advantages(log_file):
    """Extract advantage values from log."""
    advantages = []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if "advantage=" in line:
                    parts = line.split("advantage=")
                    if len(parts) > 1:
                        adv_part = parts[1].split(",")[0].strip()
                        try:
                            adv = float(adv_part)
                            advantages.append(adv)
                        except ValueError:
                            pass
    except FileNotFoundError:
        pass
    
    return advantages


if __name__ == "__main__":
    # Run the verification test
    test_grpo_improvements()
    
    print("\n" + "="*80)
    print("WHAT TO DO NEXT")
    print("="*80)
    print("\n1. Check the log file for detailed patterns:")
    print("   tail -50 grpo_improvement_test.log")
    print("\n2. View the learning curve:")
    print("   open grpo_learning_progress.png")
    print("\n3. Run with surrogate for even better results:")
    print("   # Edit this script and set use_surrogate=True")
    print("\n4. Test on different SCM structures:")
    print("   # Try chain_scm or collider_scm")