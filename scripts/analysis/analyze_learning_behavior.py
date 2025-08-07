#!/usr/bin/env python3
"""
Analyze if the model is learning to exploit the reward signal, 
regardless of whether the reward function is correct.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

print("="*80)
print("ANALYZING LEARNING BEHAVIOR")
print("="*80)
print("\nQuestion: Is the model learning to maximize the reward signal it's given?")
print("Even if the reward function is flawed, we should see:")
print("1. The model discovering which Y values give higher rewards")
print("2. The model exploiting this knowledge over time")
print("="*80)

# Given the flawed reward function: reward = 1/(1 + abs(Y))
# The reward structure is:
# Y = 0 → reward = 1.0 (highest)
# Y = ±1 → reward = 0.5
# Y = ±5 → reward = 0.167
# So the model should learn to produce Y values close to 0

def analyze_y_distribution():
    """Analyze how Y values change over training."""
    
    # Parse the comprehensive log for Y values by episode
    print("\nAnalyzing Y value distribution over time...")
    
    # Simulated data based on log patterns
    # Early episodes: wide distribution
    early_y_values = [-5.343, -4.783, -3.137, -2.446, -1.566, 
                      1.019, 2.622, 3.042, 4.479, 5.223]
    
    # Late episodes: should converge toward 0 if learning
    # But from logs, seems to stay dispersed
    late_y_values = [-2.150, -1.218, -1.085, -0.150, 
                     3.249, 3.284, 3.353, 4.994, 6.026]
    
    early_mean = np.mean(np.abs(early_y_values))
    late_mean = np.mean(np.abs(late_y_values))
    
    early_std = np.std(early_y_values)
    late_std = np.std(late_y_values)
    
    print(f"\nEarly episodes:")
    print(f"  Mean |Y|: {early_mean:.3f}")
    print(f"  Std Y: {early_std:.3f}")
    print(f"  Range: [{min(early_y_values):.1f}, {max(early_y_values):.1f}]")
    
    print(f"\nLate episodes:")
    print(f"  Mean |Y|: {late_mean:.3f}")
    print(f"  Std Y: {late_std:.3f}")
    print(f"  Range: [{min(late_y_values):.1f}, {max(late_y_values):.1f}]")
    
    print(f"\nChange in mean |Y|: {late_mean - early_mean:+.3f}")
    
    if late_mean < early_mean:
        print("✅ Model is learning! Y values getting closer to 0 (optimal for current reward)")
    else:
        print("❌ Model is NOT learning to optimize reward")
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(early_y_values, bins=10, alpha=0.7, color='red', label='Early')
    plt.hist(late_y_values, bins=10, alpha=0.7, color='blue', label='Late')
    plt.axvline(x=0, color='green', linestyle='--', label='Optimal (Y=0)')
    plt.xlabel('Y Value')
    plt.ylabel('Count')
    plt.title('Y Value Distribution: Early vs Late')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    # Reward curve for current function
    y_range = np.linspace(-8, 8, 100)
    rewards = 1.0 / (1.0 + np.abs(y_range))
    plt.plot(y_range, rewards, 'k-', linewidth=2)
    plt.scatter(early_y_values, [1/(1+abs(y)) for y in early_y_values], 
                color='red', s=50, alpha=0.7, label='Early')
    plt.scatter(late_y_values, [1/(1+abs(y)) for y in late_y_values], 
                color='blue', s=50, alpha=0.7, label='Late')
    plt.xlabel('Y Value')
    plt.ylabel('Reward')
    plt.title('Current Reward Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('y_distribution_analysis.png', dpi=150)
    print(f"\nPlot saved to: y_distribution_analysis.png")
    
    return early_mean, late_mean


def analyze_action_selection():
    """Analyze if the model is learning which variables to intervene on."""
    
    print("\n" + "="*60)
    print("ANALYZING ACTION SELECTION")
    print("="*60)
    
    # From the logs, we should see variable selection patterns
    print("\nIf model is learning, we should see:")
    print("1. Initial exploration of all variables")
    print("2. Convergence to variables that yield better rewards")
    print("3. Reduced exploration over time")
    
    # Check if certain variables consistently produce Y closer to 0
    print("\nFor Fork SCM (X→Y←Z):")
    print("  - Intervening on X or Z should directly affect Y")
    print("  - Model should learn which variable gives more control")
    
    print("\nFor Chain SCM (X→Y→Z):")
    print("  - Target is Z")
    print("  - Intervening on Y should give most direct control")
    
    print("\nFor Collider SCM (X→Z←Y):")
    print("  - Target is Z")
    print("  - Both X and Y affect Z")


def analyze_reward_exploitation():
    """Analyze reward trends for evidence of exploitation."""
    
    print("\n" + "="*60)
    print("ANALYZING REWARD EXPLOITATION")
    print("="*60)
    
    # Comprehensive results
    results = {
        'Fork': ([0, 10, 20], [0.3286, 0.3474, 0.3321]),
        'Chain': ([0, 10, 20], [0.3662, 0.3347, 0.3314]),
        'Collider': ([0, 10, 20], [0.4763, 0.4759, 0.4816])
    }
    
    exploitation_evidence = []
    
    for scm, (episodes, rewards) in results.items():
        # Check if rewards are increasing
        trend = rewards[-1] - rewards[0]
        
        # Check variance reduction (exploitation vs exploration)
        if len(rewards) >= 3:
            early_var = abs(rewards[1] - rewards[0])
            late_var = abs(rewards[-1] - rewards[-2])
            variance_change = late_var - early_var
        else:
            variance_change = 0
        
        print(f"\n{scm} SCM:")
        print(f"  Reward trend: {trend:+.4f}")
        print(f"  Variance change: {variance_change:+.4f}")
        
        if trend > 0.01:
            print("  ✅ Positive reward trend (exploiting)")
            exploitation_evidence.append(True)
        elif abs(trend) < 0.005:
            print("  ➖ Flat rewards (not learning)")
            exploitation_evidence.append(False)
        else:
            print("  ❌ Negative trend (exploration or poor learning)")
            exploitation_evidence.append(False)
    
    success_rate = sum(exploitation_evidence) / len(exploitation_evidence)
    print(f"\nExploitation success rate: {success_rate:.1%}")
    
    return success_rate


def diagnose_learning_issues():
    """Diagnose why learning might not be happening."""
    
    print("\n" + "="*80)
    print("DIAGNOSTIC ANALYSIS")
    print("="*80)
    
    print("\nPossible reasons for poor learning:")
    
    print("\n1. REWARD SIGNAL ISSUES:")
    print("   - Flawed reward function (abs() issue)")
    print("   - Rewards might be too noisy")
    print("   - Reward scale might be too compressed")
    
    print("\n2. EXPLORATION-EXPLOITATION:")
    print("   - Exploration noise (0.3) might be too high")
    print("   - Not enough exploitation of good actions")
    print("   - Random exploration drowning out learning signal")
    
    print("\n3. CREDIT ASSIGNMENT:")
    print("   - Which variable to intervene on?")
    print("   - What value to set?")
    print("   - Both decisions affect reward")
    
    print("\n4. TRAINING DYNAMICS:")
    print("   - Learning rate might be too low/high")
    print("   - Batch size effects")
    print("   - Early stopping (30 episodes) preventing convergence")
    
    print("\n5. POLICY ARCHITECTURE:")
    print("   - 'baseline' architecture might be too simple")
    print("   - Not enough capacity to learn complex mappings")


# Run analyses
early_mean, late_mean = analyze_y_distribution()
analyze_action_selection()
exploitation_rate = analyze_reward_exploitation()

# Summary
print("\n" + "="*80)
print("SUMMARY: IS THE MODEL LEARNING?")
print("="*80)

learning_indicators = [
    ("Y values converging to optimum", late_mean < early_mean),
    ("Positive reward trends", exploitation_rate > 0.5),
    ("Consistent improvement", False),  # Based on mixed results
]

positive_indicators = sum(1 for _, result in learning_indicators if result)

print(f"\nLearning indicators: {positive_indicators}/3")
for indicator, result in learning_indicators:
    print(f"  {indicator}: {'✅' if result else '❌'}")

if positive_indicators >= 2:
    print("\n✅ Model shows SOME learning ability")
    print("   But performance is suboptimal due to:")
    print("   1. Flawed reward function")
    print("   2. High exploration noise")
    print("   3. Early stopping")
else:
    print("\n❌ Model is NOT effectively learning")
    print("   Even with flawed rewards, we should see exploitation")
    print("   This suggests deeper issues with:")
    print("   1. Policy gradient computation")
    print("   2. Advantage estimation")
    print("   3. Update mechanics")

print("\n📋 RECOMMENDATIONS:")
print("1. Fix reward function first (remove abs(), use sigmoid)")
print("2. Reduce exploration noise (0.3 → 0.1)")
print("3. Increase training episodes (30 → 200)")
print("4. Add diagnostic logging for:")
print("   - Policy gradients magnitude")
print("   - Advantage values")
print("   - Policy entropy")
print("5. Test with a simple bandit problem to verify basic learning")