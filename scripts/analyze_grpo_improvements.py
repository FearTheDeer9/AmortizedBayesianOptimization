#!/usr/bin/env python3
"""
Analyze GRPO training logs to quantify improvements from fixes.
"""

import re
from collections import defaultdict, Counter
import numpy as np


def analyze_log_file(log_file_path):
    """Analyze a GRPO training log file."""
    
    with open(log_file_path, 'r') as f:
        log_content = f.read()
    
    # Extract key metrics
    results = {
        'interventions': extract_interventions(log_content),
        'rewards': extract_rewards(log_content),
        'baselines': extract_baselines(log_content),
        'advantages': extract_advantages(log_content),
        'exploration': extract_exploration_patterns(log_content)
    }
    
    # Analyze and print results
    print_analysis(results)
    
    return results


def extract_interventions(log_content):
    """Extract which variables were intervened on."""
    pattern = r"Applied perfect intervention on \['(\w+)'\]"
    matches = re.findall(pattern, log_content)
    return matches


def extract_rewards(log_content):
    """Extract episode rewards."""
    pattern = r"Episode (\d+): mean_reward=([\d.]+)"
    matches = re.findall(pattern, log_content)
    return [(int(ep), float(reward)) for ep, reward in matches]


def extract_baselines(log_content):
    """Extract baseline values and types."""
    # Non-intervention baseline
    pattern1 = r"\[BASELINE\] Using non-intervention baseline: ([\d.-]+) \(from (\d+) obs samples\)"
    non_intervention = re.findall(pattern1, log_content)
    
    # Fallback baseline
    pattern2 = r"\[BASELINE\] No obs target values, using mean rewards"
    fallback_count = len(re.findall(pattern2, log_content))
    
    return {
        'non_intervention': non_intervention,
        'fallback_count': fallback_count
    }


def extract_advantages(log_content):
    """Extract advantage values."""
    pattern = r"advantage=([\d.-]+)"
    matches = re.findall(pattern, log_content)
    return [float(adv) for adv in matches]


def extract_exploration_patterns(log_content):
    """Extract exploration behavior."""
    pattern = r"\[EXPLORATION\] Original logits: \[([\d\s.-inf]+)\], Selected: (\d+)"
    matches = re.findall(pattern, log_content)
    
    selections = [int(selected) for _, selected in matches]
    return selections


def print_analysis(results):
    """Print analysis of the results."""
    
    print("="*80)
    print("GRPO TRAINING ANALYSIS")
    print("="*80)
    
    # Intervention diversity
    interventions = results['interventions']
    if interventions:
        intervention_counts = Counter(interventions)
        total = sum(intervention_counts.values())
        
        print("\n1. INTERVENTION DIVERSITY:")
        for var, count in intervention_counts.items():
            pct = (count / total) * 100
            print(f"   {var}: {count} ({pct:.1f}%)")
        
        # Calculate diversity score (entropy)
        probs = [count/total for count in intervention_counts.values()]
        entropy = -sum(p * np.log(p) for p in probs if p > 0)
        max_entropy = np.log(len(intervention_counts))
        diversity_score = entropy / max_entropy if max_entropy > 0 else 0
        
        print(f"   Diversity score: {diversity_score:.3f} (1.0 = perfect diversity)")
    
    # Learning progress
    rewards = results['rewards']
    if len(rewards) > 5:
        episodes, values = zip(*rewards)
        early_rewards = [v for e, v in rewards if e < 10]
        late_rewards = [v for e, v in rewards if e >= len(rewards) - 10]
        
        if early_rewards and late_rewards:
            early_avg = np.mean(early_rewards)
            late_avg = np.mean(late_rewards)
            improvement = ((late_avg - early_avg) / abs(early_avg)) * 100 if early_avg != 0 else 0
            
            print("\n2. LEARNING PROGRESS:")
            print(f"   Early episodes: {early_avg:.3f}")
            print(f"   Late episodes: {late_avg:.3f}")
            print(f"   Improvement: {improvement:+.1f}%")
    
    # Baseline usage
    baselines = results['baselines']
    if baselines['non_intervention'] or baselines['fallback_count'] > 0:
        print("\n3. BASELINE ANALYSIS:")
        print(f"   Non-intervention baseline used: {len(baselines['non_intervention'])} times")
        print(f"   Fallback to mean rewards: {baselines['fallback_count']} times")
        
        if baselines['non_intervention']:
            baseline_values = [float(val) for val, _ in baselines['non_intervention']]
            print(f"   Average baseline: {np.mean(baseline_values):.3f}")
    
    # Advantage analysis
    advantages = results['advantages']
    if advantages:
        print("\n4. ADVANTAGE ANALYSIS:")
        print(f"   Mean advantage: {np.mean(advantages):.3f}")
        print(f"   Std advantage: {np.std(advantages):.3f}")
        print(f"   Max advantage: {max(advantages):.3f}")
        print(f"   Min advantage: {min(advantages):.3f}")
    
    # Exploration patterns
    selections = results['exploration']
    if selections:
        print("\n5. EXPLORATION PATTERNS:")
        selection_counts = Counter(selections)
        for var_idx, count in sorted(selection_counts.items()):
            pct = (count / len(selections)) * 100
            print(f"   Variable {var_idx}: {count} ({pct:.1f}%)")


def compare_before_after():
    """Compare metrics before and after fixes."""
    
    print("\n" + "="*80)
    print("EXPECTED IMPROVEMENTS FROM FIXES")
    print("="*80)
    
    print("\nBEFORE FIXES:")
    print("- Diversity score: ~0.0 (stuck on one variable)")
    print("- Learning improvement: ~0% (no progress)")
    print("- Advantages: ~0.000 (just noise)")
    print("- Variable selection: 100% on variable 0")
    
    print("\nAFTER FIXES:")
    print("- Diversity score: >0.5 (explores multiple variables)")
    print("- Learning improvement: >10% (actual learning)")
    print("- Advantages: >1.0 magnitude (meaningful signal)")
    print("- Variable selection: Mixed across valid variables")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Analyze provided log file
        log_file = sys.argv[1]
        print(f"Analyzing log file: {log_file}")
        analyze_log_file(log_file)
    else:
        # Just show expected improvements
        compare_before_after()
        print("\nUsage: python analyze_grpo_improvements.py <log_file>")
        print("Example: python analyze_grpo_improvements.py checkpoints/test_grpo_fixes/training.log")