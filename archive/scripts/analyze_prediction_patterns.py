#!/usr/bin/env python3
"""Analyze prediction patterns from 12-pair evaluation to understand F1/SHD uniformity."""

import json
import sys
from pathlib import Path
import numpy as np

def analyze_prediction_patterns(results_path):
    """Analyze why all surrogates show identical F1/SHD scores."""
    
    with open(results_path) as f:
        results = json.load(f)
    
    print("\n" + "="*80)
    print("PREDICTION PATTERN ANALYSIS")
    print("="*80)
    
    # 1. Extract F1/SHD scores by policy and surrogate
    print("\n1. F1/SHD Scores by Policy and Surrogate:")
    print("-" * 60)
    
    policy_scores = {}
    for method, data in results.items():
        if 'aggregate_metrics' not in data:
            continue
        
        # Parse method name
        if '+' in method:
            policy, surrogate = method.split('+')
        else:
            policy = method
            surrogate = 'none'
        
        if policy not in policy_scores:
            policy_scores[policy] = {}
        
        policy_scores[policy][surrogate] = {
            'f1': data['aggregate_metrics']['mean_f1_score'],
            'shd': data['aggregate_metrics']['mean_shd']
        }
    
    # Display in matrix form
    surrogates = ['none', 'bc_static', 'bc_active', 'untrained']
    policies = ['Random', 'Oracle', 'BC', 'GRPO']
    
    print("\nF1 Scores:")
    print(f"{'Policy':<10} | " + " | ".join(f"{s:<12}" for s in surrogates))
    print("-" * 70)
    for policy in policies:
        if policy.lower() in policy_scores:
            row = f"{policy:<10} | "
            for surrogate in surrogates:
                score = policy_scores[policy.lower()].get(surrogate, {}).get('f1', '-')
                if isinstance(score, float):
                    row += f"{score:<12.3f} | "
                else:
                    row += f"{score:<12} | "
            print(row)
    
    print("\nSHD Scores:")
    print(f"{'Policy':<10} | " + " | ".join(f"{s:<12}" for s in surrogates))
    print("-" * 70)
    for policy in policies:
        if policy.lower() in policy_scores:
            row = f"{policy:<10} | "
            for surrogate in surrogates:
                score = policy_scores[policy.lower()].get(surrogate, {}).get('shd', '-')
                if isinstance(score, float):
                    row += f"{score:<12.3f} | "
                else:
                    row += f"{score:<12} | "
            print(row)
    
    # 2. Analyze per-SCM predictions
    print("\n\n2. Per-SCM Prediction Analysis:")
    print("-" * 60)
    
    scm_predictions = {}
    for method, data in results.items():
        if 'scm_results' not in data:
            continue
        
        for scm_name, scm_data in data['scm_results'].items():
            if scm_name not in scm_predictions:
                scm_predictions[scm_name] = {
                    'true_parents': scm_data.get('true_parents', []),
                    'predictions': {}
                }
            
            f1 = scm_data.get('f1_score', 0.0)
            shd = scm_data.get('shd', float('inf'))
            scm_predictions[scm_name]['predictions'][method] = {
                'f1': f1,
                'shd': shd
            }
    
    for scm_name, scm_data in scm_predictions.items():
        print(f"\nSCM: {scm_name}")
        print(f"True parents: {scm_data['true_parents']}")
        
        # Group by unique predictions
        unique_predictions = {}
        for method, scores in scm_data['predictions'].items():
            key = (scores['f1'], scores['shd'])
            if key not in unique_predictions:
                unique_predictions[key] = []
            unique_predictions[key].append(method)
        
        print(f"Unique prediction patterns: {len(unique_predictions)}")
        for (f1, shd), methods in unique_predictions.items():
            print(f"  F1={f1:.3f}, SHD={shd} -> {', '.join(methods)}")
    
    # 3. Analyze surrogate impact
    print("\n\n3. Surrogate Impact Analysis:")
    print("-" * 60)
    
    for policy in ['random', 'oracle', 'bc', 'grpo']:
        print(f"\n{policy.upper()} Policy:")
        
        # Get results without surrogate
        base_key = policy.capitalize() if policy != 'grpo' else 'GRPO'
        if base_key not in results:
            continue
        
        base_f1 = results[base_key]['aggregate_metrics']['mean_f1_score']
        base_shd = results[base_key]['aggregate_metrics']['mean_shd']
        base_traj = results[base_key]['aggregate_metrics']['mean_trajectory_value']
        
        print(f"  Without surrogate: F1={base_f1:.3f}, SHD={base_shd:.3f}, Traj={base_traj:.3f}")
        
        # Get results with surrogates
        for surrogate in ['bc_static', 'bc_active', 'untrained']:
            surrogate_key = f"{policy}+{surrogate}"
            if surrogate_key in results:
                s_f1 = results[surrogate_key]['aggregate_metrics']['mean_f1_score']
                s_shd = results[surrogate_key]['aggregate_metrics']['mean_shd']
                s_traj = results[surrogate_key]['aggregate_metrics']['mean_trajectory_value']
                
                f1_delta = s_f1 - base_f1
                shd_delta = s_shd - base_shd
                traj_delta = s_traj - base_traj
                
                print(f"  With {surrogate}: F1={s_f1:.3f} ({f1_delta:+.3f}), "
                      f"SHD={s_shd:.3f} ({shd_delta:+.3f}), "
                      f"Traj={s_traj:.3f} ({traj_delta:+.3f})")
    
    # 4. Hypothesis testing
    print("\n\n4. Hypothesis Testing:")
    print("-" * 60)
    
    # Check if all surrogates produce identical F1/SHD
    surrogate_f1s = set()
    surrogate_shds = set()
    
    for method, data in results.items():
        if '+' in method and 'aggregate_metrics' in data:
            surrogate_f1s.add(round(data['aggregate_metrics']['mean_f1_score'], 6))
            surrogate_shds.add(round(data['aggregate_metrics']['mean_shd'], 6))
    
    if len(surrogate_f1s) == 1:
        print("✗ All surrogate methods produce IDENTICAL F1 scores:", list(surrogate_f1s)[0])
    else:
        print("✓ Surrogate methods produce different F1 scores:", sorted(surrogate_f1s))
    
    if len(surrogate_shds) == 1:
        print("✗ All surrogate methods produce IDENTICAL SHD scores:", list(surrogate_shds)[0])
    else:
        print("✓ Surrogate methods produce different SHD scores:", sorted(surrogate_shds))
    
    # Check if surrogates always improve structure learning
    improvements = []
    for policy in ['random', 'oracle']:
        base_key = policy.capitalize()
        if base_key not in results:
            continue
        
        base_f1 = results[base_key]['aggregate_metrics']['mean_f1_score']
        
        for surrogate in ['bc_static']:
            surrogate_key = f"{policy}+{surrogate}"
            if surrogate_key in results:
                s_f1 = results[surrogate_key]['aggregate_metrics']['mean_f1_score']
                improvements.append(s_f1 - base_f1)
    
    if all(imp > 0 for imp in improvements):
        print(f"✓ BC surrogate always improves F1 score (avg improvement: {np.mean(improvements):.3f})")
    else:
        print("✗ BC surrogate does not always improve F1 score")
    
    # 5. Recommendations
    print("\n\n5. Analysis Summary:")
    print("-" * 60)
    
    if len(surrogate_f1s) == 1 and len(surrogate_shds) == 1:
        print("⚠️  ISSUE: All policies produce identical structure predictions when using surrogates.")
        print("   This suggests the surrogate predictions are dominating the policy decisions.")
        print("   Possible causes:")
        print("   - Surrogate predictions are very confident (close to 0 or 1)")
        print("   - Policies are not properly integrating surrogate information")
        print("   - All surrogates are making the same predictions")
        print("\n   Recommendations:")
        print("   - Check surrogate prediction confidence levels")
        print("   - Verify policies are using 5-channel input correctly")
        print("   - Test with different surrogate checkpoints")
    else:
        print("✓ Different surrogate types produce different predictions.")
        print("  This suggests the system is working as intended.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        results_path = "evaluation_results/test_12_pairs/evaluation_results.json"
    
    if not Path(results_path).exists():
        print(f"Results file not found: {results_path}")
        sys.exit(1)
    
    analyze_prediction_patterns(results_path)