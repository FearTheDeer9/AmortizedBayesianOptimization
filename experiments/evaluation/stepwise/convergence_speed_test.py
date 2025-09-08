#!/usr/bin/env python3
"""
Test convergence speed of different models.

This script evaluates how quickly different policies converge to optimal
target values as a function of the number of interventions.
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess
from datetime import datetime

def run_single_evaluation(
    policy_path: str,
    surrogate_path: str,
    num_interventions: int,
    structure: str = "chain",
    num_vars: int = 3,
    num_episodes: int = 10,
    seed: int = 42
) -> Dict:
    """Run evaluation with specified number of interventions."""
    
    output_dir = Path(f"test_debug/convergence_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    cmd = [
        'python', 'experiments/evaluation/stepwise/full_evaluation.py',
        '--policy-path', policy_path,
        '--surrogate-path', surrogate_path,
        '--structures', structure,
        '--num-vars', str(num_vars),
        '--num-episodes', str(num_episodes),
        '--num-interventions', str(num_interventions),
        '--initial-observations', '10',
        '--output-dir', str(output_dir),
        '--seed', str(seed)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running evaluation: {result.stderr}")
        return None
    
    # Parse results from output
    lines = result.stdout.split('\n')
    
    # Extract metrics
    metrics = {
        'num_interventions': num_interventions,
        'f1_scores': [],
        'parent_rates': [],
        'target_values': []
    }
    
    for line in lines:
        if 'F1=' in line and 'Parent Rate=' in line:
            # Parse episode result line
            parts = line.split(',')
            for part in parts:
                if 'F1=' in part:
                    f1 = float(part.split('=')[1])
                    metrics['f1_scores'].append(f1)
                elif 'Parent Rate=' in part:
                    rate = float(part.split('=')[1].replace('%', ''))
                    metrics['parent_rates'].append(rate)
                elif 'Target=' in part:
                    target = float(part.split('=')[1])
                    metrics['target_values'].append(target)
    
    # Calculate averages
    if metrics['f1_scores']:
        metrics['avg_f1'] = np.mean(metrics['f1_scores'])
        metrics['std_f1'] = np.std(metrics['f1_scores'])
    if metrics['parent_rates']:
        metrics['avg_parent_rate'] = np.mean(metrics['parent_rates'])
        metrics['std_parent_rate'] = np.std(metrics['parent_rates'])
    if metrics['target_values']:
        metrics['avg_target'] = np.mean(metrics['target_values'])
        metrics['std_target'] = np.std(metrics['target_values'])
        metrics['best_target'] = min(metrics['target_values'])  # For minimization
    
    return metrics


def test_convergence_speed(
    models: List[Tuple[str, str, str]],  # [(name, policy_path, surrogate_path)]
    intervention_steps: List[int] = [5, 10, 15, 20, 30, 40, 50],
    structure: str = "chain",
    num_vars: int = 3,
    num_episodes: int = 10,
    seed: int = 42
) -> Dict:
    """Test convergence speed for multiple models."""
    
    results = {}
    
    for model_name, policy_path, surrogate_path in models:
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"Policy: {policy_path}")
        print(f"Surrogate: {surrogate_path}")
        print('='*60)
        
        model_results = []
        
        for num_interventions in intervention_steps:
            print(f"\n  Testing with {num_interventions} interventions...")
            
            metrics = run_single_evaluation(
                policy_path=policy_path,
                surrogate_path=surrogate_path,
                num_interventions=num_interventions,
                structure=structure,
                num_vars=num_vars,
                num_episodes=num_episodes,
                seed=seed
            )
            
            if metrics:
                model_results.append(metrics)
                print(f"    F1: {metrics.get('avg_f1', 0):.3f} ± {metrics.get('std_f1', 0):.3f}")
                print(f"    Parent Rate: {metrics.get('avg_parent_rate', 0):.1f}% ± {metrics.get('std_parent_rate', 0):.1f}%")
                print(f"    Best Target: {metrics.get('best_target', 0):.3f}")
        
        results[model_name] = model_results
    
    return results


def plot_convergence(results: Dict, save_path: Path = None):
    """Plot convergence curves for all models."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: F1 Score vs Interventions
    ax = axes[0, 0]
    for model_name, model_results in results.items():
        if model_results:
            x = [r['num_interventions'] for r in model_results]
            y = [r.get('avg_f1', 0) for r in model_results]
            err = [r.get('std_f1', 0) for r in model_results]
            ax.errorbar(x, y, yerr=err, marker='o', label=model_name, linewidth=2)
    
    ax.set_xlabel('Number of Interventions')
    ax.set_ylabel('F1 Score')
    ax.set_title('Structure Learning Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Plot 2: Parent Selection Rate vs Interventions
    ax = axes[0, 1]
    for model_name, model_results in results.items():
        if model_results:
            x = [r['num_interventions'] for r in model_results]
            y = [r.get('avg_parent_rate', 0) for r in model_results]
            err = [r.get('std_parent_rate', 0) for r in model_results]
            ax.errorbar(x, y, yerr=err, marker='s', label=model_name, linewidth=2)
    
    ax.set_xlabel('Number of Interventions')
    ax.set_ylabel('Parent Selection Rate (%)')
    ax.set_title('Parent Selection Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    # Plot 3: Best Target Value vs Interventions
    ax = axes[1, 0]
    for model_name, model_results in results.items():
        if model_results:
            x = [r['num_interventions'] for r in model_results]
            y = [r.get('best_target', 0) for r in model_results]
            ax.plot(x, y, marker='^', label=model_name, linewidth=2)
    
    ax.set_xlabel('Number of Interventions')
    ax.set_ylabel('Best Target Value (lower is better)')
    ax.set_title('Target Optimization Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary Statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create summary table
    summary_text = "Summary at 20 interventions:\n\n"
    for model_name, model_results in results.items():
        # Find results for 20 interventions
        for r in model_results:
            if r['num_interventions'] == 20:
                summary_text += f"{model_name}:\n"
                summary_text += f"  F1: {r.get('avg_f1', 0):.3f}\n"
                summary_text += f"  Parent Rate: {r.get('avg_parent_rate', 0):.1f}%\n"
                summary_text += f"  Best Target: {r.get('best_target', 0):.3f}\n\n"
                break
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
           fontfamily='monospace')
    
    plt.suptitle('Convergence Speed Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()


def main():
    """Run convergence speed test."""
    
    print("="*70)
    print("CONVERGENCE SPEED TEST")
    print("="*70)
    
    # Define models to test
    models = [
        # New model
        ("GRPO Enhanced (181212)", 
         "imperial-vm-checkpoints/grpo_enhanced_20250907_181212/final_policy.pkl",
         "imperial-vm-checkpoints/avici_style_20250907_034427/best_model.pkl"),
        
        # Previous model for comparison
        ("GRPO Enhanced (034435)",
         "imperial-vm-checkpoints/grpo_enhanced_20250907_034435/final_policy.pkl",
         "imperial-vm-checkpoints/avici_style_20250907_034427/best_model.pkl"),
    ]
    
    # Test convergence at different intervention counts
    intervention_steps = [5, 10, 15, 20, 30]
    
    # Run tests
    results = test_convergence_speed(
        models=models,
        intervention_steps=intervention_steps,
        structure="chain",  # Test on chain structures
        num_vars=3,
        num_episodes=5,  # Fewer episodes for speed
        seed=42
    )
    
    # Save results
    results_file = Path("test_debug/convergence_speed_results.json")
    results_file.parent.mkdir(exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Plot results
    plot_path = Path("test_debug/convergence_speed_plot.png")
    plot_convergence(results, save_path=plot_path)
    
    # Print final summary
    print("\n" + "="*70)
    print("CONVERGENCE SUMMARY")
    print("="*70)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        
        # Find improvement from 5 to 20 interventions
        early = None
        late = None
        for r in model_results:
            if r['num_interventions'] == 5:
                early = r
            elif r['num_interventions'] == 20:
                late = r
        
        if early and late:
            f1_improvement = late.get('avg_f1', 0) - early.get('avg_f1', 0)
            target_improvement = early.get('best_target', 0) - late.get('best_target', 0)
            
            print(f"  F1 improvement (5→20): {f1_improvement:+.3f}")
            print(f"  Target improvement (5→20): {target_improvement:+.3f}")
            print(f"  Final F1: {late.get('avg_f1', 0):.3f}")
            print(f"  Final target: {late.get('best_target', 0):.3f}")


if __name__ == "__main__":
    main()