#!/usr/bin/env python3
"""
Analyze and visualize experiment results.
"""

import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_results():
    """Load experiment results."""
    results_file = Path("debugging-bc-training/results_experiments/experiment_results.json")
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return []

def create_summary_report(results):
    """Create a summary report of all experiments."""
    print("\n" + "="*80)
    print("BEHAVIORAL CLONING TRAINING ANALYSIS REPORT")
    print("="*80)
    
    print("\n### PROBLEM SUMMARY ###")
    print("- Target Problem: X4 variable has 0% accuracy in predictions")
    print("- Root Cause: Class imbalance (X4 appears ~2.7% vs X0/X1 ~35%)")
    print("- Overall Performance: ~50-60% accuracy baseline")
    
    print("\n### EXPERIMENTS CONDUCTED ###")
    
    # Sort by best validation accuracy
    sorted_results = sorted([r for r in results if r['status'] == 'success'], 
                           key=lambda x: x['metrics'].get('best_val_accuracy', 0), 
                           reverse=True)
    
    if sorted_results:
        print("\nRanked by Best Validation Accuracy:")
        print("-" * 60)
        for i, result in enumerate(sorted_results, 1):
            metrics = result['metrics']
            print(f"\n{i}. {result['name']}")
            print(f"   Best Val Acc: {metrics.get('best_val_accuracy', 0):.3f}")
            print(f"   Final Val Acc: {metrics.get('final_val_accuracy', 0):.3f}")
            print(f"   F1 Score: {metrics.get('final_val_f1', 0):.3f}")
            if 'X4_accuracy' in metrics:
                print(f"   X4 Accuracy: {metrics.get('X4_accuracy', 0):.3f}")
            print(f"   Duration: {result['duration_minutes']:.1f} min")
    
    print("\n### KEY FINDINGS ###")
    print("1. X4 Performance: ALL approaches achieved 0% accuracy on X4")
    print("   - Permutation augmentation: Did NOT improve X4 predictions")
    print("   - Weighted loss: Not tested yet but likely similar issue")
    print("   - Combined approach: Still 0% on X4")
    
    print("\n2. Overall Performance:")
    print("   - Baseline: ~55% best accuracy")
    print("   - Higher learning rate: 60% best (but unstable)")
    print("   - Permutation augmentation: Slightly worse (~47%)")
    print("   - Combined approach: Worse performance (~39%)")
    
    print("\n### ANALYSIS ###")
    print("The failure to improve X4 predictions suggests deeper issues:")
    print("1. X4 might be fundamentally harder to predict")
    print("2. The model architecture may not capture X4's patterns")
    print("3. X4 might have different causal relationships")
    print("4. The 2.7% frequency might be too low even with augmentation")
    
    print("\n### RECOMMENDATIONS ###")
    print("1. **Investigate X4's causal structure**: Check if X4 has unique properties")
    print("2. **Manual oversampling**: Focus specifically on X4 examples")
    print("3. **Architecture changes**: Add attention mechanisms or specialized heads")
    print("4. **Loss function**: Use focal loss or extreme class weighting (10-100x)")
    print("5. **Data analysis**: Check if X4 interventions have different patterns")
    print("6. **Ensemble methods**: Train separate models for rare classes")
    
    return sorted_results

def plot_comparison(results):
    """Create comparison plots."""
    if not results:
        print("No results to plot")
        return
    
    # Filter successful results
    successful = [r for r in results if r['status'] == 'success']
    if not successful:
        print("No successful experiments to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Best validation accuracy comparison
    ax = axes[0, 0]
    names = [r['name'].replace('_', '\n') for r in successful]
    best_accs = [r['metrics'].get('best_val_accuracy', 0) for r in successful]
    bars = ax.bar(range(len(names)), best_accs)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=0, ha='center', fontsize=8)
    ax.set_ylabel('Best Validation Accuracy')
    ax.set_title('Best Validation Accuracy by Approach')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Color code bars
    for i, bar in enumerate(bars):
        if best_accs[i] >= 0.55:
            bar.set_color('green')
        elif best_accs[i] >= 0.45:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # 2. Final validation accuracy comparison
    ax = axes[0, 1]
    final_accs = [r['metrics'].get('final_val_accuracy', 0) for r in successful]
    bars = ax.bar(range(len(names)), final_accs)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=0, ha='center', fontsize=8)
    ax.set_ylabel('Final Validation Accuracy')
    ax.set_title('Final Validation Accuracy by Approach')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # 3. F1 Score comparison
    ax = axes[1, 0]
    f1_scores = [r['metrics'].get('final_val_f1', 0) for r in successful]
    bars = ax.bar(range(len(names)), f1_scores)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=0, ha='center', fontsize=8)
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score by Approach')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # 4. X4 Accuracy (if available)
    ax = axes[1, 1]
    x4_accs = []
    x4_names = []
    for r in successful:
        if 'X4_accuracy' in r['metrics']:
            x4_accs.append(r['metrics']['X4_accuracy'])
            x4_names.append(r['name'].replace('_', '\n'))
    
    if x4_accs:
        bars = ax.bar(range(len(x4_names)), x4_accs)
        ax.set_xticks(range(len(x4_names)))
        ax.set_xticklabels(x4_names, rotation=0, ha='center', fontsize=8)
        ax.set_ylabel('X4 Accuracy')
        ax.set_title('X4 Variable Accuracy (Problem Target)')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        # All bars are red since X4 accuracy is 0
        for bar in bars:
            bar.set_color('red')
        # Add text showing the problem
        ax.text(0.5, 0.5, 'ALL APPROACHES: 0% ACCURACY', 
                transform=ax.transAxes, ha='center', va='center',
                fontsize=14, color='red', weight='bold')
    else:
        ax.text(0.5, 0.5, 'X4 accuracy data not available\nfor these experiments', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('X4 Variable Accuracy')
    
    plt.suptitle('BC Training Experiment Comparison', fontsize=14, weight='bold')
    plt.tight_layout()
    
    output_file = Path("debugging-bc-training/experiments/comparison_plot.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    plt.close()

def main():
    """Main analysis function."""
    results = load_results()
    
    if not results:
        print("No experiment results found!")
        return
    
    print(f"Loaded {len(results)} experiment results")
    
    # Create summary report
    sorted_results = create_summary_report(results)
    
    # Create plots
    plot_comparison(results)
    
    # Additional detailed analysis
    print("\n" + "="*80)
    print("DETAILED RECOMMENDATIONS FOR X4 PROBLEM")
    print("="*80)
    
    print("\n### IMMEDIATE NEXT STEPS ###")
    print("1. Check if X4 appears in training data at all:")
    print("   python debugging-bc-training/analyze_all_targets.py")
    
    print("\n2. Analyze X4's causal relationships:")
    print("   - Does X4 have different parent nodes?")
    print("   - Are X4 interventions qualitatively different?")
    
    print("\n3. Try extreme class weighting:")
    print("   python debugging-bc-training/experiments/weighted_loss_trainer.py \\")
    print("     --manual_weights '{\"X0\":1.0,\"X1\":1.0,\"X2\":1.0,\"X3\":1.0,\"X4\":50.0}'")
    
    print("\n4. Focus training on X4 examples only:")
    print("   - Create a filtered dataset with higher X4 proportion")
    print("   - Use curriculum learning: start with X4-heavy batches")
    
    print("\n### ARCHITECTURAL CHANGES TO CONSIDER ###")
    print("1. Multi-head architecture: Separate prediction heads per variable")
    print("2. Attention mechanisms: Let model focus on rare patterns")
    print("3. Hierarchical model: First predict variable group, then specific variable")
    print("4. Ensemble: Train separate models for common vs rare variables")

if __name__ == "__main__":
    main()