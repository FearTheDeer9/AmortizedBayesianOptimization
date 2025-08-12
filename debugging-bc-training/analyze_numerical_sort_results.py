#!/usr/bin/env python3
"""
Analyze the results of training with numerical sorting fix.
Compare against previous training runs to see improvements.
"""

import pickle
from pathlib import Path
import numpy as np

def analyze_results():
    """Analyze and compare training results."""
    
    print("="*80)
    print("NUMERICAL SORTING TRAINING RESULTS ANALYSIS")
    print("="*80)
    
    # Load results from numerical sorting training
    numerical_results_path = Path("numerical_sort_results/metrics_history.pkl")
    
    if not numerical_results_path.exists():
        print(f"Results not found at: {numerical_results_path}")
        print("Training may still be running...")
        return
    
    with open(numerical_results_path, 'rb') as f:
        numerical_data = pickle.load(f)
    
    # Load previous results for comparison (if available)
    previous_results = {}
    previous_paths = [
        ("Permutation", Path("permutation_output/metrics_history.pkl")),
        ("Gradient Analysis", Path("gradient_analysis_output/metrics_history.pkl")),
        ("Original", Path("results_optimal/metrics_history.pkl"))
    ]
    
    for name, path in previous_paths:
        if path.exists():
            with open(path, 'rb') as f:
                previous_results[name] = pickle.load(f)
    
    # Analyze numerical sorting results
    print("\n1. NUMERICAL SORTING RESULTS")
    print("-" * 60)
    
    if 'epoch_metrics' in numerical_data and numerical_data['epoch_metrics']:
        epochs = len(numerical_data['epoch_metrics'])
        
        # Get best and final metrics
        best_acc = max(m.get('val_accuracy', 0) for m in numerical_data['epoch_metrics'])
        final_metrics = numerical_data['epoch_metrics'][-1]
        final_acc = final_metrics.get('val_accuracy', 0)
        final_loss = final_metrics.get('val_loss', 0)
        
        # Get initial metrics for comparison
        initial_metrics = numerical_data['epoch_metrics'][0]
        initial_loss = initial_metrics.get('val_loss', 0)
        
        print(f"Total epochs trained: {epochs}")
        print(f"Initial val loss: {initial_loss:.3f}")
        print(f"Final val loss: {final_loss:.3f}")
        print(f"Loss reduction: {initial_loss - final_loss:.3f}")
        print(f"Best validation accuracy: {best_acc:.3f}")
        print(f"Final validation accuracy: {final_acc:.3f}")
    
    # Per-variable performance
    print("\n2. PER-VARIABLE PERFORMANCE")
    print("-" * 60)
    
    if 'per_variable_stats' in numerical_data:
        print("\nVariable | Attempts | Correct | Accuracy")
        print("-" * 45)
        
        total_attempts = 0
        total_correct = 0
        
        for var_name in sorted(numerical_data['per_variable_stats'].keys()):
            stats = numerical_data['per_variable_stats'][var_name]
            attempts = stats.get('attempts', 0)
            correct = stats.get('correct', 0)
            
            if attempts > 0:
                acc = correct / attempts
                print(f"{var_name:8s} | {attempts:8d} | {correct:7d} | {acc:.3f}")
                total_attempts += attempts
                total_correct += correct
            else:
                print(f"{var_name:8s} | {attempts:8d} | {correct:7d} | N/A")
        
        if total_attempts > 0:
            overall_acc = total_correct / total_attempts
            print(f"\nOverall accuracy: {overall_acc:.3f}")
    
    # Compare with previous runs
    if previous_results:
        print("\n3. COMPARISON WITH PREVIOUS RUNS")
        print("-" * 60)
        print("\nRun Name         | Best Acc | Final Loss | Notes")
        print("-" * 60)
        
        for name, data in previous_results.items():
            if 'epoch_metrics' in data and data['epoch_metrics']:
                best_acc = max(m.get('val_accuracy', 0) for m in data['epoch_metrics'])
                final_loss = data['epoch_metrics'][-1].get('val_loss', 0)
                print(f"{name:16s} | {best_acc:8.3f} | {final_loss:10.3f} |")
        
        # Add numerical sorting results
        if 'epoch_metrics' in numerical_data and numerical_data['epoch_metrics']:
            best_acc = max(m.get('val_accuracy', 0) for m in numerical_data['epoch_metrics'])
            final_loss = numerical_data['epoch_metrics'][-1].get('val_loss', 0)
            print(f"{'Numerical Sort':16s} | {best_acc:8.3f} | {final_loss:10.3f} | ← NEW")
    
    # Check specific improvements
    print("\n4. KEY IMPROVEMENTS CHECK")
    print("-" * 60)
    
    if 'epoch_metrics' in numerical_data and numerical_data['epoch_metrics']:
        initial_loss = numerical_data['epoch_metrics'][0].get('val_loss', 0)
        final_loss = numerical_data['epoch_metrics'][-1].get('val_loss', 0)
        
        # Check if loss dropped to reasonable range
        if final_loss < 3.0:
            print("✓ Loss dropped to reasonable range (<3.0)")
        else:
            print(f"✗ Loss still high ({final_loss:.2f}), expected <3.0")
        
        # Check if accuracy improved
        final_acc = numerical_data['epoch_metrics'][-1].get('val_accuracy', 0)
        if final_acc > 0.70:
            print(f"✓ Accuracy improved to {final_acc:.3f} (>70%)")
        else:
            print(f"⚠ Accuracy at {final_acc:.3f}, expected >70%")
        
        # Check X6/X8 performance (these are the remapped X4s)
        if 'per_variable_stats' in numerical_data:
            for var in ['X6', 'X8']:
                if var in numerical_data['per_variable_stats']:
                    stats = numerical_data['per_variable_stats'][var]
                    if stats['attempts'] > 0:
                        acc = stats['correct'] / stats['attempts']
                        if acc > 0.3:
                            print(f"✓ {var} accuracy: {acc:.3f} (improved!)")
                        else:
                            print(f"⚠ {var} accuracy: {acc:.3f} (still low)")
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    if 'epoch_metrics' in numerical_data and numerical_data['epoch_metrics']:
        final_loss = numerical_data['epoch_metrics'][-1].get('val_loss', 0)
        final_acc = numerical_data['epoch_metrics'][-1].get('val_accuracy', 0)
        
        if final_loss < 3.0 and final_acc > 0.65:
            print("""
✓ SUCCESS! The numerical sorting fix appears to have worked:
- Loss dropped to reasonable range
- Accuracy improved significantly
- Training signals are now consistent
""")
        else:
            print("""
⚠ PARTIAL SUCCESS: Some improvements seen but not as much as expected.
This could be due to:
- Need for more training epochs
- Other issues in the data pipeline
- Need for additional fixes beyond sorting
""")

if __name__ == "__main__":
    analyze_results()