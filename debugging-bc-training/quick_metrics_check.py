#!/usr/bin/env python3
"""
Quick script to check saved metrics and diagnose training issues.
"""

import pickle
import numpy as np
from pathlib import Path
import sys

def check_metrics(metrics_file='debugging-bc-training/results/metrics_history.pkl'):
    """Load and analyze saved metrics."""
    
    metrics_path = Path(metrics_file)
    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_file}")
        return
    
    with open(metrics_path, 'rb') as f:
        data = pickle.load(f)
    
    print("="*60)
    print("METRICS DIAGNOSTIC REPORT")
    print("="*60)
    
    # Check epoch metrics
    if 'epoch_metrics' in data:
        epochs = data['epoch_metrics']
        print(f"\n📊 Training Summary:")
        print(f"  Total epochs: {len(epochs)}")
        
        if epochs:
            # Check for inf losses
            train_accs = [e.get('train_accuracy', 0) for e in epochs]
            val_accs = [e.get('val_accuracy', 0) for e in epochs]
            val_f1s = [e.get('val_f1', 0) for e in epochs]
            
            print(f"\n📈 Performance Metrics:")
            print(f"  Best val accuracy: {max(val_accs):.3f} (epoch {np.argmax(val_accs)+1})")
            print(f"  Final val accuracy: {val_accs[-1]:.3f}")
            print(f"  Best val F1: {max(val_f1s):.3f}")
            print(f"  Accuracy range: [{min(val_accs):.3f}, {max(val_accs):.3f}]")
            
            # Check for training issues
            print(f"\n⚠️ Training Issues:")
            issues_found = False
            
            # Check if performance plateaued
            if len(val_accs) > 5:
                last_5 = val_accs[-5:]
                if max(last_5) - min(last_5) < 0.01:
                    print(f"  - Performance plateaued in last 5 epochs")
                    issues_found = True
            
            # Check if accuracy is too low
            if max(val_accs) < 0.3:
                print(f"  - Very low accuracy (max {max(val_accs):.3f})")
                print(f"    → Model may not be learning")
                issues_found = True
            
            # Check if we have high variance
            if len(val_accs) > 2:
                acc_std = np.std(val_accs)
                if acc_std > 0.1:
                    print(f"  - High variance in accuracy (std={acc_std:.3f})")
                    print(f"    → Training may be unstable")
                    issues_found = True
            
            if not issues_found:
                print(f"  No major issues detected")
            
            # Print epoch-by-epoch details
            print(f"\n📝 Epoch Details:")
            for i, epoch in enumerate(epochs[:10]):  # First 10 epochs
                print(f"  Epoch {i+1}:")
                print(f"    Train acc: {epoch.get('train_accuracy', 0):.3f}")
                print(f"    Val acc: {epoch.get('val_accuracy', 0):.3f}")
                print(f"    Val F1: {epoch.get('val_f1', 0):.3f}")
                if 'val_precision' in epoch:
                    print(f"    Val precision: {epoch['val_precision']:.3f}")
                    print(f"    Val recall: {epoch['val_recall']:.3f}")
    
    # Check per-variable stats
    if 'per_variable_stats' in data and data['per_variable_stats']:
        print(f"\n🎯 Per-Variable Performance:")
        for var_name, stats in list(data['per_variable_stats'].items())[:10]:
            if stats['attempts'] > 0:
                acc = stats['correct'] / stats['attempts']
                print(f"  {var_name}: {acc:.3f} ({stats['attempts']} attempts)")
    
    # Check embeddings
    if 'embeddings_history' in data and data['embeddings_history']:
        print(f"\n🧠 Embeddings:")
        print(f"  Saved for epochs: {list(data['embeddings_history'].keys())}")
        
        # Check embedding shapes
        for epoch, embs in list(data['embeddings_history'].items())[:1]:
            if embs:
                print(f"  Shape at epoch {epoch}: {embs[0].shape}")
    
    # Check confusion matrices
    if 'confusion_matrices' in data and data['confusion_matrices']:
        print(f"\n🔄 Confusion Matrices:")
        print(f"  Available for epochs: {list(data['confusion_matrices'].keys())}")
        
        # Get latest confusion matrix
        latest_epoch = max(data['confusion_matrices'].keys())
        cm_data = data['confusion_matrices'][latest_epoch]
        cm = cm_data['matrix']
        
        # Calculate accuracy from confusion matrix
        if cm.size > 0:
            accuracy = np.trace(cm) / cm.sum()
            print(f"  Accuracy from confusion matrix (epoch {latest_epoch}): {accuracy:.3f}")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if epochs and max(val_accs) < 0.3:
        print("1. Model is not learning well. Consider:")
        print("   - Reducing learning rate (try 1e-4 or 1e-5)")
        print("   - Checking data quality and labels")
        print("   - Simplifying model architecture")
    elif epochs and max(val_accs) > 0.8:
        print("1. Model is performing well!")
        print("   - Consider deploying or testing on harder examples")
    else:
        print("1. Model is learning but could improve. Consider:")
        print("   - Training for more epochs")
        print("   - Tuning hyperparameters")
        print("   - Adding regularization")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_metrics(sys.argv[1])
    else:
        check_metrics()