#!/usr/bin/env python3
"""
Hyperparameter search for BC training.
Systematically tests different configurations to find optimal settings.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import itertools
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from gradient_analyzer import GradientAnalyzerTrainer
from variable_permutation_trainer import VariablePermutationTrainer


# Hyperparameter search space
SEARCH_SPACE = {
    'learning_rate': [3e-4, 1e-3, 3e-3, 1e-2],
    'hidden_dim': [64, 128, 256],
    'gradient_clip': [None, 0.5, 1.0, 5.0],
    'batch_size': [16, 32],
    'use_permutation': [True, False]
}

# Quick search space for testing
QUICK_SEARCH_SPACE = {
    'learning_rate': [3e-4, 3e-3],
    'hidden_dim': [128, 256],
    'gradient_clip': [1.0, None],
    'batch_size': [32],
    'use_permutation': [True, False]
}


def run_single_experiment(config: Dict[str, Any], 
                         demo_path: str,
                         max_demos: int = 50,
                         max_epochs: int = 30) -> Dict[str, Any]:
    """Run a single training experiment with given configuration."""
    
    print(f"\n{'='*60}")
    print(f"Running experiment with:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Choose trainer based on configuration
        if config['use_permutation']:
            trainer = VariablePermutationTrainer(
                permute_every_epoch=True,
                track_permutation_stats=True,
                hidden_dim=config['hidden_dim'],
                learning_rate=config['learning_rate'],
                batch_size=config['batch_size'],
                gradient_clip=config['gradient_clip'] if config['gradient_clip'] else 1.0,
                max_epochs=max_epochs,
                seed=42
            )
        else:
            trainer = GradientAnalyzerTrainer(
                hidden_dim=config['hidden_dim'],
                learning_rate=config['learning_rate'],
                batch_size=config['batch_size'],
                gradient_clip=config['gradient_clip'] if config['gradient_clip'] else 1.0,
                max_epochs=max_epochs,
                seed=42
            )
        
        # Train
        results = trainer.train(
            demonstrations_path=demo_path,
            max_demos=max_demos,
            output_dir=None  # Don't save individual results
        )
        
        # Extract key metrics
        metrics = results.get('metrics', {})
        gradient_analysis = results.get('gradient_analysis', {})
        
        experiment_result = {
            'config': config,
            'success': True,
            'training_time': time.time() - start_time,
            'final_train_loss': metrics.get('final_train_loss', float('inf')),
            'best_val_loss': metrics.get('best_val_loss', float('inf')),
            'best_val_accuracy': metrics.get('best_val_accuracy', 0.0),
            'final_val_accuracy': metrics.get('latest_metrics', {}).get('val_accuracy', 0.0),
            'gradient_issue': gradient_analysis.get('gradient_issue', 'UNKNOWN'),
            'gradient_norm_mean': gradient_analysis.get('gradient_norm', {}).get('mean', 0.0),
            'learning_signal': gradient_analysis.get('learning_signal', {}).get('mean', 0.0)
        }
        
        # Check for X4 performance if available
        if 'per_variable_stats' in metrics and 'X4' in metrics['per_variable_stats']:
            x4_stats = metrics['per_variable_stats']['X4']
            if x4_stats['attempts'] > 0:
                experiment_result['x4_accuracy'] = x4_stats['correct'] / x4_stats['attempts']
            else:
                experiment_result['x4_accuracy'] = 0.0
        
        print(f"✓ Experiment completed: val_acc={experiment_result['best_val_accuracy']:.3f}")
        
    except Exception as e:
        print(f"✗ Experiment failed: {e}")
        experiment_result = {
            'config': config,
            'success': False,
            'error': str(e),
            'training_time': time.time() - start_time
        }
    
    return experiment_result


def grid_search(search_space: Dict[str, List[Any]],
                demo_path: str,
                max_demos: int = 50,
                max_epochs: int = 30,
                output_file: str = None) -> List[Dict[str, Any]]:
    """Perform grid search over hyperparameter space."""
    
    # Generate all combinations
    keys = list(search_space.keys())
    values = list(search_space.values())
    combinations = list(itertools.product(*values))
    
    print(f"Testing {len(combinations)} configurations...")
    
    results = []
    best_config = None
    best_accuracy = 0.0
    
    for i, combo in enumerate(combinations):
        config = dict(zip(keys, combo))
        print(f"\n[{i+1}/{len(combinations)}] Testing configuration...")
        
        result = run_single_experiment(config, demo_path, max_demos, max_epochs)
        results.append(result)
        
        # Track best configuration
        if result['success'] and result.get('best_val_accuracy', 0) > best_accuracy:
            best_accuracy = result['best_val_accuracy']
            best_config = config
        
        # Save intermediate results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
    
    return results, best_config, best_accuracy


def analyze_results(results: List[Dict[str, Any]]):
    """Analyze and summarize search results."""
    
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("="*80)
    
    # Filter successful experiments
    successful = [r for r in results if r['success']]
    
    if not successful:
        print("No successful experiments!")
        return
    
    # Sort by validation accuracy
    successful.sort(key=lambda x: x.get('best_val_accuracy', 0), reverse=True)
    
    print(f"\nTop 5 configurations by validation accuracy:")
    print("-" * 80)
    
    for i, result in enumerate(successful[:5]):
        config = result['config']
        print(f"\n{i+1}. Validation Accuracy: {result['best_val_accuracy']:.3f}")
        print(f"   Config: LR={config['learning_rate']}, Hidden={config['hidden_dim']}, "
              f"Clip={config['gradient_clip']}, Batch={config['batch_size']}, "
              f"Permute={config['use_permutation']}")
        print(f"   Gradient status: {result.get('gradient_issue', 'N/A')}")
        print(f"   Gradient norm: {result.get('gradient_norm_mean', 0):.6f}")
        print(f"   Learning signal: {result.get('learning_signal', 0):.6f}")
        if 'x4_accuracy' in result:
            print(f"   X4 accuracy: {result['x4_accuracy']:.3f}")
    
    # Analyze impact of each hyperparameter
    print("\n" + "="*60)
    print("HYPERPARAMETER IMPACT ANALYSIS")
    print("="*60)
    
    for param in ['learning_rate', 'hidden_dim', 'gradient_clip', 'use_permutation']:
        print(f"\n{param}:")
        
        # Group by parameter value
        param_groups = {}
        for result in successful:
            value = result['config'][param]
            if value not in param_groups:
                param_groups[value] = []
            param_groups[value].append(result['best_val_accuracy'])
        
        # Calculate statistics
        for value, accuracies in sorted(param_groups.items()):
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            print(f"  {value}: {mean_acc:.3f} ± {std_acc:.3f} (n={len(accuracies)})")
    
    # Gradient analysis
    print("\n" + "="*60)
    print("GRADIENT ANALYSIS")
    print("="*60)
    
    gradient_issues = {'VANISHING': [], 'NORMAL': [], 'EXPLODING': []}
    for result in successful:
        issue = result.get('gradient_issue', 'UNKNOWN')
        if issue in gradient_issues:
            gradient_issues[issue].append(result['config'])
    
    for issue, configs in gradient_issues.items():
        if configs:
            print(f"\n{issue} gradients ({len(configs)} configs):")
            if issue == 'VANISHING':
                lrs = set(c['learning_rate'] for c in configs)
                print(f"  Common learning rates: {sorted(lrs)}")
                print("  → Recommendation: Increase learning rate")
            elif issue == 'EXPLODING':
                clips = set(c['gradient_clip'] for c in configs)
                print(f"  Common gradient clips: {sorted(clips)}")
                print("  → Recommendation: Reduce learning rate or increase clipping")


def main():
    """Main hyperparameter search function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter search for BC training')
    parser.add_argument('--demo_path', default='expert_demonstrations/raw/raw_demonstrations')
    parser.add_argument('--max_demos', type=int, default=50)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--quick', action='store_true', help='Use smaller search space')
    parser.add_argument('--output', default='debugging-bc-training/hyperparam_results.json')
    
    args = parser.parse_args()
    
    # Choose search space
    search_space = QUICK_SEARCH_SPACE if args.quick else SEARCH_SPACE
    
    print(f"Starting hyperparameter search")
    print(f"Search space: {len(list(itertools.product(*search_space.values())))} configurations")
    print(f"Quick mode: {args.quick}")
    
    # Run search
    results, best_config, best_accuracy = grid_search(
        search_space,
        args.demo_path,
        args.max_demos,
        args.max_epochs,
        args.output
    )
    
    # Analyze results
    analyze_results(results)
    
    # Print best configuration
    if best_config:
        print("\n" + "="*80)
        print("BEST CONFIGURATION")
        print("="*80)
        print(f"Validation Accuracy: {best_accuracy:.3f}")
        print("Configuration:")
        for key, value in best_config.items():
            print(f"  {key}: {value}")
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()