#!/usr/bin/env python3
"""
Compare different policy architectures and std configurations.
"""

import sys
import os
from pathlib import Path
import numpy as np
import time
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.experiments.benchmark_scms import create_chain_scm
from src.causal_bayes_opt.training.joint_acbo_trainer import JointACBOTrainer


class ArchitectureComparisonTrainer(JointACBOTrainer):
    """Track performance for architecture comparison."""
    
    def __init__(self, config):
        super().__init__(config=config)
        self.x1_values = []
        self.target_values = []
        
    def _select_best_grpo_intervention(self, candidates):
        """Track interventions."""
        best = super()._select_best_grpo_intervention(candidates)
        
        if best['variable'] == 'X1':
            self.x1_values.append(best['value'])
        if 'target_value' in best:
            self.target_values.append(best['target_value'])
            
        return best


def run_single_configuration(
    config_name: str,
    architecture: str,
    use_fixed_std: bool,
    fixed_std: float = 0.5
) -> Dict[str, Any]:
    """Run training with a single configuration."""
    
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print(f"{'='*60}")
    print(f"Architecture: {architecture}")
    print(f"Fixed std: {use_fixed_std} ({fixed_std if use_fixed_std else 'learned'})")
    
    # Create SCM
    scm = create_chain_scm(
        chain_length=3,
        variable_ranges={
            'X0': (-2.0, 2.0),
            'X1': (-5.0, 5.0),
            'X2': (-10.0, 10.0)
        }
    )
    
    # Config
    config = {
        'max_episodes': 10,  # Moderate scale
        'obs_per_episode': 10,
        'max_interventions': 30,
        
        'policy_architecture': architecture,
        'use_fixed_std': use_fixed_std,
        'fixed_std': fixed_std,
        
        'episodes_per_phase': 1000,
        'use_surrogate': True,
        'use_grpo_rewards': True,
        
        'learning_rate': 5e-4,
        
        'grpo_config': {
            'group_size': 10,
            'entropy_coefficient': 0.001,
            'clip_ratio': 0.2,
            'gradient_clip': 1.0,
            'ppo_epochs': 4
        },
        
        'joint_training': {
            'loss_weights': {
                'policy': {
                    'target_delta': 0.9,
                    'direct_parent': 0.1,
                    'information_gain': 0.0
                }
            }
        },
        
        'checkpoint_dir': f'comparison_{config_name.replace(" ", "_")}',
        'verbose': False
    }
    
    # Run training
    start_time = time.time()
    trainer = ArchitectureComparisonTrainer(config)
    results = trainer.train([scm])
    training_time = time.time() - start_time
    
    # Analyze results
    analysis = {
        'config_name': config_name,
        'architecture': architecture,
        'use_fixed_std': use_fixed_std,
        'fixed_std': fixed_std if use_fixed_std else None,
        'training_time': training_time,
        'x1_values': trainer.x1_values,
        'target_values': trainer.target_values
    }
    
    # Compute metrics
    if trainer.x1_values:
        x1_vals = trainer.x1_values
        analysis['x1_mean'] = np.mean(x1_vals)
        analysis['x1_std'] = np.std(x1_vals)
        analysis['x1_min'] = min(x1_vals)
        analysis['x1_max'] = max(x1_vals)
        analysis['x1_range'] = max(x1_vals) - min(x1_vals)
        analysis['x1_range_pct'] = analysis['x1_range'] / 10.0 * 100
        
        # Count extreme values
        extreme_low = sum(1 for v in x1_vals if v < -3)
        extreme_high = sum(1 for v in x1_vals if v > 3)
        analysis['x1_extreme_pct'] = (extreme_low + extreme_high) / len(x1_vals) * 100
    
    if trainer.target_values:
        target_vals = trainer.target_values
        analysis['target_mean'] = np.mean(target_vals)
        analysis['target_best'] = min(target_vals)
        analysis['target_worst'] = max(target_vals)
    
    return analysis


def compare_architectures():
    """Compare different architecture configurations."""
    
    print("\n" + "="*80)
    print("ARCHITECTURE COMPARISON EXPERIMENT")
    print("="*80)
    
    # Define configurations to test
    configurations = [
        # Original with learned std
        {
            'name': 'Original + Learned Std',
            'architecture': 'permutation_invariant',
            'use_fixed_std': False
        },
        # Original with fixed std
        {
            'name': 'Original + Fixed Std',
            'architecture': 'permutation_invariant', 
            'use_fixed_std': True,
            'fixed_std': 0.5
        },
        # Simplified with fixed std
        {
            'name': 'Simplified + Fixed Std',
            'architecture': 'simple_permutation_invariant',
            'use_fixed_std': True,
            'fixed_std': 0.5
        },
        # Simplified with range-based std
        {
            'name': 'Simplified + Range-based Std',
            'architecture': 'simple_permutation_invariant',
            'use_fixed_std': True,
            'fixed_std': 1.0  # Larger std for more exploration
        },
        # Simple baseline
        {
            'name': 'Simple MLP + Fixed Std',
            'architecture': 'simple',
            'use_fixed_std': True,
            'fixed_std': 0.5
        }
    ]
    
    # Run each configuration
    results = []
    for config in configurations:
        result = run_single_configuration(
            config_name=config['name'],
            architecture=config['architecture'],
            use_fixed_std=config['use_fixed_std'],
            fixed_std=config.get('fixed_std', 0.5)
        )
        results.append(result)
        
        # Print immediate results
        print(f"\n📊 Results for {config['name']}:")
        if 'x1_range_pct' in result:
            print(f"  X1 range utilization: {result['x1_range_pct']:.1f}%")
            print(f"  X1 extreme values: {result['x1_extreme_pct']:.1f}%")
        if 'target_best' in result:
            print(f"  Best target: {result['target_best']:.3f}")
            print(f"  Mean target: {result['target_mean']:.3f}")
        print(f"  Training time: {result['training_time']:.1f}s")
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    # Create comparison table
    print("\n📊 Performance Comparison:")
    print("-"*80)
    print(f"{'Configuration':<30} {'Range %':<10} {'Best Target':<12} {'Mean Target':<12} {'Time (s)':<10}")
    print("-"*80)
    
    for result in results:
        name = result['config_name'][:30]
        range_pct = f"{result.get('x1_range_pct', 0):.1f}%" if 'x1_range_pct' in result else "N/A"
        best_target = f"{result.get('target_best', 0):.3f}" if 'target_best' in result else "N/A"
        mean_target = f"{result.get('target_mean', 0):.3f}" if 'target_mean' in result else "N/A"
        time_str = f"{result['training_time']:.1f}"
        
        print(f"{name:<30} {range_pct:<10} {best_target:<12} {mean_target:<12} {time_str:<10}")
    
    # Find winner
    print("\n" + "="*80)
    print("WINNER ANALYSIS")
    print("="*80)
    
    # Best by target value
    best_target_result = min(results, key=lambda x: x.get('target_best', float('inf')))
    print(f"\n🏆 Best target value: {best_target_result['config_name']}")
    print(f"   Achieved: {best_target_result.get('target_best', 'N/A'):.3f}")
    
    # Best by range utilization
    best_range_result = max(results, key=lambda x: x.get('x1_range_pct', 0))
    print(f"\n🏆 Best range utilization: {best_range_result['config_name']}")
    print(f"   Achieved: {best_range_result.get('x1_range_pct', 0):.1f}%")
    
    # Fastest
    fastest_result = min(results, key=lambda x: x['training_time'])
    print(f"\n🏆 Fastest training: {fastest_result['config_name']}")
    print(f"   Time: {fastest_result['training_time']:.1f}s")
    
    # Overall recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    # Score each configuration
    for result in results:
        score = 0
        # Target performance (most important)
        if 'target_best' in result:
            score += (10 - result['target_best']) * 10  # Lower is better
        # Range utilization
        if 'x1_range_pct' in result:
            score += result['x1_range_pct'] / 10
        # Speed bonus
        score += 10 / result['training_time']
        result['overall_score'] = score
    
    best_overall = max(results, key=lambda x: x.get('overall_score', 0))
    print(f"\n✅ Recommended configuration: {best_overall['config_name']}")
    print(f"   Overall score: {best_overall.get('overall_score', 0):.1f}")
    print("\nRationale:")
    print(f"  - Target performance: {best_overall.get('target_best', 'N/A'):.3f}")
    print(f"  - Range utilization: {best_overall.get('x1_range_pct', 0):.1f}%")
    print(f"  - Training speed: {best_overall['training_time']:.1f}s")
    
    return results


if __name__ == "__main__":
    results = compare_architectures()