#!/usr/bin/env python3
"""
Test the best GRPO configuration on mixed SCMs with rotation.
This simulates the real-world amortized setting where the model must generalize across different causal structures.
"""

import sys
sys.path.append('.')

import json
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.causal_bayes_opt.training.grpo_enhanced_trainer import GRPOEnhancedTrainer, ENHANCED_TARGET_VALUES
from src.causal_bayes_opt.experiments.benchmark_scms import (
    create_chain_scm, create_fork_scm, create_collider_scm,
    create_diamond_scm, create_butterfly_scm, create_sparse_scm
)


def test_on_mixed_scms(reward_type: str = None, episodes: int = 100):
    """
    Test GRPO with the best reward type on mixed SCMs with rotation.
    
    Args:
        reward_type: Reward type to use (if None, loads from best_reward_config.json)
        episodes: Number of training episodes
    """
    
    # Load best reward type if not provided
    if reward_type is None:
        # Check for override first (for better balance of metrics)
        if os.path.exists('best_reward_config_override.json'):
            with open('best_reward_config_override.json', 'r') as f:
                config_data = json.load(f)
                reward_type = config_data['reward_type']
                logger.info(f"Using override reward type for better balance: {reward_type}")
                logger.info(f"Reason: {config_data.get('reason', 'N/A')}")
        elif os.path.exists('best_reward_config.json'):
            with open('best_reward_config.json', 'r') as f:
                config_data = json.load(f)
                reward_type = config_data['reward_type']
                logger.info(f"Loaded best reward type from previous test: {reward_type}")
        else:
            reward_type = 'continuous'  # Default fallback
            logger.info(f"No best config found, using default: {reward_type}")
    
    logger.info("="*80)
    logger.info("MIXED SCM TRAINING TEST")
    logger.info(f"Reward type: {reward_type}")
    logger.info(f"Episodes: {episodes}")
    logger.info("="*80)
    
    # Configuration with all our fixes
    config = {
        'policy_architecture': 'alternating_attention',
        'use_grpo_rewards': True,
        'group_size': 8,
        'learning_rate': 3e-4,
        'n_episodes': episodes,
        'episode_length': 10,  # Longer episodes for better exploration
        'batch_size': 64,  # Larger batch for stability across SCMs
        'optimization_direction': "MINIMIZE",
        'use_surrogate': False,
        'seed': 42,
        'grpo_config': {
            'ppo_epochs': 4,  # Epoch-based training fix
            'clip_ratio': 0.2,
            'entropy_coeff': 0.01
        },
        'grpo_reward_config': {
            'reward_type': reward_type,
            'reward_weights': {
                'variable_selection': 0.3,
                'value_selection': 0.7,
                'parent_bonus': 0.2,
                'improvement_bonus': 0.1
            },
            'improvement_threshold': 0.1
        }
    }
    
    trainer = GRPOEnhancedTrainer(**config)
    
    # Create all SCM types
    scms = {
        'chain': create_chain_scm(),
        'fork': create_fork_scm(),
        'collider': create_collider_scm(),
        'diamond': create_diamond_scm(),
        'butterfly': create_butterfly_scm(),
        'sparse': create_sparse_scm()
    }
    
    logger.info(f"Training on {len(scms)} different SCM structures:")
    for name in scms.keys():
        logger.info(f"  - {name}")
    
    # Clear target value tracking
    ENHANCED_TARGET_VALUES.clear()
    
    # Train with rotation
    result = trainer.train(scms)
    
    # Analyze results per SCM
    logger.info("\n" + "="*80)
    logger.info("PER-SCM ANALYSIS")
    logger.info("="*80)
    
    scm_metrics = {}
    
    # Group target values by SCM
    if ENHANCED_TARGET_VALUES:
        scm_values = {name: [] for name in scms.keys()}
        
        for entry in ENHANCED_TARGET_VALUES:
            scm_name = entry.get('scm_name', 'unknown')
            if scm_name in scm_values:
                scm_values[scm_name].append(entry['value'])
        
        # Analyze each SCM
        for scm_name, values in scm_values.items():
            if len(values) > 10:
                n = len(values)
                early_mean = np.mean(values[:n//3])
                late_mean = np.mean(values[-n//3:])
                improvement = early_mean - late_mean  # For minimization
                improvement_pct = (improvement / abs(early_mean)) * 100 if early_mean != 0 else 0
                
                scm_metrics[scm_name] = {
                    'early_mean': early_mean,
                    'late_mean': late_mean,
                    'absolute_improvement': improvement,
                    'improvement_pct': improvement_pct,
                    'n_interventions': len(values)
                }
                
                logger.info(f"\n{scm_name.upper()} SCM:")
                logger.info(f"  Interventions: {len(values)}")
                logger.info(f"  Early mean: {early_mean:.3f}")
                logger.info(f"  Late mean: {late_mean:.3f}")
                logger.info(f"  Improvement: {improvement:.3f} ({improvement_pct:.1f}%)")
    
    # Overall metrics
    logger.info("\n" + "="*80)
    logger.info("OVERALL PERFORMANCE")
    logger.info("="*80)
    
    if ENHANCED_TARGET_VALUES:
        all_values = [entry['value'] for entry in ENHANCED_TARGET_VALUES]
        n = len(all_values)
        
        # Calculate overall improvement
        early_overall = np.mean(all_values[:n//4])
        late_overall = np.mean(all_values[-n//4:])
        overall_improvement = early_overall - late_overall
        overall_improvement_pct = (overall_improvement / abs(early_overall)) * 100 if early_overall != 0 else 0
        
        logger.info(f"Total interventions: {n}")
        logger.info(f"Early mean (all SCMs): {early_overall:.3f}")
        logger.info(f"Late mean (all SCMs): {late_overall:.3f}")
        logger.info(f"Overall improvement: {overall_improvement:.3f} ({overall_improvement_pct:.1f}%)")
    
    # Parent selection analysis
    if hasattr(trainer, 'reward_history') and trainer.reward_history:
        from src.causal_bayes_opt.acquisition.grpo_rewards import analyze_reward_distribution
        
        early_rewards = trainer.reward_history[:200]
        late_rewards = trainer.reward_history[-200:]
        
        if early_rewards and late_rewards:
            early_analysis = analyze_reward_distribution(early_rewards)
            late_analysis = analyze_reward_distribution(late_rewards)
            
            logger.info(f"\nParent selection:")
            logger.info(f"  Early: {early_analysis['binary_signals']['parent_selection_rate']:.1%}")
            logger.info(f"  Late: {late_analysis['binary_signals']['parent_selection_rate']:.1%}")
            
            # Note about misleading metric
            logger.info(f"\nImprovement rate (misleading metric):")
            logger.info(f"  Early: {early_analysis['binary_signals']['improvement_rate']:.1%}")
            logger.info(f"  Late: {late_analysis['binary_signals']['improvement_rate']:.1%}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Per-SCM improvement bars
    ax = axes[0, 0]
    if scm_metrics:
        scm_names = list(scm_metrics.keys())
        improvements = [scm_metrics[name]['improvement_pct'] for name in scm_names]
        bars = ax.bar(range(len(scm_names)), improvements)
        ax.set_xticks(range(len(scm_names)))
        ax.set_xticklabels(scm_names, rotation=45)
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Per-SCM Absolute Improvement')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Color code
        for bar, imp in zip(bars, improvements):
            if imp > 50:
                bar.set_color('green')
            elif imp > 25:
                bar.set_color('lightgreen')
            elif imp > 0:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
    
    # 2. Learning curves per SCM
    ax = axes[0, 1]
    if ENHANCED_TARGET_VALUES:
        for scm_name in scms.keys():
            scm_specific = [e['value'] for e in ENHANCED_TARGET_VALUES if e.get('scm_name') == scm_name]
            if len(scm_specific) > 5:
                # Smooth the curve
                window = min(10, len(scm_specific) // 5)
                smoothed = [np.mean(scm_specific[max(0, i-window):i+1]) for i in range(len(scm_specific))]
                ax.plot(smoothed, label=scm_name, alpha=0.7)
        ax.set_xlabel('Interventions')
        ax.set_ylabel('Target Value')
        ax.set_title('Learning Curves per SCM')
        ax.legend()
    
    # 3. Overall learning curve
    ax = axes[0, 2]
    if ENHANCED_TARGET_VALUES:
        all_values = [entry['value'] for entry in ENHANCED_TARGET_VALUES]
        ax.plot(all_values, alpha=0.3, color='gray')
        # Smoothed
        window = 50
        smoothed = [np.mean(all_values[max(0, i-window):i+1]) for i in range(len(all_values))]
        ax.plot(smoothed, linewidth=2, color='blue', label='Smoothed')
        ax.set_xlabel('Intervention')
        ax.set_ylabel('Target Value')
        ax.set_title('Overall Learning Progress')
        ax.legend()
    
    # 4. Intervention distribution
    ax = axes[1, 0]
    if scm_metrics:
        scm_names = list(scm_metrics.keys())
        n_interventions = [scm_metrics[name]['n_interventions'] for name in scm_names]
        ax.pie(n_interventions, labels=scm_names, autopct='%1.1f%%')
        ax.set_title('Intervention Distribution Across SCMs')
    
    # 5. Early vs Late comparison
    ax = axes[1, 1]
    if scm_metrics:
        scm_names = list(scm_metrics.keys())
        x = np.arange(len(scm_names))
        width = 0.35
        
        early_means = [scm_metrics[name]['early_mean'] for name in scm_names]
        late_means = [scm_metrics[name]['late_mean'] for name in scm_names]
        
        ax.bar(x - width/2, early_means, width, label='Early', alpha=0.7)
        ax.bar(x + width/2, late_means, width, label='Late', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(scm_names, rotation=45)
        ax.set_ylabel('Mean Target Value')
        ax.set_title('Early vs Late Performance')
        ax.legend()
    
    # 6. Summary
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = f"MIXED SCM TRAINING RESULTS\n\n"
    summary_text += f"Reward Type: {reward_type}\n"
    summary_text += f"Episodes: {episodes}\n"
    summary_text += f"SCMs: {len(scms)}\n\n"
    
    if 'overall_improvement_pct' in locals():
        summary_text += f"Overall Improvement: {overall_improvement_pct:.1f}%\n"
        summary_text += f"Final Mean: {late_overall:.3f}\n\n"
    
    # Best and worst SCM
    if scm_metrics:
        sorted_scms = sorted(scm_metrics.items(), key=lambda x: x[1]['improvement_pct'], reverse=True)
        summary_text += f"Best SCM: {sorted_scms[0][0]} ({sorted_scms[0][1]['improvement_pct']:.1f}%)\n"
        summary_text += f"Worst SCM: {sorted_scms[-1][0]} ({sorted_scms[-1][1]['improvement_pct']:.1f}%)\n"
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('mixed_scm_training_results.png', dpi=150)
    logger.info("\nVisualization saved to mixed_scm_training_results.png")
    
    # Save results
    results = {
        'reward_type': reward_type,
        'episodes': episodes,
        'scm_metrics': scm_metrics,
        'overall_improvement': overall_improvement if 'overall_improvement' in locals() else 0,
        'overall_improvement_pct': overall_improvement_pct if 'overall_improvement_pct' in locals() else 0
    }
    
    with open('mixed_scm_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\nResults saved to mixed_scm_results.json")
    
    # Final verdict
    logger.info("\n" + "="*80)
    logger.info("VERDICT")
    logger.info("="*80)
    
    if 'overall_improvement_pct' in locals() and overall_improvement_pct > 30:
        logger.info(f"✓ SUCCESS: {overall_improvement_pct:.1f}% improvement across {len(scms)} SCMs!")
        logger.info("The model successfully generalizes across different causal structures.")
    else:
        logger.info("⚠ More work needed for better generalization across SCMs.")
    
    return results


def compare_to_baseline():
    """Compare our improved GRPO to the baseline from before our fixes."""
    
    logger.info("\n" + "="*80)
    logger.info("BASELINE COMPARISON")
    logger.info("="*80)
    
    # Load results if they exist
    if os.path.exists('mixed_scm_results.json'):
        with open('mixed_scm_results.json', 'r') as f:
            results = json.load(f)
        
        logger.info("\nOur improved GRPO:")
        logger.info(f"  Overall improvement: {results['overall_improvement_pct']:.1f}%")
        
        logger.info("\nBaseline (before fixes):")
        logger.info("  Parent selection: ~43% → 100%")
        logger.info("  Value selection: ~random (46% 'improvement')")
        logger.info("  Actual improvement: ~0%")
        
        logger.info("\nImprovements from our fixes:")
        logger.info("  ✓ Fixed log probability computation (includes value component)")
        logger.info("  ✓ Implemented epoch-based training (fixes ratio=1.0 issue)")
        logger.info("  ✓ Better reward functions (continuous signals)")
        logger.info("  ✓ Proper group-based advantages")
        logger.info(f"  ✓ Result: {results['overall_improvement_pct']:.1f}% actual improvement!")
    else:
        logger.info("Run test_on_mixed_scms() first to generate results.")


if __name__ == "__main__":
    # First, try to load best reward type from comparison
    best_reward = None
    if os.path.exists('best_reward_config.json'):
        with open('best_reward_config.json', 'r') as f:
            config = json.load(f)
            best_reward = config['reward_type']
            logger.info(f"Using best reward type from comparison: {best_reward}")
    
    # Run the mixed SCM test
    results = test_on_mixed_scms(reward_type=best_reward, episodes=100)
    
    # Compare to baseline
    compare_to_baseline()