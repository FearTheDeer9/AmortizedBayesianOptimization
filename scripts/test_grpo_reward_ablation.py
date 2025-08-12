#!/usr/bin/env python3
"""
Ablation study for different reward functions in GRPO.
Tests which reward formulation best addresses the zero loss issue.
"""

import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import logging
from collections import defaultdict
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('jax').setLevel(logging.WARNING)

from src.causal_bayes_opt.training.grpo_enhanced_trainer import GRPOEnhancedTrainer
from src.causal_bayes_opt.experiments.benchmark_scms import create_chain_scm


def collect_metrics_during_training(trainer):
    """Extract key metrics from trainer after training."""
    metrics = {
        'reward_variance': [],
        'advantage_symmetry': [],
        'loss_magnitude': [],
        'unique_rewards': [],
        'policy_entropy': [],
        'parent_selection_rate': 0.0,
        'final_performance': 0.0
    }
    
    # Extract from reward history if available
    if hasattr(trainer, 'reward_history') and trainer.reward_history:
        from src.causal_bayes_opt.acquisition.grpo_rewards import analyze_reward_distribution
        analysis = analyze_reward_distribution(trainer.reward_history)
        metrics['parent_selection_rate'] = analysis['binary_signals']['parent_selection_rate']
    
    # Extract from gradient history
    if hasattr(trainer, 'gradient_history') and trainer.gradient_history:
        for hist in trainer.gradient_history:
            if 'effective_update' in hist:
                metrics['loss_magnitude'].append(hist['effective_update'])
    
    return metrics


def run_ablation_study():
    """Run ablation study comparing different reward functions."""
    
    # Define configurations to test
    configs = [
        ("Binary (Baseline)", {
            'reward_type': 'binary',
            'description': 'Original binary rewards with thresholding'
        }),
        ("Continuous", {
            'reward_type': 'continuous', 
            'description': 'Continuous improvement ratio'
        }),
        ("Scaled Binary", {
            'reward_type': 'scaled_binary',
            'description': 'Binary + small continuous component'
        }),
        ("Direct Value", {
            'reward_type': 'direct_value',
            'description': 'Direct target value optimization'
        })
    ]
    
    # Storage for results
    all_results = {}
    
    # Run each configuration
    for config_name, config_overrides in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {config_name}")
        logger.info(f"Description: {config_overrides['description']}")
        logger.info(f"{'='*60}")
        
        # Base configuration
        config = {
            'policy_architecture': 'alternating_attention',
            'use_grpo_rewards': True,
            'group_size': 4,
            'learning_rate': 3e-4,
            'n_episodes': 10,  # Reduced for faster testing
            'episode_length': 5,
            'batch_size': 16,
            'use_early_stopping': False,
            'optimization_direction': "MINIMIZE",
            'use_surrogate': False,
            'seed': 42,
            'grpo_reward_config': {
                'reward_type': config_overrides['reward_type'],  # Key parameter
                'reward_weights': {
                    'variable_selection': 0.3,
                    'value_selection': 0.7,
                    'parent_bonus': 0.2,
                    'improvement_bonus': 0.3
                },
                'improvement_threshold': 0.1
            }
        }
        
        # Create trainer
        trainer = GRPOEnhancedTrainer(**config)
        
        # Create SCM
        scms = {'chain': create_chain_scm()}
        
        # Collect diagnostic data during training
        diagnostic_data = defaultdict(list)
        
        # Monkey-patch logging to capture diagnostics
        original_info = logger.info
        def capture_info(msg):
            original_info(msg)
            # Capture specific diagnostic patterns
            if "Unique rewards:" in msg:
                parts = msg.split("values: ")[-1].strip("[]")
                unique_count = len(parts.split(", "))
                diagnostic_data['unique_rewards'].append(unique_count)
            elif "Advantages symmetric:" in msg:
                is_symmetric = "True" in msg
                diagnostic_data['advantage_symmetry'].append(is_symmetric)
            elif "Reward variance:" in msg:
                variance = float(msg.split("Reward variance: ")[-1])
                diagnostic_data['reward_variance'].append(variance)
            elif "Policy loss:" in msg and "Episode" in msg:
                loss = float(msg.split("Policy loss: ")[-1])
                diagnostic_data['policy_loss'].append(loss)
        
        logger.info = capture_info
        
        # Train
        try:
            result = trainer.train(scms)
            
            # Restore original logger
            logger.info = original_info
            
            # Collect final metrics
            metrics = collect_metrics_during_training(trainer)
            metrics['diagnostic_data'] = dict(diagnostic_data)
            
            # Store results
            all_results[config_name] = {
                'config': config_overrides,
                'metrics': metrics,
                'result': result
            }
            
            # Log summary
            logger.info(f"\n[SUMMARY] {config_name}:")
            logger.info(f"  Average unique rewards: {np.mean(diagnostic_data['unique_rewards']) if diagnostic_data['unique_rewards'] else 0:.2f}")
            logger.info(f"  Symmetry rate: {np.mean(diagnostic_data['advantage_symmetry']) if diagnostic_data['advantage_symmetry'] else 0:.2%}")
            logger.info(f"  Average reward variance: {np.mean(diagnostic_data['reward_variance']) if diagnostic_data['reward_variance'] else 0:.6f}")
            logger.info(f"  Average policy loss: {np.mean(diagnostic_data['policy_loss']) if diagnostic_data['policy_loss'] else 0:.6f}")
            
        except Exception as e:
            logger.error(f"Error in {config_name}: {e}")
            logger.info = original_info
            all_results[config_name] = {'error': str(e)}
    
    # Plot comparison
    plot_ablation_results(all_results)
    
    # Save results
    save_ablation_results(all_results)
    
    return all_results


def plot_ablation_results(results):
    """Plot comparison of different reward functions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Extract data for plotting
    config_names = []
    unique_rewards = []
    symmetry_rates = []
    reward_variances = []
    policy_losses = []
    
    for name, data in results.items():
        if 'error' not in data:
            config_names.append(name)
            diag = data['metrics'].get('diagnostic_data', {})
            unique_rewards.append(np.mean(diag.get('unique_rewards', [])) if diag.get('unique_rewards') else 0)
            symmetry_rates.append(np.mean(diag.get('advantage_symmetry', [])) if diag.get('advantage_symmetry') else 0)
            reward_variances.append(np.mean(diag.get('reward_variance', [])) if diag.get('reward_variance') else 0)
            policy_losses.append(np.mean(np.abs(diag.get('policy_loss', []))) if diag.get('policy_loss') else 0)
    
    # Plot 1: Unique rewards
    ax = axes[0]
    bars = ax.bar(range(len(config_names)), unique_rewards)
    ax.set_xticks(range(len(config_names)))
    ax.set_xticklabels(config_names, rotation=45, ha='right')
    ax.set_ylabel('Average Unique Rewards')
    ax.set_title('Reward Diversity')
    ax.axhline(y=4, color='green', linestyle='--', alpha=0.5, label='Max possible')
    
    # Plot 2: Symmetry rate
    ax = axes[1]
    bars = ax.bar(range(len(config_names)), symmetry_rates)
    ax.set_xticks(range(len(config_names)))
    ax.set_xticklabels(config_names, rotation=45, ha='right')
    ax.set_ylabel('Advantage Symmetry Rate')
    ax.set_title('Symmetry Issues')
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Ideal')
    
    # Plot 3: Reward variance
    ax = axes[2]
    bars = ax.bar(range(len(config_names)), reward_variances)
    ax.set_xticks(range(len(config_names)))
    ax.set_xticklabels(config_names, rotation=45, ha='right')
    ax.set_ylabel('Average Reward Variance')
    ax.set_title('Reward Spread')
    
    # Plot 4: Policy loss magnitude
    ax = axes[3]
    bars = ax.bar(range(len(config_names)), policy_losses)
    ax.set_xticks(range(len(config_names)))
    ax.set_xticklabels(config_names, rotation=45, ha='right')
    ax.set_ylabel('Average |Policy Loss|')
    ax.set_title('Learning Signal Strength')
    
    plt.suptitle('GRPO Reward Function Ablation Study', fontsize=14)
    plt.tight_layout()
    plt.savefig('grpo_reward_ablation.png', dpi=150, bbox_inches='tight')
    logger.info("Plots saved to grpo_reward_ablation.png")


def save_ablation_results(results):
    """Save ablation results to file for later analysis."""
    # Convert to serializable format
    serializable_results = {}
    for name, data in results.items():
        if 'error' in data:
            serializable_results[name] = {'error': data['error']}
        else:
            serializable_results[name] = {
                'config': data['config'],
                'metrics': {
                    'parent_selection_rate': data['metrics'].get('parent_selection_rate', 0),
                    'diagnostic_summary': {
                        'unique_rewards': np.mean(data['metrics'].get('diagnostic_data', {}).get('unique_rewards', [])) 
                                        if data['metrics'].get('diagnostic_data', {}).get('unique_rewards') else 0,
                        'symmetry_rate': np.mean(data['metrics'].get('diagnostic_data', {}).get('advantage_symmetry', [])) 
                                       if data['metrics'].get('diagnostic_data', {}).get('advantage_symmetry') else 0,
                        'reward_variance': np.mean(data['metrics'].get('diagnostic_data', {}).get('reward_variance', [])) 
                                         if data['metrics'].get('diagnostic_data', {}).get('reward_variance') else 0,
                        'policy_loss': np.mean(data['metrics'].get('diagnostic_data', {}).get('policy_loss', [])) 
                                     if data['metrics'].get('diagnostic_data', {}).get('policy_loss') else 0
                    }
                }
            }
    
    with open('grpo_ablation_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    logger.info("Results saved to grpo_ablation_results.json")


if __name__ == "__main__":
    results = run_ablation_study()
    
    # Print final comparison
    logger.info("\n" + "="*60)
    logger.info("FINAL COMPARISON")
    logger.info("="*60)
    
    for name, data in results.items():
        if 'error' not in data:
            diag = data['metrics'].get('diagnostic_data', {})
            logger.info(f"\n{name}:")
            logger.info(f"  Unique rewards: {np.mean(diag.get('unique_rewards', [])) if diag.get('unique_rewards') else 0:.2f}")
            logger.info(f"  Symmetry rate: {np.mean(diag.get('advantage_symmetry', [])) if diag.get('advantage_symmetry') else 0:.2%}")
            logger.info(f"  Reward variance: {np.mean(diag.get('reward_variance', [])) if diag.get('reward_variance') else 0:.6f}")
            logger.info(f"  |Policy loss|: {np.mean(np.abs(diag.get('policy_loss', []))) if diag.get('policy_loss') else 0:.6f}")