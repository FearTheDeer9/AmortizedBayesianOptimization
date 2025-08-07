#!/usr/bin/env python3
"""
Simple test that logs actual target values by modifying reward calculation.
"""

import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.causal_bayes_opt.training.unified_grpo_trainer import UnifiedGRPOTrainer
from src.causal_bayes_opt.training.grpo_enhanced_trainer import GRPOEnhancedTrainer
from src.causal_bayes_opt.experiments.benchmark_scms import (
    create_fork_scm, create_chain_scm, create_collider_scm
)
from src.causal_bayes_opt.data_structures.sample import get_values
from src.causal_bayes_opt.acquisition.better_rewards import compute_better_clean_reward, RunningStats
from src.causal_bayes_opt.acquisition.grpo_rewards import compute_grpo_reward, GRPORewardComponents
import src.causal_bayes_opt.training.unified_grpo_trainer


# Global storage for target values and discrimination metrics
TARGET_VALUES = []
DISCRIMINATION_METRICS = []
REWARD_COMPONENTS = []  # Track detailed reward components


def compute_grpo_reward_with_logging(
    scm,
    intervention,
    outcome,
    target_variable,
    buffer_before,
    config=None,
    group_outcomes=None
):
    """Wrapper that logs target values and reward components for GRPO rewards."""
    # Extract target value
    outcome_values = get_values(outcome)
    if target_variable in outcome_values:
        target_value = float(outcome_values[target_variable])
        TARGET_VALUES.append({
            'value': target_value,
            'target_var': target_variable,
            'intervention': intervention
        })
        logger.info(f"[LOGGED] Target {target_variable} = {target_value:.3f} after intervention on {intervention.get('targets', 'unknown')}")
    
    # Call original function
    reward_comp = compute_grpo_reward(
        scm=scm,
        intervention=intervention,
        outcome=outcome,
        target_variable=target_variable,
        buffer_before=buffer_before,
        config=config,
        group_outcomes=group_outcomes
    )
    
    # Store reward components
    REWARD_COMPONENTS.append(reward_comp)
    
    # Log reward decomposition occasionally
    if len(REWARD_COMPONENTS) % 50 == 1:
        logger.info(
            f"[REWARD DETAILS] "
            f"Parent: {reward_comp.parent_intervention:.3f}, "
            f"Improvement: {reward_comp.target_improvement:.3f}, "
            f"Total: {reward_comp.total_reward:.3f}, "
            f"IsParent: {reward_comp.correct_parent}"
        )
    
    return reward_comp


def compute_better_clean_reward_with_logging(
    buffer_before,
    intervention,
    outcome,
    target_variable,
    config=None,
    stats=None,
    posterior_before=None,
    posterior_after=None
):
    """Wrapper that logs target values before computing reward."""
    # Extract target value
    outcome_values = get_values(outcome)
    if target_variable in outcome_values:
        target_value = float(outcome_values[target_variable])
        TARGET_VALUES.append({
            'value': target_value,
            'target_var': target_variable,
            'intervention': intervention
        })
        logger.info(f"[LOGGED] Target {target_variable} = {target_value:.3f} after intervention on {intervention.get('targets', 'unknown')}")
    else:
        logger.warning(f"[WARNING] Target variable {target_variable} not found in outcome: {outcome_values.keys()}")
    
    # Call original function
    return compute_better_clean_reward(
        buffer_before=buffer_before,
        intervention=intervention,
        outcome=outcome,
        target_variable=target_variable,
        config=config,
        stats=stats,
        posterior_before=posterior_before,
        posterior_after=posterior_after
    )


# Store original function for verification
original_compute_reward = src.causal_bayes_opt.training.unified_grpo_trainer.compute_better_clean_reward


# Store original _run_grpo_episode method
original_run_grpo_episode = src.causal_bayes_opt.training.unified_grpo_trainer.UnifiedGRPOTrainer._run_grpo_episode


def _run_grpo_episode_with_tracking(self, episode_idx, scm, scm_name, key):
    """Wrapper to track discrimination metrics during episodes."""
    # Check if we should log discrimination metrics (every 5 episodes)
    if episode_idx % 5 == 0:
        # Access internal state to get logits info
        # This is a bit hacky but necessary for debugging
        self._track_discrimination = True
        self._current_episode_idx = episode_idx
        self._current_scm_name = scm_name
    else:
        self._track_discrimination = False
    
    # Call original method
    result = original_run_grpo_episode(self, episode_idx, scm, scm_name, key)
    
    return result


# Monkey-patch the episode runner
src.causal_bayes_opt.training.unified_grpo_trainer.UnifiedGRPOTrainer._run_grpo_episode = _run_grpo_episode_with_tracking


def test_and_plot():
    """Run tests and plot results."""
    
    # MINIMAL TEST: Single configuration, single SCM
    configs = [
        ("Enhanced GRPO (chain only)", {
            'trainer_class': GRPOEnhancedTrainer,
            'policy_architecture': 'alternating_attention',
            'use_grpo_rewards': True,
            'group_size': 4,
        }),
    ]
    
    all_results = {}
    
    for config_name, overrides in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {config_name}")
        logger.info(f"{'='*60}")
        
        # Clear target values for this configuration
        TARGET_VALUES.clear()
        REWARD_COMPONENTS.clear()
        logger.info(f"Cleared TARGET_VALUES and REWARD_COMPONENTS, starting fresh for {config_name}")
        
        # Extract trainer class
        trainer_class = overrides.pop('trainer_class', UnifiedGRPOTrainer)
        
        # Default config - increased episodes for single SCM
        config = {
            'learning_rate': 3e-4,
            'n_episodes': 100,  # More episodes for single SCM
            'episode_length': 10,
            'batch_size': 16,
            'use_early_stopping': False,
            'reward_weights': {
                'optimization': 1.0,
                'discovery': 0.0,
                'efficiency': 0.0,
                'info_gain': 0.0
            },
            'optimization_direction': "MINIMIZE",
            'use_surrogate': False,
            'seed': 42,
            'convergence_config': {
                'patience': 1000,
                'min_episodes': 100,
                'max_episodes_per_scm': 100  # Allow all episodes on single SCM
            },
            'grpo_reward_config': {
                'reward_weights': {
                    'variable_selection': 0.3,  # Reduced since this is mostly solved
                    'value_selection': 0.7,     # Increased to emphasize value learning
                    'parent_bonus': 0.3,
                    'improvement_bonus': 0.3    # Increased to reward improvements more
                },
                'improvement_threshold': 0.05   # Lowered threshold for easier improvement detection
            }
        }
        
        # Apply overrides
        config.update(overrides)
        
        # Apply monkey-patches based on trainer class
        if trainer_class.__name__ != 'GRPOEnhancedTrainer':
            # Patch the unified trainer reward function for baseline
            import src.causal_bayes_opt.training.unified_grpo_trainer as unified_trainer_module
            unified_trainer_module.compute_better_clean_reward = compute_better_clean_reward_with_logging
            logger.info("Patched compute_better_clean_reward for baseline tracking")
        else:
            logger.info("Enhanced trainer will log target values directly")
        
        # Also patch the episode runner for discrimination metrics
        import src.causal_bayes_opt.training.unified_grpo_trainer as unified_trainer_module2
        unified_trainer_module2.UnifiedGRPOTrainer._run_grpo_episode = _run_grpo_episode_with_tracking
        
        # Create trainer
        trainer = trainer_class(**config)
        
        # Create SCMs - ONLY CHAIN for minimal test
        scms = {
            'chain': create_chain_scm(), 
        }
        
        # Log SCM structure for clarity
        logger.info("\n[SCM STRUCTURE] Chain SCM:")
        logger.info("  X0 (no parents)")
        logger.info("  X1 (parent: X0)")
        logger.info("  X2 (parent: X1) - TARGET")
        logger.info("  Optimal: Intervene on X1 (direct parent of target)")
        
        # Train
        result = trainer.train(scms)
        
        # Log how many values were collected
        logger.info(f"Training complete. Collected {len(TARGET_VALUES)} target values from baseline")
        
        # For enhanced trainer, also get values from its tracking
        if trainer_class.__name__ == 'GRPOEnhancedTrainer':
            from src.causal_bayes_opt.training.grpo_enhanced_trainer import ENHANCED_TARGET_VALUES
            logger.info(f"Enhanced trainer collected {len(ENHANCED_TARGET_VALUES)} target values")
            # Use enhanced values if available
            if ENHANCED_TARGET_VALUES:
                config_target_values = ENHANCED_TARGET_VALUES.copy()
                # Clear for next run
                ENHANCED_TARGET_VALUES.clear()
            else:
                config_target_values = TARGET_VALUES.copy()
        else:
            # Make a copy of TARGET_VALUES for this configuration
            config_target_values = TARGET_VALUES.copy()
        
        # Group target values by SCM
        scm_values = {'fork': [], 'chain': [], 'collider': []}
        current_scm = 'fork'
        scm_counter = {'fork': 0, 'chain': 0, 'collider': 0}
        
        for i, tv in enumerate(config_target_values):
            # Determine which SCM based on target variable
            if tv['target_var'] == 'Y':
                current_scm = 'fork'
            elif tv['target_var'] == 'X2':
                current_scm = 'chain'
            elif tv['target_var'] == 'Z':
                current_scm = 'collider'
            
            scm_values[current_scm].append(tv['value'])
            scm_counter[current_scm] += 1
        
        # Calculate improvements
        improvements = {}
        for scm_name, values in scm_values.items():
            if len(values) >= 10:
                early_mean = np.mean(values[:10])
                late_mean = np.mean(values[-10:])
                improvement = early_mean - late_mean  # Positive = good for MINIMIZE
                improvements[scm_name] = improvement
                
                logger.info(f"{scm_name}: {len(values)} samples, "
                           f"early={early_mean:.3f}, late={late_mean:.3f}, "
                           f"improvement={improvement:+.3f}")
            else:
                improvements[scm_name] = 0.0
                logger.info(f"{scm_name}: Only {len(values)} samples (too few)")
        
        # Analyze reward components if available
        reward_analysis = {}
        if REWARD_COMPONENTS:
            from src.causal_bayes_opt.acquisition.grpo_rewards import analyze_reward_distribution
            reward_analysis = analyze_reward_distribution(REWARD_COMPONENTS)
            
            logger.info(f"\n[REWARD COMPONENT ANALYSIS] {config_name}:")
            logger.info(f"  Total reward samples: {reward_analysis['n_samples']}")
            logger.info(f"  Parent selection rate: {reward_analysis['binary_signals']['parent_selection_rate']:.3f}")
            logger.info(f"  Improvement rate: {reward_analysis['binary_signals']['improvement_rate']:.3f}")
            logger.info(f"  Target improvement mean: {reward_analysis['target_improvement']['mean']:.3f}")
            logger.info(f"  Parent intervention mean: {reward_analysis['parent_intervention']['mean']:.3f}")
        
        all_results[config_name] = {
            'scm_values': scm_values,
            'improvements': improvements,
            'avg_improvement': np.mean(list(improvements.values())),
            'discrimination_metrics': DISCRIMINATION_METRICS.copy(),
            'reward_analysis': reward_analysis,
            'trainer': trainer if 'Enhanced' in config_name else None  # Store trainer for enhanced configs
        }
        
        # Analyze discrimination metrics
        if DISCRIMINATION_METRICS:
            logger.info(f"\n[DISCRIMINATION ANALYSIS] {config_name}:")
            
            # Group by episode
            episode_groups = defaultdict(list)
            for metric in DISCRIMINATION_METRICS:
                episode_groups[metric['episode']].append(metric)
            
            # Compute average metrics over time
            episodes = sorted(episode_groups.keys())
            avg_logit_spread = []
            avg_embedding_var = []
            parent_selection_rate = []
            parent_logit_advantage = []
            
            for ep in episodes:
                metrics = episode_groups[ep]
                avg_logit_spread.append(np.mean([m['logit_spread'] for m in metrics]))
                avg_embedding_var.append(np.mean([m['embedding_variance'] for m in metrics]))
                
                # Calculate parent selection rate
                parent_selections = [m['is_direct_parent'] for m in metrics]
                parent_selection_rate.append(np.mean(parent_selections))
                
                # Average parent logit advantage
                parent_advantages = [m['parent_logit_advantage'] for m in metrics]
                parent_logit_advantage.append(np.mean(parent_advantages))
            
            logger.info(f"  Episodes tracked: {episodes}")
            logger.info(f"  Logit spread trend: {[f'{x:.3f}' for x in avg_logit_spread]}")
            logger.info(f"  Embedding variance trend: {[f'{x:.3f}' for x in avg_embedding_var]}")
            logger.info(f"  Parent selection rate: {[f'{x:.3f}' for x in parent_selection_rate]}")
            logger.info(f"  Parent logit advantage: {[f'{x:.3f}' for x in parent_logit_advantage]}")
            
            # Check if discrimination is improving
            if len(avg_logit_spread) > 1:
                logit_trend = np.polyfit(range(len(avg_logit_spread)), avg_logit_spread, 1)[0]
                logger.info(f"  Logit spread slope: {logit_trend:.4f} {'(improving)' if logit_trend > 0 else '(worsening)'}")
                
                parent_trend = np.polyfit(range(len(parent_selection_rate)), parent_selection_rate, 1)[0]
                logger.info(f"  Parent selection trend: {parent_trend:.4f} {'(improving)' if parent_trend > 0 else '(worsening)'}")
            
            # Analyze by SCM type
            scm_parent_rates = defaultdict(list)
            for metric in DISCRIMINATION_METRICS:
                scm_parent_rates[metric['scm']].append(metric['is_direct_parent'])
            
            logger.info(f"\n  Parent selection rates by SCM:")
            for scm, selections in scm_parent_rates.items():
                rate = np.mean(selections) if selections else 0
                logger.info(f"    {scm}: {rate:.3f} ({sum(selections)}/{len(selections)} interventions on parents)")
        
        # Clear for next config
        DISCRIMINATION_METRICS.clear()
    
    # Plot results - create enough subplots for all configurations
    n_configs = len(all_results)
    n_cols = min(3, n_configs)  # Max 3 columns
    n_rows = (n_configs + n_cols - 1) // n_cols  # Calculate needed rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), squeeze=False)
    axes = axes.flatten()
    
    colors = {'fork': 'blue', 'chain': 'green', 'collider': 'red'}
    
    for idx, (config_name, results) in enumerate(all_results.items()):
        ax = axes[idx]
        
        for scm_name, values in results['scm_values'].items():
            if not values:
                continue
                
            # Plot values over time
            ax.plot(range(len(values)), values, 'o-', alpha=0.6, 
                   label=f'{scm_name} (n={len(values)})', 
                   color=colors[scm_name], markersize=3)
            
            # Add trend line if enough data
            if len(values) >= 10:
                z = np.polyfit(range(len(values)), values, 1)
                p = np.poly1d(z)
                ax.plot(range(len(values)), p(range(len(values))), '--', 
                       color=colors[scm_name], alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Target Value')
        ax.set_title(f'{config_name}\nAvg improvement: {results["avg_improvement"]:+.3f}')
        if ax.get_legend_handles_labels()[0]:  # Only add legend if there are labeled items
            ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    
    # Hide any unused subplots
    for idx in range(len(all_results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Target Value Trajectories (MINIMIZE: lower is better)', fontsize=14)
    plt.tight_layout()
    plt.savefig('target_values_logged.png', dpi=150, bbox_inches='tight')
    logger.info(f"\nPlot saved to target_values_logged.png")
    
    # Summary bar plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    config_names = list(all_results.keys())
    avg_improvements = [r['avg_improvement'] for r in all_results.values()]
    
    colors_bar = plt.cm.viridis(np.linspace(0, 1, len(config_names)))
    bars = ax2.bar(range(len(config_names)), avg_improvements, color=colors_bar)
    
    # Add value labels
    for bar, val in zip(bars, avg_improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{val:+.3f}', ha='center', va='bottom')
    
    ax2.set_xticks(range(len(config_names)))
    ax2.set_xticklabels(config_names, rotation=45, ha='right')
    ax2.set_ylabel('Average Improvement (positive = better)')
    ax2.set_title('Target Value Improvement by Configuration')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('target_improvements_logged.png', dpi=150, bbox_inches='tight')
    logger.info(f"Summary saved to target_improvements_logged.png")
    
    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
    
    for config_name, results in all_results.items():
        logger.info(f"\n{config_name}:")
        logger.info(f"  Average improvement: {results['avg_improvement']:+.3f}")
        for scm, imp in results['improvements'].items():
            logger.info(f"  {scm}: {imp:+.3f}")
    
    # Plot discrimination metrics over time
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    axes3 = axes3.flatten()
    
    for config_name, results in all_results.items():
        if 'discrimination_metrics' in results and results['discrimination_metrics']:
            # Group by episode
            episode_groups = defaultdict(list)
            for metric in results['discrimination_metrics']:
                episode_groups[metric['episode']].append(metric)
            
            episodes = sorted(episode_groups.keys())
            avg_logit_spread = []
            avg_embedding_var = []
            parent_selection_rate = []
            parent_logit_advantage = []
            
            for ep in episodes:
                metrics = episode_groups[ep]
                avg_logit_spread.append(np.mean([m['logit_spread'] for m in metrics]))
                avg_embedding_var.append(np.mean([m['embedding_variance'] for m in metrics]))
                
                # Calculate parent selection rate
                parent_selections = [m['is_direct_parent'] for m in metrics]
                parent_selection_rate.append(np.mean(parent_selections))
                
                # Average parent logit advantage
                parent_advantages = [m['parent_logit_advantage'] for m in metrics]
                parent_logit_advantage.append(np.mean(parent_advantages))
            
            # Plot logit spread
            axes3[0].plot(episodes, avg_logit_spread, 'o-', label=config_name, markersize=6)
            
            # Plot embedding variance
            axes3[1].plot(episodes, avg_embedding_var, 'o-', label=config_name, markersize=6)
            
            # Plot parent selection rate
            axes3[2].plot(episodes, parent_selection_rate, 'o-', label=config_name, markersize=6)
            axes3[2].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect parent selection')
            axes3[2].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random selection')
            
            # Plot parent logit advantage
            axes3[3].plot(episodes, parent_logit_advantage, 'o-', label=config_name, markersize=6)
            axes3[3].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    axes3[0].set_xlabel('Episode')
    axes3[0].set_ylabel('Average Logit Spread (std)')
    axes3[0].set_title('Variable Discrimination in Policy Network')
    axes3[0].legend()
    axes3[0].grid(True, alpha=0.3)
    
    axes3[1].set_xlabel('Episode')
    axes3[1].set_ylabel('Average Embedding Variance')
    axes3[1].set_title('Input Embedding Diversity')
    axes3[1].legend()
    axes3[1].grid(True, alpha=0.3)
    
    axes3[2].set_xlabel('Episode')
    axes3[2].set_ylabel('Parent Selection Rate')
    axes3[2].set_title('Rate of Intervening on Direct Parents')
    axes3[2].legend()
    axes3[2].grid(True, alpha=0.3)
    axes3[2].set_ylim(-0.1, 1.1)
    
    axes3[3].set_xlabel('Episode')
    axes3[3].set_ylabel('Parent Logit Advantage')
    axes3[3].set_title('Average Logit Advantage for Parent Variables')
    axes3[3].legend()
    axes3[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('discrimination_metrics.png', dpi=150, bbox_inches='tight')
    logger.info(f"\nDiscrimination metrics saved to discrimination_metrics.png")
    
    # Plot reward component analysis if available
    configs_with_rewards = {k: v for k, v in all_results.items() if v.get('reward_analysis')}
    enhanced_configs = {k: v for k, v in all_results.items() if 'Enhanced' in k}
    
    if configs_with_rewards:
        fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
        axes4 = axes4.flatten()
        
        # Parent selection rates
        ax = axes4[0]
        config_names = list(configs_with_rewards.keys())
        parent_rates = [v['reward_analysis']['binary_signals']['parent_selection_rate'] 
                       for v in configs_with_rewards.values()]
        improvement_rates = [v['reward_analysis']['binary_signals']['improvement_rate'] 
                            for v in configs_with_rewards.values()]
        
        x = np.arange(len(config_names))
        width = 0.35
        bars1 = ax.bar(x - width/2, parent_rates, width, label='Parent Selection Rate')
        bars2 = ax.bar(x + width/2, improvement_rates, width, label='Improvement Rate')
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Rate')
        ax.set_title('Binary Signal Rates (Enhanced GRPO)')
        ax.set_xticks(x)
        ax.set_xticklabels(config_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Reward component means
        ax = axes4[1]
        components = ['target_improvement', 'parent_intervention', 'value_optimization', 'structure_discovery']
        
        for i, (config_name, data) in enumerate(configs_with_rewards.items()):
            analysis = data['reward_analysis']
            values = [analysis.get(comp, {}).get('mean', 0) for comp in components]
            
            x_pos = np.arange(len(components)) + i * 0.2
            ax.bar(x_pos, values, width=0.15, label=config_name)
        
        ax.set_xlabel('Reward Component')
        ax.set_ylabel('Mean Value')
        ax.set_title('Reward Component Breakdown')
        ax.set_xticks(np.arange(len(components)) + 0.3)
        ax.set_xticklabels([c.replace('_', ' ').title() for c in components], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Performance vs parent selection rate
        ax = axes4[2]
        improvements = [v['avg_improvement'] for v in configs_with_rewards.values()]
        ax.scatter(parent_rates, improvements, s=100)
        for i, name in enumerate(config_names):
            ax.annotate(name, (parent_rates[i], improvements[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Parent Selection Rate')
        ax.set_ylabel('Average Improvement')
        ax.set_title('Performance vs Parent Selection')
        ax.grid(True, alpha=0.3)
        
        # Summary comparison
        ax = axes4[3]
        ax.axis('off')
        summary_text = "Enhanced GRPO Summary:\n\n"
        for config_name, data in configs_with_rewards.items():
            analysis = data['reward_analysis']
            summary_text += f"{config_name}:\n"
            summary_text += f"  Parent rate: {analysis['binary_signals']['parent_selection_rate']:.3f}\n"
            summary_text += f"  Improvement: {data['avg_improvement']:+.3f}\n\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Enhanced GRPO Reward Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig('enhanced_grpo_analysis.png', dpi=150, bbox_inches='tight')
        logger.info(f"\nEnhanced GRPO analysis saved to enhanced_grpo_analysis.png")
    
    # Plot component-wise reward evolution and gradient analysis for enhanced configs
    if enhanced_configs:
        # Get the trainer instance from results (if available)
        for config_name, data in enhanced_configs.items():
            trainer = data.get('trainer')
            if trainer and hasattr(trainer, 'component_reward_history'):
                fig5, axes5 = plt.subplots(2, 3, figsize=(18, 10))
                axes5 = axes5.flatten()
                
                # Plot 1-4: Component reward evolution
                components = ['target_improvement', 'parent_intervention', 
                             'value_optimization', 'structure_discovery']
                
                for i, comp in enumerate(components):
                    ax = axes5[i]
                    values = trainer.component_reward_history.get(comp, [])
                    if values:
                        # Moving average
                        window = min(50, len(values) // 10)
                        if window > 1:
                            ma = np.convolve(values, np.ones(window)/window, mode='valid')
                            ax.plot(ma, label=f'MA({window})', linewidth=2)
                        ax.plot(values, alpha=0.3, label='Raw')
                        
                        ax.set_title(f'{comp.replace("_", " ").title()}')
                        ax.set_xlabel('Sample')
                        ax.set_ylabel('Reward')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        # Add trend line
                        if len(values) > 10:
                            z = np.polyfit(range(len(values)), values, 1)
                            trend = z[0]
                            ax.text(0.02, 0.98, f'Trend: {trend:.4f}', 
                                   transform=ax.transAxes, verticalalignment='top')
                
                # Plot 5: Gradient norms
                ax = axes5[4]
                if hasattr(trainer, 'gradient_history') and trainer.gradient_history:
                    episodes = [g['episode'] for g in trainer.gradient_history]
                    grad_norms = [g['grad_norm'] for g in trainer.gradient_history]
                    effective_updates = [g['effective_update'] for g in trainer.gradient_history]
                    
                    ax.plot(episodes, grad_norms, 'o-', label='Gradient Norm')
                    ax.plot(episodes, effective_updates, 's-', label='Effective Update (Grad * LR)')
                    ax.set_xlabel('Episode')
                    ax.set_ylabel('Magnitude')
                    ax.set_title('Gradient Analysis')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_yscale('log')
                
                # Plot 6: Parameter changes
                ax = axes5[5]
                if hasattr(trainer, 'param_change_history') and trainer.param_change_history:
                    episodes = [p['episode'] for p in trainer.param_change_history]
                    relative_changes = [p['relative_change'] * 100 for p in trainer.param_change_history]
                    
                    ax.plot(episodes, relative_changes, 'o-')
                    ax.set_xlabel('Episode')
                    ax.set_ylabel('Relative Change (%)')
                    ax.set_title('Parameter Update Magnitude')
                    ax.grid(True, alpha=0.3)
                    
                    # Add warning if updates are too small
                    avg_change = np.mean(relative_changes) if relative_changes else 0
                    if avg_change < 0.01:  # Less than 0.01%
                        ax.text(0.5, 0.5, 'WARNING: Updates may be too small!', 
                               transform=ax.transAxes, ha='center', va='center',
                               color='red', fontsize=12, weight='bold')
                
                plt.suptitle(f'Component Rewards & Gradient Analysis: {config_name}', fontsize=14)
                plt.tight_layout()
                plt.savefig(f'component_analysis_{config_name.replace(" ", "_").lower()}.png', 
                           dpi=150, bbox_inches='tight')
                logger.info(f"\nComponent analysis saved for {config_name}")


if __name__ == "__main__":
    test_and_plot()