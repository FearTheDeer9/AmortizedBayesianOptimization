#!/usr/bin/env python3
"""
Simple script to plot target value trajectories by extracting from logs.
"""

import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import logging
import re
from collections import defaultdict

# Set up logging to capture trainer output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a custom handler to capture specific log messages
class TargetValueCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.target_values = []
        
    def emit(self, record):
        # Look for outcome value logs
        if "Outcome:" in record.getMessage():
            match = re.search(r'Outcome:\s*([-\d.]+)', record.getMessage())
            if match:
                value = float(match.group(1))
                # Extract episode number if available
                episode_match = re.search(r'Episode\s*(\d+)', record.getMessage())
                episode = int(episode_match.group(1)) if episode_match else len(self.target_values)
                
                self.target_values.append({
                    'episode': episode,
                    'value': value,
                    'timestamp': record.created
                })

from src.causal_bayes_opt.training.unified_grpo_trainer import UnifiedGRPOTrainer
from src.causal_bayes_opt.experiments.benchmark_scms import (
    create_fork_scm, create_chain_scm, create_collider_scm
)


def run_and_plot():
    """Run training with different configs and plot results."""
    
    configs = [
        ('Baseline (LR=3e-4)', {'learning_rate': 3e-4}),
        ('Higher LR (3e-3)', {'learning_rate': 3e-3}),
        ('Much Higher LR (3e-2)', {'learning_rate': 3e-2}),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    all_results = {}
    
    for config_idx, (config_name, config_overrides) in enumerate(configs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {config_name}")
        logger.info(f"{'='*60}")
        
        # Add custom handler
        capture_handler = TargetValueCapture()
        logging.getLogger('src.causal_bayes_opt').addHandler(capture_handler)
        
        # Create trainer with verbose logging
        trainer = UnifiedGRPOTrainer(
            learning_rate=config_overrides.get('learning_rate', 3e-4),
            n_episodes=60,  # 20 per SCM
            episode_length=10,
            batch_size=16,
            use_early_stopping=False,
            reward_weights={
                'optimization': 1.0,
                'discovery': 0.0,
                'efficiency': 0.0,
                'info_gain': 0.0
            },
            optimization_direction="MINIMIZE",
            use_surrogate=False,
            seed=42,
            convergence_config={
                'patience': 1000,
                'min_episodes': 100,
                'max_episodes_per_scm': 20
            }
        )
        
        # Create SCMs
        scms = {
            'fork': create_fork_scm(),
            'chain': create_chain_scm(), 
            'collider': create_collider_scm()
        }
        
        # Train
        result = trainer.train(scms)
        
        # Extract metrics from training
        all_metrics = result.get('all_metrics', [])
        
        # Group by SCM
        scm_metrics = defaultdict(list)
        for m in all_metrics:
            scm_name = m.get('scm_type', 'unknown')
            scm_metrics[scm_name].append(m)
        
        # Store results
        all_results[config_name] = {
            'metrics': scm_metrics,
            'captured_values': capture_handler.target_values
        }
        
        # Remove handler
        logging.getLogger('src.causal_bayes_opt').removeHandler(capture_handler)
        
        # Plot for each SCM
        ax = axes[config_idx]
        colors = {'fork': 'blue', 'chain': 'green', 'collider': 'red'}
        
        for scm_name, metrics in scm_metrics.items():
            if not metrics:
                continue
                
            episodes = [m['episode'] for m in metrics]
            
            # Try to extract target values from metrics
            # If not available, use mean rewards as proxy
            values = []
            for m in metrics:
                # Look for actual target value in metrics
                if 'target_value' in m:
                    values.append(m['target_value'])
                elif 'mean_outcome' in m:
                    values.append(m['mean_outcome'])
                else:
                    # Use negative mean reward as proxy (since high reward = low target for MINIMIZE)
                    values.append(-m.get('mean_reward', 0))
            
            if values:
                # Plot scatter
                ax.scatter(episodes, values, alpha=0.4, s=20, 
                          color=colors.get(scm_name, 'gray'), label=scm_name)
                
                # Add trend line
                if len(values) >= 5:
                    z = np.polyfit(episodes, values, 1)
                    p = np.poly1d(z)
                    ax.plot(episodes, p(episodes), '--', 
                           color=colors.get(scm_name, 'gray'), linewidth=2)
                    
                    # Calculate improvement
                    initial_mean = np.mean(values[:min(5, len(values))])
                    final_mean = np.mean(values[-min(5, len(values)):])
                    improvement = initial_mean - final_mean  # Positive = improvement for MINIMIZE
                    
                    logger.info(f"{scm_name}: initial={initial_mean:.3f}, final={final_mean:.3f}, "
                               f"improvement={improvement:.3f}, slope={z[0]:.4f}")
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Target Value (or -Reward)')
        ax.set_title(config_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Target Value Trajectories (MINIMIZE: lower is better)', fontsize=14)
    plt.tight_layout()
    plt.savefig('target_trajectories_simple.png', dpi=150, bbox_inches='tight')
    logger.info(f"\nPlot saved to target_trajectories_simple.png")
    
    # Create summary comparison
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    scm_names = ['fork', 'chain', 'collider']
    x = np.arange(len(scm_names))
    width = 0.25
    
    for i, config_name in enumerate(all_results.keys()):
        improvements = []
        
        for scm_name in scm_names:
            metrics = all_results[config_name]['metrics'].get(scm_name, [])
            if len(metrics) >= 5:
                # Use mean rewards as proxy
                initial_rewards = [m['mean_reward'] for m in metrics[:5]]
                final_rewards = [m['mean_reward'] for m in metrics[-5:]]
                
                # For adaptive rewards, stable or slightly decreasing rewards 
                # actually indicate improvement in target values
                reward_change = np.mean(final_rewards) - np.mean(initial_rewards)
                improvements.append(-reward_change)  # Negative because lower target = higher difficulty
            else:
                improvements.append(0)
        
        ax2.bar(x + i*width, improvements, width, label=config_name, alpha=0.8)
    
    ax2.set_xlabel('SCM Type')
    ax2.set_ylabel('Inferred Improvement (from reward trends)')
    ax2.set_title('Performance Comparison Across Configurations')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(scm_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('target_improvements_summary.png', dpi=150, bbox_inches='tight')
    logger.info(f"Summary saved to target_improvements_summary.png")
    
    return all_results


if __name__ == "__main__":
    results = run_and_plot()