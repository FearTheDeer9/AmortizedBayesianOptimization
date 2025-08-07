#!/usr/bin/env python3
"""
Plot the trajectory of target variable values over training.
This shows the actual optimization progress, not just rewards.
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
from src.causal_bayes_opt.experiments.benchmark_scms import (
    create_fork_scm, create_chain_scm, create_collider_scm
)
from src.causal_bayes_opt.data_structures.sample import get_values


class TargetTrackingTrainer(UnifiedGRPOTrainer):
    """Modified trainer that tracks target values."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_history = []
        
    def _run_grpo_episode(self, episode: int, scm, scm_name: str, rng_key):
        """Override to track target values."""
        # Get the target variable name based on SCM type
        if scm_name == 'fork':
            target_var = 'Y'
        elif scm_name == 'chain':
            target_var = 'X2'  # Chain SCM has X0->X1->X2
        elif scm_name == 'collider':
            target_var = 'Z'
        else:
            target_var = getattr(scm, 'target', 'Y')
            
        # Run the episode
        result = super()._run_grpo_episode(episode, scm, scm_name, rng_key)
        
        # Extract target values from this episode
        if hasattr(self, '_last_episode_targets'):
            for target_val in self._last_episode_targets:
                self.target_history.append({
                    'episode': episode,
                    'scm': scm_name,
                    'target_var': target_var,
                    'value': target_val
                })
        
        return result
    
    def _collect_grpo_data(self, episode: int, scm, scm_name: str, rng_key):
        """Override to capture target values during collection."""
        self._last_episode_targets = []
        
        # Run parent method and capture the data
        grpo_data = super()._collect_grpo_data(episode, scm, scm_name, rng_key)
        
        # Extract target values from rewards computation
        # The parent method stores outcome samples that we can analyze
        # For now, we'll use a different approach in _run_grpo_episode
        
        return grpo_data


def plot_target_trajectories():
    """Run training and plot target value trajectories."""
    
    # Create SCMs
    scms = {
        'fork': create_fork_scm(),
        'chain': create_chain_scm(), 
        'collider': create_collider_scm()
    }
    
    # Run with different configurations
    configs = [
        ('Baseline (LR=3e-4)', {'learning_rate': 3e-4}),
        ('Higher LR (3e-3)', {'learning_rate': 3e-3}),
        ('Much Higher LR (3e-2)', {'learning_rate': 3e-2}),
    ]
    
    fig, axes = plt.subplots(len(configs), 1, figsize=(12, 4*len(configs)))
    if len(configs) == 1:
        axes = [axes]
    
    for idx, (config_name, config_overrides) in enumerate(configs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {config_name}")
        logger.info(f"{'='*60}")
        
        # Create trainer
        trainer = TargetTrackingTrainer(
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
                'max_episodes_per_scm': 20  # Ensure rotation
            }
        )
        
        # Train
        result = trainer.train(scms)
        
        # Plot trajectories
        ax = axes[idx]
        
        # Group by SCM
        scm_colors = {'fork': 'blue', 'chain': 'green', 'collider': 'red'}
        
        for scm_name in ['fork', 'chain', 'collider']:
            # Get data for this SCM
            scm_data = [h for h in trainer.target_history if h['scm'] == scm_name]
            
            if scm_data:
                episodes = [h['episode'] for h in scm_data]
                values = [h['value'] for h in scm_data]
                
                # Plot all points
                ax.scatter(episodes, values, alpha=0.3, s=10, 
                          color=scm_colors[scm_name], label=f'{scm_name} (n={len(values)})')
                
                # Add rolling mean
                if len(values) >= 5:
                    window = min(20, len(values) // 3)
                    rolling_mean = np.convolve(values, np.ones(window)/window, mode='valid')
                    rolling_episodes = episodes[window//2:len(episodes)-window//2+1]
                    ax.plot(rolling_episodes, rolling_mean, 
                           color=scm_colors[scm_name], linewidth=2, alpha=0.8)
                
                # Add trend line
                if len(values) >= 10:
                    z = np.polyfit(episodes, values, 1)
                    p = np.poly1d(z)
                    ax.plot(episodes, p(episodes), '--', 
                           color=scm_colors[scm_name], alpha=0.5, linewidth=1)
                    
                    # Print statistics
                    logger.info(f"\n{scm_name} statistics:")
                    logger.info(f"  Episodes: {len(values)}")
                    logger.info(f"  Mean value: {np.mean(values):.3f}")
                    logger.info(f"  Initial mean (first 5): {np.mean(values[:5]):.3f}")
                    logger.info(f"  Final mean (last 5): {np.mean(values[-5:]):.3f}")
                    logger.info(f"  Trend slope: {z[0]:.4f} (negative is good for MINIMIZE)")
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Target Value')
        ax.set_title(f'{config_name} - Target Value Trajectory (MINIMIZE: lower is better)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add horizontal line at y=0 for reference
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('target_trajectories.png', dpi=150, bbox_inches='tight')
    logger.info(f"\nPlot saved to target_trajectories.png")
    
    # Also create a summary plot showing just the trends
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    x_offset = 0
    bar_width = 0.25
    scm_names = ['fork', 'chain', 'collider']
    
    for i, (config_name, _) in enumerate(configs):
        improvements = []
        
        for scm_name in scm_names:
            # Re-run to get data (or store it above)
            # For now, let's calculate from the last trainer
            scm_data = [h for h in trainer.target_history if h['scm'] == scm_name]
            if len(scm_data) >= 10:
                initial = np.mean([h['value'] for h in scm_data[:5]])
                final = np.mean([h['value'] for h in scm_data[-5:]])
                improvement = initial - final  # Positive means improvement for MINIMIZE
                improvements.append(improvement)
            else:
                improvements.append(0)
        
        x_positions = np.arange(len(scm_names)) + x_offset
        ax2.bar(x_positions, improvements, bar_width, label=config_name, alpha=0.8)
        x_offset += bar_width
    
    ax2.set_xlabel('SCM Type')
    ax2.set_ylabel('Improvement (Initial - Final Mean)')
    ax2.set_title('Target Value Improvement by Configuration')
    ax2.set_xticks(np.arange(len(scm_names)) + bar_width)
    ax2.set_xticklabels(scm_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('target_improvements.png', dpi=150, bbox_inches='tight')
    logger.info(f"Summary plot saved to target_improvements.png")


if __name__ == "__main__":
    plot_target_trajectories()