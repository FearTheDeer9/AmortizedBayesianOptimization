#!/usr/bin/env python3
"""
Test GRPO fixes by tracking actual target values instead of rewards.
This addresses the issue that adaptive rewards make "improvement" hard to measure.
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


class TargetValueTrainer(UnifiedGRPOTrainer):
    """Modified trainer that tracks target values instead of just rewards."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_value_history = []
        
    def _collect_grpo_data(self, episode: int, scm, scm_name: str, rng_key) -> dict:
        """Override to track target values during data collection."""
        # Call parent method
        grpo_data = super()._collect_grpo_data(episode, scm, scm_name, rng_key)
        
        # The parent method collects transitions with outcomes
        # We need to extract target values from those outcomes
        # This happens in the reward calculation section
        
        return grpo_data
    
    def _run_grpo_episode(self, episode: int, scm, scm_name: str, rng_key):
        """Override to capture target values from the episode."""
        # Store current episode and SCM info for logging
        self.current_episode = episode
        self.current_scm_name = scm_name
        
        # Determine target variable based on SCM
        if scm_name == 'fork':
            self.current_target_var = 'Y'
        elif scm_name == 'chain':
            self.current_target_var = 'X2'
        elif scm_name == 'collider':
            self.current_target_var = 'Z'
        else:
            self.current_target_var = getattr(scm, 'target', 'Y')
        
        # Run parent method
        result = super()._run_grpo_episode(episode, scm, scm_name, rng_key)
        
        return result
    
    def _collect_grpo_data(self, episode: int, scm, scm_name: str, rng_key) -> dict:
        """Override parent's data collection to log target values."""
        # Import here to avoid circular dependency
        import jax
        import jax.numpy as jnp
        from ..utils.tensor_utils import buffer_to_tensor_clean
        
        grpo_batch_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'old_log_probs': [],
            'masks': [],
            'advantages': []
        }
        
        # Initialize buffer for this episode
        buffer = self._initialize_buffer(scm)
        
        # Collect transitions with GRPO batch approach
        self.rng_key, collect_key = jax.random.split(self.rng_key)
        
        for step in range(self.episode_length):
            step_key, collect_key = jax.random.split(collect_key)
            
            # Convert buffer to tensor for policy input
            tensor = buffer_to_tensor_clean(buffer, self.num_timesteps, standardize=True)
            
            # Get action from policy with old parameters (for importance sampling)
            policy_key, value_key, step_key = jax.random.split(step_key, 3)
            
            # Find target variable index using mapper
            mapper = self._get_variable_mapper(scm)
            target_idx = mapper.target_idx
            
            # Get policy output
            policy_output = self.policy_fn.apply(
                self.policy_params, policy_key, tensor, mapper.target_idx
            )
            
            # Sample variable to intervene on
            var_logits = policy_output['variable_logits']
            var_probs = jax.nn.softmax(var_logits)
            selected_idx = jax.random.categorical(policy_key, var_logits)
            selected_var = mapper.idx_to_var[int(selected_idx)]
            
            # Sample intervention value
            value_params = policy_output['value_params'][selected_idx]
            value_mean, value_log_std = value_params[0], value_params[1]
            value_std = jnp.exp(value_log_std)
            
            intervention_value = value_mean + value_std * jax.random.normal(value_key)
            
            # Log probability for this action
            var_log_prob = jnp.log(var_probs[selected_idx])
            value_log_prob = -0.5 * ((intervention_value - value_mean) / value_std) ** 2 - value_log_std - 0.5 * jnp.log(2 * jnp.pi)
            total_log_prob = var_log_prob + value_log_prob
            
            # Apply intervention
            intervention_dict = {selected_var: float(intervention_value)}
            buffer = self._apply_intervention(buffer, scm, intervention_dict)
            
            # Compute reward
            if buffer.size > 0:
                # Get current state for reward computation
                outcome_sample = buffer.get_recent_samples(1)[0]
                outcome_intervention = outcome_sample.intervention
                
                # Map clean reward weights
                clean_weights = {
                    'target': self.reward_weights.get('optimization', 0.8),
                    'diversity': self.reward_weights.get('discovery', 0.1),
                    'efficiency': self.reward_weights.get('efficiency', 0.1),
                }
                
                # Get posterior info if using surrogate
                posterior_before = None
                posterior_after = None
                if self.use_surrogate and self.surrogate_predict_fn is not None:
                    pass  # Simplified for now
                
                # Import reward computation
                from ..acquisition.better_rewards import compute_better_clean_reward
                
                reward_info = compute_better_clean_reward(
                    buffer_before=buffer,
                    intervention={
                        'targets': frozenset([selected_var]),
                        'values': {selected_var: float(intervention_value)}
                    },
                    outcome=outcome_sample,
                    target_variable=self.current_target_var,
                    config={
                        'optimization_direction': self.optimization_direction,
                        'reward_type': 'adaptive_sigmoid',
                        'temperature_factor': 2.0,
                        'weights': clean_weights
                    },
                    stats=self.reward_stats,
                    posterior_before=posterior_before,
                    posterior_after=posterior_after
                )
                
                # Extract target value and log it
                target_value = float(get_values(outcome_sample)[self.current_target_var])
                self.target_value_history.append({
                    'episode': episode,
                    'step': step,
                    'scm': scm_name,
                    'target_var': self.current_target_var,
                    'value': target_value,
                    'intervention': intervention_dict,
                    'reward': reward_info['total']
                })
                
                reward = reward_info['total']
            else:
                reward = 0.0
            
            # Store transition
            grpo_batch_data['states'].append(tensor)
            grpo_batch_data['actions'].append({
                'variable_idx': selected_idx,
                'value': intervention_value
            })
            grpo_batch_data['rewards'].append(reward)
            grpo_batch_data['old_log_probs'].append(total_log_prob)
            grpo_batch_data['masks'].append(1.0)  # No early termination
        
        # Convert to arrays
        grpo_batch_data['rewards'] = jnp.array(grpo_batch_data['rewards'])
        grpo_batch_data['old_log_probs'] = jnp.array(grpo_batch_data['old_log_probs'])
        grpo_batch_data['masks'] = jnp.array(grpo_batch_data['masks'])
        
        return grpo_batch_data


def test_config(config_name: str, **kwargs):
    """Test a single configuration and return results."""
    
    # Default configuration
    default_config = {
        'learning_rate': 3e-4,
        'n_episodes': 51,  # 17 per SCM
        'episode_length': 10,
        'batch_size': 16,
        'use_early_stopping': False,
        'reward_weights': {
            'optimization': 0.8,
            'discovery': 0.2,
            'efficiency': 0.0,
            'info_gain': 0.0
        },
        'optimization_direction': "MINIMIZE",
        'use_surrogate': False,
        'seed': 42,
        'convergence_config': {
            'enabled': False,
            'patience': 1000,
            'min_episodes': 50,
            'max_episodes_per_scm': 17  # ~50/3 to ensure rotation through all SCMs
        }
    }
    
    # Apply configuration overrides
    config = {**default_config, **kwargs}
    
    # Log configuration changes
    logger.info(f"Configuration changes:")
    for key, value in kwargs.items():
        if key in default_config and value != default_config[key]:
            logger.info(f"  {key}: {default_config[key]} -> {value}")
    
    # Create trainer
    trainer = TargetValueTrainer(**config)
    
    # Create SCMs
    scms = {
        'fork': create_fork_scm(),
        'chain': create_chain_scm(), 
        'collider': create_collider_scm()
    }
    
    # Train
    result = trainer.train(scms)
    all_metrics = result.get('all_metrics', [])
    
    # Analyze target values instead of rewards
    scm_values = defaultdict(list)
    for entry in trainer.target_value_history:
        scm_values[entry['scm']].append(entry['value'])
    
    improvements = {}
    for scm_name, values in scm_values.items():
        if len(values) >= 2:
            early = np.mean(values[:5]) if len(values) >= 5 else values[0]
            late = np.mean(values[-5:]) if len(values) >= 5 else values[-1]
            # For MINIMIZE: improvement = early - late (positive is good)
            improvements[scm_name] = early - late
            
            # Also calculate trend
            if len(values) >= 5:
                x = np.arange(len(values))
                slope, _ = np.polyfit(x, values, 1)
                logger.info(f"  {scm_name}: early_mean={early:.3f}, late_mean={late:.3f}, "
                           f"improvement={improvements[scm_name]:.3f}, slope={slope:.4f}")
        else:
            improvements[scm_name] = 0.0
    
    avg_improvement = np.mean(list(improvements.values()))
    
    logger.info(f"\nResults for {config_name}:")
    logger.info(f"  Average improvement: {avg_improvement:+.3f} (positive = target decreased)")
    
    return {
        'config_name': config_name,
        'improvements': improvements,
        'avg_improvement': avg_improvement,
        'all_metrics': all_metrics,
        'target_values': trainer.target_value_history
    }


def plot_results(results):
    """Create plots comparing different configurations."""
    
    # Plot 1: Target value trajectories
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = {'fork': 'blue', 'chain': 'green', 'collider': 'red'}
    
    for idx, result in enumerate(results):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        config_name = result['config_name']
        target_values = result['target_values']
        
        # Group by SCM
        scm_data = defaultdict(list)
        for entry in target_values:
            scm_data[entry['scm']].append({
                'episode': entry['episode'],
                'value': entry['value']
            })
        
        # Plot each SCM
        for scm_name, data in scm_data.items():
            if not data:
                continue
                
            episodes = [d['episode'] for d in data]
            values = [d['value'] for d in data]
            
            # Scatter plot
            ax.scatter(episodes, values, alpha=0.3, s=10, 
                      color=colors.get(scm_name, 'gray'), label=scm_name)
            
            # Add rolling mean
            if len(values) >= 10:
                window = min(20, len(values) // 3)
                rolling_mean = np.convolve(values, np.ones(window)/window, mode='valid')
                rolling_episodes = episodes[window//2:len(episodes)-window//2+1]
                ax.plot(rolling_episodes, rolling_mean, 
                       color=colors.get(scm_name, 'gray'), linewidth=2, alpha=0.8)
            
            # Add trend line
            if len(values) >= 5:
                z = np.polyfit(episodes, values, 1)
                p = np.poly1d(z)
                ax.plot(episodes, p(episodes), '--', 
                       color=colors.get(scm_name, 'gray'), alpha=0.7, linewidth=1)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Target Value')
        ax.set_title(f'{config_name}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    
    plt.suptitle('Target Value Trajectories (MINIMIZE: lower is better)', fontsize=14)
    plt.tight_layout()
    plt.savefig('grpo_target_values.png', dpi=150, bbox_inches='tight')
    logger.info("\nPlot saved to grpo_target_values.png")
    
    # Plot 2: Summary comparison
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Average improvements
    config_names = [r['config_name'] for r in results]
    avg_improvements = [r['avg_improvement'] for r in results]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    bars = ax1.bar(range(len(config_names)), avg_improvements, color=colors)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, avg_improvements)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    ax1.set_xticks(range(len(config_names)))
    ax1.set_xticklabels(config_names, rotation=45, ha='right')
    ax1.set_ylabel('Average Target Value Improvement')
    ax1.set_title('Average Improvement by Configuration\n(Positive = Target Decreased)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # SCM-specific improvements
    scm_names = ['fork', 'chain', 'collider']
    x = np.arange(len(scm_names))
    width = 0.8 / len(results)
    
    for i, result in enumerate(results):
        improvements = [result['improvements'].get(scm, 0) for scm in scm_names]
        ax2.bar(x + i * width - 0.4 + width/2, improvements, width, 
                label=result['config_name'], alpha=0.8, color=colors[i])
    
    ax2.set_xlabel('SCM Type')
    ax2.set_ylabel('Target Value Improvement')
    ax2.set_title('Improvements by SCM and Configuration')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scm_names)
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('grpo_improvements_summary.png', dpi=150, bbox_inches='tight')
    logger.info("\nPlot saved to grpo_improvements_summary.png")


def main():
    logger.info("="*80)
    logger.info("TESTING GRPO WITH TARGET VALUE TRACKING")
    logger.info("="*80)
    
    # Test configurations
    configs = [
        ("Baseline (Simple)", {}),
        ("Attention Policy", {'policy_architecture': 'attention'}),
        ("Alternating Attention", {'policy_architecture': 'alternating_attention'}),
        ("Alternating Attention (LR=3e-3)", {
            'policy_architecture': 'alternating_attention',
            'learning_rate': 3e-3
        }),
        ("Alternating Attention + Strong Reward", {
            'policy_architecture': 'alternating_attention',
            'learning_rate': 3e-3,
            'reward_weights': {
                'optimization': 1.0,
                'discovery': 0.0,
                'efficiency': 0.0,
                'info_gain': 0.0
            }
        }),
    ]
    
    results = []
    for config_name, config_overrides in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing configuration: {config_name}")
        logger.info(f"{'='*60}")
        
        try:
            result = test_config(config_name, **config_overrides)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to test {config_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Sort by average improvement
    results.sort(key=lambda x: x['avg_improvement'], reverse=True)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY OF ALL CONFIGURATIONS")
    logger.info("="*80)
    
    for i, result in enumerate(results):
        logger.info(f"\n{i+1}. {result['config_name']}:")
        logger.info(f"   Average improvement: {result['avg_improvement']:+.3f}")
        logger.info(f"   By SCM: " + 
                   ", ".join(f"{k}={v:+.3f}" for k, v in result['improvements'].items()))
    
    if results:
        logger.info("\n" + "="*60)
        logger.info(f"BEST CONFIGURATION: {results[0]['config_name']}")
        logger.info(f"Average improvement: {results[0]['avg_improvement']:+.3f}")
        logger.info("="*60)
        
        plot_results(results)
    
    logger.info("\n" + "="*80)
    logger.info("KEY INSIGHTS:")
    logger.info("1. Positive improvement means target values decreased (good for MINIMIZE)")
    logger.info("2. Negative improvement means target values increased (bad)")
    logger.info("3. Zero improvement means no learning occurred")
    logger.info("4. Slope < 0 indicates continuous improvement over time")
    logger.info("="*80)


if __name__ == "__main__":
    main()