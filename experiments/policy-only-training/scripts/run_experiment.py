#!/usr/bin/env python3
"""Run single GRPO experiment with specified configuration."""

import sys
import os
import argparse
import json
import importlib
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.training.joint_acbo_trainer import JointACBOTrainer
from src.causal_bayes_opt.experiments.benchmark_scms import create_chain_scm, create_fork_scm, create_collider_scm


def load_config(config_name):
    """Load configuration by name."""
    try:
        config_module = importlib.import_module(f'experiments.joint-grpo-target-training.configs.{config_name}')
        return config_module.get_config()
    except ImportError as e:
        print(f"Error loading config '{config_name}': {e}")
        sys.exit(1)


def create_scm(scm_type, num_nodes=3):
    """Create SCM of specified type."""
    if scm_type == 'chain':
        return create_chain_scm(chain_length=num_nodes)
    elif scm_type == 'fork':
        return create_fork_scm()
    elif scm_type == 'collider':
        return create_collider_scm()
    else:
        raise ValueError(f"Unknown SCM type: {scm_type}")


def analyze_within_episode_improvements(episode_metrics):
    """Analyze within-episode learning patterns."""
    improvements = []
    for metrics in episode_metrics:
        if 'within_episode' in metrics:
            improvements.append(metrics['within_episode']['improvement'])
    
    if not improvements:
        return None
    
    return {
        'mean_improvement': float(np.mean(improvements)),
        'std_improvement': float(np.std(improvements)),
        'positive_episodes': sum(1 for i in improvements if i > 0),
        'total_episodes': len(improvements),
        'success_rate': sum(1 for i in improvements if i > 0) / len(improvements),
        'all_improvements': improvements
    }


def main():
    parser = argparse.ArgumentParser(description='Run GRPO experiment')
    parser.add_argument('--config', type=str, default='target_only',
                      choices=['target_only', 'target_heavy', 'balanced', 'exploration_enhanced'],
                      help='Configuration to use')
    parser.add_argument('--episodes', type=int, default=None,
                      help='Override number of episodes')
    parser.add_argument('--interventions', type=int, default=None,
                      help='Override interventions per episode')
    parser.add_argument('--scm', type=str, default='chain',
                      choices=['chain', 'fork', 'collider'],
                      help='SCM type to use')
    parser.add_argument('--num_scms', type=int, default=1,
                      help='Number of different SCMs to train on')
    parser.add_argument('--output', type=str, default=None,
                      help='Output file for results')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"\n{'='*70}")
    print(f"GRPO EXPERIMENT: {args.config}")
    print(f"{'='*70}\n")
    
    config = load_config(args.config)
    
    # Override settings if specified
    if args.episodes:
        config['max_episodes'] = args.episodes
    if args.interventions:
        config['max_interventions'] = args.interventions
    
    print("Configuration:")
    print(f"  Config name: {args.config}")
    print(f"  Episodes: {config['max_episodes']}")
    print(f"  Interventions per episode: {config['max_interventions']}")
    print(f"  SCM type: {args.scm}")
    print(f"  GRPO group size: {config['grpo_config']['group_size']}")
    
    if 'grpo_reward_weights' in config:
        weights = config['grpo_reward_weights']
        print(f"\nReward weights:")
        for key, value in weights.items():
            print(f"  {key}: {value}")
    
    print(f"\n{'-'*70}\n")
    
    # Create SCMs
    scms = [create_scm(args.scm) for _ in range(args.num_scms)]
    
    # Initialize trainer
    trainer = JointACBOTrainer(config=config)
    
    # Run training
    print("Starting training...")
    results = trainer.train(scms)
    
    print(f"\n{'-'*70}")
    print("TRAINING COMPLETE")
    print(f"{'-'*70}\n")
    
    # Analyze results
    if 'episode_metrics' in results:
        metrics = results['episode_metrics']
        
        # Within-episode analysis
        within_analysis = analyze_within_episode_improvements(metrics)
        
        if within_analysis:
            print("\n📈 Within-Episode Learning Analysis:")
            print(f"  Mean improvement: {within_analysis['mean_improvement']:+.4f} ± {within_analysis['std_improvement']:.4f}")
            print(f"  Success rate: {within_analysis['success_rate']:.1%} ({within_analysis['positive_episodes']}/{within_analysis['total_episodes']})")
            
            if within_analysis['mean_improvement'] > 0:
                print("  ✅ Policy shows learning within episodes!")
            else:
                print("  ⚠️ No consistent within-episode improvement")
        
        # Overall reward progression
        all_rewards = [m.get('mean_reward', 0) for m in metrics]
        if len(all_rewards) >= 2:
            print(f"\n📊 Overall Reward Progression:")
            print(f"  First episode: {all_rewards[0]:.4f}")
            print(f"  Last episode: {all_rewards[-1]:.4f}")
            print(f"  Change: {all_rewards[-1] - all_rewards[0]:+.4f}")
        
        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            save_data = {
                'config_name': args.config,
                'config': config,
                'within_episode_analysis': within_analysis,
                'episode_metrics': metrics,
                'reward_progression': all_rewards
            }
            
            with open(output_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"\n💾 Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()