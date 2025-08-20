#!/usr/bin/env python3
"""
Evaluation Script for Production GRPO Models

Evaluates trained models on curriculum levels and held-out SCMs.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pickle
import json
from typing import Dict, Any, List, Tuple
import jax
import jax.numpy as jnp
import haiku as hk

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.training.curriculum_factory import SCMCurriculumFactory
from src.causal_bayes_opt.experiments.benchmark_scms import (
    create_chain_scm, create_butterfly_scm, create_mixed_coeff_scm
)
from src.causal_bayes_opt.data_structures.scm import (
    sample_observational_data, perform_perfect_intervention,
    get_variables, get_target, get_parents
)
from src.causal_bayes_opt.policies.clean_policy_factory import create_clean_grpo_policy


class ModelEvaluator:
    """Evaluates trained GRPO models."""
    
    def __init__(self, checkpoint_path: str):
        """Load model from checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")
        
        with open(checkpoint_path, 'rb') as f:
            self.checkpoint = pickle.load(f)
        
        self.policy_params = self.checkpoint['policy_params']
        self.config = self.checkpoint.get('config', {})
        
        # Initialize policy
        self._init_policy()
        
        print(f"Loaded model from episode {self.checkpoint.get('episode', 'unknown')}")
        if 'curriculum_levels' in self.checkpoint:
            levels = self.checkpoint['curriculum_levels']
            if levels:
                print(f"Model trained up to curriculum level {max(levels)}")
    
    def _init_policy(self):
        """Initialize policy network."""
        # Use same architecture as training
        architecture = self.config.get('policy_architecture', 'simple_permutation_invariant')
        use_fixed_std = self.config.get('use_fixed_std', True)
        fixed_std = self.config.get('fixed_std', 0.5)
        
        # Create policy using same factory
        self.policy = create_clean_grpo_policy(
            architecture=architecture,
            hidden_dim=self.config.get('hidden_dim', 256),
            use_fixed_std=use_fixed_std,
            fixed_std=fixed_std
        )
        
        # Transform for JAX
        self.policy = hk.without_apply_rng(hk.transform(self.policy))
    
    def evaluate_on_scm(self, scm, num_episodes: int = 10, 
                        interventions_per_episode: int = 20) -> Dict[str, Any]:
        """Evaluate policy on a single SCM."""
        
        variables = list(get_variables(scm))
        target_var = get_target(scm)
        true_parents = set(get_parents(scm, target_var))
        
        episode_results = []
        
        for episode in range(num_episodes):
            # Sample observational data
            obs_data = sample_observational_data(scm, num_samples=100)
            
            intervention_values = []
            target_values = []
            selected_variables = []
            
            for step in range(interventions_per_episode):
                # Create simplified state tensor (would be more complex in production)
                state = jnp.zeros((1, len(variables), 5))  # Batch=1, vars, channels
                
                # Get policy action
                action_params = self.policy.apply(self.policy_params, state)
                
                # Extract variable and value
                var_probs = action_params['variable_probs'][0]
                selected_var_idx = int(jnp.argmax(var_probs))
                selected_var = variables[selected_var_idx]
                
                value_mean = float(action_params['value_params'][0, selected_var_idx, 0])
                
                # Apply intervention
                outcome = perform_perfect_intervention(scm, {selected_var: value_mean})
                target_value = float(outcome[target_var])
                
                # Track results
                selected_variables.append(selected_var)
                intervention_values.append(value_mean)
                target_values.append(target_value)
            
            # Calculate episode metrics
            parent_selection_rate = sum(1 for v in selected_variables if v in true_parents) / len(selected_variables)
            avg_target = np.mean(target_values)
            best_target = min(target_values)
            
            if intervention_values:
                value_range = max(intervention_values) - min(intervention_values)
                range_utilization = value_range / 10.0 * 100  # Assuming [-5, 5] range
            else:
                range_utilization = 0
            
            episode_results.append({
                'avg_target': avg_target,
                'best_target': best_target,
                'parent_selection_rate': parent_selection_rate,
                'range_utilization': range_utilization
            })
        
        # Aggregate results
        return {
            'num_variables': len(variables),
            'true_parents': list(true_parents),
            'avg_target': np.mean([r['avg_target'] for r in episode_results]),
            'best_target': min([r['best_target'] for r in episode_results]),
            'parent_selection_rate': np.mean([r['parent_selection_rate'] for r in episode_results]),
            'range_utilization': np.mean([r['range_utilization'] for r in episode_results]),
            'std_target': np.std([r['avg_target'] for r in episode_results])
        }
    
    def evaluate_curriculum_levels(self, levels: List[int] = None) -> Dict[int, Dict]:
        """Evaluate on specific curriculum levels."""
        
        if levels is None:
            levels = [1, 3, 5, 7, 10]
        
        # Create curriculum factory
        curriculum = SCMCurriculumFactory(
            start_level=1,
            max_level=max(levels),
            mode="fixed",
            seed=123  # Different seed from training
        )
        
        results = {}
        
        for level in levels:
            print(f"\nEvaluating curriculum level {level}...")
            
            # Get SCM from curriculum
            scm = curriculum.get_scm(level=level)
            
            # Evaluate
            level_results = self.evaluate_on_scm(scm)
            level_results['level'] = level
            level_results['stage_name'] = curriculum.stages[level-1].name
            
            results[level] = level_results
            
            print(f"  Level {level} ({level_results['stage_name']}):")
            print(f"    Avg target: {level_results['avg_target']:.3f}")
            print(f"    Parent selection: {level_results['parent_selection_rate']*100:.1f}%")
            print(f"    Range utilization: {level_results['range_utilization']:.1f}%")
        
        return results
    
    def evaluate_held_out_scms(self) -> Dict[str, Dict]:
        """Evaluate on SCMs not in training curriculum."""
        
        held_out_scms = {
            'chain_6': create_chain_scm(6),
            'butterfly': create_butterfly_scm(),
            'mixed_coeff': create_mixed_coeff_scm()
        }
        
        results = {}
        
        print("\nEvaluating held-out SCMs...")
        
        for name, scm in held_out_scms.items():
            print(f"\n  Evaluating {name}...")
            scm_results = self.evaluate_on_scm(scm)
            scm_results['scm_name'] = name
            results[name] = scm_results
            
            print(f"    Avg target: {scm_results['avg_target']:.3f}")
            print(f"    Parent selection: {scm_results['parent_selection_rate']*100:.1f}%")
        
        return results
    
    def compare_to_random(self, scm, num_episodes: int = 10) -> Dict[str, Any]:
        """Compare trained policy to random baseline."""
        
        variables = list(get_variables(scm))
        target_var = get_target(scm)
        
        random_targets = []
        
        for _ in range(num_episodes):
            obs_data = sample_observational_data(scm, num_samples=100)
            episode_targets = []
            
            for _ in range(20):  # 20 interventions
                # Random intervention
                random_var = np.random.choice(variables)
                random_value = np.random.uniform(-5, 5)
                
                outcome = perform_perfect_intervention(scm, {random_var: random_value})
                episode_targets.append(float(outcome[target_var]))
            
            random_targets.append(np.mean(episode_targets))
        
        # Get trained policy results
        policy_results = self.evaluate_on_scm(scm, num_episodes=num_episodes)
        
        return {
            'random_avg': np.mean(random_targets),
            'policy_avg': policy_results['avg_target'],
            'improvement': (np.mean(random_targets) - policy_results['avg_target']) / abs(np.mean(random_targets)) * 100,
            'random_std': np.std(random_targets),
            'policy_std': policy_results['std_target']
        }


def run_comprehensive_evaluation(checkpoint_path: str, output_dir: str = "results"):
    """Run comprehensive evaluation suite."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(checkpoint_path)
    
    # 1. Evaluate on curriculum levels
    print("\n📊 Evaluating Curriculum Levels...")
    curriculum_results = evaluator.evaluate_curriculum_levels([1, 2, 3, 5, 7, 10])
    
    # 2. Evaluate on held-out SCMs
    print("\n🎯 Evaluating Held-Out SCMs...")
    held_out_results = evaluator.evaluate_held_out_scms()
    
    # 3. Compare to random baseline
    print("\n🎲 Comparing to Random Baseline...")
    test_scm = create_chain_scm(4)
    comparison = evaluator.compare_to_random(test_scm)
    
    print(f"\n  Random avg: {comparison['random_avg']:.3f} ± {comparison['random_std']:.3f}")
    print(f"  Policy avg: {comparison['policy_avg']:.3f} ± {comparison['policy_std']:.3f}")
    print(f"  Improvement: {comparison['improvement']:.1f}%")
    
    # Save results
    results = {
        'checkpoint': checkpoint_path,
        'curriculum_levels': curriculum_results,
        'held_out_scms': held_out_results,
        'random_comparison': comparison
    }
    
    results_file = output_path / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {results_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    print("\n📈 Performance by Curriculum Level:")
    for level, res in sorted(curriculum_results.items()):
        print(f"  Level {level:2d}: Target = {res['avg_target']:6.3f}, "
              f"Parent Selection = {res['parent_selection_rate']*100:5.1f}%")
    
    print("\n🎯 Held-Out SCM Performance:")
    for name, res in held_out_results.items():
        print(f"  {name:12s}: Target = {res['avg_target']:6.3f}, "
              f"Parent Selection = {res['parent_selection_rate']*100:5.1f}%")
    
    if comparison['improvement'] > 0:
        print(f"\n✅ Policy outperforms random baseline by {comparison['improvement']:.1f}%")
    else:
        print(f"\n⚠️ Policy underperforms random baseline by {-comparison['improvement']:.1f}%")
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained GRPO models')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/quick_test/latest.pkl',
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("Please train a model first using train_production.py")
        return
    
    # Run evaluation
    results = run_comprehensive_evaluation(args.checkpoint, args.output)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()