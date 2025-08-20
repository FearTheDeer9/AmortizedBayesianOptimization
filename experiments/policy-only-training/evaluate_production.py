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
        
        # Initialize policy function
        self._init_policy()
        
    def _init_policy(self):
        """Initialize policy function from checkpoint."""
        # Get architecture from config
        architecture = self.config.get('policy_architecture', 'simple_permutation_invariant')
        use_fixed_std = self.config.get('use_fixed_std', True)
        fixed_std = self.config.get('fixed_std', 1.5)
        hidden_dim = self.config.get('hidden_dim', 256)
        
        # Create policy function
        policy_fn = create_clean_grpo_policy(
            hidden_dim=hidden_dim,
            architecture=architecture,
            use_fixed_std=use_fixed_std,
            fixed_std=fixed_std
        )
        
        self.policy = hk.without_apply_rng(hk.transform(policy_fn))
        
    def evaluate_on_scm(self, scm, num_interventions: int = 30) -> Dict[str, Any]:
        """Evaluate policy on a single SCM."""
        results = {
            'scm_type': str(type(scm).__name__),
            'num_vars': len(scm.variable_names),
            'interventions': [],
            'target_values': [],
            'range_utilization': {},
            'correct_parent_rate': 0.0
        }
        
        # Track values per variable
        variable_values = {var: [] for var in scm.variable_names}
        correct_parent_selections = 0
        
        # Get target variable (usually last)
        target_var = scm.variable_names[-1]
        target_idx = len(scm.variable_names) - 1
        
        # Initial observations
        observations = []
        for _ in range(10):
            obs = scm.sample()
            observations.append(obs)
        
        # Run interventions
        for i in range(num_interventions):
            # Create tensor input (simplified - would use actual converter in production)
            tensor_input = self._create_tensor_input(observations, scm)
            
            # Get policy action
            output = self.policy.apply(self.policy_params, tensor_input, target_idx)
            
            # Sample intervention
            variable_logits = output['variable_logits']
            value_params = output['value_params']
            
            # Select variable
            var_probs = jax.nn.softmax(variable_logits)
            var_idx = int(jnp.argmax(var_probs))
            selected_var = scm.variable_names[var_idx]
            
            # Select value
            mean = value_params[var_idx, 0]
            log_std = value_params[var_idx, 1]
            value = float(mean)  # Use mean for deterministic evaluation
            
            # Clip to valid range
            var_range = scm.variable_ranges.get(selected_var, (-5, 5))
            value = np.clip(value, var_range[0], var_range[1])
            
            # Record intervention
            variable_values[selected_var].append(value)
            
            # Perform intervention and get outcome
            outcome = scm.intervene({selected_var: value})
            target_value = outcome[target_var]
            
            results['interventions'].append({
                'variable': selected_var,
                'value': value,
                'target': target_value
            })
            results['target_values'].append(target_value)
            
            # Check if correct parent selected
            true_parents = scm.get_parents(target_var)
            if selected_var in true_parents:
                correct_parent_selections += 1
            
            # Add to observations
            observations.append(outcome)
            if len(observations) > 20:
                observations.pop(0)
        
        # Compute metrics
        results['avg_target'] = float(np.mean(results['target_values']))
        results['best_target'] = float(np.min(results['target_values']))
        results['correct_parent_rate'] = correct_parent_selections / num_interventions
        
        # Compute range utilization per variable
        for var, values in variable_values.items():
            if values:
                var_range = scm.variable_ranges.get(var, (-5, 5))
                range_size = var_range[1] - var_range[0]
                actual_range = max(values) - min(values)
                utilization = (actual_range / range_size) * 100
                results['range_utilization'][var] = {
                    'min': float(min(values)),
                    'max': float(max(values)),
                    'utilization_pct': float(utilization),
                    'num_interventions': len(values)
                }
        
        return results
    
    def _create_tensor_input(self, observations, scm) -> jnp.ndarray:
        """Create tensor input from observations (simplified)."""
        # This is a simplified version - real implementation would use proper converter
        T = len(observations)
        n_vars = len(scm.variable_names)
        
        # Create basic 5-channel tensor
        tensor = jnp.zeros((T, n_vars, 5))
        
        # Fill with observation values (channel 0)
        for t, obs in enumerate(observations):
            for v, var in enumerate(scm.variable_names):
                tensor = tensor.at[t, v, 0].set(obs.get(var, 0))
        
        # Add some basic statistics (channels 1-4)
        # This is simplified - real implementation would compute proper posteriors
        tensor = tensor.at[:, :, 1].set(0.5)  # Placeholder
        
        return tensor
    
    def evaluate_suite(self) -> Dict[str, Any]:
        """Evaluate on comprehensive test suite."""
        logger.info("Running comprehensive evaluation suite...")
        
        test_scms = {
            'simple_chain_3': create_chain_scm(3),
            'simple_chain_4': create_chain_scm(4),
            'medium_chain_5': create_chain_scm(5),
            'fork': create_fork_scm(),  # Fixed X->Y<-Z structure
            'diamond': create_diamond_scm(),  # X->Y->W<-Z<-X structure
            'collider': create_collider_scm(),  # Fixed X->Z<-Y structure
            'sparse_5': create_sparse_scm(num_vars=5, edge_prob=0.3),
            'sparse_7': create_sparse_scm(num_vars=7, edge_prob=0.3),
            'dense_6': create_dense_scm(num_vars=6, edge_prob=0.5),
            'dense_8': create_dense_scm(num_vars=8, edge_prob=0.4),
        }
        
        all_results = {}
        
        for name, scm in test_scms.items():
            logger.info(f"Evaluating on {name}...")
            result = self.evaluate_on_scm(scm)
            all_results[name] = result
            
            # Log summary
            logger.info(f"  Avg target: {result['avg_target']:.3f}")
            logger.info(f"  Best target: {result['best_target']:.3f}")
            logger.info(f"  Parent accuracy: {result['correct_parent_rate']:.1%}")
        
        # Compute aggregate metrics
        aggregate = self._compute_aggregate_metrics(all_results)
        
        return {
            'checkpoint': str(self.checkpoint.get('episode', 'unknown')),
            'individual_results': all_results,
            'aggregate': aggregate
        }
    
    def _compute_aggregate_metrics(self, results: Dict) -> Dict[str, float]:
        """Compute aggregate metrics across all SCMs."""
        all_targets = []
        all_parent_rates = []
        all_utilizations = []
        
        for scm_results in results.values():
            all_targets.extend(scm_results['target_values'])
            all_parent_rates.append(scm_results['correct_parent_rate'])
            
            for var_util in scm_results['range_utilization'].values():
                all_utilizations.append(var_util['utilization_pct'])
        
        return {
            'mean_target': float(np.mean(all_targets)),
            'std_target': float(np.std(all_targets)),
            'best_target': float(np.min(all_targets)),
            'mean_parent_accuracy': float(np.mean(all_parent_rates)),
            'mean_range_utilization': float(np.mean(all_utilizations)) if all_utilizations else 0.0,
            'num_scms_evaluated': len(results)
        }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate GRPO Models')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='results/evaluation.json',
                       help='Output path for results')
    parser.add_argument('--num_interventions', type=int, default=30,
                       help='Number of interventions per SCM')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick evaluation (fewer SCMs)')
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return
    
    # Create evaluator
    evaluator = ModelEvaluator(args.checkpoint)
    
    # Run evaluation
    if args.quick:
        # Quick test on single SCM
        scm = create_chain_scm(3)
        results = evaluator.evaluate_on_scm(scm, num_interventions=10)
        logger.info(f"Quick eval - Target: {results['avg_target']:.3f}")
    else:
        # Full evaluation suite
        results = evaluator.evaluate_suite()
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        
        # Print summary
        agg = results['aggregate']
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Mean target value: {agg['mean_target']:.3f} ± {agg['std_target']:.3f}")
        print(f"Best target value: {agg['best_target']:.3f}")
        print(f"Parent accuracy: {agg['mean_parent_accuracy']:.1%}")
        print(f"Range utilization: {agg['mean_range_utilization']:.1f}%")
        print(f"SCMs evaluated: {agg['num_scms_evaluated']}")


if __name__ == "__main__":
    main()