#!/usr/bin/env python3
"""Debug whether policies are actually using the 5-channel tensor input."""

import jax.numpy as jnp
from pathlib import Path
import logging

from src.causal_bayes_opt.data_structures.scm import get_target
from src.causal_bayes_opt.environments.registry import SCMRegistry
from src.causal_bayes_opt.evaluation.model_interfaces import (
    create_random_baseline,
    create_bc_acquisition, 
    create_optimal_oracle_acquisition
)
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.training.five_channel_converter import buffer_to_five_channel_tensor

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_5channel_integration():
    """Test if policies actually use the 5-channel input differently."""
    
    # Create a simple SCM
    registry = SCMRegistry()
    scm = registry.create('fork')
    
    target = get_target(scm)
    variables = ['X', 'Y', 'Z']
    
    # Create buffer with some data
    buffer = ExperienceBuffer()
    samples = sample_from_linear_scm(scm, 100, seed=42)
    for sample in samples:
        buffer.add_observation(sample)
    
    # Create a dummy surrogate that returns specific predictions
    def confident_surrogate(tensor, target_var, variables):
        """Surrogate with confident predictions."""
        return {
            'marginal_parent_probs': {
                'X': 0.9,  # High confidence X is parent
                'Y': 0.0,
                'Z': 0.8   # High confidence Z is parent
            },
            'entropy': 0.2
        }
    
    def uncertain_surrogate(tensor, target_var, variables):
        """Surrogate with uncertain predictions."""
        return {
            'marginal_parent_probs': {
                'X': 0.6,  # Low confidence
                'Y': 0.0,
                'Z': 0.4   # Low confidence
            },
            'entropy': 0.9
        }
    
    # Test different policies with different surrogates
    policies = {
        'Random': create_random_baseline(seed=42),
        'Oracle': create_optimal_oracle_acquisition(scm, optimization_direction='MINIMIZE', seed=42),
        'BC': create_bc_acquisition(Path('checkpoints/acbo/bc_acquisition/1733589468'))
    }
    
    surrogates = {
        'None': None,
        'Confident': confident_surrogate,
        'Uncertain': uncertain_surrogate
    }
    
    print("\n" + "="*80)
    print("5-CHANNEL INTEGRATION TEST")
    print("="*80)
    
    for policy_name, policy_fn in policies.items():
        print(f"\n{policy_name} Policy:")
        print("-" * 60)
        
        for surrogate_name, surrogate_fn in surrogates.items():
            # Convert to tensor
            if surrogate_fn is None:
                # 3-channel tensor
                from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
                tensor, var_order = buffer_to_three_channel_tensor(buffer, target, standardize=True)
                posterior = None
            else:
                # 5-channel tensor with surrogate predictions
                tensor, var_order, metadata = buffer_to_five_channel_tensor(
                    buffer, target, surrogate_fn=surrogate_fn, standardize=True
                )
                posterior = surrogate_fn(tensor[:, :, :3], target, var_order)  # Get posterior for policy
            
            # Get intervention from policy
            intervention = policy_fn(tensor, posterior, target, var_order)
            
            # Log results
            targets = list(intervention.get('targets', []))
            values = [intervention['values'][t] for t in targets] if targets else []
            
            print(f"\n  With {surrogate_name} surrogate:")
            print(f"    Tensor shape: {tensor.shape}")
            print(f"    Intervention: {targets} = {values}")
            
            if surrogate_fn is not None and tensor.shape[2] == 5:
                # Check if surrogate predictions are in the tensor
                last_timestep = tensor[-1]  # [n_vars, 5]
                print(f"    Surrogate signals in tensor:")
                for i, var in enumerate(var_order):
                    parent_prob = last_timestep[i, 3]
                    recency = last_timestep[i, 4]
                    print(f"      {var}: parent_prob={parent_prob:.3f}, recency={recency:.3f}")
    
    # Test if policies behave differently with different surrogate predictions
    print("\n\n" + "="*60)
    print("POLICY BEHAVIOR ANALYSIS")
    print("="*60)
    
    # Count unique interventions per policy
    for policy_name, policy_fn in policies.items():
        interventions = []
        
        for surrogate_fn in surrogates.values():
            if surrogate_fn is None:
                from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
                tensor, var_order = buffer_to_three_channel_tensor(buffer, target, standardize=True)
                posterior = None
            else:
                tensor, var_order, metadata = buffer_to_five_channel_tensor(
                    buffer, target, surrogate_fn=surrogate_fn, standardize=True
                )
                posterior = surrogate_fn(tensor[:, :, :3], target, var_order)
            
            intervention = policy_fn(tensor, posterior, target, var_order)
            intervention_key = (
                tuple(sorted(intervention.get('targets', []))),
                tuple(intervention['values'].get(t, 0) for t in sorted(intervention.get('targets', [])))
            )
            interventions.append(intervention_key)
        
        unique_interventions = len(set(interventions))
        print(f"\n{policy_name}: {unique_interventions} unique interventions across {len(surrogates)} surrogate types")
        
        if unique_interventions == 1:
            print(f"  ⚠️  Policy ignores surrogate predictions!")
        else:
            print(f"  ✓ Policy adapts based on surrogate predictions")

if __name__ == "__main__":
    test_5channel_integration()