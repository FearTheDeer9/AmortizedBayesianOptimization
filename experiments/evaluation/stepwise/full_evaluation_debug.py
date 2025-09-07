#!/usr/bin/env python3
"""
Debug version of full_evaluation.py with extensive printing of tensor values.
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.utils.checkpoint_utils import load_checkpoint
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.training.four_channel_converter import buffer_to_four_channel_tensor
from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
from src.causal_bayes_opt.data_structures.scm import get_parents, get_variables, get_target
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.avici_integration.continuous.model import ContinuousParentSetPredictionModel
from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
from src.causal_bayes_opt.environments.sampling import sample_with_intervention
from src.causal_bayes_opt.data_structures.sample import get_values
from src.causal_bayes_opt.policies.clean_policy_factory import create_quantile_policy

def debug_evaluate_single_episode():
    """Run one episode with detailed debugging of all 4 channels."""
    
    # Use the exact paths from the user's request
    policy_path = Path("imperial-vm-checkpoints/grpo_enhanced_20250907_034435/final_policy.pkl")
    surrogate_path = Path("checkpoints/avici_runs/avici_style_20250903_154909/checkpoint_step_1000.pkl")
    
    print("="*80)
    print("DEBUG EVALUATION - DETAILED CHANNEL ANALYSIS")
    print("="*80)
    
    # Load surrogate
    surrogate_checkpoint = load_checkpoint(surrogate_path)
    surrogate_params = surrogate_checkpoint['params']
    surrogate_architecture = surrogate_checkpoint.get('architecture', {})
    
    def surrogate_fn(x, target_idx, is_training):
        model = ContinuousParentSetPredictionModel(
            hidden_dim=surrogate_architecture.get('hidden_dim', 128),
            num_heads=surrogate_architecture.get('num_heads', 8),
            num_layers=surrogate_architecture.get('num_layers', 8),
            key_size=surrogate_architecture.get('key_size', 32),
            dropout=0.0,
            use_temperature_scaling=True,
            temperature_init=0.0
        )
        return model(x, target_idx, is_training)
    
    surrogate_net = hk.transform(surrogate_fn)
    
    # Load policy
    policy_checkpoint = load_checkpoint(policy_path)
    policy_params = policy_checkpoint['params']
    policy_architecture = policy_checkpoint.get('architecture', {})
    
    policy_fn = create_quantile_policy(
        hidden_dim=policy_architecture.get('hidden_dim', 256)
    )
    policy_net = hk.without_apply_rng(hk.transform(policy_fn))
    
    # Create SCM - chain structure as requested
    factory = VariableSCMFactory(seed=42, noise_scale=0.5)
    scm = factory.create_variable_scm(num_variables=5, structure_type='chain')
    
    variables = list(get_variables(scm))
    target_var = get_target(scm)
    true_parents = set(get_parents(scm, target_var))
    variable_ranges = scm.get('metadata', {}).get('variable_ranges', {})
    
    print(f"\nSCM Configuration:")
    print(f"  Variables: {variables}")
    print(f"  Target: {target_var}")
    print(f"  True parents: {true_parents}")
    
    # Initialize buffer
    buffer = ExperienceBuffer()
    
    # Add initial data (matching evaluation settings)
    # 20 observations
    samples = sample_from_linear_scm(scm, n_samples=20, seed=42)
    for sample in samples:
        buffer.add_observation(sample)
    
    # 10 initial random interventions
    rng_key = random.PRNGKey(42)
    non_target_vars = [v for v in variables if v != target_var]
    
    for i in range(10):
        rng_key, var_key, value_key = random.split(rng_key, 3)
        selected_var_idx = random.choice(var_key, len(non_target_vars))
        selected_var = non_target_vars[int(selected_var_idx)]
        
        var_range = variable_ranges.get(selected_var, (-2.0, 2.0))
        intervened_value = float(random.uniform(value_key, 
                                               minval=var_range[0], 
                                               maxval=var_range[1]))
        
        intervention = create_perfect_intervention(
            targets=frozenset([selected_var]),
            values={selected_var: intervened_value}
        )
        
        samples = sample_with_intervention(
            scm, intervention, n_samples=1,
            seed=42 + i + 5000
        )
        
        for sample in samples:
            buffer.add_intervention({selected_var: intervened_value}, sample)
    
    print(f"\nBuffer Status:")
    print(f"  Observations: {len(buffer._observations)}")
    print(f"  Interventions: {len(buffer._interventions)}")
    print(f"  Total samples: {len(buffer._observations) + len(buffer._interventions)}")
    
    # Create surrogate wrapper
    def surrogate_wrapper(tensor_3ch, target_var_name, variable_list):
        """Wrapper for surrogate predictions."""
        target_idx = list(variable_list).index(target_var_name)
        rng_key_surrogate = random.PRNGKey(42)
        predictions = surrogate_net.apply(surrogate_params, rng_key_surrogate, tensor_3ch, target_idx, False)
        parent_probs = predictions.get('parent_probabilities', jnp.full(len(variable_list), 0.5))
        return {'parent_probs': parent_probs}
    
    # Convert buffer to tensor
    print(f"\n" + "="*60)
    print("CONVERTING BUFFER TO 4-CHANNEL TENSOR")
    print("="*60)
    
    tensor_4ch, mapper, diagnostics = buffer_to_four_channel_tensor(
        buffer, target_var, 
        surrogate_fn=surrogate_wrapper,
        max_history_size=None,
        standardize=True
    )
    
    print(f"\nTensor Shape: {tensor_4ch.shape}")
    print(f"Mapper Variables (in order): {mapper.variables}")
    print(f"Target Index: {mapper.get_index(target_var)}")
    
    # Print detailed channel information for the LAST timestep (most recent)
    print(f"\n" + "="*60)
    print("DETAILED 4-CHANNEL VALUES (LAST TIMESTEP)")
    print("="*60)
    
    last_timestep = tensor_4ch[-1]  # Shape: [n_vars, 4]
    
    print("\nPER-VARIABLE CHANNEL VALUES:")
    print("-" * 60)
    print(f"{'Variable':<10} {'Ch0:Value':<12} {'Ch1:Target':<12} {'Ch2:Interv':<12} {'Ch3:ParProb':<12} {'IsParent?':<10}")
    print("-" * 60)
    
    for i, var in enumerate(mapper.variables):
        ch0 = float(last_timestep[i, 0])  # Value
        ch1 = float(last_timestep[i, 1])  # Target indicator
        ch2 = float(last_timestep[i, 2])  # Intervention indicator
        ch3 = float(last_timestep[i, 3])  # Parent probability
        is_parent = "YES" if var in true_parents else "NO"
        is_target = "(TARGET)" if var == target_var else ""
        
        print(f"{var:<10} {ch0:>11.4f} {ch1:>11.1f} {ch2:>11.1f} {ch3:>11.4f} {is_parent:<10} {is_target}")
    
    # Print statistics across timesteps
    print(f"\n" + "="*60)
    print("CHANNEL STATISTICS ACROSS ALL TIMESTEPS")
    print("="*60)
    
    for ch_idx, ch_name in enumerate(["Values", "Target", "Intervention", "Parent Probs"]):
        print(f"\nChannel {ch_idx} ({ch_name}):")
        channel_data = tensor_4ch[:, :, ch_idx]
        
        # Per-variable statistics
        for i, var in enumerate(mapper.variables):
            var_data = channel_data[:, i]
            is_parent = "(PARENT)" if var in true_parents else ""
            is_target = "(TARGET)" if var == target_var else ""
            
            if ch_idx == 0:  # Values - show mean and std
                print(f"  {var}: mean={float(jnp.mean(var_data)):>7.3f}, std={float(jnp.std(var_data)):>7.3f} {is_parent} {is_target}")
            elif ch_idx == 1:  # Target - should be constant
                unique_vals = jnp.unique(var_data)
                print(f"  {var}: {float(unique_vals[0]):.1f} (constant) {is_parent} {is_target}")
            elif ch_idx == 2:  # Intervention - show frequency
                intervention_rate = float(jnp.mean(var_data))
                print(f"  {var}: {intervention_rate:.2%} of timesteps {is_parent} {is_target}")
            elif ch_idx == 3:  # Parent probs - should be constant from surrogate
                unique_vals = jnp.unique(var_data)
                if len(unique_vals) == 1:
                    print(f"  {var}: {float(unique_vals[0]):.4f} (constant) {is_parent} {is_target}")
                else:
                    print(f"  {var}: varies {float(jnp.min(var_data)):.4f} to {float(jnp.max(var_data)):.4f} {is_parent} {is_target}")
    
    # Now run the policy
    print(f"\n" + "="*60)
    print("POLICY INFERENCE")
    print("="*60)
    
    target_idx_mapper = mapper.get_index(target_var)
    policy_output = policy_net.apply(policy_params, tensor_4ch, target_idx_mapper)
    
    print(f"\nPolicy Output Keys: {policy_output.keys()}")
    print(f"\nQuantile Scores (shape {policy_output['quantile_scores'].shape}):")
    print("-" * 60)
    print(f"{'Variable':<10} {'Q1 (25%)':<12} {'Q2 (50%)':<12} {'Q3 (75%)':<12} {'IsParent?':<10}")
    print("-" * 60)
    
    quantile_scores = policy_output['quantile_scores']
    for i, var in enumerate(mapper.variables):
        scores = quantile_scores[i]
        is_parent = "YES" if var in true_parents else "NO"
        is_target = "(TARGET)" if var == target_var else ""
        
        if jnp.any(jnp.isinf(scores)):
            print(f"{var:<10} {'[MASKED]':<12} {'[MASKED]':<12} {'[MASKED]':<12} {is_parent:<10} {is_target}")
        else:
            print(f"{var:<10} {float(scores[0]):>11.4f} {float(scores[1]):>11.4f} {float(scores[2]):>11.4f} {is_parent:<10} {is_target}")
    
    # Compute selection probabilities
    print(f"\n" + "="*60)
    print("SELECTION PROBABILITIES")
    print("="*60)
    
    flat_scores = quantile_scores.flatten()
    probs = jax.nn.softmax(flat_scores)
    
    # Group by variable
    print("\nProbability by Variable (sum across quantiles):")
    for i, var in enumerate(mapper.variables):
        var_prob_sum = float(jnp.sum(probs[i*3:(i+1)*3]))
        is_parent = "(PARENT)" if var in true_parents else ""
        is_target = "(TARGET)" if var == target_var else ""
        
        if var != target_var:
            print(f"  {var}: {var_prob_sum:.3f} {is_parent}")
            # Show breakdown by quantile
            for q in range(3):
                q_prob = float(probs[i*3 + q])
                if q_prob > 0.001:
                    print(f"    Q{q+1} ({25*(q+1)}%): {q_prob:.3f}")
    
    # Sample to see what would be selected
    print(f"\n" + "="*60)
    print("SIMULATED SELECTIONS (10 samples)")
    print("="*60)
    
    rng_key = random.PRNGKey(42)
    parent_count = 0
    for sample_idx in range(10):
        rng_key, sample_key = random.split(rng_key)
        selected_flat_idx = random.choice(sample_key, len(flat_scores), p=probs)
        
        selected_var_idx = selected_flat_idx // 3
        selected_quantile_idx = selected_flat_idx % 3
        selected_var = mapper.variables[int(selected_var_idx)]
        
        is_parent = selected_var in true_parents
        if is_parent:
            parent_count += 1
        
        quantile_name = ["25%", "50%", "75%"][int(selected_quantile_idx)]
        print(f"  Sample {sample_idx+1}: {selected_var} @ {quantile_name} {'(PARENT)' if is_parent else ''}")
    
    print(f"\nParent selection rate: {parent_count/10:.0%}")
    print(f"Expected random rate: {len(true_parents)/len(non_target_vars):.0%}")

if __name__ == "__main__":
    debug_evaluate_single_episode()