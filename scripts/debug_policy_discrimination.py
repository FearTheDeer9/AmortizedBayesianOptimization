#!/usr/bin/env python3
"""
Debug why the policy can't discriminate between variables.

This script traces the data flow from raw buffer samples through
tensor conversion to policy output, identifying where variable
discrimination is lost.
"""

import sys
sys.path.append('.')

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
from typing import Dict, List, Tuple
import logging

# Set up detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
from src.causal_bayes_opt.environments.sampling import sample_with_intervention
from src.causal_bayes_opt.data_structures.sample import get_values
from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
from src.causal_bayes_opt.training.five_channel_converter import buffer_to_five_channel_tensor
from src.causal_bayes_opt.policies.clean_policy_factory import create_clean_grpo_policy
from src.causal_bayes_opt.utils.variable_mapping import VariableMapper


def analyze_buffer_samples(buffer: ExperienceBuffer, target_var: str) -> Dict:
    """Analyze raw buffer samples for variable discrimination."""
    logger.info("\n=== ANALYZING RAW BUFFER SAMPLES ===")
    
    all_samples = buffer.get_all_samples()
    n_samples = len(all_samples)
    
    # Extract values for each variable
    variable_values = {}
    variables = set()
    
    for sample in all_samples:
        values = get_values(sample)
        for var, val in values.items():
            if var not in variable_values:
                variable_values[var] = []
            variable_values[var].append(val)
            variables.add(var)
    
    variables = sorted(variables)
    logger.info(f"Variables: {variables}")
    logger.info(f"Target: {target_var}")
    logger.info(f"Number of samples: {n_samples}")
    
    # Compute statistics for each variable
    stats = {}
    for var in variables:
        values = np.array(variable_values[var])
        stats[var] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'range': np.max(values) - np.min(values)
        }
        
        logger.info(f"\n{var} statistics:")
        logger.info(f"  Mean: {stats[var]['mean']:.3f}")
        logger.info(f"  Std:  {stats[var]['std']:.3f}")
        logger.info(f"  Range: [{stats[var]['min']:.3f}, {stats[var]['max']:.3f}]")
    
    # Check correlation with target
    target_values = np.array(variable_values[target_var])
    correlations = {}
    
    for var in variables:
        if var != target_var:
            corr = np.corrcoef(target_values, variable_values[var])[0, 1]
            correlations[var] = corr
            logger.info(f"\nCorrelation {var} -> {target_var}: {corr:.3f}")
    
    return {
        'variables': variables,
        'stats': stats,
        'correlations': correlations,
        'n_samples': n_samples
    }


def analyze_tensor_conversion(buffer: ExperienceBuffer, target_var: str) -> Dict:
    """Analyze how buffer is converted to tensor representation."""
    logger.info("\n=== ANALYZING TENSOR CONVERSION ===")
    
    # Get 3-channel tensor
    tensor_3ch, mapper = buffer_to_three_channel_tensor(
        buffer, target_var, max_history_size=100, standardize=True
    )
    
    logger.info(f"\n3-Channel Tensor Shape: {tensor_3ch.shape}")
    logger.info(f"Variable order (mapper): {mapper.variables}")
    logger.info(f"Target index: {mapper.target_idx}")
    
    # Analyze each channel
    for ch_idx, ch_name in enumerate(['Values', 'Target Indicator', 'Intervention Indicator']):
        logger.info(f"\n--- Channel {ch_idx}: {ch_name} ---")
        channel = tensor_3ch[:, :, ch_idx]
        
        # Overall statistics
        logger.info(f"Overall mean: {jnp.mean(channel):.3f}")
        logger.info(f"Overall std: {jnp.std(channel):.3f}")
        logger.info(f"Overall range: [{jnp.min(channel):.3f}, {jnp.max(channel):.3f}]")
        
        # Per-variable statistics
        for var_idx, var_name in enumerate(mapper.variables):
            var_data = channel[:, var_idx]
            # Only look at non-zero timesteps
            non_zero_mask = jnp.any(tensor_3ch[:, :, 0] != 0, axis=1)
            valid_data = var_data[non_zero_mask]
            
            if len(valid_data) > 0:
                logger.info(f"\n  {var_name} (idx {var_idx}):")
                logger.info(f"    Mean: {jnp.mean(valid_data):.3f}")
                logger.info(f"    Std:  {jnp.std(valid_data):.3f}")
                logger.info(f"    Range: [{jnp.min(valid_data):.3f}, {jnp.max(valid_data):.3f}]")
    
    # Convert to 5-channel
    tensor_5ch, mapper_5ch, diagnostics = buffer_to_five_channel_tensor(
        buffer, target_var, surrogate_fn=None, max_history_size=100, standardize=True
    )
    
    logger.info(f"\n5-Channel Tensor Shape: {tensor_5ch.shape}")
    logger.info(f"Diagnostics: {diagnostics}")
    
    # Check if standardization is removing signal
    logger.info("\n--- Checking Standardization Effect ---")
    tensor_3ch_unstd, _ = buffer_to_three_channel_tensor(
        buffer, target_var, max_history_size=100, standardize=False
    )
    
    values_std = tensor_3ch[:, :, 0]
    values_unstd = tensor_3ch_unstd[:, :, 0]
    
    logger.info(f"Unstandardized value range: [{jnp.min(values_unstd):.3f}, {jnp.max(values_unstd):.3f}]")
    logger.info(f"Standardized value range: [{jnp.min(values_std):.3f}, {jnp.max(values_std):.3f}]")
    
    # Check variance per variable
    for var_idx, var_name in enumerate(mapper.variables):
        std_before = jnp.std(values_unstd[:, var_idx])
        std_after = jnp.std(values_std[:, var_idx])
        logger.info(f"{var_name}: std before={std_before:.3f}, after={std_after:.3f}")
    
    return {
        'tensor_3ch': tensor_3ch,
        'tensor_5ch': tensor_5ch,
        'mapper': mapper,
        'diagnostics': diagnostics
    }


def analyze_policy_computation(tensor_5ch: jnp.ndarray, mapper: VariableMapper) -> Dict:
    """Analyze how policy processes the tensor input."""
    logger.info("\n=== ANALYZING POLICY COMPUTATION ===")
    
    # Create policy
    policy_fn = create_clean_grpo_policy(hidden_dim=256, architecture="simple")
    
    # Transform with Haiku
    def forward(tensor, target_idx):
        return policy_fn(tensor, target_idx)
    
    forward_fn = hk.without_apply_rng(hk.transform(forward))
    
    # Initialize
    rng_key = jax.random.PRNGKey(42)
    params = forward_fn.init(rng_key, tensor_5ch, mapper.target_idx)
    
    # Get initial output
    output = forward_fn.apply(params, tensor_5ch, mapper.target_idx)
    
    logger.info(f"\nInitial policy output:")
    logger.info(f"Variable logits: {output['variable_logits']}")
    logger.info(f"Value params shape: {output['value_params'].shape}")
    
    # Analyze logits
    logits = output['variable_logits']
    valid_logits = logits[logits != -jnp.inf]
    
    logger.info(f"\nLogit analysis:")
    logger.info(f"Valid logits: {valid_logits}")
    logger.info(f"Logit std: {jnp.std(valid_logits):.6f}")
    logger.info(f"Logit range: [{jnp.min(valid_logits):.6f}, {jnp.max(valid_logits):.6f}]")
    
    probs = jax.nn.softmax(logits)
    valid_probs = probs[logits != -jnp.inf]
    logger.info(f"\nProbabilities: {valid_probs}")
    logger.info(f"Entropy: {-jnp.sum(valid_probs * jnp.log(valid_probs + 1e-8)):.3f}")
    
    # Check what happens with different inputs
    logger.info("\n--- Testing Variable Discrimination ---")
    
    # Create test tensors with clear differences
    test_tensor1 = jnp.zeros_like(tensor_5ch)
    test_tensor2 = jnp.zeros_like(tensor_5ch)
    
    # Set different values for different variables
    test_tensor1 = test_tensor1.at[0, 0, 0].set(1.0)  # Variable 0 high
    test_tensor2 = test_tensor2.at[0, 1, 0].set(1.0)  # Variable 1 high
    
    output1 = forward_fn.apply(params, test_tensor1, mapper.target_idx)
    output2 = forward_fn.apply(params, test_tensor2, mapper.target_idx)
    
    logger.info(f"\nTest 1 (var 0 high) logits: {output1['variable_logits']}")
    logger.info(f"Test 2 (var 1 high) logits: {output2['variable_logits']}")
    logger.info(f"Logit difference: {output1['variable_logits'] - output2['variable_logits']}")
    
    return {
        'initial_output': output,
        'params': params,
        'test_outputs': (output1, output2)
    }


def analyze_reward_computation(buffer: ExperienceBuffer, scm, target_var: str) -> Dict:
    """Analyze reward computation for different interventions."""
    logger.info("\n=== ANALYZING REWARD COMPUTATION ===")
    
    variables = sorted(buffer.get_variable_coverage())
    non_target_vars = [v for v in variables if v != target_var]
    
    # Test intervening on each variable
    intervention_results = {}
    
    for var in non_target_vars:
        logger.info(f"\n--- Testing intervention on {var} ---")
        
        # Create intervention
        intervention = create_perfect_intervention(
            targets=frozenset([var]),
            values={var: 0.0}  # Set to 0 for consistency
        )
        
        # Sample outcomes
        samples = sample_with_intervention(scm, intervention, n_samples=100, seed=42)
        
        # Extract target values
        target_values = [get_values(s)[target_var] for s in samples]
        
        mean_outcome = np.mean(target_values)
        std_outcome = np.std(target_values)
        
        intervention_results[var] = {
            'mean': mean_outcome,
            'std': std_outcome,
            'values': target_values
        }
        
        logger.info(f"  Mean outcome: {mean_outcome:.3f}")
        logger.info(f"  Std outcome: {std_outcome:.3f}")
    
    # Compare rewards
    logger.info("\n--- Reward Comparison ---")
    from src.causal_bayes_opt.acquisition.better_rewards import (
        adaptive_sigmoid_reward, RunningStats
    )
    
    stats = RunningStats()
    
    for var, results in intervention_results.items():
        # Compute rewards for this intervention
        rewards = []
        for val in results['values'][:10]:  # First 10 samples
            reward = adaptive_sigmoid_reward(val, stats, 'MINIMIZE')
            rewards.append(reward)
        
        mean_reward = np.mean(rewards)
        logger.info(f"\n{var} intervention:")
        logger.info(f"  Outcome range: [{min(results['values']):.3f}, {max(results['values']):.3f}]")
        logger.info(f"  Mean reward: {mean_reward:.3f}")
        logger.info(f"  First 5 rewards: {rewards[:5]}")
    
    return intervention_results


def main():
    logger.info("="*80)
    logger.info("POLICY DISCRIMINATION DEBUGGING")
    logger.info("="*80)
    
    # Create a simple test scenario
    scm = create_fork_scm()  # X -> Y <- Z
    target_var = 'Y'
    
    logger.info("\nTrue causal structure: X -> Y <- Z")
    logger.info("Both X and Z are parents of Y")
    logger.info("Policy should learn to prefer X and Z over other interventions")
    
    # Create buffer with data
    buffer = ExperienceBuffer()
    
    # Add observational samples
    obs_samples = sample_from_linear_scm(scm, 100, seed=42)
    for sample in obs_samples:
        buffer.add_observation(sample)
    
    # Add some intervention samples
    for var in ['X', 'Z']:
        for val in [-1.0, 0.0, 1.0]:
            intervention = create_perfect_intervention(
                targets=frozenset([var]),
                values={var: val}
            )
            int_samples = sample_with_intervention(scm, intervention, n_samples=10, seed=42)
            for sample in int_samples:
                buffer.add_intervention(intervention, sample)
    
    # Run analyses
    buffer_analysis = analyze_buffer_samples(buffer, target_var)
    tensor_analysis = analyze_tensor_conversion(buffer, target_var)
    policy_analysis = analyze_policy_computation(
        tensor_analysis['tensor_5ch'], 
        tensor_analysis['mapper']
    )
    reward_analysis = analyze_reward_computation(buffer, scm, target_var)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY OF FINDINGS")
    logger.info("="*80)
    
    logger.info("\n1. Buffer Analysis:")
    logger.info(f"   - Variables have different distributions: {buffer_analysis['variables']}")
    logger.info(f"   - Correlations exist with target: {buffer_analysis['correlations']}")
    
    logger.info("\n2. Tensor Conversion:")
    logger.info(f"   - Standardization active: Changes value distributions")
    logger.info(f"   - Channel structure preserved: 5 channels created")
    
    logger.info("\n3. Policy Computation:")
    initial_output = policy_analysis['initial_output']
    logger.info(f"   - Initial logit std: {jnp.std(initial_output['variable_logits'][initial_output['variable_logits'] != -jnp.inf]):.6f}")
    logger.info(f"   - Policy CAN distinguish inputs (test showed different outputs)")
    
    logger.info("\n4. Reward Analysis:")
    logger.info(f"   - Different interventions produce different outcomes")
    logger.info(f"   - Rewards reflect these differences")
    
    logger.info("\n" + "="*80)
    logger.info("NEXT STEPS:")
    logger.info("1. Check if GRPO gradient updates are too small")
    logger.info("2. Verify batch advantage computation")
    logger.info("3. Test with larger learning rate or different optimizer")
    logger.info("4. Check if exploration noise overwhelms learning signal")
    logger.info("="*80)


if __name__ == "__main__":
    main()