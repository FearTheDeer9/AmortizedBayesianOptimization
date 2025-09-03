#!/usr/bin/env python3
"""
Simple debug evaluation script with clean output.
Focuses on showing input/output channels clearly without JAX compilation noise.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Suppress JAX compilation messages
os.environ['JAX_LOG_LEVEL'] = 'WARNING'

import jax
import jax.numpy as jnp

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
from src.causal_bayes_opt.data_structures.scm import get_variables, get_target, get_parents
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
from experiments.evaluation.core.model_loader import ModelLoader


def print_separator(title=""):
    """Print a visual separator."""
    if title:
        print(f"\n{'='*20} {title} {'='*20}")
    else:
        print(f"{'='*60}")


def print_tensor_info(tensor, name="Tensor"):
    """Print tensor information in a clean, readable format."""
    print(f"\n{name}:")
    print(f"  Shape: {tensor.shape}")
    
    if len(tensor.shape) >= 3:
        n_channels = tensor.shape[-1]
        print(f"  Channels: {n_channels}")
        
        # Channel 0: Values
        if n_channels >= 1:
            vals = tensor[..., 0]
            print(f"  Ch0 (values):     min={float(vals.min()):.3f}, max={float(vals.max()):.3f}, mean={float(vals.mean()):.3f}, std={float(vals.std()):.3f}")
        
        # Channel 1: Target indicators
        if n_channels >= 2:
            targets = tensor[..., 1]
            print(f"  Ch1 (targets):    sum={int(targets.sum())}, mean={float(targets.mean()):.3f}")
        
        # Channel 2: Intervention indicators
        if n_channels >= 3:
            intervs = tensor[..., 2]
            print(f"  Ch2 (intervened): sum={int(intervs.sum())}, mean={float(intervs.mean()):.3f}")
        
        # Channel 3: Parent probabilities
        if n_channels >= 4:
            probs = tensor[..., 3]
            print(f"  Ch3 (parent probs): min={float(probs.min()):.3f}, max={float(probs.max()):.3f}, mean={float(probs.mean()):.3f}")
            print(f"                      unique values: {len(np.unique(probs))}")


def test_single_step(policy_path, surrogate_path=None):
    """Test a single evaluation step with clean debug output."""
    
    print_separator("SETUP")
    
    # Create a simple SCM
    scm_factory = VariableSCMFactory(seed=42, noise_scale=0.1)
    scm = scm_factory.create_variable_scm(
        num_variables=5,
        structure_type='fork',
        target_variable=None
    )
    
    target = get_target(scm)
    true_parents = set(get_parents(scm, target))
    variables = list(get_variables(scm))
    
    print(f"SCM Configuration:")
    print(f"  Variables: {variables}")
    print(f"  Target: {target}")
    print(f"  True parents: {true_parents}")
    
    # Create buffer and add observational data
    buffer = ExperienceBuffer()
    obs_samples = sample_from_linear_scm(scm, 20, seed=42)
    for sample in obs_samples:
        buffer.add_observation(sample)
    
    print(f"\nBuffer state:")
    print(f"  Observations: {buffer.num_observations()}")
    print(f"  Interventions: {buffer.num_interventions()}")
    
    print_separator("DATA PREPARATION")
    
    # Get 3-channel tensor
    tensor_3ch, _ = buffer_to_three_channel_tensor(buffer, target)
    print_tensor_info(tensor_3ch, "Raw 3-channel tensor")
    
    # Standardize values channel
    mean = float(tensor_3ch[:, :, 0].mean())
    std = float(tensor_3ch[:, :, 0].std()) + 1e-8
    tensor_3ch = tensor_3ch.at[:, :, 0].set((tensor_3ch[:, :, 0] - mean) / std)
    
    print(f"\nStandardization applied:")
    print(f"  Mean: {mean:.3f}")
    print(f"  Std: {std:.3f}")
    
    print_tensor_info(tensor_3ch, "Standardized 3-channel tensor")
    
    # Load and test surrogate if provided
    if surrogate_path and Path(surrogate_path).exists():
        print_separator("SURROGATE MODEL")
        
        try:
            params, architecture, surrogate_fn = ModelLoader.load_surrogate(Path(surrogate_path))
            print(f"Loaded surrogate with architecture: {architecture.get('encoder_type', 'unknown')}")
            
            # Get surrogate predictions
            print("\nCalling surrogate with:")
            print(f"  Input shape: {tensor_3ch.shape}")
            print(f"  Target: {target}")
            
            output = surrogate_fn(tensor_3ch, target, variables)
            
            if 'parent_probabilities' in output:
                probs = output['parent_probabilities']
                print(f"\nSurrogate output:")
                print(f"  Parent probabilities: {[f'{v}:{float(p):.3f}' for v, p in zip(variables, probs)]}")
                
                # Compute F1
                predicted_parents = {variables[i] for i in range(len(variables)) 
                                   if variables[i] != target and float(probs[i]) > 0.5}
                tp = len(true_parents & predicted_parents)
                fp = len(predicted_parents - true_parents)
                fn = len(true_parents - predicted_parents)
                f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
                
                print(f"\nStructure learning metrics:")
                print(f"  Predicted parents: {predicted_parents}")
                print(f"  True parents: {true_parents}")
                print(f"  F1 score: {f1:.3f}")
        except Exception as e:
            print(f"Surrogate error: {e}")
    
    # Load and test policy
    print_separator("POLICY MODEL")
    
    try:
        # Load policy with SCM metadata for quantile ranges
        scm_metadata = scm.get('metadata', {}) if hasattr(scm, 'get') else {}
        policy_fn = ModelLoader.load_policy(Path(policy_path), seed=42, scm_metadata=scm_metadata)
        print("Loaded quantile policy")
        
        # Convert to 4-channel for policy (add parent probabilities)
        tensor_4ch = jnp.zeros((tensor_3ch.shape[0], tensor_3ch.shape[1], 4))
        tensor_4ch = tensor_4ch.at[:, :, :3].set(tensor_3ch)
        
        # Add parent probabilities from surrogate if available
        if surrogate_path and 'parent_probabilities' in locals():
            # Broadcast parent probs to all timesteps
            for t in range(tensor_4ch.shape[0]):
                tensor_4ch = tensor_4ch.at[t, :, 3].set(probs)
            print("\nAdded surrogate parent probabilities to channel 3")
        else:
            tensor_4ch = tensor_4ch.at[:, :, 3].set(0.5)  # Uniform if no surrogate
            print("\nAdded uniform parent probabilities (0.5) to channel 3")
        
        print_tensor_info(tensor_4ch, "4-channel tensor for policy")
        
        # Get policy decision
        print("\nCalling policy...")
        posterior = {'parent_probs': probs} if surrogate_path and 'probs' in locals() else None
        intervention = policy_fn(tensor_4ch, posterior, target, variables)
        
        print(f"\nPolicy decision:")
        print(f"  Intervene on: {intervention['targets']}")
        print(f"  Values: {intervention['values']}")
        
    except Exception as e:
        print(f"Policy error: {e}")
        import traceback
        traceback.print_exc()
    
    print_separator()
    print("Debug evaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple debug evaluation with clean output")
    parser.add_argument('--policy', type=Path, required=True, help='Path to policy checkpoint')
    parser.add_argument('--surrogate', type=Path, help='Path to surrogate checkpoint')
    
    args = parser.parse_args()
    
    # Ensure clean output
    os.environ['JAX_LOG_LEVEL'] = 'WARNING'
    
    test_single_step(args.policy, args.surrogate)