#!/usr/bin/env python3
"""
Evaluate policy trained with ground truth coefficients.
Modified from full_evaluation.py to use coefficient information in 4th channel.
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
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.utils.checkpoint_utils import load_checkpoint
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
from src.causal_bayes_opt.data_structures.scm import get_parents, get_variables, get_target
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
from src.causal_bayes_opt.environments.sampling import sample_with_intervention
from src.causal_bayes_opt.data_structures.sample import get_values

# Import from full_evaluation
from full_evaluation import (
    MetricsTracker,
    create_quantile_policy,
    extract_parent_probabilities
)


def buffer_to_coefficient_tensor(
    buffer: ExperienceBuffer,
    variables: List[str],
    target_var: str,
    scm: Dict,
    mode: str = 'last'
) -> jnp.ndarray:
    """
    Convert buffer to 4-channel tensor with ground truth coefficients in 4th channel.
    
    Args:
        buffer: Experience buffer
        variables: List of variable names
        target_var: Target variable name
        scm: SCM dictionary with coefficient information
        mode: 'all' for all samples, 'last' for last sample only
    
    Returns:
        4-channel tensor: [values, target_indicator, intervention_indicator, coefficients]
    """
    
    if mode == 'last':
        # Only use the last sample
        if len(buffer.observational_samples) == 0:
            raise ValueError("Buffer has no samples")
        
        last_sample = buffer.observational_samples[-1]
        values = get_values(last_sample)
        
        # Create value vector
        value_vector = jnp.array([values[var] for var in variables])
        
        # Create target indicator
        target_idx = variables.index(target_var)
        target_indicator = jnp.zeros(len(variables))
        target_indicator = target_indicator.at[target_idx].set(1.0)
        
        # Create intervention indicator (last intervention if any)
        intervention_indicator = jnp.zeros(len(variables))
        if buffer.interventional_samples:
            last_intervention = buffer.interventional_samples[-1]
            intervention_var = last_intervention['intervention']['variable']
            if intervention_var in variables:
                int_idx = variables.index(intervention_var)
                intervention_indicator = intervention_indicator.at[int_idx].set(1.0)
        
        # Create coefficient channel with ground truth values
        coefficient_channel = jnp.zeros(len(variables))
        true_parents = set(get_parents(scm, target_var))
        
        # Extract coefficients from SCM metadata
        all_coefficients = scm.get('metadata', {}).get('coefficients', {})
        for edge_str, coeff in all_coefficients.items():
            # Parse edge string
            if isinstance(edge_str, str) and ',' in edge_str:
                edge_str = edge_str.strip('()')
                parts = [p.strip().strip("'").strip('"') for p in edge_str.split(',')]
                if len(parts) == 2:
                    from_var, to_var = parts
                    if to_var == target_var and from_var in true_parents:
                        parent_idx = variables.index(from_var)
                        coefficient_channel = coefficient_channel.at[parent_idx].set(coeff)
            elif isinstance(edge_str, tuple) and len(edge_str) == 2:
                from_var, to_var = edge_str
                if to_var == target_var and from_var in true_parents:
                    parent_idx = variables.index(from_var)
                    coefficient_channel = coefficient_channel.at[parent_idx].set(coeff)
        
        # Stack into 4-channel tensor
        tensor = jnp.stack([
            value_vector,
            target_indicator,
            intervention_indicator,
            coefficient_channel
        ], axis=-1)
        
        # Add batch dimension
        return jnp.expand_dims(tensor, 0)
    
    else:
        raise NotImplementedError(f"Mode '{mode}' not implemented for coefficient tensor")


def evaluate_episode_with_coefficients(
    policy_net, policy_params,
    scm, 
    num_interventions: int = 40,
    initial_observations: int = 20,
    initial_interventions: int = 10,
    seed: int = 42,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate one episode using ground truth coefficients in 4th channel.
    """
    
    # Set up logging
    logger = logging.getLogger(__name__)
    
    # Get SCM information
    variables = list(get_variables(scm))
    target_var = get_target(scm)
    true_parents = set(get_parents(scm, target_var))
    
    # Initialize RNG
    rng_key = random.PRNGKey(seed)
    
    # Initialize buffer for data collection
    buffer = ExperienceBuffer(variables=variables)
    
    # Collect initial observational samples
    for _ in range(initial_observations):
        rng_key, sample_key = random.split(rng_key)
        sample = sample_from_linear_scm(scm, sample_key)
        buffer.add_observational_sample(sample)
    
    # Collect initial random interventional samples
    for _ in range(initial_interventions):
        rng_key, int_key, sample_key = random.split(rng_key, 3)
        
        # Random intervention
        int_var = random.choice(int_key, jnp.array(variables))
        int_value = random.normal(sample_key) * 2.0
        
        intervention = create_perfect_intervention(
            variable=int_var,
            value=float(int_value)
        )
        
        rng_key, sample_key = random.split(rng_key)
        sample = sample_with_intervention(scm, intervention, sample_key)
        buffer.add_interventional_sample(sample, intervention)
    
    # Tracking metrics
    episode_data = {
        'target_values': [],
        'selected_variables': [],
        'parent_selections': [],
        'scm_info': {
            'num_variables': len(variables),
            'target': target_var,
            'true_parents': list(true_parents),
            'structure_type': scm.get('metadata', {}).get('structure_type', 'unknown')
        }
    }
    
    # Policy-driven interventions
    for step in range(num_interventions):
        # Create input tensor with coefficients
        policy_input = buffer_to_coefficient_tensor(
            buffer, variables, target_var, scm, mode='last'
        )
        
        # Get policy action
        rng_key, policy_key = random.split(rng_key)
        action_logits = policy_net.apply(policy_params, policy_input)
        
        # Sample action
        action_probs = jax.nn.softmax(action_logits[0])
        rng_key, action_key = random.split(rng_key)
        selected_idx = int(random.choice(action_key, len(variables), p=action_probs))
        selected_var = variables[selected_idx]
        
        # Perform intervention
        rng_key, int_key = random.split(rng_key)
        int_value = random.normal(int_key) * 2.0
        
        intervention = create_perfect_intervention(
            variable=selected_var,
            value=float(int_value)
        )
        
        rng_key, sample_key = random.split(rng_key)
        sample = sample_with_intervention(scm, intervention, sample_key)
        buffer.add_interventional_sample(sample, intervention)
        
        # Record metrics
        values = get_values(sample)
        episode_data['target_values'].append(float(values[target_var]))
        episode_data['selected_variables'].append(selected_var)
        episode_data['parent_selections'].append(selected_var in true_parents)
        
        if verbose and step % 10 == 0:
            logger.info(f"Step {step}: Selected {selected_var}, "
                       f"Target value: {values[target_var]:.3f}, "
                       f"Is parent: {selected_var in true_parents}")
    
    return episode_data


def main():
    """Main evaluation function."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Paths
    policy_path = Path("/Users/harellidar/Documents/Imperial/Individual_Project/causal_bayes_opt-worktree/"
                       "checkpoints/grpo_enhanced/grpo_enhanced_20250909_111252/episode_int_100/policy.pkl")
    
    if not policy_path.exists():
        logger.error(f"Policy not found at {policy_path}")
        return
    
    # Load policy
    logger.info("Loading coefficient-trained policy...")
    policy_checkpoint = load_checkpoint(policy_path)
    policy_params = policy_checkpoint['policy']
    policy_architecture = policy_checkpoint.get('architecture', {})
    
    # Create policy network
    policy_fn = create_quantile_policy(
        hidden_dim=policy_architecture.get('hidden_dim', 512)
    )
    policy_net = hk.without_apply_rng(hk.transform(policy_fn))
    
    # Evaluation configuration
    num_episodes = 30
    num_interventions = 40
    structure_type = 'chain'
    num_vars = 8
    
    # Results storage
    results = {
        'episodes': [],
        'metadata': {
            'policy_path': str(policy_path),
            'policy_architecture': policy_architecture,
            'num_episodes': num_episodes,
            'num_interventions': num_interventions,
            'structure_type': structure_type,
            'num_vars': num_vars
        }
    }
    
    # SCM Factory
    scm_factory = VariableSCMFactory()
    
    # Run evaluation
    logger.info(f"Evaluating {num_episodes} episodes...")
    for episode_idx in range(num_episodes):
        # Create SCM
        scm = scm_factory.create(
            structure_type=structure_type,
            num_variables=num_vars,
            seed=42 + episode_idx
        )
        
        # Evaluate episode
        episode_data = evaluate_episode_with_coefficients(
            policy_net, policy_params,
            scm,
            num_interventions=num_interventions,
            seed=42 + episode_idx,
            verbose=(episode_idx == 0)  # Only verbose for first episode
        )
        
        results['episodes'].append(episode_data)
        
        if (episode_idx + 1) % 5 == 0:
            logger.info(f"Completed {episode_idx + 1}/{num_episodes} episodes")
    
    # Save results
    output_dir = Path("thesis_results/capacity_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"coefficient_evaluation_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {output_path}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("COEFFICIENT POLICY EVALUATION SUMMARY")
    print("="*70)
    
    # Calculate metrics
    all_target_values = []
    all_parent_rates = []
    convergence_steps = []
    
    for episode in results['episodes']:
        target_values = episode['target_values']
        parent_selections = episode['parent_selections']
        
        all_target_values.extend(target_values)
        all_parent_rates.append(np.mean(parent_selections) * 100)
        
        # Find convergence (first time reaching < -5.0)
        conv_step = next((i for i, v in enumerate(target_values) if v < -5.0), len(target_values))
        convergence_steps.append(conv_step)
    
    print(f"\n1. TARGET OPTIMIZATION:")
    print(f"   Final values: {np.mean([ep['target_values'][-1] for ep in results['episodes']]):.2f} ± "
          f"{np.std([ep['target_values'][-1] for ep in results['episodes']]):.2f}")
    print(f"   Best achieved: {min(all_target_values):.2f}")
    print(f"   Average across trajectory: {np.mean(all_target_values):.2f}")
    
    print(f"\n2. PARENT SELECTION:")
    print(f"   Average rate: {np.mean(all_parent_rates):.1f}% ± {np.std(all_parent_rates):.1f}%")
    
    print(f"\n3. CONVERGENCE SPEED:")
    converged = [s for s in convergence_steps if s < num_interventions]
    if converged:
        print(f"   Steps to reach -5.0: {np.mean(converged):.1f}")
        print(f"   Episodes converged: {len(converged)}/{len(convergence_steps)} "
              f"({len(converged)/len(convergence_steps)*100:.0f}%)")
    else:
        print(f"   No episodes reached -5.0 threshold")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()