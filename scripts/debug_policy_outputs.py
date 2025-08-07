#!/usr/bin/env python3
"""
Debug policy outputs to understand why it chooses wrong interventions.
Simpler approach that directly analyzes policy behavior.
"""

import sys
sys.path.append('.')

import numpy as np
import jax
import jax.numpy as jnp
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.causal_bayes_opt.training.unified_grpo_trainer import UnifiedGRPOTrainer
from src.causal_bayes_opt.experiments.benchmark_scms import (
    create_fork_scm, create_chain_scm, create_collider_scm
)
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.data_structures.sample import create_sample, get_values
from src.causal_bayes_opt.training.five_channel_converter import buffer_to_five_channel_tensor
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.data_structures.scm import get_variables, get_target, get_parents


def debug_policy_on_scm(trainer, scm, scm_name):
    """Debug policy behavior on a specific SCM."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Debugging {scm_name} SCM")
    logger.info(f"{'='*60}")
    
    # Get SCM info
    variables = get_variables(scm)
    target_var = get_target(scm)
    parents = get_parents(scm, target_var)
    
    logger.info(f"Variables: {variables}")
    logger.info(f"Target: {target_var}")
    logger.info(f"Parents of target: {parents}")
    
    # Create variable mapper
    mapper = trainer._get_variable_mapper(scm)
    logger.info(f"Variable mapping: {mapper.var_to_idx}")
    logger.info(f"Target index: {mapper.target_idx}")
    
    # Initialize buffer and sample initial data
    buffer = ExperienceBuffer(max_size=100)
    
    # Sample 5 initial observations
    for _ in range(5):
        values = sample_from_linear_scm(scm)
        sample = create_sample(
            observations=values,
            outcome_values=values,
            interventions=frozenset(),
            intervention_values={}
        )
        buffer.add(sample)
    
    # Convert to tensor
    tensor = buffer_to_five_channel_tensor(buffer, trainer.num_timesteps)
    logger.info(f"\nTensor shape: {tensor.shape}")
    
    # Show current values
    current_values = tensor[0, :, 0]  # Most recent values
    logger.info("\nCurrent variable values:")
    for var, idx in mapper.var_to_idx.items():
        logger.info(f"  {var}: {current_values[idx]:.3f}")
    
    # Show all channels for each variable
    logger.info("\nAll channels (most recent timestep):")
    channel_names = ['values', 'target_ind', 'intervention_ind', 'marginal_probs', 'recency']
    for var, idx in mapper.var_to_idx.items():
        logger.info(f"\n{var}:")
        for ch_idx, ch_name in enumerate(channel_names):
            logger.info(f"  {ch_name}: {tensor[0, idx, ch_idx]:.3f}")
    
    # Get policy output
    rng_key = jax.random.PRNGKey(42)
    policy_output = trainer.policy_fn.apply(
        trainer.policy_params, rng_key, tensor, mapper.target_idx
    )
    
    # Analyze variable selection
    var_logits = policy_output['variable_logits']
    var_probs = jax.nn.softmax(var_logits)
    
    logger.info("\nPolicy outputs:")
    logger.info("Variable selection probabilities:")
    for var, idx in mapper.var_to_idx.items():
        is_target = " (TARGET)" if var == target_var else ""
        is_parent = " (PARENT)" if var in parents else ""
        logger.info(f"  {var}: logit={var_logits[idx]:.3f}, prob={var_probs[idx]:.3f}{is_target}{is_parent}")
    
    # Find which variable is most likely to be selected
    max_prob_idx = jnp.argmax(var_probs)
    max_prob_var = mapper.idx_to_var[int(max_prob_idx)]
    logger.info(f"\nMost likely selection: {max_prob_var} (prob={var_probs[max_prob_idx]:.3f})")
    
    if max_prob_var == target_var:
        logger.warning("WARNING: Policy wants to intervene on target!")
    elif max_prob_var not in parents:
        logger.warning(f"WARNING: {max_prob_var} is not a parent of {target_var}")
    
    # Analyze value parameters
    logger.info("\nValue parameters for each variable:")
    for var, idx in mapper.var_to_idx.items():
        value_params = policy_output['value_params'][idx]
        mean, log_std = value_params[0], value_params[1]
        std = jnp.exp(log_std)
        logger.info(f"  {var}: mean={mean:.3f}, std={std:.3f}")
    
    # Simulate what would happen with different interventions
    logger.info("\nSimulating interventions:")
    for var in variables:
        if var == target_var:
            continue
            
        # Sample intervention value from policy
        idx = mapper.var_to_idx[var]
        value_params = policy_output['value_params'][idx]
        mean = value_params[0]
        
        # Apply intervention
        intervention_dict = {var: float(mean)}
        
        # Sample outcome
        outcome_values = sample_from_linear_scm(
            scm, 
            intervention_targets=set([var]),
            intervention_values=intervention_dict
        )
        
        target_value = outcome_values[target_var]
        logger.info(f"  Intervene {var}={mean:.3f} -> {target_var}={target_value:.3f}")


def main():
    """Debug policy behavior."""
    
    # Create trainer
    trainer = UnifiedGRPOTrainer(
        learning_rate=3e-3,
        n_episodes=1,  # We won't train, just analyze initial policy
        episode_length=1,
        batch_size=1,
        use_early_stopping=False,
        reward_weights={
            'optimization': 1.0,
            'discovery': 0.0,
            'efficiency': 0.0,
            'info_gain': 0.0
        },
        optimization_direction="MINIMIZE",
        use_surrogate=False,
        seed=42
    )
    
    # Create SCMs
    scms = {
        'fork': create_fork_scm(),
        'chain': create_chain_scm(), 
        'collider': create_collider_scm()
    }
    
    # Debug each SCM
    for scm_name, scm in scms.items():
        debug_policy_on_scm(trainer, scm, scm_name)
    
    # Now train for a few episodes and check again
    logger.info("\n" + "="*80)
    logger.info("Training for 30 episodes...")
    logger.info("="*80)
    
    trainer.max_episodes = 30  # 10 per SCM
    trainer.convergence_config = {
        'patience': 1000,
        'min_episodes': 100,
        'max_episodes_per_scm': 10
    }
    
    result = trainer.train(scms)
    
    logger.info("\n" + "="*80)
    logger.info("After training - checking policy behavior again")
    logger.info("="*80)
    
    # Debug each SCM again
    for scm_name, scm in scms.items():
        debug_policy_on_scm(trainer, scm, scm_name)


if __name__ == "__main__":
    main()