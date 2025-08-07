#!/usr/bin/env python3
"""
Detailed debugging script to understand why the model chooses wrong interventions.
Adds print statements and breakpoints to analyze:
1. Variable embeddings at each layer
2. Logit values and their variance
3. Gradient flow and reinforcement
4. Why interventions increase target values
"""

import sys
sys.path.append('.')

import numpy as np
import jax
import jax.numpy as jnp
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.causal_bayes_opt.training.unified_grpo_trainer import UnifiedGRPOTrainer
from src.causal_bayes_opt.experiments.benchmark_scms import (
    create_fork_scm, create_chain_scm, create_collider_scm
)
from src.causal_bayes_opt.training.five_channel_converter import buffer_to_five_channel_tensor
from src.causal_bayes_opt.data_structures.sample import get_values


class DebugGRPOTrainer(UnifiedGRPOTrainer):
    """Modified trainer with detailed debugging output."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_data = defaultdict(list)
        self.episode_count = 0
        
    def _collect_grpo_data(self, episode: int, scm, scm_name: str, rng_key) -> dict:
        """Override to add debugging."""
        self.episode_count += 1
        
        # Get target variable
        if scm_name == 'fork':
            target_var = 'Y'
        elif scm_name == 'chain':
            target_var = 'X2'
        elif scm_name == 'collider':
            target_var = 'Z'
        else:
            target_var = getattr(scm, 'target', 'Y')
            
        logger.info(f"\n{'='*60}")
        logger.info(f"Episode {episode}, SCM: {scm_name}, Target: {target_var}")
        logger.info(f"{'='*60}")
        
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
        
        # Get variable mapper
        mapper = self._get_variable_mapper(scm)
        logger.info(f"Variables: {list(mapper.var_to_idx.keys())}")
        logger.info(f"Target index: {mapper.target_idx}")
        
        # Collect transitions
        self.rng_key, collect_key = jax.random.split(self.rng_key)
        
        for step in range(min(3, self.episode_length)):  # Only debug first 3 steps
            step_key, collect_key = jax.random.split(collect_key)
            
            logger.info(f"\n--- Step {step} ---")
            
            # Convert buffer to tensor (using 5-channel format)
            tensor = buffer_to_five_channel_tensor(buffer, self.num_timesteps)
            logger.info(f"Tensor shape: {tensor.shape}")
            
            # Extract variable values from tensor (first row is current values)
            current_values = tensor[0, :, 0]  # [num_vars, channels] -> just values channel
            logger.info(f"Current variable values:")
            for var, idx in mapper.var_to_idx.items():
                logger.info(f"  {var}: {current_values[idx]:.3f}")
            
            # Get policy output
            policy_key, value_key, step_key = jax.random.split(step_key, 3)
            
            # Debug: Apply policy step by step
            policy_output = self.policy_fn.apply(
                self.policy_params, policy_key, tensor, mapper.target_idx
            )
            
            # Analyze variable selection
            var_logits = policy_output['variable_logits']
            var_probs = jax.nn.softmax(var_logits)
            
            logger.info(f"\nVariable selection:")
            for var, idx in mapper.var_to_idx.items():
                logger.info(f"  {var}: logit={var_logits[idx]:.3f}, prob={var_probs[idx]:.3f}")
            
            # Sample variable
            selected_idx = jax.random.categorical(policy_key, var_logits)
            selected_var = mapper.idx_to_var[int(selected_idx)]
            logger.info(f"\nSelected variable: {selected_var} (idx={selected_idx})")
            
            # Debug: Why was this variable selected?
            if selected_var == target_var:
                logger.warning(f"WARNING: Selected target variable {target_var}!")
            
            # Analyze value parameters
            value_params = policy_output['value_params'][selected_idx]
            value_mean, value_log_std = value_params[0], value_params[1]
            value_std = jnp.exp(value_log_std)
            
            logger.info(f"\nValue parameters for {selected_var}:")
            logger.info(f"  Mean: {value_mean:.3f}")
            logger.info(f"  Std: {value_std:.3f}")
            logger.info(f"  Current value: {current_values[selected_idx]:.3f}")
            
            # Sample intervention value
            intervention_value = value_mean + value_std * jax.random.normal(value_key)
            logger.info(f"  Sampled intervention: {intervention_value:.3f}")
            
            # Apply intervention
            intervention_dict = {selected_var: float(intervention_value)}
            buffer = self._apply_intervention(buffer, scm, intervention_dict)
            
            # Get outcome
            if buffer.size > 0:
                outcome_sample = buffer.get_recent_samples(1)[0]
                outcome_values = get_values(outcome_sample)
                target_value = outcome_values.get(target_var, 0.0)
                
                logger.info(f"\nOutcome after intervention:")
                logger.info(f"  Target {target_var}: {target_value:.3f}")
                
                # Compute reward
                from ..acquisition.better_rewards import compute_better_clean_reward
                
                reward_info = compute_better_clean_reward(
                    buffer_before=buffer,
                    intervention={
                        'targets': frozenset([selected_var]),
                        'values': {selected_var: float(intervention_value)}
                    },
                    outcome=outcome_sample,
                    target_variable=target_var,
                    config={
                        'optimization_direction': self.optimization_direction,
                        'reward_type': 'adaptive_sigmoid',
                        'temperature_factor': 2.0,
                        'weights': {
                            'target': self.reward_weights.get('optimization', 1.0),
                            'diversity': 0.0,
                            'efficiency': 0.0,
                        }
                    },
                    stats=self.reward_stats,
                    posterior_before=None,
                    posterior_after=None
                )
                
                reward = reward_info['total']
                logger.info(f"  Reward: {reward:.3f}")
                
                # Store debug data
                self.debug_data['interventions'].append({
                    'episode': episode,
                    'step': step,
                    'scm': scm_name,
                    'selected_var': selected_var,
                    'intervention_value': float(intervention_value),
                    'target_value': float(target_value),
                    'reward': float(reward),
                    'var_probs': {var: float(var_probs[idx]) for var, idx in mapper.var_to_idx.items()},
                    'var_logits': {var: float(var_logits[idx]) for var, idx in mapper.var_to_idx.items()},
                })
            
            # Log probability
            var_log_prob = jnp.log(var_probs[selected_idx])
            value_log_prob = -0.5 * ((intervention_value - value_mean) / value_std) ** 2 - value_log_std - 0.5 * jnp.log(2 * jnp.pi)
            total_log_prob = var_log_prob + value_log_prob
            
            # Store transition
            grpo_batch_data['states'].append(tensor)
            grpo_batch_data['actions'].append({
                'variable_idx': selected_idx,
                'value': intervention_value
            })
            grpo_batch_data['rewards'].append(reward)
            grpo_batch_data['old_log_probs'].append(total_log_prob)
            grpo_batch_data['masks'].append(1.0)
        
        # Convert to arrays
        grpo_batch_data['rewards'] = jnp.array(grpo_batch_data['rewards'])
        grpo_batch_data['old_log_probs'] = jnp.array(grpo_batch_data['old_log_probs'])
        grpo_batch_data['masks'] = jnp.array(grpo_batch_data['masks'])
        
        return grpo_batch_data
    
    def _debug_policy_internals(self, tensor, target_idx):
        """Debug what happens inside the policy network."""
        # This would require modifying the policy to return intermediate activations
        # For now, we'll focus on input/output analysis
        pass


def analyze_debug_data(debug_data):
    """Analyze the collected debug data to find patterns."""
    
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS OF INTERVENTION CHOICES")
    logger.info("="*80)
    
    # Group by SCM
    scm_groups = defaultdict(list)
    for entry in debug_data['interventions']:
        scm_groups[entry['scm']].append(entry)
    
    for scm_name, entries in scm_groups.items():
        logger.info(f"\n{scm_name.upper()} SCM Analysis:")
        
        # Count variable selections
        var_counts = defaultdict(int)
        for entry in entries:
            var_counts[entry['selected_var']] += 1
        
        logger.info("\nVariable selection frequency:")
        for var, count in sorted(var_counts.items()):
            pct = 100 * count / len(entries)
            logger.info(f"  {var}: {count} times ({pct:.1f}%)")
        
        # Analyze target value changes
        target_changes = []
        for i in range(1, len(entries)):
            if entries[i]['episode'] == entries[i-1]['episode']:
                change = entries[i]['target_value'] - entries[i-1]['target_value']
                target_changes.append(change)
        
        if target_changes:
            logger.info(f"\nTarget value changes:")
            logger.info(f"  Mean change: {np.mean(target_changes):.3f}")
            logger.info(f"  Positive changes: {sum(c > 0 for c in target_changes)} / {len(target_changes)}")
        
        # Analyze logit patterns
        logger.info("\nAverage logits by variable:")
        var_logits_sum = defaultdict(list)
        for entry in entries:
            for var, logit in entry['var_logits'].items():
                var_logits_sum[var].append(logit)
        
        for var, logits in sorted(var_logits_sum.items()):
            logger.info(f"  {var}: mean={np.mean(logits):.3f}, std={np.std(logits):.3f}")
        
        # Look for patterns in wrong interventions
        bad_interventions = [e for e in entries if e['target_value'] > 0]  # For MINIMIZE
        if bad_interventions:
            logger.info(f"\nBad interventions (increased target): {len(bad_interventions)} / {len(entries)}")
            
            # What variables were chosen?
            bad_var_counts = defaultdict(int)
            for entry in bad_interventions:
                bad_var_counts[entry['selected_var']] += 1
            
            logger.info("Variables chosen in bad interventions:")
            for var, count in sorted(bad_var_counts.items()):
                pct = 100 * count / len(bad_interventions)
                logger.info(f"  {var}: {count} times ({pct:.1f}%)")


def main():
    """Run debugging analysis."""
    
    logger.info("="*80)
    logger.info("DEBUGGING GRPO INTERVENTION CHOICES")
    logger.info("="*80)
    
    # Create simple trainer with minimal config
    trainer = DebugGRPOTrainer(
        learning_rate=3e-2,  # High LR to see changes
        n_episodes=9,  # 3 per SCM for debugging
        episode_length=10,
        batch_size=8,
        use_early_stopping=False,
        reward_weights={
            'optimization': 1.0,
            'discovery': 0.0,
            'efficiency': 0.0,
            'info_gain': 0.0
        },
        optimization_direction="MINIMIZE",
        use_surrogate=False,
        seed=42,
        convergence_config={
            'patience': 1000,
            'min_episodes': 100,
            'max_episodes_per_scm': 3
        }
    )
    
    # Create SCMs
    scms = {
        'fork': create_fork_scm(),
        'chain': create_chain_scm(), 
        'collider': create_collider_scm()
    }
    
    # Train with debugging
    result = trainer.train(scms)
    
    # Analyze collected data
    analyze_debug_data(trainer.debug_data)
    
    # Additional analysis: Check if embeddings are distinguishable
    logger.info("\n" + "="*80)
    logger.info("CHECKING VARIABLE DISTINGUISHABILITY")
    logger.info("="*80)
    
    # Get a sample tensor and check embeddings
    fork_scm = create_fork_scm()
    buffer = trainer._initialize_buffer(fork_scm)
    tensor = buffer_to_five_channel_tensor(buffer, trainer.num_timesteps)
    mapper = trainer._get_variable_mapper(fork_scm)
    
    logger.info("\nInitial tensor values (Fork SCM):")
    current_values = tensor[0, :, 0]
    for var, idx in mapper.var_to_idx.items():
        logger.info(f"  {var}: {current_values[idx]:.3f}")
    
    # Check channel information
    logger.info("\nChannel information (standardized):")
    for channel_idx, channel_name in enumerate(['values', 'target_ind', 'intervention_ind', 'marginal_probs', 'recency']):
        logger.info(f"\n{channel_name}:")
        channel_data = tensor[0, :, channel_idx]
        for var, idx in mapper.var_to_idx.items():
            logger.info(f"  {var}: {channel_data[idx]:.3f}")


if __name__ == "__main__":
    main()