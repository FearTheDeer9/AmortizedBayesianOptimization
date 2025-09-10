#!/usr/bin/env python3
"""
AVICI-style Surrogate Model Training.

Trains a causal discovery surrogate model using diverse graph generation.
Uses homogeneous batches (same number of variables) for efficient training.

Example usage:
    # Basic training for 1000 steps
    python train_avici_style.py --num-steps 1000 --batch-size 32
    
    # Continue training from checkpoint
    python train_avici_style.py --checkpoint checkpoint.pkl --num-steps 500
    
    # Train on specific structures with custom learning rate
    python train_avici_style.py --structure-types chain fork --lr 1e-4 --num-steps 2000
    
    # Train with time limit (useful for cluster jobs)
    python train_avici_style.py --max-time-minutes 120 --checkpoint-output final.pkl
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
import optax

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from optimizer_utils import create_adaptive_optimizer, create_curriculum_optimizer_config
from src.causal_bayes_opt.utils.checkpoint_utils import save_checkpoint
from src.causal_bayes_opt.data_structures.scm import get_parents, get_variables, get_target

# Add BC utilities import
sys.path.insert(0, str(Path(__file__).parent.parent))
from bc_utils import (
    load_expert_demonstrations, 
    run_bc_warmup,
    compute_kl_regularization
)
from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
import pyrsistent as pyr
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.data_structures.sample import create_sample
from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
from src.causal_bayes_opt.environments.sampling import sample_with_intervention
from src.causal_bayes_opt.avici_integration.continuous.model import ContinuousParentSetPredictionModel
from src.causal_bayes_opt.policies.permutation_invariant_alternating_policy import create_permutation_invariant_alternating_policy
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
from src.causal_bayes_opt.utils.variable_mapping import VariableMapper


def generate_homogeneous_graph_batch(rng_key, batch_size: int = 32, min_vars: int = 3, max_vars: int = 100,                structure_types: List[str] = None) -> Tuple[int, List[Dict]]:
    """
    Generate homogeneous batch where all graphs have the same number of variables.
    This follows AVICI's approach of avoiding padding by using same-size batches.
    
    Args:
        rng_key: Random key for generation
        batch_size: Number of graphs in batch
        min_vars: Minimum number of variables
        max_vars: Maximum number of variables
        structure_types: List of structure types to sample from. 
                        Options: 'random', 'chain', 'fork', 'collider', 'mixed', 'scale_free', 'two_layer'
                        Default: ['random', 'chain', 'fork', 'collider', 'mixed']
    
    Returns:
        num_vars: Number of variables for all graphs in this batch
        graphs: List of graph configurations with identical num_vars
    """
    # Default structure types if not specified
    if structure_types is None:
        structure_types = ['random', 'chain', 'fork', 'collider', 'mixed']
    
    # Sample number of variables for this entire batch
    rng_key, vars_key = random.split(rng_key)
    num_vars = int(random.uniform(vars_key, minval=min_vars, maxval=max_vars + 1))
    
    graphs = []
    
    for i in range(batch_size):
        # Sample graph type from provided list
        rng_key, type_key = random.split(rng_key)
        structure = structure_types[int(random.choice(type_key, len(structure_types)))]
        
        # Set edge density based on structure type
        if structure == 'random':
            # Erdos-Renyi with varying edge density (1-3 edges per var)
            rng_key, density_key = random.split(rng_key)
            edges_per_var = random.uniform(density_key, minval=1.0, maxval=3.0)
            edge_density = min(edges_per_var / (num_vars - 1), 0.5)  # Cap at 0.5
        elif structure == 'chain':
            # Chain structure (hardest for causal discovery)
            edge_density = 1.0 / (num_vars - 1) if num_vars > 1 else 0.0
        elif structure in ['fork', 'collider']:
            # Fork/Collider structures
            edge_density = 0.3
        elif structure == 'mixed':
            # Mixed structure
            edge_density = 0.25
        elif structure == 'scale_free':
            # Scale-free networks tend to be sparser
            edge_density = 0.2
        elif structure == 'two_layer':
            # Two-layer hierarchical structure
            edge_density = 0.35
        else:
            # Default for unknown types
            edge_density = 0.3
        
        graphs.append({
            'num_vars': num_vars,  # All graphs in batch have same num_vars
            'structure': structure,
            'edge_density': edge_density
        })
    
    return num_vars, graphs


def create_scm_with_factory(config: Dict, factory: VariableSCMFactory, rng_key) -> pyr.PMap:
    """Create SCM using VariableSCMFactory with random target selection."""
    num_vars = config['num_vars']
    structure = config['structure']
    edge_density = config['edge_density']
    
    # Map structure names from training config to factory structure types
    structure_mapping = {
        'random': 'random',
        'chain': 'chain',
        'fork': 'fork',
        'collider': 'collider',
        'mixed': 'mixed'
    }
    
    factory_structure = structure_mapping.get(structure, 'random')
    
    # Randomize target selection for ALL structures to create diversity
    variables = [f"X{i}" for i in range(num_vars)]
    if factory_structure == 'chain':
        # For chains, randomize which direction the chain goes
        target_idx = int(random.choice(rng_key, num_vars))
        target_variable = variables[target_idx]
    else:
        # For other structures, also randomize target
        target_idx = int(random.choice(rng_key, num_vars)) 
        target_variable = variables[target_idx]
    
    # Create SCM using factory
    scm = factory.create_variable_scm(
        num_variables=num_vars,
        structure_type=factory_structure,
        target_variable=target_variable,  # Randomize target for diversity
        edge_density=edge_density
    )
    
    return scm




def initialize_models(hidden_dim: int = 128, 
                     num_layers: int = 8,
                     num_heads: int = 8,
                     key_size: int = 32,
                     dropout: float = 0.1,
                     key: jax.random.PRNGKey = None,
                     max_vars: int = 100) -> Tuple:
    """Initialize policy and surrogate models with AVICI configuration."""
    # Create dummy data for initialization
    dummy_buffer = ExperienceBuffer()
    dummy_values = {f'X{i}': 0.0 for i in range(max_vars)}
    dummy_sample = create_sample(dummy_values, intervention_type=None)
    dummy_buffer.add_observation(dummy_sample)
    dummy_tensor, _ = buffer_to_three_channel_tensor(dummy_buffer, 'X0')
    
    # Initialize policy
    policy_key, surrogate_key = random.split(key)
    policy_fn = create_permutation_invariant_alternating_policy(hidden_dim)
    policy_net = hk.without_apply_rng(hk.transform(policy_fn))
    policy_params = policy_net.init(policy_key, dummy_tensor, 0)
    
    # Initialize surrogate with AVICI architecture
    def surrogate_fn(x, target_idx, is_training):
        model = ContinuousParentSetPredictionModel(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            key_size=key_size,
            dropout=dropout,
            use_temperature_scaling=True,
            temperature_init=0.0
        )
        return model(x, target_idx, is_training)
    
    surrogate_net = hk.transform(surrogate_fn)
    surrogate_params = surrogate_net.init(surrogate_key, dummy_tensor, 0, True)
    
    return policy_net, policy_params, surrogate_net, surrogate_params


def compute_surrogate_loss(params, net, buffer, target_idx, target_var, true_parents, variables, rng_key, use_weighted_loss=True):
    """Compute BCE loss for surrogate predictions."""
    tensor, _ = buffer_to_three_channel_tensor(buffer, target_var)
    predictions = net.apply(params, rng_key, tensor, target_idx, True)
    
    if 'parent_probabilities' in predictions:
        pred_probs = predictions['parent_probabilities']
    else:
        raw_logits = predictions.get('attention_logits', jnp.zeros(len(variables)))
        pred_probs = jax.nn.sigmoid(raw_logits)
    
    pred_probs = jnp.clip(pred_probs, 1e-7, 1 - 1e-7)
    
    # Create ground truth labels
    labels = []
    for i, var in enumerate(variables):
        if i != target_idx:
            label = 1.0 if var in true_parents else 0.0
            labels.append(label)
        else:
            labels.append(0.0)  # Target can't be its own parent
    
    labels = jnp.array(labels)
    
    # Binary cross-entropy loss with optional weighting
    d = len(variables)
    if use_weighted_loss:
        # Assume 1/5 of possible edges are present (20% positive class)
        pos_weight = 4.0  # (1 - 0.2) / 0.2 = 0.8 / 0.2 = 4.0
        bce_loss = -(pos_weight * labels * jnp.log(pred_probs) + (1 - labels) * jnp.log(1 - pred_probs))
    else:
        bce_loss = -(labels * jnp.log(pred_probs) + (1 - labels) * jnp.log(1 - pred_probs))
    
    # Normalize by number of possible parent edges for target (d - 1, excluding self)
    loss = jnp.sum(bce_loss) / (d - 1)
    
    return loss, pred_probs, labels


def compute_vectorized_surrogate_loss(params, net, batch_tensors, batch_target_indices, 
                                     batch_true_parents, variables, rng_key, use_weighted_loss=True,
                                     bc_reference_params=None, kl_weight=0.0):
    """Compute BCE loss for vectorized batch of surrogate predictions with optional KL regularization."""
    # batch_tensors: [batch_size, N, d, 3]
    # batch_target_indices: [batch_size] 
    # batch_true_parents: List of sets of parent variable names
    
    # Single forward pass for entire batch
    predictions = net.apply(params, rng_key, batch_tensors, batch_target_indices, True)
    
    if 'parent_probabilities' in predictions:
        pred_probs = predictions['parent_probabilities']  # [batch_size, d]
    else:
        raw_logits = predictions.get('attention_logits', jnp.zeros((len(batch_tensors), len(variables))))
        pred_probs = jax.nn.sigmoid(raw_logits)
    
    pred_probs = jnp.clip(pred_probs, 1e-7, 1 - 1e-7)
    
    # Create ground truth labels for entire batch
    batch_size, d = pred_probs.shape
    labels_list = []
    
    # Need to get variables from each SCM's mapper, not just first SCM
    # This is a limitation of the current vectorized approach - all SCMs must have same variables
    # For now, use the variables list but ensure consistent ordering
    variables_list = sorted(list(variables))  # Ensure consistent ordering
    
    for b in range(batch_size):
        target_idx = batch_target_indices[b]
        true_parents = batch_true_parents[b]
        
        labels = []
        for i, var in enumerate(variables_list):
            if i != target_idx:
                label = 1.0 if var in true_parents else 0.0
                labels.append(label)
            else:
                labels.append(0.0)  # Target can't be its own parent
        
        labels_list.append(jnp.array(labels))
    
    labels = jnp.stack(labels_list)  # [batch_size, d]
    
    # Binary cross-entropy loss with optional weighting
    if use_weighted_loss:
        # Assume 1/5 of possible edges are present (20% positive class)
        pos_weight = 4.0  # (1 - 0.2) / 0.2 = 0.8 / 0.2 = 4.0
        bce_loss = -(pos_weight * labels * jnp.log(pred_probs) + (1 - labels) * jnp.log(1 - pred_probs))
    else:
        bce_loss = -(labels * jnp.log(pred_probs) + (1 - labels) * jnp.log(1 - pred_probs))
    
    # Normalize by number of possible parent edges for target (d - 1, excluding self)
    bce_loss_value = jnp.mean(jnp.sum(bce_loss, axis=-1) / (d - 1))
    
    # Add KL regularization if BC reference params provided
    total_loss = bce_loss_value
    if bc_reference_params is not None and kl_weight > 0:
        # Get predictions from reference model
        ref_predictions = net.apply(bc_reference_params, rng_key, batch_tensors, 
                                   batch_target_indices, False)  # is_training=False
        
        if 'parent_probabilities' in ref_predictions:
            ref_probs = ref_predictions['parent_probabilities']
        else:
            ref_logits = ref_predictions.get('attention_logits', jnp.zeros((len(batch_tensors), d)))
            ref_probs = jax.nn.sigmoid(ref_logits)
        
        ref_probs = jnp.clip(ref_probs, 1e-7, 1 - 1e-7)
        
        # Compute KL divergence for each example in batch
        # KL(current || reference) for Bernoulli distributions
        kl_pos = pred_probs * jnp.log(pred_probs / ref_probs)
        kl_neg = (1 - pred_probs) * jnp.log((1 - pred_probs) / (1 - ref_probs))
        kl_div = kl_pos + kl_neg
        
        # Mask out target variables for KL computation
        mask = jnp.ones((batch_size, d))
        for b in range(batch_size):
            mask = mask.at[b, batch_target_indices[b]].set(0.0)
        
        masked_kl = kl_div * mask
        kl_loss = jnp.mean(jnp.sum(masked_kl, axis=-1) / jnp.sum(mask, axis=-1))
        
        # Add weighted KL to total loss
        total_loss = bce_loss_value + kl_weight * kl_loss
    
    return total_loss, pred_probs, labels


def train_batch_vectorized(scm_batch: List,
                           policy_net, policy_params,
                           surrogate_net, surrogate_params,
                           optimizer, opt_state,
                           min_datapoints: int,
                           max_datapoints: int,
                           min_obs_ratio: float,
                           max_obs_ratio: float,
                           rng_key,
                           use_weighted_loss: bool = True,
                           bc_reference_params=None,
                           kl_weight: float = 0.0) -> Tuple:
    """Train on a homogeneous batch of SCMs with vectorized processing."""
    
    batch_size = len(scm_batch)
    
    # All SCMs have same number of variables
    first_scm = scm_batch[0]
    variables = get_variables(first_scm)
    num_vars = len(variables)
    
    # Prepare batch data structures
    batch_tensors = []
    batch_target_indices = []
    batch_true_parents = []
    batch_mappers = []
    
    for scm in scm_batch:
        target_var = get_target(scm)
        mapper = VariableMapper(variables, target_variable=target_var)
        target_idx = mapper.target_idx
        true_parents = get_parents(scm, target_var)
        
        # Generate data for this SCM
        buffer = ExperienceBuffer()
        
        # Sample data configuration for this SCM
        rng_key, datapoint_key, ratio_key = random.split(rng_key, 3)
        total_datapoints = int(random.uniform(datapoint_key, minval=min_datapoints, maxval=max_datapoints))
        obs_ratio = float(random.uniform(ratio_key, minval=min_obs_ratio, maxval=max_obs_ratio))
        
        num_observations = int(total_datapoints * obs_ratio)
        num_interventions = total_datapoints - num_observations
        
        # Generate observational data
        rng_key, obs_key = random.split(rng_key)
        obs_seed = int(obs_key[0]) % 1000000
        samples = sample_from_linear_scm(scm, n_samples=num_observations, seed=obs_seed)
        for sample in samples:
            buffer.add_observation(sample)
        
        # Generate interventional data
        if num_interventions > 0:
            rng_key, int_key = random.split(rng_key)
            int_keys = random.split(int_key, num_interventions * 3)
            
            valid_mask = jnp.ones(len(variables)).at[target_idx].set(0)
            valid_indices = jnp.where(valid_mask)[0]
            
            intervention_indices = random.choice(
                int_keys[0], valid_indices, shape=(num_interventions,)
            )
            intervention_values = random.normal(
                int_keys[1], shape=(num_interventions,)
            ) * 2.0
            
            interventions_by_var = {}
            for i in range(num_interventions):
                var_idx = int(intervention_indices[i])
                var_name = mapper.get_name(var_idx)
                
                if var_name not in interventions_by_var:
                    interventions_by_var[var_name] = []
                interventions_by_var[var_name].append(float(intervention_values[i]))
            
            seed_counter = 0
            for var_name, values in interventions_by_var.items():
                for value in values:
                    intervention = create_perfect_intervention(
                        targets=frozenset([var_name]),
                        values={var_name: value}
                    )
                    
                    post_seed = int(int_keys[2 + seed_counter % len(int_keys)][0]) % 1000000
                    seed_counter += 1
                    
                    post_data = sample_with_intervention(scm, intervention, 1, seed=post_seed)
                    if post_data:
                        buffer.add_intervention(intervention, post_data[0])
        
        # Convert to tensor and store
        tensor, _ = buffer_to_three_channel_tensor(buffer, target_var)
        batch_tensors.append(tensor)
        batch_target_indices.append(target_idx)
        batch_true_parents.append(true_parents)
        batch_mappers.append(mapper)
    
    # Stack all tensors into batch format
    # Find max N across all tensors for padding
    max_N = max(tensor.shape[0] for tensor in batch_tensors)
    
    # Pad all tensors to same N
    padded_tensors = []
    for tensor in batch_tensors:
        N, d, channels = tensor.shape
        if N < max_N:
            # Pad with zeros at the beginning (our convention)
            padding = jnp.zeros((max_N - N, d, channels))
            padded_tensor = jnp.concatenate([padding, tensor], axis=0)
        else:
            padded_tensor = tensor
        padded_tensors.append(padded_tensor)
    
    batch_data = jnp.stack(padded_tensors)  # [batch_size, max_N, d, 3]
    batch_target_indices = jnp.array(batch_target_indices)  # [batch_size]
    
    # Compute loss and gradients with vectorized approach
    def loss_fn(params):
        rng_key_loss = random.PRNGKey(0)  # Deterministic for gradient
        loss, pred_probs, labels = compute_vectorized_surrogate_loss(
            params, surrogate_net, batch_data, batch_target_indices,
            batch_true_parents, variables, rng_key_loss, use_weighted_loss,
            bc_reference_params, kl_weight
        )
        return loss, (pred_probs, labels)
    
    (avg_loss, (batch_pred_probs, batch_labels)), grads = jax.value_and_grad(loss_fn, has_aux=True)(surrogate_params)
    
    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state, surrogate_params)
    surrogate_params = optax.apply_updates(surrogate_params, updates)
    
    # Compute batch metrics
    batch_predictions = batch_pred_probs > 0.5
    batch_tp = jnp.sum(batch_predictions * batch_labels, axis=-1)  # [batch_size]
    batch_fp = jnp.sum(batch_predictions * (1 - batch_labels), axis=-1)
    batch_fn = jnp.sum((1 - batch_predictions) * batch_labels, axis=-1)
    
    batch_precision = batch_tp / (batch_tp + batch_fp + 1e-8)
    batch_recall = batch_tp / (batch_tp + batch_fn + 1e-8) 
    batch_f1 = 2 * batch_precision * batch_recall / (batch_precision + batch_recall + 1e-8)
    
    # Compute AUROC and AUPRC for each item in batch
    batch_auroc = []
    batch_auprc = []
    for i in range(batch_size):
        # Convert to numpy for sklearn
        labels_np = np.array(batch_labels[i])
        probs_np = np.array(batch_pred_probs[i])
        
        # Only compute if there are both positive and negative examples
        if labels_np.sum() > 0 and labels_np.sum() < len(labels_np):
            try:
                from sklearn.metrics import roc_auc_score, average_precision_score

                auroc = roc_auc_score(labels_np, probs_np)
                auprc = average_precision_score(labels_np, probs_np)
            except:
                auroc = 0.5  # Random baseline
                auprc = labels_np.sum() / len(labels_np)  # Proportion of positives
        else:
            # All same class - metrics undefined
            auroc = 0.5  # Undefined, use random baseline
            auprc = labels_np.sum() / len(labels_np) if labels_np.sum() > 0 else 0.0
        
        batch_auroc.append(auroc)
        batch_auprc.append(auprc)
    
    # Create metrics for each SCM in batch
    metrics = []
    for i, (scm, mapper) in enumerate(zip(scm_batch, batch_mappers)):
        variables_list = list(variables)
        target_var = get_target(scm)
        
        # Generate data info (we could track this during generation for accuracy)
        rng_key, sample_key = random.split(rng_key)
        total_datapoints = int(random.uniform(sample_key, minval=min_datapoints, maxval=max_datapoints))
        obs_ratio = 0.7  # Approximate, for logging
        
        metrics.append({
            'num_vars': num_vars,
            'structure': scm.get('metadata', {}).get('structure', 'unknown'),
            'total_datapoints': total_datapoints,  # Approximate 
            'num_observations': int(total_datapoints * obs_ratio),
            'num_interventions': int(total_datapoints * (1 - obs_ratio)),
            'obs_ratio': obs_ratio,
            'loss': float(avg_loss),  # Same loss for all in batch
            'f1': float(batch_f1[i]),
            'precision': float(batch_precision[i]),
            'recall': float(batch_recall[i]),
            'auroc': float(batch_auroc[i]),
            'auprc': float(batch_auprc[i])
        })
    
    avg_f1 = float(jnp.mean(batch_f1))
    avg_auroc = float(np.mean(batch_auroc))
    avg_auprc = float(np.mean(batch_auprc))
    
    return surrogate_params, opt_state, avg_loss, avg_f1, avg_auroc, avg_auprc, metrics


def train_batch(scm_batch: List, 
               policy_net, policy_params,
               surrogate_net, surrogate_params,
               optimizer, opt_state, 
               min_datapoints: int,
               max_datapoints: int,
               min_obs_ratio: float,
               max_obs_ratio: float,
               rng_key) -> Tuple:
    """Train on a batch of diverse SCMs."""
    
    total_loss = 0.0
    num_updates = 0
    metrics = []
    
    for scm in scm_batch:
        variables = get_variables(scm)
        target_var = get_target(scm)
        mapper = VariableMapper(variables, target_variable=target_var)
        target_idx = mapper.target_idx
        true_parents = get_parents(scm, target_var)
        
        # Generate variable amount of data with variable ratios
        buffer = ExperienceBuffer()
        
        # Randomly sample total datapoints and observation ratio for this SCM
        rng_key, datapoint_key, ratio_key = random.split(rng_key, 3)
        total_datapoints = int(random.uniform(datapoint_key, minval=min_datapoints, maxval=max_datapoints))
        obs_ratio = float(random.uniform(ratio_key, minval=min_obs_ratio, maxval=max_obs_ratio))
        
        num_observations = int(total_datapoints * obs_ratio)
        num_interventions = total_datapoints - num_observations
        
        
        # Observational data
        rng_key, obs_key = random.split(rng_key)
        obs_seed = int(obs_key[0]) % 1000000
        samples = sample_from_linear_scm(scm, n_samples=num_observations, seed=obs_seed)
        for sample in samples:
            buffer.add_observation(sample)
        
        # Optimized interventional data generation - vectorized approach
        if num_interventions > 0:
            # Pre-generate all intervention parameters at once
            rng_key, int_key = random.split(rng_key)
            int_keys = random.split(int_key, num_interventions * 3)
            
            # Create valid indices mask (exclude target)
            valid_mask = jnp.ones(len(variables)).at[target_idx].set(0)
            valid_indices = jnp.where(valid_mask)[0]
            
            # Vectorized generation of intervention targets and values
            intervention_indices = random.choice(
                int_keys[0], valid_indices, shape=(num_interventions,)
            )
            intervention_values = random.normal(
                int_keys[1], shape=(num_interventions,)
            ) * 2.0
            
            # Group interventions by target variable for more efficient processing
            interventions_by_var = {}
            for i in range(num_interventions):
                var_idx = int(intervention_indices[i])
                var_name = mapper.get_name(var_idx)
                
                if var_name == target_var:  # Double-check, shouldn't happen
                    continue
                    
                if var_name not in interventions_by_var:
                    interventions_by_var[var_name] = []
                interventions_by_var[var_name].append(float(intervention_values[i]))
            
            
            # Process interventions by variable (more cache-friendly)
            seed_counter = 0
            for var_name, values in interventions_by_var.items():
                for value in values:
                    intervention = create_perfect_intervention(
                        targets=frozenset([var_name]),
                        values={var_name: value}
                    )
                    
                    # Use deterministic seed based on counter
                    post_seed = int(int_keys[2 + seed_counter % len(int_keys)][0]) % 1000000
                    seed_counter += 1
                    
                    post_data = sample_with_intervention(scm, intervention, 1, seed=post_seed)
                    if post_data:
                        buffer.add_intervention(intervention, post_data[0])
        
        
        # Compute loss and gradients
        def loss_fn(params):
            rng_key_loss = random.PRNGKey(0)  # Deterministic for gradient
            loss, pred_probs, labels = compute_surrogate_loss(
                params, surrogate_net, buffer, target_idx, 
                target_var, true_parents, mapper.variables, rng_key_loss
            )
            return loss, (pred_probs, labels)
        
        (loss, (pred_probs, labels)), grads = jax.value_and_grad(loss_fn, has_aux=True)(surrogate_params)
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state, surrogate_params)
        surrogate_params = optax.apply_updates(surrogate_params, updates)
        
        # Compute metrics
        predictions = pred_probs > 0.5
        tp = jnp.sum(predictions * labels)
        fp = jnp.sum(predictions * (1 - labels))
        fn = jnp.sum((1 - predictions) * labels)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        
        metrics.append({
            'num_vars': len(variables),
            'structure': scm.get('metadata', {}).get('structure', 'unknown'),
            'total_datapoints': total_datapoints,
            'num_observations': num_observations,
            'num_interventions': num_interventions,
            'obs_ratio': obs_ratio,
            'loss': float(loss),
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall)
        })
        
        total_loss += loss
        num_updates += 1
    
    avg_loss = total_loss / max(num_updates, 1)
    avg_f1 = np.mean([m['f1'] for m in metrics])  # Use numpy, not jax.numpy for Python list
    
    return surrogate_params, opt_state, avg_loss, avg_f1, metrics


def main():
    parser = argparse.ArgumentParser(
        description='AVICI-style surrogate training with diverse graph generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  %(prog)s --num-steps 1000 --batch-size 32
  
  # Continue from checkpoint
  %(prog)s --checkpoint model.pkl --num-steps 500
  
  # Train on specific structures
  %(prog)s --structure-types chain fork scale_free --num-steps 2000
  
Note: Learning rate is scaled by sqrt(batch_size) following AVICI's approach.
        """
    )
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden dimension (AVICI uses 128)')
    parser.add_argument('--num-layers', type=int, default=8,
                       help='Number of layers (AVICI uses 8)')
    parser.add_argument('--num-heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--key-size', type=int, default=32,
                       help='Key size for attention')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (note: learning rate is scaled by sqrt(batch_size))')
    parser.add_argument('--num-observations', type=int,  default=200, help='Number of observations per graph (deprecated, use min/max-datapoints)')
    parser.add_argument('--min-datapoints', type=int, default=100,
                       help='Minimum total datapoints (obs + interventions) per graph')
    parser.add_argument('--max-datapoints', type=int, default=1000,
                       help='Maximum total datapoints (obs + interventions) per graph')
    parser.add_argument('--min-obs-ratio', type=float, default=0.5,
                       help='Minimum ratio of observations (0.5 = 50% observations)')
    parser.add_argument('--max-obs-ratio', type=float, default=0.9,
                       help='Maximum ratio of observations (0.9 = 90% observations)')
    parser.add_argument('--lr', type=float, default=3e-5,
                       help='Base learning rate (AVICI default: 3e-5, scaled by sqrt(batch_size))')
    parser.add_argument('--num-steps', type=int, default=5000,
                       help='Number of training steps')
    parser.add_argument('--min-vars', type=int, default=3,
                       help='Minimum number of variables')
    parser.add_argument('--max-vars', type=int, default=100,
                       help='Maximum number of variables')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--use-weighted-loss', action='store_true',
                       help='Disable weighted BCE loss (weighted loss is now default)')
    parser.add_argument('--save-freq', type=int, default=100,
                       help='Save checkpoint every N steps')
    parser.add_argument('--log-freq', type=int, default=100,
                       help='Log metrics every N steps')
    parser.add_argument('--max-time-minutes', type=int, default=None,
                       help='Maximum training time in minutes')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint-output', type=str, default=None,
                       help='Explicit path for final checkpoint output')
    parser.add_argument('--structure-types', type=str, nargs='+', 
                       default=['random', 'chain', 'fork', 'collider', 'mixed'],
                       choices=['random', 'chain', 'fork', 'true_fork', 'collider', 'mixed', 'scale_free', 'two_layer'],
                       help='SCM structure types to train on')
    
    # Behavioral Cloning warm-up arguments
    parser.add_argument('--bc-warm-up', action='store_true',
                       help='Enable behavioral cloning warm-up using expert demonstrations')
    parser.add_argument('--bc-steps', type=int, default=500,
                       help='Number of BC warm-up steps')
    parser.add_argument('--kl-weight', type=float, default=0.01,
                       help='Weight for KL regularization after BC warm-up')
    parser.add_argument('--demo-path', type=str, 
                       default='expert_demonstrations/raw/raw_demonstrations',
                       help='Path to expert demonstrations')
    parser.add_argument('--max-demos', type=int, default=None,
                       help='Maximum number of demonstration files to load (for testing)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("AVICI-STYLE TRAINING WITH DIVERSE GRAPH GENERATION")
    print("="*70)
    
    print("\nConfiguration:")
    print(f"  Model: {args.hidden_dim} hidden, {args.num_layers} layers")
    print(f"  Training: {args.num_steps} steps, batch size {args.batch_size}")
    print(f"  Variables: {args.min_vars}-{args.max_vars}")
    print(f"  Structure types: {args.structure_types}")
    print(f"  Datapoints: {args.min_datapoints}-{args.max_datapoints} per graph")
    print(f"  Observation ratio: {args.min_obs_ratio:.1%}-{args.max_obs_ratio:.1%}")
    print(f"  Learning rate: {args.lr}")
    
    if args.bc_warm_up:
        print(f"\nBC Warm-up Configuration:")
        print(f"  BC steps: {args.bc_steps}")
        print(f"  KL weight: {args.kl_weight}")
        print(f"  Demo path: {args.demo_path}")
        if args.max_demos:
            print(f"  Max demos: {args.max_demos} (limited for testing)")
    
    # Create checkpoint directory
    run_name = f"avici_style_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = Path("checkpoints/avici_runs") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config['run_name'] = run_name
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize RNG
    rng_key = random.PRNGKey(args.seed)
    
    # Initialize SCM factory for consistent graph generation
    scm_factory = VariableSCMFactory(
        seed=args.seed,
        noise_scale=1.0,
        coefficient_range=(-2.0, 2.0),
        vary_intervention_ranges=True,
        use_output_bounds=True
    )
    
    # Track time if limit specified
    start_time = time.time() if args.max_time_minutes else None
    
    # Initialize models (always need full initialization for function signatures)
    rng_key, init_key = random.split(rng_key)
    policy_net, policy_params, surrogate_net, surrogate_params = initialize_models(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        key_size=args.key_size,
        dropout=args.dropout,
        key=init_key,
        max_vars=args.max_vars
    )
    
    # Determine starting step (checkpoint-based approach)
    start_step = 0
    loaded_opt_state = None  # Track if we have a saved optimizer state
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"\nLoading checkpoint from: {args.checkpoint}")
        from src.causal_bayes_opt.utils.checkpoint_utils import load_checkpoint
        checkpoint = load_checkpoint(Path(args.checkpoint))
        surrogate_params = checkpoint['params']  # Override with loaded params
        
        
        # Check if optimizer state is available
        if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
            loaded_opt_state = checkpoint['optimizer_state']
            print(f"  Loaded optimizer state for continual learning")
        else:
            print(f"  No optimizer state found, will reinitialize optimizer")
        
        # Simple checkpoint-based continuation
        last_step = checkpoint.get('metadata', {}).get('step', 0)
        start_step = last_step + 1  # Continue from next step
        print(f"  Loaded surrogate checkpoint from step {last_step}")
        print(f"  Will continue training from step {start_step}")
        print(f"  Policy params remain dummy (not used in surrogate training)")
    else:
        print("\nTraining surrogate from scratch")
    
    # Initialize optimizer with AVICI's approach
    import math
    learning_rate = math.sqrt(args.batch_size) * args.lr  # AVICI's sqrt batch size scaling
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # AVICI uses gradient clipping
        optax.lamb(learning_rate)        # AVICI uses LAMB optimizer
    )
    
    # Use loaded optimizer state if available, otherwise initialize fresh
    if loaded_opt_state is not None:
        opt_state = loaded_opt_state
        print(f"  Resuming optimizer from saved state")
    else:
        opt_state = optimizer.init(surrogate_params)
        print(f"  Initialized fresh optimizer state")
    
    print(f"Using AVICI optimizer settings:")
    print(f"  Base LR: {args.lr} -> Scaled LR: {learning_rate:.6f}")
    print(f"  Optimizer: LAMB with gradient clipping (1.0)")
    
    # Run BC warm-up if enabled
    bc_reference_params = None
    if args.bc_warm_up:
        print(f"\n{'='*70}")
        print("BEHAVIORAL CLONING WARM-UP PHASE")
        print(f"{'='*70}")
        
        # Load expert demonstrations
        demo_path = Path(project_root) / args.demo_path
        expert_demos = load_expert_demonstrations(str(demo_path), max_demos=args.max_demos)
        
        if not expert_demos:
            print("WARNING: No expert demonstrations found, skipping BC warm-up")
            args.bc_warm_up = False
        else:
            print(f"Loaded {len(expert_demos)} expert demonstrations")
            
            # Run BC warm-up
            rng_key, bc_key = random.split(rng_key)
            surrogate_params, opt_state, bc_metrics = run_bc_warmup(
                surrogate_net, surrogate_params, expert_demos,
                optimizer, opt_state, args.bc_steps, args.batch_size,
                bc_key, use_weighted_loss=not args.use_weighted_loss
            )
            
            # Store BC-trained params as reference for KL regularization
            bc_reference_params = jax.tree.map(lambda x: x, surrogate_params)
            
            print(f"BC warm-up complete. Final loss: {bc_metrics['final_bc_loss']:.4f}")
            print(f"{'='*70}\n")
    
    print(f"\nTraining for {args.num_steps} steps...")
    
    # Training metrics
    metrics_history = []
    best_f1 = 0.0
    
    # Calculate end step for continuation
    end_step = start_step + args.num_steps
    print(f"Training from step {start_step} to {end_step}")
    
    for step in range(start_step, end_step):
        # Check time limit if specified
        if args.max_time_minutes and start_time:
            elapsed_minutes = (time.time() - start_time) / 60.0
            if elapsed_minutes >= args.max_time_minutes:
                print(f"\nTime limit reached ({args.max_time_minutes} minutes), stopping training")
                print(f"Completed training up to step {step}")
                break
        
        # Generate homogeneous batch of graphs (all same num_vars)
        rng_key, batch_key = random.split(rng_key)
        num_vars_batch, graph_configs = generate_homogeneous_graph_batch(
            batch_key, args.batch_size, args.min_vars, args.max_vars,
            structure_types=args.structure_types
        )
        
        # Create SCMs from configs using factory with fresh seeds for diversity
        scm_batch = []
        for j, config in enumerate(graph_configs):
            # Create a new factory instance with a different seed for each SCM
            rng_key, scm_seed_key, target_key = random.split(rng_key, 3)
            scm_seed = int(scm_seed_key[0]) % 1000000
            
            diverse_factory = VariableSCMFactory(
                seed=scm_seed,  # Different seed for each SCM
                noise_scale=1.0,
                coefficient_range=(-2.0, 2.0),
                vary_intervention_ranges=True,
                use_output_bounds=True
            )
            scm = create_scm_with_factory(config, diverse_factory, target_key)
            scm_batch.append(scm)
        
        # Train on batch with vectorized processing
        rng_key, train_key = random.split(rng_key)
        
        # Time the vectorized approach
        batch_start_time = time.time()
        
        # Pass BC reference params and KL weight if BC warm-up was done
        kl_weight = args.kl_weight if (args.bc_warm_up and bc_reference_params is not None) else 0.0
        
        surrogate_params, opt_state, avg_loss, avg_f1, avg_auroc, avg_auprc, batch_metrics = train_batch_vectorized(
            scm_batch, policy_net, policy_params,
            surrogate_net, surrogate_params,
            optimizer, opt_state,
            args.min_datapoints,
            args.max_datapoints,
            args.min_obs_ratio,
            args.max_obs_ratio,
            train_key,
            not args.use_weighted_loss,  # Invert flag: --use-weighted-loss now disables weighting
            bc_reference_params,
            kl_weight
        )
        batch_time = time.time() - batch_start_time
        
        # Log metrics
        if step % args.log_freq == 0:
            # Group metrics by size range
            size_ranges = [(2, 10), (11, 30), (31, 50), (51, 80), (81, 100)]
            size_metrics = {f"{r[0]}-{r[1]}": [] for r in size_ranges}
            
            for m in batch_metrics:
                num_vars = m['num_vars']
                for r in size_ranges:
                    if r[0] <= num_vars <= r[1]:
                        size_metrics[f"{r[0]}-{r[1]}"].append(m['f1'])
                        break
            
            # Calculate data distribution statistics
            avg_datapoints = np.mean([m['total_datapoints'] for m in batch_metrics])
            avg_obs_ratio = np.mean([m['obs_ratio'] for m in batch_metrics])
            
            print(f"\nStep {step}/{args.num_steps}")
            print(f"  Avg Loss: {avg_loss:.4f}, Avg F1: {avg_f1:.4f}")
            print(f"  Avg AUROC: {avg_auroc:.4f}, Avg AUPRC: {avg_auprc:.4f}")
            print(f"  Batch time: {batch_time:.3f}s (num_vars={num_vars_batch})")
            print(f"  Avg datapoints: {avg_datapoints:.0f} (obs ratio: {avg_obs_ratio:.1%})")
            print("  F1 by size:")
            for range_str, f1_list in size_metrics.items():
                if f1_list:
                    print(f"    {range_str}: {np.mean(f1_list):.3f}")
        
        # Save checkpoint
        if step % args.save_freq == 0 or avg_f1 > best_f1:
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                checkpoint_path = checkpoint_dir / 'best_model.pkl'
                print(f"  New best F1: {best_f1:.4f}")
            else:
                checkpoint_path = checkpoint_dir / f'checkpoint_step_{step}.pkl'
            
            
            # Save using standardized checkpoint format
            save_checkpoint(
                path=checkpoint_path,
                params=surrogate_params,
                architecture={
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'num_heads': args.num_heads,
                    'key_size': args.key_size,
                    'dropout': args.dropout,
                    'encoder_type': 'node_feature'
                },
                model_type='surrogate',
                model_subtype='continuous_parent_set',
                training_config={
                    'learning_rate': args.lr,
                    'batch_size': args.batch_size,
                    'max_vars': args.max_vars,
                    'min_vars': args.min_vars,
                    'total_steps': args.num_steps
                },
                metadata={
                    'step': step,
                    'avg_f1': float(avg_f1),
                    'best_f1': float(best_f1),
                    'dataset': 'avici_style_diverse'
                },
                metrics=batch_metrics,
                optimizer_state=opt_state  # Save optimizer state for continual learning
            )
        
        metrics_history.append({
            'step': step,
            'avg_loss': float(avg_loss),
            'avg_f1': float(avg_f1),
            'avg_auroc': float(avg_auroc),
            'avg_auprc': float(avg_auprc),
            'batch_metrics': batch_metrics
        })
    
    # Save final checkpoint if explicit path specified
    if args.checkpoint_output:
        final_checkpoint_path = Path(args.checkpoint_output)
        
        
        save_checkpoint(
            path=final_checkpoint_path,
            params=surrogate_params,
            architecture={
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'num_heads': args.num_heads,
                'key_size': args.key_size,
                'dropout': args.dropout,
                'encoder_type': 'node_feature'
            },
            model_type='surrogate',
            model_subtype='continuous_parent_set',
            training_config={
                'learning_rate': args.lr,
                'batch_size': args.batch_size,
                'max_vars': args.max_vars,
                'min_vars': args.min_vars,
                'total_steps': step
            },
            metadata={
                'step': step,
                'avg_f1': float(avg_f1) if 'avg_f1' in locals() else 0.0,
                'best_f1': float(best_f1),
                'dataset': 'avici_style_diverse'
            },
            optimizer_state=opt_state  # Save optimizer state for continual learning
        )
        print(f"Saved final checkpoint to: {final_checkpoint_path}")
    
    # Save final metrics
    with open(checkpoint_dir / 'metrics_history.json', 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    print("\n" + "="*70)
    print(f"Training complete! Best F1: {best_f1:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("="*70)


if __name__ == '__main__':
    main()