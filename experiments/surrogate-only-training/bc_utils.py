#!/usr/bin/env python3
"""Behavioral Cloning utilities for AVICI-style training with expert demonstrations."""

import jax
import jax.numpy as jnp
import jax.random as random
import optax
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging

# Import from the main package
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.training.data_preprocessing import (
    SurrogateTrainingData,
    load_demonstrations_from_path,
    preprocess_demonstration_batch,
    extract_surrogate_training_data
)

logger = logging.getLogger(__name__)


def load_expert_demonstrations(demo_path: str, max_demos: Optional[int] = None) -> List[SurrogateTrainingData]:
    """
    Load and preprocess expert demonstrations for BC training.
    
    Args:
        demo_path: Path to expert demonstrations directory
        max_demos: Maximum number of demonstrations to load (for testing)
        
    Returns:
        List of SurrogateTrainingData objects ready for BC training
    """
    logger.info(f"Loading expert demonstrations from {demo_path}")
    
    # Load raw demonstrations
    raw_demos = load_demonstrations_from_path(demo_path, max_files=max_demos)
    logger.info(f"Loaded {len(raw_demos)} demonstration files")
    
    # Preprocess to extract surrogate training data
    surrogate_data = []
    
    for demo in raw_demos:
        try:
            # Process each demonstration batch
            preprocessed = preprocess_demonstration_batch([demo])
            if preprocessed['surrogate_data']:
                surrogate_data.extend(preprocessed['surrogate_data'])
        except Exception as e:
            logger.warning(f"Failed to preprocess demonstration: {e}")
            continue
    
    logger.info(f"Extracted {len(surrogate_data)} training examples from demonstrations")
    return surrogate_data


def compute_bc_loss(pred_probs: jnp.ndarray, 
                   expert_data: SurrogateTrainingData,
                   use_weighted_loss: bool = False) -> jnp.ndarray:
    """
    Compute behavioral cloning loss against expert demonstrations.
    
    Args:
        pred_probs: Predicted parent probabilities [d]
        expert_data: Expert demonstration with marginal parent probabilities
        use_weighted_loss: Whether to weight positive examples more
        
    Returns:
        Scalar BC loss value
    """
    # Get ground truth marginal probabilities
    true_probs = jnp.zeros(len(expert_data.variables))
    for i, var in enumerate(expert_data.variables):
        if var in expert_data.marginal_parent_probs:
            true_probs = true_probs.at[i].set(
                expert_data.marginal_parent_probs[var]
            )
    
    # Mask out target variable (can't be its own parent)
    mask = jnp.ones(len(expert_data.variables))
    mask = mask.at[expert_data.target_idx].set(0.0)
    
    # Clip probabilities to avoid log(0)
    pred_probs = jnp.clip(pred_probs, 1e-7, 1 - 1e-7)
    
    # Binary cross-entropy loss
    if use_weighted_loss:
        # Weight positive examples more (assuming sparse graphs)
        pos_weight = 4.0  # Assuming ~20% edges present
        bce = -(pos_weight * true_probs * jnp.log(pred_probs) + 
                (1 - true_probs) * jnp.log(1 - pred_probs))
    else:
        bce = -(true_probs * jnp.log(pred_probs) + 
                (1 - true_probs) * jnp.log(1 - pred_probs))
    
    # Apply mask and normalize
    masked_bce = bce * mask
    loss = jnp.sum(masked_bce) / jnp.sum(mask)
    
    return loss


def compute_kl_divergence(current_probs: jnp.ndarray,
                         reference_probs: jnp.ndarray,
                         mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """
    Compute KL divergence between current and reference distributions.
    
    KL(current || reference) = sum(current * log(current/reference))
    
    Args:
        current_probs: Current model predictions [d]
        reference_probs: Reference model predictions (from BC) [d]
        mask: Optional mask for variables to include [d]
        
    Returns:
        Scalar KL divergence value
    """
    # Clip probabilities to avoid numerical issues
    current_probs = jnp.clip(current_probs, 1e-7, 1 - 1e-7)
    reference_probs = jnp.clip(reference_probs, 1e-7, 1 - 1e-7)
    
    # Compute KL for Bernoulli distributions
    # For each variable: KL = p*log(p/q) + (1-p)*log((1-p)/(1-q))
    kl_pos = current_probs * jnp.log(current_probs / reference_probs)
    kl_neg = (1 - current_probs) * jnp.log((1 - current_probs) / (1 - reference_probs))
    kl = kl_pos + kl_neg
    
    # Apply mask if provided
    if mask is not None:
        kl = kl * mask
        return jnp.sum(kl) / jnp.sum(mask)
    else:
        return jnp.mean(kl)


def run_bc_warmup(surrogate_net, surrogate_params, expert_demos: List[SurrogateTrainingData],
                  optimizer, opt_state, bc_steps: int, batch_size: int, 
                  rng_key: jax.random.PRNGKey, use_weighted_loss: bool = False) -> Tuple:
    """
    Run behavioral cloning warm-up phase.
    
    Args:
        surrogate_net: Surrogate model network
        surrogate_params: Initial model parameters
        expert_demos: List of expert demonstration data
        optimizer: Optimizer instance
        opt_state: Optimizer state
        bc_steps: Number of BC training steps
        batch_size: Batch size for BC training
        rng_key: Random key
        use_weighted_loss: Whether to use weighted BCE loss
        
    Returns:
        Updated (params, opt_state, training_metrics)
    """
    logger.info(f"Starting BC warm-up for {bc_steps} steps")
    
    bc_losses = []
    
    for step in range(bc_steps):
        # Sample batch of expert demonstrations
        rng_key, batch_key = random.split(rng_key)
        batch_indices = random.choice(batch_key, len(expert_demos), 
                                    shape=(min(batch_size, len(expert_demos)),),
                                    replace=True)
        batch = [expert_demos[i] for i in batch_indices]
        
        # Define loss function for this batch
        def loss_fn(params):
            total_loss = 0.0
            
            for expert_data in batch:
                # Forward pass
                rng_key_forward = random.PRNGKey(0)  # Deterministic for gradient
                output = surrogate_net.apply(
                    params, rng_key_forward,
                    expert_data.state_tensor,
                    expert_data.target_idx,
                    True  # is_training
                )
                
                # Extract predicted probabilities
                if 'parent_probabilities' in output:
                    pred_probs = output['parent_probabilities']
                else:
                    # Fallback to sigmoid of logits
                    raw_logits = output.get('attention_logits', jnp.zeros(len(expert_data.variables)))
                    pred_probs = jax.nn.sigmoid(raw_logits)
                
                # Compute BC loss
                bc_loss = compute_bc_loss(pred_probs, expert_data, use_weighted_loss)
                total_loss += bc_loss
            
            return total_loss / len(batch)
        
        # Compute gradients and update
        loss_val, grads = jax.value_and_grad(loss_fn)(surrogate_params)
        updates, opt_state = optimizer.update(grads, opt_state, surrogate_params)
        surrogate_params = optax.apply_updates(surrogate_params, updates)
        
        bc_losses.append(float(loss_val))
        
        # Log progress
        if step % 50 == 0:
            logger.info(f"  BC Step {step}/{bc_steps}: Loss = {loss_val:.4f}")
    
    # Compute final metrics
    avg_loss = sum(bc_losses[-10:]) / min(10, len(bc_losses))
    logger.info(f"BC warm-up complete. Final avg loss: {avg_loss:.4f}")
    
    metrics = {
        'bc_losses': bc_losses,
        'final_bc_loss': bc_losses[-1] if bc_losses else 0.0,
        'avg_final_bc_loss': avg_loss
    }
    
    return surrogate_params, opt_state, metrics


def compute_kl_regularization(current_params, reference_params, surrogate_net,
                             buffer, target_var, target_idx, variables,
                             rng_key: jax.random.PRNGKey) -> jnp.ndarray:
    """
    Compute KL regularization term between current and reference model predictions.
    
    Args:
        current_params: Current model parameters
        reference_params: Reference model parameters (from BC)
        surrogate_net: Model network
        buffer: Experience buffer with data
        target_var: Target variable name
        target_idx: Target variable index
        variables: List of all variables
        rng_key: Random key
        
    Returns:
        Scalar KL divergence value
    """
    from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
    
    # Convert buffer to tensor
    tensor, _ = buffer_to_three_channel_tensor(buffer, target_var)
    
    # Get predictions from both models
    current_output = surrogate_net.apply(
        current_params, rng_key, tensor, target_idx, False
    )
    reference_output = surrogate_net.apply(
        reference_params, rng_key, tensor, target_idx, False
    )
    
    # Extract probabilities
    if 'parent_probabilities' in current_output:
        current_probs = current_output['parent_probabilities']
    else:
        current_probs = jax.nn.sigmoid(current_output.get('attention_logits', jnp.zeros(len(variables))))
    
    if 'parent_probabilities' in reference_output:
        reference_probs = reference_output['parent_probabilities']
    else:
        reference_probs = jax.nn.sigmoid(reference_output.get('attention_logits', jnp.zeros(len(variables))))
    
    # Create mask (exclude target variable)
    mask = jnp.ones(len(variables))
    mask = mask.at[target_idx].set(0.0)
    
    # Compute KL divergence
    kl = compute_kl_divergence(current_probs, reference_probs, mask)
    
    return kl