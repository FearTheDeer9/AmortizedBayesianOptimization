#!/usr/bin/env python3
"""Train surrogate model with AVICI-style diverse graph generation."""

import sys
import os
import json
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
from src.causal_bayes_opt.data_structures.scm import get_parents, get_variables, get_target, create_scm
from src.causal_bayes_opt.mechanisms.linear import create_linear_mechanism
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


def generate_diverse_graph_batch(rng_key, batch_size: int = 32, 
                                min_vars: int = 3, max_vars: int = 100) -> List[Dict]:
    """
    Generate diverse graph types similar to AVICI's training data.
    
    Includes:
    - Erdos-Renyi with different edge densities
    - Chain structures
    - Fork/Collider structures
    - Mixed patterns
    """
    graphs = []
    
    for i in range(batch_size):
        rng_key, subkey = random.split(rng_key)
        
        # Sample number of variables uniformly
        num_vars = int(random.uniform(subkey, minval=min_vars, maxval=max_vars + 1))
        
        # Sample graph type
        rng_key, type_key = random.split(rng_key)
        graph_type_idx = random.choice(type_key, 5)
        
        if graph_type_idx == 0:
            # Erdos-Renyi with varying edge density (1-3 edges per var)
            rng_key, density_key = random.split(rng_key)
            edges_per_var = random.uniform(density_key, minval=1.0, maxval=3.0)
            edge_density = min(edges_per_var / (num_vars - 1), 0.5)  # Cap at 0.5
            structure = 'random'
        elif graph_type_idx == 1:
            # Chain structure (hardest for causal discovery)
            structure = 'chain'
            edge_density = 1.0 / (num_vars - 1) if num_vars > 1 else 0.0
        elif graph_type_idx == 2:
            # Fork structure (common cause)
            structure = 'fork'
            edge_density = 0.3
        elif graph_type_idx == 3:
            # Collider structure (common effect)
            structure = 'collider'
            edge_density = 0.3
        else:
            # Mixed structure
            structure = 'mixed'
            edge_density = 0.25
        
        graphs.append({
            'num_vars': num_vars,
            'structure': structure,
            'edge_density': edge_density
        })
    
    return graphs


def create_scm_from_config(config: Dict, rng_key) -> pyr.PMap:
    """Create SCM using unified create_scm() interface instead of VariableSCMFactory."""
    num_vars = config['num_vars']
    structure = config['structure']
    edge_density = config['edge_density']
    
    # Generate variable names
    variables = frozenset([f'X{i}' for i in range(num_vars)])
    
    # Create mechanisms for each variable
    mechanisms = {}
    edges = set()
    
    if structure == 'chain':
        # Chain: X0 -> X1 -> X2 -> ... -> X(n-1)
        for i in range(num_vars):
            if i == 0:
                # Root variable
                mechanisms[f'X{i}'] = create_linear_mechanism(
                    parents=[], coefficients={}, intercept=0.0, noise_scale=1.0
                )
            else:
                # Chain variable
                parent = f'X{i-1}'
                edges.add((parent, f'X{i}'))
                mechanisms[f'X{i}'] = create_linear_mechanism(
                    parents=[parent], 
                    coefficients={parent: 1.0}, 
                    intercept=0.0, 
                    noise_scale=1.0
                )
    
    elif structure == 'fork':
        # Fork: X0 -> X1, X0 -> X2, X0 -> X3, ...
        for i in range(num_vars):
            if i == 0:
                # Root (common cause)
                mechanisms[f'X{i}'] = create_linear_mechanism(
                    parents=[], coefficients={}, intercept=0.0, noise_scale=1.0
                )
            else:
                # Fork children
                parent = 'X0'
                edges.add((parent, f'X{i}'))
                mechanisms[f'X{i}'] = create_linear_mechanism(
                    parents=[parent],
                    coefficients={parent: 1.0},
                    intercept=0.0,
                    noise_scale=1.0
                )
    
    elif structure == 'collider':
        # Collider: X0 -> X(n-1), X1 -> X(n-1), ..., X(n-2) -> X(n-1)
        for i in range(num_vars):
            if i == num_vars - 1:
                # Collider (common effect)
                parents = [f'X{j}' for j in range(num_vars - 1)]
                for parent in parents:
                    edges.add((parent, f'X{i}'))
                coefficients = {parent: 1.0 for parent in parents}
                mechanisms[f'X{i}'] = create_linear_mechanism(
                    parents=parents,
                    coefficients=coefficients,
                    intercept=0.0,
                    noise_scale=1.0
                )
            else:
                # Root variables
                mechanisms[f'X{i}'] = create_linear_mechanism(
                    parents=[], coefficients={}, intercept=0.0, noise_scale=1.0
                )
    
    else:
        # Random/mixed structure - simplified random graph
        rng_key, edge_key = random.split(rng_key)
        
        # Create mechanisms (all root variables for simplicity)
        for i in range(num_vars):
            mechanisms[f'X{i}'] = create_linear_mechanism(
                parents=[], coefficients={}, intercept=0.0, noise_scale=1.0
            )
        
        # Add a few random edges based on edge_density
        max_edges = int(edge_density * num_vars * (num_vars - 1) / 2)
        edge_count = 0
        
        for i in range(num_vars):
            for j in range(i + 1, num_vars):
                if edge_count >= max_edges:
                    break
                rng_key, add_edge_key = random.split(rng_key)
                if random.uniform(add_edge_key) < edge_density:
                    # Add edge X_i -> X_j
                    parent, child = f'X{i}', f'X{j}'
                    edges.add((parent, child))
                    
                    # Update child mechanism to include parent
                    mechanisms[child] = create_linear_mechanism(
                        parents=[parent],
                        coefficients={parent: 1.0},
                        intercept=0.0,
                        noise_scale=1.0
                    )
                    edge_count += 1
    
    # Choose random target (not X0 to avoid trivial cases)
    target_idx = random.choice(rng_key, num_vars)
    target_var = f'X{target_idx}'
    
    # Create SCM using unified interface
    scm = create_scm(
        variables=variables,
        edges=frozenset(edges),
        mechanisms=pyr.pmap(mechanisms),
        target=target_var,
        metadata=pyr.pmap({
            'structure': structure,
            'edge_density': edge_density,
            'num_variables': num_vars
        })
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
            dropout=dropout
        )
        return model(x, target_idx, is_training)
    
    surrogate_net = hk.transform(surrogate_fn)
    surrogate_params = surrogate_net.init(surrogate_key, dummy_tensor, 0, True)
    
    return policy_net, policy_params, surrogate_net, surrogate_params


def compute_surrogate_loss(params, net, buffer, target_idx, target_var, true_parents, variables, rng_key):
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
    
    # Binary cross-entropy loss
    bce_loss = -(labels * jnp.log(pred_probs) + (1 - labels) * jnp.log(1 - pred_probs))
    return jnp.mean(bce_loss), pred_probs, labels


def train_batch(scm_batch: List, 
               policy_net, policy_params,
               surrogate_net, surrogate_params,
               optimizer, opt_state, 
               num_observations: int,
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
        
        # Generate observations (600 obs + 200 interventions like AVICI)
        buffer = ExperienceBuffer()
        
        # Observational data
        rng_key, obs_key = random.split(rng_key)
        obs_seed = int(obs_key[0]) % 1000000
        samples = sample_from_linear_scm(scm, n_samples=int(num_observations * 0.75), seed=obs_seed)
        for sample in samples:
            buffer.add_observation(sample)
        
        # Interventional data - following pattern from other scripts
        for _ in range(int(num_observations * 0.25)):
            rng_key, action_key, int_key, post_key = random.split(rng_key, 4)
            
            # Random selection for diversity (like other scripts)
            valid_mask = jnp.ones(len(variables)).at[target_idx].set(0)
            valid_indices = jnp.where(valid_mask)[0]
            selected_idx = random.choice(action_key, valid_indices)
            selected_var = mapper.get_name(int(selected_idx))
            
            # Skip if somehow selected the target
            if selected_var == target_var:
                continue
                
            intervention_value = random.normal(int_key) * 2.0
            
            intervention = create_perfect_intervention(
                targets=frozenset([selected_var]),
                values={selected_var: float(intervention_value)}
            )
            
            post_data = sample_with_intervention(scm, intervention, 1, seed=int(post_key[0]))
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
    parser = argparse.ArgumentParser(description='Train surrogate with AVICI-style diverse graphs')
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
                       help='Batch size for training')
    parser.add_argument('--num-observations', type=int,  default=200, help='Number of observations per graph')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--num-steps', type=int, default=5000,
                       help='Number of training steps')
    parser.add_argument('--min-vars', type=int, default=3,
                       help='Minimum number of variables')
    parser.add_argument('--max-vars', type=int, default=100,
                       help='Maximum number of variables')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save-freq', type=int, default=100,
                       help='Save checkpoint every N steps')
    parser.add_argument('--log-freq', type=int, default=100,
                       help='Log metrics every N steps')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("AVICI-STYLE TRAINING WITH DIVERSE GRAPH GENERATION")
    print("="*70)
    
    print("\nConfiguration:")
    print(f"  Model: {args.hidden_dim} hidden, {args.num_layers} layers")
    print(f"  Training: {args.num_steps} steps, batch size {args.batch_size}")
    print(f"  Variables: {args.min_vars}-{args.max_vars}")
    print(f"  Observations: {args.num_observations} per graph")
    print(f"  Learning rate: {args.lr}")
    
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
    
    # Initialize models
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
    
    # Initialize optimizer with cosine schedule
    schedule = optax.cosine_decay_schedule(
        init_value=args.lr,
        decay_steps=args.num_steps,
        alpha=0.1  # Final LR = 0.1 * initial LR
    )
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=0.01)
    opt_state = optimizer.init(surrogate_params)
    
    print(f"\nTraining for {args.num_steps} steps...")
    
    # Training metrics
    metrics_history = []
    best_f1 = 0.0
    
    for step in range(args.num_steps):
        # Generate diverse batch of graphs
        rng_key, batch_key = random.split(rng_key)
        graph_configs = generate_diverse_graph_batch(
            batch_key, args.batch_size, args.min_vars, args.max_vars
        )
        
        # Create SCMs from configs using unified interface
        scm_batch = []
        for config in graph_configs:
            rng_key, scm_key = random.split(rng_key)
            scm = create_scm_from_config(config, scm_key)
            scm_batch.append(scm)
        
        # Train on batch
        rng_key, train_key = random.split(rng_key)
        surrogate_params, opt_state, avg_loss, avg_f1, batch_metrics = train_batch(
            scm_batch, policy_net, policy_params,
            surrogate_net, surrogate_params,
            optimizer, opt_state,
            args.num_observations,
            train_key
        )
        
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
            
            print(f"\nStep {step}/{args.num_steps}")
            print(f"  Avg Loss: {avg_loss:.4f}, Avg F1: {avg_f1:.4f}")
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
                metrics=batch_metrics
            )
        
        metrics_history.append({
            'step': step,
            'avg_loss': float(avg_loss),
            'avg_f1': float(avg_f1),
            'batch_metrics': batch_metrics
        })
    
    # Save final metrics
    with open(checkpoint_dir / 'metrics_history.json', 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    print("\n" + "="*70)
    print(f"Training complete! Best F1: {best_f1:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("="*70)


if __name__ == '__main__':
    main()