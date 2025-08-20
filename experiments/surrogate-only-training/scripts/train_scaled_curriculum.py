#!/usr/bin/env python3
"""Train surrogate model with curriculum learning on progressively harder SCMs up to 100 variables."""

import sys
import os
import json
import pickle
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
import optax
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from optimizer_utils import create_adaptive_optimizer, create_curriculum_optimizer_config
from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
from src.causal_bayes_opt.data_structures.scm import get_parents, get_variables
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.data_structures.sample import create_sample
from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
from src.causal_bayes_opt.environments.sampling import sample_with_intervention
from src.causal_bayes_opt.avici_integration.continuous.model import ContinuousParentSetPredictionModel
from src.causal_bayes_opt.policies.permutation_invariant_alternating_policy import create_permutation_invariant_alternating_policy
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
from src.causal_bayes_opt.utils.variable_mapping import VariableMapper


class ScaledCheckpointManager:
    """Manage checkpoints for scaled training up to 100 variables."""
    
    def __init__(self, run_name: Optional[str] = None, max_vars: int = 100):
        self.run_name = run_name or f"run_100var_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = Path("checkpoints/runs") / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.max_vars = max_vars
        self.metrics_history = []
        
        # Create subdirectories
        (self.run_dir / "best_per_size").mkdir(exist_ok=True)
        
    def save_config(self, config: Dict):
        """Save training configuration."""
        with open(self.run_dir / "config.yaml", 'w') as f:
            yaml.dump(config, f)
    
    def save_stage_checkpoint(self, params, opt_state, stage_info: Dict):
        """Save checkpoint after completing a curriculum stage."""
        checkpoint = {
            'params': params,
            'opt_state': opt_state,
            'stage': stage_info['stage'],
            'scms_completed': stage_info['scms_completed'],
            'total_interventions': stage_info['total_interventions'],
            'max_vars_solved': stage_info['max_vars_solved'],
            'timestamp': datetime.now().isoformat(),
            'run_name': self.run_name
        }
        path = self.run_dir / f"stage_{stage_info['stage']}_complete.pkl"
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"  💾 Saved stage {stage_info['stage']} checkpoint")
        return path
    
    def save_best_for_size(self, params, metrics: Dict, num_vars: int):
        """Save best model for each size category."""
        size_dir = self.run_dir / "best_per_size"
        path = size_dir / f"best_{num_vars}var.pkl"
        
        # Check if better than existing
        if path.exists():
            with open(path, 'rb') as f:
                existing = pickle.load(f)
            if metrics.get('f1', 0) <= existing.get('metrics', {}).get('f1', 0):
                return False
        
        checkpoint = {
            'params': params,
            'metrics': metrics,
            'num_vars': num_vars,
            'timestamp': datetime.now().isoformat()
        }
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"  ⭐ New best for {num_vars} variables: F1={metrics.get('f1', 0):.3f}")
        return True
    
    def save_final_model(self, params, opt_state, training_log: Dict):
        """Save final trained model."""
        checkpoint = {
            'params': params,
            'opt_state': opt_state,
            'training_log': training_log,
            'timestamp': datetime.now().isoformat(),
            'run_name': self.run_name
        }
        path = self.run_dir / "final_model.pkl"
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Also save to production if this is the best
        prod_dir = Path("checkpoints/production")
        prod_dir.mkdir(exist_ok=True)
        shutil.copy(path, prod_dir / f"surrogate_100var_{self.run_name}.pkl")
        return path
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load a checkpoint for resuming training."""
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    
    def save_metrics(self, metrics: Dict):
        """Save training metrics."""
        self.metrics_history.append(metrics)
        with open(self.run_dir / "training_metrics.json", 'w') as f:
            json.dump(self.metrics_history, f, indent=2)


def generate_diverse_graph_batch(rng_key, batch_size: int = 32, 
                                min_vars: int = 2, max_vars: int = 100) -> List[Dict]:
    """
    Generate diverse graph types similar to AVICI's training data.
    
    Includes:
    - Erdos-Renyi with different edge densities
    - Scale-free (preferential attachment)
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


def get_model_config(model_size: str) -> Dict:
    """Get model configuration - using AVICI's proven sizes."""
    configs = {
        'avici': {  # AVICI's configuration that scales to 100+ vars
            'hidden_dim': 128,
            'num_layers': 8,  # Will be doubled in alternating attention
            'num_heads': 8,
            'key_size': 32,
            'dropout': 0.1,
            'widening_factor': 4
        },
        'small': {  # For quick testing
            'hidden_dim': 64,
            'num_layers': 4,
            'num_heads': 4,
            'key_size': 16,
            'dropout': 0.1,
            'widening_factor': 4
        },
        'large': {  # Only if AVICI size doesn't work
            'hidden_dim': 256,
            'num_layers': 8,
            'num_heads': 8,
            'key_size': 32,
            'dropout': 0.1,
            'widening_factor': 4
        }
    }
    return configs[model_size]


def get_training_params(num_vars: int) -> Dict:
    """Get adaptive training parameters based on SCM size."""
    if num_vars <= 20:
        return {
            'max_interventions': 50,
            'target_f1': 0.95,
            'gradient_steps': 3
        }
    elif num_vars <= 50:
        return {
            'max_interventions': 100,
            'target_f1': 0.98,
            'gradient_steps': 5
        }
    elif num_vars <= 80:
        return {
            'max_interventions': 200,
            'target_f1': 0.99,
            'gradient_steps': 8
        }
    else:  # 80-100 vars
        return {
            'max_interventions': 300,
            'target_f1': 0.99,
            'gradient_steps': 10
        }


def initialize_models(model_config: Dict, key: jax.random.PRNGKey, max_vars: int) -> Tuple:
    """Initialize policy and surrogate models for maximum variable count."""
    # Create dummy data for initialization
    dummy_buffer = ExperienceBuffer()
    dummy_values = {f'X{i}': 0.0 for i in range(max_vars)}
    dummy_sample = create_sample(dummy_values, intervention_type=None)
    dummy_buffer.add_observation(dummy_sample)
    dummy_tensor, _ = buffer_to_three_channel_tensor(dummy_buffer, 'X0')
    
    # Initialize policy
    policy_key, surrogate_key = random.split(key)
    policy_fn = create_permutation_invariant_alternating_policy(model_config['hidden_dim'])
    policy_net = hk.without_apply_rng(hk.transform(policy_fn))
    policy_params = policy_net.init(policy_key, dummy_tensor, 0)
    
    # Initialize surrogate
    def surrogate_fn(x, target_idx, is_training):
        model = ContinuousParentSetPredictionModel(
            hidden_dim=model_config['hidden_dim'],
            num_heads=model_config.get('num_heads', 8),
            num_layers=model_config['num_layers'],
            key_size=model_config.get('key_size', 32),
            dropout=model_config['dropout']
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
    
    labels = jnp.array(labels)
    mask = jnp.ones(len(variables), dtype=bool).at[target_idx].set(False)
    pred_probs_filtered = pred_probs[mask]
    
    # BCE loss
    loss = -jnp.mean(
        labels * jnp.log(pred_probs_filtered) + 
        (1 - labels) * jnp.log(1 - pred_probs_filtered)
    )
    
    return loss, pred_probs


def evaluate_predictions(pred_probs, true_parents, variables, target_idx, threshold=0.5):
    """Evaluate surrogate predictions and compute F1 score."""
    predicted_parents = set()
    for i, var in enumerate(variables):
        if i != target_idx and pred_probs[i] > threshold:
            predicted_parents.add(var)
    
    true_parent_set = set(true_parents)
    
    if len(true_parent_set) == 0 and len(predicted_parents) == 0:
        return 1.0
    elif len(true_parent_set) == 0 or len(predicted_parents) == 0:
        return 0.0
    
    tp = len(predicted_parents & true_parent_set)
    fp = len(predicted_parents - true_parent_set)
    fn = len(true_parent_set - predicted_parents)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1


def train_on_scm(scm, policy_net, policy_params, surrogate_net, surrogate_params,
                 optimizer, opt_state, training_params, rng_key):
    """Train surrogate on a single SCM until target F1 or max interventions."""
    
    # Get SCM info
    mapper = VariableMapper(list(get_variables(scm)))
    variables = mapper.variables
    num_vars = len(variables)
    
    # Initialize buffer with observations
    buffer = ExperienceBuffer()
    rng_key, obs_key = random.split(rng_key)
    obs_data = sample_from_linear_scm(scm, 20, seed=int(obs_key[0]))
    for sample in obs_data:
        buffer.add_observation(sample)
    
    interventions_count = 0
    f1_history = []
    
    for intervention_idx in range(training_params['max_interventions']):
        interventions_count += 1
        
        # Progress indicator
        if intervention_idx == 0:
            print(f"    Training (max {training_params['max_interventions']} interventions, target F1={training_params['target_f1']})...")
        
        # Select intervention using policy
        rng_key, target_key, select_key = random.split(rng_key, 3)
        target_idx = random.choice(target_key, num_vars)
        target_var = variables[int(target_idx)]
        
        tensor, _ = buffer_to_three_channel_tensor(buffer, target_var)
        output = policy_net.apply(policy_params, tensor, int(target_idx))
        
        # Random selection for diversity
        valid_mask = jnp.ones(num_vars).at[int(target_idx)].set(0)
        valid_indices = jnp.where(valid_mask)[0]
        selected_idx = random.choice(select_key, valid_indices)
        selected_var = variables[int(selected_idx)]
        
        # Perform intervention
        rng_key, int_key, post_key = random.split(rng_key, 3)
        intervention_value = random.normal(int_key) * 2.0
        
        intervention = create_perfect_intervention(
            targets=frozenset([selected_var]),
            values={selected_var: float(intervention_value)}
        )
        
        post_data = sample_with_intervention(scm, intervention, 1, seed=int(post_key[0]))
        if post_data:
            buffer.add_intervention(intervention, post_data[0])
        
        # Train surrogate on all variables
        all_f1_scores = []
        
        for target_var in variables:
            target_idx = mapper.get_index(target_var)
            true_parents = list(get_parents(scm, target_var))
            
            # Multiple gradient steps
            for _ in range(training_params['gradient_steps']):
                rng_key, loss_key = random.split(rng_key)
                
                def loss_fn(params):
                    loss, pred_probs = compute_surrogate_loss(
                        params, surrogate_net, buffer, target_idx, target_var,
                        true_parents, variables, loss_key
                    )
                    return loss, pred_probs
                
                (loss, pred_probs), grads = jax.value_and_grad(loss_fn, has_aux=True)(surrogate_params)
                updates, opt_state = optimizer.update(grads, opt_state, surrogate_params)
                surrogate_params = optax.apply_updates(surrogate_params, updates)
            
            # Evaluate F1 for this variable
            f1 = evaluate_predictions(pred_probs, true_parents, variables, target_idx)
            all_f1_scores.append(f1)
        
        # Average F1 across all variables
        avg_f1 = np.mean(all_f1_scores)
        f1_history.append(avg_f1)
        
        # Check if we've achieved target F1
        if avg_f1 >= training_params['target_f1']:
            print(f"    ✓ Achieved F1={avg_f1:.3f} after {interventions_count} interventions")
            return surrogate_params, opt_state, True, interventions_count, f1_history
        
        # Log progress periodically
        if intervention_idx % 10 == 0 or intervention_idx < 5:
            print(f"      Intervention {intervention_idx+1}: F1={avg_f1:.3f}")
    
    print(f"    ✗ Max interventions reached. Final F1={f1_history[-1]:.3f}")
    return surrogate_params, opt_state, False, interventions_count, f1_history


def main():
    parser = argparse.ArgumentParser(description='Train surrogate with scaled curriculum learning')
    parser.add_argument('--model-size', type=str, default='avici',
                       choices=['avici', 'small', 'large'],
                       help='Model size configuration')
    parser.add_argument('--lr', type=float, default=0.0003,
                       help='Learning rate')
    parser.add_argument('--stages', type=str, default='1-5',
                       help='Stages to train (e.g., "1-5" or "3-5")')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Name for this training run')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("SCALED CURRICULUM LEARNING FOR SURROGATE MODEL (100 VARIABLES)")
    print("="*70)
    
    # Parse stages
    if '-' in args.stages:
        start_stage, end_stage = map(int, args.stages.split('-'))
    else:
        start_stage = end_stage = int(args.stages)
    
    # Get model configuration
    model_config = get_model_config(args.model_size)
    
    # Configuration
    config = {
        'model_size': args.model_size,
        'model_config': model_config,
        'learning_rate': args.lr,
        'stages': args.stages,
        'seed': args.seed
    }
    
    print("\nConfiguration:")
    print(f"  Model size: {args.model_size}")
    print(f"  Hidden dim: {model_config['hidden_dim']}")
    print(f"  Num layers: {model_config['num_layers']}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Stages: {args.stages}")
    
    # Initialize checkpoint manager
    checkpoint_manager = ScaledCheckpointManager(run_name=args.run_name)
    checkpoint_manager.save_config(config)
    print(f"\nRun name: {checkpoint_manager.run_name}")
    
    # Initialize RNG
    rng_key = random.PRNGKey(config['seed'])
    
    # Create curriculum
    full_curriculum = create_scaled_curriculum()
    
    # Filter curriculum by stages
    curriculum = [scm for scm in full_curriculum if start_stage <= scm['stage'] <= end_stage]
    print(f"\nCurriculum: {len(curriculum)} SCMs (stages {start_stage}-{end_stage})")
    
    # Find max variables for model initialization
    max_vars = max(scm_cfg['num_vars'] for scm_cfg in full_curriculum)
    print(f"Maximum variables: {max_vars}")
    
    # Initialize or resume
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = checkpoint_manager.load_checkpoint(args.resume)
        surrogate_params = checkpoint['params']
        opt_state = checkpoint['opt_state']
        start_from = checkpoint.get('scms_completed', 0)
        total_interventions = checkpoint.get('total_interventions', 0)
    else:
        # Initialize models
        rng_key, init_key = random.split(rng_key)
        policy_net, policy_params, surrogate_net, surrogate_params = initialize_models(
            model_config, init_key, max_vars
        )
        
        # Initialize optimizer with smart scheduling
        optimizer_config = create_curriculum_optimizer_config(
            model_size=args.model_size,
            max_stages=end_stage,
            estimated_steps=len(curriculum) * 100  # Rough estimate
        )
        optimizer_config['learning_rate'] = config['learning_rate']  # Override with user's LR if specified
        
        optimizer = create_adaptive_optimizer(
            config=optimizer_config,
            num_training_steps=len(curriculum) * 100,
            use_curriculum_aware=True
        )
        opt_state = optimizer.init(surrogate_params)
        
        # Save optimizer config for resume
        config['optimizer_config'] = optimizer_config
        
        start_from = 0
        total_interventions = 0
        
        # Save initial model
        initial_checkpoint = {
            'params': surrogate_params,
            'opt_state': opt_state,
            'config': config,
            'trained_on': []
        }
        with open(checkpoint_manager.run_dir / 'initial_model.pkl', 'wb') as f:
            pickle.dump(initial_checkpoint, f)
        print(f"Saved initial model")
    
    # Training loop
    print("\n" + "-"*70)
    print("Starting Curriculum Training")
    print("-"*70)
    
    training_log = {
        'config': config,
        'curriculum': curriculum,
        'results': [],
        'total_interventions': total_interventions,
        'completed_scms': 0
    }
    
    factory = VariableSCMFactory(seed=config['seed'])
    current_stage = start_stage
    stage_info = {'stage': current_stage, 'scms_completed': 0, 'total_interventions': 0, 'max_vars_solved': 0}
    
    for i, scm_config in enumerate(curriculum[start_from:], start=start_from+1):
        # Check if we're entering a new stage
        if scm_config['stage'] > current_stage:
            # Save stage checkpoint
            checkpoint_manager.save_stage_checkpoint(
                surrogate_params, opt_state, stage_info
            )
            current_stage = scm_config['stage']
            stage_info = {'stage': current_stage, 'scms_completed': 0, 'total_interventions': 0, 'max_vars_solved': 0}
        
        replay_tag = " [REPLAY]" if scm_config.get('replay', False) else ""
        print(f"\nSCM {i}/{len(curriculum)}: {scm_config['structure']} with {scm_config['num_vars']} variables (Stage {scm_config['stage']}){replay_tag}")
        
        # Create SCM
        scm = factory.create_variable_scm(
            num_variables=scm_config['num_vars'],
            structure_type=scm_config['structure'],
            edge_density=scm_config.get('edge_density', 0.5)
        )
        
        # Get adaptive training parameters
        training_params = get_training_params(scm_config['num_vars'])
        
        # Train on this SCM
        rng_key, train_key = random.split(rng_key)
        start_time = datetime.now()
        
        surrogate_params, opt_state, success, interventions, f1_history = train_on_scm(
            scm, policy_net, policy_params, surrogate_net, surrogate_params,
            optimizer, opt_state, training_params, train_key
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Log results
        result = {
            'scm_index': i,
            'scm_config': scm_config,
            'success': success,
            'interventions': interventions,
            'final_f1': f1_history[-1] if f1_history else 0,
            'f1_history': f1_history,
            'time_seconds': elapsed
        }
        training_log['results'].append(result)
        training_log['total_interventions'] += interventions
        
        if success:
            training_log['completed_scms'] += 1
            stage_info['scms_completed'] += 1
            stage_info['total_interventions'] += interventions
            stage_info['max_vars_solved'] = max(stage_info['max_vars_solved'], scm_config['num_vars'])
            
            # Save best model for this size
            checkpoint_manager.save_best_for_size(
                surrogate_params,
                {'f1': f1_history[-1], 'interventions': interventions},
                scm_config['num_vars']
            )
        
        # Save metrics
        checkpoint_manager.save_metrics(result)
        
        # Early stopping if struggling with large SCMs
        if not success and scm_config['num_vars'] > 80 and not scm_config.get('replay', False):
            print(f"\nStruggling with large SCMs. Consider stopping or adjusting parameters.")
    
    # Save final stage checkpoint
    checkpoint_manager.save_stage_checkpoint(
        surrogate_params, opt_state, stage_info
    )
    
    # Save final model
    final_path = checkpoint_manager.save_final_model(
        surrogate_params, opt_state, training_log
    )
    print(f"\nSaved final model to {final_path}")
    
    # Save training log
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = results_dir / f'scaled_curriculum_training_{timestamp}.json'
    
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"Saved training log to {log_path}")
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Completed SCMs: {training_log['completed_scms']}/{len(curriculum)}")
    print(f"Total interventions: {training_log['total_interventions']}")
    
    if training_log['results']:
        successful = [r for r in training_log['results'] if r['success']]
        if successful:
            avg_interventions = np.mean([r['interventions'] for r in successful])
            print(f"Average interventions per success: {avg_interventions:.1f}")
            
            largest_solved = max(successful, key=lambda r: r['scm_config']['num_vars'])
            print(f"Largest SCM solved: {largest_solved['scm_config']['num_vars']} variables")
            
            # Performance by structure
            by_structure = {}
            for r in successful:
                struct = r['scm_config']['structure']
                if struct not in by_structure:
                    by_structure[struct] = []
                by_structure[struct].append(r['interventions'])
            
            print("\nInterventions by structure type:")
            for struct, interventions in sorted(by_structure.items()):
                print(f"  {struct}: {np.mean(interventions):.1f} ± {np.std(interventions):.1f}")


if __name__ == '__main__':
    main()