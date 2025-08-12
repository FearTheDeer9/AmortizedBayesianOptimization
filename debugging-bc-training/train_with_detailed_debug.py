#!/usr/bin/env python3
"""
Train BC policy with detailed debugging output.

This script provides comprehensive debugging information to understand
why training isn't improving, including loss breakdowns, gradient analysis,
and prediction vs expert comparisons.
"""

import sys
from pathlib import Path
import pickle
import time
from datetime import datetime
import numpy as np
import jax
import jax.numpy as jnp
import optax
from typing import Dict, List, Any, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import standard modules
from src.causal_bayes_opt.training.data_preprocessing import load_demonstrations_from_path

# Import our SMOOTHED trainer with label smoothing to prevent overconfidence
from policy_bc_trainer_smoothed import PolicyBCTrainer

# Import our fixed modules with numerical sorting
from demonstration_to_tensor_fixed import create_bc_training_dataset
from variable_mapping_fixed import VariableMapper
from numerical_sort_utils import numerical_sort_variables


def analyze_predictions(
    var_logits: jnp.ndarray,
    target_idx: int,
    variables: List[str],
    example_idx: int
) -> Dict[str, Any]:
    """Analyze model predictions vs target."""
    
    # Get softmax probabilities
    probs = jax.nn.softmax(var_logits)
    
    # Get top 3 predictions
    top_indices = jnp.argsort(probs)[-3:][::-1]
    
    analysis = {
        'example_idx': example_idx,
        'target_variable': variables[target_idx] if target_idx < len(variables) else f"idx_{target_idx}",
        'target_idx': target_idx,
        'target_prob': float(probs[target_idx]),
        'predicted_idx': int(jnp.argmax(probs)),
        'predicted_variable': variables[int(jnp.argmax(probs))],
        'predicted_prob': float(jnp.max(probs)),
        'top_3': [(variables[int(idx)], float(probs[idx])) for idx in top_indices],
        'correct': int(jnp.argmax(probs)) == target_idx,
        'loss': float(-jnp.log(jnp.maximum(probs[target_idx], 1e-10)))
    }
    
    return analysis


def analyze_gradients(grads: Any, params: Any, learning_rate: float) -> Dict[str, float]:
    """Analyze gradient statistics."""
    
    # Calculate gradient norms
    grad_norm_before = optax.global_norm(grads)
    
    # Create gradient clipper and apply it
    clipper = optax.clip_by_global_norm(5.0)
    clipped_grads, _ = clipper.update(grads, None)
    grad_norm_after = optax.global_norm(clipped_grads)
    
    # Calculate parameter norm
    param_norm = optax.global_norm(params)
    
    # Calculate effective learning rate and update magnitude
    effective_lr = float(grad_norm_after * learning_rate)
    relative_update = float(effective_lr / (param_norm + 1e-8))
    
    return {
        'grad_norm_before_clip': float(grad_norm_before),
        'grad_norm_after_clip': float(grad_norm_after),
        'clip_ratio': float(grad_norm_after / (grad_norm_before + 1e-8)),
        'param_norm': float(param_norm),
        'effective_lr': effective_lr,
        'relative_update_pct': relative_update * 100
    }


def debug_loss_calculation(
    var_logits: jnp.ndarray,
    value_params: jnp.ndarray,
    target_idx: int,
    target_value: float,
    variables: List[str]
) -> Dict[str, Any]:
    """Break down loss calculation for debugging."""
    
    # Variable selection loss
    var_probs = jax.nn.softmax(var_logits)
    var_loss = -jnp.log(jnp.maximum(var_probs[target_idx], 1e-10))
    
    # Value prediction loss using robust formulation
    value_mean = value_params[target_idx, 0]
    value_log_std = value_params[target_idx, 1]
    
    # Harmonized clipping
    log_std = jnp.clip(value_log_std, -2.0, 2.0)
    value_std = jnp.exp(log_std)
    
    # Huber-style robust loss
    error = target_value - value_mean
    huber_delta = 1.0
    normalized_error = jnp.abs(error) / value_std
    
    is_small_error = normalized_error <= huber_delta
    quadratic_loss = 0.5 * (error / value_std) ** 2
    linear_loss = huber_delta * (normalized_error - 0.5 * huber_delta)
    
    mse_term = jnp.where(is_small_error, quadratic_loss, linear_loss)
    std_regularization = 0.01 * jnp.exp(-log_std - 2.0)
    
    value_loss = 0.5 * jnp.log(2 * jnp.pi) + log_std + mse_term + std_regularization
    
    return {
        'var_logits': var_logits[:5].tolist(),  # First 5 for display
        'var_probs': var_probs[:5].tolist(),
        'target_variable': variables[target_idx] if target_idx < len(variables) else f"idx_{target_idx}",
        'target_idx': target_idx,
        'target_prob': float(var_probs[target_idx]),
        'var_loss': float(var_loss),
        'target_value': target_value,
        'predicted_mean': float(value_mean),
        'predicted_std': float(value_std),
        'value_loss': float(value_loss),
        'total_loss': float(var_loss + 0.5 * value_loss)
    }


def verify_variable_mapping(labels: List[Dict[str, Any]], sample_size: int = 5):
    """Verify that variable mapping is consistent."""
    
    print("\n" + "="*60)
    print("VARIABLE MAPPING VERIFICATION")
    print("="*60)
    
    # Sample some labels
    sample_labels = labels[:sample_size]
    
    for i, label in enumerate(sample_labels):
        variables = label.get('variables', [])
        targets = list(label.get('targets', []))
        
        if not variables or not targets:
            continue
            
        target_var = targets[0]
        
        # Check both sorting methods
        alpha_sorted = sorted(variables)
        num_sorted = numerical_sort_variables(variables)
        
        # Get indices
        if target_var in alpha_sorted and target_var in num_sorted:
            alpha_idx = alpha_sorted.index(target_var)
            num_idx = num_sorted.index(target_var)
            
            print(f"\nExample {i+1}:")
            print(f"  Variables ({len(variables)}): {variables[:5]}...")
            print(f"  Target: {target_var}")
            print(f"  Alpha index: {alpha_idx}, Numerical index: {num_idx}")
            if alpha_idx != num_idx:
                print(f"  ✓ Fixed: {target_var} moved from {alpha_idx} → {num_idx}")


def main():
    """Main training function with detailed debugging."""
    
    print("="*80)
    print("BC TRAINING WITH DETAILED DEBUGGING")
    print("="*80)
    print(f"Start time: {datetime.now()}")
    
    # Set random seed
    np.random.seed(42)
    
    # Create output directory
    output_dir = Path("debug_training_results")
    output_dir.mkdir(exist_ok=True)
    
    # Load demonstrations
    print("\n1. Loading demonstrations...")
    demos_path = Path("../expert_demonstrations/raw/raw_demonstrations")
    
    if not demos_path.exists():
        alt_paths = [
            Path("expert_demonstrations/raw/raw_demonstrations"),
            Path("../data/expert_demonstrations"),
            Path("data/expert_demonstrations")
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                demos_path = alt_path
                break
    
    raw_demos = load_demonstrations_from_path(str(demos_path), max_files=100)
    
    # Flatten demonstrations
    flat_demos = []
    for item in raw_demos:
        if hasattr(item, 'demonstrations'):
            flat_demos.extend(item.demonstrations)
        else:
            flat_demos.append(item)
    
    print(f"Loaded {len(flat_demos)} demonstrations")
    
    # Create training dataset with FIXED sorting
    print("\n2. Creating training dataset with numerical sorting...")
    all_inputs, all_labels, metadata = create_bc_training_dataset(
        flat_demos, max_trajectory_length=100
    )
    
    print(f"Created {len(all_inputs)} training examples")
    print(f"Tensor shape: {all_inputs[0].shape if all_inputs else 'N/A'}")
    
    # Verify variable mapping
    verify_variable_mapping(all_labels, sample_size=5)
    
    # Split into train/val
    print("\n3. Splitting into train/validation...")
    n_examples = len(all_inputs)
    n_train = int(0.8 * n_examples)
    
    indices = np.random.permutation(n_examples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_inputs = [all_inputs[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    val_inputs = [all_inputs[i] for i in val_indices]
    val_labels = [all_labels[i] for i in val_indices]
    
    print(f"Train: {len(train_inputs)} examples")
    print(f"Validation: {len(val_inputs)} examples")
    
    # Create trainer with label smoothing
    print("\n4. Creating trainer with label smoothing...")
    trainer = PolicyBCTrainer(
        hidden_dim=128,
        learning_rate=3e-3,
        batch_size=32,
        max_epochs=20,  # Fewer epochs for debugging
        gradient_clip=5.0,
        weight_decay=1e-4,
        label_smoothing=0.1,  # Add label smoothing to prevent overconfidence
        validation_split=0.0
    )
    
    # Initialize the model
    print("Initializing model...")
    trainer._initialize_model(all_inputs[0], metadata)
    
    # Training loop with detailed debugging
    print("\n5. Starting training with detailed debugging...")
    print("="*80)
    
    best_val_acc = 0.0
    loss_history = []
    gradient_history = []
    
    for epoch in range(trainer.max_epochs):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{trainer.max_epochs}")
        print(f"{'='*60}")
        
        # Train one epoch with debugging
        train_losses = []
        batch_predictions = []
        
        for batch_idx in range(0, len(train_inputs), trainer.batch_size):
            batch_inputs = train_inputs[batch_idx:batch_idx+trainer.batch_size]
            batch_labels = train_labels[batch_idx:batch_idx+trainer.batch_size]
            
            # Get a new random key for this batch
            trainer.key, batch_key = jax.random.split(trainer.key)
            
            # Detailed debugging for first batch of each epoch
            if batch_idx == 0:
                print(f"\n--- Detailed Debug for Batch 1 ---")
                
                # Forward pass for debugging
                outputs = trainer.net.apply(trainer.model_params, batch_key, batch_inputs[0], trainer.target_idx)
                var_logits = outputs['variable_logits']
                value_params = outputs['value_params']
                
                # Get target info
                target_vars = list(batch_labels[0]['targets'])
                target_var_name = target_vars[0] if target_vars else None
                target_value = batch_labels[0]['values'].get(target_var_name, 0.0)
                variables = batch_labels[0].get('variables', [])
                
                # Create mapper
                mapper = VariableMapper(variables, batch_labels[0].get('target_variable'))
                target_idx = mapper.get_index(target_var_name) if target_var_name else 0
                
                # Analyze prediction
                pred_analysis = analyze_predictions(var_logits, target_idx, variables, 0)
                print(f"\nPrediction Analysis:")
                print(f"  Target: {pred_analysis['target_variable']} (idx {pred_analysis['target_idx']})")
                print(f"  Predicted: {pred_analysis['predicted_variable']} (idx {pred_analysis['predicted_idx']})")
                print(f"  Target prob: {pred_analysis['target_prob']:.4f}")
                print(f"  Predicted prob: {pred_analysis['predicted_prob']:.4f}")
                print(f"  Top 3: {pred_analysis['top_3']}")
                
                # Debug loss calculation
                loss_breakdown = debug_loss_calculation(
                    var_logits, value_params, target_idx, target_value, variables
                )
                print(f"\nLoss Breakdown:")
                print(f"  Variable loss: {loss_breakdown['var_loss']:.4f}")
                print(f"  Value loss: {loss_breakdown['value_loss']:.4f}")
                print(f"  Total loss: {loss_breakdown['total_loss']:.4f}")
                
                # Gradient analysis (need to compute gradients)
                def loss_fn(params):
                    outputs = trainer.net.apply(params, batch_key, batch_inputs[0], trainer.target_idx)
                    var_logits = outputs['variable_logits']
                    var_loss = -jax.nn.log_softmax(var_logits)[target_idx]
                    return var_loss
                
                loss_val, grads = jax.value_and_grad(loss_fn)(trainer.model_params)
                grad_stats = analyze_gradients(grads, trainer.model_params, trainer.learning_rate)
                
                print(f"\nGradient Analysis:")
                print(f"  Grad norm (before clip): {grad_stats['grad_norm_before_clip']:.4f}")
                print(f"  Grad norm (after clip): {grad_stats['grad_norm_after_clip']:.4f}")
                print(f"  Clip ratio: {grad_stats['clip_ratio']:.2%}")
                print(f"  Effective LR: {grad_stats['effective_lr']:.6f}")
                print(f"  Relative update: {grad_stats['relative_update_pct']:.4f}%")
                
                gradient_history.append(grad_stats)
            
            # Regular training step
            loss = trainer._train_batch(batch_inputs, batch_labels, batch_key)
            train_losses.append(loss)
            
            # Sample predictions periodically
            if batch_idx % (10 * trainer.batch_size) == 0 and batch_idx > 0:
                print(f"  Batch {batch_idx//trainer.batch_size}: Loss = {loss:.4f}")
        
        avg_train_loss = np.mean(train_losses)
        loss_history.append(avg_train_loss)
        
        # Evaluate on validation set
        val_losses = []
        correct = 0
        total = 0
        per_var_correct = {}
        per_var_total = {}
        
        for val_input, val_label in zip(val_inputs, val_labels):
            # Get prediction
            trainer.key, eval_key = jax.random.split(trainer.key)
            outputs = trainer.net.apply(trainer.model_params, eval_key, val_input, trainer.target_idx)
            var_logits = outputs['variable_logits']
            
            # Get target
            target_vars = list(val_label['targets'])
            if not target_vars:
                continue
            target_var_name = target_vars[0]
            
            # Create mapper
            variables = val_label.get('variables', [])
            if not variables:
                continue
            
            mapper = VariableMapper(variables, val_label.get('target_variable'))
            
            try:
                actual_idx = mapper.get_index(target_var_name)
                predicted_idx = int(jnp.argmax(var_logits))
                
                # Track overall accuracy
                if predicted_idx == actual_idx:
                    correct += 1
                total += 1
                
                # Track per-variable accuracy
                if target_var_name not in per_var_correct:
                    per_var_correct[target_var_name] = 0
                    per_var_total[target_var_name] = 0
                per_var_total[target_var_name] += 1
                if predicted_idx == actual_idx:
                    per_var_correct[target_var_name] += 1
                
                # Calculate loss
                var_loss = -jax.nn.log_softmax(var_logits)[actual_idx]
                val_losses.append(float(var_loss))
            except ValueError:
                continue
        
        val_acc = correct / total if total > 0 else 0.0
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        
        # Print epoch summary
        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.3f} ({correct}/{total})")
        
        # Per-variable accuracy
        print(f"\nPer-Variable Accuracy:")
        for var in sorted(per_var_total.keys()):
            var_acc = per_var_correct[var] / per_var_total[var] if per_var_total[var] > 0 else 0
            print(f"  {var}: {var_acc:.3f} ({per_var_correct[var]}/{per_var_total[var]})")
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"✓ New best validation accuracy: {best_val_acc:.3f}")
        
        # Loss trajectory analysis
        if len(loss_history) > 3:
            recent_losses = loss_history[-3:]
            if all(abs(recent_losses[i] - recent_losses[i+1]) < 0.01 for i in range(len(recent_losses)-1)):
                print("⚠️ Warning: Loss appears to be plateauing")
        
        print(f"\nEpoch time: {time.time() - epoch_start:.2f}s")
    
    # Final analysis
    print("\n" + "="*80)
    print("TRAINING COMPLETE - FINAL ANALYSIS")
    print("="*80)
    
    print(f"\nBest validation accuracy: {best_val_acc:.3f}")
    
    # Analyze loss trajectory
    print(f"\nLoss Trajectory:")
    print(f"  Initial: {loss_history[0]:.4f}")
    print(f"  Final: {loss_history[-1]:.4f}")
    print(f"  Reduction: {loss_history[0] - loss_history[-1]:.4f}")
    
    # Analyze gradient behavior
    if gradient_history:
        avg_grad_norm = np.mean([g['grad_norm_after_clip'] for g in gradient_history])
        avg_update = np.mean([g['relative_update_pct'] for g in gradient_history])
        print(f"\nGradient Statistics:")
        print(f"  Avg gradient norm: {avg_grad_norm:.4f}")
        print(f"  Avg relative update: {avg_update:.4f}%")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()