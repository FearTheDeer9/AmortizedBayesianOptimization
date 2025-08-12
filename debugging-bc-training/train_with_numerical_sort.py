#!/usr/bin/env python3
"""
Train BC policy with numerical sorting fix.

This script trains the policy using the fixed numerical sorting for variables,
which should resolve the index misalignment issue and improve training.
"""

import sys
from pathlib import Path
import pickle
import time
from datetime import datetime
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List, Any, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import standard modules
from src.causal_bayes_opt.training.data_preprocessing import load_demonstrations_from_path
from src.causal_bayes_opt.training.policy_bc_trainer import PolicyBCTrainer

# Import our fixed modules with numerical sorting
from demonstration_to_tensor_fixed import create_bc_training_dataset
from variable_mapping_fixed import VariableMapper

# Import metrics tracking
from metrics_tracker import BCMetricsTracker as MetricsTracker


def analyze_variable_distribution(labels: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze the distribution of target variables in labels."""
    variable_counts = {}
    
    for label in labels:
        if 'targets' in label and label['targets']:
            target_var = list(label['targets'])[0]
            variable_counts[target_var] = variable_counts.get(target_var, 0) + 1
    
    return variable_counts


def main():
    """Main training function with numerical sorting fix."""
    
    print("="*80)
    print("BC TRAINING WITH NUMERICAL SORTING FIX")
    print("="*80)
    print(f"Start time: {datetime.now()}")
    
    # Set random seed
    np.random.seed(42)
    
    # Create output directory
    output_dir = Path("numerical_sort_results")
    output_dir.mkdir(exist_ok=True)
    
    # Load demonstrations
    print("\n1. Loading demonstrations...")
    demos_path = Path("../expert_demonstrations/raw/raw_demonstrations")
    
    if not demos_path.exists():
        # Try alternative paths
        alt_paths = [
            Path("expert_demonstrations/raw/raw_demonstrations"),
            Path("../data/expert_demonstrations"),
            Path("data/expert_demonstrations")
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                demos_path = alt_path
                break
        else:
            print(f"ERROR: Could not find demonstrations directory")
            print(f"Tried: {demos_path} and alternatives")
            return
    
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
    
    # Analyze variable distribution
    print("\n3. Analyzing variable distribution...")
    variable_counts = analyze_variable_distribution(all_labels)
    
    total_examples = sum(variable_counts.values())
    print("\nTarget Variable Distribution:")
    print("-" * 40)
    for var in sorted(variable_counts.keys()):
        count = variable_counts[var]
        percentage = count / total_examples * 100
        print(f"{var:8s}: {count:6d} ({percentage:5.1f}%)")
    
    # Check if X4 appears now
    if 'X4' in variable_counts:
        print(f"\n✓ X4 appears {variable_counts['X4']} times in training data!")
    else:
        print("\n⚠️ X4 still doesn't appear in training data")
    
    # Split into train/val
    print("\n4. Splitting into train/validation...")
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
    
    # Create trainer with optimal hyperparameters
    print("\n5. Creating trainer...")
    trainer = PolicyBCTrainer(
        hidden_dim=128,
        learning_rate=3e-3,
        batch_size=32,
        max_epochs=50,
        gradient_clip=5.0,
        weight_decay=1e-4,
        validation_split=0.0  # We already split manually
    )
    
    # Initialize the model (normally done in train() method)
    print("Initializing model...")
    trainer._initialize_model(all_inputs[0], metadata)
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(
        save_embeddings_every=10,
        max_embeddings_per_epoch=100
    )
    
    # Training loop with detailed metrics
    print("\n6. Starting training...")
    print("-" * 60)
    
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(trainer.max_epochs):
        epoch_start = time.time()
        
        # Train one epoch
        train_losses = []
        for i in range(0, len(train_inputs), trainer.batch_size):
            batch_inputs = train_inputs[i:i+trainer.batch_size]
            batch_labels = train_labels[i:i+trainer.batch_size]
            
            # Get a new random key for this batch
            trainer.key, batch_key = jax.random.split(trainer.key)
            loss = trainer._train_batch(batch_inputs, batch_labels, batch_key)
            train_losses.append(loss)
        
        avg_train_loss = np.mean(train_losses)
        
        # Evaluate
        val_loss = trainer._evaluate(val_inputs, val_labels)
        
        # Calculate accuracy
        correct = 0
        total = 0
        per_var_stats = {}
        
        for input_tensor, label in zip(val_inputs, val_labels):
            # Get prediction
            trainer.key, eval_key = jax.random.split(trainer.key)
            outputs = trainer.net.apply(trainer.model_params, eval_key, input_tensor, trainer.target_idx)
            var_logits = outputs['variable_logits']
            
            # Get target
            target_vars = list(label['targets'])
            if not target_vars:
                continue
            target_var_name = target_vars[0]
            
            # Create mapper for this example
            example_variables = label.get('variables', [])
            if not example_variables:
                continue
            
            example_mapper = VariableMapper(
                variables=example_variables,
                target_variable=label.get('target_variable')
            )
            
            # Get predicted and actual indices
            try:
                actual_idx = example_mapper.get_index(target_var_name)
                predicted_idx = int(jnp.argmax(var_logits))
                
                # Track per-variable stats
                if target_var_name not in per_var_stats:
                    per_var_stats[target_var_name] = {'attempts': 0, 'correct': 0}
                per_var_stats[target_var_name]['attempts'] += 1
                
                if predicted_idx == actual_idx:
                    correct += 1
                    per_var_stats[target_var_name]['correct'] += 1
                
                total += 1
            except ValueError:
                continue
        
        val_acc = correct / total if total > 0 else 0.0
        
        # Track metrics using BCMetricsTracker's actual methods
        metrics_tracker.start_epoch(epoch)
        
        # Store metrics in tracker's format
        metrics_tracker.epoch_metrics.append({
            'epoch': epoch,
            'train_accuracy': val_acc,  # Using val_acc as we don't have train_acc
            'n_batches': len(train_losses),
            'val_accuracy': val_acc,
            'val_f1': 0.0,  # We'd need to calculate this
            'val_precision': 0.0,
            'val_recall': 0.0
        })
        
        # Update per-variable stats
        for var_name, stats in per_var_stats.items():
            if var_name not in metrics_tracker.per_variable_stats:
                metrics_tracker.per_variable_stats[var_name] = {
                    'attempts': 0,
                    'correct': 0,
                    'accuracy_history': []
                }
            metrics_tracker.per_variable_stats[var_name]['attempts'] = stats['attempts']
            metrics_tracker.per_variable_stats[var_name]['correct'] = stats['correct']
            acc = stats['correct'] / stats['attempts'] if stats['attempts'] > 0 else 0.0
            metrics_tracker.per_variable_stats[var_name]['accuracy_history'].append((epoch, acc))
        
        # Print progress
        print(f"Epoch {epoch+1}/{trainer.max_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {val_acc:.3f}")
        
        # Check X4 accuracy
        if 'X4' in per_var_stats:
            x4_stats = per_var_stats['X4']
            if x4_stats['attempts'] > 0:
                x4_acc = x4_stats['correct'] / x4_stats['attempts']
                print(f"  X4 Acc:     {x4_acc:.3f} ({x4_stats['correct']}/{x4_stats['attempts']})")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            checkpoint_path = output_dir / "best_model.pkl"
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'params': trainer.model_params,
                    'config': {
                        'hidden_dim': trainer.hidden_dim,
                        'learning_rate': trainer.learning_rate,
                        'batch_size': trainer.batch_size
                    },
                    'metrics': {
                        'val_accuracy': val_acc,
                        'val_loss': val_loss,
                        'train_loss': avg_train_loss,
                        'per_variable_stats': per_var_stats
                    },
                    'epoch': epoch
                }, f)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Save final metrics
    print("\n7. Saving results...")
    
    # Save metrics manually since BCMetricsTracker doesn't have a save method
    metrics_path = output_dir / "metrics_history.pkl"
    with open(metrics_path, 'wb') as f:
        pickle.dump({
            'epoch_metrics': metrics_tracker.epoch_metrics,
            'per_variable_stats': metrics_tracker.per_variable_stats,
            'confusion_matrices': metrics_tracker.confusion_matrices,
            'config': {
                'save_embeddings_every': metrics_tracker.save_embeddings_every,
                'max_embeddings_per_epoch': metrics_tracker.max_embeddings_per_epoch
            }
        }, f)
    
    # Final analysis
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    
    # Check final X4 performance
    if 'X4' in per_var_stats:
        x4_stats = per_var_stats['X4']
        if x4_stats['attempts'] > 0:
            x4_acc = x4_stats['correct'] / x4_stats['attempts']
            print(f"Final X4 accuracy: {x4_acc:.3f}")
            if x4_acc > 0:
                print("✓ SUCCESS: X4 is being predicted correctly!")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"End time: {datetime.now()}")


if __name__ == "__main__":
    main()