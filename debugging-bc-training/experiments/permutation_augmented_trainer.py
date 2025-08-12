#!/usr/bin/env python3
"""
Permutation-augmented BC trainer that creates balanced training data
by rotating variable orderings.

This leverages permutation equivariance to create multiple versions
of each training example with different variable orderings, ensuring
all variables appear equally often at each position.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import logging

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from enhanced_bc_trainer_fixed import FixedEnhancedBCTrainer
from src.causal_bayes_opt.utils.variable_mapping import VariableMapper

logger = logging.getLogger(__name__)


def create_permuted_versions(
    input_tensor: jnp.ndarray,
    label: Dict[str, Any],
    n_permutations: int = 5
) -> List[Tuple[jnp.ndarray, Dict[str, Any]]]:
    """
    Create permuted versions of a training example.
    
    For a 5-variable problem, we create rotations:
    - Original: [X0, X1, X2, X3, X4]
    - Rotation 1: [X4, X0, X1, X2, X3]
    - Rotation 2: [X3, X4, X0, X1, X2]
    - etc.
    
    Args:
        input_tensor: Input tensor [T, n_vars, channels]
        label: Label dictionary with targets and variable names
        n_permutations: Number of permutations to create (default 5 for full rotation)
        
    Returns:
        List of (permuted_tensor, permuted_label) tuples
    """
    augmented = []
    
    # Get original variable names
    original_vars = label.get('variables', ['X0', 'X1', 'X2', 'X3', 'X4'])
    n_vars = len(original_vars)
    
    # Skip if not 5 variables (to avoid breaking other SCM types)
    if n_vars != 5:
        return [(input_tensor, label)]
    
    for rotation in range(min(n_permutations, n_vars)):
        if rotation == 0:
            # Keep original
            augmented.append((input_tensor, label))
        else:
            # Create rotation permutation
            perm = np.roll(np.arange(n_vars), -rotation)
            
            # Permute tensor dimensions
            rotated_tensor = input_tensor[:, perm, :]
            
            # Create new variable ordering
            rotated_vars = [original_vars[i] for i in perm]
            
            # Update label
            rotated_label = label.copy()
            rotated_label['variables'] = rotated_vars
            
            # Update target index if it exists
            if 'targets' in label and label['targets']:
                target_var = list(label['targets'])[0]
                if target_var in original_vars:
                    # Find where the target variable ended up after rotation
                    old_idx = original_vars.index(target_var)
                    new_idx = np.where(perm == old_idx)[0][0]
                    # Target variable name stays the same, but its position changed
                    # This is handled by the VariableMapper
            
            augmented.append((rotated_tensor, rotated_label))
    
    return augmented


class PermutationAugmentedTrainer(FixedEnhancedBCTrainer):
    """
    BC trainer with permutation-based data augmentation for balanced training.
    """
    
    def __init__(self, 
                 n_permutations: int = 5,
                 permutation_probability: float = 1.0,
                 **kwargs):
        """
        Initialize permutation-augmented trainer.
        
        Args:
            n_permutations: Number of permutations per example (5 = full rotation)
            permutation_probability: Probability of applying augmentation (1.0 = always)
            **kwargs: Arguments passed to parent trainer
        """
        super().__init__(**kwargs)
        self.n_permutations = n_permutations
        self.permutation_probability = permutation_probability
        
        logger.info(f"Initialized PermutationAugmentedTrainer with {n_permutations} permutations")
    
    def _augment_training_data(self, 
                               inputs: List[jnp.ndarray], 
                               labels: List[Dict[str, Any]]) -> Tuple[List[jnp.ndarray], List[Dict[str, Any]]]:
        """
        Augment training data with permutations.
        
        Args:
            inputs: Original input tensors
            labels: Original labels
            
        Returns:
            Augmented inputs and labels
        """
        augmented_inputs = []
        augmented_labels = []
        
        for input_tensor, label in zip(inputs, labels):
            # Decide whether to augment this example
            if np.random.random() < self.permutation_probability:
                # Create permuted versions
                permuted_versions = create_permuted_versions(
                    input_tensor, label, self.n_permutations
                )
                
                for perm_tensor, perm_label in permuted_versions:
                    augmented_inputs.append(perm_tensor)
                    augmented_labels.append(perm_label)
            else:
                # Keep original only
                augmented_inputs.append(input_tensor)
                augmented_labels.append(label)
        
        # Log augmentation statistics
        original_size = len(inputs)
        augmented_size = len(augmented_inputs)
        logger.info(f"Augmented training data: {original_size} → {augmented_size} examples "
                   f"({augmented_size/original_size:.1f}x increase)")
        
        # Check target distribution after augmentation
        target_counts = {}
        for label in augmented_labels:
            if 'targets' in label and label['targets']:
                for target in label['targets']:
                    target_counts[target] = target_counts.get(target, 0) + 1
        
        if target_counts:
            total = sum(target_counts.values())
            logger.info("Target distribution after augmentation:")
            for target, count in sorted(target_counts.items()):
                logger.info(f"  {target}: {count} ({count/total*100:.1f}%)")
        
        return augmented_inputs, augmented_labels
    
    def train(self, demonstrations_path: str, max_demos: int = None, 
              max_trajectory_length: int = 100, output_dir: str = None, **kwargs):
        """
        Train with permutation augmentation.
        
        Overrides parent train method to add augmentation step.
        """
        # First, do all the normal data loading and preprocessing
        # by calling parent's data loading code
        
        start_time = time.time()
        logger.info("Starting permutation-augmented BC training")
        
        # Load demonstrations (from parent class logic)
        from src.causal_bayes_opt.training.data_preprocessing import load_demonstrations_from_path
        from src.causal_bayes_opt.training.demonstration_to_tensor import create_bc_training_dataset
        
        logger.info(f"Loading demonstrations from {demonstrations_path}")
        raw_demos = load_demonstrations_from_path(demonstrations_path, max_files=max_demos)
        
        # Flatten demonstrations
        flat_demos = []
        for item in raw_demos:
            if hasattr(item, 'demonstrations'):
                flat_demos.extend(item.demonstrations)
            else:
                flat_demos.append(item)
        
        if max_demos and len(flat_demos) > max_demos:
            flat_demos = flat_demos[:max_demos]
        
        # Convert to training data
        logger.info("Converting demonstrations to training tensors...")
        all_inputs, all_labels, dataset_metadata = create_bc_training_dataset(
            flat_demos, max_trajectory_length
        )
        
        if not all_inputs:
            raise ValueError("No valid training data created from demonstrations")
        
        # Store variable names
        self.variable_names = dataset_metadata.get('variables', [])
        
        logger.info(f"Created {len(all_inputs)} training examples before augmentation")
        
        # AUGMENT THE TRAINING DATA
        logger.info(f"Applying permutation augmentation with {self.n_permutations} permutations...")
        all_inputs, all_labels = self._augment_training_data(all_inputs, all_labels)
        
        logger.info(f"Training on {len(all_inputs)} examples after augmentation")
        
        # Initialize model
        self._initialize_model(all_inputs[0], dataset_metadata)
        
        # Split data (with augmented dataset)
        n_val = int(len(all_inputs) * self.validation_split)
        self.key, split_key = jax.random.split(self.key)
        indices = jax.random.permutation(split_key, len(all_inputs))
        
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        train_inputs = [all_inputs[i] for i in train_indices]
        train_labels = [all_labels[i] for i in train_indices]
        val_inputs = [all_inputs[i] for i in val_indices] if n_val > 0 else []
        val_labels = [all_labels[i] for i in val_indices] if n_val > 0 else []
        
        logger.info(f"Split: {len(train_inputs)} train, {len(val_inputs)} validation")
        
        # Continue with normal training (from parent)
        # This reuses all the parent's training loop logic
        return self._continue_training(
            train_inputs, train_labels,
            val_inputs, val_labels,
            flat_demos, start_time, output_dir
        )
    
    def _continue_training(self, train_inputs, train_labels, val_inputs, val_labels,
                          flat_demos, start_time, output_dir):
        """Continue training after augmentation (reuses parent logic)."""
        # Training loop with enhanced metrics
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_params = None
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            epoch_start = time.time()
            self.metrics_tracker.start_epoch(epoch)
            
            # Train epoch
            train_loss = self._train_epoch_with_metrics(train_inputs, train_labels, epoch)
            train_losses.append(train_loss)
            
            # Skip if loss is non-finite
            if not np.isfinite(train_loss):
                logger.warning(f"Non-finite train loss at epoch {epoch+1}, skipping metrics")
                continue
            
            # Validation with metrics
            val_predictions = []
            val_targets = []
            
            if val_inputs:
                val_loss = self._evaluate(val_inputs, val_labels)
                val_losses.append(val_loss)
                
                # Collect predictions for metrics
                for val_input, val_label in zip(val_inputs, val_labels):
                    self.key, eval_key = jax.random.split(self.key)
                    outputs = self.net.apply(self.model_params, eval_key, val_input, self.target_idx)
                    var_logits = outputs['variable_logits']
                    pred_idx = int(jnp.argmax(var_logits))
                    
                    if 'targets' in val_label and val_label['targets']:
                        target_var = list(val_label['targets'])[0]
                        if 'variables' in val_label:
                            try:
                                target_idx = val_label['variables'].index(target_var)
                                val_predictions.append(pred_idx)
                                val_targets.append(target_idx)
                            except ValueError:
                                pass
                
                # End epoch with validation metrics
                if val_predictions:
                    self.metrics_tracker.end_epoch(
                        val_predictions=jnp.array(val_predictions),
                        val_targets=jnp.array(val_targets),
                        variable_names=self.variable_names
                    )
                else:
                    self.metrics_tracker.end_epoch()
                
                # Early stopping (only on finite losses)
                if np.isfinite(val_loss):
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = jax.tree.map(lambda x: x, self.model_params)
                        patience_counter = 0
                        logger.info(f"  ✓ New best val loss: {val_loss:.4f}")
                    else:
                        patience_counter += 1
                else:
                    logger.warning(f"  Skipping early stopping check due to inf val loss")
                
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                
                # Log progress
                if epoch % 1 == 0:
                    metrics_summary = self.metrics_tracker.get_metrics_summary()
                    logger.info(f"Epoch {epoch+1}/{self.max_epochs}:")
                    train_loss_str = f"{train_loss:.4f}" if np.isfinite(train_loss) else "inf"
                    val_loss_str = f"{val_loss:.4f}" if np.isfinite(val_loss) else "inf"
                    logger.info(f"  Train loss: {train_loss_str}")
                    logger.info(f"  Val loss: {val_loss_str}")
                    
                    if 'latest_metrics' in metrics_summary:
                        latest = metrics_summary['latest_metrics']
                        for metric in ['val_accuracy', 'val_f1', 'val_precision', 'val_recall']:
                            if metric in latest:
                                logger.info(f"  {metric.replace('val_', 'Val ')}: {latest[metric]:.3f}")
                    
                    logger.info(f"  Patience counter: {patience_counter}/{self.early_stopping_patience}")
        
        # Use best parameters
        if best_params is not None:
            self.model_params = best_params
        
        training_time = time.time() - start_time
        
        # Save metrics if output directory provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            metrics_file = output_path / "metrics_history.pkl"
            self.metrics_tracker.save(metrics_file)
            logger.info(f"Saved metrics to {metrics_file}")
        
        # Get final metrics summary
        metrics_summary = self.metrics_tracker.get_metrics_summary()
        
        # Filter out non-finite losses from history
        train_losses = [l for l in train_losses if np.isfinite(l)]
        val_losses = [l for l in val_losses if np.isfinite(l)]
        
        # Prepare results
        results = {
            "params": self.model_params,
            "config": {
                "hidden_dim": self.hidden_dim,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "n_permutations": self.n_permutations
            },
            "metrics": {
                "training_time": training_time,
                "epochs_trained": len(train_losses),
                "final_train_loss": train_losses[-1] if train_losses else 0.0,
                "best_val_loss": best_val_loss if val_inputs and np.isfinite(best_val_loss) else (train_losses[-1] if train_losses else 0.0),
                "train_history": train_losses,
                "val_history": val_losses,
                **metrics_summary
            },
            "metadata": {
                "trainer_type": "PermutationAugmentedTrainer",
                "model_type": "acquisition",
                "n_train_samples": len(train_inputs),
                "n_val_samples": len(val_inputs),
                "n_demonstrations": len(flat_demos),
                "augmentation_factor": len(train_inputs) / (len(flat_demos) * 5),  # Approximate
                "tensor_channels": 5,
                "uses_structural_knowledge": True,
                "variables": self.variable_names
            }
        }
        
        logger.info(f"Training completed in {training_time/60:.1f} minutes")
        logger.info(f"Final metrics summary:")
        for key, value in metrics_summary.items():
            if not isinstance(value, (dict, list)):
                logger.info(f"  {key}: {value}")
        
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train BC with permutation augmentation')
    parser.add_argument('--demo_path', default='expert_demonstrations/raw/raw_demonstrations')
    parser.add_argument('--max_demos', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_permutations', type=int, default=5)
    parser.add_argument('--output_dir', default='debugging-bc-training/results_experiments/permutation/')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    trainer = PermutationAugmentedTrainer(
        n_permutations=args.n_permutations,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        seed=args.seed
    )
    
    results = trainer.train(
        demonstrations_path=args.demo_path,
        max_demos=args.max_demos,
        output_dir=args.output_dir
    )
    
    print(f"\nPermutation augmentation training complete!")
    print(f"Results saved to {args.output_dir}")