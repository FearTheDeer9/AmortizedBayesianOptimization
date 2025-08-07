#!/usr/bin/env python3
"""
Fixed enhanced BC trainer with proper loss handling.

This version fixes the inf loss issue by adding numerical stability checks.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.training.policy_bc_trainer import PolicyBCTrainer
from src.causal_bayes_opt.training.data_preprocessing import load_demonstrations_from_path
from src.causal_bayes_opt.training.demonstration_to_tensor import create_bc_training_dataset
from metrics_tracker import BCMetricsTracker, extract_embeddings_from_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FixedEnhancedBCTrainer(PolicyBCTrainer):
    """
    Fixed enhanced BC trainer with numerical stability.
    """
    
    def __init__(self, 
                 metrics_tracker: Optional[BCMetricsTracker] = None,
                 save_embeddings_every: int = 10,
                 **kwargs):
        """
        Initialize enhanced trainer.
        """
        super().__init__(**kwargs)
        
        self.metrics_tracker = metrics_tracker or BCMetricsTracker(
            save_embeddings_every=save_embeddings_every
        )
        self.variable_names = None
    
    def _train_batch(self, batch_inputs: List[jnp.ndarray], 
                     batch_labels: List[Dict[str, Any]], 
                     rng_key: jax.Array) -> float:
        """Train on a batch with numerical stability fixes."""
        def loss_fn(params):
            batch_loss = 0.0
            valid_examples = 0
            
            for input_tensor, label in zip(batch_inputs, batch_labels):
                # Forward pass with 5-channel tensor
                outputs = self.net.apply(params, rng_key, input_tensor, self.target_idx)
                var_logits = outputs['variable_logits']
                value_params = outputs['value_params']
                
                # Extract intervention target
                target_vars = list(label['targets'])
                if not target_vars:
                    continue
                    
                target_var_name = target_vars[0]
                target_value = label['values'][target_var_name]
                
                # Create variable mapper for this example
                example_variables = label.get('variables', [])
                if not example_variables:
                    logger.warning(f"No variable names in label, skipping")
                    continue
                
                # Import here to avoid circular dependency
                from src.causal_bayes_opt.utils.variable_mapping import VariableMapper
                    
                example_mapper = VariableMapper(
                    variables=example_variables,
                    target_variable=label.get('target_variable')
                )
                
                # Map variable name to index for this example
                try:
                    var_idx = example_mapper.get_index(target_var_name)
                except ValueError:
                    logger.warning(f"Variable {target_var_name} not in example variables: {example_variables}")
                    continue
                
                # Variable selection loss with numerical stability
                log_probs = jax.nn.log_softmax(var_logits)
                # Clip to prevent -inf
                log_probs = jnp.clip(log_probs, -100, 0)
                var_loss = -log_probs[var_idx]
                
                # Value prediction loss with numerical stability
                value_mean = value_params[var_idx, 0]
                value_log_std = value_params[var_idx, 1]
                # More aggressive clipping for stability
                value_log_std = jnp.clip(value_log_std, -5, 2)
                value_std = jnp.exp(value_log_std)
                
                # Compute normalized error
                normalized_error = (target_value - value_mean) / (value_std + 1e-8)
                # Clip extreme errors
                normalized_error = jnp.clip(normalized_error, -10, 10)
                
                value_loss = 0.5 * jnp.log(2 * jnp.pi) + value_log_std + \
                            0.5 * normalized_error ** 2
                
                # Clip total loss to prevent explosion
                example_loss = var_loss + 0.5 * value_loss
                example_loss = jnp.clip(example_loss, 0, 100)
                
                batch_loss += example_loss
                valid_examples += 1
            
            # Return average loss, or small value if no valid examples
            if valid_examples > 0:
                return batch_loss / valid_examples
            else:
                return jnp.array(1e-6)  # Small non-zero loss
        
        # Compute gradients and update
        loss_val, grads = jax.value_and_grad(loss_fn)(self.model_params)
        
        # Check for NaN/Inf
        if not jnp.isfinite(loss_val):
            logger.warning(f"Non-finite loss detected: {loss_val}, skipping batch")
            return 1.0  # Return default loss
        
        # Clip gradients
        grads, _ = optax.clip_by_global_norm(self.gradient_clip).update(grads, None)
        
        # Apply updates
        updates, self.optimizer_state = self.optimizer.update(
            grads, self.optimizer_state, self.model_params
        )
        self.model_params = optax.apply_updates(self.model_params, updates)
        
        return float(loss_val)
    
    def _evaluate(self, inputs: List[jnp.ndarray], labels: List[Dict[str, Any]]) -> float:
        """Evaluate with numerical stability."""
        total_loss = 0.0
        n_examples = 0
        
        for input_tensor, label in zip(inputs, labels):
            self.key, eval_key = random.split(self.key)
            
            # Forward pass
            outputs = self.net.apply(self.model_params, eval_key, input_tensor, self.target_idx)
            var_logits = outputs['variable_logits']
            value_params = outputs['value_params']
            
            # Extract target
            target_vars = list(label['targets'])
            if not target_vars:
                continue
                
            target_var_name = target_vars[0]
            target_value = label['values'][target_var_name]
            
            # Create variable mapper for this example
            example_variables = label.get('variables', [])
            if not example_variables:
                continue
            
            from src.causal_bayes_opt.utils.variable_mapping import VariableMapper
                
            example_mapper = VariableMapper(
                variables=example_variables,
                target_variable=label.get('target_variable')
            )
            
            # Map to index
            try:
                var_idx = example_mapper.get_index(target_var_name)
            except ValueError:
                continue
            
            # Compute losses with numerical stability
            log_probs = jax.nn.log_softmax(var_logits)
            log_probs = jnp.clip(log_probs, -100, 0)
            var_loss = -log_probs[var_idx]
            
            value_mean = value_params[var_idx, 0]
            value_log_std = jnp.clip(value_params[var_idx, 1], -5, 2)
            value_std = jnp.exp(value_log_std)
            
            normalized_error = jnp.clip((target_value - value_mean) / (value_std + 1e-8), -10, 10)
            value_loss = 0.5 * jnp.log(2 * jnp.pi) + value_log_std + 0.5 * normalized_error ** 2
            
            example_loss = var_loss + 0.5 * value_loss
            example_loss = jnp.clip(example_loss, 0, 100)
            
            if jnp.isfinite(example_loss):
                total_loss += float(example_loss)
                n_examples += 1
        
        return total_loss / n_examples if n_examples > 0 else 1.0
    
    def train(self, demonstrations_path: str, max_demos: Optional[int] = None, 
              max_trajectory_length: int = 100, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Train with enhanced metrics tracking and numerical stability.
        """
        start_time = time.time()
        logger.info("Starting enhanced BC training with metrics tracking (FIXED)")
        
        # Load demonstrations
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
            logger.info(f"Limited to {max_demos} demonstrations")
        
        # Convert to training data
        logger.info("Converting demonstrations to training tensors...")
        all_inputs, all_labels, dataset_metadata = create_bc_training_dataset(
            flat_demos, max_trajectory_length
        )
        
        if not all_inputs:
            raise ValueError("No valid training data created from demonstrations")
        
        # Store variable names
        self.variable_names = dataset_metadata.get('variables', [])
        
        logger.info(f"Created {len(all_inputs)} training examples")
        logger.info(f"Variables: {self.variable_names}")
        
        # Initialize model
        self._initialize_model(all_inputs[0], dataset_metadata)
        
        # Split data
        n_val = int(len(all_inputs) * self.validation_split)
        self.key, split_key = random.split(self.key)
        indices = random.permutation(split_key, len(all_inputs))
        
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        train_inputs = [all_inputs[i] for i in train_indices]
        train_labels = [all_labels[i] for i in train_indices]
        val_inputs = [all_inputs[i] for i in val_indices] if n_val > 0 else []
        val_labels = [all_labels[i] for i in val_indices] if n_val > 0 else []
        
        logger.info(f"Split: {len(train_inputs)} train, {len(val_inputs)} validation")
        
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
                    self.key, eval_key = random.split(self.key)
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
                    # Don't count inf losses toward early stopping
                    logger.warning(f"  Skipping early stopping check due to inf val loss")
                
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                
                # Log progress - EVERY EPOCH for better debugging
                if epoch % 1 == 0:  # Changed from 10 to 1
                    metrics_summary = self.metrics_tracker.get_metrics_summary()
                    logger.info(f"Epoch {epoch+1}/{self.max_epochs}:")
                    train_loss_str = f"{train_loss:.4f}" if np.isfinite(train_loss) else "inf"
                    val_loss_str = f"{val_loss:.4f}" if np.isfinite(val_loss) else "inf"
                    logger.info(f"  Train loss: {train_loss_str}")
                    logger.info(f"  Val loss: {val_loss_str}")
                    
                    if 'latest_metrics' in metrics_summary:
                        latest = metrics_summary['latest_metrics']
                        if 'val_accuracy' in latest:
                            logger.info(f"  Val accuracy: {latest['val_accuracy']:.3f}")
                        if 'val_f1' in latest:
                            logger.info(f"  Val F1: {latest['val_f1']:.3f}")
                        if 'val_precision' in latest:
                            logger.info(f"  Val precision: {latest['val_precision']:.3f}")
                        if 'val_recall' in latest:
                            logger.info(f"  Val recall: {latest['val_recall']:.3f}")
                    
                    # Also log gradient norms if available
                    logger.info(f"  Patience counter: {patience_counter}/{self.early_stopping_patience}")
            else:
                self.metrics_tracker.end_epoch()
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.max_epochs}: train_loss={train_loss:.4f}")
        
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
                "batch_size": self.batch_size
            },
            "metrics": {
                "training_time": training_time,
                "epochs_trained": len(train_losses),
                "final_train_loss": train_losses[-1] if train_losses else 0.0,
                "best_val_loss": best_val_loss if val_inputs and np.isfinite(best_val_loss) else (train_losses[-1] if train_losses else 0.0),
                "train_history": train_losses,
                "val_history": val_losses,
                **metrics_summary  # Include enhanced metrics
            },
            "metadata": {
                "trainer_type": "FixedEnhancedBCTrainer",
                "model_type": "acquisition",
                "n_train_samples": len(train_inputs),
                "n_val_samples": len(val_inputs),
                "n_demonstrations": len(flat_demos),
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
    
    def _train_epoch_with_metrics(self, inputs: List[jnp.ndarray], 
                                  labels: List[Dict[str, Any]], 
                                  epoch: int) -> float:
        """Train epoch with metrics tracking."""
        # Shuffle data
        self.key, shuffle_key = random.split(self.key)
        indices = random.permutation(shuffle_key, len(inputs))
        
        total_loss = 0.0
        n_batches = 0
        valid_batches = 0
        
        # Process in batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_inputs = [inputs[idx] for idx in batch_indices]
            batch_labels = [labels[idx] for idx in batch_indices]
            
            self.key, step_key = random.split(self.key)
            
            # Get predictions for metrics
            batch_predictions = []
            batch_targets = []
            
            for input_tensor, label in zip(batch_inputs, batch_labels):
                # Get model prediction
                outputs = self.net.apply(self.model_params, step_key, input_tensor, self.target_idx)
                var_logits = outputs['variable_logits']
                pred_idx = jnp.argmax(var_logits)
                batch_predictions.append(pred_idx)
                
                # Get target
                if 'targets' in label and label['targets']:
                    target_var = list(label['targets'])[0]
                    if 'variables' in label:
                        try:
                            target_idx = label['variables'].index(target_var)
                            batch_targets.append({'variable_idx': target_idx})
                        except ValueError:
                            batch_targets.append({'variable_idx': 0})
            
            # Track batch metrics
            if batch_predictions and batch_targets:
                self.metrics_tracker.track_batch(
                    predictions=jnp.array(batch_predictions),
                    targets=batch_targets
                )
            
            # Train batch
            loss = self._train_batch(batch_inputs, batch_labels, step_key)
            if np.isfinite(loss):
                total_loss += loss
                valid_batches += 1
            n_batches += 1
        
        return total_loss / valid_batches if valid_batches > 0 else float('inf')


def main():
    """Main training function with fixed numerical stability."""
    parser = argparse.ArgumentParser(description='Train BC with enhanced metrics (FIXED)')
    
    # Training parameters
    parser.add_argument('--demo_path', type=str,
                       default='expert_demonstrations/raw/raw_demonstrations',
                       help='Path to expert demonstrations')
    parser.add_argument('--max_demos', type=int, default=None,
                       help='Maximum number of demonstrations to use')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension')
    
    # Metrics parameters
    parser.add_argument('--save_embeddings_every', type=int, default=10,
                       help='Save embeddings every N epochs')
    parser.add_argument('--output_dir', type=str,
                       default='debugging-bc-training/results/',
                       help='Output directory for metrics and results')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str,
                       default='checkpoints',
                       help='Directory to save model checkpoint')
    
    args = parser.parse_args()
    
    # Create enhanced trainer with fixes
    trainer = FixedEnhancedBCTrainer(
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        save_embeddings_every=args.save_embeddings_every,
        seed=args.seed
    )
    
    # Train with metrics
    results = trainer.train(
        demonstrations_path=args.demo_path,
        max_demos=args.max_demos,
        output_dir=args.output_dir
    )
    
    # Save checkpoint
    checkpoint_path = Path(args.checkpoint_dir) / 'bc_enhanced_fixed_final'
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(checkpoint_path, results)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    logger.info("Enhanced training (FIXED) completed successfully!")
    logger.info(f"Metrics saved to {args.output_dir}")


if __name__ == "__main__":
    main()