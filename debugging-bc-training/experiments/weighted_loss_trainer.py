#!/usr/bin/env python3
"""
Weighted loss BC trainer that addresses class imbalance by weighting
the loss function based on target variable frequency.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter
import numpy as np
import jax
import jax.numpy as jnp
import logging
import optax

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from enhanced_bc_trainer_fixed import FixedEnhancedBCTrainer

logger = logging.getLogger(__name__)


def compute_class_weights(labels: List[Dict[str, Any]], 
                         smoothing: float = 0.01) -> Dict[str, float]:
    """
    Compute inverse frequency class weights from training labels.
    
    Args:
        labels: Training labels
        smoothing: Smoothing factor to avoid division by zero
        
    Returns:
        Dictionary mapping variable names to weights
    """
    # Count target frequencies
    target_counts = Counter()
    for label in labels:
        if 'targets' in label and label['targets']:
            for target in label['targets']:
                target_counts[target] += 1
    
    if not target_counts:
        return {}
    
    # Compute inverse frequency weights
    total = sum(target_counts.values())
    weights = {}
    
    for target, count in target_counts.items():
        frequency = count / total
        # Inverse frequency with smoothing
        weights[target] = 1.0 / (frequency + smoothing)
    
    # Normalize so mean weight = 1.0
    mean_weight = np.mean(list(weights.values()))
    weights = {k: v / mean_weight for k, v in weights.items()}
    
    logger.info("Computed class weights:")
    for target in sorted(weights.keys()):
        logger.info(f"  {target}: {weights[target]:.2f} (count: {target_counts[target]})")
    
    return weights


class WeightedLossTrainer(FixedEnhancedBCTrainer):
    """
    BC trainer with weighted loss to handle class imbalance.
    """
    
    def __init__(self,
                 use_class_weights: bool = True,
                 manual_weights: Optional[Dict[str, float]] = None,
                 weight_smoothing: float = 0.01,
                 **kwargs):
        """
        Initialize weighted loss trainer.
        
        Args:
            use_class_weights: Whether to use class weighting
            manual_weights: Manual weight dictionary (overrides computed weights)
            weight_smoothing: Smoothing factor for weight computation
            **kwargs: Arguments passed to parent trainer
        """
        super().__init__(**kwargs)
        self.use_class_weights = use_class_weights
        self.manual_weights = manual_weights
        self.weight_smoothing = weight_smoothing
        self.class_weights = {}
        
        if manual_weights:
            self.class_weights = manual_weights
            logger.info(f"Using manual class weights: {manual_weights}")
        else:
            logger.info("Will compute class weights from training data")
    
    def _compute_weights_from_data(self, train_labels: List[Dict[str, Any]]):
        """Compute class weights from training data."""
        if not self.manual_weights and self.use_class_weights:
            self.class_weights = compute_class_weights(
                train_labels, 
                smoothing=self.weight_smoothing
            )
            
            # Add default weight for unknown classes
            self.class_weights['_default'] = 1.0
    
    def _get_weight_for_target(self, target_var_name: str) -> float:
        """Get weight for a target variable."""
        if not self.use_class_weights:
            return 1.0
        
        return self.class_weights.get(target_var_name, 
                                      self.class_weights.get('_default', 1.0))
    
    def _train_batch(self, batch_inputs: List[jnp.ndarray], 
                     batch_labels: List[Dict[str, Any]], 
                     rng_key: jax.Array) -> float:
        """Train on a batch with weighted loss."""
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
                    continue
                
                from src.causal_bayes_opt.utils.variable_mapping import VariableMapper
                    
                example_mapper = VariableMapper(
                    variables=example_variables,
                    target_variable=label.get('target_variable')
                )
                
                # Map variable name to index
                try:
                    var_idx = example_mapper.get_index(target_var_name)
                except ValueError:
                    continue
                
                # Get class weight for this target
                weight = self._get_weight_for_target(target_var_name)
                
                # Variable selection loss with class weight
                log_probs = jax.nn.log_softmax(var_logits)
                log_probs = jnp.clip(log_probs, -100, 0)
                var_loss = -weight * log_probs[var_idx]  # Apply weight here
                
                # Value prediction loss (also weighted)
                value_mean = value_params[var_idx, 0]
                value_log_std = jnp.clip(value_params[var_idx, 1], -5, 2)
                value_std = jnp.exp(value_log_std)
                
                normalized_error = jnp.clip((target_value - value_mean) / (value_std + 1e-8), -10, 10)
                value_loss = weight * (0.5 * jnp.log(2 * jnp.pi) + value_log_std + 
                                      0.5 * normalized_error ** 2)
                
                # Total weighted loss
                example_loss = var_loss + 0.5 * value_loss
                example_loss = jnp.clip(example_loss, 0, 100)
                
                batch_loss += example_loss
                valid_examples += 1
            
            # Return average loss
            if valid_examples > 0:
                return batch_loss / valid_examples
            else:
                return jnp.array(1e-6)
        
        # Compute gradients and update
        loss_val, grads = jax.value_and_grad(loss_fn)(self.model_params)
        
        # Check for NaN/Inf
        if not jnp.isfinite(loss_val):
            logger.warning(f"Non-finite loss detected: {loss_val}, skipping batch")
            return 1.0
        
        # Clip gradients
        grads, _ = optax.clip_by_global_norm(self.gradient_clip).update(grads, None)
        
        # Apply updates
        updates, self.optimizer_state = self.optimizer.update(
            grads, self.optimizer_state, self.model_params
        )
        self.model_params = optax.apply_updates(self.model_params, updates)
        
        return float(loss_val)
    
    def train(self, demonstrations_path: str, max_demos: int = None,
              max_trajectory_length: int = 100, output_dir: str = None, **kwargs):
        """
        Train with weighted loss.
        
        Overrides to compute class weights before training.
        """
        start_time = time.time()
        logger.info("Starting weighted loss BC training")
        
        # Load demonstrations
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
        
        logger.info(f"Created {len(all_inputs)} training examples")
        
        # Initialize model
        self._initialize_model(all_inputs[0], dataset_metadata)
        
        # Split data
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
        
        # COMPUTE CLASS WEIGHTS FROM TRAINING DATA
        self._compute_weights_from_data(train_labels)
        
        # Continue with normal training (rest is same as parent)
        return self._continue_training(
            train_inputs, train_labels,
            val_inputs, val_labels,
            flat_demos, start_time, output_dir
        )
    
    def _continue_training(self, train_inputs, train_labels, val_inputs, val_labels,
                          flat_demos, start_time, output_dir):
        """Continue training after computing weights (reuses parent logic)."""
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_params = None
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            self.metrics_tracker.start_epoch(epoch)
            
            # Train epoch
            train_loss = self._train_epoch_with_metrics(train_inputs, train_labels, epoch)
            train_losses.append(train_loss)
            
            if not np.isfinite(train_loss):
                logger.warning(f"Non-finite train loss at epoch {epoch+1}")
                continue
            
            # Validation
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
                
                # End epoch with metrics
                if val_predictions:
                    self.metrics_tracker.end_epoch(
                        val_predictions=jnp.array(val_predictions),
                        val_targets=jnp.array(val_targets),
                        variable_names=self.variable_names
                    )
                else:
                    self.metrics_tracker.end_epoch()
                
                # Early stopping
                if np.isfinite(val_loss):
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = jax.tree.map(lambda x: x, self.model_params)
                        patience_counter = 0
                        logger.info(f"  ✓ New best val loss: {val_loss:.4f}")
                    else:
                        patience_counter += 1
                
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                
                # Log progress
                if epoch % 5 == 0:
                    metrics_summary = self.metrics_tracker.get_metrics_summary()
                    logger.info(f"Epoch {epoch+1}/{self.max_epochs}:")
                    logger.info(f"  Train loss: {train_loss:.4f}")
                    logger.info(f"  Val loss: {val_loss:.4f}")
                    
                    if 'latest_metrics' in metrics_summary:
                        latest = metrics_summary['latest_metrics']
                        if 'val_accuracy' in latest:
                            logger.info(f"  Val accuracy: {latest['val_accuracy']:.3f}")
        
        # Use best parameters
        if best_params is not None:
            self.model_params = best_params
        
        training_time = time.time() - start_time
        
        # Save metrics
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            metrics_file = output_path / "metrics_history.pkl"
            self.metrics_tracker.save(metrics_file)
            logger.info(f"Saved metrics to {metrics_file}")
        
        # Get final metrics
        metrics_summary = self.metrics_tracker.get_metrics_summary()
        
        # Prepare results
        results = {
            "params": self.model_params,
            "config": {
                "hidden_dim": self.hidden_dim,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "use_class_weights": self.use_class_weights,
                "class_weights": self.class_weights
            },
            "metrics": {
                "training_time": training_time,
                "epochs_trained": len(train_losses),
                "final_train_loss": train_losses[-1] if train_losses else 0.0,
                "best_val_loss": best_val_loss,
                **metrics_summary
            },
            "metadata": {
                "trainer_type": "WeightedLossTrainer",
                "n_train_samples": len(train_inputs),
                "n_val_samples": len(val_inputs),
                "n_demonstrations": len(flat_demos)
            }
        }
        
        logger.info(f"Training completed in {training_time/60:.1f} minutes")
        
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train BC with weighted loss')
    parser.add_argument('--demo_path', default='expert_demonstrations/raw/raw_demonstrations')
    parser.add_argument('--max_demos', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--output_dir', default='debugging-bc-training/results_experiments/weighted/')
    parser.add_argument('--seed', type=int, default=42)
    
    # Manual weights option
    parser.add_argument('--manual_weights', type=str, default=None,
                       help='Manual weights as JSON string, e.g. \'{"X0":1.0,"X4":10.0}\'')
    
    args = parser.parse_args()
    
    # Parse manual weights if provided
    manual_weights = None
    if args.manual_weights:
        import json
        manual_weights = json.loads(args.manual_weights)
    
    trainer = WeightedLossTrainer(
        use_class_weights=True,
        manual_weights=manual_weights,
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
    
    print(f"\nWeighted loss training complete!")
    print(f"Results saved to {args.output_dir}")