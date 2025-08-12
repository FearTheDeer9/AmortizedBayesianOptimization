#!/usr/bin/env python3
"""
Train BC policy with variable permutation to break size-based shortcuts.

This script uses:
1. Label smoothing to prevent gradient vanishing
2. Variable permutation to prevent position-based shortcuts
3. Reduced epochs (10) for faster iteration
"""

import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from datetime import datetime
from collections import defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import standard modules
from src.causal_bayes_opt.training.data_preprocessing import load_demonstrations_from_path

# Import our fixed modules
from demonstration_to_tensor_permuted import create_bc_training_dataset_permuted
from policy_bc_trainer_smoothed import PolicyBCTrainer, smooth_cross_entropy_loss, robust_value_loss
from permuted_variable_mapper import PermutedVariableMapper

def train_with_permutation():
    """Train with both label smoothing and variable permutation."""
    
    print("="*80)
    print("BC TRAINING WITH PERMUTATION + LABEL SMOOTHING")
    print("="*80)
    print(f"Start time: {datetime.now()}")
    
    # Set random seed
    np.random.seed(42)
    
    # Load demonstrations
    print("\n1. Loading demonstrations...")
    demos_path = Path("../expert_demonstrations/raw/raw_demonstrations")
    
    if not demos_path.exists():
        alt_paths = [
            Path("expert_demonstrations/raw/raw_demonstrations"),
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
    
    # Create training dataset WITH PERMUTATION
    print("\n2. Creating training dataset with variable permutation...")
    all_inputs, all_labels, metadata = create_bc_training_dataset_permuted(
        flat_demos, 
        max_trajectory_length=100,
        use_permutation=True,  # Enable permutation
        base_seed=42
    )
    
    print(f"Created {len(all_inputs)} training examples")
    print(f"Tensor shape: {all_inputs[0].shape if all_inputs else 'N/A'}")
    print(f"Using variable permutation: {metadata.get('uses_permutation', False)}")
    
    # Verify permutation is working
    print("\n3. Verifying permutation effect...")
    target_positions = defaultdict(list)
    for label in all_labels[:100]:  # Check first 100
        target = list(label['targets'])[0]
        idx = label.get('permuted_target_idx', -1)
        target_positions[target].append(idx)
    
    print("Variable position variance (higher = more randomization):")
    for var in sorted(target_positions.keys()):
        positions = target_positions[var]
        if len(positions) > 1:
            variance = np.var(positions)
            print(f"  {var}: variance = {variance:.2f} (positions: {set(positions)})")
    
    # Create modified trainer that uses permuted indices
    print("\n4. Creating trainer with label smoothing...")
    
    class PermutedPolicyBCTrainer(PolicyBCTrainer):
        """Modified trainer that uses permuted indices from labels."""
        
        def _train_batch(self, batch_inputs, batch_labels, rng_key):
            """Train on batch using permuted indices."""
            def loss_fn(params):
                batch_loss = 0.0
                valid_examples = 0
                
                for input_tensor, label in zip(batch_inputs, batch_labels):
                    # Forward pass
                    outputs = self.net.apply(params, rng_key, input_tensor, self.target_idx)
                    var_logits = outputs['variable_logits']
                    value_params = outputs['value_params']
                    
                    # Get PERMUTED target index directly from label
                    var_idx = label.get('permuted_target_idx')
                    if var_idx is None:
                        continue
                    
                    # Get target value
                    target_vars = list(label.get('targets', []))
                    if not target_vars:
                        continue
                    target_var_name = target_vars[0]
                    target_value = label['values'].get(target_var_name)
                    
                    if target_value is None:
                        continue
                    
                    # Variable selection loss with label smoothing
                    var_loss = smooth_cross_entropy_loss(var_logits, var_idx, self.label_smoothing)
                    
                    # Value prediction loss
                    value_mean = value_params[var_idx, 0]
                    value_log_std = value_params[var_idx, 1]
                    value_loss = robust_value_loss(value_mean, value_log_std, target_value)
                    
                    # Combine losses
                    example_loss = var_loss + 0.5 * value_loss
                    example_loss = jnp.where(
                        jnp.isfinite(example_loss),
                        example_loss,
                        10.0
                    )
                    
                    batch_loss += example_loss
                    valid_examples += 1
                
                return batch_loss / valid_examples if valid_examples > 0 else 0.0
            
            # Compute gradients and update
            loss_val, grads = jax.value_and_grad(loss_fn)(self.model_params)
            grads, _ = optax.clip_by_global_norm(self.gradient_clip).update(grads, None)
            updates, self.optimizer_state = self.optimizer.update(
                grads, self.optimizer_state, self.model_params
            )
            self.model_params = optax.apply_updates(self.model_params, updates)
            
            return float(loss_val)
    
    # Import optax for the trainer
    import optax
    
    trainer = PermutedPolicyBCTrainer(
        hidden_dim=128,
        learning_rate=3e-3,
        batch_size=32,
        max_epochs=10,  # Reduced to 10 for faster iteration
        gradient_clip=5.0,
        weight_decay=1e-4,
        label_smoothing=0.1,
        validation_split=0.2
    )
    
    # Train the model
    print("\n5. Starting training...")
    print("="*80)
    
    results = trainer.train(
        demonstrations_path=str(demos_path),
        max_demos=100,
        max_trajectory_length=100
    )
    
    # Analyze results
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    metrics = results.get('metrics', {})
    print(f"\nTraining time: {metrics.get('training_time', 0)/60:.1f} minutes")
    print(f"Epochs trained: {metrics.get('epochs_trained', 0)}")
    print(f"Final train loss: {metrics.get('final_train_loss', 0):.4f}")
    print(f"Best val loss: {metrics.get('best_val_loss', 0):.4f}")
    
    # Check if we improved
    train_history = metrics.get('train_history', [])
    if train_history:
        improvement = train_history[0] - train_history[-1]
        print(f"Loss improvement: {improvement:.4f} ({improvement/train_history[0]*100:.1f}%)")
    
    print("\n" + "="*80)
    print("KEY EXPECTATIONS:")
    print("="*80)
    print("""
With variable permutation + label smoothing:
1. X2 accuracy should improve significantly (target: >50%)
2. No more perfect accuracy on X3/X6/X8/X10 (shortcuts broken)
3. More uniform accuracy across all variables
4. Model learns actual causal patterns, not position heuristics
""")

if __name__ == "__main__":
    train_with_permutation()