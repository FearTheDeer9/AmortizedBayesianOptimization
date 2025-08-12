#!/usr/bin/env python3
"""
Variable permutation trainer that randomly permutes variable ordering during training.
This ensures all variables (including rare ones like X4) get equal exposure at all positions.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import logging

sys.path.append(str(Path(__file__).parent.parent))

from enhanced_bc_trainer_fixed import FixedEnhancedBCTrainer
from src.causal_bayes_opt.utils.variable_mapping import VariableMapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_random_permutation(n_vars: int, rng: np.random.RandomState) -> np.ndarray:
    """Create a random permutation of variable indices."""
    return rng.permutation(n_vars)


def apply_permutation_to_tensor(tensor: jnp.ndarray, perm: np.ndarray) -> jnp.ndarray:
    """Apply permutation to the variable dimension of a tensor.
    
    Args:
        tensor: Shape [T, n_vars, channels]
        perm: Permutation indices
    
    Returns:
        Permuted tensor
    """
    return tensor[:, perm, :]


def apply_permutation_to_label(label: Dict[str, Any], perm: np.ndarray, 
                               original_vars: List[str]) -> Dict[str, Any]:
    """Apply permutation to a label dictionary.
    
    This updates the variable ordering and ensures targets map correctly.
    """
    permuted_label = label.copy()
    
    # Create permuted variable list
    permuted_vars = [original_vars[i] for i in perm]
    permuted_label['variables'] = permuted_vars
    
    # Store the permutation for later reversal
    permuted_label['_permutation'] = perm
    permuted_label['_original_vars'] = original_vars
    
    return permuted_label


class VariablePermutationTrainer(FixedEnhancedBCTrainer):
    """
    BC trainer that uses random variable permutations during training.
    
    Key insight: By randomly permuting variable order each epoch, we ensure
    that X4 (and all variables) appear equally often at each position,
    solving the class imbalance problem.
    """
    
    def __init__(self, 
                 permute_every_epoch: bool = True,
                 permute_every_batch: bool = False,
                 track_permutation_stats: bool = True,
                 **kwargs):
        """
        Initialize variable permutation trainer.
        
        Args:
            permute_every_epoch: Permute variables once per epoch
            permute_every_batch: Permute variables for each batch (more aggressive)
            track_permutation_stats: Track statistics about permutations
            **kwargs: Arguments passed to parent trainer
        """
        super().__init__(**kwargs)
        self.permute_every_epoch = permute_every_epoch
        self.permute_every_batch = permute_every_batch
        self.track_permutation_stats = track_permutation_stats
        self.rng = np.random.RandomState(self.seed)
        
        # Track permutation statistics
        if track_permutation_stats:
            self.permutation_stats = {
                'position_counts': {},  # How often each variable appears at each position
                'total_permutations': 0
            }
    
    def _apply_permutation(self, inputs: List[jnp.ndarray], 
                          labels: List[Dict[str, Any]]) -> Tuple[List[jnp.ndarray], List[Dict[str, Any]]]:
        """Apply random permutation to a batch of inputs and labels."""
        
        permuted_inputs = []
        permuted_labels = []
        
        for input_tensor, label in zip(inputs, labels):
            # Get original variables
            original_vars = label.get('variables', ['X0', 'X1', 'X2', 'X3', 'X4'])
            n_vars = len(original_vars)
            
            # Generate random permutation
            perm = create_random_permutation(n_vars, self.rng)
            
            # Apply permutation to tensor
            permuted_tensor = apply_permutation_to_tensor(input_tensor, perm)
            permuted_inputs.append(permuted_tensor)
            
            # Apply permutation to label
            permuted_label = apply_permutation_to_label(label, perm, original_vars)
            permuted_labels.append(permuted_label)
            
            # Track statistics
            if self.track_permutation_stats:
                for pos, var_idx in enumerate(perm):
                    var_name = original_vars[var_idx]
                    if var_name not in self.permutation_stats['position_counts']:
                        self.permutation_stats['position_counts'][var_name] = [0] * n_vars
                    self.permutation_stats['position_counts'][var_name][pos] += 1
                self.permutation_stats['total_permutations'] += 1
        
        return permuted_inputs, permuted_labels
    
    def _train_epoch(self, inputs: List[jnp.ndarray], 
                     labels: List[Dict[str, Any]], epoch: int) -> float:
        """Train one epoch with optional permutation."""
        
        # Apply permutation at epoch level if requested
        if self.permute_every_epoch:
            logger.info(f"Applying random permutation for epoch {epoch+1}")
            inputs, labels = self._apply_permutation(inputs, labels)
        
        # Shuffle data
        n_samples = len(inputs)
        indices = np.random.permutation(n_samples)
        
        total_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_inputs = [inputs[j] for j in batch_indices]
            batch_labels = [labels[j] for j in batch_indices]
            
            # Apply permutation at batch level if requested
            if self.permute_every_batch:
                batch_inputs, batch_labels = self._apply_permutation(batch_inputs, batch_labels)
            
            self.key, batch_key = jax.random.split(self.key)
            batch_loss = self._train_batch(batch_inputs, batch_labels, batch_key)
            
            total_loss += batch_loss
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def _train_batch(self, batch_inputs: List[jnp.ndarray], 
                     batch_labels: List[Dict[str, Any]], 
                     rng_key: jax.Array) -> float:
        """Train on a batch with permuted variables."""
        
        def loss_fn(params):
            batch_loss = 0.0
            valid_examples = 0
            
            for input_tensor, label in zip(batch_inputs, batch_labels):
                outputs = self.net.apply(params, rng_key, input_tensor, self.target_idx)
                var_logits = outputs['variable_logits']
                value_params = outputs['value_params']
                
                target_vars = list(label['targets'])
                if not target_vars:
                    continue
                    
                target_var_name = target_vars[0]
                target_value = label['values'][target_var_name]
                
                # Use the PERMUTED variable list
                example_variables = label.get('variables', [])
                if not example_variables:
                    continue
                
                # Create mapper with permuted variables
                example_mapper = VariableMapper(
                    variables=example_variables,
                    target_variable=label.get('target_variable')
                )
                
                # Get index in PERMUTED space
                try:
                    var_idx = example_mapper.get_index(target_var_name)
                except ValueError:
                    continue
                
                # Compute loss as usual
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
                
                batch_loss += example_loss
                valid_examples += 1
            
            if valid_examples > 0:
                return batch_loss / valid_examples
            else:
                return jnp.array(1e-6)
        
        # Standard gradient update
        loss_val, grads = jax.value_and_grad(loss_fn)(self.model_params)
        
        if not jnp.isfinite(loss_val):
            logger.warning(f"Non-finite loss detected: {loss_val}, skipping batch")
            return 1.0
        
        import optax
        grads, _ = optax.clip_by_global_norm(self.gradient_clip).update(grads, None)
        
        updates, self.optimizer_state = self.optimizer.update(
            grads, self.optimizer_state, self.model_params
        )
        self.model_params = optax.apply_updates(self.model_params, updates)
        
        return float(loss_val)
    
    def train(self, *args, **kwargs):
        """Train with variable permutation and show statistics."""
        result = super().train(*args, **kwargs)
        
        # Show permutation statistics
        if self.track_permutation_stats and self.permutation_stats['total_permutations'] > 0:
            print("\n" + "="*80)
            print("VARIABLE PERMUTATION STATISTICS")
            print("="*80)
            
            print(f"\nTotal permutations applied: {self.permutation_stats['total_permutations']}")
            
            print("\nPosition distribution for each variable:")
            print("(Shows how often each variable appeared at each position)")
            print("\nVariable | Pos 0 | Pos 1 | Pos 2 | Pos 3 | Pos 4 |")
            print("-" * 55)
            
            for var_name in sorted(self.permutation_stats['position_counts'].keys()):
                counts = self.permutation_stats['position_counts'][var_name]
                total = sum(counts)
                if total > 0:
                    percentages = [c/total*100 for c in counts]
                    print(f"{var_name:8s} | {percentages[0]:5.1f}% | {percentages[1]:5.1f}% | "
                          f"{percentages[2]:5.1f}% | {percentages[3]:5.1f}% | {percentages[4]:5.1f}% |")
            
            # Check if distribution is roughly uniform
            print("\nDistribution analysis:")
            all_uniform = True
            for var_name, counts in self.permutation_stats['position_counts'].items():
                total = sum(counts)
                if total > 0:
                    expected = total / len(counts)
                    max_deviation = max(abs(c - expected) / expected for c in counts)
                    if max_deviation > 0.2:  # More than 20% deviation
                        print(f"  ⚠️ {var_name}: Non-uniform distribution (max deviation: {max_deviation:.1%})")
                        all_uniform = False
            
            if all_uniform:
                print("  ✓ All variables have roughly uniform position distribution")
                print("  This should solve the X4 frequency problem!")
        
        result['permutation_stats'] = self.permutation_stats if self.track_permutation_stats else None
        
        return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='BC training with variable permutation')
    parser.add_argument('--demo_path', default='expert_demonstrations/raw/raw_demonstrations')
    parser.add_argument('--max_demos', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--permute_epoch', action='store_true', default=True,
                       help='Permute variables once per epoch')
    parser.add_argument('--permute_batch', action='store_true',
                       help='Permute variables for each batch (more aggressive)')
    parser.add_argument('--output_dir', default='debugging-bc-training/permutation_output')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    print(f"Running variable permutation training:")
    print(f"  Permute per epoch: {args.permute_epoch}")
    print(f"  Permute per batch: {args.permute_batch}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Hidden dim: {args.hidden_dim}")
    
    trainer = VariablePermutationTrainer(
        permute_every_epoch=args.permute_epoch,
        permute_every_batch=args.permute_batch,
        track_permutation_stats=True,
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
    
    print(f"\nTraining complete!")
    print(f"Results saved to {args.output_dir}")