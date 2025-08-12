#!/usr/bin/env python3
"""
Live debugging of X4 predictions during training.
Shows exact probabilities when X4 is the target.
"""

import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import logging

sys.path.append(str(Path(__file__).parent.parent))

from enhanced_bc_trainer_fixed import FixedEnhancedBCTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class X4DebugTrainer(FixedEnhancedBCTrainer):
    """Trainer that specifically debugs X4 predictions."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.x4_cases = []
        
    def _evaluate(self, val_inputs, val_labels):
        """Override evaluate to capture X4 predictions."""
        total_loss = 0.0
        n_examples = 0
        
        print("\n" + "="*60)
        print("CHECKING FOR X4 TARGETS IN VALIDATION")
        print("="*60)
        
        x4_found = False
        
        for val_input, val_label in zip(val_inputs, val_labels):
            self.key, eval_key = jax.random.split(self.key)
            
            # Forward pass
            outputs = self.net.apply(self.model_params, eval_key, val_input, self.target_idx)
            var_logits = outputs['variable_logits']
            var_probs = jax.nn.softmax(var_logits)
            value_params = outputs['value_params']
            
            # Check if this is an X4 target
            if 'targets' in val_label and val_label['targets']:
                target_var = list(val_label['targets'])[0]
                
                if target_var == 'X4':
                    x4_found = True
                    variables = val_label.get('variables', ['X0', 'X1', 'X2', 'X3', 'X4'])
                    
                    # Get true index
                    try:
                        true_idx = variables.index(target_var)
                    except ValueError:
                        continue
                    
                    print(f"\n🎯 Found X4 as target!")
                    print(f"Variables order: {variables}")
                    print(f"True target index: {true_idx}")
                    
                    # Show full probability distribution
                    print("\nProbability distribution:")
                    sorted_indices = np.argsort(var_probs)[::-1]
                    for rank, idx in enumerate(sorted_indices):
                        var_name = variables[idx] if idx < len(variables) else f"idx_{idx}"
                        prob = float(var_probs[idx])
                        logit = float(var_logits[idx])
                        marker = " <-- TARGET" if idx == true_idx else ""
                        print(f"  {rank+1}. {var_name}: prob={prob:.4f}, logit={logit:.3f}{marker}")
                    
                    # Key statistics
                    x4_prob = float(var_probs[true_idx])
                    x4_rank = list(sorted_indices).index(true_idx) + 1
                    predicted_idx = int(jnp.argmax(var_logits))
                    predicted_var = variables[predicted_idx] if predicted_idx < len(variables) else f"idx_{predicted_idx}"
                    
                    print(f"\nX4 probability: {x4_prob:.4f}")
                    print(f"X4 rank: {x4_rank}/5")
                    print(f"Model predicts: {predicted_var}")
                    print(f"Correct: {'✅' if predicted_idx == true_idx else '❌'}")
                    
                    self.x4_cases.append({
                        'prob': x4_prob,
                        'rank': x4_rank,
                        'predicted': predicted_var,
                        'correct': predicted_idx == true_idx
                    })
            
            # Calculate loss (same as parent)
            target_vars = list(val_label['targets'])
            if not target_vars:
                continue
                
            target_var_name = target_vars[0]
            target_value = val_label['values'][target_var_name]
            
            example_variables = val_label.get('variables', [])
            if not example_variables:
                continue
            
            from src.causal_bayes_opt.utils.variable_mapping import VariableMapper
            example_mapper = VariableMapper(
                variables=example_variables,
                target_variable=val_label.get('target_variable')
            )
            
            try:
                var_idx = example_mapper.get_index(target_var_name)
            except ValueError:
                continue
            
            # Compute loss
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
            
            total_loss += float(example_loss)
            n_examples += 1
        
        if not x4_found:
            print("No X4 targets found in this validation batch")
        
        return total_loss / n_examples if n_examples > 0 else 0.0
    
    def train(self, *args, **kwargs):
        """Train and show X4 summary at the end."""
        result = super().train(*args, **kwargs)
        
        # Show X4 summary
        if self.x4_cases:
            print("\n" + "="*80)
            print("X4 PREDICTION SUMMARY")
            print("="*80)
            
            probs = [c['prob'] for c in self.x4_cases]
            ranks = [c['rank'] for c in self.x4_cases]
            correct = sum(1 for c in self.x4_cases if c['correct'])
            
            print(f"\nTotal X4 cases seen: {len(self.x4_cases)}")
            print(f"Times predicted correctly: {correct} ({correct/len(self.x4_cases)*100:.1f}%)")
            
            print(f"\nX4 Probability stats:")
            print(f"  Mean: {np.mean(probs):.4f}")
            print(f"  Max: {np.max(probs):.4f}")
            print(f"  Min: {np.min(probs):.4f}")
            
            print(f"\nX4 Rank stats:")
            print(f"  Mean rank: {np.mean(ranks):.1f}/5")
            print(f"  Best rank: {np.min(ranks)}")
            print(f"  Worst rank: {np.max(ranks)}")
            
            # Distribution of predictions when X4 is target
            from collections import Counter
            predictions = Counter(c['predicted'] for c in self.x4_cases)
            print(f"\nWhat model predicts when X4 is target:")
            for var, count in predictions.most_common():
                print(f"  {var}: {count} times ({count/len(self.x4_cases)*100:.1f}%)")
        
        return result


if __name__ == "__main__":
    trainer = X4DebugTrainer(
        hidden_dim=256,
        learning_rate=3e-4,
        batch_size=32,
        max_epochs=20,  # Quick run
        seed=42
    )
    
    results = trainer.train(
        demonstrations_path='expert_demonstrations/raw/raw_demonstrations',
        max_demos=50,  # Limited for quick debugging
        output_dir='debugging-bc-training/x4_debug_output'
    )