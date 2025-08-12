#!/usr/bin/env python3
"""
Gradient analyzer for BC training.
Tracks gradient magnitudes, variance, and flow to diagnose training issues.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import optax
import logging

sys.path.append(str(Path(__file__).parent.parent))

from enhanced_bc_trainer_fixed import FixedEnhancedBCTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradientAnalyzer:
    """Tracks and analyzes gradients during training."""
    
    def __init__(self):
        self.gradient_history = {
            'batch_norms': [],
            'batch_norms_clipped': [],
            'layer_norms': {},
            'gradient_variances': [],
            'loss_values': [],
            'learning_signals': []
        }
        
    def analyze_gradients(self, grads, params, loss_val):
        """Analyze gradient statistics."""
        stats = {}
        
        # Overall gradient norm
        grad_norm = optax.global_norm(grads)
        stats['global_norm'] = float(grad_norm)
        
        # Per-layer analysis
        layer_stats = {}
        for path, grad in jax.tree_util.tree_flatten_with_path(grads)[0]:
            layer_name = '/'.join(str(p) for p in path)
            
            if grad is not None and grad.size > 0:
                layer_norm = float(jnp.linalg.norm(grad))
                layer_mean = float(jnp.mean(jnp.abs(grad)))
                layer_std = float(jnp.std(grad))
                layer_max = float(jnp.max(jnp.abs(grad)))
                
                # Get corresponding parameter for relative magnitude
                param = jax.tree_util.tree_leaves([params])[0]  # Simplified
                param_norm = float(jnp.linalg.norm(param)) if hasattr(param, 'shape') else 1.0
                
                layer_stats[layer_name] = {
                    'norm': layer_norm,
                    'mean_abs': layer_mean,
                    'std': layer_std,
                    'max_abs': layer_max,
                    'relative_norm': layer_norm / (param_norm + 1e-8)
                }
        
        stats['layer_stats'] = layer_stats
        stats['loss'] = float(loss_val)
        
        # Learning signal strength (gradient norm relative to loss)
        stats['learning_signal'] = grad_norm / (abs(loss_val) + 1e-8)
        
        return stats
    
    def update_history(self, stats, clipped_norm=None):
        """Update gradient history."""
        self.gradient_history['batch_norms'].append(stats['global_norm'])
        if clipped_norm is not None:
            self.gradient_history['batch_norms_clipped'].append(float(clipped_norm))
        
        self.gradient_history['loss_values'].append(stats['loss'])
        self.gradient_history['learning_signals'].append(stats['learning_signal'])
        
        # Track per-layer norms
        for layer_name, layer_stat in stats['layer_stats'].items():
            if layer_name not in self.gradient_history['layer_norms']:
                self.gradient_history['layer_norms'][layer_name] = []
            self.gradient_history['layer_norms'][layer_name].append(layer_stat['norm'])
    
    def get_summary(self):
        """Get summary statistics."""
        summary = {}
        
        if self.gradient_history['batch_norms']:
            norms = np.array(self.gradient_history['batch_norms'])
            summary['gradient_norm'] = {
                'mean': float(np.mean(norms)),
                'std': float(np.std(norms)),
                'min': float(np.min(norms)),
                'max': float(np.max(norms)),
                'median': float(np.median(norms))
            }
            
            # Check for vanishing/exploding gradients
            if summary['gradient_norm']['mean'] < 1e-4:
                summary['gradient_issue'] = 'VANISHING'
            elif summary['gradient_norm']['mean'] > 10:
                summary['gradient_issue'] = 'EXPLODING'
            else:
                summary['gradient_issue'] = 'NORMAL'
        
        if self.gradient_history['batch_norms_clipped']:
            clipped = np.array(self.gradient_history['batch_norms_clipped'])
            original = np.array(self.gradient_history['batch_norms'][:len(clipped)])
            summary['clipping_ratio'] = float(np.mean(clipped / (original + 1e-8)))
        
        if self.gradient_history['learning_signals']:
            signals = np.array(self.gradient_history['learning_signals'])
            summary['learning_signal'] = {
                'mean': float(np.mean(signals)),
                'std': float(np.std(signals))
            }
        
        # Loss variance (indicates training stability)
        if len(self.gradient_history['loss_values']) > 10:
            recent_losses = self.gradient_history['loss_values'][-10:]
            summary['loss_variance'] = float(np.var(recent_losses))
            summary['loss_trend'] = 'decreasing' if recent_losses[-1] < recent_losses[0] else 'increasing'
        
        return summary


class GradientAnalyzerTrainer(FixedEnhancedBCTrainer):
    """BC trainer with gradient analysis."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gradient_analyzer = GradientAnalyzer()
        self.log_frequency = kwargs.get('log_frequency', 10)
        
    def _train_batch(self, batch_inputs: List[jnp.ndarray], 
                     batch_labels: List[Dict[str, Any]], 
                     rng_key: jax.Array) -> float:
        """Train batch with gradient analysis."""
        
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
                
                example_variables = label.get('variables', [])
                if not example_variables:
                    continue
                
                from src.causal_bayes_opt.utils.variable_mapping import VariableMapper
                    
                example_mapper = VariableMapper(
                    variables=example_variables,
                    target_variable=label.get('target_variable')
                )
                
                try:
                    var_idx = example_mapper.get_index(target_var_name)
                except ValueError:
                    continue
                
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
        
        # Compute gradients
        loss_val, grads = jax.value_and_grad(loss_fn)(self.model_params)
        
        # Analyze gradients BEFORE clipping
        grad_stats = self.gradient_analyzer.analyze_gradients(grads, self.model_params, loss_val)
        
        # Clip gradients
        clipped_grads, _ = optax.clip_by_global_norm(self.gradient_clip).update(grads, None)
        clipped_norm = optax.global_norm(clipped_grads)
        
        # Update history
        self.gradient_analyzer.update_history(grad_stats, clipped_norm)
        
        # Log periodically
        if len(self.gradient_analyzer.gradient_history['batch_norms']) % self.log_frequency == 0:
            summary = self.gradient_analyzer.get_summary()
            logger.info(f"\n{'='*60}")
            logger.info("GRADIENT ANALYSIS")
            logger.info(f"{'='*60}")
            logger.info(f"Gradient norm: {summary['gradient_norm']['mean']:.6f} ± {summary['gradient_norm']['std']:.6f}")
            logger.info(f"Gradient status: {summary['gradient_issue']}")
            
            if 'clipping_ratio' in summary:
                logger.info(f"Clipping ratio: {summary['clipping_ratio']:.3f}")
                if summary['clipping_ratio'] < 0.5:
                    logger.warning("⚠️ Aggressive clipping detected - gradients being cut by >50%")
            
            if 'learning_signal' in summary:
                logger.info(f"Learning signal: {summary['learning_signal']['mean']:.6f}")
                if summary['learning_signal']['mean'] < 0.01:
                    logger.warning("⚠️ Weak learning signal - consider increasing learning rate")
            
            if 'loss_variance' in summary:
                logger.info(f"Loss variance: {summary['loss_variance']:.6f}")
                logger.info(f"Loss trend: {summary['loss_trend']}")
                if summary['loss_variance'] > 1.0:
                    logger.warning("⚠️ High loss variance - training may be unstable")
        
        # Apply updates
        updates, self.optimizer_state = self.optimizer.update(
            clipped_grads, self.optimizer_state, self.model_params
        )
        self.model_params = optax.apply_updates(self.model_params, updates)
        
        return float(loss_val)
    
    def train(self, *args, **kwargs):
        """Train and provide final gradient analysis."""
        result = super().train(*args, **kwargs)
        
        # Final gradient analysis
        print("\n" + "="*80)
        print("FINAL GRADIENT ANALYSIS REPORT")
        print("="*80)
        
        summary = self.gradient_analyzer.get_summary()
        
        print(f"\nGradient Statistics:")
        if 'gradient_norm' in summary:
            stats = summary['gradient_norm']
            print(f"  Mean norm: {stats['mean']:.6f}")
            print(f"  Std dev: {stats['std']:.6f}")
            print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
            print(f"  Status: {summary['gradient_issue']}")
        
        print(f"\nRecommendations:")
        if summary.get('gradient_issue') == 'VANISHING':
            print("  ⚠️ VANISHING GRADIENTS DETECTED")
            print("  Try:")
            print("  - Increase learning rate (current: 3e-4 → try 1e-3 or 3e-3)")
            print("  - Reduce model depth or use skip connections")
            print("  - Check for dead ReLUs or saturated activations")
            print("  - Remove or reduce gradient clipping")
        elif summary.get('gradient_issue') == 'EXPLODING':
            print("  ⚠️ EXPLODING GRADIENTS DETECTED")
            print("  Try:")
            print("  - Decrease learning rate")
            print("  - Increase gradient clipping (current: 1.0)")
            print("  - Check for numerical instabilities")
            print("  - Use gradient normalization")
        else:
            print("  ✓ Gradient flow appears normal")
        
        if summary.get('clipping_ratio', 1.0) < 0.5:
            print(f"\n  ⚠️ Gradients being clipped aggressively ({summary['clipping_ratio']:.1%} of original)")
            print("  Consider increasing gradient_clip value")
        
        if summary.get('learning_signal', {}).get('mean', 1.0) < 0.01:
            print(f"\n  ⚠️ Weak learning signal ({summary['learning_signal']['mean']:.6f})")
            print("  Model may not be learning effectively")
        
        # Add gradient summary to results
        result['gradient_analysis'] = summary
        
        return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='BC training with gradient analysis')
    parser.add_argument('--demo_path', default='expert_demonstrations/raw/raw_demonstrations')
    parser.add_argument('--max_demos', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--output_dir', default='debugging-bc-training/gradient_analysis_output')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    print(f"Running gradient analysis with:")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Gradient clip: {args.gradient_clip}")
    
    trainer = GradientAnalyzerTrainer(
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        gradient_clip=args.gradient_clip,
        seed=args.seed
    )
    
    results = trainer.train(
        demonstrations_path=args.demo_path,
        max_demos=args.max_demos,
        output_dir=args.output_dir
    )
    
    print(f"\nTraining complete!")
    print(f"Results saved to {args.output_dir}")