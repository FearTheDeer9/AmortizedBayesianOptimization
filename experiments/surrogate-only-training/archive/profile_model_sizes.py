#!/usr/bin/env python3
"""Profile different model sizes to determine optimal configuration."""

import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
import optax

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
from src.causal_bayes_opt.data_structures.scm import get_parents, get_variables
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.data_structures.sample import create_sample
from src.causal_bayes_opt.avici_integration.continuous.model import ContinuousParentSetPredictionModel
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
from src.causal_bayes_opt.utils.variable_mapping import VariableMapper


def profile_model(model_config: Dict, test_sizes: list = [10, 30, 50, 70, 100]):
    """Profile a model configuration on different SCM sizes."""
    print(f"\n{'='*60}")
    print(f"Profiling Model: hidden_dim={model_config['hidden_dim']}, "
          f"layers={model_config['num_layers']}, heads={model_config['num_heads']}")
    print(f"{'='*60}")
    
    results = {}
    rng_key = random.PRNGKey(42)
    factory = VariableSCMFactory(seed=42)
    
    for num_vars in test_sizes:
        print(f"\nTesting with {num_vars} variables...")
        
        # Create SCM
        scm = factory.create_variable_scm(
            num_variables=num_vars,
            structure_type='mixed',
            edge_density=0.3
        )
        
        mapper = VariableMapper(list(get_variables(scm)))
        variables = mapper.variables
        
        # Create buffer with some data
        buffer = ExperienceBuffer()
        obs_data = sample_from_linear_scm(scm, 20, seed=42)
        for sample in obs_data:
            buffer.add_observation(sample)
        
        # Initialize model
        tensor, _ = buffer_to_three_channel_tensor(buffer, variables[0])
        
        def surrogate_fn(x, target_idx, is_training):
            model = ContinuousParentSetPredictionModel(
                hidden_dim=model_config['hidden_dim'],
                num_heads=model_config['num_heads'],
                num_layers=model_config['num_layers'],
                dropout=0.1
            )
            return model(x, target_idx, is_training)
        
        try:
            # Start memory tracking
            tracemalloc.start()
            
            # Initialize
            rng_key, init_key = random.split(rng_key)
            surrogate_net = hk.transform(surrogate_fn)
            surrogate_params = surrogate_net.init(init_key, tensor, 0, True)
            
            # Count parameters
            param_count = sum(p.size for p in jax.tree_util.tree_leaves(surrogate_params))
            
            # Initialize optimizer
            optimizer = optax.adam(0.001)
            opt_state = optimizer.init(surrogate_params)
            
            # Time forward pass
            start_time = time.time()
            for _ in range(5):  # Average over 5 runs
                rng_key, fwd_key = random.split(rng_key)
                output = surrogate_net.apply(surrogate_params, fwd_key, tensor, 0, True)
            forward_time = (time.time() - start_time) / 5
            
            # Time backward pass
            start_time = time.time()
            for _ in range(5):
                rng_key, loss_key = random.split(rng_key)
                
                def loss_fn(params):
                    predictions = surrogate_net.apply(params, loss_key, tensor, 0, True)
                    pred_probs = predictions.get('parent_probabilities', jnp.zeros(num_vars))
                    # Simple dummy loss
                    return jnp.mean(pred_probs)
                
                loss, grads = jax.value_and_grad(loss_fn)(surrogate_params)
                updates, opt_state = optimizer.update(grads, opt_state, surrogate_params)
                surrogate_params = optax.apply_updates(surrogate_params, updates)
            
            backward_time = (time.time() - start_time) / 5
            
            # Get memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            results[num_vars] = {
                'param_count': param_count,
                'forward_time': forward_time,
                'backward_time': backward_time,
                'memory_mb': peak / 1024 / 1024,
                'success': True
            }
            
            print(f"  ✓ Parameters: {param_count:,}")
            print(f"  ✓ Forward pass: {forward_time*1000:.2f}ms")
            print(f"  ✓ Backward pass: {backward_time*1000:.2f}ms")
            print(f"  ✓ Peak memory: {peak/1024/1024:.1f}MB")
            
        except Exception as e:
            tracemalloc.stop()
            results[num_vars] = {
                'success': False,
                'error': str(e)
            }
            print(f"  ✗ Failed: {e}")
            
    return results


def main():
    print("\n" + "="*70)
    print("MODEL SIZE PROFILING FOR SURROGATE TRAINING")
    print("="*70)
    
    # Define model configurations
    model_configs = {
        'small': {
            'hidden_dim': 256,
            'num_layers': 4,
            'num_heads': 4
        },
        'medium': {
            'hidden_dim': 512,
            'num_layers': 6,
            'num_heads': 8
        },
        'large': {
            'hidden_dim': 768,
            'num_layers': 8,
            'num_heads': 12
        },
        'xlarge': {
            'hidden_dim': 1024,
            'num_layers': 10,
            'num_heads': 16
        }
    }
    
    # Test sizes - include the challenging ones
    test_sizes = [10, 30, 50, 70, 100]
    
    all_results = {}
    
    for name, config in model_configs.items():
        all_results[name] = profile_model(config, test_sizes)
    
    # Summary and recommendations
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    # Compare models
    print("\nModel Comparison (100 variables):")
    print("-" * 50)
    print(f"{'Model':<10} {'Params':<12} {'Fwd(ms)':<10} {'Bwd(ms)':<10} {'Mem(MB)':<10}")
    print("-" * 50)
    
    for name in model_configs:
        if 100 in all_results[name] and all_results[name][100]['success']:
            r = all_results[name][100]
            print(f"{name:<10} {r['param_count']:>11,} {r['forward_time']*1000:>9.1f} "
                  f"{r['backward_time']*1000:>9.1f} {r['memory_mb']:>9.1f}")
        else:
            print(f"{name:<10} {'FAILED':>11}")
    
    # Scaling analysis
    print("\nScaling Analysis:")
    print("-" * 50)
    
    for name in model_configs:
        successful_sizes = [s for s in test_sizes 
                          if s in all_results[name] and all_results[name][s]['success']]
        if successful_sizes:
            max_size = max(successful_sizes)
            
            # Check if time scales reasonably
            if 10 in successful_sizes and max_size in successful_sizes:
                time_ratio = (all_results[name][max_size]['backward_time'] / 
                            all_results[name][10]['backward_time'])
                print(f"{name}: Max size={max_size}, Time scaling={time_ratio:.1f}x")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    # Find best model for 100 variables
    valid_models = []
    for name in model_configs:
        if 100 in all_results[name] and all_results[name][100]['success']:
            r = all_results[name][100]
            score = r['backward_time']  # Optimize for speed
            valid_models.append((name, score, r))
    
    if valid_models:
        valid_models.sort(key=lambda x: x[1])
        best_model = valid_models[0][0]
        
        print(f"\n✅ RECOMMENDED: '{best_model}' model")
        print(f"   - Handles 100 variables successfully")
        print(f"   - Training time: {valid_models[0][2]['backward_time']*1000:.1f}ms per step")
        print(f"   - Memory usage: {valid_models[0][2]['memory_mb']:.1f}MB")
        
        if len(valid_models) > 1:
            print(f"\n🔄 ALTERNATIVE: '{valid_models[1][0]}' model")
            print(f"   - Use if you need better accuracy and have more time")
            print(f"   - {(valid_models[1][1]/valid_models[0][1]):.1f}x slower but potentially more accurate")
    
    # Memory constraints
    print("\n💾 Memory Considerations:")
    for name in ['medium', 'large', 'xlarge']:
        if 100 in all_results[name] and all_results[name][100]['success']:
            mem = all_results[name][100]['memory_mb']
            print(f"   {name}: ~{mem:.0f}MB per model instance")
    
    # Training time estimates
    print("\n⏰ Training Time Estimates (100-var curriculum):")
    print("   Assuming ~10,000 gradient steps total:")
    for name in ['medium', 'large', 'xlarge']:
        if 100 in all_results[name] and all_results[name][100]['success']:
            time_per_step = all_results[name][100]['backward_time']
            total_hours = (time_per_step * 10000) / 3600
            print(f"   {name}: ~{total_hours:.1f} hours")


if __name__ == '__main__':
    main()