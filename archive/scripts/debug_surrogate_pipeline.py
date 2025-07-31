#!/usr/bin/env python3
"""Debug the complete surrogate pipeline from prediction to F1 calculation."""

import logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

from pathlib import Path
import jax.numpy as jnp

from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
from src.causal_bayes_opt.evaluation.model_interfaces import create_bc_surrogate
from src.causal_bayes_opt.data_structures.scm import get_parents, get_variables

# Create SCM
scm = create_fork_scm(noise_scale=1.0)
variables = list(get_variables(scm))
target = scm.get('target')
true_parents = list(get_parents(scm, target))

print("\n=== SCM INFORMATION ===")
print(f"Variables: {variables}")
print(f"Target: {target}")
print(f"True parents of {target}: {true_parents}")

# Create some test data
buffer = ExperienceBuffer()
samples = sample_from_linear_scm(scm, 20, seed=42)
for sample in samples:
    buffer.add_observation(sample)

# Convert to tensor
tensor, var_order = buffer_to_three_channel_tensor(buffer, target, standardize=True)
print(f"\nTensor variable order: {var_order}")
print(f"Tensor shape: {tensor.shape}")

# Load BC surrogate
checkpoint_path = Path('checkpoints/validation/bc_surrogate_final')
bc_surrogate_fn, _ = create_bc_surrogate(checkpoint_path, allow_updates=False)

print("\n=== BC SURROGATE PREDICTION ===")
# Call surrogate with variable names
if hasattr(bc_surrogate_fn, 'predict_with_variables'):
    posterior = bc_surrogate_fn.predict_with_variables(tensor, target, var_order)
    print("Called with variable names")
else:
    posterior = bc_surrogate_fn(tensor, target, var_order)
    print("Called with standard signature")
print(f"Posterior type: {type(posterior)}")

# Check what's in the posterior
if hasattr(posterior, 'metadata'):
    print(f"Has metadata: {type(posterior.metadata)}")
    if 'marginal_parent_probs' in posterior.metadata:
        marginals = posterior.metadata['marginal_parent_probs']
        print(f"\nMarginal parent probabilities:")
        for var, prob in marginals.items():
            is_parent = var in true_parents
            print(f"  {var}: {prob:.3f} {'✓ TRUE PARENT' if is_parent else ''}")

# Now simulate what the universal evaluator does
print("\n=== UNIVERSAL EVALUATOR EXTRACTION ===")
from src.causal_bayes_opt.avici_integration.parent_set.posterior import ParentSetPosterior

if isinstance(posterior, ParentSetPosterior):
    print("Posterior is ParentSetPosterior")
    
    # Check metadata extraction
    if hasattr(posterior, 'metadata') and 'marginal_parent_probs' in posterior.metadata:
        raw_marginals = dict(posterior.metadata['marginal_parent_probs'])
        print(f"\nRaw marginals from metadata: {raw_marginals}")
        
        # Apply variable mapping
        marginals = {}
        for var_name, prob in raw_marginals.items():
            if var_name.startswith('X') and var_name[1:].isdigit():
                idx = int(var_name[1:])
                if idx < len(var_order):
                    actual_name = var_order[idx]
                    marginals[actual_name] = prob
                    print(f"  Mapped {var_name} (index {idx}) -> {actual_name}")
            else:
                if var_name in var_order:
                    marginals[var_name] = prob
        
        print(f"\nMapped marginals: {marginals}")
        
        # Check F1 calculation
        print("\n=== F1 CALCULATION ===")
        threshold = 0.5
        predicted_parents = {var for var, prob in marginals.items() if prob > threshold}
        print(f"Predicted parents (prob > {threshold}): {predicted_parents}")
        print(f"True parents: {set(true_parents)}")
        
        # Calculate metrics
        true_parents_set = set(true_parents)
        tp = len(predicted_parents & true_parents_set)
        fp = len(predicted_parents - true_parents_set)
        fn = len(true_parents_set - predicted_parents)
        
        print(f"\nConfusion matrix:")
        print(f"  True positives: {tp}")
        print(f"  False positives: {fp}") 
        print(f"  False negatives: {fn}")
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0
            
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0
            
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
            
        print(f"\nMetrics:")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1 Score: {f1:.3f}")