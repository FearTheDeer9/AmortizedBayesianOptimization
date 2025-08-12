#!/usr/bin/env python3
"""
Debug X4 predictions by examining the actual probability distributions
when X4 is the target variable.
"""

import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import pickle

sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.training.data_preprocessing import load_demonstrations_from_path
from src.causal_bayes_opt.training.demonstration_to_tensor import create_bc_training_dataset
from src.causal_bayes_opt.utils.variable_mapping import VariableMapper

def analyze_x4_predictions():
    """Analyze predictions specifically when X4 is the target."""
    
    print("="*80)
    print("X4 PREDICTION ANALYSIS")
    print("="*80)
    
    # Load the trained model parameters
    results_dir = Path("debugging-bc-training/results_fixed")
    if not results_dir.exists():
        results_dir = Path("debugging-bc-training/results")
    
    params_file = results_dir / "model_params.pkl"
    if not params_file.exists():
        print(f"Model params not found at {params_file}")
        return
    
    print(f"Loading model from {params_file}")
    with open(params_file, 'rb') as f:
        model_data = pickle.load(f)
        model_params = model_data['params']
    
    # Load demonstrations and create dataset
    print("\nLoading demonstrations...")
    demos_path = 'expert_demonstrations/raw/raw_demonstrations'
    raw_demos = load_demonstrations_from_path(demos_path, max_files=100)
    
    # Flatten
    flat_demos = []
    for item in raw_demos:
        if hasattr(item, 'demonstrations'):
            flat_demos.extend(item.demonstrations)
        else:
            flat_demos.append(item)
    
    # Create dataset
    all_inputs, all_labels, metadata = create_bc_training_dataset(
        flat_demos[:100], max_trajectory_length=100
    )
    
    print(f"Created {len(all_inputs)} examples")
    
    # Find X4 target examples
    x4_examples = []
    for i, label in enumerate(all_labels):
        if 'targets' in label and label['targets']:
            target_var = list(label['targets'])[0]
            if target_var == 'X4':
                x4_examples.append((i, all_inputs[i], label))
    
    print(f"\nFound {len(x4_examples)} examples where X4 is the target")
    
    if not x4_examples:
        print("No X4 examples found!")
        return
    
    # Initialize model
    from src.causal_bayes_opt.policies.acquisition_policy import AcquisitionPolicy
    
    policy = AcquisitionPolicy(
        hidden_dim=256,
        encoder_layers=3,
        target_variable='Y',
        seed=42
    )
    
    # Initialize with dummy input
    dummy_input = all_inputs[0]
    target_idx = 4  # Y is at index 4
    key = jax.random.PRNGKey(42)
    _ = policy.net.init(key, dummy_input, target_idx)
    
    print("\n" + "="*60)
    print("ANALYZING X4 PREDICTIONS")
    print("="*60)
    
    # Analyze each X4 example
    for idx, (example_idx, input_tensor, label) in enumerate(x4_examples[:10]):  # First 10
        print(f"\n--- Example {idx+1} (index {example_idx}) ---")
        
        # Get model predictions
        outputs = policy.net.apply(model_params, key, input_tensor, target_idx)
        var_logits = outputs['variable_logits']
        var_probs = jax.nn.softmax(var_logits)
        
        # Get variable names and mapper
        variables = label.get('variables', ['X0', 'X1', 'X2', 'X3', 'X4'])
        mapper = VariableMapper(
            variables=variables,
            target_variable=label.get('target_variable', 'Y')
        )
        
        # Get true target index
        target_var = list(label['targets'])[0]
        true_idx = mapper.get_index(target_var)
        
        print(f"True target: {target_var} (index {true_idx})")
        print(f"Target value: {label['values'][target_var]:.3f}")
        
        # Show probability distribution
        print("\nPredicted probabilities:")
        sorted_indices = np.argsort(var_probs)[::-1]
        for rank, var_idx in enumerate(sorted_indices):
            var_name = variables[var_idx] if var_idx < len(variables) else f"idx_{var_idx}"
            prob = var_probs[var_idx]
            marker = " <-- TRUE" if var_idx == true_idx else ""
            print(f"  {rank+1}. {var_name}: {prob:.4f} (logit={var_logits[var_idx]:.3f}){marker}")
        
        # Analyze the problem
        predicted_idx = int(jnp.argmax(var_logits))
        predicted_var = variables[predicted_idx] if predicted_idx < len(variables) else f"idx_{predicted_idx}"
        
        print(f"\nPredicted: {predicted_var} (prob={var_probs[predicted_idx]:.4f})")
        print(f"X4 probability: {var_probs[true_idx]:.4f}")
        print(f"X4 rank: {list(sorted_indices).index(true_idx) + 1} out of {len(variables)}")
        
        # Check if X4 is at least getting some probability
        if var_probs[true_idx] < 0.01:
            print("⚠️  X4 has <1% probability!")
        elif var_probs[true_idx] < 0.1:
            print("⚠️  X4 has <10% probability")
        
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS FOR ALL X4 EXAMPLES")
    print("="*60)
    
    x4_probs = []
    x4_ranks = []
    predictions = []
    
    for example_idx, input_tensor, label in x4_examples:
        outputs = policy.net.apply(model_params, key, input_tensor, target_idx)
        var_logits = outputs['variable_logits']
        var_probs = jax.nn.softmax(var_logits)
        
        variables = label.get('variables', ['X0', 'X1', 'X2', 'X3', 'X4'])
        mapper = VariableMapper(
            variables=variables,
            target_variable=label.get('target_variable', 'Y')
        )
        
        target_var = list(label['targets'])[0]
        true_idx = mapper.get_index(target_var)
        
        # Get X4 probability and rank
        x4_prob = var_probs[true_idx]
        x4_probs.append(float(x4_prob))
        
        sorted_indices = np.argsort(var_probs)[::-1]
        rank = list(sorted_indices).index(true_idx) + 1
        x4_ranks.append(rank)
        
        # Get prediction
        pred_idx = int(jnp.argmax(var_logits))
        pred_var = variables[pred_idx] if pred_idx < len(variables) else f"idx_{pred_idx}"
        predictions.append(pred_var)
    
    if x4_probs:
        print(f"\nX4 Probability Statistics ({len(x4_probs)} examples):")
        print(f"  Mean: {np.mean(x4_probs):.4f}")
        print(f"  Median: {np.median(x4_probs):.4f}")
        print(f"  Max: {np.max(x4_probs):.4f}")
        print(f"  Min: {np.min(x4_probs):.4f}")
        print(f"  Std: {np.std(x4_probs):.4f}")
        
        print(f"\nX4 Rank Statistics:")
        print(f"  Mean rank: {np.mean(x4_ranks):.1f} / 5")
        print(f"  Best rank: {np.min(x4_ranks)}")
        print(f"  Worst rank: {np.max(x4_ranks)}")
        print(f"  Times ranked 1st: {sum(1 for r in x4_ranks if r == 1)}")
        print(f"  Times ranked top 2: {sum(1 for r in x4_ranks if r <= 2)}")
        
        print(f"\nPrediction Distribution (when X4 is true):")
        from collections import Counter
        pred_counts = Counter(predictions)
        for var, count in pred_counts.most_common():
            pct = count / len(predictions) * 100
            print(f"  {var}: {count} ({pct:.1f}%)")
        
        # Check if it's close
        close_calls = sum(1 for p in x4_probs if p > 0.15)
        print(f"\nClose calls (X4 prob > 0.15): {close_calls} / {len(x4_probs)}")
        
        very_close = sum(1 for p in x4_probs if p > 0.25)
        print(f"Very close (X4 prob > 0.25): {very_close} / {len(x4_probs)}")

if __name__ == "__main__":
    analyze_x4_predictions()