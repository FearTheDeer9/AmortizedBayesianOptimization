#!/usr/bin/env python3
"""
Analyze what the model actually predicts when it should predict X2.
"""

import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.training.data_preprocessing import load_demonstrations_from_path
from demonstration_to_tensor_fixed import create_bc_training_dataset
from variable_mapping_fixed import VariableMapper
from policy_bc_trainer_fixed import PolicyBCTrainer

def analyze_x2_predictions():
    print("="*80)
    print("ANALYZING X2 PREDICTION ERRORS")
    print("="*80)
    
    # Load demonstrations
    demos_path = Path("../expert_demonstrations/raw/raw_demonstrations")
    if not demos_path.exists():
        demos_path = Path("expert_demonstrations/raw/raw_demonstrations")
    
    raw_demos = load_demonstrations_from_path(str(demos_path), max_files=100)
    
    # Flatten
    flat_demos = []
    for item in raw_demos:
        if hasattr(item, 'demonstrations'):
            flat_demos.extend(item.demonstrations)
        else:
            flat_demos.append(item)
    
    # Create dataset
    all_inputs, all_labels, metadata = create_bc_training_dataset(
        flat_demos, max_trajectory_length=100
    )
    
    # Create and initialize trainer
    trainer = PolicyBCTrainer(
        hidden_dim=128,
        learning_rate=3e-3,
        batch_size=32,
        seed=42
    )
    
    # Initialize model
    trainer._initialize_model(all_inputs[0], metadata)
    
    # Load trained weights if available (or use random initialization for analysis)
    checkpoint_path = Path("debug_training_results/policy_checkpoint.pkl")
    if checkpoint_path.exists():
        import pickle
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
            if 'params' in checkpoint:
                trainer.model_params = checkpoint['params']
                print("Loaded trained model parameters")
    else:
        print("Using random initialization (no checkpoint found)")
    
    # Analyze predictions for X2 targets
    x2_predictions = defaultdict(lambda: defaultdict(int))
    x2_by_scm_size = defaultdict(lambda: defaultdict(int))
    
    # Also track other variables for comparison
    other_predictions = defaultdict(lambda: defaultdict(int))
    
    print("\nAnalyzing predictions...")
    
    for i, (input_tensor, label) in enumerate(zip(all_inputs[:500], all_labels[:500])):
        variables = label.get('variables', [])
        targets = list(label.get('targets', []))
        
        if not targets:
            continue
            
        target_var = targets[0]
        n_vars = len(variables)
        
        # Get model prediction
        key = jax.random.PRNGKey(i)
        outputs = trainer.net.apply(trainer.model_params, key, input_tensor, trainer.target_idx)
        var_logits = outputs['variable_logits']
        
        # Get predicted variable
        pred_idx = int(jnp.argmax(var_logits))
        if pred_idx < len(variables):
            pred_var = variables[pred_idx]
        else:
            pred_var = f"OOB_{pred_idx}"
        
        # Track predictions
        if target_var == 'X2':
            x2_predictions[target_var][pred_var] += 1
            x2_by_scm_size[n_vars][pred_var] += 1
        else:
            other_predictions[target_var][pred_var] += 1
    
    # Print confusion matrix for X2
    print("\n" + "="*60)
    print("CONFUSION MATRIX FOR X2")
    print("="*60)
    
    x2_total = sum(x2_predictions['X2'].values())
    if x2_total > 0:
        print(f"\nWhen target is X2 ({x2_total} cases), model predicts:")
        for pred_var, count in sorted(x2_predictions['X2'].items(), key=lambda x: x[1], reverse=True):
            pct = (count / x2_total) * 100
            print(f"  {pred_var}: {count} times ({pct:.1f}%)")
    
    # Breakdown by SCM size
    print("\n" + "="*60)
    print("X2 PREDICTIONS BY SCM SIZE")
    print("="*60)
    
    for scm_size in sorted(x2_by_scm_size.keys()):
        predictions = x2_by_scm_size[scm_size]
        total = sum(predictions.values())
        if total > 0:
            print(f"\nIn {scm_size}-variable SCMs ({total} X2 targets):")
            for pred_var, count in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
                pct = (count / total) * 100
                print(f"  Predicts {pred_var}: {count} times ({pct:.1f}%)")
    
    # Compare with other variables
    print("\n" + "="*60)
    print("COMPARISON WITH OTHER VARIABLES")
    print("="*60)
    
    for target_var in ['X0', 'X1', 'X3']:
        predictions = other_predictions.get(target_var, {})
        total = sum(predictions.values())
        if total > 0:
            correct = predictions.get(target_var, 0)
            accuracy = (correct / total) * 100
            print(f"\n{target_var} accuracy: {accuracy:.1f}% ({correct}/{total})")
            
            # Show top mispredictions
            if accuracy < 100:
                print(f"  When target is {target_var}, model predicts:")
                for pred_var, count in sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]:
                    pct = (count / total) * 100
                    print(f"    {pred_var}: {pct:.1f}%")
    
    # The hypothesis
    print("\n" + "="*60)
    print("HYPOTHESIS: INDEX SHIFT PROBLEM")
    print("="*60)
    
    print("""
Based on the pattern, I suspect:
1. The model might be learning position-based patterns
2. X2 is at index 2 in the sorted list
3. But the model might be biased toward predicting indices 0 or 1
4. This could explain why X0 and X1 have better accuracy

Let's check if there's a systematic index shift...
""")
    
    # Check if predictions are shifted
    index_shifts = defaultdict(int)
    
    for i, (input_tensor, label) in enumerate(zip(all_inputs[:200], all_labels[:200])):
        variables = label.get('variables', [])
        targets = list(label.get('targets', []))
        
        if not targets or targets[0] != 'X2':
            continue
            
        target_var = targets[0]
        
        # Create mapper
        mapper = VariableMapper(variables, label.get('target_variable'))
        target_idx = mapper.get_index(target_var)
        
        # Get model prediction
        key = jax.random.PRNGKey(i + 1000)
        outputs = trainer.net.apply(trainer.model_params, key, input_tensor, trainer.target_idx)
        var_logits = outputs['variable_logits']
        pred_idx = int(jnp.argmax(var_logits))
        
        # Calculate shift
        shift = pred_idx - target_idx
        index_shifts[shift] += 1
    
    print("\nIndex shift analysis (predicted_idx - target_idx) for X2:")
    for shift, count in sorted(index_shifts.items()):
        print(f"  Shift {shift:+d}: {count} times")
    
    if -2 in index_shifts and index_shifts[-2] > len(index_shifts) // 3:
        print("\n⚠️ FOUND IT: Model consistently predicts index 0 when it should predict index 2!")

if __name__ == "__main__":
    analyze_x2_predictions()