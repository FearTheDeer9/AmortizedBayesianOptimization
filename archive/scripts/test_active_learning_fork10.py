#!/usr/bin/env python3
"""Test BC active learning on a 10-variable fork SCM with 50 interventions.

This test is designed to show the difference between static and active learning
surrogates. With 10 variables and only 2 true parents, a pre-trained surrogate
will struggle due to distribution shift, while active learning should adapt.
"""

import logging
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))

from src.causal_bayes_opt.experiments.test_scms import create_simple_linear_scm
from src.causal_bayes_opt.evaluation.universal_evaluator import create_universal_evaluator
from src.causal_bayes_opt.evaluation.model_interfaces import (
    create_random_acquisition,
    create_bc_surrogate,
    create_bc_active_learning_wrapper
)

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def create_fork_scm_size10(noise_scale: float = 1.0, seed: int = 42):
    """Create a 10-variable fork SCM: X1 -> Y <- X9, with X2-X8 as distractors."""
    np.random.seed(seed)
    
    # Variable names
    variables = [f'X{i}' for i in range(1, 10)] + ['Y']
    
    # Edges: Only X1 -> Y and X9 -> Y
    edges = [('X1', 'Y'), ('X9', 'Y')]
    
    # Coefficients for the edges
    coefficients = {
        ('X1', 'Y'): 2.0,
        ('X9', 'Y'): -1.5
    }
    
    # Noise scales for all variables
    noise_scales = {var: noise_scale for var in variables}
    
    # Random intercepts for distractors
    intercepts = {f'X{i}': np.random.randn() * 0.5 for i in range(1, 10)}
    intercepts['Y'] = 0.0
    
    # Create SCM using factory
    scm = create_simple_linear_scm(
        variables=variables,
        edges=edges,
        coefficients=coefficients,
        noise_scales=noise_scales,
        target='Y',
        intercepts=intercepts
    )
    
    logger.info(f"Created 10-variable fork SCM:")
    logger.info(f"  Variables: {variables}")
    logger.info(f"  True parents of Y: X1, X9")
    logger.info(f"  Coefficients: X1->Y: 2.0, X9->Y: -1.5")
    logger.info(f"  Distractors: X2, X3, X4, X5, X6, X7, X8")
    
    return scm

def plot_learning_curves(static_history, active_history, output_file='fork10_learning_curves.png'):
    """Plot F1 and prediction evolution over interventions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract F1 scores over time
    static_f1s = []
    active_f1s = []
    
    for step in static_history:
        if hasattr(step, 'f1_score') and step.f1_score is not None:
            static_f1s.append(step.f1_score)
        elif step.marginals:
            # Calculate F1 from marginals
            marginals = dict(step.marginals)
            predicted = [var for var, prob in marginals.items() if prob > 0.5]
            true_parents = {'X1', 'X9'}
            
            tp = len([p for p in predicted if p in true_parents])
            fp = len([p for p in predicted if p not in true_parents])
            fn = len(true_parents) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            static_f1s.append(f1)
    
    for step in active_history:
        if hasattr(step, 'f1_score') and step.f1_score is not None:
            active_f1s.append(step.f1_score)
        elif step.marginals:
            # Calculate F1 from marginals
            marginals = dict(step.marginals)
            predicted = [var for var, prob in marginals.items() if prob > 0.5]
            true_parents = {'X1', 'X9'}
            
            tp = len([p for p in predicted if p in true_parents])
            fp = len([p for p in predicted if p not in true_parents])
            fn = len(true_parents) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            active_f1s.append(f1)
    
    # Plot F1 evolution
    steps = list(range(1, len(static_f1s) + 1))
    ax1.plot(steps, static_f1s, 'b-', label='Static BC', linewidth=2)
    ax1.plot(steps, active_f1s, 'r-', label='Active BC', linewidth=2)
    ax1.set_xlabel('Intervention Step')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('Structure Learning Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot marginal probabilities for true parents
    static_x1_probs = []
    static_x9_probs = []
    active_x1_probs = []
    active_x9_probs = []
    
    for step in static_history:
        if step.marginals:
            marginals = dict(step.marginals)
            static_x1_probs.append(marginals.get('X1', 0))
            static_x9_probs.append(marginals.get('X9', 0))
    
    for step in active_history:
        if step.marginals:
            marginals = dict(step.marginals)
            active_x1_probs.append(marginals.get('X1', 0))
            active_x9_probs.append(marginals.get('X9', 0))
    
    steps = list(range(1, len(static_x1_probs) + 1))
    ax2.plot(steps, static_x1_probs, 'b--', label='Static X1', linewidth=2)
    ax2.plot(steps, static_x9_probs, 'b:', label='Static X9', linewidth=2)
    ax2.plot(steps, active_x1_probs, 'r--', label='Active X1', linewidth=2)
    ax2.plot(steps, active_x9_probs, 'r:', label='Active X9', linewidth=2)
    ax2.axhline(y=0.5, color='k', linestyle='-', alpha=0.3, label='Threshold')
    ax2.set_xlabel('Intervention Step')
    ax2.set_ylabel('Parent Probability')
    ax2.set_title('True Parent Probabilities')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    logger.info(f"Saved learning curves to {output_file}")

def main():
    """Test BC active learning on 10-variable fork SCM."""
    
    # Create large fork SCM
    scm = create_fork_scm_size10(noise_scale=1.0, seed=42)
    
    # BC surrogate checkpoint
    checkpoint_path = Path('checkpoints/validation/bc_surrogate_final')
    if not checkpoint_path.exists():
        logger.error(f"BC checkpoint not found at {checkpoint_path}")
        return
    
    # Evaluation config with 50 interventions
    eval_config = {
        'n_initial_obs': 100,  # More initial data
        'max_interventions': 50,  # Many interventions to see learning
        'n_intervention_samples': 50,
        'optimization_direction': 'MINIMIZE',
        'seed': 42
    }
    
    # Use random policy (not oracle)
    random_policy = create_random_acquisition(seed=42)
    evaluator = create_universal_evaluator()
    
    print("=" * 70)
    print("BC ACTIVE LEARNING TEST: 10-VARIABLE FORK SCM")
    print("=" * 70)
    print(f"SCM: 10 variables, true parents of Y: X1, X9")
    print(f"Task: {eval_config['max_interventions']} interventions")
    print(f"Challenge: Pre-trained surrogate must adapt to new distribution")
    print("=" * 70)
    
    # Test 1: Static BC
    print("\n1. STATIC BC SURROGATE (no learning)...")
    bc_static_predict, _ = create_bc_surrogate(checkpoint_path, allow_updates=False)
    
    static_result = evaluator.evaluate(
        acquisition_fn=random_policy,
        scm=scm,
        config=eval_config,
        surrogate_fn=bc_static_predict,
        seed=42
    )
    
    print(f"\nStatic Results:")
    print(f"  Final F1: {static_result.final_metrics['final_f1']:.3f}")
    print(f"  Final SHD: {static_result.final_metrics['final_shd']}")
    print(f"  Active learning: {static_result.final_metrics.get('active_learning_enabled', False)}")
    
    # Extract final predictions
    if static_result.history and static_result.history[-1].marginals:
        static_final_marginals = dict(static_result.history[-1].marginals)
        print(f"\n  Final predictions (top 5):")
        sorted_vars = sorted(static_final_marginals.items(), key=lambda x: x[1], reverse=True)[:5]
        for var, prob in sorted_vars:
            marker = "✓" if var in ['X1', 'X9'] else "✗"
            print(f"    {var}: {prob:.3f} {marker}")
    
    # Test 2: Active BC
    print("\n\n2. ACTIVE BC SURROGATE (with learning)...")
    predict_fn, update_fn, initial_params, initial_opt_state = create_bc_active_learning_wrapper(
        checkpoint_path, scm, learning_rate=1e-3, seed=42
    )
    
    active_result = evaluator.evaluate(
        acquisition_fn=random_policy,
        scm=scm,
        config=eval_config,
        surrogate_fn=predict_fn,
        surrogate_update_fn=update_fn,
        surrogate_params=initial_params,
        surrogate_opt_state=initial_opt_state,
        seed=42
    )
    
    print(f"\nActive Results:")
    print(f"  Final F1: {active_result.final_metrics['final_f1']:.3f}")
    print(f"  Final SHD: {active_result.final_metrics['final_shd']}")
    print(f"  Active learning: {active_result.final_metrics.get('active_learning_enabled', False)}")
    
    # Extract final predictions
    if active_result.history and active_result.history[-1].marginals:
        active_final_marginals = dict(active_result.history[-1].marginals)
        print(f"\n  Final predictions (top 5):")
        sorted_vars = sorted(active_final_marginals.items(), key=lambda x: x[1], reverse=True)[:5]
        for var, prob in sorted_vars:
            marker = "✓" if var in ['X1', 'X9'] else "✗"
            print(f"    {var}: {prob:.3f} {marker}")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    f1_improvement = active_result.final_metrics['final_f1'] - static_result.final_metrics['final_f1']
    shd_improvement = static_result.final_metrics['final_shd'] - active_result.final_metrics['final_shd']
    
    print(f"\nStructure Learning:")
    print(f"  F1 Score: {static_result.final_metrics['final_f1']:.3f} → "
          f"{active_result.final_metrics['final_f1']:.3f} "
          f"({'↑' if f1_improvement > 0 else '↓'}{abs(f1_improvement):.3f})")
    print(f"  SHD: {static_result.final_metrics['final_shd']} → "
          f"{active_result.final_metrics['final_shd']} "
          f"({'↓' if shd_improvement > 0 else '↑'}{abs(shd_improvement)})")
    
    # Check prediction evolution
    static_predictions_unique = len(set(
        str(dict(step.marginals)) for step in static_result.history if step.marginals
    ))
    active_predictions_unique = len(set(
        str(dict(step.marginals)) for step in active_result.history if step.marginals
    ))
    
    print(f"\nPrediction Diversity:")
    print(f"  Static: {static_predictions_unique} unique predictions")
    print(f"  Active: {active_predictions_unique} unique predictions")
    
    # Final verdict
    print(f"\nConclusion:")
    if f1_improvement > 0.1:
        print("  ✓ Active learning SIGNIFICANTLY improved structure discovery!")
        print(f"  ✓ F1 improvement: {f1_improvement:.3f}")
        print(f"  ✓ Active learning successfully adapted to the 10-variable distribution")
    elif f1_improvement > 0:
        print("  ✓ Active learning slightly improved structure discovery")
        print(f"  ✓ F1 improvement: {f1_improvement:.3f}")
    else:
        print("  ✗ Active learning did not improve structure discovery")
        print("  ✗ This suggests the implementation may need debugging")
    
    # Plot learning curves
    plot_learning_curves(static_result.history, active_result.history)
    
    print("\n" + "=" * 70)
    print("Test complete! Check fork10_learning_curves.png for visualization.")

if __name__ == "__main__":
    main()