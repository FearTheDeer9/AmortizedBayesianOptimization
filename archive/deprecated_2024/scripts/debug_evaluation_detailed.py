#!/usr/bin/env python3
"""
Detailed debugging of ACBO evaluation to understand exactly what's happening.

This script:
1. Loads trained models from checkpoints
2. Evaluates with detailed logging of interventions and outcomes
3. Shows the improvement calculation step by step
4. Validates the checkpoint loading and dynamic pairing
"""

import logging
import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp

sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.evaluation.universal_evaluator import UniversalACBOEvaluator
from src.causal_bayes_opt.evaluation.model_interfaces import (
    create_grpo_acquisition,
    create_bc_acquisition,
    create_bc_surrogate,
    create_random_acquisition,
    create_optimal_oracle_acquisition
)
from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_with_details(name, acquisition_fn, scm, surrogate_fn=None, n_interventions=5):
    """Evaluate a method with detailed logging of each step."""
    
    print(f"\n{'='*80}")
    print(f"EVALUATING: {name}")
    print(f"{'='*80}")
    
    # Show SCM details
    print(f"\nSCM Details:")
    print(f"  Variables: {scm['variables']}")
    print(f"  Target: {scm['target']}")
    print(f"  True parents of target: {[p for p, c in scm['edges'] if c == scm['target']]}")
    
    # Create evaluator
    evaluator = UniversalACBOEvaluator()
    
    # Evaluation config
    config = {
        'n_observational': 20,
        'max_interventions': n_interventions,
        'n_intervention_samples': 20,
        'optimization_direction': 'MINIMIZE'
    }
    
    # Custom evaluation with detailed logging
    from src.causal_bayes_opt.environments.sampling import sample_with_intervention
    from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
    from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
    from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
    from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
    from src.causal_bayes_opt.training.five_channel_converter import buffer_to_five_channel_tensor
    
    # Initialize
    buffer = ExperienceBuffer()
    
    # Collect observational data
    obs_samples = sample_from_linear_scm(scm, config['n_observational'], seed=42)
    
    for sample in obs_samples:
        buffer.add_observation(sample)
    
    # Calculate initial value
    initial_values = [s['values'][scm['target']] for s in obs_samples]
    initial_value = np.mean(initial_values)
    print(f"\nInitial target value: {initial_value:.4f} (mean of {len(initial_values)} observations)")
    print(f"  Range: [{np.min(initial_values):.4f}, {np.max(initial_values):.4f}]")
    
    # Track best value
    best_value = initial_value
    trajectory = [initial_value]
    
    print(f"\nStarting interventions (optimization_direction=MINIMIZE):")
    print("-" * 60)
    
    for i in range(n_interventions):
        # Get current tensor representation
        if surrogate_fn is not None:
            # Use 5-channel with surrogate
            tensor, var_order, metadata = buffer_to_five_channel_tensor(
                buffer, scm['target'], surrogate_fn=surrogate_fn
            )
            print(f"\nStep {i+1}: Using 5-channel tensor (with surrogate predictions)")
        else:
            # Use 3-channel without surrogate
            tensor, var_order = buffer_to_three_channel_tensor(buffer, scm['target'])
            print(f"\nStep {i+1}: Using 3-channel tensor (no surrogate)")
        
        # Get intervention recommendation
        # Handle different acquisition function signatures
        if 'random' in name.lower() or 'oracle' in name.lower():
            # Random and Oracle have different signature
            intervention = acquisition_fn(tensor, None, scm['target'], list(var_order))
        else:
            # GRPO/BC use simpler signature
            intervention = acquisition_fn(tensor, scm['target'])
        
        if not intervention or not intervention.get('targets'):
            print("  No intervention recommended")
            continue
        
        # Show intervention details
        targets = intervention['targets']
        values = intervention['values']
        print(f"  Intervention: {list(targets)[0]} = {values[list(targets)[0]]:.3f}")
        
        # Show surrogate predictions if available
        if surrogate_fn is not None and metadata and 'marginal_probs' in metadata:
            probs = metadata['marginal_probs']
            sorted_vars = sorted(
                [(v, p) for v, p in probs.items() if v != scm['target']],
                key=lambda x: x[1],
                reverse=True
            )
            print(f"  Surrogate predictions (top 3): {sorted_vars[:3]}")
        
        # Execute intervention
        intervention_obj = create_perfect_intervention(
            targets=targets,
            values=values
        )
        int_samples = sample_with_intervention(
            scm, intervention_obj, config['n_intervention_samples'], seed=100+i
        )
        
        # Add to buffer
        for sample in int_samples:
            buffer.add_intervention(intervention_obj, sample)
        
        # Calculate outcome
        outcome_values = [s['values'][scm['target']] for s in int_samples]
        outcome_mean = np.mean(outcome_values)
        
        # Update best value
        if outcome_mean < best_value:  # MINIMIZE
            best_value = outcome_mean
            print(f"  NEW BEST! {outcome_mean:.4f}")
        else:
            print(f"  Outcome: {outcome_mean:.4f}")
        
        trajectory.append(outcome_mean)
        
        # Show running improvement
        current_improvement = initial_value - best_value
        print(f"  Current improvement: {initial_value:.4f} - {best_value:.4f} = {current_improvement:.4f}")
    
    # Final results
    print(f"\n{'='*40}")
    print(f"FINAL RESULTS for {name}:")
    print(f"  Initial value: {initial_value:.4f}")
    print(f"  Best value: {best_value:.4f}")
    print(f"  Final improvement: {initial_value - best_value:.4f}")
    print(f"  Trajectory: {[f'{v:.3f}' for v in trajectory]}")
    
    return {
        'initial_value': initial_value,
        'best_value': best_value,
        'improvement': initial_value - best_value,
        'trajectory': trajectory
    }


def main():
    """Run detailed evaluation of all methods."""
    
    print("="*100)
    print("DETAILED ACBO EVALUATION DEBUGGING")
    print("="*100)
    
    # Create test SCM
    scm = create_fork_scm(noise_scale=1.0)
    
    # Define checkpoint paths
    checkpoint_base = Path("checkpoints/validation")
    
    # Test 1: Verify checkpoint loading
    print("\n1. TESTING CHECKPOINT LOADING:")
    print("-" * 40)
    
    # Try loading GRPO
    grpo_path = checkpoint_base / "unified_grpo_final"
    if grpo_path.exists():
        print(f"✓ GRPO checkpoint found: {grpo_path}")
        grpo_fn = create_grpo_acquisition(grpo_path)
        print("✓ GRPO loaded successfully")
    else:
        print(f"✗ GRPO checkpoint not found: {grpo_path}")
        grpo_fn = None
    
    # Try loading BC policy
    bc_path = checkpoint_base / "bc_final"
    if bc_path.exists():
        print(f"✓ BC policy checkpoint found: {bc_path}")
        bc_fn = create_bc_acquisition(bc_path)
        print("✓ BC policy loaded successfully")
    else:
        print(f"✗ BC policy checkpoint not found: {bc_path}")
        bc_fn = None
    
    # Try loading BC surrogate
    surrogate_path = checkpoint_base / "bc_surrogate_final"
    if surrogate_path.exists():
        print(f"✓ BC surrogate checkpoint found: {surrogate_path}")
        surrogate_fn, _ = create_bc_surrogate(surrogate_path)
        print("✓ BC surrogate loaded successfully")
    else:
        print(f"✗ BC surrogate checkpoint not found: {surrogate_path}")
        surrogate_fn = None
    
    # Create baselines
    random_fn = create_random_acquisition()
    oracle_fn = create_optimal_oracle_acquisition(scm)
    
    print("\n2. RUNNING DETAILED EVALUATIONS:")
    
    # Evaluate each method
    results = {}
    
    # Random baseline
    results['random'] = evaluate_with_details("Random Baseline", random_fn, scm)
    
    # Oracle baseline
    results['oracle'] = evaluate_with_details("Oracle (knows structure)", oracle_fn, scm)
    
    # GRPO without surrogate
    if grpo_fn:
        results['grpo'] = evaluate_with_details("GRPO (no surrogate)", grpo_fn, scm)
    
    # GRPO with surrogate
    if grpo_fn and surrogate_fn:
        results['grpo_surrogate'] = evaluate_with_details(
            "GRPO + BC Surrogate", grpo_fn, scm, surrogate_fn=surrogate_fn
        )
    
    # BC without surrogate
    if bc_fn:
        results['bc'] = evaluate_with_details("BC Policy (no surrogate)", bc_fn, scm)
    
    # BC with surrogate
    if bc_fn and surrogate_fn:
        results['bc_surrogate'] = evaluate_with_details(
            "BC Policy + BC Surrogate", bc_fn, scm, surrogate_fn=surrogate_fn
        )
    
    # Summary
    print("\n" + "="*100)
    print("SUMMARY OF IMPROVEMENTS:")
    print("="*100)
    print(f"{'Method':<30} {'Initial':>10} {'Best':>10} {'Improvement':>15}")
    print("-" * 65)
    
    for method, res in results.items():
        print(f"{method:<30} {res['initial_value']:>10.4f} {res['best_value']:>10.4f} {res['improvement']:>15.4f}")
    
    # Analyze improvement calculation
    print("\n" + "="*100)
    print("IMPROVEMENT CALCULATION ANALYSIS:")
    print("="*100)
    print("\nWe calculate improvement as: initial_value - best_value")
    print("For MINIMIZE optimization:")
    print("  - Positive improvement = we reduced the target (good)")
    print("  - Negative improvement = we increased the target (bad)")
    print("\nIf Oracle has the LOWEST trajectory values, it should have the HIGHEST improvement!")
    
    # Show trajectories
    print("\nTrajectory comparison:")
    max_len = max(len(res['trajectory']) for res in results.values())
    for i in range(max_len):
        line = f"Step {i}: "
        for method, res in results.items():
            if i < len(res['trajectory']):
                line += f"{method}={res['trajectory'][i]:.3f}  "
        print(line)


if __name__ == "__main__":
    main()