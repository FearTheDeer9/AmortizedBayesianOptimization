#!/usr/bin/env python3
"""
Diagnostic script to investigate suspicious ACBO validation results.

This script addresses three key concerns:
1. Why does random baseline achieve -3.492 average (seems too good)?
2. Why do all methods show identical F1/SHD scores (suggests frozen surrogate)?
3. Are improvement calculations correct?
"""

import logging
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.experiments.benchmark_scms import (
    create_chain_scm, create_collider_scm, create_fork_scm
)
from src.causal_bayes_opt.evaluation.model_interfaces import (
    create_random_acquisition, create_grpo_acquisition, 
    create_bc_acquisition, create_bc_surrogate
)
from src.causal_bayes_opt.environments.sampling import sample_with_intervention
from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.utils.posterior_validator import PosteriorValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_scm_baseline(scm_dict, n_trials=100, n_interventions=10):
    """Analyze baseline performance on a specific SCM."""
    logger.info(f"\nAnalyzing SCM with {len(scm_dict['variables'])} variables")
    logger.info(f"Target: {scm_dict['target']}")
    logger.info(f"Graph: {scm_dict['graph']}")
    
    # Get initial target value - sample from SCM without intervention
    no_intervention = create_perfect_intervention(targets=set(), values={})
    initial_samples = sample_with_intervention(scm_dict, no_intervention, 100, seed=42)
    initial_target = np.mean([s['variables'][scm_dict['target']] for s in initial_samples])
    logger.info(f"Initial target value: {initial_target:.3f}")
    
    # Analyze random intervention performance
    random_improvements = []
    best_improvements = []
    worst_improvements = []
    
    for trial in range(n_trials):
        trial_best = initial_target
        trial_worst = initial_target
        
        for _ in range(n_interventions):
            # Random intervention
            var_idx = np.random.randint(0, len(scm_dict['variables']))
            var_name = scm_dict['variables'][var_idx]
            value = np.random.randn()
            
            # Sample with intervention
            intervention = create_perfect_intervention(targets={var_name}, values={var_name: value})
            samples = sample_with_intervention(scm_dict, intervention, 100, seed=trial*100 + _)
            current_target = np.mean([s['variables'][scm_dict['target']] for s in samples])
            
            trial_best = min(trial_best, current_target)
            trial_worst = max(trial_worst, current_target)
        
        improvement = initial_target - trial_best
        random_improvements.append(improvement)
        best_improvements.append(initial_target - trial_best)
        worst_improvements.append(initial_target - trial_worst)
    
    # Calculate statistics
    mean_improvement = np.mean(random_improvements)
    std_improvement = np.std(random_improvements)
    median_improvement = np.median(random_improvements)
    
    logger.info(f"\nRandom intervention statistics ({n_trials} trials):")
    logger.info(f"  Mean improvement: {mean_improvement:.3f} ± {std_improvement:.3f}")
    logger.info(f"  Median improvement: {median_improvement:.3f}")
    logger.info(f"  Best improvement: {max(random_improvements):.3f}")
    logger.info(f"  Worst 'improvement': {min(random_improvements):.3f}")
    logger.info(f"  % trials with positive improvement: {100 * sum(i > 0 for i in random_improvements) / n_trials:.1f}%")
    
    # Analyze causal structure impact
    parents = [p for p, c in scm_dict['graph'] if c == scm_dict['target']]
    logger.info(f"\nCausal structure:")
    logger.info(f"  Direct parents of target: {parents}")
    logger.info(f"  Intervening on parents vs non-parents:")
    
    # Test parent vs non-parent interventions
    parent_improvements = []
    non_parent_improvements = []
    
    for trial in range(n_trials):
        # Try intervening on a parent
        if parents:
            parent = np.random.choice(parents)
            value = np.random.randn()
            intervention = create_perfect_intervention(targets={parent}, values={parent: value})
            samples = sample_with_intervention(scm_dict, intervention, 100, seed=1000 + trial)
            parent_target = np.mean([s['variables'][scm_dict['target']] for s in samples])
            parent_improvements.append(initial_target - parent_target)
        
        # Try non-parent
        non_parents = [v for v in scm_dict['variables'] if v not in parents and v != scm_dict['target']]
        if non_parents:
            non_parent = np.random.choice(non_parents)
            value = np.random.randn()
            intervention = create_perfect_intervention(targets={non_parent}, values={non_parent: value})
            samples = sample_with_intervention(scm_dict, intervention, 100, seed=2000 + trial)
            non_parent_target = np.mean([s['variables'][scm_dict['target']] for s in samples])
            non_parent_improvements.append(initial_target - non_parent_target)
    
    if parent_improvements:
        logger.info(f"    Parent interventions: {np.mean(parent_improvements):.3f} ± {np.std(parent_improvements):.3f}")
    if non_parent_improvements:
        logger.info(f"    Non-parent interventions: {np.mean(non_parent_improvements):.3f} ± {np.std(non_parent_improvements):.3f}")
    
    return mean_improvement, std_improvement


def test_surrogate_predictions(surrogate_checkpoint_path):
    """Test if surrogate is making meaningful predictions."""
    logger.info("\n" + "="*60)
    logger.info("Testing Surrogate Predictions")
    logger.info("="*60)
    
    # Load surrogate
    surrogate_fn, _ = create_bc_surrogate(surrogate_checkpoint_path)
    
    # Create test SCM
    scm = create_chain_scm(chain_length=5)
    buffer = ExperienceBuffer()
    
    # Collect some observations - no intervention first
    no_intervention = create_perfect_intervention(targets=set(), values={})
    obs_samples = sample_with_intervention(scm, no_intervention, 10, seed=42)
    for sample in obs_samples:
        buffer.add(
            scm=scm,
            value=sample['variables'][scm['target']],
            intervention={},
            observation=sample['variables']
        )
    
    # Do a few interventions
    for i in range(3):
        var_idx = i % (len(scm['variables']) - 1)  # Don't intervene on target
        var_name = scm['variables'][var_idx]
        value = 1.0
        
        intervention = create_perfect_intervention(targets={var_name}, values={var_name: value})
        int_samples = sample_with_intervention(scm, intervention, 10, seed=100 + i)
        for sample in int_samples:
            buffer.add(
                scm=scm,
                value=sample['variables'][scm['target']],
                intervention={var_name: value},
                observation=sample['variables']
            )
    
    # Convert to tensor
    from src.causal_bayes_opt.training.data_preprocessing import buffer_to_tensor
    tensor_data, var_order = buffer_to_tensor(buffer, scm['target'])
    
    # Get predictions for different targets
    logger.info("\nTesting surrogate predictions for different targets:")
    for target in scm['variables']:
        posterior = surrogate_fn(tensor_data, target)
        
        logger.info(f"\nTarget: {target}")
        logger.info(f"  Posterior type: {type(posterior)}")
        
        # Try to extract marginal probabilities
        is_valid, issues, marginals = PosteriorValidator.validate_posterior(
            posterior, var_order, target
        )
        
        if is_valid and marginals:
            # Sort by probability
            sorted_probs = sorted(
                [(v, p) for v, p in marginals.items() if v != target],
                key=lambda x: x[1],
                reverse=True
            )
            logger.info(f"  Top predictions: {sorted_probs[:3]}")
            
            # Check if predictions change with different data
            if i == 0:
                first_marginals = marginals
            else:
                # Compare with first prediction
                changes = []
                for var in var_order:
                    if var != target and var in first_marginals:
                        change = abs(marginals[var] - first_marginals[var])
                        if change > 0.01:
                            changes.append((var, change))
                
                if changes:
                    logger.info(f"  Predictions changed from first: {changes}")
                else:
                    logger.info("  ⚠️  Predictions unchanged (possible frozen model)")
        else:
            logger.info(f"  ❌ Invalid posterior: {issues}")


def test_improvement_calculation():
    """Test that improvement calculations are correct."""
    logger.info("\n" + "="*60)
    logger.info("Testing Improvement Calculations")
    logger.info("="*60)
    
    # Create simple SCM where X0 -> X1 (target)
    scm = {
        'name': 'simple_chain',
        'variables': ['X0', 'X1'],
        'target': 'X1',
        'graph': [('X0', 'X1')],
        'functions': {
            'X0': lambda z, pa: 2.0 * z,
            'X1': lambda z, pa: pa['X0'] + z
        },
        'noise_scale': 1.0,
        'interventional_constraints': {}
    }
    
    # Test specific scenarios
    logger.info("\nScenario 1: Baseline")
    no_intervention = create_perfect_intervention(targets=set(), values={})
    baseline_samples = sample_with_intervention(scm, no_intervention, 100, seed=42)
    initial_value = np.mean([s['variables']['X1'] for s in baseline_samples])
    initial_x0 = np.mean([s['variables']['X0'] for s in baseline_samples])
    logger.info(f"  Initial X0: {initial_x0:.3f}")
    logger.info(f"  Initial X1 (target): {initial_value:.3f}")
    
    logger.info("\nScenario 2: Good intervention (X0 = -2)")
    good_intervention = create_perfect_intervention(targets={'X0'}, values={'X0': -2.0})
    good_samples = sample_with_intervention(scm, good_intervention, 100, seed=43)
    improved_value = np.mean([s['variables']['X1'] for s in good_samples])
    improvement = initial_value - improved_value
    logger.info(f"  After intervention X0=-2:")
    logger.info(f"  X1 (target): {improved_value:.3f}")
    logger.info(f"  Improvement: {improvement:.3f} (should be positive)")
    
    logger.info("\nScenario 3: Bad intervention (X0 = 2)")
    bad_intervention = create_perfect_intervention(targets={'X0'}, values={'X0': 2.0})
    bad_samples = sample_with_intervention(scm, bad_intervention, 100, seed=44)
    worse_value = np.mean([s['variables']['X1'] for s in bad_samples])
    degradation = initial_value - worse_value
    logger.info(f"  After intervention X0=2:")
    logger.info(f"  X1 (target): {worse_value:.3f}")
    logger.info(f"  Improvement: {degradation:.3f} (should be negative)")


def main():
    """Run diagnostics on ACBO validation results."""
    logger.info("="*80)
    logger.info("ACBO VALIDATION DIAGNOSTICS")
    logger.info("="*80)
    
    # 1. Analyze random baseline on different SCM types
    logger.info("\n1. ANALYZING RANDOM BASELINE PERFORMANCE")
    logger.info("-"*40)
    
    scm_types = [
        ("Chain", create_chain_scm(chain_length=5)),
        ("Collider", create_collider_scm()),
        ("Fork", create_fork_scm())
    ]
    
    all_improvements = []
    for name, scm in scm_types:
        logger.info(f"\n{name} SCM:")
        mean_imp, std_imp = analyze_scm_baseline(scm, n_trials=100, n_interventions=10)
        all_improvements.append(mean_imp)
    
    overall_mean = np.mean(all_improvements)
    logger.info(f"\nOverall mean improvement across SCM types: {overall_mean:.3f}")
    logger.info(f"Note: Reported -3.492 seems suspiciously good compared to {overall_mean:.3f}")
    
    # 2. Test surrogate predictions
    surrogate_path = Path("checkpoints/validation/bc_surrogate_final")
    if surrogate_path.exists():
        test_surrogate_predictions(surrogate_path)
    else:
        logger.warning(f"\nSurrogate checkpoint not found at {surrogate_path}")
        logger.warning("Run training script first to generate checkpoints")
    
    # 3. Test improvement calculations
    test_improvement_calculation()
    
    # 4. Summary and recommendations
    logger.info("\n" + "="*80)
    logger.info("SUMMARY AND RECOMMENDATIONS")
    logger.info("="*80)
    
    logger.info("\n1. Random Baseline Issue:")
    logger.info("   - Expected random improvement should be close to 0")
    logger.info("   - Large negative values suggest calculation error or")
    logger.info("   - optimization_direction confusion (minimize vs maximize)")
    
    logger.info("\n2. Identical F1/SHD Scores:")
    logger.info("   - Check if surrogate receives intervention outcomes during training")
    logger.info("   - Verify surrogate parameters are updating")
    logger.info("   - Consider longer training (100 episodes may be too short)")
    
    logger.info("\n3. Next Steps:")
    logger.info("   - Check optimization_direction in evaluation config")
    logger.info("   - Log raw target values before/after interventions")
    logger.info("   - Verify surrogate training includes graph supervision")
    logger.info("   - Test with more training episodes (1000+)")


if __name__ == "__main__":
    main()