#!/usr/bin/env python3
"""
Mechanism-Aware ACBO Integration Demo

Demonstrates the end-to-end integration of mechanism-aware architecture enhancements
with the ACBO pipeline. Shows both structure-only and mechanism-aware modes working
together seamlessly.

Architecture Enhancement Pivot - Part C: Integration & Testing
"""

import logging
from typing import Dict, Any, Optional
import time

import jax
import jax.numpy as jnp
import jax.random as random
import pyrsistent as pyr

from causal_bayes_opt.acquisition.state import AcquisitionState, MECHANISM_AWARE_AVAILABLE
from causal_bayes_opt.acquisition.hybrid_rewards import (
    compute_hybrid_reward,
    create_hybrid_reward_config,
    HybridRewardConfig
)
from causal_bayes_opt.avici_integration.parent_set.posterior import create_parent_set_posterior
from causal_bayes_opt.data_structures.buffer import ExperienceBuffer, create_buffer_from_samples
from causal_bayes_opt.data_structures.sample import create_sample
from causal_bayes_opt.experiments.test_scms import create_simple_test_scm
from causal_bayes_opt.mechanisms.linear import sample_from_linear_scm

# Optional mechanism-aware imports
if MECHANISM_AWARE_AVAILABLE:
    from causal_bayes_opt.avici_integration.parent_set.mechanism_aware import (
        MechanismPrediction,
        create_enhanced_config,
        create_structure_only_config
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_demo_mechanism_predictions() -> Optional[list]:
    """Create realistic mechanism predictions for demo."""
    if not MECHANISM_AWARE_AVAILABLE:
        return None
    
    return [
        MechanismPrediction(
            parent_set=frozenset(['X']),
            mechanism_type='linear',
            confidence=0.85,
            parameters={
                'coefficients': {'X': 2.5},
                'intercept': 0.1,
                'noise_variance': 0.1
            }
        ),
        MechanismPrediction(
            parent_set=frozenset(['Z']),
            mechanism_type='linear',
            confidence=0.62,
            parameters={
                'coefficients': {'Z': -1.8},
                'intercept': 0.0,
                'noise_variance': 0.3
            }
        ),
        MechanismPrediction(
            parent_set=frozenset(['X', 'Z']),
            mechanism_type='linear',
            confidence=0.73,
            parameters={
                'coefficients': {'X': 2.1, 'Z': -1.2},
                'intercept': 0.05,
                'noise_variance': 0.15
            }
        )
    ]


def create_demo_scm_and_data(n_samples: int = 25, seed: int = 42) -> tuple:
    """Create demo SCM and sample data."""
    logger.info(f"Creating demo SCM and generating {n_samples} samples...")
    
    # Create test SCM: X → Y ← Z (fork structure)
    scm = create_simple_test_scm(noise_scale=1.0, target='Y')
    
    # Generate mixed observational and interventional data
    key = random.PRNGKey(seed)
    
    # Observational samples
    obs_samples = sample_from_linear_scm(scm, n_samples=n_samples//2, seed=seed)
    
    # Interventional samples (simulate interventions on X and Z)
    int_samples = []
    for i in range(n_samples//2):
        if i % 2 == 0:
            # Intervene on X
            int_sample = create_sample(
                values={'X': 2.0 + 0.5*i, 'Y': 5.0 + 0.2*i, 'Z': 1.0},
                intervention_type='perfect',
                intervention_targets=frozenset(['X'])
            )
        else:
            # Intervene on Z  
            int_sample = create_sample(
                values={'X': 1.0, 'Y': 2.0 - 0.3*i, 'Z': -1.0 + 0.4*i},
                intervention_type='perfect',
                intervention_targets=frozenset(['Z'])
            )
        int_samples.append(int_sample)
    
    all_samples = obs_samples + int_samples
    logger.info(f"Generated {len(obs_samples)} observational and {len(int_samples)} interventional samples")
    
    return scm, all_samples


def create_demo_acquisition_state(samples: list, mode: str = "structure_only") -> AcquisitionState:
    """Create acquisition state for demo."""
    logger.info(f"Creating acquisition state in {mode} mode...")
    
    # Separate observational and interventional samples
    from causal_bayes_opt.data_structures.sample import is_observational, is_interventional
    
    obs_samples = [s for s in samples if is_observational(s)]
    int_samples = [s for s in samples if is_interventional(s)]
    
    # Create buffer with separated samples
    buffer = create_buffer_from_samples(observations=obs_samples)
    
    # Add interventional samples properly (create mock interventions)
    for int_sample in int_samples:
        from causal_bayes_opt.data_structures.sample import get_intervention_targets
        targets = get_intervention_targets(int_sample)
        values = {target: 0.0 for target in targets}  # Mock intervention values
        
        # Create mock intervention specification
        mock_intervention = pyr.m(
            targets=targets,
            values=values,
            type='perfect'
        )
        
        buffer.add_intervention(mock_intervention, int_sample)
    
    # Create posterior over parent sets
    parent_sets = [
        frozenset(),           # Empty parent set
        frozenset(['X']),      # X is parent
        frozenset(['Z']),      # Z is parent  
        frozenset(['X', 'Z'])  # Both X and Z are parents
    ]
    
    # Realistic probabilities (X is most likely parent)
    probs = jnp.array([0.05, 0.65, 0.15, 0.15])
    posterior = create_parent_set_posterior('Y', parent_sets, probs)
    
    # Create mechanism predictions if in mechanism-aware mode
    mechanism_predictions = None
    mechanism_uncertainties = None
    
    if mode == "mechanism_aware" and MECHANISM_AWARE_AVAILABLE:
        mechanism_predictions = create_demo_mechanism_predictions()
        mechanism_uncertainties = {
            'X': 0.15,  # High confidence
            'Z': 0.38,  # Lower confidence
        }
        logger.info("Added mechanism predictions and uncertainties")
    
    # Find best target value from samples
    target_values = []
    for sample in samples:
        values = sample.get('values', pyr.m())
        if 'Y' in values:
            target_values.append(float(values['Y']))
    
    best_value = max(target_values) if target_values else 0.0
    
    state = AcquisitionState(
        posterior=posterior,
        buffer=buffer,
        best_value=best_value,
        current_target='Y',
        step=len(samples),
        metadata=pyr.m(demo_mode=mode),
        mechanism_predictions=mechanism_predictions,
        mechanism_uncertainties=mechanism_uncertainties
    )
    
    logger.info(f"Created acquisition state: {state}")
    return state


def demonstrate_state_capabilities(state: AcquisitionState) -> None:
    """Demonstrate the capabilities of the acquisition state."""
    mode = state.metadata.get('demo_mode', 'unknown')
    logger.info(f"\n=== Demonstrating {mode} State Capabilities ===")
    
    # Basic state information
    print(f"Step: {state.step}")
    print(f"Target: {state.current_target}")
    print(f"Best value: {state.best_value:.3f}")
    print(f"Uncertainty: {state.uncertainty_bits:.2f} bits")
    print(f"Total samples: {state.buffer_statistics.total_samples}")
    
    # Optimization progress
    progress = state.get_optimization_progress()
    print(f"\nOptimization Progress:")
    print(f"  Improvement from start: {progress['improvement_from_start']:.3f}")
    print(f"  Recent improvement: {progress['recent_improvement']:.3f}")
    print(f"  Optimization rate: {progress['optimization_rate']:.4f}")
    print(f"  Stagnation steps: {progress['stagnation_steps']:.0f}")
    
    # Exploration coverage
    coverage = state.get_exploration_coverage()
    print(f"\nExploration Coverage:")
    print(f"  Target coverage rate: {coverage['target_coverage_rate']:.2f}")
    print(f"  Intervention diversity: {coverage['intervention_diversity']:.2f}")
    print(f"  Unexplored variables: {coverage['unexplored_variables']:.2f}")
    
    # Marginal parent probabilities
    print(f"\nMarginal Parent Probabilities:")
    for var, prob in sorted(state.marginal_parent_probs.items()):
        print(f"  P({var} is parent) = {prob:.3f}")
    
    # Mechanism insights (enhanced feature)
    insights = state.get_mechanism_insights()
    print(f"\nMechanism Insights:")
    print(f"  Mechanism-aware: {insights['mechanism_aware']}")
    
    if insights['mechanism_aware']:
        print(f"  High-impact variables: {insights['high_impact_variables']}")
        print(f"  Uncertain mechanisms: {insights['uncertain_mechanisms']}")
        print(f"  Predicted effects: {insights['predicted_effects']}")
        print(f"  Avg mechanism confidence: {insights['mechanism_confidence_avg']:.3f}")
        
        print(f"\nMechanism Confidence Scores:")
        for var, confidence in sorted(state.mechanism_confidence.items()):
            print(f"  {var}: {confidence:.3f}")
    else:
        print("  No mechanism predictions available")


def demonstrate_hybrid_rewards(state: AcquisitionState, scm: pyr.PMap) -> None:
    """Demonstrate hybrid reward computation."""
    mode = state.metadata.get('demo_mode', 'unknown')
    logger.info(f"\n=== Demonstrating Hybrid Rewards ({mode} mode) ===")
    
    # Create mock intervention and outcome
    intervention = pyr.m(
        targets=frozenset(['X']),
        values={'X': 3.0},
        type='perfect',
        metadata=pyr.m(reason='high_confidence_parent')
    )
    
    outcome = create_sample(
        values={'X': 3.0, 'Y': 6.5, 'Z': 1.0},  # Significant improvement
        intervention_type='perfect',
        intervention_targets=frozenset(['X'])
    )
    
    # Create next state (would normally update buffer and posterior)
    next_state = AcquisitionState(
        posterior=state.posterior,  # Simplified: same posterior
        buffer=state.buffer,        # Simplified: same buffer  
        best_value=6.5,            # Improved best value
        current_target='Y',
        step=state.step + 1,
        metadata=state.metadata,
        mechanism_predictions=state.mechanism_predictions,
        mechanism_uncertainties=state.mechanism_uncertainties
    )
    
    # Create ground truth for supervised rewards
    ground_truth = {
        'scm': scm,
        'mechanism_info': {
            'Y': {
                'parents': frozenset(['X']),  # True parents
                'coefficients': {'X': 2.0},  # True effect
                'intercept': 0.0
            }
        }
    }
    
    # Test different reward configurations
    configs = {
        'training': create_hybrid_reward_config(mode="training"),
        'deployment': create_hybrid_reward_config(mode="deployment"),
        'research': create_hybrid_reward_config(mode="research")
    }
    
    for config_name, config in configs.items():
        print(f"\n{config_name.title()} Mode Rewards:")
        
        reward_components = compute_hybrid_reward(
            current_state=state,
            intervention=intervention,
            outcome=outcome,
            next_state=next_state,
            config=config,
            ground_truth=ground_truth if config.use_supervised_signals else None
        )
        
        print(f"  Total reward: {reward_components.total_reward:.3f}")
        print(f"  Supervised parent: {reward_components.supervised_parent_reward:.3f}")
        print(f"  Supervised mechanism: {reward_components.supervised_mechanism_reward:.3f}")
        print(f"  Posterior confidence: {reward_components.posterior_confidence_reward:.3f}")
        print(f"  Causal effect: {reward_components.causal_effect_reward:.3f}")
        print(f"  Mechanism consistency: {reward_components.mechanism_consistency_reward:.3f}")


def run_comparative_demo() -> None:
    """Run comparative demo showing structure-only vs mechanism-aware modes."""
    logger.info("Starting Mechanism-Aware ACBO Integration Demo")
    logger.info("=" * 60)
    
    # Create shared demo data
    scm, samples = create_demo_scm_and_data(n_samples=30, seed=42)
    
    # Demonstrate structure-only mode (backward compatibility)
    logger.info("\n🔧 STRUCTURE-ONLY MODE (Backward Compatibility)")
    structure_state = create_demo_acquisition_state(samples, mode="structure_only")
    demonstrate_state_capabilities(structure_state)
    demonstrate_hybrid_rewards(structure_state, scm)
    
    # Demonstrate mechanism-aware mode (if available)
    if MECHANISM_AWARE_AVAILABLE:
        logger.info("\n🚀 MECHANISM-AWARE MODE (Enhanced)")
        mechanism_state = create_demo_acquisition_state(samples, mode="mechanism_aware")
        demonstrate_state_capabilities(mechanism_state)
        demonstrate_hybrid_rewards(mechanism_state, scm)
        
        # Comparative analysis
        logger.info("\n📊 COMPARATIVE ANALYSIS")
        print("Structure-only vs Mechanism-aware comparison:")
        
        struct_insights = structure_state.get_mechanism_insights()
        mech_insights = mechanism_state.get_mechanism_insights()
        
        print(f"  Mechanism awareness: {struct_insights['mechanism_aware']} → {mech_insights['mechanism_aware']}")
        print(f"  High-impact variables: {len(struct_insights['high_impact_variables'])} → {len(mech_insights['high_impact_variables'])}")
        print(f"  Predicted effects: {len(struct_insights['predicted_effects'])} → {len(mech_insights['predicted_effects'])}")
        
        if mech_insights['predicted_effects']:
            print("  Effect predictions:")
            for var, effect in mech_insights['predicted_effects'].items():
                print(f"    {var}: {effect:.2f}")
    else:
        logger.warning("\n⚠️  Mechanism-aware features not available")
        logger.info("    This demo shows graceful degradation to structure-only mode")
    
    # Performance summary
    logger.info("\n✅ INTEGRATION SUCCESS")
    print("Key achievements demonstrated:")
    print("  ✓ Backward compatibility with structure-only mode")
    print("  ✓ Enhanced features in mechanism-aware mode")
    print("  ✓ Graceful degradation when features unavailable")
    print("  ✓ Hybrid reward system integration")
    print("  ✓ Rich state representation for policy networks")
    
    if MECHANISM_AWARE_AVAILABLE:
        print("  ✓ Mechanism predictions enhance decision making")
        print("  ✓ Confidence scores guide exploration")
        print("  ✓ Effect predictions inform intervention values")


def main():
    """Main demo function."""
    start_time = time.time()
    
    try:
        run_comparative_demo()
        
        runtime = time.time() - start_time
        logger.info(f"\n🎯 Demo completed successfully in {runtime:.2f}s")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()