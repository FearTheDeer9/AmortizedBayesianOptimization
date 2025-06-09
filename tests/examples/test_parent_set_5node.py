#!/usr/bin/env python3
"""
Enhanced validation test showcasing ParentSetPosterior API with a 5-node SCM.

This test validates the complete parent set prediction system using the clean
ParentSetPosterior API with rich analysis capabilities on a more complex 5-node SCM.

Tests the 5-node SCM: A → B → D ← E, A → C → D
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

import jax
import jax.numpy as jnp
import jax.random as random
import optax

from causal_bayes_opt.experiments.test_scms import create_simple_linear_scm
from causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
# Clean ParentSetPosterior API
from causal_bayes_opt.avici_integration import (
    create_training_batch,
    create_parent_set_model,
    predict_parent_posterior,          # Main prediction API
    summarize_posterior,               # Rich analysis
    get_marginal_parent_probabilities, # Marginal analysis
    get_most_likely_parents,           # Clean extraction
    get_parent_set_probability,        # Direct queries
    compare_posteriors,                # Comparison utilities
    create_parent_set_posterior,       # Manual creation
    compute_loss,
    create_train_step
)
from causal_bayes_opt.avici_integration.testing.debug_tools import (
    debug_parent_set_enumeration,
    debug_training_step,
    debug_logits_and_probabilities
)


def create_five_node_test_scm(noise_scale: float = 1.0, target: str = "D"):
    """
    Create a 5-node test SCM with complex structure.
    
    Structure: A → B → D ← E, A → C → D
    
    This creates a diamond-like structure where:
    - A is a root node affecting both B and C
    - B and C are intermediate nodes, both paths lead to D
    - E is another root node that directly affects D
    - D is the confluence point with three parents: B, C, and E
    """
    return create_simple_linear_scm(
        variables=['A', 'B', 'C', 'D', 'E'],
        edges=[
            ('A', 'B'),  # A influences B
            ('A', 'C'),  # A influences C
            ('B', 'D'),  # B influences D
            ('C', 'D'),  # C influences D
            ('E', 'D'),  # E influences D
        ],
        coefficients={
            ('A', 'B'): 1.5,
            ('A', 'C'): 2.0,
            ('B', 'D'): 1.0,
            ('C', 'D'): -0.8,
            ('E', 'D'): 1.2,
        },
        noise_scales={var: noise_scale for var in ['A', 'B', 'C', 'D', 'E']},
        target=target
    )


def create_simple_config():
    """Simple config for quick testing."""
    return {
        'model_kwargs': {
            'layers': 3,      # Increased for more complex SCM
            'dim': 64,        # Increased for more complex SCM
            'key_size': 16,
            'num_heads': 4,
            'dropout': 0.0,   # No dropout for testing
        },
        'learning_rate': 1e-3,  # Fixed learning rate
        'batch_size': 32,       # Larger batch for 5-node
        'gradient_clip_norm': 1.0,
        'max_parent_size': 5,   # Allow up to 5 parents (all nodes)
    }


def create_improved_optimizer(config):
    """Create optimizer with gradient clipping."""
    return optax.chain(
        optax.clip_by_global_norm(config['gradient_clip_norm']),
        optax.adam(learning_rate=config['learning_rate'])
    )


def test_posterior_api_features():
    """Demonstrate the comprehensive ParentSetPosterior API features with 5-node SCM."""
    print("🔮 PARENT SET POSTERIOR API FEATURES (5-NODE SCM)")
    print("=" * 60)
    print("Demonstrating comprehensive ParentSetPosterior capabilities")
    print("with a more complex 5-node causal structure")
    print("=" * 60)
    
    # Setup
    config = create_simple_config()
    scm = create_five_node_test_scm(noise_scale=1.0, target="D")
    from causal_bayes_opt.data_structures.scm import get_variables
    variables = sorted(get_variables(scm))
    
    print(f"\n📊 SCM STRUCTURE:")
    print(f"  Variables: {variables}")
    print(f"  Causal graph: A → B → D ← E")
    print(f"                A → C → D")
    print(f"  Target variable: D")
    print(f"  Expected parents of D: {{B, C, E}}")
    
    net = create_parent_set_model(
        model_kwargs=config['model_kwargs'],
        max_parent_size=config['max_parent_size']
    )
    
    # Generate test data
    samples = sample_from_linear_scm(scm, n_samples=config['batch_size'], seed=42)
    batch = create_training_batch(scm, samples, "D")
    params = net.init(random.PRNGKey(42), batch['x'], variables, "D", True)
    
    print(f"\n🎯 CORE API:")
    print("-" * 30)
    
    # Main prediction API
    posterior = predict_parent_posterior(net, params, batch['x'], variables, "D")
    print(f"✅ predict_parent_posterior() -> ParentSetPosterior")
    print(f"  Target: {posterior.target_variable}")
    print(f"  Parent sets: {len(posterior.parent_set_probs)}")
    print(f"  Uncertainty: {posterior.uncertainty:.3f} nats")
    
    # Rich analysis capabilities
    print(f"\n✨ ANALYSIS FEATURES:")
    
    # 1. Rich summary
    summary = summarize_posterior(posterior)
    print(f"📋 summarize_posterior():")
    print(f"  Most likely: {summary['most_likely_parents']}")
    print(f"  Confidence: {summary['most_likely_probability']:.3f}")
    print(f"  Uncertainty: {summary['uncertainty_bits']:.2f} bits")
    print(f"  Concentration: {summary['concentration']:.3f}")
    
    # 2. Marginal probabilities
    marginals = get_marginal_parent_probabilities(posterior, variables)
    print(f"\n📈 get_marginal_parent_probabilities():")
    for var, prob in marginals.items():
        is_true_parent = var in {'B', 'C', 'E'}
        mark = "✓" if is_true_parent else " "
        print(f"  {mark} P({var} is parent of D) = {prob:.3f}")
    
    # 3. Top-k extraction
    top_5 = get_most_likely_parents(posterior, k=5)
    print(f"\n🏆 get_most_likely_parents(k=5):")
    for i, ps in enumerate(top_5):
        ps_str = set(ps) if ps else "{}"
        prob = posterior.parent_set_probs[ps]
        is_correct = set(ps) == {'B', 'C', 'E'}
        mark = "✓" if is_correct else " "
        print(f"  {mark} {i+1}. {ps_str}: {prob:.3f}")
    
    # 4. Direct probability queries
    true_parents_prob = get_parent_set_probability(posterior, frozenset(['B', 'C', 'E']))
    partial_prob = get_parent_set_probability(posterior, frozenset(['B', 'C']))
    print(f"\n🔍 get_parent_set_probability():")
    print(f"  P(parents = {{B,C,E}}) = {true_parents_prob:.3f} (TRUE)")
    print(f"  P(parents = {{B,C}}) = {partial_prob:.3f} (PARTIAL)")
    
    print(f"\n✅ All ParentSetPosterior features demonstrated!")
    assert posterior is not None
    assert posterior.target_variable == "D"
    assert len(posterior.parent_set_probs) > 0
    assert isinstance(posterior.uncertainty, float)


def test_enhanced_training_workflow():
    """Test the enhanced training workflow with posterior analysis on 5-node SCM."""
    print(f"\n🎯 ENHANCED TRAINING WITH POSTERIOR ANALYSIS (5-NODE SCM)")
    print("=" * 70)
    
    # Create SCM and expected results
    config = create_simple_config()
    scm = create_five_node_test_scm(noise_scale=1.0, target="D")
    from causal_bayes_opt.data_structures.scm import get_variables
    variables = sorted(get_variables(scm))
    
    print(f"Testing 5-node SCM: A → B → D ← E, A → C → D")
    print(f"Variables: {variables}")
    
    # Test cases with ground truth
    test_cases = [
        ('A', frozenset()),                      # Root variable
        ('E', frozenset()),                      # Root variable  
        ('B', frozenset(['A'])),                 # B has parent A
        ('C', frozenset(['A'])),                 # C has parent A
        ('D', frozenset(['B', 'C', 'E']))        # D has three parents
    ]
    
    net = create_parent_set_model(
        model_kwargs=config['model_kwargs'],
        max_parent_size=config['max_parent_size']
    )
    
    for target_var, expected_parents in test_cases:
        print(f"\n" + "="*50)
        print(f"🎯 TARGET: {target_var}")
        print(f"Expected: {set(expected_parents) if expected_parents else '{}'}")
        print("="*50)
        
        # Generate data and initialize
        samples = sample_from_linear_scm(scm, n_samples=config['batch_size'], seed=42)
        batch = create_training_batch(scm, samples, target_var)
        params = net.init(random.PRNGKey(42), batch['x'], variables, target_var, True)
        
        # Initial prediction
        print(f"\n🔮 INITIAL PREDICTION:")
        initial_posterior = predict_parent_posterior(net, params, batch['x'], variables, target_var)
        initial_summary = summarize_posterior(initial_posterior)
        
        print(f"  Most likely: {initial_summary['most_likely_parents']}")
        print(f"  Confidence: {initial_summary['most_likely_probability']:.3f}")
        print(f"  Uncertainty: {initial_summary['uncertainty_bits']:.2f} bits")
        
        # Quick training
        print(f"\n🏃 TRAINING (10 steps):")  # More steps for complex SCM
        optimizer = create_improved_optimizer(config)
        opt_state = optimizer.init(params)
        train_step_fn = create_train_step(net, optimizer)
        
        for step in range(10):
            step_samples = sample_from_linear_scm(scm, n_samples=config['batch_size'], seed=42+step)
            step_batch = create_training_batch(scm, step_samples, target_var)
            
            params, opt_state, loss = train_step_fn(
                params, opt_state, step_batch['x'], variables, target_var, expected_parents
            )
            if step % 2 == 0:  # Print every other step
                print(f"  Step {step}: loss = {loss:.4f}")
        
        # Final prediction with analysis
        print(f"\n🎯 FINAL PREDICTION:")
        final_posterior = predict_parent_posterior(net, params, batch['x'], variables, target_var)
        final_summary = summarize_posterior(final_posterior)
        
        print(f"  Most likely: {final_summary['most_likely_parents']}")
        print(f"  Confidence: {final_summary['most_likely_probability']:.3f}")
        print(f"  Uncertainty: {final_summary['uncertainty_bits']:.2f} bits")
        
        # Check improvement
        if final_summary['most_likely_parents'] == set(expected_parents):
            print(f"  ✅ SUCCESS: Correct prediction!")
        else:
            print(f"  ⚠️ PARTIAL: Expected {set(expected_parents)}, got {final_summary['most_likely_parents']}")
        
        # Compare initial vs final
        print(f"\n📊 IMPROVEMENT ANALYSIS:")
        try:
            comparison = compare_posteriors(initial_posterior, final_posterior)
            print(f"  KL divergence: {comparison['symmetric_kl_divergence']:.3f}")
            print(f"  Change in uncertainty: {final_summary['uncertainty_bits'] - initial_summary['uncertainty_bits']:.2f} bits")
            
            conf_improvement = final_summary['most_likely_probability'] - initial_summary['most_likely_probability']
            print(f"  Confidence change: {conf_improvement:+.3f}")
            
        except Exception as e:
            print(f"  Comparison skipped: {e}")
        
        # Show marginal probabilities for complex cases
        if target_var in ['B', 'C', 'D']:
            marginals = get_marginal_parent_probabilities(final_posterior, variables)
            print(f"\n📈 MARGINAL PARENT PROBABILITIES:")
            for var, prob in marginals.items():
                is_true_parent = var in expected_parents
                status = "✅" if is_true_parent else "  "
                print(f"  {status} P({var} is parent) = {prob:.3f}")
    
    print(f"\n✅ Enhanced training workflow completed!")


def test_complex_parent_sets():
    """Test handling of complex parent sets in 5-node SCM."""
    print(f"\n🎯 COMPLEX PARENT SET ANALYSIS (5-NODE SCM)")
    print("=" * 60)
    
    # Create a ground truth posterior for D with multiple plausible parent sets
    parent_sets = [
        frozenset(['B', 'C', 'E']),     # Correct answer - high probability
        frozenset(['B', 'C']),          # Missing E - medium probability
        frozenset(['B', 'E']),          # Missing C - medium probability  
        frozenset(['C', 'E']),          # Missing B - medium probability
        frozenset(['A', 'B', 'C', 'E']), # Including indirect parent - low
        frozenset(['B', 'C', 'D']),     # Including self - very low
        frozenset(['A', 'C']),          # Only indirect paths - low
        frozenset(),                    # Empty set - very low
    ]
    
    # Create probabilities that reflect realistic uncertainty
    probs = jnp.array([0.40, 0.15, 0.15, 0.15, 0.05, 0.02, 0.05, 0.03])
    
    ground_truth = create_parent_set_posterior(
        target_variable="D",
        parent_sets=parent_sets,
        probabilities=probs,
        metadata={'source': 'ground_truth', 'scm': '5-node complex'}
    )
    
    print(f"📋 GROUND TRUTH POSTERIOR FOR D:")
    gt_summary = summarize_posterior(ground_truth)
    print(f"  Most likely: {gt_summary['most_likely_parents']}")
    print(f"  Confidence: {gt_summary['most_likely_probability']:.3f}")
    print(f"  Uncertainty: {gt_summary['uncertainty_bits']:.2f} bits")
    print(f"  Concentration: {gt_summary['concentration']:.3f}")
    
    # Show top 5 parent sets
    print(f"\n🏆 TOP 5 PARENT SETS:")
    top_5 = get_most_likely_parents(ground_truth, k=5)
    for i, ps in enumerate(top_5):
        ps_str = set(ps) if ps else "{}"
        prob = ground_truth.parent_set_probs[ps]
        print(f"  {i+1}. {ps_str}: {prob:.3f}")
    
    # Analyze marginal probabilities
    variables = ['A', 'B', 'C', 'D', 'E']
    marginals = get_marginal_parent_probabilities(ground_truth, variables)
    print(f"\n📈 MARGINAL PARENT PROBABILITIES:")
    true_parents = {'B', 'C', 'E'}
    for var, prob in sorted(marginals.items(), key=lambda x: x[1], reverse=True):
        is_true = var in true_parents
        status = "✅" if is_true else "❌"
        print(f"  {status} P({var} is parent) = {prob:.3f}")
    
    # Generate a prediction from our model and compare
    config = create_simple_config()
    scm = create_five_node_test_scm(noise_scale=1.0, target="D")
    
    net = create_parent_set_model(max_parent_size=config['max_parent_size'])
    samples = sample_from_linear_scm(scm, n_samples=config['batch_size'], seed=42)
    batch = create_training_batch(scm, samples, "D")
    params = net.init(random.PRNGKey(42), batch['x'], variables, "D", True)
    
    predicted = predict_parent_posterior(net, params, batch['x'], variables, "D")
    
    print(f"\n🔮 MODEL PREDICTION:")
    pred_summary = summarize_posterior(predicted)
    print(f"  Most likely: {pred_summary['most_likely_parents']}")
    print(f"  Confidence: {pred_summary['most_likely_probability']:.3f}")
    print(f"  Uncertainty: {pred_summary['uncertainty_bits']:.2f} bits")
    
    # Compare predictions
    print(f"\n📊 COMPARISON METRICS:")
    try:
        comparison = compare_posteriors(predicted, ground_truth)
        print(f"  KL divergence (pred → truth): {comparison['kl_divergence_1_to_2']:.3f}")
        print(f"  KL divergence (truth → pred): {comparison['kl_divergence_2_to_1']:.3f}")
        print(f"  Symmetric KL divergence: {comparison['symmetric_kl_divergence']:.3f}")
        print(f"  Total variation distance: {comparison['total_variation_distance']:.3f}")
        print(f"  Overlap: {comparison['overlap']:.3f}")
        
        # Interpretation
        if comparison['symmetric_kl_divergence'] < 0.5:
            print(f"  ✅ GOOD: Predictions are close to ground truth")
        elif comparison['symmetric_kl_divergence'] < 1.0:
            print(f"  ⚠️ FAIR: Predictions are somewhat close")
        else:
            print(f"  ❌ POOR: Predictions are far from ground truth")
            
    except Exception as e:
        print(f"  Comparison failed: {e}")
    
    print(f"\n✅ Complex parent set analysis completed!")


def main():
    """Run all enhanced tests with ParentSetPosterior API on 5-node SCM."""
    print("🚀 PARENT SET PREDICTION TESTS - 5-NODE SCM")
    print("=" * 70)
    print("Clean ParentSetPosterior API with rich analysis capabilities")
    print("Testing on a more complex 5-node causal structure")
    print("=" * 70)
    
    try:
        # Test 1: ParentSetPosterior API features
        posterior = test_posterior_api_features()
        
        # Test 2: Enhanced training workflow  
        test_enhanced_training_workflow()
        
        # Test 3: Complex parent set analysis
        test_complex_parent_sets()
        
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)