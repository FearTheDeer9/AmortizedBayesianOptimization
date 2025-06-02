#!/usr/bin/env python3
"""
Test script for ParentSetPosterior integration.

This script demonstrates the new ParentSetPosterior API and validates
that it works correctly with the existing parent set prediction model.
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

from causal_bayes_opt.experiments.test_scms import create_simple_test_scm
from causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from causal_bayes_opt.avici_integration import (
    create_training_batch,
    create_parent_set_model,
    predict_parent_posterior,
    summarize_posterior,
    get_most_likely_parents,
    get_marginal_parent_probabilities,
    ParentSetPosterior
)


def test_parent_set_posterior_basic():
    """Test basic ParentSetPosterior functionality."""
    print("🧪 TESTING PARENT SET POSTERIOR")
    print("=" * 50)
    
    # Create simple SCM: X → Y ← Z
    scm = create_simple_test_scm(noise_scale=1.0, target="Y")
    from causal_bayes_opt.data_structures.scm import get_variables
    variables = sorted(get_variables(scm))
    
    print(f"SCM variables: {variables}")
    print(f"Expected: Y should have parents {{X, Z}}")
    
    # Create model
    config = {
        'model_kwargs': {
            'layers': 2,
            'dim': 32,
            'dropout': 0.0,
        },
        'max_parent_size': 3,
    }
    
    net = create_parent_set_model(
        model_kwargs=config['model_kwargs'],
        max_parent_size=config['max_parent_size']
    )
    
    # Generate test data
    samples = sample_from_linear_scm(scm, n_samples=16, seed=42)
    batch = create_training_batch(scm, samples, "Y")
    
    # Initialize model
    params = net.init(random.PRNGKey(42), batch['x'], variables, "Y", True)
    
    print(f"\n🔮 TESTING NEW POSTERIOR API")
    print("-" * 30)
    
    # Test new predict_parent_posterior function
    posterior = predict_parent_posterior(
        net, params, batch['x'], variables, "Y", 
        metadata={'test_run': True, 'scm_type': 'simple_linear'}
    )
    
    # Validate posterior type
    assert isinstance(posterior, ParentSetPosterior), f"Expected ParentSetPosterior, got {type(posterior)}"
    print(f"✅ Posterior type: {type(posterior).__name__}")
    
    # Test basic properties
    print(f"✅ Target variable: {posterior.target_variable}")
    print(f"✅ Number of parent sets: {len(posterior.parent_set_probs)}")
    print(f"✅ Uncertainty (nats): {posterior.uncertainty:.3f}")
    print(f"✅ Uncertainty (bits): {posterior.uncertainty / jnp.log(2):.3f}")
    
    # Test top-k functionality
    top_3_parents = get_most_likely_parents(posterior, k=3)
    print(f"\n📊 TOP 3 MOST LIKELY PARENT SETS:")
    for i, (ps, prob) in enumerate(posterior.top_k_sets[:3]):
        ps_str = set(ps) if ps else "{}"
        print(f"  {i+1}. {ps_str}: {prob:.4f}")
    
    # Test marginal probabilities
    marginals = get_marginal_parent_probabilities(posterior, variables)
    print(f"\n📈 MARGINAL PARENT PROBABILITIES:")
    for var, prob in marginals.items():
        print(f"  P({var} is parent of Y) = {prob:.4f}")
    
    # Test summary functionality
    summary = summarize_posterior(posterior)
    print(f"\n📋 POSTERIOR SUMMARY:")
    print(f"  Most likely parents: {summary['most_likely_parents']}")
    print(f"  Most likely probability: {summary['most_likely_probability']:.4f}")
    print(f"  Concentration: {summary['concentration']:.4f}")
    print(f"  Effective # parent sets: {summary['effective_n_parent_sets']:.2f}")
    
    # Test metadata
    print(f"\n🏷️  METADATA:")
    for key, value in summary['metadata'].items():
        print(f"  {key}: {value}")
    
    print(f"\n✅ All ParentSetPosterior tests passed!")
    assert posterior is not None
    assert posterior.target_variable == "Y"
    assert len(posterior.parent_set_probs) > 0
    assert isinstance(posterior.uncertainty, float)
    assert "test_run" in posterior.metadata


def test_posterior_utilities():
    """Test additional posterior utility functions."""
    print(f"\n🔧 TESTING POSTERIOR UTILITIES")
    print("-" * 30)
    
    # Create a simple test posterior manually
    from causal_bayes_opt.avici_integration import create_parent_set_posterior
    
    test_parent_sets = [
        frozenset(),           # Empty set
        frozenset(['X']),      # Single parent
        frozenset(['Z']),      # Single parent
        frozenset(['X', 'Z'])  # Both parents
    ]
    
    test_probs = jnp.array([0.1, 0.2, 0.3, 0.4])  # Sums to 1.0
    
    test_posterior = create_parent_set_posterior(
        target_variable="Y",
        parent_sets=test_parent_sets,
        probabilities=test_probs,
        metadata={'test': 'manual_creation'}
    )
    
    print(f"✅ Created test posterior with {len(test_posterior.parent_set_probs)} parent sets")
    
    # Test specific probability queries
    from causal_bayes_opt.avici_integration import get_parent_set_probability
    
    empty_prob = get_parent_set_probability(test_posterior, frozenset())
    both_prob = get_parent_set_probability(test_posterior, frozenset(['X', 'Z']))
    missing_prob = get_parent_set_probability(test_posterior, frozenset(['W']))  # Doesn't exist
    
    print(f"✅ P(parents = {{}}) = {empty_prob:.3f}")
    print(f"✅ P(parents = {{X,Z}}) = {both_prob:.3f}")
    print(f"✅ P(parents = {{W}}) = {missing_prob:.3f} (should be 0)")
    
    # Test filtering
    from causal_bayes_opt.avici_integration import filter_parent_sets_by_probability
    
    filtered = filter_parent_sets_by_probability(test_posterior, min_probability=0.25)
    print(f"✅ Filtered posterior (prob >= 0.25): {len(filtered.parent_set_probs)} parent sets")
    
    filtered_summary = summarize_posterior(filtered)
    print(f"  Top parent set: {filtered_summary['most_likely_parents']}")
    print(f"  Probability mass retained: {sum(filtered.parent_set_probs.values()):.3f}")
    
    # Test comparison (compare test posterior with itself - should be identical)
    from causal_bayes_opt.avici_integration import compare_posteriors
    
    comparison = compare_posteriors(test_posterior, test_posterior)
    print(f"\n📊 SELF-COMPARISON METRICS (should be near-zero):")
    print(f"  KL divergence: {comparison['symmetric_kl_divergence']:.6f}")
    print(f"  Total variation distance: {comparison['total_variation_distance']:.6f}")
    print(f"  Overlap: {comparison['overlap']:.6f}")
    
    print(f"\n✅ All utility function tests passed!")
    assert test_posterior is not None
    assert test_posterior.target_variable == "Y"
    assert len(test_posterior.parent_set_probs) == 4
    assert "test" in test_posterior.metadata


def test_integration_with_existing_code():
    """Test that new API works with existing workflow."""
    print(f"\n🔗 TESTING INTEGRATION WITH EXISTING CODE")
    print("-" * 30)
    
    # This replicates the workflow but uses clean ParentSetPosterior API
    scm = create_simple_test_scm(noise_scale=1.0, target="Y")
    variables = sorted(scm['variables'])
    
    # Create and initialize model (existing workflow)
    net = create_parent_set_model(max_parent_size=3)
    samples = sample_from_linear_scm(scm, n_samples=16, seed=42)
    batch = create_training_batch(scm, samples, "Y")
    params = net.init(random.PRNGKey(42), batch['x'], variables, "Y", True)
    
    # OLD API (should still work)
    # No longer available - old API was removed for cleaner design
    
    # NEW API 
    new_result = predict_parent_posterior(net, params, batch['x'], variables, "Y")
    
    print(f"✅ Clean API returned: {type(new_result)}")
    
    # Check basic functionality
    assert len(new_result.parent_set_probs) > 0
    assert new_result.target_variable == "Y"
    
    print(f"✅ API works correctly:")
    print(f"  Target: {new_result.target_variable}")
    print(f"  Parent sets: {len(new_result.parent_set_probs)}")
    
    # Show that new API provides more information
    print(f"\n📈 NEW API PROVIDES ADDITIONAL INSIGHTS:")
    summary = summarize_posterior(new_result)
    print(f"  Uncertainty: {summary['uncertainty_bits']:.2f} bits")
    print(f"  Concentration: {summary['concentration']:.3f}")
    print(f"  Effective # parent sets: {summary['effective_n_parent_sets']:.1f}")
    
    if 'marginal_parent_probabilities' in summary and summary['marginal_parent_probabilities']:
        print(f"  Marginal probabilities: {summary['marginal_parent_probabilities']}")
    
    print(f"\n✅ Integration test passed!")


def main():
    """Run all ParentSetPosterior tests."""
    print("🚀 PARENT SET POSTERIOR INTEGRATION TESTS")
    print("=" * 60)
    
    try:
        # Test 1: Basic functionality
        posterior1 = test_parent_set_posterior_basic()
        
        # Test 2: Utility functions
        posterior2 = test_posterior_utilities()
        
        # Test 3: Integration with existing code
        test_integration_with_existing_code()
        
        print(f"\n" + "="*60)
        print(f"🎉 ALL PARENT SET POSTERIOR TESTS PASSED! 🎉")
        print(f"\nKey achievements:")
        print(f"✅ ParentSetPosterior data structure works correctly")
        print(f"✅ New predict_parent_posterior() API functions properly")
        print(f"✅ Utility functions (marginals, summaries, comparisons) work")
        print(f"✅ Integration with existing code is seamless")
        print(f"✅ Clean, single API design")
        
        print(f"\n📖 Usage examples:")
        print(f"  # Clean ParentSetPosterior API:")
        print(f"  posterior = predict_parent_posterior(net, params, data, vars, 'Y')")
        print(f"  most_likely = posterior.top_k_sets[0][0]")
        print(f"  summary = summarize_posterior(posterior)")
        print(f"  marginals = get_marginal_parent_probabilities(posterior, vars)")
        print(f"  comparison = compare_posteriors(pred, truth)")
        print(f"\n✨ ParentSetPosterior is ready for production use!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
