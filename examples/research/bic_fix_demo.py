#!/usr/bin/env python3
"""
Educational Demo: BIC Scoring Fix for Likelihood Overfitting

This script provides an educational demonstration of the BIC scoring fix
that prevents likelihood overfitting in causal discovery. It shows:

1. The problem: How raw likelihood scoring leads to overfitting
2. The solution: How BIC scoring prevents overfitting with complexity penalty
3. The evidence: Side-by-side comparison with clear metrics

Perfect for understanding why BIC scoring is crucial for robust causal discovery.
"""

import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'examples')

from typing import Dict, Any
import time

try:
    # Try relative imports first (when used as module)
    from .demo_learning import DemoConfig
    from .complete_workflow_demo import run_zero_obs_fixed_intervention_test
    from .demo_evaluation import compute_f1_metrics
except ImportError:
    # Fall back to absolute imports (when run directly)
    from demo_learning import DemoConfig
    from complete_workflow_demo import run_zero_obs_fixed_intervention_test
    from demo_evaluation import compute_f1_metrics


def print_section_header(title: str, emoji: str = "🔬"):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"{emoji} {title}")
    print(f"{'='*70}")


def print_comparison_table(title: str, metrics: Dict[str, Dict[str, Any]]):
    """Print a formatted comparison table."""
    print(f"\n📊 {title}")
    print("-" * 70)
    print(f"{'Metric':<30} {'BIC Scoring':<20} {'Likelihood Scoring':<20}")
    print("-" * 70)
    
    for metric, values in metrics.items():
        bic_val = values.get('bic', 'N/A')
        likelihood_val = values.get('likelihood', 'N/A')
        
        # Format values appropriately
        if isinstance(bic_val, bool):
            bic_str = str(bic_val)
            likelihood_str = str(likelihood_val)
        elif isinstance(bic_val, (int, float)):
            if isinstance(bic_val, float):
                bic_str = f"{bic_val:.3f}"
                likelihood_str = f"{likelihood_val:.3f}"
            else:
                bic_str = str(bic_val)
                likelihood_str = str(likelihood_val)
        else:
            bic_str = str(bic_val)
            likelihood_str = str(likelihood_val)
        
        print(f"{metric:<30} {bic_str:<20} {likelihood_str:<20}")


def explain_bic_scoring():
    """Explain what BIC scoring is and why it helps."""
    print_section_header("Understanding BIC Scoring", "🧠")
    
    print("📚 What is BIC (Bayesian Information Criterion)?")
    print()
    print("BIC is a model selection criterion that balances:")
    print("   • Goodness of fit (how well model explains data)")
    print("   • Model complexity (number of parameters)")
    print()
    print("Formula: BIC_score = log_likelihood - 0.5 * n_params * log(n_samples)")
    print()
    print("💡 Why does this prevent overfitting?")
    print()
    print("Without BIC (raw likelihood):")
    print("   ❌ Models with more parents always fit data better")
    print("   ❌ Can learn spurious relationships from noise")
    print("   ❌ Overly confident in wrong structures")
    print()
    print("With BIC penalty:")
    print("   ✅ Penalizes models for having more parameters")
    print("   ✅ Prefers simpler models unless complexity is justified")
    print("   ✅ More robust to noise and spurious correlations")
    print()
    print("🎯 In our context:")
    print("   • Parent set with 3 variables has more parameters than parent set with 1")
    print("   • BIC prevents the model from choosing larger parent sets")
    print("     just because they can fit noise better")
    print("   • Results in more accurate causal discovery")


def run_bic_demonstration():
    """Run the main BIC vs likelihood demonstration."""
    print_section_header("BIC vs Likelihood Demonstration", "🔬")
    
    print("🧪 EXPERIMENTAL SETUP:")
    print("   • SCM: A → B → D ← C ← E (target D, true parents B,C)")
    print("   • Strategy: Fixed interventions on disconnected variable X")
    print("   • Problem: X provides NO causal information about D")
    print("   • Question: Will the model learn spurious relationships?")
    print()
    print("🎯 PREDICTION:")
    print("   • Likelihood scoring: Will overfit and learn wrong structure")
    print("   • BIC scoring: Will resist overfitting and be more conservative")
    print()
    
    # Test configuration
    config_base = {
        'n_observational_samples': 0,
        'n_intervention_steps': 8,
        'learning_rate': 1e-3,
        'random_seed': 42
    }
    
    print("🚀 Running experiments...")
    print()
    
    # Run BIC test
    print("Testing BIC scoring...")
    bic_config = DemoConfig(**config_base, scoring_method="bic")
    start_time = time.time()
    bic_results = run_zero_obs_fixed_intervention_test(bic_config)
    bic_time = time.time() - start_time
    
    # Run likelihood test
    print("Testing likelihood scoring...")
    likelihood_config = DemoConfig(**config_base, scoring_method="likelihood")
    start_time = time.time()
    likelihood_results = run_zero_obs_fixed_intervention_test(likelihood_config)
    likelihood_time = time.time() - start_time
    
    return bic_results, likelihood_results, bic_time, likelihood_time


def analyze_overfitting_behavior(bic_results: Dict, likelihood_results: Dict):
    """Analyze the overfitting behavior in detail."""
    print_section_header("Overfitting Analysis", "🔍")
    
    # Extract final marginal probabilities
    bic_marginals = bic_results['final_marginal_probs']
    likelihood_marginals = likelihood_results['final_marginal_probs']
    
    true_parents = set(bic_results['true_parents'])  # Should be {'B', 'C'}
    all_variables = set(bic_marginals.keys())
    false_variables = all_variables - true_parents
    
    print("🧮 Parent Probability Analysis:")
    print()
    print(f"{'Variable':<12} {'True Parent?':<15} {'BIC Prob':<12} {'Likelihood Prob':<18} {'Difference'}")
    print("-" * 75)
    
    total_false_positive_difference = 0
    
    for var in sorted(all_variables):
        is_true_parent = var in true_parents
        bic_prob = bic_marginals[var]
        likelihood_prob = likelihood_marginals[var]
        difference = likelihood_prob - bic_prob
        
        if not is_true_parent and likelihood_prob > 0.5:
            total_false_positive_difference += difference
        
        print(f"{var:<12} {str(is_true_parent):<15} {bic_prob:<12.3f} {likelihood_prob:<18.3f} {difference:+.3f}")
    
    print()
    print("🚨 Overfitting Indicators:")
    
    # Count false positives (non-parents with high probability)
    bic_false_positives = sum(1 for var in false_variables if bic_marginals[var] > 0.7)
    likelihood_false_positives = sum(1 for var in false_variables if likelihood_marginals[var] > 0.7)
    
    print(f"   • False positives (prob > 0.7):")
    print(f"     - BIC: {bic_false_positives}")
    print(f"     - Likelihood: {likelihood_false_positives}")
    print(f"     - Reduction: {likelihood_false_positives - bic_false_positives}")
    
    # Check over-confidence on true parents
    true_parent_bic_probs = [bic_marginals[var] for var in true_parents]
    true_parent_likelihood_probs = [likelihood_marginals[var] for var in true_parents]
    
    avg_true_parent_bic = sum(true_parent_bic_probs) / len(true_parent_bic_probs)
    avg_true_parent_likelihood = sum(true_parent_likelihood_probs) / len(true_parent_likelihood_probs)
    
    print(f"   • Average probability for true parents:")
    print(f"     - BIC: {avg_true_parent_bic:.3f}")
    print(f"     - Likelihood: {avg_true_parent_likelihood:.3f}")
    
    # Overall assessment
    overfitting_prevented = (
        bic_false_positives < likelihood_false_positives and
        total_false_positive_difference > 0.1
    )
    
    print()
    if overfitting_prevented:
        print("✅ OVERFITTING PREVENTION SUCCESSFUL!")
        print("   BIC scoring successfully reduced false positive parent assignments")
    else:
        print("❌ OVERFITTING PREVENTION FAILED")
        print("   BIC scoring did not show expected conservative behavior")
    
    return overfitting_prevented


def demonstrate_bic_fix():
    """Run the complete BIC fix demonstration."""
    print("🎓 EDUCATIONAL DEMO: BIC Scoring Fix for Causal Discovery")
    print("=" * 70)
    print("This demo shows how BIC scoring prevents likelihood overfitting")
    print("in causal structure learning with uninformative interventions.")
    print(f"Estimated runtime: ~1-2 minutes")
    
    start_time = time.time()
    
    # Step 1: Explain the theory
    explain_bic_scoring()
    
    # Step 2: Run the demonstration
    bic_results, likelihood_results, bic_time, likelihood_time = run_bic_demonstration()
    
    # Step 3: Compare key metrics
    print_section_header("Key Metrics Comparison", "📊")
    
    # Extract convergence metrics
    bic_converged = bic_results['converged_to_truth']['converged']
    likelihood_converged = likelihood_results['converged_to_truth']['converged']
    
    bic_accuracy = bic_results['converged_to_truth']['final_accuracy']
    likelihood_accuracy = likelihood_results['converged_to_truth']['final_accuracy']
    
    # Compute F1 scores
    bic_f1 = compute_f1_metrics(
        bic_results['final_marginal_probs'], 
        bic_results['true_parents']
    )['f1']
    
    likelihood_f1 = compute_f1_metrics(
        likelihood_results['final_marginal_probs'], 
        likelihood_results['true_parents']
    )['f1']
    
    # Count false positives
    true_parents = set(bic_results['true_parents'])
    all_vars = set(bic_results['final_marginal_probs'].keys())
    false_vars = all_vars - true_parents
    
    bic_false_pos = sum(1 for var in false_vars 
                       if bic_results['final_marginal_probs'][var] > 0.7)
    likelihood_false_pos = sum(1 for var in false_vars 
                              if likelihood_results['final_marginal_probs'][var] > 0.7)
    
    metrics = {
        'Converged to Truth': {'bic': bic_converged, 'likelihood': likelihood_converged},
        'Final Accuracy': {'bic': bic_accuracy, 'likelihood': likelihood_accuracy},
        'F1 Score': {'bic': bic_f1, 'likelihood': likelihood_f1},
        'False Positives': {'bic': bic_false_pos, 'likelihood': likelihood_false_pos},
        'Runtime (seconds)': {'bic': bic_time, 'likelihood': likelihood_time}
    }
    
    print_comparison_table("Performance Comparison", metrics)
    
    # Step 4: Detailed overfitting analysis
    overfitting_prevented = analyze_overfitting_behavior(bic_results, likelihood_results)
    
    # Step 5: Final summary
    total_time = time.time() - start_time
    
    print_section_header("Demo Summary", "🎯")
    
    print(f"Demo completed in {total_time:.1f} seconds")
    print()
    
    if overfitting_prevented:
        print("🎉 DEMONSTRATION SUCCESSFUL!")
        print()
        print("Key findings:")
        print("   ✅ BIC scoring prevents overfitting to uninformative data")
        print("   ✅ Likelihood scoring leads to spurious parent assignments")
        print("   ✅ BIC provides more conservative, accurate causal discovery")
        print()
        print("📚 Educational takeaways:")
        print("   • Information criteria are crucial for robust causal discovery")
        print("   • Raw likelihood can be misleading with complex models")
        print("   • Model complexity penalties prevent spurious relationships")
        print("   • BIC balances fit quality with model simplicity")
        
    else:
        print("⚠️ UNEXPECTED RESULTS")
        print("The demonstration did not show the expected overfitting prevention.")
        print("This may indicate configuration issues or environment differences.")
    
    print()
    print("💡 NEXT STEPS:")
    print("   1. Run the full verification suite: python examples/verify_intervention_strategies.py")
    print("   2. Compare intervention strategies: python examples/intervention_strategy_comparison.py")
    print("   3. Read the research summary: python examples/intervention_strategy_summary.py")
    
    return {
        'overfitting_prevented': overfitting_prevented,
        'bic_results': bic_results,
        'likelihood_results': likelihood_results,
        'metrics': metrics,
        'runtime_seconds': total_time
    }


if __name__ == "__main__":
    results = demonstrate_bic_fix()
    
    # Exit with appropriate code
    if results['overfitting_prevented']:
        print(f"\n✨ BIC scoring demonstration completed successfully!")
        sys.exit(0)
    else:
        print(f"\n⚠️ Demonstration showed unexpected results.")
        sys.exit(1)