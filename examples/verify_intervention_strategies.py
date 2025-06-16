#!/usr/bin/env python3
"""
Verification Script: Intervention Strategy Comparison

Quick validation script to verify that:
1. BIC scoring prevents likelihood overfitting 
2. Random interventions outperform fixed interventions
3. F1 scores and convergence speed differences are as expected

This script provides a user-friendly way to verify the core findings
from our causal discovery research.
"""

import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'examples')

from typing import Dict, Any
import time

try:
    # Try relative imports first (when used as module)
    from .demo_learning import DemoConfig
    from .complete_workflow_demo import (
        run_zero_obs_random_intervention_test,
        run_zero_obs_fixed_intervention_test
    )
    from .demo_evaluation import (
        compute_f1_progress, analyze_convergence_speed,
        compute_f1_metrics
    )
except ImportError:
    # Fall back to absolute imports (when run directly)
    from demo_learning import DemoConfig
    from complete_workflow_demo import (
        run_zero_obs_random_intervention_test,
        run_zero_obs_fixed_intervention_test
    )
    from demo_evaluation import (
        compute_f1_progress, analyze_convergence_speed,
        compute_f1_metrics
    )


def print_section_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"🔬 {title}")
    print(f"{'='*60}")


def print_test_status(test_name: str, status: str, details: str = ""):
    """Print test status with consistent formatting."""
    status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
    print(f"{status_emoji} {test_name}: {status}")
    if details:
        print(f"   {details}")


def verify_bic_prevents_overfitting() -> Dict[str, Any]:
    """Verify that BIC scoring prevents likelihood overfitting."""
    print_section_header("BIC Overfitting Prevention Test")
    
    print("Testing BIC vs likelihood scoring with fixed interventions on disconnected variable...")
    print("Expected: BIC should prevent learning spurious relationships")
    
    # Configuration for BIC test
    bic_config = DemoConfig(
        n_observational_samples=0,
        n_intervention_steps=8,  # Reduced for speed
        learning_rate=1e-3,
        scoring_method="bic",
        random_seed=42
    )
    
    # Configuration for likelihood test  
    likelihood_config = DemoConfig(
        n_observational_samples=0,
        n_intervention_steps=8,
        learning_rate=1e-3,
        scoring_method="likelihood",
        random_seed=42
    )
    
    print(f"\nRunning BIC scoring test...")
    start_time = time.time()
    bic_results = run_zero_obs_fixed_intervention_test(bic_config)
    bic_time = time.time() - start_time
    
    print(f"Running likelihood scoring test...")
    start_time = time.time()
    likelihood_results = run_zero_obs_fixed_intervention_test(likelihood_config)
    likelihood_time = time.time() - start_time
    
    # Extract key metrics
    bic_converged = bic_results['converged_to_truth']['converged']
    bic_accuracy = bic_results['converged_to_truth']['final_accuracy']
    bic_false_positives = len([v for v in bic_results['final_marginal_probs'].values() 
                              if v > 0.7]) - len(bic_results['true_parents'])
    
    likelihood_converged = likelihood_results['converged_to_truth']['converged']
    likelihood_accuracy = likelihood_results['converged_to_truth']['final_accuracy']
    likelihood_false_positives = len([v for v in likelihood_results['final_marginal_probs'].values() 
                                     if v > 0.7]) - len(likelihood_results['true_parents'])
    
    print(f"\n📊 Results Comparison:")
    print(f"{'Metric':<25} {'BIC':<15} {'Likelihood':<15} {'Expected'}")
    print("-" * 65)
    print(f"{'Converged to truth':<25} {str(bic_converged):<15} {str(likelihood_converged):<15} BIC=False")
    print(f"{'Final accuracy':<25} {bic_accuracy:<15.3f} {likelihood_accuracy:<15.3f} BIC<Likelihood")
    print(f"{'False positives':<25} {bic_false_positives:<15} {likelihood_false_positives:<15} BIC<Likelihood")
    print(f"{'Runtime (seconds)':<25} {bic_time:<15.1f} {likelihood_time:<15.1f} Similar")
    
    # Validation
    bic_prevents_overfitting = (
        not bic_converged and  # BIC should not converge to wrong structure
        bic_false_positives < likelihood_false_positives and  # Fewer false positives
        bic_accuracy <= likelihood_accuracy  # BIC is more conservative
    )
    
    print(f"\n🎯 Validation:")
    if bic_prevents_overfitting:
        print_test_status("BIC Overfitting Prevention", "PASS", 
                         "BIC successfully prevents spurious learning from uninformative interventions")
    else:
        print_test_status("BIC Overfitting Prevention", "FAIL", 
                         "BIC did not show expected conservative behavior")
    
    return {
        'test_name': 'BIC Overfitting Prevention',
        'passed': bic_prevents_overfitting,
        'bic_results': bic_results,
        'likelihood_results': likelihood_results,
        'metrics': {
            'bic_converged': bic_converged,
            'likelihood_converged': likelihood_converged,
            'bic_false_positives': bic_false_positives,
            'likelihood_false_positives': likelihood_false_positives
        }
    }


def verify_intervention_strategy_performance() -> Dict[str, Any]:
    """Verify that random interventions outperform fixed interventions."""
    print_section_header("Intervention Strategy Performance Test")
    
    print("Testing random vs fixed intervention strategies...")
    print("Expected: Random interventions should achieve higher F1 scores and faster convergence")
    
    # Configuration for both tests
    config = DemoConfig(
        n_observational_samples=0,
        n_intervention_steps=10,  # Reduced for speed
        learning_rate=1e-3,
        scoring_method="bic",
        random_seed=42
    )
    
    print(f"\nRunning random intervention test...")
    start_time = time.time()
    random_results = run_zero_obs_random_intervention_test(config)
    random_time = time.time() - start_time
    
    print(f"Running fixed intervention test...")
    start_time = time.time()
    fixed_results = run_zero_obs_fixed_intervention_test(config)
    fixed_time = time.time() - start_time
    
    # Compute F1 metrics over time
    random_marginal_progress = random_results['marginal_prob_progress']
    fixed_marginal_progress = fixed_results['marginal_prob_progress']
    
    random_true_parents = random_results['true_parents']
    fixed_true_parents = fixed_results['true_parents']
    
    random_f1_progress = compute_f1_progress(random_marginal_progress, random_true_parents)
    fixed_f1_progress = compute_f1_progress(fixed_marginal_progress, fixed_true_parents)
    
    # Analyze convergence speed
    random_speed_analysis = analyze_convergence_speed(random_f1_progress)
    fixed_speed_analysis = analyze_convergence_speed(fixed_f1_progress)
    
    # Extract key metrics
    random_final_f1 = random_speed_analysis['final_f1']
    fixed_final_f1 = fixed_speed_analysis['final_f1']
    f1_advantage = random_final_f1 - fixed_final_f1
    
    random_converged = random_results['converged_to_truth']['converged']
    fixed_converged = fixed_results['converged_to_truth']['converged']
    
    # Steps to reach F1 = 0.5
    random_steps_to_half = random_speed_analysis['convergence_steps'].get('f1_0.5', None)
    fixed_steps_to_half = fixed_speed_analysis['convergence_steps'].get('f1_0.5', None)
    
    if random_steps_to_half is not None and fixed_steps_to_half is not None:
        speed_advantage = fixed_steps_to_half - random_steps_to_half
    else:
        speed_advantage = "N/A"
    
    print(f"\n📊 Performance Comparison:")
    print(f"{'Metric':<25} {'Random':<15} {'Fixed':<15} {'Expected'}")
    print("-" * 65)
    print(f"{'Final F1 Score':<25} {random_final_f1:<15.3f} {fixed_final_f1:<15.3f} Random>Fixed")
    print(f"{'Converged to truth':<25} {str(random_converged):<15} {str(fixed_converged):<15} Random=True")
    print(f"{'Steps to F1=0.5':<25} {str(random_steps_to_half):<15} {str(fixed_steps_to_half):<15} Random<Fixed")
    if speed_advantage != "N/A":
        print(f"{'Speed advantage':<25} {f'+{speed_advantage} steps':<15} {'for random':<15} Random faster")
    print(f"{'Runtime (seconds)':<25} {random_time:<15.1f} {fixed_time:<15.1f} Similar")
    
    # Validation
    strategy_performance_valid = (
        random_final_f1 > fixed_final_f1 and  # Better F1 score
        f1_advantage > 0.05 and  # Meaningful improvement
        random_converged  # Should converge to truth
    )
    
    print(f"\n🎯 Validation:")
    if strategy_performance_valid:
        print_test_status("Intervention Strategy Performance", "PASS",
                         f"Random interventions outperform by {f1_advantage:.3f} F1 points")
    else:
        print_test_status("Intervention Strategy Performance", "FAIL",
                         f"Expected advantage not observed (advantage: {f1_advantage:.3f})")
    
    return {
        'test_name': 'Intervention Strategy Performance',
        'passed': strategy_performance_valid,
        'random_results': random_results,
        'fixed_results': fixed_results,
        'metrics': {
            'f1_advantage': f1_advantage,
            'random_final_f1': random_final_f1,
            'fixed_final_f1': fixed_final_f1,
            'speed_advantage': speed_advantage,
            'random_converged': random_converged,
            'fixed_converged': fixed_converged
        }
    }


def run_verification_suite() -> Dict[str, Any]:
    """Run the complete verification suite."""
    print("🧪 ACBO INTERVENTION STRATEGY VERIFICATION SUITE")
    print("=" * 70)
    print("This script verifies two key findings:")
    print("1. BIC scoring prevents likelihood overfitting in causal discovery")
    print("2. Random interventions outperform fixed interventions")
    print(f"Estimated runtime: ~2-3 minutes")
    
    start_time = time.time()
    
    # Run tests
    test_results = []
    
    try:
        bic_test = verify_bic_prevents_overfitting()
        test_results.append(bic_test)
    except Exception as e:
        print_test_status("BIC Overfitting Prevention", "ERROR", f"Exception: {e}")
        test_results.append({'test_name': 'BIC Overfitting Prevention', 'passed': False, 'error': str(e)})
    
    try:
        strategy_test = verify_intervention_strategy_performance()
        test_results.append(strategy_test)
    except Exception as e:
        print_test_status("Intervention Strategy Performance", "ERROR", f"Exception: {e}")
        test_results.append({'test_name': 'Intervention Strategy Performance', 'passed': False, 'error': str(e)})
    
    total_time = time.time() - start_time
    
    # Summary
    print_section_header("VERIFICATION SUMMARY")
    
    passed_tests = sum(1 for test in test_results if test.get('passed', False))
    total_tests = len(test_results)
    
    print(f"Tests completed in {total_time:.1f} seconds")
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    print()
    
    for test in test_results:
        status = "PASS" if test.get('passed', False) else "FAIL"
        if 'error' in test:
            status = "ERROR"
        print_test_status(test['test_name'], status)
    
    print()
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("✅ BIC scoring successfully prevents overfitting")
        print("✅ Random interventions significantly outperform fixed interventions")
        print("✅ The ACBO framework behaves as expected")
        print()
        print("📚 Key findings validated:")
        print("   • Intervention diversity is crucial for causal discovery")
        print("   • Information criteria (BIC) prevent spurious structure learning")
        print("   • Random exploration outperforms fixed policies")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("Please check the detailed output above for specific issues.")
        print("This may indicate:")
        print("   • Configuration problems")
        print("   • Environment differences") 
        print("   • Code changes affecting behavior")
    
    return {
        'summary': {
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'all_passed': passed_tests == total_tests,
            'runtime_seconds': total_time
        },
        'test_results': test_results
    }


if __name__ == "__main__":
    results = run_verification_suite()
    
    # Exit with appropriate code
    if results['summary']['all_passed']:
        sys.exit(0)
    else:
        sys.exit(1)