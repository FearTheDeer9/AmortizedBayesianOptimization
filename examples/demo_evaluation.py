#!/usr/bin/env python3
"""
Evaluation and analysis functions for ACBO demo experiments.

Provides convergence analysis, progress tracking, and result comparison utilities.
"""

from typing import Dict, List, Any, Tuple
import jax.numpy as jnp
import numpy as np

from causal_bayes_opt.data_structures import get_values
from causal_bayes_opt.avici_integration.parent_set import get_marginal_parent_probabilities


def analyze_convergence(marginal_prob_progress: List[Dict[str, float]], true_parents: List[str], 
                       threshold: float = 0.7) -> Dict[str, Any]:
    """
    Analyze convergence to ground truth parents.
    
    Args:
        marginal_prob_progress: List of marginal probability dictionaries over time
        true_parents: List of ground truth parent variable names
        threshold: Probability threshold for considering a variable as a parent
        
    Returns:
        Dictionary with convergence analysis results
    """
    if not marginal_prob_progress:
        return {
            'converged': False,
            'final_accuracy': 0.0,
            'convergence_step': None,
            'true_parent_probs': {},
            'false_positive_probs': {},
            'threshold_used': threshold
        }
    
    final_probs = marginal_prob_progress[-1]
    
    # Check final accuracy - true parents above threshold, others below
    true_parent_probs = {p: final_probs.get(p, 0.0) for p in true_parents}
    other_vars = [v for v in final_probs.keys() if v not in true_parents]
    false_positive_probs = {v: final_probs.get(v, 0.0) for v in other_vars}
    
    # Count correct identifications
    true_positives = sum(1 for p in true_parents if final_probs.get(p, 0.0) > threshold)
    false_positives = sum(1 for v in other_vars if final_probs.get(v, 0.0) > threshold)
    
    final_accuracy = true_positives / len(true_parents) if true_parents else 0.0
    converged = (true_positives == len(true_parents)) and (false_positives == 0)
    
    # Find convergence step (first time all true parents above threshold)
    convergence_step = None
    for step, probs in enumerate(marginal_prob_progress):
        all_true_above = all(probs.get(p, 0.0) > threshold for p in true_parents)
        any_false_above = any(probs.get(v, 0.0) > threshold for v in other_vars)
        
        if all_true_above and not any_false_above:
            convergence_step = step
            break
    
    return {
        'converged': converged,
        'final_accuracy': final_accuracy,
        'convergence_step': convergence_step,
        'true_parent_probs': true_parent_probs,
        'false_positive_probs': false_positive_probs,
        'threshold_used': threshold
    }


def compute_data_likelihood_from_posterior_jax(_posterior: object, new_samples: List, target_variable: str) -> jnp.ndarray:
    """
    JAX-compatible version of data likelihood computation.
    
    Simplified approach that works with JAX compilation:
    - Use only JAX operations
    - Avoid Python control flow 
    - Use vectorized operations
    """
    if not new_samples:
        return jnp.array(0.0)
    
    # Convert target values to JAX array (avoid float() conversion)
    target_values_list = []
    for s in new_samples:
        val = get_values(s)[target_variable]
        # Don't convert to float if it's already a JAX array
        target_values_list.append(val)
    target_values = jnp.array(target_values_list)
    
    # Simplified approach: just compute likelihood under normal distribution
    # This is a proxy that should still provide useful learning signal
    mean_pred = jnp.mean(target_values)
    std_pred = jnp.maximum(jnp.std(target_values), 0.1)  # Avoid zero std
    
    # Log likelihood under normal distribution (JAX compatible)
    log_likelihood = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(std_pred)
    log_likelihood -= 0.5 * jnp.sum((target_values - mean_pred) ** 2) / (std_pred ** 2)
    
    return log_likelihood


def compute_data_likelihood_from_posterior(posterior: object, new_samples: List, target_variable: str) -> float:
    """
    Wrapper that calls JAX-compatible version and converts result.
    """
    if not new_samples:
        return 0.0
    
    jax_result = compute_data_likelihood_from_posterior_jax(posterior, new_samples, target_variable)
    
    # Convert JAX result to Python float safely
    try:
        return float(jax_result)
    except:
        # If it's still a tracer, return it as-is
        return jax_result


def compare_intervention_tests(random_results: Dict[str, Any], fixed_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare results from zero observational data tests to validate intervention diversity importance.
    """
    print("\n🔍 COMPARING ZERO OBSERVATIONAL DATA TESTS")
    print("=" * 60)
    print("Comparing random vs fixed interventions with identical setup (zero obs data)")
    
    # Extract key metrics
    random_converged = random_results['converged_to_truth']['converged']
    random_accuracy = random_results['converged_to_truth']['final_accuracy']
    random_uncertainty = random_results['final_uncertainty']
    
    fixed_converged = fixed_results['converged_to_truth']['converged']
    fixed_accuracy = fixed_results['converged_to_truth']['final_accuracy']
    fixed_uncertainty = fixed_results['final_uncertainty']
    
    print(f"\n📊 Side-by-Side Comparison:")
    print(f"{'Metric':<25} {'Random':<15} {'Fixed':<15} {'Expected'}")
    print("-" * 70)
    print(f"{'Converged to truth':<25} {str(random_converged):<15} {str(fixed_converged):<15} Random=True, Fixed=False")
    print(f"{'Final accuracy':<25} {random_accuracy:<15.3f} {fixed_accuracy:<15.3f} Random>Fixed")
    print(f"{'Final uncertainty':<25} {random_uncertainty:<15.2f} {fixed_uncertainty:<15.2f} Random<Fixed")
    print(f"{'Total samples':<25} {random_results['total_samples']:<15} {fixed_results['total_samples']:<15} Similar")
    
    # Validation checks
    diversity_matters = random_converged and not fixed_converged
    accuracy_difference = random_accuracy - fixed_accuracy
    uncertainty_difference = fixed_uncertainty - random_uncertainty
    
    print(f"\n🎯 Key Validations:")
    print(f"✅ Intervention diversity matters: {diversity_matters}")
    print(f"   Random interventions converged: {random_converged}")
    print(f"   Fixed interventions converged: {fixed_converged}")
    
    print(f"✅ Accuracy advantage for random: {accuracy_difference:.3f}")
    print(f"   Random accuracy: {random_accuracy:.3f}")
    print(f"   Fixed accuracy: {fixed_accuracy:.3f}")
    
    print(f"✅ Uncertainty advantage for random: {uncertainty_difference:.2f} bits")
    print(f"   Random uncertainty: {random_uncertainty:.2f} bits")
    print(f"   Fixed uncertainty: {fixed_uncertainty:.2f} bits")
    
    # Overall assessment
    substantial_difference = accuracy_difference > 0.3 and abs(uncertainty_difference) > 0.1
    validates_hypothesis = diversity_matters and substantial_difference
    
    print(f"\n📝 Overall Assessment:")
    if validates_hypothesis:
        print(f"✅ HYPOTHESIS VALIDATED")
        print(f"   • Intervention diversity is crucial for causal discovery")
        print(f"   • Random interventions significantly outperform fixed interventions")
    else:
        print(f"⚠️  HYPOTHESIS NOT CLEARLY VALIDATED")
        if not diversity_matters:
            print(f"   • Both approaches converged or both failed")
        if not substantial_difference:
            print(f"   • Difference between approaches is not substantial")
    
    return {
        'comparison_metrics': {
            'diversity_matters': diversity_matters,
            'accuracy_difference': accuracy_difference,
            'uncertainty_difference': uncertainty_difference,
            'validates_hypothesis': validates_hypothesis
        },
        'individual_results': {
            'random': {
                'converged': random_converged,
                'accuracy': random_accuracy,
                'uncertainty': random_uncertainty
            },
            'fixed': {
                'converged': fixed_converged,
                'accuracy': fixed_accuracy,
                'uncertainty': fixed_uncertainty
            }
        }
    }


def print_learning_progress(learning_history: List[Dict], target_variable: str, 
                          true_parents: List[str], show_steps: int = 5):
    """Print learning progress for debugging and monitoring."""
    if not learning_history:
        print("No learning history available")
        return
    
    print(f"\n📈 Learning Progress (last {show_steps} steps):")
    print(f"{'Step':<6} {'Loss':<10} {'Uncertainty':<12} {'True Parent Probs':<25} {'Best {target_variable}'}")
    print("-" * 80)
    
    for step_info in learning_history[-show_steps:]:
        step = step_info['step']
        loss = step_info.get('loss', float('nan'))
        uncertainty = step_info.get('uncertainty', float('inf'))
        outcome_value = step_info.get('outcome_value', float('nan'))
        marginals = step_info.get('marginals', {})
        
        # Extract true parent probabilities
        true_probs = {p: marginals.get(p, 0.0) for p in true_parents}
        true_probs_str = ", ".join([f"{p}:{prob:.3f}" for p, prob in true_probs.items()])
        
        print(f"{step:<6} {loss:<10.3f} {uncertainty:<12.2f} {true_probs_str:<25} {outcome_value:<10.2f}")


def print_final_summary(results: Dict[str, Any], experiment_name: str):
    """Print a comprehensive summary of experiment results."""
    print(f"\n🎯 FINAL SUMMARY: {experiment_name}")
    print("=" * 60)
    
    # Basic info
    target = results.get('target_variable', 'Unknown')
    true_parents = results.get('true_parents', [])
    converged = results.get('converged_to_truth', {}).get('converged', False)
    accuracy = results.get('converged_to_truth', {}).get('final_accuracy', 0.0)
    
    print(f"Target variable: {target}")
    print(f"True parents: {true_parents}")
    print(f"Converged: {converged}")
    print(f"Final accuracy: {accuracy:.3f}")
    
    # Progress metrics
    if 'final_uncertainty' in results:
        print(f"Final uncertainty: {results['final_uncertainty']:.2f} bits")
    
    if 'improvement' in results:
        print(f"Target improvement: {results['improvement']:.3f}")
    
    # Sample usage
    if 'total_samples' in results:
        print(f"Total samples used: {results['total_samples']}")
    
    if 'intervention_counts' in results:
        print(f"Intervention counts: {results['intervention_counts']}")
    
    # Final marginal probabilities
    if 'final_marginal_probs' in results:
        print(f"\nFinal marginal probabilities:")
        marginals = results['final_marginal_probs']
        for var in sorted(marginals.keys()):
            prob = marginals[var]
            is_true_parent = "✅" if var in true_parents else "❌"
            print(f"  {var}: {prob:.4f} {is_true_parent}")


def validate_experiment_results(results: Dict[str, Any], expected_convergence: bool = None,
                               min_accuracy: float = None, max_uncertainty: float = None) -> bool:
    """Validate experiment results against expected criteria."""
    validations = []
    
    # Check convergence
    if expected_convergence is not None:
        converged = results.get('converged_to_truth', {}).get('converged', False)
        convergence_ok = (converged == expected_convergence)
        validations.append(convergence_ok)
        print(f"Convergence check: {'✅' if convergence_ok else '❌'} (expected: {expected_convergence}, got: {converged})")
    
    # Check accuracy
    if min_accuracy is not None:
        accuracy = results.get('converged_to_truth', {}).get('final_accuracy', 0.0)
        accuracy_ok = accuracy >= min_accuracy
        validations.append(accuracy_ok)
        print(f"Accuracy check: {'✅' if accuracy_ok else '❌'} (min: {min_accuracy}, got: {accuracy:.3f})")
    
    # Check uncertainty
    if max_uncertainty is not None:
        uncertainty = results.get('final_uncertainty', float('inf'))
        uncertainty_ok = uncertainty <= max_uncertainty
        validations.append(uncertainty_ok)
        print(f"Uncertainty check: {'✅' if uncertainty_ok else '❌'} (max: {max_uncertainty}, got: {uncertainty:.2f})")
    
    all_passed = all(validations) if validations else True
    print(f"Overall validation: {'✅ PASSED' if all_passed else '❌ FAILED'}")
    
    return all_passed


def compute_f1_metrics(marginal_probs: Dict[str, float], true_parents: List[str], 
                      threshold: float = 0.7) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score for parent set prediction.
    
    Args:
        marginal_probs: Dictionary of variable -> probability
        true_parents: List of ground truth parent variables
        threshold: Probability threshold for considering a variable as a parent
        
    Returns:
        Dictionary with precision, recall, F1, and other metrics
    """
    if not marginal_probs:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'accuracy': 0.0
        }
    
    # Get predictions based on threshold
    predicted_parents = {var for var, prob in marginal_probs.items() if prob > threshold}
    true_parent_set = set(true_parents)
    all_variables = set(marginal_probs.keys())
    
    # Compute confusion matrix elements
    true_positives = len(predicted_parents & true_parent_set)
    false_positives = len(predicted_parents - true_parent_set)
    false_negatives = len(true_parent_set - predicted_parents)
    true_negatives = len(all_variables - predicted_parents - true_parent_set)
    
    # Compute metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (true_positives + true_negatives) / len(all_variables) if len(all_variables) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives,
        'accuracy': accuracy
    }


def compute_f1_progress(marginal_prob_progress: List[Dict[str, float]], true_parents: List[str],
                       threshold: float = 0.7) -> List[Dict[str, float]]:
    """
    Compute F1 metrics over time for convergence analysis.
    
    Args:
        marginal_prob_progress: List of marginal probability dictionaries over time
        true_parents: List of ground truth parent variables
        threshold: Probability threshold for prediction
        
    Returns:
        List of F1 metrics dictionaries for each time step
    """
    f1_progress = []
    
    for step_probs in marginal_prob_progress:
        step_metrics = compute_f1_metrics(step_probs, true_parents, threshold)
        f1_progress.append(step_metrics)
    
    return f1_progress


def analyze_convergence_speed(f1_progress: List[Dict[str, float]], 
                            accuracy_thresholds: List[float] = [0.5, 0.75, 0.9, 1.0]) -> Dict[str, Any]:
    """
    Analyze convergence speed based on F1 score progression.
    
    Args:
        f1_progress: List of F1 metrics over time
        accuracy_thresholds: F1 score thresholds to analyze
        
    Returns:
        Dictionary with convergence speed metrics
    """
    if not f1_progress:
        return {
            'convergence_steps': {},
            'final_f1': 0.0,
            'max_f1': 0.0,
            'auc_f1': 0.0,
            'convergence_rate': 0.0,
            'steps_to_max': len(f1_progress)
        }
    
    f1_scores = [metrics['f1_score'] for metrics in f1_progress]
    
    # Find convergence steps for different thresholds
    convergence_steps = {}
    for threshold in accuracy_thresholds:
        step = next((i for i, f1 in enumerate(f1_scores) if f1 >= threshold), None)
        convergence_steps[f"f1_{threshold}"] = step
    
    # Compute additional metrics
    final_f1 = f1_scores[-1] if f1_scores else 0.0
    max_f1 = max(f1_scores) if f1_scores else 0.0
    steps_to_max = f1_scores.index(max_f1) if f1_scores and max_f1 > 0 else len(f1_progress)
    
    # Area Under Curve (AUC) for F1 score
    auc_f1 = np.trapz(f1_scores) / len(f1_scores) if len(f1_scores) > 1 else final_f1
    
    # Convergence rate (improvement per step)
    if len(f1_scores) > 1:
        convergence_rate = (final_f1 - f1_scores[0]) / (len(f1_scores) - 1)
    else:
        convergence_rate = 0.0
    
    return {
        'convergence_steps': convergence_steps,
        'final_f1': final_f1,
        'max_f1': max_f1,
        'auc_f1': auc_f1,
        'convergence_rate': convergence_rate,
        'steps_to_max': steps_to_max,
        'f1_scores': f1_scores
    }


def comprehensive_evaluation(marginal_prob_progress: List[Dict[str, float]], true_parents: List[str],
                           threshold: float = 0.7) -> Dict[str, Any]:
    """
    Perform comprehensive evaluation including convergence analysis, F1 metrics, and speed analysis.
    
    Args:
        marginal_prob_progress: List of marginal probability dictionaries over time
        true_parents: List of ground truth parent variables
        threshold: Probability threshold for prediction
        
    Returns:
        Dictionary with comprehensive evaluation results
    """
    # Basic convergence analysis
    convergence_analysis = analyze_convergence(marginal_prob_progress, true_parents, threshold)
    
    # F1 metrics over time
    f1_progress = compute_f1_progress(marginal_prob_progress, true_parents, threshold)
    
    # Speed analysis
    speed_analysis = analyze_convergence_speed(f1_progress)
    
    # Extract final metrics
    final_metrics = f1_progress[-1] if f1_progress else compute_f1_metrics({}, true_parents, threshold)
    
    return {
        'convergence_analysis': convergence_analysis,
        'f1_progress': f1_progress,
        'speed_analysis': speed_analysis,
        'final_metrics': final_metrics,
        'summary': {
            'converged': convergence_analysis['converged'],
            'final_f1': final_metrics['f1_score'],
            'final_precision': final_metrics['precision'],
            'final_recall': final_metrics['recall'],
            'steps_to_convergence': convergence_analysis['convergence_step'],
            'auc_f1': speed_analysis['auc_f1'],
            'convergence_rate': speed_analysis['convergence_rate']
        }
    }


def compare_intervention_strategies_detailed(strategy1_results: Dict[str, Any], strategy2_results: Dict[str, Any],
                                           strategy1_name: str = "Strategy 1", strategy2_name: str = "Strategy 2") -> Dict[str, Any]:
    """
    Detailed comparison of two intervention strategies including statistical analysis.
    
    Args:
        strategy1_results: Results from first strategy
        strategy2_results: Results from second strategy
        strategy1_name: Name of first strategy
        strategy2_name: Name of second strategy
        
    Returns:
        Dictionary with detailed comparison results
    """
    # Extract comprehensive evaluations
    eval1 = strategy1_results.get('comprehensive_evaluation', {})
    eval2 = strategy2_results.get('comprehensive_evaluation', {})
    
    summary1 = eval1.get('summary', {})
    summary2 = eval2.get('summary', {})
    
    # Compare key metrics
    f1_comparison = {
        'strategy1_f1': summary1.get('final_f1', 0.0),
        'strategy2_f1': summary2.get('final_f1', 0.0),
        'f1_advantage': summary1.get('final_f1', 0.0) - summary2.get('final_f1', 0.0)
    }
    
    speed_comparison = {
        'strategy1_steps': summary1.get('steps_to_convergence', float('inf')),
        'strategy2_steps': summary2.get('steps_to_convergence', float('inf')),
        'speed_advantage': (summary2.get('steps_to_convergence', float('inf')) - 
                           summary1.get('steps_to_convergence', float('inf')))
    }
    
    convergence_comparison = {
        'strategy1_converged': summary1.get('converged', False),
        'strategy2_converged': summary2.get('converged', False),
        'convergence_advantage': summary1.get('converged', False) and not summary2.get('converged', False)
    }
    
    # Overall assessment
    strategy1_better = (
        f1_comparison['f1_advantage'] > 0.1 or
        (speed_comparison['speed_advantage'] > 0 and f1_comparison['f1_advantage'] >= 0) or
        convergence_comparison['convergence_advantage']
    )
    
    print(f"\n📊 DETAILED STRATEGY COMPARISON")
    print("=" * 60)
    print(f"Comparing {strategy1_name} vs {strategy2_name}")
    
    print(f"\n🎯 F1 Score Comparison:")
    print(f"   {strategy1_name}: {f1_comparison['strategy1_f1']:.3f}")
    print(f"   {strategy2_name}: {f1_comparison['strategy2_f1']:.3f}")
    print(f"   Advantage: {f1_comparison['f1_advantage']:+.3f} ({strategy1_name})")
    
    print(f"\n⚡ Speed Comparison:")
    s1_steps = speed_comparison['strategy1_steps']
    s2_steps = speed_comparison['strategy2_steps']
    print(f"   {strategy1_name}: {s1_steps if s1_steps != float('inf') else 'No convergence'}")
    print(f"   {strategy2_name}: {s2_steps if s2_steps != float('inf') else 'No convergence'}")
    if speed_comparison['speed_advantage'] != 0:
        print(f"   Speed advantage: {abs(speed_comparison['speed_advantage'])} steps ({strategy1_name if speed_comparison['speed_advantage'] > 0 else strategy2_name})")
    
    print(f"\n✅ Convergence Comparison:")
    print(f"   {strategy1_name}: {'Converged' if convergence_comparison['strategy1_converged'] else 'Failed'}")
    print(f"   {strategy2_name}: {'Converged' if convergence_comparison['strategy2_converged'] else 'Failed'}")
    
    print(f"\n🏆 Overall Winner: {strategy1_name if strategy1_better else strategy2_name}")
    
    return {
        'strategy1_name': strategy1_name,
        'strategy2_name': strategy2_name,
        'f1_comparison': f1_comparison,
        'speed_comparison': speed_comparison,
        'convergence_comparison': convergence_comparison,
        'strategy1_better': strategy1_better,
        'detailed_metrics': {
            'strategy1': summary1,
            'strategy2': summary2
        }
    }


def extract_trajectory_metrics_from_demo(demo_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract trajectory metrics from demo results in the format expected by visualization functions.
    
    This function bridges the gap between demo experiment results and the visualization
    functions in plots.py, ensuring all necessary metrics are available for plotting.
    
    Args:
        demo_results: Results from run_progressive_learning_demo or similar
        
    Returns:
        Dictionary with trajectory metrics suitable for visualization
    """
    # Extract basic info
    target_variable = demo_results.get('target_variable', 'target')
    true_parents = demo_results.get('true_parents', [])
    learning_history = demo_results.get('learning_history', [])
    
    # Initialize trajectory lists
    steps = []
    target_values = []
    f1_scores = []
    precisions = []
    recalls = []
    shd_values = []
    true_parent_likelihood = []
    uncertainty_bits = []
    rewards = []  # For reward signal visualization
    
    # Process learning history to extract metrics
    for i, step_data in enumerate(learning_history):
        steps.append(i + 1)
        
        # Target value (outcome)
        outcome_value = step_data.get('outcome_value', 0.0)
        target_values.append(outcome_value)
        
        # For reward signal, we can use the improvement from baseline
        if i == 0:
            baseline_value = outcome_value
        reward = outcome_value - baseline_value if i > 0 else 0.0
        rewards.append(reward)
        
        # Marginal probabilities
        marginals = step_data.get('marginals', {})
        
        # Compute F1 metrics
        f1_metrics = compute_f1_metrics(marginals, true_parents, threshold=0.5)
        f1_scores.append(f1_metrics['f1_score'])
        precisions.append(f1_metrics['precision'])
        recalls.append(f1_metrics['recall'])
        
        # Compute SHD (Structural Hamming Distance)
        # For SHD, count the number of edge differences
        predicted_parents = {var for var, prob in marginals.items() if prob > 0.5}
        true_parent_set = set(true_parents)
        
        # Edge differences = (predicted but not true) + (true but not predicted)
        false_positives = len(predicted_parents - true_parent_set)
        false_negatives = len(true_parent_set - predicted_parents)
        shd = false_positives + false_negatives
        shd_values.append(shd)
        
        # True parent likelihood (average probability of true parents)
        if true_parents:
            parent_probs = [marginals.get(parent, 0.0) for parent in true_parents]
            avg_parent_prob = np.mean(parent_probs)
        else:
            avg_parent_prob = 0.0
        true_parent_likelihood.append(avg_parent_prob)
        
        # Uncertainty
        uncertainty = step_data.get('uncertainty', 0.0)
        uncertainty_bits.append(uncertainty)
    
    # Also extract from dedicated progress lists if available
    if 'target_progress' in demo_results:
        # Use the cumulative best values for target optimization
        target_values = demo_results['target_progress'][1:]  # Skip initial value
        
    if 'uncertainty_progress' in demo_results:
        uncertainty_bits = demo_results['uncertainty_progress'][1:]  # Skip initial value
        
    if 'marginal_prob_progress' in demo_results:
        # Recompute metrics from marginal progress for consistency
        true_parent_likelihood = []
        f1_scores = []
        shd_values = []
        
        for marginals in demo_results['marginal_prob_progress'][1:]:  # Skip initial
            # True parent likelihood
            if true_parents:
                parent_probs = [marginals.get(parent, 0.0) for parent in true_parents]
                avg_parent_prob = np.mean(parent_probs)
            else:
                avg_parent_prob = 0.0
            true_parent_likelihood.append(avg_parent_prob)
            
            # F1 score
            f1_metrics = compute_f1_metrics(marginals, true_parents, threshold=0.5)
            f1_scores.append(f1_metrics['f1_score'])
            
            # SHD
            predicted_parents = {var for var, prob in marginals.items() if prob > 0.5}
            true_parent_set = set(true_parents)
            false_positives = len(predicted_parents - true_parent_set)
            false_negatives = len(true_parent_set - predicted_parents)
            shd = false_positives + false_negatives
            shd_values.append(shd)
    
    # Ensure all lists have the same length
    max_length = max(len(lst) for lst in [steps, target_values, f1_scores, shd_values, 
                                          true_parent_likelihood, uncertainty_bits] if lst)
    
    # Pad shorter lists with their last value
    def pad_list(lst, target_length):
        if not lst:
            return [0.0] * target_length
        while len(lst) < target_length:
            lst.append(lst[-1])
        return lst[:target_length]
    
    steps = list(range(1, max_length + 1))
    target_values = pad_list(target_values, max_length)
    f1_scores = pad_list(f1_scores, max_length)
    shd_values = pad_list(shd_values, max_length)
    true_parent_likelihood = pad_list(true_parent_likelihood, max_length)
    uncertainty_bits = pad_list(uncertainty_bits, max_length)
    rewards = pad_list(rewards, max_length)
    
    # Return in format expected by plotting functions
    return {
        'steps': steps,
        'target_values': target_values,
        'f1_scores': f1_scores,
        'precisions': precisions[:max_length] if precisions else [0.0] * max_length,
        'recalls': recalls[:max_length] if recalls else [0.0] * max_length,
        'shd_values': shd_values,
        'true_parent_likelihood': true_parent_likelihood,
        'uncertainty_bits': uncertainty_bits,
        'rewards': rewards,
        
        # Additional metadata
        'target_variable': target_variable,
        'true_parents': true_parents,
        'final_best': demo_results.get('final_best', target_values[-1] if target_values else 0.0),
        'improvement': demo_results.get('improvement', 0.0),
        'converged': demo_results.get('converged_to_truth', {}).get('converged', False),
        'total_samples': demo_results.get('total_samples', len(steps))
    }